#include "hyle/compute/apple.h"
#include "hyle/compute/apple_memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <vector>
#include <thread>
#include <optional>
#include <tuple>

#include <gtest/gtest.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <stdexcept>
#include <mlx/backend/metal/event.h>

#include "../../src/compute_route_apple/apple_carrier.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/model_ops.h"
#include "recurrent_fixture.h"

namespace {
namespace a = hyle::compute::apple;
namespace ad = a::detail;
namespace c = hyle::compute;
namespace mx = mlx::core;

// A test-only gate in the original FIFO, with no extra thread or GPU work.
// Every assertion exit releases and joins it before its stack storage dies.
class FifoGate final : public ad::Job {
public:
  FifoGate() {
    execute = [](ad::Job *job) noexcept {
      auto &gate = *static_cast<FifoGate *>(job);
      std::unique_lock lock(gate.mutex_);
      gate.entered_ = true;
      gate.changed_.notify_all();
      gate.changed_.wait(lock, [&] { return gate.open_; });
      gate.finished_ = true;
      gate.changed_.notify_all();
    };
  }
  ~FifoGate() { open_and_join(); }
  hyle::Status enqueue(const std::shared_ptr<ad::Runtime> &runtime) noexcept {
    auto status = runtime->enqueue(*this);
    admitted_ = status.ok();
    return status;
  }
  void wait_entered() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return entered_; });
  }
  bool wait_entered_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return entered_; });
  }
  void open_and_join() {
    if (!admitted_) return;
    std::unique_lock lock(mutex_);
    open_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return finished_; });
    admitted_ = false;
  }
private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool admitted_ = false, entered_ = false, open_ = false, finished_ = false;
};

// Block real GPU work while the original FIFO remains free to encode Jobs.
// Destruction always releases the event, including assertion exits.
class GpuEventGate final : public ad::Job {
public:
  explicit GpuEventGate(std::shared_ptr<ad::Runtime> runtime) : runtime_(std::move(runtime)) {
    execute = [](ad::Job *job) noexcept {
      auto &gate = *static_cast<GpuEventGate *>(job);
      auto event = mx::Event(*gate.runtime_->stream);
      event.set_value(1);
      event.wait(*gate.runtime_->stream);
      auto &encoder = mx::metal::get_command_encoder(*gate.runtime_->stream);
      encoder.end_encoding();
      encoder.commit();
      { std::lock_guard lock(gate.mutex_); gate.event_ = std::move(event); }
      gate.changed_.notify_all();
    };
  }
  hyle::Status enqueue() noexcept {
    auto status = runtime_->enqueue(*this);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return event_.valid(); });
    return hyle::Status::Ok();
  }
  void open() {
    std::lock_guard lock(mutex_);
    if (event_.valid()) event_.cast<mx::metal::EventImpl>().signal(1);
  }
  ~GpuEventGate() { open(); }
private:
  std::shared_ptr<ad::Runtime> runtime_;
  std::mutex mutex_;
  std::condition_variable changed_;
  mx::Event event_;
};

// Let MLX commit actual prefix math at its real operation threshold, then hold
// only the remaining GPU work. A first-buffer Source callback would prepend a
// blocking retirement Job while this graph's real suffix still waits.
class SplitGpuFrontier final {
public:
  explicit SplitGpuFrontier(std::shared_ptr<ad::Runtime> runtime)
      : runtime_(std::move(runtime)) {
    capture_.owner = this;
    capture_.execute = [](ad::Job *job) noexcept {
      auto &self = *static_cast<Capture *>(job)->owner;
      auto &stream = *self.runtime_->stream;
      const auto limits = mx::metal::device(stream.device).get_max_ops_mb_per_buffer();
      auto first = NS::RetainPtr(mx::metal::get_command_encoder(stream).get_command_buffer());
      {
        std::lock_guard lock(self.mutex_);
        self.first_ = std::move(first);
        self.maximum_ops_ = std::get<0>(limits);
        self.recorded_ = true;
      }
      self.changed_.notify_all();
    };
  }
  hyle::Status capture() {
    { std::lock_guard lock(mutex_); recorded_ = false; }
    const auto status = runtime_->enqueue(capture_);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return recorded_; });
    return hyle::Status::Ok();
  }
  int maximum_ops() { std::lock_guard lock(mutex_); return maximum_ops_; }
  void arm() { armed_.store(true, std::memory_order_release); }
  void encode(mx::Stream stream) {
    if (!armed_.exchange(false, std::memory_order_acq_rel)) return;
    NS::SharedPtr<MTL::CommandBuffer> first;
    { std::lock_guard lock(mutex_); first = first_; }
    // Softmax is not a fusable primitive in locked MLX. Its real dispatches
    // have naturally exceeded the queried threshold, so MLX's gpu::eval has
    // already committed this buffer and installed its own handlers correctly.
    if (!first || first->status() == MTL::CommandBufferStatusNotEnqueued)
      throw std::runtime_error("natural MLX operation threshold did not commit the first buffer");
    first->waitUntilCompleted();
    auto event = mx::Event(stream);
    event.set_value(1);
    event.wait(stream);
    {
      std::lock_guard lock(mutex_);
      event_ = std::move(event);
      if (open_) event_.cast<mx::metal::EventImpl>().signal(1);
    }
    changed_.notify_all();
  }
  bool wait_entered_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return event_.valid(); });
  }
  void open() {
    std::lock_guard lock(mutex_);
    open_ = true;
    if (event_.valid()) event_.cast<mx::metal::EventImpl>().signal(1);
  }
private:
  struct Capture final : ad::Job { SplitGpuFrontier *owner = nullptr; } capture_;
  std::shared_ptr<ad::Runtime> runtime_;
  NS::SharedPtr<MTL::CommandBuffer> first_;
  std::atomic<bool> armed_{false};
  std::mutex mutex_;
  std::condition_variable changed_;
  mx::Event event_;
  int maximum_ops_ = 0;
  bool open_ = false, recorded_ = false;
};
class SplitGpuCheckpoint final : public mx::Primitive {
public:
  SplitGpuCheckpoint(mx::Stream stream, std::shared_ptr<SplitGpuFrontier> frontier)
      : Primitive(stream), frontier_(std::move(frontier)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::logic_error("split GPU checkpoint requires its original Metal stream");
  }
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    frontier_->encode(stream());
    outputs[0].copy_shared_buffer(inputs[0]);
  }
  const char *name() const override { return "HyleTestSplitGpuPrefix"; }
private:
  std::shared_ptr<SplitGpuFrontier> frontier_;
};

void exercise_device_prefix_last_buffer(bool cancel) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  auto runtime = ad::ContextAccess::runtime(*context);
  auto frontier = std::make_shared<SplitGpuFrontier>(runtime);
  ASSERT_TRUE(frontier->capture().ok());
  const auto maximum_ops = frontier->maximum_ops();
  ASSERT_GT(maximum_ops, 0);
  ASSERT_LE(maximum_ops, 256) << "bounded fixture requires at most 258 real softmax dispatches";
  const auto prefix_ops = (std::max)(std::size_t{32}, static_cast<std::size_t>(maximum_ops) + 2);
  struct OpenOnExit final {
    std::shared_ptr<SplitGpuFrontier> frontier;
    ~OpenOnExit() { frontier->open(); }
  } release{frontier};
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = ad::ContextAccess::compile_device_prefix(runtime,
      [frontier, prefix_ops](const auto &inputs, mx::Stream stream) {
        auto prefix = inputs[0];
        for (std::size_t i = 0; i < prefix_ops; ++i)
          prefix = mx::softmax(prefix, -1, true, stream);
        auto middle = mx::array(prefix.shape(), prefix.dtype(),
            std::make_shared<SplitGpuCheckpoint>(stream, frontier), {prefix});
        return std::vector<mx::array>{mx::square(middle, stream)};
      }, specs);
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot) << slot.error().message();
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(slot->bind(inputs).ok());
  ASSERT_TRUE(frontier->capture().ok()); // After every cold compile/finalize.
  frontier->arm();
  auto work = slot->submit();
  ASSERT_TRUE(work) << work.error().message();
  ASSERT_TRUE(frontier->wait_entered_for(std::chrono::seconds(2)));
  FifoGate owner;
  const auto admitted = owner.enqueue(runtime);
  const bool forward = admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
  const auto pending = work->completion().poll();
  if (cancel) EXPECT_TRUE(work->request_cancel().ok());
  frontier->open();
  owner.open_and_join();
  ASSERT_TRUE(admitted.ok()) << admitted.message();
  EXPECT_TRUE(forward) << "first-buffer callback blocked the original owner before the actual prefix ended";
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  const auto terminal = work->completion().wait();
  EXPECT_TRUE(cancel ? terminal.IsCancelled() : terminal.ok()) << terminal.message();
  if (!cancel) {
    auto output = ad::InvocationAccess::output(*slot, 0);
    ASSERT_TRUE(output) << output.error().message();
    const std::array<float, 4> expected{.0625f, .0625f, .0625f, .0625f};
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_FLOAT_EQ(output->data<float>()[i], expected[i]);
  }
  auto fresh = program->create_invocation();
  ASSERT_TRUE(fresh) << fresh.error().message();
  ASSERT_TRUE(fresh->bind(inputs).ok());
  auto reused = fresh->submit();
  ASSERT_TRUE(reused) << reused.error().message();
  EXPECT_TRUE(reused->completion().wait().ok());
  ASSERT_TRUE(context->close().ok());
  { std::lock_guard lock(runtime->queue_mutex); EXPECT_EQ(runtime->native_tails, 0u); }
}
TEST(AppleLateInput, DevicePrefixRetirementFollowsLastBufferWithoutBlockingTheOriginalOwner) {
  exercise_device_prefix_last_buffer(false);
}
TEST(AppleLateInput, DevicePrefixLastBufferCancelRetiresBeforeFreshSlotReuse) {
  exercise_device_prefix_last_buffer(true);
}
TEST(AppleLateInput, DevicePrefixReadyViewNeedsNoNativeRetirementBuffer) {
  auto context = a::Context::create().value();
  auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = ad::ContextAccess::compile_device_prefix(runtime,
      [](const auto &inputs, mx::Stream) { return std::vector<mx::array>{inputs[0]}; }, specs);
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation().value();
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto work = slot.submit().value();
  ASSERT_TRUE(work.completion().wait().ok());
  auto output = ad::InvocationAccess::output(slot, 0).value();
  EXPECT_EQ(output.data_shared_ptr(), inputs[0].data_shared_ptr());
  { std::lock_guard lock(runtime->queue_mutex);
    EXPECT_EQ(runtime->native_tail_peak, 0u); EXPECT_EQ(runtime->native_tails, 0u); }
  EXPECT_TRUE(context.close().ok());
}
TEST(AppleLateInput, DevicePrefixPendingViewKeepsItsOriginalSourceAndActualTail) {
  auto context = a::Context::create().value();
  auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto first = ad::ContextAccess::compile_device_prefix(runtime,
      [](const auto &inputs, mx::Stream stream) {
        return std::vector<mx::array>{mx::square(inputs[0], stream)};
      }, specs).value();
  auto view = ad::ContextAccess::compile_device_prefix(runtime,
      [](const auto &inputs, mx::Stream) { return std::vector<mx::array>{inputs[0]}; }, specs).value();
  auto producer = first.create_invocation().value(), consumer = view.create_invocation().value();
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer.bind(inputs).ok());
  std::array<ad::InputBinding, 1> late;
  late[0].source = ad::InvocationAccess::source(producer, 0).value();
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(consumer, late).ok());
  GpuEventGate gpu(runtime);
  ASSERT_TRUE(gpu.enqueue().ok());
  auto produced = producer.submit().value(), consumed = consumer.submit().value();
  FifoGate owner;
  const auto admitted = owner.enqueue(runtime);
  const bool forward = admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
  const auto pending = consumed.completion().poll();
  gpu.open(); owner.open_and_join();
  ASSERT_TRUE(admitted.ok()); EXPECT_TRUE(forward);
  ASSERT_TRUE(pending); EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  ASSERT_TRUE(consumed.completion().wait().ok());
  ASSERT_TRUE(produced.completion().wait().ok());
  auto source_output = ad::InvocationAccess::output(producer, 0).value();
  auto view_output = ad::InvocationAccess::output(consumer, 0).value();
  EXPECT_EQ(source_output.data_shared_ptr(), view_output.data_shared_ptr());
  const std::array<float, 4> expected{1.f, 4.f, 9.f, 16.f};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_FLOAT_EQ(view_output.data<float>()[i], expected[i]);
  ASSERT_TRUE(context.close().ok());
  { std::lock_guard lock(runtime->queue_mutex); EXPECT_EQ(runtime->native_tails, 0u); }
}

// Hold only the callback which publishes the original prefix's completion.
// Its actual GPU event can signal first, as it does during ordinary MLX eval.
class PrefixCallbackGate final : public std::enable_shared_from_this<PrefixCallbackGate> {
public:
  void arm() { armed_.store(true, std::memory_order_release); }
  void install(mx::Stream stream) {
    if (!armed_.exchange(false, std::memory_order_acq_rel)) return;
    auto *command = mx::metal::get_command_encoder(stream).get_command_buffer();
    if (!command) throw std::runtime_error("original prefix command buffer is absent");
    command->addCompletedHandler([self = shared_from_this()](MTL::CommandBuffer *) {
      std::unique_lock lock(self->mutex_);
      self->entered_ = true; self->changed_.notify_all();
      self->changed_.wait(lock, [&] { return self->open_; });
    });
  }
  bool wait_entered_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return entered_; });
  }
  void open() {
    std::lock_guard lock(mutex_); open_ = true; changed_.notify_all();
  }
private:
  std::atomic<bool> armed_{false};
  std::mutex mutex_;
  std::condition_variable changed_;
  bool entered_ = false, open_ = false;
};
class PrefixCallbackCheckpoint final : public mx::Primitive {
public:
  PrefixCallbackCheckpoint(mx::Stream stream, std::shared_ptr<PrefixCallbackGate> gate)
      : Primitive(stream), gate_(std::move(gate)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::logic_error("prefix callback checkpoint requires its original GPU");
  }
  void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override {
    outputs[0].copy_shared_buffer(inputs[0]);
    gate_->install(stream());
  }
  const char *name() const override { return "HyleTestPrefixCallbackCheckpoint"; }
private:
  std::shared_ptr<PrefixCallbackGate> gate_;
};
class DetachSignaledSource final : public ad::Job {
public:
  DetachSignaledSource(std::shared_ptr<ad::Runtime> runtime, ad::InputSource source,
      std::shared_ptr<PrefixCallbackGate> callback, bool inspect_address)
      : runtime_(std::move(runtime)), source_(std::move(source)), callback_(std::move(callback)),
        inspect_address_(inspect_address) {
    execute = [](ad::Job *job) noexcept { static_cast<DetachSignaledSource *>(job)->run(); };
  }
  hyle::Status enqueue() { return runtime_->enqueue(*this); }
  bool wait_entered_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return entered_ || done_; });
  }
  bool wait_done_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return done_; });
  }
  hyle::Status status() { std::lock_guard lock(mutex_); return status_; }
private:
  void run() noexcept {
    hyle::Status status;
    try {
      std::array<a::NativeStorage, 4> words;
      std::array<std::optional<mx::array>, 4> owners;
      std::array<ad::DeviceCheckVisit, 8> visited{};
      ad::DeviceCheckInputs checks{words, owners, 0, visited};
      auto epoch = ad::InvocationAccess::source_epoch(source_, runtime_);
      if (!epoch) throw std::runtime_error(std::string(epoch.error().message()));
      auto actual = ad::InvocationAccess::device_source_output(source_, runtime_, *epoch, checks);
      if (!actual) throw std::runtime_error(std::string(actual.error().message()));
      auto value = std::move(*actual);
      if (value.status() != mx::array::evaluated || !value.event().valid())
        throw std::runtime_error("original producer has no actual evaluated GPU event");
      std::optional<a::NativeStorage> original_address;
      if (inspect_address_) {
        auto address = ad::MemoryAccess::submitted_storage_info(value);
        if (!address) throw std::runtime_error(std::string(address.error().message()));
        original_address = *address;
      }
      { std::lock_guard lock(mutex_); entered_ = true; changed_.notify_all(); }
      // The test's independent caller opens the real GPU gate. The public
      // event wait does not wait for the deliberately held completed handler.
      value.event().wait();
      if (!value.event().is_signaled() || !callback_->wait_entered_for(std::chrono::seconds(2)))
        throw std::runtime_error("original GPU event/held callback frontier was not reached");
      value.detach_event(); // The same legal operation used by MLX transforms.
      if (value.status() != mx::array::evaluated || value.event().valid())
        throw std::runtime_error("detaching the signaled original event changed raw evaluation state");
      if (inspect_address_) {
        // This is the original pinned Source, before the next Context input
        // resolves it. Ready normalization grants no different native bytes.
        auto address = ad::MemoryAccess::submitted_storage_info(value);
        if (!address) throw std::runtime_error(std::string(address.error().message()));
        if (address->buffer != original_address->buffer || address->offset_bytes != original_address->offset_bytes ||
            address->view_bytes != original_address->view_bytes || address->backing_bytes != original_address->backing_bytes)
          throw std::runtime_error("event normalization changed the original native address/span");
      }
      auto unchanged = ad::InvocationAccess::source_epoch(source_, runtime_);
      if (!unchanged || *unchanged != *epoch)
        throw std::runtime_error("event normalization replaced the original Source epoch");
    } catch (const std::exception &failure) { status = hyle::Status::Internal(failure.what()); }
    { std::lock_guard lock(mutex_); status_ = std::move(status); done_ = true; changed_.notify_all(); }
  }
  std::shared_ptr<ad::Runtime> runtime_;
  ad::InputSource source_;
  std::shared_ptr<PrefixCallbackGate> callback_;
  std::mutex mutex_;
  std::condition_variable changed_;
  hyle::Status status_;
  bool entered_ = false, done_ = false, inspect_address_ = false;
};
void exercise_device_prefix_detached_source(bool inspect_address) {
  auto context = a::Context::create().value();
  auto runtime = ad::ContextAccess::runtime(context);
  auto callback = std::make_shared<PrefixCallbackGate>();
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto first = ad::ContextAccess::compile_device_prefix(runtime,
      [callback](const auto &inputs, mx::Stream stream) {
        auto value = mx::square(inputs[0], stream);
        return std::vector<mx::array>{mx::array(value.shape(), value.dtype(),
            std::make_shared<PrefixCallbackCheckpoint>(stream, callback), {value})};
      }, specs).value();
  auto second = ad::ContextAccess::compile_device_prefix(runtime,
      [](const auto &inputs, mx::Stream stream) { return std::vector<mx::array>{mx::square(inputs[0], stream)}; }, specs).value();
  auto producer = first.create_invocation().value(), consumer = second.create_invocation().value();
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer.bind(inputs).ok());
  const auto original = ad::InvocationAccess::source(producer, 0).value();
  std::array<ad::InputBinding, 1> late;
  late[0].source = original;
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(consumer, late).ok());
  GpuEventGate gpu(runtime);
  FifoGate before, after;
  DetachSignaledSource detach(runtime, original, callback, inspect_address);
  struct OpenOnExit final {
    GpuEventGate &gpu; FifoGate &before; FifoGate &after; std::shared_ptr<PrefixCallbackGate> callback;
    ~OpenOnExit() { gpu.open(); callback->open(); before.open_and_join(); after.open_and_join(); }
  } release{gpu, before, after, callback};
  ASSERT_TRUE(gpu.enqueue().ok());
  ASSERT_TRUE(before.enqueue(runtime).ok()); before.wait_entered();
  callback->arm();
  auto produced = producer.submit().value();
  ASSERT_TRUE(detach.enqueue().ok());
  auto consumed = consumer.submit().value();
  before.open_and_join();
  const bool entered = detach.wait_entered_for(std::chrono::seconds(2));
  gpu.open();
  const bool detached = detach.wait_done_for(std::chrono::seconds(2));
  const auto after_admitted = after.enqueue(runtime);
  if (after_admitted.ok()) (void)after.wait_entered_for(std::chrono::seconds(2));
  const auto pending_producer = produced.completion().poll();
  callback->open(); after.open_and_join();
  const auto consumer_terminal = consumed.completion().wait();
  const auto producer_terminal = produced.completion().wait();
  ASSERT_TRUE(entered); ASSERT_TRUE(detached);
  ASSERT_TRUE(detach.status().ok()) << detach.status().message();
  ASSERT_TRUE(after_admitted.ok());
  ASSERT_TRUE(pending_producer); EXPECT_EQ(*pending_producer, c::CompletionPoll::kPending);
  ASSERT_TRUE(producer_terminal.ok()) << producer_terminal.message();
  ASSERT_TRUE(consumer_terminal.ok()) << consumer_terminal.message();
  {
    auto output = ad::InvocationAccess::output(consumer, 0).value();
    const std::array<float, 4> expected{1.f, 16.f, 81.f, 256.f};
    for (std::size_t at = 0; at < expected.size(); ++at) EXPECT_FLOAT_EQ(output.data<float>()[at], expected[at]);
  }
  produced = {}; consumed = {}; // Release original result readers before reuse.
  ASSERT_TRUE(producer.bind(inputs).ok());
  auto reused_producer = producer.submit();
  ASSERT_TRUE(reused_producer) << reused_producer.error().message();
  auto reused_consumer = consumer.submit();
  ASSERT_TRUE(reused_consumer) << reused_consumer.error().message();
  ASSERT_TRUE(reused_consumer->completion().wait().ok());
  ASSERT_TRUE(reused_producer->completion().wait().ok());
  ASSERT_TRUE(context.close().ok());
  { std::lock_guard lock(runtime->queue_mutex); EXPECT_EQ(runtime->native_tails, 0u); }
}
TEST(AppleLateInput, DevicePrefixSignaledDetachedEventKeepsItsOriginalSourceEpoch) {
  exercise_device_prefix_detached_source(false);
}
TEST(AppleLateInput, DevicePrefixDetachedReadySourceRetainsItsOriginalNativeAddress) {
  exercise_device_prefix_detached_source(true);
}

void exercise_native_signal_source(bool close_pending) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  apple_test::Recurrent fixture(3, 2, 64);
  const auto expected = fixture.oracle();
  auto first = context->prepare(c::GatedDeltaNet{}, fixture.specs());
  ASSERT_TRUE(first) << first.error().message();
  const std::array specs{a::InputSpec{fixture.inputs.front().shape(), mx::float32}};
  auto second = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, specs);
  ASSERT_TRUE(second) << second.error().message();
  auto producer = first->create_invocation(), consumer = second->create_invocation();
  ASSERT_TRUE(producer); ASSERT_TRUE(consumer);
  auto source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  std::array<ad::InputBinding, 1> late;
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset();
  ASSERT_TRUE(producer->bind(fixture.inputs).ok());
  auto runtime = ad::ContextAccess::runtime(*context);
  for (int repeat = 0; repeat < (close_pending ? 1 : 3); ++repeat) {
    std::optional<FifoGate> gate;
    if (close_pending) {
      gate.emplace();
      ASSERT_TRUE(gate->enqueue(runtime).ok());
      gate->wait_entered();
    }
    auto produced = producer->submit(), consumed = consumer->submit();
    ASSERT_TRUE(produced) << produced.error().message();
    ASSERT_TRUE(consumed) << consumed.error().message();
    if (close_pending) {
      hyle::Status closed;
      std::thread closer([&] { closed = context->close(); });
      // The original owner is gated. Close must reject new public admission
      // and still accept the actual GPU tail of these already queued Jobs.
      for (;;) {
        bool closing;
        { std::lock_guard lock(runtime->queue_mutex); closing = !runtime->accepting; }
        if (closing) break;
        std::this_thread::yield();
      }
      gate->open_and_join();
      closer.join();
      ASSERT_TRUE(closed.ok()) << closed.message();
    }
    ASSERT_TRUE(consumed->completion().wait().ok());
    ASSERT_TRUE(produced->completion().wait().ok());
    auto output = ad::InvocationAccess::output(*consumer, 0);
    ASSERT_TRUE(output) << output.error().message();
    ASSERT_EQ(output->size(), expected[0].size());
    for (std::size_t index = 0; index < expected[0].size(); ++index)
      EXPECT_NEAR(output->data<float>()[index], 2 * expected[0][index], 1.e-5);
    {
      std::lock_guard lock(runtime->queue_mutex);
      EXPECT_GT(runtime->native_tail_peak, 0u);
      EXPECT_EQ(runtime->native_tails, 0u);
    }
  }
}

TEST(AppleLateInput, NativeSignalFeedsMlxAndReusesTheOriginalProducer) {
  exercise_native_signal_source(false);
}
TEST(AppleLateInput, CloseDrainsQueuedNativeSignalsBeforeStoppingTheOwner) {
  exercise_native_signal_source(true);
}

void exercise_mixed_plan(bool pending_chain, bool close_pending = false, bool cancel_pending = false, std::size_t order_count = 0) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  apple_test::Recurrent fixture(3, 2, 64);
  const auto expected = fixture.oracle(2);
  auto specs = fixture.specs();
  ad::RegionPlan plan;
  plan.values = specs;
  plan.inputs = {0, 1, 2, 3, 4, 5};
  plan.values.push_back(specs[0]); // MLX query scaling.
  plan.values.push_back(specs[2]); // Original native scores.
  plan.values.push_back(specs[5]); // Complete native state.
  plan.values.push_back(specs[2]); // Final MLX scores.
  plan.outputs = {9, 8};
  ad::Region first;
  first.inputs = {0}; first.outputs = {6};
  first.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
  };
  ad::Region native;
  native.kind = ad::RegionKind::kGgml;
  native.inputs = {6, 1, 2, 3, 4, 5}; native.outputs = {7, 8};
  native.operation = c::Operation(c::GatedDeltaNet{});
  for (const auto &input : fixture.inputs) {
    std::vector<std::int64_t> shape(input.shape().begin(), input.shape().end());
    native.native_inputs.push_back(c::TensorDesc::dense(c::DataType::kFloat32, shape).value());
  }
  ad::Region last;
  last.inputs = {7}; last.outputs = {9};
  last.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(3.f), stream)};
  };
  plan.regions = {std::move(first), std::move(native), std::move(last)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  std::optional<a::Invocation> consumer;
  auto runtime = ad::ContextAccess::runtime(*context);
  apple_test::Recurrent following(3, 2, 64);
  for (std::size_t i = 0; i < expected[0].size(); ++i)
    following.data[0][i] = float(3 * expected[0][i]);
  const auto next_expected = following.oracle(2);
  if (pending_chain) {
    auto next = program->create_invocation();
    ASSERT_TRUE(next) << next.error().message();
    consumer.emplace(std::move(*next));
    if (order_count) {
      ASSERT_TRUE(consumer->bind(fixture.inputs).ok());
      ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, order_count).ok());
    }
    std::vector<ad::InputBinding> bindings(fixture.inputs.size());
    auto source = ad::InvocationAccess::source(*invocation, 0);
    ASSERT_TRUE(source);
    bindings[0].source = std::move(*source);
    for (std::size_t i = 1; i < bindings.size(); ++i) bindings[i].value = fixture.inputs[i];
    if (!order_count) ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, bindings).ok());
  }
  std::vector<a::Invocation> preceding;
  for (std::size_t i = 1; i < order_count; ++i) {
    auto slot = program->create_invocation(); ASSERT_TRUE(slot);
    ASSERT_TRUE(slot->bind(fixture.inputs).ok()); preceding.push_back(std::move(*slot));
  }
  for (int repeat = 0; repeat < (close_pending || cancel_pending ? 1 : 3); ++repeat) {
    ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
    std::optional<GpuEventGate> gpu_gate;
    if (pending_chain) {
      gpu_gate.emplace(runtime);
      ASSERT_TRUE(gpu_gate->enqueue().ok());
    }
    auto result = invocation->submit();
    ASSERT_TRUE(result) << result.error().message();
    if (pending_chain) {
      std::vector<c::OwnedResult<a::Results>> preceding_results;
      preceding_results.reserve(preceding.size());
      std::vector<c::detail::CompletionDependency> dependencies;
      if (order_count) {
        dependencies.push_back(c::detail::CompletionAccess::dependency(result->completion()));
        for (auto &slot : preceding) {
          auto work = slot.submit(); ASSERT_TRUE(work);
          preceding_results.push_back(std::move(*work));
          dependencies.push_back(c::detail::CompletionAccess::dependency(preceding_results.back().completion()));
        }
      }
      auto next = order_count
          ? ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies)
          : consumer->submit();
      ASSERT_TRUE(next) << next.error().message();
      FifoGate owner_gate;
      ASSERT_TRUE(owner_gate.enqueue(runtime).ok());
      const bool owner_free = owner_gate.wait_entered_for(std::chrono::seconds(2));
      EXPECT_TRUE(owner_free) << "Mixed Plans waited on the host before encoding the next Job";
      EXPECT_EQ(result->completion().poll().value(), c::CompletionPoll::kPending);
      EXPECT_EQ(next->completion().poll().value(), c::CompletionPoll::kPending);
      if (cancel_pending) EXPECT_TRUE(result->request_cancel().ok());
      // Release the real GPU before any blocking assertion/close path.
      gpu_gate->open();
      if (close_pending && owner_free) {
        hyle::Status closed;
        std::thread closer([&] { closed = context->close(); });
        for (;;) {
          bool closing;
          { std::lock_guard lock(runtime->queue_mutex); closing = !runtime->accepting; }
          if (closing) break;
          std::this_thread::yield();
        }
        owner_gate.open_and_join();
        closer.join();
        ASSERT_TRUE(closed.ok()) << closed.message();
      } else owner_gate.open_and_join();
      const auto child_status = next->completion().wait();
      if (cancel_pending) {
        EXPECT_TRUE(child_status.IsCancelled()) << child_status.message();
        EXPECT_TRUE(result->completion().wait().IsCancelled());
        EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
        EXPECT_FALSE(ad::InvocationAccess::output(*consumer, 0));
        ASSERT_TRUE(context->close().ok());
        return;
      }
      ASSERT_TRUE(child_status.ok()) << child_status.message();
      auto next_scores = ad::InvocationAccess::output(*consumer, 0);
      auto next_state = ad::InvocationAccess::output(*consumer, 1);
      ASSERT_TRUE(next_scores); ASSERT_TRUE(next_state);
      const auto &expected_next = order_count ? expected : next_expected;
      for (std::size_t i = 0; i < expected_next[0].size(); ++i)
        EXPECT_NEAR(next_scores->data<float>()[i], 3 * expected_next[0][i], 1.e-5);
      for (std::size_t i = 0; i < expected_next[1].size(); ++i)
        EXPECT_NEAR(next_state->data<float>()[i], expected_next[1][i], 1.e-5);
      for (auto &work : preceding_results) ASSERT_TRUE(work.completion().wait().ok());
    }
    ASSERT_TRUE(result->completion().wait().ok());
    auto scores = ad::InvocationAccess::output(*invocation, 0);
    auto state = ad::InvocationAccess::output(*invocation, 1);
    ASSERT_TRUE(scores); ASSERT_TRUE(state);
    for (std::size_t i = 0; i < expected[0].size(); ++i)
      EXPECT_NEAR(scores->data<float>()[i], 3 * expected[0][i], 1.e-5);
    for (std::size_t i = 0; i < expected[1].size(); ++i)
      EXPECT_NEAR(state->data<float>()[i], expected[1][i], 1.e-5);
  }
  ASSERT_TRUE(context->close().ok());
}

TEST(AppleLateInput, MixedMlxNativeMlxPlanPreservesValuesAndReusesItsSlot) {
  exercise_mixed_plan(false);
}
TEST(AppleLateInput, MixedPlansEncodePendingSourcesWithoutBlockingTheirOwner) {
  exercise_mixed_plan(true);
}
TEST(AppleLateInput, CloseDrainsPendingMixedPlanSourcesOnTheirOriginalOwner) {
  exercise_mixed_plan(true, true);
}

TEST(AppleLateInput, CancelDrainsPendingMixedPlanAndRejectsDependentPublication) {
  exercise_mixed_plan(true, false, true);
}

TEST(AppleLateInput, NativeOrderEdgesKeepIndependentMathematicalInputs) {
  exercise_mixed_plan(true, false, false, 1);
}
TEST(AppleLateInput, NativeOrderFanInExceedsTheTensorInputPortCount) {
  exercise_mixed_plan(true, false, false, 9);
}
TEST(AppleLateInput, CloseDrainsOriginalOrderOnlyProducerAndConsumer) {
  exercise_mixed_plan(true, true, false, 1);
}
TEST(AppleLateInput, CancelledOrderOnlyProducerRejectsIndependentOutput) {
  exercise_mixed_plan(true, false, true, 1);
}

void exercise_multiple_native_prefixes(bool order_only) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const auto x = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 64}).value();
  const auto w = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}).value();
  ad::RegionPlan plan;
  plan.values = {{{1, 64}, mx::float32}, {{64, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 64}, mx::float32}};
  plan.inputs = {0, 1}; plan.outputs = {5, 2};
  ad::Region first, middle, last, output;
  first.kind = last.kind = ad::RegionKind::kGgml;
  first.operation = last.operation = c::model_ops::dense_linear_f32();
  first.native_inputs = last.native_inputs = {x, w};
  first.inputs = {0, 1}; first.outputs = {2};
  middle.inputs = {2}; middle.outputs = {3};
  middle.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(3.f), stream)};
  };
  last.inputs = {3, 1}; last.outputs = {4};
  output.inputs = {4}; output.outputs = {5};
  output.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(.5f), stream)};
  };
  plan.regions = {std::move(first), std::move(middle), std::move(last), std::move(output)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  std::optional<a::Invocation> consumer;
  if (order_only) {
    auto next = program->create_invocation(); ASSERT_TRUE(next);
    consumer = std::move(*next);
    ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, 1).ok());
  }
  std::array<float, 64> input;
  std::array<float, 64 * 64> weights{};
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(i + 1); weights[i * 64 + i] = 2.f;
  }
  const std::array inputs{mx::array(input.data(), {1, 64}), mx::array(weights.data(), {64, 64})};
  auto runtime = ad::ContextAccess::runtime(*context);
  // Exercise ready execution before the repeated pending-source episodes.
  ASSERT_TRUE(invocation->bind(inputs).ok());
  { auto warm = invocation->submit(); ASSERT_TRUE(warm); ASSERT_TRUE(warm->completion().wait().ok()); }
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE(invocation->bind(inputs).ok());
    if (consumer) ASSERT_TRUE(consumer->bind(inputs).ok());
    GpuEventGate gpu_gate(runtime);
    ASSERT_TRUE(gpu_gate.enqueue().ok());
    auto result = invocation->submit();
    ASSERT_TRUE(result);
    std::optional<c::OwnedResult<a::Results>> dependent;
    if (consumer) {
      const std::array dependencies{c::detail::CompletionAccess::dependency(result->completion())};
      auto submitted = ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies);
      ASSERT_TRUE(submitted); dependent = std::move(*submitted);
    }
    FifoGate owner_gate;
    ASSERT_TRUE(owner_gate.enqueue(runtime).ok());
    const bool owner_free = owner_gate.wait_entered_for(std::chrono::seconds(2));
    auto &completion = dependent ? dependent->completion() : result->completion();
    const auto pending = completion.poll();
    gpu_gate.open();
    owner_gate.open_and_join();
    EXPECT_TRUE(owner_free) << "The first prefix cannot retire the still-pending complete Plan";
    ASSERT_TRUE(pending) << pending.error().message(); EXPECT_EQ(*pending, c::CompletionPoll::kPending);
    ASSERT_TRUE(result->completion().wait().ok());
    ASSERT_TRUE(completion.wait().ok());
    auto &observed = consumer ? *consumer : *invocation;
    auto final = ad::InvocationAccess::output(observed, 0);
    auto early = ad::InvocationAccess::output(observed, 1);
    ASSERT_TRUE(final); ASSERT_TRUE(early);
    for (std::size_t i = 0; i < input.size(); ++i) {
      EXPECT_FLOAT_EQ(final->data<float>()[i], 6 * input[i]);
      EXPECT_FLOAT_EQ(early->data<float>()[i], 2 * input[i]);
    }
    std::lock_guard lock(runtime->queue_mutex);
    EXPECT_EQ(runtime->native_tails, 0u);
  }
  ASSERT_TRUE(context->close().ok());
}
TEST(AppleLateInput, MultipleNativePrefixesRetireOnTheLastOriginalCallback) {
  exercise_multiple_native_prefixes(false);
}
TEST(AppleLateInput, MultipleNativeConsumerPrefixesKeepBothOriginalQueueFrontiers) {
  exercise_multiple_native_prefixes(true);
}

void exercise_deferred_input_error(bool order_only) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array<std::int64_t, 2> table_shape{4, 64}, dot_shape{1, 64};
  const std::array<std::int64_t, 2> weight_shape{2, 64};
  const std::array<std::int64_t, 1> index_shape{1};
  const std::array descriptions{
      c::TensorDesc::dense(c::DataType::kFloat32, table_shape).value(),
      c::TensorDesc::dense(c::DataType::kInt32, index_shape).value()};
  ad::RegionPlan plan;
  plan.values = {{{4, 64}, mx::float32}, {{1}, mx::int32}, {{2, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 2}, mx::float32}};
  plan.inputs = {0, 1, 2}; plan.outputs = {4};
  ad::Region checked;
  checked.inputs = {0, 1}; checked.outputs = {3};
  checked.mlx = [descriptions](const auto &args, mx::Stream stream) {
    return ad::lower_mlx_float(c::Operation(c::Gather{}), descriptions, args, stream).value();
  };
  ad::Region native;
  native.kind = ad::RegionKind::kGgml;
  native.inputs = {3, 2}; native.outputs = {4};
  native.operation = c::model_ops::dense_linear_f32();
  native.native_inputs = {
      c::TensorDesc::dense(c::DataType::kFloat32, dot_shape).value(),
      c::TensorDesc::dense(c::DataType::kFloat32, weight_shape).value()};
  plan.regions = {std::move(checked), std::move(native)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  std::array<float, 256> table_values; table_values.fill(1.f);
  std::array<float, 128> weight_values; weight_values.fill(3.f / 64);
  const mx::array table(table_values.data(), {4, 64}), weight(weight_values.data(), {2, 64});
  const std::array source_specs{a::InputSpec{{1, 2}, mx::float32}};
  auto next_program = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, source_specs);
  ASSERT_TRUE(next_program);
  for (const auto index : {8, 0}) {
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();
    const std::array inputs{table, mx::array({index}, mx::int32), weight};
    ASSERT_TRUE(invocation->bind(inputs).ok());
    auto consumer = next_program->create_invocation();
    ASSERT_TRUE(consumer);
    if (order_only) {
      ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, 1).ok());
      ASSERT_TRUE(consumer->bind(std::array{mx::array({2.f, 3.f}, {1, 2})}).ok());
    } else {
      std::array<ad::InputBinding, 1> late;
      late[0].source = ad::InvocationAccess::source(*invocation, 0).value();
      ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
    }
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::ContextAccess::runtime(*context)).ok());
    gate.wait_entered();
    auto result = invocation->submit();
    ASSERT_TRUE(result);
    const std::array dependencies{c::detail::CompletionAccess::dependency(result->completion())};
    auto consumed = order_only
        ? ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies)
        : consumer->submit();
    ASSERT_TRUE(consumed);
    gate.open_and_join();
    const auto child_status = consumed->completion().wait();
    const auto status = result->completion().wait();
    if (index == 8) {
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_TRUE(child_status.IsOutOfRange()) << child_status.message();
      EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
      EXPECT_FALSE(ad::InvocationAccess::output(*consumer, 0));
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      auto output = ad::InvocationAccess::output(*invocation, 0);
      ASSERT_TRUE(output);
      EXPECT_FLOAT_EQ(output->data<float>()[0], 3.f);
      EXPECT_FLOAT_EQ(output->data<float>()[1], 3.f);
      ASSERT_TRUE(child_status.ok()) << child_status.message();
      auto child = ad::InvocationAccess::output(*consumer, 0);
      ASSERT_TRUE(child);
      EXPECT_FLOAT_EQ(child->data<float>()[0], order_only ? 4.f : 6.f);
      EXPECT_FLOAT_EQ(child->data<float>()[1], 6.f);
    }
  }
  ASSERT_TRUE(context->close().ok());
}
TEST(AppleLateInput, DeferredMlxInputErrorPreventsPublicationAndKeepsTheRuntimeUsable) {
  exercise_deferred_input_error(false);
}
TEST(AppleLateInput, OrderOnlyErrorPreventsIndependentConsumerPublication) {
  exercise_deferred_input_error(true);
}

TEST(AppleLateInput, CopyAndMathPinOriginalOrderOnlyEpisodesAfterPublicHandleAbandonment) {
  namespace m = hyle::mem;
  auto context = a::Context::create().value();
  const auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = context.prepare(c::Unary{c::UnaryKind::kNeg}, specs).value();
  auto producer = program.create_invocation().value(), consumer = program.create_invocation().value();
  ASSERT_TRUE(producer.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f})}).ok());
  ASSERT_TRUE(consumer.bind(std::array{mx::array({5.f, 6.f, 7.f, 8.f})}).ok());
  ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(consumer, 1).ok());
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto memory = a::MemoryResource::create(context).value();
  auto source = m::Buffer::create(memory.ref(), 16, alignof(float)).value();
  auto target = m::Buffer::create(m::system_resource(), 16, alignof(float)).value();
  const std::array<float, 4> values{10.f, 20.f, 30.f, 40.f};
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, m::system_allocator(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::reserve_copy_dependencies(*copy, 1).ok());
  ASSERT_TRUE(ad::bind_host_copy_recording(*copy,
      c::ConstTensorView::create(description, source.view()).value(), target.view(), {}).ok());
  FifoGate gate;
  ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
  auto produced = producer.submit().value();
  const std::array before_copy{c::detail::CompletionAccess::dependency(produced.completion())};
  auto copied = ad::submit_copy_recording(*copy, before_copy).value();
  const std::array before_math{c::detail::CompletionAccess::dependency(copied)};
  auto consumed = ad::InvocationAccess::submit_ordered(consumer, {}, nullptr, nullptr, before_math);
  ASSERT_TRUE(consumed) << consumed.error().message();
  produced = {}; copied = {};
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
  EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).IsUnavailable());
  auto replay = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(replay); EXPECT_TRUE(replay.error().IsFailedPrecondition());
  gate.open_and_join();
  ASSERT_TRUE(consumed->completion().wait().ok());
  std::array<float, 4> observed{};
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, values);
  const auto output = ad::InvocationAccess::output(consumer, 0).value();
  EXPECT_FLOAT_EQ(output.data<float>()[0], -5.f);
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).ok());
  EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).ok());
  ASSERT_TRUE(context.close().ok());
}

TEST(AppleLateInput, EmptyCopyStillPublishesItsOriginalOrderDependencyFailure) {
  auto context = a::Context::create().value();
  auto program = context.prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}}).value();
  auto producer = program.create_invocation().value();
  ASSERT_TRUE(producer.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}), mx::array({9}, mx::int32)}).ok());
  const auto runtime = ad::ContextAccess::runtime(context);
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, hyle::mem::system_allocator(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::reserve_copy_dependencies(*copy, 1).ok());
  ASSERT_TRUE(ad::bind_copy_recording(*copy, {}).ok());
  FifoGate gate;
  ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
  auto produced = producer.submit().value();
  const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};
  auto copied = ad::submit_copy_recording(*copy, dependencies);
  ASSERT_TRUE(copied) << copied.error().message();
  EXPECT_EQ(copied->poll().value(), c::CompletionPoll::kPending);
  gate.open_and_join();
  EXPECT_TRUE(copied->wait().IsOutOfRange());
  EXPECT_TRUE(produced.completion().wait().IsOutOfRange());
  ASSERT_TRUE(context.close().ok());
}

TEST(AppleLateInput, RetiredCopyInputReleaseKeepsNativeReaderEpochAndVerdict) {
  for (const bool fail : {false,true}) {
    auto context=a::Context::create().value();
    auto runtime=ad::ContextAccess::runtime(context);
    auto program=context.prepare(c::Gather{},
        std::array{a::InputSpec{{4},mx::float32},a::InputSpec{{1},mx::int32}}).value();
    auto producer=program.create_invocation().value();
    ASSERT_TRUE(producer.bind(std::array{mx::array({1.f,2.f,3.f,4.f}),mx::array({fail?9:0},mx::int32)}).ok());
    auto memory=a::MemoryResource::create(context).value();
    const auto description=c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t,1>{1}).value();
    auto source=hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value();
    auto target=hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value();
    const std::array<float,1> input{17.f},sentinel{-7.f};
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(input))).ok());
    ASSERT_TRUE(target.view().write(std::as_bytes(std::span(sentinel))).ok());
    auto copy=ad::prepare_copy_recording_runtime(runtime,1,hyle::mem::system_allocator(),sizeof(float),
        ad::CopyRecordingMode::kReplayable).value();
    ASSERT_TRUE(ad::reserve_copy_dependencies(*copy,1).ok());
    ASSERT_TRUE(ad::bind_host_copy_recording(*copy,
        c::ConstTensorView::create(description,source.view()).value(),target.view(),{}).ok());
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());gate.wait_entered();
    auto produced=producer.submit().value();
    const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};
    auto copied=ad::submit_copy_recording(*copy,dependencies).value();
    std::uint64_t epoch=0;
    auto pinned=ad::pin_copy_dependency(runtime,c::detail::CompletionAccess::dependency(copied),&epoch).value();
    gate.open_and_join();
    const auto terminal=copied.wait();
    EXPECT_EQ(terminal.ok(),!fail);
    EXPECT_EQ(produced.completion().wait().code(),terminal.code());
    EXPECT_TRUE(ad::copy_recording_retired_gate(*copy).ok());
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    ASSERT_TRUE(ad::clear_copy_recording(*copy).ok());
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    struct Observation final:ad::Job {
      ad::CopyRecording *copy;
      std::uint64_t epoch;
      hyle::Status terminal;
      std::atomic<bool> done{false};
      Observation(ad::CopyRecording &selected,std::uint64_t generation):copy(&selected),epoch(generation) {
        execute=[](ad::Job *job) noexcept {
          auto &call=*static_cast<Observation *>(job);
          call.terminal=ad::observe_copy_dependency(*call.copy,call.epoch);
          call.done.store(true,std::memory_order_release);call.done.notify_all();
        };
      }
    } observation(*copy,epoch);
    ASSERT_TRUE(runtime->enqueue(observation).ok());
    observation.done.wait(false,std::memory_order_acquire);
    EXPECT_EQ(observation.terminal.code(),terminal.code());
    EXPECT_EQ(observation.terminal.message(),terminal.message());
    ad::release_copy_dependency(*pinned);pinned.reset();
    EXPECT_EQ(ad::copy_recording_reuse_gate(*copy).code(),terminal.code());
    std::array<float,1> actual{};
    ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual,fail?sentinel:input);
    ASSERT_TRUE(context.close().ok());
  }
}

void exercise_pending_gpu_copy(unsigned mode) {
  namespace m=hyle::mem;
  auto context=a::Context::create().value();
  const auto runtime=ad::ContextAccess::runtime(context);
  auto memory=a::MemoryResource::create(context).value();
  apple_test::Recurrent fixture(3,2,64);
  const auto expected=fixture.oracle();
  auto program=context.prepare(c::GatedDeltaNet{},fixture.specs()).value();
  auto consumer=program.create_invocation().value();
  ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(consumer,1).ok());
  const auto bytes=fixture.inputs.front().nbytes();
  auto source=m::Buffer::create(memory.ref(),bytes,256).value();
  auto intermediate=m::Buffer::create(memory.ref(),bytes,256).value();
  auto target=m::Buffer::create(memory.ref(),bytes,256).value();
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(
      fixture.inputs.front().data<float>(),fixture.inputs.front().size()))).ok());
  const std::vector<std::int64_t> dimensions(fixture.inputs.front().shape().begin(),fixture.inputs.front().shape().end());
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,dimensions).value();
  auto inputs=fixture.inputs;
  inputs.front()=ad::MemoryAccess::borrow_tensor(c::ConstTensorView::create(description,
      mode==4?intermediate.view():target.view()).value()).value();
  m::QuotaAllocator metadata(m::system_allocator());
  auto first=ad::prepare_copy_recording_runtime(runtime,1,metadata.get(),bytes,ad::CopyRecordingMode::kReplayable).value();
  auto second=ad::prepare_copy_recording_runtime(runtime,1,metadata.get(),bytes,ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::reserve_copy_dependencies(*second,1).ok());
  const std::array<c::detail::CopyBackendRange,1> to_intermediate{{{intermediate.view(),source.view()}}};
  const std::array<c::detail::CopyBackendRange,1> to_target{{{target.view(),intermediate.view()}}};
  ASSERT_TRUE(ad::bind_copy_recording(*first,to_intermediate).ok());
  ASSERT_TRUE(ad::bind_copy_recording(*second,mode==4?
      std::span<const c::detail::CopyBackendRange>{}:std::span<const c::detail::CopyBackendRange>{to_target}).ok());
  const auto cold_metadata=metadata.used();
  for (unsigned replay=0;replay<(mode==0?2u:1u);++replay) {
    ASSERT_TRUE(consumer.bind(inputs).ok());
    GpuEventGate gpu(runtime);
    ASSERT_TRUE(gpu.enqueue().ok());
    auto produced=ad::submit_copy_recording(*first).value();
    const std::array before_copy{c::detail::CompletionAccess::dependency(produced)};
    auto copied=ad::submit_copy_recording(*second,before_copy).value();
    const std::array before_math{c::detail::CompletionAccess::dependency(copied)};
    auto consumed=ad::InvocationAccess::submit_ordered(consumer,{},nullptr,nullptr,before_math).value();
    FifoGate witness;
    ASSERT_TRUE(witness.enqueue(runtime).ok());
    const bool encoded=witness.wait_entered_for(std::chrono::seconds(2));
    if (!encoded) gpu.open();
    if (encoded) {
      EXPECT_EQ(produced.poll().value(),c::CompletionPoll::kPending);
      EXPECT_EQ(copied.poll().value(),c::CompletionPoll::kPending);
      EXPECT_EQ(consumed.completion().poll().value(),c::CompletionPoll::kPending);
      EXPECT_TRUE(ad::copy_recording_reuse_gate(*first).IsUnavailable());
      EXPECT_TRUE(ad::copy_recording_reuse_gate(*second).IsUnavailable());
      EXPECT_TRUE(ad::observe_copy_recording_on_owner(*first).IsUnavailable());
    }
    witness.open_and_join();
    ASSERT_TRUE(encoded) << "original FIFO waited for a GPU copy instead of importing its real event";
    if (mode==1) { produced={};copied={}; }
    if (mode==2) EXPECT_TRUE(consumed.request_cancel().ok());
    std::thread closer;
    hyle::Status closed;
    std::atomic<bool> close_returned{false};
    if (mode==3) {
      closer=std::thread([&]{closed=context.close();close_returned.store(true,std::memory_order_release);});
      bool closing=false;
      const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(2);
      do {
        { std::lock_guard lock(runtime->queue_mutex); closing=!runtime->accepting; }
        if(!closing) std::this_thread::yield();
      } while(!closing && std::chrono::steady_clock::now()<limit);
      EXPECT_TRUE(closing);
      EXPECT_FALSE(close_returned.load(std::memory_order_acquire));
    }
    gpu.open();
    const auto terminal=consumed.completion().wait();
    if (mode!=1) {
      EXPECT_TRUE(produced.wait().ok());
      EXPECT_TRUE(copied.wait().ok());
    }
    if (closer.joinable()) closer.join();
    EXPECT_TRUE(mode==2?terminal.IsCancelled():terminal.ok()) << terminal.message();
    EXPECT_EQ(metadata.used(),cold_metadata);
    EXPECT_EQ(ad::copy_recording_report(*first).host_waits,0u);
    EXPECT_EQ(ad::copy_recording_report(*second).host_waits,0u);
    if(mode==4) {
      EXPECT_EQ(ad::copy_recording_report(*second).native_calls,0u);
      EXPECT_EQ(ad::copy_recording_report(*second).phase_completions,1u);
    }
    if (mode!=2) {
      auto actual=ad::InvocationAccess::output(consumer,0).value();
      ASSERT_EQ(actual.size(),expected[0].size());
      for(std::size_t i=0;i<actual.size();++i) EXPECT_NEAR(actual.data<float>()[i],expected[0][i],1.e-5);
    } else EXPECT_FALSE(ad::InvocationAccess::output(consumer,0));
    if (mode==3) EXPECT_TRUE(closed.ok()) << closed.message();
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*first).ok());
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*second).ok());
  EXPECT_TRUE(ad::observe_copy_recording_on_owner(*first).ok());
  { std::lock_guard lock(runtime->queue_mutex); EXPECT_EQ(runtime->native_tails,0u); }
}
TEST(AppleLateInput, PendingGpuBlitsFeedTheOriginalNativeQueueWithoutBlockingTheirOwner) {
  exercise_pending_gpu_copy(0);
}
TEST(AppleLateInput, PendingGpuCopyChainSurvivesPublicCompletionAbandonment) {
  exercise_pending_gpu_copy(1);
}
TEST(AppleLateInput, PendingGpuCopyReaderCancelKeepsTheOriginalCopyVerdicts) {
  exercise_pending_gpu_copy(2);
}
TEST(AppleLateInput, CloseDrainsTheOriginalGpuCopyCallbacksAndNativeConsumer) {
  exercise_pending_gpu_copy(3);
}
TEST(AppleLateInput, EmptyGpuCopyPhaseImportsItsOriginalPendingOrderWithoutOwnerDrain) {
  exercise_pending_gpu_copy(4);
}

// Inject only at actual eval, after cold compilation and the native prefix.
class LateEncodingFailure final : public mx::Primitive {
public:
  LateEncodingFailure(mx::Stream stream, std::shared_ptr<std::atomic<bool>> fail)
      : Primitive(stream), fail_(std::move(fail)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::runtime_error("GPU encoding checkpoint cannot use CPU");
  }
  void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override {
    if (fail_->load()) throw std::runtime_error("injected late mixed-Plan encoding failure");
    outputs[0].copy_shared_buffer(inputs[0]);
  }
  const char *name() const override { return "HyleLatePlanEncodingFailure"; }
private:
  std::shared_ptr<std::atomic<bool>> fail_;
};

TEST(AppleLateInput, EarlierDeviceCheckPrecedesLaterEncodingFailure) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array<std::int64_t, 2> table_shape{4, 64}, dot_shape{1, 64};
  const std::array<std::int64_t, 2> weight_shape{2, 64};
  const std::array<std::int64_t, 1> index_shape{1};
  const std::array descriptions{
      c::TensorDesc::dense(c::DataType::kFloat32, table_shape).value(),
      c::TensorDesc::dense(c::DataType::kInt32, index_shape).value()};
  auto fail = std::make_shared<std::atomic<bool>>(false);
  ad::RegionPlan plan;
  plan.values = {{{4, 64}, mx::float32}, {{1}, mx::int32}, {{2, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 2}, mx::float32}};
  plan.inputs = {0, 1, 2}; plan.outputs = {4};
  ad::Region checked;
  checked.inputs = {0, 1}; checked.outputs = {3};
  checked.mlx = [descriptions](const auto &args, mx::Stream stream) {
    return ad::lower_mlx_float(c::Operation(c::Gather{}), descriptions, args, stream).value();
  };
  ad::Region native;
  native.kind = ad::RegionKind::kGgml;
  native.inputs = {3, 2}; native.outputs = {4};
  native.operation = c::model_ops::dense_linear_f32();
  native.native_inputs = {
      c::TensorDesc::dense(c::DataType::kFloat32, dot_shape).value(),
      c::TensorDesc::dense(c::DataType::kFloat32, weight_shape).value()};
  plan.values.push_back({{1, 2}, mx::float32}); plan.outputs = {5};
  ad::Region finish;
  finish.inputs = {4}; finish.outputs = {5};
  finish.mlx = [fail](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::array(args[0].shape(), args[0].dtype(),
        std::make_shared<LateEncodingFailure>(stream, fail), {args[0]})};
  };
  plan.regions = {std::move(checked), std::move(native), std::move(finish)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  std::array<float, 256> table_values; table_values.fill(1.f);
  std::array<float, 128> weight_values; weight_values.fill(3.f / 64);
  const mx::array table(table_values.data(), {4, 64}), weight(weight_values.data(), {2, 64});
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  const std::array inputs{table, mx::array({8}, mx::int32), weight};
  ASSERT_TRUE(invocation->bind(inputs).ok());
  fail->store(true);
  auto result = invocation->submit();
  ASSERT_TRUE(result);
  const auto status = result->completion().wait();
  EXPECT_TRUE(status.IsOutOfRange()) << status.message();
  EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
  { std::lock_guard lock(ad::ContextAccess::runtime(*context)->queue_mutex);
    EXPECT_EQ(ad::ContextAccess::runtime(*context)->native_tails, 0u); }
  context->close().IgnoreError(); // The actual encoding exception poisons this Runtime.
}

void exercise_retired_binding_detach(bool alias) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  auto runtime = ad::ContextAccess::runtime(*context);
  auto memory = a::MemoryResource::create(*context);
  ASSERT_TRUE(memory);
  auto pool = c::LeasePool::create(memory->ref(), 16, {.max_blocks = 1});
  ASSERT_TRUE(pool);
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{4}).value();
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto first = ad::ContextAccess::compile(runtime,
      [alias](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{alias ? args[0] : mx::add(args[0], mx::array(1.f), stream)};
      }, specs);
  auto second = ad::ContextAccess::compile(runtime,
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, specs);
  ASSERT_TRUE(first); ASSERT_TRUE(second);
  auto producer = first->create_invocation(), consumer = second->create_invocation();
  ASSERT_TRUE(producer); ASSERT_TRUE(consumer);
  auto source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  auto dependency = ad::InvocationAccess::source_dependency(*source, runtime);
  ASSERT_TRUE(dependency);
  std::array<ad::InputBinding, 1> late;
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset();
  {
    auto writer = pool->allocate(16);
    ASSERT_TRUE(writer);
    const std::array<float, 4> initial{1, 2, 3, 4};
    ASSERT_TRUE(writer->view()->write(std::as_bytes(std::span(initial))).ok());
    auto ready = writer->publish();
    ASSERT_TRUE(ready);
    auto array = memory->binding(*ready, desc);
    ASSERT_TRUE(array);
    const std::array values{*array};
    ASSERT_TRUE(producer->bind(values).ok());
  } // Only original native arrays now keep the aliased allocation alive.
  c::detail::ResultTransfer retired;
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
    auto produced = producer->submit();
    ASSERT_TRUE(produced);
    EXPECT_TRUE(ad::InvocationAccess::can_detach_inputs(*producer, {}).IsUnavailable());
    gate.open_and_join();
    ASSERT_TRUE(produced->completion().wait().ok());
    ASSERT_TRUE(dependency->terminal_hint->ok()) << dependency->terminal_hint->message();
    retired = c::detail::ResultAccess::take_storage(std::move(*produced));
  }
  // An escaped result is still a real observer, never discounted as private.
  EXPECT_TRUE(ad::InvocationAccess::can_detach_inputs(*producer, {}).IsFailedPrecondition());
  auto escaped = retired.owner;
  EXPECT_TRUE(ad::InvocationAccess::can_detach_inputs(*producer, retired.owner).IsFailedPrecondition());
  escaped.reset();
  const float *original_output=nullptr;
  {
    auto output=ad::InvocationAccess::output(*producer,0);
    ASSERT_TRUE(output)<<output.error().message();
    original_output=output->data<float>();
  } // Keep only its address; the test must not retain the output allocation.
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
    auto pending = consumer->submit();
    ASSERT_TRUE(pending);
    ASSERT_TRUE(ad::InvocationAccess::can_detach_inputs(*producer, retired.owner).ok());
    retired.owner.reset();
    ad::InvocationAccess::detach_inputs(*producer);
    const std::array changed{mx::array({5.f, 6.f, 7.f, 8.f})};
    EXPECT_TRUE(producer->bind(changed).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(*producer).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::release_inputs(*producer).IsUnavailable());
    if(alias)EXPECT_EQ(pool->stats().live_blocks, 1u);
    // A generated output need not retain its retired input allocation. Prove
    // that the consumer's pin keeps the original output itself, not that DAG
    // detachment happens at one incidental time in the MLX allocator.
    {
      auto output=ad::InvocationAccess::output(*producer,0);
      ASSERT_TRUE(output)<<output.error().message();
      EXPECT_EQ(output->data<float>(),original_output);
    }
    EXPECT_FALSE(producer->submit());
    gate.open_and_join();
    ASSERT_TRUE(pending->completion().wait().ok());
    {
      auto output = ad::InvocationAccess::output(*consumer, 0);
      ASSERT_TRUE(output) << output.error().message();
      for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(output->data<float>()[i], 2.f * (i + (alias ? 1.f : 2.f)));
    }
    // An evaluated consumer result can still own its original input DAG.
    // Return that consumer's own private result and bindings as well.
    auto consumed = c::detail::ResultAccess::take_storage(std::move(*pending));
    ASSERT_TRUE(ad::InvocationAccess::can_detach_inputs(*consumer, consumed.owner).ok());
    consumed.owner.reset();
    ad::InvocationAccess::detach_inputs(*consumer);
  }
  ASSERT_TRUE(ad::InvocationAccess::reuse_gate(*producer).ok());
  source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset();
  const std::array changed{mx::array({5.f, 6.f, 7.f, 8.f})};
  ASSERT_TRUE(producer->bind(changed).ok());
  auto produced = producer->submit();
  ASSERT_TRUE(produced);
  ASSERT_TRUE(produced->completion().wait().ok());
  ASSERT_TRUE(dependency->terminal_hint->ok()) << dependency->terminal_hint->message();
  auto consumed = consumer->submit();
  ASSERT_TRUE(consumed);
  ASSERT_TRUE(consumed->completion().wait().ok());
  auto output = ad::InvocationAccess::output(*consumer, 0);
  ASSERT_TRUE(output);
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(output->data<float>()[i], 2.f * (i + (alias ? 5.f : 6.f)));
}

TEST(AppleLateInput, DetachRetiredBindingsPreservesPinnedGeneratedOutput) {
  exercise_retired_binding_detach(false);
}
TEST(AppleLateInput, DetachRetiredBindingsPreservesPinnedAliasedOutput) {
  exercise_retired_binding_detach(true);
}

TEST(AppleLateInput, PinsActualProducerUntilConsumerRetiresAndRejectsStaleEpoch) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  std::atomic<int> traces{0};
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto first = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [&](const auto &args, mx::Stream stream) {
        ++traces;
        return std::vector<mx::array>{mx::add(args[0], mx::array(1.f), stream)};
      }, specs);
  auto second = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [&](const auto &args, mx::Stream stream) {
        ++traces;
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, specs);
  ASSERT_TRUE(first); ASSERT_TRUE(second);
  auto producer = first->create_invocation();
  auto consumer = second->create_invocation();
  ASSERT_TRUE(producer); ASSERT_TRUE(consumer);
  auto source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  std::array<ad::InputBinding, 1> late;
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset(); // Only the consumer's cold binding retains this Job.
  const int cold_traces = traces.load();
  const std::array values{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer->bind(values).ok());
  {
    auto produced = producer->submit();
    ASSERT_TRUE(produced);
    ASSERT_TRUE(produced->completion().wait().ok());
  } // No ResultStorage owner is retained by the test.
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::InvocationAccess::runtime(*producer)).ok());
    gate.wait_entered();
    auto pending = consumer->submit();
    ASSERT_TRUE(pending);
    std::array<ad::InputBinding, 1> ready;
    ready[0].value = values[0];
    EXPECT_TRUE(producer->bind(values).IsUnavailable());
    auto replay = producer->submit();
    ASSERT_FALSE(replay);
    EXPECT_TRUE(replay.error().IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*producer, ready).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::release_inputs(*producer).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(*producer).IsUnavailable());
    gate.open_and_join();
    ASSERT_TRUE(pending->completion().wait().ok());
    auto output = ad::InvocationAccess::output(*consumer, 0);
    ASSERT_TRUE(output);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(output->data<float>()[i], 4.f + 2.f * i);
  }
  auto stale = consumer->submit();
  ASSERT_FALSE(stale);
  EXPECT_TRUE(stale.error().IsFailedPrecondition());
  ASSERT_TRUE(ad::InvocationAccess::reuse_gate(*producer).ok());
  const std::array changed{mx::array({5.f, 6.f, 7.f, 8.f})};
  ASSERT_TRUE(producer->bind(changed).ok());
  {
    auto produced = producer->submit();
    ASSERT_TRUE(produced);
    ASSERT_TRUE(produced->completion().wait().ok());
  }
  *producer = a::Invocation{}; // The consumer retains the actual source Job.
  auto repeated = consumer->submit();
  ASSERT_TRUE(repeated);
  ASSERT_TRUE(repeated->completion().wait().ok());
  EXPECT_EQ(traces.load(), cold_traces);
  ASSERT_TRUE(context->close().ok());
  auto output = ad::InvocationAccess::output(*consumer, 0);
  ASSERT_TRUE(output);
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(output->data<float>()[i], 12.f + 2.f * i);
}

TEST(AppleLateInput, ReplayCopyPinsNewEpochsAndKeepsFailedProducerBytesUntouched) {
  namespace m = hyle::mem;
  auto context = a::Context::create().value();
  const auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = context.prepare(c::Unary{c::UnaryKind::kNeg}, specs).value();
  auto producer = program.create_invocation().value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto target = m::Buffer::create(m::system_resource(), 16, alignof(float)).value();
  m::QuotaAllocator metadata(m::system_allocator());
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, metadata.get(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  auto source = ad::InvocationAccess::source(producer, 0).value();
  ASSERT_TRUE(ad::bind_host_copy_recording(*copy, std::move(source), description, target.view()).ok());
  const auto cold_metadata = metadata.used();
  const std::array first_values{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer.bind(first_values).ok());
  {
    auto produced = producer.submit().value();
    ASSERT_TRUE(produced.completion().wait().ok());
  }
  c::Completion old_completion;
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());
    gate.wait_entered();
    old_completion = ad::submit_copy_recording(*copy).value();
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    EXPECT_TRUE(producer.bind(first_values).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).IsUnavailable());
    gate.open_and_join();
    ASSERT_TRUE(old_completion.wait().ok());
  }
  std::array<float, 4> observed{};
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, (std::array<float, 4>{-1, -2, -3, -4}));
  ASSERT_TRUE(ad::copy_recording_reuse_gate(*copy).ok());
  auto stale = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(stale);
  EXPECT_TRUE(stale.error().IsFailedPrecondition());
  ASSERT_TRUE(ad::InvocationAccess::reuse_gate(producer).ok());

  ASSERT_TRUE(producer.bind(std::array{mx::array({5.f, 6.f, 7.f, 8.f})}).ok());
  {
    auto produced = producer.submit().value();
    ASSERT_TRUE(produced.completion().wait().ok());
  }
  // No new bind/record/map: the settled first Completion remains alive and
  // keeps its own terminal state while the original recording consumes epoch 2.
  auto second = ad::submit_copy_recording(*copy).value();
  ASSERT_TRUE(old_completion.wait().ok());
  EXPECT_EQ(old_completion.poll().value(), c::CompletionPoll::kComplete);
  EXPECT_TRUE(old_completion.request_cancel().ok());
  ASSERT_TRUE(second.wait().ok());
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, (std::array<float, 4>{-5, -6, -7, -8}));
  EXPECT_EQ(metadata.used(), cold_metadata);
  auto repeated = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(repeated);
  EXPECT_TRUE(repeated.error().IsFailedPrecondition());

  auto gather = context.prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}}).value();
  auto bad = gather.create_invocation().value();
  const auto one = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{1}).value();
  auto untouched = m::Buffer::create(m::system_resource(), sizeof(float), alignof(float)).value();
  const std::array<float, 1> sentinel{123.f};
  ASSERT_TRUE(untouched.view().write(std::as_bytes(std::span(sentinel))).ok());
  auto failed_copy = ad::prepare_copy_recording_runtime(runtime, 1, m::system_allocator(), sizeof(float),
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::bind_host_copy_recording(*failed_copy,
      ad::InvocationAccess::source(bad, 0).value(), one, untouched.view()).ok());
  ASSERT_TRUE(bad.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}), mx::array({9}, mx::int32)}).ok());
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());
    gate.wait_entered();
    auto produced = bad.submit().value();
    auto consumed = ad::submit_copy_recording(*failed_copy).value();
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(bad).IsUnavailable());
    gate.open_and_join();
    EXPECT_TRUE(produced.completion().wait().IsOutOfRange());
    EXPECT_TRUE(consumed.wait().IsOutOfRange());
  }
  std::array<float, 1> unchanged{};
  ASSERT_TRUE(untouched.view().read(std::as_writable_bytes(std::span(unchanged))).ok());
  EXPECT_EQ(unchanged, sentinel);
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*failed_copy).IsOutOfRange());
  failed_copy.reset();
  ASSERT_TRUE(context.close().ok()); // Checked failure still drains the original owner.
}

TEST(AppleLateInput, RejectsCyclesPortsAndForeignRuntimeAndPropagatesCheckedFailure) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array specs{a::InputSpec{{1}, mx::float32}};
  const auto identity = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::negative(args[0], stream)};
  };
  auto program = context->compile(identity, specs);
  ASSERT_TRUE(program);
  auto first = program->create_invocation(), second = program->create_invocation();
  ASSERT_TRUE(first); ASSERT_TRUE(second);
  auto first_source = ad::InvocationAccess::source(*first, 0);
  auto second_source = ad::InvocationAccess::source(*second, 0);
  ASSERT_TRUE(first_source); ASSERT_TRUE(second_source);
  EXPECT_FALSE(ad::InvocationAccess::source(*first, 1));
  std::array<ad::InputBinding, 1> binding;
  binding[0].source = *first_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*first, binding).IsInvalidArgument());
  binding[0].source = *second_source;
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*first, binding).ok());
  // This legitimate edge uses the later-created source slot.
  binding[0].source = *first_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*second, binding).IsInvalidArgument());
  auto foreign = a::Context::create();
  ASSERT_TRUE(foreign);
  auto foreign_program = foreign->compile(identity, specs);
  ASSERT_TRUE(foreign_program);
  auto foreign_slot = foreign_program->create_invocation();
  ASSERT_TRUE(foreign_slot);
  auto foreign_source = ad::InvocationAccess::source(*foreign_slot, 0);
  ASSERT_TRUE(foreign_source);
  binding[0].source = *foreign_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*second, binding).IsUnimplemented());

  auto gather = context->prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}});
  ASSERT_TRUE(gather);
  auto bad = gather->create_invocation(), dependent = program->create_invocation();
  ASSERT_TRUE(bad); ASSERT_TRUE(dependent);
  auto bad_source = ad::InvocationAccess::source(*bad, 0);
  ASSERT_TRUE(bad_source);
  binding[0].source = *bad_source;
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*dependent, binding).ok());
  ASSERT_TRUE(bad->bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}),
                                 mx::array({9}, mx::int32)}).ok());
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::InvocationAccess::runtime(*bad)).ok());
    gate.wait_entered();
    auto produced = bad->submit();
    auto consumed = dependent->submit();
    ASSERT_TRUE(produced); ASSERT_TRUE(consumed);
    gate.open_and_join();
    EXPECT_TRUE(produced->completion().wait().IsOutOfRange());
    EXPECT_TRUE(consumed->completion().wait().IsOutOfRange());
    EXPECT_FALSE(ad::InvocationAccess::output(*dependent, 0));
  }
  // Checked input errors retire their slots, not the shared Runtime.
  ASSERT_TRUE(second->bind(std::array{mx::array({3.f})}).ok());
  auto independent = second->submit();
  ASSERT_TRUE(independent);
  ASSERT_TRUE(independent->completion().wait().ok());
  auto output = ad::InvocationAccess::output(*second, 0);
  ASSERT_TRUE(output);
  EXPECT_FLOAT_EQ(output->data<float>()[0], -3.f);
}
TEST(AppleLateInput, OriginalMlxPrefixEncodesPendingMathAndCheckedCopiesWithoutAnOwnerWait) {
  auto context=a::Context::create();ASSERT_TRUE(context)<<context.error().message();
  auto runtime=ad::ContextAccess::runtime(*context);
  const std::array specs{a::InputSpec{{4},mx::float32},a::InputSpec{{1},mx::int32}};
  auto first=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return a::lower(c::Operation{c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked}},args,stream);
  },specs);
  const std::array one{a::InputSpec{{1},mx::float32}};
  auto second=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return std::vector<mx::array>{mx::multiply(args[0],mx::array(2.f),stream)};
  },one);
  ASSERT_TRUE(first)<<first.error().message();ASSERT_TRUE(second)<<second.error().message();
  auto memory=a::MemoryResource::create(*context);ASSERT_TRUE(memory);
  auto target=hyle::mem::Buffer::create(memory->ref(),sizeof(float),256);
  auto other=hyle::mem::Buffer::create(memory->ref(),sizeof(float),256);
  ASSERT_TRUE(target);ASSERT_TRUE(other);
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{1}).value();
  // A checked GPU failure must preserve the same Context for the following
  // valid pair; the failed slots themselves remain retired. Both original
  // producers/consumers encode while the actual GPU is held.
  for(const bool ranges:{false,true})for(const std::int32_t index:{2,9,1}){
    auto producer=first->create_invocation(),consumer=second->create_invocation();
    ASSERT_TRUE(producer);ASSERT_TRUE(consumer);
    auto source=ad::InvocationAccess::source(*producer,0);ASSERT_TRUE(source);
    std::array<ad::InputBinding,1> late;late[0].source=std::move(*source);
    ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer,late).ok());late[0].source.reset();
    const std::array<float,1> sentinel{-17.f};
    ASSERT_TRUE(target->view().write(std::as_bytes(std::span(sentinel))).ok());
    ASSERT_TRUE(other->view().write(std::as_bytes(std::span(sentinel))).ok());
    auto copy=ad::prepare_copy_recording_runtime(runtime,2,hyle::mem::system_allocator(),sizeof(float)*2,
        ad::CopyRecordingMode::kReplayable).value();
    auto provisioned=ad::prepare_device_copy_recording(*copy);
    ASSERT_TRUE(provisioned)<<provisioned.error().message();ASSERT_TRUE(*provisioned);
    EXPECT_FALSE(ad::submit_copy_recording(*copy)); // Metadata is not a bound copy or execution permission.
    auto copied_source=ad::InvocationAccess::source(*producer,0).value();
    if(ranges){
      const std::array spans{ad::HostCopyRange{0,target->view()},ad::HostCopyRange{0,other->view()}};
      ASSERT_TRUE(ad::bind_device_copy_ranges(*copy,std::move(copied_source),description,spans).ok());
    }else ASSERT_TRUE(ad::bind_device_copy_recording(*copy,std::move(copied_source),description,target->view()).ok());
    const std::array values{mx::array({1.f,2.f,3.f,4.f}),mx::array({index})};
    ASSERT_TRUE(producer->bind(values).ok());
    GpuEventGate gpu(runtime);ASSERT_TRUE(gpu.enqueue().ok());
    auto a_work=producer->submit();ASSERT_TRUE(a_work)<<a_work.error().message();
    auto b_work=consumer->submit();ASSERT_TRUE(b_work)<<b_work.error().message();
    auto copied=ad::submit_copy_recording(*copy);ASSERT_TRUE(copied)<<copied.error().message();
    FifoGate owner;
    const auto admitted=owner.enqueue(runtime);
    const bool free=admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
    const auto a_pending=a_work->completion().poll(),b_pending=b_work->completion().poll(),copy_pending=copied->poll();
    const auto rebind=producer->bind(values);
    gpu.open();owner.open_and_join();
    ASSERT_TRUE(admitted.ok())<<admitted.message();EXPECT_TRUE(free);
    ASSERT_TRUE(a_pending);ASSERT_TRUE(b_pending);ASSERT_TRUE(copy_pending);
    EXPECT_EQ(*a_pending,c::CompletionPoll::kPending);EXPECT_EQ(*b_pending,c::CompletionPoll::kPending);
    EXPECT_EQ(*copy_pending,c::CompletionPoll::kPending);
    EXPECT_TRUE(rebind.IsFailedPrecondition());
    const auto a_done=a_work->completion().wait(),b_done=b_work->completion().wait();
    const auto copy_done=copied->wait();
    if(index==9){EXPECT_TRUE(a_done.IsOutOfRange());EXPECT_TRUE(b_done.IsOutOfRange());EXPECT_TRUE(copy_done.IsOutOfRange());}
    else{
      ASSERT_TRUE(a_done.ok())<<a_done.message();ASSERT_TRUE(b_done.ok())<<b_done.message();
      ASSERT_TRUE(copy_done.ok())<<copy_done.message();
      auto result=ad::InvocationAccess::output(*consumer,0);ASSERT_TRUE(result);
      EXPECT_TRUE(result->is_available());EXPECT_FLOAT_EQ(result->data<float>()[0],2.f*(index+1));
    }
    std::array<float,1> observed{};
    ASSERT_TRUE(target->view().read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_FLOAT_EQ(observed[0],index==9?sentinel[0]:float(index+1));
    ASSERT_TRUE(other->view().read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_FLOAT_EQ(observed[0],!ranges || index==9?sentinel[0]:float(index+1));
    EXPECT_EQ(ad::copy_recording_report(*copy).native_calls,1u);
    EXPECT_EQ(ad::copy_recording_report(*copy).host_waits,0u);
  }
  {std::lock_guard lock(runtime->queue_mutex);EXPECT_EQ(runtime->native_tails,0u);}
  ASSERT_TRUE(context->close().ok());
}
TEST(AppleLateInput, SharedPendingAncestorsAreCollectedOnceBeforeNativeCopy) {
  auto context=a::Context::create().value();auto runtime=ad::ContextAccess::runtime(context);
  const std::array gather_specs{a::InputSpec{{4},mx::float32},a::InputSpec{{1},mx::int32}};
  auto first=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return a::lower(c::Operation{c::Gather{}},args,stream);
  },gather_specs).value();
  const std::array pair{a::InputSpec{{1},mx::float32},a::InputSpec{{1},mx::float32}};
  auto twice=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return std::vector<mx::array>{mx::add(args[0],args[1],stream)};
  },pair).value();
  auto memory=a::MemoryResource::create(context).value();
  auto target=hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value();
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{1}).value();
  for(const bool fail:{false,true}){
    auto root=first.create_invocation().value();
    ASSERT_TRUE(root.bind(std::array{mx::array({1.f,2.f,3.f,4.f}),mx::array({fail?9:2},mx::int32)}).ok());
    std::vector<a::Invocation> chain;chain.reserve(24);
    for(int i=0;i<24;++i){
      auto source=ad::InvocationAccess::source(i?chain.back():root,0).value();
      auto &node=chain.emplace_back(twice.create_invocation().value());
      std::array<ad::InputBinding,2> bindings;bindings[0].source=source;bindings[1].source=std::move(source);
      ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(node,bindings).ok());
    }
    auto copy=ad::prepare_copy_recording_runtime(runtime,1,hyle::mem::system_allocator(),sizeof(float)).value();
    ASSERT_TRUE(ad::bind_device_copy_recording(*copy,ad::InvocationAccess::source(chain.back(),0).value(),description,target.view()).ok());
    const std::array<float,1> sentinel{-123.f};
    ASSERT_TRUE(target.view().write(std::as_bytes(std::span(sentinel))).ok());
    GpuEventGate gpu(runtime);ASSERT_TRUE(gpu.enqueue().ok());
    auto produced=root.submit().value();
    std::vector<c::OwnedResult<a::Results>> pending;pending.reserve(chain.size());
    for(auto &node:chain){auto work=node.submit();ASSERT_TRUE(work)<<work.error().message();pending.push_back(std::move(*work));}
    auto copied=ad::submit_copy_recording(*copy).value();
    FifoGate owner;const auto admitted=owner.enqueue(runtime);
    const bool free=admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
    const auto poll=copied.poll();gpu.open();owner.open_and_join();
    ASSERT_TRUE(admitted.ok());EXPECT_TRUE(free);ASSERT_TRUE(poll);EXPECT_EQ(*poll,c::CompletionPoll::kPending);
    const auto result=copied.wait();
    EXPECT_EQ(result.IsOutOfRange(),fail);EXPECT_EQ(produced.completion().wait().code(),result.code());
    for(auto &work:pending)EXPECT_EQ(work.completion().wait().code(),result.code());
    std::array<float,1> actual{};ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_FLOAT_EQ(actual[0],fail?sentinel[0]:3.f*float(std::uint32_t{1}<<24));
    EXPECT_EQ(ad::copy_recording_report(*copy).native_calls,1u);
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(AppleLateInput, CheckedStoreGroupKeepsEachOriginalWriterAndVerdict) {
  namespace m = hyle::mem;
  auto context = a::Context::create().value();
  auto runtime = ad::ContextAccess::runtime(context);
  auto memory = a::MemoryResource::create(context).value();
  const std::array specs{a::InputSpec{{4}, mx::float32}, a::InputSpec{{4}, mx::int32}};
  auto program = ad::ContextAccess::compile_device_prefix(runtime, [](const auto &args, mx::Stream stream) {
    return a::lower(c::Operation{c::Gather{}}, args, stream);
  }, specs).value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  constexpr std::size_t writers = 3;
  const std::array values{11.f, 23.f, 37.f, 41.f};
  const std::array sentinel{-1.f, -2.f, -3.f, -4.f};
  auto target = m::Buffer::create(memory.ref(), sizeof(sentinel), 256).value();
  for (unsigned mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    const bool failed_guard = mode == 1;
    const bool overlapping_writes = mode == 2;
    const bool cancelled = mode == 3;
    ASSERT_TRUE(target.write(0, std::as_bytes(std::span(sentinel))).ok());
    auto producer = program.create_invocation().value();
    ASSERT_TRUE(producer.bind(std::array{mx::array(values.data(), {4}, mx::float32),
        mx::array({0, 1, failed_guard ? 99 : 2, 3}, mx::int32)}).ok());
    std::array<std::shared_ptr<ad::CopyRecording>, writers> copies;
    for (std::size_t i = 0; i < writers; ++i) {
      copies[i] = ad::prepare_copy_recording_runtime(runtime, 1, m::system_allocator(), sizeof(float),
          ad::CopyRecordingMode::kReplayable).value();
      ASSERT_TRUE(ad::prepare_device_copy_recording(*copies[i]).value());
      ASSERT_TRUE(ad::reserve_copy_dependencies(*copies[i], 1).ok());
      const auto position = overlapping_writes && i == 1 ? 0 : i;
      const std::array spans{ad::HostCopyRange{i * sizeof(float),
          target.view().subview(position * sizeof(float), sizeof(float)).value()}};
      ASSERT_TRUE(ad::bind_device_copy_ranges(*copies[i],
          ad::InvocationAccess::source(producer, 0).value(), description, spans).ok());
    }
    GpuEventGate gpu(runtime);
    ASSERT_TRUE(gpu.enqueue().ok());
    FifoGate admission;
    ASSERT_TRUE(admission.enqueue(runtime).ok());
    admission.wait_entered();
    auto produced = producer.submit().value();
    std::array<c::Completion, writers> completed;
    completed[0] = ad::submit_copy_recording(*copies[0]).value();
    for (std::size_t i = 1; i < writers; ++i) {
      const std::array dependency{c::detail::CompletionAccess::dependency(completed[i - 1])};
      completed[i] = ad::submit_copy_recording(*copies[i], dependency).value();
    }
    if (cancelled) EXPECT_TRUE(completed[1].request_cancel().ok());
    FifoGate witness;
    auto queued = witness.enqueue(runtime);
    admission.open_and_join();
    const bool pending = queued.ok() && !cancelled && witness.wait_entered_for(std::chrono::seconds(2));
    if (pending) {
      for (std::size_t i = 0; i < writers; ++i) {
        EXPECT_EQ(completed[i].poll().value(), c::CompletionPoll::kPending);
        EXPECT_TRUE(ad::copy_recording_reuse_gate(*copies[i]).IsUnavailable());
      }
    }
    gpu.open();
    witness.open_and_join();
    ASSERT_TRUE(queued.ok()) << queued.message();
    EXPECT_TRUE(pending || cancelled);
    const auto producer_done = produced.completion().wait();
    EXPECT_EQ(producer_done.IsOutOfRange(), failed_guard);
    EXPECT_EQ(producer_done.ok(), !failed_guard);
    for (std::size_t i = 0; i < writers; ++i) {
      const auto terminal = completed[i].wait();
      EXPECT_EQ(terminal.IsOutOfRange(), failed_guard) << terminal.message();
      EXPECT_EQ(terminal.IsCancelled(), cancelled && i != 0) << terminal.message();
      EXPECT_EQ(terminal.ok(), !failed_guard && (!cancelled || i == 0)) << terminal.message();
    }
    std::uint64_t dispatches = 0;
    for (const auto &copy : copies) dispatches += ad::copy_recording_report(*copy).native_calls;
    EXPECT_EQ(dispatches, overlapping_writes ? 2u : 1u);
    auto expected = sentinel;
    if (!failed_guard) {
      expected[0] = values[overlapping_writes ? 1 : 0];
      if (!cancelled) {
        if (!overlapping_writes) expected[1] = values[1];
        expected[2] = values[2];
      }
    }
    std::array<float, sentinel.size()> actual{};
    ASSERT_TRUE(target.read(0, std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, expected);
  }
  ASSERT_TRUE(context.close().ok());
  { std::lock_guard lock(runtime->queue_mutex); EXPECT_EQ(runtime->native_tails, 0u); }
}

TEST(AppleLateInput, PendingCheckedByteRangesPreserveBitsAndEveryUnownedByte) {
  auto context=a::Context::create().value();auto runtime=ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4},mx::float32},a::InputSpec{{4},mx::int32}};
  auto program=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return a::lower(c::Operation{c::Gather{}},args,stream);
  },specs).value();
  auto memory=a::MemoryResource::create(context).value();
  auto target=hyle::mem::Buffer::create(memory.ref(),24,256).value();
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{4}).value();
  const auto payload=std::bit_cast<std::array<float,4>>(std::array<std::uint32_t,4>{0x7fc10001,0x80000000,1,0xff800000});
  for(const bool fail:{false,true}){
    auto producer=program.create_invocation().value();
    ASSERT_TRUE(producer.bind(std::array{mx::array(payload.data(),{4},mx::float32),
        mx::array({0,1,fail?99:2,3},mx::int32)}).ok());
    auto copy=ad::prepare_copy_recording_runtime(runtime,2,hyle::mem::system_allocator(),16).value();
    const std::array ranges{ad::HostCopyRange{1,target.view().subview(3,3).value()},
        ad::HostCopyRange{7,target.view().subview(11,5).value()}};
    ASSERT_TRUE(ad::bind_device_copy_ranges(*copy,ad::InvocationAccess::source(producer,0).value(),description,ranges).ok());
    const std::array<std::byte,24> sentinel=[] {std::array<std::byte,24> b;b.fill(std::byte{0xa5});return b;}();
    ASSERT_TRUE(target.view().write(sentinel).ok());
    GpuEventGate gpu(runtime);ASSERT_TRUE(gpu.enqueue().ok());
    auto produced=producer.submit().value();auto copied=ad::submit_copy_recording(*copy).value();
    FifoGate owner;const auto admitted=owner.enqueue(runtime);
    const bool free=admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
    const auto poll=copied.poll();gpu.open();owner.open_and_join();
    ASSERT_TRUE(admitted.ok());EXPECT_TRUE(free);ASSERT_TRUE(poll);EXPECT_EQ(*poll,c::CompletionPoll::kPending);
    const auto terminal=copied.wait();EXPECT_EQ(terminal.IsOutOfRange(),fail);
    EXPECT_EQ(produced.completion().wait().code(),terminal.code());
    auto expected=sentinel;
    if(!fail){const auto bytes=std::as_bytes(std::span(payload));
      std::copy_n(bytes.begin()+1,3,expected.begin()+3);std::copy_n(bytes.begin()+7,5,expected.begin()+11);}
    std::array<std::byte,24> actual{};ASSERT_TRUE(target.view().read(actual).ok());EXPECT_EQ(actual,expected);
    EXPECT_EQ(ad::copy_recording_report(*copy).native_calls,1u);
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(AppleLateInput, UnequalLargeNativeCopiesKeepBothFinalTailsAndCheckedFailure) {
  constexpr int count=65537;
  auto context=a::Context::create().value();auto runtime=ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{count},mx::float32},a::InputSpec{{1},mx::int32}};
  auto program=ad::ContextAccess::compile_device_prefix(runtime,
      [](const auto &args,mx::Stream stream)->hyle::Result<std::vector<mx::array>>{
        auto checked=a::lower(c::Operation{c::Gather{}},args,stream);
        if(!checked)return std::unexpected(checked.error());
        return std::vector{mx::copy(args[0],stream),checked->front()};
      },specs).value();
  auto memory=a::MemoryResource::create(context).value();
  std::vector<float> payload(count);
  for(int i=0;i<count;++i)payload[i]=std::bit_cast<float>(0x3f000001u+std::uint32_t(i));
  const auto bytes=std::as_bytes(std::span(payload));
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{count}).value();
  auto large=hyle::mem::Buffer::create(memory.ref(),bytes.size()+32,256).value();
  auto small=hyle::mem::Buffer::create(memory.ref(),8195+32,256).value();
  for(const bool fail:{false,true}){
    auto producer=program.create_invocation().value();
    ASSERT_TRUE(producer.bind(std::array{mx::array(payload.data(),{count},mx::float32),mx::array({fail?count+1:0},mx::int32)}).ok());
    std::vector<std::byte> expected_large(large.size(),std::byte{0x5c}),expected_small(small.size(),std::byte{0xa5});
    ASSERT_TRUE(large.view().write(expected_large).ok());ASSERT_TRUE(small.view().write(expected_small).ok());
    auto copy=ad::prepare_copy_recording_runtime(runtime,2,hyle::mem::system_allocator(),bytes.size()+8195).value();
    const std::array ranges{ad::HostCopyRange{0,large.view().subview(16,bytes.size()).value()},
        ad::HostCopyRange{3,small.view().subview(11,8195).value()}};
    ASSERT_TRUE(ad::bind_device_copy_ranges(*copy,ad::InvocationAccess::source(producer,0).value(),description,ranges).ok());
    GpuEventGate gpu(runtime);ASSERT_TRUE(gpu.enqueue().ok());
    auto produced=producer.submit().value();auto copied=ad::submit_copy_recording(*copy).value();
    FifoGate owner;const auto admitted=owner.enqueue(runtime);
    const bool free=admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
    const auto poll=copied.poll();gpu.open();owner.open_and_join();
    ASSERT_TRUE(admitted.ok());EXPECT_TRUE(free);ASSERT_TRUE(poll);EXPECT_EQ(*poll,c::CompletionPoll::kPending);
    const auto terminal=copied.wait();EXPECT_EQ(terminal.IsOutOfRange(),fail);
    EXPECT_EQ(produced.completion().wait().code(),terminal.code());
    if(!fail){std::copy(bytes.begin(),bytes.end(),expected_large.begin()+16);
      std::copy_n(bytes.begin()+3,8195,expected_small.begin()+11);}
    std::vector<std::byte> actual_large(large.size()),actual_small(small.size());
    ASSERT_TRUE(large.view().read(actual_large).ok());ASSERT_TRUE(small.view().read(actual_small).ok());
    EXPECT_EQ(actual_large,expected_large);EXPECT_EQ(actual_small,expected_small);
  }
  ASSERT_TRUE(context.close().ok());
}
TEST(AppleLateInput, ReusingCpuTraversalScratchNeverReplacesPendingGpuChecks) {
  auto context=a::Context::create().value();auto runtime=ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4},mx::float32},a::InputSpec{{1},mx::int32}};
  auto program=ad::ContextAccess::compile_device_prefix(runtime,[](const auto &args,mx::Stream stream){
    return a::lower(c::Operation{c::Gather{}},args,stream);
  },specs).value();
  auto memory=a::MemoryResource::create(context).value();
  const auto description=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{1}).value();
  std::array targets{hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value(),
      hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value()};
  std::array producers{program.create_invocation().value(),program.create_invocation().value()};
  std::array<std::shared_ptr<ad::CopyRecording>,2> copies;
  const std::array<float,1> sentinel{-73.f};
  for(std::size_t i=0;i<2;++i){
    ASSERT_TRUE(targets[i].view().write(std::as_bytes(std::span(sentinel))).ok());
    ASSERT_TRUE(producers[i].bind(std::array{mx::array({1.f,2.f,3.f,4.f}),mx::array({i?2:9},mx::int32)}).ok());
    copies[i]=ad::prepare_copy_recording_runtime(runtime,1,hyle::mem::system_allocator(),sizeof(float)).value();
    ASSERT_TRUE(ad::bind_device_copy_recording(*copies[i],ad::InvocationAccess::source(producers[i],0).value(),description,targets[i].view()).ok());
  }
  GpuEventGate gpu(runtime);ASSERT_TRUE(gpu.enqueue().ok());
  std::array<c::OwnedResult<a::Results>,2> jobs;
  std::array<c::Completion,2> copied;
  for(std::size_t i=0;i<2;++i){jobs[i]=producers[i].submit().value();copied[i]=ad::submit_copy_recording(*copies[i]).value();}
  FifoGate owner;const auto admitted=owner.enqueue(runtime);
  const bool free=admitted.ok() && owner.wait_entered_for(std::chrono::seconds(2));
  const std::array polls{copied[0].poll(),copied[1].poll()};gpu.open();owner.open_and_join();
  ASSERT_TRUE(admitted.ok());EXPECT_TRUE(free);
  for(const auto &poll:polls){ASSERT_TRUE(poll);EXPECT_EQ(*poll,c::CompletionPoll::kPending);}
  for(std::size_t i=0;i<2;++i){
    const auto result=copied[i].wait();EXPECT_EQ(result.IsOutOfRange(),i==0);
    EXPECT_EQ(jobs[i].completion().wait().code(),result.code());
    std::array<float,1> actual{};ASSERT_TRUE(targets[i].view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_FLOAT_EQ(actual[0],i?3.f:sentinel[0]);
  }
  ASSERT_TRUE(context.close().ok());
}
} // namespace
