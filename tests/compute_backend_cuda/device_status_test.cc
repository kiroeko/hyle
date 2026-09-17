#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <span>
#include <thread>
#include <vector>

#include "../../src/compute_backend_cuda/cuda_internal.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/mem/stats.h"

cudaError_t launch_index_validation(cudaStream_t, const int *, std::size_t, int,
                                    unsigned *) noexcept;
cudaError_t launch_scan_validation(cudaStream_t, const int *, std::size_t,
                                   unsigned *) noexcept;
cudaError_t launch_increment_kernel(cudaStream_t, void *, std::size_t) noexcept;
cudaError_t prepare_validation_kernels() noexcept;

namespace {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;
using compute::detail::CapturedCommands;
using compute::detail::CommandCaptureAccess;
using compute::detail::EnqueueResult;
using compute::detail::QueueSequence;

struct Fixture final {
  mem::StatsAllocator stats =
      mem::StatsAllocator::create(mem::system_allocator()).value();
  compute::Context context =
      compute::Context::create(cuda::device(0).value(),
                               {.metadata_allocator = stats.get()})
          .value();
  cuda::MemoryResource memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  compute::Queue first = context.create_queue().value();
  compute::Queue second = context.create_queue().value();
  mem::Buffer values =
      mem::Buffer::create(memory.ref(), 257 * sizeof(int), 256).value();

  Fixture() {
    EXPECT_EQ(prepare_validation_kernels(), cudaSuccess);
    EXPECT_TRUE(first.reserve_completions(4).ok());
    EXPECT_TRUE(second.reserve_completions(4).ok());
  }
  ~Fixture() {
    context.drain().IgnoreError();
    values = {};
    first = {};
    second = {};
    memory = {};
    context = {};
    // An abandoned Completion is retired by the real CUDA reaper. Keep the
    // borrowed allocator alive until that asynchronous teardown is observed.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!stats.balanced() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_TRUE(stats.balanced());
  }
  hyle::Status write(std::span<const int> data) {
    return values.write(0, std::as_bytes(data));
  }
  hyle::Status check(QueueSequence &sequence, std::size_t count,
                     bool scan = false, std::size_t element_offset = 0) {
    auto pointer = cuda::pointer<int>(values.view());
    if (!pointer)
      return pointer.error();
    unsigned *word = nullptr;
    auto initialized =
        cuda::detail::acquire_device_status(sequence.queue(), &word);
    const auto status = initialized.status;
    sequence.record(std::move(initialized));
    if (!status.ok())
      return status;
    const auto stream = cuda::stream(sequence.queue()).value();
    const auto recorded =
        scan ? launch_scan_validation(stream, *pointer + element_offset, count,
                                      word)
             : launch_index_validation(stream, *pointer + element_offset, count,
                                       1024, word);
    sequence.record(EnqueueResult::accepted_command(cuda::status(recorded)));
    return cuda::status(recorded);
  }
  hyle::Result<compute::Completion> submit(compute::Queue &queue,
                                           std::size_t count, bool scan = false,
                                           std::size_t element_offset = 0) {
    auto sequence = QueueSequence::begin(queue);
    if (!sequence)
      return std::unexpected(sequence.error());
    auto checked = check(*sequence, count, scan, element_offset);
    if (!checked.ok() && sequence->terminal().ok())
      sequence->record(EnqueueResult::rejected(checked));
    return sequence->finish();
  }
  hyle::Result<CapturedCommands> capture(compute::Queue &queue,
                                         std::size_t count, bool scan = false) {
    auto sequence = QueueSequence::begin(queue).value();
    auto extension = CommandCaptureAccess::query(queue).value();
    auto begun = CommandCaptureAccess::begin(extension, sequence);
    if (!begun.ok())
      return std::unexpected(begun);
    auto checked = check(sequence, count, scan);
    if (!checked.ok()) {
      CommandCaptureAccess::abort(extension, sequence).IgnoreError();
      sequence.finish().value().wait().IgnoreError();
      return std::unexpected(checked);
    }
    auto captured = CommandCaptureAccess::end(extension, sequence, stats.get());
    const auto finished = sequence.finish().value().wait();
    if (!finished.ok())
      return std::unexpected(finished);
    return captured;
  }
};

struct Gate final {
  std::atomic<bool> released{false};
  std::atomic<const char *> phase{"unlabeled command"};
  const char *stalled_at = nullptr;
  ~Gate() { EXPECT_EQ(stalled_at, nullptr) << stalled_at; }
  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<Gate *>(raw);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.released.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        gate.stalled_at = gate.phase.load(std::memory_order_acquire);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  void release() {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};
struct Release final {
  Gate &gate;
  cudaStream_t stream;
  ~Release() {
    gate.release();
    (void)cudaStreamSynchronize(stream);
  }
};

TEST(CudaDeviceStatus, GpuChecksReportOnlyAfterCompletionAndFreshRunsRecover) {
  Fixture fixture;
  const auto before = fixture.stats.snapshot();
  for (const auto data : {std::array{1, 2, 1023}, std::array{1, -1, 2},
                          std::array{1, 1024, 2}, std::array{0, 1, 2}}) {
    ASSERT_TRUE(fixture.write(data).ok());
    Gate gate;
    Release release{gate, cuda::stream(fixture.first).value()};
    ASSERT_EQ(cudaLaunchHostFunc(cuda::stream(fixture.first).value(),
                                 &Gate::wait, &gate),
              cudaSuccess);
    auto done = fixture.submit(fixture.first, data.size());
    ASSERT_TRUE(done.has_value()) << done.error().message();
    EXPECT_EQ(done->poll().value(), compute::CompletionPoll::kPending);
    gate.release();
    const auto result = done->wait();
    if (data[1] < 0 || data[1] >= 1024) {
      EXPECT_TRUE(result.IsOutOfRange()) << result.message();
      EXPECT_NE(result.message().find("index"), std::string_view::npos);
    } else
      EXPECT_TRUE(result.ok()) << result.message();
  }
  for (const auto data :
       {std::array{INT_MAX, 1, -1}, std::array{INT_MIN, -1, 1},
        std::array{INT_MAX, -1, 1}}) {
    ASSERT_TRUE(fixture.write(data).ok());
    auto done = fixture.submit(fixture.first, data.size(), true);
    ASSERT_TRUE(done.has_value());
    const auto result = done->wait();
    if (data[0] == INT_MAX && data[1] == -1)
      EXPECT_TRUE(result.ok());
    else {
      EXPECT_TRUE(result.IsOutOfRange());
      EXPECT_NE(result.message().find("overflows"), std::string_view::npos);
    }
  }
  EXPECT_EQ(fixture.stats.snapshot().total_allocs, before.total_allocs);
}

TEST(CudaDeviceStatus, OnePrefixKeepsItsFirstDeviceFailure) {
  Fixture fixture;
  constexpr std::array data{INT_MAX, 1};
  ASSERT_TRUE(fixture.write(data).ok());
  for (const bool scan_first : {false, true}) {
    auto sequence = QueueSequence::begin(fixture.first).value();
    ASSERT_TRUE(fixture.check(sequence, data.size(), scan_first).ok());
    ASSERT_TRUE(fixture.check(sequence, data.size(), !scan_first).ok());
    auto done = sequence.finish().value();
    const auto status = done.wait();
    EXPECT_TRUE(status.IsOutOfRange());
    EXPECT_NE(status.message().find(scan_first ? "overflows" : "index"),
              std::string_view::npos);
  }
}

TEST(CudaDeviceStatus,
     ImportsRetainSourceWordsAfterEarlyObservationAndSlotReuse) {
  Fixture fixture;
  for (const bool abandon : {false, true}) {
    // Stage both inputs cold. Gate the source only until the dependency has
    // been admitted; CUDA host callbacks must not wait on unrelated future
    // GPU work. Keep the destination unobserved while the source pool churns.
    ASSERT_TRUE(fixture.write(std::array{-1, 4, 5, 1, 4, 5}).ok());
    Gate gate;
    Release release{gate, cuda::stream(fixture.first).value()};
    ASSERT_EQ(cudaLaunchHostFunc(cuda::stream(fixture.first).value(),
                                 &Gate::wait, &gate),
              cudaSuccess);
    auto source = fixture.submit(fixture.first, 3).value();
    auto *source_slot = static_cast<cuda::detail::EventSlot *>(
        compute::detail::CompletionAccess::dependency(source).state);
    ASSERT_NE(source_slot, nullptr);
    EXPECT_EQ(source.poll().value(), compute::CompletionPoll::kPending);
    auto sequence = QueueSequence::begin(fixture.second).value();
    gate.phase.store("import source status", std::memory_order_release);
    auto waited = sequence.enqueue_wait(source);
    ASSERT_TRUE(waited.status.ok()) << waited.status.message();
    EXPECT_TRUE(waited.accepted);
    sequence.record(std::move(waited));
    gate.phase.store("finish destination", std::memory_order_release);
    auto destination = sequence.finish().value();
    EXPECT_EQ(destination.poll().value(), compute::CompletionPoll::kPending);
    gate.phase.store("observe or abandon source", std::memory_order_release);
    if (abandon) {
      source = {};
      gate.release();
    } else {
      gate.release();
      EXPECT_TRUE(source.wait().IsOutOfRange());
    }
    ASSERT_TRUE(fixture.first.drain().ok());
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (source_slot->references.load(std::memory_order_acquire) > 1 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    ASSERT_EQ(source_slot->references.load(std::memory_order_acquire), 1U);
    for (int repeat = 0; repeat < 40; ++repeat) {
      auto fresh = fixture.submit(fixture.first, 3, false, 3).value();
      ASSERT_TRUE(fresh.wait().ok());
    }
    // The destination may already have completed on the GPU. Inspecting the
    // retained device word also catches premature reuse even in that ordering.
    unsigned retained_error = 0;
    ASSERT_EQ(cudaMemcpy(&retained_error, source_slot->device_status,
                         sizeof(retained_error), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(retained_error,
              static_cast<unsigned>(
                  cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    ASSERT_TRUE(fixture.context.is_open());
    EXPECT_TRUE(destination.wait().IsOutOfRange());
  }
}

TEST(CudaDeviceStatus, CapturedSlotsResetStateAndRetainArtifactsThroughClose) {
  Fixture fixture;
  auto first = fixture.capture(fixture.first, 3).value();
  auto second = fixture.capture(fixture.second, 3, true).value();
  const auto before = fixture.stats.snapshot();
  for (int repeat = 0; repeat < 3; ++repeat) {
    const std::array data =
        repeat == 1 ? std::array{INT_MAX, 1, -1} : std::array{1, 2, 3};
    ASSERT_TRUE(fixture.write(data).ok());
    auto a = QueueSequence::begin(fixture.first).value();
    a.record(first.enqueue(a));
    auto done_a = a.finish().value();
    auto b = QueueSequence::begin(fixture.second).value();
    b.record(second.enqueue(b));
    auto done_b = b.finish().value();
    if (repeat == 1) {
      EXPECT_TRUE(done_a.wait().IsOutOfRange());
      EXPECT_TRUE(done_b.wait().IsOutOfRange());
    } else {
      EXPECT_TRUE(done_a.wait().ok());
      EXPECT_TRUE(done_b.wait().ok());
    }
  }
  EXPECT_EQ(fixture.stats.snapshot().total_allocs, before.total_allocs);
  ASSERT_TRUE(fixture.write(std::array{-1, 2, 3}).ok());
  Gate gate;
  Release release{gate, cuda::stream(fixture.first).value()};
  ASSERT_EQ(cudaLaunchHostFunc(cuda::stream(fixture.first).value(), &Gate::wait,
                               &gate),
            cudaSuccess);
  auto sequence = QueueSequence::begin(fixture.first).value();
  sequence.record(first.enqueue(sequence));
  auto done = sequence.finish().value();
  EXPECT_EQ(done.poll().value(), compute::CompletionPoll::kPending);
  first.reset();
  ASSERT_TRUE(fixture.context.close().ok());
  gate.release();
  EXPECT_TRUE(done.wait().IsOutOfRange());
}

TEST(CudaDeviceStatus, MixedCapturesAndRuntimeWorkAggregateIntoOnePrefix) {
  Fixture fixture;
  auto indices = fixture.capture(fixture.first, 2).value();
  auto scan = fixture.capture(fixture.first, 2, true).value();
  ASSERT_TRUE(fixture.write(std::array{INT_MAX, 1}).ok());
  auto sequence = QueueSequence::begin(fixture.first).value();
  sequence.record(scan.enqueue(sequence));
  ASSERT_TRUE(sequence.terminal().ok());
  sequence.record(indices.enqueue(sequence));
  ASSERT_TRUE(sequence.terminal().ok());
  ASSERT_TRUE(fixture.check(sequence, 2).ok());
  auto done = sequence.finish().value();
  auto status = done.wait();
  EXPECT_TRUE(status.IsOutOfRange());
  EXPECT_NE(status.message().find("overflows"), std::string_view::npos);
}

TEST(CudaDeviceStatus, PureCaptureAndPrefixAddNoStatusCommands) {
  Fixture fixture;
  ASSERT_TRUE(fixture.write(std::array{0, 0, 0}).ok());
  auto sequence = QueueSequence::begin(fixture.first).value();
  auto *state = static_cast<cuda::detail::QueueState *>(
      compute::detail::ContextAccess::backend_queue(fixture.first));
  auto *slot = state->active_sequence;
  auto extension = CommandCaptureAccess::query(fixture.first).value();
  ASSERT_TRUE(CommandCaptureAccess::begin(extension, sequence).ok());
  auto pointer = cuda::pointer<int>(fixture.values.view()).value();
  const auto stream = cuda::stream(fixture.first).value();
  ASSERT_EQ(launch_increment_kernel(stream, pointer, sizeof(int)), cudaSuccess);
  sequence.record(EnqueueResult::accepted_command());
  cudaStreamCaptureStatus capture_status{};
  unsigned long long capture_id = 0;
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id,
                                     &graph, nullptr, nullptr, nullptr),
            cudaSuccess);
  std::size_t nodes = 0;
  ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &nodes), cudaSuccess);
  EXPECT_EQ(nodes, 1U);
  auto captured =
      CommandCaptureAccess::end(extension, sequence, fixture.stats.get())
          .value();
  auto prepared = sequence.finish().value();
  EXPECT_FALSE(slot->status_used);
  EXPECT_FALSE(slot->status_copied);
  ASSERT_TRUE(prepared.wait().ok());
  auto replay = QueueSequence::begin(fixture.first).value();
  slot = state->active_sequence;
  replay.record(captured.enqueue(replay));
  auto done = replay.finish().value();
  EXPECT_FALSE(slot->status_used);
  EXPECT_FALSE(slot->status_copied);
  EXPECT_EQ(slot->status_references, nullptr);
  ASSERT_TRUE(done.wait().ok());
}

TEST(CudaDeviceStatus, CapturedStatusRejectsOverlappingReuseAndAbortIsClean) {
  Fixture fixture;
  ASSERT_TRUE(fixture.write(std::array{1, 2, 3}).ok());
  auto extension = CommandCaptureAccess::query(fixture.first).value();
  auto recording = QueueSequence::begin(fixture.first).value();
  ASSERT_TRUE(CommandCaptureAccess::begin(extension, recording).ok());
  ASSERT_TRUE(fixture.check(recording, 3).ok());
  ASSERT_TRUE(CommandCaptureAccess::abort(extension, recording).ok());
  ASSERT_TRUE(recording.finish().value().wait().ok());
  auto *backend = static_cast<cuda::detail::QueueState *>(
      compute::detail::ContextAccess::backend_queue(fixture.first));
  EXPECT_FALSE(backend->capture_owned);
  EXPECT_FALSE(backend->capture_status_used);
  EXPECT_EQ(backend->capture_device_status, nullptr);
  auto captured = fixture.capture(fixture.first, 3).value();
  Gate gate;
  Release release{gate, cuda::stream(fixture.first).value()};
  ASSERT_EQ(cudaLaunchHostFunc(cuda::stream(fixture.first).value(), &Gate::wait,
                               &gate),
            cudaSuccess);
  auto sequence = QueueSequence::begin(fixture.first).value();
  sequence.record(captured.enqueue(sequence));
  auto pending = sequence.finish().value();
  auto overlapping = QueueSequence::begin(fixture.second).value();
  auto refused = captured.enqueue(overlapping);
  EXPECT_FALSE(refused.accepted);
  EXPECT_TRUE(refused.status.IsFailedPrecondition());
  overlapping.record(std::move(refused));
  EXPECT_FALSE(overlapping.finish().has_value());
  gate.release();
  ASSERT_TRUE(pending.wait().ok());
  auto fresh = QueueSequence::begin(fixture.second).value();
  fresh.record(captured.enqueue(fresh));
  ASSERT_TRUE(fresh.finish().value().wait().ok());
}

TEST(CudaDeviceStatus,
     ReservedReferenceBudgetRejectsWithoutLosingAcceptedWork) {
  Fixture fixture;
  ASSERT_TRUE(fixture.write(std::array{1, 2, 3}).ok());
  auto source = fixture.submit(fixture.first, 3).value();
  auto *backend = static_cast<cuda::detail::QueueState *>(
      compute::detail::ContextAccess::backend_queue(fixture.second));
  const auto before = fixture.stats.snapshot();
  auto sequence = QueueSequence::begin(fixture.second).value();
  for (std::size_t i = 0; i < 2 * backend->capacity; ++i) {
    auto wait = sequence.enqueue_wait(source);
    ASSERT_TRUE(wait.status.ok()) << wait.status.message();
    sequence.record(std::move(wait));
  }
  auto exhausted = sequence.enqueue_wait(source);
  EXPECT_FALSE(exhausted.accepted);
  EXPECT_TRUE(exhausted.status.IsResourceExhausted());
  sequence.record(std::move(exhausted));
  auto accepted = sequence.finish();
  ASSERT_TRUE(accepted.has_value());
  EXPECT_TRUE(accepted->wait().IsResourceExhausted());
  EXPECT_EQ(fixture.stats.snapshot().total_allocs, before.total_allocs);
  auto fresh = QueueSequence::begin(fixture.second).value();
  auto wait = fresh.enqueue_wait(source);
  ASSERT_TRUE(wait.status.ok());
  fresh.record(std::move(wait));
  ASSERT_TRUE(fresh.finish().value().wait().ok());
  EXPECT_TRUE(source.wait().ok());
}

TEST(CudaDeviceStatus, KnownTailFailureAndUnknownDeviceCodeRemainErrors) {
  Fixture fixture;
  ASSERT_TRUE(fixture.write(std::array{-1, 2, 3}).ok());
  auto sequence = QueueSequence::begin(fixture.first).value();
  ASSERT_TRUE(fixture.check(sequence, 3).ok());
  sequence.record(
      EnqueueResult::rejected(hyle::Status::InvalidArgument("tail rejected")));
  EXPECT_TRUE(sequence.finish().value().wait().IsInvalidArgument());
  auto unknown = QueueSequence::begin(fixture.first).value();
  unsigned *word = nullptr;
  auto acquired = cuda::detail::acquire_device_status(fixture.first, &word);
  ASSERT_TRUE(acquired.status.ok());
  unknown.record(std::move(acquired));
  const auto written = cudaMemsetAsync(word, 0x7f, sizeof(unsigned),
                                       cuda::stream(fixture.first).value());
  unknown.record(EnqueueResult::accepted_command(cuda::status(written)));
  EXPECT_TRUE(unknown.finish().value().wait().IsInternal());
}

struct FailAllocator final {
  bool fail = false;
  mem::AllocatorRef base = mem::system_allocator();
  mem::AllocatorRef ref() noexcept {
    return {[](void *raw, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<FailAllocator *>(raw);
              return self.fail ? nullptr : self.base.allocate(bytes, alignment);
            },
            [](void *raw, void *pointer, std::size_t bytes,
               std::size_t alignment) noexcept {
              static_cast<FailAllocator *>(raw)->base.deallocate(pointer, bytes,
                                                                 alignment);
            },
            [](void *raw, void *pointer, std::size_t old_bytes,
               std::size_t bytes, std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<FailAllocator *>(raw);
              return self.fail && bytes != 0
                         ? nullptr
                         : self.base.reallocate(pointer, old_bytes, bytes,
                                                alignment);
            },
            this};
  }
};

TEST(CudaDeviceStatus, ColdMetadataOomRollsBackPoolAndNativeCapturedStorage) {
  FailAllocator failure;
  auto stats = mem::StatsAllocator::create(failure.ref()).value();
  {
    auto context = compute::Context::create(cuda::device(0).value(),
                                            {.metadata_allocator = stats.get()})
                       .value();
    auto queue = context.create_queue().value();
    const auto before = stats.snapshot();
    failure.fail = true;
    EXPECT_TRUE(queue.reserve_completions(1).IsResourceExhausted());
    EXPECT_EQ(stats.snapshot().live_count, before.live_count);
    failure.fail = false;
    ASSERT_TRUE(queue.reserve_completions(1).ok());
    auto *state = static_cast<cuda::detail::QueueState *>(
        compute::detail::ContextAccess::backend_queue(queue));
    auto extension = CommandCaptureAccess::query(queue).value();
    auto recording = QueueSequence::begin(queue).value();
    ASSERT_TRUE(CommandCaptureAccess::begin(extension, recording).ok());
    unsigned *word = nullptr;
    recording.record(cuda::detail::acquire_device_status(queue, &word));
    ASSERT_TRUE(recording.terminal().ok());
    const auto retained = stats.snapshot().live_count;
    failure.fail = true;
    auto captured =
        CommandCaptureAccess::end(extension, recording, stats.get());
    ASSERT_FALSE(captured.has_value());
    EXPECT_TRUE(captured.error().IsResourceExhausted());
    failure.fail = false;
    EXPECT_EQ(stats.snapshot().live_count, retained);
    EXPECT_FALSE(state->capture_owned);
    EXPECT_FALSE(state->capture_status_used);
    EXPECT_EQ(state->capture_device_status, nullptr);
    ASSERT_TRUE(recording.finish().value().wait().ok());
    auto fresh = QueueSequence::begin(queue).value();
    fresh.record(cuda::detail::acquire_device_status(queue, &word));
    ASSERT_TRUE(fresh.finish().value().wait().ok());
    ASSERT_TRUE(context.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
