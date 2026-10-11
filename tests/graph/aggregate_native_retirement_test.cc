#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <memory>
#include <span>

#include "../compute/completion_internal.h"
#include "plan_state.h"
#include "submission_internal.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/graph/leased_invocation.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace cuda = compute::cuda;

struct MetadataCounter final {
  std::size_t allocations = 0;
  hyle::mem::AllocatorRef ref() noexcept {
    return {
        [](void *owner, std::size_t bytes, std::size_t alignment) noexcept {
          ++static_cast<MetadataCounter *>(owner)->allocations;
          return hyle::mem::system_allocator().allocate(bytes, alignment);
        },
        [](void *, void *pointer, std::size_t bytes, std::size_t alignment) noexcept {
          hyle::mem::system_allocator().deallocate(pointer, bytes, alignment);
        },
        [](void *owner, void *pointer, std::size_t old_bytes, std::size_t bytes,
           std::size_t alignment) noexcept {
          ++static_cast<MetadataCounter *>(owner)->allocations;
          return hyle::mem::system_allocator().reallocate(pointer, old_bytes, bytes, alignment);
        }, this};
  }
};

struct Gate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *opaque) {
    static_cast<Gate *>(opaque)->released.wait(false, std::memory_order_acquire);
  }
  void release() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
  ~Gate() {
    release();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

// Test-only observation decoration over an accepted real native Completion.
// It deliberately supplies no retirement proof. The original CUDA destroy
// and reaper, commands and native state remain responsible for retirement.
struct UnknownObservation final {
  compute::Completion native;
  compute::detail::CompletionRetention retirement{};
  compute::detail::CompletionLifetime forwarded{};
};
const compute::detail::CompletionVTable kUnknownObservation{
    .destroy = [](void *opaque, compute::detail::CompletionLifetime lifetime,
                  hyle::Status hint) noexcept {
      auto *observed = static_cast<UnknownObservation *>(opaque);
      observed->forwarded = lifetime;
      observed->retirement.own = {
          observed,
          [](void *state, const hyle::Status &terminal) noexcept {
            const auto &forwarded = static_cast<UnknownObservation *>(state)->forwarded;
            if (forwarded.valid()) forwarded.terminal(forwarded.state, terminal);
          },
          [](void *state) noexcept {
            auto *owned = static_cast<UnknownObservation *>(state);
            const auto forwarded = owned->forwarded;
            if (forwarded.valid()) forwarded.release(forwarded.state);
            delete owned;
          }};
      auto native = std::move(observed->native);
      compute::detail::CompletionAccess::append_terminal(native, std::move(hint));
      EXPECT_TRUE(compute::detail::CompletionAccess::attach_retention(
          native, observed->retirement).ok());
      // native's destructor can delete observed. Do not access it afterwards.
    },
    .poll = [](void *) noexcept -> hyle::Result<compute::CompletionPoll> {
      return std::unexpected(hyle::Status::Unavailable("test native observation has no retirement proof"));
    },
    .wait = [](void *) noexcept {
      return hyle::Status::Unavailable("test native observation has no retirement proof");
    }};

TEST(GraphNativeRetirement, ObservationErrorKeepsAllAcceptedCublasBranchesAlive) {
  auto device = cuda::device(0).value();
  std::array contexts{
      compute::Context::create(device, compute::cublas::provider()).value(),
      compute::Context::create(device, compute::cublas::provider()).value()};
  auto resource = cuda::MemoryResource::create(contexts[0], cuda::MemoryKind::kDevice).value();
  std::array queues{contexts[0].create_queue().value(), contexts[1].create_queue().value()};
  auto pool = compute::LeasePool::create(resource.ref(), 1024, {.max_blocks = 4}).value();
  auto a_write = pool.allocate(16).value(), b_write = pool.allocate(16).value();
  std::array outputs{pool.allocate(16).value(), pool.allocate(16).value()};
  const std::array<float, 4> a{1, 2, 3, 4}, b{1, 0, 0, 1};
  ASSERT_TRUE(a_write.view()->write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(b_write.view()->write(std::as_bytes(std::span(b))).ok());
  auto a_read = a_write.publish().value(), b_read = b_write.publish().value();
  const auto desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto a_id = builder.input(desc).value(), b_id = builder.input(desc).value();
  const std::array ids{builder.add(compute::MatMul{}, a_id, b_id).value(),
                       builder.add(compute::MatMul{}, a_id, b_id).value()};
  for (auto id : ids) ASSERT_TRUE(builder.publish(id).ok());
  auto definition = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{resource.ref()}};
  const std::array targets{
      graph::ExecutionTarget{&contexts[0], graph::StorageDomainId{0}},
      graph::ExecutionTarget{&contexts[1], graph::StorageDomainId{0}}};
  const std::array placements{
      graph::NodePlacement{definition.node_id(0).value(), graph::ExecutionTargetId{0},
                           graph::PlacementStrength::kRequired},
      graph::NodePlacement{definition.node_id(1).value(), graph::ExecutionTargetId{1},
                           graph::PlacementStrength::kRequired}};
  auto compiled = graph::compile(definition, {.storage_domains = storage, .targets = targets},
      {.node_placements = placements}).value();
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(a_id, a_read.tensor(desc).value()).ok());
  ASSERT_TRUE(invocation.bind(b_id, b_read.tensor(desc).value()).ok());
  for (std::size_t index = 0; index < ids.size(); ++index)
    ASSERT_TRUE(invocation.bind(ids[index], outputs[index].tensor(desc).value()).ok());
  const std::array bindings{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[0]},
      graph::QueueBinding{graph::ExecutionLaneId{1}, &queues[1]}};
  ASSERT_TRUE(invocation.prepare_replay(bindings).ok());
  auto warm = invocation.submit(bindings).value();
  ASSERT_TRUE(warm.wait().ok());
  std::array<Gate, 2> gates;
  for (std::size_t index = 0; index < gates.size(); ++index) {
    gates[index].stream = cuda::stream(queues[index]).value();
    ASSERT_EQ(cudaLaunchHostFunc(gates[index].stream, &Gate::wait, &gates[index]), cudaSuccess);
  }
  auto uses = compute::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 4}).value();
  const std::array output_storage{outputs[0].view().value(), outputs[1].view().value()};
  auto use = uses.begin().value();
  ASSERT_TRUE(use.read(a_read).ok()); ASSERT_TRUE(use.read(b_read).ok());
  for (const auto &output : outputs) ASSERT_TRUE(use.write(output).ok());
  // Preserve a pre-existing child attachment when the aggregate is built.
  auto child_uses = compute::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto child_use = child_uses.begin().value();
  ASSERT_TRUE(child_use.read(a_read).ok());
  auto cursor = graph::detail::SubmissionCursor::begin(invocation, bindings).value();
  auto *state = graph::detail::InvocationAccess::state(invocation);
  while (!cursor.done()) {
    ASSERT_TRUE(cursor.advance(true).has_value());
    if (child_use.valid()) {
      for (auto &native : state->segment_completions) {
        if (native.valid()) {
          ASSERT_TRUE(child_use.attach(native).ok());
          break;
        }
      }
    }
  }
  ASSERT_FALSE(child_use.valid());
  auto completion = cursor.finish().value();
  cursor = {};
  ASSERT_TRUE(use.attach(completion).ok());
  ASSERT_EQ(state->aggregate_completion.completion_count, 2u);
  for (std::size_t index = 0; index < 2; ++index) {
    auto &native = state->segment_completions[state->aggregate_segments[index]];
    auto *observed = new UnknownObservation{std::move(native)};
    native = compute::detail::CompletionAccess::make(&kUnknownObservation, observed);
  }
  EXPECT_FALSE(completion.poll().has_value());
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_EQ(uses.active_batches(), 1u);
  EXPECT_EQ(pool.stats().active_uses, 5u);
  EXPECT_EQ(child_uses.active_batches(), 1u);
  EXPECT_FALSE(outputs[0].publish().has_value());
  invocation = {}; completion = {};
  a_read.reset(); b_read.reset();
  EXPECT_EQ(pool.stats().live_blocks, 4u);
  for (auto &gate : gates) gate.release();
  ASSERT_TRUE(pool.drain().ok());
  ASSERT_TRUE(uses.drain().ok());
  ASSERT_TRUE(child_uses.drain().ok());
  EXPECT_EQ(pool.stats().live_blocks, 2u);
  // Observational failure poisons publication but must not alter native math.
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    EXPECT_FALSE(outputs[index].publish().has_value());
    std::array<float, 4> observed{};
    ASSERT_TRUE(output_storage[index].read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_EQ(observed, a);
  }
}
TEST(GraphNativeRetirement, LeasedGraphOwnsBindingsAndOldResultsPreventReplay) {
  auto owner = compute::Context::create(cuda::device(0).value(), compute::cublas::provider()).value();
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  MetadataCounter metadata;
  auto pool = compute::LeasePool::create(resource.ref(), 1024,
      {.max_blocks = 3, .allocator = metadata.ref()}).value();
  auto a_write = pool.allocate(16).value(), b_write = pool.allocate(16).value();
  auto destination = pool.allocate(16).value();
  const std::array<float, 4> a{1, 2, 3, 4}, identity{1, 0, 0, 1};
  ASSERT_TRUE(a_write.view()->write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(b_write.view()->write(std::as_bytes(std::span(identity))).ok());
  auto left = a_write.publish().value(), right = b_write.publish().value();
  const auto desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto a_id = builder.input(desc).value(), b_id = builder.input(desc).value();
  const auto product = builder.add(compute::MatMul{}, a_id, b_id).value();
  ASSERT_TRUE(builder.publish(product).ok());
  ASSERT_TRUE(builder.publish(a_id).ok()); // source-alias result
  auto definition = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{resource.ref()}};
  const std::array targets{graph::ExecutionTarget{&owner, graph::StorageDomainId{0}}};
  auto compiled = graph::compile(definition, {.storage_domains = storage, .targets = targets}).value();
  auto invocation = graph::LeasedInvocation::create(compiled.executable,
      {.allocator = metadata.ref()}).value();
  ASSERT_TRUE(invocation.bind(a_id, desc, left).ok());
  ASSERT_TRUE(invocation.bind(b_id, desc, right).ok());
  const auto wrong = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 4}).value();
  EXPECT_FALSE(invocation.bind(product, wrong, std::move(destination)).ok());
  ASSERT_TRUE(destination.valid());
  ASSERT_TRUE(invocation.bind(product, desc, std::move(destination)).ok());
  EXPECT_FALSE(destination.valid());
  EXPECT_FALSE(invocation.output(product).has_value());
  left.reset(); right.reset(); // bindings own the only business references
  auto warm = invocation.submit(queue).value();
  ASSERT_TRUE(warm.wait().ok());
  const auto cold_allocations = metadata.allocations;
  for (int iteration = 0; iteration < 8; ++iteration) {
    auto result = invocation.output(product).value();
    result = result;
    auto old_result = result;
    std::array<float, 4> observed{};
    ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_EQ(observed, a);
    EXPECT_FALSE(invocation.submit(queue).has_value());
    EXPECT_TRUE(invocation.reset().IsUnavailable());
    auto exported = result.storage_lease().value();
    result.reset(); old_result.reset();
    EXPECT_FALSE(invocation.submit(queue).has_value()); // independent read lease
    exported.reset();
    auto replay = invocation.submit(queue).value();
    ASSERT_TRUE(replay.wait().ok());
  }
  EXPECT_EQ(metadata.allocations, cold_allocations);
  Gate gate;
  gate.stream = cuda::stream(queue).value();
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto pending = invocation.submit(queue).value();
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_FALSE(invocation.output(product).has_value());
  EXPECT_TRUE(invocation.close().IsUnavailable());
  EXPECT_FALSE(invocation.submit(queue).has_value());
  // An unobserved held Completion must not prevent this adapter from draining.
  EXPECT_EQ(pool.stats().active_uses, 3u);
  gate.release();
  ASSERT_TRUE(invocation.drain().ok());
  ASSERT_TRUE(pending.wait().ok());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(metadata.allocations, cold_allocations);
  auto alias = invocation.output(a_id).value();
  auto aliased_storage = alias.storage_lease().value();
  invocation = {};
  alias.reset();
  EXPECT_EQ(pool.stats().live_blocks, 1u);
  std::array<float, 4> observed{};
  ASSERT_TRUE(aliased_storage.view()->read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, a);
  aliased_storage.reset();
  EXPECT_EQ(pool.stats().live_blocks, 0u);
}

TEST(GraphNativeRetirement, LeasedPrivateOutputRetainsOriginalWorkspaceAfterAdapterDestruction) {
  auto owner = compute::Context::create(cuda::device(0).value(), compute::cublas::provider()).value();
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  auto pool = compute::LeasePool::create(resource.ref(), 512, {.max_blocks = 2}).value();
  auto a_write = pool.allocate(16).value(), b_write = pool.allocate(16).value();
  const std::array<float, 4> a{1, 2, 3, 4}, identity{1, 0, 0, 1};
  ASSERT_TRUE(a_write.view()->write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(b_write.view()->write(std::as_bytes(std::span(identity))).ok());
  auto left = a_write.publish().value(), right = b_write.publish().value();
  const auto desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto a_id = builder.input(desc).value(), b_id = builder.input(desc).value();
  const auto product = builder.add(compute::MatMul{}, a_id, b_id).value();
  const auto slice = builder.add(compute::View::slice(0, 1, 1).value(), product).value();
  ASSERT_TRUE(builder.publish(slice).ok());
  auto definition = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{resource.ref()}};
  const std::array targets{graph::ExecutionTarget{&owner, graph::StorageDomainId{0}}};
  auto compiled = graph::compile(definition, {.storage_domains = storage, .targets = targets}).value();
  EXPECT_EQ(compiled.executable.output_ownership().value(), graph::OutputOwnership::kInvocation);
  auto invocation = graph::LeasedInvocation::create(compiled.executable).value();
  ASSERT_TRUE(invocation.bind(a_id, desc, left).ok());
  ASSERT_TRUE(invocation.bind(b_id, desc, right).ok());
  auto completion = invocation.submit(queue).value();
  ASSERT_TRUE(completion.wait().ok());
  auto result = invocation.output(slice).value();
  EXPECT_FALSE(result.storage_lease().has_value());
  left.reset(); right.reset(); invocation = {}; compiled = {};
  std::array<float, 2> observed{};
  ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, (std::array<float, 2>{3, 4}));
  EXPECT_EQ(pool.stats().live_blocks, 2u);
  result.reset();
  EXPECT_EQ(pool.stats().live_blocks, 0u);
}
#if defined(HYLE_TESTING)
TEST(GraphNativeRetirement, LeasedAcceptedPrefixSurvivesDependencyObservationRejection) {
  auto device = cuda::device(0).value();
  std::array contexts{
      compute::Context::create(device, compute::cublas::provider()).value(),
      compute::Context::create(device, compute::cublas::provider()).value()};
  auto resource = cuda::MemoryResource::create(contexts[0], cuda::MemoryKind::kDevice).value();
  std::array queues{contexts[0].create_queue().value(), contexts[1].create_queue().value()};
  auto pool = compute::LeasePool::create(resource.ref(), 1024, {.max_blocks = 3}).value();
  auto left_write = pool.allocate(16).value(), right_write = pool.allocate(16).value();
  auto destination = pool.allocate(16).value();
  const std::array<float, 4> a{1, 2, 3, 4}, b{1, 0, 0, 1};
  ASSERT_TRUE(left_write.view()->write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(right_write.view()->write(std::as_bytes(std::span(b))).ok());
  auto left = left_write.publish().value(), right = right_write.publish().value();
  const auto desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto a_id = builder.input(desc).value(), b_id = builder.input(desc).value();
  const auto first = builder.add(compute::MatMul{}, a_id, b_id).value();
  const auto output = builder.add(compute::MatMul{}, first, b_id).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto definition = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{resource.ref()}};
  const std::array targets{
      graph::ExecutionTarget{&contexts[0], graph::StorageDomainId{0}},
      graph::ExecutionTarget{&contexts[1], graph::StorageDomainId{0}}};
  const std::array placements{
      graph::NodePlacement{definition.node_id(0).value(), graph::ExecutionTargetId{0},
                           graph::PlacementStrength::kRequired},
      graph::NodePlacement{definition.node_id(1).value(), graph::ExecutionTargetId{1},
                           graph::PlacementStrength::kRequired}};
  auto compiled = graph::compile(definition, {.storage_domains = storage, .targets = targets},
      {.node_placements = placements}).value();
  auto invocation = graph::LeasedInvocation::create(compiled.executable).value();
  ASSERT_TRUE(invocation.bind(a_id, desc, left).ok());
  ASSERT_TRUE(invocation.bind(b_id, desc, right).ok());
  ASSERT_TRUE(invocation.bind(output, desc, std::move(destination)).ok());
  const std::array bindings{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[0]},
      graph::QueueBinding{graph::ExecutionLaneId{1}, &queues[1]}};
  ASSERT_TRUE(invocation.prepare_replay(bindings).ok());
  const auto report = invocation.replay_report().value();
  EXPECT_GT(report.device_wait_dependency_count + report.host_wait_dependency_count, 0u);
  auto warm = invocation.submit(bindings).value();
  ASSERT_TRUE(warm.wait().ok());
  Gate gate;
  gate.stream = cuda::stream(queues[0]).value();
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  struct HookScope final {
    graph::detail::AcceptedCompletionFn previous;
    HookScope() : previous(graph::detail::set_accepted_observation_hook(
        [](void *, std::size_t segment, compute::Completion &native) noexcept {
          if (segment != 0) return;
          auto *observed = new UnknownObservation{std::move(native)};
          native = compute::detail::CompletionAccess::make(&kUnknownObservation, observed);
        })) {}
    ~HookScope() { graph::detail::set_accepted_observation_hook(previous); }
  };
  compute::Completion completion;
  {
    HookScope fault;
    auto submitted = invocation.submit(bindings);
    ASSERT_TRUE(submitted.has_value());
    completion = std::move(*submitted);
  }
  struct RootLifetime final {
    std::atomic<unsigned> releases{0};
    std::atomic<bool> failed{false};
  };
  struct AttachedRoot final {
    std::shared_ptr<RootLifetime> marker;
    compute::detail::CompletionLifetime lifetime;
  };
  auto root = std::make_shared<RootLifetime>();
  auto *root_owner = new AttachedRoot{root, {}};
  root_owner->lifetime = {
      root_owner,
      [](void *opaque, const hyle::Status &terminal) noexcept {
        static_cast<AttachedRoot *>(opaque)->marker->failed.store(!terminal.ok());
      },
      [](void *opaque) noexcept {
        auto *owner = static_cast<AttachedRoot *>(opaque);
        owner->marker->releases.fetch_add(1);
        delete owner;
      }};
  ASSERT_TRUE(compute::detail::CompletionAccess::attach_lifetime(completion, root_owner->lifetime).ok());
  auto observed = completion.poll();
  ASSERT_FALSE(observed.has_value());
  EXPECT_EQ(root->releases.load(), 0u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_EQ(pool.stats().active_uses, 3u);
  EXPECT_FALSE(invocation.output(output).has_value());
  EXPECT_TRUE(invocation.close().IsUnavailable());
  left.reset(); right.reset(); completion = {};
  EXPECT_EQ(pool.stats().live_blocks, 3u);
  gate.release();
  auto retired = invocation.drain();
  EXPECT_FALSE(retired.ok());
  EXPECT_EQ(retired.code(), observed.error().code());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(root->releases.load(), 1u);
  EXPECT_TRUE(root->failed.load());
  EXPECT_FALSE(invocation.output(output).has_value());
  invocation = {};
  EXPECT_EQ(pool.stats().live_blocks, 0u);
}
#endif
} // namespace
