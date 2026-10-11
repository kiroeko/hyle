#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <span>

#include "../compute/completion_internal.h"
#include "plan_state.h"
#include "submission_internal.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/resource_leases.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace cuda = compute::cuda;

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
} // namespace
