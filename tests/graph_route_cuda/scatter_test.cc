#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <span>
#include <thread>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;
using Value = hyle::simd::bfloat16_t;
struct Tensor final {
  mem::Buffer buffer;
  compute::TensorView view;
  Tensor(const compute::TensorDesc &desc, mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(desc, buffer.view()).value()) {}
  template <class T> hyle::Status write(std::span<const T> values) {
    return buffer.view().write(std::as_bytes(values));
  }
  template <class T> std::vector<T> read() const {
    std::vector<T> result(buffer.size() / sizeof(T));
    EXPECT_TRUE(
        buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
    return result;
  }
};
void expect_bits(std::span<const Value> actual,
                 std::span<const Value> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    ASSERT_EQ(actual[index].bits, expected[index].bits) << index;
}
TEST(CudaDefaultGraph,
     OrderedScatterStateRoundTripsThroughCapturedGatherNormResidual) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    for (std::int64_t width : {128, 129, 2048}) {
      for (std::int64_t rows : {1, 17, 23, 128}) {
        SCOPED_TRACE(::testing::Message() << rows << '/' << width);
        const auto state_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array<std::int64_t, 2>{17, width})
                .value();
        const auto ids_desc = compute::TensorDesc::dense(
                                  compute::DataType::kInt32, std::array{rows})
                                  .value();
        const auto updates_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array{rows, width})
                .value();
        const auto weight_desc =
            compute::TensorDesc::dense(compute::DataType::kFloat32,
                                       std::array{width})
                .value();
        auto builder = graph::GraphBuilder::create().value();
        const auto s = builder.input(state_desc).value(),
                   i = builder.input(ids_desc).value(),
                   u = builder.input(updates_desc).value(),
                   w = builder.input(weight_desc).value();
        const auto updated =
            builder
                .add(compute::Scatter{.kind = compute::ScatterKind::kAdd}, s, i,
                     u)
                .value();
        const auto gathered =
            builder.add(compute::Gather{}, updated, i).value();
        const auto norm = builder.add(compute::RmsNorm{}, gathered, w).value();
        const auto residual =
            builder.add(compute::Binary{}, gathered, norm).value();
        for (auto value : {updated, gathered, norm, residual})
          ASSERT_TRUE(builder.publish(value).ok());
        auto logical = std::move(builder).finish().value();
        auto compiled = graph::cuda::compile(logical, owner, resource.ref());
        ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
        auto direct =
            compute::prepare(
                owner, compute::Scatter{.kind = compute::ScatterKind::kAdd},
                state_desc, ids_desc, updates_desc, resource.ref())
                .value();
        EXPECT_EQ(direct.provider().id, compute::cuda_native::kProviderId);
        const auto need = direct.workspace_requirement();
        auto scratch =
            mem::Buffer::create(resource.ref(), need.size, need.alignment)
                .value();
        auto direct_slot = direct.create_invocation(scratch.view()).value();
        auto advance =
            compute::prepare(owner, compute::Copy{}, state_desc, resource.ref())
                .value();
        auto advance_slot = advance.create_invocation().value();
        Tensor weights(weight_desc, resource.ref()),
            direct_output(state_desc, resource.ref());
        std::vector<float> weight_values(static_cast<std::size_t>(width));
        for (std::size_t col = 0; col < weight_values.size(); ++col)
          weight_values[col] = 1.0F + (col % 5) * 0.0625F;
        ASSERT_TRUE(weights.write<float>(weight_values).ok());
        std::array sources{Tensor(state_desc, resource.ref()),
                           Tensor(state_desc, resource.ref())};
        std::array indices{Tensor(ids_desc, resource.ref()),
                           Tensor(ids_desc, resource.ref())};
        std::array updates{Tensor(updates_desc, resource.ref()),
                           Tensor(updates_desc, resource.ref())};
        std::array states{Tensor(state_desc, resource.ref()),
                          Tensor(state_desc, resource.ref())};
        std::array gathers{Tensor(updates_desc, resource.ref()),
                           Tensor(updates_desc, resource.ref())};
        std::array norms{Tensor(updates_desc, resource.ref()),
                         Tensor(updates_desc, resource.ref())};
        std::array outputs{Tensor(updates_desc, resource.ref()),
                           Tensor(updates_desc, resource.ref())};
        std::array queues{owner.create_queue().value(),
                          owner.create_queue().value()};
        std::array slots{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
        std::array<std::vector<Value>, 2> expected;
        for (std::size_t lane = 0; lane < 2; ++lane) {
          expected[lane].resize(static_cast<std::size_t>(17 * width));
          for (std::size_t pos = 0; pos < expected[lane].size(); ++pos)
            expected[lane][pos] = half_test::encode<Value>(
                (static_cast<int>(pos % 19) - 9) * 0.125 + lane);
          ASSERT_TRUE(sources[lane].write<Value>(expected[lane]).ok());
          ASSERT_TRUE(slots[lane].bind(s, sources[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(i, indices[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(u, updates[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(w, weights.view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(updated, states[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(gathered, gathers[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(norm, norms[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(residual, outputs[lane].view).ok());
          auto captured = graph::cuda::prepare_replay(
              slots[lane], queues[lane],
              {.mode = graph::ReplayMode::kRequireCapture,
               .host_wait = graph::HostWaitPolicy::kReject});
          ASSERT_TRUE(captured.ok()) << captured.message();
          EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
          ASSERT_TRUE(queues[lane].reserve_completions(4).ok());
        }
        const auto before = stats.snapshot();
        std::vector<Value> retained_state, retained_output;
        for (int repeat = 0; repeat < 3; ++repeat) {
          std::array<compute::Completion, 2> done;
          std::array<std::vector<std::int32_t>, 2> ids;
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0)
              continue;
            ids[lane].resize(static_cast<std::size_t>(rows));
            std::vector<Value> changes(static_cast<std::size_t>(rows * width));
            for (std::size_t row = 0; row < ids[lane].size(); ++row) {
              ids[lane][row] =
                  static_cast<std::int32_t>((row % 5 + repeat * 3 + lane) % 17);
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col) {
                const auto pos = row * width + col;
                changes[pos] = half_test::encode<Value>(
                    (static_cast<int>(pos % 11) - 5) * 0.03125);
                const auto destination = ids[lane][row] * width + col;
                expected[lane][destination] = half_test::encode<Value>(
                    static_cast<float>(
                        half_test::decode(expected[lane][destination])) +
                    static_cast<float>(half_test::decode(changes[pos])));
              }
            }
            ASSERT_TRUE(indices[lane].write<std::int32_t>(ids[lane]).ok());
            ASSERT_TRUE(updates[lane].write<Value>(changes).ok());
            auto submitted = slots[lane].submit(queues[lane]);
            ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
            done[lane] = std::move(*submitted);
          }
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0) {
              expect_bits(states[lane].read<Value>(), retained_state);
              expect_bits(outputs[lane].read<Value>(), retained_output);
              continue;
            }
            ASSERT_TRUE(done[lane].wait().ok());
            expect_bits(states[lane].read<Value>(), expected[lane]);
            const auto actual_gather = gathers[lane].read<Value>();
            const auto actual_norm = norms[lane].read<Value>();
            const auto actual_output = outputs[lane].read<Value>();
            for (std::size_t row = 0; row < ids[lane].size(); ++row) {
              double squares = 0;
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col) {
                const auto value = expected[lane][ids[lane][row] * width + col];
                ASSERT_EQ(actual_gather[row * width + col].bits, value.bits);
                const auto x = half_test::decode(value);
                squares += x * x;
              }
              const double inverse = 1 / std::sqrt(squares / width + 1e-5F);
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col) {
                const auto pos = row * width + col;
                const double x = half_test::decode(
                    expected[lane][ids[lane][row] * width + col]);
                const auto norm_value =
                    half_test::encode<Value>(x * inverse * weight_values[col]);
                half_test::expect_near(actual_norm[pos],
                                       half_test::decode(norm_value));
                half_test::expect_near(actual_output[pos],
                                       x + half_test::decode(norm_value));
              }
            }
            auto direct_done =
                direct_slot
                    .submit(queues[lane],
                            std::array{sources[lane].view.as_const(),
                                       indices[lane].view.as_const(),
                                       updates[lane].view.as_const()},
                            std::array{direct_output.view})
                    .value();
            ASSERT_TRUE(direct_done.wait().ok());
            expect_bits(direct_output.read<Value>(), expected[lane]);
            // Carry the actual device result into the next state step, with the
            // same bindings/captured program and no host state reconstruction.
            auto copied =
                advance_slot
                    .submit(queues[lane], states[lane].view.as_const(),
                            sources[lane].view)
                    .value();
            ASSERT_TRUE(copied.wait().ok());
            if (lane == 0) {
              retained_state = states[lane].read<Value>();
              retained_output = actual_output;
            }
          }
        }
        EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
        EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
struct Gate final {
  std::atomic<bool> released{false}, timed_out{false};
  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<Gate *>(raw);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.released.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        gate.timed_out.store(true);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  void release() { released.store(true, std::memory_order_release); }
  ~Gate() { EXPECT_FALSE(timed_out.load()); }
};
void failure_case(bool abandon) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    auto queue = owner.create_queue().value();
    const auto td = compute::TensorDesc::dense(compute::DataType::kInt32,
                                               std::array<std::int64_t, 1>{4})
                        .value();
    const auto id = compute::TensorDesc::dense(compute::DataType::kInt64,
                                               std::array<std::int64_t, 1>{2})
                        .value();
    const auto ud = compute::TensorDesc::dense(compute::DataType::kInt32,
                                               std::array<std::int64_t, 1>{2})
                        .value();
    auto builder = graph::GraphBuilder::create().value();
    const auto t = builder.input(td).value(), i = builder.input(id).value(),
               u = builder.input(ud).value();
    const auto o = builder.add(compute::Scatter{}, t, i, u).value();
    ASSERT_TRUE(builder.publish(o).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled =
        graph::cuda::compile(logical, owner, resource.ref()).value();
    Tensor source(td, resource.ref()), indices(id, resource.ref()),
        updates(ud, resource.ref()), output(td, resource.ref());
    ASSERT_TRUE(source.write<int>(std::array{11, 12, 13, 14}).ok());
    ASSERT_TRUE(updates.write<int>(std::array{31, 32}).ok());
    ASSERT_TRUE(
        indices.write<std::int64_t>(std::array<std::int64_t, 2>{0, 4}).ok());
    ASSERT_TRUE(output.write<int>(std::array{-23, -24, -25, -26}).ok());
    const auto bind = [&](auto &slot) {
      EXPECT_TRUE(slot.bind(t, source.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(i, indices.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(u, updates.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(o, output.view).ok());
      return graph::cuda::prepare_replay(
          slot, queue,
          {.mode = graph::ReplayMode::kRequireCapture,
           .host_wait = graph::HostWaitPolicy::kReject});
    };
    const auto caller_bytes =
        resource.ref().resource().stats().value().used_bytes;
    auto slot = compiled.executable.create_invocation().value();
    ASSERT_TRUE(bind(slot).ok());
    EXPECT_EQ(output.read<int>(), (std::vector{-23, -24, -25, -26}));
    auto failed = slot.submit(queue).value();
    EXPECT_TRUE(failed.wait().IsOutOfRange());
    EXPECT_EQ(output.read<int>(), (std::vector{-23, -24, -25, -26}));
    auto retired = slot.submit(queue);
    ASSERT_FALSE(retired.has_value());
    EXPECT_TRUE(retired.error().IsFailedPrecondition());
    slot = {};
    auto fresh = compiled.executable.create_invocation().value();
    ASSERT_TRUE(bind(fresh).ok());
    ASSERT_TRUE(
        indices.write<std::int64_t>(std::array<std::int64_t, 2>{3, 1}).ok());
    auto good = fresh.submit(queue).value();
    ASSERT_TRUE(good.wait().ok());
    EXPECT_EQ(output.read<int>(), (std::vector{11, 32, 13, 31}));
    ASSERT_TRUE(
        indices.write<std::int64_t>(std::array<std::int64_t, 2>{-1, 1}).ok());
    Gate gate;
    const auto stream = cuda::stream(queue).value();
    struct Release final {
      Gate &gate;
      cudaStream_t stream;
      ~Release() {
        gate.release();
        (void)cudaStreamSynchronize(stream);
      }
    } release{gate, stream};
    ASSERT_EQ(cudaLaunchHostFunc(stream, &Gate::wait, &gate), cudaSuccess);
    auto pending = fresh.submit(queue).value();
    EXPECT_EQ(pending.poll().value(), compute::CompletionPoll::kPending);
    fresh = {};
    compiled.executable = {};
    ASSERT_TRUE(owner.close().ok());
    if (abandon)
      pending = {};
    gate.release();
    if (!abandon)
      EXPECT_TRUE(pending.wait().IsOutOfRange());
    ASSERT_TRUE(owner.drain().ok());
    EXPECT_EQ(output.read<int>(), (std::vector{11, 32, 13, 31}));
    failed = {};
    good = {};
    pending = {};
    // The caller owns the workspace resource. An abandoned completion releases
    // its retained Graph invocation asynchronously, even after GPU drain.
    const auto cleanup_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (resource.ref().resource().stats().value().used_bytes !=
               caller_bytes &&
           std::chrono::steady_clock::now() < cleanup_deadline)
      std::this_thread::yield();
    EXPECT_EQ(resource.ref().resource().stats().value().used_bytes,
              caller_bytes);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!stats.balanced() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(stats.balanced());
}
TEST(CudaDefaultGraph,
     CheckedScatterOwnsSortGraphThroughFailureCloseAndAbandon) {
  failure_case(false);
  failure_case(true);
}
} // namespace
