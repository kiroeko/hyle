#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
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
     CountOffsetsGatherAndFloatGatePrefixesUseCapturedIndependentSlots) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    constexpr std::int64_t width = 128, heads = 16, vocabulary = 1024;
    for (std::int64_t experts : {1, 17, 128, 257}) {
      for (std::int64_t tokens : {1, 17, 23, 128, 129}) {
        SCOPED_TRACE(::testing::Message() << experts << '/' << tokens);
        const auto counts_desc =
            compute::TensorDesc::dense(compute::DataType::kInt32,
                                       std::array{experts})
                .value();
        const auto table_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array{vocabulary, width})
                .value();
        const auto gathered_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array{experts, width})
                .value();
        const auto gates_desc =
            compute::TensorDesc::dense(
                compute::DataType::kFloat32,
                std::array<std::int64_t, 3>{2, tokens, heads})
                .value();
        const auto scales_desc =
            compute::TensorDesc::dense(compute::DataType::kFloat32,
                                       std::array{heads})
                .value();
        auto builder = graph::GraphBuilder::create().value();
        const auto c = builder.input(counts_desc).value(),
                   t = builder.input(table_desc).value(),
                   g = builder.input(gates_desc).value(),
                   s = builder.input(scales_desc).value();
        const auto offset =
            builder.add(compute::Scan{.exclusive = true}, c).value();
        const auto gathered = builder.add(compute::Gather{}, t, offset).value();
        const auto softplus =
            builder.add(compute::Unary{compute::UnaryKind::kSoftplus}, g)
                .value();
        const auto gate =
            builder
                .add(compute::Binary{compute::BinaryKind::kMultiply}, softplus,
                     s)
                .value();
        const auto prefix = builder.add(compute::Scan{.axis = 1}, gate).value();
        for (auto value : {offset, gathered, gate, prefix})
          ASSERT_TRUE(builder.publish(value).ok());
        auto logical = std::move(builder).finish().value();
        auto compiled = graph::cuda::compile(
            logical, owner, resource.ref(),
            {.provider_replay = graph::ProviderReplayPolicy::kRequirePrepared});
        ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
        auto direct_counts =
            compute::prepare(owner, compute::Scan{.exclusive = true},
                             counts_desc, resource.ref())
                .value();
        auto direct_gate = compute::prepare(owner, compute::Scan{.axis = 1},
                                            gates_desc, resource.ref())
                               .value();
        EXPECT_EQ(direct_counts.provider().id,
                  compute::cuda_native::kProviderId);
        EXPECT_EQ(direct_gate.provider().id, compute::cuda_native::kProviderId);
        auto direct_counts_slot = direct_counts.create_invocation().value();
        auto direct_gate_slot = direct_gate.create_invocation().value();
        Tensor table(table_desc, resource.ref()),
            scales(scales_desc, resource.ref()),
            direct_offsets(counts_desc, resource.ref()),
            direct_prefix(gates_desc, resource.ref());
        std::vector<Value> table_values(
            static_cast<std::size_t>(vocabulary * width));
        for (std::size_t index = 0; index < table_values.size(); ++index)
          table_values[index] = half_test::encode<Value>(
              (static_cast<int>(index % 97) - 48) * 0.03125);
        const std::vector<float> scale_values(static_cast<std::size_t>(heads),
                                              -0.5F);
        ASSERT_TRUE(table.write<Value>(table_values).ok());
        ASSERT_TRUE(scales.write<float>(scale_values).ok());
        std::array counts{Tensor(counts_desc, resource.ref()),
                          Tensor(counts_desc, resource.ref())};
        std::array inputs{Tensor(gates_desc, resource.ref()),
                          Tensor(gates_desc, resource.ref())};
        std::array offsets{Tensor(counts_desc, resource.ref()),
                           Tensor(counts_desc, resource.ref())};
        std::array gathers{Tensor(gathered_desc, resource.ref()),
                           Tensor(gathered_desc, resource.ref())};
        std::array gates{Tensor(gates_desc, resource.ref()),
                         Tensor(gates_desc, resource.ref())};
        std::array prefixes{Tensor(gates_desc, resource.ref()),
                            Tensor(gates_desc, resource.ref())};
        std::array queues{owner.create_queue().value(),
                          owner.create_queue().value()};
        std::array slots{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
        for (std::size_t lane = 0; lane < 2; ++lane) {
          ASSERT_TRUE(slots[lane].bind(c, counts[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(t, table.view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(g, inputs[lane].view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(s, scales.view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(offset, offsets[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(gathered, gathers[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(gate, gates[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(prefix, prefixes[lane].view).ok());
          const auto captured = graph::cuda::prepare_replay(
              slots[lane], queues[lane],
              {.mode = graph::ReplayMode::kRequireCapture,
               .host_wait = graph::HostWaitPolicy::kReject});
          ASSERT_TRUE(captured.ok()) << captured.message();
          EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
          ASSERT_TRUE(queues[lane].reserve_completions(4).ok());
        }
        const auto before = stats.snapshot();
        std::vector<std::int32_t> retained_offsets;
        std::vector<Value> retained_gather;
        std::vector<float> retained_prefix;
        for (int repeat = 0; repeat < 3; ++repeat) {
          std::array<compute::Completion, 2> done;
          std::array<std::vector<std::int32_t>, 2> counter_values;
          std::array<std::vector<float>, 2> gate_values;
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0)
              continue;
            counter_values[lane].resize(static_cast<std::size_t>(experts));
            for (std::size_t index = 0; index < counter_values[lane].size();
                 ++index)
              counter_values[lane][index] =
                  static_cast<std::int32_t>((index + repeat + lane) % 4);
            gate_values[lane].resize(
                static_cast<std::size_t>(2 * tokens * heads));
            for (std::size_t index = 0; index < gate_values[lane].size();
                 ++index)
              gate_values[lane][index] =
                  (static_cast<int>((index + repeat * 3 + lane) % 31) - 15) *
                  0.17F;
            ASSERT_TRUE(
                counts[lane].write<std::int32_t>(counter_values[lane]).ok());
            ASSERT_TRUE(inputs[lane].write<float>(gate_values[lane]).ok());
            auto submitted = slots[lane].submit(queues[lane]);
            ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
            done[lane] = std::move(*submitted);
          }
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0) {
              EXPECT_EQ(offsets[lane].read<std::int32_t>(), retained_offsets);
              expect_bits(gathers[lane].read<Value>(), retained_gather);
              EXPECT_EQ(prefixes[lane].read<float>(), retained_prefix);
              continue;
            }
            ASSERT_TRUE(done[lane].wait().ok());
            const auto actual_offsets = offsets[lane].read<std::int32_t>();
            const auto actual_gather = gathers[lane].read<Value>();
            const auto actual_gate = gates[lane].read<float>();
            const auto actual_prefix = prefixes[lane].read<float>();
            std::int64_t running = 0;
            for (std::size_t index = 0; index < actual_offsets.size();
                 ++index) {
              ASSERT_EQ(actual_offsets[index], running);
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col)
                ASSERT_EQ(actual_gather[index * width + col].bits,
                          table_values[running * width + col].bits);
              running += counter_values[lane][index];
            }
            for (std::size_t batch = 0; batch < 2; ++batch)
              for (std::size_t head = 0; head < static_cast<std::size_t>(heads);
                   ++head) {
                double accumulated = 0;
                for (std::size_t token = 0;
                     token < static_cast<std::size_t>(tokens); ++token) {
                  const auto index = (batch * tokens + token) * heads + head;
                  const double x = gate_values[lane][index];
                  const double expected =
                      (std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)))) *
                      scale_values[head];
                  accumulated += expected;
                  EXPECT_NEAR(actual_gate[index], expected,
                              3e-6 * std::abs(expected) + 1e-6);
                  EXPECT_NEAR(actual_prefix[index], accumulated,
                              6e-6 * std::abs(accumulated) + 1e-6);
                }
              }
            auto direct_i =
                direct_counts_slot
                    .submit(queues[lane], counts[lane].view.as_const(),
                            direct_offsets.view)
                    .value();
            ASSERT_TRUE(direct_i.wait().ok());
            EXPECT_EQ(direct_offsets.read<std::int32_t>(), actual_offsets);
            auto direct_f =
                direct_gate_slot
                    .submit(queues[lane], gates[lane].view.as_const(),
                            direct_prefix.view)
                    .value();
            ASSERT_TRUE(direct_f.wait().ok());
            EXPECT_EQ(direct_prefix.read<float>(), actual_prefix);
            if (lane == 0) {
              retained_offsets = actual_offsets;
              retained_gather = actual_gather;
              retained_prefix = actual_prefix;
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
    const auto desc =
        compute::TensorDesc::dense(compute::DataType::kInt32,
                                   std::array<std::int64_t, 2>{2, 3})
            .value();
    auto builder = graph::GraphBuilder::create().value();
    const auto in = builder.input(desc).value(),
               out = builder.add(compute::Scan{}, in).value();
    ASSERT_TRUE(builder.publish(out).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled =
        graph::cuda::compile(logical, owner, resource.ref()).value();
    Tensor input(desc, resource.ref()), output(desc, resource.ref());
    ASSERT_TRUE(input
                    .write<std::int32_t>(
                        std::array<std::int32_t, 6>{1, 2, 3, INT32_MAX, 0, 1})
                    .ok());
    ASSERT_TRUE(output
                    .write<std::int32_t>(
                        std::array<std::int32_t, 6>{-1, -2, -3, -4, -5, -6})
                    .ok());
    const auto bind = [&](auto &slot) {
      EXPECT_TRUE(slot.bind(in, input.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(out, output.view).ok());
      return graph::cuda::prepare_replay(
          slot, queue,
          {.mode = graph::ReplayMode::kRequireCapture,
           .host_wait = graph::HostWaitPolicy::kReject});
    };
    auto slot = compiled.executable.create_invocation().value();
    ASSERT_TRUE(bind(slot).ok());
    EXPECT_EQ(output.read<std::int32_t>(),
              (std::vector{-1, -2, -3, -4, -5, -6}));
    auto failed = slot.submit(queue).value();
    EXPECT_TRUE(failed.wait().IsOutOfRange());
    EXPECT_EQ(output.read<std::int32_t>(),
              (std::vector{-1, -2, -3, -4, -5, -6}));
    auto retired = slot.submit(queue);
    ASSERT_FALSE(retired.has_value());
    EXPECT_TRUE(retired.error().IsFailedPrecondition());
    slot = {};
    auto fresh = compiled.executable.create_invocation().value();
    ASSERT_TRUE(bind(fresh).ok());
    ASSERT_TRUE(input.write<std::int32_t>(std::array{1, 2, 3, 4, 5, 6}).ok());
    auto good = fresh.submit(queue).value();
    ASSERT_TRUE(good.wait().ok());
    EXPECT_EQ(output.read<std::int32_t>(), (std::vector{1, 3, 6, 4, 9, 15}));
    ASSERT_TRUE(input
                    .write<std::int32_t>(
                        std::array<std::int32_t, 6>{0, 0, 0, INT32_MIN, -1, 1})
                    .ok());
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
    if (!abandon) {
      EXPECT_TRUE(pending.wait().IsOutOfRange());
    }
    ASSERT_TRUE(owner.drain().ok());
    EXPECT_EQ(output.read<std::int32_t>(), (std::vector{1, 3, 6, 4, 9, 15}));
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!stats.balanced() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(stats.balanced());
}
TEST(CudaDefaultGraph,
     IntegerScanOverflowPreservesOutputRecoversAndRetainsPendingOwners) {
  failure_case(false);
  failure_case(true);
}
} // namespace
