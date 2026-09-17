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

void expect_same_bits(std::span<const Value> actual,
                      std::span<const Value> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    ASSERT_EQ(actual[index].bits, expected[index].bits) << index;
}

TEST(CudaDefaultGraph,
     CheckedEmbeddingNormResidualUsesDefaultDirectAndCapturedSlots) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    constexpr std::int64_t vocabulary = 257;
    for (std::int64_t width : {128, 129, 2048}) {
      for (std::int64_t rows : {1, 17, 23, 128}) {
        SCOPED_TRACE(::testing::Message() << rows << '/' << width);
        const auto table_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array{vocabulary, width})
                .value();
        const auto index_desc = compute::TensorDesc::dense(
                                    compute::DataType::kInt32, std::array{rows})
                                    .value();
        const auto output_desc =
            compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                       std::array{rows, width})
                .value();
        const auto weight_desc =
            compute::TensorDesc::dense(compute::DataType::kFloat32,
                                       std::array{width})
                .value();
        auto builder = graph::GraphBuilder::create().value();
        const auto table_id = builder.input(table_desc).value();
        const auto indices_id = builder.input(index_desc).value();
        const auto weights_id = builder.input(weight_desc).value();
        const auto gathered_id =
            builder.add(compute::Gather{}, table_id, indices_id).value();
        const auto norm_id =
            builder.add(compute::RmsNorm{}, gathered_id, weights_id).value();
        const auto output_id =
            builder.add(compute::Binary{}, gathered_id, norm_id).value();
        ASSERT_TRUE(builder.publish(gathered_id).ok());
        ASSERT_TRUE(builder.publish(norm_id).ok());
        ASSERT_TRUE(builder.publish(output_id).ok());
        auto logical = std::move(builder).finish().value();
        auto compiled = graph::cuda::compile(logical, owner, resource.ref());
        ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
        Tensor table(table_desc, resource.ref()),
            weights(weight_desc, resource.ref());
        std::vector<Value> table_values(
            static_cast<std::size_t>(vocabulary * width));
        for (std::size_t i = 0; i < table_values.size(); ++i)
          table_values[i] = half_test::encode<Value>(
              (static_cast<int>(i % 97) - 48) * 0.0625);
        std::vector<float> weight_values(static_cast<std::size_t>(width));
        for (std::size_t i = 0; i < weight_values.size(); ++i)
          weight_values[i] = 1.0F + (static_cast<int>(i % 13) - 6) * 0.031F;
        ASSERT_TRUE(table.write<Value>(table_values).ok());
        ASSERT_TRUE(weights.write<float>(weight_values).ok());
        auto gather_program =
            compute::prepare(owner, compute::Gather{}, table_desc, index_desc,
                             resource.ref())
                .value();
        auto norm_program =
            compute::prepare(owner, compute::RmsNorm{}, output_desc,
                             weight_desc, resource.ref())
                .value();
        auto add_program =
            compute::prepare(owner, compute::Binary{}, output_desc, output_desc,
                             resource.ref())
                .value();
        EXPECT_EQ(gather_program.provider().id,
                  compute::cuda_native::kProviderId);
        auto direct_gather_slot = gather_program.create_invocation().value();
        auto direct_norm_slot = norm_program.create_invocation().value();
        auto direct_add_slot = add_program.create_invocation().value();
        Tensor direct_gather(output_desc, resource.ref()),
            direct_norm(output_desc, resource.ref()),
            direct_output(output_desc, resource.ref());
        std::array indices{Tensor(index_desc, resource.ref()),
                           Tensor(index_desc, resource.ref())};
        std::array gathered{Tensor(output_desc, resource.ref()),
                            Tensor(output_desc, resource.ref())};
        std::array normals{Tensor(output_desc, resource.ref()),
                           Tensor(output_desc, resource.ref())};
        std::array outputs{Tensor(output_desc, resource.ref()),
                           Tensor(output_desc, resource.ref())};
        std::array queues{owner.create_queue().value(),
                          owner.create_queue().value()};
        std::array slots{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
        for (std::size_t lane = 0; lane < 2; ++lane) {
          ASSERT_TRUE(slots[lane].bind(table_id, table.view.as_const()).ok());
          ASSERT_TRUE(
              slots[lane].bind(indices_id, indices[lane].view.as_const()).ok());
          ASSERT_TRUE(
              slots[lane].bind(weights_id, weights.view.as_const()).ok());
          ASSERT_TRUE(slots[lane].bind(gathered_id, gathered[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(norm_id, normals[lane].view).ok());
          ASSERT_TRUE(slots[lane].bind(output_id, outputs[lane].view).ok());
          const auto captured = graph::cuda::prepare_replay(
              slots[lane], queues[lane],
              {.mode = graph::ReplayMode::kRequireCapture,
               .host_wait = graph::HostWaitPolicy::kReject});
          ASSERT_TRUE(captured.ok()) << captured.message();
          EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
          ASSERT_TRUE(queues[lane].reserve_completions(4).ok());
        }
        const auto before = stats.snapshot();
        std::vector<Value> retained_gather, retained_norm, retained_output;
        for (int repeat = 0; repeat < 3; ++repeat) {
          std::array<compute::Completion, 2> done;
          std::array<std::vector<std::int32_t>, 2> ids;
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0)
              continue;
            ids[lane].resize(static_cast<std::size_t>(rows));
            for (std::size_t row = 0; row < ids[lane].size(); ++row)
              ids[lane][row] = static_cast<std::int32_t>(
                  (row * 7 + repeat * 31 + lane * 13) % vocabulary);
            ASSERT_TRUE(indices[lane].write<std::int32_t>(ids[lane]).ok());
            auto submitted = slots[lane].submit(queues[lane]);
            ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
            done[lane] = std::move(*submitted);
          }
          for (std::size_t lane = 0; lane < 2; ++lane) {
            if (repeat == 2 && lane == 0) {
              expect_same_bits(gathered[lane].read<Value>(), retained_gather);
              expect_same_bits(normals[lane].read<Value>(), retained_norm);
              expect_same_bits(outputs[lane].read<Value>(), retained_output);
              continue;
            }
            ASSERT_TRUE(done[lane].wait().ok());
            const auto actual_gather = gathered[lane].read<Value>();
            const auto actual_norm = normals[lane].read<Value>();
            const auto actual_output = outputs[lane].read<Value>();
            for (std::size_t row = 0; row < ids[lane].size(); ++row) {
              double squares = 0;
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col) {
                const auto value = table_values[ids[lane][row] * width + col];
                EXPECT_EQ(actual_gather[row * width + col].bits, value.bits);
                const auto x = half_test::decode(value);
                squares += x * x;
              }
              const double inverse = 1 / std::sqrt(squares / width + 1e-5F);
              for (std::size_t col = 0; col < static_cast<std::size_t>(width);
                   ++col) {
                const auto i = row * width + col;
                const double x = half_test::decode(
                    table_values[ids[lane][row] * width + col]);
                const auto norm =
                    half_test::encode<Value>(x * inverse * weight_values[col]);
                half_test::expect_near(actual_norm[i], half_test::decode(norm));
                half_test::expect_near(actual_output[i],
                                       x + half_test::decode(norm));
              }
            }
            auto dg =
                direct_gather_slot
                    .submit(queues[lane], table.view.as_const(),
                            indices[lane].view.as_const(), direct_gather.view)
                    .value();
            ASSERT_TRUE(dg.wait().ok());
            auto dn = direct_norm_slot
                          .submit(queues[lane], direct_gather.view.as_const(),
                                  weights.view.as_const(), direct_norm.view)
                          .value();
            ASSERT_TRUE(dn.wait().ok());
            auto da =
                direct_add_slot
                    .submit(queues[lane], direct_gather.view.as_const(),
                            direct_norm.view.as_const(), direct_output.view)
                    .value();
            ASSERT_TRUE(da.wait().ok());
            expect_same_bits(direct_gather.read<Value>(), actual_gather);
            expect_same_bits(direct_norm.read<Value>(), actual_norm);
            expect_same_bits(direct_output.read<Value>(), actual_output);
            if (lane == 0) {
              retained_gather = actual_gather;
              retained_norm = actual_norm;
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
    auto builder = graph::GraphBuilder::create().value();
    const auto t = builder.input(td).value(), i = builder.input(id).value();
    const auto o = builder.add(compute::Gather{}, t, i).value();
    ASSERT_TRUE(builder.publish(o).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled =
        graph::cuda::compile(logical, owner, resource.ref()).value();
    Tensor table(td, resource.ref()), indices(id, resource.ref());
    Tensor output(compute::TensorDesc::dense(compute::DataType::kInt32,
                                             std::array<std::int64_t, 1>{2})
                      .value(),
                  resource.ref());
    ASSERT_TRUE(table.write<int>(std::array{11, 12, 13, 14}).ok());
    ASSERT_TRUE(
        indices.write<std::int64_t>(std::array<std::int64_t, 2>{0, 4}).ok());
    ASSERT_TRUE(output.write<int>(std::array{-23, -24}).ok());
    const auto bind = [&](auto &slot) {
      EXPECT_TRUE(slot.bind(t, table.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(i, indices.view.as_const()).ok());
      EXPECT_TRUE(slot.bind(o, output.view).ok());
      // Invalid data is already present: preparing capture must not execute it.
      return graph::cuda::prepare_replay(
          slot, queue,
          {.mode = graph::ReplayMode::kRequireCapture,
           .host_wait = graph::HostWaitPolicy::kReject});
    };
    auto slot = compiled.executable.create_invocation().value();
    ASSERT_TRUE(bind(slot).ok());
    EXPECT_EQ(output.read<int>(), (std::vector{-23, -24}));
    auto failed = slot.submit(queue);
    ASSERT_TRUE(failed.has_value());
    EXPECT_TRUE(failed->wait().IsOutOfRange());
    EXPECT_EQ(output.read<int>(), (std::vector{-23, -24}));
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
    EXPECT_EQ(output.read<int>(), (std::vector{14, 12}));
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
    EXPECT_EQ(output.read<int>(), (std::vector{14, 12}));
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!stats.balanced() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(stats.balanced());
}
TEST(CudaDefaultGraph,
     CheckedGatherErrorsRecoverAndRetainClosedOrAbandonedOwners) {
  failure_case(false);
  failure_case(true);
}
} // namespace
