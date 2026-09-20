#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
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
  Tensor(const compute::TensorDesc &description,
         mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(description, buffer.view()).value()) {}
  template <class T> hyle::Status write(std::span<const T> values) {
    return buffer.view().write(std::as_bytes(values));
  }
};

TEST(CudaDefaultGraph,
     FloatL2AndMaskedHalfProbabilitiesReplayIndependentInputs) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    for (const std::int64_t rows : {1, 17, 23}) {
      constexpr std::int64_t width = 128;
      constexpr float epsilon = 1e-6F;
      const auto shape = std::array{rows, width};
      const auto desc =
          compute::TensorDesc::dense(compute::DataType::kBFloat16, shape)
              .value();
      const auto wide_desc =
          compute::TensorDesc::dense(compute::DataType::kFloat32, shape)
              .value();
      const auto sum_desc =
          compute::TensorDesc::dense(compute::DataType::kFloat32,
                                     std::array<std::int64_t, 2>{rows, 1})
              .value();
      const auto mask_desc =
          compute::TensorDesc::dense(compute::DataType::kBool, shape).value();
      const compute::Reduce sum_op{
          .kind = compute::ReductionKind::kSum,
          .axes = compute::Axes::from(std::array<std::int32_t, 1>{-1}).value(),
          .keep_dimensions = true,
          .accumulation_type = compute::DataType::kFloat32};
      const compute::Softmax probability_op{
          .scale = 0.5F, .mask = compute::MaskKind::kBoolean};
      auto builder = graph::GraphBuilder::create().value();
      const auto input_id = builder.input(desc).value();
      const auto mask_id = builder.input(mask_desc).value();
      const auto wide =
          builder.add(compute::Cast{compute::DataType::kFloat32}, input_id)
              .value();
      const auto squared =
          builder
              .add(compute::Binary{compute::BinaryKind::kMultiply}, wide, wide)
              .value();
      const auto sum_id = builder.add(sum_op, squared).value();
      const auto eps =
          builder
              .add(compute::Fill{
                  compute::TensorShape::create(std::span<const std::int64_t>{})
                      .value(),
                  compute::DataType::kFloat32, compute::ScalarValue(epsilon)})
              .value();
      const auto stable_sum =
          builder.add(compute::Binary{}, sum_id, eps).value();
      const auto inverse =
          builder.add(compute::Unary{compute::UnaryKind::kRsqrt}, stable_sum)
              .value();
      const auto normalized_id =
          builder
              .add(compute::Binary{compute::BinaryKind::kMultiply}, wide,
                   inverse)
              .value();
      const auto probability_id =
          builder.add(probability_op, input_id, mask_id).value();
      ASSERT_TRUE(builder.publish(sum_id).ok());
      ASSERT_TRUE(builder.publish(normalized_id).ok());
      ASSERT_TRUE(builder.publish(probability_id).ok());
      auto logical = std::move(builder).finish().value();
      auto compiled = graph::cuda::compile(
          logical, owner, resource.ref(),
          {.provider_replay = graph::ProviderReplayPolicy::kRequirePrepared});
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      auto direct_sum =
          compute::prepare(owner, sum_op, wide_desc, resource.ref()).value();
      auto direct_probability = compute::prepare(owner, probability_op, desc,
                                                 mask_desc, resource.ref())
                                    .value();
      EXPECT_EQ(direct_sum.provider().id, compute::ggml_cuda::kProviderId);
      EXPECT_EQ(direct_probability.provider().id,
                compute::cuda_native::kProviderId);
      auto sum_slot = direct_sum.create_invocation().value();
      auto probability_slot = direct_probability.create_invocation().value();
      Tensor square_input(wide_desc, resource.ref()),
          direct_sum_output(sum_desc, resource.ref()),
          direct_probability_output(desc, resource.ref());
      std::array inputs{Tensor(desc, resource.ref()),
                        Tensor(desc, resource.ref())};
      std::array masks{Tensor(mask_desc, resource.ref()),
                       Tensor(mask_desc, resource.ref())};
      std::array sums{Tensor(sum_desc, resource.ref()),
                      Tensor(sum_desc, resource.ref())};
      std::array normalized{Tensor(wide_desc, resource.ref()),
                            Tensor(wide_desc, resource.ref())};
      std::array probabilities{Tensor(desc, resource.ref()),
                               Tensor(desc, resource.ref())};
      std::array queues{owner.create_queue().value(),
                        owner.create_queue().value()};
      std::array slots{compiled->executable.create_invocation().value(),
                       compiled->executable.create_invocation().value()};
      for (std::size_t lane = 0; lane < 2; ++lane) {
        ASSERT_TRUE(
            slots[lane].bind(input_id, inputs[lane].view.as_const()).ok());
        ASSERT_TRUE(
            slots[lane].bind(mask_id, masks[lane].view.as_const()).ok());
        ASSERT_TRUE(slots[lane].bind(sum_id, sums[lane].view).ok());
        ASSERT_TRUE(
            slots[lane].bind(normalized_id, normalized[lane].view).ok());
        ASSERT_TRUE(
            slots[lane].bind(probability_id, probabilities[lane].view).ok());
        const auto status = graph::cuda::prepare_replay(
            slots[lane], queues[lane],
            {.mode = graph::ReplayMode::kRequireCapture,
             .host_wait = graph::HostWaitPolicy::kReject});
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(slots[lane].replay_route(), graph::ReplayRoute::kCaptured);
      }
      ASSERT_TRUE(queues[0].reserve_completions(2).ok());
      const auto before = stats.snapshot();
      std::vector<Value> retained_probability;
      std::vector<float> retained_l2;
      for (int repeat = 0; repeat < 3; ++repeat) {
        std::array<compute::Completion, 2> done;
        std::array<std::vector<Value>, 2> values;
        std::array<std::vector<std::uint8_t>, 2> masks_host;
        for (std::size_t lane = 0; lane < 2; ++lane) {
          if (repeat == 2 && lane == 0)
            continue;
          values[lane].resize(static_cast<std::size_t>(rows * width));
          masks_host[lane].resize(values[lane].size());
          for (std::size_t i = 0; i < values[lane].size(); ++i) {
            const auto row = i / width, col = i % width;
            // Small magnitudes distinguish epsilon inside sum(x*x) from an
            // epsilon after sqrt or an accidental divide-by-hidden RMSNorm.
            values[lane][i] = half_test::encode<Value>(
                row == 1 ? 0
                         : (static_cast<int>(i % 31) - 15 + repeat +
                            static_cast<int>(lane)) *
                               0.000125);
            masks_host[lane][i] =
                row == 2 ? 0 : col <= (row + repeat + lane) % width;
          }
          ASSERT_TRUE(inputs[lane].write<Value>(values[lane]).ok());
          ASSERT_TRUE(masks[lane].write<std::uint8_t>(masks_host[lane]).ok());
          auto submitted = slots[lane].submit(queues[lane]);
          ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
          done[lane] = std::move(*submitted);
        }
        for (std::size_t lane = 0; lane < 2; ++lane) {
          if (repeat == 2 && lane == 0) {
            const auto probability =
                half_test::read<Value>(probabilities[lane].buffer);
            for (std::size_t i = 0; i < probability.size(); ++i)
              EXPECT_EQ(probability[i].bits, retained_probability[i].bits);
            EXPECT_EQ(half_test::read<float>(normalized[lane].buffer),
                      retained_l2);
            continue;
          }
          ASSERT_TRUE(done[lane].wait().ok());
          const auto sum = half_test::read<float>(sums[lane].buffer);
          const auto l2 = half_test::read<float>(normalized[lane].buffer);
          const auto probability =
              half_test::read<Value>(probabilities[lane].buffer);
          for (std::size_t row = 0; row < static_cast<std::size_t>(rows);
               ++row) {
            double squared_norm = 0, exponentials = 0;
            for (std::size_t col = 0; col < width; ++col) {
              const auto i = row * width + col;
              const auto value = half_test::decode(values[lane][i]);
              squared_norm += value * value;
              if (masks_host[lane][i])
                exponentials += std::exp(value * 0.5);
            }
            EXPECT_NEAR(sum[row], squared_norm, squared_norm * 4e-6 + 1e-12);
            for (std::size_t col = 0; col < width; ++col) {
              const auto i = row * width + col;
              const auto value = half_test::decode(values[lane][i]);
              const auto expected_l2 =
                  value / std::sqrt(squared_norm + epsilon);
              EXPECT_NEAR(l2[i], expected_l2,
                          std::abs(expected_l2) * 4e-6 + 2e-7);
              half_test::expect_near(probability[i],
                                     exponentials == 0 ? NAN
                                     : masks_host[lane][i]
                                         ? std::exp(value * 0.5) / exponentials
                                         : 0);
            }
          }
          if (lane == 0) {
            std::vector<float> squares(values[0].size());
            for (std::size_t i = 0; i < squares.size(); ++i) {
              const auto value =
                  static_cast<float>(half_test::decode(values[0][i]));
              squares[i] = value * value;
            }
            ASSERT_TRUE(square_input.write<float>(squares).ok());
            auto direct_done =
                sum_slot.submit(queues[0], square_input.view.as_const(),
                                direct_sum_output.view);
            ASSERT_TRUE(direct_done.has_value());
            ASSERT_TRUE(direct_done->wait().ok());
            EXPECT_EQ(half_test::read<float>(direct_sum_output.buffer), sum);
            direct_done = probability_slot.submit(
                queues[0], inputs[0].view.as_const(), masks[0].view.as_const(),
                direct_probability_output.view);
            ASSERT_TRUE(direct_done.has_value());
            ASSERT_TRUE(direct_done->wait().ok());
            const auto direct =
                half_test::read<Value>(direct_probability_output.buffer);
            for (std::size_t i = 0; i < direct.size(); ++i)
              EXPECT_EQ(direct[i].bits, probability[i].bits);
            retained_probability = probability;
            retained_l2 = l2;
          }
        }
      }
      EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
      EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
