#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
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
     MixedFloatNormAndSwishGateCaptureReusableIndependentState) {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    auto owner =
        graph::cuda::create_context({.metadata_allocator = stats.get()})
            .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    for (const std::int64_t rows : {1, 17}) {
      for (const std::int64_t width : {128, 2048}) {
        for (const bool layer : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << rows << '/' << width << '/' << layer);
          const auto desc =
              compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                         std::array{rows, width})
                  .value();
          const auto parameter_desc =
              compute::TensorDesc::dense(compute::DataType::kFloat32,
                                         std::array{width})
                  .value();
          auto builder = graph::GraphBuilder::create().value();
          const auto input_id = builder.input(desc).value();
          const auto gate_id = builder.input(desc).value();
          const auto gamma_id = builder.input(parameter_desc).value();
          const auto one =
              builder
                  .add(compute::Fill{compute::TensorShape::create(
                                         std::span<const std::int64_t>{})
                                         .value(),
                                     compute::DataType::kFloat32,
                                     compute::ScalarValue(1.0F)})
                  .value();
          const auto weight_id =
              builder.add(compute::Binary{}, gamma_id, one).value();
          const auto norm_id =
              layer ? builder.add(compute::LayerNorm{}, input_id, weight_id)
                          .value()
                    : builder.add(compute::RmsNorm{}, input_id, weight_id)
                          .value();
          const auto norm_wide =
              builder.add(compute::Cast{compute::DataType::kFloat32}, norm_id)
                  .value();
          const auto gate_wide =
              builder.add(compute::Cast{compute::DataType::kFloat32}, gate_id)
                  .value();
          const auto activated =
              builder.add(compute::Unary{compute::UnaryKind::kSilu}, gate_wide)
                  .value();
          const auto multiplied =
              builder
                  .add(compute::Binary{compute::BinaryKind::kMultiply},
                       norm_wide, activated)
                  .value();
          const auto output_id =
              builder
                  .add(compute::Cast{compute::DataType::kBFloat16}, multiplied)
                  .value();
          ASSERT_TRUE(builder.publish(norm_id).ok());
          ASSERT_TRUE(builder.publish(output_id).ok());
          auto logical = std::move(builder).finish().value();
          auto compiled = graph::cuda::compile(logical, owner, resource.ref());
          ASSERT_TRUE(compiled.has_value()) << compiled.error().message();

          std::vector<float> gamma(static_cast<std::size_t>(width)),
              effective(gamma.size());
          for (std::size_t i = 0; i < gamma.size(); ++i) {
            gamma[i] = (static_cast<int>(i % 17) - 8) * 0.0031F;
            effective[i] = 1.0F + gamma[i];
          }
          Tensor parameters(parameter_desc, resource.ref()),
              direct_scale(parameter_desc, resource.ref());
          ASSERT_TRUE(parameters.write<float>(gamma).ok());
          ASSERT_TRUE(direct_scale.write<float>(effective).ok());
          auto direct =
              layer ? compute::prepare(owner, compute::LayerNorm{}, desc,
                                       parameter_desc, resource.ref())
                    : compute::prepare(owner, compute::RmsNorm{}, desc,
                                       parameter_desc, resource.ref());
          ASSERT_TRUE(direct.has_value()) << direct.error().message();
          EXPECT_EQ(direct->provider().id, compute::cuda_native::kProviderId);
          auto direct_slot = direct->create_invocation().value();
          Tensor direct_output(desc, resource.ref());
          std::array inputs{Tensor(desc, resource.ref()),
                            Tensor(desc, resource.ref())};
          std::array gates{Tensor(desc, resource.ref()),
                           Tensor(desc, resource.ref())};
          std::array normals{Tensor(desc, resource.ref()),
                             Tensor(desc, resource.ref())};
          std::array outputs{Tensor(desc, resource.ref()),
                             Tensor(desc, resource.ref())};
          std::array queues{owner.create_queue().value(),
                            owner.create_queue().value()};
          std::array slots{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
          for (std::size_t lane = 0; lane < 2; ++lane) {
            ASSERT_TRUE(
                slots[lane].bind(input_id, inputs[lane].view.as_const()).ok());
            ASSERT_TRUE(
                slots[lane].bind(gate_id, gates[lane].view.as_const()).ok());
            ASSERT_TRUE(
                slots[lane].bind(gamma_id, parameters.view.as_const()).ok());
            ASSERT_TRUE(slots[lane].bind(norm_id, normals[lane].view).ok());
            ASSERT_TRUE(slots[lane].bind(output_id, outputs[lane].view).ok());
            const auto status = graph::cuda::prepare_replay(
                slots[lane], queues[lane],
                {.mode = graph::ReplayMode::kRequireCapture,
                 .host_wait = graph::HostWaitPolicy::kReject});
            ASSERT_TRUE(status.ok()) << status.message();
            EXPECT_EQ(slots[lane].replay_route(),
                      graph::ReplayRoute::kCaptured);
          }
          ASSERT_TRUE(queues[0].reserve_completions(2).ok());
          const auto before = stats.snapshot();
          std::vector<Value> retained;
          for (int repeat = 0; repeat < 3; ++repeat) {
            std::array<compute::Completion, 2> done;
            std::array<std::vector<Value>, 2> values, gate_values;
            for (std::size_t lane = 0; lane < 2; ++lane) {
              if (repeat == 2 && lane == 0)
                continue;
              values[lane].resize(static_cast<std::size_t>(rows * width));
              gate_values[lane].resize(values[lane].size());
              for (std::size_t i = 0; i < values[lane].size(); ++i) {
                values[lane][i] =
                    half_test::encode<Value>((static_cast<int>(i % 31) - 15 +
                                              repeat + static_cast<int>(lane)) *
                                             0.1875);
                gate_values[lane][i] = half_test::encode<Value>(
                    (static_cast<int>(i % 13) - 6 + repeat) * 0.25);
              }
              ASSERT_TRUE(inputs[lane].write<Value>(values[lane]).ok());
              ASSERT_TRUE(gates[lane].write<Value>(gate_values[lane]).ok());
              auto submitted = slots[lane].submit(queues[lane]);
              ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
              done[lane] = std::move(*submitted);
            }
            for (std::size_t lane = 0; lane < 2; ++lane) {
              if (repeat == 2 && lane == 0) {
                const auto actual =
                    half_test::read<Value>(outputs[lane].buffer);
                for (std::size_t i = 0; i < actual.size(); ++i)
                  EXPECT_EQ(actual[i].bits, retained[i].bits);
                continue;
              }
              ASSERT_TRUE(done[lane].wait().ok());
              const auto normal = half_test::read<Value>(normals[lane].buffer);
              const auto actual = half_test::read<Value>(outputs[lane].buffer);
              for (std::size_t i = 0; i < actual.size(); ++i) {
                const double gate = half_test::decode(gate_values[lane][i]);
                const float activation =
                    static_cast<float>(gate / (1 + std::exp(-gate)));
                const float expected =
                    static_cast<float>(half_test::decode(normal[i])) *
                    activation;
                half_test::expect_near(actual[i], expected);
              }
              if (lane == 0) {
                auto direct_done = direct_slot.submit(
                    queues[0], inputs[0].view.as_const(),
                    direct_scale.view.as_const(), direct_output.view);
                ASSERT_TRUE(direct_done.has_value());
                ASSERT_TRUE(direct_done->wait().ok());
                const auto direct_values =
                    half_test::read<Value>(direct_output.buffer);
                for (std::size_t i = 0; i < normal.size(); ++i)
                  EXPECT_EQ(normal[i].bits, direct_values[i].bits);
                retained = actual;
              }
            }
          }
          EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
          EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
          EXPECT_EQ(half_test::read<float>(parameters.buffer), gamma);
        }
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace
