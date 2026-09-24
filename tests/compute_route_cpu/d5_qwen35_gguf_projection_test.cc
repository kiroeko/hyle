#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
namespace c = hyle::compute;

c::TensorDesc dense(std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(c::DataType::kFloat32,
                              std::array{rows, columns})
      .value();
}

hyle::mem::Buffer buffer(std::span<const std::byte> bytes) {
  auto result = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                          bytes.size(), 64)
                    .value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

hyle::mem::Buffer output(const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                   description.required_bytes().value(), 64)
      .value();
}

TEST(CpuDefaultDirect,
     D5Qwen35PrimaryGgufExactMixedProjectionsUseNativeRegionsAndReplay) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (const auto &test : d5_gguf_test::kQwen35PrimaryProjectionCases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc = dense(1, test.reduction);
    const auto weight_desc =
        c::gguf_weight_description(test.type, c::DataType::kFloat32,
                                   test.reduction, test.columns)
            .value();
    const auto output_desc = dense(1, test.columns);
    const std::array input_descs{input_desc, weight_desc};
    const std::array output_descs{output_desc};
    const auto workload = d5_qwen35_fixture::workload(
        d5_qwen35_fixture::kPrimaryGgufProfile,
        c::WorkloadPhase::kDecode, test.region);
    auto prepared = c::prepare(context, c::Operation(c::MatMul{}), input_descs,
                               output_descs, hyle::mem::system_resource(),
                               workload);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
    EXPECT_EQ(prepared->workload(), workload);

    const auto physical =
        d5_gguf_test::values(test.columns, test.reduction, test.seed);
    const auto packed = d5_gguf_test::pack(
        test.type, physical, test.columns, test.reduction,
        weight_desc.required_bytes().value());
    auto activation_values =
        d5_gguf_test::activation(test.reduction, test.seed);
    auto activation_buffer = half_test::buffer<float>(activation_values);
    auto weight_buffer = buffer(packed.bytes);
    auto result_buffer = output(output_desc);
    const std::array inputs{
        c::ConstTensorView::create(input_desc, activation_buffer.view()).value(),
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, result_buffer.view()).value()};
    auto invocation = prepared->create_invocation().value();

    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_values)
          value = value * -0.75F + 0.001F;
        ASSERT_TRUE(activation_buffer.view()
                        .write(std::as_bytes(std::span(activation_values)))
                        .ok());
      }
      auto completion = invocation.submit(queue, inputs, outputs);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(result_buffer);
      const auto expected =
          d5_gguf_test::matmul(activation_values, packed.decoded, test.columns,
                               test.reduction);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }
  }
}

} // namespace
