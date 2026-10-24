#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/mem/buffer.h"
#include "../../src/compute_provider_ggml_cuda/native_bridge.h"
#include "../../src/compute_provider_ggml_cuda/packed_matmul_native.h"
#include "../compute_models/mmq_q6k_reference.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace ref = ggml_cuda_mmq_q6k_reference;

std::vector<std::byte> original_q6k_bytes() {
  std::vector<std::byte> result(1260);
  constexpr std::array<float, 6> scales{0.75F, -1.25F, 0.5F, 1.5F, -0.625F, 1.125F};
  for (std::size_t block = 0; block < scales.size(); ++block) {
    const auto at = block * 210;
    // Generate original packed bytes directly. No production quantizer or
    // matching pack/unpack algorithm is used to construct these weights.
    for (std::size_t i = 0; i < 128; ++i)
      result[at + i] = std::byte((i * 37 + block * 29 + 11) & 255U);
    for (std::size_t i = 0; i < 64; ++i)
      result[at + 128 + i] = std::byte((i * 71 + block * 17 + 5) & 255U);
    for (std::size_t i = 0; i < 16; ++i) {
      int scale = int((block * 5 + i * 3) % 15) - 7;
      if (scale == 0) scale = 1;
      result[at + 192 + i] = std::bit_cast<std::byte>(std::int8_t(scale));
    }
    const auto bits = ref::fp::half_bits(scales[block]);
    result[at + 208] = std::byte(bits & 255U);
    result[at + 209] = std::byte(bits >> 8U);
  }
  return result;
}

std::vector<float> actual_f32_input() {
  std::vector<float> result(ref::kRows * ref::kWidth);
  for (std::size_t row = 0; row < ref::kRows; ++row) {
    for (std::size_t group = 0; group < ref::kGroups; ++group) {
      const float maximum = float(1.013 + double(row) * .071 + double(group) * .017);
      const auto first = row * ref::kWidth + group * 32;
      result[first] = maximum;
      for (std::size_t lane = 1; lane < 32; ++lane) {
        const double code = double(int((row * 43 + group * 19 + lane * 7) % 221) - 110) +
            (lane % 2 ? .125 : -.125);
        result[first + lane] = float(code * double(maximum) / 127.0);
      }
    }
  }
  return result;
}

TEST(GgmlCudaPackedMmqWorkspace, OriginalQ6KD4UsesMeasuredCallerWorkspace) {
  // Nontrivial signed codes and scales, with non-half-exact D4 metadata.
  // The oracle uses only original bytes and the exact submitted F32 A.
  const auto encoded = original_q6k_bytes();
  const auto input = actual_f32_input();
  const auto reference = ref::evaluate(encoded, input);
  const auto wrong_half = ref::wrong_half_scale_for_fixture(encoded, input);
  ASSERT_EQ(reference.ambiguous_codes, 0U);
  ASSERT_GT(reference.non_half_scales, 0U);
  std::size_t separated = 0;
  for (std::size_t i = 0; i < reference.center.size(); ++i) {
    const double distance_low = std::nextafter(std::abs(reference.center[i] - wrong_half.center[i]), 0.0);
    separated += distance_low > ref::fp::add_upper(reference.bound[i], wrong_half.bound[i]);
  }
  ASSERT_GT(separated, 0U) << "Fixture must distinguish D4 F32 scale from an incorrect F16 store";

  auto context = c::Context::create(c::cuda::device(0).value(), c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto environment = c::cuda::environment(context).value();
  const auto native = c::ggml_cuda::bridge::device_index(environment.device_ordinal);
  // Q6_K MMVQ's original NVIDIA limit is eight rows. M16 naturally uses
  // MMQ on the actual nominated device: no force flag or fabricated CC.
  constexpr std::int64_t K = 512, N = 3, M = 16;
  const auto plan = c::ggml_cuda::internal::packed_matmul_plan(native, GGML_TYPE_Q6_K, K, N, M);
  ASSERT_EQ(plan.route, c::ggml_cuda::internal::PackedMatmulRoute::kMmq);
  ASSERT_GT(plan.activation_bytes, 0U);
  ASSERT_GT(plan.bytes, plan.activation_bytes);
  EXPECT_EQ(c::ggml_cuda::internal::packed_matmul_route(native, GGML_TYPE_Q6_K, K, N, M),
            c::ggml_cuda::internal::PackedMatmulRoute::kDeclined); // Large-bank exception still caps M4.
  const auto a_desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array{M, K}).value();
  const auto w_desc = c::gguf_weight_description(c::GgufQuantizedType::kQ6K, c::DataType::kFloat32, K, N).value();
  const auto y_desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array{M, N}).value();
  ASSERT_EQ(w_desc.required_bytes().value(), encoded.size());
  auto prepared = c::prepare(context, c::Operation(c::MatMul{}),
      std::array{a_desc, w_desc}, std::array{y_desc}, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId);
  const auto requirement = prepared->workspace_requirement();
  ASSERT_GE(requirement.size, plan.bytes);
  ASSERT_LT(requirement.size, 64U * 1024U * 1024U);
  const auto alignment = std::max(std::size_t{256}, requirement.alignment);
  const auto aligned = [alignment](std::size_t bytes) {
    return (bytes + alignment - 1) & ~(alignment - 1);
  };
  // One real allocation proves both directions of the alias boundary:
  // used workspace and three ports are disjoint, while the supplied workspace
  // view includes an unused tail that contains those very same port views.
  const auto workspace_offset = alignment;
  const auto activation_offset = aligned(workspace_offset + requirement.size);
  const auto weights_offset = aligned(activation_offset + a_desc.required_bytes().value());
  const auto output_offset = aligned(weights_offset + encoded.size());
  auto backing = m::Buffer::create(memory.ref(), output_offset + y_desc.required_bytes().value(), alignment).value();
  const auto workspace = backing.view().subview(workspace_offset).value();
  const auto activation = backing.view().subview(activation_offset, a_desc.required_bytes().value()).value();
  const auto weights = backing.view().subview(weights_offset, encoded.size()).value();
  const auto output = backing.view().subview(output_offset, y_desc.required_bytes().value()).value();
  ASSERT_TRUE(weights.write(encoded).ok());
  ASSERT_TRUE(activation.write(std::as_bytes(std::span(input))).ok());
  ASSERT_TRUE(output.fill(std::byte{0x7f}).ok());
  auto too_small = prepared->create_invocation(workspace.subview(0, requirement.size - 1).value());
  ASSERT_FALSE(too_small); // Rejected cold, before AQ/encode; no native private growth.
  auto invocation = prepared->create_invocation(workspace);
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  const auto a = c::ConstTensorView::create(a_desc, activation).value();
  const auto w = c::ConstTensorView::create(w_desc, weights).value();
  const auto y = c::TensorView::create(y_desc, output).value();
  const auto bad_a = c::ConstTensorView::create(a_desc,
      backing.view().subview(workspace_offset, a_desc.required_bytes().value()).value()).value();
  const auto bad_w = c::ConstTensorView::create(w_desc,
      backing.view().subview(workspace_offset, encoded.size()).value()).value();
  const auto last_used_aligned = workspace_offset + (requirement.size - 1) / alignment * alignment;
  const auto bad_y = c::TensorView::create(y_desc,
      backing.view().subview(last_used_aligned, y_desc.required_bytes().value()).value()).value();
  for (int alias = 0; alias < 3; ++alias) {
    auto rejected = invocation->submit(queue, alias == 0 ? bad_a : a,
        alias == 1 ? bad_w : w, alias == 2 ? bad_y : y);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
    EXPECT_NE(rejected.error().message().find("caller workspace"), std::string::npos);
    EXPECT_FALSE(invocation->in_flight());
  }
  // The caller-owned workspace and all bindings outlive Completion. wait()
  // retires the original AQ/main/fixup stream before any output is read.
  auto completion = invocation->submit(queue, a, w, y);
  ASSERT_TRUE(completion) << completion.error().message();
  const auto status = completion->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  std::vector<float> actual(std::size_t(M * N));
  ASSERT_TRUE(output.read(std::as_writable_bytes(std::span(actual))).ok());
  ref::expect_projection(actual, reference);

  ::testing::Test::RecordProperty("workspace_alias_rejections", "3");
  ::testing::Test::RecordProperty("workspace_unused_tail_ports", "same allocation, disjoint used prefix");
  ::testing::Test::RecordProperty("workspace_bytes", std::to_string(requirement.size));
  ::testing::Test::RecordProperty("native_mmq_scratch_upper_bytes", std::to_string(plan.bytes));
  ::testing::Test::RecordProperty("native_mmq_activation_bytes", std::to_string(plan.activation_bytes));
  ::testing::Test::RecordProperty("native_mmq_fixup_upper_bytes", std::to_string(plan.fixup_bytes));
  ::testing::Test::RecordProperty("d4_non_half_scale_groups", std::to_string(reference.non_half_scales));
  ::testing::Test::RecordProperty("wrong_half_scale_disjoint_outputs", std::to_string(separated));
  ::testing::Test::RecordProperty("scope", "Actual default Q6_K MMQ M16 K512 N3; independent D4 reciprocal/multiply/round/F32-store interval and measured caller workspace. Conditional fast-division envelope; no Orin execution, all-codec, real-weight or memcheck claim");
}
} // namespace
