#include "apple_ct_packed.h"
#include "apple_checks.h"

#include "../compute_packed_shared/artifact_packed_linear.h"
#include "../compute_shader_shared/ct_packed.h"
#include "../compute_shader_shared/nvfp4_f32.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
int scalar_tag(DataType type) noexcept {
  return type == DataType::kFloat16 ? 1 : type == DataType::kBFloat16 ? 2 : 0;
}
mx::Dtype raw_type(DataType type) noexcept {
  return type == DataType::kFloat32 ? mx::uint32 : mx::uint16;
}
mx::Dtype floating_type(DataType type) noexcept {
  return type == DataType::kFloat16 ? mx::float16
       : type == DataType::kBFloat16 ? mx::bfloat16 : mx::float32;
}
std::string kernel_header() {
  return std::string(R"METAL(
    inline uint hyle_mul_hi(uint a, uint b) {
      return uint((ulong(a) * ulong(b)) >> 32);
    }
    inline uint hyle_clz(uint a) { return clz(a); }
  )METAL") + shader::kNvfp4F32 + shader::kCtPacked + R"METAL(
    inline bool hyle_ct_tiny_or_special(uint value) {
      const uint magnitude = value & 0x7fffffffu;
      return (magnitude != 0u && magnitude < 0x00800000u) || magnitude >= 0x7f800000u;
    }
    inline bool hyle_ct_unsafe_add(uint left, uint right, uint result) {
      return hyle_ct_tiny_or_special(result) ||
          ((result & 0x7fffffffu) == 0u && ((left | right) & 0x7fffffffu) != 0u);
    }
    inline uint hyle_ct_weight(uint centered, uint scale, uint scale_type,
        uint decode_type, uint operand_type) {
      const uint exponent = (scale >> 23) & 255u;
      if (scale_type == 2u && exponent != 0u && exponent != 255u) {
        // A BF16 scale and integer center[-15,15] need at most twelve F32
        // significant bits. A nonzero product cannot underflow F32 here.
        // Safe Metal math preserves signed zero/overflow; all three original
        // storage boundaries still use the shared bit-defined RNE helpers.
        const uint product = as_type<uint>(as_type<float>(centered) * as_type<float>(scale));
        return hyle_ct_round(hyle_ct_round(hyle_ct_round(product, scale_type),
            decode_type), operand_type);
      }
      return hyle_ct_product(centered, scale, scale_type, decode_type, operand_type);
    }
  )METAL";
}
} // namespace

Result<std::vector<mx::array>> lower_ct_packed(
    const packed::internal::ArtifactPackedLinear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream stream) {
  if (operation.format != model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 &&
      operation.format != model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 &&
      operation.format != model_ops::WeightArtifactFormat::kGptqV1Int4)
    return std::unexpected(Status::Unimplemented("Apple packed artifact codec is not implemented"));
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(operation.infer_output_descriptions(descriptions, output));
  if (inputs.size() != descriptions.size() || stream.device.type != mx::Device::gpu)
    return std::unexpected(Status::InvalidArgument("Apple packed artifact requires matching GPU carriers"));
  const bool grouped = operation.expert_count != 0;
  if (!descriptions[0].is_plain() || descriptions[0].rank() != (grouped ? 3U : 2U) ||
      descriptions[0].dimensions()[0] <= 0)
    return std::unexpected(Status::Unimplemented("Apple packed artifact requires plain positive activation rows"));
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return std::unexpected(Status::InvalidArgument("Apple packed artifact carrier differs from logical storage"));
  }
  HYLE_ASSIGN_OR_RETURN(auto output_spec, input_spec(output[0]));
  HYLE_ASSIGN_OR_RETURN(auto items, output[0].element_count());
  HYLE_ASSIGN_OR_RETURN(auto output_bytes, output[0].required_bytes());
  (void)output_bytes;
  const auto columns = static_cast<int>(output[0].dimensions()[grouped ? 2 : 1]);
  const auto groups = static_cast<std::uint64_t>(items / 4 + (items % 4 != 0));
  const auto groups_x = (std::min)(groups, std::uint64_t{65535});
  const auto groups_y = groups / groups_x + (groups % groups_x != 0);
  if (groups_y > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return std::unexpected(Status::OutOfRange("Apple packed artifact grid exceeds native int axes"));
  const bool asymmetric = operation.format ==
      model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4;
  const bool gptq = operation.format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const auto operand_type = operation.operand_type;
  const auto scale_port = gptq ? 3U : descriptions.size() - (grouped ? 2 : 1);
  const auto scale_type = descriptions[scale_port].data_type();
  // Same-width native Views preserve all storage bits, including subnormals;
  // Metal never loads them through a floating conversion or FTZ operation.
  auto a = mx::view(inputs[0], raw_type(descriptions[0].data_type()), stream);
  auto codes = mx::view(inputs[1], mx::uint32, stream);
  auto zeros = asymmetric || gptq ? mx::view(inputs[2], mx::uint32, stream) : codes;
  auto scales = mx::view(inputs[scale_port], raw_type(scale_type), stream);
  auto indices = grouped ? inputs.back() : gptq ? inputs[4]
      : mx::reshape(mx::array(std::int32_t{0}), {1}, stream);
  auto index_status = mx::reshape(mx::array(std::int32_t{0}), {1}, stream);
  if (gptq || grouped) {
    // The complete signed ID tensor is a device dependency of the dot. No
    // selected expert address is formed before that original checked domain.
    const auto extent = grouped ? static_cast<int>(operation.expert_count)
                                : inputs[scale_port].shape(0);
    auto valid = mx::logical_and(
        mx::greater_equal(indices, mx::array(std::int32_t{0}), stream),
        mx::less(indices, mx::array(extent), stream), stream);
    index_status = mx::reshape(mx::where(mx::all(valid, false, stream),
        mx::array(std::int32_t{0}), mx::array(kIndexOutOfRange), stream), {1}, stream);
  }
  static const auto header = kernel_header();
  static const auto kernel = mx::fast::metal_kernel("hyle_artifact_packed_reference_dot",
      {"a", "codes", "zeros", "scales", "indices", "index_status"}, {"out"}, R"METAL(
    const uint lane = thread_index_in_simdgroup;
    const uint subgroup = simdgroup_index_in_threadgroup;
    const ulong group = ulong(threadgroup_position_in_grid.y) * GROUPS_X +
        threadgroup_position_in_grid.x;
    const ulong index = group * 4 + subgroup;
    const ulong columns = GROUPED ? COLUMNS : GPTQ ? codes_shape[1] : codes_shape[0];
    const ulong rows = GROUPED ? ulong(indices_shape[0]) * indices_shape[1] : a_shape[0];
    const ulong depth = GROUPED ? a_shape[2] : a_shape[1];
    if (index >= rows * columns) return;
    if ((GPTQ || GROUPED) && index_status[0] != 0) {
      if (lane == 0) out[index] = OutType(0);
      return;
    }
    const ulong row = index / columns, column = index % columns;
    const ulong token = GROUPED ? row / indices_shape[1] : row;
    const ulong slot = GROUPED ? row % indices_shape[1] : 0;
    const int expert = GROUPED ? indices[token * indices_strides[0] + slot * indices_strides[1]] : 0;
    const ulong bank_row = GROUPED ? ulong(expert) * columns + column : column;
    const ulong activation_origin = GROUPED
        ? token * a_strides[0] + (a_shape[1] == 1 ? 0 : slot) * a_strides[1]
        : row * a_strides[0];
    const ulong activation_stride = GROUPED ? a_strides[2] : a_strides[1];
    uint total = 0u;
    bool unusual = false;
    for (ulong k = lane; k < depth; k += 32) {
      const int selected = GPTQ ? indices[k * indices_strides[0]] : int(k / GROUP_SIZE);
      const ulong code_offset = GPTQ
          ? (k / 8) * codes_strides[0] + column * codes_strides[1]
          : bank_row * codes_strides[0] + (k / 8) * codes_strides[1];
      const uint word = codes[code_offset];
      const uint code = (word >> (4 * uint(k % 8))) & 15u;
      const uint zero = GPTQ
          ? ((zeros[ulong(selected) * zeros_strides[0] + (column / 8) * zeros_strides[1]] >>
                (4 * uint(column % 8))) & 15u) + 1u
          : (ASYMMETRIC
              ? (zeros[(bank_row / 8) * zeros_strides[0] + ulong(selected) * zeros_strides[1]] >>
                  (4 * uint(bank_row % 8))) & 15u
              : 8u);
      const uint centered = as_type<uint>(float(int(code) - int(zero)));
      const ulong scale_offset = GPTQ
          ? ulong(selected) * scales_strides[0] + column * scales_strides[1]
          : bank_row * scales_strides[0] + ulong(selected) * scales_strides[1];
      const uint scale = hyle_ct_widen(uint(scales[scale_offset]), SCALE_TYPE);
      const uint weight = hyle_ct_weight(centered, scale, SCALE_TYPE,
          DECODE_TYPE, OPERAND_TYPE);
      const uint value = hyle_ct_widen(uint(a[activation_origin + k * activation_stride]),
          A_TYPE);
      const uint ae = (value >> 23) & 255u, we = (weight >> 23) & 255u;
      unusual |= hyle_ct_tiny_or_special(value) || hyle_ct_tiny_or_special(weight) ||
          (((value & 0x7fffffffu) != 0u && (weight & 0x7fffffffu) != 0u) && ae + we <= 127u);
      const float product = as_type<float>(value) * as_type<float>(weight);
      const uint term = as_type<uint>(product);
      const uint next = as_type<uint>(as_type<float>(total) + as_type<float>(term));
      unusual |= hyle_ct_tiny_or_special(term) || hyle_ct_unsafe_add(total, term, next);
      total = next;
    }
    for (ushort distance = 16; distance > 0; distance /= 2) {
      const uint right = simd_shuffle_down(total, distance);
      const uint next = as_type<uint>(as_type<float>(total) + as_type<float>(right));
      if (lane < distance) unusual |= hyle_ct_unsafe_add(total, right, next);
      total = next;
    }
    // Safe native F32 is the ordinary dot. Only this SIMDgroup's FTZ-sensitive
    // or nonfinite data takes the same reduction tree through strict uint math.
    if (simd_any(unusual)) {
      total = 0u;
      for (ulong k = lane; k < depth; k += 32) {
        const int selected = GPTQ ? indices[k * indices_strides[0]] : int(k / GROUP_SIZE);
        const ulong code_offset = GPTQ
            ? (k / 8) * codes_strides[0] + column * codes_strides[1]
            : bank_row * codes_strides[0] + (k / 8) * codes_strides[1];
        const uint word = codes[code_offset];
        const uint code = (word >> (4 * uint(k % 8))) & 15u;
        const uint zero = GPTQ
            ? ((zeros[ulong(selected) * zeros_strides[0] + (column / 8) * zeros_strides[1]] >>
                  (4 * uint(column % 8))) & 15u) + 1u
            : (ASYMMETRIC
                ? (zeros[(bank_row / 8) * zeros_strides[0] + ulong(selected) * zeros_strides[1]] >>
                    (4 * uint(bank_row % 8))) & 15u
                : 8u);
        const uint centered = as_type<uint>(float(int(code) - int(zero)));
        const ulong scale_offset = GPTQ
            ? ulong(selected) * scales_strides[0] + column * scales_strides[1]
            : bank_row * scales_strides[0] + ulong(selected) * scales_strides[1];
        const uint scale = hyle_ct_widen(uint(scales[scale_offset]), SCALE_TYPE);
        const uint weight = hyle_ct_weight(centered, scale, SCALE_TYPE,
            DECODE_TYPE, OPERAND_TYPE);
        const uint value = hyle_ct_widen(uint(a[activation_origin + k * activation_stride]),
            A_TYPE);
        total = hyle_f32_add(total, hyle_f32_mul(value, weight));
      }
      for (ushort distance = 16; distance > 0; distance /= 2)
        total = hyle_f32_add(total, simd_shuffle_down(total, distance));
    }
    if (lane == 0) out[index] = OutType(hyle_ct_narrow(total, OUTPUT_TYPE));
  )METAL", header, false, false, mx::CompileOptions{});
  auto result = kernel({a, codes, zeros, scales, indices, index_status}, {output_spec.shape},
      {raw_type(output[0].data_type())},
      {static_cast<int>(groups_x * 128), static_cast<int>(groups_y), 1}, {128, 1, 1},
      {{"GROUPS_X", static_cast<int>(groups_x)}, {"GROUP_SIZE", static_cast<int>(operation.group_size)},
        {"ASYMMETRIC", asymmetric}, {"GPTQ", gptq}, {"GROUPED", grouped},
       {"COLUMNS", columns}, {"A_TYPE", scalar_tag(descriptions[0].data_type())},
       {"OUTPUT_TYPE", scalar_tag(output[0].data_type())}, {"SCALE_TYPE", scalar_tag(scale_type)},
       {"DECODE_TYPE", scalar_tag(operation.decode_type)}, {"OPERAND_TYPE", scalar_tag(operand_type)},
       {"OutType", raw_type(output[0].data_type())}}, std::nullopt, false, stream);
  auto value = mx::view(result[0], floating_type(output[0].data_type()), stream);
  if (gptq || grouped) {
    HYLE_ASSIGN_OR_RETURN(value,
        with_device_status(std::move(value), std::move(index_status), stream));
  }
  return std::vector{std::move(value)};
}
} // namespace hyle::compute::apple::detail
