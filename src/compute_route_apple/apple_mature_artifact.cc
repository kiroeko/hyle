#include "apple_mature_artifact.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_checks.h"
#include "bitsandbytes_apple.h"
#include "awq_apple.h"
#include "compressed_tensors_apple.h"
#include "../compute_shader_shared/ct_packed.h"
#include "../compute_shader_shared/nvfp4_f32.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
Result<mx::array> lower_mature_artifact(model_ops::ArtifactWeightDecode operation,
    std::span<const mx::array> inputs, mx::Shape shape, mx::Dtype dtype,
    mx::Stream stream) {
  const int group = static_cast<int>(operation.group_size);
  using Format = model_ops::WeightArtifactFormat;
  if (operation.format == Format::kGptqV1Int4) {
    const auto items = std::uint64_t(shape[0]) * shape[1];
    const auto groups = items / 128 + (items % 128 != 0);
    const auto groups_x = (std::min)(groups, std::uint64_t{65535});
    const auto groups_y = groups / groups_x + (groups % groups_x != 0);
    if (groups_y > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
      return std::unexpected(Status::OutOfRange(
          "Apple GPTQ decode grid exceeds native int axes"));
    // Natural exporter indices are not permission to skip mutable contents.
    // Complete validation must retire before any indexed SF/Z address.
    auto valid = mx::logical_and(
        mx::greater_equal(inputs[3], mx::array(std::int32_t{0}), stream),
        mx::less(inputs[3], mx::array(inputs[2].shape(0)), stream), stream);
    auto status = mx::reshape(mx::where(mx::all(valid, false, stream),
        mx::array(std::int32_t{0}), mx::array(kIndexOutOfRange), stream), {1}, stream);
    static const auto header = std::string(R"METAL(
      inline uint hyle_mul_hi(uint a, uint b) {
        return uint((ulong(a) * ulong(b)) >> 32);
      }
      inline uint hyle_clz(uint a) { return clz(a); }
    )METAL") + shader::kNvfp4F32 + shader::kCtPacked;
    static const auto kernel = mx::fast::metal_kernel(
        "hyle_gptq_v1_int4_reference_decode",
        {"codes", "zeros", "scales", "indices", "index_status"}, {"out"}, R"METAL(
      const ulong item = ulong(thread_position_in_grid.y)*GROUP_WIDTH +
          thread_position_in_grid.x;
      const ulong columns = codes_shape[1], depth = indices_shape[0];
      if (item >= columns*depth) return;
      uint result = 0u;
      if (index_status[0] == 0) {
        const ulong column = item/depth, k = item%depth;
        const int group = indices[k*indices_strides[0]];
        const uint code =
            (codes[(k/8)*codes_strides[0] + column*codes_strides[1]] >>
                (4*uint(k%8))) & 15u;
        const uint zero =
            ((zeros[ulong(group)*zeros_strides[0] + (column/8)*zeros_strides[1]] >>
                (4*uint(column%8))) & 15u) + 1u;
        const uint centered = as_type<uint>(float(int(code)-int(zero)));
        const uint scale = hyle_ct_widen(
            uint(scales[ulong(group)*scales_strides[0] + column*scales_strides[1]]), 1u);
        // Preserve the original F16 product before the requested decode cast.
        result = hyle_ct_round(hyle_ct_round(hyle_f32_mul(centered,scale),1u),
            OUTPUT_TYPE);
      }
      out[item] = OutType(hyle_ct_narrow(result,OUTPUT_TYPE));
    )METAL", header, false, false, mx::CompileOptions{});
    const int output_type = dtype == mx::float32 ? 0 : dtype == mx::float16 ? 1 : 2;
    const auto raw_type = output_type == 0 ? mx::uint32 : mx::uint16;
    auto result = kernel({mx::view(inputs[0],mx::uint32,stream),
        mx::view(inputs[1],mx::uint32,stream), mx::view(inputs[2],mx::uint16,stream),
        inputs[3],status}, {shape}, {raw_type},
        {static_cast<int>(groups_x*128),static_cast<int>(groups_y),1}, {128,1,1},
        {{"GROUP_WIDTH",static_cast<int>(groups_x*128)},
         {"OUTPUT_TYPE",output_type},{"OutType",raw_type}}, std::nullopt, false, stream);
    return with_device_status(mx::view(result[0],dtype,stream),std::move(status),stream);
  }
  if (operation.format == Format::kNf4)
    return borrowed::bitsandbytes_apple::decode_nf4(inputs[0], inputs[1], inputs[2],
                                                   group, shape, dtype, stream);
  if (operation.format == Format::kNf4DoubleQuant) {
    auto scales = borrowed::bitsandbytes_apple::nested_scales(
        inputs[1], inputs[2], inputs[3], inputs[4], stream);
    return borrowed::bitsandbytes_apple::decode_nf4(inputs[0], scales, inputs[5],
                                                   group, shape, dtype, stream);
  }
  const auto shifts = mx::multiply(mx::arange(8, mx::uint32, stream),
                                   mx::array(std::uint32_t{4}), stream);
  if (operation.format == Format::kAwqGemm4) {
    const mx::array order({std::int32_t{0}, std::int32_t{4}, std::int32_t{1},
                           std::int32_t{5}, std::int32_t{2}, std::int32_t{6},
                           std::int32_t{3}, std::int32_t{7}});
    return mx::astype(borrowed::awq_apple::decode_gemm4(inputs[0], inputs[1],
        inputs[2], group, shifts, order, stream), dtype, stream);
  }
  if (operation.format == Format::kCompressedTensorsAsymmetric4)
    return mx::astype(borrowed::compressed_tensors_apple::decode_asymmetric4(
        inputs[0], inputs[1], inputs[2], group, shifts, stream), dtype, stream);
  if (operation.format == Format::kCompressedTensorsSymmetric4)
    return mx::astype(borrowed::compressed_tensors_apple::decode_symmetric4(
        inputs[0], inputs[1], group, shifts, stream), dtype, stream);
  throw std::invalid_argument("Apple artifact weight format is not implemented");
}
} // namespace hyle::compute::apple::detail
