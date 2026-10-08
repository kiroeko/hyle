#include "apple_mature_artifact.h"
#include <cstdint>

#include "bitsandbytes_apple.h"
#include "awq_apple.h"
#include "compressed_tensors_apple.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
mx::array lower_mature_artifact(model_ops::ArtifactWeightDecode operation,
    std::span<const mx::array> inputs, mx::Shape shape, mx::Dtype dtype,
    mx::Stream stream) {
  const int group = static_cast<int>(operation.group_size);
  using Format = model_ops::WeightArtifactFormat;
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
  return mx::astype(borrowed::compressed_tensors_apple::decode_symmetric4(
      inputs[0], inputs[1], group, shifts, stream), dtype, stream);
}
} // namespace hyle::compute::apple::detail
