#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/low_precision.h"
#include "apple_internal.h"
#include "apple_checks.h"
#include "ggml_q4k_apple.h"
#include <array>
#include <limits>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
bool is_mlx_exact_gguf_gather(const Operation &operation,
                             std::span<const TensorDesc> inputs) noexcept {
  const auto *gather = operation.get_if<Gather>();
  return gather && inputs.size() == 2 &&
      (gather->axis == 0 || gather->axis == -2) && gather->bounds == IndexBoundsPolicy::kChecked &&
      inputs[0].rank() == 2 && inputs[0].block_axis() == std::size_t{1} &&
      inputs[0].data_type() == DataType::kFloat32 &&
      gguf_quantized_type(inputs[0].storage_format()) == GgufQuantizedType::kQ4K &&
      inputs[1].rank() == 1 && inputs[1].data_type() == DataType::kInt32;
}
Result<std::vector<mx::array>> lower_mlx_exact_gguf_gather(const Operation &operation,
    std::span<const TensorDesc> descriptions, std::span<const mx::array> inputs,
    mx::Stream stream) noexcept {
  try {
    if (!is_mlx_exact_gguf_gather(operation, descriptions) || inputs.size() != 2)
      return std::unexpected(Status::Unimplemented("Apple exact GGUF row carrier is not admitted"));
    std::array<TensorDesc, 1> output;
    HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, descriptions, output));
    const auto rows = descriptions[0].dimensions()[0], width = descriptions[0].dimensions()[1];
    const auto row_bytes = width / 256 * 144;
    if (rows > INT32_MAX || width > INT32_MAX || row_bytes > INT32_MAX)
      return std::unexpected(Status::OutOfRange("Apple exact GGUF row axes exceed native range"));
    auto valid = mx::logical_and(mx::greater_equal(inputs[1], mx::array(0), stream),
        mx::less(inputs[1], mx::array(static_cast<std::int32_t>(rows)), stream), stream);
    auto ids = mx::where(valid, inputs[1], mx::array(0), stream);
    auto selected = mx::take(mx::reshape(inputs[0], {static_cast<int>(rows),
                             static_cast<int>(row_bytes)}, stream), ids, 0, stream);
    auto decoded = borrowed::ggml_q4k_apple::decode_rows(selected, static_cast<int>(width), stream);
    auto status = mx::where(mx::all(valid, false, stream), mx::array(0), mx::array(kIndexOutOfRange), stream);
    HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(decoded), std::move(status), stream));
    return std::vector{std::move(checked)};
  } catch (...) { return std::unexpected(native_exception()); }
}
} // namespace hyle::compute::apple::detail
