#pragma once

#include "hyle/compute/operation.h"
#include <array>

namespace hyle::compute::cuda_native::internal {

// Same-build composite selected from the original public typed score graph.
// Dimensions and physical pitches belong to its actual ports, never the op.
struct DenseBfloatScores final {
  bool masked = false;
  DataType output_type = DataType::kBFloat16;
  static constexpr OperationId operation_id{0x48594c4543554441ULL,
                                             0x444253434f520002ULL};
  static constexpr std::string_view operation_name =
      "private-cuda-dense-bfloat-scores";
  constexpr OperationTraits operation_traits() const noexcept {
    const auto count = std::uint16_t(masked ? 4 : 3);
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    if (in.size() != (masked ? 4u : 3u) || out.size() != 1 ||
        (output_type != DataType::kBFloat16 &&
         output_type != DataType::kFloat32))
      return Status::InvalidArgument("dense BF16 score port/type differs");
    for (std::size_t i = 0; i < in.size(); ++i)
      if (!in[i].valid() || !in[i].is_plain() ||
          in[i].data_type() != (i == 3 ? DataType::kFloat16
                                      : DataType::kBFloat16))
        return Status::InvalidArgument("dense BF16 score input type differs");
    if (in[0].rank() != 4 || in[1].rank() != 4 || in[2].rank() != 4)
      return Status::InvalidArgument("dense BF16 score ranks differ");
    const auto q = in[0].dimensions(), k = in[1].dimensions(),
               w = in[2].dimensions();
    // The actual public View boundaries are ports: Kt=[B,1,D,L] and
    // Wr=[B,T,H,1]. Do not consume/reconstruct their original View producers.
    if (q[2] <= 0 || q[3] <= 0 || k[0] != q[0] || k[1] != 1 || k[2] != q[3] ||
        w[0] != q[0] || w[1] != q[1] || w[2] != q[2] || w[3] != 1)
      return Status::InvalidArgument("dense BF16 score dimensions differ");
    if (masked) {
      if (in[3].rank() != 3)
        return Status::InvalidArgument("dense BF16 score mask rank differs");
      const auto mask = in[3].dimensions();
      if ((mask[0] != 1 && mask[0] != q[0]) || mask[1] != q[1] ||
          mask[2] != k[3])
        return Status::InvalidArgument("dense BF16 score mask dimensions differ");
    }
    HYLE_ASSIGN_OR_RETURN(
        out[0], TensorDesc::dense(output_type, std::array{q[0], q[1], k[3]}));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<DenseBfloatScores>);

} // namespace hyle::compute::cuda_native::internal
