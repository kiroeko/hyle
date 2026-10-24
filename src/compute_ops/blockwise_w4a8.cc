#include "hyle/compute/model_ops.h"

#include <algorithm>
#include <array>

namespace hyle::compute::model_ops {
namespace {
Status infer(std::span<const TensorDesc> in, std::span<TensorDesc> out,
             bool grouped) noexcept {
  if (in.size() != (grouped ? 4u : 3u) || out.size() != 1)
    return Status::InvalidArgument("blockwise W4A8 port counts differ");
  for (std::size_t port = 0; port < in.size(); ++port) {
    const auto &d = in[port];
    bool shared = false;
    if (grouped && port == 0 && d.valid() && d.is_plain() && d.rank() == 3 &&
        d.data_type() == DataType::kBFloat16) {
      const auto s = d.dimensions(), b = d.byte_strides();
      shared = s[0] > 0 && s[1] > 0 && s[2] > 0 && b[2] == 2 &&
          b[1] == 0 && b[0] % 2 == 0 && (s[0] == 1 || b[0] / 2 >= s[2]);
    }
    if (!d.valid() || !d.is_plain() || (!d.is_dense() && !shared))
      return Status::InvalidArgument("blockwise W4A8 needs dense ports or shared A");
    const auto bytes = d.required_bytes();
    if (!bytes) return bytes.error();
  }
  const auto &a = in[0], &w = in[1], &sf = in[2];
  const std::size_t rank = grouped ? 3u : 2u;
  if (a.data_type() != DataType::kBFloat16 || a.rank() != rank ||
      w.data_type() != DataType::kInt32 || w.rank() != rank ||
      sf.data_type() != DataType::kFloat32 || sf.rank() != rank)
    return Status::InvalidArgument("blockwise W4A8 typed ranks differ");
  const auto t = a.dimensions()[0], k = a.dimensions().back();
  const auto n = w.dimensions()[grouped ? 1 : 0];
  const auto e = grouped ? w.dimensions()[0] : 1;
  if (t <= 0 || n <= 0 || e <= 0 || k <= 0 || k % 512 ||
      w.dimensions().back() != k / 8 ||
      sf.dimensions().back() != k / 512 ||
      sf.dimensions()[grouped ? 1 : 0] != n ||
      (grouped && sf.dimensions()[0] != e))
    return Status::InvalidArgument("blockwise W4A8 requires matching positive dimensions and group512 K");
  if (grouped) {
    const auto &ids = in[3];
    if (ids.data_type() != DataType::kInt32 || ids.rank() != 2 ||
        ids.dimensions()[0] != t || ids.dimensions()[1] <= 0 ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != ids.dimensions()[1]))
      return Status::InvalidArgument("grouped blockwise W4A8 ID/activation slots differ");
    HYLE_ASSIGN_OR_RETURN(out[0], TensorDesc::dense(DataType::kBFloat16,
        std::array{t, ids.dimensions()[1], n}));
  } else {
    HYLE_ASSIGN_OR_RETURN(out[0], TensorDesc::dense(DataType::kBFloat16,
        std::array{t, n}));
  }
  const auto bytes = out[0].required_bytes();
  if (!bytes) return bytes.error();
  return Status::Ok();
}
} // namespace

Status BlockwiseW4A8Linear::infer_output_descriptions(
    std::span<const TensorDesc> in, std::span<TensorDesc> out) const noexcept {
  return infer(in, out, false);
}
Status GroupedBlockwiseW4A8Linear::infer_output_descriptions(
    std::span<const TensorDesc> in, std::span<TensorDesc> out) const noexcept {
  return infer(in, out, true);
}
} // namespace hyle::compute::model_ops
