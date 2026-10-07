// Original bitsandbytes default GPU tensor sequence, expressed in ATen C++.
// Copyright (c) Facebook, Inc. and its affiliates. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <ATen/ATen.h>

namespace hyle::borrowed::bitsandbytes_cuda {
// _dequantize_blockwise_compute: nested block size is exactly256.
inline at::Tensor nested_scales(const at::Tensor &codes,
    const at::Tensor &absmax, const at::Tensor &codebook, const at::Tensor &offset) {
  auto flat = codes.reshape({-1});
  auto decoded = codebook.index_select(0, flat.to(at::kLong));
  auto scales = absmax.repeat_interleave(256).slice(0, 0, flat.numel());
  return decoded * scales + offset.reshape({});
}
// _dequantize_4bit_compute: high nibble first; lookup and multiply remain F32.
inline at::Tensor decode_nf4(const at::Tensor &packed,
    const at::Tensor &absmax, const at::Tensor &codebook, std::int64_t group,
    std::int64_t rows, std::int64_t columns, at::ScalarType dtype) {
  auto flat = packed.reshape({-1}).to(at::kLong);
  auto codes = at::stack({at::bitwise_right_shift(flat, 4),
                         at::bitwise_and(flat, 15)}, 1).reshape({-1});
  auto values = codebook.index_select(0, codes);
  return (values.reshape({-1, group}) * absmax.reshape({-1, 1}))
      .reshape({rows, columns}).to(dtype);
}
} // namespace hyle::borrowed::bitsandbytes_cuda
