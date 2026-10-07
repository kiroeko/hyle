// Original AutoAWQ unpack_awq/reverse_awq_order/dequantize_gemm tensor sequence.
// Copyright (c) 2023 MIT HAN Lab. MIT; see LICENSE/NOTICE.md.
#pragma once
#include <ATen/ATen.h>

namespace hyle::borrowed::awq_cuda {
inline at::Tensor unpack(const at::Tensor &words, const at::Tensor &shifts, const at::Tensor &order) {
  auto codes = at::bitwise_right_shift(words.to(at::kLong).unsqueeze(-1), shifts);
  return at::bitwise_and(codes.index_select(-1, order), 15)
      .reshape({words.size(0), -1}).to(at::kChar);
}
inline at::Tensor decode_gemm4(const at::Tensor &weight,
    const at::Tensor &zeros, const at::Tensor &scales, std::int64_t group,
    const at::Tensor &shifts, const at::Tensor &order) {
  auto values = unpack(weight, shifts, order);
  auto zero = unpack(zeros, shifts, order).repeat_interleave(group, 0);
  // Original int8 subtraction is exact in [-15,15]; original scale dtype
  // determines multiplication/rounding. No centered-int4 bias substitution.
  return ((values - zero) * scales.repeat_interleave(group, 0))
      .transpose(0, 1).contiguous();
}
} // namespace hyle::borrowed::awq_cuda
