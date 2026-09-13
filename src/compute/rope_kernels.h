#pragma once

#include <cstddef>

namespace hyle::compute::detail {

#define HYLE_COMPUTE_DECLARE_ROPE_KERNELS(suffix)                              \
  void run_rope_interleaved_##suffix(                                         \
      const float *input, const float *frequencies, float *output,             \
      std::size_t pairs, float position) noexcept;                             \
  void run_rope_half_split_##suffix(                                          \
      const float *input, const float *frequencies, float *output,             \
      std::size_t pairs, float position) noexcept

HYLE_COMPUTE_DECLARE_ROPE_KERNELS(scalar);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(sse4);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(avx2);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(avx512);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(neon);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(sve);
HYLE_COMPUTE_DECLARE_ROPE_KERNELS(rvv);

#undef HYLE_COMPUTE_DECLARE_ROPE_KERNELS

} // namespace hyle::compute::detail
