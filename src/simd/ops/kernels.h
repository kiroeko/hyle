#pragma once

#include <cstddef>

// kernels.h (internal) -- declares the per-ISA variant symbols the dispatcher takes the
// address of. Each is DEFINED in its own -march TU (kernels_scalar/sse4/avx2.cc) from the
// shared body in kernel_impl.h. Names carry an ISA suffix so the three variants coexist in
// one binary without collision (one backend per TU; only the symbol names are shared here).
//
// NOT a public header -- lives under src/simd/ops/, never installed. Consumers use ops.h.

namespace hyle::simd::detail {

float dot_f32_scalar(const float* x, const float* y, std::size_t n) noexcept;
float dot_f32_sse4(const float* x, const float* y, std::size_t n) noexcept;
float dot_f32_avx2(const float* x, const float* y, std::size_t n) noexcept;
float dot_f32_avx512(const float* x, const float* y, std::size_t n) noexcept;

void saxpy_f32_scalar(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_sse4(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_avx2(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_avx512(float a, const float* x, float* y, std::size_t n) noexcept;

}  // namespace hyle::simd::detail
