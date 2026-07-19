#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

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
float dot_f32_neon(const float* x, const float* y, std::size_t n) noexcept;
float dot_f32_sve(const float* x, const float* y, std::size_t n) noexcept;
float dot_f32_rvv(const float* x, const float* y, std::size_t n) noexcept;

void saxpy_f32_scalar(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_sse4(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_avx2(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_avx512(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_neon(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_sve(float a, const float* x, float* y, std::size_t n) noexcept;
void saxpy_f32_rvv(float a, const float* x, float* y, std::size_t n) noexcept;

double dot_f64_scalar(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_sse4(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_avx2(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_avx512(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_neon(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_sve(const double* x, const double* y, std::size_t n) noexcept;
double dot_f64_rvv(const double* x, const double* y, std::size_t n) noexcept;

#define HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(suffix)                                  \
  float dot_bf16_bf16_##suffix(const bfloat16_t*, const bfloat16_t*,              \
                                std::size_t) noexcept;                             \
  float dot_f32_bf16_##suffix(const float*, const bfloat16_t*,                    \
                               std::size_t) noexcept;                              \
  float dot_f16_f16_##suffix(const float16_t*, const float16_t*,                  \
                              std::size_t) noexcept;                               \
  std::int64_t dot_i16_i16_##suffix(const std::int16_t*, const std::int16_t*,     \
                                     std::size_t) noexcept;                       \
  std::int64_t dot_i8_i8_##suffix(const std::int8_t*, const std::int8_t*,         \
                                   std::size_t) noexcept;

HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(scalar)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(sse4)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(avx2)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(avx512)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(neon)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(sve)
HYLE_SIMD_OPS_DECLARE_MIXED_DOTS(rvv)

#undef HYLE_SIMD_OPS_DECLARE_MIXED_DOTS

// Precise-feature kernels are not broad ISA variants. The runtime table selects
// them only after CPUID/OS-state gating.
float dot_f16_f16_f16c(const float16_t*, const float16_t*, std::size_t) noexcept;
float dot_bf16_bf16_avx512bf16(const bfloat16_t*, const bfloat16_t*,
                               std::size_t) noexcept;
std::int64_t dot_i8_i8_avx_vnni(const std::int8_t*, const std::int8_t*,
                                std::size_t) noexcept;
std::int64_t dot_i8_i8_avx512_vnni(const std::int8_t*, const std::int8_t*,
                                   std::size_t) noexcept;

void saxpy_f64_scalar(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_sse4(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_avx2(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_avx512(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_neon(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_sve(double a, const double* x, double* y, std::size_t n) noexcept;
void saxpy_f64_rvv(double a, const double* x, double* y, std::size_t n) noexcept;

void add_f32_scalar(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_sse4(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_avx2(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_avx512(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_neon(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_sve(const float* x, const float* y, float* z, std::size_t n) noexcept;
void add_f32_rvv(const float* x, const float* y, float* z, std::size_t n) noexcept;

void add_f64_scalar(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_sse4(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_avx2(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_avx512(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_neon(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_sve(const double* x, const double* y, double* z, std::size_t n) noexcept;
void add_f64_rvv(const double* x, const double* y, double* z, std::size_t n) noexcept;

}  // namespace hyle::simd::detail
