/* Pre-release Native FFI schema candidate for HYLE SIMD whole-buffer products.
 * This is not a native-C product API or a frozen cross-language contract;
 * managed review may retain task-driven batches or declare FFI=N/A. Register
 * vectors and target-bound native APIs are intentionally C++ only. */
#ifndef HYLE_C_SIMD_H_
#define HYLE_C_SIMD_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h"
#include "hyle/c/simd_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_SIMD_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_SIMD_BUILD)
#    define HYLE_SIMD_C_API __declspec(dllexport)
#  else
#    define HYLE_SIMD_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_SIMD_C_API __attribute__((visibility("default")))
#else
#  define HYLE_SIMD_C_API
#endif

typedef uint32_t hyle_simd_isa;
enum {
  HYLE_SIMD_ISA_SCALAR = 0,
  HYLE_SIMD_ISA_SSE42 = 12,
  HYLE_SIMD_ISA_AVX2 = 20,
  HYLE_SIMD_ISA_AVX512 = 30,
  HYLE_SIMD_ISA_NEON = 110,
  HYLE_SIMD_ISA_SVE = 120,
  HYLE_SIMD_ISA_RVV = 210
};

typedef uint32_t hyle_simd_sort_order;
enum {
  HYLE_SIMD_SORT_ASCENDING = 0,
  HYLE_SIMD_SORT_DESCENDING = 1
};

typedef struct hyle_simd_module_info {
  hyle_simd_isa isa;
  uint32_t reserved;
  uint64_t features;
} hyle_simd_module_info;

/* hyle_simd_module_info.features bits. */
#define HYLE_SIMD_FEATURE_F16C                  (UINT64_C(1) << 0)
#define HYLE_SIMD_FEATURE_AVX512_BF16           (UINT64_C(1) << 1)
#define HYLE_SIMD_FEATURE_AVX10_2               (UINT64_C(1) << 2)
#define HYLE_SIMD_FEATURE_AVX_VNNI              (UINT64_C(1) << 3)
#define HYLE_SIMD_FEATURE_AVX512_VNNI           (UINT64_C(1) << 4)
#define HYLE_SIMD_FEATURE_MATH_EXACT_LIBM_SCALAR (UINT64_C(1) << 16)
#define HYLE_SIMD_FEATURE_DTYPE_FP8_SCALAR       (UINT64_C(1) << 17)
#define HYLE_SIMD_FEATURE_PER_OP_LOWER_TIER      (UINT64_C(1) << 18)
#define HYLE_SIMD_FEATURE_SORT_CAPPED_NETWORK    (UINT64_C(1) << 19)
#define HYLE_SIMD_FEATURE_BITPACK_LOGICAL128     (UINT64_C(1) << 20)

typedef struct hyle_simd_runtime_info {
  uint32_t struct_size;
  hyle_simd_isa isa;
  hyle_simd_module_info ops;
  hyle_simd_module_info scan;
  hyle_simd_module_info dtype;
  hyle_simd_module_info math;
  hyle_simd_module_info sort;
  hyle_simd_module_info bitpack;
} hyle_simd_runtime_info;

/* Readable prefix of the current candidate runtime-info layout. It exercises
 * size-aware binding mechanics but is not frozen for an existing caller. */
#define HYLE_SIMD_RUNTIME_INFO_V1_SIZE \
  ((uint32_t)(offsetof(hyle_simd_runtime_info, bitpack) + \
              sizeof(hyle_simd_module_info)))

#define HYLE_SIMD_RUNTIME_INFO_INIT \
  { (uint32_t)sizeof(hyle_simd_runtime_info), HYLE_SIMD_ISA_SCALAR, \
    {HYLE_SIMD_ISA_SCALAR, 0, 0}, {HYLE_SIMD_ISA_SCALAR, 0, 0}, \
    {HYLE_SIMD_ISA_SCALAR, 0, 0}, {HYLE_SIMD_ISA_SCALAR, 0, 0}, \
    {HYLE_SIMD_ISA_SCALAR, 0, 0}, {HYLE_SIMD_ISA_SCALAR, 0, 0} }

HYLE_SIMD_C_API hyle_simd_isa hyle_simd_active_isa(void);
HYLE_SIMD_C_API int hyle_simd_supports_isa(hyle_simd_isa isa);
/* Returns an owning status. `out` must use HYLE_SIMD_RUNTIME_INFO_INIT (or set
 * a supported struct_size) before the call. */
HYLE_SIMD_C_API hyle_status hyle_simd_runtime_info_get(
    hyle_simd_runtime_info* out);

/* Numeric ops. Integer dots accumulate modulo 2^64. */
HYLE_SIMD_C_API float hyle_simd_dot_f32(const float* x, const float* y, size_t n);
HYLE_SIMD_C_API double hyle_simd_dot_f64(const double* x, const double* y, size_t n);
HYLE_SIMD_C_API float hyle_simd_dot_bf16(const hyle_simd_bf16* x,
                                         const hyle_simd_bf16* y, size_t n);
HYLE_SIMD_C_API float hyle_simd_dot_f32_bf16(const float* x,
                                             const hyle_simd_bf16* y, size_t n);
HYLE_SIMD_C_API float hyle_simd_dot_f16(const hyle_simd_f16* x,
                                        const hyle_simd_f16* y, size_t n);
HYLE_SIMD_C_API int64_t hyle_simd_dot_i16(const int16_t* x, const int16_t* y,
                                          size_t n);
HYLE_SIMD_C_API int64_t hyle_simd_dot_i8(const int8_t* x, const int8_t* y,
                                         size_t n);
HYLE_SIMD_C_API void hyle_simd_saxpy_f32(float a, const float* x, float* y,
                                         size_t n);
HYLE_SIMD_C_API void hyle_simd_saxpy_f64(double a, const double* x, double* y,
                                         size_t n);
HYLE_SIMD_C_API void hyle_simd_add_f32(const float* x, const float* y, float* out,
                                       size_t n);
HYLE_SIMD_C_API void hyle_simd_add_f64(const double* x, const double* y, double* out,
                                       size_t n);

/* Opaque deterministic xoshiro state. Not cryptographically secure. */
typedef struct hyle_simd_random_o hyle_simd_random_o;
/* Returns an owning status and writes a new handle to `out`; a valid out is
 * cleared to NULL before allocation, so failure never publishes a handle. */
HYLE_SIMD_C_API hyle_status hyle_simd_random_create(
    hyle_simd_random_o** out, uint64_t seed);
HYLE_SIMD_C_API void hyle_simd_random_destroy(hyle_simd_random_o* state);
HYLE_SIMD_C_API void hyle_simd_random_fill_u64(hyle_simd_random_o* state,
                                               uint64_t* dst, size_t count);
HYLE_SIMD_C_API void hyle_simd_random_fill_f32(hyle_simd_random_o* state,
                                               float* dst, size_t count);
HYLE_SIMD_C_API void hyle_simd_random_fill_f64(hyle_simd_random_o* state,
                                               double* dst, size_t count);

#define HYLE_SIMD_DECLARE_SCAN(T, tag) \
  HYLE_SIMD_C_API void hyle_simd_inclusive_scan_##tag(T* data, size_t n); \
  HYLE_SIMD_C_API void hyle_simd_exclusive_scan_##tag(T* data, size_t n)
HYLE_SIMD_DECLARE_SCAN(uint32_t, u32);
HYLE_SIMD_DECLARE_SCAN(int32_t, i32);
HYLE_SIMD_DECLARE_SCAN(uint64_t, u64);
HYLE_SIMD_DECLARE_SCAN(int64_t, i64);
HYLE_SIMD_DECLARE_SCAN(float, f32);
HYLE_SIMD_DECLARE_SCAN(double, f64);
#undef HYLE_SIMD_DECLARE_SCAN

#define HYLE_SIMD_DECLARE_SEARCH(T, tag) \
  HYLE_SIMD_C_API size_t hyle_simd_count_equal_##tag(const T*, size_t, T); \
  HYLE_SIMD_C_API size_t hyle_simd_find_equal_##tag(const T*, size_t, T); \
  HYLE_SIMD_C_API T hyle_simd_min_value_##tag(const T*, size_t); \
  HYLE_SIMD_C_API T hyle_simd_max_value_##tag(const T*, size_t); \
  HYLE_SIMD_C_API void hyle_simd_minmax_value_##tag( \
      const T*, size_t, T* out_min, T* out_max); \
  HYLE_SIMD_C_API int hyle_simd_is_sorted_##tag( \
      const T*, size_t, hyle_simd_sort_order)
HYLE_SIMD_DECLARE_SEARCH(int8_t, i8);
HYLE_SIMD_DECLARE_SEARCH(uint8_t, u8);
HYLE_SIMD_DECLARE_SEARCH(int16_t, i16);
HYLE_SIMD_DECLARE_SEARCH(uint16_t, u16);
HYLE_SIMD_DECLARE_SEARCH(int32_t, i32);
HYLE_SIMD_DECLARE_SEARCH(uint32_t, u32);
HYLE_SIMD_DECLARE_SEARCH(int64_t, i64);
HYLE_SIMD_DECLARE_SEARCH(uint64_t, u64);
HYLE_SIMD_DECLARE_SEARCH(float, f32);
HYLE_SIMD_DECLARE_SEARCH(double, f64);
#undef HYLE_SIMD_DECLARE_SEARCH

/* f16/bf16 whole-range products. Numeric equality follows IEEE == (-0 equals
 * +0, NaN never matches); *_equal_bits compares all 16 representation bits.
 * sum accumulates and returns f32. *_sorted_total uses IEEE-754 totalOrder and
 * therefore orders signed zero and individual NaN encodings deterministically. */
#define HYLE_SIMD_DECLARE_NARROW(T, tag) \
  HYLE_SIMD_C_API size_t hyle_simd_count_equal_##tag( \
      const T*, size_t, T); \
  HYLE_SIMD_C_API size_t hyle_simd_find_equal_##tag( \
      const T*, size_t, T); \
  HYLE_SIMD_C_API size_t hyle_simd_count_equal_bits_##tag( \
      const T*, size_t, T); \
  HYLE_SIMD_C_API size_t hyle_simd_find_equal_bits_##tag( \
      const T*, size_t, T); \
  HYLE_SIMD_C_API float hyle_simd_sum_##tag(const T*, size_t); \
  HYLE_SIMD_C_API int hyle_simd_is_sorted_total_##tag( \
      const T*, size_t, hyle_simd_sort_order)
HYLE_SIMD_DECLARE_NARROW(hyle_simd_f16, f16);
HYLE_SIMD_DECLARE_NARROW(hyle_simd_bf16, bf16);
#undef HYLE_SIMD_DECLARE_NARROW

HYLE_SIMD_C_API void hyle_simd_f16_to_f32(const hyle_simd_f16*, float*, size_t);
HYLE_SIMD_C_API void hyle_simd_f32_to_f16(const float*, hyle_simd_f16*, size_t);
HYLE_SIMD_C_API void hyle_simd_bf16_to_f32(const hyle_simd_bf16*, float*, size_t);
HYLE_SIMD_C_API void hyle_simd_f32_to_bf16(const float*, hyle_simd_bf16*, size_t);
HYLE_SIMD_C_API void hyle_simd_f8e4m3_to_f32(const hyle_simd_f8e4m3*, float*, size_t);
HYLE_SIMD_C_API void hyle_simd_f32_to_f8e4m3(const float*, hyle_simd_f8e4m3*, size_t);
HYLE_SIMD_C_API void hyle_simd_f8e5m2_to_f32(const hyle_simd_f8e5m2*, float*, size_t);
HYLE_SIMD_C_API void hyle_simd_f32_to_f8e5m2(const float*, hyle_simd_f8e5m2*, size_t);
HYLE_SIMD_C_API void hyle_simd_f16_to_f8e4m3(const hyle_simd_f16*,
                                             hyle_simd_f8e4m3*, size_t);
HYLE_SIMD_C_API void hyle_simd_f16_to_f8e5m2(const hyle_simd_f16*,
                                             hyle_simd_f8e5m2*, size_t);

#define HYLE_SIMD_DECLARE_UNARY(T, name) \
  HYLE_SIMD_C_API void name(const T* src, T* dst, size_t n)
#define HYLE_SIMD_DECLARE_BINARY(T, name) \
  HYLE_SIMD_C_API void name(const T* a, const T* b, T* dst, size_t n)
#define HYLE_SIMD_DECLARE_COMPLEX(T, name) \
  HYLE_SIMD_C_API void name(const T* src_re, const T* src_im, \
                            T* dst_re, T* dst_im, size_t n)
#define HYLE_SIMD_DECLARE_CPOW(T, name) \
  HYLE_SIMD_C_API void name(const T* z_re, const T* z_im, \
                            const T* w_re, const T* w_im, \
                            T* dst_re, T* dst_im, size_t n)

HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_exp_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_log_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_tanh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_sin_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_cos_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_exp2_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_log2_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_erf_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_atan_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_cbrt_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_expm1_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_log1p_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_asinh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_acosh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_acos_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_asin_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_sinh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_cosh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_tan_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_atanh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_log10_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_erfc_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_exp10_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_lgamma_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_tgamma_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_gelu_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_gelu_tanh_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_sigmoid_f32_batch);
HYLE_SIMD_DECLARE_UNARY(float, hyle_simd_silu_f32_batch);

HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_exp_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_log_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_exp_in_range_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_log_in_range_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_tanh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_sin_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_cos_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_exp2_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_log2_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_erf_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_atan_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_cbrt_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_expm1_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_log1p_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_asinh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_acosh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_acos_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_asin_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_sinh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_cosh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_tan_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_atanh_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_log10_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_erfc_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_exp10_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_lgamma_f64_batch);
HYLE_SIMD_DECLARE_UNARY(double, hyle_simd_tgamma_f64_batch);

HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_atan2_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_pow_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_fmod_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_remainder_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_nextafter_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_fdim_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_hypot_f32_batch);
HYLE_SIMD_DECLARE_BINARY(float, hyle_simd_hypot_in_range_f32_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_atan2_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_pow_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_fmod_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_remainder_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_nextafter_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_fdim_f64_batch);
HYLE_SIMD_DECLARE_BINARY(double, hyle_simd_hypot_f64_batch);

HYLE_SIMD_C_API void hyle_simd_sincos_f32_batch(const float*, float*, float*, size_t);
HYLE_SIMD_C_API void hyle_simd_sincos_f64_batch(const double*, double*, double*, size_t);
HYLE_SIMD_C_API void hyle_simd_softmax_f32(float* data, size_t n);
HYLE_SIMD_C_API void hyle_simd_log_softmax_f32(float* data, size_t n);

HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_cexp_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_clog_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_csqrt_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_csin_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_ccos_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_ctan_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_csinh_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_ccosh_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(float, hyle_simd_ctanh_f32_batch);
HYLE_SIMD_DECLARE_CPOW(float, hyle_simd_cpow_f32_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_cexp_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_clog_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_csqrt_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_csin_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_ccos_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_ctan_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_csinh_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_ccosh_f64_batch);
HYLE_SIMD_DECLARE_COMPLEX(double, hyle_simd_ctanh_f64_batch);
HYLE_SIMD_DECLARE_CPOW(double, hyle_simd_cpow_f64_batch);

#undef HYLE_SIMD_DECLARE_CPOW
#undef HYLE_SIMD_DECLARE_COMPLEX
#undef HYLE_SIMD_DECLARE_BINARY
#undef HYLE_SIMD_DECLARE_UNARY

#define HYLE_SIMD_DECLARE_SORT(T, tag) \
  HYLE_SIMD_C_API void hyle_simd_sort_##tag(T*, size_t, hyle_simd_sort_order); \
  HYLE_SIMD_C_API void hyle_simd_partial_sort_##tag( \
      T*, size_t, size_t, hyle_simd_sort_order); \
  HYLE_SIMD_C_API void hyle_simd_select_##tag( \
      T*, size_t, size_t, hyle_simd_sort_order)
HYLE_SIMD_DECLARE_SORT(uint16_t, u16);
HYLE_SIMD_DECLARE_SORT(int16_t, i16);
HYLE_SIMD_DECLARE_SORT(uint32_t, u32);
HYLE_SIMD_DECLARE_SORT(int32_t, i32);
HYLE_SIMD_DECLARE_SORT(uint64_t, u64);
HYLE_SIMD_DECLARE_SORT(int64_t, i64);
HYLE_SIMD_DECLARE_SORT(float, f32);
HYLE_SIMD_DECLARE_SORT(double, f64);
HYLE_SIMD_DECLARE_SORT(hyle_simd_u128, u128);
HYLE_SIMD_DECLARE_SORT(hyle_simd_kv32, kv32);
HYLE_SIMD_DECLARE_SORT(hyle_simd_kv64, kv64);
#undef HYLE_SIMD_DECLARE_SORT

#define HYLE_SIMD_DECLARE_RADIX(T, tag) \
  HYLE_SIMD_C_API void hyle_simd_radix_sort_##tag( \
      T*, size_t, hyle_simd_sort_order)
HYLE_SIMD_DECLARE_RADIX(uint32_t, u32);
HYLE_SIMD_DECLARE_RADIX(int32_t, i32);
HYLE_SIMD_DECLARE_RADIX(uint64_t, u64);
HYLE_SIMD_DECLARE_RADIX(int64_t, i64);
HYLE_SIMD_DECLARE_RADIX(float, f32);
#undef HYLE_SIMD_DECLARE_RADIX

#define HYLE_SIMD_DECLARE_ARGSORT(T, tag) \
  HYLE_SIMD_C_API void hyle_simd_argsort_##tag( \
      const T*, uint32_t*, size_t, hyle_simd_sort_order)
HYLE_SIMD_DECLARE_ARGSORT(uint32_t, u32);
HYLE_SIMD_DECLARE_ARGSORT(int32_t, i32);
HYLE_SIMD_DECLARE_ARGSORT(float, f32);
#undef HYLE_SIMD_DECLARE_ARGSORT

static inline size_t hyle_simd_packed_size_u8(size_t count, unsigned bits) {
  return (count / 8u) * bits + ((count % 8u) * bits + 7u) / 8u;
}
static inline size_t hyle_simd_packed_size_u16(size_t count, unsigned bits) {
  return (count / 16u) * bits + ((count % 16u) * bits + 15u) / 16u;
}
static inline size_t hyle_simd_packed_size_u32(size_t count, unsigned bits) {
  return (count / 32u) * bits + ((count % 32u) * bits + 31u) / 32u;
}
#define HYLE_SIMD_DECLARE_BITPACK(T, tag) \
  HYLE_SIMD_C_API void hyle_simd_pack_##tag( \
      const T*, size_t, T*, unsigned); \
  HYLE_SIMD_C_API void hyle_simd_unpack_##tag( \
      const T*, size_t, T*, unsigned)
HYLE_SIMD_DECLARE_BITPACK(uint8_t, u8);
HYLE_SIMD_DECLARE_BITPACK(uint16_t, u16);
HYLE_SIMD_DECLARE_BITPACK(uint32_t, u32);
#undef HYLE_SIMD_DECLARE_BITPACK

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* HYLE_C_SIMD_H_ */
