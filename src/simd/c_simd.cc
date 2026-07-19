#include "hyle/c/simd.h"

#include <cstdlib>
#include <new>
#include <type_traits>

#include "hyle/simd/bitpack.h"
#include "hyle/simd/dtype_ops.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/random.h"
#include "hyle/simd/runtime.h"
#include "hyle/simd/scan.h"
#include "hyle/simd/search.h"
#include "hyle/simd/sort.h"

struct hyle_simd_random_o {
  explicit hyle_simd_random_o(std::uint64_t seed) noexcept : state(seed) {}
  hyle::simd::RandomState state;
};

namespace {
thread_local const char* g_last_error = "";

hyle::simd::SortOrder cpp_order(hyle_simd_sort_order order) noexcept {
  switch (order) {
    case HYLE_SIMD_SORT_ASCENDING: return hyle::simd::SortOrder::kAscending;
    case HYLE_SIMD_SORT_DESCENDING: return hyle::simd::SortOrder::kDescending;
    default: std::abort();
  }
}

hyle::simd::Isa cpp_isa(hyle_simd_isa isa) noexcept {
  return static_cast<hyle::simd::Isa>(static_cast<std::uint32_t>(isa));
}

hyle_simd_module_info c_module(hyle::simd::Runtime::Module module) noexcept {
  return {static_cast<hyle_simd_isa>(static_cast<std::uint32_t>(module.isa)),
          0, module.features};
}

static_assert(std::is_same_v<hyle::simd::float16_t, hyle_simd_f16>);
static_assert(std::is_same_v<hyle::simd::bfloat16_t, hyle_simd_bf16>);
static_assert(std::is_same_v<hyle::simd::float8_e4m3_t, hyle_simd_f8e4m3>);
static_assert(std::is_same_v<hyle::simd::float8_e5m2_t, hyle_simd_f8e5m2>);
static_assert(std::is_same_v<hyle::simd::UInt128, hyle_simd_u128>);
static_assert(std::is_same_v<hyle::simd::KeyValue32, hyle_simd_kv32>);
static_assert(std::is_same_v<hyle::simd::KeyValue64, hyle_simd_kv64>);
static_assert(std::is_standard_layout_v<hyle_simd_runtime_info>);
static_assert(std::is_trivially_copyable_v<hyle_simd_runtime_info>);
static_assert(std::is_standard_layout_v<hyle_simd_module_info>);
static_assert(std::is_trivially_copyable_v<hyle_simd_module_info>);
static_assert(sizeof(hyle_simd_module_info) == 16);
static_assert(offsetof(hyle_simd_module_info, isa) == 0);
static_assert(offsetof(hyle_simd_module_info, reserved) == 4);
static_assert(offsetof(hyle_simd_module_info, features) == 8);
static_assert(HYLE_SIMD_RUNTIME_INFO_V1_SIZE == sizeof(hyle_simd_runtime_info));
static_assert(sizeof(hyle_simd_u128) == 16 && alignof(hyle_simd_u128) == 16);
static_assert(sizeof(hyle_simd_kv32) == 8 && alignof(hyle_simd_kv32) == 8);
static_assert(sizeof(hyle_simd_kv64) == 16 && alignof(hyle_simd_kv64) == 16);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kScalar) ==
              HYLE_SIMD_ISA_SCALAR);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kSse42) ==
              HYLE_SIMD_ISA_SSE42);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kAvx2) ==
              HYLE_SIMD_ISA_AVX2);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kAvx512) ==
              HYLE_SIMD_ISA_AVX512);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kNeon) ==
              HYLE_SIMD_ISA_NEON);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kSve) ==
              HYLE_SIMD_ISA_SVE);
static_assert(static_cast<std::uint32_t>(hyle::simd::Isa::kRvv) ==
              HYLE_SIMD_ISA_RVV);
static_assert(static_cast<unsigned>(hyle::simd::SortOrder::kAscending) ==
              HYLE_SIMD_SORT_ASCENDING);
static_assert(static_cast<unsigned>(hyle::simd::SortOrder::kDescending) ==
              HYLE_SIMD_SORT_DESCENDING);
static_assert(hyle::simd::Runtime::kFeatureF16c == HYLE_SIMD_FEATURE_F16C);
static_assert(hyle::simd::Runtime::kFeatureAvx512Bf16 ==
              HYLE_SIMD_FEATURE_AVX512_BF16);
static_assert(hyle::simd::Runtime::kFeatureAvx10_2 == HYLE_SIMD_FEATURE_AVX10_2);
static_assert(hyle::simd::Runtime::kFeatureAvxVnni ==
              HYLE_SIMD_FEATURE_AVX_VNNI);
static_assert(hyle::simd::Runtime::kFeatureAvx512Vnni ==
              HYLE_SIMD_FEATURE_AVX512_VNNI);
static_assert(hyle::simd::Runtime::kFeatureMathExactLibmScalar ==
              HYLE_SIMD_FEATURE_MATH_EXACT_LIBM_SCALAR);
static_assert(hyle::simd::Runtime::kFeatureDtypeFp8Scalar ==
              HYLE_SIMD_FEATURE_DTYPE_FP8_SCALAR);
static_assert(hyle::simd::Runtime::kFeaturePerOpLowerTier ==
              HYLE_SIMD_FEATURE_PER_OP_LOWER_TIER);
static_assert(hyle::simd::Runtime::kFeatureSortCappedNetwork ==
              HYLE_SIMD_FEATURE_SORT_CAPPED_NETWORK);
static_assert(hyle::simd::Runtime::kFeatureBitpackLogical128 ==
              HYLE_SIMD_FEATURE_BITPACK_LOGICAL128);
}  // namespace

extern "C" {

hyle_simd_isa hyle_simd_active_isa(void) {
  return static_cast<hyle_simd_isa>(
      static_cast<std::uint32_t>(hyle::simd::runtime().isa));
}

int hyle_simd_supports_isa(hyle_simd_isa isa) {
  switch (isa) {
    case HYLE_SIMD_ISA_SCALAR:
    case HYLE_SIMD_ISA_SSE42:
    case HYLE_SIMD_ISA_AVX2:
    case HYLE_SIMD_ISA_AVX512:
    case HYLE_SIMD_ISA_NEON:
    case HYLE_SIMD_ISA_SVE:
    case HYLE_SIMD_ISA_RVV:
      break;
    default:
      return 0;
  }
  return hyle::simd::supports(cpp_isa(isa)) ? 1 : 0;
}

int hyle_simd_runtime_info_get(hyle_simd_runtime_info* out) {
  if (out == nullptr || out->struct_size < HYLE_SIMD_RUNTIME_INFO_V1_SIZE) {
    g_last_error = "runtime info is null or struct_size is too small";
    return 0;
  }
  const auto& runtime = hyle::simd::runtime();
  out->isa = static_cast<hyle_simd_isa>(
      static_cast<std::uint32_t>(runtime.isa));
  out->ops = c_module(runtime.ops_module);
  out->scan = c_module(runtime.scan);
  out->dtype = c_module(runtime.dtype);
  out->math = c_module(runtime.math);
  out->sort = c_module(runtime.sort);
  out->bitpack = c_module(runtime.bitpack);
  g_last_error = "";
  return 1;
}

const char* hyle_simd_last_error(void) { return g_last_error; }

float hyle_simd_dot_f32(const float* x, const float* y, size_t n) {
  return hyle::simd::dot(x, y, n);
}
double hyle_simd_dot_f64(const double* x, const double* y, size_t n) {
  return hyle::simd::dot(x, y, n);
}
float hyle_simd_dot_bf16(const hyle_simd_bf16* x,
                         const hyle_simd_bf16* y, size_t n) {
  return hyle::simd::dot(x, y, n);
}
float hyle_simd_dot_f32_bf16(const float* x, const hyle_simd_bf16* y,
                             size_t n) {
  return hyle::simd::dot(x, y, n);
}
float hyle_simd_dot_f16(const hyle_simd_f16* x, const hyle_simd_f16* y,
                        size_t n) {
  return hyle::simd::dot(x, y, n);
}
int64_t hyle_simd_dot_i16(const int16_t* x, const int16_t* y, size_t n) {
  return hyle::simd::dot(x, y, n);
}
int64_t hyle_simd_dot_i8(const int8_t* x, const int8_t* y, size_t n) {
  return hyle::simd::dot(x, y, n);
}
void hyle_simd_saxpy_f32(float a, const float* x, float* y, size_t n) {
  hyle::simd::saxpy(a, x, y, n);
}
void hyle_simd_saxpy_f64(double a, const double* x, double* y, size_t n) {
  hyle::simd::saxpy(a, x, y, n);
}
void hyle_simd_add_f32(const float* x, const float* y, float* out, size_t n) {
  hyle::simd::add(x, y, out, n);
}
void hyle_simd_add_f64(const double* x, const double* y, double* out, size_t n) {
  hyle::simd::add(x, y, out, n);
}

hyle_simd_random_o* hyle_simd_random_create(uint64_t seed) {
  auto* state = new (std::nothrow) hyle_simd_random_o(seed);
  g_last_error = state != nullptr ? "" : "random state allocation failed";
  return state;
}
void hyle_simd_random_destroy(hyle_simd_random_o* state) { delete state; }
void hyle_simd_random_fill_u64(hyle_simd_random_o* state, uint64_t* dst,
                               size_t count) {
  if (state != nullptr) hyle::simd::random_fill(state->state, dst, count);
}
void hyle_simd_random_fill_f32(hyle_simd_random_o* state, float* dst,
                               size_t count) {
  if (state != nullptr) hyle::simd::random_fill(state->state, dst, count);
}
void hyle_simd_random_fill_f64(hyle_simd_random_o* state, double* dst,
                               size_t count) {
  if (state != nullptr) hyle::simd::random_fill(state->state, dst, count);
}

#define HYLE_C_SCAN(T, tag) \
  void hyle_simd_inclusive_scan_##tag(T* data, size_t n) { \
    hyle::simd::inclusive_scan(data, n); \
  } \
  void hyle_simd_exclusive_scan_##tag(T* data, size_t n) { \
    hyle::simd::exclusive_scan(data, n); \
  }
HYLE_C_SCAN(uint32_t, u32)
HYLE_C_SCAN(int32_t, i32)
HYLE_C_SCAN(uint64_t, u64)
HYLE_C_SCAN(int64_t, i64)
HYLE_C_SCAN(float, f32)
HYLE_C_SCAN(double, f64)
#undef HYLE_C_SCAN

#define HYLE_C_SEARCH(T, tag) \
  size_t hyle_simd_count_equal_##tag(const T* data, size_t n, T value) { \
    return hyle::simd::count_equal(data, n, value); \
  } \
  size_t hyle_simd_find_equal_##tag(const T* data, size_t n, T value) { \
    return hyle::simd::find_equal(data, n, value); \
  } \
  T hyle_simd_min_value_##tag(const T* data, size_t n) { \
    return hyle::simd::min_value(data, n); \
  } \
  T hyle_simd_max_value_##tag(const T* data, size_t n) { \
    return hyle::simd::max_value(data, n); \
  } \
  void hyle_simd_minmax_value_##tag( \
      const T* data, size_t n, T* out_min, T* out_max) { \
    hyle::simd::minmax_value(data, n, out_min, out_max); \
  } \
  int hyle_simd_is_sorted_##tag( \
      const T* data, size_t n, hyle_simd_sort_order order) { \
    return hyle::simd::is_sorted( \
        data, n, cpp_order(order) == hyle::simd::SortOrder::kAscending) ? 1 : 0; \
  }
HYLE_C_SEARCH(int8_t, i8)
HYLE_C_SEARCH(uint8_t, u8)
HYLE_C_SEARCH(int16_t, i16)
HYLE_C_SEARCH(uint16_t, u16)
HYLE_C_SEARCH(int32_t, i32)
HYLE_C_SEARCH(uint32_t, u32)
HYLE_C_SEARCH(int64_t, i64)
HYLE_C_SEARCH(uint64_t, u64)
HYLE_C_SEARCH(float, f32)
HYLE_C_SEARCH(double, f64)
#undef HYLE_C_SEARCH

#define HYLE_C_DTYPE(Src, Dst, c_name, cpp_name) \
  void hyle_simd_##c_name(const Src* src, Dst* dst, size_t n) { \
    hyle::simd::cpp_name##_batch(src, dst, n); \
  }
HYLE_C_DTYPE(hyle_simd_f16, float, f16_to_f32, f16_to_f32)
HYLE_C_DTYPE(float, hyle_simd_f16, f32_to_f16, f32_to_f16)
HYLE_C_DTYPE(hyle_simd_bf16, float, bf16_to_f32, bf16_to_f32)
HYLE_C_DTYPE(float, hyle_simd_bf16, f32_to_bf16, f32_to_bf16)
HYLE_C_DTYPE(hyle_simd_f8e4m3, float, f8e4m3_to_f32, f8e4m3_to_f32)
HYLE_C_DTYPE(float, hyle_simd_f8e4m3, f32_to_f8e4m3, f32_to_f8e4m3)
HYLE_C_DTYPE(hyle_simd_f8e5m2, float, f8e5m2_to_f32, f8e5m2_to_f32)
HYLE_C_DTYPE(float, hyle_simd_f8e5m2, f32_to_f8e5m2, f32_to_f8e5m2)
HYLE_C_DTYPE(hyle_simd_f16, hyle_simd_f8e4m3, f16_to_f8e4m3, f16_to_f8e4m3)
HYLE_C_DTYPE(hyle_simd_f16, hyle_simd_f8e5m2, f16_to_f8e5m2, f16_to_f8e5m2)
#undef HYLE_C_DTYPE

#define HYLE_C_UNARY(T, name) \
  void hyle_simd_##name(const T* src, T* dst, size_t n) { \
    hyle::simd::name(src, dst, n); \
  }
#define HYLE_C_BINARY(T, name) \
  void hyle_simd_##name(const T* a, const T* b, T* dst, size_t n) { \
    hyle::simd::name(a, b, dst, n); \
  }
#define HYLE_C_COMPLEX(T, name) \
  void hyle_simd_##name(const T* src_re, const T* src_im, \
                        T* dst_re, T* dst_im, size_t n) { \
    hyle::simd::name(src_re, src_im, dst_re, dst_im, n); \
  }
#define HYLE_C_CPOW(T, name) \
  void hyle_simd_##name(const T* z_re, const T* z_im, \
                        const T* w_re, const T* w_im, \
                        T* dst_re, T* dst_im, size_t n) { \
    hyle::simd::name(z_re, z_im, w_re, w_im, dst_re, dst_im, n); \
  }

#define HYLE_C_UNARY_F32(name) HYLE_C_UNARY(float, name)
HYLE_C_UNARY_F32(exp_f32_batch)
HYLE_C_UNARY_F32(log_f32_batch)
HYLE_C_UNARY_F32(tanh_f32_batch)
HYLE_C_UNARY_F32(sin_f32_batch)
HYLE_C_UNARY_F32(cos_f32_batch)
HYLE_C_UNARY_F32(exp2_f32_batch)
HYLE_C_UNARY_F32(log2_f32_batch)
HYLE_C_UNARY_F32(erf_f32_batch)
HYLE_C_UNARY_F32(atan_f32_batch)
HYLE_C_UNARY_F32(cbrt_f32_batch)
HYLE_C_UNARY_F32(expm1_f32_batch)
HYLE_C_UNARY_F32(log1p_f32_batch)
HYLE_C_UNARY_F32(asinh_f32_batch)
HYLE_C_UNARY_F32(acosh_f32_batch)
HYLE_C_UNARY_F32(acos_f32_batch)
HYLE_C_UNARY_F32(asin_f32_batch)
HYLE_C_UNARY_F32(sinh_f32_batch)
HYLE_C_UNARY_F32(cosh_f32_batch)
HYLE_C_UNARY_F32(tan_f32_batch)
HYLE_C_UNARY_F32(atanh_f32_batch)
HYLE_C_UNARY_F32(log10_f32_batch)
HYLE_C_UNARY_F32(erfc_f32_batch)
HYLE_C_UNARY_F32(exp10_f32_batch)
HYLE_C_UNARY_F32(lgamma_f32_batch)
HYLE_C_UNARY_F32(tgamma_f32_batch)
HYLE_C_UNARY_F32(gelu_f32_batch)
HYLE_C_UNARY_F32(gelu_tanh_f32_batch)
HYLE_C_UNARY_F32(sigmoid_f32_batch)
HYLE_C_UNARY_F32(silu_f32_batch)
#undef HYLE_C_UNARY_F32

#define HYLE_C_UNARY_F64(name) HYLE_C_UNARY(double, name)
HYLE_C_UNARY_F64(exp_f64_batch)
HYLE_C_UNARY_F64(log_f64_batch)
HYLE_C_UNARY_F64(exp_in_range_f64_batch)
HYLE_C_UNARY_F64(log_in_range_f64_batch)
HYLE_C_UNARY_F64(tanh_f64_batch)
HYLE_C_UNARY_F64(sin_f64_batch)
HYLE_C_UNARY_F64(cos_f64_batch)
HYLE_C_UNARY_F64(exp2_f64_batch)
HYLE_C_UNARY_F64(log2_f64_batch)
HYLE_C_UNARY_F64(erf_f64_batch)
HYLE_C_UNARY_F64(atan_f64_batch)
HYLE_C_UNARY_F64(cbrt_f64_batch)
HYLE_C_UNARY_F64(expm1_f64_batch)
HYLE_C_UNARY_F64(log1p_f64_batch)
HYLE_C_UNARY_F64(asinh_f64_batch)
HYLE_C_UNARY_F64(acosh_f64_batch)
HYLE_C_UNARY_F64(acos_f64_batch)
HYLE_C_UNARY_F64(asin_f64_batch)
HYLE_C_UNARY_F64(sinh_f64_batch)
HYLE_C_UNARY_F64(cosh_f64_batch)
HYLE_C_UNARY_F64(tan_f64_batch)
HYLE_C_UNARY_F64(atanh_f64_batch)
HYLE_C_UNARY_F64(log10_f64_batch)
HYLE_C_UNARY_F64(erfc_f64_batch)
HYLE_C_UNARY_F64(exp10_f64_batch)
HYLE_C_UNARY_F64(lgamma_f64_batch)
HYLE_C_UNARY_F64(tgamma_f64_batch)
#undef HYLE_C_UNARY_F64

HYLE_C_BINARY(float, atan2_f32_batch)
HYLE_C_BINARY(float, pow_f32_batch)
HYLE_C_BINARY(float, fmod_f32_batch)
HYLE_C_BINARY(float, remainder_f32_batch)
HYLE_C_BINARY(float, nextafter_f32_batch)
HYLE_C_BINARY(float, fdim_f32_batch)
HYLE_C_BINARY(float, hypot_f32_batch)
HYLE_C_BINARY(float, hypot_in_range_f32_batch)
HYLE_C_BINARY(double, atan2_f64_batch)
HYLE_C_BINARY(double, pow_f64_batch)
HYLE_C_BINARY(double, fmod_f64_batch)
HYLE_C_BINARY(double, remainder_f64_batch)
HYLE_C_BINARY(double, nextafter_f64_batch)
HYLE_C_BINARY(double, fdim_f64_batch)
HYLE_C_BINARY(double, hypot_f64_batch)

void hyle_simd_sincos_f32_batch(const float* src, float* s, float* c, size_t n) {
  hyle::simd::sincos_f32_batch(src, s, c, n);
}
void hyle_simd_sincos_f64_batch(const double* src, double* s, double* c,
                                size_t n) {
  hyle::simd::sincos_f64_batch(src, s, c, n);
}
void hyle_simd_softmax_f32(float* data, size_t n) {
  hyle::simd::softmax_f32(data, n);
}
void hyle_simd_log_softmax_f32(float* data, size_t n) {
  hyle::simd::log_softmax_f32(data, n);
}

#define HYLE_C_COMPLEX_F32(name) HYLE_C_COMPLEX(float, name)
HYLE_C_COMPLEX_F32(cexp_f32_batch)
HYLE_C_COMPLEX_F32(clog_f32_batch)
HYLE_C_COMPLEX_F32(csqrt_f32_batch)
HYLE_C_COMPLEX_F32(csin_f32_batch)
HYLE_C_COMPLEX_F32(ccos_f32_batch)
HYLE_C_COMPLEX_F32(ctan_f32_batch)
HYLE_C_COMPLEX_F32(csinh_f32_batch)
HYLE_C_COMPLEX_F32(ccosh_f32_batch)
HYLE_C_COMPLEX_F32(ctanh_f32_batch)
#undef HYLE_C_COMPLEX_F32
HYLE_C_CPOW(float, cpow_f32_batch)
#define HYLE_C_COMPLEX_F64(name) HYLE_C_COMPLEX(double, name)
HYLE_C_COMPLEX_F64(cexp_f64_batch)
HYLE_C_COMPLEX_F64(clog_f64_batch)
HYLE_C_COMPLEX_F64(csqrt_f64_batch)
HYLE_C_COMPLEX_F64(csin_f64_batch)
HYLE_C_COMPLEX_F64(ccos_f64_batch)
HYLE_C_COMPLEX_F64(ctan_f64_batch)
HYLE_C_COMPLEX_F64(csinh_f64_batch)
HYLE_C_COMPLEX_F64(ccosh_f64_batch)
HYLE_C_COMPLEX_F64(ctanh_f64_batch)
#undef HYLE_C_COMPLEX_F64
HYLE_C_CPOW(double, cpow_f64_batch)

#undef HYLE_C_CPOW
#undef HYLE_C_COMPLEX
#undef HYLE_C_BINARY
#undef HYLE_C_UNARY

#define HYLE_C_SORT(T, tag) \
  void hyle_simd_sort_##tag(T* data, size_t n, hyle_simd_sort_order order) { \
    hyle::simd::sort(data, n, cpp_order(order)); \
  } \
  void hyle_simd_partial_sort_##tag( \
      T* data, size_t n, size_t k, hyle_simd_sort_order order) { \
    hyle::simd::partial_sort(data, n, k, cpp_order(order)); \
  } \
  void hyle_simd_select_##tag( \
      T* data, size_t n, size_t k, hyle_simd_sort_order order) { \
    hyle::simd::select(data, n, k, cpp_order(order)); \
  }
HYLE_C_SORT(uint16_t, u16)
HYLE_C_SORT(int16_t, i16)
HYLE_C_SORT(uint32_t, u32)
HYLE_C_SORT(int32_t, i32)
HYLE_C_SORT(uint64_t, u64)
HYLE_C_SORT(int64_t, i64)
HYLE_C_SORT(float, f32)
HYLE_C_SORT(double, f64)
HYLE_C_SORT(hyle_simd_u128, u128)
HYLE_C_SORT(hyle_simd_kv32, kv32)
HYLE_C_SORT(hyle_simd_kv64, kv64)
#undef HYLE_C_SORT

#define HYLE_C_RADIX(T, tag) \
  void hyle_simd_radix_sort_##tag( \
      T* data, size_t n, hyle_simd_sort_order order) { \
    hyle::simd::radix_sort(data, n, cpp_order(order)); \
  }
HYLE_C_RADIX(uint32_t, u32)
HYLE_C_RADIX(int32_t, i32)
HYLE_C_RADIX(uint64_t, u64)
HYLE_C_RADIX(int64_t, i64)
HYLE_C_RADIX(float, f32)
#undef HYLE_C_RADIX

#define HYLE_C_ARGSORT(T, tag) \
  void hyle_simd_argsort_##tag( \
      const T* data, uint32_t* indices, size_t n, \
      hyle_simd_sort_order order) { \
    hyle::simd::argsort(data, indices, n, cpp_order(order)); \
  }
HYLE_C_ARGSORT(uint32_t, u32)
HYLE_C_ARGSORT(int32_t, i32)
HYLE_C_ARGSORT(float, f32)
#undef HYLE_C_ARGSORT

#define HYLE_C_BITPACK(T, tag) \
  void hyle_simd_pack_##tag( \
      const T* raw, size_t n, T* packed, unsigned bits) { \
    hyle::simd::pack(raw, n, packed, bits); \
  } \
  void hyle_simd_unpack_##tag( \
      const T* packed, size_t n, T* raw, unsigned bits) { \
    hyle::simd::unpack(packed, n, raw, bits); \
  }
HYLE_C_BITPACK(uint8_t, u8)
HYLE_C_BITPACK(uint16_t, u16)
HYLE_C_BITPACK(uint32_t, u32)
#undef HYLE_C_BITPACK

}  // extern "C"
