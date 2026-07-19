#include "hyle/c/simd.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* The ABI coverage script slices preprocessed output at this marker, so
 * declarations pulled in by simd.h cannot masquerade as test references. */
enum { HYLE_C_SIMD_COVERAGE_BODY_MARKER = 0 };

static int failures;
#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
  ++failures; } } while (0)

#define TEST_SCAN(T, tag) do { \
  T v[4] = {(T)1, (T)2, (T)3, (T)4}; \
  hyle_simd_inclusive_scan_##tag(v, 4); CHECK(v[3] == (T)10); \
  hyle_simd_exclusive_scan_##tag(v, 4); CHECK(v[0] == (T)0); \
} while (0)

#define TEST_SEARCH(T, tag) do { \
  T v[4] = {(T)3, (T)1, (T)2, (T)2}, lo, hi; \
  CHECK(hyle_simd_count_equal_##tag(v, 4, (T)2) == 2); \
  CHECK(hyle_simd_find_equal_##tag(v, 4, (T)1) == 1); \
  CHECK(hyle_simd_min_value_##tag(v, 4) == (T)1); \
  CHECK(hyle_simd_max_value_##tag(v, 4) == (T)3); \
  hyle_simd_minmax_value_##tag(v, 4, &lo, &hi); \
  CHECK(lo == (T)1 && hi == (T)3); \
  CHECK(!hyle_simd_is_sorted_##tag(v, 4, HYLE_SIMD_SORT_ASCENDING)); \
} while (0)

#define CALL_UNARY(T, name) do { T x[2] = {(T)1.25, (T)1.5}, y[2]; \
  name(x, y, 2); } while (0)
#define CALL_BINARY(T, name) do { \
  T x[2] = {(T)1.25, (T)1.5}, y[2] = {(T)0.75, (T)2}, z[2]; \
  name(x, y, z, 2); } while (0)
#define CALL_COMPLEX(T, name) do { \
  T r[2] = {(T)1, (T).5}, i[2] = {(T).25, (T)-.5}, ro[2], io[2]; \
  name(r, i, ro, io, 2); } while (0)

#define CALL_ALL_F32_UNARY(M) \
  M(exp_f32_batch); M(log_f32_batch); M(tanh_f32_batch); M(sin_f32_batch); \
  M(cos_f32_batch); M(exp2_f32_batch); M(log2_f32_batch); M(erf_f32_batch); \
  M(atan_f32_batch); M(cbrt_f32_batch); M(expm1_f32_batch); M(log1p_f32_batch); \
  M(asinh_f32_batch); M(acosh_f32_batch); M(acos_f32_batch); M(asin_f32_batch); \
  M(sinh_f32_batch); M(cosh_f32_batch); M(tan_f32_batch); M(atanh_f32_batch); \
  M(log10_f32_batch); M(erfc_f32_batch); M(exp10_f32_batch); \
  M(lgamma_f32_batch); M(tgamma_f32_batch); M(gelu_f32_batch); \
  M(gelu_tanh_f32_batch); M(sigmoid_f32_batch); M(silu_f32_batch)
#define CALL_ALL_F64_UNARY(M) \
  M(exp_f64_batch); M(log_f64_batch); M(exp_in_range_f64_batch); \
  M(log_in_range_f64_batch); M(tanh_f64_batch); M(sin_f64_batch); \
  M(cos_f64_batch); M(exp2_f64_batch); M(log2_f64_batch); M(erf_f64_batch); \
  M(atan_f64_batch); M(cbrt_f64_batch); M(expm1_f64_batch); M(log1p_f64_batch); \
  M(asinh_f64_batch); M(acosh_f64_batch); M(acos_f64_batch); M(asin_f64_batch); \
  M(sinh_f64_batch); M(cosh_f64_batch); M(tan_f64_batch); M(atanh_f64_batch); \
  M(log10_f64_batch); M(erfc_f64_batch); M(exp10_f64_batch); \
  M(lgamma_f64_batch); M(tgamma_f64_batch)

int main(void) {
  hyle_simd_runtime_info info = HYLE_SIMD_RUNTIME_INFO_INIT;
  CHECK(hyle_simd_runtime_info_get(&info));
  CHECK(info.isa == hyle_simd_active_isa());
  CHECK(hyle_simd_supports_isa(HYLE_SIMD_ISA_SCALAR));
  CHECK(!hyle_simd_supports_isa(13));
  CHECK(info.math.reserved == 0);
  CHECK((info.math.features & HYLE_SIMD_FEATURE_MATH_EXACT_LIBM_SCALAR) != 0);
  CHECK((info.dtype.features & HYLE_SIMD_FEATURE_DTYPE_FP8_SCALAR) != 0);
  info.struct_size = HYLE_SIMD_RUNTIME_INFO_V1_SIZE;
  CHECK(hyle_simd_runtime_info_get(&info));
  info.struct_size = 0;
  CHECK(!hyle_simd_runtime_info_get(&info));
  CHECK(strlen(hyle_simd_last_error()) != 0);

  {
    float x[3] = {1, 2, 3}, y[3] = {4, 5, 6}, z[3];
    double dx[2] = {1, 2}, dy[2] = {3, 4}, dz[2];
    int8_t i8x[3] = {-128, 127, 3}, i8y[3] = {-128, 127, -4};
    int16_t i16x[2] = {-32768, 32767}, i16y[2] = {-32768, 32767};
    CHECK(fabsf(hyle_simd_dot_f32(x, y, 3) - 32) < 1e-6f);
    CHECK(hyle_simd_dot_f64(dx, dy, 2) == 11);
    CHECK(hyle_simd_dot_i8(i8x, i8y, 3) == 32501);
    CHECK(hyle_simd_dot_i16(i16x, i16y, 2) == 2147418113);
    hyle_simd_add_f32(x, y, z, 3); hyle_simd_saxpy_f32(2, x, z, 3);
    hyle_simd_add_f64(dx, dy, dz, 2); hyle_simd_saxpy_f64(2, dx, dz, 2);
  }
  {
    float x[3] = {1, -2, .5f}, y[3];
    hyle_simd_f16 f16[3]; hyle_simd_bf16 bf16[3];
    hyle_simd_f8e4m3 e4[3]; hyle_simd_f8e5m2 e5[3];
    hyle_simd_f32_to_f16(x, f16, 3); hyle_simd_f16_to_f32(f16, y, 3);
    hyle_simd_f32_to_bf16(x, bf16, 3); hyle_simd_bf16_to_f32(bf16, y, 3);
    hyle_simd_f32_to_f8e4m3(x, e4, 3); hyle_simd_f8e4m3_to_f32(e4, y, 3);
    hyle_simd_f32_to_f8e5m2(x, e5, 3); hyle_simd_f8e5m2_to_f32(e5, y, 3);
    hyle_simd_f16_to_f8e4m3(f16, e4, 3); hyle_simd_f16_to_f8e5m2(f16, e5, 3);
    CHECK(hyle_simd_dot_f16(f16, f16, 3) > 5);
    CHECK(hyle_simd_dot_bf16(bf16, bf16, 3) > 5);
    CHECK(hyle_simd_dot_f32_bf16(x, bf16, 3) > 5);
  }
  {
    hyle_simd_random_o* a = hyle_simd_random_create(123);
    hyle_simd_random_o* b = hyle_simd_random_create(123);
    uint64_t ua[8], ub[8]; float f[8]; double d[8];
    CHECK(a && b);
    hyle_simd_random_fill_u64(a, ua, 8); hyle_simd_random_fill_u64(b, ub, 8);
    CHECK(memcmp(ua, ub, sizeof ua) == 0);
    hyle_simd_random_fill_f32(a, f, 8); hyle_simd_random_fill_f64(a, d, 8);
    hyle_simd_random_destroy(a); hyle_simd_random_destroy(b);
    hyle_simd_random_fill_u64(NULL, ua, 8);
    hyle_simd_random_fill_f32(NULL, f, 8);
    hyle_simd_random_fill_f64(NULL, d, 8);
    hyle_simd_random_destroy(NULL);
  }

  TEST_SCAN(uint32_t, u32); TEST_SCAN(int32_t, i32);
  TEST_SCAN(uint64_t, u64); TEST_SCAN(int64_t, i64);
  TEST_SCAN(float, f32); TEST_SCAN(double, f64);
  TEST_SEARCH(int8_t, i8); TEST_SEARCH(uint8_t, u8);
  TEST_SEARCH(int16_t, i16); TEST_SEARCH(uint16_t, u16);
  TEST_SEARCH(int32_t, i32); TEST_SEARCH(uint32_t, u32);
  TEST_SEARCH(int64_t, i64); TEST_SEARCH(uint64_t, u64);
  TEST_SEARCH(float, f32); TEST_SEARCH(double, f64);

#define U32(name) CALL_UNARY(float, hyle_simd_##name)
  CALL_ALL_F32_UNARY(U32);
#undef U32
#define U64(name) CALL_UNARY(double, hyle_simd_##name)
  CALL_ALL_F64_UNARY(U64);
#undef U64
#define B32(name) CALL_BINARY(float, hyle_simd_##name)
  B32(atan2_f32_batch); B32(pow_f32_batch); B32(fmod_f32_batch);
  B32(remainder_f32_batch); B32(nextafter_f32_batch); B32(fdim_f32_batch);
  B32(hypot_f32_batch); B32(hypot_in_range_f32_batch);
#undef B32
#define B64(name) CALL_BINARY(double, hyle_simd_##name)
  B64(atan2_f64_batch); B64(pow_f64_batch); B64(fmod_f64_batch);
  B64(remainder_f64_batch); B64(nextafter_f64_batch); B64(fdim_f64_batch);
  B64(hypot_f64_batch);
#undef B64
  {
    float x[2] = {1, 2}, s[2], c[2];
    double dx[2] = {1, 2}, ds[2], dc[2];
    hyle_simd_sincos_f32_batch(x, s, c, 2);
    hyle_simd_sincos_f64_batch(dx, ds, dc, 2);
    hyle_simd_softmax_f32(x, 2); hyle_simd_log_softmax_f32(x, 2);
  }
#define C32(name) CALL_COMPLEX(float, hyle_simd_##name)
  C32(cexp_f32_batch); C32(clog_f32_batch); C32(csqrt_f32_batch);
  C32(csin_f32_batch); C32(ccos_f32_batch); C32(ctan_f32_batch);
  C32(csinh_f32_batch); C32(ccosh_f32_batch); C32(ctanh_f32_batch);
#undef C32
#define C64(name) CALL_COMPLEX(double, hyle_simd_##name)
  C64(cexp_f64_batch); C64(clog_f64_batch); C64(csqrt_f64_batch);
  C64(csin_f64_batch); C64(ccos_f64_batch); C64(ctan_f64_batch);
  C64(csinh_f64_batch); C64(ccosh_f64_batch); C64(ctanh_f64_batch);
#undef C64
  {
    float r[1] = {1}, i[1] = {.25f}, wr[1] = {2}, wi[1] = {0}, ro[1], io[1];
    double dr[1] = {1}, di[1] = {.25}, dwr[1] = {2}, dwi[1] = {0}, dro[1], dio[1];
    hyle_simd_cpow_f32_batch(r, i, wr, wi, ro, io, 1);
    hyle_simd_cpow_f64_batch(dr, di, dwr, dwi, dro, dio, 1);
  }

#define SORT(T, tag) do { T v[3] = {(T)3, (T)1, (T)2}; \
  hyle_simd_sort_##tag(v, 3, HYLE_SIMD_SORT_ASCENDING); \
  hyle_simd_partial_sort_##tag(v, 3, 2, HYLE_SIMD_SORT_DESCENDING); \
  hyle_simd_select_##tag(v, 3, 1, HYLE_SIMD_SORT_ASCENDING); } while (0)
  SORT(uint16_t, u16); SORT(int16_t, i16); SORT(uint32_t, u32);
  SORT(int32_t, i32); SORT(uint64_t, u64); SORT(int64_t, i64);
  SORT(float, f32); SORT(double, f64);
#undef SORT
  {
    hyle_simd_u128 u[3] = {{3,0},{1,0},{2,0}};
    hyle_simd_kv32 a[3] = {{0,3},{1,1},{2,2}};
    hyle_simd_kv64 b[3] = {{0,3},{1,1},{2,2}};
#define SPECIAL(tag, v) hyle_simd_sort_##tag(v,3,HYLE_SIMD_SORT_ASCENDING); \
    hyle_simd_partial_sort_##tag(v,3,2,HYLE_SIMD_SORT_ASCENDING); \
    hyle_simd_select_##tag(v,3,1,HYLE_SIMD_SORT_ASCENDING)
    SPECIAL(u128, u); SPECIAL(kv32, a); SPECIAL(kv64, b);
#undef SPECIAL
  }
  {
    uint32_t u32[3]={3,1,2}, idx[3]; int32_t i32[3]={3,1,2};
    uint64_t u64[3]={3,1,2}; int64_t i64[3]={3,1,2}; float f[3]={3,1,2};
    hyle_simd_radix_sort_u32(u32,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_radix_sort_i32(i32,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_radix_sort_u64(u64,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_radix_sort_i64(i64,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_radix_sort_f32(f,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_argsort_u32(u32,idx,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_argsort_i32(i32,idx,3,HYLE_SIMD_SORT_ASCENDING);
    hyle_simd_argsort_f32(f,idx,3,HYLE_SIMD_SORT_ASCENDING);
  }
#define PACK(T,tag) do { T x[8]={0,1,2,3,4,5,6,7}, p[8], y[8]; \
  hyle_simd_pack_##tag(x,8,p,3); hyle_simd_unpack_##tag(p,8,y,3); \
  CHECK(memcmp(x,y,sizeof x)==0); } while(0)
  PACK(uint8_t,u8); PACK(uint16_t,u16); PACK(uint32_t,u32);
#undef PACK
  CHECK(hyle_simd_packed_size_u8(8, 3) == 3);
  CHECK(hyle_simd_packed_size_u16(16, 5) == 5);
  CHECK(hyle_simd_packed_size_u32(32, 1) == 1);
  CHECK(hyle_simd_packed_size_u8(SIZE_MAX, 8) == SIZE_MAX);
  CHECK(hyle_simd_packed_size_u16(SIZE_MAX, 16) == SIZE_MAX);
  CHECK(hyle_simd_packed_size_u32(SIZE_MAX, 32) == SIZE_MAX);

  if (!failures) puts("c_simd_test: all checks passed");
  return failures ? 1 : 0;
}
