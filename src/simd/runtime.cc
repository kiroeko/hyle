#include "src/simd/runtime_internal.h"

#include "hyle/core/environment.h"
#include "src/simd/bitpack/kernels.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/dtype/conversion_kernels.h"
#include "src/simd/dtype/kernels.h"
#include "src/simd/narrow/kernels.h"
#include "src/simd/ops/kernels.h"
#include "src/simd/random/kernels.h"
#include "src/simd/scan/kernels.h"
#include "src/simd/search/kernels.h"
#include "src/simd/sort/kernels.h"

namespace hyle::simd {
namespace {

#define HYLE_SEARCH_RUNTIME_ONE(T, tag, suffix)                                  \
  &detail::count_equal_##tag##_##suffix, &detail::find_equal_##tag##_##suffix,   \
      &detail::min_value_##tag##_##suffix, &detail::max_value_##tag##_##suffix,  \
      &detail::index_of_min_##tag##_##suffix,                                    \
      &detail::index_of_max_##tag##_##suffix,                                    \
      &detail::minmax_value_##tag##_##suffix,                                   \
      &detail::is_sorted_asc_##tag##_##suffix,                                  \
      &detail::is_sorted_desc_##tag##_##suffix,

#define HYLE_SEARCH_RUNTIME(suffix)                                               \
  {HYLE_SEARCH_RUNTIME_ONE(std::int8_t, i8, suffix)                               \
   HYLE_SEARCH_RUNTIME_ONE(std::uint8_t, u8, suffix)                              \
   HYLE_SEARCH_RUNTIME_ONE(std::int16_t, i16, suffix)                             \
   HYLE_SEARCH_RUNTIME_ONE(std::uint16_t, u16, suffix)                            \
   HYLE_SEARCH_RUNTIME_ONE(std::int32_t, i32, suffix)                             \
   HYLE_SEARCH_RUNTIME_ONE(std::uint32_t, u32, suffix)                            \
   HYLE_SEARCH_RUNTIME_ONE(std::int64_t, i64, suffix)                             \
   HYLE_SEARCH_RUNTIME_ONE(std::uint64_t, u64, suffix)                            \
   HYLE_SEARCH_RUNTIME_ONE(float, f32, suffix)                                    \
   HYLE_SEARCH_RUNTIME_ONE(double, f64, suffix)}

#define HYLE_SCAN_RUNTIME_ONE(T, tag, suffix)                         \
  &detail::inclusive_scan_##tag##_##suffix,                           \
      &detail::exclusive_scan_##tag##_##suffix,

#define HYLE_SCAN_RUNTIME(suffix)                                  \
  {HYLE_SCAN_RUNTIME_ONE(std::uint32_t, u32, suffix)                \
   HYLE_SCAN_RUNTIME_ONE(std::int32_t, i32, suffix)                 \
   HYLE_SCAN_RUNTIME_ONE(std::uint64_t, u64, suffix)                \
   HYLE_SCAN_RUNTIME_ONE(std::int64_t, i64, suffix)                 \
   HYLE_SCAN_RUNTIME_ONE(float, f32, suffix)                        \
   HYLE_SCAN_RUNTIME_ONE(double, f64, suffix)}

#define HYLE_NARROW_RUNTIME_ONE(tag, suffix)                                    \
  &detail::narrow_count_equal_##tag##_##suffix,                                \
      &detail::narrow_find_equal_##tag##_##suffix,                             \
      &detail::narrow_count_equal_bits_##tag##_##suffix,                       \
      &detail::narrow_find_equal_bits_##tag##_##suffix,                        \
      &detail::narrow_sum_##tag##_##suffix,                                    \
      &detail::narrow_is_sorted_total_asc_##tag##_##suffix,                    \
      &detail::narrow_is_sorted_total_desc_##tag##_##suffix,

#define HYLE_NARROW_RUNTIME(suffix)                                            \
  {HYLE_NARROW_RUNTIME_ONE(f16, suffix) HYLE_NARROW_RUNTIME_ONE(bf16, suffix)}

#define HYLE_OPS_ROW_RUNTIME_ONE(tag, suffix)                                  \
  &detail::sub_##tag##_##suffix, &detail::mul_##tag##_##suffix,                \
      &detail::div_##tag##_##suffix, &detail::scale_##tag##_##suffix,          \
      &detail::add_scalar_##tag##_##suffix, &detail::affine_##tag##_##suffix,  \
      &detail::sum_##tag##_##suffix, &detail::abs_##tag##_##suffix,            \
      &detail::neg_##tag##_##suffix, &detail::square_##tag##_##suffix,         \
      &detail::sqrt_##tag##_##suffix,

#define HYLE_BITPACK_RUNTIME_ONE(tag, suffix)                                  \
  &detail::pack_##tag##_##suffix, &detail::unpack_##tag##_##suffix,

#define HYLE_BITPACK_RUNTIME(suffix)                                           \
  {HYLE_BITPACK_RUNTIME_ONE(u8, suffix) HYLE_BITPACK_RUNTIME_ONE(u16, suffix)  \
       HYLE_BITPACK_RUNTIME_ONE(u32, suffix)}

#define HYLE_SORT_RUNTIME_ONE(T, tag, suffix)                                  \
  &detail::sort_##tag##_##suffix, &detail::psort_##tag##_##suffix,             \
      &detail::select_##tag##_##suffix,

#define HYLE_SORT_RUNTIME(suffix)                                                \
  {HYLE_SORT_RUNTIME_ONE(std::uint16_t, u16, suffix)                            \
   HYLE_SORT_RUNTIME_ONE(std::int16_t, i16, suffix)                             \
   HYLE_SORT_RUNTIME_ONE(std::uint32_t, u32, suffix)                            \
   HYLE_SORT_RUNTIME_ONE(std::int32_t, i32, suffix)                             \
   HYLE_SORT_RUNTIME_ONE(std::uint64_t, u64, suffix)                            \
   HYLE_SORT_RUNTIME_ONE(std::int64_t, i64, suffix)                             \
   HYLE_SORT_RUNTIME_ONE(float, f32, suffix)                                    \
   HYLE_SORT_RUNTIME_ONE(double, f64, suffix)                                   \
   HYLE_SORT_RUNTIME_ONE(UInt128, u128, suffix)                                 \
   HYLE_SORT_RUNTIME_ONE(KeyValue32, kv32, suffix)                              \
   HYLE_SORT_RUNTIME_ONE(KeyValue64, kv64, suffix)}

Runtime::Sort sort_ops_for_product_isa(Isa isa) noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_SCALAR) || defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  (void)isa;
  return HYLE_SORT_RUNTIME(scalar);
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  (void)isa;
  return HYLE_SORT_RUNTIME(avx2);
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  (void)isa;
#if HYLE_COMPILER_MSVC
  return HYLE_SORT_RUNTIME(avx2);
#else
  return HYLE_SORT_RUNTIME(avx512);
#endif
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  (void)isa;
  return HYLE_SORT_RUNTIME(neon);
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
  (void)isa;
  return HYLE_SORT_RUNTIME(sve);
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
  (void)isa;
  return HYLE_SORT_RUNTIME(rvv);
#else
  switch (isa) {
#if HYLE_ARCH_X64
    case Isa::kAvx512:
#if HYLE_COMPILER_MSVC
      return HYLE_SORT_RUNTIME(avx2);
#else
      return HYLE_SORT_RUNTIME(avx512);
#endif
    case Isa::kAvx2: return HYLE_SORT_RUNTIME(avx2);
    // The bounded adaptation has no separate SSE4 body.
    case Isa::kSse42: return HYLE_SORT_RUNTIME(scalar);
#elif HYLE_ARCH_ARM64
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
    case Isa::kSve: return HYLE_SORT_RUNTIME(sve);
#endif
    case Isa::kNeon: return HYLE_SORT_RUNTIME(neon);
#elif HYLE_ARCH_RISCV64 && defined(HYLE_SIMD_HAS_RVV_VARIANT)
    case Isa::kRvv: return HYLE_SORT_RUNTIME(rvv);
#endif
    default: return HYLE_SORT_RUNTIME(scalar);
  }
#endif
}

#if defined(HYLE_SIMD_FAT_AVX512_CAPPED)
#define HYLE_SIMD_CLASSIC_CL_HIGHWAY_FALLBACK(value)                           \
  value.ops_module.isa = Isa::kAvx2;                                          \
  value.ops_module.features |= Runtime::kFeaturePerOpLowerTier;                \
  value.narrow_module.isa = Isa::kAvx2;                                       \
  value.narrow_module.features |= Runtime::kFeaturePerOpLowerTier;             \
  value.search_module.isa = Isa::kAvx2;                                       \
  value.search_module.features |= Runtime::kFeaturePerOpLowerTier;             \
  value.scan.isa = Isa::kAvx2;                                                \
  value.scan.features |= Runtime::kFeaturePerOpLowerTier;                      \
  value.random_module.isa = Isa::kAvx2;                                       \
  value.random_module.features |= Runtime::kFeaturePerOpLowerTier;             \
  value.bitpack.isa = Isa::kAvx2;                                             \
  value.bitpack.features |= Runtime::kFeaturePerOpLowerTier;                   \
  value.dtype.isa = Isa::kAvx2;                                               \
  value.dtype.features |= Runtime::kFeaturePerOpLowerTier;
#else
#define HYLE_SIMD_CLASSIC_CL_HIGHWAY_FALLBACK(value)
#endif

#define HYLE_RUNTIME_TABLE(name, tier, suffix)                                      \
  const Runtime name = [] {                                                         \
    Runtime value = {                                                               \
        Runtime::kTableVersion, sizeof(Runtime), tier,                              \
        {&detail::dot_f32_##suffix, &detail::dot_f64_##suffix,                      \
         &detail::dot_bf16_bf16_##suffix, &detail::dot_f32_bf16_##suffix,           \
         &detail::dot_f16_f16_##suffix, &detail::dot_i16_i16_##suffix,              \
         &detail::dot_i8_i8_##suffix,                                                \
         &detail::saxpy_f32_##suffix, &detail::saxpy_f64_##suffix,                  \
         &detail::add_f32_##suffix, &detail::add_f64_##suffix,                      \
         HYLE_OPS_ROW_RUNTIME_ONE(f32, suffix)                                      \
         HYLE_OPS_ROW_RUNTIME_ONE(f64, suffix)},                                    \
        {&detail::random_fill_u64_##suffix, &detail::random_fill_f32_##suffix,      \
         &detail::random_fill_f64_##suffix, &detail::random_advance_##suffix},      \
        HYLE_SEARCH_RUNTIME(suffix),                                                \
        HYLE_SCAN_RUNTIME(suffix),                                                  \
        HYLE_NARROW_RUNTIME(suffix),                                                \
        {&detail::f16_to_f32_##suffix, &detail::f32_to_f16_##suffix,               \
         &detail::bf16_to_f32_##suffix, &detail::f32_to_bf16_##suffix,             \
         &detail::f16_to_bf16_##suffix, &detail::bf16_to_f16_##suffix},            \
        HYLE_BITPACK_RUNTIME(suffix),                                              \
        sort_ops_for_product_isa(tier),                                            \
        {tier, 0},                                                                  \
        {tier, 0},                                                                  \
        {tier, 0},                                                                  \
        {tier, 0},                                                                  \
        {tier, 0},                                                                  \
        {tier, Runtime::kFeatureDtypeFp8Scalar},                                    \
        {tier, Runtime::kFeatureMathExactLibmScalar},                               \
        {tier, 0},                                                                  \
        {tier, Runtime::kFeatureBitpackLogical128}};                                \
    if constexpr (tier == Isa::kSse42) {                                            \
      value.math.isa = value.sort.isa = Isa::kScalar;                               \
    } else if constexpr (tier == Isa::kAvx2) {                                      \
      value.dtype.features |= Runtime::kFeatureF16c;                                \
    } else if constexpr (tier == Isa::kAvx512) {                                    \
      value.dtype.features |= Runtime::kFeatureF16c;                                \
      value.math.features |= Runtime::kFeaturePerOpLowerTier;                       \
      value.sort.isa = detail::sort_isa_for_product_isa(tier);                      \
      if constexpr (detail::sort_isa_for_product_isa(tier) != tier)                 \
        value.sort.features |= Runtime::kFeaturePerOpLowerTier;                     \
      /* Official Highway has no classic-cl AVX3 target. */                         \
      /* The generated avx512-named ops leg is then an AVX2 body. */                 \
      HYLE_SIMD_CLASSIC_CL_HIGHWAY_FALLBACK(value)                                  \
    } else if constexpr (tier == Isa::kSve) {                                       \
      value.sort.features |= Runtime::kFeatureSortCappedNetwork;                    \
    }                                                                               \
    return value;                                                                   \
  }()

#define HYLE_DEFINE_RUNTIME(pretty, suffix, isa) \
  HYLE_RUNTIME_TABLE(k##pretty##Runtime, Isa::k##isa, suffix);
HYLE_SIMD_FOR_EACH_COMPILED_TARGET(HYLE_DEFINE_RUNTIME)
#undef HYLE_DEFINE_RUNTIME

#undef HYLE_RUNTIME_TABLE
#undef HYLE_SIMD_CLASSIC_CL_HIGHWAY_FALLBACK
#undef HYLE_SORT_RUNTIME
#undef HYLE_SORT_RUNTIME_ONE
#undef HYLE_BITPACK_RUNTIME
#undef HYLE_BITPACK_RUNTIME_ONE
#undef HYLE_OPS_ROW_RUNTIME_ONE
#undef HYLE_SCAN_RUNTIME
#undef HYLE_SCAN_RUNTIME_ONE
#undef HYLE_NARROW_RUNTIME
#undef HYLE_NARROW_RUNTIME_ONE
#undef HYLE_SEARCH_RUNTIME
#undef HYLE_SEARCH_RUNTIME_ONE

const Runtime* runtime_for_supported_isa(Isa isa) noexcept {
  switch (isa) {
#define HYLE_RUNTIME_CASE(pretty, suffix, isa_token) \
    case Isa::k##isa_token: return &k##pretty##Runtime;
    HYLE_SIMD_FOR_EACH_COMPILED_TARGET(HYLE_RUNTIME_CASE)
#undef HYLE_RUNTIME_CASE
    default: return nullptr;
  }
}

Runtime selected_runtime() noexcept {
  const Isa tier = detail::default_dispatched_isa();
  const Runtime* base = runtime_for_supported_isa(tier);
#if defined(HYLE_SIMD_EXACT)
  Runtime result = base != nullptr ? *base : HYLE_SIMD_EXACT_RUNTIME;
#else
  Runtime result = base != nullptr ? *base : kScalarRuntime;
#endif

#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  result.math.features |= Runtime::kFeaturePerOpLowerTier;
  result.sort.isa = detail::sort_isa_for_product_isa(Isa::kAvx512);
  if (result.sort.isa != result.isa)
    result.sort.features |= Runtime::kFeaturePerOpLowerTier;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4) || defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  result.math.isa = Isa::kScalar;
  result.sort.isa = Isa::kScalar;
#elif defined(HYLE_SIMD_EXACT)
  // The generated table already describes the sole compiled Highway target.
#elif HYLE_ARCH_X64
  const auto& f = platform::cpu_features();
  const auto specialty = detail::x86_specialty_support(f);
  if (specialty.f16c)
    result.ops.dot_f16_f16 = &detail::dot_f16_f16_f16c;
  if (specialty.avx512_bf16)
    result.ops.dot_bf16_bf16 = &detail::dot_bf16_bf16_avx512bf16;
  if (specialty.avx512_vnni) {
    result.ops.dot_i8_i8 = &detail::dot_i8_i8_avx512_vnni;
    result.ops_module.features |= Runtime::kFeatureAvx512Vnni;
  } else if (specialty.avx_vnni) {
    result.ops.dot_i8_i8 = &detail::dot_i8_i8_avx_vnni;
    result.ops_module.features |= Runtime::kFeatureAvxVnni;
  }
  // Generated Highway owns every common conversion at the selected product
  // tier. F16C is retained only for AVX+F16C CPUs below AVX2, where it is a
  // genuine feature capability not representable by the canonical ladder.
  if (specialty.f16c) {
    result.dtype.features |= Runtime::kFeatureF16c;
    if (result.isa != Isa::kAvx2 && result.isa != Isa::kAvx512) {
      result.dtype_convert.f16_to_f32 = &detail::f16_to_f32_f16c;
      result.dtype_convert.f32_to_f16 = &detail::f32_to_f16_f16c;
    }
  }
  if (specialty.avx512_bf16) {
    result.dtype_convert.f32_to_bf16 = &detail::f32_to_bf16_avx512bf16;
    if (result.dtype.isa != Isa::kAvx512) {
      result.dtype.isa = Isa::kAvx512;
      result.dtype.features |= Runtime::kFeaturePerOpLowerTier;
    }
    result.dtype.features |= Runtime::kFeatureAvx512Bf16;
  }
  if (specialty.avx10_2)
    result.dtype.features |= Runtime::kFeatureAvx10_2;
  const Isa product_tier =
      result.isa == Isa::kAvx512 ? Isa::kAvx512
      : result.isa == Isa::kAvx2 ? Isa::kAvx2
                                 : Isa::kScalar;
  result.math.isa = product_tier;
  result.sort.isa = detail::sort_isa_for_product_isa(product_tier);
  if (product_tier == Isa::kAvx512)
    result.math.features |= Runtime::kFeaturePerOpLowerTier;
  if (result.sort.isa != product_tier)
    result.sort.features |= Runtime::kFeaturePerOpLowerTier;
#elif HYLE_ARCH_ARM64
  result.dtype.isa = Isa::kNeon;
#endif

#if HYLE_ARCH_ARM64
  // Math, common dtype conversions, canonical bitpack and sort have SVE product TUs.
  if (result.isa == Isa::kSve) {
    result.dtype.isa = Isa::kSve;
    result.math.isa = Isa::kSve;
    result.sort.isa = Isa::kSve;
    result.sort.features |= Runtime::kFeatureSortCappedNetwork;
    result.bitpack.isa = Isa::kSve;
  }
#elif HYLE_ARCH_RISCV64
  // RVV math, common dtype conversions and canonical bitpack have product TUs.
  result.dtype.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.math.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.sort.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.bitpack.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
#endif
  return result;
}

}  // namespace

const Runtime& runtime() noexcept {
  static const Runtime selected = selected_runtime();
  return selected;
}

const Runtime* runtime_for(Isa isa) noexcept {
  const Runtime* result = runtime_for_supported_isa(isa);
#if defined(HYLE_SIMD_EXACT)
  // Exact deployment is an assertion by the builder. Do not reintroduce CPU probing through the
  // diagnostic table; only the one compiled canonical tier exists.
  return result;
#else
  if (result == nullptr || isa == Isa::kScalar) return result;
#if HYLE_ARCH_X64
  if (isa != Isa::kSse42 && isa != Isa::kAvx2 && isa != Isa::kAvx512) return nullptr;
#elif HYLE_ARCH_ARM64
  if (isa != Isa::kNeon && isa != Isa::kSve) return nullptr;
#elif HYLE_ARCH_RISCV64
  if (isa != Isa::kRvv) return nullptr;
#else
  return nullptr;
#endif
  return platform::supports(isa) ? result : nullptr;
#endif
}

SelectedTarget selected_target(KernelFamily family) noexcept {
  const Runtime& selected = runtime();
  switch (family) {
    case KernelFamily::kOverall: return {selected.isa, 0};
    case KernelFamily::kOps:
      return {selected.ops_module.isa, selected.ops_module.features};
    case KernelFamily::kRandom:
      return {selected.random_module.isa, selected.random_module.features};
    case KernelFamily::kSearch:
      return {selected.search_module.isa, selected.search_module.features};
    case KernelFamily::kNarrow:
      return {selected.narrow_module.isa, selected.narrow_module.features};
    case KernelFamily::kScan:
      return {selected.scan.isa, selected.scan.features};
    case KernelFamily::kDtype:
      return {selected.dtype.isa, selected.dtype.features};
    case KernelFamily::kMath:
      return {selected.math.isa, selected.math.features};
    case KernelFamily::kSort:
      return {selected.sort.isa, selected.sort.features};
    case KernelFamily::kBitpack:
      return {selected.bitpack.isa, selected.bitpack.features};
  }
  return {selected.isa, 0};
}

}  // namespace hyle::simd
