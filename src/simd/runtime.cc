#include "hyle/simd/runtime.h"

#include "hyle/platform.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/ops/kernels.h"
#include "src/simd/random/kernels.h"
#include "src/simd/search/kernels.h"

namespace hyle::simd {
namespace {

#define HYLE_SEARCH_RUNTIME_ONE(T, tag, suffix)                                  \
  &detail::count_equal_##tag##_##suffix, &detail::find_equal_##tag##_##suffix,   \
      &detail::min_value_##tag##_##suffix, &detail::max_value_##tag##_##suffix,  \
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

#define HYLE_RUNTIME_TABLE(name, tier, suffix)                                      \
  const Runtime name = [] {                                                         \
    Runtime value = {                                                               \
        Runtime::kTableVersion, sizeof(Runtime), tier,                              \
        {&detail::dot_f32_##suffix, &detail::dot_f64_##suffix,                      \
         &detail::dot_bf16_bf16_##suffix, &detail::dot_f32_bf16_##suffix,           \
         &detail::dot_f16_f16_##suffix, &detail::dot_i16_i16_##suffix,              \
         &detail::dot_i8_i8_##suffix,                                                \
         &detail::saxpy_f32_##suffix, &detail::saxpy_f64_##suffix,                  \
         &detail::add_f32_##suffix, &detail::add_f64_##suffix},                     \
        {&detail::random_fill_u64_##suffix, &detail::random_fill_f32_##suffix,      \
         &detail::random_fill_f64_##suffix},                                        \
        HYLE_SEARCH_RUNTIME(suffix),                                                \
        {tier, 0},                                                                  \
        {tier, 0},                                                                  \
        {tier, Runtime::kFeatureDtypeFp8Scalar},                                    \
        {tier, Runtime::kFeatureMathExactLibmScalar},                               \
        {tier, 0}, {tier, 0}};                                                      \
    if constexpr (tier == Isa::kSse42) {                                            \
      value.dtype.isa = value.math.isa = value.sort.isa = value.bitpack.isa =       \
          Isa::kScalar;                                                             \
    } else if constexpr (tier == Isa::kAvx2) {                                      \
      value.dtype.features |= Runtime::kFeatureF16c;                                \
    } else if constexpr (tier == Isa::kAvx512) {                                    \
      value.dtype.isa = Isa::kAvx2;                                                 \
      value.dtype.features |= Runtime::kFeatureF16c;                                \
      value.math.features |= Runtime::kFeaturePerOpLowerTier;                       \
    } else if constexpr (tier == Isa::kSve) {                                       \
      value.sort.features |= Runtime::kFeatureSortCappedNetwork;                    \
      value.bitpack.features |= Runtime::kFeatureBitpackLogical128;                 \
    } else if constexpr (tier == Isa::kRvv) {                                       \
      value.bitpack.features |= Runtime::kFeatureBitpackLogical128;                 \
    }                                                                               \
    return value;                                                                   \
  }()

#define HYLE_DEFINE_RUNTIME(pretty, suffix, isa) \
  HYLE_RUNTIME_TABLE(k##pretty##Runtime, Isa::k##isa, suffix);
HYLE_SIMD_FOR_EACH_COMPILED_TARGET(HYLE_DEFINE_RUNTIME)
#undef HYLE_DEFINE_RUNTIME

#undef HYLE_RUNTIME_TABLE
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

#if !defined(HYLE_SIMD_EXACT) && defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
  // GCC's scalar dependency chain beats the single-stream Kogge-Stone kernel on this platform.
  result.scan.isa = Isa::kScalar;
#endif

#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  result.dtype.isa = Isa::kAvx2;
  result.dtype.features |= Runtime::kFeatureF16c;
  result.math.features |= Runtime::kFeaturePerOpLowerTier;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  result.dtype.isa = Isa::kAvx2;
  result.dtype.features |= Runtime::kFeatureF16c;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4) || defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  result.dtype.isa = Isa::kScalar;
  result.math.isa = Isa::kScalar;
  result.sort.isa = Isa::kScalar;
  result.bitpack.isa = Isa::kScalar;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON) || defined(HYLE_SIMD_EXACT_TARGET_SVE)
  result.dtype.isa = Isa::kNeon;
#elif defined(HYLE_ARCH_X64)
  const auto& f = cpu_features();
  if (f.f16c && f.avx) result.ops.dot_f16_f16 = &detail::dot_f16_f16_f16c;
  if (f.avx512bf16)
    result.ops.dot_bf16_bf16 = &detail::dot_bf16_bf16_avx512bf16;
  if (f.avx512vnni) {
    result.ops.dot_i8_i8 = &detail::dot_i8_i8_avx512_vnni;
    result.ops_module.features |= Runtime::kFeatureAvx512Vnni;
  } else if (f.avx_vnni) {
    result.ops.dot_i8_i8 = &detail::dot_i8_i8_avx_vnni;
    result.ops_module.features |= Runtime::kFeatureAvxVnni;
  }
  result.dtype.isa = f.avx2 ? Isa::kAvx2 : Isa::kScalar;
  if (f.f16c && f.avx) result.dtype.features |= Runtime::kFeatureF16c;
  if (f.avx512bf16) {
    result.dtype.isa = Isa::kAvx512;
    result.dtype.features |= Runtime::kFeatureAvx512Bf16;
  }
  if (f.avx10_2) result.dtype.features |= Runtime::kFeatureAvx10_2;
  const Isa product_tier =
      result.isa == Isa::kAvx512 ? Isa::kAvx512
      : result.isa == Isa::kAvx2 ? Isa::kAvx2
                                 : Isa::kScalar;
  result.math.isa = product_tier;
  result.sort.isa = product_tier;
  result.bitpack.isa = product_tier;
  if (product_tier == Isa::kAvx512)
    result.math.features |= Runtime::kFeaturePerOpLowerTier;
#elif defined(HYLE_ARCH_ARM64)
  result.dtype.isa = Isa::kNeon;
#endif

#if defined(HYLE_ARCH_ARM64)
  // Math, common dtype conversions, bitpack and sort have SVE product TUs. Bitpack freezes a
  // 128-bit logical block so its bytes do not depend on the thread's VL.
  if (result.isa == Isa::kSve) {
    result.dtype.isa = Isa::kSve;
    result.math.isa = Isa::kSve;
    result.sort.isa = Isa::kSve;
    result.sort.features |= Runtime::kFeatureSortCappedNetwork;
    result.bitpack.isa = Isa::kSve;
    result.bitpack.features |= Runtime::kFeatureBitpackLogical128;
  }
#elif defined(HYLE_ARCH_RISCV64)
  // RVV math, common dtype conversions and fixed-logical-block bitpack have product TUs.
  result.dtype.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.math.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.sort.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.bitpack.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  if (result.isa == Isa::kRvv)
    result.bitpack.features |= Runtime::kFeatureBitpackLogical128;
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
#if defined(HYLE_ARCH_X64)
  if (isa != Isa::kSse42 && isa != Isa::kAvx2 && isa != Isa::kAvx512) return nullptr;
#elif defined(HYLE_ARCH_ARM64)
  if (isa != Isa::kNeon && isa != Isa::kSve) return nullptr;
#elif defined(HYLE_ARCH_RISCV64)
  if (isa != Isa::kRvv) return nullptr;
#else
  return nullptr;
#endif
  return supports(isa) ? result : nullptr;
#endif
}

}  // namespace hyle::simd
