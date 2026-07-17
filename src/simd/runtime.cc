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
  const Runtime name = {                                                           \
      Runtime::kTableVersion, sizeof(Runtime), tier,                              \
      {&detail::dot_f32_##suffix, &detail::dot_f64_##suffix,                       \
       &detail::dot_bf16_bf16_##suffix, &detail::dot_f32_bf16_##suffix,            \
       &detail::dot_f16_f16_##suffix, &detail::dot_i16_i16_##suffix,               \
       &detail::saxpy_f32_##suffix, &detail::saxpy_f64_##suffix,                   \
       &detail::add_f32_##suffix, &detail::add_f64_##suffix},                      \
      {&detail::random_fill_u64_##suffix, &detail::random_fill_f32_##suffix,       \
       &detail::random_fill_f64_##suffix},                                         \
      HYLE_SEARCH_RUNTIME(suffix),                                                 \
      {tier, 0}, {tier, 0}, {tier, 0}, {tier, 0}, {tier, 0}}

HYLE_RUNTIME_TABLE(kScalarRuntime, Isa::kScalar, scalar);
#if defined(HYLE_ARCH_X64)
HYLE_RUNTIME_TABLE(kSse4Runtime, Isa::kSse42, sse4);
HYLE_RUNTIME_TABLE(kAvx2Runtime, Isa::kAvx2, avx2);
HYLE_RUNTIME_TABLE(kAvx512Runtime, Isa::kAvx512, avx512);
#elif defined(HYLE_ARCH_ARM64)
HYLE_RUNTIME_TABLE(kNeonRuntime, Isa::kNeon, neon);
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
HYLE_RUNTIME_TABLE(kSveRuntime, Isa::kSve, sve);
#endif
#elif defined(HYLE_ARCH_RISCV64)
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
HYLE_RUNTIME_TABLE(kRvvRuntime, Isa::kRvv, rvv);
#endif
#endif

#undef HYLE_RUNTIME_TABLE
#undef HYLE_SEARCH_RUNTIME
#undef HYLE_SEARCH_RUNTIME_ONE

const Runtime* runtime_for_supported_isa(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar: return &kScalarRuntime;
#if defined(HYLE_ARCH_X64)
    case Isa::kSse42: return &kSse4Runtime;
    case Isa::kAvx2: return &kAvx2Runtime;
    case Isa::kAvx512: return &kAvx512Runtime;
#elif defined(HYLE_ARCH_ARM64)
    case Isa::kNeon: return &kNeonRuntime;
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
    case Isa::kSve: {
      static const Runtime table = [] {
        Runtime r = kSveRuntime;
        r.sort.isa = r.bitpack.isa = Isa::kNeon;
        return r;
      }();
      return &table;
    }
#endif
#elif defined(HYLE_ARCH_RISCV64)
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
    case Isa::kRvv: {
      static const Runtime table = [] {
        Runtime r = kRvvRuntime;
        r.sort.isa = r.bitpack.isa = Isa::kScalar;
        return r;
      }();
      return &table;
    }
#endif
#endif
    default: return nullptr;
  }
}

Runtime selected_runtime() noexcept {
  const Isa tier = detail::default_dispatched_isa();
  const Runtime* base = runtime_for_supported_isa(tier);
  Runtime result = base != nullptr ? *base : kScalarRuntime;

#if !defined(HYLE_SIMD_EXACT) && defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
  // GCC's scalar dependency chain beats the single-stream Kogge-Stone kernel on this platform.
  result.scan.isa = Isa::kScalar;
#endif

#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  result.dtype.isa = Isa::kAvx2;
  result.dtype.features = Runtime::kFeatureF16c;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  result.dtype.isa = Isa::kAvx2;
  result.dtype.features = Runtime::kFeatureF16c;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4) || defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  result.dtype.isa = Isa::kScalar;
  result.dtype.features = 0;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON) || defined(HYLE_SIMD_EXACT_TARGET_SVE)
  result.dtype.isa = Isa::kNeon;
  result.dtype.features = 0;
#elif defined(HYLE_ARCH_X64)
  const auto& f = cpu_features();
  if (f.f16c && f.avx) result.ops.dot_f16_f16 = &detail::dot_f16_f16_f16c;
  if (f.avx512bf16)
    result.ops.dot_bf16_bf16 = &detail::dot_bf16_bf16_avx512bf16;
  result.dtype.isa = f.avx2 ? Isa::kAvx2 : Isa::kScalar;
  if (f.f16c && f.avx) result.dtype.features |= Runtime::kFeatureF16c;
  if (f.avx512bf16) {
    result.dtype.isa = Isa::kAvx512;
    result.dtype.features |= Runtime::kFeatureAvx512Bf16;
  }
  if (f.avx10_2) result.dtype.features |= Runtime::kFeatureAvx10_2;
#elif defined(HYLE_ARCH_ARM64)
  result.dtype.isa = Isa::kNeon;
#endif

#if defined(HYLE_ARCH_ARM64)
  // Math and common dtype conversions have length-agnostic SVE product TUs. Sort remains NEON;
  // bitpack stays NEON until its format is independent of the calling thread's SVE VL.
  if (result.isa == Isa::kSve) {
    result.dtype.isa = Isa::kSve;
    result.math.isa = Isa::kSve;
    result.sort.isa = Isa::kNeon;
    result.bitpack.isa = Isa::kNeon;
  }
#elif defined(HYLE_ARCH_RISCV64)
  // RVV math and common dtype conversions now have length-agnostic product TUs.
  result.dtype.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.math.isa = result.isa == Isa::kRvv ? Isa::kRvv : Isa::kScalar;
  result.sort.isa = Isa::kScalar;
  result.bitpack.isa = Isa::kScalar;
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
}

}  // namespace hyle::simd
