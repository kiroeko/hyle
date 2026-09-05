/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/simd/provider/operation_manifest.h"
#include "hyle/simd/target.h"

// Source-only compile-time provider selection. This header owns no register
// facade: each Candidate specialization keeps Highway, xsimd, or HYLE native
// types inside one provider-coherent body. Installed static/shared SDK users
// continue to see pointer/count HYLE APIs only.

namespace hyle::simd::provider {

enum class Provider : std::uint8_t {
  kPending,
  kHyle,
  kHighway,
  kXsimd,
  kNone,
};

enum class Decision : std::uint8_t {
  kPending,
  kKeepHyle,
  kRouteHighway,
  kRouteXsimd,
  kRemoveRedundant,
};

enum class Compiler : std::uint8_t {
  kMsvc,
  kClangCl,
  kClang,
  kAppleClang,
  kGcc,
  kOther,
};

enum class Operation : std::uint16_t {
#define HYLE_SIMD_DECLARE_OPERATION(name) k##name,
  HYLE_SIMD_OPERATION_LIST(HYLE_SIMD_DECLARE_OPERATION)
#undef HYLE_SIMD_DECLARE_OPERATION
  kCount,
};

inline constexpr std::size_t kOperationCount = HYLE_SIMD_OPERATION_COUNT;
inline constexpr char kOperationInventorySha256[] =
    HYLE_SIMD_OPERATION_INVENTORY_SHA256;
static_assert(static_cast<std::size_t>(Operation::kCount) == kOperationCount);

// A per-TU target describes the implementation body actually being compiled
// and therefore takes precedence over an artifact-wide exact target. This is
// observable for classic cl.exe exact-AVX512 artifacts whose Highway math leg
// is intentionally an AVX2 body.
#if defined(HYLE_SIMD_TARGET_AVX512)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kAvx512;
#elif defined(HYLE_SIMD_TARGET_AVX2)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kAvx2;
#elif defined(HYLE_SIMD_TARGET_SSE4)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kSse4;
#elif defined(HYLE_SIMD_TARGET_SVE)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kSve;
#elif defined(HYLE_SIMD_TARGET_NEON)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kNeon;
#elif defined(HYLE_SIMD_TARGET_RVV)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kRvv;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kAvx512;
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kAvx2;
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kSse4;
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kSve;
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kNeon;
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kRvv;
#else
inline constexpr KernelTarget kCurrentTarget = KernelTarget::kScalar;
#endif

#if defined(__clang__) && defined(_MSC_VER)
inline constexpr Compiler kCurrentCompiler = Compiler::kClangCl;
inline constexpr std::uint32_t kCurrentCompilerVersion =
    __clang_major__ * 10000u + __clang_minor__ * 100u + __clang_patchlevel__;
#elif defined(__apple_build_version__)
inline constexpr Compiler kCurrentCompiler = Compiler::kAppleClang;
inline constexpr std::uint32_t kCurrentCompilerVersion =
    __clang_major__ * 10000u + __clang_minor__ * 100u + __clang_patchlevel__;
#elif defined(__clang__)
inline constexpr Compiler kCurrentCompiler = Compiler::kClang;
inline constexpr std::uint32_t kCurrentCompilerVersion =
    __clang_major__ * 10000u + __clang_minor__ * 100u + __clang_patchlevel__;
#elif defined(_MSC_VER)
inline constexpr Compiler kCurrentCompiler = Compiler::kMsvc;
inline constexpr std::uint32_t kCurrentCompilerVersion = _MSC_FULL_VER;
#elif defined(__GNUC__)
inline constexpr Compiler kCurrentCompiler = Compiler::kGcc;
inline constexpr std::uint32_t kCurrentCompilerVersion =
    __GNUC__ * 10000u + __GNUC_MINOR__ * 100u + __GNUC_PATCHLEVEL__;
#else
inline constexpr Compiler kCurrentCompiler = Compiler::kOther;
inline constexpr std::uint32_t kCurrentCompilerVersion = 0;
#endif

struct Route final {
  Provider provider;
  Decision decision;

  [[nodiscard]] constexpr bool qualified() const noexcept {
    return decision != Decision::kPending;
  }
};

[[nodiscard]] constexpr bool consistent(Route route) noexcept {
  switch (route.decision) {
    case Decision::kPending:
      return route.provider == Provider::kPending;
    case Decision::kKeepHyle:
      return route.provider == Provider::kHyle;
    case Decision::kRouteHighway:
      return route.provider == Provider::kHighway;
    case Decision::kRouteXsimd:
      return route.provider == Provider::kXsimd;
    case Decision::kRemoveRedundant:
      return route.provider == Provider::kNone;
  }
  return false;
}

// Machine-readable semantic keys. A single lane type is sufficient for most
// primitives; conversions and reusable invariant-divisor operations carry the
// extra contract dimensions that materially affect implementation choice.
// Buffer length, mask density and benchmark title are deliberately not route
// dimensions.
struct AnyVariant final {};

template <typename Lane>
struct LaneVariant final {
  using lane_type = Lane;
};

enum class ConversionContract : std::uint8_t {
  kDefault,
  kTwoInputSaturating,
};

template <typename From, typename To,
          ConversionContract kContract = ConversionContract::kDefault>
struct ConversionVariant final {
  using from_type = From;
  using to_type = To;
  inline static constexpr ConversionContract contract = kContract;
};

template <typename Lane>
struct InvariantDivisorVariant final {
  using lane_type = Lane;
};

// Preserve the enum type as part of the route identity. Some compilers compare
// overlapping enum values while ordering `auto` non-type partial
// specializations; wrapping the value in a type makes Operation::kX and
// Family::kX unambiguously different without adding any runtime state.
template <auto kValue>
struct RouteKey final {};

// The primary is fail-closed. SIMD-15 generates only qualified
// specializations and may split one operation by lane type, ISA, compiler kind
// and compiler version.
template <typename Key, typename Variant = AnyVariant,
          KernelTarget kTarget = kCurrentTarget,
          Compiler kCompiler = kCurrentCompiler,
          std::uint32_t kCompilerVersion = kCurrentCompilerVersion>
struct RouteFor final {
  inline static constexpr Route value{Provider::kPending,
                                      Decision::kPending};
};

// Generated terminal specializations. The operation inventory remains an
// internal migration ledger; this does not recreate a public register API.
#include "hyle/simd/provider/routes_generated.h"

template <auto kOperation, typename Variant = AnyVariant,
          KernelTarget kTarget = kCurrentTarget,
          Compiler kCompiler = kCurrentCompiler,
          std::uint32_t kCompilerVersion = kCurrentCompilerVersion>
inline constexpr Route kRouteFor =
    RouteFor<RouteKey<kOperation>, Variant, kTarget, kCompiler,
             kCompilerVersion>::value;

// Resolve a route entirely in the type system. Callers instantiate their
// provider-specialized Candidate directly with kQualifiedProviderFor; there is
// no wrapper function for a compiler to outline and no register value crosses
// the selection boundary.
template <auto kOperation, typename Variant = AnyVariant,
          KernelTarget kTarget = kCurrentTarget,
          Compiler kCompiler = kCurrentCompiler,
          std::uint32_t kCompilerVersion = kCurrentCompilerVersion>
struct QualifiedProviderFor final {
  inline static constexpr Route route =
      kRouteFor<kOperation, Variant, kTarget, kCompiler, kCompilerVersion>;
  static_assert(consistent(route),
                "SIMD provider route is internally inconsistent");
  static_assert(route.qualified(),
                "SIMD provider route has not been qualified for this operation/type/target/compiler");
  static_assert(route.provider != Provider::kNone,
                "a redundant alias/composition has no directly invocable provider body");
  inline static constexpr Provider value = route.provider;
};

template <auto kOperation, typename Variant = AnyVariant,
          KernelTarget kTarget = kCurrentTarget,
          Compiler kCompiler = kCurrentCompiler,
          std::uint32_t kCompilerVersion = kCurrentCompilerVersion>
inline constexpr Provider kQualifiedProviderFor =
    QualifiedProviderFor<kOperation, Variant, kTarget, kCompiler,
                         kCompilerVersion>::value;

}  // namespace hyle::simd::provider

// Optional convenience spellings. They expand directly to Candidate<P>::run;
// the template form above remains available when a macro is undesirable.
#define HYLE_SIMD_INVOKE_CANDIDATE(provider_value, CandidateTemplate, ...) \
  CandidateTemplate<provider_value>::run(__VA_ARGS__)

#define HYLE_SIMD_INVOKE_ROUTED(operation_value, CandidateTemplate, VariantType, ...) \
  CandidateTemplate<::hyle::simd::provider::kQualifiedProviderFor<                    \
      operation_value, VariantType>>::run(                                             \
      __VA_ARGS__)
