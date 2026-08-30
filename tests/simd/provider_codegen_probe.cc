/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "hyle/simd/provider/hyle_winners-inl.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hwy/cache_control.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"
#include "xsimd/xsimd.hpp"

namespace hyle::simd::provider::codegen_probe {

namespace hn = hwy::HWY_NAMESPACE;

enum class ProbeOperation : unsigned char {
  kAdd,
  kCopy,
  kRelu,
  kTanh,
  kFusedAffineRelu,
};

struct HyleLane final {};
struct HighwayLane final {};
struct XsimdLane final {};

template <Provider>
struct AddCandidate;
template <Provider>
struct CopyCandidate;
template <Provider>
struct ReluCandidate;
template <Provider>
struct TanhCandidate;
template <Provider>
struct FusedCandidate;

template <Provider kProvider>
struct HighwayShapeCandidates {
  static HYLE_FORCE_INLINE void add(const float* a, const float* b, float* out,
                                    std::size_t n) noexcept {
    const hn::ScalableTag<float> d;
    const std::size_t width = hn::Lanes(d);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d,
                 out + i);
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
  }

  static HYLE_FORCE_INLINE void copy(const float* src, float* out,
                                     std::size_t n) noexcept {
    const hn::ScalableTag<float> d;
    const std::size_t width = hn::Lanes(d);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      hn::StoreU(hn::LoadU(d, src + i), d, out + i);
    }
    for (; i < n; ++i) out[i] = src[i];
  }

  static HYLE_FORCE_INLINE void relu(const float* src, float* out,
                                     std::size_t n) noexcept {
    const hn::ScalableTag<float> d;
    const auto zero = hn::Zero(d);
    const std::size_t width = hn::Lanes(d);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      const auto value = hn::LoadU(d, src + i);
      hn::StoreU(hn::IfThenElse(hn::Gt(value, zero), value, zero), d,
                 out + i);
    }
    for (; i < n; ++i) out[i] = src[i] > 0.0f ? src[i] : 0.0f;
  }

  static HYLE_FORCE_INLINE void tanh(const float* src, float* out,
                                     std::size_t n) noexcept {
    const hn::ScalableTag<float> d;
    const std::size_t width = hn::Lanes(d);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::tanh(src[i]);
  }

  static HYLE_FORCE_INLINE void fused(const float* src, float factor,
                                      float bias, float* out,
                                      std::size_t n) noexcept {
    const hn::ScalableTag<float> d;
    const auto vf = hn::Set(d, factor);
    const auto vb = hn::Set(d, bias);
    const auto zero = hn::Zero(d);
    const std::size_t width = hn::Lanes(d);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      const auto value = hn::MulAdd(hn::LoadU(d, src + i), vf, vb);
      hn::StoreU(hn::IfThenElse(hn::Gt(value, zero), value, zero), d,
                 out + i);
    }
    for (; i < n; ++i) {
      const float value = src[i] * factor + bias;
      out[i] = value > 0.0f ? value : 0.0f;
    }
  }
};

template <>
struct AddCandidate<Provider::kHyle> final {
  static HYLE_FORCE_INLINE void run(const float* a, const float* b, float* out,
                                    std::size_t n) noexcept {
    HighwayShapeCandidates<Provider::kHyle>::add(a, b, out, n);
  }
};
template <>
struct CopyCandidate<Provider::kHyle> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    HighwayShapeCandidates<Provider::kHyle>::copy(src, out, n);
  }
};
template <>
struct ReluCandidate<Provider::kHyle> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    HighwayShapeCandidates<Provider::kHyle>::relu(src, out, n);
  }
};
template <>
struct TanhCandidate<Provider::kHyle> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    HighwayShapeCandidates<Provider::kHyle>::tanh(src, out, n);
  }
};
template <>
struct FusedCandidate<Provider::kHyle> final {
  static HYLE_FORCE_INLINE void run(const float* src, float factor, float bias,
                                    float* out, std::size_t n) noexcept {
    HighwayShapeCandidates<Provider::kHyle>::fused(src, factor, bias, out, n);
  }
};

#define HYLE_DEFINE_HIGHWAY_CANDIDATE(candidate, method)                       \
  template <>                                                                  \
  struct candidate<Provider::kHighway> final {                                 \
    template <class... Args>                                                   \
    static HYLE_FORCE_INLINE void run(Args... args) noexcept {                 \
      HighwayShapeCandidates<Provider::kHighway>::method(args...);             \
    }                                                                          \
  }

HYLE_DEFINE_HIGHWAY_CANDIDATE(AddCandidate, add);
HYLE_DEFINE_HIGHWAY_CANDIDATE(CopyCandidate, copy);
HYLE_DEFINE_HIGHWAY_CANDIDATE(ReluCandidate, relu);
HYLE_DEFINE_HIGHWAY_CANDIDATE(TanhCandidate, tanh);
HYLE_DEFINE_HIGHWAY_CANDIDATE(FusedCandidate, fused);
#undef HYLE_DEFINE_HIGHWAY_CANDIDATE

template <>
struct AddCandidate<Provider::kXsimd> final {
  static HYLE_FORCE_INLINE void run(const float* a, const float* b, float* out,
                                    std::size_t n) noexcept {
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      (Batch::load_unaligned(a + i) + Batch::load_unaligned(b + i))
          .store_unaligned(out + i);
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
  }
};
template <>
struct CopyCandidate<Provider::kXsimd> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      Batch::load_unaligned(src + i).store_unaligned(out + i);
    }
    for (; i < n; ++i) out[i] = src[i];
  }
};
template <>
struct ReluCandidate<Provider::kXsimd> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    const Batch zero(0.0f);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      const Batch value = Batch::load_unaligned(src + i);
      xsimd::select(value > zero, value, zero).store_unaligned(out + i);
    }
    for (; i < n; ++i) out[i] = src[i] > 0.0f ? src[i] : 0.0f;
  }
};
template <>
struct TanhCandidate<Provider::kXsimd> final {
  static HYLE_FORCE_INLINE void run(const float* src, float* out,
                                    std::size_t n) noexcept {
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      xsimd::tanh(Batch::load_unaligned(src + i)).store_unaligned(out + i);
    }
    for (; i < n; ++i) out[i] = std::tanh(src[i]);
  }
};
template <>
struct FusedCandidate<Provider::kXsimd> final {
  static HYLE_FORCE_INLINE void run(const float* src, float factor, float bias,
                                    float* out, std::size_t n) noexcept {
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    const Batch vf(factor);
    const Batch vb(bias);
    const Batch zero(0.0f);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      const Batch value = xsimd::fma(Batch::load_unaligned(src + i), vf, vb);
      xsimd::select(value > zero, value, zero).store_unaligned(out + i);
    }
    for (; i < n; ++i) {
      const float value = src[i] * factor + bias;
      out[i] = value > 0.0f ? value : 0.0f;
    }
  }
};

}  // namespace hyle::simd::provider::codegen_probe

namespace hyle::simd::provider {

#define HYLE_SIMD_PROBE_ROUTE(operation, lane, selected_provider, decision)    \
  template <>                                                                  \
  struct RouteFor<RouteKey<codegen_probe::ProbeOperation::operation>,          \
                  codegen_probe::lane> final {                                 \
    inline static constexpr Route value{Provider::selected_provider,           \
                                        Decision::decision};                   \
  }

#define HYLE_SIMD_PROBE_PROVIDER_ROUTES(lane, selected_provider, decision)     \
  HYLE_SIMD_PROBE_ROUTE(kAdd, lane, selected_provider, decision);              \
  HYLE_SIMD_PROBE_ROUTE(kCopy, lane, selected_provider, decision);             \
  HYLE_SIMD_PROBE_ROUTE(kRelu, lane, selected_provider, decision);             \
  HYLE_SIMD_PROBE_ROUTE(kTanh, lane, selected_provider, decision);             \
  HYLE_SIMD_PROBE_ROUTE(kFusedAffineRelu, lane, selected_provider, decision)

HYLE_SIMD_PROBE_PROVIDER_ROUTES(HyleLane, kHyle, kKeepHyle);
HYLE_SIMD_PROBE_PROVIDER_ROUTES(HighwayLane, kHighway, kRouteHighway);
HYLE_SIMD_PROBE_PROVIDER_ROUTES(XsimdLane, kXsimd, kRouteXsimd);

#undef HYLE_SIMD_PROBE_PROVIDER_ROUTES
#undef HYLE_SIMD_PROBE_ROUTE

}  // namespace hyle::simd::provider

namespace hyle::simd::provider::codegen_probe {

#define HYLE_SIMD_DEFINE_BINARY_PAIR(token, provider_name, lane, operation,    \
                                     candidate)                                \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_direct(           \
      const float* a, const float* b, float* out, std::size_t n) noexcept {    \
    candidate<Provider::provider_name>::run(a, b, out, n);                     \
  }                                                                            \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_routed(           \
      const float* a, const float* b, float* out, std::size_t n) noexcept {    \
    HYLE_SIMD_INVOKE_ROUTED(ProbeOperation::operation, candidate, lane,        \
                            a, b, out, n);                                     \
  }

#define HYLE_SIMD_DEFINE_UNARY_PAIR(token, provider_name, lane, operation,     \
                                    candidate)                                 \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_direct(           \
      const float* src, float* out, std::size_t n) noexcept {                  \
    candidate<Provider::provider_name>::run(src, out, n);                      \
  }                                                                            \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_routed(           \
      const float* src, float* out, std::size_t n) noexcept {                  \
    HYLE_SIMD_INVOKE_ROUTED(ProbeOperation::operation, candidate, lane,        \
                            src, out, n);                                      \
  }

#define HYLE_SIMD_DEFINE_FUSED_PAIR(token, provider_name, lane)                \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_direct(           \
      const float* src, float factor, float bias, float* out,                  \
      std::size_t n) noexcept {                                                \
    FusedCandidate<Provider::provider_name>::run(src, factor, bias, out, n);   \
  }                                                                            \
  extern "C" HYLE_NOINLINE void hyle_simd_codegen_##token##_routed(           \
      const float* src, float factor, float bias, float* out,                  \
      std::size_t n) noexcept {                                                \
    HYLE_SIMD_INVOKE_ROUTED(ProbeOperation::kFusedAffineRelu, FusedCandidate,  \
                            lane, src, factor, bias, out, n);                  \
  }

#define HYLE_SIMD_DEFINE_PROVIDER_PAIRS(prefix, provider_name, lane)           \
  HYLE_SIMD_DEFINE_BINARY_PAIR(prefix##_add, provider_name, lane, kAdd,        \
                               AddCandidate)                                   \
  HYLE_SIMD_DEFINE_UNARY_PAIR(prefix##_copy, provider_name, lane, kCopy,       \
                              CopyCandidate)                                   \
  HYLE_SIMD_DEFINE_UNARY_PAIR(prefix##_relu, provider_name, lane, kRelu,       \
                              ReluCandidate)                                   \
  HYLE_SIMD_DEFINE_UNARY_PAIR(prefix##_tanh, provider_name, lane, kTanh,       \
                              TanhCandidate)                                   \
  HYLE_SIMD_DEFINE_FUSED_PAIR(prefix##_fused, provider_name, lane)

HYLE_SIMD_DEFINE_PROVIDER_PAIRS(hyle, kHyle, HyleLane)
HYLE_SIMD_DEFINE_PROVIDER_PAIRS(highway, kHighway, HighwayLane)
HYLE_SIMD_DEFINE_PROVIDER_PAIRS(xsimd, kXsimd, XsimdLane)

template <Provider kProvider>
HYLE_FORCE_INLINE void masked_load(const std::int32_t* src,
                                   const std::int32_t* mask,
                                   std::int32_t* out,
                                   std::size_t n) noexcept {
  const hn::ScalableTag<std::int32_t> d;
  const auto zero = hn::Zero(d);
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto active = hn::Gt(hn::LoadU(d, mask + i), zero);
    if constexpr (kProvider == Provider::kHighway) {
      hn::StoreU(hn::MaskedLoad(active, d, src + i), d, out + i);
    } else {
      hn::StoreU(hn::IfThenElseZero(active, hn::LoadU(d, src + i)), d,
                 out + i);
    }
  }
  for (; i < n; ++i) out[i] = mask[i] > 0 ? src[i] : 0;
}

extern "C" HYLE_NOINLINE void hyle_simd_peer_masked_load_hyle(
    const std::int32_t* src, const std::int32_t* mask, std::int32_t* out,
    std::size_t n) noexcept {
  masked_load<Provider::kHyle>(src, mask, out, n);
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_masked_load_highway(
    const std::int32_t* src, const std::int32_t* mask, std::int32_t* out,
    std::size_t n) noexcept {
  masked_load<Provider::kHighway>(src, mask, out, n);
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_masked_load_xsimd(
    const std::int32_t* src, const std::int32_t* mask, std::int32_t* out,
    std::size_t n) noexcept {
  using Batch = xsimd::batch<std::int32_t>;
  constexpr std::size_t width = Batch::size;
  const Batch zero(0);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto active = Batch::load_unaligned(mask + i) > zero;
    Batch::load(src + i, active, xsimd::unaligned_mode{})
        .store_unaligned(out + i);
  }
  for (; i < n; ++i) out[i] = mask[i] > 0 ? src[i] : 0;
}

extern "C" HYLE_NOINLINE void hyle_simd_peer_safe_copy_n_hyle(
    const std::int32_t* src, std::int32_t* out, std::size_t n) noexcept {
  const hn::ScalableTag<std::int32_t> d;
  const std::size_t count = n < hn::Lanes(d) ? n : hn::Lanes(d);
  for (std::size_t i = 0; i < count; ++i) out[i] = src[i];
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_safe_copy_n_highway(
    const std::int32_t* src, std::int32_t* out, std::size_t n) noexcept {
  const hn::ScalableTag<std::int32_t> d;
  hn::SafeCopyN(n < hn::Lanes(d) ? n : hn::Lanes(d), d, src, out);
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_safe_fill_n_hyle(
    std::int32_t value, std::int32_t* out, std::size_t n) noexcept {
  const hn::ScalableTag<std::int32_t> d;
  const std::size_t count = n < hn::Lanes(d) ? n : hn::Lanes(d);
  for (std::size_t i = 0; i < count; ++i) out[i] = value;
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_safe_fill_n_highway(
    std::int32_t value, std::int32_t* out, std::size_t n) noexcept {
  const hn::ScalableTag<std::int32_t> d;
  hn::SafeFillN(n < hn::Lanes(d) ? n : hn::Lanes(d), value, d, out);
}

extern "C" HYLE_NOINLINE void hyle_simd_peer_stream_hyle(
    std::uint64_t value, std::uint64_t* aligned, std::size_t n) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const auto vector = hn::Set(d, value);
  const std::size_t width = hn::Lanes(d);
  for (std::size_t i = 0; width <= n - i; i += width) {
#if defined(HYLE_SIMD_TARGET_NEON)
    hyle_winner::neon::stream(vector, d, aligned + i);
#else
    hn::Stream(vector, d, aligned + i);
#endif
  }
}
extern "C" HYLE_NOINLINE void hyle_simd_peer_stream_highway(
    std::uint64_t value, std::uint64_t* aligned, std::size_t n) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const auto vector = hn::Set(d, value);
  const std::size_t width = hn::Lanes(d);
  for (std::size_t i = 0; width <= n - i; i += width) {
    hn::Stream(vector, d, aligned + i);
  }
}

#undef HYLE_SIMD_DEFINE_PROVIDER_PAIRS
#undef HYLE_SIMD_DEFINE_FUSED_PAIR
#undef HYLE_SIMD_DEFINE_UNARY_PAIR
#undef HYLE_SIMD_DEFINE_BINARY_PAIR

}  // namespace hyle::simd::provider::codegen_probe
