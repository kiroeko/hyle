// SIMD-08 bounded activation/softmax integration instrument.
//
// The primary rows instantiate the shipping target helpers in this TU and
// compare them with an independently spelled direct Highway composition under
// the same ISA, pointer/count, accuracy, alias, tail and resource contract.
// The shared parity engine supplies placement replicas, reverse order, A/A
// calibration and CPU pinning. Public/body rows are informational: they expose
// the one allowed batch-boundary dispatch without mixing it into algorithm
// parity.

#include <nanobench.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "hyle/simd/math.h"
#include "op_parity_engine.h"

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_HIGHWAY_MATH_SUFFIX parity
#include "src/simd/math/highway_math_kernel-inl.h"
#undef HYLE_HIGHWAY_MATH_SUFFIX
#undef HWY_COMPILE_ONLY_STATIC

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;
namespace hp = hyle_highway_math_detail_parity;

namespace {

constexpr std::size_t kResidentCount = 4096;
constexpr std::size_t kStreamingCount = 1u << 20;

template <class VectorFn, class ScalarFn>
HWY_INLINE void direct_unary(const float* src, float* dst, std::size_t n,
                             VectorFn vector_fn,
                             ScalarFn scalar_fn) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    hn::StoreU(vector_fn(d, hn::LoadU(d, src + i)), d, dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_fn(src[i]);
}

HWY_INLINE void peer_add(const float* x, const float* y, float* dst,
                         std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    hn::StoreU(hn::Add(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d,
               dst + i);
  }
  for (; i < n; ++i) dst[i] = x[i] + y[i];
}

HWY_INLINE void candidate_exp(const float* src, float* dst,
                              std::size_t n) noexcept {
  hp::unary_loop<hp::UnaryDomain::kExp>(
      src, dst, n, [](auto d, auto x) { return hn::Exp(d, x); },
      [](float x) { return std::exp(x); });
}

HWY_INLINE void peer_exp(const float* src, float* dst,
                         std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  const auto maximum = hn::Set(d, 104.0f);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto x = hn::LoadU(d, src + i);
    if (hn::AllFalse(d, hn::Or(hn::IsNaN(x), hn::IsInf(x))) &&
        hn::AllTrue(d, hn::Le(x, maximum))) {
      hn::StoreU(hn::Exp(d, x), d, dst + i);
    } else {
      for (std::size_t lane = 0; lane < width; ++lane) {
        dst[i + lane] = std::exp(src[i + lane]);
      }
    }
  }
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}

HWY_INLINE void candidate_gelu_tanh(const float* src, float* dst,
                                    std::size_t n) noexcept {
  hp::unary_loop<hp::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hp::activation_gelu_tanh(d, x); },
      [](float x) { return hp::activation_gelu_tanh_scalar(x); });
}

HWY_INLINE void peer_gelu_tanh(const float* src, float* dst,
                               std::size_t n) noexcept {
  direct_unary(
      src, dst, n,
      [](auto d, auto x) {
        const auto one = hn::Set(d, 1.0f);
        const auto inner = hn::Mul(
            hn::Mul(hn::Set(d, 0.79788456080286535588f), x),
            hn::MulAdd(hn::Set(d, 0.044715f), hn::Mul(x, x), one));
        const auto result = hn::Mul(
            hn::Mul(hn::Set(d, 0.5f), x),
            hn::Add(one, hn::Tanh(d, inner)));
        return hn::IfThenElse(
            hn::And(hn::IsInf(x), hn::Lt(x, hn::Zero(d))), hn::Zero(d),
            result);
      },
      [](float x) {
        if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
        const float inner =
            0.79788456080286535588f * x * std::fma(0.044715f, x * x, 1.0f);
        return 0.5f * x * (1.0f + std::tanh(inner));
      });
}

HWY_INLINE void candidate_silu(const float* src, float* dst,
                               std::size_t n) noexcept {
  hp::unary_loop<hp::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hp::activation_silu(d, x); },
      [](float x) { return hp::activation_silu_scalar(x); });
}

HWY_INLINE void peer_silu(const float* src, float* dst,
                          std::size_t n) noexcept {
  direct_unary(
      src, dst, n,
      [](auto d, auto x) {
        const auto one = hn::Set(d, 1.0f);
        const auto e = hn::Exp(d, hn::Neg(hn::Abs(x)));
        const auto denominator = hn::Add(one, e);
        const auto sigmoid = hn::IfThenElse(
            hn::Lt(x, hn::Zero(d)), hn::Div(e, denominator),
            hn::Div(one, denominator));
        const auto result = hn::Mul(x, sigmoid);
        return hn::IfThenElse(
            hn::And(hn::IsInf(x), hn::Lt(x, hn::Zero(d))), hn::Zero(d),
            result);
      },
      [](float x) {
        if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
        const float e = std::exp(-std::abs(x));
        const float sigmoid =
            x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
        return x * sigmoid;
      });
}

HWY_INLINE bool peer_row_max(const float* src, std::size_t n,
                             float& maximum) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  const auto positive_infinity =
      hn::Set(d, std::numeric_limits<float>::infinity());
  auto maximum_v =
      hn::Set(d, -std::numeric_limits<float>::infinity());
  bool invalid = false;
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto x = hn::LoadU(d, src + i);
    invalid = invalid ||
              !hn::AllFalse(d, hn::Or(hn::IsNaN(x),
                                      hn::Eq(x, positive_infinity)));
    maximum_v = hn::Max(maximum_v, x);
  }
  maximum = hn::ReduceMax(d, maximum_v);
  for (; i < n; ++i) {
    const float x = src[i];
    if (std::isnan(x) || x == std::numeric_limits<float>::infinity()) {
      invalid = true;
    }
    if (x > maximum) maximum = x;
  }
  return !invalid && std::isfinite(maximum);
}

HWY_INLINE void peer_softmax(const float* src, float* dst,
                             std::size_t n) noexcept {
  if (n == 0) return;
  float maximum = 0.0f;
  if (!peer_row_max(src, n, maximum)) {
    const float value = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t i = 0; i < n; ++i) dst[i] = value;
    return;
  }

  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  const auto maximum_v = hn::Set(d, maximum);
  double sum = 0.0;
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto value =
        hn::Exp(d, hn::Sub(hn::LoadU(d, src + i), maximum_v));
    hn::StoreU(value, d, dst + i);
    sum += static_cast<double>(hn::ReduceSum(d, value));
  }
  for (; i < n; ++i) {
    const float value = std::exp(src[i] - maximum);
    dst[i] = value;
    sum += static_cast<double>(value);
  }
  const float inverse_scalar = static_cast<float>(1.0 / sum);
  const auto inverse = hn::Set(d, inverse_scalar);
  i = 0;
  for (; width <= n - i; i += width) {
    hn::StoreU(hn::Mul(hn::LoadU(d, dst + i), inverse), d, dst + i);
  }
  for (; i < n; ++i) dst[i] *= inverse_scalar;
}

std::size_t epoch_iterations(std::size_t n) {
  return n == kResidentCount ? 8192 : 32;
}

template <class Candidate, class Peer>
void run_pair(const std::string& title, std::size_t n, Candidate candidate,
              Peer peer) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(n).minEpochIterations(
      epoch_iterations(n));
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        candidate();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        peer();
        hyle_bench::mark(rep);
      });
}

template <class Public, class Body>
void run_public_info(const std::string& title, std::size_t n,
                     Public public_call, Body body_call) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(n).minEpochIterations(
      epoch_iterations(n));
  HYLE_RUN_INFO_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        public_call();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        body_call();
        hyle_bench::mark(rep);
      });
}

void run_size(std::size_t n) {
  std::vector<float> input(n), other(n), output(n);
  for (std::size_t i = 0; i < n; ++i) {
    input[i] =
        8.0f * std::sin(static_cast<float>(i) * 0.0137f) +
        0.5f * std::cos(static_cast<float>(i) * 0.071f);
    other[i] =
        4.0f * std::cos(static_cast<float>(i) * 0.0191f) - 0.25f;
  }
  const std::string suffix = " (" + std::to_string(n) + ")";

  if (n == kResidentCount) {
    run_pair("Add f32 (4096)", n,
             [&] { peer_add(input.data(), other.data(), output.data(), n); },
             [&] { peer_add(input.data(), other.data(), output.data(), n); });
  }
  run_pair("activation exp f32" + suffix, n,
           [&] { candidate_exp(input.data(), output.data(), n); },
           [&] { peer_exp(input.data(), output.data(), n); });
  run_pair("activation tanh-gelu f32" + suffix, n,
           [&] { candidate_gelu_tanh(input.data(), output.data(), n); },
           [&] { peer_gelu_tanh(input.data(), output.data(), n); });
  run_pair("activation SiLU f32" + suffix, n,
           [&] { candidate_silu(input.data(), output.data(), n); },
           [&] { peer_silu(input.data(), output.data(), n); });
  run_pair("row softmax f32" + suffix, n,
           [&] { hp::activation_softmax(input.data(), output.data(), n); },
           [&] { peer_softmax(input.data(), output.data(), n); });

  run_public_info("public/body activation exp" + suffix, n,
                  [&] { hs::exp_f32_batch(input.data(), output.data(), n); },
                  [&] {
                    hs::detail::exp_f32_highway_parity(
                        input.data(), output.data(), n);
                  });
  run_public_info("public/body tanh-gelu" + suffix, n,
                  [&] {
                    hs::gelu_tanh_f32_batch(input.data(), output.data(), n);
                  },
                  [&] {
                    hs::detail::gelu_tanh_f32_hyle_parity(
                        input.data(), output.data(), n);
                  });
  run_public_info("public/body SiLU" + suffix, n,
                  [&] { hs::silu_f32_batch(input.data(), output.data(), n); },
                  [&] {
                    hs::detail::silu_f32_hyle_parity(
                        input.data(), output.data(), n);
                  });
  run_public_info("public/body softmax" + suffix, n,
                  [&] { hs::softmax_f32(input.data(), output.data(), n); },
                  [&] {
                    hs::detail::softmax_f32_hyle_parity(
                        input.data(), output.data(), n);
                  });
}

}  // namespace

int main() {
  hyle_bench::pin_and_boost();
  std::printf(
      "SIMD-08 activation parity: shipping target helper vs direct Highway\n");
  run_size(kResidentCount);
  run_size(kStreamingCount);
}
