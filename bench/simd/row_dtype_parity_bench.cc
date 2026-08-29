// SIMD-07 bounded integration instrument.
//
// The candidate is the exact shipping helper instantiated and force-inlined in
// every placement replica in this TU; the peer is the equivalent direct
// Highway loop, with the same ISA, pointers, count, aliasing and numeric
// contract. This matters: benchmarking one fixed candidate callee against one
// fixed peer callee merely measures their code addresses. A separate
// informational pair measures the public whole-buffer facade against its
// generated target function. The shared parity engine supplies pinning,
// replicated code placement, A/A calibration and the reverse-order/layout bank
// selected by HYLE_BENCH_HWYFIRST.

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "hwy/highway.h"

#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"
#include "hyle/simd/ops.h"
#include "op_parity_engine.h"

// Instantiate the shipping templates at this TU's one CMake-pinned Highway
// target. The unique suffix avoids colliding with the linked product library.
#define HYLE_SIMD_VARIANT_SUFFIX parity
#include "src/simd/ops/kernel_impl.h"
#include "src/simd/dtype/conversion_kernel_impl.h"
#undef HYLE_SIMD_VARIANT_SUFFIX

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr std::size_t kResidentCount = 4096;
constexpr std::size_t kStreamingCount = 1u << 20;

template <typename T, class VectorOp, class ScalarOp>
HWY_INLINE void direct_binary(const T* x, const T* y, T* out,
                              std::size_t n, VectorOp vector_op,
                              ScalarOp scalar_op) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(vector_op(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d,
               out + i);
  }
  for (; i < n; ++i) out[i] = scalar_op(x[i], y[i]);
}

HWY_INLINE void direct_add(const float* x, const float* y, float* out,
                           std::size_t n) noexcept {
  direct_binary(x, y, out, n,
                [](auto a, auto b) { return hn::Add(a, b); },
                [](float a, float b) { return a + b; });
}

HWY_INLINE void direct_multiply(const float* x, const float* y, float* out,
                                std::size_t n) noexcept {
  direct_binary(x, y, out, n,
                [](auto a, auto b) { return hn::Mul(a, b); },
                [](float a, float b) { return a * b; });
}

HWY_INLINE void direct_affine(const float* x, float factor, float bias,
                              float* out, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const auto vf = hn::Set(d, factor);
  const auto vb = hn::Set(d, bias);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::MulAdd(hn::LoadU(d, x + i), vf, vb), d, out + i);
  }
  for (; i < n; ++i) out[i] = x[i] * factor + bias;
}

HWY_INLINE float direct_sum(const float* x, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::Add(sum0, hn::LoadU(d, x + i));
    sum1 = hn::Add(sum1, hn::LoadU(d, x + i + lanes));
    sum2 = hn::Add(sum2, hn::LoadU(d, x + i + lanes * 2));
    sum3 = hn::Add(sum3, hn::LoadU(d, x + i + lanes * 3));
  }
  for (; lanes <= n - i; i += lanes) sum0 = hn::Add(sum0, hn::LoadU(d, x + i));
  float result =
      hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += x[i];
  return result;
}

template <typename HNarrow, typename Storage, class D>
HWY_INLINE auto direct_load_as_f32(D df, const Storage* data) noexcept {
  static_assert(sizeof(HNarrow) == sizeof(Storage));
  const hn::Rebind<HNarrow, D> dn;
  const hn::Rebind<std::uint16_t, D> du16;
#if HWY_TARGET == HWY_SCALAR
  std::uint16_t source_bits;
  std::memcpy(&source_bits, data, sizeof(source_bits));
  const auto packed = hn::Set(du16, source_bits);
#else
  const hn::Repartition<unsigned char, decltype(du16)> du8;
  const auto packed = hn::BitCast(
      du16, hn::LoadU(du8, reinterpret_cast<const unsigned char*>(data)));
#endif
  const auto promoted = hn::PromoteTo(df, hn::BitCast(dn, packed));
  if constexpr (!std::is_same_v<HNarrow, hwy::float16_t>) {
    return promoted;
  } else {
    // Match HYLE's pre-existing exact f16 NaN payload contract. AVX2's
    // VCVTPH2PS is otherwise the direct result; the rare repair is entered only
    // for a NaN vector. Portable targets also repair infinity exponent 31.
    const hn::Rebind<std::uint32_t, D> du32;
    const auto source = hn::PromoteTo(du32, packed);
    const auto exponent =
        hn::And(hn::ShiftRight<10>(source), hn::Set(du32, 0x1Fu));
    const auto mantissa = hn::ShiftLeft<13>(
        hn::And(source, hn::Set(du32, 0x03FFu)));
#if HWY_TARGET <= HWY_AVX2
    const auto special_mask = hn::And(
        hn::Eq(exponent, hn::Set(du32, 0x1Fu)),
        hn::Ne(mantissa, hn::Zero(du32)));
    if (hn::AllFalse(du32, special_mask)) return promoted;
#else
    const auto special_mask = hn::Eq(exponent, hn::Set(du32, 0x1Fu));
#endif
    const auto sign =
        hn::ShiftLeft<16>(hn::And(source, hn::Set(du32, 0x8000u)));
    const auto special = hn::Or(
        sign, hn::Or(hn::Set(du32, 0x7F800000u), mantissa));
    return hn::BitCast(
        df, hn::IfThenElse(special_mask, special,
                           hn::BitCast(du32, promoted)));
  }
}

template <class D, class V>
HWY_INLINE void direct_store_bf16(D d, V value,
                                  hs::bfloat16_t* destination) noexcept {
  (void)d;
  const hn::Rebind<std::uint16_t, D> du16;
#if HWY_TARGET == HWY_SCALAR
  const std::uint16_t bits = hn::GetLane(hn::BitCast(du16, value));
  std::memcpy(destination, &bits, sizeof(bits));
#else
  const hn::Repartition<unsigned char, decltype(du16)> du8;
  hn::StoreU(hn::BitCast(du8, hn::BitCast(du16, value)), du8,
             reinterpret_cast<unsigned char*>(destination));
#endif
}

HWY_INLINE void direct_f16_to_f32(const hs::float16_t* src, float* dst,
                                  std::size_t n) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(direct_load_as_f32<hwy::float16_t>(df, src + i), df,
               dst + i);
  }
  for (; i < n; ++i) dst[i] = hs::f16_to_f32(src[i]);
}

HWY_INLINE void direct_f32_to_bf16(const float* src, hs::bfloat16_t* dst,
                                   std::size_t n) noexcept {
  const hn::ScalableTag<float> df;
#if HWY_TARGET != HWY_SCALAR
  const hn::Repartition<hwy::bfloat16_t, decltype(df)> db_full;
#endif
  const hn::Rebind<hwy::bfloat16_t, decltype(df)> db_half;
  const std::size_t lanes = hn::Lanes(df);
  std::size_t i = 0;
#if HWY_TARGET != HWY_SCALAR
  for (; lanes <= (n - i) / 2; i += 2 * lanes) {
    const auto a = hn::LoadU(df, src + i);
    const auto b = hn::LoadU(df, src + i + lanes);
    if (!hn::AllFalse(df, hn::IsNaN(a)) ||
        !hn::AllFalse(df, hn::IsNaN(b))) {
      for (std::size_t lane = 0; lane < 2 * lanes; ++lane)
        dst[i + lane] = hs::f32_to_bf16(src[i + lane]);
      continue;
    }
    direct_store_bf16(db_full, hn::OrderedDemote2To(db_full, a, b), dst + i);
  }
#endif
  for (; lanes <= n - i; i += lanes) {
    const auto value = hn::LoadU(df, src + i);
    if (!hn::AllFalse(df, hn::IsNaN(value))) {
      for (std::size_t lane = 0; lane < lanes; ++lane)
        dst[i + lane] = hs::f32_to_bf16(src[i + lane]);
    } else {
      direct_store_bf16(db_half, hn::DemoteTo(db_half, value), dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = hs::f32_to_bf16(src[i]);
}

HWY_INLINE float direct_bf16_dot(const hs::bfloat16_t* x,
                                 const hs::bfloat16_t* y,
                                 std::size_t n) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(df);
  auto sum1 = hn::Zero(df);
  auto sum2 = hn::Zero(df);
  auto sum3 = hn::Zero(df);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::MulAdd(direct_load_as_f32<hwy::bfloat16_t>(df, x + i),
                      direct_load_as_f32<hwy::bfloat16_t>(df, y + i), sum0);
    sum1 = hn::MulAdd(direct_load_as_f32<hwy::bfloat16_t>(df, x + i + lanes),
                      direct_load_as_f32<hwy::bfloat16_t>(df, y + i + lanes),
                      sum1);
    sum2 = hn::MulAdd(
        direct_load_as_f32<hwy::bfloat16_t>(df, x + i + lanes * 2),
        direct_load_as_f32<hwy::bfloat16_t>(df, y + i + lanes * 2), sum2);
    sum3 = hn::MulAdd(
        direct_load_as_f32<hwy::bfloat16_t>(df, x + i + lanes * 3),
        direct_load_as_f32<hwy::bfloat16_t>(df, y + i + lanes * 3), sum3);
  }
  for (; lanes <= n - i; i += lanes) {
    sum0 = hn::MulAdd(direct_load_as_f32<hwy::bfloat16_t>(df, x + i),
                      direct_load_as_f32<hwy::bfloat16_t>(df, y + i), sum0);
  }
  float result =
      hn::ReduceSum(df, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += hs::bf16_to_f32(x[i]) * hs::bf16_to_f32(y[i]);
  return result;
}

HWY_INLINE void candidate_add(const float* x, const float* y, float* out,
                              std::size_t n) noexcept {
  hs::detail::binary_row(
      x, y, out, n, [](auto a, auto b) { return hn::Add(a, b); },
      [](float a, float b) { return a + b; });
}
HWY_INLINE void candidate_multiply(const float* x, const float* y, float* out,
                                   std::size_t n) noexcept {
  hs::detail::binary_row(
      x, y, out, n, [](auto a, auto b) { return hn::Mul(a, b); },
      [](float a, float b) { return a * b; });
}
HWY_INLINE void candidate_affine(const float* x, float factor, float bias,
                                 float* out, std::size_t n) noexcept {
  hs::detail::affine_row(x, factor, bias, out, n);
}
HWY_INLINE float candidate_sum(const float* x, std::size_t n) noexcept {
  return hs::detail::sum_row(x, n);
}
HWY_INLINE void candidate_f16_to_f32(const hs::float16_t* src, float* dst,
                                     std::size_t n) noexcept {
  hs::detail::convert_to_f32<hwy::float16_t>(src, dst, n, hs::f16_to_f32);
}
HWY_INLINE void candidate_f32_to_bf16(const float* src,
                                      hs::bfloat16_t* dst,
                                      std::size_t n) noexcept {
  hs::detail::convert_from_f32<hwy::bfloat16_t>(src, dst, n,
                                                hs::f32_to_bf16);
}
HWY_INLINE float candidate_bf16_dot(const hs::bfloat16_t* x,
                                    const hs::bfloat16_t* y,
                                    std::size_t n) noexcept {
  return hs::detail::dot_narrow<hwy::bfloat16_t>(x, y, n,
                                                 hs::bf16_to_f32);
}

std::size_t epoch_iterations(std::size_t n) {
  // Keep an epoch above the observed Windows timer/noise floor. The first
  // qualification probe requested roughly 59k resident iterations. A later
  // streaming probe observed memory/frequency bursts and requested roughly
  // 2.8k iterations, so round both upward rather than admitting an unstable
  // sample as a product verdict.
  return n == kResidentCount ? 65536 : 3072;
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
  std::vector<float> x(n), y(n), out(n);
  std::vector<hs::float16_t> f16(n);
  std::vector<hs::bfloat16_t> bf16_x(n), bf16_y(n), bf16_out(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = (static_cast<int>(i % 251) - 125) * (1.0f / 128.0f);
    y[i] = (static_cast<int>((i * 17) % 241) - 120) * (1.0f / 128.0f);
    f16[i] = hs::f32_to_f16(x[i]);
    bf16_x[i] = hs::f32_to_bf16(x[i]);
    bf16_y[i] = hs::f32_to_bf16(y[i]);
  }
  const std::string suffix = " (" + std::to_string(n) + ")";

  // A/A is deliberately the one source-proven calibration title understood by
  // op-parity-report.py. Both sides invoke the same direct Highway body.
  if (n == kResidentCount) {
    run_pair("Add f32 (4096)", n,
             [&] { direct_add(x.data(), y.data(), out.data(), n); },
             [&] { direct_add(x.data(), y.data(), out.data(), n); });
  }

  run_pair("row add f32" + suffix, n,
           [&] { candidate_add(x.data(), y.data(), out.data(), n); },
           [&] { direct_add(x.data(), y.data(), out.data(), n); });
  run_pair("row multiply f32" + suffix, n,
           [&] { candidate_multiply(x.data(), y.data(), out.data(), n); },
           [&] { direct_multiply(x.data(), y.data(), out.data(), n); });
  run_pair("row affine f32" + suffix, n,
           [&] { candidate_affine(x.data(), 1.25f, -0.5f, out.data(), n); },
           [&] { direct_affine(x.data(), 1.25f, -0.5f, out.data(), n); });
  run_pair("row sum f32" + suffix, n,
           [&] { nb::doNotOptimizeAway(candidate_sum(x.data(), n)); },
           [&] { nb::doNotOptimizeAway(direct_sum(x.data(), n)); });
  run_pair("convert f16-to-f32" + suffix, n,
           [&] { candidate_f16_to_f32(f16.data(), out.data(), n); },
           [&] { direct_f16_to_f32(f16.data(), out.data(), n); });
  run_pair("convert f32-to-bf16" + suffix, n,
           [&] { candidate_f32_to_bf16(x.data(), bf16_out.data(), n); },
           [&] { direct_f32_to_bf16(x.data(), bf16_out.data(), n); });
  run_pair("mixed bf16 dot" + suffix, n,
           [&] {
             nb::doNotOptimizeAway(
                 candidate_bf16_dot(bf16_x.data(), bf16_y.data(), n));
           },
           [&] {
             nb::doNotOptimizeAway(
                 direct_bf16_dot(bf16_x.data(), bf16_y.data(), n));
           });

  // These rows do not decide algorithm parity. They isolate the only intended
  // middleware cost: one public whole-buffer dispatch/call versus its selected
  // target body. No element-loop dispatch is permitted.
  run_public_info("public/body row add" + suffix, n,
                  [&] { hs::add(x.data(), y.data(), out.data(), n); },
                  [&] {
                    hs::detail::add_f32_parity(x.data(), y.data(), out.data(),
                                               n);
                  });
  run_public_info("public/body f16-to-f32" + suffix, n,
                  [&] { hs::convert(f16.data(), out.data(), n); },
                  [&] {
                    hs::detail::f16_to_f32_parity(f16.data(), out.data(), n);
                  });
  run_public_info("public/body bf16 dot" + suffix, n,
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::dot(bf16_x.data(), bf16_y.data(), n));
                  },
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::detail::dot_bf16_bf16_parity(
                            bf16_x.data(), bf16_y.data(), n));
                  });
}

}  // namespace

int main() {
  hyle_bench::pin_and_boost();
  std::printf("SIMD-07 row/dtype parity: inlined shipping helper vs direct Highway\n");
  run_size(kResidentCount);
  run_size(kStreamingCount);
}
