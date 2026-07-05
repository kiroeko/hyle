#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "hyle/simd/ops.h"

// ops_test -- validates the RUNTIME-DISPATCHED ops (simd-04). Unlike vec_test (compiled
// per-ISA), this links the ONE dispatcher and checks: (1) results match a naive scalar
// reference regardless of which variant the host selected; (2) active_variant() really
// picked the host's best compiled tier -- proving best_isa() is now LIVE (consumed to
// select a kernel), not the dead detector the simd-01 audit flagged.

using hyle::simd::active_variant;
using hyle::simd::best_isa;
using hyle::simd::dot_f32;
using hyle::simd::Isa;
using hyle::simd::saxpy_f32;

namespace {
std::vector<float> gen(std::size_t n, float base) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = base + static_cast<float>((i * 7 + 1) % 17) * 0.25f;
  return v;
}
double ref_dot(const std::vector<float>& x, const std::vector<float>& y) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += double(x[i]) * y[i];
  return s;
}
}  // namespace

// dot result matches a DOUBLE-precision reference across sizes that exercise full vectors
// + a tail (n not a multiple of any backend width: 1,7,8,31,64,1000). Tolerance is
// RELATIVE to the magnitude: fp reorder (tree-reduce vs sequential) drifts ~1e-5 relative,
// but a real logic bug (wrong lane, off-by-one tail) shifts the result by ~one element's
// worth = a large relative error, so this still catches bugs. Exact per-op math is pinned
// separately in vec_test (integer oracles); here we validate the DISPATCH produces a sane
// result from whichever variant the host picked.
TEST(Ops, DotMatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                        std::size_t{64}, std::size_t{1000}}) {
    auto x = gen(n, 0.5f), y = gen(n, -0.3f);
    const double ref = ref_dot(x, y);
    float got = dot_f32(x.data(), y.data(), n);
    const float tol = static_cast<float>(std::abs(ref)) * 1e-4f + 1e-4f;
    EXPECT_NEAR(got, static_cast<float>(ref), tol) << "n=" << n;
  }
}

TEST(Ops, DotEmptyIsZero) {
  EXPECT_EQ(dot_f32(nullptr, nullptr, 0), 0.0f);
}

// saxpy result matches naive reference; tail handled.
TEST(Ops, SaxpyMatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{31}, std::size_t{257}}) {
    auto x = gen(n, 1.0f);
    auto y = gen(n, 2.0f);
    std::vector<float> expect(n);
    const float a = 1.5f;
    for (std::size_t i = 0; i < n; ++i) expect[i] = a * x[i] + y[i];
    saxpy_f32(a, x.data(), y.data(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(y[i], expect[i]) << "n=" << n << " i=" << i;
  }
}

// THE simd-04 point: the dispatcher selected the host's best COMPILED tier. We ship
// scalar/sse4/avx2/avx512, so on an x86 host active_variant() must equal best_isa() mapped
// to {scalar,sse42,avx2,avx512} -- never below what the CPU supports, never above what we
// compiled. This proves best_isa() is actually consumed (was dead before simd-04).
TEST(Ops, ActiveVariantMatchesBestCompiledTier) {
  const Isa active = active_variant();
#if defined(HYLE_ARCH_X64)
  const auto best = static_cast<std::uint32_t>(best_isa());
  Isa expect;
  if (best >= static_cast<std::uint32_t>(Isa::kAvx512)) expect = Isa::kAvx512;
  else if (best >= static_cast<std::uint32_t>(Isa::kAvx2)) expect = Isa::kAvx2;
  else if (best >= static_cast<std::uint32_t>(Isa::kSse42)) expect = Isa::kSse42;
  else expect = Isa::kScalar;
  EXPECT_EQ(active, expect);
  // On any modern x86-64 dev box this should be a real SIMD tier, not the floor.
  EXPECT_GE(static_cast<std::uint32_t>(active), static_cast<std::uint32_t>(Isa::kSse42));
#elif defined(HYLE_ARCH_ARM64)
  // aarch64 ships scalar + neon (simd-03 real-run landing); NEON is the ARMv8 baseline, so
  // any real arm64 CPU dispatches to kNeon, not the scalar floor.
  if (static_cast<std::uint32_t>(best_isa()) >= static_cast<std::uint32_t>(Isa::kNeon))
    EXPECT_EQ(active, Isa::kNeon);
  else
    EXPECT_EQ(active, Isa::kScalar);
#else
  EXPECT_EQ(active, Isa::kScalar);  // unknown arch: portable floor only
#endif
}

// Dispatch is stable: repeated calls return the same cached selection (no re-resolve).
TEST(Ops, ActiveVariantStable) {
  EXPECT_EQ(active_variant(), active_variant());
}
