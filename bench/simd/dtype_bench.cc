// dtype cvt bench (task #21): fp16/bf16 <-> f32 batch conversions. README claims
// "fp16/bf16 dtype(F16C)" but it had NO benchmark -- so we never proved the F16C hardware
// path engages or that it beats the scalar reference. The AI hot path (KV-cache, mixed
// precision) converts whole arrays, so this matters.
// GROUPS:
//   * HYLE *_batch        -- dispatched: F16C hardware kernel for fp16 (if cpu has it),
//                            scalar core for bf16 (no AVX2 bf16 convert; header is honest).
//   * HYLE scalar loop    -- per-element dtype.h reference (the always-available baseline =
//                            what you get with no F16C). Shows the hardware speedup.
//   * Highway (SOTA)      -- gated -DHYLE_BENCH_HIGHWAY=ON. Highway also lowers to F16C on
//                            an f16c host, so HYLE-vs-Highway is the "is our wiring SOTA?"
//                            check (a tie = our F16C path is as good as the gold standard).
// NOTE: std has no batch fp16/bf16 conversion (std::float16_t is C++23, no array convert),
// so there is no std group here -- the honest reference set is hardware-vs-scalar-vs-Highway.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"
#include "hyle/simd/runtime.h"

#if defined(HYLE_BENCH_HIGHWAY)
#  include "hwy/highway.h"
namespace hn = hwy::HWY_NAMESPACE;
#endif

using hyle::simd::bf16_to_f32;
using hyle::simd::bfloat16_t;
using hyle::simd::f16_to_f32;
using hyle::simd::f32_to_bf16;
using hyle::simd::f32_to_f16;
using hyle::simd::float16_t;
using hyle::simd::f32_to_f8e4m3;
using hyle::simd::f32_to_f8e5m2;
using hyle::simd::f8e4m3_to_f32;
using hyle::simd::f8e5m2_to_f32;
using hyle::simd::float8_e4m3_t;
using hyle::simd::float8_e5m2_t;

namespace {
namespace nb = ankerl::nanobench;
constexpr std::size_t kN = 4096;  // fits L1; isolates convert throughput

const char* tier() {
  return hyle::simd::selected_target(hyle::simd::KernelFamily::kDtype).isa ==
                 hyle::simd::Isa::kAvx2
             ? "AVX2/F16C"
             : "scalar";
}
}  // namespace

int main() {
  std::printf("HYLE dtype fp16 tier live: %s\n\n", tier());

  std::mt19937_64 rng(1);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> f32(kN);
  for (auto& v : f32) v = dist(rng);
  std::vector<float16_t> f16(kN);
  std::vector<bfloat16_t> bf16(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    f16[i] = f32_to_f16(f32[i]);
    bf16[i] = f32_to_bf16(f32[i]);
  }
  std::vector<float> out_f32(kN);
  std::vector<float16_t> out_f16(kN);

  // ---- fp16 -> f32 (F16C hardware path) ----
  {
    nb::Bench b;
    b.title("f16->f32 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f16_to_f32_batch (F16C)", [&] {
      hyle::simd::f16_to_f32_batch(f16.data(), out_f32.data(), kN);
      nb::doNotOptimizeAway(out_f32.data());
    });
    b.run("HYLE scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) out_f32[i] = f16_to_f32(f16[i]);
      nb::doNotOptimizeAway(out_f32.data());
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway PromoteTo f16->f32 (SOTA)", [&] {
      const hn::ScalableTag<float> df;
      const hn::Rebind<hwy::float16_t, decltype(df)> dh;
      const std::size_t w = hn::Lanes(df);
      std::size_t i = 0;
      for (; i + w <= kN; i += w) {
        auto v = hn::LoadU(dh, reinterpret_cast<const hwy::float16_t*>(f16.data()) + i);
        hn::StoreU(hn::PromoteTo(df, v), df, out_f32.data() + i);
      }
      for (; i < kN; ++i) out_f32[i] = f16_to_f32(f16[i]);
      nb::doNotOptimizeAway(out_f32.data());
    });
#endif
  }

  // ---- f32 -> fp16 (F16C hardware path) ----
  {
    nb::Bench b;
    b.title("f32->f16 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f32_to_f16_batch (F16C)", [&] {
      hyle::simd::f32_to_f16_batch(f32.data(), out_f16.data(), kN);
      nb::doNotOptimizeAway(out_f16.data());
    });
    b.run("HYLE scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) out_f16[i] = f32_to_f16(f32[i]);
      nb::doNotOptimizeAway(out_f16.data());
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway DemoteTo f32->f16 (SOTA)", [&] {
      const hn::ScalableTag<float> df;
      const hn::Rebind<hwy::float16_t, decltype(df)> dh;
      const std::size_t w = hn::Lanes(df);
      std::size_t i = 0;
      for (; i + w <= kN; i += w) {
        auto v = hn::LoadU(df, f32.data() + i);
        hn::StoreU(hn::DemoteTo(dh, v), dh,
                   reinterpret_cast<hwy::float16_t*>(out_f16.data()) + i);
      }
      for (; i < kN; ++i) out_f16[i] = f32_to_f16(f32[i]);
      nb::doNotOptimizeAway(out_f16.data());
    });
#endif
  }

  // ---- bf16 <-> f32 (AVX2 8-wide integer; bf16 = top 16 bits of f32) ----
  {
    std::vector<float> bout(kN);
    std::vector<bfloat16_t> bf_out(kN);
    nb::Bench b;
    b.title("bf16->f32 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE bf16_to_f32_batch (AVX2)", [&] {
      hyle::simd::bf16_to_f32_batch(bf16.data(), bout.data(), kN);
      nb::doNotOptimizeAway(bout.data());
    });
    b.run("HYLE scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) bout[i] = bf16_to_f32(bf16[i]);
      nb::doNotOptimizeAway(bout.data());
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway PromoteTo bf16->f32 (SOTA)", [&] {
      const hn::ScalableTag<float> df;
      const hn::Rebind<hwy::bfloat16_t, decltype(df)> dbf;
      const std::size_t w = hn::Lanes(df);
      std::size_t i = 0;
      for (; i + w <= kN; i += w) {
        auto v = hn::LoadU(dbf, reinterpret_cast<const hwy::bfloat16_t*>(bf16.data()) + i);
        hn::StoreU(hn::PromoteTo(df, v), df, bout.data() + i);
      }
      for (; i < kN; ++i) bout[i] = bf16_to_f32(bf16[i]);
      nb::doNotOptimizeAway(bout.data());
    });
#endif

    nb::Bench b2;
    b2.title("f32->bf16 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b2.run("HYLE f32_to_bf16_batch (AVX2)", [&] {
      hyle::simd::f32_to_bf16_batch(f32.data(), bf_out.data(), kN);
      nb::doNotOptimizeAway(bf_out.data());
    });
    b2.run("HYLE scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) bf_out[i] = f32_to_bf16(f32[i]);
      nb::doNotOptimizeAway(bf_out.data());
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b2.run("Highway DemoteTo f32->bf16 (naive 1-vec)", [&] {
      const hn::ScalableTag<float> df;
      const hn::Rebind<hwy::bfloat16_t, decltype(df)> dbf;
      const std::size_t w = hn::Lanes(df);
      std::size_t i = 0;
      for (; i + w <= kN; i += w) {
        auto v = hn::LoadU(df, f32.data() + i);
        hn::StoreU(hn::DemoteTo(dbf, v), dbf,
                   reinterpret_cast<hwy::bfloat16_t*>(bf_out.data()) + i);
      }
      for (; i < kN; ++i) bf_out[i] = f32_to_bf16(f32[i]);
      nb::doNotOptimizeAway(bf_out.data());
    });
    // Highway PRODUCTION shape: OrderedDemote2To (2 f32 vecs -> 1 full bf16 vec,
    // one packus+vpermq for 16 elems + full store). The naive DemoteTo above is the
    // 8->8 convenience form (packus+vpermq per 8 + half store) that HYLE used to
    // mirror -- racing only against it hid the production ceiling (same trap as the
    // dot single-accumulator case). This is the honest SOTA yardstick.
    b2.run("Highway OrderedDemote2To (SOTA 2-vec)", [&] {
      const hn::ScalableTag<float> df;
      const hn::Repartition<hwy::bfloat16_t, decltype(df)> dbf;
      const std::size_t w = hn::Lanes(df);
      std::size_t i = 0;
      for (; i + 2 * w <= kN; i += 2 * w) {
        auto a = hn::LoadU(df, f32.data() + i);
        auto b = hn::LoadU(df, f32.data() + i + w);
        auto packed = hn::OrderedDemote2To(dbf, a, b);
        hn::StoreU(packed, dbf,
                   reinterpret_cast<hwy::bfloat16_t*>(bf_out.data()) + i);
      }
      for (; i < kN; ++i) bf_out[i] = f32_to_bf16(f32[i]);
      nb::doNotOptimizeAway(bf_out.data());
    });
#endif
  }

  // fp8 (OCP E4M3FN / E5M2) <-> f32 (directive3 fill; dtype_bench only had fp16/bf16). No CPU has
  // hardware fp8 convert (AVX10.2 f16->fp8 is not bit-identical + clang-20-only, see gap-inventory),
  // so the batch is the SCALAR bit-math core -- benched vs a naive per-element loop to show the
  // batch's auto-vectorized throughput. Both formats, both directions.
  std::vector<float8_e4m3_t> e4m3(kN);
  std::vector<float8_e5m2_t> e5m2(kN);
  for (std::size_t i = 0; i < kN; ++i) { e4m3[i] = f32_to_f8e4m3(f32[i]); e5m2[i] = f32_to_f8e5m2(f32[i]); }
  {
    nb::Bench b;
    b.title("f32->f8e4m3 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f32_to_f8e4m3_batch", [&] {
      hyle::simd::f32_to_f8e4m3_batch(f32.data(), e4m3.data(), kN);
      nb::doNotOptimizeAway(e4m3.data());
    });
    b.run("scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) e4m3[i] = f32_to_f8e4m3(f32[i]);
      nb::doNotOptimizeAway(e4m3.data());
    });
  }
  {
    std::vector<float> out(kN);
    nb::Bench b;
    b.title("f8e4m3->f32 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f8e4m3_to_f32_batch", [&] {
      hyle::simd::f8e4m3_to_f32_batch(e4m3.data(), out.data(), kN);
      nb::doNotOptimizeAway(out.data());
    });
    b.run("scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) out[i] = f8e4m3_to_f32(e4m3[i]);
      nb::doNotOptimizeAway(out.data());
    });
  }
  {
    nb::Bench b;
    b.title("f32->f8e5m2 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f32_to_f8e5m2_batch", [&] {
      hyle::simd::f32_to_f8e5m2_batch(f32.data(), e5m2.data(), kN);
      nb::doNotOptimizeAway(e5m2.data());
    });
    b.run("scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) e5m2[i] = f32_to_f8e5m2(f32[i]);
      nb::doNotOptimizeAway(e5m2.data());
    });
  }
  {
    std::vector<float> out(kN);
    nb::Bench b;
    b.title("f8e5m2->f32 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f8e5m2_to_f32_batch", [&] {
      hyle::simd::f8e5m2_to_f32_batch(e5m2.data(), out.data(), kN);
      nb::doNotOptimizeAway(out.data());
    });
    b.run("scalar loop (reference)", [&] {
      for (std::size_t i = 0; i < kN; ++i) out[i] = f8e5m2_to_f32(e5m2[i]);
      nb::doNotOptimizeAway(out.data());
    });
  }

  // f16 -> fp8 (source direction missing from earlier dtype_bench; vs scalar + vs f32 path cost)
  {
    nb::Bench b;
    b.title("f16->f8e4m3 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f16_to_f8e4m3_batch", [&] {
      hyle::simd::f16_to_f8e4m3_batch(f16.data(), e4m3.data(), kN);
      nb::doNotOptimizeAway(e4m3.data());
    });
    b.run("via f16->f32->f8e4m3 batch", [&] {
      hyle::simd::f16_to_f32_batch(f16.data(), out_f32.data(), kN);
      hyle::simd::f32_to_f8e4m3_batch(out_f32.data(), e4m3.data(), kN);
      nb::doNotOptimizeAway(e4m3.data());
    });
  }
  {
    nb::Bench b;
    b.title("f16->f8e5m2 batch (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE f16_to_f8e5m2_batch", [&] {
      hyle::simd::f16_to_f8e5m2_batch(f16.data(), e5m2.data(), kN);
      nb::doNotOptimizeAway(e5m2.data());
    });
    b.run("via f16->f32->f8e5m2 batch", [&] {
      hyle::simd::f16_to_f32_batch(f16.data(), out_f32.data(), kN);
      hyle::simd::f32_to_f8e5m2_batch(out_f32.data(), e5m2.data(), kN);
      nb::doNotOptimizeAway(e5m2.data());
    });
  }

  return 0;
}
