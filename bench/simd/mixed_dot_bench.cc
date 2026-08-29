#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/highway.h"

#include "hyle/simd/ops.h"
#include "hyle/simd/runtime.h"

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

void run_size(std::size_t n) {
  static_assert(sizeof(hs::bfloat16_t) == sizeof(hwy::bfloat16_t));
  std::vector<float> f32(n);
  std::vector<hs::bfloat16_t> bf0(n), bf1(n);
  std::vector<hs::float16_t> f16a(n), f16b(n);
  std::vector<std::int16_t> i16a(n), i16b(n);
  std::vector<std::int8_t> i8a(n), i8b(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float a = (static_cast<int>(i % 251) - 125) * (1.0f / 128.0f);
    const float b = (static_cast<int>((i * 17) % 241) - 120) * (1.0f / 128.0f);
    f32[i] = a + 0.0001f;
    bf0[i] = hs::f32_to_bf16(a);
    bf1[i] = hs::f32_to_bf16(b);
    f16a[i] = hs::f32_to_f16(a);
    f16b[i] = hs::f32_to_f16(b);
    i16a[i] = static_cast<std::int16_t>(static_cast<int>(i % 32749) - 16374);
    i16b[i] = static_cast<std::int16_t>(16373 - static_cast<int>((i * 7) % 32747));
    i8a[i] = static_cast<std::int8_t>(static_cast<int>(i % 255) - 127);
    i8b[i] = static_cast<std::int8_t>(127 - static_cast<int>((i * 7) % 255));
  }
  const auto* hbf0 = reinterpret_cast<const hwy::bfloat16_t*>(bf0.data());
  const auto* hbf1 = reinterpret_cast<const hwy::bfloat16_t*>(bf1.data());
  const hn::ScalableTag<hwy::bfloat16_t> hdb;
  const hn::ScalableTag<float> hdf;
  const std::size_t iterations = n <= 4096 ? 2000 : 256;

  {
    nb::Bench b;
    b.title("bf16 x bf16 -> f32 dot, n=" + std::to_string(n))
        .unit("element").batch(n).relative(true).minEpochIterations(iterations);
    b.run("Highway contrib/dot", [&] {
      nb::doNotOptimizeAway(hn::Dot::Compute<hn::Dot::kAtLeastOneVector>(
          hdb, hbf0, hbf1, n));
    });
    b.run("HYLE public", [&] {
      nb::doNotOptimizeAway(hs::dot(bf0.data(), bf1.data(), n));
    });
  }
  {
    nb::Bench b;
    b.title("f32 x bf16 -> f32 dot, n=" + std::to_string(n))
        .unit("element").batch(n).relative(true).minEpochIterations(iterations);
    b.run("Highway contrib/dot", [&] {
      nb::doNotOptimizeAway(hn::Dot::Compute<hn::Dot::kAtLeastOneVector>(
          hdf, f32.data(), hbf1, n));
    });
    b.run("HYLE public", [&] {
      nb::doNotOptimizeAway(hs::dot(f32.data(), bf1.data(), n));
    });
  }
  {
    nb::Bench b;
    b.title("HYLE storage/exact dot, n=" + std::to_string(n))
        .unit("element").batch(n).relative(true).minEpochIterations(iterations);
    b.run("f16 scalar contract", [&] {
      float sum = 0.0f;
      for (std::size_t i = 0; i < n; ++i)
        sum += hs::f16_to_f32(f16a[i]) * hs::f16_to_f32(f16b[i]);
      nb::doNotOptimizeAway(sum);
    });
    b.run("f16 HYLE public", [&] {
      nb::doNotOptimizeAway(hs::dot(f16a.data(), f16b.data(), n));
    });
    b.run("i16-i64 scalar contract", [&] {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += static_cast<std::uint64_t>(
            static_cast<std::int64_t>(i16a[i]) * i16b[i]);
      nb::doNotOptimizeAway(sum);
    });
    b.run("i16-i64 HYLE public", [&] {
      nb::doNotOptimizeAway(hs::dot(i16a.data(), i16b.data(), n));
    });
    b.run("i8-i64 scalar contract", [&] {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += static_cast<std::uint64_t>(
            static_cast<std::int64_t>(i8a[i]) * i8b[i]);
      nb::doNotOptimizeAway(sum);
    });
    b.run("i8-i64 HYLE public", [&] {
      nb::doNotOptimizeAway(hs::dot(i8a.data(), i8b.data(), n));
    });
  }
}

}  // namespace

int main() {
  std::printf("HYLE ops ISA=%u; direct Highway TU is target-bound by CMake\n",
              static_cast<unsigned>(
                  hs::selected_target(hs::KernelFamily::kOps).isa));
  run_size(4096);
  run_size(1u << 20);
}
