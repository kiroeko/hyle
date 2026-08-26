#include <nanobench.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/simd/math.h"
#include "hyle/simd/runtime.h"
#include "src/simd/math/complex_hyle-inl.h"
#include "src/simd/math/complex_typed_hyle-inl.h"
#include "xsimd/xsimd.hpp"

namespace {
namespace nb = ankerl::nanobench;
using Complex = std::complex<float>;
using Batch = xsimd::batch<Complex>;
using UnaryFn = void (*)(const float*, const float*, float*, float*, std::size_t) noexcept;

#define HYLE_COMPLEX_UNARY(name)                                                               \
  HYLE_NOINLINE void scalar_##name(const float* sr, const float* si, float* dr, float* di,      \
                                    std::size_t n) noexcept {                                    \
    for (std::size_t i = 0; i < n; ++i) {                                                       \
      const Complex value = std::name(Complex(sr[i], si[i]));                                   \
      dr[i] = value.real();                                                                      \
      di[i] = value.imag();                                                                      \
    }                                                                                            \
  }                                                                                              \
  HYLE_NOINLINE void xsimd_##name(const float* sr, const float* si, float* dr, float* di,        \
                                   std::size_t n) noexcept {                                     \
    constexpr std::size_t width = Batch::size;                                                   \
    std::size_t i = 0;                                                                           \
    for (; i + width <= n; i += width)                                                           \
      xsimd::name(Batch::load_unaligned(sr + i, si + i)).store_unaligned(dr + i, di + i);       \
    for (; i < n; ++i) {                                                                         \
      const Complex value = std::name(Complex(sr[i], si[i]));                                   \
      dr[i] = value.real();                                                                      \
      di[i] = value.imag();                                                                      \
    }                                                                                            \
  }

HYLE_COMPLEX_UNARY(exp)
HYLE_COMPLEX_UNARY(log)
HYLE_COMPLEX_UNARY(sqrt)
HYLE_COMPLEX_UNARY(sin)
HYLE_COMPLEX_UNARY(cos)
HYLE_COMPLEX_UNARY(tan)
HYLE_COMPLEX_UNARY(sinh)
HYLE_COMPLEX_UNARY(cosh)
HYLE_COMPLEX_UNARY(tanh)
#undef HYLE_COMPLEX_UNARY

HYLE_NOINLINE void candidate_sin(const float* sr, const float* si, float* dr, float* di,
                                 std::size_t n) noexcept {
  hyle::simd::detail::complex_sincos_hyle<false>(sr, si, dr, di, n);
}
HYLE_NOINLINE void candidate_cos(const float* sr, const float* si, float* dr, float* di,
                                 std::size_t n) noexcept {
  hyle::simd::detail::complex_sincos_hyle<true>(sr, si, dr, di, n);
}
HYLE_NOINLINE void candidate_tanh(const float* sr, const float* si, float* dr, float* di,
                                  std::size_t n) noexcept {
  hyle::simd::detail::complex_ctanh_hyle(sr, si, dr, di, n);
}
HYLE_NOINLINE void candidate_tan(const float* sr, const float* si, float* dr, float* di,
                                 std::size_t n) noexcept {
  hyle::simd::detail::complex_tan_typed_hyle(sr, si, dr, di, n);
}
HYLE_NOINLINE void candidate_sinh(const float* sr, const float* si, float* dr, float* di,
                                  std::size_t n) noexcept {
  hyle::simd::detail::complex_sinhcosh_typed_hyle<float, false>(sr, si, dr, di, n);
}
HYLE_NOINLINE void candidate_cosh(const float* sr, const float* si, float* dr, float* di,
                                  std::size_t n) noexcept {
  hyle::simd::detail::complex_sinhcosh_typed_hyle<float, true>(sr, si, dr, di, n);
}

HYLE_NOINLINE void scalar_pow(const float* zr, const float* zi, const float* wr, const float* wi,
                              float* dr, float* di, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const Complex value = std::pow(Complex(zr[i], zi[i]), Complex(wr[i], wi[i]));
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

HYLE_NOINLINE void candidate_pow(const float* zr, const float* zi, const float* wr, const float* wi,
                                 float* dr, float* di, std::size_t n) noexcept {
  hyle::simd::detail::complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}

HYLE_NOINLINE void xsimd_pow(const float* zr, const float* zi, const float* wr, const float* wi,
                             float* dr, float* di, std::size_t n) noexcept {
  constexpr std::size_t width = Batch::size;
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    xsimd::pow(Batch::load_unaligned(zr + i, zi + i),
               Batch::load_unaligned(wr + i, wi + i))
        .store_unaligned(dr + i, di + i);
  }
  for (; i < n; ++i) {
    const Complex value = std::pow(Complex(zr[i], zi[i]), Complex(wr[i], wi[i]));
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

std::vector<float> make_plane(std::size_t n, float lo, float hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> values(std::max<std::size_t>(n, 1));
  for (float& value : values) value = dist(rng);
  return values;
}

struct UnaryCase {
  const char* name;
  UnaryFn scalar;
  UnaryFn direct;
  UnaryFn candidate;
  UnaryFn public_api;
};

void run_unary(const UnaryCase& op, std::size_t n) {
  auto re = make_plane(n, -3.0f, 3.0f, 0xC001 + n);
  auto im = make_plane(n, -2.0f, 2.0f, 0xC002 + n);
  const char* distribution = std::getenv("HYLE_BENCH_DIST");
  if (distribution != nullptr && std::strcmp(distribution, "real") == 0) {
    std::fill(im.begin(), im.end(), 0.0f);
  } else if (distribution != nullptr && std::strcmp(distribution, "tiny") == 0) {
    for (std::size_t i = 0; i < n; ++i) im[i] = (i & 1) ? -1e-8f : 1e-8f;
  } else if (distribution != nullptr && std::strcmp(distribution, "sparse-special") == 0) {
    for (std::size_t i = 0; i < n; i += Batch::size)
      im[i] = std::numeric_limits<float>::infinity();
  } else if (distribution != nullptr && std::strcmp(distribution, "near-pole") == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      re[i] = (i & 1) ? -0.01f : 0.01f;
      im[i] = (i & 2) ? -1.5707963f + 0.001f : 1.5707963f - 0.001f;
    }
  } else if (distribution != nullptr && std::strcmp(distribution, "large-real") == 0) {
    for (std::size_t i = 0; i < n; ++i) re[i] = (i & 1) ? -45.0f : 45.0f;
  } else if (distribution != nullptr && std::strcmp(distribution, "near-cut") == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      re[i] = -0.5f - 0.01f * static_cast<float>(i % 8);
      im[i] = (i & 1) ? -1e-5f : 1e-5f;
    }
  } else if (distribution != nullptr && std::strcmp(distribution, "zero-base") == 0) {
    std::fill(re.begin(), re.end(), 0.0f);
    for (std::size_t i = 0; i < n; ++i) im[i] = (i & 1) ? -0.0f : 0.0f;
  }
  std::vector<float> out_re(std::max<std::size_t>(n, 1));
  std::vector<float> out_im(std::max<std::size_t>(n, 1));

  nb::Bench bench;
  const std::string title = std::string(op.name) + " complex n=" + std::to_string(n) +
                            " " + (distribution == nullptr ? "random" : distribution);
  bench.title(title).unit("complex").relative(true).batch(std::max<std::size_t>(n, 1));
  bench.run("std scalar direct", [&] {
    op.scalar(re.data(), im.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
  bench.run("xsimd direct", [&] {
    op.direct(re.data(), im.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
  if (op.candidate != nullptr) {
    bench.run("HYLE candidate direct", [&] {
      op.candidate(re.data(), im.data(), out_re.data(), out_im.data(), n);
      nb::doNotOptimizeAway(out_re.data());
    });
  }
  bench.run("HYLE public", [&] {
    op.public_api(re.data(), im.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
}

void run_pow(std::size_t n) {
  auto zr = make_plane(n, 0.25f, 3.0f, 0xD001 + n);
  auto zi = make_plane(n, -2.0f, 2.0f, 0xD002 + n);
  auto wr = make_plane(n, -2.0f, 2.0f, 0xD003 + n);
  auto wi = make_plane(n, -1.0f, 1.0f, 0xD004 + n);
  const char* distribution = std::getenv("HYLE_BENCH_DIST");
  if (distribution != nullptr && std::strcmp(distribution, "near-cut") == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      zr[i] = -0.5f - 0.01f * static_cast<float>(i % 8);
      zi[i] = (i & 1) ? -0.0f : 0.0f;
    }
  } else if (distribution != nullptr && std::strcmp(distribution, "zero-base") == 0) {
    std::fill(zr.begin(), zr.end(), 0.0f);
    std::fill(zi.begin(), zi.end(), 0.0f);
  } else if (distribution != nullptr && std::strcmp(distribution, "large-base") == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      zr[i] = (i & 1) ? -1.0e18f : 1.0e18f;
      zi[i] = (i & 2) ? -0.5f : 0.5f;
    }
  }
  std::vector<float> out_re(std::max<std::size_t>(n, 1));
  std::vector<float> out_im(std::max<std::size_t>(n, 1));

  nb::Bench bench;
  const std::string title = "pow complex n=" + std::to_string(n) + " " +
                            (distribution == nullptr ? "random" : distribution);
  bench.title(title).unit("complex").relative(true).batch(std::max<std::size_t>(n, 1));
  bench.run("std scalar direct", [&] {
    scalar_pow(zr.data(), zi.data(), wr.data(), wi.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
  bench.run("xsimd direct", [&] {
    xsimd_pow(zr.data(), zi.data(), wr.data(), wi.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
  bench.run("HYLE candidate direct", [&] {
    candidate_pow(zr.data(), zi.data(), wr.data(), wi.data(), out_re.data(), out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
  bench.run("HYLE public", [&] {
    hyle::simd::cpow_f32_batch(zr.data(), zi.data(), wr.data(), wi.data(), out_re.data(),
                               out_im.data(), n);
    nb::doNotOptimizeAway(out_re.data());
  });
}

bool selected_text(const char* variable, const char* value) {
  const char* text = std::getenv(variable);
  return text == nullptr || std::strcmp(text, value) == 0;
}

bool selected_number(const char* variable, std::size_t value) {
  const char* text = std::getenv(variable);
  return text == nullptr || static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) == value;
}

}  // namespace

int main() {
  const UnaryCase ops[] = {
      {"exp", scalar_exp, xsimd_exp, nullptr, hyle::simd::cexp_f32_batch},
      {"log", scalar_log, xsimd_log, nullptr, hyle::simd::clog_f32_batch},
      {"sqrt", scalar_sqrt, xsimd_sqrt, nullptr, hyle::simd::csqrt_f32_batch},
      {"sin", scalar_sin, xsimd_sin, candidate_sin, hyle::simd::csin_f32_batch},
      {"cos", scalar_cos, xsimd_cos, candidate_cos, hyle::simd::ccos_f32_batch},
      {"tan", scalar_tan, xsimd_tan, candidate_tan, hyle::simd::ctan_f32_batch},
      {"sinh", scalar_sinh, xsimd_sinh, candidate_sinh, hyle::simd::csinh_f32_batch},
      {"cosh", scalar_cosh, xsimd_cosh, candidate_cosh, hyle::simd::ccosh_f32_batch},
      {"tanh", scalar_tanh, xsimd_tanh, candidate_tanh, hyle::simd::ctanh_f32_batch},
  };
  std::printf("HYLE math tier=%u, xsimd complex lanes=%zu\n",
              static_cast<unsigned>(hyle::simd::runtime().math.isa), Batch::size);
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
                        std::size_t{4}, std::size_t{7}, std::size_t{8}, std::size_t{9},
                        std::size_t{15}, std::size_t{16}, std::size_t{17}, std::size_t{31},
                        std::size_t{32}, std::size_t{33}, std::size_t{64}, std::size_t{127},
                        std::size_t{128}, std::size_t{256}, std::size_t{1024},
                        std::size_t{4096}}) {
    if (!selected_number("HYLE_BENCH_N", n)) continue;
    for (const UnaryCase& op : ops)
      if (selected_text("HYLE_BENCH_OP", op.name)) run_unary(op, n);
    if (selected_text("HYLE_BENCH_OP", "pow")) run_pow(n);
  }
  return 0;
}
