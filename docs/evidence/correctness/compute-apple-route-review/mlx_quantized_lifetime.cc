// Research only: quantized GPU composition and result/input ownership.
#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
constexpr int kK = 256, kH = 256, kGroup = 64;
using Function = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

std::vector<float> values(int n, int seed) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = ((i * 17 + seed) % 31 - 15) / 256.0f;
  return v;
}

struct Weight {
  mx::array codes, scales, biases;
  std::vector<double> decoded;
};

Weight pack(int rows, int cols, int seed, mx::Stream stream) {
  const auto host = values(rows * cols, seed);
  auto q = mx::quantize(mx::array(host.begin(), {rows, cols}),
                        kGroup, 4, "affine", std::nullopt, stream);
  mx::async_eval(q);
  mx::synchronize(stream);
  const auto* codes = q.at(0).data<std::uint32_t>();
  const float* scales = q.at(1).data<float>();
  const float* biases = q.at(2).data<float>();
  std::vector<double> decoded(rows * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const auto group = row * (cols / kGroup) + col / kGroup;
      const auto nibble = (codes[row * (cols / 8) + col / 8] >> (4 * (col % 8))) & 15;
      decoded[row * cols + col] = double(scales[group]) * nibble + biases[group];
    }
  }
  return {q[0], q[1], q[2], std::move(decoded)};
}

std::vector<double> oracle(const std::vector<float>& x, int m,
                           const Weight& g, const Weight& u, const Weight& d) {
  std::vector<double> h(m * kH), y(m * kK);
  for (int row = 0; row < m; ++row) {
    for (int j = 0; j < kH; ++j) {
      double a = 0, b = 0;
      for (int k = 0; k < kK; ++k) {
        a += double(x[row * kK + k]) * g.decoded[j * kK + k];
        b += double(x[row * kK + k]) * u.decoded[j * kK + k];
      }
      h[row * kH + j] = a / (1.0 + std::exp(-a)) * b;
    }
    for (int j = 0; j < kK; ++j)
      for (int k = 0; k < kH; ++k)
        y[row * kK + j] += h[row * kH + k] * d.decoded[j * kH + k];
  }
  return y;
}

double check(const mx::array& output, const std::vector<double>& expected) {
  const float* actual = output.data<float>();
  double worst = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double error = std::abs(actual[i] - expected[i]);
    worst = std::max(worst, error);
    if (!std::isfinite(actual[i]) || error > 1e-4 * std::max(1.0, std::abs(expected[i])))
      throw std::runtime_error("quantized FFN oracle mismatch");
  }
  return worst;
}

int main() {
  try {
    const auto stream = mx::new_stream(mx::Device::gpu);
    mx::set_default_device(mx::Device::gpu);
    const Weight g = pack(kH, kK, 1, stream), u = pack(kH, kK, 7, stream);
    const Weight d = pack(kK, kH, 19, stream);
    auto custom = mx::fast::metal_kernel(
        "hyle_research_q4_swiglu", {"gate", "up"}, {"out"},
        "uint i = thread_position_in_grid.x;\n"
        "out[i] = gate[i] / (1.0f + exp(-gate[i])) * up[i];\n");
    std::printf("MLX=0.32.2 GPU=Metal bits=4 group=%d K=%d H=%d tolerance=1e-4\n",
                kGroup, kK, kH);
    for (int m : {1, 7}) {
      int traces = 0;
      Function body = [&, m](const auto& args) {
        ++traces;
        auto gate = mx::quantized_matmul(args.at(0), g.codes, g.scales, g.biases,
                                         true, kGroup, 4, "affine", stream);
        auto up = mx::quantized_matmul(args.at(0), u.codes, u.scales, u.biases,
                                       true, kGroup, 4, "affine", stream);
        auto h = custom({gate, up}, {{m, kH}}, {mx::float32}, {m * kH, 1, 1},
                        {256, 1, 1}, {}, std::nullopt, false, stream);
        return std::vector<mx::array>{mx::quantized_matmul(
            h.at(0), d.codes, d.scales, d.biases, true, kGroup, 4, "affine", stream)};
      };
      auto compiled = mx::compile(std::move(body), false);
      std::optional<mx::array> retained;
      std::vector<double> retained_expected;
      for (int run = 0; run < 3; ++run) {
        const auto input = values(m * kK, 3 + run * 5);
        const auto expected = oracle(input, m, g, u, d);
        std::vector<mx::array> output;
        {
          mx::array x(input.begin(), {m, kK});
          output = compiled({x});
          mx::async_eval(output);
        }  // The caller's temporary input array is gone before observation.
        mx::synchronize(stream);
        const auto worst = check(output.at(0), expected);
        if (traces != 1) throw std::runtime_error("same-shape body retraced");
        if (!retained) { retained = output.at(0); retained_expected = expected; }
        check(*retained, retained_expected);
        std::printf("M=%d run=%d traces=%d max_abs_error=%.12g retained=PASS result=PASS\n",
                    m, run, traces, worst);
      }
    }
    bool rejected = false;
    try {
      auto bad = mx::zeros({1, kK - 1}, mx::float32, stream);
      auto unused = mx::quantized_matmul(bad, g.codes, g.scales, g.biases,
                                         true, kGroup, 4, "affine", stream);
      (void)unused;
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid K was not rejected before eval");
    mx::synchronize(stream);
    std::puts("result=PASS invalid_K=REJECTED_BEFORE_EVAL intermediate_host_waits=0 timing=NOT_RUN");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "result=FAIL error=%s\n", error.what());
    return 1;
  }
}
