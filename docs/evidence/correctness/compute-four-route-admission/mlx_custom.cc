// Research only: a native Metal insertion inside an MLX-owned compiled region.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
constexpr int kM = 4, kK = 1024, kH = 256;
using Function = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

std::vector<float> values(int count, int seed) {
  std::vector<float> result(count);
  for (int i = 0; i < count; ++i)
    result[i] = static_cast<float>((i * 17 + seed) % 31 - 15) / 256.0f;
  return result;
}

std::vector<double> oracle(const std::vector<float>& x,
                           const std::vector<float>& g,
                           const std::vector<float>& u,
                           const std::vector<float>& d) {
  std::vector<double> h(kM * kH), y(kM * kK);
  for (int m = 0; m < kM; ++m) {
    for (int j = 0; j < kH; ++j) {
      double a = 0, b = 0;
      for (int k = 0; k < kK; ++k) {
        a += static_cast<double>(x[m * kK + k]) * g[k * kH + j];
        b += static_cast<double>(x[m * kK + k]) * u[k * kH + j];
      }
      h[m * kH + j] = a / (1.0 + std::exp(-a)) * b;
    }
    for (int j = 0; j < kK; ++j)
      for (int k = 0; k < kH; ++k)
        y[m * kK + j] += h[m * kH + k] * d[k * kK + j];
  }
  return y;
}

int main() {
  try {
    const auto stream = mx::new_stream(mx::Device::gpu);
    mx::set_default_device(mx::Device::gpu);
    const auto g = values(kK * kH, 1), u = values(kK * kH, 7);
    const auto d = values(kH * kK, 19);
    mx::array ga(g.begin(), {kK, kH}), ua(u.begin(), {kK, kH});
    mx::array da(d.begin(), {kH, kK});
    auto custom = mx::fast::metal_kernel(
        "hyle_research_swiglu", {"gate", "up"}, {"out"},
        "uint i = thread_position_in_grid.x;\n"
        "out[i] = (gate[i] / (1.0f + exp(-gate[i]))) * up[i];\n");
    int traces = 0;
    Function body = [&, ga, ua, da, custom](const auto& args) {
      ++traces;
      auto gate = mx::matmul(args.at(0), ga, stream);
      auto up = mx::matmul(args.at(0), ua, stream);
      // No host extraction, separate native queue or wait at either boundary.
      auto h = custom({gate, up}, {{kM, kH}}, {mx::float32},
                      {kM * kH, 1, 1}, {256, 1, 1}, {}, std::nullopt,
                      false, stream);
      return std::vector<mx::array>{mx::matmul(h.at(0), da, stream)};
    };
    auto compiled = mx::compile(std::move(body), false);
    std::printf("MLX=0.32.2 route=GPU M=%d K=%d H=%d tolerance=1e-4\n", kM, kK, kH);
    for (int run = 0; run < 6; ++run) {
      const auto input = values(kM * kK, 3 + run * 5);
      const auto expected = oracle(input, g, u, d);
      mx::array x(input.begin(), {kM, kK});
      auto output = compiled({x});
      mx::async_eval(output);
      // One terminal observation; ready-only array.wait is not an error verdict.
      mx::synchronize(stream);
      double worst = 0;
      const float* actual = output.at(0).data<float>();
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]);
        worst = std::max(worst, error);
        if (!std::isfinite(actual[i]) ||
            error > 1e-4 * std::max(1.0, std::abs(expected[i])))
          throw std::runtime_error("custom Metal FFN oracle mismatch");
      }
      if (traces != 1) throw std::runtime_error("same-thread compiled body retraced");
      std::printf("run=%d traces=%d max_abs_error=%.12g result=PASS\n", run, traces, worst);
    }
    mx::synchronize(stream);
    std::puts("result=PASS native_insertion=inside_MLX explicit_intermediate_waits=0 timing=NOT_RUN");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "result=FAIL error=%s\n", error.what());
    return 1;
  }
}
