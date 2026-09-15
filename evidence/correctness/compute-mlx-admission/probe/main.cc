// Research fixture: public MLX C++ API only, not a shipping HYLE provider.
#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

namespace mx = mlx::core;
namespace {
constexpr int kRows = 4;
constexpr int kWidth = 1024;
constexpr int kHidden = 256;
constexpr double kTolerance = 1e-4;
using Function = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

std::vector<float> weights(int rows, int columns, int seed) {
  std::vector<float> values(static_cast<std::size_t>(rows) * columns);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const int raw = static_cast<int>((i * 17 + (i / columns) * 13 + seed) % 31) - 15;
    values[i] = static_cast<float>(raw) / 256.0f;
  }
  return values;
}

std::vector<double> oracle(const float* x, const std::vector<float>& gate,
                           const std::vector<float>& up,
                           const std::vector<float>& down) {
  std::vector<double> hidden(kRows * kHidden);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kHidden; ++column) {
      double g = 0;
      double u = 0;
      for (int k = 0; k < kWidth; ++k) {
        g += static_cast<double>(x[row * kWidth + k]) * gate[k * kHidden + column];
        u += static_cast<double>(x[row * kWidth + k]) * up[k * kHidden + column];
      }
      hidden[row * kHidden + column] = (g / (1.0 + std::exp(-g))) * u;
    }
  }
  std::vector<double> result(kRows * kWidth);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      for (int k = 0; k < kHidden; ++k) {
        result[row * kWidth + column] +=
            hidden[row * kHidden + k] * down[k * kWidth + column];
      }
    }
  }
  return result;
}

void run_phase(const Function& compiled, const std::vector<float>& gate,
               const std::vector<float>& up, const std::vector<float>& down,
               int phase, std::atomic<int>& released, std::atomic<int>& traced) {
  const long page = sysconf(_SC_PAGESIZE);
  if (page <= 0) throw std::runtime_error("page size unavailable");
  constexpr std::size_t bytes = kRows * kWidth * sizeof(float);
  if (bytes % static_cast<std::size_t>(page) != 0) {
    throw std::runtime_error("fixture input is not a whole number of host pages");
  }
  for (int run = 0; run < 3; ++run) {
    void* storage = nullptr;
    if (posix_memalign(&storage, static_cast<std::size_t>(page), bytes) != 0) {
      throw std::bad_alloc();
    }
    auto* input = static_cast<float*>(storage);
    for (int i = 0; i < kRows * kWidth; ++i) {
      input[i] = static_cast<float>((i * 11 + run * 5 + phase * 7) % 47 - 23) / 64.0f;
    }
    const auto expected = oracle(input, gate, up, down);
    const int before = released.load();
    {
      // The public constructor can copy; observe rather than assume borrowing.
      mx::array x(storage, {kRows, kWidth}, mx::float32, [&released](void* p) {
        std::free(p);
        ++released;
      });
      const bool borrowed = x.data<float>() == input;
      auto output = compiled({x});
      if (output.size() != 1) throw std::runtime_error("wrong result arity");
      mx::async_eval(output);
      // array.wait's already-signaled fast path need not check an event error.
      // Use the public stream's explicit terminal error check for this probe.
      output[0].wait();
      mx::synchronize(mx::default_stream(mx::Device::gpu));
      double max_error = 0;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const double actual = output[0].data<float>()[i];
        const double error = std::abs(actual - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual) ||
            error > kTolerance * std::max(1.0, std::abs(expected[i]))) {
          throw std::runtime_error("FFN oracle mismatch");
        }
      }
      if (borrowed && released.load() != before) {
        throw std::runtime_error("borrowed input released while caller array lives");
      }
      std::cout << "phase=" << phase << " run=" << run
                << " borrowed=" << borrowed << " traces=" << traced.load()
                << " max_abs_error=" << max_error << " result=PASS\n";
    }
  }
}
}  // namespace

int main() {
  try {
    std::cout << "MLX v0.32.2 official C++ runtime; FFN F32 M=" << kRows
              << " K=" << kWidth << " H=" << kHidden
              << " tolerance=" << kTolerance << "\n";
    const auto gate = weights(kWidth, kHidden, 1);
    const auto up = weights(kWidth, kHidden, 7);
    const auto down = weights(kHidden, kWidth, 19);
    std::atomic<int> traced{0};
    std::atomic<int> released{0};
    {
      mx::array gate_array(gate.begin(), {kWidth, kHidden}, mx::float32);
      mx::array up_array(up.begin(), {kWidth, kHidden}, mx::float32);
      mx::array down_array(down.begin(), {kHidden, kWidth}, mx::float32);
      Function body = [&, gate_array, up_array, down_array](const auto& args) {
        ++traced;
        const auto stream = mx::default_stream(mx::Device::gpu);
        auto g = mx::matmul(args.at(0), gate_array, stream);
        auto u = mx::matmul(args.at(0), up_array, stream);
        auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g, stream), stream), u, stream);
        return std::vector<mx::array>{mx::matmul(h, down_array, stream)};
      };
      auto compiled = mx::compile(std::move(body), false);
      run_phase(compiled, gate, up, down, 0, released, traced);
      if (traced.load() != 1) throw std::runtime_error("same-thread cache expectation failed");
      std::exception_ptr worker_error;
      std::thread worker([&] {
        try {
          run_phase(compiled, gate, up, down, 1, released, traced);
          mx::clear_streams();
        } catch (...) {
          worker_error = std::current_exception();
        }
      });
      worker.join();
      if (worker_error) std::rethrow_exception(worker_error);
      if (traced.load() != 2) throw std::runtime_error("cross-thread retrace expectation failed");
    }
    mx::clear_streams();
    if (released.load() != 6) throw std::runtime_error("input owners were retained after drain");
    std::cout << "traces=" << traced.load() << " input_releases=" << released.load()
              << " thread_local_compile_cache=CONFIRMED result=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "result=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
