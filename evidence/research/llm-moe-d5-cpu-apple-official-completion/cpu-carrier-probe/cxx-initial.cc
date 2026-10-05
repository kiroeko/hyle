#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cstdio>
#include <fstream>
#include <vector>

// Qualification only: native ATen calls, no HYLE product or Python runtime.
// The rceil tensor sequence corresponds to the pinned TorchAO original
// _f32_to_e8m0_rceil and _e8m0_scale_to_reciprocal_fp32 bodies.
at::Tensor rceil(const at::Tensor &value) {
  auto bits = value.to(at::kFloat).view(at::kInt);
  auto exponent = at::bitwise_and(at::bitwise_right_shift(bits, 23), 255);
  auto mantissa = at::bitwise_and(bits, 0x7fffff);
  auto up = at::where(exponent == 0, mantissa > 0x400000, mantissa != 0);
  return at::where(at::isfinite(value), exponent + up.to(at::kInt), 255).to(at::kByte);
}
at::Tensor reciprocal_scale(const at::Tensor &exponent) {
  auto biased = (254 - exponent.to(at::kInt)).to(at::kByte);
  auto bits = at::bitwise_left_shift(biased.to(at::kInt), 23);
  bits = at::where(biased == 0, 0x00400000, bits);
  bits = at::where(biased == 255, 0x7f800001, bits);
  return bits.view(at::kFloat);
}
std::vector<float> read(const char *name) {
  std::ifstream file(name, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(name);
  std::vector<float> result(static_cast<std::size_t>(file.tellg()) / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(float));
  return result;
}
int main() {
  try {
    at::set_num_threads(2); // qualification process only, not a context setting
    at::NoGradGuard guard;
    auto data = read("all_finite_bf16.bin");
    auto x = at::from_blob(data.data(), {static_cast<long long>(data.size()/128),128}, at::kFloat);
    for (int power : {0, 1}) {
      auto maximum = x.abs().amax(-1, true).clamp_min(power ? 1e-4 : 1e-10);
      auto scale = power ? reciprocal_scale(rceil(maximum * (1.0/448.0))).reciprocal()
                         : maximum / 448.0;
      auto codes = (x / scale).clamp(-448,448).to(at::kFloat8_e4m3fn);
      std::ofstream code_file(power ? "cxx-power-codes.bin" : "cxx-ordinary-codes.bin", std::ios::binary);
      code_file.write(reinterpret_cast<char *>(codes.const_data_ptr()), codes.numel());
      std::ofstream scale_file(power ? "cxx-power-scales.bin" : "cxx-ordinary-scales.bin", std::ios::binary);
      scale_file.write(reinterpret_cast<char *>(scale.const_data_ptr()), scale.numel()*sizeof(float));
      std::printf("CPU FP8 mode=%d elements=%lld\n",power, static_cast<long long>(codes.numel()));
    }
    auto a = at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({2,2});
    auto b = at::tensor({2.0f, -1.0f, 1.0f, 3.0f}).reshape({2,2});
    auto product = at::matmul(a,b);
    auto expected = at::tensor({4.0f, 5.0f, 10.0f, 9.0f}).reshape({2,2});
    if (!at::equal(product, expected)) return 2;
    std::puts("Native CPU matmul PASS; no Python or CUDA linkage");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
