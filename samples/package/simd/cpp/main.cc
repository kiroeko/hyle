#include "hyle/platform/cpu.h"
#include "hyle/simd/bitpack.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/math.h"
#include "hyle/simd/random.h"
#include "hyle/simd/runtime.h"
#include "hyle/simd/scan.h"
#include "hyle/simd/search.h"
#include "hyle/simd/sort.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

float hyle_package_expert_sum(const float*, std::size_t) noexcept;
std::size_t hyle_package_xsimd_width() noexcept;

using HylePackageBiasReluFn = void (*)(float*, std::size_t, float) noexcept;
void hyle_package_bias_relu_scalar(float*, std::size_t, float) noexcept;
#if HYLE_ARCH_X64
void hyle_package_bias_relu_sse4(float*, std::size_t, float) noexcept;
void hyle_package_bias_relu_avx2(float*, std::size_t, float) noexcept;
void hyle_package_bias_relu_avx512(float*, std::size_t, float) noexcept;
#elif HYLE_ARCH_ARM64
void hyle_package_bias_relu_neon(float*, std::size_t, float) noexcept;
void hyle_package_bias_relu_sve(float*, std::size_t, float) noexcept;
#elif HYLE_ARCH_RISCV64
void hyle_package_bias_relu_rvv(float*, std::size_t, float) noexcept;
#endif

int main() {
  std::array<float, 9> input{-4.0f, -1.0f, -0.0f, 0.0f, 0.25f,
                             1.0f,  2.0f,  4.0f,  8.0f};
  std::array<float, input.size()> output{};

  hyle::simd::exp_in_range_f32_batch(input.data(), output.data(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float expected = std::exp(input[i]);
    if (std::abs(output[i] - expected) > std::abs(expected) * 8e-7f + 1e-6f)
      return 1;
  }

  hyle::simd::relu_f32_batch(input.data(), output.data(), input.size());
  if (output[0] != 0.0f || std::signbit(output[2]) ||
      output.back() != input.back())
    return 2;

  hyle::simd::softmax_f32(input.data(), output.data(), input.size());
  float probability_sum = 0.0f;
  for (float probability : output) probability_sum += probability;
  if (std::abs(probability_sum - 1.0f) > 2e-6f) return 3;

  std::array<float, input.size()> gate_output{};
  hyle::simd::swiglu_f32_batch(input.data(), output.data(), gate_output.data(),
                               input.size());
  hyle::simd::rms_norm_f32(gate_output.data(), gate_output.data(),
                           gate_output.size(), 1e-5f);
  for (float value : gate_output) {
    if (!std::isfinite(value)) return 16;
  }

  const std::array<std::uint32_t, 7> scan_input{3, 1, 4, 1, 5, 9, 2};
  std::array<std::uint32_t, scan_input.size()> scan_output{};
  hyle::simd::inclusive_scan(scan_input.data(), scan_output.data(),
                             scan_input.size());
  if (scan_output != std::array<std::uint32_t, 7>{3, 4, 8, 9, 14, 23, 25})
    return 4;
  hyle::simd::exclusive_scan(scan_output.data(), scan_output.size(), 7u);
  if (scan_output != std::array<std::uint32_t, 7>{7, 10, 14, 22, 31, 45, 68})
    return 5;

  const std::array<float, 8> search_input{1.0f, 9.0f, -2.0f, 9.0f,
                                          4.0f, 9.0f, 0.0f,  3.0f};
  if (hyle::simd::count_equal(search_input.data(), search_input.size(), 9.0f) !=
          3 ||
      hyle::simd::index_of_max(search_input.data(), search_input.size()) != 1)
    return 6;

  constexpr unsigned kBits = 11;
  std::array<std::uint16_t, 129> raw{};
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<std::uint16_t>((i * 37 + 11) & 0x7ffu);
  std::array<std::uint16_t,
             hyle::simd::packed_size<std::uint16_t>(raw.size(), kBits)>
      packed{};
  std::array<std::uint16_t, raw.size()> decoded{};
  hyle::simd::pack(std::span<const std::uint16_t>{raw},
                   std::span<std::uint16_t>{packed}, kBits);
  hyle::simd::unpack(std::span<const std::uint16_t>{packed},
                     std::span<std::uint16_t>{decoded}, kBits);
  if (hyle::simd::kBitpackFormatVersion != 1 || decoded != raw) return 7;

  hyle::simd::RandomState random{12345};
  hyle::simd::RandomState copy = random;
  std::array<float, 33> random_a{};
  std::array<float, random_a.size()> random_b{};
  hyle::simd::random_fill(random, std::span<float>{random_a});
  hyle::simd::random_fill(copy, std::span<float>{random_b});
  if (random_a != random_b) return 8;

  random.seed(12345);
  auto worker = random.split();
  hyle::simd::RandomState expected{12345};
  worker.advance(17);
  std::array<float, 17> discarded{};
  hyle::simd::random_fill(expected, std::span<float>{discarded});
  hyle::simd::random_fill(worker, std::span<float>{random_a});
  hyle::simd::random_fill(expected, std::span<float>{random_b});
  if (random_a != random_b) return 9;

  std::array<float, 6> top_keys{3.0f, -2.0f, 9.0f, 1.0f, 7.0f, 4.0f};
  hyle::simd::partial_sort(top_keys.data(), top_keys.size(), 2,
                           hyle::simd::SortOrder::kDescending);
  if (top_keys[0] != 9.0f || top_keys[1] != 7.0f) return 10;

  std::array<std::uint32_t, 7> stable_keys{4, 1, 4, 2, 1, 3, 1};
  std::array<std::uint32_t, stable_keys.size()> radix_scratch{};
  hyle::simd::radix_sort(stable_keys.data(), stable_keys.size(), radix_scratch);
  if (stable_keys != std::array<std::uint32_t, 7>{1, 1, 1, 2, 3, 4, 4})
    return 11;

  const std::array<float, 4> scores{0.5f, 2.0f, -1.0f, 2.0f};
  std::array<std::uint32_t, scores.size()> indices{};
  std::array<std::uint32_t, scores.size()> argsort_scratch{};
  hyle::simd::argsort(scores.data(), indices.data(), scores.size(),
                      argsort_scratch, hyle::simd::SortOrder::kDescending);
  if (indices != std::array<std::uint32_t, 4>{1, 3, 0, 2}) return 12;

  std::array<hyle::simd::ScoredIndexF32, 2> selected{};
  hyle::simd::top_k_f32(scores.data(), scores.size(), selected.data(),
                        selected.size());
  if (selected[0].index != 1 || selected[1].index != 3 ||
      selected[0].score != 2.0f || selected[1].score != 2.0f)
    return 17;

  // The package must stage and load the runtime-selected implementation, not
  // merely make the public header compile.
  const char* active = hyle::simd::kernel_target_name(
      hyle::simd::selected_target(hyle::simd::KernelFamily::kMath).target);
  const char* ordering = hyle::simd::kernel_target_name(
      hyle::simd::selected_target(hyle::simd::KernelFamily::kSort).target);
  if (active == nullptr || active[0] == '\0' || ordering == nullptr ||
      ordering[0] == '\0')
    return 13;

  const std::array<float, 5> expert_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  if (hyle_package_expert_sum(expert_values.data(), expert_values.size()) !=
          15.0f ||
      hyle_package_xsimd_width() == 0)
    return 14;

  std::array<float, 4> fat_values{-2.0f, -0.5f, 1.0f, 4.0f};
#if HYLE_ARCH_X64
  static const HylePackageBiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_package_bias_relu_scalar, &hyle_package_bias_relu_sse4,
      &hyle_package_bias_relu_avx2, &hyle_package_bias_relu_avx512);
#elif HYLE_ARCH_ARM64
  static const HylePackageBiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_package_bias_relu_scalar, &hyle_package_bias_relu_neon,
      &hyle_package_bias_relu_sve);
#elif HYLE_ARCH_RISCV64
  static const HylePackageBiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_package_bias_relu_scalar, &hyle_package_bias_relu_rvv);
#else
  static const HylePackageBiasReluFn bias_relu =
      HYLE_SIMD_SELECT_FAT_KERNEL(&hyle_package_bias_relu_scalar);
#endif
  bias_relu(fat_values.data(), fat_values.size(), 1.0f);
  return fat_values != std::array<float, 4>{0.0f, 0.5f, 2.0f, 5.0f} ? 15 : 0;
}
