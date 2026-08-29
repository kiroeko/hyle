#include "hyle/simd/math.h"
#include "hyle/simd/runtime.h"
#include "hyle/platform/cpu.h"

#include <array>
#include <cmath>
#include <cstddef>

int main() {
  std::array<float, 9> input{-4.0f, -1.0f, -0.0f, 0.0f, 0.25f,
                             1.0f,  2.0f,  4.0f, 8.0f};
  std::array<float, input.size()> output{};

  hyle::simd::exp_in_range_f32_batch(input.data(), output.data(),
                                      input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float expected = std::exp(input[i]);
    if (std::abs(output[i] - expected) >
        std::abs(expected) * 8e-7f + 1e-6f)
      return 1;
  }

  // The package must stage and load the runtime-selected implementation, not
  // merely make the public header compile.
  const char* active = hyle::platform::isa_name(
      hyle::simd::selected_target(hyle::simd::KernelFamily::kMath).isa);
  return active == nullptr || active[0] == '\0' ? 2 : 0;
}
