#pragma once

#include "cpu_program_abi.h"

#include "hyle/error/result.h"
#include "hyle/platform/cpu.h"

namespace hyle::compute::ggml_runtime {

inline constexpr bool
cpu_avx2_program_eligible(const platform::CpuFeatureSet &facts) noexcept {
  using platform::CpuFeature;
  constexpr platform::CpuFeatureSet required{
      CpuFeature::kX86Avx, CpuFeature::kX86Avx2, CpuFeature::kX86Fma,
      CpuFeature::kX86F16c, CpuFeature::kX86Bmi2};
  return facts.has_all(required);
}
Result<const CpuProgram *> select_cpu_program() noexcept;

} // namespace hyle::compute::ggml_runtime
