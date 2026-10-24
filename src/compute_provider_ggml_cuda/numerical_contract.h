#pragma once

#include <cstdint>

namespace hyle::compute::ggml_cuda::internal {

// Private native UNARY metadata; zero retains upstream Unary/GLU behavior.
inline constexpr std::uint32_t kUnaryContractParam = 1;
inline constexpr std::int32_t kStableSiluContract = 0x48595349;

// Private CUDA RoPE carrier for the existing explicit-frequency contract.
// src0=[D,H,S,B], src1=I32[S,1|B], src2=F32[pairs,1|S].
// Rotary dimensions are 2*pairs; no duplicate narrow integer metadata.
// Layout uses upstream NORMAL(interleaved) or NEOX(first/second half).
// All remaining op_params bytes are zero, not upstream YaRN parameters.
inline constexpr std::uint32_t kExplicitRopeContractParam = 0;
inline constexpr std::int32_t kExplicitRopeContract = 0x48595250;
inline constexpr std::uint32_t kExplicitRopeLayoutParam = 2;

} // namespace hyle::compute::ggml_cuda::internal
