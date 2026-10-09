#pragma once

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {
Status lower_mlx(ProgramState &, const PlanStep &) noexcept;
Status ensure_iree_device(ProgramState &) noexcept;
Status lower_native_cast(ProgramState &, std::uint32_t input,
                          std::uint32_t output) noexcept;
} // namespace hyle::compute::vulkan::detail
