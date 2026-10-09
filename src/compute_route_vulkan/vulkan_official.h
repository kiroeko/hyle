#pragma once

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {
bool official_float_operation(const ProgramState &, const PlanStep &) noexcept;
Status lower_official_float(ProgramState &, const PlanStep &) noexcept;
} // namespace hyle::compute::vulkan::detail
