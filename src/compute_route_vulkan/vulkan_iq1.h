#pragma once

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {
bool iq1_storage_description(const TensorDesc &) noexcept;
bool iq1_operation(const ProgramState &, const PlanStep &) noexcept;
Status lower_iq1(ProgramState &, const PlanStep &) noexcept;
} // namespace hyle::compute::vulkan::detail
