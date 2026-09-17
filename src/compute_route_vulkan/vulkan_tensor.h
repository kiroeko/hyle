#pragma once

#include <optional>

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {
std::optional<std::uint32_t> tensor_storage_type(DataType type) noexcept;
Status lower_tensor(ProgramState &program, const PlanStep &step);
} // namespace hyle::compute::vulkan::detail
