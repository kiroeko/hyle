#pragma once

#include <optional>

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {
std::optional<std::uint32_t> tensor_storage_type(DataType type) noexcept;
Status lower_tensor(ProgramState &program, const PlanStep &step);
Status lower_index_check(ProgramState &program, std::uint32_t input,
                         std::uint32_t output, std::uint64_t extent,
                         bool checked);
Status lower_checked_copy(ProgramState &program, std::uint32_t input,
                          std::uint32_t output);
Status lower_integer_scan(ProgramState &program, std::uint32_t input,
                          std::uint32_t output, bool exclusive);
Status lower_scatter_dispatch(ProgramState &program, std::uint32_t mode,
                              std::span<const std::uint32_t> inputs,
                              std::uint32_t output, std::uint32_t count,
                              std::uint32_t extent, std::uint32_t inner);
} // namespace hyle::compute::vulkan::detail
