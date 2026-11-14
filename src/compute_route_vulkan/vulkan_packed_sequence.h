#pragma once

#include "hyle/error/status.h"

namespace hyle::compute::vulkan::detail {
struct ProgramState;
struct PlanStep;
Status lower_packed_conv(ProgramState &, const PlanStep &) noexcept;
Status lower_packed_gdn(ProgramState &, const PlanStep &) noexcept;
} // namespace hyle::compute::vulkan::detail
