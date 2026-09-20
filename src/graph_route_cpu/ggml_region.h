#pragma once

#include "hyle/compute/providers/ggml_cpu.h"

namespace hyle::graph::cpu::internal {

compute::Provider ggml_regions(compute::ggml_cpu::Options) noexcept;

} // namespace hyle::graph::cpu::internal
