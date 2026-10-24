#pragma once

#include "hyle/compute/provider.h"

namespace hyle::graph::cpu::internal {
compute::Provider ct_packed_regions(compute::Provider) noexcept;
} // namespace hyle::graph::cpu::internal
