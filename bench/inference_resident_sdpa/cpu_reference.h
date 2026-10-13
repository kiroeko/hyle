#pragma once

#include "hyle/compute/provider.h"
#include "hyle/error/result.h"

namespace hyle::bench::resident {
// A private numerical comparator, carried by the existing public typed-kernel
// factory and PreparedOperation lifecycle. No product Provider SPI is added.
Result<compute::Provider> cpu_reference_provider() noexcept;
} // namespace hyle::bench::resident
