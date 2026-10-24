#pragma once

#include "hyle/compute/provider.h"

namespace hyle::graph::cuda::internal {
compute::Provider fp8_shared_prologue_provider() noexcept;
}
