#pragma once
#include "hyle/compute/provider.h"
namespace hyle::bench::resident {
// Same original BatchProgram ownership/Queue/lifecycle, independent pinned
// ggml flash graph. This private comparator adds no product entry point.
compute::Provider cpu_ragged_reference_provider() noexcept;
}
