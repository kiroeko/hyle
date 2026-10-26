#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::slots::detail {
// Compile-local assembly facts. The returned artifact copies these values;
// no adapter or provider lookup is kept on the execution path.
struct StoreAdapter final {
  mem::MemoryResourceRef resource;
  compute::DeviceClass device_class = compute::DeviceClass::kCpu;
};
// Same-build module assembly, not a public provider selection API.
HYLE_GRAPH_API const RegionProviderVTable *store_provider() noexcept;
} // namespace hyle::graph::slots::detail
