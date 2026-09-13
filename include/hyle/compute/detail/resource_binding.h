#pragma once

#include <cstddef>

#include "hyle/mem/resource.h"

namespace hyle::compute::detail {

// Provider-neutral cold resource constraint shared by Compute proposal and
// Graph region planning. `minimum_alignment` is the binding alignment the
// planner must request when it allocates storage; it is not compared with a
// resource's unconditional minimum allocation alignment. Exact preparation
// always revalidates the selected route with the owning backend/provider.
struct ResourceBindingRequirement final {
  mem::MemoryProperty required_properties = mem::MemoryProperty::kNone;
  mem::MemoryProperty forbidden_properties = mem::MemoryProperty::kNone;
  std::size_t minimum_alignment = 1;
  mem::MemoryResourceRef exact_resource{};
  mem::MemoryDomainId exact_domain{};
  bool requires_exact_domain = false;
};

} // namespace hyle::compute::detail
