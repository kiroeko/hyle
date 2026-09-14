#pragma once

#include <cstddef>

#include "hyle/mem/resource.h"

namespace hyle::compute {

// Cold storage constraint for a proposed kernel. The planner requests
// minimum_alignment when allocating a binding; it is not compared with the
// resource's unconditional minimum allocation alignment. Exact preparation
// revalidates both this requirement and backend accessibility.
struct ResourceRequirement final {
  mem::MemoryProperty required_properties = mem::MemoryProperty::kNone;
  mem::MemoryProperty forbidden_properties = mem::MemoryProperty::kNone;
  std::size_t minimum_alignment = 1;
  mem::MemoryResourceRef exact_resource{};
  mem::MemoryDomainId exact_domain{};
  bool requires_exact_domain = false;
};

} // namespace hyle::compute
