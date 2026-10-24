#pragma once

#include "hyle/compute/resource_budget.h"
#include "hyle/mem/allocator.h"

namespace hyle::graph {

// Cold resource-binding metadata for an Invocation. Provider programs and
// native workspace remain subject to the Executable's compile-time budgets.
struct ResourceOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  compute::ResourceBudget resource_budget{};
};

} // namespace hyle::graph
