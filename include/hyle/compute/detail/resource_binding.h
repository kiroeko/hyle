#pragma once

#include "hyle/compute/resource_requirement.h"

namespace hyle::compute::detail {

// One representation and matcher for public custom kernels and private
// operation/region planning; this is not a second resource contract.
using ResourceBindingRequirement = ResourceRequirement;

} // namespace hyle::compute::detail
