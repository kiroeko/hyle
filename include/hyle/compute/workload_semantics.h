#pragma once

#include "hyle/compute/workload.h"

namespace hyle::compute::workload_semantics {

// A target-model text forward with ordinary state publication. Proposal,
// verification and repair graphs use their own semantics.
inline constexpr RegionSemanticId kTargetTextForward{
    .namespace_id = 0x68796c652e746578ULL, // hyle.tex
    .region_id = 1,
    .version = 1};

} // namespace hyle::compute::workload_semantics
