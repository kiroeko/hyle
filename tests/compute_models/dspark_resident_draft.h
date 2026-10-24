#pragma once

#include "cpu_c4_model_branches.h"
#include "model_resident_fragment.h"

// DSpark's three actual router frontiers use the same cold Fragment slicing
// algorithm as other original-model consumers. This layer supplies only the
// model's typed boundary and stable region identity; execution stays ordinary
// Graph/Invocation with explicit caller-owned leases.
namespace dspark_resident_draft {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace official=d5_deepseek_official_fragments;
using Location=model_resident_fragment::Location;
using Phase=model_resident_fragment::Phase;
using ResidentDraft=model_resident_fragment::ResidentDraft<official::RoutedBoundary,3>;
inline ResidentDraft split(const sem::Fragment &full,const std::array<official::RoutedBoundary,3> &routers) {
  return model_resident_fragment::split(full,routers,0x0c040200u);
}
} // namespace dspark_resident_draft
