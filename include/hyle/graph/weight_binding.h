#pragma once

#include <cstdint>

#include "hyle/graph/graph.h"

namespace hyle::graph {

// One graph input and one bank in an immutable native WeightBatch. A native
// span bind resolves every bank and owner before atomically replacing the
// inputs, so weights, scales and an explicit ID remap come from one upload.
struct WeightBinding final {
  ValueId value{};
  std::uint64_t id = 0;
};

} // namespace hyle::graph
