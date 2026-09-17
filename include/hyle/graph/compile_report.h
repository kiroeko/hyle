#pragma once

#include <cstdint>

#include "hyle/compute/operation.h"

namespace hyle::graph {

// Cold semantic metadata copied from a logical node. This contains no
// Operation, custom payload, tensor, graph generation or native owner.
// Traits record the schema's precision and accumulation policy; they are not
// a claim about unobservable vendor intermediates or performance.
struct CompileNodeReport final {
  std::uint32_t ordinal = 0;
  compute::OperationKind kind = compute::OperationKind::kView;
  compute::OperationTraits traits{};
};

} // namespace hyle::graph
