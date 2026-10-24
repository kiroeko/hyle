#pragma once

#include <span>

#include "graph_state.h"

namespace hyle::graph::detail {

// Prove View relationships over the admitted program, then form conservative
// descriptor envelopes. These descriptors need not be an admitted point.
Status prove_view_domains(const GraphState &graph,
                          std::span<compute::TensorDesc> descriptions,
                          mem::AllocatorRef allocator) noexcept;

// The whole-domain prover admits reshape only from proven dense input. Exact
// reshapes may retain strides; they use concrete descriptors, not this seam.
// Other inferred View layouts use the explicit bounded-strided Provider seam.
bool has_strided_view_layout(const GraphState &graph,
                             std::size_t value) noexcept;

} // namespace hyle::graph::detail
