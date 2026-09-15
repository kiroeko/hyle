#pragma once

#include <span>

#include "graph_state.h"

namespace hyle::graph::detail {

// Prove View relationships over the admitted program, then form conservative
// descriptor envelopes. These descriptors need not be an admitted point.
Status prove_view_domains(const GraphState &graph,
                          std::span<compute::TensorDesc> descriptions,
                          mem::AllocatorRef allocator) noexcept;

// Only reshape promises canonical dense strides for every concrete shape.
// Other inferred View layouts use the explicit bounded-strided Provider seam.
bool has_strided_view_layout(const GraphState &graph,
                             std::size_t value) noexcept;

} // namespace hyle::graph::detail
