#pragma once

#include "hyle/graph/executable.h"
#include "hyle/graph/detail/region_adapter.h"
#include "hyle/container/small_vector.h"

namespace hyle::graph::detail {

// Shared cold resource-footprint admission, including cross-fragment aliases.
using EffectBindings = container::SmallVector<EffectResourceBinding, 0, mem::AllocatorRef>;
HYLE_GRAPH_API Status collect_effect_bindings(
    const Graph &, std::span<const EffectResourceBinding>,
    std::span<const ConstantBinding>, mem::AllocatorRef, EffectBindings &) noexcept;
HYLE_GRAPH_API Status validate_effect_tensor_alias(
    std::span<const EffectResourceBinding>, mem::ConstBufferView) noexcept;

} // namespace hyle::graph::detail
