#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/tensor_desc.h"
#include "hyle/compute/visibility.h"
#include "shape_program.h"

namespace hyle::compute::detail {

enum class ProviderTensorLayout : std::uint8_t {
  // Preserve fixed layouts; reconstruct changing extents as dense storage.
  kDerived = 0,
  // Plain, non-negative strides supplied at cold binding, including zero
  // strides. The provider must support the entire family within maximum_bytes,
  // not merely the envelope's strides. This does not promise injectivity.
  kBoundedStrided,
};

// One entry per axis of the corresponding envelope TensorDesc. Fixed axes
// retain their descriptor extent; other axes name an input or derived value.
struct ProviderTensorAxes final {
  constexpr ProviderTensorAxes() noexcept { values.fill(kFixedShapeAxis); }
  std::array<std::uint32_t, kMaxTensorRank> values{};
  ProviderTensorLayout layout = ProviderTensorLayout::kDerived;
  // Reachable storage span, not logical element bytes; zero is a real bound.
  // Must be zero for kDerived, which uses its existing descriptor contract.
  std::size_t maximum_bytes = 0;
};

// Borrowed for one cold call. Empty spans are the exact fast path. Otherwise
// every input/output has an axis map, and the descriptors supplied alongside
// this view describe the envelope, not necessarily an admitted point. Dynamic
// layouts are dense or blocked-dense by default; a bounded strided family is
// explicit. Exact tensors retain arbitrary legal strides/opaque formats.
struct ProviderShapeDomain final {
  std::span<const ShapeDimension> dimensions{};
  std::span<const ProviderTensorAxes> inputs{};
  std::span<const ProviderTensorAxes> outputs{};
  std::span<const ShapeGuard> guards{};

  constexpr ShapeProgramView program() const noexcept {
    return {dimensions, guards};
  }

  constexpr bool empty() const noexcept {
    return program().empty() && inputs.empty() && outputs.empty();
  }
  constexpr bool exact() const noexcept {
    for (const auto &map : inputs)
      if (map.layout != ProviderTensorLayout::kDerived)
        return false;
    for (const auto &map : outputs)
      if (map.layout != ProviderTensorLayout::kDerived)
        return false;
    return program().exact();
  }
};

// Metadata-only checks, allocation-free on success; error diagnostics may
// allocate through Status. An empty domain leaves exact TensorDesc validation
// to the operation schema. The schema/provider must separately prove operation
// semantics for the entire domain; checking its endpoints is not such a proof.
HYLE_COMPUTE_API Status validate_shape_domain(
    ProviderShapeDomain domain, std::span<const TensorDesc> upper_inputs,
    std::span<const TensorDesc> upper_outputs) noexcept;

// Reconstruct concrete dense descriptors before work acceptance, without
// inference, provider search, JIT or success-path allocation. Caller supplies
// scratch for every dimension, with input slots filled and derived slots
// ignored; on failure discard values/destinations (which may be partly set).
// Each destination may be identical to its paired upper span; all other
// overlap between destinations and source/domain storage is prohibited.
// Bounded-strided families require supplied descriptors through the checked
// binding below; this function does not guess their concrete strides.
HYLE_COMPUTE_API Status bind_shape_domain(
    ProviderShapeDomain domain, std::span<const TensorDesc> upper_inputs,
    std::span<const TensorDesc> upper_outputs, std::span<std::int64_t> values,
    std::span<TensorDesc> inputs, std::span<TensorDesc> outputs) noexcept;

// Validate descriptors inferred by the caller's cold metadata program. The
// normal derived layout must still equal the reconstructed descriptor exactly.
// Bounded-strided families check shape/type/format/rank and reachable bytes;
// operation relationships, aliases and provider eligibility remain separate.
// No payload access or success-path allocation; values may change on failure.
HYLE_COMPUTE_API Status validate_shape_binding(
    ProviderShapeDomain domain, std::span<const TensorDesc> upper_inputs,
    std::span<const TensorDesc> upper_outputs, std::span<std::int64_t> values,
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs) noexcept;

enum class ProviderShapeCoverage : std::uint8_t {
  kExactPoint = 0,
  // This commits the provider to every admitted point and all equality/layout
  // constraints in the request, including schema semantics, resource bounds,
  // alias guarantees and a no-JIT per-Invocation shape binding route.
  kEntireDomain,
};

} // namespace hyle::compute::detail
