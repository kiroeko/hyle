#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/tensor_desc.h"
#include "hyle/compute/visibility.h"

namespace hyle::compute::detail {

// Provider-neutral projection of a Graph symbol. No Graph identity or graph
// ownership crosses this seam. Endpoints need not be divisible by multiple_of;
// first()/last() name the first/last admitted point, including zero extents.
struct ProviderShapeRange final {
  std::int64_t minimum = 1;
  std::int64_t maximum = 1;
  std::int64_t multiple_of = 1;

  constexpr bool valid() const noexcept {
    return minimum >= 0 && maximum >= minimum && multiple_of > 0 &&
           maximum - maximum % multiple_of >= minimum;
  }
  // Requires valid(); the admitted upper point proves this addition fits.
  constexpr std::int64_t first() const noexcept {
    const std::int64_t remainder = minimum % multiple_of;
    return minimum + (remainder == 0 ? 0 : multiple_of - remainder);
  }
  constexpr std::int64_t last() const noexcept {
    return maximum - maximum % multiple_of;
  }
  constexpr bool contains(std::int64_t value) const noexcept {
    return valid() && value >= minimum && value <= maximum &&
           value % multiple_of == 0;
  }
  constexpr bool exact() const noexcept { return valid() && first() == last(); }
  friend constexpr bool operator==(ProviderShapeRange,
                                   ProviderShapeRange) noexcept = default;
};

inline constexpr std::uint32_t kFixedShapeAxis = UINT32_MAX;

// One entry per axis of the corresponding upper-point TensorDesc. Fixed axes
// retain their descriptor extent. Repeated indices impose equality, while two
// distinct indices remain independent even when their ranges are identical.
struct ProviderTensorAxes final {
  constexpr ProviderTensorAxes() noexcept { symbols.fill(kFixedShapeAxis); }
  std::array<std::uint32_t, kMaxTensorRank> symbols{};
};

// Borrowed for one cold call. Empty spans are the exact fast path. Otherwise
// every input/output has an axis map, and the descriptors supplied alongside
// this view describe the Cartesian upper point. Dynamic layouts are dense or
// blocked-dense; exact tensors retain arbitrary legal strides/opaque formats.
struct ProviderShapeDomain final {
  std::span<const ProviderShapeRange> symbols{};
  std::span<const ProviderTensorAxes> inputs{};
  std::span<const ProviderTensorAxes> outputs{};

  constexpr bool empty() const noexcept {
    return symbols.empty() && inputs.empty() && outputs.empty();
  }
  constexpr bool exact() const noexcept {
    for (ProviderShapeRange range : symbols) {
      if (!range.exact())
        return false;
    }
    return true;
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
// scratch; on failure discard the destinations (which may be partly set).
// Each destination may be identical to its paired upper span; all other
// overlap between destinations and source/domain storage is prohibited.
HYLE_COMPUTE_API Status bind_shape_domain(
    ProviderShapeDomain domain, std::span<const TensorDesc> upper_inputs,
    std::span<const TensorDesc> upper_outputs,
    std::span<const std::int64_t> point, std::span<TensorDesc> inputs,
    std::span<TensorDesc> outputs) noexcept;

enum class ProviderShapeCoverage : std::uint8_t {
  kExactPoint = 0,
  // This commits the provider to every admitted point and all equality/layout
  // constraints in the request, including schema semantics, resource bounds,
  // alias guarantees and a no-JIT per-Invocation shape binding route.
  kEntireDomain,
};

} // namespace hyle::compute::detail
