#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"

namespace hyle::compute::detail {

struct ProviderShapeRange final {
  std::int64_t minimum = 1;
  std::int64_t maximum = 1;
  std::int64_t multiple_of = 1;

  constexpr bool valid() const noexcept {
    return minimum >= 0 && maximum >= minimum && multiple_of > 0 &&
           maximum - maximum % multiple_of >= minimum;
  }
  // Requires valid(); the admitted upper value proves this addition fits.
  constexpr std::int64_t first() const noexcept {
    const auto remainder = minimum % multiple_of;
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

// An ordinal names a previous program value, never a Graph-owned identity.
struct ShapeOperand final {
  std::uint32_t ordinal = kFixedShapeAxis;
  std::int64_t constant = 0;

  static constexpr ShapeOperand fixed(std::int64_t value) noexcept {
    return {kFixedShapeAxis, value};
  }
  static constexpr ShapeOperand value(std::uint32_t ordinal) noexcept {
    return {ordinal, 0};
  }
  constexpr bool is_fixed() const noexcept {
    return ordinal == kFixedShapeAxis;
  }
  friend constexpr bool operator==(ShapeOperand,
                                   ShapeOperand) noexcept = default;
};

enum class ShapeDimensionKind : std::uint8_t {
  kInput = 0,
  kAdd,
  kMultiply,
  kExactDivide,
  kBroadcast,
};

struct ShapeDimension final {
  ProviderShapeRange bounds{};
  ShapeDimensionKind kind = ShapeDimensionKind::kInput;
  ShapeOperand left{};
  ShapeOperand right{};
};

enum class ShapeGuardKind : std::uint8_t { kEqual = 0, kLessEqual };

struct ShapeGuard final {
  ShapeGuardKind kind = ShapeGuardKind::kEqual;
  ShapeOperand left{};
  ShapeOperand right{};
};

struct ShapeProgramView final {
  std::span<const ShapeDimension> dimensions{};
  std::span<const ShapeGuard> guards{};

  constexpr bool empty() const noexcept {
    return dimensions.empty() && guards.empty();
  }
  constexpr bool exact() const noexcept {
    for (const auto &dimension : dimensions) {
      if (!dimension.bounds.exact())
        return false;
    }
    return true;
  }
  constexpr std::size_t input_count() const noexcept {
    std::size_t count = 0;
    for (const auto &dimension : dimensions)
      count += dimension.kind == ShapeDimensionKind::kInput;
    return count;
  }
};

// Conservative envelope, not a satisfiability proof. Divisibility/broadcast
// conditions and explicit guards can exclude points inside the envelope.
HYLE_COMPUTE_API Result<ProviderShapeRange>
infer_shape_range(ShapeDimensionKind kind, ProviderShapeRange left,
                  ProviderShapeRange right) noexcept;

// Bounds on derived dimensions must equal infer_shape_range(), and operands
// may only refer backwards. O(D+G), no allocation on success; Status may
// allocate.
HYLE_COMPUTE_API Status
validate_shape_program(ShapeProgramView program) noexcept;

// Caller fills input slots; derived slots are overwritten. Exactly D values
// are required. Program storage must not overlap scratch. On failure discard
// all scratch; no provider work or tensor payload is accessed here.
HYLE_COMPUTE_API Status evaluate_shape_program(
    ShapeProgramView program, std::span<std::int64_t> values) noexcept;

} // namespace hyle::compute::detail
