#include "shape_program.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace hyle::compute::detail {
namespace {

constexpr auto kMaximum = (std::numeric_limits<std::int64_t>::max)();

Result<std::int64_t> calculate(ShapeDimensionKind kind, std::int64_t left,
                               std::int64_t right) noexcept {
  switch (kind) {
  case ShapeDimensionKind::kAdd:
    if (left > kMaximum - right)
      return std::unexpected(Status::OutOfRange("shape sum overflows"));
    return left + right;
  case ShapeDimensionKind::kMultiply:
    if (right != 0 && left > kMaximum / right)
      return std::unexpected(Status::OutOfRange("shape product overflows"));
    return left * right;
  case ShapeDimensionKind::kExactDivide:
    if (right == 0 || left % right != 0) {
      return std::unexpected(Status::InvalidArgument(
          "shape quotient requires exact nonzero division"));
    }
    return left / right;
  case ShapeDimensionKind::kBroadcast:
    if (left == right || right == 1)
      return left;
    if (left == 1)
      return right;
    return std::unexpected(
        Status::InvalidArgument("shape broadcast extents are incompatible"));
  default:
    return std::unexpected(Status::InvalidArgument("invalid shape expression"));
  }
}

constexpr ProviderShapeRange fixed_range(std::int64_t value) noexcept {
  return {value, value, value == 0 ? 1 : value};
}

constexpr std::int64_t divisor(ProviderShapeRange range) noexcept {
  return range.exact() ? range.first() : range.multiple_of;
}

bool valid_operand(ShapeOperand operand, std::size_t limit) noexcept {
  return operand.is_fixed() ? operand.constant >= 0
                            : operand.constant == 0 && operand.ordinal < limit;
}

ProviderShapeRange operand_range(ShapeProgramView program,
                                 ShapeOperand operand) noexcept {
  return operand.is_fixed() ? fixed_range(operand.constant)
                            : program.dimensions[operand.ordinal].bounds;
}

std::int64_t operand_value(std::span<const std::int64_t> values,
                           ShapeOperand operand) noexcept {
  return operand.is_fixed() ? operand.constant : values[operand.ordinal];
}

bool guard_holds(ShapeGuardKind kind, std::int64_t left,
                 std::int64_t right) noexcept {
  return kind == ShapeGuardKind::kEqual ? left == right : left <= right;
}

} // namespace

Result<ProviderShapeRange>
infer_shape_range(ShapeDimensionKind kind, ProviderShapeRange left,
                  ProviderShapeRange right) noexcept {
  if (!left.valid() || !right.valid()) {
    return std::unexpected(
        Status::InvalidArgument("invalid shape operand bounds"));
  }
  if (left.exact() && right.exact()) {
    auto value = calculate(kind, left.first(), right.first());
    if (!value)
      return std::unexpected(std::move(value).error());
    return fixed_range(*value);
  }
  ProviderShapeRange result{};
  switch (kind) {
  case ShapeDimensionKind::kAdd:
  case ShapeDimensionKind::kMultiply: {
    auto upper = calculate(kind, left.last(), right.last());
    if (!upper)
      return std::unexpected(std::move(upper).error());
    // Nonnegative add/multiply are monotone; a representable upper bound
    // proves the lower arithmetic safe as well.
    result.minimum = kind == ShapeDimensionKind::kAdd
                         ? left.first() + right.first()
                         : left.first() * right.first();
    result.maximum = *upper;
    result.multiple_of = kind == ShapeDimensionKind::kAdd
                             ? std::gcd(divisor(left), divisor(right))
                             : divisor(left) * divisor(right);
    if (result.multiple_of == 0)
      result.multiple_of = 1;
    break;
  }
  case ShapeDimensionKind::kExactDivide: {
    if (right.last() == 0) {
      return std::unexpected(
          Status::InvalidArgument("shape divisor is always zero"));
    }
    const auto positive_minimum =
        right.first() == 0 ? right.multiple_of : right.first();
    result.minimum = left.first() / right.last();
    result.maximum = left.last() / positive_minimum;
    if (right.exact() && divisor(left) % right.first() == 0)
      result.multiple_of =
          (std::max)(std::int64_t{1}, divisor(left) / right.first());
    break;
  }
  case ShapeDimensionKind::kBroadcast:
    if (left.contains(1) && !right.contains(1))
      return right;
    if (right.contains(1) && !left.contains(1))
      return left;
    if (left.contains(1)) {
      result.minimum = (std::min)(left.first(), right.first());
      result.maximum = (std::max)(left.last(), right.last());
    } else {
      result.minimum = (std::max)(left.first(), right.first());
      result.maximum = (std::min)(left.last(), right.last());
    }
    result.multiple_of =
        (std::max)(std::int64_t{1}, std::gcd(divisor(left), divisor(right)));
    break;
  default:
    return std::unexpected(
        Status::InvalidArgument("invalid derived shape kind"));
  }
  if (!result.valid()) {
    return std::unexpected(
        Status::InvalidArgument("shape envelope has no legal extent"));
  }
  return result;
}

Status validate_shape_program(ShapeProgramView program) noexcept {
  if (program.dimensions.size() > kFixedShapeAxis) {
    return Status::ResourceExhausted("shape program exceeds ordinal capacity");
  }
  for (std::size_t index = 0; index < program.dimensions.size(); ++index) {
    const auto &dimension = program.dimensions[index];
    if (!dimension.bounds.valid())
      return Status::InvalidArgument("shape dimension bounds are invalid");
    if (dimension.kind == ShapeDimensionKind::kInput) {
      if (dimension.left != ShapeOperand{} || dimension.right != ShapeOperand{})
        return Status::InvalidArgument("input shape dimension has operands");
      continue;
    }
    if (!valid_operand(dimension.left, index) ||
        !valid_operand(dimension.right, index)) {
      return Status::InvalidArgument(
          "shape expression has invalid or forward operands");
    }
    auto inferred = infer_shape_range(dimension.kind,
                                      operand_range(program, dimension.left),
                                      operand_range(program, dimension.right));
    if (!inferred)
      return std::move(inferred).error();
    if (*inferred != dimension.bounds) {
      return Status::InvalidArgument(
          "derived shape bounds do not match its expression");
    }
  }
  for (const auto &guard : program.guards) {
    if ((guard.kind != ShapeGuardKind::kEqual &&
         guard.kind != ShapeGuardKind::kLessEqual) ||
        !valid_operand(guard.left, program.dimensions.size()) ||
        !valid_operand(guard.right, program.dimensions.size())) {
      return Status::InvalidArgument("shape guard is invalid");
    }
    const auto left = operand_range(program, guard.left);
    const auto right = operand_range(program, guard.right);
    if (left.exact() && right.exact() &&
        !guard_holds(guard.kind, left.first(), right.first())) {
      return Status::InvalidArgument("exact shape violates its guard");
    }
  }
  return Status::Ok();
}

Status evaluate_shape_program(ShapeProgramView program,
                              std::span<std::int64_t> values) noexcept {
  if (values.size() != program.dimensions.size())
    return Status::InvalidArgument(
        "shape program scratch size does not match dimensions");
  Status valid = validate_shape_program(program);
  if (!valid.ok())
    return valid;
  for (std::size_t index = 0; index < program.dimensions.size(); ++index) {
    const auto &dimension = program.dimensions[index];
    if (dimension.kind != ShapeDimensionKind::kInput) {
      auto value =
          calculate(dimension.kind, operand_value(values, dimension.left),
                    operand_value(values, dimension.right));
      if (!value)
        return std::move(value).error();
      values[index] = *value;
    }
    if (!dimension.bounds.contains(values[index]))
      return Status::OutOfRange("shape value is outside its dimension bounds");
  }
  for (const auto &guard : program.guards) {
    if (!guard_holds(guard.kind, operand_value(values, guard.left),
                     operand_value(values, guard.right))) {
      return Status::InvalidArgument("shape values violate a program guard");
    }
  }
  return Status::Ok();
}

} // namespace hyle::compute::detail
