#include "hyle/compute/tensor.h"

#include <limits>
#include <utility>

#include "hyle/math/scalar.h"

namespace hyle::compute {
namespace {

Status validate_schema(DataType type,
                       std::span<const std::int64_t> dimensions) noexcept {
  if (data_type_size(type) == 0) {
    return Status::InvalidArgument("tensor data type is unknown");
  }
  if (dimensions.size() > kMaxTensorRank) {
    return Status::InvalidArgument("tensor rank exceeds kMaxTensorRank");
  }
  for (const std::int64_t dimension : dimensions) {
    if (dimension < 0) {
      return Status::InvalidArgument("tensor dimension is negative");
    }
  }
  return Status::Ok();
}

}  // namespace

std::size_t data_type_size(DataType type) noexcept {
  switch (type) {
    case DataType::kFloat16:
    case DataType::kBFloat16:
      return 2;
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kFloat64:
    case DataType::kInt64:
      return 8;
    case DataType::kInt8:
    case DataType::kUInt8:
      return 1;
    case DataType::kUnknown:
      break;
  }
  return 0;
}

Result<TensorDesc> TensorDesc::dense(
    DataType type, std::span<const std::int64_t> dimensions) noexcept {
  Status validation = validate_schema(type, dimensions);
  if (!validation.ok()) return std::unexpected(std::move(validation));

  TensorDesc result;
  result.type_ = type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    result.dimensions_[index] = dimensions[index];
  }

  std::int64_t stride = 1;
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    const std::size_t position = index - 1;
    result.strides_[position] = stride;
    const auto next = math::try_multiply(stride, dimensions[position]);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("dense tensor stride overflows int64"));
    }
    stride = *next;
  }

  auto bytes = result.required_bytes();
  if (!bytes) return std::unexpected(std::move(bytes).error());
  return result;
}

Result<TensorDesc> TensorDesc::strided(
    DataType type, std::span<const std::int64_t> dimensions,
    std::span<const std::int64_t> element_strides) noexcept {
  Status validation = validate_schema(type, dimensions);
  if (!validation.ok()) return std::unexpected(std::move(validation));
  if (element_strides.size() != dimensions.size()) {
    return std::unexpected(Status::InvalidArgument(
        "tensor dimensions and strides have different ranks"));
  }
  for (const std::int64_t stride : element_strides) {
    if (stride < 0) {
      return std::unexpected(
          Status::InvalidArgument("tensor stride is negative"));
    }
  }

  TensorDesc result;
  result.type_ = type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    result.dimensions_[index] = dimensions[index];
    result.strides_[index] = element_strides[index];
  }

  auto bytes = result.required_bytes();
  if (!bytes) return std::unexpected(std::move(bytes).error());
  return result;
}

Result<std::size_t> TensorDesc::element_count() const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor description is invalid"));
  }
  std::size_t count = 1;
  for (std::size_t index = 0; index < rank_; ++index) {
    const auto dimension =
        math::try_integer_cast<std::size_t>(dimensions_[index]);
    if (!dimension) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    }
    const auto next = math::try_multiply(count, *dimension);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("tensor element count overflows size_t"));
    }
    count = *next;
  }
  return count;
}

Result<std::size_t> TensorDesc::required_bytes() const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor description is invalid"));
  }
  for (std::size_t index = 0; index < rank_; ++index) {
    if (dimensions_[index] == 0) return std::size_t{0};
  }

  std::size_t last_element = 0;
  for (std::size_t index = 0; index < rank_; ++index) {
    const auto dimension =
        math::try_integer_cast<std::size_t>(dimensions_[index]);
    const auto stride = math::try_integer_cast<std::size_t>(strides_[index]);
    if (!dimension || !stride) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension or stride exceeds size_t"));
    }
    const auto extent = math::try_multiply(*dimension - 1, *stride);
    if (!extent) {
      return std::unexpected(
          Status::OutOfRange("tensor strided extent overflows size_t"));
    }
    const auto next = math::try_add(last_element, *extent);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("tensor strided offset overflows size_t"));
    }
    last_element = *next;
  }

  const auto storage_elements = math::try_add(last_element, std::size_t{1});
  if (!storage_elements) {
    return std::unexpected(
        Status::OutOfRange("tensor storage extent overflows size_t"));
  }
  const auto bytes =
      math::try_multiply(*storage_elements, data_type_size(type_));
  if (!bytes) {
    return std::unexpected(
        Status::OutOfRange("tensor byte extent overflows size_t"));
  }
  return *bytes;
}

bool TensorDesc::is_dense() const noexcept {
  if (!valid()) return false;
  const auto canonical = dense(type_, dimensions());
  return canonical.has_value() && *canonical == *this;
}

Result<ConstTensorView> ConstTensorView::create(
    TensorDesc description, mem::ConstBufferView storage) noexcept {
  if (!description.valid()) {
    return std::unexpected(
        Status::InvalidArgument("tensor description is invalid"));
  }
  if (!storage.valid()) {
    return std::unexpected(Status::InvalidArgument("tensor storage is invalid"));
  }
  auto required = description.required_bytes();
  if (!required) return std::unexpected(std::move(required).error());
  if (*required > storage.size()) {
    return std::unexpected(
        Status::OutOfRange("tensor storage is smaller than its description"));
  }
  auto exact = storage.subview(0, *required);
  if (!exact) return std::unexpected(std::move(exact).error());
  return ConstTensorView(description, *exact);
}

Result<TensorView> TensorView::create(TensorDesc description,
                                      mem::BufferView storage) noexcept {
  if (!description.valid()) {
    return std::unexpected(
        Status::InvalidArgument("tensor description is invalid"));
  }
  if (!storage.valid()) {
    return std::unexpected(Status::InvalidArgument("tensor storage is invalid"));
  }
  auto required = description.required_bytes();
  if (!required) return std::unexpected(std::move(required).error());
  if (*required > storage.size()) {
    return std::unexpected(
        Status::OutOfRange("tensor storage is smaller than its description"));
  }
  auto exact = storage.subview(0, *required);
  if (!exact) return std::unexpected(std::move(exact).error());
  return TensorView(description, *exact);
}

}  // namespace hyle::compute
