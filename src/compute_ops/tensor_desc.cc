#include "hyle/compute/tensor_desc.h"

#include <limits>
#include <utility>

#include "hyle/math/scalar.h"

namespace hyle::compute {
namespace {

constexpr std::uint8_t kNoBlockAxis = static_cast<std::uint8_t>(kMaxTensorRank);

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

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

Status validate_strides(std::span<const std::int64_t> dimensions,
                        std::span<const std::int64_t> byte_strides) noexcept {
  if (byte_strides.size() != dimensions.size()) {
    return Status::InvalidArgument(
        "tensor dimensions and byte strides have different ranks");
  }
  for (const std::int64_t stride : byte_strides) {
    if (stride < 0) {
      return Status::InvalidArgument("tensor byte stride is negative");
    }
  }
  return Status::Ok();
}

bool has_zero_extent(std::span<const std::int64_t> dimensions) noexcept {
  for (const std::int64_t dimension : dimensions) {
    if (dimension == 0)
      return true;
  }
  return false;
}

Result<std::int64_t> stride_as_int64(std::size_t stride) noexcept {
  const auto converted = math::try_integer_cast<std::int64_t>(stride);
  if (!converted) {
    return std::unexpected(
        Status::OutOfRange("tensor byte stride overflows int64"));
  }
  return *converted;
}

Status
validate_materialized_description(const TensorDesc &description) noexcept {
  auto elements = description.element_count();
  if (!elements)
    return std::move(elements).error();
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::move(bytes).error();
  return Status::Ok();
}

} // namespace

std::size_t data_type_size(DataType type) noexcept {
  switch (type) {
  case DataType::kBool:
  case DataType::kFloat8E4M3FN:
  case DataType::kFloat8E5M2:
  case DataType::kInt8:
  case DataType::kUInt8:
    return 1;
  case DataType::kFloat16:
  case DataType::kBFloat16:
  case DataType::kInt16:
  case DataType::kUInt16:
    return 2;
  case DataType::kFloat32:
  case DataType::kInt32:
  case DataType::kUInt32:
    return 4;
  case DataType::kFloat64:
  case DataType::kInt64:
  case DataType::kUInt64:
    return 8;
  case DataType::kUnknown:
    break;
  }
  return 0;
}

Result<TensorShape>
TensorShape::create(std::span<const std::int64_t> dimensions) noexcept {
  if (dimensions.size() > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("tensor shape rank exceeds kMaxTensorRank"));
  }
  TensorShape result;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (dimensions[index] < 0) {
      return std::unexpected(
          Status::InvalidArgument("tensor shape dimension is negative"));
    }
    result.dimensions_[index] = dimensions[index];
  }
  result.rank_plus_one_ = static_cast<std::uint8_t>(dimensions.size() + 1);
  return result;
}

TensorShape TensorShape::scalar() noexcept {
  TensorShape result;
  result.rank_plus_one_ = 1;
  return result;
}

Result<std::size_t> TensorShape::element_count() const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor shape is invalid"));
  }
  for (const std::int64_t dimension : dimensions()) {
    if (dimension == 0)
      return std::size_t{0};
  }
  std::size_t count = 1;
  for (const std::int64_t dimension : dimensions()) {
    auto extent = math::try_integer_cast<std::size_t>(dimension);
    if (!extent) {
      return std::unexpected(
          Status::OutOfRange("tensor shape dimension exceeds size_t"));
    }
    auto next = math::try_multiply(count, *extent);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("tensor shape element count overflows size_t"));
    }
    count = *next;
  }
  return count;
}

Result<StorageFormat> StorageFormat::blocked(StorageFormatId id,
                                             std::size_t elements_per_block,
                                             std::size_t bytes_per_block,
                                             std::size_t alignment) noexcept {
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("blocked storage format identity is empty"));
  }
  if (elements_per_block == 0 || bytes_per_block == 0) {
    return std::unexpected(Status::InvalidArgument(
        "blocked storage format requires non-zero block dimensions"));
  }
  if (!is_power_of_two(alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "storage format alignment must be a non-zero power of two"));
  }
  constexpr std::size_t kMaximum = (std::numeric_limits<std::uint32_t>::max)();
  if (elements_per_block > kMaximum || bytes_per_block > kMaximum ||
      alignment > kMaximum) {
    return std::unexpected(
        Status::OutOfRange("blocked storage format metadata exceeds uint32"));
  }

  StorageFormat result;
  result.kind_ = StorageFormatKind::kBlocked;
  result.id_ = id;
  result.elements_per_block_ = static_cast<std::uint32_t>(elements_per_block);
  result.bytes_per_block_ = static_cast<std::uint32_t>(bytes_per_block);
  result.required_alignment_ = static_cast<std::uint32_t>(alignment);
  return result;
}

Result<StorageFormat> StorageFormat::opaque(StorageFormatId id,
                                            std::size_t alignment) noexcept {
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("opaque storage format identity is empty"));
  }
  if (!is_power_of_two(alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "storage format alignment must be a non-zero power of two"));
  }
  constexpr std::size_t kMaximum = (std::numeric_limits<std::uint32_t>::max)();
  if (alignment > kMaximum) {
    return std::unexpected(
        Status::OutOfRange("opaque storage alignment exceeds uint32"));
  }

  StorageFormat result;
  result.kind_ = StorageFormatKind::kOpaque;
  result.id_ = id;
  result.elements_per_block_ = 0;
  result.bytes_per_block_ = 0;
  result.required_alignment_ = static_cast<std::uint32_t>(alignment);
  return result;
}

Result<TensorDesc>
TensorDesc::dense(DataType type,
                  std::span<const std::int64_t> dimensions) noexcept {
  Status validation = validate_schema(type, dimensions);
  if (!validation.ok())
    return std::unexpected(std::move(validation));

  std::array<std::int64_t, kMaxTensorRank> byte_strides{};
  if (has_zero_extent(dimensions)) {
    return strided_bytes(type, dimensions,
                         std::span(byte_strides.data(), dimensions.size()));
  }
  std::size_t stride = data_type_size(type);
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    const std::size_t position = index - 1;
    auto stored_stride = stride_as_int64(stride);
    if (!stored_stride)
      return std::unexpected(std::move(stored_stride).error());
    byte_strides[position] = *stored_stride;

    const auto dimension =
        math::try_integer_cast<std::size_t>(dimensions[position]);
    if (!dimension) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    }
    const auto next = math::try_multiply(stride, *dimension);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("dense tensor byte stride overflows size_t"));
    }
    stride = *next;
  }
  return strided_bytes(type, dimensions,
                       std::span(byte_strides.data(), dimensions.size()));
}

Result<TensorDesc>
TensorDesc::strided_bytes(DataType type,
                          std::span<const std::int64_t> dimensions,
                          std::span<const std::int64_t> byte_strides) noexcept {
  Status validation = validate_schema(type, dimensions);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  validation = validate_strides(dimensions, byte_strides);
  if (!validation.ok())
    return std::unexpected(std::move(validation));

  TensorDesc result;
  result.type_ = type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.block_axis_ = kNoBlockAxis;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    result.dimensions_[index] = dimensions[index];
    result.strides_[index] = byte_strides[index];
  }

  validation = validate_materialized_description(result);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  return result;
}

Result<TensorDesc>
TensorDesc::blocked_dense(DataType logical_type,
                          std::span<const std::int64_t> dimensions,
                          StorageFormat format) noexcept {
  Status validation = validate_schema(logical_type, dimensions);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (format.kind() != StorageFormatKind::kBlocked) {
    return std::unexpected(Status::InvalidArgument(
        "blocked tensor requires a blocked storage format"));
  }
  if (dimensions.empty()) {
    return std::unexpected(Status::InvalidArgument(
        "blocked tensor requires at least one logical axis"));
  }

  const std::size_t block_axis = dimensions.size() - 1;
  const auto blocked_extent =
      math::try_integer_cast<std::size_t>(dimensions[block_axis]);
  if (!blocked_extent) {
    return std::unexpected(
        Status::OutOfRange("tensor dimension exceeds size_t"));
  }
  if (*blocked_extent % format.elements_per_block() != 0) {
    return std::unexpected(Status::InvalidArgument(
        "blocked tensor axis is not divisible by elements per block"));
  }

  std::array<std::int64_t, kMaxTensorRank> byte_strides{};
  if (has_zero_extent(dimensions)) {
    return blocked_strided(logical_type, dimensions, format, block_axis,
                           std::span(byte_strides.data(), dimensions.size()));
  }
  std::size_t stride = format.bytes_per_block();
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    const std::size_t position = index - 1;
    auto stored_stride = stride_as_int64(stride);
    if (!stored_stride)
      return std::unexpected(std::move(stored_stride).error());
    byte_strides[position] = *stored_stride;

    auto units = math::try_integer_cast<std::size_t>(dimensions[position]);
    if (!units) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    }
    if (position == block_axis)
      *units /= format.elements_per_block();
    const auto next = math::try_multiply(stride, *units);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("blocked tensor byte stride overflows size_t"));
    }
    stride = *next;
  }
  return blocked_strided(logical_type, dimensions, format, block_axis,
                         std::span(byte_strides.data(), dimensions.size()));
}

Result<TensorDesc> TensorDesc::blocked_strided(
    DataType logical_type, std::span<const std::int64_t> dimensions,
    StorageFormat format, std::size_t block_axis,
    std::span<const std::int64_t> byte_strides) noexcept {
  Status validation = validate_schema(logical_type, dimensions);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  validation = validate_strides(dimensions, byte_strides);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (format.kind() != StorageFormatKind::kBlocked) {
    return std::unexpected(Status::InvalidArgument(
        "blocked tensor requires a blocked storage format"));
  }
  if (block_axis >= dimensions.size()) {
    return std::unexpected(
        Status::InvalidArgument("blocked tensor axis is out of range"));
  }
  const auto blocked_extent =
      math::try_integer_cast<std::size_t>(dimensions[block_axis]);
  if (!blocked_extent) {
    return std::unexpected(
        Status::OutOfRange("tensor dimension exceeds size_t"));
  }
  if (*blocked_extent % format.elements_per_block() != 0) {
    return std::unexpected(Status::InvalidArgument(
        "blocked tensor axis is not divisible by elements per block"));
  }

  TensorDesc result;
  result.type_ = logical_type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.block_axis_ = static_cast<std::uint8_t>(block_axis);
  result.format_ = format;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    result.dimensions_[index] = dimensions[index];
    result.strides_[index] = byte_strides[index];
  }

  validation = validate_materialized_description(result);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  return result;
}

Result<TensorDesc> TensorDesc::opaque(DataType logical_type,
                                      std::span<const std::int64_t> dimensions,
                                      StorageFormat format,
                                      std::size_t required_bytes) noexcept {
  Status validation = validate_schema(logical_type, dimensions);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (format.kind() != StorageFormatKind::kOpaque) {
    return std::unexpected(
        Status::InvalidArgument("opaque tensor requires an opaque format"));
  }

  TensorDesc result;
  result.type_ = logical_type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.block_axis_ = kNoBlockAxis;
  result.format_ = format;
  result.opaque_bytes_ = required_bytes;
  for (std::size_t index = 0; index < dimensions.size(); ++index)
    result.dimensions_[index] = dimensions[index];

  auto elements = result.element_count();
  if (!elements)
    return std::unexpected(std::move(elements).error());
  if (*elements != 0 && required_bytes == 0) {
    return std::unexpected(Status::InvalidArgument(
        "non-empty opaque tensor requires a non-zero byte extent"));
  }
  return result;
}

Result<TensorDesc>
TensorDesc::permuted(std::span<const std::size_t> axes) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor description is invalid"));
  }
  if (axes.size() != rank_) {
    return std::unexpected(Status::InvalidArgument(
        "tensor permutation rank does not match the description"));
  }

  std::array<bool, kMaxTensorRank> seen{};
  for (const std::size_t source : axes) {
    if (source >= rank_ || seen[source]) {
      return std::unexpected(Status::InvalidArgument(
          "tensor permutation contains an invalid or duplicate axis"));
    }
    seen[source] = true;
  }
  if (format_.kind() == StorageFormatKind::kOpaque) {
    return std::unexpected(Status::Unimplemented(
        "opaque tensor layout cannot be generically permuted"));
  }

  TensorDesc result = *this;
  result.block_axis_ = kNoBlockAxis;
  for (std::size_t destination = 0; destination < axes.size(); ++destination) {
    const std::size_t source = axes[destination];
    result.dimensions_[destination] = dimensions_[source];
    result.strides_[destination] = strides_[source];
    if (source == block_axis_)
      result.block_axis_ = static_cast<std::uint8_t>(destination);
  }
  return result;
}

Result<TensorDesc> TensorDesc::transposed() const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor description is invalid"));
  }
  if (rank_ != 2) {
    return std::unexpected(
        Status::InvalidArgument("matrix transpose requires rank 2"));
  }
  constexpr std::array<std::size_t, 2> axes{1, 0};
  return permuted(axes);
}

std::size_t TensorDesc::required_alignment() const noexcept {
  if (!valid())
    return 1;
  if (format_.kind() == StorageFormatKind::kPlain)
    return data_type_size(type_);
  return format_.required_alignment();
}

Result<std::size_t> TensorDesc::element_count() const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("tensor description is invalid"));
  }
  for (std::size_t index = 0; index < rank_; ++index) {
    if (dimensions_[index] == 0)
      return std::size_t{0};
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
  if (format_.kind() == StorageFormatKind::kOpaque)
    return opaque_bytes_;
  for (std::size_t index = 0; index < rank_; ++index) {
    if (dimensions_[index] == 0)
      return std::size_t{0};
  }

  std::size_t last_unit = 0;
  for (std::size_t index = 0; index < rank_; ++index) {
    auto units = math::try_integer_cast<std::size_t>(dimensions_[index]);
    const auto stride = math::try_integer_cast<std::size_t>(strides_[index]);
    if (!units || !stride) {
      return std::unexpected(
          Status::OutOfRange("tensor dimension or byte stride exceeds size_t"));
    }
    if (index == block_axis_)
      *units /= format_.elements_per_block();
    const auto extent = math::try_multiply(*units - 1, *stride);
    if (!extent) {
      return std::unexpected(
          Status::OutOfRange("tensor strided extent overflows size_t"));
    }
    const auto next = math::try_add(last_unit, *extent);
    if (!next) {
      return std::unexpected(
          Status::OutOfRange("tensor strided offset overflows size_t"));
    }
    last_unit = *next;
  }

  const std::size_t unit_bytes = format_.kind() == StorageFormatKind::kBlocked
                                     ? format_.bytes_per_block()
                                     : data_type_size(type_);
  const auto bytes = math::try_add(last_unit, unit_bytes);
  if (!bytes) {
    return std::unexpected(
        Status::OutOfRange("tensor byte extent overflows size_t"));
  }
  return *bytes;
}

bool TensorDesc::is_dense() const noexcept {
  if (!valid() || format_.kind() == StorageFormatKind::kOpaque)
    return false;
  if (format_.kind() == StorageFormatKind::kPlain) {
    const auto canonical = dense(type_, dimensions());
    return canonical.has_value() && *canonical == *this;
  }
  if (rank_ == 0 || block_axis_ != rank_ - 1)
    return false;
  const auto canonical = blocked_dense(type_, dimensions(), format_);
  return canonical.has_value() && *canonical == *this;
}

} // namespace hyle::compute
