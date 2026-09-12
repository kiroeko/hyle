#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

#include "hyle/compute/ops_visibility.h"
#include "hyle/error/result.h"

namespace hyle::compute {

inline constexpr std::size_t kMaxTensorRank = 8;

enum class DataType : std::uint8_t {
  kUnknown = 0,
  kBool,
  kFloat8E4M3FN,
  kFloat8E5M2,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
};

HYLE_COMPUTE_OPS_API std::size_t data_type_size(DataType type) noexcept;

enum class StorageFormatKind : std::uint8_t {
  kPlain = 0,
  kBlocked,
  kOpaque,
};

// Process- and binary-independent identity for a physical encoding. Provider
// extensions should publish named constants or typed factories so ordinary
// users do not have to manufacture these values themselves.
struct StorageFormatId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t format_id = 0;

  constexpr bool valid() const noexcept {
    return namespace_id != 0 || format_id != 0;
  }
  constexpr explicit operator bool() const noexcept { return valid(); }
  friend constexpr bool operator==(StorageFormatId,
                                   StorageFormatId) noexcept = default;
};

// Self-contained physical encoding value. Plain storage derives its scalar
// width from TensorDesc::data_type(); blocked storage describes one repeated
// fixed-size block; opaque storage exposes only identity and alignment.
class HYLE_COMPUTE_OPS_API StorageFormat final {
public:
  constexpr StorageFormat() noexcept = default;

  static Result<StorageFormat> blocked(StorageFormatId id,
                                       std::size_t elements_per_block,
                                       std::size_t bytes_per_block,
                                       std::size_t alignment = 1) noexcept;
  static Result<StorageFormat> opaque(StorageFormatId id,
                                      std::size_t alignment = 1) noexcept;

  constexpr StorageFormatKind kind() const noexcept { return kind_; }
  constexpr StorageFormatId id() const noexcept { return id_; }
  constexpr std::size_t elements_per_block() const noexcept {
    return elements_per_block_;
  }
  constexpr std::size_t bytes_per_block() const noexcept {
    return bytes_per_block_;
  }
  constexpr std::size_t required_alignment() const noexcept {
    return required_alignment_;
  }
  constexpr bool is_plain() const noexcept {
    return kind_ == StorageFormatKind::kPlain;
  }

  friend constexpr bool operator==(const StorageFormat &,
                                   const StorageFormat &) noexcept = default;

private:
  StorageFormatKind kind_ = StorageFormatKind::kPlain;
  StorageFormatId id_{};
  std::uint32_t elements_per_block_ = 1;
  std::uint32_t bytes_per_block_ = 0;
  std::uint32_t required_alignment_ = 1;
};

// Self-contained logical tensor description plus its physical storage layout.
// Strides are bytes. For blocked storage the block axis stride advances one
// complete block; every other axis stride advances one logical coordinate.
// Negative strides are deliberately outside the public contract. Zero strides
// can describe broadcast inputs but an operation must explicitly support them.
class HYLE_COMPUTE_OPS_API TensorDesc final {
public:
  TensorDesc() noexcept = default;

  static Result<TensorDesc>
  dense(DataType type, std::span<const std::int64_t> dimensions) noexcept;
  static Result<TensorDesc>
  strided_bytes(DataType type, std::span<const std::int64_t> dimensions,
                std::span<const std::int64_t> byte_strides) noexcept;
  static Result<TensorDesc>
  blocked_dense(DataType logical_type, std::span<const std::int64_t> dimensions,
                StorageFormat format) noexcept;
  static Result<TensorDesc>
  blocked_strided(DataType logical_type,
                  std::span<const std::int64_t> dimensions,
                  StorageFormat format, std::size_t block_axis,
                  std::span<const std::int64_t> byte_strides) noexcept;
  static Result<TensorDesc> opaque(DataType logical_type,
                                   std::span<const std::int64_t> dimensions,
                                   StorageFormat format,
                                   std::size_t required_bytes) noexcept;

  // Return a logical view description with axes reordered. No storage is
  // moved: dimensions, byte strides and any block axis are permuted together.
  // Opaque layouts reject generic permutation because only their provider can
  // prove that a new logical view is zero-copy.
  Result<TensorDesc> permuted(std::span<const std::size_t> axes) const noexcept;

  // Natural rank-2 convenience for matrix weights stored in the opposite
  // orientation. Equivalent to permuted({1, 0}).
  Result<TensorDesc> transposed() const noexcept;

  bool valid() const noexcept { return type_ != DataType::kUnknown; }
  explicit operator bool() const noexcept { return valid(); }
  DataType data_type() const noexcept { return type_; }
  const StorageFormat &storage_format() const noexcept { return format_; }
  bool is_plain() const noexcept { return format_.is_plain(); }
  std::size_t required_alignment() const noexcept;
  std::optional<std::size_t> block_axis() const noexcept {
    return block_axis_ < kMaxTensorRank
               ? std::optional<std::size_t>(block_axis_)
               : std::nullopt;
  }
  std::size_t rank() const noexcept { return rank_; }
  std::span<const std::int64_t> dimensions() const noexcept {
    return {dimensions_.data(), rank_};
  }
  std::span<const std::int64_t> byte_strides() const noexcept {
    return format_.kind() == StorageFormatKind::kOpaque
               ? std::span<const std::int64_t>{}
               : std::span<const std::int64_t>{strides_.data(), rank_};
  }

  Result<std::size_t> element_count() const noexcept;
  Result<std::size_t> required_bytes() const noexcept;
  bool is_dense() const noexcept;

  friend constexpr bool operator==(const TensorDesc &left,
                                   const TensorDesc &right) noexcept {
    if (left.type_ != right.type_ || left.rank_ != right.rank_ ||
        left.block_axis_ != right.block_axis_ ||
        left.opaque_bytes_ != right.opaque_bytes_ ||
        left.format_ != right.format_) {
      return false;
    }
    for (std::size_t index = 0; index < left.rank_; ++index) {
      if (left.dimensions_[index] != right.dimensions_[index] ||
          left.strides_[index] != right.strides_[index]) {
        return false;
      }
    }
    return true;
  }

private:
  std::array<std::int64_t, kMaxTensorRank> dimensions_{};
  std::array<std::int64_t, kMaxTensorRank> strides_{};
  StorageFormat format_{};
  std::size_t opaque_bytes_ = 0;
  DataType type_ = DataType::kUnknown;
  std::uint8_t rank_ = 0;
  std::uint8_t block_axis_ = static_cast<std::uint8_t>(kMaxTensorRank);
};

static_assert(std::is_trivially_copyable_v<StorageFormatId>);
static_assert(std::is_trivially_copyable_v<StorageFormat>);
static_assert(std::is_trivially_copyable_v<TensorDesc>);
static_assert(sizeof(TensorDesc) <= 192);

} // namespace hyle::compute
