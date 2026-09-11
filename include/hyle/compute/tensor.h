#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute {

inline constexpr std::size_t kMaxTensorRank = 8;

enum class DataType : std::uint8_t {
  kUnknown = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
  kInt8,
  kUInt8,
  kInt32,
  kInt64,
};

HYLE_COMPUTE_API std::size_t data_type_size(DataType type) noexcept;

// Self-contained logical tensor description. Strides are in elements, not
// bytes. Negative strides are deliberately rejected in v1; zero strides can
// describe broadcast inputs but are not accepted by the first RMSNorm route.
class HYLE_COMPUTE_API TensorDesc final {
 public:
  TensorDesc() noexcept = default;

  static Result<TensorDesc> dense(
      DataType type, std::span<const std::int64_t> dimensions) noexcept;
  static Result<TensorDesc> strided(
      DataType type, std::span<const std::int64_t> dimensions,
      std::span<const std::int64_t> element_strides) noexcept;

  // Return a logical view description with axes reordered. No storage is
  // moved: dimensions and element strides are permuted together. `axes` must
  // contain every source axis exactly once.
  Result<TensorDesc> permuted(
      std::span<const std::size_t> axes) const noexcept;

  // Natural rank-2 convenience for matrix weights stored in the opposite
  // orientation. Equivalent to permuted({1, 0}).
  Result<TensorDesc> transposed() const noexcept;

  bool valid() const noexcept { return type_ != DataType::kUnknown; }
  explicit operator bool() const noexcept { return valid(); }
  DataType data_type() const noexcept { return type_; }
  std::size_t rank() const noexcept { return rank_; }
  std::span<const std::int64_t> dimensions() const noexcept {
    return {dimensions_.data(), rank_};
  }
  std::span<const std::int64_t> element_strides() const noexcept {
    return {strides_.data(), rank_};
  }

  Result<std::size_t> element_count() const noexcept;
  Result<std::size_t> required_bytes() const noexcept;
  bool is_dense() const noexcept;

  friend constexpr bool operator==(const TensorDesc& left,
                                   const TensorDesc& right) noexcept {
    if (left.type_ != right.type_ || left.rank_ != right.rank_) return false;
    for (std::size_t index = 0; index < left.rank_; ++index) {
      if (left.dimensions_[index] != right.dimensions_[index] ||
          left.strides_[index] != right.strides_[index]) {
        return false;
      }
    }
    return true;
  }

 private:
  DataType type_ = DataType::kUnknown;
  std::uint8_t rank_ = 0;
  std::array<std::int64_t, kMaxTensorRank> dimensions_{};
  std::array<std::int64_t, kMaxTensorRank> strides_{};
};

class TensorView;

class ConstTensorView final {
 public:
  ConstTensorView() noexcept = default;

  HYLE_COMPUTE_API static Result<ConstTensorView> create(
      TensorDesc description, mem::ConstBufferView storage) noexcept;

  bool valid() const noexcept { return description_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  const TensorDesc& description() const noexcept { return description_; }
  mem::ConstBufferView storage() const noexcept { return storage_; }

 private:
  friend class TensorView;
  ConstTensorView(TensorDesc description,
                  mem::ConstBufferView storage) noexcept
      : description_(description), storage_(storage) {}

  TensorDesc description_{};
  mem::ConstBufferView storage_{};
};

class TensorView final {
 public:
  TensorView() noexcept = default;

  HYLE_COMPUTE_API static Result<TensorView> create(
      TensorDesc description, mem::BufferView storage) noexcept;

  bool valid() const noexcept { return description_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  const TensorDesc& description() const noexcept { return description_; }
  mem::BufferView storage() const noexcept { return storage_; }
  ConstTensorView as_const() const noexcept {
    return ConstTensorView(description_, mem::ConstBufferView(storage_));
  }

 private:
  TensorView(TensorDesc description, mem::BufferView storage) noexcept
      : description_(description), storage_(storage) {}

  TensorDesc description_{};
  mem::BufferView storage_{};
};

}  // namespace hyle::compute
