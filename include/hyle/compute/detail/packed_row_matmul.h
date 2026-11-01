#pragma once
#include "hyle/compute/provider.h"
#include "hyle/compute/tensor.h"

namespace hyle::graph::batch::detail { struct QueryRowsAccess; }
namespace hyle::compute::detail {
// Coarse source capability only. Exact shapes and the original host-owned
// offsets proof are still qualified by the selected prepared implementation.
inline constexpr ProviderExtensionId kPackedRowMatMulSourceExtensionId{
    0x48594c45504b4d4dULL, 0x5352434543415053ULL, 1};
struct PackedRowMatMulSourceVTable final {
  bool (*available)(void *provider_context) noexcept = nullptr;
};

// Selected-operation cold access, using the existing OperationAccessVTable.
// The Graph adapter must first verify its original opaque Batch row control.
// This borrowed tuple grants no storage permission or lifetime of its own.
inline constexpr ProviderExtensionId kPackedRowMatMulRowsExtensionId{
    0x48594c45504b4d4dULL, 0x524f5753434f4e46ULL, 2};

// Borrowed cold facts minted only by the original Batch QueryRows source.
// Its existing Graph configuration retains and seals that immutable metadata
// through real retirement. This proof creates no owner, lease or new epoch.
class PackedRowMatMulRowsProof final {
 public:
  PackedRowMatMulRowsProof() noexcept = default;
  ConstTensorView offsets() const noexcept { return offsets_; }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t capacity() const noexcept { return capacity_; }
  bool valid() const noexcept {
    return offsets_.valid() && capacity_ != 0 && rows_ <= capacity_;
  }

 private:
  friend struct hyle::graph::batch::detail::QueryRowsAccess;
  PackedRowMatMulRowsProof(ConstTensorView offsets, std::size_t rows,
                          std::size_t capacity) noexcept
      : offsets_(offsets), rows_(rows), capacity_(capacity) {}
  ConstTensorView offsets_;
  std::size_t rows_ = 0, capacity_ = 0;
};
struct PackedRowMatMulRowsConfiguration final {
  ConstTensorView offsets;
  std::size_t rows = 0;
  bool clear = false;
  PackedRowMatMulRowsProof proof{};
};
} // namespace hyle::compute::detail
