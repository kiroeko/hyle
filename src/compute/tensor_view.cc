#include "hyle/compute/tensor.h"

#include <utility>

namespace hyle::compute {
namespace {

Status validate_view_alignment(const TensorDesc& description,
                               std::size_t storage_offset,
                               std::size_t required_bytes) noexcept {
  if (required_bytes == 0) return Status::Ok();
  const std::size_t alignment = description.required_alignment();
  if (alignment == 0 || storage_offset % alignment != 0) {
    return Status::InvalidArgument(
        "tensor storage offset does not satisfy its format alignment");
  }
  return Status::Ok();
}

}  // namespace

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
  Status alignment =
      validate_view_alignment(description, storage.offset(), *required);
  if (!alignment.ok()) return std::unexpected(std::move(alignment));
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
  Status alignment =
      validate_view_alignment(description, storage.offset(), *required);
  if (!alignment.ok()) return std::unexpected(std::move(alignment));
  auto exact = storage.subview(0, *required);
  if (!exact) return std::unexpected(std::move(exact).error());
  return TensorView(description, *exact);
}

}  // namespace hyle::compute
