#pragma once

#include <type_traits>

#include "hyle/compute/tensor_desc.h"
#include "hyle/compute/visibility.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute {

class TensorView;

// A TensorView never owns payload. Mem supplies the provider-neutral storage
// identity; the matching backend/provider interprets an opaque resource.
class ConstTensorView final {
 public:
  ConstTensorView() noexcept = default;

  HYLE_COMPUTE_API static Result<ConstTensorView>
  create(TensorDesc description, mem::ConstBufferView storage) noexcept;

  bool valid() const noexcept { return description_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  const TensorDesc& description() const noexcept { return description_; }
  mem::ConstBufferView storage() const noexcept { return storage_; }

 private:
  friend class TensorView;
  ConstTensorView(TensorDesc description, mem::ConstBufferView storage) noexcept
      : description_(description), storage_(storage) {}

  TensorDesc description_{};
  mem::ConstBufferView storage_{};
};

class TensorView final {
 public:
  TensorView() noexcept = default;

  HYLE_COMPUTE_API static Result<TensorView>
  create(TensorDesc description, mem::BufferView storage) noexcept;

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

static_assert(std::is_trivially_copyable_v<TensorView>);
static_assert(std::is_trivially_copyable_v<ConstTensorView>);

}  // namespace hyle::compute
