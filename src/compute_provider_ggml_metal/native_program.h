#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "hyle/compute/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GGML_METAL_INTERNAL_API
#elif defined(HYLE_COMPUTE_PROVIDER_GGML_METAL_BUILD)
#define HYLE_GGML_METAL_INTERNAL_API HYLE_EXPORT
#else
#define HYLE_GGML_METAL_INTERNAL_API HYLE_IMPORT
#endif

namespace hyle::compute::ggml_metal {

// Private Apple-native boundary. metal_buffer is an id<MTLBuffer>, with its
// owning array retained by the Apple invocation. No host-pointer remapping.
struct BufferView final {
  void *metal_buffer = nullptr;
  std::size_t offset = 0;
};

// One original backend/graph and its bindings per invocation. Caller completes
// preceding GPU work before run(); run() checks and drains this backend before
// returning. Thread-compatible, with no Graph or MLX dependency.
class NativeProgram final {
public:
  NativeProgram(const NativeProgram &) = delete;
  NativeProgram &operator=(const NativeProgram &) = delete;
  HYLE_GGML_METAL_INTERNAL_API ~NativeProgram();
  HYLE_GGML_METAL_INTERNAL_API static Result<std::unique_ptr<NativeProgram>>
  create(Operation operation, std::span<const TensorDesc> inputs,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  static Result<std::unique_ptr<NativeProgram>>
  create(std::span<const TensorDesc> inputs,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept {
    return create(Operation(GatedDeltaNet{}), inputs, allocator);
  }
  HYLE_GGML_METAL_INTERNAL_API static Status
  validate(Operation operation, std::span<const TensorDesc> inputs) noexcept;
  static Status validate(std::span<const TensorDesc> inputs) noexcept {
    return validate(Operation(GatedDeltaNet{}), inputs);
  }
  // Input ports followed by one contiguous output allocation. Multi-output
  // native regions publish non-overlapping views that retain that allocation.
  HYLE_GGML_METAL_INTERNAL_API Status
  run(std::span<const BufferView> ports) noexcept;
  HYLE_GGML_METAL_INTERNAL_API std::size_t output_count() const noexcept;
  HYLE_GGML_METAL_INTERNAL_API const TensorDesc &
  output_description(std::size_t index) const noexcept;
  HYLE_GGML_METAL_INTERNAL_API std::size_t
  output_offset(std::size_t index) const noexcept;
  HYLE_GGML_METAL_INTERNAL_API std::size_t score_bytes() const noexcept;
  HYLE_GGML_METAL_INTERNAL_API std::size_t output_bytes() const noexcept;

private:
  struct State;
  explicit NativeProgram(std::unique_ptr<State> state) noexcept;
  std::unique_ptr<State> state_;
};
} // namespace hyle::compute::ggml_metal
