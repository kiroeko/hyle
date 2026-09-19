#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace hyle::compute::cuda {

class ExternalBuffer;

inline constexpr BackendId kBackendId{0x48594c4500000001ULL,
                                      0x4355444100000001ULL};

// CUDA Runtime and Driver failures retain both the portable StatusCode and
// the exact vendor code.  These helpers are also the natural return bridge for
// a typed native command submitted through submit().
HYLE_COMPUTE_BACKEND_CUDA_API Status status(cudaError_t error) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Status status(CUresult error) noexcept;

HYLE_COMPUTE_BACKEND_CUDA_API Backend backend() noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<Device>
device(std::size_t ordinal) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<int>
device_ordinal(Device device) noexcept;

// Captured once for this Context. Driver version is the CUDA API version
// supported by the installed driver (cuDriverGetVersion), not its vendor
// package/build number. Versions use CUDA's native integer encoding; the
// runtime and compiled toolkit versions are separate facts. No SDK query is
// performed by environment() or by submission. Reprepare for changed facts.
struct Environment final {
  std::array<char, 256> device_name{};
  std::array<std::byte, 16> device_uuid{};
  int device_ordinal = -1;
  int sm_major = 0;
  int sm_minor = 0;
  int driver_cuda_version = 0;
  int runtime_version = 0;
  int build_toolkit_version = 0;

  friend bool operator==(const Environment &, const Environment &) = default;
};

// An owned cold snapshot, valid independently of Context lifetime. An invalid,
// closed or non-CUDA Context is rejected through the native backend seam.
HYLE_COMPUTE_BACKEND_CUDA_API Result<Environment>
environment(const Context &context) noexcept;

enum class MemoryKind : unsigned char {
  kDevice = 0,
  kPinnedHost,
  kManaged,
};

// Stable owner for one CUDA allocation domain.  ref() is intentionally
// non-owning, matching every HYLE MemoryResourceRef: this owner and its Context
// must outlive Buffers and in-flight work created from the reference.
class HYLE_COMPUTE_BACKEND_CUDA_API MemoryResource final {
public:
  MemoryResource() noexcept = default;
  MemoryResource(const MemoryResource &) = delete;
  MemoryResource &operator=(const MemoryResource &) = delete;
  MemoryResource(MemoryResource &&other) noexcept;
  MemoryResource &operator=(MemoryResource &&other) noexcept;
  ~MemoryResource();

  static Result<MemoryResource> create(const Context &context,
                                       MemoryKind kind) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  mem::MemoryResourceRef ref() const noexcept;
  MemoryKind kind() const noexcept;
  Device device() const noexcept;
  Status trim() noexcept;
  void reset() noexcept;

private:
  explicit MemoryResource(void *state) noexcept : state_(state) {}
  void *state_ = nullptr;
};

// Typed native interop is opt-in: ordinary Compute headers never include CUDA
// declarations.  Returned handles are borrowed from their HYLE owner.
HYLE_COMPUTE_BACKEND_CUDA_API Result<cudaExecutionContext_t>
execution_context(const Context &context) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<CUcontext>
primary_context(const Context &context) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<cudaStream_t>
stream(const Queue &queue) noexcept;

// Import an existing non-default stream from the exact primary execution
// context. Borrow never destroys it; adopt destroys it after all retained
// Completion state terminates. The external owner must cover every borrowed
// Queue and Completion.
HYLE_COMPUTE_BACKEND_CUDA_API Result<Queue>
borrow_stream(const Context &context, cudaStream_t stream) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<Queue>
adopt_stream(const Context &context, cudaStream_t stream) noexcept;

using StreamCommand = Status (*)(cudaStream_t stream, void *context) noexcept;

// The callback is invoked exactly once during host-side enqueue and is never
// retained. A successful return means the callback recorded one or more
// commands on the supplied non-default stream; the returned Completion covers
// the complete ordered Queue prefix through those commands.
HYLE_COMPUTE_BACKEND_CUDA_API Result<Completion>
submit(Queue &queue, StreamCommand command, void *context) noexcept;

template <class Function>
  requires std::is_nothrow_invocable_v<Function &, cudaStream_t> &&
           (std::same_as<std::invoke_result_t<Function &, cudaStream_t>, void> ||
            std::same_as<std::invoke_result_t<Function &, cudaStream_t>, Status> ||
            std::same_as<std::invoke_result_t<Function &, cudaStream_t>,
                         cudaError_t>)
Result<Completion> submit(Queue &queue, Function &&function) noexcept {
  using Callable = std::remove_reference_t<Function>;
  auto invoke = [](cudaStream_t native_stream, void *context) noexcept -> Status {
    auto &callable = *static_cast<Callable *>(context);
    if constexpr (std::same_as<std::invoke_result_t<Callable &, cudaStream_t>,
                               void>) {
      callable(native_stream);
      return Status::Ok();
    } else if constexpr (std::same_as<
                             std::invoke_result_t<Callable &, cudaStream_t>,
                             cudaError_t>) {
      return status(callable(native_stream));
    } else {
      return callable(native_stream);
    }
  };
  return submit(queue, invoke, std::addressof(function));
}

namespace detail {
HYLE_COMPUTE_BACKEND_CUDA_API Result<void *>
mutable_pointer(mem::BufferView view, std::size_t alignment,
                std::size_t element_size) noexcept;
HYLE_COMPUTE_BACKEND_CUDA_API Result<const void *>
const_pointer(mem::ConstBufferView view, std::size_t alignment,
              std::size_t element_size) noexcept;
Result<ExternalBuffer> make_external(const Context &context, void *pointer,
                                     std::size_t size, bool owning) noexcept;
} // namespace detail

template <class T>
  requires(!std::is_void_v<T>)
Result<T *> pointer(mem::BufferView view) noexcept {
  auto raw = detail::mutable_pointer(view, alignof(T), sizeof(T));
  if (!raw)
    return std::unexpected(std::move(raw).error());
  return static_cast<T *>(*raw);
}

template <class T>
  requires(!std::is_void_v<T>)
Result<const T *> pointer(mem::ConstBufferView view) noexcept {
  auto raw = detail::const_pointer(view, alignof(T), sizeof(T));
  if (!raw)
    return std::unexpected(std::move(raw).error());
  return static_cast<const T *>(*raw);
}

class HYLE_COMPUTE_BACKEND_CUDA_API ExternalBuffer final {
public:
  ExternalBuffer() noexcept = default;
  ExternalBuffer(const ExternalBuffer &) = delete;
  ExternalBuffer &operator=(const ExternalBuffer &) = delete;
  ExternalBuffer(ExternalBuffer &&other) noexcept;
  ExternalBuffer &operator=(ExternalBuffer &&other) noexcept;
  ~ExternalBuffer();

  static Result<ExternalBuffer> borrow(const Context &context, void *pointer,
                                       std::size_t size) noexcept;
  static Result<ExternalBuffer> adopt(const Context &context, void *pointer,
                                      std::size_t size) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  mem::BufferView view() noexcept;
  mem::ConstBufferView view() const noexcept;
  std::size_t size() const noexcept;
  Device device() const noexcept;
  void reset() noexcept;

private:
  friend Result<ExternalBuffer> detail::make_external(
      const Context &context, void *pointer, std::size_t size,
      bool owning) noexcept;
  explicit ExternalBuffer(void *state) noexcept : state_(state) {}
  void *state_ = nullptr;
};

} // namespace hyle::compute::cuda
