#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_ROUTE_VULKAN_API
#elif defined(HYLE_COMPUTE_ROUTE_VULKAN_BUILD)
#define HYLE_COMPUTE_ROUTE_VULKAN_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_ROUTE_VULKAN_API HYLE_IMPORT
#endif

#if defined(HYLE_COMPUTE_VULKAN)
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "hyle/compute/operation.h"
#include "hyle/compute/result.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/quota_allocator.h"

namespace hyle::compute::vulkan {
namespace detail {
struct Buffer;
struct Runtime;
struct Access;
struct Pipeline;
struct ProgramState;
struct InvocationState;
struct RecorderState;
struct Run;
struct ResultValues;
} // namespace detail

// Cold facts used by fixed, qualified implementation rules. A compiled shader
// or an available extension alone is not a performance qualification.
struct Environment final {
  std::string device_name;
  std::string driver_name;
  std::string driver_info;
  std::string shader_compiler;
  std::string shader_revision;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t driver_id = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t loader_version = 0;
  std::uint32_t api_version = 0;
  std::uint32_t header_version = VK_HEADER_VERSION;
  std::uint32_t subgroup_size = 0;
  std::uint32_t shared_memory_bytes = 0;
  bool subgroup_arithmetic = false;
  bool storage_update_after_bind = false;
  bool portability_subset = false;
  bool software_device = false;
};

struct ContextOptions final {
  // Null selects the platform Vulkan loader. This is a native loader path,
  // not an implementation preference list.
  const char *loader_path = nullptr;
  std::uint32_t device_index = (std::numeric_limits<std::uint32_t>::max)();
  bool allow_software_device = false;
  bool validation = false;
  // Optional accounting of actual VkDeviceMemory allocations, including
  // staging. The quota must outlive all tensors/programs/results. Native
  // driver and pipeline allocations are outside this byte budget.
  mem::QuotaAllocator *memory_quota = nullptr;
};

struct NativeDevice final {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
};

// A native buffer owner plus its logical view. Copies retain the allocation;
// they never copy payload. Separate handles are thread-compatible. Mutating
// payload requires exclusive application access and completed prior GPU work.
// Captured constants stay immutable for the lifetime of their Program.
class Tensor final {
public:
  Tensor() noexcept = default;
  bool valid() const noexcept { return buffer_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  const TensorDesc &description() const noexcept { return description_; }
  HYLE_COMPUTE_ROUTE_VULKAN_API VkBuffer native_buffer() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API VkDeviceSize native_offset() const noexcept;
  // Export this token with native_buffer, including when the Tensor came from
  // a generated result. A retained token also prevents that slot's overwrite.
  HYLE_COMPUTE_ROUTE_VULKAN_API
  std::shared_ptr<const void> native_owner() const noexcept;
  // Pure metadata: preserves native storage and retains its owner. Slices may
  // have an unaligned logical byte origin; typed tensor inputs handle it.
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  view(const compute::View &operation) const noexcept;

private:
  friend struct detail::Access;
  Tensor(TensorDesc description,
         std::shared_ptr<detail::Buffer> buffer) noexcept
      : description_(description), buffer_(std::move(buffer)) {}
  TensorDesc description_{};
  std::shared_ptr<detail::Buffer> buffer_;
};

// One native compute queue, serialized only at its Vulkan host synchronization
// boundary. Independent invocation slots have independent command/descriptor
// pools and fences. Context methods may be used concurrently on an unmoved
// handle; moving/destroying a handle requires external synchronization.
class Context final {
public:
  Context() noexcept = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API Context &operator=(Context &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~Context();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Context>
  create(ContextOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_ROUTE_VULKAN_API const Environment &environment() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API NativeDevice native_device() const noexcept;

  // Uninitialized storage for native producers. Reading before initialization
  // is a caller error. Ordinary host input construction uses upload instead.
  // Native capacity rounds logical bytes up to four, with a four-byte minimum
  // for empty tensors. Transfers consume exactly the logical byte span.
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  allocate(const TensorDesc &description) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  upload(const TensorDesc &description,
         std::span<const std::byte> bytes) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  upload(const Tensor &destination,
         std::span<const std::byte> bytes) const noexcept;
  // Explicit synchronous transfer; no implicit host shadow is maintained.
  // Download remains valid after close, while tensor/device owners survive.
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  download(const Tensor &source, std::span<std::byte> bytes) const noexcept;

  // Ready, visible buffer in this exact VkDevice and queue family. The owner
  // must retain its buffer and memory; external producers must have finished.
  // No pending semaphore, ownership transfer or cross-device bridge is implied.
  // available_bytes must cover the same four-byte padded capacity as allocate;
  // offset must satisfy both the native storage alignment and four bytes.
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  borrow_ready(const TensorDesc &description, VkBuffer buffer,
               VkDeviceSize offset, VkDeviceSize available_bytes,
               std::shared_ptr<const void> owner) const noexcept;
  // Expert access to the same queue under its host lock. Do not call back into
  // this Context. The callback must finish its work and establish visibility
  // before returning; later HYLE submissions do not infer an external fence.
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  with_queue(const std::function<Status(VkQueue)> &function) const noexcept;

  // Nonblocking retirement, including abandoned submissions. Unknown native
  // observation failures retain pending resources and return their diagnosis.
  HYLE_COMPUTE_ROUTE_VULKAN_API Status collect() const noexcept;
  // Stop accepting compute and drain. A persistent native observation failure
  // retains unresolved work; callers may retry close/collect. Device loss is a
  // native terminal failure, never successful output. Destruction calls close.
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close() noexcept;

private:
  friend struct detail::Access;
  explicit Context(std::shared_ptr<detail::Runtime> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::Runtime> state_;
};

struct SpecializationConstant final {
  std::uint32_t id = 0;
  std::uint32_t value = 0;
};

// Native compute shader: one set of sequential storage-buffer bindings and
// one push-constant range. The caller supplies valid SPIR-V for this device's
// enabled features. Resources remain native; a shader may implement a fused
// fragment rather than one HYLE operation.
struct SpirvDescription final {
  std::span<const std::uint32_t> code;
  std::uint32_t storage_bindings = 0;
  std::uint32_t push_constant_bytes = 0;
  std::span<const SpecializationConstant> specialization;
};

class Kernel final {
public:
  Kernel() noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Kernel>
  create(const Context &context, const SpirvDescription &description) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

private:
  friend struct detail::Access;
  explicit Kernel(std::shared_ptr<detail::Pipeline> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::Pipeline> state_;
};

// A runtime input specification or a ready immutable captured constant, in
// the operation's ordinary argument order. Constant RHS packing is cold and
// shared; a runtime MatMul RHS pays its required layout conversion on the GPU.
class Argument final {
public:
  Argument(TensorDesc input) noexcept : description_(input) {}
  Argument(Tensor constant) noexcept
      : description_(constant.description()), constant_(std::move(constant)) {}
  const TensorDesc &description() const noexcept { return description_; }
  const Tensor &constant() const noexcept { return constant_; }

private:
  TensorDesc description_;
  Tensor constant_;
};

// Original device buffers, available after successful completion. Copying a
// Tensor out of this view retains it and prevents that Invocation from
// overwriting its allocation, even after the ResultLease is released.
class Results final {
public:
  Results() noexcept = default;
  Results(const Results &) = delete;
  Results &operator=(const Results &) = delete;
  Results(Results &&) noexcept = default;
  Results &operator=(Results &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API std::span<const Tensor>
  tensors() const noexcept;

private:
  friend struct detail::Access;
  explicit Results(std::shared_ptr<detail::ResultValues> values) noexcept
      : values_(std::move(values)) {}
  std::shared_ptr<detail::ResultValues> values_;
};

class Invocation;

// Immutable pipelines, constants and native recording recipe. Copies share
// those artifacts; every create_invocation allocates its own mutable resources.
class Program final {
public:
  Program() noexcept = default;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_ROUTE_VULKAN_API std::span<const TensorDesc>
  input_descriptions() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API std::span<const TensorDesc>
  output_descriptions() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Invocation>
  create_invocation(mem::AllocatorRef result_allocator =
                        mem::system_allocator()) const noexcept;

private:
  friend struct detail::Access;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::ProgramState> state_;
};

// Thread-compatible slot. bind updates idle native bindings and retains input
// owners. submit performs no HYLE planning, shader compilation, timing or
// implementation selection. Managed providers may encode their fixed graph
// and manage private runtime resources when executing it.
class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  Invocation(Invocation &&) noexcept = default;
  Invocation &operator=(Invocation &&) noexcept = default;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  bind(std::span<const Tensor> inputs) noexcept;
  template <class... Inputs>
    requires((std::same_as<Inputs, Tensor> && ...))
  Status bind(const Inputs &...inputs) noexcept {
    const std::array<Tensor, sizeof...(Inputs)> values{inputs...};
    return bind(std::span<const Tensor>(values));
  }
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit() noexcept;

private:
  friend struct detail::Access;
  explicit Invocation(std::shared_ptr<detail::InvocationState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::InvocationState> state_;
};

HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare(const Context &context, const Operation &operation,
        std::span<const Argument> arguments) noexcept;

template <class Descriptor, class... Arguments>
  requires std::constructible_from<Operation, Descriptor> &&
           (std::constructible_from<Argument, Arguments> && ...)
Result<Program> prepare(const Context &context, Descriptor operation,
                        Arguments &&...arguments) noexcept {
  const std::array<Argument, sizeof...(Arguments)> values{
      Argument(std::forward<Arguments>(arguments))...};
  return prepare(context, Operation(operation),
                 std::span<const Argument>(values));
}

// Expert native pipeline with fixed caller destinations. Completion protects
// native work, but does not promise immutable generated output: the application
// decides when those destinations can be overwritten or read externally.
class RecordedProgram final {
public:
  RecordedProgram() noexcept = default;
  RecordedProgram(const RecordedProgram &) = delete;
  RecordedProgram &operator=(const RecordedProgram &) = delete;
  RecordedProgram(RecordedProgram &&) noexcept = default;
  RecordedProgram &operator=(RecordedProgram &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion> submit() noexcept;

private:
  friend class Recorder;
  explicit RecordedProgram(std::shared_ptr<detail::Run> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::Run> state_;
};

// Cold imperative composition of prepared calculations and native SPIR-V.
// No generic DAG, runtime provider lookup or kernel executor is introduced.
class Recorder final {
public:
  Recorder() noexcept = default;
  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;
  Recorder(Recorder &&) noexcept = default;
  Recorder &operator=(Recorder &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Recorder>
  create(const Context &context) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  append(const Program &program, std::span<const Tensor> inputs,
         std::span<const Tensor> outputs) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status
  dispatch(const Kernel &kernel, std::span<const Tensor> bindings,
           std::array<std::uint32_t, 3> groups,
           std::span<const std::byte> push_constants = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<RecordedProgram> finish() noexcept;

private:
  explicit Recorder(std::shared_ptr<detail::RecorderState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::RecorderState> state_;
};

} // namespace hyle::compute::vulkan
#endif
