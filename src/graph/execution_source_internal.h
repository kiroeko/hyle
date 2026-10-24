#pragma once

#include "hyle/graph/detail/execution_source.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/compute/tensor.h"
#include "hyle/graph/region_provider.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph { struct RegionProviderVTable; struct ProviderRegionPrepared; }

namespace hyle::graph::detail {
using SourceCall = Status (*)(void *) noexcept;
// Immutable cold transfer only: compiled constants or an owned request-layout
// version. Exact source/destination allocations stay owned and charged by the
// caller through successful native retirement; this is never hot replay work.
struct ConstantCopy final {
  compute::TensorView destination;
  compute::ConstTensorView source;
};
// Cold source-port facts, shared unchanged by transfer qualification and
// preparation. Resource accessibility cannot supply a future native handle.
// The actual producer source retains its original owner; external caller
// bindings have no producer and keep the default caller/TensorView contract.
struct CopySourceDescription final {
  compute::TensorDesc description;
  mem::MemoryResourceRef resource{};
  RegionOutputContract value{};
  ExecutionSource producer{};
};
struct ExecutionSourceState;
struct ExecutionSourceVTable final {
  void (*destroy)(ExecutionSourceState *) noexcept;
  bool (*is_open)(const ExecutionSourceState &) noexcept;
  bool (*same_device)(const ExecutionSourceState &, const ExecutionSourceState &) noexcept;
  compute::BackendId (*backend_id)(const ExecutionSourceState &) noexcept;
  compute::DeviceClass (*device_class)(const ExecutionSourceState &) noexcept;
  Result<compute::DeviceProperties> (*device_properties)(const ExecutionSourceState &) noexcept;
  Status (*validate_resource)(const ExecutionSourceState &, mem::MemoryResourceRef) noexcept;
  Status (*call)(const ExecutionSourceState &, SourceCall, void *) noexcept;
  const compute::detail::ContextRef *(*compute)(const ExecutionSourceState &) noexcept;
  // Actual retained native owner context, typed privately by its capability
  // (for example a shared Runtime holder). Borrowed only while source lives;
  // prepared artifacts retain the actual owner, never a no-op shared pointer.
  void *(*native_context)(const ExecutionSourceState &) noexcept;
  const RegionProviderVTable *(*region_provider)(const ExecutionSourceState &) noexcept;
  // Pure route qualification precedes all accepted bytes. A selected batch
  // succeeds only after its original native work is terminal. On failure the
  // original native owners protect any accepted prefix through retirement;
  // failure never permits retrying possibly accepted work elsewhere.
  Result<bool> (*can_copy_constant)(const ExecutionSourceState &,
                                  const ConstantCopy &) noexcept;
  Status (*copy_constants)(const ExecutionSourceState &,
                          std::span<const ConstantCopy>, mem::AllocatorRef) noexcept;
  // Batch-reused physical scratch, excluding metadata already charged through
  // the supplied allocator and native driver rounding/control overhead.
  Result<std::size_t> (*constant_copy_scratch_bytes)(const ExecutionSourceState &,
      std::span<const ConstantCopy>) noexcept;
  // Readonly imports may have an owning resource distinct from the nominated
  // allocation resource. Only the actual native source may prove that family;
  // this never changes caller-output, leaf, copy or cold scheduling contracts.
  bool (*matches_read_resource)(const ExecutionSourceState &,
      mem::MemoryResourceRef nominated, mem::MemoryResourceRef actual) noexcept = nullptr;
  // Replay transfers are a separate native capability from cold constants.
  // Resources are qualified without allocation; the selected artifact uses
  // the same original invocation/binding/completion protocol as a region.
  Result<bool> (*can_copy)(const ExecutionSourceState &,
      const CopySourceDescription &, mem::MemoryResourceRef destination) noexcept = nullptr;
  Status (*prepare_copy)(const ExecutionSourceState &, const CopySourceDescription &,
      mem::MemoryResourceRef destination,
      mem::AllocatorRef, ProviderRegionPrepared *) noexcept = nullptr;
};
struct ExecutionSourceState : util::RefCounted<ExecutionSourceState> {
  ExecutionSourceState(const ExecutionSourceVTable *table_value,
      mem::AllocatorRef allocator_value, const void *identity_value) noexcept
      : table(table_value), allocator(allocator_value), identity(identity_value) {}
  void destroy() noexcept { table->destroy(this); }
  const ExecutionSourceVTable *table;
  mem::AllocatorRef allocator;
  const void *identity;
};

struct ExecutionSourceAccess final {
  static ExecutionSource adopt(ExecutionSourceState *state) noexcept { return ExecutionSource(state); }
  static ExecutionSourceState *state(const ExecutionSource &source) noexcept { return source.state_; }
  static const void *identity_key(const ExecutionSource &source) noexcept {
    return source.state_ ? source.state_->identity : nullptr;
  }
  HYLE_GRAPH_API static Result<const compute::detail::ContextRef *>
  compute(const ExecutionSource &) noexcept;
  HYLE_GRAPH_API static bool owns(const ExecutionSource &, const compute::Queue &) noexcept;
  HYLE_GRAPH_API static bool owns(const ExecutionSource &,
      const compute::detail::QueueRef &) noexcept;
  HYLE_GRAPH_API static bool owns(const ExecutionSource &,
      const compute::detail::ProviderExtensionRef &) noexcept;
  HYLE_GRAPH_API static Status call(const ExecutionSource &, SourceCall, void *) noexcept;
  HYLE_GRAPH_API static Result<void *> native_context(const ExecutionSource &) noexcept;
  HYLE_GRAPH_API static Result<GraphRegionSource> region(const ExecutionSource &) noexcept;
  HYLE_GRAPH_API static Result<bool> can_copy_constant(
      const ExecutionSource &, const ConstantCopy &) noexcept;
  HYLE_GRAPH_API static Status copy_constants(const ExecutionSource &,
      std::span<const ConstantCopy>, mem::AllocatorRef) noexcept;
  HYLE_GRAPH_API static Result<std::size_t> constant_copy_scratch_bytes(
      const ExecutionSource &, std::span<const ConstantCopy>) noexcept;
  HYLE_GRAPH_API static bool matches_read_resource(const ExecutionSource &,
      mem::MemoryResourceRef nominated, mem::MemoryResourceRef actual) noexcept;
  HYLE_GRAPH_API static Result<bool> can_copy(const ExecutionSource &,
      const CopySourceDescription &, mem::MemoryResourceRef destination) noexcept;
  HYLE_GRAPH_API static Status prepare_copy(const ExecutionSource &,
      const CopySourceDescription &, mem::MemoryResourceRef destination, mem::AllocatorRef,
      ProviderRegionPrepared *) noexcept;
};

} // namespace hyle::graph::detail
