#include "cuda_internal.h"

#include <cudaTypedefs.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hyle/compute/detail/copy_backend.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::cuda::detail {
namespace {
#if defined(HYLE_TESTING)
thread_local CopyNativeHooks copy_hooks;
#endif
using compute::detail::CopyBackendRange;
using compute::detail::EnqueueResult;
template<class T> using Array = std::vector<T, mem::StdAllocator<T>>;

struct BatchProgram final {
  // Exactly the SDK's 13.0 ABI, not the 12.8 failIdx signature. The requested
  // version stays fixed even if the installed driver gains another revision.
  PFN_cuMemcpyBatchAsync_v13000 batch = nullptr;
};
struct BatchSlot final {
  explicit BatchSlot(mem::AllocatorRef allocator) noexcept
      : destinations(mem::StdAllocator<CUdeviceptr>(allocator)),
        sources(mem::StdAllocator<CUdeviceptr>(allocator)),
        sizes(mem::StdAllocator<std::size_t>(allocator)),
        ranges(mem::StdAllocator<CopyBackendRange>(allocator)) {
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  }
  Array<CUdeviceptr> destinations, sources;
  Array<std::size_t> sizes;
  Array<CopyBackendRange> ranges;
  std::size_t count = 0;
  QueueState *queue = nullptr; // borrowed original slot endpoint
  CUmemcpyAttributes attributes{};
  std::size_t attribute_index = 0;
  bool use_batch = false;
};

Status prepare(void *opaque, mem::AllocatorRef allocator, void **output,
               std::size_t *metadata) noexcept {
  auto &context = *static_cast<ContextState *>(opaque);
  ScopedPrimaryContext scope(context.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  void *entry = nullptr;
  CUdriverProcAddressQueryResult query = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  const auto found =
#if defined(HYLE_TESTING)
      copy_hooks.proc_address ? copy_hooks.proc_address("cuMemcpyBatchAsync", &entry, 13000,
                                      CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query) :
#endif
      cuGetProcAddress("cuMemcpyBatchAsync", &entry, 13000,
                                      CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query);
  if (found != CUDA_SUCCESS && found != CUDA_ERROR_NOT_FOUND &&
      found != CUDA_ERROR_NOT_SUPPORTED)
    return driver_status(found);
  auto *storage = allocator.allocate(sizeof(BatchProgram), alignof(BatchProgram));
  if (!storage) return Status::ResourceExhausted("CUDA copy program metadata");
  auto *program = std::construct_at(static_cast<BatchProgram *>(storage));
  if (found == CUDA_SUCCESS && query == CU_GET_PROC_ADDRESS_SUCCESS)
    program->batch = reinterpret_cast<PFN_cuMemcpyBatchAsync_v13000>(entry);
  *output = program;
  *metadata = sizeof(BatchProgram);
  return Status::Ok();
}
void destroy_program(void *opaque, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<BatchProgram *>(opaque);
  std::destroy_at(program);
  allocator.deallocate(program, sizeof(BatchProgram), alignof(BatchProgram));
}
Status create_slot(void *opaque, void *queue_opaque, void *program_opaque,
                   std::uint32_t capacity, std::uint32_t phase_capacity, mem::AllocatorRef allocator,
                   void **output, std::size_t *metadata) noexcept {
  auto &queue = *static_cast<QueueState *>(queue_opaque);
  if (queue.context != opaque)
    return Status::InvalidArgument("CUDA copy queue context mismatch");
  std::size_t existing;
  { std::lock_guard guard(queue.slots_lock); existing = queue.capacity; }
  if (phase_capacity == 0 || phase_capacity > 64 || phase_capacity > SIZE_MAX - existing)
    return Status::ResourceExhausted("CUDA copy phase completion capacity overflow");
  // Add this slot's capacity at cold creation, rather than asking each slot
  // for the same minimum and allowing a second in-flight slot to grow on hot
  // acquire. Rounded native event storage remains an explicit opaque cost.
  HYLE_RETURN_IF_ERROR(queue.reserve(existing + phase_capacity));
  constexpr auto row_bytes = 2 * sizeof(CUdeviceptr) + sizeof(std::size_t) + sizeof(CopyBackendRange);
  if (capacity == 0 || capacity > (SIZE_MAX - sizeof(BatchSlot)) / row_bytes)
    return Status::ResourceExhausted("CUDA copy descriptor capacity overflow");
  auto *storage = allocator.allocate(sizeof(BatchSlot), alignof(BatchSlot));
  if (!storage) return Status::ResourceExhausted("CUDA copy slot metadata");
  auto *slot = std::construct_at(static_cast<BatchSlot *>(storage), allocator);
  try {
    slot->destinations.resize(capacity); slot->sources.resize(capacity);
    slot->sizes.resize(capacity); slot->ranges.resize(capacity);
  } catch (...) {
    std::destroy_at(slot);
    allocator.deallocate(slot, sizeof(BatchSlot), alignof(BatchSlot));
    return Status::ResourceExhausted("CUDA copy descriptor metadata");
  }
  // Imported legacy queues retain the original same-stream memcpy route.
  slot->use_batch = static_cast<BatchProgram *>(program_opaque)->batch != nullptr &&
                    queue.stream != nullptr && queue.stream != cudaStreamLegacy;
  slot->queue = &queue;
  *output = slot;
  *metadata = sizeof(BatchSlot) + (slot->destinations.capacity() + slot->sources.capacity()) * sizeof(CUdeviceptr) +
              slot->sizes.capacity() * sizeof(std::size_t) + slot->ranges.capacity() * sizeof(CopyBackendRange);
  return Status::Ok();
}
void destroy_slot(void *opaque, mem::AllocatorRef allocator) noexcept {
  auto *slot = static_cast<BatchSlot *>(opaque);
  std::destroy_at(slot);
  allocator.deallocate(slot, sizeof(BatchSlot), alignof(BatchSlot));
}
Result<compute::detail::PhysicalCopyResource> physical(void *opaque,
                                                       mem::ConstBufferView view) noexcept {
  const auto *selected = resource_const(view.resource());
  if (!view.valid() || !selected || selected->context() != opaque)
    return std::unexpected(Status::InvalidArgument("CUDA copy binding context mismatch"));
  const auto address = reinterpret_cast<std::uintptr_t>(byte_pointer(view));
  if (view.size() > SIZE_MAX - address)
    return std::unexpected(Status::OutOfRange("CUDA copy physical coordinate overflow"));
  // Device, pinned and managed pointers share this exact CUDA context's UVA.
  // Wrapper/resource addresses never substitute for the actual byte address.
  return compute::detail::PhysicalCopyResource{
      reinterpret_cast<std::uintptr_t>(opaque), 0, address, view.size(), 1};
}
Status bind_slot(void *opaque, void *, void *slot_opaque,
                 std::span<const CopyBackendRange> ranges) noexcept {
  auto &slot = *static_cast<BatchSlot *>(slot_opaque);
  slot.count = 0; // Any rejected update invalidates the preceding replay.
  if (ranges.size() > slot.ranges.size())
    return Status::ResourceExhausted("CUDA copy replay descriptor capacity");
  // Full preflight precedes overwriting the retained, fixed descriptor arrays.
  for (const auto &range : ranges) {
    HYLE_ASSIGN_OR_RETURN(auto source, physical(opaque, range.source));
    HYLE_ASSIGN_OR_RETURN(auto destination, physical(opaque, range.destination));
    if (source.capacity != destination.capacity)
      return Status::InvalidArgument("CUDA copy descriptor sizes differ");
  }
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    slot.ranges[index] = ranges[index];
    slot.sources[index] = reinterpret_cast<CUdeviceptr>(byte_pointer(ranges[index].source));
    slot.destinations[index] = reinterpret_cast<CUdeviceptr>(byte_pointer(ranges[index].destination));
    slot.sizes[index] = ranges[index].source.size();
  }
  slot.count = ranges.size();
  return Status::Ok();
}
EnqueueResult enqueue(void *opaque, void *queue_opaque, void *program_opaque,
                      void *slot_opaque, std::size_t first, std::size_t count,
                      std::uint64_t *calls) noexcept {
  auto &queue = *static_cast<QueueState *>(queue_opaque);
  auto &slot = *static_cast<BatchSlot *>(slot_opaque);
  if (&queue != slot.queue || queue.context != opaque || first > slot.count || count > slot.count - first)
    return EnqueueResult::rejected(Status::InvalidArgument("CUDA copy phase slot is invalid"));
  if (count == 0) return EnqueueResult::no_op();
  if (slot.use_batch) {
    ScopedPrimaryContext scope(queue.context->primary);
    if (!scope) return EnqueueResult::rejected(scope.status());
    ++*calls;
    const auto result = static_cast<BatchProgram *>(program_opaque)->batch(
        slot.destinations.data() + first, slot.sources.data() + first,
        slot.sizes.data() + first, count, &slot.attributes,
        &slot.attribute_index, 1, reinterpret_cast<CUstream>(queue.stream));
    // A native error may report earlier asynchronous work. Conservatively
    // preserve this sequence's real prefix; never retry using another route.
    return EnqueueResult::accepted_command(driver_status(result));
  }
  for (std::size_t index = first; index < first + count; ++index) {
    ++*calls;
    const auto result =
#if defined(HYLE_TESTING)
        copy_hooks.memcpy_async ? copy_hooks.memcpy_async(byte_pointer(slot.ranges[index].destination),
        byte_pointer(slot.ranges[index].source), slot.sizes[index], cudaMemcpyDefault, queue.stream) :
#endif
        cudaMemcpyAsync(byte_pointer(slot.ranges[index].destination),
        byte_pointer(slot.ranges[index].source), slot.sizes[index],
        cudaMemcpyDefault, queue.stream);
    if (result != cudaSuccess) return EnqueueResult::accepted_command(runtime_status(result));
  }
  return EnqueueResult::accepted_command();
}
const compute::detail::CopyBackendVTable table{
    &prepare, &destroy_program, &create_slot, &destroy_slot, &physical, &bind_slot, &enqueue};
} // namespace

#if defined(HYLE_TESTING)
CopyNativeHooks set_copy_native_hooks(CopyNativeHooks hooks) noexcept {
  return std::exchange(copy_hooks, hooks);
}
#endif
const compute::detail::CopyBackendVTable *copy_batch_interface() noexcept { return &table; }
} // namespace hyle::compute::cuda::detail
