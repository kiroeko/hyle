#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../compute/completion_internal.h"
#include "../compute/context_state.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/platform/thread.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"
#include "native_context.h"

namespace hyle::compute::cuda::detail {

inline constexpr std::size_t kCudaAllocationAlignment = 256;
inline constexpr std::size_t kEventSlotsPerBlock = 16;

extern const char kMemoryDomainAuthority;

struct ContextState;
struct QueueState;
struct EventSlot;
struct CapturedCommandsState;

struct StatusReference final {
  StatusReference *next = nullptr;
  EventSlot *source = nullptr;
  CapturedCommandsState *capture = nullptr;
};

enum class StreamOwnership : unsigned char {
  kOwned = 0,
  kBorrowed,
  kAdopted,
};

enum class FenceKind : unsigned char {
  kReserved = 0,
  kEvent,
  kStreamFallback,
};

struct EventSlot final {
  QueueState *owner = nullptr;
  cudaEvent_t event = nullptr;
  EventSlot *next = nullptr;
  FenceKind fence = FenceKind::kReserved;
  compute::detail::CompletionLifetime lifetime{};
  Status terminal_hint{};
  std::atomic<std::size_t> references{0};
  unsigned *device_status = nullptr;
  unsigned *host_status = nullptr;
  StatusReference *status_references = nullptr;
  bool status_used = false;
  bool status_copied = false;
  bool report_device_status = false;
};

struct EventBlock final {
  EventBlock *next = nullptr;
  std::size_t initialized = 0;
  std::array<EventSlot, kEventSlotsPerBlock> slots{};
  unsigned *device_status = nullptr;
  unsigned *host_status = nullptr;
  // Cold capacity for imported status sources and captured programs. These
  // are independent of EventSlot's free/reaper links and never grow at submit.
  std::array<StatusReference, 2 * kEventSlotsPerBlock> status_references{};
};

struct ContextState final : util::RefCounted<ContextState> {
  ContextState(Device device_value, int ordinal_value,
               cudaExecutionContext_t execution_value, CUcontext primary_value,
               mem::AllocatorRef allocator_value) noexcept
      : device(device_value), ordinal(ordinal_value),
        execution(execution_value), primary(primary_value),
        allocator(allocator_value) {}

  void destroy() noexcept;

  Device device{};
  int ordinal = -1;
  cudaExecutionContext_t execution = nullptr;
  CUcontext primary = nullptr;
  mem::AllocatorRef allocator{};
  util::SpinLock queues_lock{};
  QueueState *queues = nullptr;
  const void *status_merge_kernel = device_status_merge_kernel();
};

struct CapturedCommandsState final : util::RefCounted<CapturedCommandsState> {
  void destroy() noexcept;
  ContextState *context = nullptr;
  cudaGraphExec_t executable = nullptr;
  mem::AllocatorRef allocator{};
  unsigned *device_status = nullptr;
  std::atomic<bool> status_busy{false};
};

struct QueueState final : util::RefCounted<QueueState> {
  QueueState(ContextState *context_value, cudaStream_t stream_value,
             StreamOwnership ownership_value,
             mem::AllocatorRef allocator_value) noexcept
      : context(context_value), stream(stream_value),
        ownership(ownership_value), allocator(allocator_value) {
    context->incref();
  }

  void destroy() noexcept;
  Status reserve(std::size_t minimum) noexcept;
  Result<EventSlot *> acquire() noexcept;
  void release(EventSlot *slot) noexcept;
  Result<StatusReference *> acquire_status_reference() noexcept;
  void release_status_reference(StatusReference *reference) noexcept;

  ContextState *context = nullptr;
  cudaStream_t stream = nullptr;
  StreamOwnership ownership = StreamOwnership::kOwned;
  mem::AllocatorRef allocator{};
  util::SpinLock slots_lock{};
  EventBlock *blocks = nullptr;
  EventSlot *free_slots = nullptr;
  StatusReference *free_status_references = nullptr;
  std::size_t capacity = 0;
  QueueState *context_next = nullptr;
  // Protected by Compute's one-active-QueueSequence admission contract.
  bool capture_owned = false;
  EventSlot *active_sequence = nullptr;
  unsigned *capture_device_status = nullptr;
  bool capture_status_used = false;
};

class CudaResourceBase : public mem::MemoryResource {
public:
  CudaResourceBase(compute::detail::ContextState *owner, ContextState *context,
                   MemoryKind kind) noexcept;
  ~CudaResourceBase() override;

  MemoryKind kind() const noexcept { return kind_; }
  Device device() const noexcept { return context_->device; }
  int ordinal() const noexcept { return context_->ordinal; }
  ContextState *context() const noexcept { return context_; }

  mem::MemoryProperties properties() const noexcept override;
  Result<mem::ResourceHostMapping>
  map_host(void *native_handle, std::size_t offset, std::size_t size,
           mem::HostAccess access) noexcept override;
  Status unmap_host(void *native_handle, void *token, std::size_t offset,
                    std::size_t size, mem::HostAccess access) noexcept override;
  Status flush(void *native_handle, std::size_t offset,
               std::size_t size) noexcept override;
  Status invalidate(void *native_handle, std::size_t offset,
                    std::size_t size) noexcept override;
  Status read(void *native_handle, std::size_t offset,
              std::span<std::byte> destination) noexcept override;
  Status write(void *native_handle, std::size_t offset,
               std::span<const std::byte> source) noexcept override;
  Status fill(void *native_handle, std::size_t offset, std::size_t size,
              std::byte value) noexcept override;
  Status copy(void *destination_handle, std::size_t destination_offset,
              mem::MemoryResource &source_resource, void *source_handle,
              std::size_t source_offset, std::size_t size) noexcept override;

protected:
  compute::detail::ContextState *owner_ = nullptr;
  ContextState *context_ = nullptr;
  MemoryKind kind_ = MemoryKind::kDevice;
};

CudaResourceBase *resource(mem::MemoryResourceRef value) noexcept;
const CudaResourceBase *resource_const(mem::MemoryResourceRef value) noexcept;
void *byte_pointer(mem::BufferView view) noexcept;
const void *byte_pointer(mem::ConstBufferView view) noexcept;

Result<QueueState *> create_queue_state(ContextState *context,
                                        cudaStream_t stream,
                                        StreamOwnership ownership,
                                        mem::AllocatorRef allocator) noexcept;

Status runtime_status(cudaError_t error) noexcept;
Status driver_status(CUresult error) noexcept;
Status combine_terminal(Status observed, Status hint) noexcept;

compute::detail::EnqueueResult
acquire_device_status(QueueState *queue, unsigned **device_word) noexcept;
void settle_status_references(EventSlot *slot) noexcept;
Status completed_device_status(const EventSlot *slot) noexcept;

Result<CompletionPoll> poll_slot(EventSlot *slot) noexcept;
Status wait_slot(EventSlot *slot) noexcept;
void finish_abandoned(EventSlot *slot, Status observed) noexcept;

class CompletionReaper final {
public:
  CompletionReaper() noexcept = default;
  CompletionReaper(const CompletionReaper &) = delete;
  CompletionReaper &operator=(const CompletionReaper &) = delete;
  ~CompletionReaper();

  Status start() noexcept;
  void retire(EventSlot *slot, compute::detail::CompletionLifetime lifetime,
              Status terminal_hint) noexcept;

  static CompletionReaper &instance() noexcept;

private:
  static void run_entry(void *context) noexcept;
  void run() noexcept;
  EventSlot *take_all() noexcept;
  void append(EventSlot *head, EventSlot *tail) noexcept;

  util::SpinLock lock_{};
  EventSlot *pending_ = nullptr;
  platform::Parker parker_{};
  platform::Thread thread_{};
  std::atomic<bool> stopping_{false};
  bool started_ = false;
};

} // namespace hyle::compute::cuda::detail
