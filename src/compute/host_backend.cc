#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "backend.h"
#include "hyle/compute/types.h"
#include "hyle/error/status.h"

// The HOST fallback backend (coding-conventions §10: an independent composition
// unit, one TU, filling the manual vtable). It is ALWAYS present -- it lets the
// whole compute completion contract compile and run on any platform with no GPU
// (mac, CI, a machine with CUDA not built) and is the DETERMINISTIC backend
// compute-03 will use to test fiber wakeup without a real device.
//
// Semantics: a host "stream" runs work INLINE at enqueue time (there is no async
// engine), so an Event recorded on it is ALWAYS immediately kReady. This honors the
// same record/query/sync API the CUDA backend does -- the caller's code path is
// identical -- it just never observes NotReady. device_alloc is plain malloc;
// memcpy_async is a synchronous memcpy. No per-object heap allocation: the stream/
// event native handles are non-null SENTINELS (there is no per-instance state to
// keep -- inline execution means an event carries no pending work).

namespace hyle::compute {
namespace {

// Non-null sentinels: the dispatch layer only checks native != null for validity
// and passes it back to us verbatim; the host backend keeps no per-handle state,
// so a distinct address per kind is all we need (and it's free -- no allocation).
char g_host_stream_tag;
char g_host_event_tag;
void* const kHostStream = &g_host_stream_tag;
void* const kHostEvent = &g_host_event_tag;

std::size_t host_device_count() noexcept { return 1; }

DeviceType host_device_type(int) noexcept { return DeviceType::kCpu; }

const char* host_device_name(int) noexcept { return "cpu"; }

DeviceCaps host_device_caps(int) noexcept {
  // No real asynchrony, but host memory is directly usable (host_buffer) and the
  // event contract is honored synchronously (events=true).
  return DeviceCaps{/*async=*/false, /*host_buffer=*/true, /*events=*/true};
}

void host_device_memory(int, std::uint64_t* free, std::uint64_t* total) noexcept {
  // Host RAM is not meaningfully "device memory"; report 0 rather than probing the
  // OS (informational only, and compute-01 doesn't need it). A later milestone can
  // fill this from platform sysinfo if a consumer wants it.
  if (free != nullptr) *free = 0;
  if (total != nullptr) *total = 0;
}

Status host_stream_create(int, void** out_native) noexcept {
  *out_native = kHostStream;
  return Status();  // OK
}

void host_stream_destroy(void*) noexcept {}  // sentinel: nothing to free

Status host_stream_sync(void*) noexcept { return Status(); }  // inline: already done

Status host_event_create(int, void** out_native) noexcept {
  *out_native = kHostEvent;
  return Status();
}

void host_event_destroy(void*) noexcept {}

void host_event_record(void*, void*) noexcept {}  // inline: nothing pending to mark

EventStatus host_event_query(void*) noexcept {
  return EventStatus::kReady;  // inline execution => always complete
}

void host_event_sync(void*) noexcept {}  // already complete

Status host_device_alloc(int, std::size_t bytes, void** out_ptr) noexcept {
  void* p = std::malloc(bytes == 0 ? 1 : bytes);  // malloc(0) is impl-defined; give 1
  if (p == nullptr) return Status::ResourceExhausted("host: device_alloc");
  *out_ptr = p;
  return Status();
}

void host_device_free(int, void* ptr) noexcept { std::free(ptr); }

Status host_memcpy_async(void*, void* dst, const void* src,
                         std::size_t bytes) noexcept {
  std::memcpy(dst, src, bytes);  // synchronous on host
  return Status();
}

// ---- buffer (compute-02) ----------------------------------------------------
// On the host backend both BufferKinds are just plain RAM (there is no separate
// device, and no page-locking benefit without a GPU DMA engine). is_host()=true for
// both -- the memory is directly host-dereferenceable. set/get/copy/clear are plain
// memcpy/memset (the "device" pointer IS a host pointer here).

std::size_t host_buffer_alignment(BufferKind) noexcept {
  return alignof(std::max_align_t);
}

std::size_t host_buffer_max_size(int, BufferKind) noexcept {
  return SIZE_MAX;  // bounded only by the system allocator
}

bool host_buffer_is_host(BufferKind) noexcept {
  return true;  // plain RAM either way -- directly dereferenceable
}

Status host_buffer_alloc(int, BufferKind, std::size_t bytes,
                         void** out_ptr) noexcept {
  void* p = std::malloc(bytes == 0 ? 1 : bytes);
  if (p == nullptr) return Status::ResourceExhausted("host: buffer_alloc");
  *out_ptr = p;
  return Status();
}

void host_buffer_free(int, BufferKind, void* ptr) noexcept { std::free(ptr); }

Status host_buffer_set(void* dst_base, std::size_t offset, const void* src,
                       std::size_t size) noexcept {
  std::memcpy(static_cast<char*>(dst_base) + offset, src, size);
  return Status();
}

Status host_buffer_get(const void* src_base, std::size_t offset, void* dst,
                       std::size_t size) noexcept {
  std::memcpy(dst, static_cast<const char*>(src_base) + offset, size);
  return Status();
}

Status host_buffer_copy(void* dst_base, std::size_t dst_off, const void* src_base,
                        std::size_t src_off, std::size_t size) noexcept {
  std::memcpy(static_cast<char*>(dst_base) + dst_off,
              static_cast<const char*>(src_base) + src_off, size);
  return Status();
}

Status host_buffer_clear(void* base, std::size_t offset, std::uint8_t value,
                         std::size_t size) noexcept {
  std::memset(static_cast<char*>(base) + offset, value, size);
  return Status();
}

constexpr ComputeBackend kHostBackend = {
    /*name=*/"host",
    /*device_count=*/host_device_count,
    /*device_type=*/host_device_type,
    /*device_name=*/host_device_name,
    /*device_caps=*/host_device_caps,
    /*device_memory=*/host_device_memory,
    /*stream_create=*/host_stream_create,
    /*stream_destroy=*/host_stream_destroy,
    /*stream_sync=*/host_stream_sync,
    /*event_create=*/host_event_create,
    /*event_destroy=*/host_event_destroy,
    /*event_record=*/host_event_record,
    /*event_query=*/host_event_query,
    /*event_sync=*/host_event_sync,
    /*device_alloc=*/host_device_alloc,
    /*device_free=*/host_device_free,
    /*memcpy_async=*/host_memcpy_async,
    /*buffer_alignment=*/host_buffer_alignment,
    /*buffer_max_size=*/host_buffer_max_size,
    /*buffer_is_host=*/host_buffer_is_host,
    /*buffer_alloc=*/host_buffer_alloc,
    /*buffer_free=*/host_buffer_free,
    /*buffer_set=*/host_buffer_set,
    /*buffer_get=*/host_buffer_get,
    /*buffer_copy=*/host_buffer_copy,
    /*buffer_clear=*/host_buffer_clear,
};

}  // namespace

const ComputeBackend* host_vtable() noexcept { return &kHostBackend; }

}  // namespace hyle::compute
