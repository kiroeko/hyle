#include "hyle/compute/device.h"

#include <cstdint>
#include <cstdlib>

#include "backend.h"
#include "hyle/compute/buffer.h"
#include "hyle/compute/device_pool.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/pinned_pool.h"
#include "hyle/compute/stream.h"
#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

// compute-01 dispatch layer: the public Device/Stream/Event API forwarding through
// the manual backend vtable (backend.h). Pure C++, no CUDA -- the CUDA calls live
// in cuda_backend.cu (nvcc), the inline fallback in host_backend.cc. This file
// only picks a backend (cold path) and forwards.

namespace hyle::compute {
namespace {

// The process-wide device table, built once (Meyers singleton -- charter §, no
// leaky global). Each entry maps a GLOBAL device index to {backend vtable, the
// index LOCAL to that backend}. GPUs (CUDA) come first so default_device() prefers
// them; the host CPU device is always last, so the table is never empty.
struct DeviceEntry {
  const ComputeBackend* vt;
  int local;  // ordinal within vt's own device list
};

struct DeviceTable {
  static constexpr std::size_t kMax = 32;
  DeviceEntry entries[kMax];
  std::size_t count = 0;

  DeviceTable() noexcept {
    // GPUs first (CUDA backend), if compiled in and any device is present.
#if defined(HYLE_COMPUTE_CUDA)
    add_backend(cuda_vtable());
#endif
    // Vulkan compute backend (compute-06) next: cross-vendor GPU. On NVIDIA it ranks
    // below CUDA (CUDA preferred for N-cards); on AMD/Intel/Apple it's the GPU path.
    // vulkan_vtable() may be null (no loader/device at runtime) -- add_backend tolerates.
#if defined(HYLE_COMPUTE_VULKAN)
    add_backend(vulkan_vtable());
#endif
    // Host CPU fallback always last -- guarantees >= 1 device.
    add_backend(host_vtable());
  }

  void add_backend(const ComputeBackend* vt) noexcept {
    if (vt == nullptr) return;
    std::size_t n = vt->device_count();
    for (std::size_t i = 0; i < n && count < kMax; ++i) {
      entries[count++] = DeviceEntry{vt, static_cast<int>(i)};
    }
  }
};

const DeviceTable& table() noexcept {
  static const DeviceTable t;
  return t;
}

// Resolve a public Device (whose ordinal_ is the GLOBAL index) to its backend +
// local ordinal. The global index is minted only by enumerate()/default_device(),
// so it is always in range here -- an out-of-range Device is a caller bug.
DeviceEntry resolve(const Device& d) noexcept {
  const DeviceTable& t = table();
  int g = d.ordinal();
  HYLE_FATAL_ASSERT(g >= 0 && static_cast<std::size_t>(g) < t.count);
  return t.entries[g];
}

}  // namespace

// ---- Device -----------------------------------------------------------------

const char* device_type_name(DeviceType t) noexcept {
  switch (t) {
    case DeviceType::kCpu: return "cpu";
    case DeviceType::kGpu: return "gpu";
    case DeviceType::kIGpu: return "igpu";
  }
  return "unknown";  // unreachable for the closed enum; keeps the compiler happy
}

std::size_t enumerate(Device* out, std::size_t max) noexcept {
  const DeviceTable& t = table();
  for (std::size_t i = 0; i < t.count && i < max; ++i) {
    const DeviceEntry& e = t.entries[i];
    out[i] = Device(e.vt->device_type(e.local), static_cast<int>(i));
  }
  return t.count;
}

Device default_device() noexcept {
  // Entry 0 is the first GPU if any (CUDA added first), else the host CPU.
  const DeviceTable& t = table();
  const DeviceEntry& e = t.entries[0];
  return Device(e.vt->device_type(e.local), 0);
}

const char* Device::name() const noexcept {
  DeviceEntry e = resolve(*this);
  return e.vt->device_name(e.local);
}

const char* Device::backend() const noexcept {
  DeviceEntry e = resolve(*this);
  return e.vt->name;
}

DeviceCaps Device::caps() const noexcept {
  DeviceEntry e = resolve(*this);
  return e.vt->device_caps(e.local);
}

void Device::memory(std::uint64_t* free, std::uint64_t* total) const noexcept {
  DeviceEntry e = resolve(*this);
  e.vt->device_memory(e.local, free, total);
}

// ---- Stream -----------------------------------------------------------------
// State is inline {vt_, native_, local_} (no pimpl -- see stream.h perf note).

Stream::Stream(Stream&& o) noexcept
    : device_(o.device_), vt_(o.vt_), native_(o.native_), local_(o.local_) {
  o.vt_ = nullptr;
  o.native_ = nullptr;
}

Stream& Stream::operator=(Stream&& o) noexcept {
  if (this != &o) {
    if (vt_ != nullptr) vt_->stream_destroy(native_);
    device_ = o.device_;
    vt_ = o.vt_;
    native_ = o.native_;
    local_ = o.local_;
    o.vt_ = nullptr;
    o.native_ = nullptr;
  }
  return *this;
}

Stream::~Stream() {
  if (vt_ != nullptr) vt_->stream_destroy(native_);
}

Result<Stream> Stream::create(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  void* native = nullptr;
  Status st = e.vt->stream_create(e.local, &native);
  if (!st.ok()) return std::unexpected(std::move(st));
  Stream s;
  s.device_ = device;
  s.vt_ = e.vt;
  s.native_ = native;
  s.local_ = e.local;
  return s;
}

void Stream::record(Event& event) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr && event.vt_ != nullptr);
  // Contract: stream and event on the SAME backend (same device). A cross-device
  // record is a caller bug, not a runtime error.
  HYLE_FATAL_ASSERT(vt_ == event.vt_ && local_ == event.local_);
  // A timing event (create_timing) writes a device timestamp -> its own vtable slot; the
  // plain completion event stays on the cheap no-timing path. One bool, cold vs GPU time.
  if (event.timing_) vt_->event_record_timing(native_, event.native_);
  else vt_->event_record(native_, event.native_);
}

Status Stream::sync() noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return vt_->stream_sync(native_);
}

// ---- Event ------------------------------------------------------------------

namespace {
// Destroy an event's native handle via the right slot: a timing event may have a distinct
// native layout (Vulkan) so it MUST use event_destroy_timing, not event_destroy.
void destroy_event(const ComputeBackend* vt, void* native, bool timing) noexcept {
  if (vt == nullptr) return;
  if (timing) vt->event_destroy_timing(native);
  else vt->event_destroy(native);
}
}  // namespace

Event::Event(Event&& o) noexcept
    : vt_(o.vt_), native_(o.native_), local_(o.local_), timing_(o.timing_) {
  o.vt_ = nullptr;
  o.native_ = nullptr;
}

Event& Event::operator=(Event&& o) noexcept {
  if (this != &o) {
    destroy_event(vt_, native_, timing_);
    vt_ = o.vt_;
    native_ = o.native_;
    local_ = o.local_;
    timing_ = o.timing_;
    o.vt_ = nullptr;
    o.native_ = nullptr;
  }
  return *this;
}

Event::~Event() {
  destroy_event(vt_, native_, timing_);
}

Result<Event> Event::create(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  void* native = nullptr;
  Status st = e.vt->event_create(e.local, &native);
  if (!st.ok()) return std::unexpected(std::move(st));
  Event ev;
  ev.vt_ = e.vt;
  ev.native_ = native;
  ev.local_ = e.local;
  return ev;
}

Result<Event> Event::create_timing(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  // A backend without a timing slot (none currently, but future backends may skip it)
  // reports Unimplemented rather than crashing on a null fn-ptr.
  if (e.vt->event_create_timing == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no GPU timing"));
  void* native = nullptr;
  Status st = e.vt->event_create_timing(e.local, &native);
  if (!st.ok()) return std::unexpected(std::move(st));
  Event ev;
  ev.vt_ = e.vt;
  ev.native_ = native;
  ev.local_ = e.local;
  ev.timing_ = true;  // Stream::record routes this to event_record_timing
  return ev;
}

EventStatus Event::query() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return timing_ ? vt_->event_query_timing(native_) : vt_->event_query(native_);
}

void Event::sync() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  if (timing_) vt_->event_sync_timing(native_);
  else vt_->event_sync(native_);
}

Result<std::uint64_t> elapsed_ns(const Event& start, const Event& stop) noexcept {
  HYLE_FATAL_ASSERT(start.vt_ != nullptr && stop.vt_ != nullptr);
  // Both events must be from the same backend (cross-backend elapsed is meaningless).
  if (start.vt_ != stop.vt_)
    return std::unexpected(Status::InvalidArgument("elapsed_ns: events on different backends"));
  if (start.vt_->event_elapsed_ns == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no GPU timing"));
  std::uint64_t ns = 0;
  Status st = start.vt_->event_elapsed_ns(start.native_, stop.native_, &ns);
  if (!st.ok()) return std::unexpected(std::move(st));
  return ns;
}

// ---- BufferType -------------------------------------------------------------

BufferType device_buffer_type(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  return BufferType(e.vt, e.local, BufferKind::kDevice);
}

BufferType host_buffer_type(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  return BufferType(e.vt, e.local, BufferKind::kHostPinned);
}

std::size_t BufferType::alignment() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return vt_->buffer_alignment(kind_);
}

std::size_t BufferType::max_size() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return vt_->buffer_max_size(ordinal_, kind_);
}

bool BufferType::is_host() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return vt_->buffer_is_host(kind_);
}

// ---- Buffer -----------------------------------------------------------------
// Inline state {vt_, ptr_, size_, ordinal_, kind_, is_host_} -- no pimpl/heap.

Buffer::Buffer(Buffer&& o) noexcept
    : vt_(o.vt_),
      ptr_(o.ptr_),
      size_(o.size_),
      ordinal_(o.ordinal_),
      kind_(o.kind_),
      is_host_(o.is_host_) {
  o.vt_ = nullptr;
  o.ptr_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    if (vt_ != nullptr) vt_->buffer_free(ordinal_, kind_, ptr_);
    vt_ = o.vt_;
    ptr_ = o.ptr_;
    size_ = o.size_;
    ordinal_ = o.ordinal_;
    kind_ = o.kind_;
    is_host_ = o.is_host_;
    o.vt_ = nullptr;
    o.ptr_ = nullptr;
  }
  return *this;
}

Buffer::~Buffer() {
  if (vt_ != nullptr) vt_->buffer_free(ordinal_, kind_, ptr_);
}

Result<Buffer> Buffer::create(const BufferType& type, std::size_t bytes) noexcept {
  HYLE_FATAL_ASSERT(type.valid());
  // Caller contract: a single Buffer must fit the backend's max single-allocation
  // size; a larger logical buffer is the caller's split into multiple Buffers.
  HYLE_FATAL_ASSERT(bytes <= type.vt_->buffer_max_size(type.ordinal_, type.kind_));
  void* ptr = nullptr;
  Status st = type.vt_->buffer_alloc(type.ordinal_, type.kind_, bytes, &ptr);
  if (!st.ok()) return std::unexpected(std::move(st));
  Buffer b;
  b.vt_ = type.vt_;
  b.ptr_ = ptr;
  b.size_ = bytes;
  b.ordinal_ = type.ordinal_;
  b.kind_ = type.kind_;
  b.is_host_ = type.vt_->buffer_is_host(type.kind_);
  return b;
}

Status Buffer::set(std::size_t offset, const void* src, std::size_t size) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  // Bounds (partial-write, KV-cache). Subtract-then-compare, NOT `offset + size <= size_`:
  // the sum can wrap size_t (e.g. offset=SIZE_MAX-2, size=10 -> 7) and let an out-of-range
  // write pass the check, then overrun in the backend memcpy. size_ - offset can't wrap
  // once offset <= size_ holds.
  HYLE_FATAL_ASSERT(offset <= size_ && size <= size_ - offset);
  return vt_->buffer_set(ptr_, offset, src, size);
}

Status Buffer::get(std::size_t offset, void* dst, std::size_t size) const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  HYLE_FATAL_ASSERT(offset <= size_ && size <= size_ - offset);  // wrap-safe (see set())
  return vt_->buffer_get(ptr_, offset, dst, size);
}

Status Buffer::copy_from(const Buffer& src, std::size_t dst_offset,
                         std::size_t src_offset, std::size_t size) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr && src.vt_ != nullptr);
  // Same-backend copy (cross-backend is a later concern). A cross-backend copy is a
  // caller bug here, not a runtime fallback.
  HYLE_FATAL_ASSERT(vt_ == src.vt_);
  HYLE_FATAL_ASSERT(dst_offset <= size_ && size <= size_ - dst_offset &&      // wrap-safe
                    src_offset <= src.size_ && size <= src.size_ - src_offset);
  return vt_->buffer_copy(ptr_, dst_offset, src.ptr_, src_offset, size);
}

Status Buffer::clear(std::size_t offset, std::uint8_t value,
                     std::size_t size) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  HYLE_FATAL_ASSERT(offset <= size_ && size <= size_ - offset);  // wrap-safe (see set())
  return vt_->buffer_clear(ptr_, offset, value, size);
}

// ---- pinned staging (compute-02b): REUSE mem/, don't reinvent ----------------
// pinned_allocator wraps the backend's pinned (cudaMallocHost) alloc/free as a
// mem::Allocator, so a pooled pinned block pool is just mem::MemoryArena over it and
// a contiguous staging region is just mem::Arena -- all mem/ reused. state = the
// backend vtable (pinned host RAM is NOT device-bound -- cudaMallocHost is process-
// wide host memory usable by any GPU -- so no ordinal is needed; ordinal 0 is passed
// to the kind-tagged buffer_alloc). alignment is ignored: cudaMallocHost returns
// page-aligned memory (>=512B), far exceeding any scalar alignment; the host
// fallback's malloc is max_align. The Device/backend must outlive the allocator.

namespace {
void* pinned_alloc_fn(void* state, std::size_t size, std::size_t /*alignment*/) noexcept {
  auto* vt = static_cast<const ComputeBackend*>(state);
  void* p = nullptr;
  Status st = vt->buffer_alloc(0, BufferKind::kHostPinned, size == 0 ? 1 : size, &p);
  return st.ok() ? p : nullptr;  // mem::Allocator contract: nullptr on failure
}
void pinned_free_fn(void* state, void* ptr, std::size_t /*size*/,
                    std::size_t /*alignment*/) noexcept {
  if (ptr == nullptr) return;
  auto* vt = static_cast<const ComputeBackend*>(state);
  vt->buffer_free(0, BufferKind::kHostPinned, ptr);
}
void* pinned_realloc_fn(void* /*state*/, void* /*ptr*/, std::size_t /*old*/,
                        std::size_t /*new_size*/, std::size_t /*align*/) noexcept {
  // Pinned staging is bump/pool use (mem::Arena/MemoryArena never realloc a backing
  // slab); a realloc on pinned memory is not a supported path -- return nullptr.
  return nullptr;
}
}  // namespace

mem::Allocator pinned_allocator(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  // state is the const vtable pointer; cast away const for the void* state slot
  // (the fns cast it back to const -- they never mutate the vtable).
  return mem::Allocator(&pinned_alloc_fn, &pinned_free_fn, &pinned_realloc_fn,
                        const_cast<ComputeBackend*>(e.vt));
}

// ---- PinnedStaging -----------------------------------------------------------

PinnedStaging::PinnedStaging(PinnedStaging&& o) noexcept
    : vt_(o.vt_), ordinal_(o.ordinal_), base_(o.base_), cap_(o.cap_), used_(o.used_) {
  o.vt_ = nullptr;
  o.base_ = nullptr;
}

PinnedStaging& PinnedStaging::operator=(PinnedStaging&& o) noexcept {
  if (this != &o) {
    if (base_ != nullptr) vt_->buffer_free(ordinal_, BufferKind::kHostPinned, base_);
    vt_ = o.vt_;
    ordinal_ = o.ordinal_;
    base_ = o.base_;
    cap_ = o.cap_;
    used_ = o.used_;
    o.vt_ = nullptr;
    o.base_ = nullptr;
  }
  return *this;
}

PinnedStaging::~PinnedStaging() {
  if (base_ != nullptr) vt_->buffer_free(ordinal_, BufferKind::kHostPinned, base_);
}

Result<PinnedStaging> PinnedStaging::create(const Device& device,
                                            std::size_t bytes) noexcept {
  DeviceEntry e = resolve(device);
  void* p = nullptr;
  Status st = e.vt->buffer_alloc(e.local, BufferKind::kHostPinned,
                                 bytes == 0 ? 1 : bytes, &p);
  if (!st.ok()) return std::unexpected(std::move(st));
  PinnedStaging s;
  s.vt_ = e.vt;
  s.ordinal_ = e.local;
  s.base_ = p;
  s.cap_ = bytes == 0 ? 1 : bytes;
  s.used_ = 0;
  return s;
}

void* PinnedStaging::allocate(std::size_t bytes, std::size_t alignment) noexcept {
  HYLE_FATAL_ASSERT(base_ != nullptr);
  // Bump, aligned, from the pinned block (contiguous -> lets commit_to batch).
  std::uintptr_t cur = reinterpret_cast<std::uintptr_t>(base_) + used_;
  std::uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);
  std::size_t new_used = (aligned - reinterpret_cast<std::uintptr_t>(base_)) + bytes;
  if (new_used > cap_) return nullptr;  // full
  used_ = new_used;
  return reinterpret_cast<void*>(aligned);
}

Status PinnedStaging::commit_to(Buffer& dst, std::size_t dst_offset) noexcept {
  HYLE_FATAL_ASSERT(base_ != nullptr);
  // One H2D copy of the whole contiguous prepared run = the batched submit.
  return dst.set(dst_offset, base_, used_);
}

// ---- DevicePool (compute-02c): VRAM sub-allocator dispatch -------------------
// Forwards to the backend's stateful pool (VMM or leg, ported from ggml). The pool
// handle is opaque; DevicePool holds {vt, handle, ordinal} inline.

DevicePool::DevicePool(DevicePool&& o) noexcept
    : vt_(o.vt_), handle_(o.handle_), ordinal_(o.ordinal_) {
  o.vt_ = nullptr;
  o.handle_ = nullptr;
}

DevicePool& DevicePool::operator=(DevicePool&& o) noexcept {
  if (this != &o) {
    if (vt_ != nullptr) vt_->pool_destroy(handle_);
    vt_ = o.vt_;
    handle_ = o.handle_;
    ordinal_ = o.ordinal_;
    o.vt_ = nullptr;
    o.handle_ = nullptr;
  }
  return *this;
}

DevicePool::~DevicePool() {
  if (vt_ != nullptr) vt_->pool_destroy(handle_);
}

Result<DevicePool> DevicePool::create(const Device& device) noexcept {
  DeviceEntry e = resolve(device);
  void* handle = nullptr;
  Status st = e.vt->pool_create(e.local, &handle);
  if (!st.ok()) return std::unexpected(std::move(st));
  DevicePool p;
  p.vt_ = e.vt;
  p.handle_ = handle;
  p.ordinal_ = e.local;
  return p;
}

bool DevicePool::is_vmm() const noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  return vt_->pool_is_vmm(handle_);
}

Result<void*> DevicePool::alloc(std::size_t bytes) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  void* ptr = nullptr;
  std::size_t actual = 0;
  Status st = vt_->pool_alloc(handle_, bytes, &ptr, &actual);
  if (!st.ok()) return std::unexpected(std::move(st));
  return ptr;
}

void DevicePool::free(void* ptr, std::size_t bytes) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  vt_->pool_free(handle_, ptr, bytes);
}

// ---- Kernel (compute-04): launch dispatch -----------------------------------

Kernel::Kernel(Kernel&& o) noexcept
    : vt_(o.vt_), module_(o.module_), func_(o.func_), ordinal_(o.ordinal_) {
  o.vt_ = nullptr;
  o.module_ = nullptr;
  o.func_ = nullptr;
}

Kernel& Kernel::operator=(Kernel&& o) noexcept {
  if (this != &o) {
    if (vt_ != nullptr) vt_->kernel_destroy(module_, func_);
    vt_ = o.vt_;
    module_ = o.module_;
    func_ = o.func_;
    ordinal_ = o.ordinal_;
    o.vt_ = nullptr;
    o.module_ = nullptr;
    o.func_ = nullptr;
  }
  return *this;
}

Kernel::~Kernel() {
  if (vt_ != nullptr) vt_->kernel_destroy(module_, func_);
}

Result<Kernel> Kernel::compile(const Device& device, const char* source,
                               const char* entry) noexcept {
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  Status st = e.vt->kernel_compile(e.local, source, entry, &module, &func);
  if (!st.ok()) return std::unexpected(std::move(st));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Result<Kernel> Kernel::load_ptx(const Device& device, const char* ptx,
                                const char* entry) noexcept {
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  // PTX is NUL-terminated text: pass code_size=0 (the CUDA backend reads it as a C string);
  // no spec constants on the CUDA path.
  Status st = e.vt->kernel_load_ptx(e.local, ptx, /*code_size=*/0, entry,
                                    nullptr, 0, nullptr, 0, &module, &func);
  if (!st.ok()) return std::unexpected(std::move(st));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Result<Kernel> Kernel::load_spirv(const Device& device, const SpirvBlob& blob,
                                  const char* entry) noexcept {
  return load_spirv(device, blob, entry, nullptr, 0, nullptr, 0);
}

Result<Kernel> Kernel::load_spirv(const Device& device, const SpirvBlob& blob,
                                  const char* entry, const SpecConstant* specs,
                                  unsigned spec_count, const void* spec_data,
                                  unsigned spec_data_size) noexcept {
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  // SPIR-V is binary: pass the exact byte size (no length-prefix hack). Same vtable slot as
  // PTX; the Vulkan backend uses code_size, the CUDA backend rejects SPIR-V shape via entry.
  Status st = e.vt->kernel_load_ptx(e.local, reinterpret_cast<const char*>(blob.words),
                                    blob.byte_size, entry, specs, spec_count,
                                    spec_data, spec_data_size, &module, &func);
  if (!st.ok()) return std::unexpected(std::move(st));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Result<Kernel> Kernel::load_host(const Device& device, HostFn fn) noexcept {
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  Status st = e.vt->kernel_load_host(e.local, reinterpret_cast<void*>(fn), &module,
                                     &func);
  if (!st.ok()) return std::unexpected(std::move(st));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Status launch(Stream& stream, const Kernel& kernel, const LaunchConfig& config,
              const KernelArgs& args) noexcept {
  HYLE_FATAL_ASSERT(stream.vt_ != nullptr && kernel.vt_ != nullptr);
  // Stream and kernel must be on the same backend (same device family).
  HYLE_FATAL_ASSERT(stream.vt_ == kernel.vt_);
  return stream.vt_->kernel_launch(stream.native_, kernel.module_, kernel.func_,
                                   &config, &args);
}

Status launch_batch(Stream& stream, const BatchItem* items, unsigned count) noexcept {
  HYLE_FATAL_ASSERT(stream.vt_ != nullptr);
  if (count == 0) return Status();
  HYLE_FATAL_ASSERT(items != nullptr && count <= 64);
  // Unpack the items into the parallel arrays the vtable slot takes (stack, bounded 64).
  const void* modules[64];
  const void* funcs[64];
  LaunchConfig configs[64];
  KernelArgs args[64];
  for (unsigned i = 0; i < count; ++i) {
    const Kernel* k = items[i].kernel;
    HYLE_FATAL_ASSERT(k != nullptr && k->vt_ == stream.vt_);  // all on stream's backend
    modules[i] = k->module_;
    funcs[i] = k->func_;
    configs[i] = items[i].config;
    args[i] = items[i].args;
  }
  return stream.vt_->kernel_launch_batch(stream.native_, modules, funcs, configs, args, count);
}

// ---- RecordedLaunch (compute-06 #16) ----------------------------------------

RecordedLaunch::RecordedLaunch(RecordedLaunch&& o) noexcept
    : vt_(o.vt_), recorded_(o.recorded_), stream_(o.stream_) {
  o.vt_ = nullptr;
  o.recorded_ = nullptr;
  o.stream_ = nullptr;
}

RecordedLaunch& RecordedLaunch::operator=(RecordedLaunch&& o) noexcept {
  if (this != &o) {
    if (vt_ != nullptr) vt_->recorded_destroy(stream_, recorded_);
    vt_ = o.vt_;
    recorded_ = o.recorded_;
    stream_ = o.stream_;
    o.vt_ = nullptr;
    o.recorded_ = nullptr;
    o.stream_ = nullptr;
  }
  return *this;
}

RecordedLaunch::~RecordedLaunch() {
  if (vt_ != nullptr) vt_->recorded_destroy(stream_, recorded_);
}

Result<RecordedLaunch> RecordedLaunch::record(Stream& stream, const Kernel& kernel,
                                              const LaunchConfig& config,
                                              const KernelArgs& args) noexcept {
  HYLE_FATAL_ASSERT(stream.vt_ != nullptr && kernel.vt_ != nullptr);
  HYLE_FATAL_ASSERT(stream.vt_ == kernel.vt_);  // same backend as launch
  // A backend without a recording path (host: inline execution, recording gives nothing) has
  // a null slot -> Unimplemented, so the caller can fall back to launch().
  if (stream.vt_->recorded_create == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no record/replay"));
  void* rec = nullptr;
  Status st = stream.vt_->recorded_create(stream.native_, kernel.module_, kernel.func_,
                                          &config, &args, &rec);
  if (!st.ok()) return std::unexpected(std::move(st));
  RecordedLaunch r;
  r.vt_ = stream.vt_;
  r.recorded_ = rec;
  r.stream_ = stream.native_;
  return r;
}

Status RecordedLaunch::replay(Stream& stream) noexcept {
  HYLE_FATAL_ASSERT(vt_ != nullptr);
  // Must replay on the SAME stream it was recorded on (the cb belongs to that pool).
  HYLE_FATAL_ASSERT(stream.vt_ == vt_ && stream.native_ == stream_);
  return vt_->recorded_replay(stream.native_, recorded_);
}

}  // namespace hyle::compute
