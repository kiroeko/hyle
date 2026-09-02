#include "hyle/compute/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <string_view>

#include "backend.h"
#include "hyle/compute/device_pool.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/memory.h"
#include "hyle/compute/stream.h"
#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#if defined(HYLE_HAS_FCONTEXT)
#  include "hyle/platform/thread.h"
#  include "hyle/sched/fiber.h"
#endif

// compute-01 dispatch layer: the public Device/Stream/Event API forwarding through
// the manual backend vtable (backend.h). Pure C++, no CUDA -- the CUDA calls live
// in cuda_backend.cu (nvcc), the inline fallback in host_backend.cc. This file
// only picks a backend (cold path) and forwards.

namespace hyle::compute {
namespace {

class BackendMemoryResource final : public mem::MemoryResource {
 public:
  void initialize(const ComputeBackend* backend, int ordinal,
                  BackendMemoryKind kind) noexcept {
    HYLE_FATAL_ASSERT(vt_ == nullptr && backend != nullptr);
    vt_ = backend;
    ordinal_ = ordinal;
    kind_ = kind;
  }

  bool supported() const noexcept {
    return vt_ != nullptr && vt_->buffer_supported(ordinal_, kind_);
  }

  std::string_view name() const noexcept override {
    switch (kind_) {
      case BackendMemoryKind::kDevice:
        return "hyle.compute.device";
      case BackendMemoryKind::kHostPinned:
        return "hyle.compute.host";
      case BackendMemoryKind::kManaged:
        return "hyle.compute.managed";
    }
    return "hyle.compute.invalid";
  }

  mem::MemoryProperties properties() const noexcept override {
    HYLE_FATAL_ASSERT(vt_ != nullptr);
    mem::MemoryProperty flags = mem::MemoryProperty::kDeviceAccessible;
    if (vt_->buffer_is_host(kind_)) {
      flags |= mem::MemoryProperty::kHostAccessible;
      if (vt_->buffer_is_coherent(kind_)) {
        flags |= mem::MemoryProperty::kHostDeviceCoherent;
      }
    }
    if (kind_ == BackendMemoryKind::kHostPinned &&
        vt_->device_caps(ordinal_).host_buffer) {
      flags |= mem::MemoryProperty::kHostPinned;
    }
    if (kind_ == BackendMemoryKind::kManaged) {
      flags |= mem::MemoryProperty::kManaged;
    }
    return mem::MemoryProperties{
        .flags = flags,
        .minimum_alignment = vt_->buffer_alignment(kind_),
        .maximum_allocation = vt_->buffer_max_size(ordinal_, kind_),
        .domain = mem::MemoryDomainId{
            vt_, static_cast<std::uintptr_t>(ordinal_)},
    };
  }

  Result<mem::ResourceAllocation> allocate(
      std::size_t size, std::size_t alignment) noexcept override {
    if (!supported()) {
      return std::unexpected(Status::Unimplemented(
          "compute backend does not support this memory provider"));
    }
    const std::size_t guaranteed = vt_->buffer_alignment(kind_);
    if (alignment > guaranteed) {
      return std::unexpected(Status::InvalidArgument(
          "compute provider cannot guarantee the requested over-alignment"));
    }
    if (size == 0) {
      return mem::ResourceAllocation{nullptr, 0, alignment};
    }
    void* native = nullptr;
    void* host = nullptr;
    Status status =
        vt_->buffer_alloc(ordinal_, kind_, size, &native, &host);
    if (!status.ok()) {
      return std::unexpected(
          std::move(status).WithContext("allocating compute memory"));
    }
    if (native == nullptr ||
        (host != nullptr) != vt_->buffer_is_host(kind_) ||
        host != vt_->buffer_host_data(kind_, native)) {
      if (native != nullptr) vt_->buffer_free(ordinal_, kind_, native);
      return std::unexpected(Status::Internal(
          "compute backend returned an invalid allocation token"));
    }
    return mem::ResourceAllocation{native, size, alignment};
  }

  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle != nullptr) {
      vt_->buffer_free(ordinal_, kind_, allocation.native_handle);
    }
  }

  Status read(void* native_handle, std::size_t offset,
              std::span<std::byte> destination) noexcept override {
    if (destination.empty()) return Status::Ok();
    return vt_->buffer_get(native_handle, offset, destination.data(),
                           destination.size());
  }

  Status write(void* native_handle, std::size_t offset,
               std::span<const std::byte> source) noexcept override {
    if (source.empty()) return Status::Ok();
    return vt_->buffer_set(native_handle, offset, source.data(), source.size());
  }

  Status fill(void* native_handle, std::size_t offset, std::size_t size,
              std::byte value) noexcept override {
    if (size == 0) return Status::Ok();
    return vt_->buffer_clear(native_handle, offset,
                             std::to_integer<std::uint8_t>(value), size);
  }

  Status copy(void* destination_handle, std::size_t destination_offset,
              mem::MemoryResource& source_resource, void* source_handle,
              std::size_t source_offset, std::size_t size) noexcept override {
    if (size == 0) return Status::Ok();
    if (destination_handle == source_handle) {
      if (destination_offset == source_offset) return Status::Ok();
      const std::size_t distance = destination_offset < source_offset
                                       ? source_offset - destination_offset
                                       : destination_offset - source_offset;
      if (distance < size) {
        return Status::InvalidArgument(
            "overlapping compute buffer copy is unsupported");
      }
    }

    auto* source = dynamic_cast<BackendMemoryResource*>(&source_resource);
    if (source != nullptr && source->vt_ == vt_ &&
        source->ordinal_ == ordinal_) {
      if (source->vt_->buffer_is_host(source->kind_)) {
        HYLE_RETURN_IF_ERROR(source->flush(source_handle, source_offset, size));
      }
      HYLE_RETURN_IF_ERROR(vt_->buffer_copy(
          destination_handle, destination_offset, source_handle,
          source_offset, size));
      if (vt_->buffer_is_host(kind_)) {
        return invalidate(destination_handle, destination_offset, size);
      }
      return Status::Ok();
    }

    if (!source_resource.properties().has(
            mem::MemoryProperty::kHostAccessible)) {
      return Status::Unimplemented(
          "cross-domain compute copy requires a host-accessible source");
    }
    HYLE_RETURN_IF_ERROR(
        source_resource.invalidate(source_handle, source_offset, size));
    auto mapping = source_resource.map_host(
        source_handle, source_offset, size, mem::HostAccess::kRead);
    if (!mapping) return std::move(mapping).error();
    if (mapping->data == nullptr) {
      source_resource
          .unmap_host(source_handle, mapping->token, source_offset, size,
                      mem::HostAccess::kRead)
          .IgnoreError();
      return Status::Internal(
          "source resource returned a null host mapping");
    }
    Status result =
        vt_->buffer_set(destination_handle, destination_offset, mapping->data,
                        size);
    result.Update(source_resource.unmap_host(
        source_handle, mapping->token, source_offset, size,
        mem::HostAccess::kRead));
    return result;
  }

  Result<mem::ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      mem::HostAccess) noexcept override {
    if (!vt_->buffer_is_host(kind_)) {
      return std::unexpected(Status::FailedPrecondition(
          "compute memory provider is not host accessible"));
    }
    void* base = vt_->buffer_host_data(kind_, native_handle);
    if (size != 0 && base == nullptr) {
      return std::unexpected(
          Status::Internal("compute host mapping is unavailable"));
    }
    return mem::ResourceHostMapping{
        size == 0 ? nullptr : static_cast<std::byte*>(base) + offset,
        nullptr};
  }

  Status unmap_host(void*, void*, std::size_t, std::size_t,
                    mem::HostAccess) noexcept override {
    return Status::Ok();
  }

  Status flush(void* native_handle, std::size_t offset,
               std::size_t size) noexcept override {
    if (!vt_->buffer_is_host(kind_)) {
      return Status::FailedPrecondition(
          "device-only compute memory cannot be host-flushed");
    }
    if (size == 0) return Status::Ok();
    return vt_->buffer_flush_host(native_handle, offset, size);
  }

  Status invalidate(void* native_handle, std::size_t offset,
                    std::size_t size) noexcept override {
    if (!vt_->buffer_is_host(kind_)) {
      return Status::FailedPrecondition(
          "device-only compute memory cannot be host-invalidated");
    }
    if (size == 0) return Status::Ok();
    return vt_->buffer_invalidate_host(native_handle, offset, size);
  }

  Result<mem::ResourceStats> stats() const noexcept override {
    if (kind_ != BackendMemoryKind::kDevice) {
      return mem::ResourceStats{};
    }
    std::uint64_t free = 0;
    std::uint64_t total = 0;
    vt_->device_memory(ordinal_, &free, &total);
    constexpr std::uint64_t kSizeMax =
        static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)());
    const std::size_t capacity =
        static_cast<std::size_t>((std::min)(total, kSizeMax));
    const std::size_t available =
        static_cast<std::size_t>((std::min)(free, kSizeMax));
    return mem::ResourceStats{
        .capacity_bytes = capacity,
        .used_bytes = available <= capacity ? capacity - available : SIZE_MAX,
        .reserved_bytes = SIZE_MAX,
        .peak_used_bytes = SIZE_MAX,
    };
  }

 private:
  const ComputeBackend* vt_ = nullptr;
  int ordinal_ = 0;
  BackendMemoryKind kind_ = BackendMemoryKind::kDevice;
};

constexpr std::size_t memory_kind_index(BackendMemoryKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

template <class T>
class NoDestructor {
 public:
  NoDestructor() noexcept { ::new (storage_) T(); }
  T& get() noexcept { return *reinterpret_cast<T*>(storage_); }
  const T& get() const noexcept {
    return *reinterpret_cast<const T*>(storage_);
  }

 private:
  alignas(T) std::byte storage_[sizeof(T)];
};

// The process-wide device table, built once (Meyers singleton -- charter §, no
// leaky global). Each entry maps a GLOBAL device index to {backend vtable, the
// index LOCAL to that backend}. GPUs (CUDA) come first so default_device() prefers
// them; the host CPU device is always last, so the table is never empty.
struct DeviceEntry {
  const ComputeBackend* vt;
  int local;  // ordinal within vt's own device list
};

#if defined(HYLE_HAS_FCONTEXT)
void probe_device_table(void* arg) noexcept;
#endif

struct DeviceTable {
  static constexpr std::size_t kMax = 32;
  DeviceEntry entries[kMax];
  BackendMemoryResource resources[kMax][3];
  std::size_t count = 0;

  DeviceTable() noexcept {
#if defined(HYLE_HAS_FCONTEXT)
    // Third-party GPU initialization can exceed the default fiber stack. Only
    // the process-first probe is redirected to an ordinary OS thread.
    if (sched::Fiber::current() != nullptr) {
      platform::ThreadOpts opts;
      opts.name = "hyle.gpu-init";
      auto thread = platform::Thread::spawn(&probe_device_table, this, opts);
      HYLE_FATAL_ASSERT(thread.has_value());
      thread->join();
      return;
    }
#endif
    probe();
  }

  void probe() noexcept {
    // GPUs first (CUDA backend), if compiled in and any device is present.
    // Reserve the final slot while probing optional backends: without this, 32
    // GPU entries (or duplicated CUDA+Vulkan views) would silently evict the
    // host fallback despite the public always-present contract.
#if defined(HYLE_COMPUTE_CUDA)
    add_backend(cuda_vtable(), 1);
#endif
    // Vulkan compute backend (compute-06) next: cross-vendor GPU. On NVIDIA it ranks
    // below CUDA (CUDA preferred for N-cards); on AMD/Intel/Apple it's the GPU path.
    // vulkan_vtable() may be null (no loader/device at runtime) -- add_backend tolerates.
#if defined(HYLE_COMPUTE_VULKAN)
    add_backend(vulkan_vtable(), 1);
#endif
    // Host CPU fallback always last -- guarantees >= 1 device.
    add_backend(host_vtable());
    HYLE_FATAL_ASSERT(count != 0 && entries[count - 1].vt == host_vtable());
  }

  void add_backend(const ComputeBackend* vt,
                   std::size_t tail_reserve = 0) noexcept {
    if (vt == nullptr) return;
    HYLE_FATAL_ASSERT(tail_reserve < kMax);
    const std::size_t limit = kMax - tail_reserve;
    std::size_t n = vt->device_count();
    for (std::size_t i = 0; i < n && count < limit; ++i) {
      const int ordinal = static_cast<int>(i);
      entries[count] = DeviceEntry{vt, ordinal};
      resources[count][memory_kind_index(BackendMemoryKind::kDevice)]
          .initialize(vt, ordinal, BackendMemoryKind::kDevice);
      resources[count][memory_kind_index(BackendMemoryKind::kHostPinned)]
          .initialize(vt, ordinal, BackendMemoryKind::kHostPinned);
      resources[count][memory_kind_index(BackendMemoryKind::kManaged)]
          .initialize(vt, ordinal, BackendMemoryKind::kManaged);
      ++count;
    }
  }
};

#if defined(HYLE_HAS_FCONTEXT)
void probe_device_table(void* arg) noexcept {
  static_cast<DeviceTable*>(arg)->probe();
}
#endif

const DeviceTable& table() noexcept {
  // Providers are process-lifetime borrows. Deliberately do not destroy this
  // table during global teardown, when user-owned static Buffers may still
  // return allocations through it.
  static NoDestructor<DeviceTable> t;
  return t.get();
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

BackendMemoryResource& resolve_resource(
    const Device& device, BackendMemoryKind kind) noexcept {
  DeviceEntry entry = resolve(device);
  (void)entry;
  DeviceTable& mutable_table = const_cast<DeviceTable&>(table());
  return mutable_table.resources[static_cast<std::size_t>(device.ordinal())]
                                [memory_kind_index(kind)];
}

// Validate the pointer-bearing raw ABI before any backend dereferences it. The typed shell
// always produces this shape, but raw KernelArgs is public specifically for callers that
// author offsets themselves; malformed metadata must return InvalidArgument, not become
// backend-specific UB (notably a null descriptor array on CUDA or VkBuffer read on Vulkan).
Status validate_kernel_args(const KernelArgs& args) noexcept {
  if (args.buffer_count != 0 && args.buffers == nullptr)
    return Status::InvalidArgument("compute launch: null buffer argument array");
  for (unsigned i = 0; i < args.buffer_count; ++i) {
    const BufferBinding binding = args.buffers[i];
    if (binding.native_handle == nullptr || binding.size == 0 ||
        binding.offset > SIZE_MAX - binding.size) {
      return Status::InvalidArgument(
          "compute launch: invalid buffer binding");
    }
  }
  if (args.scalar_bytes != 0 && args.scalars == nullptr)
    return Status::InvalidArgument("compute launch: null scalar blob");
  if (args.scalar_count != 0 && (args.fields == nullptr || args.scalars == nullptr))
    return Status::InvalidArgument("compute launch: null scalar field metadata");
  if ((args.scalar_count == 0) != (args.scalar_bytes == 0))
    return Status::InvalidArgument("compute launch: inconsistent scalar shape");

  for (unsigned i = 0; i < args.scalar_count; ++i) {
    const ScalarField field = args.fields[i];
    if (field.size == 0 || field.offset > args.scalar_bytes ||
        field.size > args.scalar_bytes - field.offset)
      return Status::InvalidArgument("compute launch: scalar field outside blob");
  }

  const unsigned total = args.buffer_count + args.scalar_count;
  if (total < args.buffer_count)
    return Status::InvalidArgument("compute launch: argument count overflow");
  if (args.order == nullptr) {
    if (args.arg_count != 0)
      return Status::InvalidArgument("compute launch: null argument-order table");
    return Status();
  }
  if (args.arg_count != total)
    return Status::InvalidArgument("compute launch: invalid argument-order count");
  for (unsigned i = 0; i < args.arg_count; ++i) {
    const KernelArgRef ref = args.order[i];
    const bool valid =
        (ref.kind == KernelArgKind::kBuffer && ref.index < args.buffer_count) ||
        (ref.kind == KernelArgKind::kScalar && ref.index < args.scalar_count);
    if (!valid)
      return Status::InvalidArgument("compute launch: invalid argument-order entry");
    for (unsigned j = 0; j < i; ++j)
      if (args.order[j].kind == ref.kind && args.order[j].index == ref.index)
        return Status::InvalidArgument("compute launch: duplicate argument-order entry");
  }
  return Status();
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
  if (out == nullptr) max = 0;
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

int device_backend_ordinal(const Device& device) noexcept {
  return resolve(device).local;
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("creating compute stream on an invalid device"));
  DeviceEntry e = resolve(device);
  void* native = nullptr;
  Status st = e.vt->stream_create(e.local, &native);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("creating compute stream"));
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("creating compute event on an invalid device"));
  DeviceEntry e = resolve(device);
  void* native = nullptr;
  Status st = e.vt->event_create(e.local, &native);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("creating compute event"));
  Event ev;
  ev.vt_ = e.vt;
  ev.native_ = native;
  ev.local_ = e.local;
  return ev;
}

Result<Event> Event::create_timing(const Device& device) noexcept {
  if (!device.valid())
    return std::unexpected(Status::InvalidArgument(
        "creating compute timing event on an invalid device"));
  DeviceEntry e = resolve(device);
  // A backend without a timing slot (none currently, but future backends may skip it)
  // reports Unimplemented rather than crashing on a null fn-ptr.
  if (e.vt->event_create_timing == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no GPU timing"));
  void* native = nullptr;
  Status st = e.vt->event_create_timing(e.local, &native);
  if (!st.ok())
    return std::unexpected(
        std::move(st).WithContext("creating compute timing event"));
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
  if (!start.timing_ || !stop.timing_)
    return std::unexpected(
        Status::InvalidArgument("elapsed_ns: both events must be timing events"));
  // Same backend is insufficient on a multi-GPU system: native timestamps from two
  // devices have unrelated counters and native handle ownership.
  if (start.vt_ != stop.vt_ || start.local_ != stop.local_)
    return std::unexpected(
        Status::InvalidArgument("elapsed_ns: events on different devices"));
  if (start.vt_->event_elapsed_ns == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no GPU timing"));
  std::uint64_t ns = 0;
  Status st = start.vt_->event_elapsed_ns(start.native_, stop.native_, &ns);
  if (!st.ok())
    return std::unexpected(
        std::move(st).WithContext("reading compute event duration"));
  return ns;
}

// ---- Device storage providers ------------------------------------------------

Result<mem::MemoryResourceRef> device_memory_resource(
    const Device& device) noexcept {
  if (!device.valid()) {
    return std::unexpected(
        Status::InvalidArgument("selecting memory for an invalid device"));
  }
  BackendMemoryResource& resource =
      resolve_resource(device, BackendMemoryKind::kDevice);
  HYLE_FATAL_ASSERT(resource.supported());
  return mem::MemoryResourceRef(resource);
}

Result<mem::MemoryResourceRef> host_memory_resource(
    const Device& device) noexcept {
  if (!device.valid()) {
    return std::unexpected(
        Status::InvalidArgument("selecting memory for an invalid device"));
  }
  BackendMemoryResource& resource =
      resolve_resource(device, BackendMemoryKind::kHostPinned);
  HYLE_FATAL_ASSERT(resource.supported());
  return mem::MemoryResourceRef(resource);
}

Result<mem::MemoryResourceRef> managed_memory_resource(
    const Device& device) noexcept {
  if (!device.valid()) {
    return std::unexpected(
        Status::InvalidArgument("selecting memory for an invalid device"));
  }
  BackendMemoryResource& resource =
      resolve_resource(device, BackendMemoryKind::kManaged);
  if (!resource.supported()) {
    return std::unexpected(Status::Unimplemented(
        "selected compute backend has no managed memory provider"));
  }
  return mem::MemoryResourceRef(resource);
}

bool memory_resource_belongs_to(mem::MemoryResourceRef resource,
                                const Device& device) noexcept {
  if (!resource.valid() || !device.valid()) return false;
  return resource == mem::MemoryResourceRef(
                         resolve_resource(device, BackendMemoryKind::kDevice)) ||
         resource == mem::MemoryResourceRef(resolve_resource(
                         device, BackendMemoryKind::kHostPinned)) ||
         (resolve_resource(device, BackendMemoryKind::kManaged).supported() &&
          resource == mem::MemoryResourceRef(resolve_resource(
                          device, BackendMemoryKind::kManaged)));
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("creating device pool on an invalid device"));
  DeviceEntry e = resolve(device);
  void* handle = nullptr;
  Status st = e.vt->pool_create(e.local, &handle);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("creating device pool"));
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
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("allocating from device pool"));
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("compiling compute kernel on an invalid device"));
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  Status st = e.vt->kernel_compile(e.local, source, entry, &module, &func);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("compiling compute kernel"));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Result<Kernel> Kernel::load_ptx(const Device& device, const char* ptx,
                                const char* entry) noexcept {
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("loading PTX kernel on an invalid device"));
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  // PTX is NUL-terminated text: pass code_size=0 (the CUDA backend reads it as a C string);
  // no spec constants on the CUDA path.
  Status st = e.vt->kernel_load_ptx(e.local, ptx, /*code_size=*/0, entry,
                                    nullptr, 0, nullptr, 0, &module, &func);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("loading PTX kernel"));
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("loading SPIR-V kernel on an invalid device"));
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  // SPIR-V is binary: pass the exact byte size (no length-prefix hack). Same vtable slot as
  // PTX; the Vulkan backend consumes code_size, while the CUDA backend rejects any non-zero
  // code_size (binary = SPIR-V) as Unimplemented (SPIR-V is Vulkan-only).
  Status st = e.vt->kernel_load_ptx(e.local, reinterpret_cast<const char*>(blob.words),
                                    blob.byte_size, entry, specs, spec_count,
                                    spec_data, spec_data_size, &module, &func);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("loading SPIR-V kernel"));
  Kernel k;
  k.vt_ = e.vt;
  k.module_ = module;
  k.func_ = func;
  k.ordinal_ = e.local;
  return k;
}

Result<Kernel> Kernel::load_host(const Device& device, HostFn fn) noexcept {
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("loading host kernel on an invalid device"));
  DeviceEntry e = resolve(device);
  void* module = nullptr;
  void* func = nullptr;
  Status st = e.vt->kernel_load_host(e.local, reinterpret_cast<void*>(fn), &module,
                                     &func);
  if (!st.ok())
    return std::unexpected(std::move(st).WithContext("loading host kernel"));
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
  // Native modules/functions are device-owned, not merely backend-owned.
  HYLE_FATAL_ASSERT(stream.vt_ == kernel.vt_ &&
                    stream.local_ == kernel.ordinal_);
  HYLE_RETURN_IF_ERROR(validate_kernel_args(args));
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
    HYLE_FATAL_ASSERT(k != nullptr && k->vt_ == stream.vt_ &&
                      k->ordinal_ == stream.local_);
    HYLE_RETURN_IF_ERROR(validate_kernel_args(items[i].args));
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
  HYLE_FATAL_ASSERT(stream.vt_ == kernel.vt_ &&
                    stream.local_ == kernel.ordinal_);
  HYLE_RETURN_IF_ERROR(validate_kernel_args(args));
  // A backend without a recording path (host: inline execution, recording gives nothing) has
  // a null slot -> Unimplemented, so the caller can fall back to launch().
  if (stream.vt_->recorded_create == nullptr)
    return std::unexpected(Status::Unimplemented("compute: backend has no record/replay"));
  void* rec = nullptr;
  Status st = stream.vt_->recorded_create(stream.native_, kernel.module_, kernel.func_,
                                          &config, &args, &rec);
  if (!st.ok())
    return std::unexpected(
        std::move(st).WithContext("recording compute kernel launch"));
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
