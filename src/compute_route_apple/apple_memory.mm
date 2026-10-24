#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/paged_arena_access.h"

#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include "hyle/mem/std_allocator.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/core/assert.h"

#include "apple_internal.h"
#include "apple_copy_internal.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace detail {
namespace { const char memory_authority = 0; }

struct MemoryState : mem::MemoryResource {
  std::string_view name() const noexcept override { return "MLX shared Metal"; }
  mem::MemoryProperties properties() const noexcept override {
    return {.flags = mem::MemoryProperty::kHostAccessible |
                     mem::MemoryProperty::kHostAddressIdentity |
                     mem::MemoryProperty::kDeviceAccessible |
                     mem::MemoryProperty::kHostDeviceCoherent,
            .minimum_alignment = alignof(std::max_align_t),
            .maximum_allocation = SIZE_MAX,
            .domain = {&memory_authority, 0}};
  }
  Result<mem::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      return std::unexpected(Status::InvalidArgument("Apple allocation alignment must be a power of two"));
    if (bytes == 0) return mem::ResourceAllocation{nullptr, 0, alignment};
    try {
      auto buffer = mx::allocator::malloc(bytes);
      if (buffer.ptr() == nullptr)
        return std::unexpected(Status::ResourceExhausted("MLX buffer allocation failed"));
      id<MTLBuffer> metal = (__bridge id<MTLBuffer>)buffer.ptr();
      if (metal.storageMode != MTLStorageModeShared ||
          (reinterpret_cast<std::uintptr_t>(metal.contents) & (alignment - 1)) != 0) {
        mx::allocator::free(buffer);
        return std::unexpected(Status::InvalidArgument("MLX backing cannot satisfy shared host alignment"));
      }
      return mem::ResourceAllocation{buffer.ptr(), bytes, alignment};
    } catch (...) { return std::unexpected(native_exception()); }
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle != nullptr)
      mx::allocator::free(mx::allocator::Buffer(allocation.native_handle));
  }
  Result<mem::ResourceHostMapping> map_host(void *handle, std::size_t offset,
                                           std::size_t bytes, mem::HostAccess) noexcept override {
    if (bytes == 0) return mem::ResourceHostMapping{};
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)handle;
    if (metal.storageMode != MTLStorageModeShared || offset > metal.length || bytes > metal.length - offset)
      return std::unexpected(Status::InvalidArgument("Apple host map exceeds native shared backing"));
    return mem::ResourceHostMapping{static_cast<std::byte *>(metal.contents) + offset, nullptr};
  }
  Status unmap_host(void *, void *, std::size_t, std::size_t, mem::HostAccess) noexcept override {
    return Status::Ok();
  }
  Status flush(void *, std::size_t, std::size_t) noexcept override { return Status::Ok(); }
  Status invalidate(void *, std::size_t, std::size_t) noexcept override { return Status::Ok(); }
};

struct ArrayStorage final {
  ResourceBudget::Reservation charge;
  ResourceBudget::Reservation private_metadata;
  mem::Buffer buffer;
};
struct LeasedArrayStorage final {
  ResourceBudget::Reservation private_metadata;
  ReadLease source;
};
template<class Frame>struct PagedArrayStorage final {
  ResourceBudget::Reservation private_metadata;
  Frame frame;
};
struct ImportedArrayStorage final : MemoryState {
  ImportedArrayStorage(mx::array source, mem::AllocatorRef selected) noexcept
      : owner(std::move(source)), allocator(selected) {}
  mx::array owner;
  mem::AllocatorRef allocator;
  ResourceBudget::Reservation private_metadata;
  mem::MemoryProperties properties() const noexcept override {
    auto result = MemoryState::properties();
    // Imported readonly byte views make no new allocation-alignment promise.
    // The native carrier origin is checked separately when binding it.
    result.minimum_alignment = 1;
    return result;
  }
  Result<mem::ResourceAllocation> allocate(std::size_t, std::size_t) noexcept override {
    return std::unexpected(Status::Unimplemented("imported MLX array cannot allocate storage"));
  }
  Result<mem::ResourceHostMapping> map_host(void *handle, std::size_t offset,
      std::size_t bytes, mem::HostAccess access) noexcept override {
    if (access != mem::HostAccess::kRead)
      return std::unexpected(Status::PermissionDenied("imported MLX array is readonly"));
    return MemoryState::map_host(handle, offset, bytes, access);
  }
  // The actual MLX array owns/free its buffer. This token is a readonly
  // retained view, not a second MLX allocator ownership claim.
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    HYLE_FATAL_ASSERT(allocation.native_handle == owner.buffer().ptr());
  }
};
bool is_copy_resource(mem::MemoryResourceRef resource) noexcept {
  return resource.valid() && dynamic_cast<MemoryState *>(&resource.resource()) != nullptr;
}
namespace {
Status constant_admission(const std::shared_ptr<Runtime> &runtime) noexcept {
  if (!runtime) return Status::InvalidArgument("Apple constant copy requires its native Runtime");
  std::lock_guard lock(runtime->queue_mutex);
  if (!runtime->accepting) return Status::FailedPrecondition("Apple constant source is closed");
  return runtime->failure;
}
Result<bool> constant_native_range(const std::shared_ptr<Runtime> &runtime,
    mem::ConstBufferView view, std::size_t bytes, bool writable) noexcept {
  const auto *memory = dynamic_cast<MemoryState *>(&view.resource().resource());
  if (!memory) return false;
  const auto *imported = dynamic_cast<const ImportedArrayStorage *>(memory);
  if (imported && (writable || imported->owner.status() != mx::array::available)) return false;
  if (bytes > view.size()) return std::unexpected(Status::OutOfRange("Apple constant span exceeds logical storage"));
  if (!bytes && !view.native_handle()) return true;
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)view.native_handle();
  if (buffer == nil || buffer.storageMode != MTLStorageModeShared ||
      view.offset() > buffer.length || view.size() > buffer.length - view.offset())
    return std::unexpected(Status::OutOfRange("Apple constant span exceeds shared Metal backing"));
  const auto expected = mlx::core::metal::device(runtime->stream->device).mtl_device();
  return reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device) ==
      reinterpret_cast<std::uintptr_t>(expected);
}
} // namespace
Result<bool> MemoryAccess::can_copy_constant(const std::shared_ptr<Runtime> &runtime,
    const ConstantCopy &copy) noexcept {
  HYLE_RETURN_IF_ERROR(constant_admission(runtime));
  if (!copy.destination.valid() || !copy.source.valid() ||
      !copy.destination.storage().valid() || !copy.source.storage().valid() ||
      copy.destination.description() != copy.source.description())
    return std::unexpected(Status::InvalidArgument("Apple constant copy descriptions differ"));
  if (!copy.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible)) return false;
  HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
  if (bytes > copy.source.storage().size())
    return std::unexpected(Status::OutOfRange("Apple constant source span is too short"));
  try {
    HYLE_ASSIGN_OR_RETURN(auto destination, constant_native_range(runtime, copy.destination.storage(), bytes, true));
    if (!destination) return false;
    if (is_copy_resource(copy.source.storage().resource())) {
      HYLE_ASSIGN_OR_RETURN(auto source, constant_native_range(runtime, copy.source.storage(), bytes, false));
      if (!source) return false;
      const auto from = copy.source.storage();
      const auto to = copy.destination.storage();
      if (from.native_handle() == to.native_handle() && from.offset() != to.offset() &&
          (from.offset() < to.offset() ? to.offset() - from.offset() < bytes
                                      : from.offset() - to.offset() < bytes)) return false;
    }
    return true;
  } catch (...) { return std::unexpected(native_exception()); }
}
Status MemoryAccess::copy_constants(const std::shared_ptr<Runtime> &runtime,
    std::span<const ConstantCopy> copies, mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid()) return Status::InvalidArgument("Apple constant metadata allocator is invalid");
  // Qualify every actual writable Metal target before touching any payload.
  // Source readiness/immutability and target exclusivity are the original
  // cold ConstantBinding/Graph-allocation contracts, not inferred GPU fences.
  for (const auto &copy : copies) {
    HYLE_ASSIGN_OR_RETURN(auto supported, can_copy_constant(runtime, copy));
    if (!supported) return Status::Unimplemented("Apple constant copy has no writable native target or readable source");
  }
  HYLE_RETURN_IF_ERROR(constant_admission(runtime));
  for (const auto &copy : copies) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
    if (!bytes) continue;
    HYLE_ASSIGN_OR_RETURN(auto target, copy.destination.storage().subview(0, bytes));
    HYLE_ASSIGN_OR_RETURN(auto source, copy.source.storage().subview(0, bytes));
    if (is_copy_resource(source.resource()) && source.native_handle() == target.native_handle() &&
        source.offset() == target.offset()) continue;
    HYLE_ASSIGN_OR_RETURN(auto mapping, target.map_host(mem::HostAccess::kWrite));
    HYLE_ASSIGN_OR_RETURN(auto destination, mapping.writable_bytes());
    // Preserve the source resource's real read/invalidate/error protocol.
    // This is a synchronous shared-memory write, exactly as copy_from_host;
    // it neither evaluates an Array nor accepts work on a second executor.
    auto status = source.read(destination);
    if (status.ok()) status = mapping.flush();
    status.Update(mapping.close());
    HYLE_RETURN_IF_ERROR(status);
  }
  return Status::Ok();
}
namespace {
template<class Release>
Result<mx::array> readonly_tensor(ConstTensorView tensor, const InputSpec &spec,
                                  std::size_t bytes, Release release) noexcept {
  const auto storage = tensor.storage();
  try {
    mx::array result(spec.shape, spec.dtype, nullptr, {});
    const bool strided = tensor.description().is_plain() && !tensor.description().is_dense();
    if (storage.offset() % result.itemsize() != 0 || bytes % result.itemsize() != 0 ||
        (!strided && bytes != result.nbytes()) ||
        storage.offset() / result.itemsize() > static_cast<std::size_t>(INT64_MAX))
      return std::unexpected(Status::InvalidArgument("Apple input carrier origin/size differs"));
    auto strides = result.strides();
    auto flags = result.flags();
    const auto data_size = bytes / result.itemsize();
    if (strided) {
      // Import only whole-element metadata views of the original physical
      // span. Zero strides alias that span; padding/depth strides never cause
      // an activation expansion, an allocation or a writable native alias.
      const auto unit = static_cast<std::int64_t>(result.itemsize());
      const auto byte_strides = tensor.description().byte_strides();
      std::array<std::size_t, kMaxTensorRank> packed_axes{};
      std::size_t packed_count = 0;
      for (std::size_t axis = 0; axis < strides.size(); ++axis) {
        if (byte_strides[axis] < 0 || byte_strides[axis] % unit != 0)
          return std::unexpected(Status::InvalidArgument("Apple input stride must align to whole elements"));
        strides[axis] = static_cast<std::size_t>(byte_strides[axis] / unit);
        if (strides[axis] != 0 && spec.shape[axis] > 1)
          packed_axes[packed_count++] = axis;
      }
      // Equal element-count/span alone cannot prove no gaps: e.g. shape
      // [2,2,2], strides [1,1,5] contains both overlaps and holes. Prove a
      // complete packed chain in physical stride order, ignoring broadcasts.
      std::sort(packed_axes.begin(), packed_axes.begin() + packed_count,
          [&](auto left, auto right) { return strides[left] < strides[right]; });
      std::size_t packed_stride = 1;
      bool packed = true;
      for (std::size_t index = 0; index < packed_count; ++index) {
        const auto axis = packed_axes[index];
        packed &= strides[axis] == packed_stride;
        const auto extent = static_cast<std::size_t>(spec.shape[axis]);
        if (packed_stride > SIZE_MAX / extent) packed = false;
        else packed_stride *= extent;
      }
      flags.contiguous = packed && packed_stride == data_size;
      flags.row_contiguous = flags.col_contiguous = true;
      std::size_t row_stride = 1, column_stride = 1;
      for (std::size_t axis = 0; axis < strides.size(); ++axis) {
        const auto reverse = strides.size() - axis - 1;
        if (spec.shape[reverse] != 1) {
          flags.row_contiguous &= strides[reverse] == row_stride;
          const auto extent = static_cast<std::size_t>(spec.shape[reverse]);
          if (extent && row_stride > SIZE_MAX / extent) flags.row_contiguous = false;
          else row_stride *= extent;
        }
        if (spec.shape[axis] != 1) {
          flags.col_contiguous &= strides[axis] == column_stride;
          const auto extent = static_cast<std::size_t>(spec.shape[axis]);
          if (extent && column_stride > SIZE_MAX / extent) flags.col_contiguous = false;
          else column_stride *= extent;
        }
      }
    }
    if (bytes == 0) {
      result.set_data(mx::allocator::Buffer(nullptr), std::move(release));
      result.set_status(mx::array::available);
      return result;
    }
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)storage.native_handle();
    if (metal == nil || metal.storageMode != MTLStorageModeShared ||
        storage.offset() > metal.length || bytes > metal.length - storage.offset())
      return std::unexpected(Status::InvalidArgument("Apple input exceeds shared Metal backing"));
    mx::array base(mx::Shape{1}, mx::uint8, nullptr, {});
    // Retain only the nominated readonly owner; never claim a second native free.
    base.set_data(mx::allocator::Buffer(storage.native_handle()), std::move(release));
    result.copy_shared_buffer(base, strides, flags, data_size,
        static_cast<std::int64_t>(storage.offset() / result.itemsize()));
    result.set_status(mx::array::available);
    return result;
  } catch (...) { return std::unexpected(native_exception()); }
}
} // namespace
Result<mx::array> MemoryAccess::borrow_tensor(ConstTensorView tensor) noexcept {
  if (!tensor.valid() || !is_copy_resource(tensor.storage().resource()))
    return std::unexpected(Status::InvalidArgument("Apple input requires original Metal storage"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(tensor.description()));
  HYLE_ASSIGN_OR_RETURN(auto bytes, tensor.description().required_bytes());
  return readonly_tensor(tensor, spec, bytes, [](mx::allocator::Buffer) {});
}
bool copy_source_resource_matches(mem::MemoryResourceRef nominated,
                                  mem::MemoryResourceRef actual) noexcept {
  if (actual == nominated) return true;
  if (!actual.valid() || !nominated.valid() ||
      dynamic_cast<ImportedArrayStorage *>(&actual.resource()) == nullptr ||
      dynamic_cast<MemoryState *>(&nominated.resource()) == nullptr)
    return false;
  return actual.domain() == nominated.domain();
}
Result<CopySpan> copy_span(mem::ConstBufferView view, mem::HostAccess access) noexcept {
  if (!view.valid() || dynamic_cast<MemoryState *>(&view.resource().resource()) == nullptr)
    return std::unexpected(Status::InvalidArgument("Apple copy requires original Metal resource"));
  if (access != mem::HostAccess::kRead &&
      dynamic_cast<ImportedArrayStorage *>(&view.resource().resource()) != nullptr)
    return std::unexpected(Status::PermissionDenied("imported MLX array is readonly"));
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)view.native_handle();
  if (buffer == nil || buffer.storageMode != MTLStorageModeShared ||
      view.offset() > buffer.length || view.size() > buffer.length - view.offset())
    return std::unexpected(Status::OutOfRange("Apple copy exceeds owned shared Metal backing"));
  return CopySpan{view.native_handle(), {reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device),
      reinterpret_cast<std::uintptr_t>(view.native_handle()), view.offset(), view.size(), 4}};
}
} // namespace detail

MemoryResource::MemoryResource() noexcept = default;
MemoryResource::MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept : state_(std::move(state)) {}
MemoryResource::MemoryResource(MemoryResource &&) noexcept = default;
MemoryResource &MemoryResource::operator=(MemoryResource &&) noexcept = default;
MemoryResource::~MemoryResource() = default;

Result<MemoryResource> MemoryResource::create(const Context &context) noexcept {
  if (!context.valid()) return std::unexpected(Status::InvalidArgument("Apple memory requires a valid Context"));
  try { return MemoryResource(std::make_unique<detail::MemoryState>()); }
  catch (...) { return std::unexpected(detail::native_exception()); }
}
mem::MemoryResourceRef MemoryResource::ref() const noexcept {
  return state_ ? mem::MemoryResourceRef(*state_) : mem::MemoryResourceRef{};
}
Result<mx::array> MemoryResource::allocate(const TensorDesc &description, ResourceBudget budget,
                                         ResourceBudgetCategory category) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Apple memory resource is empty"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  if (description.is_plain() && !description.is_dense())
    return std::unexpected(Status::InvalidArgument("Apple owned arrays require a dense plain carrier"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  try {
    auto metadata = mem::system_allocator();
    if (budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(metadata, budget.metadata_allocator());
    }
    auto storage = std::allocate_shared<detail::ArrayStorage>(
        mem::StdAllocator<detail::ArrayStorage>(metadata));
    if (budget.valid()) {
      // MLX owns its array descriptors. Do not pretend those opaque native
      // allocations are included in our requested payload/control bytes.
      HYLE_ASSIGN_OR_RETURN(storage->private_metadata, budget.reserve_unattributed(std::nullopt));
      HYLE_ASSIGN_OR_RETURN(storage->charge, budget.reserve(ref(), bytes, category));
    }
    HYLE_ASSIGN_OR_RETURN(storage->buffer, mem::Buffer::create(ref(), bytes, alignof(std::max_align_t)));
    mx::array result(spec.shape, spec.dtype, nullptr, {});
    // Evaluate the handle before moving its owner into the deleter: function
    // argument evaluation order must not decide whether storage is null.
    const auto handle = storage->buffer.native_handle();
    result.set_data(mx::allocator::Buffer(handle),
                    [storage = std::move(storage)](mx::allocator::Buffer) {});
    result.set_status(mx::array::available);
    return result;
  } catch (...) { return std::unexpected(detail::native_exception()); }
}
Result<mx::array> MemoryResource::binding(ReadLease source, const TensorDesc &description,
                                        LeaseImportOptions options) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Apple memory resource is empty"));
  return detail::MemoryAccess::owned_tensor(std::move(source), description, options);
}
Result<mx::array> detail::MemoryAccess::owned_tensor(ReadLease source,
    const TensorDesc &description, LeaseImportOptions options) noexcept {
  if (!options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple lease binding resource/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, source.tensor(description));
  const auto storage = tensor.storage();
  if (dynamic_cast<detail::MemoryState *>(&storage.resource().resource()) == nullptr)
    return std::unexpected(Status::InvalidArgument("Apple lease requires original Metal resource"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  try {
    auto allocator = options.allocator;
    if (options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
    }
    auto owner = std::allocate_shared<detail::LeasedArrayStorage>(
        mem::StdAllocator<detail::LeasedArrayStorage>(allocator));
    if (options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(owner->private_metadata,
                           options.resource_budget.reserve_unattributed(std::nullopt));
    }
    if (!bytes) {
      mx::array result(spec.shape, spec.dtype, nullptr, {});
      // No native payload is read. Retain the exact original zero-span lease
      // through the array descriptor, without inventing a nonempty backing.
      owner->source = std::move(source);
      result.set_data(mx::allocator::Buffer(nullptr),
                      [retained = std::move(owner)](mx::allocator::Buffer) {});
      result.set_status(mx::array::available);
      return result;
    }
    // Preserve the same original lease while interpreting its checked physical
    // span with the readonly stride metadata used by ordinary borrowed views.
    owner->source = std::move(source);
    return detail::readonly_tensor(tensor, spec, bytes,
        [retained = std::move(owner)](mx::allocator::Buffer) {});
  } catch (...) { return std::unexpected(detail::native_exception()); }
}

template<class Frame>static Result<mx::array> paged_binding(mem::MemoryResourceRef memory,
    Frame frame,LeaseImportOptions options)noexcept{
  if(!memory.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Apple paged binding requires a resource/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto tensor,compute::detail::PagedArenaAccess::tensor(frame));
  const auto storage=tensor.storage();auto description=tensor.description();
  if constexpr(std::is_same_v<Frame,PagedRowCacheRead>){const auto cap=frame.layout().schema().capacity();
    HYLE_ASSIGN_OR_RETURN(description,row_cache_carrier_description(cap.encoding,std::int64_t(std::uint64_t(cap.max_pages)*cap.page_size)));}
  if(storage.resource()!=memory || dynamic_cast<detail::MemoryState*>(&storage.resource().resource())==nullptr ||
      !description.is_dense())return std::unexpected(Status::InvalidArgument("Apple page arena belongs to another resource/carrier"));
  HYLE_ASSIGN_OR_RETURN(auto spec,input_spec(description));HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
  if(!bytes)return std::unexpected(Status::InvalidArgument("empty Apple page arena"));
  try{
    auto allocator=options.allocator;if(options.resource_budget.valid()){
      HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());
    }
    auto owner=std::allocate_shared<detail::PagedArrayStorage<Frame>>(mem::StdAllocator<detail::PagedArrayStorage<Frame>>(allocator));
    if(options.resource_budget.valid()){
      HYLE_ASSIGN_OR_RETURN(owner->private_metadata,options.resource_budget.reserve_unattributed(std::nullopt));
    }
    mx::array result(spec.shape,spec.dtype,nullptr,{});
    if(storage.offset()%result.itemsize()!=0 || bytes!=result.nbytes() ||
       storage.offset()/result.itemsize()>static_cast<std::size_t>(INT64_MAX))
      return std::unexpected(Status::InvalidArgument("Apple paged carrier origin/size mismatch"));
    id<MTLBuffer> metal=(__bridge id<MTLBuffer>)storage.native_handle();
    if(metal==nil || metal.storageMode!=MTLStorageModeShared || storage.offset()>metal.length || bytes>metal.length-storage.offset())
      return std::unexpected(Status::InvalidArgument("Apple paged capacity exceeds shared Metal backing"));
    owner->frame=std::move(frame);
    mx::array base(mx::Shape{1},mx::uint8,nullptr,{});
    base.set_data(mx::allocator::Buffer(storage.native_handle()),[retained=std::move(owner)](mx::allocator::Buffer){});
    result.copy_shared_buffer(base,result.strides(),result.flags(),result.size(),static_cast<std::int64_t>(storage.offset()/result.itemsize()));
    result.set_status(mx::array::available);return result;
  }catch(...){return std::unexpected(detail::native_exception());}
}

Result<mx::array> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedArenaRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(memory.ref(),std::move(frame),options);
}
Result<mx::array> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedRowCacheRead frame,LeaseImportOptions options)noexcept{return paged_binding(memory.ref(),std::move(frame),options);}
Result<mx::array> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedArenaPendingRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(memory.ref(),std::move(frame),options);
}
Result<mx::array> detail::PagedBatchAccess::binding(mem::MemoryResourceRef memory,
    PagedArenaRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(memory,std::move(frame),options);
}
Result<mx::array> detail::PagedBatchAccess::binding(mem::MemoryResourceRef memory,
    PagedArenaPendingRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(memory,std::move(frame),options);
}

Result<ReadLease> MemoryResource::import_readonly(const mx::array &source, const TensorDesc &description,
                                                LeaseImportOptions options) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Apple memory resource is empty"));
  return detail::MemoryAccess::export_readonly(source, description, options);
}
Status detail::MemoryAccess::match_readonly(const mx::array &source,
    const TensorDesc &description) noexcept {
  if (source.status() != mx::array::available)
    return Status::InvalidArgument("Apple import requires a ready native array");
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if (source.shape() != spec.shape || source.dtype() != spec.dtype || bytes == 0)
    return Status::InvalidArgument("Apple import description/carrier mismatch");
  if (!description.is_plain()) {
    if (!source.flags().row_contiguous || source.nbytes() != bytes)
      return Status::InvalidArgument("Apple encoded import requires its exact dense carrier");
    return Status::Ok();
  }
  const auto strides = description.byte_strides();
  const auto unit = source.itemsize();
  if (!unit) return Status::InvalidArgument("Apple readonly view has no element width");
  for (std::size_t axis = 0; axis < strides.size(); ++axis) {
    if (strides[axis] < 0)
      return Status::InvalidArgument("Apple readonly view has a negative byte stride");
    if (source.shape()[axis] <= 1) continue; // Unused singleton strides name the same bytes.
    if (source.strides()[axis] > SIZE_MAX / unit ||
        source.strides()[axis] * unit != static_cast<std::size_t>(strides[axis]))
      return Status::InvalidArgument("Apple readonly view stride differs from its native array");
  }
  return Status::Ok();
}
Result<ReadLease> detail::MemoryAccess::export_readonly(const mx::array &source,
    const TensorDesc &description, LeaseImportOptions options) noexcept {
  if (!options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Apple import metadata allocator is invalid"));
  HYLE_RETURN_IF_ERROR(match_readonly(source, description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto info, storage_info(source));
  if (info.view_bytes != bytes)
    return std::unexpected(Status::InvalidArgument("Apple readonly view physical span differs"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(detail::ImportedArrayStorage), alignof(detail::ImportedArrayStorage));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Apple import owner metadata"));
  auto *owner = std::construct_at(static_cast<detail::ImportedArrayStorage *>(storage), source, allocator);
  const auto destroy_owner = [](void *opaque) noexcept {
    auto *state = static_cast<detail::ImportedArrayStorage *>(opaque);
    const auto selected = state->allocator;
    std::destroy_at(state);
    selected.deallocate(state, sizeof(detail::ImportedArrayStorage), alignof(detail::ImportedArrayStorage));
  };
  if (options.resource_budget.valid()) {
    auto opaque = options.resource_budget.reserve_unattributed(std::nullopt);
    if (!opaque) { destroy_owner(owner); return std::unexpected(std::move(opaque).error()); }
    owner->private_metadata = std::move(*opaque);
  }
  auto backing = mem::Buffer::adopt(mem::MemoryResourceRef(*owner),
      {const_cast<void *>(info.buffer), info.backing_bytes, 1});
  if (!backing) { destroy_owner(owner); return std::unexpected(std::move(backing).error()); }
  auto imported = ReadLease::adopt(std::move(*backing), {.allocator = allocator});
  if (!imported) {
    backing->reset().IgnoreError();
    destroy_owner(owner);
    return std::unexpected(std::move(imported).error());
  }
  compute::detail::LeaseAccess::retain_resource_owner(*imported, owner, destroy_owner);
  return imported->subview(info.offset_bytes, bytes);
}

Result<NativeStorage> MemoryResource::storage_info(const mx::array &array) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Apple memory resource is empty"));
  return detail::MemoryAccess::storage_info(array);
}
Result<NativeStorage> detail::MemoryAccess::storage_info(const mx::array &array) noexcept {
  if (array.status() != mx::array::available)
    return std::unexpected(Status::FailedPrecondition("Apple storage requires a ready native array"));
  if (array.offset() < 0)
    return std::unexpected(Status::OutOfRange("Apple native array offset is negative"));
  // Logical nbytes excludes gaps and can understate an alias's permissions.
  // Bound the complete actual address span before inspecting any input owner.
  std::size_t last = 0;
  bool empty = false;
  for (const auto extent : array.shape()) {
    if (extent < 0)
      return std::unexpected(Status::InvalidArgument("Apple native array extent is negative"));
    empty |= extent == 0;
  }
  for (std::size_t axis = 0; !empty && axis < array.shape().size(); ++axis) {
    const auto extent = array.shape()[axis];
    const auto count = static_cast<std::size_t>(extent - 1);
    const auto stride = array.strides()[axis];
    if (stride != 0 && count > (SIZE_MAX - last) / stride)
      return std::unexpected(Status::OutOfRange("Apple native view address span overflows"));
    last += count * stride;
  }
  const auto unit = array.itemsize();
  if (!unit || (!empty && (last == SIZE_MAX || last + 1 > SIZE_MAX / unit)))
    return std::unexpected(Status::OutOfRange("Apple native view byte span overflows"));
  const auto bytes = empty ? std::size_t{0} : (last + 1) * unit;
  const auto handle = array.buffer().ptr();
  if (handle == nullptr) {
    if (bytes == 0) return NativeStorage{};
    return std::unexpected(Status::InvalidArgument("Apple array has no native backing"));
  }
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
  const auto offset = static_cast<std::size_t>(array.offset());
  if (buffer.storageMode != MTLStorageModeShared || offset > buffer.length ||
      bytes > buffer.length - offset)
    return std::unexpected(Status::InvalidArgument("Apple array exceeds shared native backing"));
  return NativeStorage{handle, offset, bytes, buffer.length};
}

namespace {
Status match_dense(const mx::array &array, const TensorDesc &description) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if ((description.is_plain() && !description.is_dense()) ||
      array.shape() != spec.shape || array.dtype() != spec.dtype ||
      !array.flags().row_contiguous || array.nbytes() != bytes)
    return Status::InvalidArgument("Apple boundary requires the exact dense native carrier");
  return Status::Ok();
}
} // namespace

Result<detail::CopySpan> detail::result_copy_span(const Results&result,std::size_t index,
    const TensorDesc&description)noexcept{
  const auto arrays=result.arrays();
  if(index>=arrays.size())return std::unexpected(Status::OutOfRange("Apple result copy output index"));
  const auto&array=arrays[index];
  if(array.status()!=mx::array::available)return std::unexpected(Status::Unavailable("Apple result backing is not ready on its original owner"));
  HYLE_RETURN_IF_ERROR(match_dense(array,description));
  if(array.offset()<0)return std::unexpected(Status::OutOfRange("Apple result copy origin is negative"));
  id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)array.buffer().ptr();
  const auto offset=static_cast<std::size_t>(array.offset());
  if(buffer==nil || buffer.storageMode!=MTLStorageModeShared || offset>buffer.length || array.nbytes()>buffer.length-offset)
    return std::unexpected(Status::InvalidArgument("Apple result copy exceeds original shared backing"));
  return CopySpan{const_cast<void*>(array.buffer().ptr()),{reinterpret_cast<std::uintptr_t>((__bridge void*)buffer.device),
    reinterpret_cast<std::uintptr_t>(array.buffer().ptr()),offset,array.nbytes(),4}};
}

Status MemoryResource::copy_from_host(const mx::array &destination, ConstTensorView source) const noexcept {
  if (!source.valid()) return Status::InvalidArgument("invalid Apple boundary source");
  HYLE_RETURN_IF_ERROR(match_dense(destination, source.description()));
  HYLE_ASSIGN_OR_RETURN(auto storage, storage_info(destination));
  if (storage.view_bytes == 0) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto mapping, source.storage().map_host());
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  std::memcpy(static_cast<std::byte *>(buffer.contents) + storage.offset_bytes,
              mapping.bytes().data(), storage.view_bytes);
  return mapping.close();
}
Status MemoryResource::copy_to_host(const mx::array &source, TensorView destination) const noexcept {
  if (!destination.valid()) return Status::InvalidArgument("invalid Apple boundary destination");
  HYLE_RETURN_IF_ERROR(match_dense(source, destination.description()));
  HYLE_ASSIGN_OR_RETURN(auto storage, storage_info(source));
  if (storage.view_bytes == 0) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto mapping, destination.storage().map_host(mem::HostAccess::kWrite));
  HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  std::memcpy(bytes.data(), static_cast<const std::byte *>(buffer.contents) + storage.offset_bytes,
              storage.view_bytes);
  return mapping.close();
}
} // namespace hyle::compute::apple
