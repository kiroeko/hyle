#include "hyle/mem/mapping.h"

#include <utility>

#include "hyle/platform/fs.h"

namespace hyle::mem {
namespace {

Result<platform::MappedFile::Mode> file_mode(MappingAccess access) noexcept {
  switch (access) {
    case MappingAccess::kReadOnly:
      return platform::MappedFile::Mode::kReadOnly;
    case MappingAccess::kReadWrite:
      return platform::MappedFile::Mode::kReadWrite;
    case MappingAccess::kCopyOnWrite:
      return platform::MappedFile::Mode::kCopyOnWrite;
  }
  return std::unexpected(Status::InvalidArgument("invalid mapping access"));
}

Result<platform::MappedFile::Advice> file_advice(
    MappingAdvice advice) noexcept {
  switch (advice) {
    case MappingAdvice::kNormal:
      return platform::MappedFile::Advice::kNormal;
    case MappingAdvice::kRandom:
      return platform::MappedFile::Advice::kRandom;
    case MappingAdvice::kSequential:
      return platform::MappedFile::Advice::kSequential;
    case MappingAdvice::kWillNeed:
      return platform::MappedFile::Advice::kWillNeed;
    case MappingAdvice::kDontNeed:
      return platform::MappedFile::Advice::kDontNeed;
    case MappingAdvice::kHugePage:
    case MappingAdvice::kNoHugePage:
      return std::unexpected(Status::Unimplemented(
          "huge-page advice is not a portable file-mapping contract"));
  }
  return std::unexpected(Status::InvalidArgument("invalid mapping advice"));
}

Result<platform::MappedFile::LockOptions> file_lock_options(
    ResidencyOptions options) noexcept {
  platform::MappedFile::LockOptions out;
  switch (options.mode) {
    case ResidencyMode::kEager:
      out.mode = platform::MappedFile::LockMode::kEager;
      break;
    case ResidencyMode::kOnFault:
      out.mode = platform::MappedFile::LockMode::kOnFault;
      break;
    default:
      return std::unexpected(Status::InvalidArgument("invalid residency mode"));
  }
  switch (options.limit_policy) {
    case ResidencyLimitPolicy::kPreserve:
      out.limit_policy = platform::MappedFile::LockLimitPolicy::kPreserve;
      break;
    case ResidencyLimitPolicy::kAdjustIfNeeded:
      out.limit_policy =
          platform::MappedFile::LockLimitPolicy::kAdjustIfNeeded;
      break;
    default:
      return std::unexpected(
          Status::InvalidArgument("invalid residency limit policy"));
  }
  return out;
}

Result<platform::VmLockOptions> vm_lock_options(
    ResidencyOptions options) noexcept {
  platform::VmLockOptions out;
  switch (options.mode) {
    case ResidencyMode::kEager:
      out.mode = platform::VmLockMode::kEager;
      break;
    case ResidencyMode::kOnFault:
      out.mode = platform::VmLockMode::kOnFault;
      break;
    default:
      return std::unexpected(Status::InvalidArgument("invalid residency mode"));
  }
  switch (options.limit_policy) {
    case ResidencyLimitPolicy::kPreserve:
      out.limit_policy = platform::VmLockLimitPolicy::kPreserve;
      break;
    case ResidencyLimitPolicy::kAdjustIfNeeded:
      out.limit_policy = platform::VmLockLimitPolicy::kAdjustIfNeeded;
      break;
    default:
      return std::unexpected(
          Status::InvalidArgument("invalid residency limit policy"));
  }
  return out;
}

bool mapping_range(const void* data, std::size_t total, std::size_t offset,
                   std::size_t requested, void** begin,
                   std::size_t* size) noexcept {
  if (data == nullptr || offset > total ||
      (requested != 0 && requested > total - offset)) {
    return false;
  }
  *begin = static_cast<std::byte*>(const_cast<void*>(data)) + offset;
  *size = requested != 0 ? requested : total - offset;
  return true;
}

class BorrowedMappedResource final : public MemoryResource {
 public:
  explicit BorrowedMappedResource(bool writable) noexcept
      : writable_(writable) {}

  std::string_view name() const noexcept override {
    return writable_ ? "hyle.mapping.rw" : "hyle.mapping.ro";
  }
  MemoryProperties properties() const noexcept override {
    return MemoryProperties{
        .flags = MemoryProperty::kHostAccessible |
                 MemoryProperty::kHostDeviceCoherent |
                 MemoryProperty::kPersistent,
        .minimum_alignment = 1,
        .maximum_allocation = SIZE_MAX,
        .domain = MemoryDomainId{this, 0},
    };
  }
  Result<ResourceAllocation> allocate(std::size_t,
                                      std::size_t) noexcept override {
    return std::unexpected(Status::Unimplemented(
        "borrowed mapping resource cannot allocate storage"));
  }
  void deallocate(ResourceAllocation) noexcept override {}
  Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess access) noexcept override {
    if (!writable_ && access != HostAccess::kRead) {
      return std::unexpected(
          Status::PermissionDenied("file mapping is read-only"));
    }
    return ResourceHostMapping{
        size == 0
            ? nullptr
            : static_cast<void*>(static_cast<std::byte*>(native_handle) +
                                 offset),
        nullptr};
  }

 private:
  bool writable_;
};

MemoryResourceRef mapped_resource(bool writable) noexcept {
  static BorrowedMappedResource read_only(false);
  static BorrowedMappedResource read_write(true);
  return MemoryResourceRef(writable ? static_cast<MemoryResource&>(read_write)
                                    : static_cast<MemoryResource&>(read_only));
}

}  // namespace

Result<VirtualMapping> VirtualMapping::reserve(std::size_t size) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::VmRegion region,
                        platform::VmRegion::reserve(size));
  return VirtualMapping(std::move(region));
}

Result<VirtualMapping> VirtualMapping::allocate_read_write(
    std::size_t size) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::VmRegion region,
                        platform::VmRegion::allocate_rw(size));
  return VirtualMapping(std::move(region));
}

Result<VirtualMapping> VirtualMapping::reserve_numa(std::size_t size,
                                                    int node) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::VmRegion region,
                        platform::VmRegion::reserve_numa(size, node));
  return VirtualMapping(std::move(region));
}

Result<VirtualMapping> VirtualMapping::allocate_numa_interleaved(
    std::size_t size, std::span<const unsigned> nodes,
    std::size_t bytes_per_chunk) noexcept {
  HYLE_ASSIGN_OR_RETURN(
      platform::VmRegion region,
      platform::VmRegion::allocate_numa_interleaved(size, nodes,
                                                    bytes_per_chunk));
  return VirtualMapping(std::move(region));
}

Status VirtualMapping::commit(std::size_t offset, std::size_t size) noexcept {
  return region_.commit(offset, size);
}

Status VirtualMapping::decommit(std::size_t offset, std::size_t size) noexcept {
  return region_.decommit(offset, size);
}

Status VirtualMapping::protect(std::size_t offset, std::size_t size,
                               MappingProtection protection) noexcept {
  platform::Protect native;
  switch (protection) {
    case MappingProtection::kNone:
      native = platform::Protect::kNone;
      break;
    case MappingProtection::kRead:
      native = platform::Protect::kRead;
      break;
    case MappingProtection::kReadWrite:
      native = platform::Protect::kReadWrite;
      break;
    default:
      return Status::InvalidArgument("invalid mapping protection");
  }
  return region_.protect(offset, size, native);
}

Status VirtualMapping::advise(MappingAdvice advice, std::size_t offset,
                              std::size_t size) noexcept {
  void* begin = nullptr;
  std::size_t span = 0;
  if (!mapping_range(data(), this->size(), offset, size, &begin, &span)) {
    return valid() ? Status::OutOfRange("mapping advice range is out of bounds")
                   : Status::FailedPrecondition("mapping is invalid");
  }
  return platform::vm_advise(begin, span,
                             static_cast<platform::VmAdvice>(advice));
}

Status VirtualMapping::lock(std::size_t offset, std::size_t size,
                            ResidencyOptions options) noexcept {
  void* begin = nullptr;
  std::size_t span = 0;
  if (!mapping_range(data(), this->size(), offset, size, &begin, &span)) {
    return valid() ? Status::OutOfRange("mapping lock range is out of bounds")
                   : Status::FailedPrecondition("mapping is invalid");
  }
  HYLE_ASSIGN_OR_RETURN(platform::VmLockOptions native,
                        vm_lock_options(options));
  return platform::vm_lock(begin, span, native);
}

Status VirtualMapping::unlock(std::size_t offset, std::size_t size) noexcept {
  void* begin = nullptr;
  std::size_t span = 0;
  if (!mapping_range(data(), this->size(), offset, size, &begin, &span)) {
    return valid() ? Status::OutOfRange("mapping unlock range is out of bounds")
                   : Status::FailedPrecondition("mapping is invalid");
  }
  return platform::vm_unlock(begin, span);
}

Status VirtualMapping::release() noexcept { return region_.release(); }

Result<FileMapping> FileMapping::open(const char* path, MappingAccess access,
                                      std::size_t offset,
                                      std::size_t length) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::MappedFile::Mode mode, file_mode(access));
  HYLE_ASSIGN_OR_RETURN(platform::MappedFile mapping,
                        platform::MappedFile::open(path, mode, offset, length));
  return FileMapping(std::move(mapping), access);
}

Result<FileMapping> FileMapping::map(platform::FileHandle file,
                                     MappingAccess access,
                                     std::size_t offset,
                                     std::size_t length) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::MappedFile::Mode mode, file_mode(access));
  HYLE_ASSIGN_OR_RETURN(
      platform::MappedFile mapping,
      platform::MappedFile::map(std::move(file), mode, offset, length));
  return FileMapping(std::move(mapping), access);
}

ConstBufferView FileMapping::view() const noexcept {
  if (!valid()) return {};
  return ConstBufferView(mapped_resource(false), const_cast<void*>(data()), 0,
                         size());
}

Result<std::span<std::byte>> FileMapping::writable_bytes() noexcept {
  return mapping_.writable_bytes();
}

Status FileMapping::flush(std::size_t offset, std::size_t length) noexcept {
  return mapping_.flush(offset, length);
}

Status FileMapping::advise(MappingAdvice advice, std::size_t offset,
                           std::size_t length) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::MappedFile::Advice native,
                        file_advice(advice));
  return mapping_.advise(native, offset, length);
}

Status FileMapping::lock_prefix(std::size_t length,
                                ResidencyOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::MappedFile::LockOptions native,
                        file_lock_options(options));
  return mapping_.lock_prefix(length, native);
}

Status FileMapping::lock(ResidencyOptions options) noexcept {
  return lock_prefix(size(), options);
}

Status FileMapping::unlock() noexcept { return mapping_.unlock(); }

void FileMapping::reset() noexcept { mapping_.reset(); }

Result<SharedMapping> SharedMapping::create(const char* name,
                                            std::size_t size) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::SharedMemoryRegion mapping,
                        platform::SharedMemoryRegion::create(name, size));
  return SharedMapping(std::move(mapping));
}

Result<SharedMapping> SharedMapping::open(const char* name) noexcept {
  HYLE_ASSIGN_OR_RETURN(platform::SharedMemoryRegion mapping,
                        platform::SharedMemoryRegion::open(name));
  return SharedMapping(std::move(mapping));
}

Status SharedMapping::unlink(const char* name) noexcept {
  return platform::SharedMemoryRegion::unlink(name);
}

BufferView SharedMapping::view() noexcept {
  if (!valid()) return {};
  return BufferView(mapped_resource(true), data(), 0, size());
}

ConstBufferView SharedMapping::view() const noexcept {
  if (!valid()) return {};
  return ConstBufferView(mapped_resource(false), const_cast<void*>(data()), 0,
                         size());
}

Status SharedMapping::flush() noexcept { return mapping_.flush(); }

Status SharedMapping::reset() noexcept { return mapping_.reset(); }

}  // namespace hyle::mem
