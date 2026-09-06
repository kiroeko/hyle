#include "hyle/mem/mapping.h"

namespace hyle::mem {
namespace {

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
          Status::PermissionDenied("mapping is read-only"));
    }
    return ResourceHostMapping{
        size == 0 ? nullptr
                  : static_cast<void*>(
                        static_cast<std::byte*>(native_handle) + offset),
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

std::size_t resolved_length(std::size_t total, std::size_t offset,
                            std::size_t length) noexcept {
  return length == std::dynamic_extent ? total - offset : length;
}

}  // namespace

Result<ConstBufferView> FileMapping::view(std::size_t offset,
                                          std::size_t length) const noexcept {
  HYLE_ASSIGN_OR_RETURN(const void* pointer, address(offset, length));
  const std::size_t span = resolved_length(size_, offset, length);
  return ConstBufferView(mapped_resource(false), const_cast<void*>(pointer), 0,
                         span);
}

Result<BufferView> FileMapping::writable_view(std::size_t offset,
                                              std::size_t length) noexcept {
  HYLE_ASSIGN_OR_RETURN(void* pointer, writable_address(offset, length));
  const std::size_t span = resolved_length(size_, offset, length);
  return BufferView(mapped_resource(true), pointer, 0, span);
}

ConstBufferView SharedMapping::view() const noexcept {
  if (!valid()) return {};
  return ConstBufferView(mapped_resource(false), mapping_, 0, size_);
}

Result<BufferView> SharedMapping::writable_view() noexcept {
  HYLE_ASSIGN_OR_RETURN(void* pointer, writable_data());
  return BufferView(mapped_resource(true), pointer, 0, size_);
}

}  // namespace hyle::mem
