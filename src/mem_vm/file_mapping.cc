#include "hyle/mem/mapping.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/mem/allocator.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hyle::mem {
namespace detail {

struct FileMappingFragment {
  void* data = nullptr;
  std::size_t size = 0;
};

}  // namespace detail
namespace {

using Fragment = detail::FileMappingFragment;

struct NativeRange {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool valid_access(MappingAccess access) noexcept {
  return access == MappingAccess::kReadOnly ||
         access == MappingAccess::kReadWrite ||
         access == MappingAccess::kCopyOnWrite;
}

bool valid_lock_options(ResidencyOptions options) noexcept {
  return (options.mode == ResidencyMode::kEager ||
          options.mode == ResidencyMode::kOnFault) &&
         (options.limit_policy == ResidencyLimitPolicy::kPreserve ||
          options.limit_policy == ResidencyLimitPolicy::kAdjustIfNeeded);
}

bool resolve_range(std::size_t total, std::size_t offset, std::size_t length,
                   std::size_t* span) noexcept {
  if (offset > total) return false;
  if (length == std::dynamic_extent) {
    *span = total - offset;
    return true;
  }
  if (length > total - offset) return false;
  *span = length;
  return true;
}

std::uintptr_t align_down(std::uintptr_t value, std::size_t alignment) noexcept {
  return value - value % alignment;
}

std::uintptr_t align_up(std::uintptr_t value, std::size_t alignment) noexcept {
  const std::uintptr_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

template <class T>
T* allocate_array(std::size_t count) noexcept {
  if (count == 0 ||
      count > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
    return nullptr;
  }
  return static_cast<T*>(
      system_allocator().allocate(count * sizeof(T), alignof(T)));
}

template <class T>
void deallocate_array(T* data, std::size_t count) noexcept {
  if (data != nullptr) {
    system_allocator().deallocate(data, count * sizeof(T), alignof(T));
  }
}

std::size_t current_fragment_count(const void* map_base,
                                   const Fragment* fragments,
                                   std::size_t count) noexcept {
  return fragments == nullptr ? (map_base == nullptr ? 0u : 1u) : count;
}

Fragment current_fragment_at(void* map_base, std::size_t map_len,
                             const Fragment* fragments,
                             std::size_t index) noexcept {
  return fragments == nullptr ? Fragment{map_base, map_len} : fragments[index];
}

// Partial release keeps fragments sorted by address. Locate the first fragment
// whose end is after `address` so checked access and small range controls do not
// rescan every earlier model fragment.
std::size_t first_fragment_ending_after(
    void* map_base, std::size_t map_len, const Fragment* fragments,
    std::size_t count, std::uintptr_t address) noexcept {
  if (fragments == nullptr) return 0;
  std::size_t first = 0;
  std::size_t last = count;
  while (first < last) {
    const std::size_t middle = first + (last - first) / 2;
    const Fragment fragment =
        current_fragment_at(map_base, map_len, fragments, middle);
    const std::uintptr_t end =
        reinterpret_cast<std::uintptr_t>(fragment.data) + fragment.size;
    if (end <= address) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first;
}

template <class Operation>
Status for_each_intersection(
    void* logical_base, std::size_t logical_size, void* map_base,
    std::size_t map_len, const Fragment* fragments, std::size_t fragment_count,
    std::size_t offset, std::size_t length, bool page_expand,
    Operation&& operation) noexcept {
  std::size_t span = 0;
  if (!resolve_range(logical_size, offset, length, &span)) {
    return Status::OutOfRange("mapped file range is outside logical extent");
  }
  if (span == 0) return Status::Ok();

  const std::uintptr_t requested_begin =
      reinterpret_cast<std::uintptr_t>(logical_base) + offset;
  const std::uintptr_t requested_end = requested_begin + span;
  const std::size_t page = platform::page_size();
  if (page == 0) {
    return Status::Unavailable("mapped file page size is unavailable");
  }

  bool visited = false;
  const std::size_t count =
      current_fragment_count(map_base, fragments, fragment_count);
  const std::size_t first = first_fragment_ending_after(
      map_base, map_len, fragments, count, requested_begin);
  for (std::size_t index = first; index < count; ++index) {
    const Fragment fragment =
        current_fragment_at(map_base, map_len, fragments, index);
    const std::uintptr_t fragment_begin =
        reinterpret_cast<std::uintptr_t>(fragment.data);
    const std::uintptr_t fragment_end = fragment_begin + fragment.size;
    if (fragment_begin >= requested_end) break;
    std::uintptr_t begin = (std::max)(requested_begin, fragment_begin);
    std::uintptr_t end = (std::min)(requested_end, fragment_end);
    if (begin >= end) continue;
    if (page_expand) {
      begin = (std::max)(align_down(begin, page), fragment_begin);
      end = (std::min)(align_up(end, page), fragment_end);
    }
    if (begin >= end) continue;
    visited = true;
    HYLE_RETURN_IF_ERROR(operation(reinterpret_cast<void*>(begin), end - begin));
  }
  return visited
             ? Status::Ok()
             : Status::FailedPrecondition(
                   "mapped file range contains no live pages");
}

std::size_t subtract_ranges(const Fragment* input, std::size_t input_count,
                            const NativeRange* removed,
                            std::size_t removed_count,
                            Fragment* output) noexcept {
  std::size_t output_count = 0;
  std::size_t removed_index = 0;
  for (std::size_t index = 0; index < input_count; ++index) {
    const std::uintptr_t fragment_begin =
        reinterpret_cast<std::uintptr_t>(input[index].data);
    const std::uintptr_t fragment_end = fragment_begin + input[index].size;
    std::uintptr_t cursor = fragment_begin;
    while (removed_index < removed_count &&
           removed[removed_index].end <= cursor) {
      ++removed_index;
    }
    std::size_t scan = removed_index;
    while (scan < removed_count && removed[scan].begin < fragment_end) {
      if (removed[scan].begin > cursor) {
        output[output_count++] = Fragment{
            reinterpret_cast<void*>(cursor), removed[scan].begin - cursor};
      }
      cursor = (std::max)(cursor, removed[scan].end);
      if (cursor >= fragment_end) break;
      ++scan;
    }
    if (cursor < fragment_end) {
      output[output_count++] = Fragment{
          reinterpret_cast<void*>(cursor), fragment_end - cursor};
    }
  }
  return output_count;
}

#if HYLE_OS_WINDOWS
Status native_mapping_error(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
}
#else
Status native_mapping_error(const char* operation, int code) noexcept {
  Status status = status_from_errno(code, operation);
  return code == ENOTDIR
             ? Status::FromNative(StatusCode::NotFound,
                                  status.native_error(), status.message())
             : status;
}
#endif

}  // namespace

Result<FileMapping> FileMapping::open(const char* path, MappingAccess access,
                                      std::size_t offset,
                                      std::size_t length) noexcept {
  if (!valid_access(access)) {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: invalid access"));
  }
  const unsigned flags = access == MappingAccess::kReadWrite
                             ? platform::kOpenRead | platform::kOpenWrite
                             : platform::kOpenRead;
  auto file = platform::open_file(path, flags);
  if (!file) {
    return std::unexpected(
        std::move(file).error().WithContext("mapped file open"));
  }
  return map(std::move(*file), access, offset, length);
}

Result<FileMapping> FileMapping::map(platform::FileHandle file,
                                     MappingAccess access,
                                     std::size_t offset,
                                     std::size_t length) noexcept {
  if (!file.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file map: handle is closed"));
  }
  if (!valid_access(access)) {
    return std::unexpected(
        Status::InvalidArgument("mapped file map: invalid access"));
  }

  FileMapping output;
#if HYLE_OS_WINDOWS
  HANDLE native_file = reinterpret_cast<HANDLE>(file.get());
  LARGE_INTEGER native_size{};
  if (!::GetFileSizeEx(native_file, &native_size)) {
    return std::unexpected(native_mapping_error("GetFileSizeEx",
                                                 ::GetLastError()));
  }
  if (native_size.QuadPart < 0) {
    return std::unexpected(
        Status::OutOfRange("mapped file has a negative size"));
  }
  const std::uint64_t file_size =
      static_cast<std::uint64_t>(native_size.QuadPart);
  if (offset > file_size ||
      (length != 0 && length > file_size - offset)) {
    return std::unexpected(Status::OutOfRange(
        "mapped file requested range extends past end of file"));
  }
  const std::uint64_t remaining = file_size - offset;
  if (remaining > (std::numeric_limits<std::size_t>::max)()) {
    return std::unexpected(
        Status::OutOfRange("mapped file range exceeds size_t"));
  }
  const std::size_t visible_size =
      length == 0 ? static_cast<std::size_t>(remaining) : length;
  if (visible_size == 0) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file mapping is empty"));
  }

  const DWORD protection = access == MappingAccess::kReadWrite
                               ? PAGE_READWRITE
                           : access == MappingAccess::kCopyOnWrite
                               ? PAGE_WRITECOPY
                               : PAGE_READONLY;
  const std::uint64_t maximum =
      static_cast<std::uint64_t>(offset) + visible_size;
  HANDLE mapping = ::CreateFileMappingW(
      native_file, nullptr, protection, static_cast<DWORD>(maximum >> 32),
      static_cast<DWORD>(maximum & 0xffffffffu), nullptr);
  if (mapping == nullptr) {
    return std::unexpected(native_mapping_error("CreateFileMappingW",
                                                 ::GetLastError()));
  }

  SYSTEM_INFO system{};
  ::GetSystemInfo(&system);
  const std::uint64_t granularity = system.dwAllocationGranularity;
  const std::uint64_t aligned_offset = offset / granularity * granularity;
  const std::size_t leading =
      static_cast<std::size_t>(offset - aligned_offset);
  if (visible_size > (std::numeric_limits<std::size_t>::max)() - leading) {
    ::CloseHandle(mapping);
    return std::unexpected(
        Status::OutOfRange("mapped file native view is too large"));
  }
  const DWORD desired_access = access == MappingAccess::kReadWrite
                                   ? FILE_MAP_WRITE
                               : access == MappingAccess::kCopyOnWrite
                                   ? FILE_MAP_COPY
                                   : FILE_MAP_READ;
  void* base = ::MapViewOfFile(
      mapping, desired_access, static_cast<DWORD>(aligned_offset >> 32),
      static_cast<DWORD>(aligned_offset & 0xffffffffu),
      visible_size + leading);
  if (base == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(mapping);
    return std::unexpected(native_mapping_error("MapViewOfFile", code));
  }
  output.addr_ = static_cast<std::byte*>(base) + leading;
  output.size_ = visible_size;
  output.map_base_ = base;
  output.map_len_ = visible_size + leading;
  output.h0_ = file.release();
  output.h1_ = reinterpret_cast<std::intptr_t>(mapping);
#else
  const int descriptor = static_cast<int>(file.get());
  struct ::stat native_size {};
  if (::fstat(descriptor, &native_size) != 0) {
    return std::unexpected(native_mapping_error("fstat", errno));
  }
  if (native_size.st_size < 0) {
    return std::unexpected(
        Status::OutOfRange("mapped file has a negative size"));
  }
  const std::uint64_t file_size =
      static_cast<std::uint64_t>(native_size.st_size);
  if (offset > file_size ||
      (length != 0 && length > file_size - offset)) {
    return std::unexpected(Status::OutOfRange(
        "mapped file requested range extends past end of file"));
  }
  const std::uint64_t remaining = file_size - offset;
  if (remaining > (std::numeric_limits<std::size_t>::max)()) {
    return std::unexpected(
        Status::OutOfRange("mapped file range exceeds size_t"));
  }
  const std::size_t visible_size =
      length == 0 ? static_cast<std::size_t>(remaining) : length;
  if (visible_size == 0) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file mapping is empty"));
  }

  const std::size_t page = platform::page_size();
  if (page == 0) {
    return std::unexpected(
        Status::Unavailable("mapped file page size is unavailable"));
  }
  const std::uint64_t aligned_offset = offset / page * page;
  if (aligned_offset >
      static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("mapped file offset exceeds off_t"));
  }
  const std::size_t leading =
      static_cast<std::size_t>(offset - aligned_offset);
  if (visible_size > (std::numeric_limits<std::size_t>::max)() - leading) {
    return std::unexpected(
        Status::OutOfRange("mapped file native view is too large"));
  }
  const int protection = access == MappingAccess::kReadOnly
                             ? PROT_READ
                             : PROT_READ | PROT_WRITE;
  const int flags = access == MappingAccess::kCopyOnWrite
                        ? MAP_PRIVATE
                        : MAP_SHARED;
  void* base = ::mmap(nullptr, visible_size + leading, protection, flags,
                      descriptor, static_cast<off_t>(aligned_offset));
  if (base == MAP_FAILED) {
    return std::unexpected(native_mapping_error("mmap", errno));
  }
  output.addr_ = static_cast<std::byte*>(base) + leading;
  output.size_ = visible_size;
  output.map_base_ = base;
  output.map_len_ = visible_size + leading;
#endif
  output.access_ = access;
  output.mapped_size_ = output.size_;
  return output;
}

std::size_t FileMapping::fragment_count() const noexcept {
  return valid() ? current_fragment_count(map_base_, fragments_, fragment_count_)
                 : 0u;
}

bool FileMapping::contains(std::size_t offset, std::size_t length) const noexcept {
  if (!valid()) return false;
  std::size_t span = 0;
  if (!resolve_range(size_, offset, length, &span)) return false;
  if (span == 0) return true;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(addr_) + offset;
  const std::uintptr_t end = begin + span;
  const std::size_t count = fragment_count();
  const std::size_t index = first_fragment_ending_after(
      map_base_, map_len_, fragments_, count, begin);
  if (index == count) return false;
  const Fragment fragment =
      current_fragment_at(map_base_, map_len_, fragments_, index);
  const std::uintptr_t fragment_begin =
      reinterpret_cast<std::uintptr_t>(fragment.data);
  const std::uintptr_t fragment_end = fragment_begin + fragment.size;
  return begin >= fragment_begin && end <= fragment_end;
}

Result<const void*> FileMapping::address(std::size_t offset,
                                         std::size_t length) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file is not open"));
  }
  std::size_t span = 0;
  if (!resolve_range(size_, offset, length, &span)) {
    return std::unexpected(
        Status::OutOfRange("mapped file range is outside logical extent"));
  }
  if (span != 0 && !contains(offset, span)) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file range crosses a released hole"));
  }
  return static_cast<const void*>(static_cast<const std::byte*>(addr_) + offset);
}

Result<void*> FileMapping::writable_address(std::size_t offset,
                                            std::size_t length) noexcept {
  if (access_ == MappingAccess::kReadOnly) {
    return std::unexpected(
        Status::PermissionDenied("mapped file is read-only"));
  }
  HYLE_ASSIGN_OR_RETURN(const void* pointer,
                        static_cast<const FileMapping&>(*this).address(offset,
                                                                       length));
  return const_cast<void*>(pointer);
}

Result<std::span<const std::byte>> FileMapping::bytes(
    std::size_t offset, std::size_t length) const noexcept {
  HYLE_ASSIGN_OR_RETURN(const void* pointer, address(offset, length));
  std::size_t span = 0;
  (void)resolve_range(size_, offset, length, &span);
  return std::span<const std::byte>{static_cast<const std::byte*>(pointer), span};
}

Result<std::span<std::byte>> FileMapping::writable_bytes(
    std::size_t offset, std::size_t length) noexcept {
  HYLE_ASSIGN_OR_RETURN(void* pointer, writable_address(offset, length));
  std::size_t span = 0;
  (void)resolve_range(size_, offset, length, &span);
  return std::span<std::byte>{static_cast<std::byte*>(pointer), span};
}

Status FileMapping::advise(MappingAdvice advice, std::size_t offset,
                           std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file is not open");
  }
  return for_each_intersection(
      addr_, size_, map_base_, map_len_, fragments_, fragment_count_, offset,
      length, true, [advice](void* data, std::size_t size) noexcept {
#if HYLE_OS_WINDOWS
        // DiscardVirtualMemory is not a file-cache eviction primitive: it
        // requires PAGE_READWRITE storage and makes the discarded bytes
        // undefined until the application rewrites them. Never apply those
        // anonymous/private semantics to a file-backed owner.
        if (advice == MappingAdvice::kDontNeed) {
          return Status::Unimplemented(
              "file-mapping dont-need advice is unavailable on Windows");
        }
#endif
        return platform::vm_advise(data, size, advice);
      });
}

Status FileMapping::flush(std::size_t offset, std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file is not open");
  }
  if (access_ != MappingAccess::kReadWrite) {
    return Status::FailedPrecondition(
        "only a shared read-write file mapping can be flushed");
  }
  std::size_t span = 0;
  if (!resolve_range(size_, offset, length, &span)) {
    return Status::OutOfRange("mapped file range is outside logical extent");
  }
  // An explicitly empty flush has no durability work. In particular, avoid
  // turning it into a whole-file FlushFileBuffers call on Windows.
  if (span == 0) return Status::Ok();
  Status status = for_each_intersection(
      addr_, size_, map_base_, map_len_, fragments_, fragment_count_, offset,
      length, true, [](void* data, std::size_t size) noexcept {
#if HYLE_OS_WINDOWS
        return ::FlushViewOfFile(data, size)
                   ? Status::Ok()
                   : native_mapping_error("FlushViewOfFile", ::GetLastError());
#else
        return ::msync(data, size, MS_SYNC) == 0
                   ? Status::Ok()
                   : native_mapping_error("msync", errno);
#endif
      });
#if HYLE_OS_WINDOWS
  if (status.ok() &&
      !::FlushFileBuffers(reinterpret_cast<HANDLE>(h0_))) {
    status = native_mapping_error("FlushFileBuffers", ::GetLastError());
  }
#endif
  return status;
}

Status FileMapping::populate(std::size_t offset, std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file is not open");
  }
  const std::size_t page = platform::page_size();
  return for_each_intersection(
      addr_, size_, map_base_, map_len_, fragments_, fragment_count_, offset,
      length, false, [page](void* data, std::size_t size) noexcept {
        auto* begin = static_cast<volatile const unsigned char*>(data);
        unsigned char observed = 0;
        std::size_t cursor = 0;
        while (cursor < size) {
          observed ^= begin[cursor];
          const std::uintptr_t address =
              reinterpret_cast<std::uintptr_t>(data) + cursor;
          const std::size_t advance = page - address % page;
          if (advance >= size - cursor) break;
          cursor += advance;
        }
        (void)observed;
        return Status::Ok();
      });
}

Result<std::size_t> FileMapping::release_range(std::size_t offset,
                                               std::size_t length) noexcept {
  const MappingRange range{offset, length};
  return release_ranges(std::span<const MappingRange>(&range, 1));
}

Result<std::size_t> FileMapping::release_ranges(
    std::span<const MappingRange> ranges) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file is not open"));
  }
  if (ranges.empty()) return std::size_t{0};
  for (const MappingRange range : ranges) {
    std::size_t span = 0;
    if (!resolve_range(size_, range.offset, range.size, &span)) {
      return std::unexpected(Status::OutOfRange(
          "mapped file release range is outside logical extent"));
    }
  }
  // A fully released logical owner remains valid so callers can safely replay
  // a release plan. Do not allocate metadata (and therefore cannot misreport
  // OOM) once no native fragments remain.
  if (fragment_count() == 0) return std::size_t{0};
  const std::size_t page = platform::page_size();
  if (page == 0) {
    return std::unexpected(
        Status::Unavailable("mapped file page size is unavailable"));
  }

  bool releases_page = false;
  for (const MappingRange range : ranges) {
    std::size_t span = 0;
    (void)resolve_range(size_, range.offset, range.size, &span);
    if (span == 0) continue;
    const std::uintptr_t requested_begin =
        reinterpret_cast<std::uintptr_t>(addr_) + range.offset;
    const std::uintptr_t requested_end = requested_begin + span;
    releases_page = releases_page ||
                    align_up(requested_begin, page) <
                        align_down(requested_end, page);
  }
  if (!releases_page) return std::size_t{0};

#if HYLE_OS_WINDOWS
  // A Win32 file view can only be unmapped as a whole. Bounds and native-page
  // coverage were established above without allocating/sorting metadata; now
  // report the unsupported operation without changing owner state.
  return std::unexpected(Status::Unimplemented(
      "partial file-mapping release is unavailable for a Windows whole view"));
#else

  NativeRange* normalized = allocate_array<NativeRange>(ranges.size());
  if (normalized == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "mapped file release range metadata allocation failed"));
  }
  std::size_t normalized_count = 0;
  for (const MappingRange range : ranges) {
    std::size_t span = 0;
    if (!resolve_range(size_, range.offset, range.size, &span)) {
      deallocate_array(normalized, ranges.size());
      return std::unexpected(Status::OutOfRange(
          "mapped file release range is outside logical extent"));
    }
    if (span == 0) continue;
    const std::uintptr_t requested_begin =
        reinterpret_cast<std::uintptr_t>(addr_) + range.offset;
    const std::uintptr_t requested_end = requested_begin + span;
    const std::uintptr_t begin = align_up(requested_begin, page);
    const std::uintptr_t end = align_down(requested_end, page);
    if (begin < end) normalized[normalized_count++] = {begin, end};
  }
  if (normalized_count == 0) {
    deallocate_array(normalized, ranges.size());
    return std::size_t{0};
  }
  std::sort(normalized, normalized + normalized_count,
            [](NativeRange left, NativeRange right) noexcept {
              return left.begin < right.begin ||
                     (left.begin == right.begin && left.end < right.end);
            });
  std::size_t merged_count = 0;
  for (std::size_t index = 0; index < normalized_count; ++index) {
    if (merged_count == 0 ||
        normalized[index].begin > normalized[merged_count - 1].end) {
      normalized[merged_count++] = normalized[index];
    } else {
      normalized[merged_count - 1].end =
          (std::max)(normalized[merged_count - 1].end,
                     normalized[index].end);
    }
  }

  const std::size_t live_count = fragment_count();
  if (live_count > (std::numeric_limits<std::size_t>::max)() - merged_count) {
    deallocate_array(normalized, ranges.size());
    return std::unexpected(
        Status::OutOfRange("mapped file fragment count overflow"));
  }
  const std::size_t intersection_capacity = live_count + merged_count;
  NativeRange* intersections =
      allocate_array<NativeRange>(intersection_capacity);
  Fragment* live = allocate_array<Fragment>(live_count);
  if (intersections == nullptr || live == nullptr) {
    deallocate_array(intersections, intersection_capacity);
    deallocate_array(live, live_count);
    deallocate_array(normalized, ranges.size());
    return std::unexpected(Status::ResourceExhausted(
        "mapped file fragment metadata allocation failed"));
  }
  for (std::size_t index = 0; index < live_count; ++index) {
    live[index] = current_fragment_at(map_base_, map_len_, fragments_, index);
  }

  std::size_t intersection_count = 0;
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < live_count && right < merged_count) {
    const std::uintptr_t live_begin =
        reinterpret_cast<std::uintptr_t>(live[left].data);
    const std::uintptr_t live_end = live_begin + live[left].size;
    const std::uintptr_t begin =
        (std::max)(live_begin, normalized[right].begin);
    const std::uintptr_t end =
        (std::min)(live_end, normalized[right].end);
    if (begin < end) intersections[intersection_count++] = {begin, end};
    if (live_end <= normalized[right].end) {
      ++left;
    } else {
      ++right;
    }
  }
  deallocate_array(normalized, ranges.size());
  if (intersection_count == 0) {
    deallocate_array(intersections, intersection_capacity);
    deallocate_array(live, live_count);
    return std::size_t{0};
  }

  const std::uintptr_t locked_end =
      reinterpret_cast<std::uintptr_t>(addr_) + locked_prefix_size_;
  for (std::size_t index = 0; index < intersection_count; ++index) {
    if (locked_prefix_size_ != 0 && intersections[index].begin < locked_end) {
      deallocate_array(intersections, intersection_capacity);
      deallocate_array(live, live_count);
      return std::unexpected(Status::FailedPrecondition(
          "unlock the file mapping before releasing its locked prefix"));
    }
  }

  const std::size_t output_capacity = live_count + intersection_count;
  Fragment* remaining = allocate_array<Fragment>(output_capacity);
  if (remaining == nullptr) {
    deallocate_array(intersections, intersection_capacity);
    deallocate_array(live, live_count);
    return std::unexpected(Status::ResourceExhausted(
        "mapped file remaining-fragment allocation failed"));
  }

  std::size_t completed = 0;
  Status failure;
  for (; completed < intersection_count; ++completed) {
    const NativeRange range = intersections[completed];
    if (::munmap(reinterpret_cast<void*>(range.begin),
                 range.end - range.begin) != 0) {
      failure = native_mapping_error("munmap(partial file mapping)", errno);
      break;
    }
  }
  const std::size_t remaining_count =
      subtract_ranges(live, live_count, intersections, completed, remaining);

  const std::uintptr_t logical_begin =
      reinterpret_cast<std::uintptr_t>(addr_);
  const std::uintptr_t logical_end = logical_begin + size_;
  std::size_t released = 0;
  for (std::size_t index = 0; index < completed; ++index) {
    const std::uintptr_t begin =
        (std::max)(logical_begin, intersections[index].begin);
    const std::uintptr_t end =
        (std::min)(logical_end, intersections[index].end);
    if (begin < end) released += end - begin;
  }
  mapped_size_ -= released;
  deallocate_array(fragments_, fragment_capacity_);
  fragments_ = remaining;
  fragment_count_ = remaining_count;
  fragment_capacity_ = output_capacity;
  deallocate_array(intersections, intersection_capacity);
  deallocate_array(live, live_count);
  if (!failure.ok()) return std::unexpected(std::move(failure));
  return released;
#endif
}

Status FileMapping::lock_prefix(std::size_t length,
                                ResidencyOptions options) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file is not open");
  }
  if (!valid_lock_options(options)) {
    return Status::InvalidArgument("mapped file lock options are invalid");
  }
  if (length > size_) {
    return Status::OutOfRange("mapped file lock prefix exceeds logical size");
  }
  if (length != 0 && !contains(0, length)) {
    return Status::FailedPrecondition(
        "mapped file lock prefix crosses a released hole");
  }
  if (locked_prefix_size_ != 0 && options.mode != locked_mode_) {
    return Status::FailedPrecondition(
        "mapped file lock mode cannot change while locked");
  }
  // A request already covered by the monotonic prefix is an idempotent no-op;
  // it never shrinks the native lock. This keeps callers from having to cache
  // the largest prior request while still making a mode change explicit.
  if (length <= locked_prefix_size_) return Status::Ok();

  const std::size_t count = fragment_count();
  NativeRange* ranges = allocate_array<NativeRange>(count);
  if (ranges == nullptr) {
    return Status::ResourceExhausted(
        "mapped file lock range metadata allocation failed");
  }
  const std::size_t page = platform::page_size();
  const std::uintptr_t requested_begin =
      reinterpret_cast<std::uintptr_t>(addr_) + locked_prefix_size_;
  const std::uintptr_t requested_end =
      reinterpret_cast<std::uintptr_t>(addr_) + length;
  std::size_t range_count = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const Fragment fragment =
        current_fragment_at(map_base_, map_len_, fragments_, index);
    const std::uintptr_t fragment_begin =
        reinterpret_cast<std::uintptr_t>(fragment.data);
    const std::uintptr_t fragment_end = fragment_begin + fragment.size;
    std::uintptr_t begin = (std::max)(requested_begin, fragment_begin);
    std::uintptr_t end = (std::min)(requested_end, fragment_end);
    if (begin >= end) continue;
    begin = locked_prefix_size_ == 0
                ? (std::max)(align_down(begin, page), fragment_begin)
                : (std::max)(align_up(begin, page), fragment_begin);
    end = (std::min)(align_up(end, page), fragment_end);
    if (begin < end) ranges[range_count++] = {begin, end};
  }

  std::size_t completed = 0;
  Status failure;
  for (; completed < range_count; ++completed) {
    const NativeRange range = ranges[completed];
    failure = platform::vm_lock(reinterpret_cast<void*>(range.begin),
                                range.end - range.begin, options);
    if (!failure.ok()) break;
  }
  if (!failure.ok()) {
    for (std::size_t index = 0; index < completed; ++index) {
      const NativeRange range = ranges[index];
      platform::vm_unlock(reinterpret_cast<void*>(range.begin),
                          range.end - range.begin)
          .IgnoreError();
    }
    deallocate_array(ranges, count);
    return failure;
  }
  deallocate_array(ranges, count);
  locked_prefix_size_ = length;
  locked_mode_ = options.mode;
  return Status::Ok();
}

Status FileMapping::lock(ResidencyOptions options) noexcept {
  return lock_prefix(size_, options);
}

Status FileMapping::unlock() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file is not open");
  }
  if (locked_prefix_size_ == 0) return Status::Ok();
  Status first;
  const Status visited = for_each_intersection(
      addr_, size_, map_base_, map_len_, fragments_, fragment_count_, 0,
      locked_prefix_size_, true,
      [&first](void* data, std::size_t size) noexcept {
        Status status = platform::vm_unlock(data, size);
        if (first.ok() && !status.ok()) first = status;
        return Status::Ok();
      });
  if (!visited.ok() && first.ok()) first = visited;
  if (first.ok()) {
    locked_prefix_size_ = 0;
    locked_mode_ = ResidencyMode::kEager;
  }
  return first;
}

Status FileMapping::reset() noexcept {
  Status first;
#if HYLE_OS_WINDOWS
  if (map_base_ != nullptr && !::UnmapViewOfFile(map_base_)) {
    first = native_mapping_error("UnmapViewOfFile", ::GetLastError());
  }
  if (h1_ != -1 && !::CloseHandle(reinterpret_cast<HANDLE>(h1_)) &&
      first.ok()) {
    first = native_mapping_error("CloseHandle(file mapping)",
                                 ::GetLastError());
  }
  if (h0_ != -1 && !::CloseHandle(reinterpret_cast<HANDLE>(h0_)) &&
      first.ok()) {
    first = native_mapping_error("CloseHandle(mapped file)",
                                 ::GetLastError());
  }
#else
  const std::size_t count = fragment_count();
  for (std::size_t index = 0; index < count; ++index) {
    const Fragment fragment =
        current_fragment_at(map_base_, map_len_, fragments_, index);
    if (::munmap(fragment.data, fragment.size) != 0 && first.ok()) {
      first = native_mapping_error("munmap(file mapping)", errno);
    }
  }
#endif
  deallocate_array(fragments_, fragment_capacity_);
  addr_ = nullptr;
  size_ = 0;
  map_base_ = nullptr;
  map_len_ = 0;
  h0_ = -1;
  h1_ = -1;
  access_ = MappingAccess::kReadOnly;
  locked_prefix_size_ = 0;
  locked_mode_ = ResidencyMode::kEager;
  fragments_ = nullptr;
  fragment_count_ = 0;
  fragment_capacity_ = 0;
  mapped_size_ = 0;
  return first;
}

FileMapping::~FileMapping() { reset().IgnoreError(); }

FileMapping::FileMapping(FileMapping&& other) noexcept
    : addr_(std::exchange(other.addr_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      map_base_(std::exchange(other.map_base_, nullptr)),
      map_len_(std::exchange(other.map_len_, 0)),
      h0_(std::exchange(other.h0_, -1)),
      h1_(std::exchange(other.h1_, -1)),
      access_(std::exchange(other.access_, MappingAccess::kReadOnly)),
      locked_prefix_size_(std::exchange(other.locked_prefix_size_, 0)),
      locked_mode_(std::exchange(other.locked_mode_, ResidencyMode::kEager)),
      fragments_(std::exchange(other.fragments_, nullptr)),
      fragment_count_(std::exchange(other.fragment_count_, 0)),
      fragment_capacity_(std::exchange(other.fragment_capacity_, 0)),
      mapped_size_(std::exchange(other.mapped_size_, 0)) {}

FileMapping& FileMapping::operator=(FileMapping&& other) noexcept {
  if (this != &other) {
    reset().IgnoreError();
    addr_ = std::exchange(other.addr_, nullptr);
    size_ = std::exchange(other.size_, 0);
    map_base_ = std::exchange(other.map_base_, nullptr);
    map_len_ = std::exchange(other.map_len_, 0);
    h0_ = std::exchange(other.h0_, -1);
    h1_ = std::exchange(other.h1_, -1);
    access_ = std::exchange(other.access_, MappingAccess::kReadOnly);
    locked_prefix_size_ = std::exchange(other.locked_prefix_size_, 0);
    locked_mode_ = std::exchange(other.locked_mode_, ResidencyMode::kEager);
    fragments_ = std::exchange(other.fragments_, nullptr);
    fragment_count_ = std::exchange(other.fragment_count_, 0);
    fragment_capacity_ = std::exchange(other.fragment_capacity_, 0);
    mapped_size_ = std::exchange(other.mapped_size_, 0);
  }
  return *this;
}

}  // namespace hyle::mem
