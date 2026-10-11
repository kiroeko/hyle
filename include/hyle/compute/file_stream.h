#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/batch_transfer.h"
#include "hyle/io/file.h"
#include "hyle/mem/mapping.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_STREAM_API
#elif defined(HYLE_COMPUTE_STREAM_BUILD)
#define HYLE_COMPUTE_STREAM_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_STREAM_API HYLE_IMPORT
#endif

namespace hyle::compute {
namespace detail {
struct StreamFileState;
struct StreamMappingState;
struct FileStreamState;
struct FileStreamAccess;
}

struct FileStreamOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
  std::uint32_t max_requests = 4, max_groups_per_request = 16;
  std::uint32_t max_ranges_per_group = 16, in_flight_slots = 2;
  std::size_t staging_bytes_per_slot = 1024u * 1024u;
  std::size_t staging_alignment = 256;
  std::size_t max_active_backing_bytes = 128u * 1024u * 1024u;
};

// Retained open-file owner, or explicit borrow of a stable File address.
// Capacity bounds admission; actual EOF/short transfers still come from IO.
// IoContext, native resources and injected allocators remain borrowed through
// every actual use, including abandoned requests. Adopt failure keeps File.
class StreamFile final {
public:
  StreamFile() noexcept = default;
  HYLE_COMPUTE_STREAM_API StreamFile(const StreamFile &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamFile &operator=(const StreamFile &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamFile(StreamFile &&) noexcept;
  HYLE_COMPUTE_STREAM_API StreamFile &operator=(StreamFile &&) noexcept;
  HYLE_COMPUTE_STREAM_API ~StreamFile();
  HYLE_COMPUTE_STREAM_API static Result<StreamFile> adopt(
      io::File &&file, std::uint64_t capacity_bytes,
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  HYLE_COMPUTE_STREAM_API static Result<StreamFile> borrow(
      io::File &file, std::uint64_t capacity_bytes,
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_STREAM_API std::uint64_t capacity_bytes() const noexcept;
  HYLE_COMPUTE_STREAM_API void reset() noexcept;
private:
  friend struct detail::FileStreamAccess;
  explicit StreamFile(detail::StreamFileState *state) noexcept : state_(state) {}
  detail::StreamFileState *state_ = nullptr;
};

// Immutable mapping sources must not be modified/unmapped/released while used.
// Mapping offsets are relative to its existing logical range. No populate or
// whole-file read is implicit. A borrowed mapping stays at its original address.
class StreamSource final {
public:
  StreamSource() noexcept = default;
  HYLE_COMPUTE_STREAM_API StreamSource(const StreamSource &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamSource &operator=(const StreamSource &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamSource(StreamSource &&) noexcept;
  HYLE_COMPUTE_STREAM_API StreamSource &operator=(StreamSource &&) noexcept;
  HYLE_COMPUTE_STREAM_API ~StreamSource();
  static StreamSource file(StreamFile file) noexcept {
    StreamSource source; source.file_ = std::move(file); return source;
  }
  HYLE_COMPUTE_STREAM_API static Result<StreamSource> adopt(
      mem::FileMapping &&mapping,
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  HYLE_COMPUTE_STREAM_API static Result<StreamSource> borrow(
      const mem::FileMapping &mapping,
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return file_.valid() || mapping_ != nullptr; }
  HYLE_COMPUTE_STREAM_API std::uint64_t capacity_bytes() const noexcept;
private:
  friend struct detail::FileStreamAccess;
  StreamFile file_;
  detail::StreamMappingState *mapping_ = nullptr;
};

struct StreamReadRange final {
  std::uint64_t file_offset = 0;
  WriteLease destination;
  std::size_t resource_offset = 0, bytes = 0, physical_bytes = 0;
  std::optional<TensorDesc> description;
};
struct StreamReadGroup final {
  std::uint64_t tag = 0;
  std::span<StreamReadRange> ranges;
};
struct StreamWriteRange final {
  ReadLease source;
  WriteLease pending_source;
  Completion producer;
  std::size_t resource_offset = 0, bytes = 0, physical_bytes = 0;
  std::uint64_t file_offset = 0;
  HYLE_COMPUTE_STREAM_API static StreamWriteRange ready(ReadLease source, std::uint64_t file_offset,
      std::size_t resource_offset, std::size_t bytes,
      std::size_t physical_bytes = 0) noexcept;
  HYLE_COMPUTE_STREAM_API static StreamWriteRange produced(WriteLease &&source, Completion &&producer,
      std::uint64_t file_offset, std::size_t resource_offset, std::size_t bytes,
      std::size_t physical_bytes = 0) noexcept;
};
struct StreamWriteGroup final {
  std::uint64_t tag = 0;
  std::span<StreamWriteRange> ranges;
};

// Partial progress and first error survive terminal failure. Retirement is
// separate: an unknown native observation cannot release or reuse storage.
struct StreamResult final {
  Status status{};
  std::uint64_t read_bytes = 0, copied_bytes = 0, written_bytes = 0;
  std::uint32_t ready_groups = 0, total_groups = 0;
  bool complete = false, retirement_proven = true;
};
struct FileStreamReport final {
  // Known composite/copy-controller arrays. Original IO operation, Mem lease
  // control and BlockingPool job allocations remain in their injected allocator
  // ledgers; this field does not claim they allocate zero bytes.
  std::size_t staging_bytes = 0, metadata_bytes = 0;
  std::size_t active_backing_bytes = 0, peak_backing_bytes = 0;
  std::size_t external_backing_bytes = 0;
  std::uint32_t live_requests = 0, active_slots = 0, peak_slots = 0;
  std::uint64_t io_calls = 0, mapping_copy_bytes = 0;
  std::uint64_t copied_physical_bytes = 0; // successful native-retired copies, including explicit padding
  std::uint64_t native_calls = 0, native_waits = 0, phase_completions = 0, host_waits = 0;
  bool native_metadata_unknown = false;
};

class StreamReadyGroup final {
public:
  StreamReadyGroup() noexcept = default;
  HYLE_COMPUTE_STREAM_API StreamReadyGroup(const StreamReadyGroup &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamReadyGroup &operator=(const StreamReadyGroup &) noexcept;
  HYLE_COMPUTE_STREAM_API StreamReadyGroup(StreamReadyGroup &&) noexcept;
  HYLE_COMPUTE_STREAM_API StreamReadyGroup &operator=(StreamReadyGroup &&) noexcept;
  HYLE_COMPUTE_STREAM_API ~StreamReadyGroup();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_STREAM_API std::uint64_t tag() const noexcept;
  HYLE_COMPUTE_STREAM_API std::uint32_t range_count() const noexcept;
  HYLE_COMPUTE_STREAM_API Result<ReadLease> output(std::uint32_t index) const noexcept;
private:
  friend class StreamTicket;
  StreamReadyGroup(detail::FileStreamState *, std::uint32_t request, std::uint32_t group) noexcept;
  void reset() noexcept;
  detail::FileStreamState *state_ = nullptr;
  std::uint32_t request_ = 0, group_ = 0;
};

// Thread-compatible observation; cancellation and copied ReadyGroup/ReadLease
// drops may cross threads. Dropping a pending ticket requests cancellation,
// retains accepted work through real retirement and performs no hidden wait.
class StreamTicket final {
public:
  StreamTicket() noexcept = default;
  StreamTicket(const StreamTicket &) = delete;
  StreamTicket &operator=(const StreamTicket &) = delete;
  HYLE_COMPUTE_STREAM_API StreamTicket(StreamTicket &&) noexcept;
  HYLE_COMPUTE_STREAM_API StreamTicket &operator=(StreamTicket &&) noexcept;
  HYLE_COMPUTE_STREAM_API ~StreamTicket();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_STREAM_API Result<StreamReadyGroup> next_ready() noexcept;
  HYLE_COMPUTE_STREAM_API Result<StreamReadyGroup> wait_next_ready() noexcept;
  HYLE_COMPUTE_STREAM_API StreamResult result() const noexcept;
  HYLE_COMPUTE_STREAM_API StreamResult wait() noexcept;
  HYLE_COMPUTE_STREAM_API void request_cancel() noexcept;
private:
  friend class FileResourcePipeline;
  StreamTicket(detail::FileStreamState *, std::uint32_t request) noexcept;
  void reset() noexcept;
  detail::FileStreamState *state_ = nullptr;
  std::uint32_t request_ = 0;
};

// Created by the selected route's create_file_pipeline factory. No runtime is
// started implicitly: it borrows the supplied IoContext's Scheduler/pool and
// the original route/Queue resources. Admission and control are thread-compatible.
class FileResourcePipeline final {
public:
  FileResourcePipeline() noexcept = default;
  FileResourcePipeline(const FileResourcePipeline &) = delete;
  FileResourcePipeline &operator=(const FileResourcePipeline &) = delete;
  HYLE_COMPUTE_STREAM_API FileResourcePipeline(FileResourcePipeline &&) noexcept;
  HYLE_COMPUTE_STREAM_API FileResourcePipeline &operator=(FileResourcePipeline &&) noexcept;
  HYLE_COMPUTE_STREAM_API ~FileResourcePipeline();
  bool valid() const noexcept { return state_ != nullptr; }
  // Success alone consumes destination/producer owners; all preflight errors
  // leave the input spans intact. Descriptor spans need not outlive admission.
  HYLE_COMPUTE_STREAM_API Result<StreamTicket> read(StreamSource source,
      std::span<StreamReadGroup> groups, io::OperationOptions options = {}) noexcept;
  HYLE_COMPUTE_STREAM_API Result<StreamTicket> write(StreamFile sink,
      std::span<StreamWriteGroup> groups, io::OperationOptions options = {}) noexcept;
  HYLE_COMPUTE_STREAM_API void stop_admission() noexcept;
  HYLE_COMPUTE_STREAM_API Status collect() noexcept;
  HYLE_COMPUTE_STREAM_API Status close() noexcept;
  HYLE_COMPUTE_STREAM_API Status drain() noexcept;
  HYLE_COMPUTE_STREAM_API FileStreamReport report() const noexcept;
private:
  friend struct detail::FileStreamAccess;
  explicit FileResourcePipeline(detail::FileStreamState *state) noexcept : state_(state) {}
  void reset() noexcept;
  detail::FileStreamState *state_ = nullptr;
};

} // namespace hyle::compute
