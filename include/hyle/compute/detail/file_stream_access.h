#pragma once

#include "hyle/compute/file_stream.h"
#include "hyle/compute/detail/copy_ranges.h"

namespace hyle::compute::detail {

struct StreamCopyResult final {
  Status status{};
  CopyBatchReport report{};
  bool retirement_proven = true;
};

// One original prepared native copy slot per file staging slot. This private
// table only hands off a complete byte-copy episode; it cannot execute kernels,
// schedule arbitrary work, expose a Queue or create another runtime.
struct StreamCopyEndpoint final {
  void *owner = nullptr;
  void (*destroy)(void *) noexcept = nullptr;
  Result<PhysicalCopyResource> (*physical)(void *, mem::ConstBufferView, bool) noexcept = nullptr;
  StreamCopyResult (*copy)(void *, std::uint32_t, ReadLease &, WriteLease *,
      WriteLease &, Completion *, std::size_t, std::size_t, std::size_t, std::size_t, bool) noexcept = nullptr;
  Status (*observe)(void *, std::uint32_t) noexcept = nullptr;
  bool (*in_flight)(void *, std::uint32_t) noexcept = nullptr;
  Status (*reset)(void *, std::uint32_t) noexcept = nullptr;
  CopyBatchReport (*report)(void *) noexcept = nullptr;
  StreamCopyEndpoint() noexcept = default;
  StreamCopyEndpoint(const StreamCopyEndpoint &) = delete;
  StreamCopyEndpoint &operator=(const StreamCopyEndpoint &) = delete;
  HYLE_COMPUTE_STREAM_API StreamCopyEndpoint(StreamCopyEndpoint &&other) noexcept;
  HYLE_COMPUTE_STREAM_API StreamCopyEndpoint &operator=(StreamCopyEndpoint &&other) noexcept;
  HYLE_COMPUTE_STREAM_API ~StreamCopyEndpoint();
};

struct FileStreamAccess final {
  static HYLE_COMPUTE_STREAM_API Result<FileResourcePipeline> create(
      io::IoContext &, mem::MemoryResourceRef staging,
      std::span<const CopyResourceSpec> targets, FileStreamOptions,
      StreamCopyEndpoint &&) noexcept;
  static io::File *file(const StreamFile &) noexcept;
  static io::File *file(const StreamSource &) noexcept;
  static const mem::FileMapping *mapping(const StreamSource &) noexcept;
};

} // namespace hyle::compute::detail
