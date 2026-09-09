#pragma once

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/error/result.h"
#include "hyle/io/io_context.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_WATCH_API
#elif defined(HYLE_IO_WATCH_BUILD)
#define HYLE_IO_WATCH_API HYLE_EXPORT
#else
#define HYLE_IO_WATCH_API HYLE_IMPORT
#endif

namespace hyle::io {

namespace detail {
struct FileWatcherState;
}

enum class FileWatchTargetKind : std::uint8_t {
  kFile,
  kDirectory,
};

// A borrowed factory input. FileWatcher::create copies the path, so the input
// storage may be released as soon as create returns. File targets may be absent
// while their parent exists; directory targets must exist at creation. Child
// name matching uses the UTF-8 spelling reported by the native directory
// stream; case-folded or Unicode-normalized aliases are not portable.
class FileWatchTarget {
 public:
  static constexpr FileWatchTarget file(std::string_view path) noexcept {
    return FileWatchTarget(FileWatchTargetKind::kFile, path);
  }
  static constexpr FileWatchTarget directory(std::string_view path) noexcept {
    return FileWatchTarget(FileWatchTargetKind::kDirectory, path);
  }

  constexpr FileWatchTargetKind kind() const noexcept { return kind_; }
  constexpr std::string_view path() const noexcept { return path_; }

 private:
  constexpr FileWatchTarget(FileWatchTargetKind kind,
                            std::string_view path) noexcept
      : kind_(kind), path_(path) {}

  FileWatchTargetKind kind_ = FileWatchTargetKind::kFile;
  std::string_view path_{};
};

// Portable event vocabulary. Native mechanisms may report several facts for
// one path at once; consumers should test bits rather than compare equality.
enum class FileWatchChange : std::uint32_t {
  kNone = 0,
  kContent = 1u << 0,
  kMetadata = 1u << 1,
  kCreated = 1u << 2,
  kRemoved = 1u << 3,
  kMovedFrom = 1u << 4,
  kMovedTo = 1u << 5,
  kRootInvalidated = 1u << 6,
};

constexpr FileWatchChange operator|(FileWatchChange left,
                                    FileWatchChange right) noexcept {
  return static_cast<FileWatchChange>(static_cast<std::uint32_t>(left) |
                                      static_cast<std::uint32_t>(right));
}
constexpr FileWatchChange operator&(FileWatchChange left,
                                    FileWatchChange right) noexcept {
  return static_cast<FileWatchChange>(static_cast<std::uint32_t>(left) &
                                      static_cast<std::uint32_t>(right));
}
constexpr FileWatchChange& operator|=(FileWatchChange& left,
                                     FileWatchChange right) noexcept {
  left = left | right;
  return left;
}
constexpr bool contains(FileWatchChange set, FileWatchChange value) noexcept {
  return (static_cast<std::uint32_t>(set) &
          static_cast<std::uint32_t>(value)) != 0;
}

struct FileWatchEvent {
  // Stable index in the target span passed to create().
  std::size_t target_index = 0;
  FileWatchChange changes = FileWatchChange::kNone;
  // Empty for an exact-file target. A directory target receives one direct
  // child basename; recursive descendants are never exposed.
  std::string_view relative_path{};
  // Best-effort native rename correlation. Zero means unavailable. Consumers
  // must remain correct by rescanning/re-reading without it.
  std::uint64_t rename_correlation = 0;
};

class FileWatchBatchView {
 public:
  constexpr FileWatchBatchView() noexcept = default;

  constexpr std::span<const FileWatchEvent> events() const noexcept {
    return events_;
  }
  constexpr std::uint64_t generation() const noexcept { return generation_; }
  constexpr bool rescan_required() const noexcept { return rescan_required_; }
  constexpr bool empty() const noexcept { return events_.empty(); }

 private:
  friend struct detail::FileWatcherState;
  constexpr FileWatchBatchView(std::span<const FileWatchEvent> events,
                               std::uint64_t generation,
                               bool rescan_required) noexcept
      : events_(events), generation_(generation),
        rescan_required_(rescan_required) {}

  std::span<const FileWatchEvent> events_{};
  std::uint64_t generation_ = 0;
  bool rescan_required_ = false;
};

struct FileWatchOptions {
  // Per native read buffer on Windows and the single inotify read buffer on
  // Linux. macOS FSEvents owns its kernel buffer but still honors the public
  // event/path bounds below.
  std::size_t native_buffer_bytes = 16 * 1024;
  std::size_t max_batch_events = 256;
  std::size_t max_batch_path_bytes = 16 * 1024;
};

// Bounded, non-recursive, cross-platform file-change stream. The watcher is a
// single-owner IO object: create(), next(), close(), move assignment over a
// live owner, and destruction all run on owner_worker. close() is idempotent
// on that owner and cancels/joins all native pumps. It
// must finish before the IoContext or Scheduler is stopped.
class FileWatcher {
 public:
  HYLE_IO_WATCH_API static Result<FileWatcher> create(
      mem::AllocatorRef allocator, IoContext& io, unsigned owner_worker,
      std::span<const FileWatchTarget> targets,
      FileWatchOptions options = {}) noexcept;

  FileWatcher() noexcept = default;
  HYLE_IO_WATCH_API ~FileWatcher();
  HYLE_IO_WATCH_API FileWatcher(FileWatcher&& other) noexcept;
  HYLE_IO_WATCH_API FileWatcher& operator=(FileWatcher&& other) noexcept;
  FileWatcher(const FileWatcher&) = delete;
  FileWatcher& operator=(const FileWatcher&) = delete;

  HYLE_IO_WATCH_API bool valid() const noexcept;
  HYLE_IO_WATCH_API unsigned owner_worker() const noexcept;

  struct NextAwaitable {
    FileWatcher* self = nullptr;
    OperationOptions options{};
    FileWatchBatchView batch{};
    Status status{};
    bool has_batch = false;

    bool await_ready() const noexcept { return false; }
    HYLE_IO_WATCH_API bool
    await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_WATCH_API Result<FileWatchBatchView> await_resume() noexcept;
  };

  // At most one next() may be outstanding. The returned view remains valid
  // until the next next(), close(), or destruction. Cancelling one wait does
  // not close the watcher or discard a concurrently sealed batch.
  NextAwaitable next(OperationOptions options = {}) noexcept {
    return NextAwaitable{this, options};
  }

  HYLE_IO_WATCH_API Status close() noexcept;

 private:
  explicit FileWatcher(detail::FileWatcherState* state,
                       mem::AllocatorRef allocator) noexcept
      : state_(state), allocator_(allocator) {}

  detail::FileWatcherState* state_ = nullptr;
  mem::AllocatorRef allocator_{};
};

}  // namespace hyle::io
