#pragma once

#include "hyle/io/watch.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "hyle/container/flat_hash_map.h"
#include "hyle/container/pod_vector.h"
#include "hyle/container/small_vector.h"
#include "hyle/core/environment.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/join.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

namespace hyle::io::detail {

inline constexpr std::size_t kNoWatchIndex =
    (std::numeric_limits<std::size_t>::max)();

// Keep the shipping pump on the small IO tier. Sanitizer instrumentation can
// expand simdutf's native-name validation frame beyond 16 KiB; use the already
// defined default tier only in instrumented builds so the safety verifier does
// not overflow the guarded stack it is meant to inspect.
inline constexpr std::size_t kWatchPumpStackSize =
#if HYLE_SANITIZER_ANY
    mem::kFiberStackDefault;
#else
    mem::kFiberStackSmall;
#endif

inline std::uint64_t watch_hash_bytes(std::string_view value) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

struct WatchStringHash {
  std::uint64_t operator()(std::string_view value) const noexcept {
    return watch_hash_bytes(value);
  }
};

struct ExactTargetKey {
  std::size_t root = 0;
  std::string_view basename{};
};

struct ExactTargetHash {
  std::uint64_t operator()(const ExactTargetKey& key) const noexcept {
    std::uint64_t hash = watch_hash_bytes(key.basename);
    hash ^= static_cast<std::uint64_t>(key.root) + 0x9e3779b97f4a7c15ull +
            (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct ExactTargetEqual {
  bool operator()(const ExactTargetKey& left,
                  const ExactTargetKey& right) const noexcept {
    return left.root == right.root && left.basename == right.basename;
  }
};

struct EventKey {
  std::size_t target = 0;
  std::string_view relative_path{};
};

struct EventKeyHash {
  std::uint64_t operator()(const EventKey& key) const noexcept {
    std::uint64_t hash = watch_hash_bytes(key.relative_path);
    hash ^= static_cast<std::uint64_t>(key.target) + 0x9e3779b97f4a7c15ull +
            (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct EventKeyEqual {
  bool operator()(const EventKey& left, const EventKey& right) const noexcept {
    return left.target == right.target &&
           left.relative_path == right.relative_path;
  }
};

struct StoredWatchTarget {
  FileWatchTargetKind kind = FileWatchTargetKind::kFile;
  std::size_t root = 0;
  std::size_t basename_offset = 0;
  std::size_t basename_size = 0;
  std::size_t next_same_exact = kNoWatchIndex;
  std::size_t next_directory = kNoWatchIndex;
  std::size_t next_tree = kNoWatchIndex;
};

struct NativeWatchRoot {
  explicit NativeWatchRoot(mem::AllocatorRef allocator) noexcept
      : native_buffer(allocator), native_path_buffer(allocator) {}
  NativeWatchRoot(NativeWatchRoot&&) noexcept = default;
  NativeWatchRoot& operator=(NativeWatchRoot&&) noexcept = default;
  NativeWatchRoot(const NativeWatchRoot&) = delete;
  NativeWatchRoot& operator=(const NativeWatchRoot&) = delete;

  std::size_t path_offset = 0;
  std::size_t path_size = 0;
  std::size_t first_directory = kNoWatchIndex;
  std::size_t first_tree = kNoWatchIndex;
  std::intptr_t native_handle = -1;
  int native_watch = -1;
  // DWORD alignment is a documented ReadDirectoryChangesW requirement. The
  // same storage type is harmless on POSIX and avoids relying on an allocator
  // returning stronger alignment than requested for std::byte.
  container::PodVector<std::uint32_t> native_buffer;
  // Windows converts one FILE_NOTIFY_INFORMATION name here. It is allocated
  // once at create; ordinary events never allocate.
  container::PodVector<char> native_path_buffer;
  sched::JoinHandle<void> pump;
  std::uint64_t pending_rename = 0;
  bool active = false;
};

#if HYLE_OS_LINUX
struct NativeWatchDirectory {
  explicit NativeWatchDirectory(mem::AllocatorRef allocator) noexcept
      : path(allocator) {}
  NativeWatchDirectory(NativeWatchDirectory&&) noexcept = default;
  NativeWatchDirectory& operator=(NativeWatchDirectory&&) noexcept = default;
  NativeWatchDirectory(const NativeWatchDirectory&) = delete;
  NativeWatchDirectory& operator=(const NativeWatchDirectory&) = delete;

  container::PodVector<char> path;
  int native_watch = -1;
  bool recursive = false;
  bool scanned = false;
};
#endif

struct WatchBatchStorage {
  explicit WatchBatchStorage(mem::AllocatorRef allocator) noexcept
      : events(allocator), paths(allocator), index(allocator) {}

  bool reserve(std::size_t event_capacity,
               std::size_t path_capacity) noexcept;
  void reset() noexcept;
  bool ready() const noexcept { return rescan || !events.empty(); }

  container::PodVector<FileWatchEvent> events;
  container::PodVector<char> paths;
  container::FlatHashMap<EventKey, std::size_t, EventKeyHash, EventKeyEqual,
                         mem::AllocatorRef>
      index;
  bool rescan = false;
};

struct FileWatcherState {
  FileWatcherState(mem::AllocatorRef allocator, IoContext& context,
                   unsigned owner_worker, FileWatchOptions configured) noexcept;
  ~FileWatcherState();
  FileWatcherState(const FileWatcherState&) = delete;
  FileWatcherState& operator=(const FileWatcherState&) = delete;

  Status initialize(std::span<const FileWatchTarget> inputs) noexcept;
  Result<FileWatchBatchView> next(OperationOptions options) noexcept;
  Status close() noexcept;

  std::string_view path_slice(std::size_t offset,
                              std::size_t size) const noexcept;
  std::string_view root_path(std::size_t root) const noexcept;
  std::string_view target_basename(std::size_t target) const noexcept;
#if HYLE_OS_LINUX
  std::string_view native_directory_path(std::size_t directory) const noexcept;
#endif

  // Native producers hold monitor while calling these functions and notify
  // once after parsing a complete native buffer.
  bool emit_for_root_locked(std::size_t root, std::string_view basename,
                            FileWatchChange changes,
                            std::uint64_t rename_correlation) noexcept;
  bool emit_absolute_path_locked(std::string_view path,
                                 FileWatchChange changes,
                                 std::uint64_t rename_correlation) noexcept;
  bool emit_target_locked(std::size_t target, std::string_view relative_path,
                          FileWatchChange changes,
                          std::uint64_t rename_correlation) noexcept;
  bool invalidate_root_locked(std::size_t root) noexcept;
  bool require_rescan_locked() noexcept;
  void publish_locked(bool changed) noexcept;
  void fail(Status error) noexcept;
  void report_armed() noexcept;
  void report_start_failure(Status error) noexcept;

  mem::AllocatorRef allocator;
  IoContext* io = nullptr;
  sched::Scheduler* scheduler = nullptr;
  unsigned owner = 0;
  FileWatchOptions options{};

  container::PodVector<char> path_storage;
  container::SmallVector<StoredWatchTarget, 8, mem::AllocatorRef> targets;
  container::SmallVector<NativeWatchRoot, 4, mem::AllocatorRef> roots;
  container::FlatHashMap<std::string_view, std::size_t, WatchStringHash,
                         std::equal_to<>, mem::AllocatorRef>
      root_by_path;
  container::FlatHashMap<ExactTargetKey, std::size_t, ExactTargetHash,
                         ExactTargetEqual, mem::AllocatorRef>
      exact_targets;
  container::FlatHashMap<int, std::size_t, container::Hash<int>,
                         std::equal_to<>, mem::AllocatorRef>
      native_root_by_watch;
#if HYLE_OS_LINUX
  container::SmallVector<NativeWatchDirectory, 0, mem::AllocatorRef>
      native_directories;
  container::FlatHashMap<std::string_view, std::size_t, WatchStringHash,
                         std::equal_to<>, mem::AllocatorRef>
      native_directory_by_path;
#endif
  WatchBatchStorage banks[2];
  unsigned pending_bank = 0;
  int sealed_bank = -1;

  util::SpinLock monitor;
  sched::ConditionVariable changed;
  sched::CancelToken native_cancel;
  Status failure{};
  std::uint64_t generation = 0;
  std::uint64_t next_rename_correlation = 1;
  std::size_t expected_arms = 0;
  std::size_t armed = 0;
  bool waiting = false;
  bool closing = false;
  bool closed = false;
  bool has_tree_targets = false;

  // Linux owns one inotify descriptor/read pump. macOS stores an opaque
  // FSEventStreamRef and serial dispatch queue here. Windows uses each root's
  // handle/buffer/pump fields.
  std::intptr_t native_stream = -1;
  container::PodVector<std::uint32_t> shared_native_buffer;
#if HYLE_OS_LINUX
  container::PodVector<char> shared_path_buffer;
#endif
  sched::JoinHandle<void> shared_pump;
  void* apple_stream = nullptr;
  void* apple_queue = nullptr;
};

Status initialize_native_watch(FileWatcherState& state) noexcept;
Status shutdown_native_watch(FileWatcherState& state) noexcept;

}  // namespace hyle::io::detail
