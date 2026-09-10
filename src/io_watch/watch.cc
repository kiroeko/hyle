#include "hyle/io/watch.h"

#include "watch_internal.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#include "hyle/container/path.h"
#include "hyle/container/string_algo.h"
#include "hyle/core/assert.h"
#include "hyle/core/environment.h"
#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/scheduler.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace hyle::io {
namespace detail {
namespace {

using container::StringView;

#if !HYLE_OS_WINDOWS
Status canonical_watch_root(std::string_view input,
                            mem::AllocatorRef allocator,
                            container::PodVector<char>* output) noexcept {
  if (input.size() >= PATH_MAX)
    return Status::OutOfRange("file watch root exceeds native path limits");
  container::PodVector<char> terminated(allocator);
  if (!terminated.try_resize(input.size() + 1))
    return Status::ResourceExhausted(
        "file watch root canonicalization allocation failed");
  if (!input.empty())
    std::memcpy(terminated.data(), input.data(), input.size());
  terminated[input.size()] = '\0';

  char resolved[PATH_MAX];
  errno = 0;
  if (::realpath(terminated.data(), resolved) == nullptr)
    return status_from_errno(errno, "canonicalize FileWatcher root");
  const std::size_t length = std::strlen(resolved);
  if (!output->try_resize(length + 1))
    return Status::ResourceExhausted(
        "file watch canonical root allocation failed");
  std::memcpy(output->data(), resolved, length + 1);
  return Status::Ok();
}
#endif

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t* output) noexcept {
  if (right > (std::numeric_limits<std::size_t>::max)() - left)
    return false;
  *output = left + right;
  return true;
}

Status append_path(container::PodVector<char>& storage,
                   std::string_view path, std::size_t* offset) noexcept {
  std::size_t with_nul = 0;
  std::size_t required = 0;
  if (!checked_add(path.size(), 1, &with_nul) ||
      !checked_add(storage.size(), with_nul, &required))
    return Status::OutOfRange("file watch path storage overflows");
  const std::size_t start = storage.size();
  if (!storage.try_resize(required))
    return Status::ResourceExhausted(
        "file watch path storage allocation failed");
  if (!path.empty())
    std::memcpy(storage.data() + start, path.data(), path.size());
  storage[start + path.size()] = '\0';
  *offset = start;
  return Status::Ok();
}

#if HYLE_OS_WINDOWS

Status absolute_watch_path(std::string_view input, mem::AllocatorRef allocator,
                           container::PodVector<char>* output) noexcept {
  if (input.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    return Status::OutOfRange("file watch path exceeds Win32 conversion limits");

  const int wide_count = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, input.data(), static_cast<int>(input.size()),
      nullptr, 0);
  if (wide_count <= 0)
    return status_from_win32(::GetLastError(),
                             "decode file watch UTF-8 path");
  container::PodVector<wchar_t> wide(allocator);
  if (!wide.try_resize(static_cast<std::size_t>(wide_count) + 1))
    return Status::ResourceExhausted(
        "file watch UTF-16 path allocation failed");
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
                            static_cast<int>(input.size()), wide.data(),
                            wide_count) != wide_count)
    return status_from_win32(::GetLastError(),
                             "decode file watch UTF-8 path");
  wide[static_cast<std::size_t>(wide_count)] = L'\0';

  DWORD required = ::GetFullPathNameW(wide.data(), 0, nullptr, nullptr);
  if (required == 0)
    return status_from_win32(::GetLastError(),
                             "resolve file watch absolute path");
  container::PodVector<wchar_t> absolute(allocator);
  if (!absolute.try_resize(static_cast<std::size_t>(required) + 1))
    return Status::ResourceExhausted(
        "file watch absolute path allocation failed");
  DWORD written = ::GetFullPathNameW(wide.data(),
                                     static_cast<DWORD>(absolute.size()),
                                     absolute.data(), nullptr);
  if (written == 0 || written >= absolute.size())
    return written == 0
               ? status_from_win32(::GetLastError(),
                                   "resolve file watch absolute path")
               : Status::OutOfRange("file watch absolute path changed size");

  const int utf8_count = ::WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, absolute.data(), static_cast<int>(written),
      nullptr, 0, nullptr, nullptr);
  if (utf8_count <= 0)
    return status_from_win32(::GetLastError(),
                             "encode file watch absolute path");
  if (!output->try_resize(static_cast<std::size_t>(utf8_count) + 1))
    return Status::ResourceExhausted(
        "file watch absolute UTF-8 path allocation failed");
  if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, absolute.data(),
                            static_cast<int>(written), output->data(),
                            utf8_count, nullptr, nullptr) != utf8_count)
    return status_from_win32(::GetLastError(),
                             "encode file watch absolute path");
  std::size_t size = static_cast<std::size_t>(utf8_count);
  for (std::size_t i = 0; i < size; ++i) {
    if ((*output)[i] == '\\') (*output)[i] = '/';
  }
  // Preserve drive roots (C:/); every other trailing separator is redundant
  // for root grouping and makes parent/basename matching ambiguous.
  if (size > 3 && (*output)[size - 1] == '/') --size;
  (*output)[size] = '\0';
  output->resize(size + 1);
  return Status::Ok();
}

#else

Status absolute_watch_path(std::string_view input, mem::AllocatorRef allocator,
                           container::PodVector<char>* output) noexcept {
  container::PodVector<char> joined(allocator);
  std::size_t joined_size = input.size();
  if (!container::is_absolute(StringView(input))) {
    Result<std::size_t> cwd_size = platform::current_directory(nullptr, 0);
    if (!cwd_size.has_value()) return std::move(cwd_size).error();
    container::PodVector<char> cwd(allocator);
    if (!cwd.try_resize(*cwd_size + 1))
      return Status::ResourceExhausted(
          "file watch current-directory allocation failed");
    Result<std::size_t> copied =
        platform::current_directory(cwd.data(), cwd.size());
    if (!copied.has_value()) return std::move(copied).error();
    if (*copied >= cwd.size())
      return Status::Unavailable(
          "file watch current directory changed while resolving path");
    const StringView parts[]{StringView(cwd.data()), StringView(input)};
    joined_size = container::join_path(parts, std::span<char>{});
    if (joined_size == (std::numeric_limits<std::size_t>::max)() ||
        !joined.try_resize(joined_size + 1))
      return Status::ResourceExhausted(
          "file watch absolute path allocation failed");
    (void)container::join_path(parts, joined);
  } else {
    if (!joined.try_resize(input.size() + 1))
      return Status::ResourceExhausted(
          "file watch absolute path allocation failed");
    if (!input.empty())
      std::memcpy(joined.data(), input.data(), input.size());
    joined[input.size()] = '\0';
  }

  if (joined_size > StringView::npos)
    return Status::OutOfRange(
        "file watch absolute path exceeds the supported UTF-8 range");

  if (!output->try_resize((std::max)(joined_size, std::size_t{1}) + 1))
    return Status::ResourceExhausted(
        "file watch normalized path allocation failed");
  auto normalized = container::normalize_path(
      StringView(joined.data(), static_cast<std::uint32_t>(joined_size), true),
      *output);
  if (!normalized.has_value())
    return Status::Internal("file watch path normalization failed");
  output->resize(normalized->size() + 1);
  return Status::Ok();
}

#endif

Status validate_options(const FileWatchOptions& options) noexcept {
  if (options.native_buffer_bytes < 1024 ||
      options.native_buffer_bytes > 64 * 1024)
    return Status::OutOfRange(
        "native_buffer_bytes must be between 1024 and 65536");
  if (options.max_batch_events == 0)
    return Status::InvalidArgument("max_batch_events must be nonzero");
  if (options.max_batch_path_bytes == 0)
    return Status::InvalidArgument("max_batch_path_bytes must be nonzero");
  if (options.max_watched_directories == 0)
    return Status::InvalidArgument("max_watched_directories must be nonzero");
  return Status::Ok();
}

void wake_cancelled_next(void* opaque) noexcept {
  auto* state = static_cast<FileWatcherState*>(opaque);
  std::lock_guard<util::SpinLock> lock(state->monitor);
  state->changed.notify_all();
}

}  // namespace

bool WatchBatchStorage::reserve(std::size_t event_capacity,
                                std::size_t path_capacity) noexcept {
  return events.try_reserve(event_capacity) && paths.try_reserve(path_capacity) &&
         index.try_reserve(event_capacity);
}

void WatchBatchStorage::reset() noexcept {
  index.clear();
  events.clear();
  paths.clear();
  rescan = false;
}

FileWatcherState::FileWatcherState(mem::AllocatorRef selected_allocator,
                                   IoContext& context,
                                   unsigned owner_worker,
                                   FileWatchOptions configured) noexcept
    : allocator(selected_allocator), io(&context),
      scheduler(&IoContextAccess::scheduler(context)), owner(owner_worker),
      options(configured), path_storage(selected_allocator),
      targets(selected_allocator), roots(selected_allocator),
      root_by_path(selected_allocator), exact_targets(selected_allocator),
      native_root_by_watch(selected_allocator),
#if HYLE_OS_LINUX
      native_directories(selected_allocator),
      native_directory_by_path(selected_allocator),
#endif
      banks{WatchBatchStorage(selected_allocator),
            WatchBatchStorage(selected_allocator)},
      shared_native_buffer(selected_allocator)
#if HYLE_OS_LINUX
      , shared_path_buffer(selected_allocator)
#endif
      {}

FileWatcherState::~FileWatcherState() {
  HYLE_FATAL_ASSERT(closed);
  HYLE_FATAL_ASSERT(native_stream == -1 && apple_stream == nullptr &&
                    apple_queue == nullptr);
}

std::string_view FileWatcherState::path_slice(std::size_t offset,
                                              std::size_t size) const noexcept {
  HYLE_FATAL_ASSERT(offset <= path_storage.size() &&
                    size <= path_storage.size() - offset);
  return std::string_view(path_storage.data() + offset, size);
}

std::string_view FileWatcherState::root_path(std::size_t root) const noexcept {
  HYLE_FATAL_ASSERT(root < roots.size());
  return path_slice(roots[root].path_offset, roots[root].path_size);
}

std::string_view FileWatcherState::target_basename(
    std::size_t target) const noexcept {
  HYLE_FATAL_ASSERT(target < targets.size());
  return path_slice(targets[target].basename_offset,
                    targets[target].basename_size);
}

#if HYLE_OS_LINUX
std::string_view FileWatcherState::native_directory_path(
    std::size_t directory) const noexcept {
  HYLE_FATAL_ASSERT(directory < native_directories.size());
  const auto& path = native_directories[directory].path;
  HYLE_FATAL_ASSERT(!path.empty() && path.back() == '\0');
  return std::string_view(path.data(), path.size() - 1);
}

#endif

Status FileWatcherState::initialize(
    std::span<const FileWatchTarget> inputs) noexcept {
  Status options_status = validate_options(options);
  if (!options_status.ok()) return options_status;
  if (!allocator.valid())
    return Status::InvalidArgument(
        "FileWatcher requires a complete allocator");
  if (io == nullptr || !io->valid())
    return Status::FailedPrecondition(
        "FileWatcher requires a valid IoContext");
  if (owner >= io->num_workers())
    return Status::OutOfRange("FileWatcher owner worker is out of range");
  if (sched::current_worker_index() != static_cast<int>(owner))
    return Status::FailedPrecondition(
        "FileWatcher must be created on its owner worker");
  if (inputs.empty())
    return Status::InvalidArgument("FileWatcher requires at least one target");
  if (!targets.try_reserve(inputs.size()) || !roots.try_reserve(inputs.size()))
    return Status::ResourceExhausted(
        "FileWatcher target table allocation failed");
  // path_storage may relocate while target paths are normalized, so a map
  // keyed by string_view cannot safely be built until the final storage is
  // stable. Group roots through stable hashes plus an explicit collision chain
  // instead: expected O(targets), without over-reserving PATH_MAX per input.
  container::FlatHashMap<std::uint64_t, std::size_t,
                         container::Hash<std::uint64_t>, std::equal_to<>,
                         mem::AllocatorRef>
      root_hash_heads(allocator);
  container::SmallVector<std::size_t, 4, mem::AllocatorRef>
      root_hash_next(allocator);
  if (!root_hash_heads.try_reserve(inputs.size()) ||
      !root_hash_next.try_reserve(inputs.size()))
    return Status::ResourceExhausted(
        "FileWatcher root grouping allocation failed");
  if (!banks[0].reserve(options.max_batch_events,
                        options.max_batch_path_bytes) ||
      !banks[1].reserve(options.max_batch_events,
                        options.max_batch_path_bytes))
    return Status::ResourceExhausted(
        "FileWatcher batch storage allocation failed");

  for (const FileWatchTarget& input : inputs) {
    if (input.kind() != FileWatchTargetKind::kFile &&
        input.kind() != FileWatchTargetKind::kDirectory &&
        input.kind() != FileWatchTargetKind::kDirectoryTree)
      return Status::InvalidArgument("FileWatcher target kind is invalid");
    const std::string_view raw = input.path();
    if (raw.empty() || raw.find('\0') != std::string_view::npos)
      return Status::InvalidArgument(
          "FileWatcher target path must be non-empty UTF-8 without NUL");
    if (raw.size() > StringView::npos ||
        !container::utf8_validate(StringView(raw)))
      return Status::InvalidArgument(
          "FileWatcher target path is not valid representable UTF-8");

    container::PodVector<char> absolute(allocator);
    Status resolved = absolute_watch_path(raw, allocator, &absolute);
    if (!resolved.ok()) return resolved.WithContext("FileWatcher target");
    const std::string_view full(absolute.data(), absolute.size() - 1);

    if (input.kind() == FileWatchTargetKind::kDirectoryTree) {
      Result<platform::FileInfo> tree_info = platform::stat(absolute.data());
      if (!tree_info.has_value())
        return std::move(tree_info).error().WithContext(
            "FileWatcher recursive root");
      if (tree_info->is_symlink)
        return Status::FailedPrecondition(
            "FileWatcher recursive root must not be a symlink or junction");
    }

    std::string_view root_name = full;
    std::string_view basename{};
    if (input.kind() == FileWatchTargetKind::kFile) {
      const std::size_t separator = full.find_last_of('/');
      if (separator == std::string_view::npos || separator + 1 == full.size())
        return Status::InvalidArgument(
            "FileWatcher file target must name a basename");
      root_name = separator == 0 ? full.substr(0, 1)
                                 : full.substr(0, separator);
#if HYLE_OS_WINDOWS
      if (separator == 2 && full[1] == ':') root_name = full.substr(0, 3);
#endif
      basename = full.substr(separator + 1);
    }

#if !HYLE_OS_WINDOWS
    // Native change streams identify the existing directory object rather
    // than preserving a lexical alias. In particular, macOS reports /private/
    // var for the common /var symlink and inotify reuses a watch descriptor
    // for two aliases of the same directory. Canonicalize only the root (which
    // must exist); an exact file target itself is still allowed to be absent.
    container::PodVector<char> canonical_root(allocator);
    Status canonical =
        canonical_watch_root(root_name, allocator, &canonical_root);
    if (!canonical.ok())
      return canonical.WithContext("FileWatcher target root");
    root_name = std::string_view(canonical_root.data(),
                                 canonical_root.size() - 1);
#endif

    const std::uint64_t root_hash = watch_hash_bytes(root_name);
    std::size_t root_index = kNoWatchIndex;
    std::size_t* hash_head = root_hash_heads.find(root_hash);
    for (std::size_t candidate =
             hash_head == nullptr ? kNoWatchIndex : *hash_head;
         candidate != kNoWatchIndex; candidate = root_hash_next[candidate]) {
      if (root_path(candidate) == root_name) {
        root_index = candidate;
        break;
      }
    }
    if (root_index == kNoWatchIndex) {
      std::size_t path_offset = 0;
      Status copied = append_path(path_storage, root_name, &path_offset);
      if (!copied.ok()) return copied;
      if (!roots.try_emplace_back(allocator))
        return Status::ResourceExhausted(
            "FileWatcher root table allocation failed");
      root_index = roots.size() - 1;
      roots[root_index].path_offset = path_offset;
      roots[root_index].path_size = root_name.size();
      if (!root_hash_next.try_push_back(
              hash_head == nullptr ? kNoWatchIndex : *hash_head))
        return Status::ResourceExhausted(
            "FileWatcher root collision table allocation failed");
      if (hash_head != nullptr) {
        *hash_head = root_index;
      } else {
        auto [entry, result] =
            root_hash_heads.try_emplace(root_hash, root_index);
        if (result == decltype(root_hash_heads)::InsertResult::kOutOfMemory)
          return Status::ResourceExhausted(
              "FileWatcher root grouping allocation failed");
        (void)entry;
      }

      Result<platform::FileInfo> info =
          platform::stat(path_storage.data() + path_offset);
      if (!info.has_value())
        return std::move(info).error().WithContext("FileWatcher root");
      if (!info->is_directory())
        return Status::FailedPrecondition(
            "FileWatcher native root is not a directory");
    }

    std::size_t basename_offset = 0;
    Status copied = append_path(path_storage, basename, &basename_offset);
    if (!copied.ok()) return copied;
    if (!targets.try_emplace_back())
      return Status::ResourceExhausted(
          "FileWatcher target table allocation failed");
    StoredWatchTarget& target = targets.back();
    target.kind = input.kind();
    target.root = root_index;
    target.basename_offset = basename_offset;
    target.basename_size = basename.size();
  }

  if (roots.size() > options.max_watched_directories)
    return Status::ResourceExhausted(
        "FileWatcher native directory limit is smaller than its roots");

  if (!root_by_path.try_reserve(roots.size()) ||
      !exact_targets.try_reserve(targets.size()))
    return Status::ResourceExhausted(
        "FileWatcher lookup table allocation failed");
  for (std::size_t root = 0; root < roots.size(); ++root) {
    auto [entry, result] = root_by_path.try_emplace(root_path(root), root);
    if (result == decltype(root_by_path)::InsertResult::kOutOfMemory)
      return Status::ResourceExhausted(
          "FileWatcher root lookup allocation failed");
    (void)entry;
  }
  for (std::size_t index = 0; index < targets.size(); ++index) {
    StoredWatchTarget& target = targets[index];
    if (target.kind == FileWatchTargetKind::kDirectory) {
      target.next_directory = roots[target.root].first_directory;
      roots[target.root].first_directory = index;
      continue;
    }
    if (target.kind == FileWatchTargetKind::kDirectoryTree) {
      target.next_tree = roots[target.root].first_tree;
      roots[target.root].first_tree = index;
      has_tree_targets = true;
      continue;
    }
    const ExactTargetKey key{target.root, target_basename(index)};
    if (std::size_t* existing = exact_targets.find(key)) {
      target.next_same_exact = *existing;
      *existing = index;
    } else {
      auto [entry, result] = exact_targets.try_emplace(key, index);
      if (result == decltype(exact_targets)::InsertResult::kOutOfMemory)
        return Status::ResourceExhausted(
            "FileWatcher exact-target lookup allocation failed");
      (void)entry;
    }
  }

  Result<sched::CancelToken> cancellation =
      sched::CancelToken::try_create_root(allocator);
  if (!cancellation.has_value()) return std::move(cancellation).error();
  native_cancel = std::move(*cancellation);

  Status native = initialize_native_watch(*this);
  if (!native.ok()) return native;

  if (expected_arms != 0) {
    std::unique_lock<util::SpinLock> lock(monitor);
    changed.wait(lock, [&] {
      return armed == expected_arms || !failure.ok();
    });
    if (!failure.ok()) return failure;
  }
  return Status::Ok();
}

bool FileWatcherState::emit_for_root_locked(
    std::size_t root, std::string_view relative, FileWatchChange changes,
    std::uint64_t rename_correlation) noexcept {
  if (root >= roots.size() || changes == FileWatchChange::kNone) return false;
  // POSIX directory entries are arbitrary bytes and native macOS events can
  // also fail to round-trip through UTF-8. The public path contract is UTF-8;
  // an unrepresentable entry therefore becomes an explicit rescan request
  // instead of leaking malformed text or silently missing an exact target.
  if (relative.empty() || relative.size() > container::StringView::npos ||
      !container::utf8_validate(container::StringView(relative)))
    return require_rescan_locked();
  bool changed_batch = false;

  for (std::size_t target = roots[root].first_tree;
       target != kNoWatchIndex; target = targets[target].next_tree)
    changed_batch = emit_target_locked(target, relative, changes,
                                       rename_correlation) || changed_batch;

  if (relative.find('/') != std::string_view::npos) return changed_batch;

  for (std::size_t target = roots[root].first_directory;
       target != kNoWatchIndex; target = targets[target].next_directory)
    changed_batch = emit_target_locked(target, relative, changes,
                                       rename_correlation) || changed_batch;

  const ExactTargetKey exact{root, relative};
  if (const std::size_t* first = exact_targets.find(exact)) {
    for (std::size_t target = *first; target != kNoWatchIndex;
         target = targets[target].next_same_exact)
      changed_batch = emit_target_locked(target, {}, changes,
                                         rename_correlation) || changed_batch;
  }
  return changed_batch;
}

bool FileWatcherState::emit_absolute_path_locked(
    std::string_view path, FileWatchChange changes,
    std::uint64_t rename_correlation) noexcept {
  bool output = false;
  std::size_t immediate_root = kNoWatchIndex;
  const std::size_t separator = path.find_last_of('/');
  if (separator != std::string_view::npos && separator + 1 < path.size()) {
    const std::string_view parent =
        separator == 0 ? path.substr(0, 1) : path.substr(0, separator);
    if (const std::size_t* root = root_by_path.find(parent)) {
      if (roots[*root].active) {
        immediate_root = *root;
        output = emit_for_root_locked(*root, path.substr(separator + 1),
                                      changes, rename_correlation) || output;
      }
    }
  }
  if (!has_tree_targets) return output;
  for (std::size_t root = 0; root < roots.size(); ++root) {
    if (root == immediate_root || !roots[root].active ||
        roots[root].first_tree == kNoWatchIndex)
      continue;
    const std::string_view prefix = root_path(root);
    if (path.size() <= prefix.size() ||
        path.compare(0, prefix.size(), prefix) != 0)
      continue;
    std::size_t relative = prefix.size();
    if (prefix.empty() || prefix.back() != '/') {
      if (path[relative] != '/') continue;
      ++relative;
    }
    if (relative == path.size()) continue;
    output = emit_for_root_locked(root, path.substr(relative),
                                  changes, rename_correlation) || output;
  }
  return output;
}

bool FileWatcherState::emit_target_locked(
    std::size_t target_index, std::string_view relative,
    FileWatchChange changes, std::uint64_t rename_correlation) noexcept {
  if (target_index >= targets.size() || changes == FileWatchChange::kNone)
    return false;
  WatchBatchStorage& bank = banks[pending_bank];
  const EventKey query{target_index, relative};
  if (std::size_t* existing = bank.index.find(query)) {
    FileWatchEvent& event = bank.events[*existing];
    const FileWatchChange previous = event.changes;
    const std::uint64_t previous_correlation = event.rename_correlation;
    event.changes |= changes;
    if (rename_correlation != 0) {
      if (event.rename_correlation == 0)
        event.rename_correlation = rename_correlation;
      else if (event.rename_correlation != rename_correlation)
        event.rename_correlation = 0;
    }
    return event.changes != previous ||
           event.rename_correlation != previous_correlation;
  }
  if (bank.events.size() == options.max_batch_events ||
      relative.size() > options.max_batch_path_bytes - bank.paths.size())
    return require_rescan_locked();
  const std::size_t path_offset = bank.paths.size();
  if (!bank.paths.try_resize(path_offset + relative.size()))
    return require_rescan_locked();
  if (!relative.empty())
    std::memcpy(bank.paths.data() + path_offset, relative.data(),
                relative.size());
  const std::string_view stored =
      relative.empty()
          ? std::string_view{}
          : std::string_view(bank.paths.data() + path_offset, relative.size());
  if (!bank.events.try_push_back(
          FileWatchEvent{target_index, changes, stored,
                         rename_correlation}))
    return require_rescan_locked();
  const std::size_t event_index = bank.events.size() - 1;
  auto [entry, result] =
      bank.index.try_emplace(EventKey{target_index, stored}, event_index);
  if (result == decltype(bank.index)::InsertResult::kOutOfMemory)
    (void)require_rescan_locked();
  (void)entry;
  return true;
}

bool FileWatcherState::invalidate_root_locked(std::size_t root) noexcept {
  if (root >= roots.size()) return require_rescan_locked();
  bool output = require_rescan_locked();
  for (std::size_t target = 0; target < targets.size(); ++target) {
    if (targets[target].root != root) continue;
    output = emit_target_locked(target, {},
                                FileWatchChange::kRootInvalidated, 0) || output;
  }
  return output;
}

bool FileWatcherState::require_rescan_locked() noexcept {
  WatchBatchStorage& bank = banks[pending_bank];
  if (bank.rescan) return false;
  bank.rescan = true;
  return true;
}

void FileWatcherState::publish_locked(bool did_change) noexcept {
  if (!did_change) return;
  changed.notify_all();
}

void FileWatcherState::fail(Status error) noexcept {
  if (error.ok()) return;
  std::lock_guard<util::SpinLock> lock(monitor);
  if (failure.ok()) failure = std::move(error);
  changed.notify_all();
}

void FileWatcherState::report_armed() noexcept {
  std::lock_guard<util::SpinLock> lock(monitor);
  ++armed;
  changed.notify_all();
}

void FileWatcherState::report_start_failure(Status error) noexcept {
  std::lock_guard<util::SpinLock> lock(monitor);
  if (failure.ok()) failure = std::move(error);
  changed.notify_all();
}

Result<FileWatchBatchView> FileWatcherState::next(
    OperationOptions operation) noexcept {
  if (sched::current_worker_index() != static_cast<int>(owner))
    return make_unexpected(Status::FailedPrecondition(
        "FileWatcher::next must run on its owner worker"));

  auto run = [&]() noexcept -> Result<FileWatchBatchView> {
    std::unique_lock<util::SpinLock> lock(monitor);
    if (waiting)
      return make_unexpected(Status::FailedPrecondition(
          "FileWatcher already has an outstanding next"));
    waiting = true;
    if (sealed_bank >= 0) {
      banks[static_cast<unsigned>(sealed_bank)].reset();
      sealed_bank = -1;
    }

    auto finish_wait = [&]() noexcept {
      waiting = false;
      changed.notify_all();
    };
    auto ready = [&]() noexcept {
      return banks[pending_bank].ready() || !failure.ok() || closing || closed ||
             (operation.cancel != nullptr &&
              operation.cancel->is_cancelled());
    };
    while (!ready()) {
      if (operation.deadline.has_value()) {
        if (!changed.wait_until(lock, *operation.deadline, ready)) break;
      } else {
        changed.wait(lock, ready);
      }
    }

    // A sealed event batch wins a cancel/deadline race. The consumer can then
    // re-read authoritative state and decide whether to issue another next.
    if (banks[pending_bank].ready()) {
      sealed_bank = static_cast<int>(pending_bank);
      pending_bank = 1u - pending_bank;
      ++generation;
      WatchBatchStorage& sealed = banks[static_cast<unsigned>(sealed_bank)];
      FileWatchBatchView view(
          std::span<const FileWatchEvent>(sealed.events.data(),
                                          sealed.events.size()),
          generation, sealed.rescan);
      finish_wait();
      return view;
    }
    Status terminal;
    if (!failure.ok())
      terminal = failure;
    else if (closing || closed)
      terminal = Status::Cancelled("FileWatcher is closed");
    else if (operation.cancel != nullptr &&
             operation.cancel->is_cancelled())
      terminal = Status::Cancelled("FileWatcher wait cancelled");
    else
      terminal = Status::DeadlineExceeded("FileWatcher wait deadline expired");
    finish_wait();
    return make_unexpected(std::move(terminal));
  };

  if (operation.cancel != nullptr) {
    sched::CancelCallback callback(*operation.cancel, wake_cancelled_next, this);
    return run();
  }
  return run();
}

Status FileWatcherState::close() noexcept {
  if (sched::current_worker_index() != static_cast<int>(owner))
    return Status::FailedPrecondition(
        "FileWatcher::close must run on its owner worker");
  {
    std::unique_lock<util::SpinLock> lock(monitor);
    if (closed) return Status::Ok();
    if (closing) {
      changed.wait(lock, [&] { return closed; });
      return Status::Ok();
    }
    closing = true;
    changed.notify_all();
  }

  if (native_cancel.valid()) native_cancel.cancel();
  Status native_status = shutdown_native_watch(*this);

  {
    std::unique_lock<util::SpinLock> lock(monitor);
    changed.wait(lock, [&] { return !waiting; });
    banks[0].reset();
    banks[1].reset();
    sealed_bank = -1;
    pending_bank = 0;
    closed = true;
    closing = false;
    changed.notify_all();
  }
  return native_status;
}

}  // namespace detail

Result<FileWatcher> FileWatcher::create(
    mem::AllocatorRef allocator, IoContext& io, unsigned owner_worker,
    std::span<const FileWatchTarget> targets,
    FileWatchOptions options) noexcept {
  if (!allocator.valid())
    return make_unexpected(Status::InvalidArgument(
        "FileWatcher::create requires a complete allocator"));
  // Reject owner-contract violations before allocating FileWatcherState. A
  // partially constructed native owner can only be rolled back on its owner
  // worker; allocating first would turn this ordinary typed failure into an
  // assertion while cleanup tried to call close() from the wrong thread.
  if (!io.valid())
    return make_unexpected(Status::FailedPrecondition(
        "FileWatcher::create requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange(
        "FileWatcher owner worker is out of range"));
  if (sched::current_worker_index() != static_cast<int>(owner_worker))
    return make_unexpected(Status::FailedPrecondition(
        "FileWatcher must be created on its owner worker"));
  detail::FileWatcherState* state =
      allocator.construct<detail::FileWatcherState>(allocator, io,
                                                     owner_worker, options);
  if (state == nullptr)
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher state allocation failed"));
  Status initialized = state->initialize(targets);
  if (!initialized.ok()) {
    (void)state->close();
    allocator.destroy(state);
    return make_unexpected(std::move(initialized));
  }
  return FileWatcher(state, allocator);
}

FileWatcher::~FileWatcher() {
  if (state_ == nullptr) return;
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(state_->owner));
  const Status closed = state_->close();
  HYLE_FATAL_ASSERT(closed.ok());
  allocator_.destroy(state_);
}

FileWatcher::FileWatcher(FileWatcher&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      allocator_(other.allocator_) {}

FileWatcher& FileWatcher::operator=(FileWatcher&& other) noexcept {
  if (this == &other) return *this;
  if (state_ != nullptr) {
    HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                      static_cast<int>(state_->owner));
    const Status closed = state_->close();
    HYLE_FATAL_ASSERT(closed.ok());
    allocator_.destroy(state_);
  }
  state_ = std::exchange(other.state_, nullptr);
  allocator_ = other.allocator_;
  return *this;
}

bool FileWatcher::valid() const noexcept {
  if (state_ == nullptr) return false;
  std::lock_guard<util::SpinLock> lock(state_->monitor);
  return !state_->closed && !state_->closing;
}

unsigned FileWatcher::owner_worker() const noexcept {
  return state_ == nullptr ? 0 : state_->owner;
}

bool FileWatcher::NextAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  if (self == nullptr || self->state_ == nullptr) {
    status = Status::FailedPrecondition("FileWatcher is invalid");
    return false;
  }
  Result<FileWatchBatchView> result = self->state_->next(options);
  if (result.has_value()) {
    batch = *result;
    has_batch = true;
  } else {
    status = std::move(result).error();
  }
  return false;
}

Result<FileWatchBatchView> FileWatcher::NextAwaitable::await_resume() noexcept {
  if (!has_batch) return make_unexpected(std::move(status));
  return batch;
}

Status FileWatcher::close() noexcept {
  return state_ == nullptr ? Status::Ok() : state_->close();
}

}  // namespace hyle::io
