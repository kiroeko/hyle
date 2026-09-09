#include "watch_internal.h"

#include "hyle/core/environment.h"

#if HYLE_OS_APPLE

#include <CoreServices/CoreServices.h>
#include <dispatch/dispatch.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace hyle::io::detail {
namespace {

constexpr FSEventStreamCreateFlags kCreateFlags =
    kFSEventStreamCreateFlagNoDefer |
    kFSEventStreamCreateFlagWatchRoot |
    kFSEventStreamCreateFlagFileEvents;

bool requires_rescan(FSEventStreamEventFlags flags) noexcept {
  return (flags & (kFSEventStreamEventFlagMustScanSubDirs |
                   kFSEventStreamEventFlagUserDropped |
                   kFSEventStreamEventFlagKernelDropped |
                   kFSEventStreamEventFlagEventIdsWrapped |
                   kFSEventStreamEventFlagRootChanged |
                   kFSEventStreamEventFlagUnmount)) != 0;
}

FileWatchChange translate_flags(FSEventStreamEventFlags flags) noexcept {
  FileWatchChange changes = FileWatchChange::kNone;
  if ((flags & kFSEventStreamEventFlagItemCreated) != 0)
    changes |= FileWatchChange::kCreated;
  if ((flags & kFSEventStreamEventFlagItemRemoved) != 0)
    changes |= FileWatchChange::kRemoved;
  if ((flags & kFSEventStreamEventFlagItemModified) != 0)
    changes |= FileWatchChange::kContent;
  if ((flags & (kFSEventStreamEventFlagItemInodeMetaMod |
                kFSEventStreamEventFlagItemFinderInfoMod |
                kFSEventStreamEventFlagItemChangeOwner |
                kFSEventStreamEventFlagItemXattrMod)) != 0)
    changes |= FileWatchChange::kMetadata;
  if ((flags & kFSEventStreamEventFlagItemRenamed) != 0) {
    // FSEvents does not pair or identify the direction of a rename. Expose
    // both portable possibilities and leave correlation zero so consumers
    // re-read authoritative state instead of inventing an identity.
    changes |= FileWatchChange::kMovedFrom | FileWatchChange::kMovedTo;
  }
  return changes;
}

std::string_view parent_path(std::string_view path,
                             std::string_view* basename) noexcept {
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string_view::npos) {
    *basename = path;
    return {};
  }
  *basename = path.substr(separator + 1);
  return separator == 0 ? path.substr(0, 1) : path.substr(0, separator);
}

void publish_fsevents(ConstFSEventStreamRef, void* opaque,
                      std::size_t event_count, void* event_paths,
                      const FSEventStreamEventFlags event_flags[],
                      const FSEventStreamEventId[]) noexcept {
  auto& state = *static_cast<FileWatcherState*>(opaque);
  auto** paths = static_cast<char**>(event_paths);
  bool changed = false;
  bool root_was_invalidated = false;
  std::lock_guard<util::SpinLock> lock(state.monitor);

  for (std::size_t index = 0; index < event_count; ++index) {
    const FSEventStreamEventFlags flags = event_flags[index];
    if (requires_rescan(flags))
      changed = state.require_rescan_locked() || changed;

    const char* raw = paths[index];
    if (raw == nullptr) {
      changed = state.require_rescan_locked() || changed;
      continue;
    }
    const std::string_view path(raw);

    // A root event is coarse or reports that the root moved/disappeared.
    // Either way consumers must rebuild their authoritative view.
    if (const std::size_t* root = state.root_by_path.find(path)) {
      changed = state.require_rescan_locked() || changed;
      if ((flags & (kFSEventStreamEventFlagRootChanged |
                    kFSEventStreamEventFlagUnmount)) != 0) {
        state.roots[*root].active = false;
        root_was_invalidated = true;
        changed = state.invalidate_root_locked(*root) || changed;
      }
      continue;
    }

    std::string_view basename;
    const std::string_view parent = parent_path(path, &basename);
    const std::size_t* root = state.root_by_path.find(parent);
    if (root == nullptr || basename.empty()) {
      // FSEvents is recursive internally. A parent not present in the root
      // table is a deeper descendant and is deliberately filtered out.
      continue;
    }
    const FileWatchChange event = translate_flags(flags);
    if (event != FileWatchChange::kNone)
      changed = state.emit_for_root_locked(*root, basename, event, 0) ||
                changed;
  }
  if (root_was_invalidated) {
    bool any_active = false;
    for (const NativeWatchRoot& root : state.roots)
      any_active = any_active || root.active;
    if (!any_active && state.failure.ok())
      state.failure =
          Status::NotFound("all FileWatcher roots were invalidated");
  }
  state.publish_locked(changed);
}

void dispatch_barrier(void*) noexcept {}

}  // namespace

Status initialize_native_watch(FileWatcherState& state) noexcept {
  CFMutableArrayRef paths = CFArrayCreateMutable(
      kCFAllocatorDefault, static_cast<CFIndex>(state.roots.size()),
      &kCFTypeArrayCallBacks);
  if (paths == nullptr)
    return Status::ResourceExhausted(
        "FileWatcher FSEvents path array allocation failed");

  Status status;
  for (std::size_t index = 0; index < state.roots.size(); ++index) {
    const std::string_view path = state.root_path(index);
    CFStringRef string = CFStringCreateWithBytes(
        kCFAllocatorDefault, reinterpret_cast<const UInt8*>(path.data()),
        static_cast<CFIndex>(path.size()), kCFStringEncodingUTF8, false);
    if (string == nullptr) {
      status = Status::InvalidArgument(
          "FileWatcher root cannot be represented by CoreFoundation");
      break;
    }
    CFArrayAppendValue(paths, string);
    CFRelease(string);
  }
  if (!status.ok()) {
    CFRelease(paths);
    return status;
  }

  FSEventStreamContext context{};
  context.info = &state;
  FSEventStreamRef stream = FSEventStreamCreate(
      kCFAllocatorDefault, publish_fsevents, &context, paths,
      kFSEventStreamEventIdSinceNow, 0.05, kCreateFlags);
  CFRelease(paths);
  if (stream == nullptr)
    return Status::ResourceExhausted(
        "FileWatcher FSEvents stream allocation failed");

  dispatch_queue_t queue = dispatch_queue_create(
      "dev.hyle.io.file-watch", DISPATCH_QUEUE_SERIAL);
  if (queue == nullptr) {
    FSEventStreamInvalidate(stream);
    FSEventStreamRelease(stream);
    return Status::ResourceExhausted(
        "FileWatcher dispatch queue allocation failed");
  }

  FSEventStreamSetDispatchQueue(stream, queue);
  if (!FSEventStreamStart(stream)) {
    FSEventStreamInvalidate(stream);
    FSEventStreamRelease(stream);
#if !OS_OBJECT_USE_OBJC
    dispatch_release(queue);
#endif
    return Status::Unavailable("FileWatcher FSEvents stream failed to start");
  }

  state.apple_stream = stream;
  state.apple_queue = queue;
  for (NativeWatchRoot& root : state.roots) root.active = true;
  // FSEventStreamStart synchronously registers the stream; unlike the IOCP and
  // inotify coroutine pumps there is no first suspended native operation to
  // await before create() returns.
  state.expected_arms = 0;
  return Status::Ok();
}

Status shutdown_native_watch(FileWatcherState& state) noexcept {
  auto stream = static_cast<FSEventStreamRef>(state.apple_stream);
  auto queue = static_cast<dispatch_queue_t>(state.apple_queue);
  if (stream != nullptr) {
    FSEventStreamStop(stream);
    FSEventStreamInvalidate(stream);
  }
  // Drain the private callback queue before releasing the stream or allowing
  // FileWatcherState to die. The owner scheduler is never this private queue.
  if (queue != nullptr) dispatch_sync_f(queue, nullptr, dispatch_barrier);
  if (stream != nullptr) FSEventStreamRelease(stream);
#if !OS_OBJECT_USE_OBJC
  if (queue != nullptr) dispatch_release(queue);
#endif
  state.apple_stream = nullptr;
  state.apple_queue = nullptr;
  for (NativeWatchRoot& root : state.roots) root.active = false;
  return Status::Ok();
}

}  // namespace hyle::io::detail

#endif  // HYLE_OS_APPLE
