#include "hyle/compute/detail/command_capture.h"

#include <utility>

namespace hyle::compute::detail {
namespace {
struct CaptureLaunch final {
  const CommandCaptureVTable *expected;
  void *state;
};
EnqueueResult launch_captured(void *backend_context, const void *interface_value,
                             void *backend_queue, void *opaque) noexcept {
  auto &call = *static_cast<CaptureLaunch *>(opaque);
  const auto *table = static_cast<const CommandCaptureVTable *>(interface_value);
  if (table != call.expected || table == nullptr || table->launch == nullptr)
    return EnqueueResult::rejected(
        Status::Internal("command capture launch table changed"));
  return table->launch(backend_context, backend_queue, call.state);
}
} // namespace

CapturedCommands::CapturedCommands(CapturedCommands &&other) noexcept
    : extension_(std::move(other.extension_)), table_(other.table_),
      state_(other.state_), allocator_(other.allocator_) {
  other.table_ = nullptr;
  other.state_ = nullptr;
  other.allocator_ = {};
}

CapturedCommands &
CapturedCommands::operator=(CapturedCommands &&other) noexcept {
  if (this != &other) {
    reset();
    extension_ = std::move(other.extension_);
    table_ = other.table_;
    state_ = other.state_;
    allocator_ = other.allocator_;
    other.table_ = nullptr;
    other.state_ = nullptr;
    other.allocator_ = {};
  }
  return *this;
}

CapturedCommands::~CapturedCommands() { reset(); }

void CapturedCommands::reset() noexcept {
  if (table_ != nullptr && state_ != nullptr && table_->destroy != nullptr)
    table_->destroy(state_, allocator_);
  extension_ = {};
  table_ = nullptr;
  state_ = nullptr;
  allocator_ = {};
}

EnqueueResult CapturedCommands::enqueue(QueueSequence &sequence) noexcept {
  if (!valid()) {
    return EnqueueResult::rejected(
        Status::FailedPrecondition("captured compute commands are invalid"));
  }
  CaptureLaunch episode{table_, state_};
  return BackendExtensionAccess::enqueue_admitted(
      extension_, sequence, &launch_captured, &episode);
}

Result<BackendExtensionRef>
CommandCaptureAccess::query(const Queue &queue) noexcept {
  return BackendExtensionAccess::query(queue, kCommandCaptureExtensionId);
}

Status CommandCaptureAccess::begin(const BackendExtensionRef &extension,
                                   QueueSequence &sequence) noexcept {
  if (sequence.recording_)
    return Status::FailedPrecondition("compute sequence is already recording");
  EnqueueResult result = BackendExtensionAccess::enqueue_admitted(
      extension, sequence,
      +[](void *backend_context, const void *interface_value,
          void *backend_queue, void *) noexcept -> EnqueueResult {
        const auto *table =
            static_cast<const CommandCaptureVTable *>(interface_value);
        if (table == nullptr || table->begin == nullptr ||
            table->end == nullptr || table->abort == nullptr ||
            table->destroy == nullptr || table->launch == nullptr) {
          return EnqueueResult::rejected(
              Status::Internal("command capture interface is incomplete"));
        }
        Status status = table->begin(backend_context, backend_queue);
        return status.ok() ? EnqueueResult::no_op()
                           : EnqueueResult::rejected(std::move(status));
      },
      nullptr);
  if (result.status.ok())
    sequence.recording_ = true;
  return std::move(result.status);
}

Result<CapturedCommands>
CommandCaptureAccess::end(const BackendExtensionRef &extension,
                          QueueSequence &sequence,
                          mem::AllocatorRef allocator,
                          CaptureRebuildStorage rebuild) noexcept {
  if (!sequence.recording_)
    return std::unexpected(Status::FailedPrecondition("compute sequence is not recording"));
  if (!allocator.valid()) {
    return std::unexpected(
        Status::InvalidArgument("command capture allocator is invalid"));
  }
  struct Episode final {
    mem::AllocatorRef allocator{};
    const CommandCaptureVTable *table = nullptr;
    void *state = nullptr;
    bool called = false;
    CapturedCommands *rebuild = nullptr;
  } episode{allocator, nullptr, nullptr, false, &rebuild.storage_};
  EnqueueResult result = BackendExtensionAccess::enqueue_admitted(
      extension, sequence,
      +[](void *backend_context, const void *interface_value,
          void *backend_queue, void *opaque) noexcept -> EnqueueResult {
        auto &call = *static_cast<Episode *>(opaque);
        call.table = static_cast<const CommandCaptureVTable *>(interface_value);
        if (call.table == nullptr || call.table->end == nullptr) {
          return EnqueueResult::rejected(
              Status::Internal("command capture end callback is missing"));
        }
        call.called = true;
        void *storage = call.rebuild->table_ == call.table
            ? call.rebuild->state_ : nullptr;
        Status status = call.table->end(backend_context, backend_queue,
                                        call.allocator, storage, &call.state);
        return status.ok() ? EnqueueResult::no_op()
                           : EnqueueResult::rejected(std::move(status));
      },
      &episode);
  // The backend end contract terminates recording even when finalization fails.
  // A rejected foreign extension never called it and cannot clear this state.
  if (episode.called)
    sequence.recording_ = false;
  if (!result.status.ok()) {
    if (episode.table != nullptr && episode.table->destroy != nullptr &&
        episode.state != nullptr) {
      episode.table->destroy(episode.state, allocator);
    }
    return std::unexpected(std::move(result.status));
  }
  if (episode.table == nullptr || episode.state == nullptr) {
    return std::unexpected(
        Status::Internal("command capture returned no replay artifact"));
  }
  return CapturedCommands(extension, episode.table, episode.state, allocator);
}

CaptureRebuildStorage CommandCaptureAccess::recycle_after_retirement(
    CapturedCommands commands) noexcept {
  CaptureRebuildStorage result;
  if (commands.valid() && commands.table_->recycle_after_retirement != nullptr &&
      commands.table_->recycle_after_retirement(commands.state_))
    result.storage_ = std::move(commands);
  return result;
}

Status CommandCaptureAccess::abort(const BackendExtensionRef &extension,
                                   QueueSequence &sequence) noexcept {
  if (!sequence.recording_)
    return Status::FailedPrecondition("compute sequence is not recording");
  bool called = false;
  EnqueueResult result = BackendExtensionAccess::enqueue_admitted(
      extension, sequence,
      +[](void *backend_context, const void *interface_value,
          void *backend_queue, void *opaque) noexcept -> EnqueueResult {
        const auto *table =
            static_cast<const CommandCaptureVTable *>(interface_value);
        if (table == nullptr || table->abort == nullptr) {
          return EnqueueResult::rejected(
              Status::Internal("command capture abort callback is missing"));
        }
        *static_cast<bool *>(opaque) = true;
        Status status = table->abort(backend_context, backend_queue);
        return status.ok() ? EnqueueResult::no_op()
                           : EnqueueResult::rejected(std::move(status));
      },
      &called);
  if (called)
    sequence.recording_ = false;
  return std::move(result.status);
}

} // namespace hyle::compute::detail
