#include "hyle/compute/transfer.h"

#include <cstdint>
#include <utility>

#include "backend_internal.h"
#include "context_state.h"
#include "hyle/compute/detail/queue_sequence.h"

namespace hyle::compute {
namespace {

Status validate_views(const detail::QueueState &queue,
                      mem::BufferView destination,
                      mem::ConstBufferView source) noexcept {
  if (!destination.valid() || !source.valid())
    return Status::InvalidArgument("compute transfer view is invalid");
  if (destination.size() != source.size()) {
    return Status::InvalidArgument(
        "compute transfer source and destination sizes differ");
  }
  const detail::ContextState &context = *queue.context;
  Status status = context.backend_vtable->validate_resource(
      context.device, destination.resource());
  if (!status.ok())
    return status;
  status = context.backend_vtable->validate_resource(context.device,
                                                     source.resource());
  if (!status.ok())
    return status;

  if (destination.resource() == source.resource() &&
      destination.native_handle() == source.native_handle()) {
    const std::size_t destination_begin = destination.offset();
    const std::size_t destination_end = destination_begin + destination.size();
    const std::size_t source_begin = source.offset();
    const std::size_t source_end = source_begin + source.size();
    const bool exact = destination_begin == source_begin &&
                       destination_end == source_end;
    if (!exact && destination_begin < source_end &&
        source_begin < destination_end) {
      return Status::InvalidArgument(
          "compute async copy ranges overlap");
    }
  }
  return Status::Ok();
}

} // namespace

Result<TransferProperties>
transfer_properties(const Queue &queue, mem::BufferView destination,
                    mem::ConstBufferView source) noexcept {
  detail::QueueState *state = detail::ContextAccess::state(queue);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("queue is invalid"));
  if (!queue.is_open())
    return std::unexpected(Status::FailedPrecondition("queue is closed"));
  Status status = validate_views(*state, destination, source);
  if (!status.ok())
    return std::unexpected(std::move(status));
  if (destination.size() == 0 ||
      (destination.resource() == source.resource() &&
       destination.native_handle() == source.native_handle() &&
       destination.offset() == source.offset())) {
    return TransferProperties{.mode = TransferMode::kImmediate};
  }
  return state->context->backend_vtable->transfer_properties(
      state->context->backend_state, destination, source);
}

Result<Completion> copy_async(Queue &queue, mem::BufferView destination,
                              mem::ConstBufferView source) noexcept {
  auto sequence = detail::QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());
  detail::QueueState *state = detail::ContextAccess::state(queue);
  Status status = validate_views(*state, destination, source);
  if (!status.ok()) {
    sequence->record(detail::EnqueueResult::rejected(std::move(status)));
    return sequence->finish();
  }
  if (destination.size() == 0 ||
      (destination.resource() == source.resource() &&
       destination.native_handle() == source.native_handle() &&
       destination.offset() == source.offset())) {
    sequence->record(detail::EnqueueResult::no_op());
  } else {
    sequence->record(state->context->backend_vtable->copy_async(
        state->backend_state, destination, source));
  }
  return sequence->finish();
}

Result<Completion> fill_async(Queue &queue, mem::BufferView destination,
                              std::byte value) noexcept {
  auto sequence = detail::QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());
  detail::QueueState *state = detail::ContextAccess::state(queue);
  if (!destination.valid()) {
    sequence->record(detail::EnqueueResult::rejected(
        Status::InvalidArgument("compute fill destination is invalid")));
    return sequence->finish();
  }
  Status status = state->context->backend_vtable->validate_resource(
      state->context->device, destination.resource());
  if (!status.ok()) {
    sequence->record(detail::EnqueueResult::rejected(std::move(status)));
  } else if (destination.empty()) {
    sequence->record(detail::EnqueueResult::no_op());
  } else {
    sequence->record(state->context->backend_vtable->fill_async(
        state->backend_state, destination, value));
  }
  return sequence->finish();
}

} // namespace hyle::compute
