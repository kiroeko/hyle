#include "execution_source_internal.h"

#include <utility>

namespace hyle::graph {
namespace detail {
namespace {
extern const ExecutionSourceVTable compute_source_table;
struct ComputeSource final : ExecutionSourceState {
  ComputeSource(compute::detail::ContextRef retained, mem::AllocatorRef allocator) noexcept
      : ExecutionSourceState(&compute_source_table, allocator, retained.identity_key()),
        context(std::move(retained)) {}
  compute::detail::ContextRef context;
};
const ComputeSource &compute_state(const ExecutionSourceState &state) noexcept {
  return static_cast<const ComputeSource &>(state);
}
const ExecutionSourceVTable compute_source_table{
    [](ExecutionSourceState *state) noexcept {
      const auto allocator = state->allocator;
      allocator.destroy(static_cast<ComputeSource *>(state));
    },
    [](const ExecutionSourceState &state) noexcept { return compute_state(state).context.is_open(); },
    [](const ExecutionSourceState &a, const ExecutionSourceState &b) noexcept {
      return compute_state(a).context.device() == compute_state(b).context.device();
    },
    [](const ExecutionSourceState &state) noexcept { return compute_state(state).context.device().backend_id(); },
    [](const ExecutionSourceState &state) noexcept { return compute_state(state).context.device().device_class(); },
    [](const ExecutionSourceState &state) noexcept { return compute_state(state).context.device().properties(); },
    [](const ExecutionSourceState &state, mem::MemoryResourceRef resource) noexcept {
      return compute_state(state).context.validate_resource(resource);
    },
    [](const ExecutionSourceState &state, SourceCall function, void *opaque) noexcept {
      return compute_state(state).context.call(function, opaque);
    },
    [](const ExecutionSourceState &state) noexcept -> const compute::detail::ContextRef * {
      return &compute_state(state).context;
    },
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
} // namespace

Result<const compute::detail::ContextRef *>
ExecutionSourceAccess::compute(const ExecutionSource &source) noexcept {
  if (!source.state_)
    return std::unexpected(Status::FailedPrecondition("execution source is empty"));
  const auto *table = source.state_->table;
  if (!table->compute)
    return std::unexpected(Status::Unimplemented("native execution source has no Compute provider slots"));
  return table->compute(*source.state_);
}
bool ExecutionSourceAccess::owns(const ExecutionSource &source, const compute::Queue &queue) noexcept {
  const auto retained = compute(source);
  return retained && (*retained)->owns(queue);
}
bool ExecutionSourceAccess::owns(const ExecutionSource &source,
    const compute::detail::QueueRef &queue) noexcept {
  const auto retained = compute(source);
  return retained && (*retained)->owns(queue);
}
bool ExecutionSourceAccess::owns(const ExecutionSource &source,
    const compute::detail::ProviderExtensionRef &extension) noexcept {
  const auto retained = compute(source);
  return retained && (*retained)->owns(extension);
}
Status ExecutionSourceAccess::call(const ExecutionSource &source, SourceCall function, void *opaque) noexcept {
  if (!source.state_)
    return Status::FailedPrecondition("execution source is empty");
  if (!function)
    return Status::InvalidArgument("execution source call is empty");
  return source.state_->table->call(*source.state_, function, opaque);
}
Result<void *> ExecutionSourceAccess::native_context(const ExecutionSource &source) noexcept {
  if (!source.state_)
    return std::unexpected(Status::FailedPrecondition("execution source is empty"));
  const auto *table = source.state_->table;
  if (!table->native_context)
    return std::unexpected(Status::Unimplemented("Compute source has no native Runtime capability"));
  auto *context = table->native_context(*source.state_);
  if (!context)
    return std::unexpected(Status::Internal("native execution source lost its Runtime"));
  return context;
}
Result<GraphRegionSource> ExecutionSourceAccess::region(const ExecutionSource &source) noexcept {
  if (!source.state_)
    return std::unexpected(Status::FailedPrecondition("execution source is empty"));
  const auto *table = source.state_->table;
  if (!table->region_provider)
    return std::unexpected(Status::Unimplemented("execution source has no native region capability"));
  const auto *interface_value = table->region_provider(*source.state_);
  if (!interface_value)
    return std::unexpected(Status::Unimplemented("execution source has no native region capability"));
  return GraphRegionSource::native(source, interface_value);
}

Result<bool> ExecutionSourceAccess::can_copy_constant(
    const ExecutionSource &source, const ConstantCopy &copy) noexcept {
  if (!source.state_)
    return std::unexpected(Status::FailedPrecondition("execution source is empty"));
  const auto *table = source.state_->table;
  if (!table->can_copy_constant || !table->copy_constants || !table->constant_copy_scratch_bytes)
    return false;
  if (!source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is closed"));
  return table->can_copy_constant(*source.state_, copy);
}

Status ExecutionSourceAccess::copy_constants(const ExecutionSource &source,
    std::span<const ConstantCopy> copies, mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid())
    return Status::InvalidArgument("constant copy allocator is invalid");
  if (!source.state_)
    return Status::FailedPrecondition("execution source is empty");
  const auto *table = source.state_->table;
  if (!table->can_copy_constant || !table->copy_constants || !table->constant_copy_scratch_bytes)
    return Status::Unimplemented("execution source has no native constant transfer");
  struct Call final {
    const ExecutionSourceState &state;
    std::span<const ConstantCopy> copies;
    mem::AllocatorRef allocator;
  } call{*source.state_, copies, allocator};
  return ExecutionSourceAccess::call(source, [](void *opaque) noexcept {
    const auto &call = *static_cast<const Call *>(opaque);
    return call.state.table->copy_constants(call.state, call.copies, call.allocator);
  }, &call);
}

Result<std::size_t> ExecutionSourceAccess::constant_copy_scratch_bytes(
    const ExecutionSource &source, std::span<const ConstantCopy> copies) noexcept {
  if (!source.state_ || !source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is empty or closed"));
  const auto query = source.state_->table->constant_copy_scratch_bytes;
  if (!query)
    return std::unexpected(Status::Unimplemented("execution source has no constant copy scratch contract"));
  return query(*source.state_, copies);
}

bool ExecutionSourceAccess::matches_read_resource(const ExecutionSource &source,
    mem::MemoryResourceRef nominated, mem::MemoryResourceRef actual) noexcept {
  if (!source.state_ || !nominated.valid() || !actual.valid())
    return false;
  if (nominated == actual)
    return true;
  const auto match = source.state_->table->matches_read_resource;
  return match && match(*source.state_, nominated, actual);
}

Result<bool> ExecutionSourceAccess::can_copy(const ExecutionSource &source,
    const CopySourceDescription &from, mem::MemoryResourceRef to) noexcept {
  if (!source.state_ || !source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is empty or closed"));
  if (!from.description.valid() || !from.resource.valid() || !to.valid() || !from.value.valid() ||
      (from.value.kind != RegionOutputKind::kCaller && !from.producer.valid()))
    return std::unexpected(Status::InvalidArgument("transfer source contract or resource is invalid"));
  const auto *table = source.state_->table;
  if (!table->can_copy || !table->prepare_copy)
    return false;
  return table->can_copy(*source.state_, from, to);
}

Status ExecutionSourceAccess::prepare_copy(const ExecutionSource &source,
    const CopySourceDescription &from, mem::MemoryResourceRef to, mem::AllocatorRef allocator,
    ProviderRegionPrepared *result) noexcept {
  if (!result || !allocator.valid())
    return Status::InvalidArgument("native transfer preparation arguments are invalid");
  HYLE_ASSIGN_OR_RETURN(const bool supported, can_copy(source, from, to));
  if (!supported)
    return Status::Unimplemented("execution source cannot transfer between these resources");
  struct Call final {
    const ExecutionSourceState &state;
    const CopySourceDescription &from;
    mem::MemoryResourceRef to;
    mem::AllocatorRef allocator;
    ProviderRegionPrepared *result;
  } call{*source.state_, from, to, allocator, result};
  return ExecutionSourceAccess::call(source, [](void *opaque) noexcept {
    const auto &call = *static_cast<const Call *>(opaque);
    return call.state.table->prepare_copy(call.state, call.from,
        call.to, call.allocator, call.result);
  }, &call);
}

Result<GraphRegionSource> GraphRegionSource::shared(ExecutionSource source,
    compute::detail::ProviderExtensionRef extension) noexcept {
  if (!extension.valid() || !ExecutionSourceAccess::owns(source, extension))
    return std::unexpected(Status::InvalidArgument("region capability belongs to another execution source"));
  if (!source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is closed"));
  GraphRegionSource result;
  result.source_ = std::move(source);
  result.extension_ = std::move(extension);
  return result;
}
Result<GraphRegionSource> GraphRegionSource::native(ExecutionSource source,
    const void *interface_value) noexcept {
  if (!interface_value)
    return std::unexpected(Status::InvalidArgument("native region capability is empty"));
  HYLE_ASSIGN_OR_RETURN(auto context, ExecutionSourceAccess::native_context(source));
  (void)context;
  if (!source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is closed"));
  GraphRegionSource result;
  result.source_ = std::move(source);
  result.interface_ = interface_value;
  return result;
}
Result<GraphRegionSource> GraphRegionSource::shared_queue(ExecutionSource source,
    const void *interface_value) noexcept {
  if (!interface_value)
    return std::unexpected(Status::InvalidArgument("shared Queue region interface is empty"));
  HYLE_ASSIGN_OR_RETURN(auto context, ExecutionSourceAccess::compute(source));
  (void)context;
  if (!source.is_open())
    return std::unexpected(Status::FailedPrecondition("execution source is closed"));
  GraphRegionSource result;
  result.source_ = std::move(source);
  result.interface_ = interface_value;
  return result;
}
bool GraphRegionSource::is_shared_queue() const noexcept {
  return valid() && !extension_.valid() && ExecutionSourceAccess::compute(source_).has_value();
}
bool GraphRegionSource::is_native() const noexcept {
  return valid() && !extension_.valid() && !is_shared_queue();
}
Status GraphRegionSource::call(compute::detail::ProviderExtensionCallFn function, void *opaque) const noexcept {
  if (!valid() || !function)
    return Status::InvalidArgument("region source call is incomplete");
  if (extension_.valid())
    return compute::detail::ProviderExtensionAccess::call(extension_, function, opaque);
  void *context = nullptr;
  if (!is_shared_queue()) {
    HYLE_ASSIGN_OR_RETURN(context, ExecutionSourceAccess::native_context(source_));
  }
  struct Call final {
    compute::detail::ProviderExtensionCallFn function;
    void *context;
    const void *interface_value;
    void *opaque;
  } call{function, context, interface_, opaque};
  return ExecutionSourceAccess::call(source_, [](void *value) noexcept {
    auto &call = *static_cast<Call *>(value);
    return call.function(call.context, call.interface_value, call.opaque);
  }, &call);
}
} // namespace detail

ExecutionSource::ExecutionSource(const ExecutionSource &other) noexcept : state_(other.state_) {
  if (state_) state_->incref();
}
ExecutionSource &ExecutionSource::operator=(const ExecutionSource &other) noexcept {
  if (this != &other) {
    if (other.state_) other.state_->incref();
    if (state_) state_->decref();
    state_ = other.state_;
  }
  return *this;
}
ExecutionSource::ExecutionSource(ExecutionSource &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
ExecutionSource &ExecutionSource::operator=(ExecutionSource &&other) noexcept {
  if (this != &other) {
    if (state_) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
ExecutionSource::~ExecutionSource() { if (state_) state_->decref(); }

Result<ExecutionSource> ExecutionSource::acquire(const compute::Context &context) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto retained, compute::detail::ContextRef::acquire(context));
  const auto allocator = retained.metadata_allocator();
  if (!allocator.valid())
    return std::unexpected(Status::InvalidArgument("execution source allocator is invalid"));
  auto *state = allocator.construct<detail::ComputeSource>(std::move(retained), allocator);
  if (!state)
    return std::unexpected(Status::ResourceExhausted("execution source owner metadata"));
  return detail::ExecutionSourceAccess::adopt(state);
}
bool ExecutionSource::is_open() const noexcept { return state_ && state_->table->is_open(*state_); }
bool ExecutionSource::same_owner(const ExecutionSource &other) const noexcept {
  return state_ && other.state_ && state_->table == other.state_->table && state_->identity == other.state_->identity;
}
bool ExecutionSource::same_device(const ExecutionSource &other) const noexcept {
  return state_ && other.state_ && state_->table == other.state_->table && state_->table->same_device(*state_, *other.state_);
}
compute::BackendId ExecutionSource::backend_id() const noexcept {
  return state_ ? state_->table->backend_id(*state_) : compute::BackendId{};
}
compute::DeviceClass ExecutionSource::device_class() const noexcept {
  return state_ ? state_->table->device_class(*state_) : compute::DeviceClass::kUnknown;
}
Result<compute::DeviceProperties> ExecutionSource::device_properties() const noexcept {
  if (!state_)
    return std::unexpected(Status::FailedPrecondition("execution source is empty"));
  return state_->table->device_properties(*state_);
}
mem::AllocatorRef ExecutionSource::metadata_allocator() const noexcept { return state_ ? state_->allocator : mem::AllocatorRef{}; }
Status ExecutionSource::validate_resource(mem::MemoryResourceRef resource) const noexcept {
  if (!state_) return Status::FailedPrecondition("execution source is empty");
  return state_->table->validate_resource(*state_, resource);
}
} // namespace hyle::graph
