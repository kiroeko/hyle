#include "hyle/graph/apple.h"
#include "native_region.h"

#include "../graph/execution_source_internal.h"
#include "../compute_route_apple/apple_copy_internal.h"
#include "hyle/core/assert.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/mem/std_allocator.h"

#include <mutex>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace hyle::graph::apple {
namespace ca = compute::apple;
namespace gd = graph::detail;
namespace {
constexpr compute::BackendId apple_backend{0x48594c4500000001ULL, 0x4150504c00000001ULL};
extern const gd::ExecutionSourceVTable source_table;
struct AppleSource final : gd::ExecutionSourceState {
  AppleSource(std::shared_ptr<ca::detail::Runtime> owner, CustomLowering custom,
              mem::AllocatorRef allocator) noexcept
      : ExecutionSourceState(&source_table, allocator, owner.get()),
        context{std::move(owner), std::move(custom)} {
    // Copies retain this configured owner. Independent custom capabilities on
    // the same native Runtime are distinct cold compilation sources.
    if (context.custom) identity = &context;
  }
  detail::SourceContext context;
};
const AppleSource &native(const gd::ExecutionSourceState &state) noexcept {
  return static_cast<const AppleSource &>(state);
}
Status admission(const AppleSource &source) noexcept {
  std::lock_guard lock(source.context.runtime->queue_mutex);
  if (!source.context.runtime->accepting) return Status::FailedPrecondition("Apple execution source is closed");
  return source.context.runtime->failure;
}
struct CopyArtifact final {
  std::shared_ptr<ca::detail::Runtime> runtime;
  gd::CopySourceDescription source;
  mem::MemoryResourceRef destination;
  std::size_t bytes = 0;
};
struct CopySlot final {
  std::shared_ptr<ca::detail::CopyRecording> recording;
  RegionOutputReference reference;
  mem::ConstBufferView source;
  mem::BufferView destination;
  bool bound = false, sealed = false;
};
bool same_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
Status copy_bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = *static_cast<const CopyArtifact *>(prepared);
  auto &slot = *static_cast<CopySlot *>(invocation);
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0].description != artifact.source.description || !outputs[0].valid() ||
      outputs[0].description() != artifact.source.description ||
      outputs[0].storage().resource() != artifact.destination)
    return Status::InvalidArgument("Apple replay copy differs from its exact ports");
  HYLE_RETURN_IF_ERROR(ca::detail::copy_recording_reuse_gate(*slot.recording));
  slot.sealed = false;
  const auto destination = outputs[0].storage();
  if (artifact.source.value.kind == RegionOutputKind::kDeferredGenerated) {
    HYLE_ASSIGN_OR_RETURN(auto source, detail::native_output_source(
        inputs[0].reference, artifact.runtime, artifact.source.description));
    if (slot.bound && ca::detail::copy_recording_matches_source(*slot.recording, source) &&
        same_range(slot.destination, destination)) return Status::Ok();
    slot.bound = false;
    HYLE_RETURN_IF_ERROR(ca::detail::bind_host_copy_recording(*slot.recording,
        std::move(source), artifact.source.description, destination));
    slot.reference = inputs[0].reference; slot.source = {};
  } else {
    const auto &input = inputs[0];
    if (input.reference.valid() || !input.tensor.valid() ||
        input.tensor.description() != artifact.source.description ||
        !ca::detail::copy_source_resource_matches(artifact.source.resource, input.tensor.storage().resource()))
      return Status::InvalidArgument("Apple fixed replay copy source differs from its exact carrier");
    const auto source = input.tensor.storage();
    if (slot.bound && same_range(slot.source, source) && same_range(slot.destination, destination))
      return Status::Ok();
    slot.bound = false;
    compute::ReadLease owner;
    if (input.lease) {
      HYLE_ASSIGN_OR_RETURN(auto owned, input.lease->view());
      if (owned.resource() != source.resource() || owned.native_handle() != source.native_handle() ||
          source.offset() < owned.offset() || source.offset() - owned.offset() > owned.size() ||
          source.size() > owned.size() - (source.offset() - owned.offset()))
        return Status::InvalidArgument("Apple replay copy lease does not cover its actual source");
      HYLE_ASSIGN_OR_RETURN(owner, input.lease->subview(source.offset() - owned.offset(), source.size()));
    }
    HYLE_RETURN_IF_ERROR(ca::detail::bind_host_copy_recording(*slot.recording,
        input.tensor, destination, std::move(owner)));
    slot.source = source; slot.reference = {};
  }
  slot.destination = destination; slot.bound = true;
  return Status::Ok();
}
Status copy_seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<CopySlot *>(invocation);
  if (!slot.bound) return Status::FailedPrecondition("Apple replay copy is not cold-bound");
  HYLE_RETURN_IF_ERROR(ca::detail::copy_recording_reuse_gate(*slot.recording));
  slot.sealed = true;
  return Status::Ok();
}
Result<compute::Completion> copy_submit(void *prepared, void *invocation,
    std::span<const compute::detail::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = *static_cast<const CopyArtifact *>(prepared);
  auto &slot = *static_cast<CopySlot *>(invocation);
  if (!slot.sealed || inputs.size() != 1 || outputs.size() != 1 ||
      outputs[0].description() != artifact.source.description ||
      !same_range(outputs[0].storage(), slot.destination) ||
      (artifact.source.value.kind != RegionOutputKind::kDeferredGenerated &&
       (!inputs[0].valid() || inputs[0].description() != artifact.source.description ||
        !same_range(inputs[0].storage(), slot.source))))
    return std::unexpected(Status::FailedPrecondition("Apple replay copy ports are not sealed"));
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(ca::detail::qualify_recording_dependency(artifact.runtime, dependency));
  return ca::detail::submit_copy_recording(*slot.recording);
}
const gd::RegionBindingReleaseVTable copy_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      return ca::detail::copy_recording_retired_gate(*static_cast<CopySlot *>(invocation)->recording);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<CopySlot *>(invocation);
      HYLE_FATAL_ASSERT(ca::detail::clear_copy_recording(*slot.recording).ok());
      slot.reference = {}; slot.source = {}; slot.destination = {};
      slot.bound = false; slot.sealed = false;
    }};
const ProviderRegionPreparedVTable copy_table{
    .destroy_prepared = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<CopyArtifact *>(state));
    },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = [](void *prepared, mem::AllocatorRef allocator,
        mem::BufferView, void **output) noexcept -> Status {
      *output = nullptr;
      const auto &artifact = *static_cast<const CopyArtifact *>(prepared);
      auto *slot = allocator.construct<CopySlot>();
      if (!slot) return Status::ResourceExhausted("Apple replay copy slot metadata");
      *output = slot;
      HYLE_ASSIGN_OR_RETURN(slot->recording, ca::detail::prepare_copy_recording_runtime(
          artifact.runtime, 1, allocator, std::max(artifact.bytes, std::size_t{1}),
          ca::detail::CopyRecordingMode::kReplayable));
      return Status::Ok();
    },
    .destroy_invocation = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<CopySlot *>(state));
    },
    .delegated_submit = &copy_submit,
    .endpoint = [](void *prepared, void *) noexcept {
      return detail::region_ordered_endpoint(static_cast<CopyArtifact *>(prepared)->runtime);
    },
    .can_import_endpoint = [](void *prepared, void *, compute::detail::CompletionEndpoint source) noexcept {
      return detail::imports_region_ordered_endpoint(static_cast<CopyArtifact *>(prepared)->runtime, source);
    },
    .prepare_bindings = &copy_bind, .seal_bindings = &copy_seal,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      return id == gd::kRegionBindingReleaseExtensionId ? &copy_release_table : nullptr;
    }};
Result<bool> can_copy(const gd::ExecutionSourceState &state,
    const gd::CopySourceDescription &source, mem::MemoryResourceRef destination) noexcept {
  const auto &owner = native(state);
  HYLE_RETURN_IF_ERROR(admission(owner));
  if (!source.value.valid() || !source.description.valid() || !source.resource.valid() || !destination.valid() ||
      !destination.properties().has(mem::MemoryProperty::kHostAccessible) ||
      !ca::detail::is_copy_resource(source.resource) ||
      (source.description.is_plain() && !source.description.is_dense())) return false;
  if (source.value.kind == RegionOutputKind::kDeferredGenerated) {
    const auto *producer = gd::ExecutionSourceAccess::state(source.producer);
    return producer && producer->table == &source_table && native(*producer).context.runtime == owner.context.runtime;
  }
  return source.value.projection == RegionOutputProjection::kTensorView;
}
Status prepare_copy(const gd::ExecutionSourceState &state, const gd::CopySourceDescription &source,
    mem::MemoryResourceRef destination, mem::AllocatorRef allocator, ProviderRegionPrepared *result) noexcept {
  if (!result) return Status::InvalidArgument("Apple replay copy requires an artifact output");
  *result = {};
  if (!allocator.valid()) return Status::InvalidArgument("Apple replay copy allocator is invalid");
  HYLE_ASSIGN_OR_RETURN(auto supported, can_copy(state, source, destination));
  if (!supported) return Status::InvalidArgument("Apple replay copy has no native producer/host destination route");
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto carrier, ca::input_spec(source.description));
  (void)carrier;
  auto *artifact = allocator.construct<CopyArtifact>(CopyArtifact{native(state).context.runtime, source, destination, bytes});
  if (!artifact) return Status::ResourceExhausted("Apple replay copy artifact metadata");
  result->vtable = &copy_table; result->state = artifact;
  result->properties = {.persistent_bytes = sizeof(CopyArtifact), .copy_bytes = bytes,
      .runtime = {.mode = compute::ProviderRuntimeMode::kDelegated,
          .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
          .replay_class = compute::ProviderReplayClass::kPrepared,
          .uses_private_threads = true},
      .invocation_private_bytes = std::nullopt};
  return Status::Ok();
}
const gd::ExecutionSourceVTable source_table{
    [](gd::ExecutionSourceState *state) noexcept {
      state->allocator.destroy(static_cast<AppleSource *>(state));
    },
    [](const gd::ExecutionSourceState &state) noexcept { return admission(native(state)).ok(); },
    [](const gd::ExecutionSourceState &a, const gd::ExecutionSourceState &b) noexcept {
      return native(a).context.runtime->stream->device == native(b).context.runtime->stream->device;
    },
    [](const gd::ExecutionSourceState &) noexcept { return apple_backend; },
    [](const gd::ExecutionSourceState &) noexcept { return compute::DeviceClass::kGpu; },
    [](const gd::ExecutionSourceState &state) noexcept -> Result<compute::DeviceProperties> {
      return compute::DeviceProperties{.backend = apple_backend,
          .device_class = compute::DeviceClass::kGpu, .backend_name = "MLX Metal",
          .name = native(state).context.runtime->environment.device_name,
          .unified_memory = true, .asynchronous_queues = true};
    },
    [](const gd::ExecutionSourceState &, mem::MemoryResourceRef resource) noexcept {
      return ca::detail::is_copy_resource(resource) ? Status::Ok()
          : Status::InvalidArgument("Apple source requires its original shared Metal resource");
    },
    [](const gd::ExecutionSourceState &state, gd::SourceCall function, void *opaque) noexcept -> Status {
      HYLE_RETURN_IF_ERROR(admission(native(state)));
      // Original native preparation/submission owns its admission and FIFO.
      // Never hold queue_mutex across a cold owner Job or invent a Queue.
      return function(opaque);
    },
    nullptr,
    [](const gd::ExecutionSourceState &state) noexcept -> void * {
      // Borrow this actual typed configuration only while Source lives.
      // Artifacts retain the Runtime and resulting compiled callable.
      return const_cast<detail::SourceContext *>(&native(state).context);
    },
    [](const gd::ExecutionSourceState &) noexcept { return detail::native_region_provider(); },
    [](const gd::ExecutionSourceState &state, const gd::ConstantCopy &copy) noexcept {
      return ca::detail::MemoryAccess::can_copy_constant(native(state).context.runtime,
          {copy.destination, copy.source});
    },
    [](const gd::ExecutionSourceState &state, std::span<const gd::ConstantCopy> copies,
       mem::AllocatorRef allocator) noexcept -> Status {
      if (!allocator.valid()) return Status::InvalidArgument("Apple constant metadata allocator is invalid");
      try {
        std::vector<ca::detail::ConstantCopy, mem::StdAllocator<ca::detail::ConstantCopy>> batch{
            mem::StdAllocator<ca::detail::ConstantCopy>(allocator)};
        batch.reserve(copies.size());
        for (const auto &copy : copies) batch.push_back({copy.destination, copy.source});
        return ca::detail::MemoryAccess::copy_constants(native(state).context.runtime, batch, allocator);
      } catch (const std::bad_alloc &) {
        return Status::ResourceExhausted("Apple constant batch metadata");
      }
    },
    [](const gd::ExecutionSourceState &, std::span<const gd::ConstantCopy>) noexcept
        -> Result<std::size_t> { return std::size_t{0}; },
    [](const gd::ExecutionSourceState &, mem::MemoryResourceRef nominated,
       mem::MemoryResourceRef actual) noexcept {
      return ca::detail::copy_source_resource_matches(nominated, actual);
    }, &can_copy, &prepare_copy};
}

Result<ExecutionSource> execution_source(const ca::Context &context, SourceOptions options) noexcept {
  if (!context || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Apple execution source requires a Context and allocator"));
  const auto &runtime = ca::detail::ContextAccess::runtime(context);
  {
    std::lock_guard lock(runtime->queue_mutex);
    if (!runtime->accepting) return std::unexpected(Status::FailedPrecondition("Apple execution source is closed"));
    HYLE_RETURN_IF_ERROR(runtime->failure);
  }
  auto *source = options.allocator.construct<AppleSource>(runtime, std::move(options.custom), options.allocator);
  if (!source) return std::unexpected(Status::ResourceExhausted("Apple execution source metadata"));
  return gd::ExecutionSourceAccess::adopt(source);
}
} // namespace hyle::graph::apple
