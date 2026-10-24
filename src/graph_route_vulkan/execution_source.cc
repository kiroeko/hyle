#include "hyle/graph/vulkan.h"
#include "vulkan_source.h"
#include "native_region.h"

#include "../graph/execution_source_internal.h"
#include "../compute_route_vulkan/vulkan_composition_access.h"
#include "../compute_route_vulkan/vulkan_copy_internal.h"
#include "../compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_route_vulkan/vulkan_program_internal.h"

#include <mutex>
#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace hyle::graph::vulkan {
namespace cv = compute::vulkan;
namespace gd = graph::detail;
namespace {
// This is a real Vulkan endpoint identity, not a Compute Backend factory.
constexpr compute::BackendId vulkan_backend{0x48594c4500000001ULL, 0x56554c4b00000001ULL};
extern const gd::ExecutionSourceVTable source_table;
struct VulkanSource final : gd::ExecutionSourceState {
  VulkanSource(std::shared_ptr<cv::detail::Runtime> retained, CustomLowering custom,
               mem::AllocatorRef allocator) noexcept
      : ExecutionSourceState(&source_table, allocator, retained.get()),
        context{std::move(retained), std::move(custom)} {
    // Separate configured lowering capabilities on one real Runtime. Plain
    // wrappers continue to identify their original actual Runtime owner.
    if (context.custom) identity = &context;
  }
  detail::SourceContext context;
};
const VulkanSource &native(const gd::ExecutionSourceState &state) noexcept {
  return static_cast<const VulkanSource &>(state);
}
Status admission(const VulkanSource &state) noexcept {
  std::lock_guard lock(state.context.runtime->mutex);
  if (state.context.runtime->closed)
    return Status::FailedPrecondition("Vulkan execution source is closed");
  return state.context.runtime->failure;
}

// A copy is an original native Run with one fixed pair, not a Queue adapter.
// Only the cold binding callback can record commands or acquire backing pins.
struct CopyArtifact final {
  std::shared_ptr<cv::detail::Runtime> runtime;
  compute::TensorDesc description;
  mem::MemoryResourceRef source, destination;
  std::size_t bytes = 0;
};
struct CopySlot final {
  std::shared_ptr<cv::detail::CopyRecording> recording;
  mem::ConstBufferView source;
  mem::BufferView destination;
  compute::detail::PhysicalCopyResource source_physical, destination_physical;
  bool bound = false;
  bool sealed = false;
};
bool same_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
bool same_physical(const compute::detail::PhysicalCopyResource &a,
    const compute::detail::PhysicalCopyResource &b) noexcept {
  return a.address_space == b.address_space && a.backing == b.backing &&
      a.origin == b.origin && a.capacity == b.capacity && a.copy_alignment == b.copy_alignment;
}
Status copy_bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
                 std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = *static_cast<const CopyArtifact *>(prepared);
  auto &slot = *static_cast<CopySlot *>(invocation);
  if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0].tensor.valid() ||
      inputs[0].description != artifact.description ||
      inputs[0].tensor.description() != artifact.description ||
      outputs[0].description() != artifact.description ||
      inputs[0].tensor.storage().resource() != artifact.source ||
      outputs[0].storage().resource() != artifact.destination)
    return Status::InvalidArgument("Vulkan replay copy binding differs from its exact ports");
  HYLE_RETURN_IF_ERROR(cv::detail::copy_recording_reuse_gate(*slot.recording));
  const auto source = inputs[0].tensor.storage();
  const auto destination = outputs[0].storage();
  HYLE_ASSIGN_OR_RETURN(auto read, source.subview(0, artifact.bytes));
  HYLE_ASSIGN_OR_RETURN(auto write, destination.subview(0, artifact.bytes));
  HYLE_ASSIGN_OR_RETURN(auto physical_source, cv::detail::MemoryAccess::copy_span(artifact.runtime, read));
  HYLE_ASSIGN_OR_RETURN(auto physical_destination, cv::detail::MemoryAccess::copy_span(artifact.runtime, write));
  slot.sealed = false;
  // Mem native_handle is an AllocationToken address, which can be reused
  // after a generated producer replaces its backing. The original recording
  // pins the actual VkBuffers, so their canonical physical spans are the
  // stable cold cache identity. Keep current wrappers for hot binding checks.
  if (slot.bound && same_physical(physical_source.physical, slot.source_physical) &&
      same_physical(physical_destination.physical, slot.destination_physical)) {
    slot.source = source;
    slot.destination = destination;
    return Status::Ok();
  }
  slot.bound = false;
  const std::array<compute::detail::CopyBackendRange, 1> ranges{{{write, read}}};
  // Stay on the Runtime's original compute queue. Host writes into staging
  // already flush through Mem; the recorded barrier preserves both host and
  // preceding native queue writes without a private transfer-queue handoff.
  HYLE_RETURN_IF_ERROR(cv::detail::bind_copy_recording(*slot.recording, ranges));
  slot.source = source;
  slot.destination = destination;
  slot.source_physical = physical_source.physical;
  slot.destination_physical = physical_destination.physical;
  slot.bound = true;
  return Status::Ok();
}
Status copy_seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<CopySlot *>(invocation);
  if (!slot.bound) return Status::FailedPrecondition("Vulkan replay copy is not cold-bound");
  HYLE_RETURN_IF_ERROR(cv::detail::copy_recording_reuse_gate(*slot.recording));
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
      inputs[0].description() != artifact.description ||
      outputs[0].description() != artifact.description ||
      !same_range(inputs[0].storage(), slot.source) ||
      !same_range(outputs[0].storage(), slot.destination))
    return std::unexpected(Status::FailedPrecondition("Vulkan replay copy spans are not sealed"));
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(cv::detail::qualify_copy_dependency(artifact.runtime, dependency));
  return cv::detail::submit_copy_recording(*slot.recording);
}
const ProviderRegionPreparedVTable copy_table{
    .destroy_prepared = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<CopyArtifact *>(state));
    },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept {
      return compute::AliasSupport{};
    },
    .create_invocation = [](void *prepared, mem::AllocatorRef allocator,
                            mem::BufferView, void **output) noexcept -> Status {
      *output = nullptr;
      const auto &artifact = *static_cast<const CopyArtifact *>(prepared);
      auto *slot = allocator.construct<CopySlot>();
      if (!slot) return Status::ResourceExhausted("Vulkan replay copy slot metadata");
      *output = slot;
      HYLE_ASSIGN_OR_RETURN(slot->recording, cv::detail::prepare_copy_recording_runtime(
          artifact.runtime, 1, allocator, std::max(artifact.bytes, std::size_t{1}),
          cv::detail::CopyRecordingMode::kReplayable));
      return Status::Ok();
    },
    .destroy_invocation = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<CopySlot *>(state));
    },
    .delegated_submit = &copy_submit,
    .endpoint = [](void *prepared, void *) noexcept {
      return detail::region_ordered_endpoint(static_cast<CopyArtifact *>(prepared)->runtime);
    },
    .can_import_endpoint = [](void *prepared, void *,
                             compute::detail::CompletionEndpoint source) noexcept {
      return detail::imports_region_ordered_endpoint(
          static_cast<CopyArtifact *>(prepared)->runtime, source);
    },
    .prepare_bindings = &copy_bind,
    .seal_bindings = &copy_seal};
Result<bool> can_copy(const gd::ExecutionSourceState &state,
    const gd::CopySourceDescription &source, mem::MemoryResourceRef destination) noexcept {
  const auto &owner = native(state);
  HYLE_RETURN_IF_ERROR(admission(owner));
  if (source.value.kind == RegionOutputKind::kDeferredGenerated ||
      source.value.projection != RegionOutputProjection::kTensorView) return false;
  return cv::detail::MemoryAccess::validate_resource(owner.context.runtime, source.resource).ok() &&
      cv::detail::MemoryAccess::validate_resource(owner.context.runtime, destination).ok();
}
Status prepare_copy(const gd::ExecutionSourceState &state,
    const gd::CopySourceDescription &source,
    mem::MemoryResourceRef destination, mem::AllocatorRef allocator,
    ProviderRegionPrepared *result) noexcept {
  if (!result) return Status::InvalidArgument("Vulkan replay copy requires an artifact output");
  *result = {};
  if (!allocator.valid()) return Status::InvalidArgument("Vulkan replay copy allocator is invalid");
  HYLE_ASSIGN_OR_RETURN(auto supported, can_copy(state, source, destination));
  if (!supported) return Status::InvalidArgument("Vulkan replay copy resources belong to another Runtime");
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description.required_bytes());
  if (bytes % 4 != 0)
    return Status::Unimplemented("Vulkan replay copy byte span must satisfy native four-byte alignment");
  auto *artifact = allocator.construct<CopyArtifact>(CopyArtifact{
      native(state).context.runtime, source.description, source.resource, destination, bytes});
  if (!artifact) return Status::ResourceExhausted("Vulkan replay copy artifact metadata");
  result->vtable = &copy_table;
  result->state = artifact;
  result->properties = {.persistent_bytes = sizeof(CopyArtifact), .copy_bytes = bytes,
      .runtime = {.mode = compute::ProviderRuntimeMode::kDelegated,
          .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
          .replay_class = compute::ProviderReplayClass::kPrepared},
      .invocation_private_bytes = std::nullopt};
  return Status::Ok();
}
const gd::ExecutionSourceVTable source_table{
    [](gd::ExecutionSourceState *state) noexcept {
      const auto allocator = state->allocator;
      allocator.destroy(static_cast<VulkanSource *>(state));
    },
    [](const gd::ExecutionSourceState &state) noexcept { return admission(native(state)).ok(); },
    [](const gd::ExecutionSourceState &a, const gd::ExecutionSourceState &b) noexcept {
      return native(a).context.runtime->device == native(b).context.runtime->device;
    },
    [](const gd::ExecutionSourceState &) noexcept { return vulkan_backend; },
    [](const gd::ExecutionSourceState &state) noexcept {
      return native(state).context.runtime->device->properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
          ? compute::DeviceClass::kCpu : compute::DeviceClass::kGpu;
    },
    [](const gd::ExecutionSourceState &state) noexcept -> Result<compute::DeviceProperties> {
      const auto &device = *native(state).context.runtime->device;
      return compute::DeviceProperties{.backend=vulkan_backend,
          .device_class=device.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU
              ? compute::DeviceClass::kCpu : compute::DeviceClass::kGpu,.backend_name="Vulkan",
          .name=device.environment.device_name,.unified_memory=false,.asynchronous_queues=true};
    },
    [](const gd::ExecutionSourceState &state, mem::MemoryResourceRef resource) noexcept {
      return cv::detail::MemoryAccess::validate_resource(native(state).context.runtime, resource);
    },
    [](const gd::ExecutionSourceState &state, gd::SourceCall function, void *opaque) noexcept -> Status {
      HYLE_RETURN_IF_ERROR(admission(native(state)));
      // Never hold Runtime::mutex across native calls: prepare/bind/submit
      // enter its original lock themselves. The retained actual Runtime keeps
      // admission races safe; final native acceptance still checks closed.
      return function(opaque);
    },
    nullptr,
    [](const gd::ExecutionSourceState &state) noexcept -> void * {
      // Borrow immutable cold configuration only while Source is retained.
      // Artifacts copy the real Runtime and resolved pipelines, not the callback.
      return const_cast<detail::SourceContext *>(&native(state).context);
    },
    [](const gd::ExecutionSourceState &) noexcept { return detail::native_region_provider(); },
    [](const gd::ExecutionSourceState &state, const gd::ConstantCopy &copy) noexcept {
      return cv::detail::MemoryAccess::can_copy_constant(native(state).context.runtime,
          {copy.destination, copy.source});
    },
    [](const gd::ExecutionSourceState &state, std::span<const gd::ConstantCopy> copies,
       mem::AllocatorRef allocator) noexcept -> Status {
      if (!allocator.valid()) return Status::InvalidArgument("invalid Vulkan constant metadata allocator");
      try {
        std::vector<cv::detail::ConstantCopy, mem::StdAllocator<cv::detail::ConstantCopy>> batch{
            mem::StdAllocator<cv::detail::ConstantCopy>(allocator)};
        batch.reserve(copies.size());
        for (const auto &copy : copies) batch.push_back({copy.destination, copy.source});
        return cv::detail::MemoryAccess::copy_constants(native(state).context.runtime, batch, allocator);
      } catch (const std::bad_alloc &) {
        return Status::ResourceExhausted("Vulkan constant batch metadata");
      }
    },
    [](const gd::ExecutionSourceState &state, std::span<const gd::ConstantCopy> copies) noexcept
        -> Result<std::size_t> {
      std::size_t maximum = 0;
      for (const auto &copy : copies) {
        HYLE_ASSIGN_OR_RETURN(auto bytes,
            cv::detail::MemoryAccess::constant_copy_staging_bytes(native(state).context.runtime,
                {copy.destination, copy.source}));
        maximum = std::max(maximum, bytes);
      }
      return maximum;
    }, nullptr, &can_copy, &prepare_copy};

Result<const VulkanSource *> checked_source(const ExecutionSource &source) noexcept {
  const auto *state = gd::ExecutionSourceAccess::state(source);
  if (!state || state->table != &source_table)
    return std::unexpected(Status::InvalidArgument("source is not this actual Vulkan endpoint"));
  return &native(*state);
}
Status matches(const ExecutionSource &source, const cv::Invocation &invocation) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto state, checked_source(source));
  const auto &slot = cv::detail::Access::invocation_state(invocation);
  if (!slot || slot->program->runtime != state->context.runtime)
    return Status::InvalidArgument("Vulkan invocation belongs to another actual source");
  return Status::Ok();
}
} // namespace

Result<ExecutionSource> execution_source(const cv::Context &context, SourceOptions options) noexcept {
  if (!context.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Vulkan execution source requires a Context and allocator"));
  const auto &runtime = cv::detail::Access::runtime(context);
  {
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed)
      return std::unexpected(Status::FailedPrecondition("Vulkan execution source is closed"));
    HYLE_RETURN_IF_ERROR(runtime->failure);
  }
  auto *state = options.allocator.construct<VulkanSource>(
      runtime, std::move(options.custom), options.allocator);
  if (!state)
    return std::unexpected(Status::ResourceExhausted("Vulkan execution source metadata"));
  return gd::ExecutionSourceAccess::adopt(state);
}
namespace detail {
Result<cv::Tensor> output_backing(const ExecutionSource &source, cv::Invocation &invocation,
                                  std::size_t index) noexcept {
  HYLE_RETURN_IF_ERROR(matches(source, invocation));
  return cv::detail::CompositionAccess::output(invocation, index);
}
Status seal_outputs(const ExecutionSource &source, cv::Invocation &invocation) noexcept {
  HYLE_RETURN_IF_ERROR(matches(source, invocation));
  return cv::detail::CompositionAccess::seal(invocation);
}
} // namespace detail
} // namespace hyle::graph::vulkan
