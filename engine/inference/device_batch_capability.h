#pragma once

#include "hyle/compute/detail/batch_device_lengths.h"
#include "hyle/compute/detail/device_lengths_paged_append_backend.h"

namespace hyle::inference::detail {

// Cold configuration qualification through the actual provider. The temporary
// native descriptor owns metadata only; the caller's original Queue is used.
// Actual layouts/recording still require their ordinary cold qualification.
inline Result<bool> supports_captured_device_batch(const compute::Context &context,
    compute::Queue &queue, compute::BatchCapacity capacity,
    const compute::BatchProgramOptions &options) noexcept {
  namespace c = compute;
  namespace d = compute::detail;
  auto append = d::BackendExtensionAccess::query(context, d::kDeviceLengthsPagedAppendExtensionId);
  if (!append) {
    if (append.error().IsUnimplemented()) return false;
    return std::unexpected(append.error());
  }
  const auto append_status = d::BackendExtensionAccess::call(*append,
      [](void *, const void *interface_value, void *) noexcept -> Status {
        const auto *table = static_cast<const d::DeviceLengthsPagedAppendVTable *>(interface_value);
        if (!table || !table->prepare || !table->destroy || !table->enqueue || !table->supports_command_capture)
          return Status::Unimplemented("captured device-length append is unavailable");
        return Status::Ok();
      }, nullptr);
  if (!append_status.ok()) {
    if (append_status.IsUnimplemented()) return false;
    return std::unexpected(append_status);
  }
  for (std::size_t index = 0; index < context.provider_count(); ++index) {
    auto lengths = d::ProviderExtensionAccess::query(context, index, d::kBatchDeviceLengthsExtensionId);
    if (!lengths) {
      if (lengths.error().IsUnimplemented()) continue;
      return std::unexpected(lengths.error());
    }
    auto planning = d::ProviderExtensionAccess::query(*lengths, d::kBatchPlanningExtensionId);
    if (!planning) {
      if (planning.error().IsUnimplemented()) continue;
      return std::unexpected(planning.error());
    }
    struct NativeDescriptor final {
      const d::BatchProviderVTable *table = nullptr;
      void *value = nullptr;
      mem::AllocatorRef allocator;
      ~NativeDescriptor() { if (value) table->destroy(value, allocator); }
    } descriptor;
    descriptor.allocator = options.allocator;
    if (options.resource_budget.valid()) {
      const auto allocator = options.resource_budget.metadata_allocator();
      if (!allocator) return std::unexpected(allocator.error());
      descriptor.allocator = *allocator;
    }
    struct Qualification final {
      c::BatchCapacity capacity;
      const c::BatchProgramOptions &options;
      NativeDescriptor &descriptor;
    } qualification{capacity, options, descriptor};
    const auto status = d::ProviderExtensionAccess::call(*planning,
        [](void *native, const void *interface_value, void *opaque) noexcept -> Status {
          const auto *table = static_cast<const d::BatchProviderVTable *>(interface_value);
          if (!table || !table->qualify || !table->create || !table->destroy ||
              !table->update || !table->plan || !table->run ||
              (table->uses_caller_workspace && !table->workspace) ||
              !table->runtime || !table->supports_command_capture)
            return Status::Unimplemented("captured paged attention is unavailable");
          const auto runtime = table->runtime(native);
          if (!runtime.valid()) return Status::FailedPrecondition("batch provider runtime is invalid");
          if (!runtime.capture_compatible || runtime.mode != c::ProviderRuntimeMode::kSharedQueue ||
              runtime.workspace_owner != c::ProviderWorkspaceOwner::kProvider ||
              runtime.replay_class != c::ProviderReplayClass::kManaged ||
              runtime.blocks_host || runtime.uses_private_threads)
            return Status::Unimplemented("batch provider does not offer managed command capture");
          const auto &call = *static_cast<const Qualification *>(opaque);
          const auto qualified = table->qualify(native, call.capacity, call.options);
          if (!qualified.ok()) return qualified;
          call.descriptor.table = table;
          return table->create(native, call.capacity, call.options,
              call.descriptor.allocator, &call.descriptor.value);
        }, &qualification);
    if (!status.ok()) {
      if (status.IsUnimplemented()) continue;
      return std::unexpected(status);
    }
    if (!descriptor.value) return std::unexpected(Status::Internal("batch capability returned no descriptor"));
    struct Configuration final { void *native; c::Queue &queue; } configuration{descriptor.value, queue};
    const auto configured = d::ProviderExtensionAccess::call(*lengths,
        [](void *, const void *interface_value, void *opaque) noexcept -> Status {
          const auto *table = static_cast<const d::BatchDeviceLengthsVTable *>(interface_value);
          if (!table || !table->configure || !table->qualify || !table->update)
            return Status::Unimplemented("device-length configuration is unavailable");
          const auto &call = *static_cast<const Configuration *>(opaque);
          return table->configure(call.native, call.queue);
        }, &configuration);
    if (configured.ok()) return true;
    if (!configured.IsUnimplemented()) return std::unexpected(configured);
  }
  return false;
}

} // namespace hyle::inference::detail
