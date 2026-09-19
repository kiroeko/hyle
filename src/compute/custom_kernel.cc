#include "hyle/compute/custom_kernel.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "hyle/util/ref_counted.h"
#include "provider_internal.h"

namespace hyle::compute::detail {

struct CustomKernelAccess final {
  static ProviderPrepared release(CustomKernel &kernel) noexcept {
    const auto properties = kernel.properties_;
    return {.vtable = std::exchange(kernel.table_, nullptr),
            .state = std::exchange(kernel.state_, nullptr),
            .workspace = properties.workspace,
            .runtime = properties.runtime,
            .persistent_bytes = properties.persistent_bytes,
            .invocation_private_bytes = properties.invocation_private_bytes};
  }
};

namespace {

struct FactoryOwner final : util::RefCounted<FactoryOwner> {
  ProviderVTable table{};
  const CustomFactoryCallbacks *callbacks = nullptr;
  BackendId backend{};
  mem::AllocatorRef allocator{};
  std::size_t bytes = 0;
  std::size_t alignment = 0;
  std::size_t object_offset = 0;

  void *object() noexcept {
    return reinterpret_cast<std::byte *>(this) + object_offset;
  }
  void destroy() noexcept {
    callbacks->destroy(object());
    const auto owner = allocator;
    const auto allocation_size = bytes;
    const auto allocation_alignment = alignment;
    std::destroy_at(this);
    owner.deallocate(this, allocation_size, allocation_alignment);
  }
};

struct FactoryContext final {
  FactoryOwner *owner = nullptr; // retained by ContextState::ProviderSlot
  Device device{};
};

FactoryOwner *owner_of(const Provider &provider) noexcept {
  return ProviderAccess::payload<FactoryOwner *>(provider);
}

Status create_context(const Provider &provider, Device device, void *,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  auto *owner = owner_of(provider);
  if (device.backend_id() != owner->backend)
    return Status::Unimplemented(
        "custom kernel factory does not support this backend");
  void *storage =
      allocator.allocate(sizeof(FactoryContext), alignof(FactoryContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("custom kernel context allocation failed");
  *output = ::new (storage) FactoryContext{owner, device};
  return Status::Ok();
}

void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  std::destroy_at(static_cast<FactoryContext *>(state));
  allocator.deallocate(state, sizeof(FactoryContext), alignof(FactoryContext));
}

Status propose(void *state, const ProviderOperationRequest &request,
               std::span<ResourceBindingRequirement> inputs,
               std::span<ResourceBindingRequirement> outputs,
               ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.empty())
    return Status::Unimplemented(
        "custom kernel factory requires an exact shape");
  const auto &context = *static_cast<FactoryContext *>(state);
  const KernelRequest public_request{context.device, request.operation,
                                     request.inputs, request.outputs, request.workload};
  auto result = context.owner->callbacks->propose(
      context.owner->object(), public_request, inputs, outputs);
  if (!result)
    return std::move(result).error();
  if (!result->runtime.valid() ||
      result->runtime.mode != ProviderRuntimeMode::kSharedQueue ||
      result->runtime.workspace_owner != ProviderWorkspaceOwner::kCaller ||
      !result->cost.valid())
    return Status::Internal(
        "custom kernel factory returned invalid shared runtime facts");
  proposal->route = result->route;
  proposal->runtime = result->runtime;
  proposal->workspace_resource = result->workspace_resource;
  proposal->cost = result->cost;
  proposal->shape_coverage = ProviderShapeCoverage::kExactPoint;
  proposal->profile_key = result->profile_key;
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               ProviderPrepared *prepared) noexcept {
  const auto &context = *static_cast<FactoryContext *>(state);
  const KernelPrepareRequest public_request{
      .device = context.device,
      .operation = request.operation,
      .inputs = request.inputs,
      .outputs = request.outputs,
      .resources = {request.input_resources, request.output_resources,
                    request.workspace_resource},
      .route = request.route,
      .workload = request.workload,
  };
  auto result = context.owner->callbacks->prepare(context.owner->object(),
                                                  public_request, allocator);
  if (!result.outcome.valid())
    return ProviderPrepareOutcome::error(
        Status::Internal("custom kernel returned invalid prepare outcome"));
  if (result.outcome.kind == ProviderPrepareOutcomeKind::kPrepared) {
    if (!result.kernel)
      return ProviderPrepareOutcome::error(
          Status::Internal("custom kernel preparation returned no kernel"));
    *prepared = CustomKernelAccess::release(result.kernel);
  }
  // A partial kernel in any non-success result is destroyed before the
  // outcome can authorize another candidate. Never infer decline from code.
  return std::move(result.outcome);
}

} // namespace

Result<Provider>
make_custom_provider(ProviderIdentity identity, BackendId backend,
                     const CustomFactoryCallbacks *callbacks, std::size_t size,
                     std::size_t alignment, void *source,
                     void (*construct)(void *, void *) noexcept,
                     mem::AllocatorRef allocator) noexcept {
  if (!identity.valid() || !backend.valid() || size == 0 ||
      !std::has_single_bit(alignment) || source == nullptr ||
      construct == nullptr || callbacks == nullptr ||
      callbacks->destroy == nullptr || callbacks->propose == nullptr ||
      callbacks->prepare == nullptr)
    return std::unexpected(
        Status::InvalidArgument("invalid custom kernel factory"));

  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  if (sizeof(FactoryOwner) > maximum - (alignment - 1))
    return std::unexpected(
        Status::OutOfRange("custom kernel factory alignment overflows"));
  const std::size_t offset =
      (sizeof(FactoryOwner) + alignment - 1) & ~(alignment - 1);
  std::size_t bytes = offset;
  for (std::size_t extra :
       {size, identity.name.size(), identity.version.size()}) {
    if (extra > maximum - bytes)
      return std::unexpected(
          Status::OutOfRange("custom kernel factory size overflows"));
    bytes += extra;
  }
  if (!allocator.valid())
    allocator = mem::system_allocator();
  const auto allocation_alignment =
      (std::max)(alignment, alignof(FactoryOwner));
  void *storage = allocator.allocate(bytes, allocation_alignment);
  if (storage == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("custom kernel factory allocation failed"));
  auto *owner = ::new (storage) FactoryOwner;
  owner->allocator = allocator;
  owner->callbacks = callbacks;
  owner->backend = backend;
  owner->bytes = bytes;
  owner->alignment = allocation_alignment;
  owner->object_offset = offset;
  auto *name = reinterpret_cast<char *>(storage) + offset + size;
  auto *version = name + identity.name.size();
  if (!identity.name.empty())
    std::memcpy(name, identity.name.data(), identity.name.size());
  if (!identity.version.empty())
    std::memcpy(version, identity.version.data(), identity.version.size());
  owner->table = {
      .identity = {identity.id,
                   {name, identity.name.size()},
                   {version, identity.version.size()}},
      .create_context = &create_context,
      .destroy_context = &destroy_context,
      .propose = &propose,
      .prepare = &prepare,
      .drain_context = [](void *) noexcept { return Status::Ok(); },
      .retain_configuration =
          [](const Provider &provider) noexcept {
            owner_of(provider)->incref();
          },
      .release_configuration =
          [](const Provider &provider) noexcept {
            owner_of(provider)->decref();
          },
  };
  // Copy text before moving a factory that might itself own those strings.
  construct(owner->object(), source);
  return ProviderAccess::make(&owner->table, owner);
}

} // namespace hyle::compute::detail
