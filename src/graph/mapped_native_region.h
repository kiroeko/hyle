#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/container/small_vector.h"
#include "../compute/provider_internal.h"

#include <cstdint>
#include <new>
#include <span>

namespace hyle::graph::detail::mapped_native_region {
namespace c = compute;

// Cold-sized port mapping over the original SharedQueue carrier. Callers own
// their exact boundary and native qualification; submission never allocates.
inline constexpr std::size_t kInlinePorts = 4;
struct Artifact final {
  c::detail::ProviderPrepared native{};
  container::SmallVector<std::size_t, kInlinePorts, mem::AllocatorRef> ports;
  container::SmallVector<std::size_t, kInlinePorts, mem::AllocatorRef> output_ports;
  std::size_t boundary_count = 0;
  bool identity_outputs = true;
  explicit Artifact(mem::AllocatorRef allocator) noexcept
      : ports(allocator), output_ports(allocator) {}
};
struct Invocation final {
  const c::detail::ProviderPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  container::SmallVector<c::ConstTensorView, kInlinePorts, mem::AllocatorRef> ordered;
  container::SmallVector<c::TensorView, kInlinePorts, mem::AllocatorRef> destinations;
  explicit Invocation(mem::AllocatorRef allocator) noexcept
      : ordered(allocator), destinations(allocator) {}
};
inline Result<std::size_t> artifact_bytes(std::size_t inputs, std::size_t outputs) noexcept {
  // try_resize grows to the exact count; inline mappings are already in Artifact.
  const auto in = inputs > kInlinePorts ? inputs : 0;
  const auto out = outputs > kInlinePorts ? outputs : 0;
  if (in > SIZE_MAX - out || in + out > (SIZE_MAX - sizeof(Artifact)) / sizeof(std::size_t))
    return std::unexpected(Status::OutOfRange("mapped native region metadata bytes overflow"));
  return sizeof(Artifact) + (in + out) * sizeof(std::size_t);
}

inline void destroy(void *value, mem::AllocatorRef allocator) noexcept {
  auto *a = static_cast<Artifact *>(value);
  if (!a) return;
  if (a->native.vtable && a->native.vtable->destroy_prepared)
    a->native.vtable->destroy_prepared(a->native.state, allocator);
  a->~Artifact(); allocator.deallocate(a, sizeof(Artifact), alignof(Artifact));
}
inline c::AliasSupport alias(void *value, std::size_t output, std::size_t input) noexcept {
  auto &a = *static_cast<Artifact *>(value);
  std::size_t native_output = 0;
  while (native_output < a.output_ports.size() && a.output_ports[native_output] != output)
    ++native_output;
  if (native_output == a.output_ports.size()) return {};
  for (std::size_t n = 0; n < a.ports.size(); ++n)
    if (a.ports[n] == input) return a.native.vtable->alias_support(a.native.state, native_output, n);
  return {};
}
inline Status create_invocation(void *value, mem::AllocatorRef allocator,
                               mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  auto &a = *static_cast<Artifact *>(value);
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage) return Status::ResourceExhausted("mapped native region invocation allocation failed");
  auto *invocation = ::new (storage) Invocation(allocator);
  invocation->vtable = a.native.vtable;
  *output = invocation;
  if (!invocation->ordered.try_resize(a.ports.size()) ||
      (!a.identity_outputs && !invocation->destinations.try_resize(a.output_ports.size())))
    return Status::ResourceExhausted("mapped native region invocation ports allocation failed");
  return a.native.vtable->create_invocation(a.native.state, allocator, workspace, &invocation->state);
}
inline void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *i = static_cast<Invocation *>(value);
  if (!i) return;
  if (i->state) i->vtable->destroy_invocation(i->state, allocator);
  i->~Invocation(); allocator.deallocate(i, sizeof(Invocation), alignof(Invocation));
}
inline c::detail::EnqueueResult submit(void *value, void *lane, c::detail::QueueSequence &sequence,
    std::span<const c::ConstTensorView> inputs, std::span<const c::TensorView> outputs) noexcept {
  const auto &a = *static_cast<Artifact *>(value);
  if (inputs.size() != a.boundary_count || outputs.size() != a.output_ports.size())
    return c::detail::EnqueueResult::rejected(Status::InvalidArgument("mapped native submit port count differs"));
  auto &invocation = *static_cast<Invocation *>(lane);
  for (std::size_t n = 0; n < a.ports.size(); ++n) invocation.ordered[n] = inputs[a.ports[n]];
  if (a.identity_outputs)
    return a.native.vtable->shared_submit(a.native.state, static_cast<Invocation *>(lane)->state,
        sequence, std::span(invocation.ordered), outputs);
  for (std::size_t n = 0; n < a.output_ports.size(); ++n)
    invocation.destinations[n] = outputs[a.output_ports[n]];
  return a.native.vtable->shared_submit(a.native.state, static_cast<Invocation *>(lane)->state,
      sequence, std::span(invocation.ordered), std::span(invocation.destinations));
}
inline constexpr ProviderRegionPreparedVTable kPrepared{
    .destroy_prepared = &destroy, .alias_support = &alias, .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation, .shared_submit = &submit};

inline c::ProviderPrepareOutcome prepare(const c::detail::ProviderVTable &provider, void *context,
    const c::detail::ProviderPrepareRequest &request, const c::detail::ProviderOperationProposal &proposal,
    std::span<const std::size_t> ports, std::size_t boundary_count,
    mem::AllocatorRef allocator, ProviderRegionPrepared *result,
    std::span<const std::size_t> output_ports = {}) noexcept {
  *result = {};
  if (ports.empty() || ports.size() != request.inputs.size() ||
      request.input_resources.size() != ports.size() || request.outputs.empty() ||
      request.output_resources.size() != request.outputs.size() ||
      (!output_ports.empty() && output_ports.size() != request.outputs.size()) ||
      boundary_count == 0 || boundary_count > ports.size())
    return c::ProviderPrepareOutcome::error(Status::InvalidArgument("mapped native region port count differs"));
  for (const auto port : ports)
    if (port >= boundary_count)
      return c::ProviderPrepareOutcome::error(Status::InvalidArgument("mapped native region input is outside its boundary"));
  for (std::size_t n = 0; n < output_ports.size(); ++n) {
    if (output_ports[n] >= output_ports.size())
      return c::ProviderPrepareOutcome::error(Status::InvalidArgument("mapped native region output is outside its boundary"));
    for (std::size_t before = 0; before < n; ++before)
      if (output_ports[before] == output_ports[n])
        return c::ProviderPrepareOutcome::error(Status::InvalidArgument("mapped native region repeats an output"));
  }
  if (!proposal.cost.dispatch_count || !proposal.cost.copy_bytes || !proposal.cost.synchronization_count ||
      proposal.runtime.mode != c::ProviderRuntimeMode::kSharedQueue)
    return c::ProviderPrepareOutcome::error(Status::Internal("mapped native runtime facts are incomplete"));
  const auto bytes = artifact_bytes(ports.size(), request.outputs.size());
  if (!bytes) return c::ProviderPrepareOutcome::error(bytes.error());
  void *storage = allocator.allocate(sizeof(Artifact), alignof(Artifact));
  if (!storage) return c::ProviderPrepareOutcome::error(
      Status::ResourceExhausted("mapped native region artifact allocation failed"));
  auto *a = ::new (storage) Artifact(allocator);
  if (!a->ports.try_resize(ports.size()) || !a->output_ports.try_resize(request.outputs.size())) {
    destroy(a, allocator);
    return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("mapped native region port mapping allocation failed"));
  }
  a->boundary_count = boundary_count;
  for (std::size_t n = 0; n < a->ports.size(); ++n) a->ports[n] = ports[n];
  for (std::size_t n = 0; n < a->output_ports.size(); ++n) {
    a->output_ports[n] = output_ports.empty() ? n : output_ports[n];
    a->identity_outputs = a->identity_outputs && a->output_ports[n] == n;
  }
  const auto outcome = provider.prepare(context, request, allocator, &a->native);
  if (outcome.kind != c::ProviderPrepareOutcomeKind::kPrepared) {
    destroy(a, allocator); return outcome;
  }
  if (!a->native.vtable || !a->native.vtable->destroy_prepared || !a->native.vtable->alias_support ||
      !a->native.vtable->create_invocation || !a->native.vtable->destroy_invocation ||
      !a->native.vtable->shared_submit || a->native.vtable->prepare_bindings) {
    destroy(a, allocator);
    return c::ProviderPrepareOutcome::error(Status::Internal("mapped native artifact requires an unsupported binding route"));
  }
  auto persistent = a->native.persistent_bytes;
  if (persistent) {
    if (*persistent > SIZE_MAX - *bytes) {
      destroy(a, allocator);
      return c::ProviderPrepareOutcome::error(Status::OutOfRange("mapped native region metadata bytes overflow"));
    }
    *persistent += *bytes;
  }
  // Invocation is tracked by the supplied allocator, outside this opaque bound.
  *result = {.vtable = &kPrepared, .state = a,
      .properties = {.workspace = a->native.workspace, .persistent_bytes = persistent,
          .copy_bytes = *proposal.cost.copy_bytes, .synchronization_count = *proposal.cost.synchronization_count,
          .dispatch_count = *proposal.cost.dispatch_count, .runtime = a->native.runtime,
          .invocation_private_bytes = a->native.invocation_private_bytes}};
  return outcome;
}
} // namespace hyle::graph::detail::mapped_native_region
