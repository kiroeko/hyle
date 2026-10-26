#include "hyle/graph/executable.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/result.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/input_source.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/completion_retention.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/append_lease.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/spin_lock.h"
#include "plan_state.h"
#include "input_sources.h"
#include "resource_episode.h"
#include "submission_internal.h"
#include "submission_lifetime.h"
#include "../compute/provider_internal.h"

namespace hyle::graph {
namespace detail {
enum class BindingScope : std::uint8_t { kCaller, kConfiguration, kProjected, kWrittenProjection, kWriteEffect };

struct LeaseBinding final {
  ValueId value{};
  ValueId projection{};
  compute::TensorDesc description{};
  compute::ReadLease read{};
  compute::WriteLease write{};
  const compute::WriteLease *projected_writer=nullptr;
  const compute::detail::AppendLease *projected_appender=nullptr;
  std::uint32_t writer_step=UINT32_MAX;
  bool exclusive = false;
  bool input = false;
  bool permissionless = false;
  bool borrowed = true;
  BindingScope scope = BindingScope::kCaller;
  bool permission_changes_preserve_commands = false;
};

struct ResourceEpisode final {
  explicit ResourceEpisode(mem::AllocatorRef selected) noexcept
      : allocator(selected), bindings(mem::StdAllocator<LeaseBinding>(selected)),
        binding_slots(mem::StdAllocator<std::uint32_t>(selected)),
        input_lease_candidates(mem::StdAllocator<std::uint32_t>(selected)),
        unbind_steps(mem::StdAllocator<std::uint8_t>(selected)),
        unbind_instances(mem::StdAllocator<std::uint8_t>(selected)),
        nominated_writers(mem::StdAllocator<std::uint32_t>(selected)),
        permissions(selected) {}
  ~ResourceEpisode() {
    if (payload) payload_table->destroy(payload);
  }
  mem::AllocatorRef allocator;
  std::vector<LeaseBinding, mem::StdAllocator<LeaseBinding>> bindings;
  // The external ValueId records form an immutable prefix. Only anonymous
  // projected-page guards are appended/erased after it. Slots index identity,
  // never permissions, leases, addresses or readiness from an old episode.
  std::vector<std::uint32_t, mem::StdAllocator<std::uint32_t>> binding_slots;
  // Call-scoped ordinals into bindings, rebuilt under preparation_uses. The
  // original lookup still checks every selected lease and actual range live.
  std::vector<std::uint32_t, mem::StdAllocator<std::uint32_t>> input_lease_candidates;
  // Fixed cold scratch; selective withdrawal never grows metadata or changes
  // InvocationState/PreparedRegion layout in other modules.
  std::vector<std::uint8_t, mem::StdAllocator<std::uint8_t>> unbind_steps;
  std::vector<std::uint8_t, mem::StdAllocator<std::uint8_t>> unbind_instances;
  // One bound original external-write NodeId ordinal per selected region.
  // Permissions remain solely in bindings; this identity lets retired release
  // reject a pure sibling without retaining the whole source Graph or scanning
  // every permission once per nominated node.
  std::vector<std::uint32_t, mem::StdAllocator<std::uint32_t>> nominated_writers;
  std::size_t fixed_binding_count = 0;
  // Publication stays lazy at the first successful result(). Later ports in
  // the same episode need not rescan writers already converted to reads.
  bool writer_publication_complete = false;
  LeasePermissionSet permissions;
  compute::LeaseUsePool uses;
  compute::LeaseUseBatch batch;
  // Actual shared capacity of bindings, permissions and LeaseUsePool, rather
  // than the last caller's extra count or vector's geometric over-allocation.
  std::uint32_t permission_capacity = 0;
  bool configuration_spans = false;
  const ResourcePayloadVTable *payload_table = nullptr;
  void *payload = nullptr;
};

struct SubmissionLifetime final {
  explicit SubmissionLifetime(mem::AllocatorRef selected) noexcept
      : allocator(selected),
        retentions(mem::StdAllocator<compute::detail::CompletionRetention>(selected)) {}

  mem::AllocatorRef allocator;
  std::vector<compute::detail::CompletionRetention,
              mem::StdAllocator<compute::detail::CompletionRetention>> retentions;
  compute::detail::CompletionLifetime observation_retirement{};
  util::SpinLock lock;
  Status first_error;
  std::atomic<std::size_t> remaining{0};
  bool accepted = false;
  OrderedGraphSubmission ordered{};
};

SubmissionLifetimeOwner::SubmissionLifetimeOwner(SubmissionLifetimeOwner &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
SubmissionLifetimeOwner &SubmissionLifetimeOwner::operator=(SubmissionLifetimeOwner &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
SubmissionLifetimeOwner::~SubmissionLifetimeOwner() { reset(); }
void SubmissionLifetimeOwner::reset() noexcept {
  if (auto *lifetime = std::exchange(state_, nullptr)) {
    const auto allocator = lifetime->allocator;
    allocator.destroy(lifetime);
  }
}

ResourceEpisodeOwner::ResourceEpisodeOwner(ResourceEpisodeOwner &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
ResourceEpisodeOwner &ResourceEpisodeOwner::operator=(ResourceEpisodeOwner &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
ResourceEpisodeOwner::~ResourceEpisodeOwner() { reset(); }
void ResourceEpisodeOwner::reset() noexcept {
  if (auto *episode = std::exchange(state_, nullptr)) {
    const auto allocator = episode->allocator;
    allocator.destroy(episode);
  }
}

Status resource_control_ready(const InvocationState *state) noexcept {
  if (!state) return Status::FailedPrecondition("graph invocation is empty");
  if (state->closed) return Status::FailedPrecondition("graph invocation is closed");
  const auto phase = state->phase.load(std::memory_order_acquire);
  if (phase == InvocationPhase::kRunning)
    return Status::Unavailable("graph invocation is in flight");
  if (phase == InvocationPhase::kFailed)
    return Status::FailedPrecondition("failed graph invocation must be recreated");
  if (state->preparation_uses.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("graph invocation bindings are being prepared");
  if (state->results.load(std::memory_order_acquire))
    return Status::Unavailable("graph results still retain this invocation");
  if (!state->observation.available())
    return Status::Unavailable("old graph completion retains its observation slot");
  return Status::Ok();
}

Status reset_resource_payload(Invocation &invocation) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (!state->resources || !state->resources.get()->payload) return Status::Ok();
  auto &episode = *state->resources.get();
  return episode.payload_table->reset(episode.payload, invocation);
}

Result<void *> RegionAccess::resource_payload(Invocation &invocation,
    const ResourcePayloadVTable *table) noexcept {
  auto *state = InvocationAccess::state(invocation);
  if (!state)
    return std::unexpected(Status::FailedPrecondition("graph invocation is empty"));
  if (!table || !table->reset || !table->destroy)
    return std::unexpected(Status::InvalidArgument("resource payload implementation is incomplete"));
  if (!state->resources || !state->resources.get()->payload) return nullptr;
  auto &episode = *state->resources.get();
  if (episode.payload_table != table)
    return std::unexpected(Status::InvalidArgument("resource payload belongs to another implementation"));
  return episode.payload;
}

Result<ResourceEpisode *> require_resources(InvocationState *state) noexcept {
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (!state->resources)
    return std::unexpected(Status::FailedPrecondition(
        "owned bindings require prepare_resources or factory.resources"));
  return state->resources.get();
}
Status RegionAccess::attach_resource_payload(Invocation &invocation,
    const ResourcePayloadVTable *table, void *payload) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_ASSIGN_OR_RETURN(auto *episode, require_resources(state));
  if (!table || !table->reset || !table->destroy || !payload)
    return Status::InvalidArgument("resource payload requires an implementation and owner");
  if (episode->payload)
    return Status::FailedPrecondition("graph invocation already owns a resource payload");
  episode->payload_table = table;
  episode->payload = payload;
  return Status::Ok();
}
const LeaseBinding *find_binding(const InvocationState *state, ValueId value) noexcept {
  if (!state || !state->resources || !value.valid()) return nullptr;
  const auto &episode = *state->resources.get();
  if (value.ordinal() >= episode.binding_slots.size()) return nullptr;
  const auto slot = episode.binding_slots[value.ordinal()];
  if (slot >= episode.fixed_binding_count) return nullptr;
  const auto &binding = episode.bindings[slot];
  // Equal ordinals from a different Graph/generation never name this record.
  return binding.value == value ? &binding : nullptr;
}
LeaseBinding *find_binding(InvocationState *state, ValueId value) noexcept {
  return const_cast<LeaseBinding *>(find_binding(
      static_cast<const InvocationState *>(state), value));
}
Status check_binding_replacement(const InvocationState &state, ValueId value) noexcept {
  const auto *binding = find_binding(&state, value);
  if (binding && (binding->scope == BindingScope::kConfiguration ||
                  binding->scope == BindingScope::kWrittenProjection))
    return Status::FailedPrecondition(
        "graph input belongs to its cold configuration; reset before ordinary rebinding");
  return Status::Ok();
}
void invalidate_native_input_bindings(InvocationState &state,
                                      ValueId value = {}) noexcept {
  const auto &plan = *state.plan;
  const auto origin = [&](std::uint32_t input) noexcept {
    while (plan.alias_sources[input] != kNoInPlaceSource)
      input = plan.alias_sources[input];
    return input;
  };
  const auto root = value.valid()
      ? origin(plan.external_instances[value.ordinal()]) : kNoWorkspaceRequest;
  for (std::size_t index = 0; index < state.step_bindings.size(); ++index) {
    if (!state.step_bindings[index].native_inputs)
      continue;
    const auto &step = plan.steps[index];
    bool affected = !value.valid();
    for (std::size_t port = 0; !affected && port < step.input_count; ++port)
      affected = origin(plan.node_inputs[step.input_begin + port]) == root;
    if (affected)
      invalidate_step_recording(state, static_cast<std::uint32_t>(index));
  }
}
void clear_binding_owner(LeaseBinding &binding) noexcept {
  binding.read.reset(); binding.write.reset(); binding.description = {};
  binding.projection = {};
  binding.projected_writer=nullptr;binding.projected_appender=nullptr;binding.writer_step=UINT32_MAX;
  binding.permission_changes_preserve_commands=false;
  binding.exclusive = false; binding.permissionless = false; binding.borrowed = true;
  binding.scope = BindingScope::kCaller;
}
bool RegionAccess::borrows_tensor_views_only(const PreparedRegion &region) noexcept {
  if (!region.prepares_bindings()) return true;
  const auto &prepared = region.prepared_;
  const auto *table = prepared.vtable->query_extension
      ? static_cast<const RegionBindingReleaseVTable *>(prepared.vtable->query_extension(
          prepared.state, kRegionBindingReleaseExtensionId)) : nullptr;
  return table && table->borrows_tensor_views_only;
}
Status RegionAccess::withdraw_bindings(Invocation &invocation,
    std::span<const ValueId> values) noexcept {
  return withdraw_bindings_impl(invocation, values, false);
}
Status RegionAccess::release_read_projection(Invocation &invocation,
    ValueId value) noexcept {
  return withdraw_bindings_impl(invocation, std::span(&value, 1), true);
}
std::span<const ExternalInputConsumer>
projected_input_consumers(const PlanState &, std::uint32_t) noexcept;
Status RegionAccess::withdraw_bindings_impl(Invocation &invocation,
    std::span<const ValueId> values, bool ready_projection) noexcept {
  auto *state = InvocationAccess::state(invocation);
  if (ready_projection) {
    if (!outcome_available(invocation) ||
        state->preparation_uses.load(std::memory_order_acquire) != 0)
      return Status::Unavailable("read projection still has native work/results/observations/preparation");
  } else {
    HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  }
  if (values.empty()) return Status::Ok();
  if (!state->resources)
    return Status::FailedPrecondition("unbind requires its original resource episode");
  auto *episode = state->resources.get();
  const auto &plan = *state->plan;
  // Untaken original write/pending page owners are not ordinary caller ports.
  // Reject without asking the payload to reset or returning any source control.
  for (const auto &binding : episode->bindings) {
    if (ready_projection) break;
    if (binding.scope == BindingScope::kWrittenProjection ||
        (binding.scope == BindingScope::kProjected && !binding.read.valid() &&
         !binding.permissionless))
      return Status::FailedPrecondition("take original page/source controls before unbinding caller ports");
  }
  for (auto value : values) {
    const auto *binding = find_binding(state, value);
    if (!binding) return Status::InvalidArgument("unbind needs this executable's caller input or output");
    if (ready_projection) {
      if (!binding->input || binding->scope != BindingScope::kProjected || binding->borrowed ||
          (!binding->read.valid() && !binding->permissionless) ||
          binding->projected_writer || binding->projected_appender || has_input_source(*state, value))
        return Status::InvalidArgument("input has no original ready read projection");
      if (projected_input_consumers(plan, value.ordinal()).empty())
        return Status::InvalidArgument("read projection has no original physical consumer");
    } else if (binding->scope != BindingScope::kCaller || binding->projection.valid() ||
        binding->projected_writer || binding->projected_appender || has_input_source(*state, value)) {
      return Status::FailedPrecondition("unbind cannot withdraw a configuration or source-owned port");
    }
  }
  auto &steps = episode->unbind_steps;
  auto &instances = episode->unbind_instances;
  HYLE_FATAL_ASSERT(steps.size() == plan.steps.size() && instances.size() == state->readable.size());
  std::fill(steps.begin(), steps.end(), std::uint8_t{});
  std::fill(instances.begin(), instances.end(), std::uint8_t{});
  const auto root = [&](std::uint32_t instance) noexcept {
    while (plan.alias_sources[instance] != kNoInPlaceSource)
      instance = plan.alias_sources[instance];
    return instance;
  };
  for (auto value : values) instances[root(plan.external_instances[value.ordinal()])] = 1;
  constexpr std::uint8_t release_native = 1, discard_recording = 2;
  const bool idle = !state->closed &&
      state->phase.load(std::memory_order_acquire) == InvocationPhase::kIdle;
  if (ready_projection) {
    const auto *binding = find_binding(state, values.front());
    for (const auto &consumer : projected_input_consumers(plan, values.front().ordinal())) {
      if (!idle || !binding->permission_changes_preserve_commands ||
          !state->step_bindings[consumer.step].stable_binding_identity)
        steps[consumer.step] |= discard_recording;
    }
  }
  // Only native binding owners propagate through generated output carriers.
  // Materialized caller-buffer results do not retain their mathematical inputs.
  bool changed;
  do {
    changed = false;
    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
      const auto &step = plan.steps[index];
      if ((steps[index] & release_native) || !step.region.valid() || !step.region.prepares_bindings()) continue;
      bool affected = false;
      for (std::size_t port = 0; port < step.input_count; ++port)
        affected |= instances[root(plan.node_inputs[step.input_begin + port])] != 0;
      for (std::size_t port = 0; port < step.output_count; ++port)
        affected |= instances[root(plan.node_outputs[step.output_begin + port])] != 0;
      if (!affected) continue;
      steps[index] |= release_native;
      changed = true;
      for (std::size_t port = 0; port < step.output_count; ++port)
        if (step.region.output_kind(port) != RegionOutputKind::kCaller)
          instances[root(plan.node_outputs[step.output_begin + port])] = 1;
    }
  } while (changed);
  const auto release_table = [&](std::size_t index) noexcept {
    const auto &prepared = plan.steps[index].region.prepared_;
    return prepared.vtable->query_extension
        ? static_cast<const RegionBindingReleaseVTable *>(prepared.vtable->query_extension(
            prepared.state, kRegionBindingReleaseExtensionId)) : nullptr;
  };
  // Every fallible native check precedes all recording, binding and owner changes.
  for (std::size_t index = 0; index < steps.size(); ++index) {
    if (!(steps[index] & release_native)) continue;
    const auto *table = release_table(index);
    if (!table || !table->validate || !table->release)
      return Status::Unimplemented("selected native region has no binding-release contract");
    HYLE_RETURN_IF_ERROR(table->validate(plan.steps[index].region.prepared_.state,
        state->invocations[index].region.state_));
  }
  // Failed/closed episodes cannot pass the ordinary admission gate. They are
  // already actually retired; invalidate their affected recordings directly,
  // which also revokes binding/replay readiness without reopening admission.
  if (idle) HYLE_RETURN_IF_ERROR(revoke_preparation(invocation));
  for (std::size_t index = 0; index < steps.size(); ++index)
    if ((steps[index] & discard_recording) ||
        ((steps[index] & release_native) &&
         (!idle || !release_table(index)->borrows_tensor_views_only ||
          !state->step_bindings[index].stable_binding_identity)))
      invalidate_step_recording(*state, static_cast<std::uint32_t>(index));
  // The selected physical plan emits producer-before-consumer steps. Release
  // native borrowed references in reverse order before their upstream owners.
  for (std::size_t index = steps.size(); index-- != 0;) {
    if (!(steps[index] & release_native)) continue;
    release_table(index)->release(plan.steps[index].region.prepared_.state,
        state->invocations[index].region.state_);
    state->step_bindings[index].native_inputs = false;
    for (std::size_t port = 0; port < plan.steps[index].output_count; ++port) {
      const auto kind = plan.steps[index].region.output_kind(port);
      if (kind != RegionOutputKind::kBoundGenerated && kind != RegionOutputKind::kDeferredGenerated) continue;
      const auto output = plan.node_outputs[plan.steps[index].output_begin + port];
      state->native_values[output] = {};
      state->readable[output] = {};
      state->writable[output] = {};
    }
  }
  episode->input_lease_candidates.clear();
  episode->permissions.reset();
  episode->writer_publication_complete = false;
  if (ready_projection) {
    const auto value = values.front();
    const auto physical = root(plan.external_instances[value.ordinal()]);
    // bind_read_projection may expose declared transparent aliases. Clear all
    // of that original root's concrete views before releasing any ready lease;
    // a future cold prepare must regenerate them from a newly bound input.
    const bool aliases = std::binary_search(plan.external_input_aliases.begin(),
        plan.external_input_aliases.end(), value.ordinal());
    if (aliases) {
      for (std::size_t i = 0; i < plan.alias_sources.size(); ++i) {
        if (root(static_cast<std::uint32_t>(i)) != physical) continue;
        state->readable[i] = {}; state->writable[i] = {}; state->bound[i] = 0;
      }
    } else {
      state->readable[physical] = {}; state->writable[physical] = {}; state->bound[physical] = 0;
    }
    const auto extras = episode->bindings.begin() + episode->fixed_binding_count;
    episode->bindings.erase(std::remove_if(extras, episode->bindings.end(),
        [&](const auto &binding) { return binding.projection == value; }), episode->bindings.end());
  }
  for (auto value : values) {
    const auto physical = plan.external_instances[value.ordinal()];
    state->readable[physical] = {};
    state->writable[physical] = {};
    state->bound[physical] = 0;
    clear_binding_owner(*find_binding(state, value));
  }
  state->external_alias_dirty = true;
  reset_input_source_preparation(*state);
  return Status::Ok();
}
void reset_resource_bindings(InvocationState &state) noexcept {
  // Native imports can own carriers/arrays independently of their raw address.
  // Release their recording before withdrawing the old source or lease owner.
  invalidate_native_input_bindings(state);
  reset_input_source_bindings(state);
  if (!state.resources) return;
  auto &episode = *state.resources.get();
  episode.input_lease_candidates.clear();
  episode.writer_publication_complete = false;
  // All extra records are anonymous page guards; fixed ValueId slots never
  // move. Shrinking releases exactly the old projected permission suffix.
  episode.bindings.resize(episode.fixed_binding_count);
  std::fill(episode.nominated_writers.begin(), episode.nominated_writers.end(), UINT32_MAX);
  for (auto &binding : episode.bindings) clear_binding_owner(binding);
}
std::span<const ExternalInputConsumer>
projected_input_consumers(const PlanState &plan, std::uint32_t value) noexcept {
  const auto &consumers = plan.external_input_consumers;
  if (consumers.empty()) return {};
  const auto first = std::lower_bound(consumers.begin(), consumers.end(), value,
      [](const auto &consumer, std::uint32_t logical) noexcept {
        return consumer.value < logical;
      });
  const auto last = std::upper_bound(first, consumers.end(), value,
      [](std::uint32_t logical, const auto &consumer) noexcept {
        return logical < consumer.value;
      });
  return {first, static_cast<std::size_t>(last - first)};
}
void remember_configuration_spans(InvocationState &state) noexcept {
  if (state.resources)
    state.resources.get()->configuration_spans = state.bindings_prepared;
}
bool reusable_configuration_spans(const InvocationState &state) noexcept {
  return state.resources && state.resources.get()->configuration_spans;
}
void discard_configuration_spans(InvocationState &state) noexcept {
  if (state.resources) state.resources.get()->configuration_spans = false;
}
void reset_configuration_bindings(InvocationState &state) noexcept {
  if (state.resources) {
    auto &episode = *state.resources.get();
    bool reusable = episode.configuration_spans && !state.input_sources;
    for (const auto &binding : state.step_bindings)
      reusable = reusable && !binding.native_inputs;
    for (const auto &binding : episode.bindings) {
      if (binding.scope == BindingScope::kProjected || binding.scope == BindingScope::kWriteEffect ||
          binding.projection.valid()) reusable = false;
      if (binding.scope == BindingScope::kConfiguration &&
          std::binary_search(state.plan->external_input_aliases.begin(),
              state.plan->external_input_aliases.end(), binding.value.ordinal()))
        reusable = false;
    }
    // Drop the covering recordings before any configuration lease or anonymous
    // page guard. Caller bindings and their native import owners survive.
    const auto &plan = *state.plan;
    for (std::size_t index = 0; index < state.step_bindings.size(); ++index) {
      if (!state.step_bindings[index].native_inputs) continue;
      const auto &step = plan.steps[index];
      for (std::size_t port = 0; port < step.input_count; ++port) {
        auto input = plan.node_inputs[step.input_begin + port];
        while (plan.alias_sources[input] != kNoInPlaceSource)
          input = plan.alias_sources[input];
        const auto logical = plan.instance_values[input];
        if (logical >= episode.binding_slots.size()) continue;
        const auto slot = episode.binding_slots[logical];
        if (slot >= episode.fixed_binding_count ||
            episode.bindings[slot].scope == BindingScope::kCaller) continue;
        invalidate_step_recording(state, static_cast<std::uint32_t>(index));
        break;
      }
    }
    episode.input_lease_candidates.clear();
    for (std::size_t i = 0; i < episode.fixed_binding_count; ++i) {
      auto &binding = episode.bindings[i];
      if (binding.scope == BindingScope::kCaller) continue;
      const auto instance = state.plan->external_instances[binding.value.ordinal()];
      state.readable[instance] = {};
      state.writable[instance] = {};
      state.bound[instance] = 0;
      clear_binding_owner(binding);
    }
    episode.bindings.resize(episode.fixed_binding_count);
    std::fill(episode.nominated_writers.begin(), episode.nominated_writers.end(), UINT32_MAX);
    episode.configuration_spans = reusable;
  }
  // Sources remain bound; dependency routes and readiness belong to the old
  // preparation and must be derived again for the next layout episode.
  reset_input_source_preparation(state);
}
void note_borrowed_binding(InvocationState &state, ValueId value) noexcept {
  const auto *previous = find_binding(&state, value);
  const auto previous_writer=previous?previous->writer_step:UINT32_MAX;
  if (state.input_sources || (previous &&
      (previous->read.valid() || previous->write.valid() ||
       previous->scope != BindingScope::kCaller)))
    invalidate_native_input_bindings(state, value);
  note_borrowed_input_source(state, value);
  if (!state.resources) return;
  auto &episode = *state.resources.get();
  if (previous_writer < episode.nominated_writers.size())
    episode.nominated_writers[previous_writer] = UINT32_MAX;
  auto &bindings = episode.bindings;
  const auto extras = bindings.begin() + episode.fixed_binding_count;
  bindings.erase(std::remove_if(extras, bindings.end(), [&](const auto &binding) {
    return binding.projection == value;
  }), bindings.end());
  if (auto *binding = find_binding(&state, value)) {
    binding->read.reset(); binding->write.reset(); binding->description = {};
    binding->projection = {};
    binding->permission_changes_preserve_commands = false;
    binding->exclusive = false; binding->permissionless = false; binding->borrowed = true;
    binding->scope = BindingScope::kCaller;
  }
}

Status RegionAccess::bind_configuration(Invocation &invocation, ValueId value,
    compute::TensorDesc description, compute::ReadLease lease) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_ASSIGN_OR_RETURN(auto *episode, require_resources(state));
  (void)episode;
  auto *binding = find_binding(state, value);
  if (!binding || !binding->input)
    return Status::InvalidArgument("cold configuration must bind a graph input");
  HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
  HYLE_ASSIGN_OR_RETURN(auto tensor,
      compute::ConstTensorView::from_initialized_storage(description, storage));
  bool same_spans = episode->configuration_spans;
  if (same_spans) {
    bool consumed = false;
    const auto physical = state->plan->external_instances[value.ordinal()];
    for (const auto &consumer : projected_input_consumers(*state->plan, value.ordinal())) {
      // Other-storage replicas retain the original complete path. Their input
      // spans name workspace rather than this external configuration tensor.
      if (consumer.instance != physical) { same_spans = false; break; }
      const auto &step = state->plan->steps[consumer.step];
      const auto &prior = state->submission_inputs[step.input_begin + consumer.port];
      const auto before = prior.storage(), after = tensor.storage();
      same_spans = same_spans && prior.valid() && prior.description() == tensor.description() &&
          before.resource() == after.resource() && before.native_handle() == after.native_handle() &&
          before.offset() == after.offset() && before.size() == after.size();
      consumed = true;
    }
    same_spans = same_spans && consumed &&
        !std::binary_search(state->plan->external_input_aliases.begin(),
            state->plan->external_input_aliases.end(), value.ordinal());
  }
  // Reuse the original binding validation/commit. Only this synchronized cold
  // call suspends the replacement guard; a rejecting bind has not invalidated
  // replay or released the old resource owner.
  const auto previous = std::exchange(binding->scope, BindingScope::kCaller);
  const auto status = invocation.bind(value, tensor);
  binding = find_binding(state, value);
  if (!status.ok()) {
    binding->scope = previous;
    return status;
  }
  binding->read = std::move(lease); binding->description = description;
  binding->exclusive = false; binding->permissionless = false; binding->borrowed = false;
  binding->scope = BindingScope::kConfiguration;
  // The original bind above revoked replay and validated the actual alias,
  // resource and description. Only its previously eligible physical facts
  // survive a same-span configuration replacement; no owner is cached here.
  episode->configuration_spans = same_spans;
  return Status::Ok();
}

Status RegionAccess::reserve_reads(Invocation &invocation, std::uint32_t extra,
                                 ResourceOptions options) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_ASSIGN_OR_RETURN(auto *episode, require_resources(state));
  // Repeated layout preparation reuses the same fixed permission/use storage.
  // Growth is cold and idle-gated; existing leases do not refer to this pool.
  // Keep the old pool until every replacement allocation has succeeded.
  if (extra > UINT32_MAX - 1 || episode->bindings.size() > UINT32_MAX - 1 - extra)
    return Status::ResourceExhausted("graph permission capacity overflow");
  const auto required = static_cast<std::uint32_t>(episode->bindings.size() + extra);
  if (required <= episode->permission_capacity) {
    try { episode->input_lease_candidates.reserve(episode->bindings.capacity()); }
    catch (...) { return Status::ResourceExhausted("graph fixed read lookup metadata"); }
    return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = required + 1,
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  try {
    episode->bindings.reserve(required);
    episode->input_lease_candidates.reserve(episode->bindings.capacity());
  }
  catch (...) { return Status::ResourceExhausted("graph fixed permission metadata"); }
  HYLE_RETURN_IF_ERROR(episode->permissions.reserve(required));
  episode->uses = std::move(uses);
  episode->permission_capacity = required;
  return Status::Ok();
}
Status RegionAccess::bind_projected_impl(Invocation&invocation,ValueId value,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease> pages,
    std::span<const ProjectedReader> readers,
    bool pending)noexcept{
  auto*state=InvocationAccess::state(invocation);
  HYLE_ASSIGN_OR_RETURN(auto*episode, require_resources(state));
  auto*binding=find_binding(state,value);
  if(!binding || !binding->input || !tensor.valid() || readers.empty())
    return Status::InvalidArgument("projected graph requires an input and actual page permissions");
  const bool replacing = !pending && !binding->borrowed && binding->projection == value;
  if(!binding->borrowed && !replacing)
    return Status::FailedPrecondition("projected graph input needs an explicit reset before rebinding");
  const auto&plan=*state->plan;
  const auto logical=value.ordinal();const auto physical=plan.external_instances[logical];
  if((plan.binding_roles[logical]&kExternalOutput)!=0)
    return Status::InvalidArgument("projected capacity cannot be a graph output");
  if(!plan.external_input_aliases.empty() && std::binary_search(plan.external_input_aliases.begin(),
      plan.external_input_aliases.end(),logical))
    return Status::InvalidArgument("projected capacity cannot have a physical alias descendant");
  for(std::size_t i=0;i<readers.size();++i){
    const auto&proof=readers[i];
    if(proof.step>=plan.steps.size())
      return Status::InvalidArgument("projected reader step is foreign");
    const auto&step=plan.steps[proof.step];
    if(step.kind!=PlanStepKind::kRegion || proof.input_port>=step.input_count ||
       !step.region.matches_artifact(proof.implementation,proof.prepared) ||
       plan.node_inputs[step.input_begin+proof.input_port]!=physical)
      return Status::InvalidArgument("projected reader differs from its selected artifact/port");
    for(std::size_t j=0;j<i;++j)
      if(readers[j].step==proof.step && readers[j].input_port==proof.input_port)
        return Status::InvalidArgument("projected reader physical port is duplicated");
    if(!pending && pages.empty()){
      const auto&prepared=step.region.prepared_;
      const auto*interface=prepared.vtable->query_extension ?
          static_cast<const RegionProjectedReadVTable*>(prepared.vtable->query_extension(
              prepared.state,kRegionProjectedReadExtensionId)) : nullptr;
      if(!interface || !interface->inspect)
        return Status::Unimplemented("selected reader cannot prove an empty input footprint");
      compute::detail::OperationPreparedAccess access;
      HYLE_RETURN_IF_ERROR(interface->inspect(prepared,
          state->invocations[proof.step].region.state_,proof.input_port,&access));
      if(access.projected_input!=proof.input_port || !access.allows_empty_footprint)
        return Status::InvalidArgument("selected reader still requires actual input permissions");
    }
  }
  const auto consumers=projected_input_consumers(plan,logical);
  for(const auto&consumer:consumers){
    const auto&step=plan.steps[consumer.step];
    if(consumer.instance!=physical || step.kind!=PlanStepKind::kRegion ||
       std::none_of(readers.begin(),readers.end(),[&](const auto&proof)noexcept{
         return proof.step==consumer.step && proof.input_port==consumer.port;
       }))
      return Status::InvalidArgument("projected capacity has an unqualified physical reader");
  }
  if(consumers.empty())return Status::InvalidArgument("projected capacity has no qualified reader");
  const auto old_extra = std::count_if(episode->bindings.begin(), episode->bindings.end(),
      [&](const auto &entry) { return !entry.value.valid() && entry.projection == value; });
  if(!pages.empty() && pages.size()-1>episode->bindings.capacity()-(episode->bindings.size()-old_extra))
    return Status::ResourceExhausted("projected graph permission capacity exhausted");
  const auto storage=tensor.storage();
  for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() ||
       view.offset()<storage.offset() || view.offset()-storage.offset()>storage.size() ||
       view.size()>storage.size()-(view.offset()-storage.offset()))
      return Status::InvalidArgument("projected page permission is outside its native capacity");
  }
  HYLE_RETURN_IF_ERROR(invocation.bind(value,tensor));
  // The ordinary bind has now invalidated replay and removed the old page
  // permissions. Every rejecting check preceded it; this commit cannot grow
  // the fixed permission table or disturb an old binding on failure.
  binding=find_binding(state,value);
  if(!pages.empty())binding->read=pages.front();
  binding->description=tensor.description();binding->exclusive=false;binding->borrowed=false;
  binding->permissionless=pages.empty()&&!pending;
  binding->projection=pending?ValueId{}:value;
  binding->scope=BindingScope::kProjected;
  binding->permission_changes_preserve_commands=std::all_of(readers.begin(),readers.end(),
      [](const auto&reader){return reader.permission_changes_preserve_commands;});
  for(std::size_t i=1;i<pages.size();++i)episode->bindings.push_back({.projection=value,.read=pages[i],.borrowed=false,.scope=BindingScope::kProjected});
  return Status::Ok();
}
Status RegionAccess::bind_projected(Invocation&invocation,ValueId value,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease> pages,
    std::span<const ProjectedReader> readers)noexcept{
  return bind_projected_impl(invocation,value,tensor,pages,readers,false);
}
Status RegionAccess::bind_pending_projected(Invocation&invocation,ValueId value,
    compute::ConstTensorView tensor,std::span<const ProjectedReader> readers)noexcept{
  return bind_projected_impl(invocation,value,tensor,{},readers,true);
}
Result<bool> RegionAccess::has_input_binding(const InvocationState*state,
    ValueId value,compute::ConstTensorView expected)noexcept{
  if(!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("input-binding parent invocation is empty"));
  const auto&plan=*state->plan;
  if(!value.valid() || GraphIdentity::generation(value)!=plan.generation ||
      value.kind()!=ValueKind::kInput || value.ordinal()>=plan.binding_roles.size() ||
      !(plan.binding_roles[value.ordinal()]&kExternalInput) || !expected.valid())
    return std::unexpected(Status::InvalidArgument("input-binding query needs its original input and tensor"));
  const auto physical=plan.external_instances[value.ordinal()];
  if(physical==kNoWorkspaceRequest || physical>=state->bound.size() ||
      physical>=state->readable.size())
    return std::unexpected(Status::Internal("original input has no physical binding"));
  if(!state->bound[physical] || !state->readable[physical].valid())return false;
  const auto actual=state->readable[physical];
  const auto left=actual.storage(),right=expected.storage();
  return actual.description()==expected.description() && left.resource()==right.resource() &&
      left.native_handle()==right.native_handle() && left.offset()==right.offset() && left.size()==right.size();
}
Result<bool> RegionAccess::has_read_projection(const InvocationState*state,
    ValueId value,std::span<const compute::ReadLease>pages)noexcept{
  if(!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("read-projection parent invocation is empty"));
  const auto&plan=*state->plan;
  if(GraphIdentity::generation(value)!=plan.generation || value.kind()!=ValueKind::kInput ||
      value.ordinal()>=plan.binding_roles.size() || !(plan.binding_roles[value.ordinal()]&kExternalInput) || pages.empty())
    return std::unexpected(Status::InvalidArgument("read-projection query needs its original input and ready footprint"));
  if(!state->resources)return false;
  const auto*binding=find_binding(state,value);
  if(!binding || !binding->input)
    return std::unexpected(Status::InvalidArgument("read projection is not an original resource input"));
  if(binding->scope!=BindingScope::kProjected){
    if(binding->scope==BindingScope::kCaller && binding->borrowed &&
        !binding->read.valid() && !binding->write.valid() && !binding->permissionless)return false;
    return std::unexpected(Status::InvalidArgument("read-projection query differs from its current binding scope"));
  }
  // The general ReadProjection entry keeps its root by value, while the
  // direct ready entry also tags projection=value. Both retain real guards;
  // a pending/empty footprint has no ready root and cannot qualify here.
  if((binding->projection.valid() && binding->projection!=value) ||
      !binding->read.valid() || binding->exclusive || binding->permissionless)
    return std::unexpected(Status::InvalidArgument("read-projection query differs from its current ready footprint"));
  const auto match=[](const compute::ReadLease&bound,const compute::ReadLease&expected)noexcept->Status{
    if(!bound.valid() || !expected.valid() ||
        compute::detail::LeaseAccess::identity(bound)!=compute::detail::LeaseAccess::identity(expected) ||
        bound.generation()!=expected.generation())
      return Status::InvalidArgument("read-projection guard differs from its original owner");
    HYLE_ASSIGN_OR_RETURN(auto left,bound.view());
    HYLE_ASSIGN_OR_RETURN(auto right,expected.view());
    if(left.resource()!=right.resource() || left.native_handle()!=right.native_handle() ||
        left.offset()!=right.offset() || left.size()!=right.size())
      return Status::InvalidArgument("read-projection guard differs from its original bounds");
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(match(binding->read,pages.front()));
  std::size_t matched=1;
  for(const auto&extra:state->resources.get()->bindings){
    if(extra.value.valid() || extra.projection!=value)continue;
    if(extra.scope!=BindingScope::kProjected || matched>=pages.size())
      return std::unexpected(Status::InvalidArgument("read-projection guard count or scope differs from its original owner"));
    HYLE_RETURN_IF_ERROR(match(extra.read,pages[matched]));
    ++matched;
  }
  if(matched!=pages.size())
    return std::unexpected(Status::InvalidArgument("read-projection guard count differs from its original owner"));
  return true;
}
namespace {
Status validate_write_ranges(compute::ConstTensorView tensor,
    std::span<const compute::ReadLease>pages,
    std::span<const compute::WriteLease*const>writers,
    std::span<const compute::detail::AppendLease*const>appenders,
    std::span<const compute::ReadLease>controls)noexcept{
  if(!tensor.valid() || (writers.empty() && appenders.empty()))
    return Status::InvalidArgument("projected write needs capacity and original mutable page permissions");
  const auto storage=tensor.storage();
  const auto inside=[&](auto range)noexcept{
    return range.resource()==storage.resource() && range.native_handle()==storage.native_handle() &&
        range.offset()>=storage.offset() && range.offset()-storage.offset()<=storage.size() &&
        range.size()<=storage.size()-(range.offset()-storage.offset());
  };
  for(std::size_t i=0;i<writers.size();++i){
    if(!writers[i] || !writers[i]->valid())return Status::InvalidArgument("projected page writer is empty");
    HYLE_ASSIGN_OR_RETURN(auto range,writers[i]->view());
    if(!inside(range))return Status::InvalidArgument("projected writer is outside its original capacity");
    const auto root=compute::detail::LeaseAccess::identity(*writers[i]);
    for(std::size_t j=0;j<i;++j)if(compute::detail::LeaseAccess::identity(*writers[j])==root)
      return Status::InvalidArgument("projected page writer is duplicated");
    for(const auto&read:pages)if(compute::detail::LeaseAccess::identity(read)==root)
      return Status::InvalidArgument("projected page mixes ready and mutable permissions");
  }
  for(std::size_t i=0;i<appenders.size();++i){
    const auto*append=appenders[i];
    if(!append || !append->valid() || !append->reserved_bytes())
      return Status::InvalidArgument("projected append authority has no original reserved suffix");
    HYLE_ASSIGN_OR_RETURN(auto range,append->view());
    if(!inside(range))return Status::InvalidArgument("projected suffix is outside its original capacity");
    const auto root=compute::detail::LeaseAccess::identity(*append);
    for(std::size_t j=0;j<i;++j)if(compute::detail::LeaseAccess::identity(*appenders[j])==root)
      return Status::InvalidArgument("projected append authority is duplicated");
    for(const auto*page:writers)if(compute::detail::LeaseAccess::identity(*page)==root)
      return Status::InvalidArgument("projected suffix cannot share an ordinary whole-page writer");
    bool covered=false;
    for(const auto&read:pages)if(compute::detail::LeaseAccess::identity(read)==root){
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*append,read));covered=true;
    }
    if(!covered)return Status::InvalidArgument("projected append has no original committed prefix permission");
  }
  for(const auto&control:controls){
    HYLE_ASSIGN_OR_RETURN(auto view,control.view());(void)view;
    const auto root=compute::detail::LeaseAccess::identity(control);
    for(const auto*page:writers)if(compute::detail::LeaseAccess::identity(*page)==root)
      return Status::InvalidArgument("write control cannot share an ordinary page writer");
    for(const auto*append:appenders)if(compute::detail::LeaseAccess::identity(*append)==root)
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*append,control));
  }
  return Status::Ok();
}
Status validate_write_conflicts(const ResourceEpisode&episode,
    std::span<const compute::ReadLease>pages,
    std::span<const compute::WriteLease*const>writers,
    std::span<const compute::detail::AppendLease*const>appenders,
    std::span<const compute::ReadLease>controls)noexcept{
  // Inspect old reads only against new mutable roots, and old mutable roots
  // against new reads. Read/read pairs need no scan or duplicate owner table.
  for(const auto&binding:episode.bindings){
    const auto*old_append=binding.projected_appender;
    const auto*old_write=binding.projected_writer;
    const bool ordinary_write=old_write || binding.exclusive;
    const void*root=old_append?compute::detail::LeaseAccess::identity(*old_append)
        :old_write?compute::detail::LeaseAccess::identity(*old_write)
        :binding.write.valid()?compute::detail::LeaseAccess::identity(binding.write)
        :binding.read.valid()?compute::detail::LeaseAccess::identity(binding.read):nullptr;
    if(!root)continue;
    for(const auto*page:writers)if(compute::detail::LeaseAccess::identity(*page)==root && !ordinary_write)
      return Status::InvalidArgument("projected writer conflicts with an existing read or append permission");
    for(const auto*append:appenders)if(compute::detail::LeaseAccess::identity(*append)==root){
      if(ordinary_write)return Status::InvalidArgument("projected append conflicts with an existing ordinary writer");
      if(old_append && old_append!=append)return Status::InvalidArgument("projected append repeats a different authority");
      if(binding.read.valid())HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*append,binding.read));
    }
    if(ordinary_write || old_append)for(const auto reads:{pages,controls})for(const auto&read:reads){
      if(compute::detail::LeaseAccess::identity(read)!=root)continue;
      if(ordinary_write)return Status::InvalidArgument("projected read conflicts with an existing ordinary writer");
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*old_append,read));
    }
  }
  return Status::Ok();
}
} // namespace
Status RegionAccess::qualify_ordered_writer(InvocationState&state,std::uint32_t index,
    const compute::detail::CompletionEndpoint*reader)noexcept{
  const auto&selected=state.plan->steps[index];
  if(selected.runtime_properties().mode!=compute::ProviderRuntimeMode::kDelegated)
    return Status::Unimplemented("projected native write needs its original delegated implementation");
  const auto&prepared=selected.region.prepared_;
  const auto*capability=prepared.vtable->query_extension?
      static_cast<const RegionOrderedWriteVTable*>(prepared.vtable->query_extension(
          prepared.state,kRegionOrderedWriteExtensionId)):nullptr;
  if(!capability || !capability->qualify ||
      !state.invocations[index].region.delegated_endpoint().valid() || (reader && !reader->valid()))
    return Status::Unimplemented("native external write has no exact ordered-effect qualification");
  struct Call final{const RegionOrderedWriteVTable*table;const ProviderRegionPrepared*prepared;
    void*slot;const compute::detail::CompletionEndpoint*reader;};
  Call call{capability,&prepared,state.invocations[index].region.state_,reader};
  return selected.region.source_.call([](void*,const void*,void*opaque)noexcept->Status{
    auto&c=*static_cast<Call*>(opaque);return c.table->qualify(*c.prepared,c.slot,c.reader);
  },&call);
}
Status RegionAccess::validate_projected_writer(InvocationState&state,const Graph&graph,
    ProjectedWriter writer)noexcept{
  const auto&plan=*state.plan;
  if(!graph.valid() || GraphIdentity::generation(writer.node)!=plan.generation ||
      writer.node.ordinal()>=plan.node_steps.size() || plan.node_steps[writer.node.ordinal()]!=writer.step ||
      writer.step>=plan.steps.size() || writer.step>=state.invocations.size() ||
      !state.invocations[writer.step].region.valid())
    return Status::InvalidArgument("projected writer belongs to another Graph/plan or has no original slot");
  const auto*logical=GraphAccess::state(graph);
  const auto&selected=plan.steps[writer.step];
  if(!logical || logical->generation!=plan.generation || selected.kind!=PlanStepKind::kRegion ||
      !selected.node_count || selected.node_set_begin>plan.step_nodes.size() ||
      selected.node_count>plan.step_nodes.size()-selected.node_set_begin ||
      !selected.region.matches_artifact(writer.implementation,writer.prepared))
    return Status::InvalidArgument("projected write lacks its actual selected external-write implementation");
  bool nominated=false;
  const auto nodes=std::span<const std::uint32_t>(plan.step_nodes).subspan(selected.node_set_begin,selected.node_count);
  for(const auto node:nodes){
    if(node>=logical->nodes.size() || node>=plan.node_steps.size() || plan.node_steps[node]!=writer.step)
      return Status::InvalidArgument("projected writer region contains a foreign original node");
    const auto traits=logical->nodes[node].traits();
    if(node==writer.node.ordinal()){
      if(nominated || traits.effect!=compute::EffectKind::kExternalSideEffect)
        return Status::InvalidArgument("projected writer is not the original external-write node");
      nominated=true;
    }else if(traits.effect!=compute::EffectKind::kPure || traits.resource_count!=0)
      return Status::InvalidArgument("projected writer region contains another effect or resource access");
  }
  if(!nominated)return Status::InvalidArgument("projected writer is outside its selected original region");
  if(selected.runtime_properties().mode!=compute::ProviderRuntimeMode::kSharedQueue)
    HYLE_RETURN_IF_ERROR(qualify_ordered_writer(state,writer.step,nullptr));
  return Status::Ok();
}
Status RegionAccess::bind_written_projection(Invocation&invocation,const Graph&graph,
    ValueId value,compute::ConstTensorView tensor,std::span<const compute::ReadLease>pages,
    std::span<const compute::WriteLease*const>writers,
    std::span<const compute::detail::AppendLease*const>appenders,ProjectedWriter writer,
    std::span<const ProjectedReader>readers,std::span<const compute::ReadLease>controls,
    ResourceOptions options)noexcept{
  auto*state=InvocationAccess::state(invocation);HYLE_ASSIGN_OR_RETURN(auto*episode,require_resources(state));
  const auto&plan=*state->plan;
  HYLE_RETURN_IF_ERROR(validate_projected_writer(*state,graph,writer));
  if(episode->nominated_writers[writer.step]!=UINT32_MAX)
    return Status::FailedPrecondition("original projected writer needs release before rebinding");
  for(const auto&reader:readers){
    if(reader.step>=plan.steps.size() || reader.step<=writer.step)
      return Status::InvalidArgument("projected write must precede every selected reader");
    const auto&step=plan.steps[reader.step];
    if(step.kind!=PlanStepKind::kRegion || step.node_count!=1 ||
        !step.region.matches_artifact(reader.implementation,reader.prepared) ||
        reader.step>=state->invocations.size() || !state->invocations[reader.step].region.valid())
      return Status::Unimplemented("same-Graph projected write needs its exact singleton reader");
    const auto writer_mode=plan.steps[writer.step].runtime_properties().mode;
    const auto reader_mode=step.runtime_properties().mode;
    if(writer_mode!=compute::ProviderRuntimeMode::kSharedQueue ||
        reader_mode!=compute::ProviderRuntimeMode::kSharedQueue){
      if(reader_mode!=compute::ProviderRuntimeMode::kDelegated)
        return Status::Unimplemented("native projected write needs its qualified original reader Runtime");
      const auto endpoint=state->invocations[reader.step].region.delegated_endpoint();
      HYLE_RETURN_IF_ERROR(qualify_ordered_writer(*state,writer.step,&endpoint));
      if(!state->invocations[reader.step].region.can_import_endpoint(
          state->invocations[writer.step].region.delegated_endpoint()))
        return Status::InvalidArgument("native projected reader cannot import its actual writer endpoint");
    }
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(plan.step_nodes[step.node_set_begin]));
    HYLE_ASSIGN_OR_RETURN(auto count,graph.node_dependency_count(node));bool ordered=false;
    for(std::size_t i=0;i<count;++i){HYLE_ASSIGN_OR_RETURN(auto prerequisite,graph.node_dependency(node,i));ordered|=prerequisite==writer.node;}
    if(!ordered)return Status::InvalidArgument("projected reader lacks its explicit original Graph write dependency");
  }
  HYLE_RETURN_IF_ERROR(validate_write_ranges(tensor,pages,writers,appenders,controls));
  HYLE_RETURN_IF_ERROR(validate_write_conflicts(*episode,pages,writers,appenders,controls));
  if(pages.size()>UINT32_MAX || writers.size()>UINT32_MAX-pages.size() ||
      appenders.size()>UINT32_MAX-pages.size()-writers.size() ||
      controls.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size() ||
      episode->bindings.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size()-controls.size())
    return Status::ResourceExhausted("projected writer permission count overflows");
  HYLE_RETURN_IF_ERROR(reserve_reads(invocation,std::uint32_t(pages.size()+writers.size()+appenders.size()+controls.size()),options));
  // The common read projection validates all physical consumers, aliases and
  // actual ready ranges before committing. Pending here means no ready lease
  // is invented for pages written earlier in this very Graph.
  HYLE_RETURN_IF_ERROR(bind_projected_impl(invocation,value,tensor,pages,readers,true));
  auto*binding=find_binding(state,value);binding->scope=BindingScope::kWrittenProjection;
  binding->projection=value;binding->permissionless=pages.empty();binding->writer_step=writer.step;
  // bind_projected_impl has checked every exact artifact/physical port and
  // complete consumer coverage. Never cache page owners or a prepared epoch.
  binding->permission_changes_preserve_commands=std::all_of(readers.begin(),readers.end(),
      [](const auto&reader){return reader.permission_changes_preserve_commands;});
  for(const auto*page:writers)episode->bindings.push_back({.projection=value,.projected_writer=page,.borrowed=false,.scope=BindingScope::kWrittenProjection});
  for(const auto*append:appenders)episode->bindings.push_back({.projection=value,.projected_appender=append,.borrowed=false,.scope=BindingScope::kWrittenProjection});
  // These are original prepared-program control buffers, never arena pages or
  // write authority. The same episode admits/retains their actual read uses.
  for(const auto&control:controls)episode->bindings.push_back({.projection=value,.read=control,.borrowed=false,.scope=BindingScope::kWrittenProjection});
  episode->nominated_writers[writer.step]=writer.node.ordinal();
  return Status::Ok();
}
Status RegionAccess::bind_write_effect(Invocation&invocation,const Graph&graph,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease>pages,
    std::span<const compute::WriteLease*const>writers,
    std::span<const compute::detail::AppendLease*const>appenders,ProjectedWriter writer,
    std::span<const compute::ReadLease>controls,ResourceOptions options)noexcept{
  auto*state=InvocationAccess::state(invocation);HYLE_ASSIGN_OR_RETURN(auto*episode,require_resources(state));
  HYLE_RETURN_IF_ERROR(validate_projected_writer(*state,graph,writer));
  if(episode->nominated_writers[writer.step]!=UINT32_MAX)
    return Status::FailedPrecondition("original write effect needs release before rebinding");
  HYLE_RETURN_IF_ERROR(validate_write_ranges(tensor,pages,writers,appenders,controls));
  const auto storage=tensor.storage();
  for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() ||
        view.offset()<storage.offset() || view.offset()-storage.offset()>storage.size() ||
        view.size()>storage.size()-(view.offset()-storage.offset()))
      return Status::InvalidArgument("write-effect ready page is outside its original capacity");
  }
  HYLE_RETURN_IF_ERROR(validate_write_conflicts(*episode,pages,writers,appenders,controls));
  if(pages.size()>UINT32_MAX || writers.size()>UINT32_MAX-pages.size() ||
      appenders.size()>UINT32_MAX-pages.size()-writers.size() ||
      controls.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size() ||
      episode->bindings.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size()-controls.size())
    return Status::ResourceExhausted("write-effect permission count overflows");
  HYLE_RETURN_IF_ERROR(reserve_reads(invocation,std::uint32_t(pages.size()+writers.size()+appenders.size()+controls.size()),options));
  HYLE_RETURN_IF_ERROR(revoke_preparation(invocation));
  // No caller input or capacity lease is installed. All fallible validation
  // and growth precedes these infallible anonymous ownership records.
  episode->input_lease_candidates.clear();
  for(const auto&page:pages)episode->bindings.push_back({.read=page,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto*page:writers)episode->bindings.push_back({.projected_writer=page,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto*append:appenders)episode->bindings.push_back({.projected_appender=append,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto&control:controls)episode->bindings.push_back({.read=control,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  episode->nominated_writers[writer.step]=writer.node.ordinal();
  return Status::Ok();
}
bool RegionAccess::successful_outcome(const Invocation&invocation)noexcept{
  const auto*state=InvocationAccess::state(invocation);
  return outcome_available(invocation) && state->phase.load(std::memory_order_acquire)==InvocationPhase::kIdle &&
      state->outputs_ready.load(std::memory_order_acquire);
}
Result<bool> RegionAccess::has_write_effect(const InvocationState*state,
    ProjectedWriter writer,std::span<const compute::WriteLease*const>writers)noexcept{
  if(!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("write-effect parent invocation is empty"));
  const auto&plan=*state->plan;
  if(GraphIdentity::generation(writer.node)!=plan.generation ||
      writer.node.ordinal()>=plan.node_steps.size() ||
      plan.node_steps[writer.node.ordinal()]!=writer.step ||
      writer.step>=plan.steps.size() || writer.step>=state->invocations.size())
    return std::unexpected(Status::InvalidArgument("write-effect proof belongs to another original plan"));
  const auto&selected=plan.steps[writer.step];
  if(selected.kind!=PlanStepKind::kRegion || !state->invocations[writer.step].region.valid() ||
      !selected.region.matches_artifact(writer.implementation,writer.prepared))
    return std::unexpected(Status::InvalidArgument("write-effect proof differs from its selected artifact"));
  if(!state->resources)return false;
  const auto&episode=*state->resources.get();
  if(writer.step>=episode.nominated_writers.size())
    return std::unexpected(Status::Internal("write-effect identity table does not cover its original plan"));
  const auto nominated=episode.nominated_writers[writer.step];
  if(nominated==UINT32_MAX)return false;
  if(nominated!=writer.node.ordinal() || writers.empty())
    return std::unexpected(Status::InvalidArgument("write-effect proof is not the original nominated writer"));
  std::size_t matched=0;
  for(const auto&binding:episode.bindings){
    if(binding.scope!=BindingScope::kWriteEffect || binding.writer_step!=writer.step)continue;
    if(binding.projected_appender)
      return std::unexpected(Status::InvalidArgument("ordinary writer proof cannot substitute an append authority"));
    if(!binding.projected_writer)continue;
    if(matched>=writers.size() || binding.projected_writer!=writers[matched] ||
        !writers[matched] || !writers[matched]->valid())
      return std::unexpected(Status::InvalidArgument("write-effect authority differs from its original owner"));
    ++matched;
  }
  if(matched!=writers.size())
    return std::unexpected(Status::InvalidArgument("write-effect authority count differs from its original owner"));
  return true;
}
Status RegionAccess::visit_retired(Invocation&invocation,RegionAccessFn callback,void*user)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!callback)return Status::InvalidArgument("retired region visitor is empty");
  if(!outcome_available(invocation))return Status::Unavailable("Graph still retains native work/results/observations");
  const auto&plan=*state->plan;
  for(std::size_t i=0;i<plan.steps.size();++i){const auto&step=plan.steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    RegionAccessView view{std::uint32_t(i),plan.generation,std::span<const std::uint32_t>(plan.step_nodes).subspan(step.node_set_begin,step.node_count),
      step.region.source_.shared_extension(),step.region.prepared_.state,state->invocations[i].region.state_,step.lane,{}, {},step.region.prepared_.vtable};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
Status RegionAccess::visit_retired(Invocation&invocation,NodeId node,RegionAccessFn callback,void*user)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!callback)return Status::InvalidArgument("retired region visitor is empty");
  if(!outcome_available(invocation))return Status::Unavailable("Graph still retains native work/results/observations");
  const auto&plan=*state->plan;
  if(GraphIdentity::generation(node)!=plan.generation || node.ordinal()>=plan.node_steps.size())
    return Status::InvalidArgument("retired region access node is foreign");
  const auto index=plan.node_steps[node.ordinal()];
  if(index>=plan.steps.size())return Status::Unimplemented("node has no selected region");
  const auto&step=plan.steps[index];
  if(step.kind!=PlanStepKind::kRegion || !step.region.valid())return Status::Unimplemented("node has no selected region");
  if(step.node_set_begin>plan.step_nodes.size() || step.node_count>plan.step_nodes.size()-step.node_set_begin ||
      index>=state->invocations.size() || !state->invocations[index].region.valid())
    return Status::Internal("selected retired region node set or slot is invalid");
  const auto nodes=std::span<const std::uint32_t>(plan.step_nodes).subspan(step.node_set_begin,step.node_count);
  if(std::ranges::find(nodes,node.ordinal())==nodes.end())return Status::InvalidArgument("node is outside its selected region");
  RegionAccessView view{index,plan.generation,nodes,step.region.source_.shared_extension(),
    step.region.prepared_.state,state->invocations[index].region.state_,step.lane,{}, {},step.region.prepared_.vtable};
  return callback(user,view);
}
bool RegionAccess::writer_preserves_commands(const InvocationState&state,std::uint32_t writer_index)noexcept{
  const auto&writer=state.plan->steps[writer_index];
  if(state.closed || state.phase.load(std::memory_order_acquire)!=InvocationPhase::kIdle ||
      !state.step_bindings[writer_index].stable_binding_identity || writer.kind!=PlanStepKind::kRegion)
    return false;
  const auto&prepared=writer.region.prepared_;
  const auto*table=prepared.vtable->query_extension?
      static_cast<const RegionProjectedWriteVTable*>(prepared.vtable->query_extension(
          prepared.state,kRegionProjectedWriteExtensionId)):nullptr;
  return table && table->preserves_commands &&
      table->preserves_commands(prepared,state.invocations[writer_index].region.state_);
}
Status RegionAccess::release_writer_preparation(Invocation&invocation,std::uint32_t writer_index)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!state || writer_index>=state->plan->steps.size() || writer_index>=state->step_bindings.size() ||
      writer_index>=state->invocations.size())return Status::InvalidArgument("write effect selected step is invalid");
  if(writer_preserves_commands(*state,writer_index))HYLE_RETURN_IF_ERROR(revoke_preparation(invocation));
  else invalidate_step_recording(*state,writer_index);
  return Status::Ok();
}
Status RegionAccess::release_write_effect(Invocation&invocation,NodeId node)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!outcome_available(invocation))return Status::Unavailable("write effect still has native work/results/observations");
  const auto&plan=*state->plan;
  if(GraphIdentity::generation(node)!=plan.generation || node.ordinal()>=plan.node_steps.size() || !state->resources)
    return Status::InvalidArgument("write effect belongs to another Graph/plan");
  const auto writer_index=plan.node_steps[node.ordinal()];
  auto&episode=*state->resources.get();
  if(writer_index>=episode.nominated_writers.size() || episode.nominated_writers[writer_index]!=node.ordinal())
    return Status::InvalidArgument("node is not the original bound external writer");
  if(std::none_of(episode.bindings.begin(),episode.bindings.end(),[&](const auto&binding){
      return binding.scope==BindingScope::kWriteEffect && binding.writer_step==writer_index;
    }))return Status::InvalidArgument("node has no original write-effect binding");
  HYLE_RETURN_IF_ERROR(release_writer_preparation(invocation,writer_index));
  episode.input_lease_candidates.clear();
  const auto first=episode.bindings.begin()+episode.fixed_binding_count;
  episode.bindings.erase(std::remove_if(first,episode.bindings.end(),[&](const auto&binding){
    return binding.scope==BindingScope::kWriteEffect && binding.writer_step==writer_index;
  }),episode.bindings.end());
  episode.nominated_writers[writer_index]=UINT32_MAX;
  return Status::Ok();
}
Status RegionAccess::release_written_projection(Invocation&invocation,ValueId value)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!outcome_available(invocation))return Status::Unavailable("projected write still has native work/results/observations");
  auto*binding=find_binding(state,value);
  if(!binding || binding->scope!=BindingScope::kWrittenProjection)
    return Status::InvalidArgument("input has no original projected write binding");
  const auto writer_index=binding->writer_step;
  auto&episode=*state->resources.get();
  if(writer_index>=episode.nominated_writers.size() || episode.nominated_writers[writer_index]==UINT32_MAX)
    return Status::InvalidArgument("input has no original nominated projected writer");
  HYLE_RETURN_IF_ERROR(release_writer_preparation(invocation,writer_index));
  for(const auto&consumer:projected_input_consumers(*state->plan,value.ordinal())){
    // Keep only physical recordings qualified by BOTH the selected reader
    // and the actual capture backend. They remain dormant: the writer above
    // revoked replay/outputs, and every old page guard is withdrawn below.
    // The next cold prepare must admit fresh permissions/token and recheck
    // exact arguments; genuine native changes retain their original callback.
    if(!binding->permission_changes_preserve_commands ||
       !state->step_bindings[consumer.step].stable_binding_identity)
      invalidate_step_recording(*state,consumer.step);
  }
  const auto first=episode.bindings.begin()+episode.fixed_binding_count;
  episode.bindings.erase(std::remove_if(first,episode.bindings.end(),[&](const auto&entry){return entry.projection==value;}),episode.bindings.end());
  episode.nominated_writers[writer_index]=UINT32_MAX;
  binding=find_binding(state,value);const auto physical=state->plan->external_instances[value.ordinal()];
  state->readable[physical]={};state->bound[physical]=0;clear_binding_owner(*binding);
  return Status::Ok();
}
Status RegionAccess::release_projected_writes(Invocation&invocation,
    std::span<const NodeId>nodes)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!outcome_available(invocation))
    return Status::Unavailable("projected writes still have native work/results/observations");
  if(nodes.empty())return Status::Ok();
  if(!state->resources)return Status::InvalidArgument("projected writes have no original resource episode");
  const auto&plan=*state->plan;auto&episode=*state->resources.get();
  auto&marks=episode.unbind_steps;
  HYLE_FATAL_ASSERT(marks.size()==plan.steps.size());
  constexpr std::uint8_t selected=1,bound=2,invalidate=4;
  std::fill(marks.begin(),marks.end(),std::uint8_t{});
  // Scratch marks index original selected steps, never addresses or retained
  // authorities. Reject the complete list before touching native preparation.
  for(const auto node:nodes){
    if(GraphIdentity::generation(node)!=plan.generation || node.ordinal()>=plan.node_steps.size())
      return Status::InvalidArgument("projected writer belongs to another Graph/plan");
    const auto step=plan.node_steps[node.ordinal()];
    if(step>=marks.size() || step>=state->invocations.size() || step>=state->step_bindings.size() ||
        plan.steps[step].kind!=PlanStepKind::kRegion || !plan.steps[step].region.valid())
      return Status::InvalidArgument("projected writer has no original selected region");
    const auto&writer=plan.steps[step];
    if(!writer.node_count || writer.node_set_begin>plan.step_nodes.size() ||
        writer.node_count>plan.step_nodes.size()-writer.node_set_begin)
      return Status::InvalidArgument("projected writer region node extent is invalid");
    const auto members=std::span<const std::uint32_t>(plan.step_nodes).subspan(writer.node_set_begin,writer.node_count);
    if(std::ranges::find(members,node.ordinal())==members.end() ||
        step>=episode.nominated_writers.size() || episode.nominated_writers[step]!=node.ordinal())
      return Status::InvalidArgument("projected writer is not its original nominated region member");
    if(marks[step]&selected)return Status::InvalidArgument("projected writer is duplicated");
    marks[step]|=selected;
  }
  for(const auto&entry:episode.bindings)
    if((entry.scope==BindingScope::kWrittenProjection && entry.value.valid()) ||
        entry.scope==BindingScope::kWriteEffect)
      if(entry.writer_step<marks.size() && (marks[entry.writer_step]&selected))
        marks[entry.writer_step]|=bound;
  bool revoke=false;
  for(const auto node:nodes){
    const auto step=plan.node_steps[node.ordinal()];
    if(!(marks[step]&bound))return Status::InvalidArgument("node has no original projected write binding");
    // Qualify before invalidation through the same helper as single release.
    if(writer_preserves_commands(*state,step))revoke=true;else marks[step]|=invalidate;
  }
  for(std::size_t i=0;i<episode.fixed_binding_count;++i){
    const auto&entry=episode.bindings[i];
    if(entry.scope!=BindingScope::kWrittenProjection || entry.writer_step>=marks.size() ||
        !(marks[entry.writer_step]&selected))continue;
    for(const auto&consumer:projected_input_consumers(plan,entry.value.ordinal()))
      if(!entry.permission_changes_preserve_commands ||
          !state->step_bindings[consumer.step].stable_binding_identity)
        marks[consumer.step]|=invalidate;
  }
  if(revoke)HYLE_RETURN_IF_ERROR(revoke_preparation(invocation));
  // No rejecting operation remains. Preserve only recordings qualified by
  // the original writer/reader and actual capture backend, as in single take.
  for(std::size_t i=0;i<marks.size();++i)
    if(marks[i]&invalidate)invalidate_step_recording(*state,static_cast<std::uint32_t>(i));
  episode.input_lease_candidates.clear();
  const auto first=episode.bindings.begin()+episode.fixed_binding_count;
  episode.bindings.erase(std::remove_if(first,episode.bindings.end(),[&](const auto&entry){
    if(entry.scope==BindingScope::kWriteEffect)
      return entry.writer_step<marks.size() && (marks[entry.writer_step]&selected);
    if(!entry.projection.valid())return false;
    const auto*owner=find_binding(state,entry.projection);
    return owner && owner->scope==BindingScope::kWrittenProjection &&
        owner->writer_step<marks.size() && (marks[owner->writer_step]&selected);
  }),episode.bindings.end());
  for(std::size_t i=0;i<episode.fixed_binding_count;++i){
    auto&entry=episode.bindings[i];
    if(entry.scope!=BindingScope::kWrittenProjection || entry.writer_step>=marks.size() ||
        !(marks[entry.writer_step]&selected))continue;
    const auto physical=plan.external_instances[entry.value.ordinal()];
    state->readable[physical]={};state->bound[physical]=0;clear_binding_owner(entry);
  }
  for(const auto node:nodes)episode.nominated_writers[plan.node_steps[node.ordinal()]]=UINT32_MAX;
  return Status::Ok();
}
bool RegionAccess::outcome_available(const Invocation&invocation)noexcept{
  const auto*state=InvocationAccess::state(invocation);
  return state && state->phase.load(std::memory_order_acquire)!=InvocationPhase::kRunning &&
      !state->results.load(std::memory_order_acquire) && state->observation.available();
}
Status RegionAccess::bind_read_projection(Invocation&invocation,ReadProjection projection,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease> pages)noexcept{
  auto*state=InvocationAccess::state(invocation);HYLE_ASSIGN_OR_RETURN(auto*episode,require_resources(state));
  const auto&plan=*state->plan;const auto capacity=projection.capacity;
  if(GraphIdentity::generation(capacity)!=plan.generation || capacity.kind()!=ValueKind::kInput ||
      capacity.ordinal()>=plan.external_instances.size() || projection.readers.empty() || !tensor.valid() || pages.empty())
    return Status::InvalidArgument("read projection requires its exact plan/input/footprint");
  auto*binding=find_binding(state,capacity);
  if(!binding || !binding->input)return Status::InvalidArgument("read projection is not a plan input");
  if(binding->read.valid() || binding->write.valid() || binding->permissionless)
    return Status::FailedPrecondition("read projection needs reset before rebinding");
  const auto root=plan.external_instances[capacity.ordinal()];
  const auto allowed=[&](std::uint32_t value)noexcept{
    return value==capacity.ordinal() || std::ranges::any_of(projection.aliases,[&](ValueId alias){return alias.ordinal()==value;});
  };
  const auto origin=[&](std::uint32_t input)noexcept{
    while(plan.alias_sources[input]!=kNoInPlaceSource)input=plan.alias_sources[input];
    return input;
  };
  for(std::size_t i=0;i<projection.aliases.size();++i){const auto alias=projection.aliases[i];
    if(GraphIdentity::generation(alias)!=plan.generation || alias==capacity || alias.ordinal()>=plan.binding_roles.size() ||
        (plan.binding_roles[alias.ordinal()]&kExternalOutput))return Status::InvalidArgument("read projection alias is foreign or published");
    for(std::size_t j=0;j<i;++j)if(alias==projection.aliases[j])return Status::InvalidArgument("read projection alias is duplicated");
  }
  if(plan.binding_roles[capacity.ordinal()]&kExternalOutput)return Status::InvalidArgument("projected capacity is published");
  for(std::size_t i=0;i<plan.alias_sources.size();++i){
    if(origin(std::uint32_t(i))!=root)continue;
    if(!allowed(plan.instance_values[i]))return Status::InvalidArgument("projected capacity has an undeclared physical alias");
    std::size_t offset=0;auto current=std::uint32_t(i);
    while(current!=root){if(plan.alias_offsets[current]>SIZE_MAX-offset)return Status::InvalidArgument("projection alias byte offset overflow");
      offset+=plan.alias_offsets[current];current=plan.alias_sources[current];}
    HYLE_ASSIGN_OR_RETURN(auto bytes,plan.instance_descriptions[i].required_bytes());
    if(offset>tensor.storage().size() || bytes>tensor.storage().size()-offset)
      return Status::InvalidArgument("projected alias bytes exceed the original capacity");
  }
  for(std::size_t i=0;i<projection.readers.size();++i){const auto reader=projection.readers[i];
    if(GraphIdentity::generation(reader)!=plan.generation || reader.ordinal()>=plan.node_steps.size())
      return Status::InvalidArgument("projected reader is foreign");
    const auto step=plan.node_steps[reader.ordinal()];
    if(step>=plan.steps.size() || (plan.steps[step].kind!=PlanStepKind::kOperation && plan.steps[step].kind!=PlanStepKind::kRegion))
      return Status::InvalidArgument("projected reader has no selected prepared implementation");
    bool reads=false;for(std::size_t port=0;port<plan.steps[step].input_count;++port)
      reads|=origin(plan.node_inputs[plan.steps[step].input_begin+port])==root;
    if(!reads)return Status::InvalidArgument("projected reader does not consume this physical root");
    for(std::size_t j=0;j<i;++j)if(reader==projection.readers[j])return Status::InvalidArgument("projected reader is duplicated");
  }
  bool consumed=false;
  for(std::size_t i=0;i<plan.steps.size();++i){const auto&step=plan.steps[i];
    const bool declared=std::ranges::any_of(projection.readers,[&](NodeId reader){return plan.node_steps[reader.ordinal()]==i;});
    for(std::size_t port=0;port<step.input_count;++port){const auto input=plan.node_inputs[step.input_begin+port];if(origin(input)!=root)continue;
      if(!allowed(plan.instance_values[input]))return Status::InvalidArgument("projection physical input is not a declared alias");
      if(step.kind==PlanStepKind::kView && !step.materializes){
        for(std::size_t out=0;out<step.output_count;++out){const auto value=plan.node_outputs[step.output_begin+out];
          if(origin(value)!=root || !allowed(plan.instance_values[value]))return Status::InvalidArgument("projection view escapes its declared aliases");}
        continue;
      }
      if(!declared)return Status::InvalidArgument("projected root has an unqualified physical consumer");
      compute::detail::OperationPreparedAccess access;
      if(step.kind==PlanStepKind::kOperation){
        HYLE_ASSIGN_OR_RETURN(access,compute::detail::OperationPreparationAccess::inspect_access(step.operation,compute::detail::kProjectedInputReadExtensionId));
      }else if(step.kind==PlanStepKind::kRegion){
        const auto&prepared=step.region.prepared_;
        const auto*interface=prepared.vtable->query_extension ?
            static_cast<const RegionProjectedReadVTable*>(prepared.vtable->query_extension(
                prepared.state,kRegionProjectedReadExtensionId)) : nullptr;
        if(!interface || !interface->inspect)
          return Status::Unimplemented("selected region has no projected input inspector");
        HYLE_RETURN_IF_ERROR(interface->inspect(prepared,
            state->invocations[i].region.state_,std::uint32_t(port),&access));
      }else return Status::InvalidArgument("projected root has a materializing/copy consumer");
      if(access.projected_input!=port)return Status::Unimplemented("selected implementation does not project this physical input port");
      consumed=true;
    }
  }
  if(!consumed)return Status::InvalidArgument("projected root has no selected checked reader");
  if(pages.size()-1>episode->bindings.capacity()-episode->bindings.size())return Status::ResourceExhausted("read projection guard capacity exhausted");
  const auto storage=tensor.storage();for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() || view.offset()<storage.offset() ||
        view.offset()-storage.offset()>storage.size() || view.size()>storage.size()-(view.offset()-storage.offset()))
      return Status::InvalidArgument("projected page guard outside original capacity");}
  HYLE_RETURN_IF_ERROR(invocation.bind(capacity,tensor));
  binding->read=pages.front();binding->description=tensor.description();binding->exclusive=false;binding->borrowed=false;
  binding->scope=BindingScope::kProjected;
  for(std::size_t i=1;i<pages.size();++i)episode->bindings.push_back({.projection=capacity,.read=pages[i],.borrowed=false,.scope=BindingScope::kProjected});
  return Status::Ok();
}
Result<compute::WriteLease> RegionAccess::take_write(Invocation&invocation,ValueId value)noexcept{
  auto*state=InvocationAccess::state(invocation);
  HYLE_ASSIGN_OR_RETURN(auto*episode,require_resources(state));
  (void)episode;
  if(state->phase.load(std::memory_order_acquire)!=InvocationPhase::kIdle)
    return std::unexpected(Status::FailedPrecondition("failed graph output cannot be reclaimed"));
  auto*binding=find_binding(state,value);if(!binding || binding->input)
    return std::unexpected(Status::InvalidArgument("only caller output writer can be reclaimed"));
  return std::move(binding->write);
}
Result<RegionNodeAccess> RegionAccess::projection_access(
    const PlanState &plan, const InvocationState *invocation,
    ResourceProjection projection) noexcept {
  if (!projection.capability.valid() || projection.access_nodes.empty())
    return std::unexpected(Status::InvalidArgument("projection needs its selected access implementations"));
  RegionNodeAccess reader_access;
  bool found = false;
  for (const auto node : projection.access_nodes) {
    if (GraphIdentity::generation(node) != plan.generation || node.ordinal() >= plan.node_steps.size())
      return std::unexpected(Status::InvalidArgument("projection access node is foreign"));
    HYLE_ASSIGN_OR_RETURN(auto access,
        inspect_projection_node(plan, invocation, node, projection.capability));
    if (node == projection.reader) { reader_access = access; found = true; }
  }
  if (!found || reader_access.access.projected_input == compute::detail::OperationPreparedAccess::kNoInput)
    return std::unexpected(Status::Unimplemented("selected reader does not project a physical input"));
  const auto &reader = plan.steps[plan.node_steps[projection.reader.ordinal()]];
  if (reader_access.access.projected_input >= reader.input_count)
    return std::unexpected(Status::InvalidArgument("projection prepared input port is outside its selected step"));
  const auto physical = plan.node_inputs[reader.input_begin + reader_access.access.projected_input];
  if (plan.instance_values[physical] != projection.capacity.ordinal())
    return std::unexpected(Status::InvalidArgument("projection capacity differs from selected prepared input"));
  return reader_access;
}
Result<mem::MemoryResourceRef> RegionAccess::projection_resource(const Executable&executable,
    ResourceProjection projection,ExecutionLaneId*reader_lane)noexcept{
  const auto*plan=ExecutableAccess::state(executable);
  const auto capacity=projection.capacity,metadata=projection.metadata;
  const auto decoder=projection.reader;
  if(!plan || capacity==metadata ||
      capacity.kind()!=ValueKind::kInput || metadata.kind()!=ValueKind::kInput ||
      GraphIdentity::generation(capacity)!=plan->generation || GraphIdentity::generation(metadata)!=plan->generation ||
      capacity.ordinal()>=plan->external_instances.size() || metadata.ordinal()>=plan->external_instances.size() ||
      !(plan->binding_roles[capacity.ordinal()]&kExternalInput) ||
      !(plan->binding_roles[metadata.ordinal()]&kExternalInput))
    return std::unexpected(Status::InvalidArgument("projection needs its exact plan and input identities"));
  HYLE_ASSIGN_OR_RETURN(auto access,projection_access(*plan,nullptr,projection));
  const auto bank_port=access.access.projected_input;
  const auto root=plan->external_instances[capacity.ordinal()];
  if(!plan->external_input_aliases.empty() && std::binary_search(plan->external_input_aliases.begin(),
      plan->external_input_aliases.end(),capacity.ordinal()))
    return std::unexpected(Status::InvalidArgument("row capacity cannot have a physical alias descendant"));
  const auto consumers=projected_input_consumers(*plan,capacity.ordinal());
  for(const auto&consumer:consumers){const auto&step=plan->steps[consumer.step];
    if(consumer.instance!=root || consumer.step!=plan->node_steps[decoder.ordinal()] ||
        (step.kind!=PlanStepKind::kOperation && step.kind!=PlanStepKind::kRegion) || consumer.port!=bank_port)
      return std::unexpected(Status::InvalidArgument("row capacity has a copy or unrelated physical reader"));
  }
  if(consumers.empty() || (plan->binding_roles[capacity.ordinal()]&kExternalOutput)!=0)
    return std::unexpected(Status::InvalidArgument("row capacity is published or has no checked reader"));
  const auto a=plan->external_storage[capacity.ordinal()],b=plan->external_storage[metadata.ordinal()];
  if(a>=plan->storage_domains.size() || b>=plan->storage_domains.size() ||
      plan->storage_domains[a].resource!=plan->storage_domains[b].resource)
    return std::unexpected(Status::Unimplemented("row capacity and metadata require the same native resource"));
  if(reader_lane)*reader_lane=plan->steps[plan->node_steps[decoder.ordinal()]].lane;
  return plan->storage_domains[a].resource;
}
Status RegionAccess::bind_projection(Invocation&invocation,ResourceProjection projection,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease>pages,bool pending)noexcept{
  auto*state=InvocationAccess::state(invocation);HYLE_ASSIGN_OR_RETURN(auto*episode, require_resources(state));
  HYLE_ASSIGN_OR_RETURN(auto resource,projection_resource(invocation.executable(),projection));
  HYLE_ASSIGN_OR_RETURN(auto access,projection_access(*state->plan,state,projection));
  if(pending && !access.allows_pending)
    return Status::Unimplemented("selected projected reader has no pending source admission");
  const auto capacity=projection.capacity;
  auto*binding=find_binding(state,capacity);
  const auto&description=state->plan->domain_graph.valid()?state->concrete_descriptions[capacity.ordinal()]:state->plan->descriptions[capacity.ordinal()];
  if(!binding || !binding->input || !tensor.valid() || tensor.description()!=description || tensor.storage().resource()!=resource ||
      (pages.empty() && !pending && !access.access.allows_empty_footprint) || (pending && !pages.empty()))
    return Status::InvalidArgument("row projection needs its native capacity and exact page guards");
  if(binding->read.valid() || binding->write.valid() || binding->permissionless)return Status::FailedPrecondition("row projection needs reset before rebinding");
  if(!pages.empty() && pages.size()-1>episode->bindings.capacity()-episode->bindings.size())return Status::ResourceExhausted("row projection page guard capacity");
  const auto storage=tensor.storage();for(const auto&page:pages){HYLE_ASSIGN_OR_RETURN(auto view,page.view());
    if(view.resource()!=storage.resource() || view.native_handle()!=storage.native_handle() || view.offset()<storage.offset() ||
        view.offset()-storage.offset()>storage.size() || view.size()>storage.size()-(view.offset()-storage.offset()))
      return Status::InvalidArgument("row page guard outside original capacity");}
  HYLE_RETURN_IF_ERROR(invocation.bind(capacity,tensor));
  binding=find_binding(state,capacity);
  if(!pages.empty())binding->read=pages.front();
  binding->description=description;binding->exclusive=false;binding->borrowed=false;binding->permissionless=pages.empty()&&!pending;
  binding->scope=BindingScope::kProjected;
  for(std::size_t i=1;i<pages.size();++i)episode->bindings.push_back({.projection=capacity,.read=pages[i],.borrowed=false,.scope=BindingScope::kProjected});
  return Status::Ok();
}
Status prepare_submission_lifetime(InvocationState &state,
                                   mem::AllocatorRef allocator) noexcept {
  HYLE_RETURN_IF_ERROR(resource_control_ready(&state));
  if (!allocator.valid())
    return Status::InvalidArgument("graph submission allocator is invalid");
  if (state.submission_lifetime) return Status::Ok();
  if (state.plan->segments.size() == SIZE_MAX)
    return Status::ResourceExhausted("graph submission retention capacity overflow");
  auto *lifetime = allocator.construct<SubmissionLifetime>(allocator);
  if (!lifetime)
    return Status::ResourceExhausted("graph submission lifetime metadata");
  SubmissionLifetimeOwner candidate(lifetime);
  try { lifetime->retentions.resize(state.plan->segments.size() + 1); }
  catch (...) { return Status::ResourceExhausted("graph fixed submission metadata"); }
  state.submission_lifetime = std::move(candidate);
  return Status::Ok();
}

Status begin_submission_lifetime(InvocationState &state,
                                 OrderedGraphSubmission ordered) noexcept {
  if (!state.submission_lifetime)
    return Status::FailedPrecondition("graph submission lifetime is not prepared");
  auto &lifetime = *state.submission_lifetime.get();
  if (state.phase.load(std::memory_order_acquire) != InvocationPhase::kRunning ||
      lifetime.remaining.load(std::memory_order_acquire) != 0)
    return Status::FailedPrecondition("graph submission lifetime requires an admitted cursor");
  if (!ordered.valid())
    return Status::InvalidArgument("ordered graph submission requires source owner, input qualification and mapping");
  if (ordered.map && !state.resources)
    return Status::FailedPrecondition("ordered graph submission requires prepared resources");
  if (!state.observation.available())
    return Status::Unavailable("old graph completion retains its observation slot");
  lifetime.observation_retirement = state.observation.reserve_retirement();
  {
    std::lock_guard guard(lifetime.lock);
    lifetime.first_error = Status::Ok();
    lifetime.accepted = false;
    lifetime.ordered = ordered;
    lifetime.remaining.store(1, std::memory_order_relaxed);
    retain(&state);
  }
  return Status::Ok();
}

void record_submission_error(InvocationState &state, const Status &error) noexcept {
  if (!error.ok()) {
    HYLE_FATAL_ASSERT(state.submission_lifetime);
    auto &lifetime = *state.submission_lifetime.get();
    std::lock_guard guard(lifetime.lock);
    lifetime.first_error.Update(error);
  }
}

Status submission_lifetime_status(const InvocationState &state) noexcept {
  if (!state.submission_lifetime) return Status::Ok();
  auto &lifetime = *state.submission_lifetime.get();
  std::lock_guard guard(lifetime.lock);
  return lifetime.first_error;
}

void child_terminal(void *opaque, const Status &terminal) noexcept {
  auto &state = *static_cast<InvocationState *>(opaque);
  record_submission_error(state, terminal);
}
void release_submission_lifetime(void *opaque) noexcept {
  auto &state = *static_cast<InvocationState *>(opaque);
  auto &lifetime = *state.submission_lifetime.get();
  if (lifetime.remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Status terminal;
  {
    std::lock_guard guard(lifetime.lock);
    terminal = lifetime.first_error;
  }
  // Successful ordered consumers prove their immutable producers were reached.
  // Reconcile their original cached diagnostics before publishing this slot.
  auto source_terminal = reconcile_input_sources(state, terminal);
  if (!source_terminal.ok()) {
    record_submission_error(state, source_terminal);
    terminal.Update(source_terminal);
  }
  const bool accepted = lifetime.accepted;
  lifetime.ordered = {};
  // All accepted children have actually released their guards. Finish every
  // resource mutation before making the original invocation idle/failed.
  if (state.resources) {
    auto &episode = *state.resources.get();
    if (accepted && episode.batch.valid()) {
      auto retired = compute::Completion::completed(terminal);
      HYLE_FATAL_ASSERT(episode.batch.attach(retired).ok());
    } else {
      episode.batch = {};
    }
    episode.permissions.reset();
  }
  const auto observation = std::exchange(lifetime.observation_retirement, {});
  InvocationAccess::retire(state, terminal, accepted);
  state.progress.notify();
  observation.terminal(observation.state, terminal);
  observation.release(observation.state);
  // No lifetime/resource access after publication: another cold call may reuse
  // them immediately. Only the original owner's final reference remains.
  release(&state);
}
void retain_submission_completion(void *opaque, std::size_t segment,
                                  compute::Completion &completion) noexcept {
  auto &state = *static_cast<InvocationState *>(opaque);
  auto &lifetime = *state.submission_lifetime.get();
  HYLE_FATAL_ASSERT(lifetime.remaining.load(std::memory_order_relaxed) != 0);
  HYLE_FATAL_ASSERT(segment < lifetime.retentions.size());
  auto &node = lifetime.retentions[segment];
  HYLE_FATAL_ASSERT(!node.own.valid());
  // The final observation guard is not another accepted native command.
  // Preserve zero-command rejection/retry even when an ordered mapper owns
  // a real external prefix that still needs retirement.
  if (segment < state.plan->segments.size()) lifetime.accepted = true;
  lifetime.remaining.fetch_add(1, std::memory_order_relaxed);
  node.own = {&state, &child_terminal, &release_submission_lifetime};
  HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(completion, node).ok());
}

Status admit_submission_sequence(void *opaque, ExecutionLaneId lane,
                                  compute::detail::QueueSequence &sequence) noexcept {
  auto &state = *static_cast<InvocationState *>(opaque);
  const auto &ordered = state.submission_lifetime.get()->ordered;
  if (!ordered.admit) return Status::Ok();
  return ordered.admit(ordered.owner, lane, sequence, state.resources.get()->batch);
}

void finish_submission_lifetime(InvocationState &state, const Status &terminal) noexcept {
  HYLE_FATAL_ASSERT(state.submission_lifetime &&
      state.submission_lifetime.get()->remaining.load(std::memory_order_relaxed) != 0);
  record_submission_error(state, terminal);
  release_submission_lifetime(&state);
}

bool submission_admission_enabled(const InvocationState &state) noexcept {
  return state.submission_lifetime && state.submission_lifetime.get()->ordered.admit;
}

Result<compute::ReadLease> output_storage(InvocationState &state,
    compute::ConstTensorView tensor, std::uint32_t instance) noexcept {
  if (state.resources) {
    const auto wanted = tensor.storage();
    HYLE_ASSIGN_OR_RETURN(const auto bytes, tensor.description().required_bytes());
    const auto &plan = *state.plan;
    // A physical value may be a replica or an alias of a different external
    // binding. Only its exact external instance can name that binding directly;
    // the live storage/range checks below remain the authority for its lease.
    if (instance < plan.instance_values.size()) {
      const auto logical = plan.instance_values[instance];
      if (logical < plan.external_instances.size() && logical < plan.binding_roles.size() &&
          plan.external_instances[logical] == instance) {
        const auto role = plan.binding_roles[logical];
        const auto kind = (role & kExternalInput) != 0 ? ValueKind::kInput
            : ((role & kConstant) != 0 ? ValueKind::kConstant : ValueKind::kNodeOutput);
        const auto value = GraphIdentity::value(plan.generation, kind, logical);
        if (const auto *binding = find_binding(&state, value);
            binding && binding->read.valid() && binding->description == tensor.description()) {
          HYLE_ASSIGN_OR_RETURN(auto source, binding->read.view());
          if (source.resource() == wanted.resource() &&
              source.native_handle() == wanted.native_handle() &&
              wanted.offset() >= source.offset()) {
            const auto offset = wanted.offset() - source.offset();
            if (offset <= source.size() && bytes <= source.size() - offset)
              return binding->read.subview(offset, bytes);
          }
        }
      }
    }
    // Borrowed outputs, aliases and projected/native carriers still resolve
    // their actual owner by the original qualified range search.
    for (auto &binding : state.resources.get()->bindings) {
      if (!binding.read.valid()) continue;
      HYLE_ASSIGN_OR_RETURN(auto source, binding.read.view());
      if (source.resource() != wanted.resource() || source.native_handle() != wanted.native_handle() ||
          wanted.offset() < source.offset()) continue;
      const auto offset = wanted.offset() - source.offset();
      if (offset <= source.size() && bytes <= source.size() - offset)
        return binding.read.subview(offset, bytes);
    }
  }
  return std::unexpected(Status::FailedPrecondition(
      "graph-owned or borrowed output has no independent storage lease"));
}

const compute::TensorDesc &output_description(const InvocationState &state,
    std::uint32_t instance) noexcept {
  return state.plan->domain_graph.valid()
      ? state.concrete_descriptions[state.plan->instance_values[instance]]
      : state.plan->instance_descriptions[instance];
}

Result<std::span<const std::uint32_t>>
collect_input_lease_candidates(InvocationState &invocation) noexcept {
  if (invocation.preparation_uses.load(std::memory_order_acquire) == 0)
    return std::unexpected(Status::FailedPrecondition(
        "read lookup requires guarded binding preparation"));
  if (!invocation.resources) return std::span<const std::uint32_t>{};
  auto &episode = *invocation.resources.get();
  auto &candidates = episode.input_lease_candidates;
  candidates.clear();
  if (episode.bindings.size() > UINT32_MAX ||
      candidates.capacity() < episode.bindings.size())
    return std::unexpected(Status::ResourceExhausted(
        "graph fixed read lookup capacity"));
  // Public binding/reset/projection mutations are excluded by the same
  // preparation guard as native hooks. Generated output projections change
  // readable/native_values, not this resource binding table.
  for (std::size_t index = 0; index < episode.bindings.size(); ++index) {
    const auto &binding = episode.bindings[index];
    if (!binding.exclusive && binding.read.valid())
      candidates.push_back(static_cast<std::uint32_t>(index));
  }
  return std::span<const std::uint32_t>(candidates);
}
void clear_input_lease_candidates(InvocationState &invocation) noexcept {
  if (invocation.resources)
    invocation.resources.get()->input_lease_candidates.clear();
}

const compute::ReadLease *input_lease(const InvocationState &invocation,
    compute::ConstTensorView tensor,
    std::span<const std::uint32_t> candidates) noexcept {
  if (!invocation.resources || !tensor.valid()) return nullptr;
  const auto wanted = tensor.storage();
  for (const auto index : candidates) {
    const auto &binding = invocation.resources.get()->bindings[index];
    // Exclusive bindings publish a read after success, but submit upgrades it
    // back to the original writer. A captured native read would prevent that
    // upgrade on a later cold rebind; the episode's write use owns this input.
    if (binding.exclusive || !binding.read.valid()) continue;
    const auto view = binding.read.view();
    if (!view || view->resource() != wanted.resource() ||
        view->native_handle() != wanted.native_handle() || wanted.offset() < view->offset())
      continue;
    const auto offset = wanted.offset() - view->offset();
    if (offset <= view->size() && wanted.size() <= view->size() - offset)
      return &binding.read;
  }
  return nullptr;
}

Result<compute::ConstTensorView> output_tensor(const InvocationState &state,
    std::uint32_t instance) noexcept {
  if (instance >= state.readable.size())
    return std::unexpected(Status::Internal("graph output instance is invalid"));
  if (state.readable[instance].valid()) return state.readable[instance];
  const auto &plan = *state.plan;
  auto root = instance;
  std::size_t offset = 0;
  while (plan.alias_sources[root] != kNoInPlaceSource) {
    if (plan.domain_graph.valid())
      return std::unexpected(Status::Unimplemented("native alias has no resolved byte offset"));
    if (plan.alias_offsets[root] > SIZE_MAX - offset)
      return std::unexpected(Status::Internal("native output alias byte offset overflow"));
    offset += plan.alias_offsets[root];
    root = plan.alias_sources[root];
  }
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    if (step.kind != PlanStepKind::kRegion) continue;
    for (std::size_t port = 0; port < step.output_count; ++port) {
      if (plan.node_outputs[step.output_begin + port] != root) continue;
      HYLE_ASSIGN_OR_RETURN(auto value, state.invocations[index].region.output_value(port));
      if (value.description != output_description(state, root) || value.reference.valid())
        return std::unexpected(Status::Internal("native output projection description differs"));
      if (!value.tensor.valid()) {
        if (step.region.output_contract(port).projection == RegionOutputProjection::kTensorView)
          return std::unexpected(Status::Internal("native graph output violated its memory projection contract"));
        return std::unexpected(Status::Unimplemented("native graph output has no memory view projection"));
      }
      const auto actual = value.tensor.storage().resource();
      const auto nominated = plan.storage_domains[plan.instance_storage[root]].resource;
      const auto &source = plan.targets[step.target.ordinal()].source;
      // Only the actual source can authorize a readonly export's distinct
      // owning resource. Coherence alone does not prove the storage family.
      if (value.tensor.description() != value.description ||
          !ExecutionSourceAccess::matches_read_resource(source, nominated, actual))
        return std::unexpected(Status::Internal("native output projection storage differs"));
      HYLE_RETURN_IF_ERROR(source.validate_resource(actual));
      HYLE_ASSIGN_OR_RETURN(auto bytes, output_description(state, instance).required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto view, value.tensor.storage().subview(offset, bytes));
      return compute::ConstTensorView::create(output_description(state, instance), view);
    }
  }
  return std::unexpected(Status::Unimplemented("graph output has no memory storage projection"));
}

Result<compute::ReadLease> output_storage(InvocationState &state,
    std::uint32_t instance, const compute::ResourceBudget &retention_budget = {},
    const compute::ResourceBudget &additional = {}) noexcept {
  const auto &plan = *state.plan;
  const auto retain = [&](compute::ReadLease &lease, std::uint32_t selected) -> Status {
    if (!retention_budget.valid() && !additional.valid()) return Status::Ok();
    const auto nominated = plan.storage_domains[plan.instance_storage[selected]].resource;
    if (retention_budget.valid())
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::retain_budget(lease, nominated, retention_budget));
    if (additional.valid())
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::retain_budget(lease, nominated, additional));
    return Status::Ok();
  };
  // An external may-alias pair can be bound to distinct storage. Export the
  // actual ready owned binding before following the cold alias permission;
  // only required/native aliases need that fallback projection. The original
  // range lookup still requires the same resource, handle and covered bytes.
  if (state.readable[instance].valid()) {
    auto bound = output_storage(state, state.readable[instance], instance);
    if (bound) { HYLE_RETURN_IF_ERROR(retain(*bound, instance)); return bound; }
    if (!bound.error().IsFailedPrecondition()) return bound;
  }
  auto root = instance;
  std::size_t offset = 0;
  while (plan.alias_sources[root] != kNoInPlaceSource) {
    const auto parent = plan.alias_sources[root];
    // Resolved memory views carry the actual specialized slice offset. Exact
    // plans may also project native-only aliases using their fixed offsets.
    std::size_t relative = plan.alias_offsets[root];
    if (state.readable[root].valid() && state.readable[parent].valid()) {
      const auto child_storage = state.readable[root].storage();
      const auto parent_storage = state.readable[parent].storage();
      if (child_storage.resource() != parent_storage.resource() ||
          child_storage.native_handle() != parent_storage.native_handle() ||
          child_storage.offset() < parent_storage.offset())
        return std::unexpected(Status::Internal("graph output alias has inconsistent storage"));
      relative = child_storage.offset() - parent_storage.offset();
    } else if (plan.domain_graph.valid()) {
      return std::unexpected(Status::Unimplemented(
          "native output alias has no resolved byte offset"));
    }
    if (relative > SIZE_MAX - offset)
      return std::unexpected(Status::Internal("graph output alias byte offset overflow"));
    offset += relative;
    root = parent;
  }
  if ((plan.instance_roles[root] & kGeneratedOutput) == 0) {
    if (!state.readable[instance].valid())
      return std::unexpected(Status::Unimplemented("graph output has no memory storage projection"));
    HYLE_ASSIGN_OR_RETURN(auto lease, output_storage(state, state.readable[instance], instance));
    HYLE_RETURN_IF_ERROR(retain(lease, instance));
    return lease;
  }
  HYLE_ASSIGN_OR_RETURN(const auto bytes, output_description(state, instance).required_bytes());
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    if (step.kind != PlanStepKind::kRegion) continue;
    for (std::size_t port = 0; port < step.output_count; ++port) {
      if (plan.node_outputs[step.output_begin + port] != root) continue;
      // This readonly projection borrows the original native slot retained by
      // GraphOutput. It performs no new admission, including after close.
      HYLE_ASSIGN_OR_RETURN(auto lease, state.invocations[index].region.output_lease(port));
      HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
      const auto nominated = plan.storage_domains[plan.instance_storage[root]].resource;
      const auto &source = plan.targets[step.target.ordinal()].source;
      if (!ExecutionSourceAccess::matches_read_resource(source, nominated, storage.resource()))
        return std::unexpected(Status::Internal("native output lease storage differs from its admitted source"));
      HYLE_RETURN_IF_ERROR(source.validate_resource(storage.resource()));
      HYLE_RETURN_IF_ERROR(retain(lease, root));
      return lease.subview(offset, bytes);
    }
  }
  return std::unexpected(Status::Internal("generated graph output has no native producer"));
}
} // namespace detail

namespace detail {
struct OutputStorageRetention final {
  explicit OutputStorageRetention(compute::StorageOwner value) noexcept
      : storage(std::move(value)) {}
  static void release(void *opaque) noexcept {
    auto *self = static_cast<OutputStorageRetention *>(opaque);
    const auto owner = self->storage;
    const auto allocator = owner.allocator();
    self->~OutputStorageRetention();
    allocator.deallocate(self, sizeof(OutputStorageRetention), alignof(OutputStorageRetention));
  }
  compute::StorageOwner storage;
  compute::CompletionRetention retirement;
};
struct OutputSourceState final : util::RefCounted<OutputSourceState> {
  OutputSourceState(InvocationState *original, compute::WriteLease *lease,
      compute::ReadLease *published,
      compute::ConstTensorView view, compute::detail::QueueRef queue_value,
      compute::StorageOwner storage, compute::Completion submission) noexcept
      : producer(original), writer(lease), ready(published), tensor(view), queue(std::move(queue_value)),
        storage_owner(std::move(storage)), completion(std::move(submission)) {
    retain(producer);
    producer->results.fetch_add(1, std::memory_order_relaxed);
  }
  void destroy() noexcept {
    auto storage = storage_owner;
    const auto allocator = storage.allocator();
    auto *original = producer;
    this->~OutputSourceState();
    allocator.deallocate(this, sizeof(OutputSourceState), alignof(OutputSourceState));
    if (original) {
      original->results.fetch_sub(1, std::memory_order_release);
      release(original);
    }
  }
  InvocationState *producer = nullptr;
  compute::WriteLease *writer = nullptr;
  compute::ReadLease *ready = nullptr;
  compute::ConstTensorView tensor;
  compute::detail::QueueRef queue;
  // This real resource/allocator owner outlives both the original Completion
  // and its leased backing, including the final metadata deallocation above.
  compute::StorageOwner storage_owner;
  // A real original read permission, declared after its actual resource owner
  // so it is destroyed first. Once detached no producer binding is borrowed.
  compute::ReadLease retained_output;
  compute::Completion completion;
  std::mutex observation;
  // One-shot borrowed identity only; never a back-reference owning consumer.
  InvocationState *consumer = nullptr;
  ValueId input{};
  bool preparing = false, submitted = false;
};
namespace {
Result<compute::detail::QueueRef> output_queue(const InvocationState &state) noexcept {
  const auto &plan = *state.plan;
  if (state.progress || plan.segments.size() != 1 || plan.required_queue_count != 1)
    return std::unexpected(Status::Unimplemented("output relay requires one original shared segment"));
  const auto lane = plan.segments.front().lane;
  if (!shared_lane(plan, lane) || plan.segments.front().blocks_host ||
      lane.ordinal() >= state.queues.size() || !state.queues[lane.ordinal()].valid())
    return std::unexpected(Status::Unimplemented("output relay requires one nonblocking prepared Queue"));
  HYLE_ASSIGN_OR_RETURN(auto endpoint,
      compute::detail::QueueSequence::completion_endpoint(state.queues[lane.ordinal()]));
  if (!endpoint.valid())
    return std::unexpected(Status::Unimplemented("output relay requires an actual asynchronous endpoint"));
  return state.queues[lane.ordinal()];
}
Result<compute::CompletionPoll> poll_output(void *opaque) noexcept {
  auto &source = *static_cast<OutputSourceState *>(opaque);
  std::lock_guard guard(source.observation);
  return source.completion.poll();
}
Status wait_output(void *opaque) noexcept {
  auto &source = *static_cast<OutputSourceState *>(opaque);
  std::lock_guard guard(source.observation);
  return source.completion.wait();
}
const InputSourceVTable output_source_table{
    [](void *opaque) noexcept { static_cast<OutputSourceState *>(opaque)->incref(); },
    [](void *opaque) noexcept { static_cast<OutputSourceState *>(opaque)->decref(); },
    [](void *opaque, const compute::detail::QueueRef *queue) noexcept
        -> Result<InputDependencyRoute> {
      const auto &source = *static_cast<OutputSourceState *>(opaque);
      if (!source.preparing)
        return std::unexpected(Status::FailedPrecondition("pending output requires its metadata-only Commands preparation"));
      if (!queue || !source.queue.same_owner(*queue))
        return std::unexpected(Status::Unimplemented("output relay requires its original actual Queue"));
      return InputDependencyRoute::kDeviceWait;
    },
    &poll_output, &wait_output,
    [](void *opaque, compute::detail::QueueSequence &sequence) noexcept {
      auto &source = *static_cast<OutputSourceState *>(opaque);
      if (!source.queue.same_owner(sequence))
        return compute::detail::EnqueueResult::rejected(
            Status::InvalidArgument("output relay sequence differs from its producer Queue"));
      std::lock_guard guard(source.observation);
      auto ready = source.completion.poll();
      if (!ready) return compute::detail::EnqueueResult::rejected(ready.error());
      if (*ready == compute::CompletionPoll::kComplete)
        return compute::detail::EnqueueResult::no_op();
      auto dependency = compute::detail::RetainedCompletion::native_dependency(source.completion);
      if (!dependency) return compute::detail::EnqueueResult::rejected(dependency.error());
      // Even on the same Queue this records an actual accepted dependency.
      // InputSource then reconciles the full original producer verdict before
      // publishing consumer success; the event is never that verdict itself.
      return sequence.enqueue_resolved_wait(*dependency);
    },
    &wait_output};
Result<InputSource> output_input_source(OutputSourceState *state) noexcept {
  if (!state) return std::unexpected(Status::FailedPrecondition("output source is empty"));
  return InputSource::acquire(state, &output_source_table, state);
}
Status check_output_consumer(Invocation &invocation, ValueId input,
    OutputSourceState *source, std::span<const QueueBinding> queues,
    bool require_binding) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (!source || state == source->producer)
    return Status::InvalidArgument("output relay source is empty or refers to its own consumer");
  const auto &plan = *state->plan;
  if (state->progress || plan.segments.size() != 1 || plan.required_queue_count != 1 ||
      !shared_lane(plan, plan.segments.front().lane) || plan.segments.front().blocks_host)
    return Status::Unimplemented("output relay consumer requires one nonblocking shared segment");
  for (const auto &step : plan.steps) {
    const auto runtime = step.runtime_properties();
    if (step.emits_command() && (runtime.mode != compute::ProviderRuntimeMode::kSharedQueue ||
        runtime.replay_class != compute::ProviderReplayClass::kPrepared || runtime.blocks_host))
      return Status::Unimplemented("output relay requires prepared nonblocking shared commands");
  }
  if (!queues.empty() && (queues.size() != 1 || !queues.front().queue ||
      queues.front().lane != plan.segments.front().lane ||
      !source->queue.same_owner(*queues.front().queue)))
    return Status::InvalidArgument("output relay requires the producer's actual Queue and consumer lane");
  if (source->consumer && (source->consumer != state || source->input != input))
    return Status::FailedPrecondition("output source already names a different original consumer input");
  if (require_binding) {
    HYLE_ASSIGN_OR_RETURN(auto owner, output_input_source(source));
    if (!input_source_matches(*state, input, owner))
      return Status::FailedPrecondition("output relay input owner was reset or replaced");
    HYLE_ASSIGN_OR_RETURN(auto bound, RegionAccess::has_input_binding(state, input, source->tensor));
    if (!bound) return Status::FailedPrecondition("output relay input storage was reset or replaced");
    const auto *binding = find_binding(state, input);
    if (!binding || binding->borrowed || binding->exclusive || binding->read.valid() ||
        binding->permissionless || binding->description != source->tensor.description())
      return Status::FailedPrecondition("output relay lost its original pending read requirement");
  }
  return Status::Ok();
}
} // namespace
OutputSource::OutputSource(OutputSource &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
OutputSource &OutputSource::operator=(OutputSource &&other) noexcept {
  if (this != &other) {
    if (state_) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
OutputSource::~OutputSource() { if (state_) state_->decref(); }
Result<OutputSource> OutputSource::create(Invocation &invocation, ValueId output,
    compute::Completion &&submission, const compute::StorageOwner &storage,
    const compute::LeasePool &pool) noexcept {
  auto *state = InvocationAccess::state(invocation);
  if (!state || !storage || !state->observation.owns(submission))
    return std::unexpected(Status::InvalidArgument("output source requires its exact submitted Invocation proxy and storage owner"));
  const auto phase = state->phase.load(std::memory_order_acquire);
  if (phase != InvocationPhase::kRunning &&
      !(phase == InvocationPhase::kIdle && state->outputs_ready.load(std::memory_order_acquire)))
    return std::unexpected(Status::FailedPrecondition("output source requires an accepted successful-or-pending producer"));
  const auto &plan = *state->plan;
  const auto published = std::find(plan.output_ids.begin(), plan.output_ids.end(), output);
  if (published == plan.output_ids.end())
    return std::unexpected(Status::InvalidArgument("output source value is not an original published output"));
  auto *binding = find_binding(state, output);
  if (!binding || binding->input || !binding->exclusive || !binding->write.valid() ||
      binding->scope != BindingScope::kCaller)
    return std::unexpected(Status::Unimplemented("output relay requires its original caller-owned output writer"));
  const auto &description = binding->description;
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if (description.data_type() != compute::DataType::kInt32 ||
      !description.is_plain() || !description.is_dense() || !bytes)
    return std::unexpected(Status::Unimplemented("output relay requires a nonempty dense I32 output"));
  const auto instance = plan.external_instances[output.ordinal()];
  if (instance >= state->writable.size() || !state->writable[instance].valid())
    return std::unexpected(Status::Unimplemented("output relay requires fixed writable output storage"));
  const auto domain = plan.instance_storage[instance];
  if (domain >= plan.storage_domains.size() || !plan.storage_domains[domain].owner ||
      plan.storage_domains[domain].owner.ref() != storage.ref() ||
      plan.storage_domains[domain].owner.context_identity() != storage.context_identity())
    return std::unexpected(Status::FailedPrecondition(
        "output relay requires the original Plan to retain its actual storage owner"));
  HYLE_ASSIGN_OR_RETURN(auto writer_view, compute::detail::LeaseAccess::ordered_view(binding->write));
  const auto tensor = state->writable[instance].as_const();
  const auto view = tensor.storage();
  if (tensor.description() != description || view.resource() != storage.ref() ||
      view.resource() != writer_view.resource() || view.native_handle() != writer_view.native_handle() ||
      view.offset() != writer_view.offset() || view.size() != bytes || writer_view.size() != bytes)
    return std::unexpected(Status::InvalidArgument("output relay writer/storage/extent differs from its actual output"));
  HYLE_ASSIGN_OR_RETURN(auto allocator, compute::detail::LeaseAccess::freeze_owner_allocator(pool, binding->write));
  const auto system = mem::system_allocator();
  if (!storage.owns_allocator(allocator) && !(allocator.alloc_fn() == system.alloc_fn() &&
      allocator.free_fn() == system.free_fn() && allocator.realloc_fn() == system.realloc_fn() &&
      allocator.state() == system.state()))
    return std::unexpected(Status::InvalidArgument("output storage owner cannot retain the original pool allocator"));
  HYLE_ASSIGN_OR_RETURN(auto queue, output_queue(*state));
  HYLE_ASSIGN_OR_RETURN(auto context, ExecutionSourceAccess::compute(plan.targets[plan.lanes[plan.segments.front().lane.ordinal()].target.ordinal()].source));
  if (context->identity_key() != storage.context_identity())
    return std::unexpected(Status::InvalidArgument("output storage owner belongs to another Context"));
  // Check the exact known retained/native bridge before consuming the proxy.
  HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(submission));
  (void)dependency;
  auto *memory = storage.allocator().allocate(sizeof(OutputSourceState), alignof(OutputSourceState));
  if (!memory) return std::unexpected(Status::ResourceExhausted("output source cold owner metadata"));
  auto *retention_memory = storage.allocator().allocate(sizeof(OutputStorageRetention), alignof(OutputStorageRetention));
  if (!retention_memory) {
    storage.allocator().deallocate(memory, sizeof(OutputSourceState), alignof(OutputSourceState));
    return std::unexpected(Status::ResourceExhausted("output source storage retirement metadata"));
  }
  auto *retained_storage = ::new (retention_memory) OutputStorageRetention(storage);
  const auto attached = retained_storage->retirement.attach(submission, retained_storage,
      &OutputStorageRetention::release);
  if (!attached.ok()) {
    OutputStorageRetention::release(retained_storage);
    storage.allocator().deallocate(memory, sizeof(OutputSourceState), alignof(OutputSourceState));
    return std::unexpected(attached);
  }
  auto *owner = ::new (memory) OutputSourceState(state, &binding->write, &binding->read, tensor,
      std::move(queue), storage, std::move(submission));
  return OutputSource(owner);
}
Result<compute::CompletionPoll> OutputSource::poll() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("output source is empty"));
  return poll_output(state_);
}
Status OutputSource::wait() noexcept {
  if (!state_) return Status::FailedPrecondition("output source is empty");
  return wait_output(state_);
}
Status OutputSource::detach() noexcept {
  if (!state_) return Status::FailedPrecondition("output source is empty");
  auto &source = *state_;
  std::lock_guard guard(source.observation);
  if (!source.producer) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto observed, source.completion.poll());
  if (observed == compute::CompletionPoll::kPending)
    return Status::Unavailable("output producer is still pending");
  auto *original = source.producer;
  if (original->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
      !original->outputs_ready.load(std::memory_order_acquire) ||
      original->preparation_uses.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("successful output producer has not actually retired");
  // results kept both original binding objects stable through this point.
  // No native producer write remains, but ordered read uses may still own the
  // block. Keep the original binding's ready alias for ordinary output().
  if (!source.writer->valid() && !source.ready->valid())
    return Status::FailedPrecondition("successful output has no original read permission");
  if (source.writer->valid()) {
    HYLE_ASSIGN_OR_RETURN(auto published,
        compute::detail::LeaseAccess::publish_after_writes(*source.writer));
    *source.ready = std::move(published);
  }
  source.retained_output = *source.ready;
  // The successful Completion is already settled and caches this generation's
  // complete verdict. Never consult the producer's next observation slot.
  source.writer = nullptr; source.ready = nullptr; source.producer = nullptr;
  original->results.fetch_sub(1, std::memory_order_release);
  release(original);
  return Status::Ok();
}
Status bind_output(Invocation &invocation, ValueId input, const OutputSource &source) noexcept {
  HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, input, source.state_, {}, false));
  if (source.state_->consumer)
    return Status::FailedPrecondition("output source has already bound its one original consumer");
  auto *state = InvocationAccess::state(invocation);
  auto *binding = find_binding(state, input);
  if (!binding || !binding->input)
    return Status::InvalidArgument("output relay requires a prepared original external input resource");
  if (source.state_->submitted)
    return Status::FailedPrecondition("output source has already submitted its one original consumer");
  HYLE_ASSIGN_OR_RETURN(auto owner, output_input_source(source.state_));
  HYLE_RETURN_IF_ERROR(InputAccess::bind(invocation, input, source.state_->tensor, std::move(owner)));
  // Unlike an immutable ready weight, these bytes require the exact ordered
  // read admission below. Ordinary Invocation::submit sees a missing read and
  // rejects before native acceptance instead of treating this as a raw view.
  binding = find_binding(state, input);
  binding->borrowed = false; binding->permissionless = false;
  binding->exclusive = false; binding->description = source.state_->tensor.description();
  source.state_->consumer = state;
  source.state_->input = input;
  return Status::Ok();
}
Status prepare_output(Invocation &invocation, ValueId input, const OutputSource &source,
    std::span<const QueueBinding> queues) noexcept {
  HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, input, source.state_, queues, true));
  if (queues.size() != 1) return Status::InvalidArgument("output relay requires one actual Queue");
  if (source.state_->submitted) return Status::FailedPrecondition("output source consumer was already submitted");
  source.state_->preparing = true;
  struct EndPreparation final {
    OutputSourceState *source;
    ~EndPreparation() { source->preparing = false; }
  } end{source.state_};
  return invocation.prepare_replay(queues,
      {.mode = ReplayMode::kCommands, .host_wait = HostWaitPolicy::kReject});
}
Result<compute::Completion> submit_output(Invocation &invocation, ValueId input,
    const OutputSource &source, std::span<const QueueBinding> queues) noexcept {
  HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, input, source.state_, queues, true));
  if (queues.size() != 1 || invocation.replay_route() != ReplayRoute::kCommands)
    return std::unexpected(Status::FailedPrecondition("output relay needs its prepared Commands route"));
  if (source.state_->submitted)
    return std::unexpected(Status::FailedPrecondition("output source consumer was already submitted"));
  const OrderedGraphSubmission ordered{
      [](void *opaque, ExecutionLaneId, compute::detail::QueueSequence &sequence,
          compute::LeaseUseBatch &uses) noexcept -> Status {
        auto &owner = *static_cast<OutputSourceState *>(opaque);
        if (!owner.queue.same_owner(sequence))
          return Status::InvalidArgument("ordered output reader has another actual Queue");
        std::lock_guard guard(owner.observation);
        if (owner.retained_output.valid())
          return compute::detail::LeaseAccess::read_unique(uses, owner.retained_output);
        // result() may have published this very writer after producer success.
        // That is an original ready permission, not a synthesized conversion.
        if (owner.writer->valid())
          return compute::detail::LeaseAccess::ordered_read_unique(uses, *owner.writer);
        return compute::detail::LeaseAccess::read_unique(uses, *owner.ready);
      },
      [](void *, compute::Completion &&completion) noexcept { return std::move(completion); },
      source.state_,
      [](void *opaque, ValueId value) noexcept {
        return static_cast<OutputSourceState *>(opaque)->input == value;
      }};
  auto result = RegionAccess::submit(invocation, queues, ordered);
  if (result) source.state_->submitted = true;
  return result;
}
} // namespace detail

GraphOutput::GraphOutput(detail::InvocationState *state,
    std::uint32_t instance) noexcept : state_(state), instance_(instance) {
  detail::retain(state_);
  state_->results.fetch_add(1, std::memory_order_relaxed);
}
GraphOutput::GraphOutput(const GraphOutput &other) noexcept
    : state_(other.state_), instance_(other.instance_) {
  if (state_) { detail::retain(state_); state_->results.fetch_add(1, std::memory_order_relaxed); }
}
GraphOutput &GraphOutput::operator=(const GraphOutput &other) noexcept {
  if (this != &other) { GraphOutput copy(other); *this = std::move(copy); }
  return *this;
}
GraphOutput::GraphOutput(GraphOutput &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      instance_(std::exchange(other.instance_, UINT32_MAX)) {}
GraphOutput &GraphOutput::operator=(GraphOutput &&other) noexcept {
  if (this != &other) {
    reset(); state_ = std::exchange(other.state_, nullptr);
    instance_ = std::exchange(other.instance_, UINT32_MAX);
  }
  return *this;
}
GraphOutput::~GraphOutput() { reset(); }
void GraphOutput::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    state->results.fetch_sub(1, std::memory_order_release);
    detail::release(state);
  }
  instance_ = UINT32_MAX;
}
Result<compute::TensorDesc> GraphOutput::description() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph output is empty"));
  return detail::output_description(*state_, instance_);
}
Result<compute::ConstTensorView> GraphOutput::tensor() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph output is empty"));
  return detail::output_tensor(*state_, instance_);
}
Result<compute::ReadLease> GraphOutput::storage_lease() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph output is empty"));
  return detail::output_storage(*state_, instance_);
}
Result<compute::ReadLease> GraphOutput::storage_lease(
    const compute::ResourceBudget &retention_budget) const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph output is empty"));
  return detail::output_storage(*state_, instance_, retention_budget);
}
Result<compute::ReadLease> GraphOutput::storage_lease(
    const compute::ResourceBudget &retention_budget,
    const compute::ResourceBudget &additional) const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph output is empty"));
  return detail::output_storage(*state_, instance_, retention_budget, additional);
}

Status Invocation::prepare_resources(ResourceOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state_));
  if (!options.allocator.valid()) return Status::InvalidArgument("graph resource allocator is invalid");
  if (state_->resources) return Status::Ok();
  const auto actual = executable();
  const auto inputs = actual.input_count(), outputs = actual.output_count();
  if (inputs > UINT32_MAX - 1 || outputs > UINT32_MAX - 1 - inputs)
    return Status::ResourceExhausted("graph resource binding capacity");
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>(inputs + outputs + 1),
      .allocator = allocator, .resource_budget = options.resource_budget}));
  auto *episode = allocator.construct<detail::ResourceEpisode>(allocator);
  if (!episode) return Status::ResourceExhausted("graph resource episode metadata");
  detail::ResourceEpisodeOwner candidate(episode);
  episode->uses = std::move(uses);
  try {
    episode->bindings.resize(inputs);
    episode->bindings.reserve(inputs + outputs);
    for (std::size_t i = 0; i < inputs; ++i) {
      HYLE_ASSIGN_OR_RETURN(episode->bindings[i].value, actual.input_id(i));
      episode->bindings[i].input = true;
    }
    for (std::size_t i = 0; i < outputs; ++i) {
      HYLE_ASSIGN_OR_RETURN(auto ownership, actual.output_ownership(i));
      if (ownership != OutputOwnership::kCaller) continue;
      HYLE_ASSIGN_OR_RETURN(auto value, actual.output_id(i));
      if (std::none_of(episode->bindings.begin(), episode->bindings.end(),
                       [&](const auto &binding) { return binding.value == value; }))
        episode->bindings.push_back({.value = value});
    }
    episode->fixed_binding_count = episode->bindings.size();
    episode->input_lease_candidates.reserve(episode->bindings.capacity());
    episode->unbind_steps.resize(state_->plan->steps.size());
    episode->nominated_writers.assign(state_->plan->steps.size(), UINT32_MAX);
    episode->unbind_instances.resize(state_->readable.size());
    if (episode->fixed_binding_count != 0) {
      episode->binding_slots.assign(state_->plan->descriptions.size(), UINT32_MAX);
      for (std::size_t slot = 0; slot < episode->fixed_binding_count; ++slot)
        episode->binding_slots[episode->bindings[slot].value.ordinal()] =
            static_cast<std::uint32_t>(slot);
    }
  } catch (...) { return Status::ResourceExhausted("graph fixed resource metadata"); }
  HYLE_RETURN_IF_ERROR(detail::prepare_submission_lifetime(*state_, allocator));
  state_->resources = std::move(candidate);
  return Status::Ok();
}
Status Invocation::unbind(std::span<const ValueId> values) noexcept {
  return detail::RegionAccess::withdraw_bindings(*this, values);
}
Status Invocation::bind(ValueId value, compute::TensorDesc description,
                        compute::ReadLease lease) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *episode, detail::require_resources(state_));
  (void)episode;
  HYLE_RETURN_IF_ERROR(detail::check_binding_replacement(*state_, value));
  auto *binding = detail::find_binding(state_, value);
  if (!binding || !binding->input) return Status::InvalidArgument("read lease must bind a graph input");
  HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
  HYLE_ASSIGN_OR_RETURN(auto tensor,
      compute::ConstTensorView::from_initialized_storage(description, storage));
  HYLE_RETURN_IF_ERROR(bind(value, tensor));
  binding = detail::find_binding(state_, value);
  binding->read = std::move(lease); binding->description = description;
  binding->exclusive = false; binding->permissionless = false; binding->borrowed = false;
  return Status::Ok();
}
Status Invocation::bind(ValueId value, compute::TensorDesc description,
                        compute::WriteLease &&lease) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *episode, detail::require_resources(state_));
  (void)episode;
  HYLE_RETURN_IF_ERROR(detail::check_binding_replacement(*state_, value));
  auto *binding = detail::find_binding(state_, value);
  if (!binding) return Status::InvalidArgument("write lease must bind an external graph value");
  HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
  HYLE_RETURN_IF_ERROR(bind(value, tensor));
  binding = detail::find_binding(state_, value);
  binding->write = std::move(lease); binding->description = description;
  binding->exclusive = true; binding->permissionless = false; binding->borrowed = false;
  episode->writer_publication_complete = false;
  return Status::Ok();
}
Result<detail::SubmissionCursor> detail::begin_submission(Invocation &invocation,
    std::span<const QueueBinding> queues, OrderedGraphSubmission ordered) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (!state->submission_lifetime)
    return std::unexpected(Status::FailedPrecondition("graph submission lifetime is not prepared"));
  auto *episode = state->resources.get();
  if (!ordered.valid())
    return std::unexpected(Status::InvalidArgument("ordered graph submission requires source owner, input qualification and mapping"));
  if (ordered.map && !episode)
    return std::unexpected(Status::FailedPrecondition("ordered graph submission requires prepared resources"));
  compute::LeaseUseBatch batch;
  if (episode) {
    // Rearm before any upgrade: a later failed admission can leave a prefix
    // of original readers converted back to writers for a result() retry.
    episode->writer_publication_complete = false;
    for (auto &binding : episode->bindings) {
      if (binding.exclusive && binding.read.valid()) {
        HYLE_ASSIGN_OR_RETURN(binding.write, binding.read.try_write());
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto admitted_batch, episode->uses.begin());
    batch = std::move(admitted_batch);
    auto &permissions = episode->permissions;
    permissions.reset();
    for (const auto &binding : episode->bindings) {
      if(binding.projected_appender){
        if(!permissions.enabled())return std::unexpected(Status::Internal("projected append needs its cold permission group"));
        HYLE_RETURN_IF_ERROR(permissions.add(*binding.projected_appender));
        continue;
      }
      if(binding.projected_writer){
        if(permissions.enabled())HYLE_RETURN_IF_ERROR(permissions.add(*binding.projected_writer));
        else HYLE_RETURN_IF_ERROR(batch.write(*binding.projected_writer));
        continue;
      }
      if (binding.borrowed || binding.permissionless) continue;
      if (!binding.exclusive && !binding.read.valid() && binding.input &&
          ordered.pending_input && ordered.pending_input(ordered.owner, binding.value)) continue;
      if (permissions.enabled()) {
        HYLE_RETURN_IF_ERROR(binding.exclusive ? permissions.add(binding.write) : permissions.add(binding.read));
      } else {
        HYLE_RETURN_IF_ERROR(binding.exclusive ? batch.write(binding.write) : batch.read(binding.read));
      }
    }
    if (permissions.enabled()) HYLE_RETURN_IF_ERROR(permissions.admit(batch));
  }
  HYLE_ASSIGN_OR_RETURN(auto cursor, SubmissionCursor::begin(invocation, queues, false));
  if (episode) episode->batch = std::move(batch);
  const auto begun = begin_submission_lifetime(*state, ordered);
  if (!begun.ok()) {
    if (episode) { episode->batch = {}; episode->permissions.reset(); }
    cursor.reject_successors(begun);
    // No lifetime or native command was admitted. Close the cursor's original
    // phase/exclusion gate without entering its emergency abandonment path.
    auto rejected = cursor.finish();
    (void)rejected;
    return std::unexpected(begun);
  }
  begin_input_sources(*state);
  return cursor;
}

Result<compute::Completion> detail::finish_submission(InvocationState &state,
    SubmissionCursor &cursor) noexcept {
  record_submission_error(state, cursor.submission_error());
  auto completion = cursor.finish(true);
  const auto ordered = state.submission_lifetime.get()->ordered;
  if (!completion && ordered.map) {
    const auto error = completion.error();
    completion = ordered.map(ordered.owner, compute::Completion::completed(error));
    retain_submission_completion(&state, state.plan->segments.size(), *completion);
    return completion;
  }
  if (completion) {
    if (ordered.map) *completion = ordered.map(ordered.owner, std::move(*completion));
    retain_submission_completion(&state, state.plan->segments.size(), *completion);
  } else {
    record_submission_error(state, completion.error());
  }
  return completion;
}

Result<compute::Completion> detail::RegionAccess::submit(Invocation &invocation,
    std::span<const QueueBinding> queues, OrderedGraphSubmission ordered) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (state->progress) return state->progress.submit(invocation, queues, ordered);
  HYLE_ASSIGN_OR_RETURN(auto cursor, begin_submission(invocation, queues, ordered));
  const SharedSequenceFn before = ordered.admit ? &admit_submission_sequence : nullptr;
  while (!cursor.done()) {
    auto advanced = cursor.advance(true, &retain_submission_completion, state, before, state);
    if (!advanced) cursor.reject_successors(std::move(advanced).error());
  }
  if (cursor.submission_error().ok()) {
    const auto ready = wait_input_results(*state);
    if (!ready.ok()) cursor.reject_successors(ready);
  }
  auto completion = finish_submission(*state, cursor);
  finish_submission_lifetime(*state,
      completion ? Status::Ok() : Status{completion.error()});
  if (!completion) return std::unexpected(std::move(completion).error());
  return state->observation.publish(std::move(*completion), true);
}

Result<GraphOutput> Invocation::result(ValueId value) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("graph invocation is empty"));
  if (state_->phase.load(std::memory_order_acquire) != detail::InvocationPhase::kIdle ||
      !state_->outputs_ready.load(std::memory_order_acquire))
    return std::unexpected(Status::FailedPrecondition(
        "graph outputs require a successful completed submission"));
  const auto &plan = *state_->plan;
  if (!value.valid() || detail::GraphIdentity::generation(value) != plan.generation ||
      value.ordinal() >= plan.descriptions.size())
    return std::unexpected(Status::InvalidArgument("value does not belong to this executable"));
  const auto role = plan.binding_roles[value.ordinal()];
  const auto expected = (role & detail::kExternalInput) != 0 ? ValueKind::kInput
      : ((role & detail::kConstant) != 0 ? ValueKind::kConstant : ValueKind::kNodeOutput);
  if (value.kind() != expected)
    return std::unexpected(Status::InvalidArgument("value does not belong to this executable"));
  if ((role & detail::kExternalOutput) == 0)
    return std::unexpected(Status::InvalidArgument("value is not a published graph output"));
  const auto instance = plan.external_instances[value.ordinal()];
  if (instance == detail::kNoWorkspaceRequest || instance >= plan.instance_values.size())
    return std::unexpected(Status::Internal("published graph output has no physical instance"));
  if (state_->resources) {
    auto &episode = *state_->resources.get();
    // Anonymous projected-page records only hold reads or writer pointers;
    // every owned WriteLease remains in the fixed caller-binding prefix.
    if (!episode.writer_publication_complete) {
      for (std::size_t index = 0; index < episode.fixed_binding_count; ++index) {
        auto &binding = episode.bindings[index];
        if (binding.write.valid()) {
          HYLE_ASSIGN_OR_RETURN(binding.read, binding.write.publish());
        }
      }
      // A failed publication leaves this false so the remaining original
      // writers are retried, with the already published reads intact.
      episode.writer_publication_complete = true;
    }
  }
  return GraphOutput(state_, instance);
}
Status Invocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return in_flight() ? Status::Unavailable("graph invocation still has native work") : Status::Ok();
}
Status Invocation::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  Status terminal;
  if (state_->progress) terminal = state_->progress.wait(*state_);
  // The slot's actual native guards can retire before the public observation's
  // callback tail returns. Joining progress alone must not publish drain early.
  terminal.Update(state_->observation.wait());
  auto phase = state_->phase.load(std::memory_order_acquire);
  while (phase == detail::InvocationPhase::kRunning) {
    state_->phase.wait(phase, std::memory_order_acquire);
    phase = state_->phase.load(std::memory_order_acquire);
  }
  terminal.Update(detail::submission_lifetime_status(*state_));
  return terminal;
}
} // namespace hyle::graph
