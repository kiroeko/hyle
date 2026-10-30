#include "hyle/graph/executable.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/detail/replay_access.h"
#include "hyle/graph/result.h"
#include "hyle/graph/source.h"
#include "hyle/graph/detail/source_binding_access.h"
#include "hyle/graph/detail/input_source.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/completion_retention.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/resource_budget_access.h"
#include "hyle/compute/detail/append_lease.h"
#include "hyle/container/small_vector.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/spin_lock.h"
#include "plan_state.h"
#include "input_sources.h"
#include "resource_episode.h"
#include "submission_internal.h"
#include "submission_lifetime.h"
#include "slots_region.h"
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
  mem::ConstBufferView pending_range{};
  bool pending_effect = false;
};

struct ResourceEpisode final {
  struct MetadataQueue final {
    const void *context = nullptr;
    ExecutionLaneId lane{};
    std::shared_ptr<compute::Queue> queue;
  };
  explicit ResourceEpisode(mem::AllocatorRef selected) noexcept
      : allocator(selected), bindings(mem::StdAllocator<LeaseBinding>(selected)),
        binding_slots(mem::StdAllocator<std::uint32_t>(selected)),
        input_lease_candidates(mem::StdAllocator<std::uint32_t>(selected)),
        unbind_steps(mem::StdAllocator<std::uint8_t>(selected)),
        unbind_instances(mem::StdAllocator<std::uint8_t>(selected)),
        nominated_writers(mem::StdAllocator<std::uint32_t>(selected)),
        metadata_queues(mem::StdAllocator<MetadataQueue>(selected)),
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
  // Sparse original Context/lane owners, independent of ready permissions.
  // Configuration reset drops guards but retains these cold stable wrappers.
  std::vector<MetadataQueue, mem::StdAllocator<MetadataQueue>> metadata_queues;
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
Result<std::shared_ptr<compute::Queue>> RegionAccess::metadata_queue(
    Invocation &invocation, ExecutionLaneId lane, ResourceOptions options) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (!options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("metadata Queue needs a valid cold allocator"));
  HYLE_ASSIGN_OR_RETURN(auto source, execution_source(invocation.executable(), lane));
  HYLE_ASSIGN_OR_RETURN(const auto *context, ExecutionSourceAccess::compute(source));
  if (!context || !context->is_open())
    return std::unexpected(Status::FailedPrecondition("metadata Queue's original Context is closed"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources(options));
  auto &episode = *state->resources.get();
  for (const auto &entry : episode.metadata_queues) {
    if (entry.context != context->identity_key() || entry.lane != lane) continue;
    if (!entry.queue || !entry.queue->is_open())
      return std::unexpected(Status::FailedPrecondition("original metadata Queue is closed"));
    return entry.queue;
  }
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  try {
    episode.metadata_queues.reserve(episode.metadata_queues.size() + 1);
    HYLE_ASSIGN_OR_RETURN(auto queue, context->create_queue());
    // The budget allocator self-retains each allocation, including the shared
    // control block after this Invocation dies. Arbitrary allocator owners
    // retain their ordinary external lifetime contract.
    auto shared = std::allocate_shared<compute::Queue>(
        mem::StdAllocator<compute::Queue>(allocator), std::move(queue));
    episode.metadata_queues.push_back({context->identity_key(), lane, shared});
    return shared;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Graph metadata Queue owner"));
  } catch (...) {
    return std::unexpected(Status::Internal("Graph metadata Queue preparation failed"));
  }
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
Status RegionAccess::release_read_projections(Invocation &invocation,
    std::span<const ValueId> values) noexcept {
  return withdraw_bindings_impl(invocation, values, true);
}
std::span<const ExternalInputConsumer>
projected_input_consumers(const PlanState &, std::uint32_t) noexcept;
Status RegionAccess::withdraw_bindings_impl(Invocation &invocation,
    std::span<const ValueId> values, bool ready_projection, const InputSource *source_owner) noexcept {
  auto *state = InvocationAccess::state(invocation);
  if (ready_projection || source_owner) {
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
    if (ready_projection || source_owner) break;
    if (binding.scope == BindingScope::kWrittenProjection ||
        (binding.scope == BindingScope::kProjected && !binding.read.valid() &&
         !binding.permissionless))
      return Status::FailedPrecondition("take original page/source controls before unbinding caller ports");
  }
  for (auto value : values) {
    const auto *binding = find_binding(state, value);
    if (!binding) return Status::InvalidArgument("unbind needs this executable's caller input or output");
    if (source_owner) {
      if (!binding->input || !input_source_matches(*state, value, *source_owner) ||
          (binding->scope != BindingScope::kCaller && binding->scope != BindingScope::kProjected) ||
          binding->borrowed || binding->exclusive || binding->write.valid() ||
          binding->permissionless || binding->projected_writer || binding->projected_appender)
        return Status::FailedPrecondition("source withdrawal differs from its exact original input authority");
    } else if (ready_projection) {
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
  if (ready_projection || source_owner) {
    for (const auto value : values) {
      const auto *binding = find_binding(state, value);
      if (binding->scope != BindingScope::kProjected) continue;
      for (const auto &consumer : projected_input_consumers(plan, value.ordinal())) {
        const auto &step = plan.steps[consumer.step];
        const auto &arguments = state->step_bindings[consumer.step];
        bool preserves = binding->permission_changes_preserve_commands;
        if (idle && !preserves && arguments.stable_binding_identity && !arguments.native_inputs &&
            step.kind == PlanStepKind::kOperation) {
          HYLE_ASSIGN_OR_RETURN(preserves,
              compute::detail::OperationPreparationAccess::preserves_capture_bindings(
                  state->invocations[consumer.step].operation));
        }
        if (source_owner && idle && !preserves && arguments.stable_binding_identity &&
            step.kind == PlanStepKind::kRegion && step.region.valid()) {
          const auto &prepared = step.region.prepared_;
          const auto *capture = prepared.vtable->query_extension
              ? static_cast<const RegionCaptureBindingsVTable *>(prepared.vtable->query_extension(
                  prepared.state, kRegionCaptureBindingsExtensionId)) : nullptr;
          preserves = capture && capture->preserves_private_arguments &&
              capture->preserves_private_arguments(prepared.state,
                  state->invocations[consumer.step].region.state_);
        }
        if (!idle || !preserves || !arguments.stable_binding_identity)
          steps[consumer.step] |= discard_recording;
      }
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
  if (ready_projection || source_owner) {
    for (const auto value : values) {
      if (find_binding(state, value)->scope != BindingScope::kProjected) continue;
      const auto physical = root(plan.external_instances[value.ordinal()]);
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
  if (!state.resources) return;
  auto &spans = state.resources.get()->configuration_spans;
  if (state.bindings_prepared) {
    // Seed once from complete physical spans, before a permission-only revoke.
    // Later revokes/resets retain that candidate but cannot revive a discarded
    // one. Opaque carriers still require their complete binding preparation.
    // Source routes and readiness are prepared separately for each episode;
    // retaining these spans grants neither producer identity nor permission.
    spans = true;
    for (std::size_t i = 0; i < state.step_bindings.size(); ++i)
      if (state.step_bindings[i].native_inputs || (state.plan->steps[i].region.valid() &&
          !RegionAccess::borrows_tensor_views_only(state.plan->steps[i].region))) {
        spans = false; break;
      }
  }
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
    bool reusable = episode.configuration_spans;
    for (const auto &binding : state.step_bindings)
      reusable = reusable && !binding.native_inputs;
    for (const auto &binding : episode.bindings) {
      // Fixed projected capacities can keep only their old physical spans.
      // Fresh binds still validate every current page/writer and compare the
      // complete replacement tuple before that candidate can be used.
      if (binding.value.valid() && binding.scope != BindingScope::kCaller &&
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
  // The common read bind above kept only a previously eligible, unchanged
  // physical tuple. This owner and its generation are still freshly installed.
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
    bool pending,bool validate_only)noexcept{
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
  if(validate_only)return Status::Ok();
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
Status RegionAccess::validate_pending_projected(Invocation&invocation,ValueId value,
    compute::ConstTensorView tensor,std::span<const ProjectedReader>readers)noexcept{
  return bind_projected_impl(invocation,value,tensor,{},readers,true,true);
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
bool same_resource_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.valid() && b.valid() && a.resource() == b.resource() &&
      a.native_handle() == b.native_handle() && a.offset() == b.offset() && a.size() == b.size();
}
bool contains_resource_range(mem::ConstBufferView outer, mem::ConstBufferView inner) noexcept {
  return outer.valid() && inner.valid() && inner.size() && outer.resource() == inner.resource() &&
      outer.native_handle() == inner.native_handle() && inner.offset() >= outer.offset() &&
      inner.offset() - outer.offset() <= outer.size() &&
      inner.size() <= outer.size() - (inner.offset() - outer.offset());
}
bool overlaps_resource_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  if (!a.valid() || !b.valid() || !a.size() || !b.size() ||
      a.resource() != b.resource() || a.native_handle() != b.native_handle()) return false;
  return a.offset() <= b.offset() ? b.offset() - a.offset() < a.size() : a.offset() - b.offset() < b.size();
}
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
    if(binding.pending_range.valid()) {
      for(const auto *page:writers) {
        HYLE_ASSIGN_OR_RETURN(auto range,page->view());
        if(overlaps_resource_range(binding.pending_range,range))
          return Status::InvalidArgument("projected writer overlaps an original pending effect read");
      }
      for(const auto *append:appenders) {
        HYLE_ASSIGN_OR_RETURN(auto range,append->view());
        if(overlaps_resource_range(binding.pending_range,range))
          return Status::InvalidArgument("projected append overlaps an original pending effect read");
      }
    }
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
Status validate_pending_write_reads(const ResourceEpisode &episode, compute::ConstTensorView capacity,
    std::span<const mem::ConstBufferView> reads,
    std::span<const compute::WriteLease *const> writers,
    std::span<const compute::detail::AppendLease *const> appenders) noexcept {
  for(std::size_t i=0;i<reads.size();++i) {
    if(!contains_resource_range(capacity.storage(),reads[i]))
      return Status::InvalidArgument("pending effect read is outside its original capacity");
    for(std::size_t j=0;j<i;++j)if(same_resource_range(reads[i],reads[j]))
      return Status::InvalidArgument("pending effect read is duplicated");
    for(const auto *writer:writers) {
      HYLE_ASSIGN_OR_RETURN(auto range,writer->view());
      if(overlaps_resource_range(reads[i],range))
        return Status::InvalidArgument("pending effect read overlaps its dirty destination");
    }
    for(const auto *append:appenders) {
      HYLE_ASSIGN_OR_RETURN(auto range,append->view());
      if(overlaps_resource_range(reads[i],range))
        return Status::InvalidArgument("pending effect read overlaps its append destination");
    }
    for(const auto &old:episode.bindings) {
      if(old.projected_writer) {
        HYLE_ASSIGN_OR_RETURN(auto range,old.projected_writer->view());
        if(overlaps_resource_range(reads[i],range))
          return Status::InvalidArgument("pending effect read overlaps an existing writer");
      } else if(old.projected_appender) {
        HYLE_ASSIGN_OR_RETURN(auto range,old.projected_appender->view());
        if(overlaps_resource_range(reads[i],range))
          return Status::InvalidArgument("pending effect read overlaps an existing append");
      } else if(old.exclusive && old.write.valid()) {
        HYLE_ASSIGN_OR_RETURN(auto range,old.write.view());
        if(overlaps_resource_range(reads[i],range))
          return Status::InvalidArgument("pending effect read overlaps an existing mutable binding");
      }
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
    ResourceOptions options,std::span<const mem::ConstBufferView>pending_reads,bool requires_source)noexcept{
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
  HYLE_RETURN_IF_ERROR(validate_pending_write_reads(*episode,tensor,pending_reads,writers,appenders));
  if(pages.size()>UINT32_MAX || writers.size()>UINT32_MAX-pages.size() ||
      appenders.size()>UINT32_MAX-pages.size()-writers.size() ||
      controls.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size() ||
      episode->bindings.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size()-controls.size())
    return Status::ResourceExhausted("projected writer permission count overflows");
  const auto extra=pages.size()+writers.size()+appenders.size()+controls.size();
  const auto pending_count=pending_reads.empty() && requires_source?std::size_t(1):pending_reads.size();
  if(pending_count>UINT32_MAX-extra-episode->bindings.size())
    return Status::ResourceExhausted("projected pending effect count overflows");
  HYLE_RETURN_IF_ERROR(reserve_reads(invocation,std::uint32_t(extra+pending_count),options));
  // The common read projection validates all physical consumers, aliases and
  // actual ready ranges before committing. Pending here means no ready lease
  // is invented for pages written earlier in this very Graph.
  HYLE_RETURN_IF_ERROR(bind_projected_impl(invocation,value,tensor,pages,readers,true));
  // The common read binder appends the remaining page guards last. They
  // belong to this same written projection, including its committed tail
  // prefix; preserve that ownership for Source capture and retirement.
  const auto extra_pages=pages.empty()?std::size_t{0}:pages.size()-1;
  HYLE_FATAL_ASSERT(extra_pages<=episode->bindings.size());
  for(std::size_t i=episode->bindings.size()-extra_pages;i<episode->bindings.size();++i){
    auto&page_binding=episode->bindings[i];
    HYLE_FATAL_ASSERT(!page_binding.value.valid() && page_binding.projection==value &&
        page_binding.scope==BindingScope::kProjected && page_binding.read.valid());
    page_binding.scope=BindingScope::kWrittenProjection;page_binding.writer_step=writer.step;
  }
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
  for(const auto range:pending_reads)episode->bindings.push_back({.projection=value,.writer_step=writer.step,
      .borrowed=false,.scope=BindingScope::kWrittenProjection,.pending_range=range,.pending_effect=true});
  if(pending_reads.empty() && requires_source)episode->bindings.push_back({.projection=value,.writer_step=writer.step,
      .borrowed=false,.scope=BindingScope::kWrittenProjection,.pending_effect=true});
  episode->nominated_writers[writer.step]=writer.node.ordinal();
  return Status::Ok();
}
Status RegionAccess::bind_write_effect(Invocation&invocation,const Graph&graph,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease>pages,
    std::span<const compute::WriteLease*const>writers,
    std::span<const compute::detail::AppendLease*const>appenders,ProjectedWriter writer,
    std::span<const compute::ReadLease>controls,ResourceOptions options,
    std::span<const mem::ConstBufferView>pending_reads,bool requires_source)noexcept{
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
  HYLE_RETURN_IF_ERROR(validate_pending_write_reads(*episode,tensor,pending_reads,writers,appenders));
  if(pages.size()>UINT32_MAX || writers.size()>UINT32_MAX-pages.size() ||
      appenders.size()>UINT32_MAX-pages.size()-writers.size() ||
      controls.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size() ||
      episode->bindings.size()>UINT32_MAX-pages.size()-writers.size()-appenders.size()-controls.size())
    return Status::ResourceExhausted("write-effect permission count overflows");
  const auto extra=pages.size()+writers.size()+appenders.size()+controls.size();
  const auto pending_count=pending_reads.empty() && requires_source?std::size_t(1):pending_reads.size();
  if(pending_count>UINT32_MAX-extra-episode->bindings.size())
    return Status::ResourceExhausted("write effect pending permission count overflows");
  HYLE_RETURN_IF_ERROR(reserve_reads(invocation,std::uint32_t(extra+pending_count),options));
  HYLE_RETURN_IF_ERROR(revoke_preparation(invocation));
  // No caller input or capacity lease is installed. All fallible validation
  // and growth precedes these infallible anonymous ownership records.
  episode->input_lease_candidates.clear();
  for(const auto&page:pages)episode->bindings.push_back({.read=page,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto*page:writers)episode->bindings.push_back({.projected_writer=page,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto*append:appenders)episode->bindings.push_back({.projected_appender=append,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto&control:controls)episode->bindings.push_back({.read=control,.writer_step=writer.step,.borrowed=false,.scope=BindingScope::kWriteEffect});
  for(const auto range:pending_reads)episode->bindings.push_back({.writer_step=writer.step,
      .borrowed=false,.scope=BindingScope::kWriteEffect,.pending_range=range,.pending_effect=true});
  if(pending_reads.empty() && requires_source)episode->bindings.push_back({.writer_step=writer.step,
      .borrowed=false,.scope=BindingScope::kWriteEffect,.pending_effect=true});
  episode->nominated_writers[writer.step]=writer.node.ordinal();
  return Status::Ok();
}
bool RegionAccess::successful_outcome(const Invocation&invocation)noexcept{
  const auto*state=InvocationAccess::state(invocation);
  return outcome_available(invocation) && state->phase.load(std::memory_order_acquire)==InvocationPhase::kIdle &&
      state->outputs_ready.load(std::memory_order_acquire);
}
Result<bool> RegionAccess::has_pending_effect(const InvocationState *state, ProjectedWriter writer,
    std::span<const mem::ConstBufferView> ranges) noexcept {
  if(!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("pending effect has no original invocation"));
  const auto &plan=*state->plan;
  if(GraphIdentity::generation(writer.node)!=plan.generation || writer.node.ordinal()>=plan.node_steps.size() ||
      plan.node_steps[writer.node.ordinal()]!=writer.step || writer.step>=plan.steps.size() ||
      writer.step>=state->invocations.size() || plan.steps[writer.step].kind!=PlanStepKind::kRegion ||
      !state->invocations[writer.step].region.valid() ||
      !plan.steps[writer.step].region.matches_artifact(writer.implementation,writer.prepared))
    return std::unexpected(Status::InvalidArgument("pending effect differs from its exact selected artifact"));
  if(!state->resources)return false;
  const auto &episode=*state->resources.get();
  if(writer.step>=episode.nominated_writers.size() || episode.nominated_writers[writer.step]!=writer.node.ordinal())
    return false;
  std::size_t count=0;bool registered=false;
  for(const auto &binding:episode.bindings) {
    if(binding.writer_step!=writer.step || !binding.pending_effect)continue;
    registered=true;
    if(!binding.pending_range.valid())continue;
    if(count>=ranges.size() || !same_resource_range(binding.pending_range,ranges[count++]))
      return std::unexpected(Status::InvalidArgument("pending effect ranges differ from the original registration"));
  }
  if(count!=ranges.size() || !registered)
    return std::unexpected(Status::InvalidArgument("pending effect range set is empty or incomplete"));
  return true;
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
  const auto consumers=projected_input_consumers(plan,capacity.ordinal());
  // The compiled index includes every physical consumer, including copies
  // and other-storage replicas. With no alias descendant and no replica,
  // these are exactly the roots visited by the general whole-plan checks.
  const bool direct=projection.aliases.empty() &&
      (plan.alias_sources[root]==kNoInPlaceSource || plan.alias_sources[root]==root) &&
      !std::binary_search(plan.external_input_aliases.begin(),plan.external_input_aliases.end(),capacity.ordinal()) &&
      std::all_of(consumers.begin(),consumers.end(),[&](const auto&entry){return entry.instance==root;});
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
  if(direct){
    HYLE_ASSIGN_OR_RETURN(auto bytes,plan.instance_descriptions[root].required_bytes());
    if(bytes>tensor.storage().size())return Status::InvalidArgument("projected alias bytes exceed the original capacity");
  }
  if(!direct)for(std::size_t i=0;i<plan.alias_sources.size();++i){
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
  const auto inspect=[&](std::size_t i,std::uint32_t port)noexcept->Status{
    const auto&step=plan.steps[i];
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
          state->invocations[i].region.state_,port,&access));
    }else return Status::InvalidArgument("projected root has a materializing/copy consumer");
    if(access.projected_input!=port)return Status::Unimplemented("selected implementation does not project this physical input port");
    return Status::Ok();
  };
  bool consumed=false;
  if(direct){
    for(const auto&consumer:consumers){
      if(std::ranges::none_of(projection.readers,[&](NodeId reader){return plan.node_steps[reader.ordinal()]==consumer.step;}))
        return Status::InvalidArgument("projected root has an unqualified physical consumer");
      HYLE_RETURN_IF_ERROR(inspect(consumer.step,consumer.port));consumed=true;
    }
  }else for(std::size_t i=0;i<plan.steps.size();++i){const auto&step=plan.steps[i];
    const bool declared=std::ranges::any_of(projection.readers,[&](NodeId reader){return plan.node_steps[reader.ordinal()]==i;});
    for(std::size_t port=0;port<step.input_count;++port){const auto input=plan.node_inputs[step.input_begin+port];if(origin(input)!=root)continue;
      if(!allowed(plan.instance_values[input]))return Status::InvalidArgument("projection physical input is not a declared alias");
      if(step.kind==PlanStepKind::kView && !step.materializes){
        for(std::size_t out=0;out<step.output_count;++out){const auto value=plan.node_outputs[step.output_begin+out];
          if(origin(value)!=root || !allowed(plan.instance_values[value]))return Status::InvalidArgument("projection view escapes its declared aliases");}
        continue;
      }
      if(!declared)return Status::InvalidArgument("projected root has an unqualified physical consumer");
      HYLE_RETURN_IF_ERROR(inspect(i,std::uint32_t(port)));
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
Status RegionAccess::validate_projection_binding(Invocation&invocation,ResourceProjection projection,
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
  return Status::Ok();
}
Status RegionAccess::validate_pending_projection(Invocation &invocation,
    ResourceProjection projection, compute::ConstTensorView tensor) noexcept {
  return validate_projection_binding(invocation, projection, tensor, {}, true);
}
Status RegionAccess::validate_ready_projection(Invocation &invocation,
    ResourceProjection projection, compute::ConstTensorView tensor,
    std::span<const compute::ReadLease> pages) noexcept {
  if (pages.empty()) return Status::InvalidArgument("ready projection requires its actual guards");
  return validate_projection_binding(invocation, projection, tensor, pages, false);
}
Status RegionAccess::bind_projection(Invocation&invocation,ResourceProjection projection,
    compute::ConstTensorView tensor,std::span<const compute::ReadLease>pages,bool pending)noexcept{
  HYLE_RETURN_IF_ERROR(validate_projection_binding(invocation,projection,tensor,pages,pending));
  auto*state=InvocationAccess::state(invocation);
  auto*episode=state->resources.get();
  const auto capacity=projection.capacity;
  auto*binding=find_binding(state,capacity);
  const auto description=tensor.description();
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

struct SubmissionSourceState final : util::RefCounted<SubmissionSourceState> {
  SubmissionSourceState(InvocationState *original, compute::detail::QueueRef queue_value,
      compute::StorageOwner storage, compute::Completion submission) noexcept
      : producer(original), queue(std::move(queue_value)), metadata_owner(std::move(storage)),
        completion(std::move(submission)) {
    retain(producer);
    producer->results.fetch_add(1, std::memory_order_relaxed);
  }
  void destroy() noexcept {
    const auto storage = metadata_owner;
    const auto allocator = storage.allocator();
    auto *original = producer;
    HYLE_FATAL_ASSERT(borrows == 0);
    this->~SubmissionSourceState();
    allocator.deallocate(this, sizeof(SubmissionSourceState), alignof(SubmissionSourceState));
    if (original) {
      original->results.fetch_sub(1, std::memory_order_release);
      release(original);
    }
  }
  InvocationState *producer = nullptr;
  compute::detail::QueueRef queue;
  // Independently retained through native retirement; the local owner above
  // also covers allocator teardown after the last public/native source ref.
  compute::StorageOwner metadata_owner;
  compute::Completion completion;
  std::mutex observation;
  std::size_t borrows = 0;
};
struct OutputBindingBundle;
struct OutputSourceState final : util::RefCounted<OutputSourceState> {
  OutputSourceState(SubmissionSourceState *source, ValueId value,
      compute::WriteLease *lease, compute::ReadLease *published,
      compute::ConstTensorView view, compute::StorageOwner storage) noexcept
      : submission(util::retain_ref(source)), output(value), writer(lease), ready(published),
        tensor(view), storage_owner(std::move(storage)) {
    ++source->borrows;
  }
  void destroy() noexcept {
    const auto storage = storage_owner;
    const auto allocator = storage.allocator();
    {
      std::lock_guard guard(submission->observation);
      if (writer) --submission->borrows;
    }
    this->~OutputSourceState();
    allocator.deallocate(this, sizeof(OutputSourceState), alignof(OutputSourceState));
  }
  util::Ref<SubmissionSourceState> submission;
  ValueId output{};
  compute::WriteLease *writer = nullptr;
  compute::ReadLease *ready = nullptr;
  compute::ConstTensorView tensor;
  compute::StorageOwner storage_owner;
  // Destroy the actual read permission before its resource/allocator owners.
  compute::ReadLease retained_output;
  // Borrowed only while the actual InputSources attachment keeps this bundle
  // alive. Destruction clears it; bound_once prevents consumer-address ABA.
  OutputBindingBundle *bundle = nullptr;
  bool bound_once = false;
};
struct StoreSourceState final : util::RefCounted<StoreSourceState> {
  StoreSourceState(compute::StorageOwner storage, compute::ResourceBudget budget,
      mem::AllocatorRef selected) noexcept
      : storage_owner(std::move(storage)), metadata_budget(std::move(budget)), allocator(selected),
        tickets(selected), addresses(selected) {}
  void destroy() noexcept {
    const auto storage = storage_owner;
    const auto budget = metadata_budget;
    (void)budget;
    const auto selected = allocator;
    auto *original = prepared_parent;
    this->~StoreSourceState();
    selected.deallocate(this, sizeof(StoreSourceState), alignof(StoreSourceState));
    if (original) release(original);
  }
  compute::StorageOwner storage_owner;
  compute::ResourceBudget metadata_budget;
  mem::AllocatorRef allocator;
  ResourceOptions options{};
  // Native region slots are created once with this InvocationState; reset
  // only clears their bindings/configuration. Retaining that parent keeps the
  // embedded StoreBinding object stable, not its replaceable transaction.
  // capture re-exports/revalidates that current transaction before any seal,
  // then clears both borrows; no writer pointer ever escapes this factory.
  InvocationState *prepared_parent = nullptr;
  slots::detail::StoreBinding *prepared_binding = nullptr;
  NodeId node{};
  slots::Schema schema{};
  // The fixed read view may cover only a consecutive part of the original
  // Store. The original transaction must still contain this full count.
  std::size_t original_destination_count = 0, first_destination = 0;
  util::Ref<SubmissionSourceState> submission;
  compute::LeaseSlotLayout layout;
  compute::ConstTensorView capacity{}, indices{};
  compute::ReadLease indices_guard;
  container::SmallVector<compute::detail::PendingReadTicket, 2, mem::AllocatorRef> tickets;
  container::SmallVector<compute::LeaseSlotAddress, 2, mem::AllocatorRef> addresses;
  OutputBindingBundle *bundle = nullptr;
  bool bound_once = false;
};
// A permission origin has no execution/terminal ancestry. Descendants retain
// this fixed metadata and the actual resource/allocator owner only.
struct ResourcePermissionBlock final : util::RefCounted<ResourcePermissionBlock> {
  ResourcePermissionBlock(compute::StorageOwner owner, compute::ResourceBudget budget,
      mem::AllocatorRef selected) noexcept
      : storage_owner(std::move(owner)), metadata_budget(std::move(budget)), allocator(selected),
        ready(mem::StdAllocator<compute::ReadLease>(selected)),
        full(mem::StdAllocator<compute::detail::PendingReadTicket>(selected)),
        append(mem::StdAllocator<compute::detail::PendingAppendReadTicket>(selected)),
        bounds(mem::StdAllocator<mem::ConstBufferView>(selected)),
        published(mem::StdAllocator<compute::ReadLease>(selected)),
        inherited_uses(mem::StdAllocator<std::uint32_t>(selected)) {}
  // Called under publication. One surviving inherited slot must not retain
  // payload belonging to another page in this metadata allocation.
  void clear_slot(std::size_t index) noexcept {
    if (index < published.size()) published[index] = {};
    if (index < ready.size()) { ready[index] = {}; return; }
    index -= ready.size();
    if (index < full.size()) { full[index].reset(); return; }
    append[index - full.size()].reset();
  }
  void release_source() noexcept {
    std::lock_guard guard(publication);
    source_alive = false;
    const auto count = ready.size() + full.size() + append.size();
    for (std::size_t i = 0; i < count; ++i)
      if (i >= inherited_uses.size() || !inherited_uses[i]) clear_slot(i);
  }
  void destroy() noexcept {
    const auto owner = storage_owner;
    const auto budget = metadata_budget;
    (void)budget;
    const auto selected = allocator;
    this->~ResourcePermissionBlock();
    selected.deallocate(this, sizeof(ResourcePermissionBlock), alignof(ResourcePermissionBlock));
  }
  compute::StorageOwner storage_owner;
  compute::ResourceBudget metadata_budget;
  mem::AllocatorRef allocator;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> ready;
  std::vector<compute::detail::PendingReadTicket,
      mem::StdAllocator<compute::detail::PendingReadTicket>> full;
  std::vector<compute::detail::PendingAppendReadTicket,
      mem::StdAllocator<compute::detail::PendingAppendReadTicket>> append;
  std::vector<mem::ConstBufferView, mem::StdAllocator<mem::ConstBufferView>> bounds;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> published;
  std::vector<std::uint32_t, mem::StdAllocator<std::uint32_t>> inherited_uses;
  std::mutex publication;
  bool source_alive = true;
};
struct ResourceInheritedPermission final {
  ResourceInheritedPermission() noexcept = default;
  ResourceInheritedPermission(const ResourceInheritedPermission &) = delete;
  ResourceInheritedPermission &operator=(const ResourceInheritedPermission &) = delete;
  ResourceInheritedPermission(ResourceInheritedPermission &&other) noexcept
      : origin(std::move(other.origin)), index(other.index) {}
  ResourceInheritedPermission &operator=(ResourceInheritedPermission &&other) noexcept {
    if (this != &other) {
      reset(); origin = std::move(other.origin); index = other.index;
    }
    return *this;
  }
  ~ResourceInheritedPermission() { reset(); }
  static Result<ResourceInheritedPermission> acquire(ResourcePermissionBlock *block,
      std::size_t index) noexcept {
    std::lock_guard guard(block->publication);
    if (block->inherited_uses[index] == UINT32_MAX)
      return std::unexpected(Status::ResourceExhausted("resource inherited slot reference count"));
    ++block->inherited_uses[index];
    ResourceInheritedPermission result;
    result.origin = util::retain_ref(block); result.index = index;
    return result;
  }
  void reset() noexcept {
    auto held = std::move(origin);
    if (!held) return;
    {
      std::lock_guard guard(held->publication);
      HYLE_FATAL_ASSERT(held->inherited_uses[index] != 0);
      if (--held->inherited_uses[index] == 0 && !held->source_alive) held->clear_slot(index);
    }
    // A last metadata ref is released only after unlocking its own mutex.
  }
  util::Ref<ResourcePermissionBlock> origin;
  std::size_t index = 0;
};
struct ResourcePermissionLookup final {
  ResourcePermissionBlock *origin = nullptr;
  std::size_t index = 0;
  mem::ConstBufferView bound{};
};
struct ResourceSourceState final : util::RefCounted<ResourceSourceState> {
  ResourceSourceState(compute::StorageOwner owner, compute::ResourceBudget budget,
      mem::AllocatorRef selected) noexcept
      : storage_owner(std::move(owner)), metadata_budget(std::move(budget)), allocator(selected),
        inherited(mem::StdAllocator<ResourceInheritedPermission>(selected)),
        lookup(mem::StdAllocator<ResourcePermissionLookup>(selected)) {}
  void destroy() noexcept {
    const auto owner = storage_owner;
    const auto budget = metadata_budget;
    (void)budget;
    const auto selected = allocator;
    auto *original = prepared_parent;
    if (permissions) permissions->release_source();
    this->~ResourceSourceState();
    selected.deallocate(this, sizeof(ResourceSourceState), alignof(ResourceSourceState));
    if (original) release(original);
  }
  compute::StorageOwner storage_owner;
  compute::ResourceBudget metadata_budget;
  mem::AllocatorRef allocator;
  ResourceOptions options{};
  InvocationState *prepared_parent = nullptr;
  util::Ref<SubmissionSourceState> submission;
  compute::ConstTensorView capacity{};
  util::Ref<ResourcePermissionBlock> permissions;
  std::vector<ResourceInheritedPermission, mem::StdAllocator<ResourceInheritedPermission>> inherited;
  // Sorted by physical offset once during capture, within this one capacity.
  std::vector<ResourcePermissionLookup, mem::StdAllocator<ResourcePermissionLookup>> lookup;
  OutputBindingBundle *bundle = nullptr;
  bool bound_once = false;
};
struct OutputSourceAccess final {
  static OutputSourceState *state(const OutputSource *source) noexcept {
    return source ? source->state_ : nullptr;
  }
  static StoreSourceState *state(const StoreSource *source) noexcept {
    return source ? source->state_ : nullptr;
  }
  static ResourceSourceState *state(const ResourceSource *source) noexcept {
    return source ? source->state_ : nullptr;
  }
};
struct OutputBindingEntry final {
  ValueId input{};
  util::Ref<OutputSourceState> view{};
  util::Ref<StoreSourceState> store{};
  ResourceProjection projection{};
  util::Ref<ResourceSourceState> resource{};
  std::span<const ProjectedReader> readers{};
  std::span<const ResourceReadyInput> ready_inputs{};
  ProjectedWriter effect{};
  std::span<const mem::ConstBufferView> effect_reads{};
  std::span<const compute::ReadLease> store_ready{};
  std::span<const compute::ReadLease> resource_ready{};
};
struct OutputBindingBundle final : util::RefCounted<OutputBindingBundle> {
  explicit OutputBindingBundle(SubmissionSourceState *source) noexcept
      : submission(util::retain_ref(source)),
        entries(mem::StdAllocator<OutputBindingEntry>(source->metadata_owner.allocator())),
        replacements(mem::StdAllocator<ReplayInputBinding>(source->metadata_owner.allocator())),
        access_nodes(mem::StdAllocator<NodeId>(source->metadata_owner.allocator())),
        readers(mem::StdAllocator<ProjectedReader>(source->metadata_owner.allocator())),
        ready_inputs(mem::StdAllocator<ResourceReadyInput>(source->metadata_owner.allocator())),
        private_controls(mem::StdAllocator<compute::ReadLease>(source->metadata_owner.allocator())),
        effect_reads(mem::StdAllocator<mem::ConstBufferView>(source->metadata_owner.allocator())),
        store_ready(mem::StdAllocator<compute::ReadLease>(source->metadata_owner.allocator())),
        resource_ready(mem::StdAllocator<compute::ReadLease>(source->metadata_owner.allocator())),
        release_values(mem::StdAllocator<ValueId>(source->metadata_owner.allocator())),
        release_effects(mem::StdAllocator<NodeId>(source->metadata_owner.allocator())) {}
  void destroy() noexcept {
    const auto storage = submission->metadata_owner;
    const auto allocator = storage.allocator();
    {
      std::lock_guard guard(submission->observation);
      for (const auto &entry : entries) {
        if (entry.view && entry.view->bundle == this) entry.view->bundle = nullptr;
        if (entry.store && entry.store->bundle == this) entry.store->bundle = nullptr;
        if (entry.resource && entry.resource->bundle == this) entry.resource->bundle = nullptr;
      }
    }
    this->~OutputBindingBundle();
    allocator.deallocate(this, sizeof(OutputBindingBundle), alignof(OutputBindingBundle));
  }
  util::Ref<SubmissionSourceState> submission;
  std::vector<OutputBindingEntry, mem::StdAllocator<OutputBindingEntry>> entries;
  std::vector<ReplayInputBinding, mem::StdAllocator<ReplayInputBinding>> replacements;
  std::vector<NodeId, mem::StdAllocator<NodeId>> access_nodes;
  std::vector<ProjectedReader, mem::StdAllocator<ProjectedReader>> readers;
  std::vector<ResourceReadyInput, mem::StdAllocator<ResourceReadyInput>> ready_inputs;
  // Real private-control roots, deduplicated in cold preparation only. Each
  // individual fact is still re-queried; admission uses the original batch.
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> private_controls;
  std::vector<mem::ConstBufferView,mem::StdAllocator<mem::ConstBufferView>> effect_reads;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> store_ready;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> resource_ready;
  std::vector<ValueId, mem::StdAllocator<ValueId>> release_values;
  std::vector<NodeId, mem::StdAllocator<NodeId>> release_effects;
  InvocationState *consumer = nullptr;
  ReplayRoute consumer_route = ReplayRoute::kUnprepared;
  bool preparing = false, submitted = false, all_ready = false;
};
Result<RegionReadyControl> RegionAccess::ready_control(const InvocationState *state,
    NodeId node, std::uint32_t ordinal) noexcept {
  if (!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("ready control has no original invocation"));
  const auto &plan = *state->plan;
  if (GraphIdentity::generation(node) != plan.generation || node.ordinal() >= plan.node_steps.size())
    return std::unexpected(Status::InvalidArgument("ready control names a foreign original node"));
  const auto index = plan.node_steps[node.ordinal()];
  if (index >= plan.steps.size() || index >= state->invocations.size())
    return std::unexpected(Status::InvalidArgument("ready control has no original selected step"));
  const auto &step = plan.steps[index];
  if (step.kind != PlanStepKind::kRegion || !step.region.valid() ||
      !state->invocations[index].region.valid())
    return std::unexpected(Status::Unimplemented("ready control needs its selected original region"));
  const auto &prepared = step.region.prepared_;
  const auto *table = prepared.vtable->query_extension
      ? static_cast<const RegionReadyControlVTable *>(prepared.vtable->query_extension(
          prepared.state, kRegionReadyControlExtensionId)) : nullptr;
  if (!table || !table->inspect)
    return std::unexpected(Status::Unimplemented("selected region cannot expose its actual ready control"));
  struct Call final {
    const RegionReadyControlVTable *table;
    const ProviderRegionPrepared *prepared;
    void *invocation;
    std::uint32_t ordinal;
    Result<RegionReadyControl> result;
  } call{table, &prepared, state->invocations[index].region.state_, ordinal,
      std::unexpected(Status::Internal("ready control query was not called"))};
  HYLE_RETURN_IF_ERROR(step.region.source_.call([](void *, const void *, void *opaque) noexcept -> Status {
    auto &c = *static_cast<Call *>(opaque);
    c.result = c.table->inspect(*c.prepared, c.invocation, c.ordinal);
    return Status::Ok();
  }, &call));
  return std::move(call.result);
}
namespace {
Result<compute::detail::QueueRef> output_queue(const InvocationState &state) noexcept {
  const auto &plan = *state.plan;
  if (state.progress || plan.segments.size() != 1 || plan.required_queue_count != 1)
    return std::unexpected(Status::Unimplemented("output relay requires one original shared segment"));
  const auto lane = plan.segments.front().lane;
  if (!shared_lane(plan, lane) ||
      lane.ordinal() >= state.queues.size() || !state.queues[lane.ordinal()].valid())
    return std::unexpected(Status::Unimplemented("output relay requires one original shared Queue"));
  // The exact unobserved submit proxy prevents binding/replay mutation. These
  // are the original submission's actual ranges, not static provider promises.
  if (submitted_shared_replay_route(state) == ReplayRoute::kUnprepared)
    return std::unexpected(Status::Unimplemented(
        "output relay requires actual captured ranges and prepared nonblocking command gaps"));
  HYLE_ASSIGN_OR_RETURN(auto endpoint,
      compute::detail::QueueSequence::completion_endpoint(state.queues[lane.ordinal()]));
  if (!endpoint.valid())
    return std::unexpected(Status::Unimplemented("output relay requires an actual asynchronous endpoint"));
  return state.queues[lane.ordinal()];
}
Result<compute::CompletionPoll> poll_submission(SubmissionSourceState &source) noexcept {
  std::lock_guard guard(source.observation);
  return source.completion.poll();
}
Status wait_submission(SubmissionSourceState &source) noexcept {
  std::lock_guard guard(source.observation);
  return source.completion.wait();
}
const InputSourceVTable output_source_table{
    [](void *opaque) noexcept { static_cast<OutputBindingBundle *>(opaque)->incref(); },
    [](void *opaque) noexcept { static_cast<OutputBindingBundle *>(opaque)->decref(); },
    [](void *opaque, const compute::detail::QueueRef *queue) noexcept
        -> Result<InputDependencyRoute> {
      const auto &bundle = *static_cast<OutputBindingBundle *>(opaque);
      if (!bundle.preparing)
        return std::unexpected(Status::FailedPrecondition("pending output requires its metadata-only bundle preparation"));
      if (!queue || !bundle.submission->queue.same_owner(*queue))
        return std::unexpected(Status::Unimplemented("output relay requires its original actual Queue"));
      return InputDependencyRoute::kDeviceWait;
    },
    [](void *opaque) noexcept {
      return poll_submission(*static_cast<OutputBindingBundle *>(opaque)->submission);
    },
    [](void *opaque) noexcept {
      return wait_submission(*static_cast<OutputBindingBundle *>(opaque)->submission);
    },
    [](void *opaque, compute::detail::QueueSequence &sequence) noexcept {
      auto &source = *static_cast<OutputBindingBundle *>(opaque)->submission;
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
      // The native dependency is not the full verdict. InputSources reconciles
      // this same original Completion after the imported consumer really retires.
      return sequence.enqueue_resolved_wait(*dependency);
    },
    [](void *opaque) noexcept {
      return wait_submission(*static_cast<OutputBindingBundle *>(opaque)->submission);
    }};
Result<InputSource> output_input_source(OutputBindingBundle *bundle) noexcept {
  return InputSource::acquire(bundle, &output_source_table, bundle->submission.get());
}
Status admit_source_permissions(compute::LeaseUseBatch &uses,
    std::span<const compute::ReadLease> ready,
    std::span<const compute::detail::PendingReadTicket> full,
    std::span<const compute::detail::PendingAppendReadTicket> append) noexcept {
  for (const auto &read : ready)
    HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::read_unique(uses, read));
  for (const auto &ticket : full)
    HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::ordered_read_unique(uses, ticket));
  for (const auto &ticket : append)
    HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::ordered_read_unique(uses, ticket));
  return Status::Ok();
}
Status validate_store_ready_guards(const StoreSourceState &store,
    std::span<const compute::ReadLease> ready) noexcept {
  if (ready.empty()) return Status::Ok();
  if (ready.size() != store.tickets.size() || ready.size() != store.addresses.size())
    return Status::InvalidArgument("ready Store source needs every original destination in order");
  for (std::size_t i = 0; i < ready.size(); ++i) {
    if (!ready[i].valid() || compute::detail::LeaseAccess::identity(ready[i]) !=
        compute::detail::LeaseAccess::identity(store.tickets[i]) ||
        ready[i].generation() != store.tickets[i].generation())
      return Status::InvalidArgument("ready Store destination differs from its original ticket root or generation");
    HYLE_ASSIGN_OR_RETURN(auto address, store.layout.locate(ready[i]));
    if (address != store.addresses[i])
      return Status::InvalidArgument("ready Store destination differs from its original slot coordinates");
    HYLE_ASSIGN_OR_RETURN(auto range, ready[i].view());
    HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::matches_pending_read(store.tickets[i], range));
  }
  return Status::Ok();
}
Status validate_store_ready(const StoreSourceState &store,
    std::span<const compute::ReadLease> ready) noexcept {
  if (ready.empty()) return Status::Ok();
  auto &source = *store.submission;
  std::lock_guard guard(source.observation);
  HYLE_ASSIGN_OR_RETURN(auto observed, source.completion.poll());
  if (observed == compute::CompletionPoll::kPending || (source.producer &&
      (source.producer->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
       !source.producer->outputs_ready.load(std::memory_order_acquire) ||
       source.producer->preparation_uses.load(std::memory_order_acquire))))
    return Status::Unavailable("ready Store source requires successful actual producer retirement");
  return validate_store_ready_guards(store, ready);
}
Status validate_resource_ready_input(const InvocationState &state, const ResourceReadyInput &required,
    std::span<const ProjectedReader> readers) noexcept {
  if (required.input.valid() == required.region.valid() || !required.tensor.valid() || !required.read.valid())
    return Status::InvalidArgument("resource ready control needs one exact input or selected region");
  compute::ReadLease actual_read;
  if (required.region.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto control,
        RegionAccess::ready_control(&state, required.region, required.ordinal));
    const auto step = state.plan->node_steps[required.region.ordinal()];
    if (std::ranges::none_of(readers, [&](const auto &reader) { return reader.step == step; }))
      return Status::InvalidArgument("resource private control is outside its selected readers");
    const auto actual = control.tensor.storage(), expected = required.tensor.storage();
    if (!control.tensor.valid() || control.tensor.description() != required.tensor.description() ||
        actual.resource() != expected.resource() || actual.native_handle() != expected.native_handle() ||
        actual.offset() != expected.offset() || actual.size() != expected.size())
      return Status::FailedPrecondition("resource source private control tensor was replaced");
    actual_read = std::move(control.read);
  } else {
    if (required.ordinal)
      return Status::InvalidArgument("external ready input cannot have a private control ordinal");
    HYLE_ASSIGN_OR_RETURN(const auto same,
        RegionAccess::has_input_binding(&state, required.input, required.tensor));
    if (!same) return Status::FailedPrecondition("resource source required ready input was reset or replaced");
    const auto *binding = find_binding(&state, required.input);
    if (!binding || binding->borrowed || binding->exclusive || binding->permissionless)
      return Status::FailedPrecondition("resource source lost its original required ready binding");
    actual_read = binding->read;
  }
  if (!actual_read.valid() ||
      compute::detail::LeaseAccess::identity(actual_read) != compute::detail::LeaseAccess::identity(required.read) ||
      actual_read.generation() != required.read.generation())
    return Status::FailedPrecondition("resource source lost its original required ready guard");
  HYLE_ASSIGN_OR_RETURN(auto actual, actual_read.view());
  HYLE_ASSIGN_OR_RETURN(auto expected, required.read.view());
  const auto tensor = required.tensor.storage();
  if (actual.resource() != expected.resource() || actual.native_handle() != expected.native_handle() ||
      actual.offset() != expected.offset() || actual.size() != expected.size() ||
      tensor.resource() != expected.resource() || tensor.native_handle() != expected.native_handle() ||
      tensor.offset() < expected.offset() || tensor.offset() - expected.offset() > expected.size() ||
      tensor.size() > expected.size() - (tensor.offset() - expected.offset()))
    return Status::FailedPrecondition("resource source required ready input range was replaced");
  return Status::Ok();
}
const ResourcePermissionLookup *find_resource_permission(const ResourceSourceState &source,
    mem::ConstBufferView range) noexcept {
  if (!range.valid() || !range.size()) return nullptr;
  auto it = std::upper_bound(source.lookup.begin(), source.lookup.end(), range.offset(),
      [](std::size_t offset, const auto &entry) { return offset < entry.bound.offset(); });
  if (it == source.lookup.begin()) return nullptr;
  --it;
  return contains_resource_range(it->bound, range) ? &*it : nullptr;
}
Status covers_resource_range(const ResourceSourceState &source, mem::ConstBufferView range) noexcept {
  return find_resource_permission(source, range) ? Status::Ok()
      : Status::InvalidArgument("effect read exceeds the source's exact produced prefixes");
}
const void *permission_root(const ResourcePermissionLookup &entry) noexcept {
  const auto &origin = *entry.origin;
  auto index = entry.index;
  if (index < origin.ready.size()) return compute::detail::LeaseAccess::identity(origin.ready[index]);
  index -= origin.ready.size();
  if (index < origin.full.size()) return compute::detail::LeaseAccess::identity(origin.full[index]);
  return compute::detail::LeaseAccess::identity(origin.append[index - origin.full.size()]);
}
std::uint64_t permission_generation(const ResourcePermissionLookup &entry) noexcept {
  const auto &origin = *entry.origin;
  auto index = entry.index;
  if (index < origin.ready.size()) return origin.ready[index].generation();
  index -= origin.ready.size();
  if (index < origin.full.size()) return origin.full[index].generation();
  return origin.append[index - origin.full.size()].generation();
}
Status validate_published_permission(const ResourcePermissionLookup &entry,
    const compute::ReadLease &read) noexcept {
  if (!read.valid() || compute::detail::LeaseAccess::identity(read) != permission_root(entry) ||
      read.generation() != permission_generation(entry))
    return Status::InvalidArgument("published resource guard differs from its original root or generation");
  HYLE_ASSIGN_OR_RETURN(auto range, read.view());
  if (!contains_resource_range(range, entry.bound) ||
      !contains_resource_range(entry.origin->bounds[entry.index], range))
    return Status::InvalidArgument("published resource guard exceeds its original readable bound");
  return Status::Ok();
}
Status admit_resource_permission(compute::LeaseUseBatch &uses,
    const ResourcePermissionLookup &entry) noexcept {
  auto &origin = *entry.origin;
  std::lock_guard guard(origin.publication);
  if (origin.published[entry.index].valid())
    return compute::detail::LeaseAccess::read_unique(uses, origin.published[entry.index]);
  auto index = entry.index;
  if (index < origin.ready.size())
    return compute::detail::LeaseAccess::read_unique(uses, origin.ready[index]);
  index -= origin.ready.size();
  if (index < origin.full.size()) {
    // A full-to-append handoff may precede its ready-slot update. Preserve the
    // original admission error; retry after publish_ready records the actual
    // guard. The same code can also report a permanently closed domain.
    return compute::detail::LeaseAccess::ordered_read_unique(uses, origin.full[index]);
  }
  return compute::detail::LeaseAccess::ordered_read_unique(uses, origin.append[index - origin.full.size()]);
}
Status admit_resource_permissions(compute::LeaseUseBatch &uses,
    const ResourceSourceState &source, std::span<const mem::ConstBufferView> ranges,
    bool selective) noexcept {
  if (selective) {
    for (const auto range : ranges) {
      const auto *entry = find_resource_permission(source, range);
      if (!entry) return Status::FailedPrecondition("pending effect lost its exact admitted prefix");
      HYLE_RETURN_IF_ERROR(admit_resource_permission(uses, *entry));
    }
  } else {
    for (const auto &entry : source.lookup)
      HYLE_RETURN_IF_ERROR(admit_resource_permission(uses, entry));
  }
  return Status::Ok();
}
Result<compute::ReadLease> ready_resource_permission(const ResourcePermissionLookup &entry) noexcept {
  auto &origin = *entry.origin;
  std::lock_guard guard(origin.publication);
  const auto &published = origin.published[entry.index];
  const auto *ready = published.valid() ? &published
      : entry.index < origin.ready.size() ? &origin.ready[entry.index] : nullptr;
  if (!ready || !ready->valid()) return compute::ReadLease{};
  HYLE_RETURN_IF_ERROR(validate_published_permission(entry, *ready));
  HYLE_ASSIGN_OR_RETURN(auto range, ready->view());
  // An inherited lookup may cover less than its origin's published prefix.
  // Keep the real root/generation, but expose only this contribution's bounds.
  return ready->subview(entry.bound.offset() - range.offset(), entry.bound.size());
}
Result<bool> resource_permissions_ready(const ResourceSourceState &source) noexcept {
  for (const auto &entry : source.lookup) {
    HYLE_ASSIGN_OR_RETURN(auto ready, ready_resource_permission(entry));
    if (!ready.valid()) return false;
  }
  return true;
}
Result<bool> output_bundle_ready(const OutputBindingBundle &bundle) noexcept {
  auto &source = *bundle.submission;
  std::lock_guard guard(source.observation);
  const auto observed = source.completion.poll();
  // A failed or pending source remains on the original dependency path. A
  // successful Completion alone cannot certify any initialized input bytes.
  if (!observed || *observed == compute::CompletionPoll::kPending ||
      (source.producer &&
          (source.producer->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
           !source.producer->outputs_ready.load(std::memory_order_acquire) ||
           source.producer->preparation_uses.load(std::memory_order_acquire))))
    return false;
  for (const auto &entry : bundle.entries) {
    if (entry.view) {
      const auto &view = *entry.view;
      // detach() retains only this exact original caller-owned tensor result.
      if (!view.retained_output.valid()) return false;
      HYLE_ASSIGN_OR_RETURN(auto ready, view.retained_output.view());
      if (!same_resource_range(ready, view.tensor.storage()))
        return std::unexpected(Status::FailedPrecondition("ready output lost its original exact storage"));
    } else if (entry.store) {
      if (entry.store_ready.empty()) return false;
      HYLE_RETURN_IF_ERROR(validate_store_ready_guards(*entry.store, entry.store_ready));
    } else {
      HYLE_ASSIGN_OR_RETURN(const auto ready, resource_permissions_ready(*entry.resource));
      if (!ready) return false;
    }
  }
  return true;
}
Status validate_resource_effect(const InvocationState &state,const OutputBindingEntry &entry) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto same,RegionAccess::has_pending_effect(&state,entry.effect,entry.effect_reads));
  if(!same)return Status::FailedPrecondition("pending resource effect registration was reset or replaced");
  for(const auto range:entry.effect_reads)HYLE_RETURN_IF_ERROR(covers_resource_range(*entry.resource,range));
  return Status::Ok();
}
Status validate_bound_store(const InvocationState &state, const OutputBindingEntry &entry) noexcept {
  const auto *binding = find_binding(&state, entry.projection.capacity);
  if (!entry.store || !binding || binding->scope != BindingScope::kProjected || binding->borrowed ||
      binding->exclusive || binding->permissionless || binding->write.valid() ||
      binding->projected_writer || binding->projected_appender)
    return Status::FailedPrecondition("Store source lost its original projected capacity authority");
  HYLE_ASSIGN_OR_RETURN(auto capacity, RegionAccess::has_input_binding(&state,
      entry.projection.capacity, entry.store->capacity));
  HYLE_ASSIGN_OR_RETURN(auto metadata, RegionAccess::has_input_binding(&state,
      entry.projection.metadata, entry.store->indices));
  if (!capacity || !metadata)
    return Status::FailedPrecondition("Store source capacity or IDs input was reset or replaced");
  if (entry.store_ready.empty()) {
    if (binding->read.valid())
      return Status::FailedPrecondition("pending Store source was replaced by a different ready projection");
  } else {
    HYLE_RETURN_IF_ERROR(validate_store_ready_guards(*entry.store, entry.store_ready));
    HYLE_ASSIGN_OR_RETURN(auto ready, RegionAccess::has_read_projection(&state,
        entry.projection.capacity, entry.store_ready));
    if (!ready) return Status::FailedPrecondition("ready Store source footprint was reset or replaced");
  }
  const auto *ids = find_binding(&state, entry.projection.metadata);
  if (!ids || ids->borrowed || ids->exclusive || ids->permissionless || !ids->read.valid() ||
      compute::detail::LeaseAccess::identity(ids->read) !=
          compute::detail::LeaseAccess::identity(entry.store->indices_guard) ||
      ids->read.generation() != entry.store->indices_guard.generation())
    return Status::FailedPrecondition("Store source lost its original bounded ready IDs");
  HYLE_ASSIGN_OR_RETURN(auto actual, ids->read.view());
  if (!same_resource_range(actual, entry.store->indices.storage()))
    return Status::FailedPrecondition("Store source ready IDs range was replaced");
  return Status::Ok();
}
bool same_ready_guard(const compute::ReadLease &, const compute::ReadLease &) noexcept;
Status check_output_consumer(Invocation &invocation, OutputBindingBundle &bundle,
    std::span<const QueueBinding> queues, bool require_binding) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  auto &source = *bundle.submission;
  if (state == source.producer)
    return Status::InvalidArgument("output relay cannot consume its own original submission");
  const auto &plan = *state->plan;
  if (state->progress || plan.segments.size() != 1 || plan.required_queue_count != 1 ||
      !shared_lane(plan, plan.segments.front().lane))
    return Status::Unimplemented("output relay consumer requires one original shared segment");
  bool commands = plan.source_commands_prepared;
  if (!bundle.all_ready) {
    const bool retained = std::ranges::any_of(state->captured_ranges,
        [](const auto &range) { return range.commands.valid(); }) ||
        std::ranges::any_of(state->capture_rebuilds, [](const auto &range) {
          return range.storage.valid() || std::ranges::any_of(range.variants,
              [](const auto &variant) { return variant.commands.valid(); });
        });
    // Commands preparation destroys these native artifacts. A pending producer
    // must not incur that driver cleanup; require the existing complete tuple
    // below, even when every ordinary command is otherwise nonblocking.
    if (retained) commands = false;
  }
  auto route = bundle.consumer ? bundle.consumer_route
      : commands ? ReplayRoute::kCommands : ReplayRoute::kUnprepared;
  if (!bundle.all_ready && route != ReplayRoute::kCommands) {
    // Binding establishes source authority, not an executable final tuple.
    // Imported Store destinations need that authority before their own cold
    // bindings can be installed. Check the complete actual tuple at prepare
    // AND submit; neither path may record or execute an incomplete capture.
    if (require_binding) {
      HYLE_ASSIGN_OR_RETURN(const auto reusable,
          ReplayAccess::can_reuse_capture(invocation, bundle.replacements));
      if (!reusable)
        return Status::Unavailable("pending outputs have no complete existing capture for their final physical tuple");
    }
    const auto expected = plan.capture_command_count == plan.command_count
        ? ReplayRoute::kCaptured : ReplayRoute::kHybrid;
    if (bundle.consumer && route != expected)
      return Status::FailedPrecondition("pending output bundle changed its original replay coverage");
    route = expected;
    const auto lane = plan.segments.front().lane;
    if (!source.queue.same_owner(state->queues[lane.ordinal()]))
      return Status::InvalidArgument("pending captured outputs require the producer's actual Queue");
  } else if (!bundle.all_ready && !commands) {
    return Status::Unimplemented("output relay requires prepared nonblocking shared commands");
  }
  if (!queues.empty() && (queues.size() != 1 || !queues.front().queue ||
      queues.front().lane != plan.segments.front().lane ||
      !source.queue.same_owner(*queues.front().queue)))
    return Status::InvalidArgument("output relay requires the producer's actual Queue and consumer lane");
  if (require_binding) {
    if (bundle.consumer != state)
      return Status::FailedPrecondition("output bundle names another original consumer");
    HYLE_ASSIGN_OR_RETURN(auto owner, output_input_source(&bundle));
    for (const auto &replacement : bundle.replacements) {
      if (!input_source_matches(*state, replacement.input, owner))
        return Status::FailedPrecondition("output bundle input owner was reset or replaced");
      HYLE_ASSIGN_OR_RETURN(auto bound,
          RegionAccess::has_input_binding(state, replacement.input, replacement.tensor));
      if (!bound) return Status::FailedPrecondition("output bundle input storage was reset or replaced");
    }
    for (const auto &entry : bundle.entries) {
      if(entry.effect.node.valid()) {
        if(!effect_source_matches(*state,entry.effect.node,owner))
          return Status::FailedPrecondition("pending resource effect lost its original source owner");
        HYLE_RETURN_IF_ERROR(validate_resource_effect(*state,entry));
        continue;
      }
      const auto *binding = find_binding(state, entry.input);
      const auto tensor = entry.view ? entry.view->tensor
          : entry.store ? entry.store->capacity : entry.resource->capacity;
      const bool ready_binding = (entry.store && !entry.store_ready.empty()) ||
          (bundle.all_ready && (entry.view || entry.resource));
      if (!binding || binding->borrowed || binding->exclusive ||
          (binding->read.valid() && !ready_binding) ||
          binding->permissionless || binding->description != tensor.description() ||
          ((entry.store || entry.resource) && binding->scope != BindingScope::kProjected))
        return Status::FailedPrecondition("output bundle lost its original pending read requirement");
      if (entry.store) HYLE_RETURN_IF_ERROR(validate_bound_store(*state, entry));
      if (bundle.all_ready && entry.view) {
        if (!same_ready_guard(binding->read, entry.view->retained_output))
          return Status::FailedPrecondition("ready output lost its exact original read binding");
      }
      if (bundle.all_ready && entry.resource && !entry.resource_ready.empty()) {
        HYLE_ASSIGN_OR_RETURN(const auto same,
            RegionAccess::has_read_projection(state, entry.input, entry.resource_ready));
        if (!same) return Status::FailedPrecondition("ready resource lost its original read projection");
      }
      for (const auto &required : entry.ready_inputs)
        HYLE_RETURN_IF_ERROR(validate_resource_ready_input(*state, required, entry.readers));
    }
  } else {
    bundle.consumer_route = route;
  }
  return Status::Ok();
}
bool same_projection(ResourceProjection a, ResourceProjection b) noexcept {
  return a.capacity == b.capacity && a.metadata == b.metadata && a.reader == b.reader &&
      a.capability == b.capability && std::ranges::equal(a.access_nodes, b.access_nodes);
}
bool same_ready_guard(const compute::ReadLease &left, const compute::ReadLease &right) noexcept {
  if (compute::detail::LeaseAccess::identity(left) != compute::detail::LeaseAccess::identity(right) ||
      left.generation() != right.generation()) return false;
  const auto a = left.view(), b = right.view();
  return a && b && same_resource_range(*a, *b);
}
Result<InputSource> bound_output_bundle(Invocation &invocation) noexcept {
  const auto *state = InvocationAccess::state(invocation);
  if (!state)
    return std::unexpected(Status::FailedPrecondition("output bundle has no original consumer"));
  HYLE_ASSIGN_OR_RETURN(auto owner, bound_input_source(*state, &output_source_table));
  const auto *bundle = static_cast<const OutputBindingBundle *>(owner.state());
  if (bundle->consumer != state || owner.identity() != bundle->submission.get())
    return std::unexpected(Status::FailedPrecondition("output bundle was not bound to this original consumer"));
  return owner;
}
} // namespace

Result<StoreProjectionInputs> RegionAccess::store_source_projection(
    const InvocationState *state, ResourceProjection projection) noexcept {
  if (!state || !state->plan || !state->resources)
    return std::unexpected(Status::FailedPrecondition("Store projection has no original invocation resources"));
  auto attached = bound_input_source(*state, projection.capacity, &output_source_table);
  if (!attached)
    return std::unexpected(Status::FailedPrecondition("Store projection has no exact original Source bundle"));
  const auto &bundle = *static_cast<const OutputBindingBundle *>(attached.state());
  if (bundle.consumer != state || attached.identity() != bundle.submission.get() ||
      !input_source_matches(*state, projection.metadata, attached))
    return std::unexpected(Status::FailedPrecondition("Store projection source or IDs attachment was replaced"));
  const auto found = std::ranges::find_if(bundle.entries, [&](const auto &entry) {
    return entry.store && same_projection(entry.projection, projection);
  });
  if (found == bundle.entries.end() || found->store->bundle != &bundle ||
      found->store->submission.get() != bundle.submission.get())
    return std::unexpected(Status::InvalidArgument("Store projection differs from its actual retained Store entry"));
  HYLE_ASSIGN_OR_RETURN(auto access, projection_access(*state->plan, state, projection));
  if (found->store_ready.empty() && !access.allows_pending)
    return std::unexpected(Status::Unimplemented("selected Store reader cannot import pending source permissions"));
  HYLE_RETURN_IF_ERROR(validate_bound_store(*state, *found));
  const auto step = state->plan->node_steps[projection.reader.ordinal()];
  for (const auto &binding : state->resources.get()->bindings) {
    if (binding.writer_step != step || !binding.projected_writer) continue;
    const auto root = compute::detail::LeaseAccess::identity(*binding.projected_writer);
    for (const auto &ticket : found->store->tickets)
      if (root == compute::detail::LeaseAccess::identity(ticket))
        return std::unexpected(Status::InvalidArgument("indexed Store destination aliases its imported source slot"));
  }
  return StoreProjectionInputs{found->store->capacity, found->store->indices};
}
} // namespace detail

using namespace detail;

SubmissionSource::SubmissionSource(SubmissionSource &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
SubmissionSource &SubmissionSource::operator=(SubmissionSource &&other) noexcept {
  if (this != &other) {
    if (state_) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
SubmissionSource::~SubmissionSource() { if (state_) state_->decref(); }
Result<SubmissionSource> SubmissionSource::create(Invocation &invocation,
    compute::Completion &&submission, const compute::StorageOwner &storage) noexcept {
  auto *state = InvocationAccess::state(invocation);
  if (!state || !storage || !state->observation.owns(submission))
    return std::unexpected(Status::InvalidArgument("submission source requires its exact submitted Invocation proxy and metadata owner"));
  HYLE_ASSIGN_OR_RETURN(auto queue, output_queue(*state));
  const auto &plan = *state->plan;
  HYLE_ASSIGN_OR_RETURN(auto context, ExecutionSourceAccess::compute(
      plan.targets[plan.lanes[plan.segments.front().lane.ordinal()].target.ordinal()].source));
  if (context->identity_key() != storage.context_identity())
    return std::unexpected(Status::InvalidArgument("submission metadata owner belongs to another Context"));
  HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(submission));
  (void)dependency;
  const auto allocator = storage.allocator();
  auto *memory = allocator.allocate(sizeof(SubmissionSourceState), alignof(SubmissionSourceState));
  if (!memory) return std::unexpected(Status::ResourceExhausted("submission source cold owner metadata"));
  auto *retention_memory = allocator.allocate(sizeof(OutputStorageRetention), alignof(OutputStorageRetention));
  if (!retention_memory) {
    allocator.deallocate(memory, sizeof(SubmissionSourceState), alignof(SubmissionSourceState));
    return std::unexpected(Status::ResourceExhausted("submission source storage retirement metadata"));
  }
  auto *retained_storage = ::new (retention_memory) OutputStorageRetention(storage);
  const auto attached = retained_storage->retirement.attach(submission, retained_storage,
      &OutputStorageRetention::release);
  if (!attached.ok()) {
    OutputStorageRetention::release(retained_storage);
    allocator.deallocate(memory, sizeof(SubmissionSourceState), alignof(SubmissionSourceState));
    return std::unexpected(attached);
  }
  auto *owner = ::new (memory) SubmissionSourceState(state, std::move(queue), storage, std::move(submission));
  return SubmissionSource(owner);
}
Result<compute::CompletionPoll> SubmissionSource::poll() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("submission source is empty"));
  return poll_submission(*state_);
}
Status SubmissionSource::wait() noexcept {
  if (!state_) return Status::FailedPrecondition("submission source is empty");
  return wait_submission(*state_);
}
Status SubmissionSource::request_cancel() noexcept {
  if (!state_) return Status::FailedPrecondition("submission source is empty");
  std::lock_guard guard(state_->observation);
  return state_->completion.request_cancel();
}
Status SubmissionSource::detach() noexcept {
  if (!state_) return Status::FailedPrecondition("submission source is empty");
  auto &source = *state_;
  std::lock_guard guard(source.observation);
  auto observed = source.completion.poll();
  if (!source.producer) return Status::Ok();
  if (source.borrows)
    return Status::FailedPrecondition("submission still has borrowed output views");
  auto *original = source.producer;
  if ((observed && *observed == compute::CompletionPoll::kPending) ||
      original->phase.load(std::memory_order_acquire) == InvocationPhase::kRunning ||
      original->preparation_uses.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("submission producer has not actually retired");
  // Actual retirement closes export independently of the mathematical verdict.
  source.producer = nullptr;
  original->results.fetch_sub(1, std::memory_order_release);
  release(original);
  return Status::Ok();
}
Status SubmissionSource::visit(NodeId node, RegionAccessFn callback, void *user) const noexcept {
  if (!state_ || !callback)
    return Status::InvalidArgument("submission metadata visit needs its source and visitor");
  InvocationState *original = nullptr;
  {
    std::lock_guard guard(state_->observation);
    original = state_->producer;
    if (!original) return Status::FailedPrecondition("submission source export is closed");
    retain(original);
    original->results.fetch_add(1, std::memory_order_relaxed);
  }
  struct ReleaseVisit final {
    InvocationState *state;
    ~ReleaseVisit() {
      state->results.fetch_sub(1, std::memory_order_release);
      release(state);
    }
  } pin{original};
  return RegionAccess::visit_pinned(original, node, callback, user);
}
namespace detail {
Status RegionAccess::visit_pinned(InvocationState *original, NodeId node,
    RegionAccessFn callback, void *user) noexcept {
  const auto &plan = *original->plan;
  if (GraphIdentity::generation(node) != plan.generation || node.ordinal() >= plan.node_steps.size())
    return Status::InvalidArgument("submission metadata visitor names a foreign node");
  const auto index = plan.node_steps[node.ordinal()];
  if (index >= plan.steps.size() || index >= original->invocations.size())
    return Status::InvalidArgument("submission metadata visitor has no original selected step");
  const auto &step = plan.steps[index];
  if (step.kind != PlanStepKind::kRegion || !step.region.valid() ||
      !original->invocations[index].region.valid())
    return Status::Unimplemented("submission metadata visitor needs its selected region");
  if (step.node_set_begin > plan.step_nodes.size() ||
      step.node_count > plan.step_nodes.size() - step.node_set_begin)
    return Status::Internal("submission metadata region has invalid original nodes");
  const auto nodes = std::span<const std::uint32_t>(plan.step_nodes).subspan(
      step.node_set_begin, step.node_count);
  if (std::ranges::find(nodes, node.ordinal()) == nodes.end())
    return Status::InvalidArgument("submission metadata node is outside its selected region");
  const RegionAccessView view{index, plan.generation, nodes,
      step.region.source_.shared_extension(), step.region.prepared_.state,
      original->invocations[index].region.state_, step.lane, {}, {}, step.region.prepared_.vtable};
  return callback(user, view);
}
ResourceSource::ResourceSource(ResourceSource &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
ResourceSource &ResourceSource::operator=(ResourceSource &&other) noexcept {
  if (this != &other) {
    if (state_) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
ResourceSource::~ResourceSource() { if (state_) state_->decref(); }
bool ResourceSource::valid() const noexcept { return state_ && state_->submission; }
compute::ConstTensorView ResourceSource::capacity() const noexcept {
  return valid() ? state_->capacity : compute::ConstTensorView{};
}
std::span<const compute::ReadLease> ResourceSource::ready_pages() const noexcept {
  return valid() ? std::span<const compute::ReadLease>(state_->permissions->ready) : std::span<const compute::ReadLease>{};
}
std::span<const compute::detail::PendingReadTicket> ResourceSource::full_tickets() const noexcept {
  return valid() ? std::span<const compute::detail::PendingReadTicket>(state_->permissions->full)
      : std::span<const compute::detail::PendingReadTicket>{};
}
std::span<const compute::detail::PendingAppendReadTicket> ResourceSource::append_tickets() const noexcept {
  return valid() ? std::span<const compute::detail::PendingAppendReadTicket>(state_->permissions->append)
      : std::span<const compute::detail::PendingAppendReadTicket>{};
}
namespace {
bool source_retains_allocator(const compute::StorageOwner &, mem::AllocatorRef) noexcept;
}
Result<ResourceSource> ResourceSource::prepare(Invocation &invocation, ResourceSourceCounts counts,
    const compute::StorageOwner &storage, ResourceOptions options) noexcept {
  auto *parent = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(parent));
  if (!storage || !options.allocator.valid() || !parent->resources ||
      (counts.full == 0 && counts.append == 0))
    return std::unexpected(Status::InvalidArgument("resource source needs original resources, mutable pages and its owner"));
  if (counts.ready > UINT32_MAX || counts.full > UINT32_MAX - counts.ready ||
      counts.append > UINT32_MAX - counts.ready - counts.full ||
      counts.inherited > UINT32_MAX - counts.ready - counts.full - counts.append)
    return std::unexpected(Status::ResourceExhausted("resource source permission count overflows"));
  if (std::ranges::none_of(parent->plan->storage_domains, [&](const auto &domain) {
        return domain.owner && domain.owner.ref() == storage.ref() &&
            domain.owner.context_identity() == storage.context_identity();
      }))
    return std::unexpected(Status::FailedPrecondition("resource source Plan must retain its actual storage owner"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!source_retains_allocator(storage, allocator))
    return std::unexpected(Status::InvalidArgument("resource source owner cannot retain its metadata allocator"));
  auto *memory = allocator.allocate(sizeof(ResourceSourceState), alignof(ResourceSourceState));
  if (!memory) return std::unexpected(Status::ResourceExhausted("resource source fixed owner metadata"));
  auto owner = util::adopt_ref(::new (memory) ResourceSourceState(storage, options.resource_budget, allocator));
  owner->options = options; owner->options.allocator = allocator;
  owner->prepared_parent = parent; retain(parent);
  auto *permission_memory = allocator.allocate(sizeof(ResourcePermissionBlock), alignof(ResourcePermissionBlock));
  if (!permission_memory)
    return std::unexpected(Status::ResourceExhausted("resource source fixed permission origin"));
  owner->permissions = util::adopt_ref(::new (permission_memory)
      ResourcePermissionBlock(storage, options.resource_budget, allocator));
  try {
    auto &permissions = *owner->permissions;
    permissions.ready.resize(counts.ready); permissions.full.resize(counts.full);
    permissions.append.resize(counts.append);
    const auto own_count = counts.ready + counts.full + counts.append;
    permissions.inherited_uses.resize(own_count);
    permissions.bounds.resize(own_count); permissions.published.resize(own_count);
    owner->inherited.resize(counts.inherited);
    owner->lookup.resize(own_count + counts.inherited);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("resource source fixed permission metadata"));
  }
  return ResourceSource(owner.release());
}
Status ResourceSource::capture(const SubmissionSource &submission, ResourceContribution contribution) noexcept {
  if (!state_ || !submission.state_)
    return Status::FailedPrecondition("resource source needs cold preparation and its original submission");
  auto &view = *state_;
  auto &permissions = *view.permissions;
  auto &source = *submission.state_;
  std::lock_guard guard(source.observation);
  auto *parent = source.producer;
  if (view.submission || !parent || parent != view.prepared_parent || !parent->resources)
    return Status::FailedPrecondition("resource contribution differs from its original submitted producer");
  const auto phase = parent->phase.load(std::memory_order_acquire);
  if (phase != InvocationPhase::kRunning &&
      !(phase == InvocationPhase::kIdle && parent->outputs_ready.load(std::memory_order_acquire)))
    return Status::FailedPrecondition("resource source requires a successful-or-pending original producer");
  const auto &plan = *parent->plan;
  const auto proof = contribution.writer;
  if (GraphIdentity::generation(proof.node) != plan.generation ||
      proof.node.ordinal() >= plan.node_steps.size() || plan.node_steps[proof.node.ordinal()] != proof.step ||
      proof.step >= plan.steps.size() || proof.step >= parent->invocations.size() ||
      plan.steps[proof.step].kind != PlanStepKind::kRegion ||
      !parent->invocations[proof.step].region.valid() ||
      !plan.steps[proof.step].region.matches_artifact(proof.implementation, proof.prepared))
    return Status::InvalidArgument("resource contribution lacks its exact original selected writer");
  const auto &episode = *parent->resources.get();
  if (proof.step >= episode.nominated_writers.size() ||
      episode.nominated_writers[proof.step] != proof.node.ordinal())
    return Status::FailedPrecondition("resource contribution is not the original registered external writer");
  if (!contribution.capacity.valid() || contribution.capacity.storage().resource() != view.storage_owner.ref() ||
      source.metadata_owner.context_identity() != view.storage_owner.context_identity() ||
      contribution.ready.size() != permissions.ready.size() || contribution.full.size() != permissions.full.size() ||
      contribution.append.size() != permissions.append.size() || contribution.inherited.size() != view.inherited.size())
    return Status::InvalidArgument("resource contribution differs from its fixed geometry or actual owner");
  const auto capacity = contribution.capacity.storage();
  const auto inside = [&](mem::ConstBufferView range) noexcept {
    return range.valid() && range.resource() == capacity.resource() &&
        range.native_handle() == capacity.native_handle() && range.offset() >= capacity.offset() &&
        range.offset() - capacity.offset() <= capacity.size() &&
        range.size() <= capacity.size() - (range.offset() - capacity.offset());
  };
  const LeaseBinding *projection = nullptr;
  if (contribution.written_projection.valid()) {
    projection = find_binding(parent, contribution.written_projection);
    if (!projection || projection->scope != BindingScope::kWrittenProjection ||
        projection->projection != contribution.written_projection || projection->writer_step != proof.step)
      return Status::InvalidArgument("resource contribution names another written projection");
    HYLE_ASSIGN_OR_RETURN(const auto same,
        RegionAccess::has_input_binding(parent, contribution.written_projection, contribution.capacity));
    if (!same) return Status::FailedPrecondition("resource contribution capacity was reset or replaced");
  }
  const auto belongs = [&](const LeaseBinding &binding) noexcept {
    return projection
        ? binding.scope == BindingScope::kWrittenProjection &&
            (binding.value == contribution.written_projection || binding.projection == contribution.written_projection)
        : binding.scope == BindingScope::kWriteEffect && binding.writer_step == proof.step;
  };
  std::size_t full_count = 0, append_count = 0;
  for (const auto &binding : episode.bindings) {
    if (!belongs(binding)) continue;
    if (binding.projected_writer) {
      if (full_count >= contribution.full.size() ||
          contribution.full[full_count].writer != binding.projected_writer)
        return Status::InvalidArgument("resource contribution full writers differ from the registered set");
      const auto &full = contribution.full[full_count++];
      if(!full.readable_bytes || full.readable_bytes>full.range.size())
        return Status::InvalidArgument("resource full writer needs its exact readable prefix");
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::matches_writer_storage(*full.writer, full.range));
      if (!inside(full.range) || !source_retains_allocator(view.storage_owner,
          compute::detail::LeaseAccess::metadata_backing_allocator(*full.writer)))
        return Status::InvalidArgument("resource source full writer range or allocator is not retained");
    } else if (binding.projected_appender) {
      if (append_count >= contribution.append.size() ||
          contribution.append[append_count].appender != binding.projected_appender)
        return Status::InvalidArgument("resource contribution appenders differ from the registered set");
      const auto &append = contribution.append[append_count++];
      if (!append.appender->valid() || !append.appender->reserved_bytes() ||
          append.appender->committed_bytes() > SIZE_MAX - append.appender->reserved_bytes() ||
          append.prefix.size() != append.appender->committed_bytes() + append.appender->reserved_bytes() ||
          !inside(append.prefix))
        return Status::InvalidArgument("resource contribution append prefix differs from the reserved epoch");
      HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::matches_append_prefix_storage(
          *append.appender, append.prefix));
      bool retained = false;
      for (const auto &prefix : episode.bindings) {
        if (!belongs(prefix) || !prefix.read.valid() ||
            compute::detail::LeaseAccess::identity(prefix.read) !=
                compute::detail::LeaseAccess::identity(*append.appender)) continue;
        HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*append.appender, prefix.read));
        if (!source_retains_allocator(view.storage_owner,
            compute::detail::LeaseAccess::metadata_backing_allocator(prefix.read)))
          return Status::InvalidArgument("resource source cannot retain its append prefix allocator");
        retained = true;
      }
      if (!retained) return Status::InvalidArgument("resource append has no original registered prefix owner");
    }
  }
  if (full_count != contribution.full.size() || append_count != contribution.append.size())
    return Status::InvalidArgument("resource contribution contains unregistered mutable page authority");
  // Ready guards describe the typed successor's actual untouched pages. The
  // original writer may also read COW copy sources and private controls; those
  // do not become receiver permissions. Every supplied guard is still proved
  // against this exact original registration, without widening its range.
  for (std::size_t i = 0; i < contribution.ready.size(); ++i) {
    const auto &ready = contribution.ready[i];
    const auto root = compute::detail::LeaseAccess::identity(ready);
    HYLE_ASSIGN_OR_RETURN(auto actual, ready.view());
    if (!inside(actual) || !source_retains_allocator(view.storage_owner,
        compute::detail::LeaseAccess::metadata_backing_allocator(ready)))
      return Status::InvalidArgument("resource ready page range or allocator is not retained");
    for (std::size_t j = 0; j < i; ++j)
      if (compute::detail::LeaseAccess::identity(contribution.ready[j]) == root)
        return Status::InvalidArgument("resource source repeats a ready page root");
    for (const auto &full : contribution.full)
      if (compute::detail::LeaseAccess::identity(*full.writer) == root)
        return Status::InvalidArgument("resource source mixes ready and full permissions of one root");
    for (const auto &append : contribution.append)
      if (compute::detail::LeaseAccess::identity(*append.appender) == root)
        return Status::InvalidArgument("resource source must cover an append prefix with its one append ticket");
    bool matched = false;
    for (const auto &binding : episode.bindings) {
      if (!belongs(binding) || !binding.read.valid() ||
          compute::detail::LeaseAccess::identity(binding.read) != root ||
          ready.generation() != binding.read.generation()) continue;
      HYLE_ASSIGN_OR_RETURN(auto registered, binding.read.view());
      if (actual.resource() == registered.resource() && actual.native_handle() == registered.native_handle() &&
          actual.offset() >= registered.offset() && actual.offset() - registered.offset() <= registered.size() &&
          actual.size() <= registered.size() - (actual.offset() - registered.offset())) matched = true;
    }
    if (!matched) return Status::InvalidArgument("resource ready page exceeds its original registered guard");
  }
  struct DiscardTickets final {
    ResourceSourceState &view;
    bool committed = false;
    ~DiscardTickets() {
      if (committed) return;
      for (auto &ticket : view.permissions->full) ticket.reset();
      for (auto &ticket : view.permissions->append) ticket.reset();
      for (auto &item : view.inherited) item = {};
    }
  } rollback{view};
  const auto own_count = permissions.bounds.size();
  std::size_t cursor = 0;
  for (const auto &ready : contribution.ready) {
    HYLE_ASSIGN_OR_RETURN(permissions.bounds[cursor], ready.view());
    ++cursor;
  }
  for (const auto &full : contribution.full) {
    HYLE_ASSIGN_OR_RETURN(permissions.bounds[cursor], full.range.subview(0, full.readable_bytes));
    ++cursor;
  }
  for (const auto &append : contribution.append) permissions.bounds[cursor++] = append.prefix;
  for (std::size_t i = 0; i < own_count; ++i)
    view.lookup[i] = {view.permissions.get(), i, permissions.bounds[i]};
  // An ancestor's bounds alone are not authority. This producer must still
  // hold that exact predecessor through the nominated writer's real effect
  // attachment. Its submission pin keeps these original attachments stable.
  for (std::size_t i = 0; i < contribution.inherited.size(); ++i) {
    const auto &incoming = contribution.inherited[i];
    const auto *previous = OutputSourceAccess::state(incoming.source);
    if (!previous || previous == &view || !previous->submission ||
        previous->submission.get() == &source || !inside(incoming.prefix) || !incoming.prefix.size())
      return Status::InvalidArgument("inherited resource needs a bounded original predecessor permission");
    std::lock_guard previous_guard(previous->submission->observation);
    auto *bundle = previous->bundle;
    if (!bundle || bundle->consumer != parent || bundle->submission.get() != previous->submission.get() ||
        std::ranges::none_of(bundle->entries, [&](const auto &entry) {
          return entry.resource.get() == previous && entry.effect.node == proof.node &&
              entry.effect.step == proof.step && entry.effect.implementation == proof.implementation &&
              entry.effect.prepared == proof.prepared;
        }))
      return Status::InvalidArgument("inherited resource lacks this producer's exact predecessor effect");
    HYLE_ASSIGN_OR_RETURN(auto attachment, output_input_source(bundle));
    if (!effect_source_matches(*parent, proof.node, attachment))
      return Status::FailedPrecondition("inherited predecessor effect was reset or replaced");
    const auto *permission = find_resource_permission(*previous, incoming.prefix);
    if (!permission)
      return Status::InvalidArgument("inherited resource exceeds its predecessor's actual readable prefix");
    // Flatten to the root origin immediately, including inheritance of an
    // already inherited page. Neither Source nor terminal ancestry is retained.
    HYLE_ASSIGN_OR_RETURN(auto inherited, ResourceInheritedPermission::acquire(
        permission->origin, permission->index));
    view.inherited[i] = std::move(inherited);
    view.lookup[own_count + i] = {permission->origin, permission->index, incoming.prefix};
  }
  std::sort(view.lookup.begin(), view.lookup.end(), [](const auto &left, const auto &right) {
    return left.bound.offset() < right.bound.offset();
  });
  for (std::size_t i = 0; i < view.lookup.size(); ++i) {
    if (!view.lookup[i].bound.size() || (i &&
        overlaps_resource_range(view.lookup[i - 1].bound, view.lookup[i].bound)))
      return Status::InvalidArgument("resource contribution repeats or overlaps a readable page");
  }
  // All inherited, address and authority checks finish before the first
  // opt-in writer seal. The following exports allocate no metadata.
  for (std::size_t i = 0; i < contribution.full.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto ticket, compute::detail::LeaseAccess::pending_read(*contribution.full[i].writer));
    permissions.full[i] = std::move(ticket);
  }
  for (std::size_t i = 0; i < contribution.append.size(); ++i) {
    const auto &append = contribution.append[i];
    HYLE_ASSIGN_OR_RETURN(auto ticket,
        compute::detail::LeaseAccess::pending_append_read(*append.appender, append.prefix));
    permissions.append[i] = std::move(ticket);
  }
  for (std::size_t i = 0; i < contribution.ready.size(); ++i) permissions.ready[i] = contribution.ready[i];
  view.capacity = contribution.capacity;
  view.submission = util::retain_ref(&source);
  auto *original = std::exchange(view.prepared_parent, nullptr);
  rollback.committed = true;
  release(original);
  return Status::Ok();
}
Status ResourceSource::covers_range(mem::ConstBufferView range) const noexcept {
  if(!valid())return Status::FailedPrecondition("resource range requires its captured original source");
  return covers_resource_range(*state_,range);
}
Status ResourceSource::successful_outcome() const noexcept {
  if (!valid()) return Status::FailedPrecondition("resource source is not captured");
  auto &source = *state_->submission;
  std::lock_guard guard(source.observation);
  HYLE_ASSIGN_OR_RETURN(auto observed, source.completion.poll());
  if (observed == compute::CompletionPoll::kPending || (source.producer &&
      (source.producer->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
       !source.producer->outputs_ready.load(std::memory_order_acquire) ||
       source.producer->preparation_uses.load(std::memory_order_acquire))))
    return Status::Unavailable("resource source producer has not successfully retired");
  return Status::Ok();
}
Status ResourceSource::publish_ready(std::span<const compute::ReadLease> ready) const noexcept {
  HYLE_RETURN_IF_ERROR(successful_outcome());
  auto &view = *state_;
  if (ready.size() != view.lookup.size())
    return Status::InvalidArgument("resource publication needs its complete canonical ready footprint");
  // Canonical physical order and an exact count prove coverage without an
  // extra bitmap, allocation or quadratic root search. Check every guard
  // before changing any published permission.
  for (std::size_t i = 0; i < ready.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto range, ready[i].view());
    if (i && range.offset() <= view.lookup[i - 1].bound.offset())
      return Status::InvalidArgument("resource publication is not in canonical physical order");
    HYLE_RETURN_IF_ERROR(validate_published_permission(view.lookup[i], ready[i]));
  }
  auto &origin = *view.permissions;
  std::lock_guard guard(origin.publication);
  for (std::size_t i = 0; i < ready.size(); ++i) {
    const auto &entry = view.lookup[i];
    if (entry.origin != &origin) continue;
    // Exact own bounds were checked above, so this is a no-fail ref assignment.
    // Inherited guards may be clipped; they never overwrite their origin slot.
    origin.published[entry.index] = ready[i];
  }
  return Status::Ok();
}
} // namespace detail

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
Result<OutputSource> OutputSource::create(const SubmissionSource &submission, ValueId output,
    const compute::StorageOwner &storage, const compute::LeasePool &pool) noexcept {
  if (!submission.state_ || !storage)
    return std::unexpected(Status::InvalidArgument("output view requires a submission source and actual storage owner"));
  auto &source = *submission.state_;
  std::lock_guard guard(source.observation);
  auto *state = source.producer;
  if (!state) return std::unexpected(Status::FailedPrecondition("submission source export is closed"));
  const auto phase = state->phase.load(std::memory_order_acquire);
  if (phase != InvocationPhase::kRunning &&
      !(phase == InvocationPhase::kIdle && state->outputs_ready.load(std::memory_order_acquire)))
    return std::unexpected(Status::FailedPrecondition("output view requires an accepted successful-or-pending producer"));
  const auto &plan = *state->plan;
  if (std::find(plan.output_ids.begin(), plan.output_ids.end(), output) == plan.output_ids.end())
    return std::unexpected(Status::InvalidArgument("output source value is not an original published output"));
  auto *binding = find_binding(state, output);
  if (!binding || binding->input || !binding->exclusive || !binding->write.valid() ||
      binding->scope != BindingScope::kCaller)
    return std::unexpected(Status::Unimplemented("output relay requires its original caller-owned output writer"));
  const auto &description = binding->description;
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if (!description.is_plain() || !description.is_dense() || !bytes)
    return std::unexpected(Status::Unimplemented("output relay requires a nonempty plain dense output"));
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
  if (storage.context_identity() != source.metadata_owner.context_identity())
    return std::unexpected(Status::InvalidArgument("output storage owner belongs to another Context"));
  if (source.borrows == SIZE_MAX)
    return std::unexpected(Status::ResourceExhausted("submission output borrow count"));
  auto *memory = storage.allocator().allocate(sizeof(OutputSourceState), alignof(OutputSourceState));
  if (!memory) return std::unexpected(Status::ResourceExhausted("output view cold owner metadata"));
  return OutputSource(::new (memory) OutputSourceState(&source, output,
      &binding->write, &binding->read, tensor, storage));
}
Result<compute::CompletionPoll> OutputSource::poll() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("output source is empty"));
  return poll_submission(*state_->submission);
}
Status OutputSource::wait() noexcept {
  if (!state_) return Status::FailedPrecondition("output source is empty");
  return wait_submission(*state_->submission);
}
Status OutputSource::detach() noexcept {
  if (!state_) return Status::FailedPrecondition("output source is empty");
  auto &view = *state_;
  auto &source = *view.submission;
  std::lock_guard guard(source.observation);
  if (!view.writer) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto observed, source.completion.poll());
  if (observed == compute::CompletionPoll::kPending)
    return Status::Unavailable("output producer is still pending");
  auto *original = source.producer;
  if (!original || original->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
      !original->outputs_ready.load(std::memory_order_acquire) ||
      original->preparation_uses.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("successful output producer has not actually retired");
  if (!view.writer->valid() && !view.ready->valid())
    return Status::FailedPrecondition("successful output has no original read permission");
  if (view.writer->valid()) {
    HYLE_ASSIGN_OR_RETURN(auto published,
        compute::detail::LeaseAccess::publish_after_writes(*view.writer));
    *view.ready = std::move(published);
  }
  view.retained_output = *view.ready;
  view.writer = nullptr; view.ready = nullptr;
  --source.borrows;
  return Status::Ok();
}
namespace detail {
namespace {
bool source_retains_allocator(const compute::StorageOwner &storage, mem::AllocatorRef selected) noexcept {
  const auto actual = compute::detail::ResourceBudgetAccess::original_allocator(selected);
  const auto system = mem::system_allocator();
  return actual.valid() && (storage.owns_allocator(actual) ||
      (actual.alloc_fn() == system.alloc_fn() && actual.free_fn() == system.free_fn() &&
       actual.realloc_fn() == system.realloc_fn() && actual.state() == system.state()));
}
Status validate_store_reader(const Graph *graph, ResourceProjection projection,
    const StoreSourceState &store, const InvocationState &invocation) noexcept {
  if (!graph || !graph->valid())
    return Status::InvalidArgument("Store source binding needs its original logical Graph");
  const auto &plan = *invocation.plan;
  if (GraphIdentity::generation(projection.reader) != plan.generation ||
      projection.reader.ordinal() >= plan.node_steps.size())
    return Status::InvalidArgument("Store source reader differs from its concrete Plan");
  HYLE_ASSIGN_OR_RETURN(auto operation, graph->operation(projection.reader));
  const auto *gather = operation.get_if<compute::Gather>();
  if (!gather || (gather->axis != 0 && gather->axis != -2) ||
      gather->bounds != compute::IndexBoundsPolicy::kChecked)
    return Status::InvalidArgument("Store source reader must be its checked axis-zero Gather");
  HYLE_ASSIGN_OR_RETURN(auto capacity, graph->node_input(projection.reader, 0));
  HYLE_ASSIGN_OR_RETURN(auto metadata, graph->node_input(projection.reader, 1));
  if (capacity != projection.capacity || metadata != projection.metadata)
    return Status::InvalidArgument("Store source projection does not name its actual Gather pool and IDs");
  HYLE_ASSIGN_OR_RETURN(auto output, graph->result(projection.reader, 0));
  const auto descriptions = plan.domain_graph.valid()
      ? std::span<const compute::TensorDesc>(invocation.concrete_descriptions)
      : std::span<const compute::TensorDesc>(plan.descriptions);
  for (const auto value : {capacity, metadata, output})
    if (GraphIdentity::generation(value) != plan.generation || value.ordinal() >= descriptions.size())
      return Status::InvalidArgument("Store source logical reader differs from its concrete Plan");
  const auto &capacity_description = descriptions[capacity.ordinal()];
  const auto &ids_description = descriptions[metadata.ordinal()];
  const auto &output_description = descriptions[output.ordinal()];
  HYLE_ASSIGN_OR_RETURN(auto expected_output, compute::TensorDesc::dense(store.schema.type(),
      std::array<std::int64_t, 2>{static_cast<std::int64_t>(store.tickets.size()), store.schema.item_elements()}));
  if (capacity_description != store.capacity.description() || ids_description != store.indices.description() ||
      output_description != expected_output)
    return Status::InvalidArgument("Store source Gather must read its exact whole rows and ready IDs");
  return Status::Ok();
}
struct PrepareStoreSource final {
  NodeId node{};
  slots::detail::StoreBinding *binding = nullptr;
  slots::Schema schema{};
  std::size_t count = 0;
  ExecutionLaneId lane{};
  mem::MemoryResourceRef resource{};
};
Result<StoreSourceState *> prepare_store_source(Invocation &invocation, NodeId node,
    const compute::StorageOwner &storage, const StoreDestinationRange *selected,
    ResourceOptions options) noexcept {
  auto *parent = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(parent));
  if (!storage || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Store source needs its actual owner and cold allocator"));
  PrepareStoreSource call{.node = node};
  HYLE_RETURN_IF_ERROR(RegionAccess::visit(invocation, node,
      [](void *opaque, const RegionAccessView &view) noexcept -> Status {
        auto &call = *static_cast<PrepareStoreSource *>(opaque);
        const auto *table = view.prepared_vtable && view.prepared_vtable->query_extension
            ? static_cast<const slots::detail::StoreRegionVTable *>(
                view.prepared_vtable->query_extension(view.prepared, slots::detail::kStoreBindingExtensionId))
            : nullptr;
        if (!table || !table->inspect || !table->binding)
          return Status::Unimplemented("Store source needs its original selected Store owner");
        const ProviderRegionPrepared prepared{view.prepared_vtable, view.prepared};
        HYLE_ASSIGN_OR_RETURN(auto facts, table->inspect(prepared, view.invocation));
        auto *binding = table->binding(view.invocation);
        if (facts.store != call.node || !binding || !facts.destination_count || !facts.schema.valid())
          return Status::InvalidArgument("Store source has no exact original destination geometry");
        HYLE_ASSIGN_OR_RETURN(auto exported, binding->export_pending());
        if (!exported.transaction || exported.transaction->destination_count() != facts.destination_count)
          return Status::FailedPrecondition("Store source transaction changed during preparation");
        call.binding = binding; call.schema = facts.schema;
        call.count = facts.destination_count; call.lane = view.lane; call.resource = facts.resource;
        return Status::Ok();
      }, &call));
  const auto first = selected ? selected->first : 0;
  const auto count = selected ? selected->count : call.count;
  if (!count || first > call.count || count > call.count - first)
    return std::unexpected(Status::InvalidArgument("Store source destination range is outside its original transaction"));
  const auto &plan = *parent->plan;
  if (!call.binding || call.resource != storage.ref() ||
      std::ranges::none_of(plan.storage_domains, [&](const auto &domain) {
        return domain.owner && domain.owner.ref() == storage.ref() &&
            domain.owner.context_identity() == storage.context_identity();
      }))
    return std::unexpected(Status::FailedPrecondition("Store source Plan must retain its actual storage owner"));
  HYLE_ASSIGN_OR_RETURN(auto execution, RegionAccess::execution_source(invocation.executable(), call.lane));
  HYLE_ASSIGN_OR_RETURN(auto context, ExecutionSourceAccess::compute(execution));
  if (context->identity_key() != storage.context_identity())
    return std::unexpected(Status::InvalidArgument("Store source owner belongs to another Context"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!source_retains_allocator(storage, allocator))
    return std::unexpected(Status::InvalidArgument("Store source owner cannot retain its cold metadata allocator"));
  auto *memory = allocator.allocate(sizeof(StoreSourceState), alignof(StoreSourceState));
  if (!memory) return std::unexpected(Status::ResourceExhausted("Store source cold metadata owner"));
  auto owner = util::adopt_ref(::new (memory) StoreSourceState(storage, options.resource_budget, allocator));
  owner->options = options; owner->options.allocator = allocator;
  owner->prepared_parent = parent; retain(parent);
  owner->prepared_binding = call.binding; owner->node = node; owner->schema = call.schema;
  owner->original_destination_count = call.count; owner->first_destination = first;
  if (!owner->tickets.try_resize(count) || !owner->addresses.try_resize(count))
    return std::unexpected(Status::ResourceExhausted("Store source fixed ticket/coordinate metadata"));
  return owner.release();
}
} // namespace
} // namespace detail

StoreSource::StoreSource(StoreSource &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
StoreSource &StoreSource::operator=(StoreSource &&other) noexcept {
  if (this != &other) {
    if (state_) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
StoreSource::~StoreSource() { if (state_) state_->decref(); }
bool StoreSource::valid() const noexcept { return state_ && state_->submission; }
std::size_t StoreSource::destination_count() const noexcept { return state_ ? state_->tickets.size() : 0; }
Result<SourceBinding> StoreSource::reader_binding(Invocation &invocation, const Graph &graph,
    NodeId store, std::span<const compute::ReadLease> ready) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Store reader needs its captured original source"));
  HYLE_ASSIGN_OR_RETURN(auto projection, slots::detail::source_projection(invocation, store));
  HYLE_RETURN_IF_ERROR(validate_store_reader(&graph, projection, *state_, *InvocationAccess::state(invocation)));
  HYLE_RETURN_IF_ERROR(validate_store_ready(*state_, ready));
  return SourceBindingAccess::make({.store = this, .projection = projection, .graph = &graph, .store_ready = ready});
}
Result<StoreSource> StoreSource::prepare(Invocation &invocation, NodeId node,
    const compute::StorageOwner &storage, ResourceOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto state, prepare_store_source(invocation, node, storage, nullptr, options));
  return StoreSource(state);
}
Result<StoreSource> StoreSource::prepare(Invocation &invocation, NodeId node,
    const compute::StorageOwner &storage, StoreDestinationRange selected, ResourceOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto state, prepare_store_source(invocation, node, storage, &selected, options));
  return StoreSource(state);
}
Status StoreSource::capture(const SubmissionSource &submission) noexcept {
  if (!state_ || !submission.state_)
    return Status::FailedPrecondition("Store source requires cold preparation and a submitted source");
  auto &view = *state_;
  auto &source = *submission.state_;
  std::lock_guard guard(source.observation);
  if (view.submission || !source.producer || source.producer != view.prepared_parent ||
      !view.prepared_binding)
    return Status::FailedPrecondition("Store source does not name its original prepared submission");
  const auto phase = source.producer->phase.load(std::memory_order_acquire);
  if (phase != InvocationPhase::kRunning &&
      !(phase == InvocationPhase::kIdle && source.producer->outputs_ready.load(std::memory_order_acquire)))
    return Status::FailedPrecondition("Store source requires a successful-or-pending original producer");
  HYLE_ASSIGN_OR_RETURN(auto exported, view.prepared_binding->export_pending());
  if (!exported.transaction || exported.transaction->destination_count() != view.original_destination_count ||
      exported.transaction->destination_addresses().size() != view.original_destination_count ||
      view.first_destination > view.original_destination_count ||
      view.tickets.size() > view.original_destination_count - view.first_destination ||
      exported.capacity.storage().resource() != view.storage_owner.ref() ||
      exported.indices.storage().resource() != view.storage_owner.ref() ||
      source.metadata_owner.context_identity() != view.storage_owner.context_identity())
    return Status::InvalidArgument("Store source transaction/resource differs from its prepared geometry");
  auto layout = exported.transaction->layout();
  HYLE_ASSIGN_OR_RETURN(auto schema,
      slots::Schema::create(layout.description(), layout.slot_count(), layout.stride_bytes()));
  if (schema != view.schema || !source_retains_allocator(view.storage_owner,
          compute::detail::LeaseAccess::metadata_backing_allocator(exported.indices_guard)))
    return Status::InvalidArgument("Store source layout or ready-ID allocator is not retained");
  const auto writers = exported.transaction->destinations();
  for (const auto &writer : writers)
    if (!source_retains_allocator(view.storage_owner,
            compute::detail::LeaseAccess::metadata_backing_allocator(writer)))
      return Status::InvalidArgument("Store source owner cannot retain its destination allocator");
  if (view.first_destination > SIZE_MAX / sizeof(std::int32_t) ||
      view.tickets.size() > SIZE_MAX / sizeof(std::int32_t) ||
      view.tickets.size() > static_cast<std::uint64_t>(INT64_MAX))
    return Status::OutOfRange("Store source destination-ID range overflows");
  HYLE_ASSIGN_OR_RETURN(auto indices_guard, exported.indices_guard.subview(
      view.first_destination * sizeof(std::int32_t), view.tickets.size() * sizeof(std::int32_t)));
  HYLE_ASSIGN_OR_RETURN(auto indices_description, compute::TensorDesc::dense(compute::DataType::kInt32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(view.tickets.size())}));
  HYLE_ASSIGN_OR_RETURN(auto indices, indices_guard.tensor(indices_description));
  // All allocations, selected authority, coordinates, carrier and allocator
  // checks precede the first seal. Rejection never escapes a partial ticket set.
  struct DiscardTickets final {
    StoreSourceState &view;
    bool committed = false;
    ~DiscardTickets() { if (!committed) for (auto &ticket : view.tickets) ticket.reset(); }
  } rollback{view};
  for (std::size_t i = 0; i < view.tickets.size(); ++i) {
    const auto original = view.first_destination + i;
    HYLE_ASSIGN_OR_RETURN(auto ticket, compute::detail::LeaseAccess::pending_read(writers[original]));
    view.tickets[i] = std::move(ticket);
    view.addresses[i] = exported.transaction->destination_addresses()[original];
  }
  view.layout = std::move(layout);
  view.capacity = exported.capacity; view.indices = indices;
  view.indices_guard = std::move(indices_guard);
  view.submission = util::retain_ref(&source);
  auto *original = std::exchange(view.prepared_parent, nullptr);
  view.prepared_binding = nullptr;
  rollback.committed = true;
  // Common Source still owns the original pin/ref. This release cannot leave
  // a dangling selected-slot borrow or retain the producer after Source.detach.
  release(original);
  return Status::Ok();
}
Result<compute::ReadLease> StoreSource::publish(std::size_t destination,
    compute::WriteLease &writer) const noexcept {
  if (!valid() || destination < state_->first_destination ||
      destination - state_->first_destination >= state_->tickets.size())
    return std::unexpected(Status::InvalidArgument("Store source destination is not captured"));
  auto &source = *state_->submission;
  std::lock_guard guard(source.observation);
  HYLE_ASSIGN_OR_RETURN(auto observed, source.completion.poll());
  if (observed == compute::CompletionPoll::kPending)
    return std::unexpected(Status::Unavailable("Store source producer is still pending"));
  if (source.producer && (source.producer->phase.load(std::memory_order_acquire) != InvocationPhase::kIdle ||
      !source.producer->outputs_ready.load(std::memory_order_acquire) ||
      source.producer->preparation_uses.load(std::memory_order_acquire)))
    return std::unexpected(Status::Unavailable("Store source producer has not actually retired"));
  HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::matches_pending_read(
      state_->tickets[destination - state_->first_destination], writer));
  return compute::detail::LeaseAccess::publish_after_writes(writer);
}
SourceBinding OutputSource::reader_binding(ValueId input) const noexcept {
  return SourceBindingAccess::make({.input = input, .output = this});
}
Result<SourceBinding> StoreSource::reader_binding(Invocation &invocation, const Graph &graph,
    std::span<const NodeId> readers, std::span<const compute::ReadLease> ready) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Store reader needs its captured original source"));
  if (readers.size() != 1)
    return std::unexpected(Status::InvalidArgument("Store source needs one original checked Gather reader"));
  auto *state = InvocationAccess::state(invocation);
  if (!state || !state->plan)
    return std::unexpected(Status::FailedPrecondition("Store reader has no original consumer"));
  HYLE_ASSIGN_OR_RETURN(auto capacity, graph.node_input(readers.front(), 0));
  HYLE_ASSIGN_OR_RETURN(auto metadata, graph.node_input(readers.front(), 1));
  const ResourceProjection projection{capacity, metadata, readers.front(), readers,
      compute::detail::kProjectedInputReadExtensionId};
  HYLE_RETURN_IF_ERROR(validate_store_reader(&graph, projection, *state_, *state));
  HYLE_RETURN_IF_ERROR(validate_store_ready(*state_, ready));
  return SourceBindingAccess::make({.store = this, .projection = projection, .graph = &graph, .store_ready = ready});
}
Status bind_sources(Invocation &invocation, std::span<const SourceBinding> bindings) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (bindings.empty() || bindings.size() > state->plan->input_ids.size()+state->plan->steps.size())
    return Status::InvalidArgument("output bundle has an invalid size");
  if (has_input_source_table(*state, &output_source_table))
    return Status::FailedPrecondition("original consumer already has its complete pending submission bundle");
  const auto &first = SourceBindingAccess::get(bindings.front());
  auto *first_token = OutputSourceAccess::state(first.output);
  auto *first_store = OutputSourceAccess::state(first.store);
  auto *first_resource = OutputSourceAccess::state(first.resource);
  auto *source = first_token ? first_token->submission.get()
      : first_store ? first_store->submission.get()
      : first_resource ? first_resource->submission.get() : nullptr;
  if (!source) return Status::InvalidArgument("output bundle contains an uncaptured view");
  std::size_t incoming_count = 0, node_count = 0, ticket_count = 0, reader_count = 0, ready_input_count = 0,
      effect_read_count = 0, store_ready_count = 0;
  ResourceOptions reserve_options{source->metadata_owner.allocator(), {}};
  for (const auto &descriptor : bindings) {
    const auto &binding = SourceBindingAccess::get(descriptor);
    auto *token = OutputSourceAccess::state(binding.output);
    auto *store = OutputSourceAccess::state(binding.store);
    auto *resource = OutputSourceAccess::state(binding.resource);
    if (static_cast<unsigned>(token != nullptr) + static_cast<unsigned>(store != nullptr) +
        static_cast<unsigned>(resource != nullptr) != 1)
      return Status::InvalidArgument("output binding requires exactly one concrete view");
    if ((token && token->submission.get() != source) ||
        (store && store->submission.get() != source) ||
        (resource && resource->submission.get() != source))
      return Status::InvalidArgument("output bundle must share one original submission");
    if ((token && token->bound_once) || (store && store->bound_once) || (resource && resource->bound_once))
      return Status::FailedPrecondition("output view has already bound its original consumer");
    const auto count = resource && binding.effect.node.valid() ? 0u : store ? 2u : 1u;
    if (incoming_count > state->plan->input_ids.size() ||
        count > state->plan->input_ids.size() - incoming_count)
      return Status::InvalidArgument("output bundle has too many original input bindings");
    incoming_count += count;
    if (!store && !binding.store_ready.empty())
      return Status::InvalidArgument("only Store source entries can supply ready destinations");
    if (store) {
      HYLE_RETURN_IF_ERROR(validate_store_ready(*store, binding.store_ready));
      if (binding.store_ready.size() > SIZE_MAX - store_ready_count)
        return Status::ResourceExhausted("ready Store bundle permission count overflows");
      store_ready_count += binding.store_ready.size();
      HYLE_RETURN_IF_ERROR(validate_store_reader(binding.graph, binding.projection, *store, *state));
      if (binding.input.valid() || binding.projection.access_nodes.size() > SIZE_MAX - node_count ||
          store->tickets.size() > UINT32_MAX - ticket_count)
        return Status::InvalidArgument("Store source projection or permission count is invalid");
      node_count += binding.projection.access_nodes.size(); ticket_count += store->tickets.size();
      reserve_options = store->options;
    }
    if (resource) {
      const bool effect=binding.effect.node.valid();
      if ((effect && (binding.input.valid() ||
              !binding.readers.empty() || !binding.ready_inputs.empty())) ||
          (!effect && (!binding.input.valid() || binding.readers.empty() || !binding.effect_reads.empty())) ||
          binding.readers.size() > SIZE_MAX - reader_count ||
          binding.ready_inputs.size() > SIZE_MAX - ready_input_count ||
          binding.effect_reads.size() > SIZE_MAX-effect_read_count)
        return Status::InvalidArgument("resource source needs its complete selected reader proof");
      const auto permissions = resource->lookup.size();
      if (permissions > UINT32_MAX - ticket_count)
        return Status::ResourceExhausted("resource source bundle permission count overflows");
      reader_count += binding.readers.size(); ready_input_count += binding.ready_inputs.size();
      effect_read_count+=binding.effect_reads.size();
      ticket_count += permissions; reserve_options = resource->options;
    } else if (!binding.readers.empty() || !binding.ready_inputs.empty() ||
        binding.effect.node.valid() || !binding.effect_reads.empty()) {
      return Status::InvalidArgument("only resource sources contribute selected readers and required ready inputs");
    }
  }
  const auto allocator = source->metadata_owner.allocator();
  auto *memory = allocator.allocate(sizeof(OutputBindingBundle), alignof(OutputBindingBundle));
  if (!memory) return Status::ResourceExhausted("output bundle cold owner metadata");
  auto bundle = util::adopt_ref(::new (memory) OutputBindingBundle(source));
  try {
    bundle->entries.reserve(bindings.size()); bundle->replacements.reserve(incoming_count);
    bundle->access_nodes.reserve(node_count);
    bundle->readers.reserve(reader_count); bundle->ready_inputs.reserve(ready_input_count);
    bundle->private_controls.reserve(ready_input_count);
    bundle->effect_reads.reserve(effect_read_count);
    bundle->store_ready.reserve(store_ready_count);
    bundle->release_values.reserve(incoming_count); bundle->release_effects.reserve(bindings.size());
    for (const auto &descriptor : bindings) {
      const auto &binding = SourceBindingAccess::get(descriptor);
      if (auto *token = OutputSourceAccess::state(binding.output)) {
        bundle->entries.push_back({.input = binding.input, .view = util::retain_ref(token)});
        bundle->replacements.push_back({binding.input, token->tensor});
      } else if (auto *store = OutputSourceAccess::state(binding.store)) {
        const auto begin = bundle->access_nodes.size();
        for (const auto node : binding.projection.access_nodes) bundle->access_nodes.push_back(node);
        auto projection = binding.projection;
        projection.access_nodes = std::span<const NodeId>(bundle->access_nodes).subspan(begin,
            binding.projection.access_nodes.size());
        const auto ready_begin = bundle->store_ready.size();
        for (const auto &read : binding.store_ready) bundle->store_ready.push_back(read);
        bundle->entries.push_back({.input = projection.capacity,
            .store = util::retain_ref(store), .projection = projection,
            .store_ready = std::span<const compute::ReadLease>(bundle->store_ready).subspan(
                ready_begin, binding.store_ready.size())});
        bundle->replacements.push_back({projection.capacity, store->capacity});
        bundle->replacements.push_back({projection.metadata, store->indices});
      } else {
        auto *resource = OutputSourceAccess::state(binding.resource);
        const auto reader_begin = bundle->readers.size(), ready_begin = bundle->ready_inputs.size();
        const auto effect_begin=bundle->effect_reads.size();
        for (const auto proof : binding.readers) bundle->readers.push_back(proof);
        for (const auto &ready : binding.ready_inputs) bundle->ready_inputs.push_back(ready);
        for (const auto range:binding.effect_reads)bundle->effect_reads.push_back(range);
        bundle->entries.push_back({.input = binding.input, .resource = util::retain_ref(resource),
            .readers = std::span<const ProjectedReader>(bundle->readers).subspan(reader_begin, binding.readers.size()),
            .ready_inputs = std::span<const ResourceReadyInput>(bundle->ready_inputs).subspan(ready_begin,
                binding.ready_inputs.size()),.effect=binding.effect,
            .effect_reads=std::span<const mem::ConstBufferView>(bundle->effect_reads).subspan(effect_begin,binding.effect_reads.size())});
        if(binding.input.valid())bundle->replacements.push_back({binding.input, resource->capacity});
      }
    }
    for (const auto &replacement : bundle->replacements) bundle->release_values.push_back(replacement.input);
    for (const auto &entry : bundle->entries)
      if (entry.effect.node.valid()) bundle->release_effects.push_back(entry.effect.node);
    std::sort(bundle->release_effects.begin(), bundle->release_effects.end(),
        [](NodeId left, NodeId right) { return left.ordinal() < right.ordinal(); });
    // Exact final input identities and selected projection qualification are
    // all checked before changing the first old binding or view identity.
    for (std::size_t i = 0; i < bundle->replacements.size(); ++i) {
      const auto input = bundle->replacements[i].input;
      const auto *old = find_binding(state, input);
      if (!old || !old->input)
        return Status::InvalidArgument("output relay requires an original external input resource");
      for (std::size_t j = 0; j < i; ++j)
        if (bundle->replacements[j].input == input)
          return Status::InvalidArgument("output bundle repeats an input or Store metadata identity");
    }
    if (ticket_count)
      HYLE_RETURN_IF_ERROR(RegionAccess::reserve_reads(invocation,
          static_cast<std::uint32_t>(ticket_count), reserve_options));
    for (const auto &entry : bundle->entries) {
      if (entry.resource) {
        if(entry.effect.node.valid()) {
          HYLE_RETURN_IF_ERROR(validate_resource_effect(*state,entry));
          continue;
        }
        HYLE_RETURN_IF_ERROR(RegionAccess::validate_pending_projected(invocation,
            entry.input, entry.resource->capacity, entry.readers));
        for (const auto &ready : entry.ready_inputs) {
          for (const auto &replacement : bundle->replacements)
            if (ready.input.valid() && replacement.input == ready.input)
              return Status::InvalidArgument("resource required ready input overlaps a pending bundle replacement");
          HYLE_RETURN_IF_ERROR(validate_resource_ready_input(*state, ready, entry.readers));
          if (ready.region.valid() && std::ranges::none_of(bundle->private_controls, [&](const auto &old) {
                return compute::detail::LeaseAccess::identity(old) ==
                    compute::detail::LeaseAccess::identity(ready.read);
              }))
            bundle->private_controls.push_back(ready.read);
        }
      }
      if (!entry.store) continue;
      if (entry.store_ready.empty())
        HYLE_RETURN_IF_ERROR(RegionAccess::validate_pending_projection(invocation,
            entry.projection, entry.store->capacity));
      else HYLE_RETURN_IF_ERROR(RegionAccess::validate_ready_projection(invocation,
          entry.projection, entry.store->capacity, entry.store_ready));
      HYLE_ASSIGN_OR_RETURN(auto ids, entry.store->indices_guard.view());
      const auto expected = entry.store->indices.storage();
      if (!entry.store->indices_guard.valid() || ids.resource() != expected.resource() ||
          ids.native_handle() != expected.native_handle() || ids.offset() != expected.offset() ||
          ids.size() != expected.size())
        return Status::FailedPrecondition("Store source lost its exact ready-ID guard");
    }
    HYLE_ASSIGN_OR_RETURN(bundle->all_ready, output_bundle_ready(*bundle));
    if (bundle->all_ready) {
      std::size_t count = 0;
      for (const auto &entry : bundle->entries) if (entry.resource && !entry.effect.node.valid()) {
        if (entry.resource->lookup.size() > SIZE_MAX - count)
          return Status::ResourceExhausted("ready resource guard count overflows");
        count += entry.resource->lookup.size();
      }
      bundle->resource_ready.reserve(count);
      for (auto &entry : bundle->entries) if (entry.resource && !entry.effect.node.valid()) {
        const auto begin = bundle->resource_ready.size();
        for (const auto &permission : entry.resource->lookup) {
          HYLE_ASSIGN_OR_RETURN(auto ready, ready_resource_permission(permission));
          if (!ready.valid()) return Status::FailedPrecondition("ready resource publication was withdrawn");
          bundle->resource_ready.push_back(std::move(ready));
        }
        entry.resource_ready = std::span<const compute::ReadLease>(bundle->resource_ready).subspan(begin);
      }
    }
    HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, *bundle, {}, false));
    HYLE_ASSIGN_OR_RETURN(auto owner, output_input_source(bundle.get()));
    std::vector<InputSourceBinding, mem::StdAllocator<InputSourceBinding>> incoming{
        mem::StdAllocator<InputSourceBinding>(allocator)};
    std::vector<EffectSourceBinding,mem::StdAllocator<EffectSourceBinding>> effects{
        mem::StdAllocator<EffectSourceBinding>(allocator)};
    incoming.reserve(incoming_count);
    effects.reserve(bundle->entries.size());
    for (const auto &replacement : bundle->replacements)
      incoming.push_back({replacement.input, replacement.tensor, owner});
    for(const auto &entry:bundle->entries)if(entry.effect.node.valid())
      effects.push_back({entry.effect.node,owner});
    // Original cold reservation for exact ticket admissions, never a second
    // payload pool. Failure changes no binding; full-model callers still need
    // to reserve bundle/preparation metadata before accepting their predecessor.
    if (bundle->private_controls.size() > UINT32_MAX - ticket_count)
      return Status::ResourceExhausted("resource private control permission count overflows");
    ticket_count += bundle->private_controls.size();
    if (ticket_count)
      HYLE_RETURN_IF_ERROR(RegionAccess::reserve_reads(invocation,
          static_cast<std::uint32_t>(ticket_count), reserve_options));
    HYLE_RETURN_IF_ERROR(InputAccess::bind(invocation, incoming,effects));
    // No fallible work or native acceptance remains. External control is
    // serialized, so no observer can submit between this atomic commit and
    // installing its exact pending/ready resource facts.
    bundle->consumer = state;
    for (const auto &entry : bundle->entries) {
      if(entry.effect.node.valid()) {
        entry.resource->bundle=bundle.get();entry.resource->bound_once=true;
        continue;
      }
      auto *bound = find_binding(state, entry.input);
      bound->borrowed = false; bound->permissionless = false; bound->exclusive = false;
      if (entry.view) {
        bound->description = entry.view->tensor.description();
        if (bundle->all_ready) bound->read = entry.view->retained_output;
        entry.view->bundle = bundle.get(); entry.view->bound_once = true;
      } else if (entry.store) {
        bound->description = entry.store->capacity.description();
        bound->scope = BindingScope::kProjected;
        if (!entry.store_ready.empty()) {
          bound->read = entry.store_ready.front();
          bound->projection = entry.projection.capacity;
          for (std::size_t i = 1; i < entry.store_ready.size(); ++i)
            state->resources.get()->bindings.push_back({.projection = entry.projection.capacity,
                .read = entry.store_ready[i], .borrowed = false, .scope = BindingScope::kProjected});
        }
        auto *ids = find_binding(state, entry.projection.metadata);
        ids->read = entry.store->indices_guard;
        ids->description = entry.store->indices.description();
        ids->borrowed = false; ids->exclusive = false; ids->permissionless = false;
        entry.store->bundle = bundle.get(); entry.store->bound_once = true;
      } else {
        bound->description = entry.resource->capacity.description();
        bound->scope = BindingScope::kProjected;
        if (!entry.resource_ready.empty()) {
          bound->read = entry.resource_ready.front();
          bound->projection = entry.input;
          for (std::size_t i = 1; i < entry.resource_ready.size(); ++i)
            state->resources.get()->bindings.push_back({.projection = entry.input,
                .read = entry.resource_ready[i], .borrowed = false, .scope = BindingScope::kProjected});
        }
        bound->permission_changes_preserve_commands = std::all_of(entry.readers.begin(), entry.readers.end(),
            [](const auto &reader) { return reader.permission_changes_preserve_commands; });
        entry.resource->bundle = bundle.get(); entry.resource->bound_once = true;
      }
    }
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("output bundle cold binding metadata");
  }
  return Status::Ok();
}
Status release_sources(Invocation &invocation) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto owner, bound_output_bundle(invocation));
  auto *bundle = static_cast<OutputBindingBundle *>(owner.state());
  auto *state = InvocationAccess::state(invocation);
  if (!RegionAccess::outcome_available(invocation) ||
      state->preparation_uses.load(std::memory_order_acquire))
    return Status::Unavailable("source bundle still has native work, results, observations or preparation");
  const auto &episode = *state->resources.get();
  for (const auto &replacement : bundle->replacements) {
    if (!input_source_matches(*state, replacement.input, owner))
      return Status::FailedPrecondition("source release lost its exact original input attachment");
    HYLE_ASSIGN_OR_RETURN(auto same, RegionAccess::has_input_binding(state,
        replacement.input, replacement.tensor));
    if (!same) return Status::FailedPrecondition("source release input tuple was reset or replaced");
    for (const auto &consumer : projected_input_consumers(*state->plan, replacement.input.ordinal()))
      if (consumer.step < episode.nominated_writers.size() &&
          episode.nominated_writers[consumer.step] != UINT32_MAX)
        return Status::FailedPrecondition("take the affected original writers before releasing their source bundle");
  }
  for (const auto &entry : bundle->entries) {
    if (entry.effect.node.valid()) {
      if (!effect_source_matches(*state, entry.effect.node, owner))
        return Status::FailedPrecondition("source release effect attachment was replaced");
      if (entry.effect.step < episode.nominated_writers.size() &&
          episode.nominated_writers[entry.effect.step] != UINT32_MAX)
        return Status::FailedPrecondition("take the original effect before releasing its source bundle");
    } else if (entry.store) {
      HYLE_RETURN_IF_ERROR(validate_bound_store(*state, entry));
    }
  }
  HYLE_RETURN_IF_ERROR(RegionAccess::withdraw_bindings_impl(invocation,
      bundle->release_values, false, &owner));
  // Original withdrawal has completed every fallible/native qualification.
  // The local owner keeps this complete bundle alive through attachment drops.
  release_bound_input_sources(*state, bundle->release_values, bundle->release_effects);
  for (const auto &entry : bundle->entries) {
    if (entry.view && entry.view->bundle == bundle) entry.view->bundle = nullptr;
    if (entry.store && entry.store->bundle == bundle) entry.store->bundle = nullptr;
    if (entry.resource && entry.resource->bundle == bundle) entry.resource->bundle = nullptr;
  }
  bundle->consumer = nullptr;
  return Status::Ok();
}
Status prepare_sources(Invocation &invocation, std::span<const QueueBinding> queues) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto owner, bound_output_bundle(invocation));
  auto *bundle = static_cast<OutputBindingBundle *>(owner.state());
  HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, *bundle, queues, true));
  if (queues.size() != 1) return Status::InvalidArgument("output relay requires one actual Queue");
  if (bundle->submitted) return Status::FailedPrecondition("output bundle consumer was already submitted");
  bundle->preparing = true;
  struct EndPreparation final {
    OutputBindingBundle *bundle;
    ~EndPreparation() { bundle->preparing = false; }
  } end{bundle};
  if (bundle->all_ready) {
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues,
        {.mode = ReplayMode::kPreferCapture, .host_wait = HostWaitPolicy::kReject}));
    bundle->consumer_route = invocation.replay_route();
    return Status::Ok();
  }
  if (bundle->consumer_route != ReplayRoute::kCommands)
    return ReplayAccess::prepare_reuse(invocation, queues);
  return invocation.prepare_replay(queues,
      {.mode = ReplayMode::kCommands, .host_wait = HostWaitPolicy::kReject});
}
Result<compute::Completion> submit_sources(Invocation &invocation,
    std::span<const QueueBinding> queues) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto owner, bound_output_bundle(invocation));
  auto *bundle = static_cast<OutputBindingBundle *>(owner.state());
  HYLE_RETURN_IF_ERROR(check_output_consumer(invocation, *bundle, queues, true));
  const auto expected = bundle->consumer_route;
  if (queues.size() != 1 || invocation.replay_route() != expected)
    return std::unexpected(Status::FailedPrecondition("output relay lost its selected prepared replay route"));
  if (bundle->submitted)
    return std::unexpected(Status::FailedPrecondition("output bundle consumer was already submitted"));
  const OrderedGraphSubmission ordered{
      [](void *opaque, ExecutionLaneId, compute::detail::QueueSequence &sequence,
          compute::LeaseUseBatch &uses) noexcept -> Status {
        auto &owner = *static_cast<OutputBindingBundle *>(opaque);
        auto &source = *owner.submission;
        if (!source.queue.same_owner(sequence))
          return Status::InvalidArgument("ordered output reader has another actual Queue");
        std::lock_guard guard(source.observation);
        for (const auto &read : owner.private_controls)
          HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::read_unique(uses, read));
        for (const auto &entry : owner.entries) {
          // The normal resource episode already admits every materialized
          // ready input guard. read_unique requires its caller to deduplicate;
          // only implicit effects still need their separate range admission.
          if (owner.all_ready && !entry.effect.node.valid()) continue;
          if (entry.resource) {
            const auto status = admit_resource_permissions(uses, *entry.resource,
                entry.effect_reads, entry.effect.node.valid());
            if (!status.ok()) {
              // Admission/control failure must not replace a real producer
              // verdict (including a failure inherited through its sources).
              auto observed = source.completion.poll();
              if (!observed) return observed.error();
              return status;
            }
            continue;
          }
          if (entry.store) {
            HYLE_RETURN_IF_ERROR(admit_source_permissions(uses, entry.store_ready,
                entry.store_ready.empty() ? std::span<const compute::detail::PendingReadTicket>(entry.store->tickets)
                                          : std::span<const compute::detail::PendingReadTicket>{}, {}));
            continue;
          }
          const auto &view = *entry.view;
          if (view.retained_output.valid()) {
            HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::read_unique(uses, view.retained_output));
          } else if (view.writer->valid()) {
            HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::ordered_read_unique(uses, *view.writer));
          } else {
            HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::read_unique(uses, *view.ready));
          }
        }
        return Status::Ok();
      },
      [](void *, compute::Completion &&completion) noexcept { return std::move(completion); },
      bundle,
      [](void *opaque, ValueId value) noexcept {
        const auto &owner = *static_cast<OutputBindingBundle *>(opaque);
        return value.valid() && std::ranges::any_of(owner.entries, [&](const auto &entry) { return entry.input == value; });
      },
      [](void *opaque,std::uint32_t step,mem::ConstBufferView range) noexcept {
        const auto &owner=*static_cast<OutputBindingBundle *>(opaque);
        return std::ranges::any_of(owner.entries,[&](const auto &entry) {
          return entry.effect.node.valid() && entry.effect.step==step &&
              ((!range.valid() && entry.effect_reads.empty()) ||
               std::ranges::any_of(entry.effect_reads,[&](auto exact) {return same_resource_range(exact,range);}));
        });
      }};
  auto result = RegionAccess::submit(invocation, queues, ordered);
  if (result) bundle->submitted = true;
  return result;
}

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
      if(binding.pending_effect) {
        if(!ordered.admit || !ordered.pending_effect ||
            !ordered.pending_effect(ordered.owner,binding.writer_step,binding.pending_range))
          return std::unexpected(Status::FailedPrecondition("pending effect read needs its exact ordered source admission"));
        continue;
      }
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
