#include "constant_artifact.h"

#include <algorithm>
#include <bit>
#include <new>
#include <utility>

#include "hyle/container/hash.h"
#include "plan_state.h"
#include "execution_source_internal.h"

namespace hyle::graph::detail {
namespace {
using Outcome = compute::ProviderPrepareOutcome;

Outcome allocation_failure() noexcept {
  return Outcome::error(
      Status::ResourceExhausted("graph constant artifact allocation failed"));
}
Outcome budget_failure() noexcept {
  return Outcome::resource_declined(Status::ResourceExhausted(
      "graph constant artifact exceeds compilation storage budget"));
}

bool resolve_constant_source(
    const PlanState &plan, std::span<const ConstantBinding> constants,
    std::span<const RegionConstantOrigin> origins, ValueId value,
    RegionConstantOrigin &origin, const ConstantBinding *&physical) noexcept {
  if (!value.valid() || value.ordinal() >= plan.descriptions.size() ||
      GraphIdentity::generation(value) != plan.generation)
    return false;
  origin = {value, value, 0};
  bool explicit_origin = false;
  for (const auto &entry : origins) {
    if (entry.value != value) continue;
    if (explicit_origin) return false;
    explicit_origin = true;
    origin = entry;
  }
  if ((!explicit_origin && value.kind() != ValueKind::kConstant) ||
      (value.kind() != ValueKind::kConstant && value.kind() != ValueKind::kNodeOutput) ||
      !origin.root.valid() || origin.root.kind() != ValueKind::kConstant ||
      (value.kind() == ValueKind::kConstant &&
       (origin.root != value || origin.byte_offset != 0)))
    return false;
  const auto root = std::lower_bound(
      plan.constant_ids.begin(), plan.constant_ids.end(), origin.root.ordinal(),
      [](ValueId constant, std::uint32_t ordinal) {
        return constant.ordinal() < ordinal;
      });
  if (root == plan.constant_ids.end() || *root != origin.root ||
      origin.root.ordinal() >= plan.descriptions.size())
    return false;
  physical = nullptr;
  for (const auto &entry : constants) {
    if (entry.value != value) continue;
    if (physical) return false;
    physical = &entry;
  }
  if (!physical || !physical->tensor.valid() ||
      physical->tensor.description() != plan.descriptions[value.ordinal()])
    return false;
  const auto root_bytes = plan.descriptions[origin.root.ordinal()].required_bytes();
  const auto view_bytes = physical->tensor.description().required_bytes();
  return root_bytes && view_bytes && origin.byte_offset <= *root_bytes &&
      *view_bytes <= *root_bytes - origin.byte_offset;
}

bool queued_preparation_supported(
    const PlanState &plan, const GraphRegionSource &source,
    std::span<const ConstantBinding> constants, ValueId constant,
    mem::MemoryResourceRef destination,
    std::span<const RegionConstantOrigin> origins = {}) noexcept {
  RegionConstantOrigin origin;
  const ConstantBinding *physical = nullptr;
  if (!resolve_constant_source(plan, constants, origins, constant, origin, physical))
    return false;
  const auto storage = std::ranges::find_if(plan.storage_domains,
      [&](const auto &domain) { return domain.resource == destination; });
  if (storage == plan.storage_domains.end() || !storage->owner ||
      !physical->owner ||
      !plan.metadata_owner ||
      (!physical->lease && physical->policy != ConstantPolicy::kMaterialize))
    return false;
  const auto compute = ExecutionSourceAccess::compute(source.source());
  return compute && *compute && (*compute)->is_open() &&
      storage->owner.context_identity() == (*compute)->identity_key();
}

// One cold episode, attached to the original Completion before observation.
// Retaining the unpublished plan keeps the exact canonical allocations or
// borrowed ready leases, native source and allocator owners together. No
// Completion is retained by this owner, so success cannot create a plan cycle.
struct ConstantPreparationState final
    : util::RefCounted<ConstantPreparationState> {
  ConstantPreparationState(PlanState &value,
      util::Ref<ConstantArtifactState> result, StorageCharge temporary) noexcept
      : plan(util::retain_ref(&value)), transient(std::move(temporary)),
        artifact(std::move(result)) {}
  ~ConstantPreparationState() {
    auto &payload = artifact->payload;
    if (auto retire = std::exchange(payload.retire_preparation, nullptr))
      retire(payload.state);
  }
  void destroy() noexcept {
    const auto allocator = plan->allocator;
    const auto retained = plan->metadata_owner;
    this->~ConstantPreparationState();
    allocator.deallocate(this, sizeof(ConstantPreparationState),
                         alignof(ConstantPreparationState));
  }
  util::Ref<PlanState> plan;
  StorageCharge transient;
  util::Ref<ConstantArtifactState> artifact;
  compute::detail::CompletionRetention retention{};
};
} // namespace

std::uint64_t CanonicalConstantKeyHash::operator()(
    const CanonicalConstantKey *key) const noexcept {
  auto hash = container::Hash<std::uint64_t>{}(key->constant);
  hash = container::hash_combine(
      hash, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
                &key->resource.resource())));
  hash = container::hash_combine(hash, key->alignment);
  // One immutable family fixes each logical constant's description. Equality
  // still compares the exact descriptor; no caller source contents are hashed.
  return hash;
}

std::uint64_t ConstantArtifactKeyHash::operator()(
    const ConstantArtifactKey *key) const noexcept {
  auto hash = container::Hash<std::uint64_t>{}(
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key->owner)));
  hash = container::hash_combine(hash,
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key->native_capability)));
  hash = container::hash_combine(hash, key->provider.value());
  for (auto word : {std::uint64_t{key->constant}, std::uint64_t{key->storage},
                    key->packing.format, key->packing.configuration,
                    std::uint64_t{key->root}, std::uint64_t{key->source_offset}})
    hash = container::hash_combine(hash, word);
  // Preserve the existing empty-identity hash. Nonempty identity is exact
  // provider data; neither span addresses nor object padding participate.
  if (!key->packing.identity.empty()) {
    hash = container::hash_combine(hash, key->packing.identity.size());
    for (const auto byte : key->packing.identity)
      hash = container::hash_combine(hash, std::to_integer<std::uint64_t>(byte));
  }
  // Description equality is still checked. In one immutable family a logical
  // constant has a fixed descriptor, so hashing that descriptor again is idle.
  return hash;
}

void ConstantArtifactCache::prune_unused() noexcept {
  if (entries_) {
    for (auto it = entries_->begin(); it != entries_->end();) {
      if (it->value()->is_unique())
        it = entries_->erase(it);
      else
        ++it;
    }
  }
  if (canonical_entries_) {
    for (auto it = canonical_entries_->begin(); it != canonical_entries_->end();) {
      if (it->value()->is_unique())
        it = canonical_entries_->erase(it);
      else
        ++it;
    }
  }
}

Status ConstantArtifactCache::begin_candidate() noexcept {
  prune_unused();
  if (candidate_ == SIZE_MAX)
    return Status::ResourceExhausted(
        "graph packing candidate identity overflows");
  ++candidate_;
  return Status::Ok();
}

void ConstantArtifactCache::commit_plan() noexcept {
  prune_unused();
  if (entries_)
    for (auto &entry : *entries_)
      entry.value()->retained_by_prior_plan = true;
  if (canonical_entries_)
    for (auto &entry : *canonical_entries_)
      if (entry.value()->ready)
        entry.value()->retained_by_prior_plan = true;
}

Status ConstantArtifactCache::remember_canonical(
    PlanState &plan, std::uint32_t physical_instance,
    const util::Ref<CanonicalConstantState> &owner) noexcept {
  auto shared = plan.shared_artifact_storage;
  if (owner->last_candidate != candidate_ && owner->retained_by_prior_plan)
    HYLE_RETURN_IF_ERROR(add_compiled_storage(shared, owner->storage()));
  auto index = plan.instance_constant_owners[physical_instance];
  if (index != kNoWorkspaceRequest) {
    if (index >= plan.canonical_constants.size() ||
        plan.canonical_constants[index].get() != owner.get())
      return Status::Internal("physical constant has a different selected owner");
  } else if (owner->last_candidate == candidate_) {
    index = owner->candidate_owner_index;
    if (index >= plan.canonical_constants.size() ||
        plan.canonical_constants[index].get() != owner.get())
      return Status::Internal("canonical constant candidate owner is invalid");
  } else {
    if (plan.canonical_constants.size() >= kNoWorkspaceRequest ||
        !plan.canonical_constants.try_push_back(owner))
      return Status::ResourceExhausted("graph canonical owner arena allocation failed");
    index = static_cast<std::uint32_t>(plan.canonical_constants.size() - 1);
  }
  plan.instance_constant_owners[physical_instance] = index;
  plan.shared_artifact_storage = shared;
  owner->last_candidate = candidate_;
  owner->candidate_owner_index = index;
  return Status::Ok();
}

Result<CanonicalConstantAcquisition>
ConstantArtifactCache::acquire_canonical(
    PlanState &plan, std::size_t slot, std::uint32_t physical_instance,
    mem::MemoryResourceRef destination,
    std::size_t minimum_alignment) noexcept {
  if (candidate_ == 0)
    return std::unexpected(Status::FailedPrecondition(
        "canonical constants require an active cold candidate"));
  if (slot >= plan.constant_ids.size() ||
      slot >= plan.constant_bindings.size() ||
      physical_instance >= plan.instance_constant_owners.size() ||
      physical_instance >= plan.instance_values.size() ||
      physical_instance >= plan.instance_descriptions.size() ||
      physical_instance >= plan.instance_storage.size() || !destination.valid() ||
      !std::has_single_bit(minimum_alignment))
    return std::unexpected(Status::InvalidArgument(
        "graph canonical constant request is invalid"));
  const auto value = plan.constant_ids[slot];
  const auto binding = plan.constant_bindings[slot];
  if (!value.valid() || value.kind() != ValueKind::kConstant ||
      value.ordinal() >= plan.descriptions.size() || !binding.valid() ||
      binding.description() != plan.descriptions[value.ordinal()])
    return std::unexpected(Status::InvalidArgument(
        "graph canonical constant description is invalid"));
  if (plan.instance_values[physical_instance] != value.ordinal() ||
      plan.instance_descriptions[physical_instance] != binding.description() ||
      plan.instance_storage[physical_instance] >= plan.storage_domains.size() ||
      plan.storage_domains[plan.instance_storage[physical_instance]].resource != destination)
    return std::unexpected(Status::InvalidArgument(
        "graph canonical constant physical binding is invalid"));
  const auto properties = destination.properties();
  if (!std::has_single_bit(properties.minimum_alignment))
    return std::unexpected(Status::Internal(
        "constant memory resource reports an invalid minimum alignment"));
  const auto &description = binding.description();
  const auto alignment = (std::max)({minimum_alignment,
                                     description.required_alignment(),
                                     properties.minimum_alignment});
  const auto selected_index = plan.instance_constant_owners[physical_instance];
  if (selected_index != kNoWorkspaceRequest) {
    if (selected_index >= plan.canonical_constants.size() ||
        !plan.canonical_constants[selected_index])
      return std::unexpected(Status::FailedPrecondition(
          "canonical constant physical owner is invalid"));
    const auto &selected = plan.canonical_constants[selected_index];
    if (!selected->ready)
      return std::unexpected(Status::FailedPrecondition(
          "canonical constant copy has not completed"));
    if (selected->key.constant != value.ordinal() ||
        selected->key.description != description ||
        selected->buffer.resource() != destination ||
        selected->buffer.alignment() < alignment)
      return std::unexpected(Status::FailedPrecondition(
          "canonical constant instance was not reset for a new candidate"));
    HYLE_RETURN_IF_ERROR(remember_canonical(plan, physical_instance, selected));
    return CanonicalConstantAcquisition{selected, false};
  }

  CanonicalConstantKey key{value.ordinal(), description, destination, alignment};
  if (canonical_entries_) {
    // Allocations are power-of-two aligned. Prefer the smallest existing
    // sufficient owner without scanning unrelated constants or mutating any
    // owner already borrowed by a previously selected Plan.
    for (auto lookup = alignment;;) {
      key.alignment = lookup;
      if (const auto *found = canonical_entries_->find(&key)) {
        if (!(*found)->ready)
          return std::unexpected(Status::FailedPrecondition(
              "canonical constant copy has not completed"));
        HYLE_RETURN_IF_ERROR(remember_canonical(plan, physical_instance, *found));
        return CanonicalConstantAcquisition{*found, false};
      }
      if (lookup > SIZE_MAX / 2)
        break;
      lookup *= 2;
    }
  }
  key.alignment = alignment;
  if (!canonical_entries_)
    canonical_entries_.emplace(plan.allocator);
  if (!canonical_entries_->try_prepare_insert() ||
      plan.canonical_constants.size() >= kNoWorkspaceRequest ||
      !plan.canonical_constants.try_reserve(plan.canonical_constants.size() + 1))
    return std::unexpected(Status::ResourceExhausted(
        "graph canonical constant cache allocation failed"));
  void *memory = plan.allocator.allocate(sizeof(CanonicalConstantState),
                                         alignof(CanonicalConstantState));
  if (memory == nullptr)
    return std::unexpected(Status::ResourceExhausted(
        "graph canonical constant owner allocation failed"));
  auto owner = util::adopt_ref(
      ::new (memory) CanonicalConstantState(plan.allocator));
  owner->key = key;
  owner->metadata_owner = plan.metadata_owner;
  owner->storage_owner = plan.storage_domains[
      plan.instance_storage[physical_instance]].owner;
  owner->budget = plan.compilation_storage;
  HYLE_ASSIGN_OR_RETURN(const auto bytes, description.required_bytes());
  if (plan.compilation_storage) {
    HYLE_ASSIGN_OR_RETURN(owner->charge,
        plan.compilation_storage->reserve_resource(bytes, destination));
  }
  HYLE_ASSIGN_OR_RETURN(owner->buffer,
      mem::Buffer::create(destination, bytes, alignment));
  owner->key.alignment = owner->buffer.alignment();
  const auto [entry, inserted] =
      canonical_entries_->try_emplace(&owner->key, owner);
  if (inserted != CanonicalMap::InsertResult::kInserted)
    return std::unexpected(Status::ResourceExhausted(
        "graph canonical constant cache insertion failed"));
  HYLE_RETURN_IF_ERROR(remember_canonical(plan, physical_instance, *entry));
  return CanonicalConstantAcquisition{std::move(owner), true};
}

Outcome ConstantArtifactCache::remember(
    PlanState &plan, const util::Ref<ConstantArtifactState> &artifact,
    ConstantArtifactView *result) noexcept {
  if (artifact->last_candidate != candidate_) {
    auto shared = plan.shared_artifact_storage;
    if (artifact->retained_by_prior_plan) {
      auto status = add_compiled_storage(shared, artifact->storage());
      if (!status.ok())
        return Outcome::error(std::move(status));
    }
    if (!plan.constant_artifacts.try_push_back(artifact))
      return allocation_failure();
    plan.shared_artifact_storage = shared;
    artifact->last_candidate = candidate_;
  }
  *result = {artifact->payload.state, artifact->packed_bytes};
  return Outcome::prepared();
}

Outcome ConstantArtifactCache::acquire(
    PlanState &plan, const GraphRegionSource &source,
    ExecutionTargetId target, std::span<const ConstantBinding> physical_constants,
    ValueId constant,
    ConstantArtifactDeclaration declaration, ConstantArtifactFactory factory,
    void *context, ConstantArtifactView *result,
    CompileOptions options, std::span<const RegionConstantOrigin> origins) noexcept {
  RegionConstantOrigin origin;
  const ConstantBinding *physical = nullptr;
  if (!resolve_constant_source(plan, physical_constants, origins, constant,
                               origin, physical) ||
      !declaration.key.valid() || !declaration.storage.valid() ||
      !declaration.private_storage.valid() || factory == nullptr ||
      !target.valid() || target.ordinal() >= plan.targets.size() ||
      !source.valid() ||
      !source.source().same_owner(plan.targets[target.ordinal()].source))
    return Outcome::error(
        Status::InvalidArgument("graph constant artifact request is invalid"));
  const auto storage =
      std::find_if(plan.storage_domains.begin(), plan.storage_domains.end(),
                   [&](const StorageDomainRecord &value) noexcept {
                     return value.resource == declaration.storage;
                   });
  if (storage == plan.storage_domains.end())
    return Outcome::error(Status::InvalidArgument(
        "graph constant artifact storage is not configured"));
  const auto binding = physical->tensor;
  if (declaration.queue_preparation && !queued_preparation_supported(
          plan, source, physical_constants, constant, declaration.storage, origins))
    return Outcome::error(Status::FailedPrecondition(
        "queued constant preparation requires owned storage and ready source"));
  ConstantArtifactKey key{
      ExecutionSourceAccess::identity_key(source.source()), source.native_interface(),
      source.provider_instance_id(), constant.ordinal(),
      static_cast<std::uint32_t>(storage - plan.storage_domains.begin()),
      declaration.key, binding.description(), origin.root.ordinal(), origin.byte_offset};
  if (entries_) {
    if (auto *found = entries_->find(&key)) {
      const auto &artifact = **found;
      if (artifact.packed_bytes != declaration.packed_bytes ||
          artifact.persistent_bytes != declaration.persistent_bytes ||
          artifact.private_storage.resident_bytes !=
              declaration.private_storage.resident_bytes ||
          artifact.private_storage.peak_bytes !=
              declaration.private_storage.peak_bytes)
        return Outcome::error(Status::InvalidArgument(
            "same constant packing key has inconsistent storage bounds"));
      return remember(plan, *found, result);
    }
  }
  auto compatible = plan.targets[target.ordinal()].source.validate_resource(
      declaration.storage);
  if (!compatible.ok())
    return Outcome::error(std::move(compatible));

  StorageCharge resident, transient;
  const auto bounds = declaration.private_storage;
  if (plan.compilation_storage) {
    const auto limits = *options.compilation_storage;
    const bool unknown = !bounds.resident_bytes || !bounds.peak_bytes;
    if (unknown && limits.unknown == UnknownStoragePolicy::kReject &&
        (limits.max_resident_bytes != SIZE_MAX ||
         limits.max_peak_bytes != SIZE_MAX))
      return budget_failure();
    const auto retained = bounds.resident_bytes.value_or(0);
    const auto peak = (std::max)(retained, bounds.peak_bytes.value_or(0));
    bool budget_declined = false;
    auto charge = plan.compilation_storage->reserve_private(bounds.resident_bytes, &budget_declined);
    if (!charge)
      return budget_declined ? budget_failure() : Outcome::error(std::move(charge).error());
    resident = std::move(*charge);
    charge = plan.compilation_storage->reserve_private(
        bounds.peak_bytes ? std::optional<std::size_t>{peak - retained} : std::nullopt,
        &budget_declined);
    if (!charge)
      return budget_declined ? budget_failure() : Outcome::error(std::move(charge).error());
    transient = std::move(*charge);
    if (unknown)
      plan.compilation_storage->note_unknown();
  }
  if (!entries_)
    entries_.emplace(plan.allocator);
  if (!entries_->try_prepare_insert() ||
      !plan.constant_artifacts.try_reserve(plan.constant_artifacts.size() + 1))
    return allocation_failure();
  void *memory = plan.allocator.allocate(sizeof(ConstantArtifactState),
                                         alignof(ConstantArtifactState));
  if (memory == nullptr)
    return allocation_failure();
  auto artifact =
      util::adopt_ref(::new (memory) ConstantArtifactState(plan.allocator));
  artifact->source = source;
  artifact->metadata_owner = plan.metadata_owner;
  artifact->storage_owner = storage->owner;
  artifact->budget = plan.compilation_storage;
  artifact->charge = std::move(resident);
  artifact->key = key;
  artifact->key.packing.identity = {};
  if (!key.packing.identity.empty()) {
    if (!artifact->packing_identity.try_resize(key.packing.identity.size()))
      return allocation_failure();
    std::ranges::copy(key.packing.identity, artifact->packing_identity.begin());
    artifact->key.packing.identity = {artifact->packing_identity.data(),
                                      artifact->packing_identity.size()};
  }
  artifact->private_storage = bounds;
  artifact->packed_bytes = declaration.packed_bytes;
  artifact->persistent_bytes = declaration.persistent_bytes;
  artifact->target = target;
  // Reserve all retention metadata before a factory can accept native work.
  void *preparation_memory = plan.allocator.allocate(
      sizeof(ConstantPreparationState), alignof(ConstantPreparationState));
  if (!preparation_memory)
    return allocation_failure();
  auto preparation = util::adopt_ref(::new (preparation_memory)
      ConstantPreparationState(plan, artifact, std::move(transient)));
  compute::Queue queue;
  compute::detail::QueueSequence sequence;
  std::size_t preparation_slot = SIZE_MAX;
  if (declaration.queue_preparation) {
    auto endpoint = ExecutionSourceAccess::compute(source.source());
    if (!endpoint) return Outcome::error(std::move(endpoint).error());
    if (!*endpoint)
      return Outcome::error(Status::FailedPrecondition(
          "constant preparation has no Compute endpoint"));
    if (!preparation_queues_)
      preparation_queues_.emplace(plan.metadata_owner, plan.compilation_storage,
                                  plan.allocator);
    auto &queues = preparation_queues_->entries;
    const auto identity = (*endpoint)->identity_key();
    const auto found = std::ranges::find(queues, identity,
                                        &PreparationQueue::endpoint);
    preparation_slot = found == queues.end()
        ? queues.size() : static_cast<std::size_t>(found - queues.begin());
    if (found == queues.end() &&
        !queues.try_emplace_back(PreparationQueue{.endpoint = identity, .queue = {}}))
      return allocation_failure();
    // An in-use or failed queue is never left in the cache. Only the fully
    // observed successful episode below can return this original owner.
    queue = std::move(queues[preparation_slot].queue);
    if (!queue.is_open()) {
      auto created = (*endpoint)->create_queue();
      if (!created) return Outcome::error(std::move(created).error());
      queue = std::move(*created);
    }
    auto admitted = compute::detail::QueueSequence::begin(queue);
    if (!admitted) return Outcome::error(std::move(admitted).error());
    sequence = std::move(*admitted);
  }
  auto status = factory(context,
      {binding, declaration.storage, plan.allocator,
       declaration.queue_preparation ? &sequence : nullptr}, &artifact->payload);
  if (declaration.queue_preparation) {
    if (!status.ok())
      sequence.record(compute::detail::EnqueueResult::rejected(status));
    auto completion = sequence.finish();
    if (completion) {
      preparation->incref();
      preparation->retention.own = {preparation.get(),
          +[](void *, const Status &) noexcept {},
          +[](void *value) noexcept {
            static_cast<ConstantPreparationState *>(value)->decref();
          }};
      // A fresh, valid Completion and inactive embedded retention cannot fail.
      const auto attached = compute::detail::CompletionAccess::attach_retention(
          *completion, preparation->retention);
      HYLE_FATAL_ASSERT(attached.ok());
      const auto terminal = completion->wait();
      if (status.ok()) status = terminal;
    } else if (status.ok()) {
      status = std::move(completion).error();
    }
  }
  // Successful preparation releases its temporary native state before cache
  // publication. On observation failure the backend owns the retained episode.
  preparation = {};
  if (artifact->payload.state != nullptr &&
      artifact->payload.destroy == nullptr)
    return Outcome::error(Status::Internal(
        "constant factory returned an unowned partial payload"));
  if (!status.ok())
    return Outcome::error(std::move(status));
  if (artifact->payload.state == nullptr ||
      artifact->payload.destroy == nullptr)
    return Outcome::error(
        Status::Internal("constant factory returned no owned payload"));
  const auto [entry, inserted] =
      entries_->try_emplace(&artifact->key, artifact);
  if (inserted != Map::InsertResult::kInserted)
    return allocation_failure();
  auto outcome = remember(plan, *entry, result);
  if (outcome.kind == compute::ProviderPrepareOutcomeKind::kPrepared &&
      declaration.queue_preparation && queue.is_open())
    preparation_queues_->entries[preparation_slot].queue = std::move(queue);
  return outcome;
}

} // namespace hyle::graph::detail

namespace hyle::graph {
bool ConstantArtifactAccess::can_prepare_queued(
    ValueId constant, mem::MemoryResourceRef destination) const noexcept {
  return scope_ && !scope_->active_factory &&
      detail::queued_preparation_supported(scope_->plan, scope_->source,
          scope_->constants, constant, destination, scope_->constant_origins);
}
compute::ProviderPrepareOutcome ConstantArtifactAccess::get_or_create(
    ValueId constant, ConstantArtifactDeclaration declaration,
    ConstantArtifactFactory factory, void *context,
    ConstantArtifactView *result) const noexcept {
  if (result != nullptr)
    *result = {};
  if (scope_ == nullptr || result == nullptr || scope_->active_factory)
    return compute::ProviderPrepareOutcome::error(Status::FailedPrecondition(
        "constant artifacts require a non-recursive region compile callback"));
  scope_->active_factory = true;
  auto outcome = scope_->cache.acquire(
      scope_->plan, scope_->source, scope_->target, scope_->constants, constant, declaration,
      factory, context, result, scope_->options, scope_->constant_origins);
  scope_->active_factory = false;
  scope_->budget_declined |=
      outcome.kind ==
      compute::ProviderPrepareOutcomeKind::kDeclinedResourceBudget;
  return outcome;
}

Status ConstantArtifactAccess::bind_prepared_output(
    ValueId output, ConstantArtifactView owner,
    compute::ConstTensorView binding) const noexcept {
  if (scope_ == nullptr || scope_->active_factory)
    return Status::FailedPrecondition("prepared output requires region compile");
  auto &scope = *scope_;
  const auto port = std::ranges::find(scope.allowed_outputs, output);
  if (port == scope.allowed_outputs.end())
    return Status::InvalidArgument("prepared value is not a region output");
  const auto index = static_cast<std::size_t>(port - scope.allowed_outputs.begin());
  if (index >= scope.output_resources.size() ||
      output.ordinal() >= scope.plan.descriptions.size() ||
      !binding.valid() || binding.description() != scope.plan.descriptions[output.ordinal()] ||
      binding.storage().resource() != scope.output_resources[index] ||
      (scope.plan.binding_roles[output.ordinal()] & detail::kExternalOutput) != 0)
    return Status::InvalidArgument("prepared output description/storage differs");
  const auto artifact = std::ranges::find_if(scope.plan.constant_artifacts,
      [&](const auto &entry) {
        return entry->payload.state == owner.state && entry->packed_bytes == owner.packed_bytes &&
               entry->key.owner == detail::ExecutionSourceAccess::identity_key(scope.source.source()) &&
               entry->key.native_capability == scope.source.native_interface() &&
               entry->key.provider == scope.source.provider_instance_id() &&
               entry->target == scope.target &&
               entry->key.storage < scope.plan.storage_domains.size() &&
               scope.plan.storage_domains[entry->key.storage].resource == binding.storage().resource();
      });
  if (artifact == scope.plan.constant_artifacts.end() ||
      binding.description().required_bytes().value() > owner.packed_bytes ||
      std::ranges::any_of(scope.outputs, [&](const auto &entry) { return entry.value == output; }))
    return Status::InvalidArgument("prepared output has no unique acquired owner");
  if (!scope.outputs.try_push_back({output, binding}))
    return Status::ResourceExhausted("prepared output metadata allocation failed");
  return Status::Ok();
}
} // namespace hyle::graph
