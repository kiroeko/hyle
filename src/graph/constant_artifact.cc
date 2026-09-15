#include "constant_artifact.h"

#include <algorithm>
#include <new>
#include <utility>

#include "hyle/container/hash.h"
#include "plan_state.h"

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
} // namespace

std::uint64_t ConstantArtifactKeyHash::operator()(
    const ConstantArtifactKey *key) const noexcept {
  auto hash = container::Hash<std::uint64_t>{}(key->provider.value());
  for (auto word : {std::uint64_t{key->constant}, std::uint64_t{key->storage},
                    key->packing.format, key->packing.configuration})
    hash = container::hash_combine(hash, word);
  // Description equality is still checked. In one immutable family a logical
  // constant has a fixed descriptor, so hashing that descriptor again is idle.
  return hash;
}

void ConstantArtifactCache::prune_unused() noexcept {
  if (!entries_)
    return;
  for (auto it = entries_->begin(); it != entries_->end();) {
    if (it->value()->is_unique())
      it = entries_->erase(it);
    else
      ++it;
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
}

Outcome ConstantArtifactCache::remember(
    PlanState &plan, const util::Ref<ConstantArtifactState> &artifact,
    ConstantArtifactView *result) noexcept {
  if (artifact->last_candidate != candidate_) {
    if (artifact->retained_by_prior_plan &&
        artifact->packed_bytes > SIZE_MAX - plan.shared_packed_bytes)
      return Outcome::error(Status::ResourceExhausted(
          "graph shared packing storage sum overflows"));
    if (!plan.constant_artifacts.try_push_back(artifact))
      return allocation_failure();
    if (artifact->retained_by_prior_plan)
      plan.shared_packed_bytes += artifact->packed_bytes;
    artifact->last_candidate = candidate_;
  }
  *result = {artifact->payload.state, artifact->packed_bytes};
  return Outcome::prepared();
}

Outcome ConstantArtifactCache::acquire(
    PlanState &plan, const compute::detail::ProviderExtensionRef &extension,
    ExecutionTargetId target, ValueId constant,
    ConstantArtifactDeclaration declaration, ConstantArtifactFactory factory,
    void *context, ConstantArtifactView *result,
    CompileOptions options) noexcept {
  const auto input = std::lower_bound(
      plan.constant_ids.begin(), plan.constant_ids.end(), constant.ordinal(),
      [](ValueId value, std::uint32_t ordinal) noexcept {
        return value.ordinal() < ordinal;
      });
  if (input == plan.constant_ids.end() || *input != constant ||
      !declaration.key.valid() || !declaration.storage.valid() ||
      !declaration.private_storage.valid() || factory == nullptr ||
      !target.valid() || target.ordinal() >= plan.targets.size())
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
  const auto slot = static_cast<std::size_t>(input - plan.constant_ids.begin());
  const auto source = plan.constant_bindings[slot];
  ConstantArtifactKey key{
      extension.provider_instance_id(), constant.ordinal(),
      static_cast<std::uint32_t>(storage - plan.storage_domains.begin()),
      declaration.key, source.description()};
  if (entries_) {
    if (auto *found = entries_->find(&key)) {
      const auto &artifact = **found;
      if (artifact.packed_bytes != declaration.packed_bytes ||
          artifact.private_storage.resident_bytes !=
              declaration.private_storage.resident_bytes ||
          artifact.private_storage.peak_bytes !=
              declaration.private_storage.peak_bytes)
        return Outcome::error(Status::InvalidArgument(
            "same constant packing key has inconsistent storage bounds"));
      return remember(plan, *found, result);
    }
  }
  auto compatible = plan.targets[target.ordinal()].context.validate_resource(
      declaration.storage);
  if (!compatible.ok())
    return Outcome::error(std::move(compatible));

  mem::QuotaAllocator::Reservation resident, transient;
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
    auto charge = plan.compilation_storage->reserve(retained);
    if (!charge)
      return budget_failure();
    resident = std::move(*charge);
    charge = plan.compilation_storage->reserve(peak - retained);
    if (!charge)
      return budget_failure();
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
  artifact->extension = extension;
  artifact->budget = plan.compilation_storage;
  artifact->charge = std::move(resident);
  artifact->key = key;
  artifact->private_storage = bounds;
  artifact->packed_bytes = declaration.packed_bytes;
  artifact->target = target;
  auto status = factory(context, source, declaration.storage, plan.allocator,
                        &artifact->payload);
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
  return remember(plan, *entry, result);
}

} // namespace hyle::graph::detail

namespace hyle::graph {
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
      scope_->plan, scope_->extension, scope_->target, constant, declaration,
      factory, context, result, scope_->options);
  scope_->active_factory = false;
  scope_->budget_declined |=
      outcome.kind ==
      compute::ProviderPrepareOutcomeKind::kDeclinedResourceBudget;
  return outcome;
}
} // namespace hyle::graph
