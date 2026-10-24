#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "../../src/graph/constant_artifact.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph/plan_state.h"
#include "../../src/compute/context_state.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/storage_owner.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace d = hyle::graph::detail;
namespace m = hyle::mem;

class TrackedHostResource final : public m::MemoryResource {
public:
  std::size_t allocations = 0;
  std::size_t live_allocations = 0;
  bool fail_allocation = false;
  const c::ResourceBudget *observed_budget = nullptr;
  bool deallocated_with_constant_charge = false;

  std::string_view name() const noexcept override { return "canonical.test.host"; }
  m::MemoryProperties properties() const noexcept override {
    // Both distinct real resources share host accessibility and coherence;
    // canonical allocation identity must still remain exact.
    return m::system_resource().properties();
  }
  hyle::Result<m::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    if (fail_allocation)
      return std::unexpected(hyle::Status::ResourceExhausted(
          "canonical test allocation failure"));
    auto result = m::system_resource().resource().allocate(size, alignment);
    if (result) {
      ++allocations;
      ++live_allocations;
    }
    return result;
  }
  void deallocate(m::ResourceAllocation allocation) noexcept override {
    if (observed_budget) {
      const auto usage = observed_budget->usage(m::MemoryResourceRef(*this));
      deallocated_with_constant_charge =
          usage && usage->live_by_category[static_cast<std::size_t>(
                       c::ResourceBudgetCategory::kConstant)] >= allocation.size;
    }
    --live_allocations;
    m::system_resource().resource().deallocate(allocation);
  }
  hyle::Result<m::ResourceHostMapping>
  map_host(void *handle, std::size_t offset, std::size_t bytes,
           m::HostAccess access) noexcept override {
    return m::system_resource().resource().map_host(handle, offset, bytes, access);
  }
  hyle::Status unmap_host(void *handle, void *token, std::size_t offset,
                         std::size_t bytes, m::HostAccess access) noexcept override {
    return m::system_resource().resource().unmap_host(handle, token, offset,
                                                    bytes, access);
  }
};

struct LogicalConstant final {
  c::TensorDesc description = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{4}).value();
  g::Graph graph;
  g::ValueId value;
  m::Buffer source;
  c::ConstTensorView binding;

  LogicalConstant() {
    auto builder = g::GraphBuilder::create().value();
    value = builder.constant(description).value();
    EXPECT_TRUE(builder.publish(value).ok());
    graph = std::move(builder).finish().value();
    source = m::Buffer::create(m::system_resource(),
        description.required_bytes().value(), 64).value();
    const std::array values{1.0F, -2.0F, 3.0F, -4.0F};
    EXPECT_TRUE(source.write(0, std::as_bytes(std::span(values))).ok());
    binding = c::ConstTensorView::create(description, source.view()).value();
  }
  std::uint64_t generation() const {
    return d::GraphAccess::state(graph)->generation;
  }
};

void initialize(d::PlanState &plan, const LogicalConstant &constant,
                std::span<const m::MemoryResourceRef> resources) {
  ASSERT_TRUE(plan.constant_ids.try_push_back(constant.value));
  ASSERT_TRUE(plan.constant_bindings.try_push_back(constant.binding));
  ASSERT_TRUE(plan.descriptions.try_resize(constant.value.ordinal() + 1));
  ASSERT_TRUE(plan.external_instances.try_resize(constant.value.ordinal() + 1));
  std::fill(plan.external_instances.begin(), plan.external_instances.end(), d::kNoWorkspaceRequest);
  plan.external_instances[constant.value.ordinal()] = 0;
  plan.descriptions[constant.value.ordinal()] = constant.description;
  for (auto resource : resources) {
    const auto storage = static_cast<std::uint32_t>(plan.storage_domains.size());
    ASSERT_TRUE(plan.storage_domains.try_push_back({.resource = resource}));
    ASSERT_TRUE(plan.instance_values.try_push_back(constant.value.ordinal()));
    ASSERT_TRUE(plan.instance_descriptions.try_push_back(constant.description));
    ASSERT_TRUE(plan.instance_storage.try_push_back(storage));
    ASSERT_TRUE(plan.instance_constant_owners.try_push_back(d::kNoWorkspaceRequest));
  }
}

void copy_and_ready(d::CanonicalConstantState &owner,
                    const LogicalConstant &constant) {
  std::array<std::byte, 16> bytes{};
  ASSERT_TRUE(constant.source.view().read(bytes).ok());
  ASSERT_TRUE(owner.buffer.write(0, bytes).ok());
  owner.ready = true;
}

void expect_contents(const m::Buffer &buffer) {
  std::array<float, 4> actual{};
  ASSERT_TRUE(buffer.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array{1.0F, -2.0F, 3.0F, -4.0F}));
}

TEST(CanonicalConstantCache, ReadyOwnersKeepExactResourcesAndDescriptions) {
  TrackedHostResource first, second, foreign;
  const std::array resources{m::MemoryResourceRef(first),
                             m::MemoryResourceRef(second)};
  const LogicalConstant constant;
  d::PlanState original(m::system_allocator(), constant.generation());
  d::PlanState same(m::system_allocator(), constant.generation());
  d::PlanState another(m::system_allocator(), constant.generation());
  initialize(original, constant, resources);
  initialize(same, constant, resources);
  initialize(another, constant, resources);
  another.external_instances[constant.value.ordinal()] = 1;
  d::ConstantArtifactCache cache;
  ASSERT_TRUE(cache.begin_candidate().ok());
  auto invalid = cache.acquire_canonical(original, 0, 0, resources[0], 3);
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  auto unconfigured = cache.acquire_canonical(
      original, 0, 0, m::MemoryResourceRef(foreign), 64);
  ASSERT_FALSE(unconfigured);
  EXPECT_TRUE(unconfigured.error().IsInvalidArgument());
  EXPECT_EQ(first.allocations, 0U);

  auto created = cache.acquire_canonical(original, 0, 0, resources[0], 64);
  ASSERT_TRUE(created);
  ASSERT_TRUE(created->created);
  EXPECT_FALSE(created->owner->ready);
  EXPECT_EQ(original.owned_constant(0), nullptr);
  auto unfinished = cache.acquire_canonical(original, 0, 0, resources[0], 64);
  ASSERT_FALSE(unfinished);
  EXPECT_TRUE(unfinished.error().IsFailedPrecondition());
  EXPECT_EQ(first.allocations, 1U);
  copy_and_ready(*created->owner, constant);
  ASSERT_NE(original.owned_constant(0), nullptr);
  cache.commit_plan();

  ASSERT_TRUE(cache.begin_candidate().ok());
  auto hit = cache.acquire_canonical(same, 0, 0, resources[0], 16);
  ASSERT_TRUE(hit);
  EXPECT_FALSE(hit->created);
  EXPECT_EQ(hit->owner.get(), created->owner.get());
  EXPECT_EQ(same.shared_artifact_storage.materialized_constant_bytes, 16U);
  ASSERT_TRUE(cache.acquire_canonical(same, 0, 0, resources[0], 16));
  EXPECT_EQ(same.shared_artifact_storage.materialized_constant_bytes, 16U);
  EXPECT_EQ(first.allocations, 1U);
  expect_contents(*same.owned_constant(0));
  cache.commit_plan();

  ASSERT_EQ(resources[0].domain(), resources[1].domain());
  ASSERT_TRUE(cache.begin_candidate().ok());
  auto other = cache.acquire_canonical(another, 0, 1, resources[1], 64);
  ASSERT_TRUE(other);
  EXPECT_TRUE(other->created);
  EXPECT_NE(other->owner.get(), created->owner.get());
  EXPECT_EQ(other->owner->buffer.resource(), resources[1]);
  EXPECT_EQ(another.shared_artifact_storage.materialized_constant_bytes, 0U);
  copy_and_ready(*other->owner, constant);
  EXPECT_EQ(second.allocations, 1U);
  expect_contents(*another.owned_constant(0));

  auto different = created->owner->key;
  different.description = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{8}).value();
  EXPECT_NE(created->owner->key, different);
  auto differently_shaped = m::Buffer::create(m::system_resource(), 32, 64).value();
  same.constant_bindings[0] = c::ConstTensorView::create(
      different.description, differently_shaped.view()).value();
  auto mismatched = cache.acquire_canonical(same, 0, 0, resources[0], 16);
  ASSERT_FALSE(mismatched);
  EXPECT_TRUE(mismatched.error().IsInvalidArgument());
}

TEST(CanonicalConstantCache, PhysicalReplicasKeepPrimaryAndShareEachOwnerOnce) {
  TrackedHostResource first, second;
  const std::array resources{m::MemoryResourceRef(first),
                             m::MemoryResourceRef(second)};
  const LogicalConstant constant;
  d::PlanState original(m::system_allocator(), constant.generation());
  d::PlanState shared(m::system_allocator(), constant.generation());
  for (auto *plan : {&original, &shared}) {
    initialize(*plan, constant, resources);
    // A second physical consumer of the first resource must use the same
    // allocation arena entry, independently of the primary's logical slot.
    ASSERT_TRUE(plan->instance_values.try_push_back(constant.value.ordinal()));
    ASSERT_TRUE(plan->instance_descriptions.try_push_back(constant.description));
    ASSERT_TRUE(plan->instance_storage.try_push_back(0));
    ASSERT_TRUE(plan->instance_constant_owners.try_push_back(d::kNoWorkspaceRequest));
  }
  d::ConstantArtifactCache cache;
  ASSERT_TRUE(cache.begin_candidate().ok());
  auto primary = cache.acquire_canonical(original, 0, 0, resources[0], 64);
  ASSERT_TRUE(primary);
  EXPECT_TRUE(primary->created);
  copy_and_ready(*primary->owner, constant);
  const auto *primary_buffer = original.owned_constant(0);
  ASSERT_NE(primary_buffer, nullptr);
  auto replica = cache.acquire_canonical(original, 0, 1, resources[1], 64);
  ASSERT_TRUE(replica);
  EXPECT_TRUE(replica->created);
  EXPECT_EQ(original.owned_instance_constant(1), nullptr);
  auto unready = cache.acquire_canonical(original, 0, 1, resources[1], 64);
  ASSERT_FALSE(unready);
  EXPECT_TRUE(unready.error().IsFailedPrecondition());
  copy_and_ready(*replica->owner, constant);
  auto repeated = cache.acquire_canonical(original, 0, 2, resources[0], 16);
  ASSERT_TRUE(repeated);
  EXPECT_FALSE(repeated->created);
  EXPECT_EQ(repeated->owner.get(), primary->owner.get());
  EXPECT_EQ(original.canonical_constants.size(), 2U);
  EXPECT_EQ(original.instance_constant_owners[0], original.instance_constant_owners[2]);
  EXPECT_NE(original.instance_constant_owners[0], original.instance_constant_owners[1]);
  EXPECT_EQ(original.owned_constant(0), primary_buffer);
  EXPECT_EQ(original.owned_instance_constant(2), primary_buffer);
  EXPECT_EQ(original.owned_instance_constant(1)->resource(), resources[1]);
  EXPECT_EQ(original.shared_artifact_storage.materialized_constant_bytes, 0U);
  expect_contents(*primary_buffer);
  expect_contents(*original.owned_instance_constant(1));
  cache.commit_plan();

  ASSERT_TRUE(cache.begin_candidate().ok());
  for (const auto physical : {0U, 2U, 0U}) {
    auto selected = cache.acquire_canonical(shared, 0, physical, resources[0], 16);
    ASSERT_TRUE(selected);
    EXPECT_FALSE(selected->created);
    EXPECT_EQ(selected->owner.get(), primary->owner.get());
    EXPECT_EQ(shared.canonical_constants.size(), 1U);
    EXPECT_EQ(shared.shared_artifact_storage.materialized_constant_bytes, 16U);
  }
  auto second_owner = cache.acquire_canonical(shared, 0, 1, resources[1], 64);
  ASSERT_TRUE(second_owner);
  EXPECT_FALSE(second_owner->created);
  EXPECT_EQ(second_owner->owner.get(), replica->owner.get());
  ASSERT_TRUE(cache.acquire_canonical(shared, 0, 1, resources[1], 64));
  EXPECT_EQ(shared.canonical_constants.size(), 2U);
  EXPECT_EQ(shared.shared_artifact_storage.materialized_constant_bytes, 32U);
  EXPECT_EQ(shared.owned_constant(0), primary_buffer);
  EXPECT_EQ(first.allocations, 1U);
  EXPECT_EQ(second.allocations, 1U);
}

TEST(CanonicalConstantCache, RejectedAlignmentOwnerReleasesOnlyItsOwnBytes) {
  TrackedHostResource resource;
  const std::array resources{m::MemoryResourceRef(resource)};
  const LogicalConstant constant;
  d::PlanState original(m::system_allocator(), constant.generation());
  d::PlanState rejected(m::system_allocator(), constant.generation());
  d::PlanState stronger(m::system_allocator(), constant.generation());
  d::PlanState weaker(m::system_allocator(), constant.generation());
  initialize(original, constant, resources);
  initialize(rejected, constant, resources);
  initialize(stronger, constant, resources);
  initialize(weaker, constant, resources);
  {
    d::ConstantArtifactCache cache;
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto first = cache.acquire_canonical(original, 0, 0, resources[0], 64);
    ASSERT_TRUE(first);
    copy_and_ready(*first->owner, constant);
    cache.commit_plan();

    ASSERT_TRUE(cache.begin_candidate().ok());
    auto unused = cache.acquire_canonical(rejected, 0, 0, resources[0], 256);
    ASSERT_TRUE(unused);
    EXPECT_TRUE(unused->created);
    EXPECT_GE(unused->owner->buffer.alignment(), 256U);
    EXPECT_NE(unused->owner.get(), first->owner.get());
    copy_and_ready(*unused->owner, constant);
    rejected.canonical_constants[0].reset();
    unused->owner.reset();
    ASSERT_TRUE(cache.begin_candidate().ok());
    EXPECT_EQ(resource.live_allocations, 1U);
    expect_contents(*original.owned_constant(0));

    auto replacement = cache.acquire_canonical(stronger, 0, 0, resources[0], 256);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(replacement->created);
    copy_and_ready(*replacement->owner, constant);
    cache.commit_plan();
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto sufficient = cache.acquire_canonical(weaker, 0, 0, resources[0], 128);
    ASSERT_TRUE(sufficient);
    EXPECT_FALSE(sufficient->created);
    EXPECT_EQ(sufficient->owner.get(), replacement->owner.get());
    EXPECT_EQ(weaker.shared_artifact_storage.materialized_constant_bytes, 16U);
    EXPECT_EQ(resource.allocations, 3U);
  }
  EXPECT_EQ(resource.live_allocations, 2U);
  original.canonical_constants[0].reset();
  EXPECT_EQ(resource.live_allocations, 1U);
  expect_contents(*stronger.owned_constant(0));
  expect_contents(*weaker.owned_constant(0));
}

TEST(CanonicalConstantCache, ChargesAreSharedAndFailedAllocationsRollBack) {
  TrackedHostResource first, second;
  const std::array resources{m::MemoryResourceRef(first),
                             m::MemoryResourceRef(second)};
  const std::array limits{
      c::ResourceBudgetLimit{m::system_resource(), SIZE_MAX},
      c::ResourceBudgetLimit{resources[0], 16},
      c::ResourceBudgetLimit{resources[1], 16}};
  auto resource_budget = c::ResourceBudget::create({.limits = limits}).value();
  first.observed_budget = &resource_budget;
  second.observed_budget = &resource_budget;
  const auto backing = resource_budget.metadata_allocator().value();
  auto budget = hyle::util::adopt_ref(d::StorageBudgetState::create(
      backing, {}, {}, false, resource_budget).value());
  const LogicalConstant constant;
  {
    d::PlanState original(budget->allocator(), constant.generation());
    d::PlanState same(budget->allocator(), constant.generation());
    d::PlanState alignment_failure(budget->allocator(), constant.generation());
    d::PlanState allocation_failure(budget->allocator(), constant.generation());
    for (auto *plan : {&original, &same, &alignment_failure, &allocation_failure}) {
      plan->compilation_storage = budget;
      initialize(*plan, constant, resources);
    }
    d::ConstantArtifactCache cache;
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto created = cache.acquire_canonical(original, 0, 0, resources[0], 64);
    ASSERT_TRUE(created);
    copy_and_ready(*created->owner, constant);
    cache.commit_plan();
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto reused = cache.acquire_canonical(same, 0, 0, resources[0], 64);
    ASSERT_TRUE(reused);
    EXPECT_FALSE(reused->created);
    EXPECT_EQ(resource_budget.usage(resources[0])->live_by_category[
        static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], 16U);

    ASSERT_TRUE(cache.begin_candidate().ok());
    auto quota = cache.acquire_canonical(alignment_failure, 0, 0, resources[0], 128);
    ASSERT_FALSE(quota);
    EXPECT_TRUE(quota.error().IsResourceExhausted());
    EXPECT_EQ(first.allocations, 1U);
    EXPECT_TRUE(alignment_failure.canonical_constants.empty());
    EXPECT_EQ(alignment_failure.owned_instance_constant(0), nullptr);
    ASSERT_TRUE(cache.begin_candidate().ok());
    second.fail_allocation = true;
    auto failed = cache.acquire_canonical(allocation_failure, 0, 1, resources[1], 64);
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_TRUE(allocation_failure.canonical_constants.empty());
    EXPECT_EQ(allocation_failure.owned_instance_constant(1), nullptr);
    EXPECT_EQ(resource_budget.usage(resources[1])->live_by_category[
        static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], 0U);
    EXPECT_EQ(second.allocations, 0U);
  }
  EXPECT_EQ(first.live_allocations, 0U);
  EXPECT_TRUE(first.deallocated_with_constant_charge);
  EXPECT_EQ(resource_budget.usage(resources[0])->live_by_category[
      static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], 0U);
  EXPECT_EQ(resource_budget.usage(resources[1])->live_by_category[
      static_cast<std::size_t>(c::ResourceBudgetCategory::kConstant)], 0U);
}

struct ArtifactCounts final {
  std::size_t factories = 0;
  std::size_t destroys = 0;
};
struct ArtifactPayload final {
  ArtifactCounts *counts;
  float first = 0;
};
hyle::Status artifact_factory(void *context,
    const g::ConstantArtifactPrepareRequest &request,
    g::ConstantArtifactPayload *output) noexcept {
  auto &observed = *static_cast<ArtifactCounts *>(context);
  ++observed.factories;
  auto *payload = request.allocator.construct<ArtifactPayload>(&observed);
  if (!payload)
    return hyle::Status::ResourceExhausted("artifact-key test payload");
  *output = {payload, +[](void *value, m::AllocatorRef allocator) noexcept {
    auto *payload = static_cast<ArtifactPayload *>(value);
    ++payload->counts->destroys;
    allocator.destroy(payload);
  }};
  HYLE_ASSIGN_OR_RETURN(const auto first,
      request.source.storage().subview(0, sizeof(payload->first)));
  return first.read(
      std::as_writable_bytes(std::span(&payload->first, 1)));
}
// Synchronous cache qualification uses this actual retained source. No
// compute work, queue or provider program is needed to exercise the cache.
const d::ExecutionSourceVTable kArtifactSourceTable{
    .destroy = +[](d::ExecutionSourceState *state) noexcept {
      state->allocator.destroy(state);
    },
    .is_open = +[](const d::ExecutionSourceState &) noexcept { return true; },
    .validate_resource = +[](const d::ExecutionSourceState &,
        m::MemoryResourceRef) noexcept { return hyle::Status::Ok(); },
    .native_context = +[](const d::ExecutionSourceState &state) noexcept -> void * {
      return const_cast<d::ExecutionSourceState *>(&state);
    }};

TEST(ConstantArtifactCache, OpaqueKeyOwnsTemporaryBytesAndComparesExactContents) {
  ArtifactCounts counts;
  using Payload = ArtifactPayload;
  const auto factory = &artifact_factory;
  const auto &source_table = kArtifactSourceTable;
  {
    TrackedHostResource resource;
    const std::array resources{m::MemoryResourceRef(resource)};
    const LogicalConstant constant;
    const auto allocator = m::system_allocator();
    auto *source_state = allocator.construct<d::ExecutionSourceState>(
        &source_table, allocator, &counts);
    ASSERT_NE(source_state, nullptr);
    auto endpoint = d::ExecutionSourceAccess::adopt(source_state);
    auto source = d::GraphRegionSource::native(endpoint, &source_table).value();
    d::PlanState plan(allocator, constant.generation());
    initialize(plan, constant, resources);
    ASSERT_TRUE(plan.targets.try_push_back({.source = endpoint}));
    const std::array physical{g::ConstantBinding{
        .value = constant.value, .tensor = constant.binding}};
    d::ConstantArtifactCache cache;
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto last_status = hyle::Status::Ok();
    const auto acquire = [&](std::span<const std::byte> identity,
                             g::ConstantArtifactView *result) {
      auto outcome = cache.acquire(plan, source, g::ExecutionTargetId{0}, physical,
          constant.value,
          {.key = {0x544553544f504151ULL, 1, identity},
           .storage = resources[0], .packed_bytes = 0,
           .persistent_bytes = sizeof(Payload),
           .private_storage = {.resident_bytes = 0, .peak_bytes = 0}},
          factory, &counts, result, {});
      last_status = outcome.status;
      return outcome;
    };
    const auto prepared = c::ProviderPrepareOutcomeKind::kPrepared;
    std::array<std::byte, 80> alpha;
    alpha.fill(std::byte{0x31});
    auto beta = alpha;
    beta.back() = std::byte{0x32}; // Same length, different contents.
    g::ConstantArtifactView first, same, other, empty;
    {
      auto temporary = alpha;
      ASSERT_EQ(acquire(temporary, &first).kind, prepared) << last_status.message();
      ASSERT_EQ(plan.constant_artifacts.size(), 1U);
      EXPECT_NE(plan.constant_artifacts[0]->key.packing.identity.data(),
                temporary.data());
      temporary.fill(std::byte{0xff});
    }
    EXPECT_TRUE(std::ranges::equal(
        plan.constant_artifacts[0]->key.packing.identity, alpha));
    EXPECT_EQ(counts.factories, 1U);
    {
      auto temporary = alpha;
      ASSERT_EQ(acquire(temporary, &same).kind, prepared) << last_status.message();
      EXPECT_EQ(same.state, first.state);
      EXPECT_EQ(counts.factories, 1U);
      temporary = beta;
      ASSERT_EQ(acquire(temporary, &other).kind, prepared) << last_status.message();
      EXPECT_NE(other.state, first.state);
      EXPECT_EQ(counts.factories, 2U);
      temporary.fill(std::byte{0xff});
    }
    ASSERT_EQ(acquire(alpha, &same).kind, prepared) << last_status.message();
    EXPECT_EQ(same.state, first.state);
    ASSERT_EQ(acquire(beta, &same).kind, prepared) << last_status.message();
    EXPECT_EQ(same.state, other.state);
    EXPECT_EQ(counts.factories, 2U);
    // Empty identity preserves the original two-word key, independently of
    // whether the borrowed empty span's data pointer is null or nonnull.
    ASSERT_EQ(acquire({}, &empty).kind, prepared) << last_status.message();
    ASSERT_EQ(acquire({alpha.data(), 0}, &same).kind, prepared) << last_status.message();
    EXPECT_EQ(same.state, empty.state);
    EXPECT_EQ(counts.factories, 3U);
    EXPECT_EQ(plan.constant_artifacts.size(), 3U);
    EXPECT_EQ(plan.constant_artifacts[0]->storage().provider_persistent_bytes,
              sizeof(Payload)); // Core key bytes use the metadata allocator.
    EXPECT_EQ(counts.destroys, 0U);
  }
  EXPECT_EQ(counts.destroys, 3U);
}

TEST(ConstantArtifactCache, AliasOriginsSeparateEqualDescriptionsByRootAndOffset) {
  ArtifactCounts counts;
  {
    TrackedHostResource resource;
    const auto allocator = m::system_allocator();
    const auto root_desc = c::TensorDesc::dense(
        c::DataType::kFloat32, std::array<std::int64_t, 1>{8}).value();
    const auto view_desc = c::TensorDesc::dense(
        c::DataType::kFloat32, std::array<std::int64_t, 1>{4}).value();
    auto builder = g::GraphBuilder::create().value();
    const auto first_root = builder.constant(root_desc).value();
    const auto second_root = builder.constant(root_desc).value();
    const auto boundary = builder.add(
        c::View::slice(0, 0, 4).value(), first_root).value();
    ASSERT_TRUE(builder.publish(boundary).ok());
    const auto graph = std::move(builder).finish().value();
    const auto generation = d::GraphAccess::state(graph)->generation;
    const auto storage = m::MemoryResourceRef(resource);
    auto first_bytes = m::Buffer::create(storage, 32, 64).value();
    auto second_bytes = m::Buffer::create(storage, 32, 64).value();
    const std::array first_values{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    const std::array second_values{11.F, 12.F, 13.F, 14.F, 15.F, 16.F, 17.F, 18.F};
    ASSERT_TRUE(first_bytes.write(0, std::as_bytes(std::span(first_values))).ok());
    ASSERT_TRUE(second_bytes.write(0, std::as_bytes(std::span(second_values))).ok());
    const auto first_binding = c::ConstTensorView::create(root_desc, first_bytes.view()).value();
    const auto second_binding = c::ConstTensorView::create(root_desc, second_bytes.view()).value();
    const auto view = [&](const m::Buffer &buffer, std::size_t offset) {
      return c::ConstTensorView::create(
          view_desc, buffer.view().subview(offset, 16).value()).value();
    };
    auto *source_state = allocator.construct<d::ExecutionSourceState>(
        &kArtifactSourceTable, allocator, &counts);
    ASSERT_NE(source_state, nullptr);
    auto endpoint = d::ExecutionSourceAccess::adopt(source_state);
    auto source = d::GraphRegionSource::native(endpoint, &kArtifactSourceTable).value();
    d::PlanState plan(allocator, generation);
    ASSERT_TRUE(plan.constant_ids.try_push_back(first_root));
    ASSERT_TRUE(plan.constant_ids.try_push_back(second_root));
    ASSERT_TRUE(plan.constant_bindings.try_push_back(first_binding));
    ASSERT_TRUE(plan.constant_bindings.try_push_back(second_binding));
    ASSERT_TRUE(plan.descriptions.try_resize(boundary.ordinal() + 1));
    plan.descriptions[first_root.ordinal()] = root_desc;
    plan.descriptions[second_root.ordinal()] = root_desc;
    plan.descriptions[boundary.ordinal()] = view_desc;
    ASSERT_TRUE(plan.storage_domains.try_push_back({.resource = storage}));
    ASSERT_TRUE(plan.targets.try_push_back({.source = endpoint}));
    d::ConstantArtifactCache cache;
    ASSERT_TRUE(cache.begin_candidate().ok());
    auto last_status = hyle::Status::Ok();
    const auto acquire = [&](c::ConstTensorView binding,
        std::span<const g::RegionConstantOrigin> origins,
        g::ConstantArtifactView *result) {
      const std::array physical{g::ConstantBinding{
          .value = boundary, .tensor = binding}};
      d::ConstantArtifactScope scope{cache, plan, source, g::ExecutionTargetId{0},
          {}, physical, {}, {}, d::PreparedConstantOutputs(allocator)};
      scope.constant_origins = origins;
      auto outcome = scope.access().get_or_create(boundary,
          {.key = {0x54455354414c4941ULL, 1}, .storage = storage,
           .packed_bytes = 0, .persistent_bytes = sizeof(ArtifactPayload),
           .private_storage = {.resident_bytes = 0, .peak_bytes = 0}},
          &artifact_factory, &counts, result);
      last_status = outcome.status;
      return outcome;
    };
    const auto prepared = c::ProviderPrepareOutcomeKind::kPrepared;
    const auto error = c::ProviderPrepareOutcomeKind::kError;
    const std::array first_origin{g::RegionConstantOrigin{boundary, first_root, 0}};
    const std::array offset_origin{g::RegionConstantOrigin{boundary, first_root, 16}};
    const std::array root_origin{g::RegionConstantOrigin{boundary, second_root, 0}};
    g::ConstantArtifactView first, offset, other_root, same;
    // These exact physical facts deliberately keep the same boundary ID and
    // descriptor, isolating root/offset identity across cold specializations.
    ASSERT_EQ(acquire(view(first_bytes, 0), first_origin, &first).kind, prepared)
        << last_status.message();
    ASSERT_EQ(acquire(view(first_bytes, 16), offset_origin, &offset).kind, prepared)
        << last_status.message();
    ASSERT_EQ(acquire(view(second_bytes, 0), root_origin, &other_root).kind, prepared)
        << last_status.message();
    EXPECT_NE(first.state, offset.state);
    EXPECT_NE(first.state, other_root.state);
    EXPECT_NE(offset.state, other_root.state);
    EXPECT_EQ(static_cast<const ArtifactPayload *>(first.state)->first, 1.F);
    EXPECT_EQ(static_cast<const ArtifactPayload *>(offset.state)->first, 5.F);
    EXPECT_EQ(static_cast<const ArtifactPayload *>(other_root.state)->first, 11.F);
    EXPECT_EQ(counts.factories, 3U);
    ASSERT_EQ(acquire(view(first_bytes, 0), first_origin, &same).kind, prepared)
        << last_status.message();
    EXPECT_EQ(same.state, first.state);
    ASSERT_EQ(acquire(view(first_bytes, 16), offset_origin, &same).kind, prepared)
        << last_status.message();
    EXPECT_EQ(same.state, offset.state);
    ASSERT_EQ(acquire(view(second_bytes, 0), root_origin, &same).kind, prepared)
        << last_status.message();
    EXPECT_EQ(same.state, other_root.state);
    EXPECT_EQ(counts.factories, 3U);
    EXPECT_EQ(plan.constant_artifacts[0]->key.root, first_root.ordinal());
    EXPECT_EQ(plan.constant_artifacts[1]->key.source_offset, 16U);
    EXPECT_EQ(plan.constant_artifacts[2]->key.root, second_root.ordinal());
    // Invalid provenance must not hit any existing alias entry or call a
    // factory, even when value, dtype, shape and provider key are identical.
    EXPECT_EQ(acquire(view(first_bytes, 0), {}, &same).kind, error);
    EXPECT_EQ(same.state, nullptr);
    const std::array not_root{g::RegionConstantOrigin{boundary, boundary, 0}};
    EXPECT_EQ(acquire(view(first_bytes, 0), not_root, &same).kind, error);
    const std::array overflow{g::RegionConstantOrigin{boundary, first_root, SIZE_MAX}};
    EXPECT_EQ(acquire(view(first_bytes, 0), overflow, &same).kind, error);
    const std::array duplicate{first_origin[0], first_origin[0]};
    EXPECT_EQ(acquire(view(first_bytes, 0), duplicate, &same).kind, error);
    EXPECT_EQ(acquire(first_binding, first_origin, &same).kind, error);
    EXPECT_EQ(same.state, nullptr);
    EXPECT_EQ(counts.factories, 3U);
    EXPECT_EQ(plan.constant_artifacts.size(), 3U);
  }
  EXPECT_EQ(counts.destroys, 3U);
}

struct StorageOwnerLifetime final {
  std::size_t resource_destroys = 0;
  std::size_t allocations = 0;
  std::size_t live_metadata = 0;
  std::size_t metadata_frees = 0;
  std::size_t allocator_destroys = 0;
  std::size_t live_metadata_at_resource_destroy = 0;
  std::size_t live_metadata_at_allocator_destroy = SIZE_MAX;
};

struct OwnedTrackedResource final {
  explicit OwnedTrackedResource(StorageOwnerLifetime &selected)
      : lifetime(&selected), resource(std::make_unique<TrackedHostResource>()) {}
  OwnedTrackedResource(const OwnedTrackedResource &) = delete;
  OwnedTrackedResource &operator=(const OwnedTrackedResource &) = delete;
  OwnedTrackedResource(OwnedTrackedResource &&) noexcept = default;
  OwnedTrackedResource &operator=(OwnedTrackedResource &&) noexcept = default;
  ~OwnedTrackedResource() {
    if (resource) {
      EXPECT_EQ(resource->live_allocations, 0U);
      ++lifetime->resource_destroys;
      lifetime->live_metadata_at_resource_destroy = lifetime->live_metadata;
    }
  }
  m::MemoryResourceRef ref() const noexcept {
    return resource ? m::MemoryResourceRef(*resource) : m::MemoryResourceRef{};
  }
  StorageOwnerLifetime *lifetime;
  std::unique_ptr<TrackedHostResource> resource;
};

struct OwnedMetadataAllocator final {
  explicit OwnedMetadataAllocator(StorageOwnerLifetime &selected) noexcept
      : lifetime(selected) {}
  ~OwnedMetadataAllocator() {
    ++lifetime.allocator_destroys;
    lifetime.live_metadata_at_allocator_destroy = lifetime.live_metadata;
  }
  m::AllocatorRef ref() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
  }
  static void *allocate(void *opaque, std::size_t bytes,
                        std::size_t alignment) noexcept {
    auto &owner = *static_cast<OwnedMetadataAllocator *>(opaque);
    if (owner.fail_allocation) return nullptr;
    auto *result = m::system_allocator().allocate(bytes, alignment);
    if (result) {
      ++owner.lifetime.allocations;
      ++owner.lifetime.live_metadata;
    }
    return result;
  }
  static void deallocate(void *opaque, void *value, std::size_t bytes,
                          std::size_t alignment) noexcept {
    if (!value) return;
    auto &owner = *static_cast<OwnedMetadataAllocator *>(opaque);
    m::system_allocator().deallocate(value, bytes, alignment);
    --owner.lifetime.live_metadata;
    ++owner.lifetime.metadata_frees;
  }
  static void *reallocate(void *opaque, void *value, std::size_t old_bytes,
                          std::size_t bytes, std::size_t alignment) noexcept {
    if (!value) return allocate(opaque, bytes, alignment);
    auto &owner = *static_cast<OwnedMetadataAllocator *>(opaque);
    if (bytes != 0 && owner.fail_allocation) return nullptr;
    auto *result = m::system_allocator().reallocate(value, old_bytes, bytes, alignment);
    if (bytes == 0) {
      --owner.lifetime.live_metadata;
      ++owner.lifetime.metadata_frees;
    }
    return result;
  }
  StorageOwnerLifetime &lifetime;
  bool fail_allocation = false;
};

TEST(StorageOwner, ReadyLeaseRetainsMovedResourceAfterOriginalHandlesDie) {
  StorageOwnerLifetime lifetime;
  OwnedTrackedResource original(lifetime);
  auto *tracked = original.resource.get();
  auto context = c::Context::create(c::cpu::device()).value();
  auto acquired = c::StorageOwner::adopt(context, std::move(original));
  ASSERT_TRUE(acquired) << acquired.error().message();
  auto owner = std::move(*acquired);
  EXPECT_FALSE(original.resource);
  EXPECT_EQ(owner.ref(), m::MemoryResourceRef(*tracked));
  EXPECT_NE(owner.context_identity(), nullptr);
  EXPECT_TRUE(owner.owns_allocator(m::system_allocator()));
  auto buffer = m::Buffer::create(owner.ref(), 16, 64).value();
  const std::array<float, 4> values{1.0F, -2.0F, 3.0F, -4.0F};
  ASSERT_TRUE(buffer.write(0, std::as_bytes(std::span(values))).ok());
  auto adopted = owner.adopt_buffer(std::move(buffer));
  ASSERT_TRUE(adopted) << adopted.error().message();
  auto ready = std::move(*adopted);
  EXPECT_FALSE(buffer.valid());
  auto copied = owner;
  owner = {};
  context = {};
  copied = {};
  EXPECT_EQ(lifetime.resource_destroys, 0U);
  EXPECT_EQ(tracked->live_allocations, 1U);
  std::array<float, 4> observed{};
  ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, values);
  ready.reset();
  EXPECT_EQ(lifetime.resource_destroys, 1U);
}

TEST(StorageOwner, CustomAllocatorOutlivesResourceContextAndLastLeaseMetadataFree) {
  StorageOwnerLifetime lifetime;
  auto metadata = std::make_shared<OwnedMetadataAllocator>(lifetime);
  std::weak_ptr<OwnedMetadataAllocator> allocator_observer = metadata;
  auto context = c::Context::create(c::cpu::device(),
      {.metadata_allocator = metadata->ref()}).value();
  OwnedTrackedResource original(lifetime);
  auto *tracked = original.resource.get();
  auto unowned_allocator = c::StorageOwner::adopt(context, std::move(original));
  ASSERT_FALSE(unowned_allocator);
  EXPECT_TRUE(original.resource);
  auto acquired = c::StorageOwner::adopt(context, std::move(original), metadata);
  ASSERT_TRUE(acquired) << acquired.error().message();
  auto owner = std::move(*acquired);
  EXPECT_TRUE(owner.owns_allocator(metadata->ref()));
  auto buffer = m::Buffer::create(owner.ref(), 16, 64).value();
  metadata->fail_allocation = true;
  auto rejected = owner.adopt_buffer(std::move(buffer));
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_TRUE(buffer.valid());
  EXPECT_EQ(tracked->live_allocations, 1U);
  metadata->fail_allocation = false;
  auto adopted = owner.adopt_buffer(std::move(buffer));
  ASSERT_TRUE(adopted) << adopted.error().message();
  auto ready = std::move(*adopted);
  context = {};
  owner = {};
  metadata.reset();
  EXPECT_FALSE(allocator_observer.expired());
  EXPECT_EQ(lifetime.allocator_destroys, 0U);
  EXPECT_EQ(lifetime.resource_destroys, 0U);
  EXPECT_GT(lifetime.live_metadata, 0U);
  EXPECT_TRUE(ready.view().has_value());
  ready.reset();
  EXPECT_EQ(lifetime.resource_destroys, 1U);
  EXPECT_GT(lifetime.live_metadata_at_resource_destroy, 0U);
  EXPECT_EQ(lifetime.allocator_destroys, 1U);
  EXPECT_EQ(lifetime.live_metadata_at_allocator_destroy, 0U);
  EXPECT_EQ(lifetime.live_metadata, 0U);
  EXPECT_GT(lifetime.metadata_frees, 0U);
  EXPECT_TRUE(allocator_observer.expired());
}

TEST(StorageOwner, ExactAllocatorIdentityAndForeignBufferRejectBeforeTransfer) {
  StorageOwnerLifetime lifetime;
  auto metadata = std::make_shared<OwnedMetadataAllocator>(lifetime);
  auto context = c::Context::create(c::cpu::device(),
      {.metadata_allocator = metadata->ref()}).value();
  const auto selected = metadata->ref();
  int foreign_state = 0;
  const auto other_alloc = +[](void *, std::size_t, std::size_t) noexcept -> void * {
    return nullptr;
  };
  const auto other_free = +[](void *, void *, std::size_t, std::size_t) noexcept {};
  const auto other_realloc = +[](void *, void *, std::size_t, std::size_t,
                                 std::size_t) noexcept -> void * { return nullptr; };
  const std::array different{
      m::AllocatorRef(other_alloc, selected.free_fn(), selected.realloc_fn(), selected.state()),
      m::AllocatorRef(selected.alloc_fn(), other_free, selected.realloc_fn(), selected.state()),
      m::AllocatorRef(selected.alloc_fn(), selected.free_fn(), other_realloc, selected.state()),
      m::AllocatorRef(selected.alloc_fn(), selected.free_fn(), selected.realloc_fn(), &foreign_state)};
  struct AllocatorIdentity final {
    m::AllocatorRef value;
    m::AllocatorRef ref() const noexcept { return value; }
  };
  for (auto identity : different) {
    StorageOwnerLifetime resource_lifetime;
    OwnedTrackedResource original(resource_lifetime);
    auto rejected = c::StorageOwner::adopt(context, std::move(original),
        std::make_shared<AllocatorIdentity>(identity));
    EXPECT_FALSE(rejected);
    EXPECT_TRUE(original.resource);
  }
  OwnedTrackedResource actual(lifetime);
  auto acquired = c::StorageOwner::adopt(context, std::move(actual), metadata);
  ASSERT_TRUE(acquired) << acquired.error().message();
  auto owner = std::move(*acquired);
  TrackedHostResource foreign;
  auto buffer = m::Buffer::create(m::MemoryResourceRef(foreign), 16, 64).value();
  auto rejected = owner.adopt_buffer(std::move(buffer));
  EXPECT_FALSE(rejected);
  EXPECT_TRUE(buffer.valid());
  EXPECT_EQ(foreign.live_allocations, 1U);
}
TEST(StorageOwner, BudgetBackingRequiresActualAllocatorOwnerBeforeBufferTransfer) {
  StorageOwnerLifetime lifetime, foreign_lifetime;
  auto metadata = std::make_shared<OwnedMetadataAllocator>(lifetime);
  auto foreign_metadata = std::make_shared<OwnedMetadataAllocator>(foreign_lifetime);
  auto context = c::Context::create(c::cpu::device(),
      {.metadata_allocator = metadata->ref()}).value();
  OwnedTrackedResource resource(lifetime);
  auto owner = c::StorageOwner::adopt(context, std::move(resource), metadata).value();
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), SIZE_MAX},
                          c::ResourceBudgetLimit{owner.ref(), SIZE_MAX}};
  auto foreign_budget = c::ResourceBudget::create(
      {.limits = limits, .allocator = foreign_metadata->ref()}).value();
  auto buffer = m::Buffer::create(owner.ref(), 16, 64).value();
  auto rejected = owner.adopt_buffer(std::move(buffer), foreign_budget);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_TRUE(buffer.valid());
  auto budget = c::ResourceBudget::create(
      {.limits = limits, .allocator = metadata->ref()}).value();
  auto ready = owner.adopt_buffer(std::move(buffer), budget).value();
  EXPECT_FALSE(buffer.valid());
  context = {};
  owner = {};
  metadata.reset();
  budget = {};
  EXPECT_EQ(lifetime.allocator_destroys, 0U);
  ready.reset();
  EXPECT_EQ(lifetime.resource_destroys, 1U);
  EXPECT_EQ(lifetime.allocator_destroys, 1U);
  EXPECT_EQ(lifetime.live_metadata_at_allocator_destroy, 0U);
  EXPECT_EQ(lifetime.live_metadata, 0U);
}

struct QueuedArtifactSource final : d::ExecutionSourceState {
  static const d::ExecutionSourceVTable table;
  QueuedArtifactSource(c::detail::ContextRef retained,
                       m::AllocatorRef owner_allocator) noexcept
      : d::ExecutionSourceState(&table, owner_allocator, retained.identity_key()),
        context(std::move(retained)) {}
  c::detail::ContextRef context;
};

const d::ExecutionSourceVTable QueuedArtifactSource::table{
    .destroy = +[](d::ExecutionSourceState *state) noexcept {
      const auto owner_allocator = state->allocator;
      owner_allocator.destroy(static_cast<QueuedArtifactSource *>(state));
    },
    .is_open = +[](const d::ExecutionSourceState &state) noexcept {
      return static_cast<const QueuedArtifactSource &>(state).context.is_open();
    },
    .validate_resource = +[](const d::ExecutionSourceState &state,
        m::MemoryResourceRef resource) noexcept {
      return static_cast<const QueuedArtifactSource &>(state).context.validate_resource(resource);
    },
    .call = +[](const d::ExecutionSourceState &state,
        d::SourceCall call, void *opaque) noexcept {
      return static_cast<const QueuedArtifactSource &>(state).context.call(call, opaque);
    },
    .compute = +[](const d::ExecutionSourceState &state) noexcept
        -> const c::detail::ContextRef * {
      return &static_cast<const QueuedArtifactSource &>(state).context;
    },
    .native_context = +[](const d::ExecutionSourceState &state) noexcept -> void * {
      return const_cast<QueuedArtifactSource *>(
          &static_cast<const QueuedArtifactSource &>(state));
    }};

struct QueuedConstantOwner final {
  StorageOwnerLifetime lifetime;
  c::Context context;
  c::StorageOwner owner;
  c::ReadLease ready;
  c::ConstTensorView binding;
  d::GraphRegionSource source;

  hyle::Status initialize_owner(const LogicalConstant &constant) {
    HYLE_ASSIGN_OR_RETURN(context, c::Context::create(c::cpu::device()));
    HYLE_ASSIGN_OR_RETURN(owner,
        c::StorageOwner::adopt(context, OwnedTrackedResource(lifetime)));
    HYLE_ASSIGN_OR_RETURN(auto backing, m::Buffer::create(owner.ref(), 16, 64));
    std::array<std::byte, 16> bytes;
    HYLE_RETURN_IF_ERROR(constant.source.view().read(bytes));
    HYLE_RETURN_IF_ERROR(backing.view().write(bytes));
    HYLE_ASSIGN_OR_RETURN(ready, owner.adopt_buffer(std::move(backing)));
    HYLE_ASSIGN_OR_RETURN(binding, ready.tensor(constant.description));
    HYLE_ASSIGN_OR_RETURN(auto retained, c::detail::ContextRef::acquire(context));
    const auto allocator = retained.metadata_allocator();
    auto *state = allocator.construct<QueuedArtifactSource>(std::move(retained), allocator);
    if (!state) return hyle::Status::ResourceExhausted("queued test source metadata");
    HYLE_ASSIGN_OR_RETURN(source, d::GraphRegionSource::native(
        d::ExecutionSourceAccess::adopt(state), &QueuedArtifactSource::table));
    return hyle::Status::Ok();
  }
  void initialize_plan(d::PlanState &plan, const LogicalConstant &constant) {
    const std::array resources{owner.ref()};
    initialize(plan, constant, resources);
    plan.metadata_owner = owner;
    plan.storage_domains[0].owner = owner;
    plan.constant_bindings[0] = binding;
    ASSERT_TRUE(plan.targets.try_push_back({.source = source.source()}));
  }
};

struct QueueWitness final {
  QueueWitness() = default;
  QueueWitness(const QueueWitness &) = delete;
  QueueWitness &operator=(const QueueWitness &) = delete;
  c::detail::QueueState *state = nullptr;
  ~QueueWitness() { c::detail::release(state); }
  void retain(c::detail::QueueState *value) noexcept {
    c::detail::retain(value);
    c::detail::release(std::exchange(state, value));
  }
  c::detail::QueueState *get() const noexcept { return state; }
  c::detail::QueueState *operator->() const noexcept { return state; }
};

struct QueuedArtifactCounts final {
  ArtifactCounts payloads;
  // Keep the actual states alive so a new allocation cannot reuse an old
  // address and disguise queue recreation as reuse in these assertions.
  std::array<QueueWitness, 8> observed;
  std::size_t calls = 0;
  std::size_t queues_created = 0;
  bool fail_after_work = false;
};

hyle::Status queued_artifact_factory(void *opaque,
    const g::ConstantArtifactPrepareRequest &request,
    g::ConstantArtifactPayload *output) noexcept {
  auto &counts = *static_cast<QueuedArtifactCounts *>(opaque);
  if (!request.sequence || counts.calls == counts.observed.size())
    return hyle::Status::Internal("queued constant test sequence is missing");
  auto *state = c::detail::ContextAccess::state(*request.sequence);
  bool prior = false;
  for (std::size_t i = 0; i < counts.calls; ++i)
    prior |= counts.observed[i].get() == state;
  if (!prior) ++counts.queues_created;
  counts.observed[counts.calls++].retain(state);
  HYLE_RETURN_IF_ERROR(artifact_factory(&counts.payloads, request, output));
  // The CPU payload read above is already complete; exercise an accepted
  // prefix and its original Completion/retention path, including tail error.
  request.sequence->record(c::detail::EnqueueResult::accepted_command());
  return counts.fail_after_work
      ? hyle::Status::Unavailable("queued constant test tail failure")
      : hyle::Status::Ok();
}

c::ProviderPrepareOutcome acquire_queued(d::ConstantArtifactCache &cache,
    d::PlanState &plan, QueuedConstantOwner &owner,
    const LogicalConstant &constant, std::uint64_t variant,
    QueuedArtifactCounts &counts) {
  const std::array physical{g::ConstantBinding{constant.value, owner.binding,
      g::ConstantPolicy::kBorrow, owner.owner, owner.ready}};
  g::ConstantArtifactView result;
  return cache.acquire(plan, owner.source, g::ExecutionTargetId{0}, physical,
      constant.value,
      {.key = {0x5445535451554555ULL, variant}, .storage = owner.owner.ref(),
       .packed_bytes = 0, .persistent_bytes = sizeof(ArtifactPayload),
       .private_storage = {.resident_bytes = 0, .peak_bytes = 0},
       .queue_preparation = true},
      &queued_artifact_factory, &counts, &result, {});
}

TEST(ConstantArtifactCache, QueuedPreparationReusesTwoInterleavedComputeEndpoints) {
  QueuedArtifactCounts counts;
  const LogicalConstant constant;
  QueuedConstantOwner first, second;
  const auto first_status = first.initialize_owner(constant);
  ASSERT_TRUE(first_status.ok()) << first_status.message();
  const auto second_status = second.initialize_owner(constant);
  ASSERT_TRUE(second_status.ok()) << second_status.message();
  d::PlanState first_plan(m::system_allocator(), constant.generation());
  d::PlanState second_plan(m::system_allocator(), constant.generation());
  first.initialize_plan(first_plan, constant);
  second.initialize_plan(second_plan, constant);
  {
    d::ConstantArtifactCache cache;
    ASSERT_TRUE(cache.begin_candidate().ok());
    for (const auto variant : {1U, 2U}) {
      ASSERT_EQ(acquire_queued(cache, first_plan, first, constant, variant, counts).kind,
                c::ProviderPrepareOutcomeKind::kPrepared);
      ASSERT_EQ(acquire_queued(cache, second_plan, second, constant, variant, counts).kind,
                c::ProviderPrepareOutcomeKind::kPrepared);
    }
    EXPECT_EQ(counts.queues_created, 2U);
    EXPECT_EQ(counts.observed[0].get(), counts.observed[2].get());
    EXPECT_EQ(counts.observed[1].get(), counts.observed[3].get());
    EXPECT_NE(counts.observed[0].get(), counts.observed[1].get());
    cache.commit_plan();
    ASSERT_TRUE(cache.begin_candidate().ok());
    ASSERT_EQ(acquire_queued(cache, first_plan, first, constant, 3, counts).kind,
              c::ProviderPrepareOutcomeKind::kPrepared);
    EXPECT_EQ(counts.observed[0].get(), counts.observed[4].get());
    EXPECT_EQ(counts.queues_created, 2U);
    EXPECT_TRUE(counts.observed[0]->admission.is_open());
    EXPECT_TRUE(counts.observed[1]->admission.is_open());
  }
  EXPECT_FALSE(counts.observed[0]->admission.is_open());
  EXPECT_FALSE(counts.observed[1]->admission.is_open());
}

TEST(ConstantArtifactCache, QueuedFailureDiscardsOnlyItsOriginalEndpointQueue) {
  QueuedArtifactCounts counts;
  const LogicalConstant constant;
  QueuedConstantOwner first, second;
  const auto first_status = first.initialize_owner(constant);
  ASSERT_TRUE(first_status.ok()) << first_status.message();
  const auto second_status = second.initialize_owner(constant);
  ASSERT_TRUE(second_status.ok()) << second_status.message();
  d::PlanState first_plan(m::system_allocator(), constant.generation());
  d::PlanState second_plan(m::system_allocator(), constant.generation());
  first.initialize_plan(first_plan, constant);
  second.initialize_plan(second_plan, constant);
  d::ConstantArtifactCache cache;
  ASSERT_TRUE(cache.begin_candidate().ok());
  ASSERT_EQ(acquire_queued(cache, first_plan, first, constant, 1, counts).kind,
            c::ProviderPrepareOutcomeKind::kPrepared);
  ASSERT_EQ(acquire_queued(cache, second_plan, second, constant, 1, counts).kind,
            c::ProviderPrepareOutcomeKind::kPrepared);
  counts.fail_after_work = true;
  const auto failed = acquire_queued(cache, first_plan, first, constant, 2, counts);
  ASSERT_EQ(failed.kind, c::ProviderPrepareOutcomeKind::kError);
  EXPECT_TRUE(failed.status.IsUnavailable());
  EXPECT_FALSE(counts.observed[0]->admission.is_open());
  EXPECT_TRUE(counts.observed[1]->admission.is_open());
  EXPECT_EQ(counts.payloads.destroys, 1U);
  counts.fail_after_work = false;
  ASSERT_TRUE(cache.begin_candidate().ok());
  ASSERT_EQ(acquire_queued(cache, first_plan, first, constant, 2, counts).kind,
            c::ProviderPrepareOutcomeKind::kPrepared);
  ASSERT_EQ(acquire_queued(cache, second_plan, second, constant, 2, counts).kind,
            c::ProviderPrepareOutcomeKind::kPrepared);
  EXPECT_NE(counts.observed[0].get(), counts.observed[3].get());
  EXPECT_EQ(counts.observed[1].get(), counts.observed[4].get());
  EXPECT_EQ(counts.queues_created, 3U);
}
} // namespace
