#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

// Included after the original product Model. All graphs, finite children,
// budgets and Invocation carriers below belong to that one real Model.
namespace qwen_cold_family_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

inline void selected_shape(const q::FamilyChild &selected, std::size_t rows) {
  const auto &compiled = selected.program.compiled;
  const auto signature = compiled.executable.specialization();
  require(compiled.executable && compiled.report && signature &&
      signature == compiled.report.specialization() && !compiled.executable.is_shape_domain() &&
      !selected.program.outputs().empty(),
      "cold selection did not return one complete concrete executable/report/output child");
  bool found = false;
  for (std::size_t index = 0; index < signature.binding_count(); ++index) {
    const auto binding = q::take(signature.binding(index));
    if (binding.dimension != selected.family.last_row) continue;
    require(binding.value == static_cast<std::int64_t>(rows - 1),
        "cold selection returned a sibling's concrete token dimension");
    found = true;
  }
  require(found, "selected cold child omitted its original query-row dimension");
}

inline int run(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.intermediate == 128 && geometry.vocabulary == 64 &&
      geometry.layers == "A" && geometry.context_limit >= 16,
      "cold family case requires the cached untrained A/H64/FFN128/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, false);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      16u << 20, {.enabled = false}, 0, {.primary = q::BackendKind::kCpu});
  const std::array finite_rows{std::size_t{1}, std::size_t{2}};
  // The first real Family compile fails after its original KV tuple is ready.
  // This is still KV-only storage: no state/hidden pool has been prepared.
  const auto resource = q::mem::system_resource();
  const auto fresh_execution = q::take(model.execution_budget.usage(resource));
  const auto fresh_cache = model.cache_usage();
  const auto fresh_peaks = model.known_workspace_peak, fresh_limits = model.known_workspace_limits;
  require(!model.attention_schema.valid() && model.families.empty() &&
      !model.known_cache_backing_bytes && !model.state_storage_prepared &&
      model.attention_schemas.empty() && model.page_inventory.banks.empty() &&
      model.draft_inventory.banks.empty() && !model.hidden_pool.valid() &&
      std::ranges::all_of(model.state_pools, [](const auto &pools) {
        return !pools.history.valid() && !pools.recurrent.valid();
      }) && !model.snapshot_versions.capacity && !model.snapshot_versions.live &&
      fresh_cache.live_bytes == fresh_cache.live_by_category[
          static_cast<std::size_t>(q::c::ResourceBudgetCategory::kMetadata)] &&
      fresh_execution.live_bytes < fresh_execution.limit_bytes,
      "cold-first failure fixture was already prepared or had no metadata allowance");
  const auto expected_kv_bytes = q::cache::add(q::cache::multiply(
      q::cache::add(plan.key_bank_bytes, plan.value_bank_bytes), geometry.attention_layers()),
      q::cache::add(plan.draft_key_bank_bytes, plan.draft_value_bank_bytes));
  {
    auto pressure = q::take(model.execution_budget.reserve(resource,
        fresh_execution.limit_bytes - fresh_execution.live_bytes));
    bool exhausted = false;
    try { (void)model.prepare_family(finite_rows); }
    catch (const i::EvaluationFailure &failure) {
      exhausted = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
    }
    require(exhausted && model.attention_schema.valid() && model.families.empty() &&
        model.probe_families.empty() && model.phases.empty() && !model.active_step &&
        !model.carrier_borrowed && !model.snapshot_versions.live && !model.state_storage_prepared &&
        !model.hidden_pool.valid() && model.known_workspace_peak == fresh_peaks &&
        model.known_workspace_limits == fresh_limits,
        "cold-first budget refusal lost its KV tuple or published Family/state/execution/diagnostics");
    const auto kv_usage = model.cache_usage();
    const auto kv_metadata = kv_usage.live_by_category[
        static_cast<std::size_t>(q::c::ResourceBudgetCategory::kMetadata)];
    require(kv_usage.live_bytes - kv_metadata == expected_kv_bytes &&
        model.known_cache_backing_bytes == expected_kv_bytes &&
        kv_usage.live_bytes <= kv_usage.limit_bytes &&
        model.page_inventory.banks.size() == geometry.attention_layers() * 2 &&
        model.snapshot_versions.capacity == plan.snapshot_slots &&
        q::take(model.execution_budget.usage(resource)).live_bytes == fresh_execution.limit_bytes,
        "cold-first refusal lost exact KV-only payload/accounting or leaked execution metadata");
  }
  require(q::take(model.execution_budget.usage(resource)).live_bytes == fresh_execution.live_bytes,
      "cold-first refusal retained execution metadata after pressure release");
  require(model.prepare_family(finite_rows), "same first Family could not retry with its retained KV tuple");
  require(model.known_cache_backing_bytes == expected_kv_bytes && !model.state_storage_prepared,
      "same first Family retry duplicated KV backing or prepared unrelated state pools");
  const auto first = model.find_child(1, 1, q::GraphRole::kText);
  const auto second = model.find_child(1, 2, q::GraphRole::kText);
  require(first && second && &first->family == &second->family &&
      &first->program != &second->program && first->family.programs.size() == 2 &&
      model.families.size() == 1, "finite cold family did not publish two distinct complete children");
  auto *const original_family = &first->family;
  const auto *const original_first = &first->program;
  const auto *const original_second = &second->program;
  const auto *const original_graph = &original_family->graph;
  const auto *const original_sampling = &model.sampling_configuration();
  const auto original_unchanged = [&] {
    const auto a = model.find_child(1, 1, q::GraphRole::kText);
    const auto b = model.find_child(1, 2, q::GraphRole::kText);
    require(a && b && &a->family == original_family && &b->family == original_family &&
        &a->program == original_first && &b->program == original_second &&
        &a->family.graph == original_graph && &b->family.graph == original_graph &&
        &a->program.resource_graph() == original_graph && &b->program.resource_graph() == original_graph &&
        a->family.sampling == original_sampling && b->family.sampling == original_sampling &&
        original_family->programs.size() == 2 && original_family->chunks == std::vector<std::size_t>{1, 2} &&
        std::ranges::all_of(model.families, [](const auto &entry) { return bool(entry.second); }),
        "failed/retried cold preparation changed an original complete family or left a null inventory entry");
    selected_shape(*a, 1); selected_shape(*b, 2);
  };
  original_unchanged();

  // This ordinary validation throw occurs after choosing a new exact key.
  // It must not publish an empty inventory slot beside the complete family.
  const std::array invalid_rows{std::size_t{17}};
  const auto invalid_key = q::FamilyKey::exact(1, q::GraphRole::kText, 17);
  bool rejected = false;
  try { (void)model.prepare_family(invalid_rows, false, 1, q::GraphRole::kText, true); }
  catch (const std::runtime_error &error) {
    rejected = std::string_view(error.what()) == "finite text chunk outside nonempty context capacity";
  }
  require(rejected && model.families.size() == 1 && !model.families.contains(invalid_key),
      "invalid cold candidate left an empty exact-family entry");
  original_unchanged();

  const q::FamilyKey state_key(1, q::GraphRole::kText, 0, q::TextCompletion::kStateOnly);
  const auto before = q::take(model.execution_budget.usage(resource));
  require(before.live_bytes < before.limit_bytes, "original execution budget has no free cold metadata allowance");
  {
    // Account for the remaining requested-byte allowance through the original
    // budget. Reservation allocates no payload and does not replace an allocator
    // or synthesize a compiler result; the real legal compile must refuse it.
    auto pressure = q::take(model.execution_budget.reserve(resource, before.limit_bytes - before.live_bytes));
    bool exhausted = false;
    try {
      (void)model.prepare_family(finite_rows, false, 1, q::GraphRole::kText,
          false, q::TextCompletion::kStateOnly);
    } catch (const i::EvaluationFailure &failure) {
      exhausted = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
    }
    require(exhausted && !model.families.contains(state_key) && model.families.size() == 1 &&
        model.phases.empty() && !model.active_step && !model.carrier_borrowed,
        "real cold budget refusal published a partial family, carrier or execution loan");
    original_unchanged();
    require(q::take(model.execution_budget.usage(resource)).live_bytes == before.limit_bytes,
        "failed cold candidate leaked or released an unrelated original budget charge");
  }
  require(q::take(model.execution_budget.usage(resource)).live_bytes == before.live_bytes,
      "failed cold compile retained metadata after original budget pressure was released");
  require(model.prepare_family(finite_rows, false, 1, q::GraphRole::kText,
      false, q::TextCompletion::kStateOnly), "same cold family could not retry after its real budget refusal");
  require(model.families.size() == 2 && model.families.at(state_key) &&
      model.families.at(state_key)->programs.size() == 2,
      "cold retry did not publish exactly one complete requested family");
  original_unchanged();

  // Real bounded exact admission occurs while idle, after the original state
  // storage is prepared and before any physical carrier is created.
  model.plan_execution(finite_rows);
  model.prepare_state_storage();
  const auto before_admission = model.shape_plan.admissions();
  model.admit_exact_shape(3, 1, q::GraphRole::kText);
  const auto exact = model.find_child(1, 3, q::GraphRole::kText);
  require(exact && exact->family.admitted_exact && exact->family.programs.size() == 1 &&
      &exact->family != original_family && model.shape_plan.admissions() == before_admission + 1 &&
      model.families.size() == 3 && model.phases.empty(),
      "actual exact-shape admission did not retain one complete independent cold child");
  selected_shape(*exact, 3);
  const std::array ordered{std::size_t{3}, std::size_t{1}, std::size_t{2}, std::size_t{3}};
  model.plan_execution(ordered);
  require(model.shape_plan.uses({1, 1})[model.slot_index] == 2 &&
      model.shape_plan.uses({1, 2})[model.slot_index] == 2 &&
      model.shape_plan.uses({1, 3})[model.slot_index] == 3,
      "ordered use planning disagreed with the already admitted finite/exact child selection");

  const auto families = model.families.size(), admissions = model.shape_plan.admissions();
  const auto planned = model.shape_plan.entries();
  const std::array unlisted{std::size_t{4}}; // Within context/query bounds, absent from the inventory.
  require(!model.find_child(1, 4, q::GraphRole::kText), "cold lookup invented an unlisted child");
  rejected = false;
  try { model.plan_execution(unlisted); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && model.shape_plan.entries() == planned,
      "use planning admitted an unlisted child or changed earlier complete plans");
  rejected = false;
  try { (void)model.prepare_child(4, false); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && model.phases.empty() && model.families.size() == families &&
      model.shape_plan.admissions() == admissions, "physical lookup silently admitted an unlisted shape");

  const std::array prepared{std::size_t{1}, std::size_t{2}, std::size_t{3}};
  model.prepare_execution(prepared);
  require(model.phases.size() == 3 && model.shape_plan.admissions() == admissions,
      "preparing known finite/exact children performed another cold admission");
  const auto warm_usage = q::take(model.execution_budget.usage(resource));
  {
    auto pressure = q::take(model.execution_budget.reserve(resource, warm_usage.limit_bytes - warm_usage.live_bytes));
    for (const auto rows : prepared) {
      const auto selected = model.find_child(1, rows, q::GraphRole::kText);
      require(bool(selected), "known complete cold child disappeared during physical reuse");
      selected_shape(*selected, rows);
      auto &phase = model.prepare_child(rows, false);
      require(phase.family == &selected->family && phase.child == &selected->program &&
          phase.query_rows == rows && phase.slots[0].carrier.initialized() &&
          &model.prepare_child(rows, false) == &phase,
          "physical reuse selected a different family/program or rebuilt its original phase");
      const auto *run = q::take(phase.slots[0].carrier.borrow());
      require(!run->in_flight() && run->specialization() == selected->program.compiled.executable.specialization(),
          "original Invocation disagreed with the cold selected child specialization");
    }
    rejected = false;
    try { (void)model.prepare_child(4, false); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected && !model.find_child(1, 4, q::GraphRole::kText) && model.phases.size() == 3 &&
        model.families.size() == families && model.shape_plan.admissions() == admissions &&
        model.shape_plan.entries() == planned && !model.active_step && !model.carrier_borrowed,
        "warm lookup added a native/cold owner or admitted an unlisted member under original budget pressure");
    original_unchanged();
  }
  require(q::take(model.execution_budget.usage(resource)).live_bytes == warm_usage.live_bytes &&
      !model.snapshot_versions.live, "cold selection leaked budget or created a Session version");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"finite_children\":2,\"admitted_exact_children\":1,"
      "\"typed_budget_refusal_retry\":true,\"failed_candidate_unpublished\":true,\"stable_original_family\":true,"
      "\"first_compile_refusal_kept_kv_only\":true,\"stable_graph_and_sampling_identity\":true,"
      "\"plan_prepare_selection_equal\":true,\"unlisted_warm_rejected\":true,\"submitted_graphs\":0,\"HTTP\":0}\n";
  return 0;
}
} // namespace qwen_cold_family_case
