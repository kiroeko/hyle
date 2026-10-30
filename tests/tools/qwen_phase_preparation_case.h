#pragma once

#include "qwen_cold_family_case.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <span>
#include <utility>

namespace qwen_phase_preparation_case {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_cold_family_case::require;

inline void layouts(const q::Model &model, const q::Phase &phase, bool attached) {
  require(!phase.state_banks.empty(), "physical Phase fixture has no real recurrent banks");
  for (std::size_t index = 0; index != phase.state_banks.size(); ++index) {
    const auto &bank = phase.state_banks[index];
    const auto expected = bank.kind == q::StateBankKind::kHidden ? model.geometry.hidden_bytes()
        : q::take((bank.kind == q::StateBankKind::kRecurrent ? model.state_schema[bank.layer].recurrent
                                                         : model.state_schema[bank.layer].history).required_bytes());
    for (const auto &slot : phase.slots.slots())
      require(slot.transactions.writer_bytes(index) == (attached ? expected : 0),
          "physical Phase transaction layout crossed its original state-storage boundary");
  }
}

inline int run(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "Phase preparation needs the cached untrained GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, false);
  auto backend = q::create_execution(geometry, plan, 0, {.primary = q::BackendKind::kCpu});
  const auto resource = backend.device();
  require(resource == q::mem::system_resource(), "Phase budget fixture lost its actual CPU allocation resource");
  const std::array limits{q::c::ResourceBudgetLimit{resource, 64u << 20}};
  auto budget = q::take(q::c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  q::Model model(std::move(backend), budget, std::move(source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20, {.enabled = false}, 0);
  const std::array rows{std::size_t{1}, std::size_t{3}};
  require(model.prepare_family(rows), "Phase budget fixture did not prepare its original complete Family");
  model.plan_execution(rows); model.freeze_shape_plan();
  const auto selected = model.find_child(1, 1, q::GraphRole::kText);
  require(selected && model.families.size() == 1 && model.phases.empty() && !model.state_storage_prepared &&
      std::ranges::all_of(model.state_pools, [](const auto &pools) {
        return !pools.history.valid() && !pools.recurrent.valid();
      }), "physical failure fixture already prepared its Phase or recurrent storage");
  const auto *const original_family = &selected->family;
  const auto *const original_child = &selected->program;
  const auto kv_backing = model.known_cache_backing_bytes;
  const auto before = q::take(budget.usage(resource));
  require(before.live_bytes < before.limit_bytes && model.allocation_budget.valid(),
      "physical failure fixture has no original allocation allowance");
  {
    // Charge only the original tracked requested-byte scope. Family/cache are
    // already complete; the real Phase Tensor/readback path must refuse.
    auto pressure = q::take(budget.reserve(resource, before.limit_bytes - before.live_bytes));
    bool exhausted = false;
    try { (void)model.prepare_child(1, false); }
    catch (const i::EvaluationFailure &failure) {
      exhausted = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
    }
    const auto same = model.find_child(1, 1, q::GraphRole::kText);
    require(exhausted && model.phases.empty() && model.probe_phases.empty() &&
        same && &same->family == original_family && &same->program == original_child &&
        model.families.size() == 1 && !model.state_storage_prepared &&
        model.known_cache_backing_bytes == kv_backing && !model.active_step && !model.carrier_borrowed &&
        !model.snapshot_versions.live && q::take(budget.usage(resource)).live_bytes == before.limit_bytes,
        "real physical allocation refusal published a Phase or changed original tracked ownership");
  }
  require(q::take(budget.usage(resource)).live_bytes == before.live_bytes,
      "failed physical candidate did not restore its original tracked budget live charge");
  auto &late = model.prepare_child(1, false);
  require(model.phases.size() == 1 && late.family == original_family && late.child == original_child &&
      late.slots.size() == 1 && !late.slots[0].carrier.initialized() && !late.slots.pending() &&
      !model.state_storage_prepared, "same physical key did not rebuild its complete uninitialized Phase");
  q::check(late.slots[0].io.check_program(late.child->compiled));
  layouts(model, late, false);
  model.prepare_state_storage();
  layouts(model, late, true);
  require(model.cache_usage().live_bytes - model.cache_usage().live_by_category[
      static_cast<std::size_t>(q::c::ResourceBudgetCategory::kMetadata)] == model.known_cache_backing_bytes &&
      model.known_cache_backing_bytes == plan.backing_bytes,
      "canonical state preparation lost the exact actual cache payload plan");
  auto &early = model.prepare_child(3, false);
  require(&early != &late && model.phases.size() == 2 && early.slots.size() == 1 &&
      !early.slots[0].carrier.initialized(), "state-first physical preparation changed the earlier Phase");
  layouts(model, early, true);
  model.initialize_slot(late, 0); model.initialize_slot(early, 0);
  const auto *const late_run = q::take(late.slots[0].carrier.borrow());
  const auto *const early_run = q::take(early.slots[0].carrier.borrow());
  const auto warm = q::take(budget.usage(resource));
  {
    auto pressure = q::take(budget.reserve(resource, warm.limit_bytes - warm.live_bytes));
    require(&model.prepare_child(1, false) == &late && &model.prepare_child(3, false) == &early,
        "warm physical lookup rebuilt an original Phase under allocation pressure");
    model.initialize_slot(late, 0); model.initialize_slot(early, 0);
    require(q::take(late.slots[0].carrier.borrow()) == late_run &&
        q::take(early.slots[0].carrier.borrow()) == early_run &&
        q::take(budget.usage(resource)).live_bytes == warm.limit_bytes,
        "warm original initialization allocated tracked storage or replaced a peer Invocation");
  }
  require(q::take(budget.usage(resource)).live_bytes == warm.live_bytes &&
      model.phases.size() == 2 && !model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "physical reuse changed tracked budget, publication or mutable model ownership");
  std::cout << "{\"status\":\"PASS\",\"case\":\"phase-preparation\",\"backend\":\"cpu\","
      "\"allocation_budget_refusal_retry\":true,\"failed_phase_unpublished\":true,\"original_family_unchanged\":true,"
      "\"phase_before_state_layout\":true,\"state_before_phase_layout\":true,\"warm_original_invocation\":true,"
      "\"tracked_budget_live_restored\":true,\"submitted_graphs\":0,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_phase_preparation_case
