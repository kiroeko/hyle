#pragma once

#include "qwen_cold_family_case.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <tuple>
#include <utility>

// Included after the real product Model. This exercises its original budgets,
// cold factories, publication maps and Invocation carriers; no substitute owner.
namespace qwen_auxiliary_preparation_case {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_cold_family_case::require;

inline int run(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 &&
      geometry.context_limit >= 16 && geometry.attention_layers(),
      "auxiliary preparation requires the cached H64/V64/C16 MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20, {.enabled = true}, 0,
      {.primary = q::BackendKind::kCpu});
  const auto resource = q::mem::system_resource();
  const auto idle = [&] {
    return !model.active_step && !model.carrier_borrowed &&
        !model.snapshot_versions.live && !model.state_storage_prepared;
  };
  const auto retry = [&](auto prepare, auto unpublished) {
    require(unpublished() && idle(), "auxiliary failure fixture was already published or active");
    const auto before = q::take(model.execution_budget.usage(resource));
    require(before.live_bytes < before.limit_bytes, "auxiliary fixture has no execution metadata allowance");
    {
      auto pressure = q::take(model.execution_budget.reserve(resource,
          before.limit_bytes - before.live_bytes));
      bool exhausted = false;
      try { prepare(); }
      catch (const i::EvaluationFailure &failure) {
        exhausted = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
      }
      require(exhausted && unpublished() && idle() &&
          q::take(model.execution_budget.usage(resource)).live_bytes == before.limit_bytes,
          "cold budget refusal published an auxiliary program or retained execution metadata");
    }
    require(q::take(model.execution_budget.usage(resource)).live_bytes == before.live_bytes,
        "failed auxiliary candidate did not restore tracked execution live bytes");
    prepare();
    require(!unpublished() && idle(), "same auxiliary key did not retry with its original idle owner");
  };
  retry([&] { model.prepare_sampling_frontier(); },
      [&] { return !model.sampling_configuration().frontier; });
  auto *const frontier = model.sampling_configuration().frontier.get();
  require(frontier != nullptr && !frontier->carrier.initialized(),
      "frontier cold preparation acquired its per-sample Invocation early");
  q::check(frontier->io.check_program(frontier->programs.child(0).compiled));

  const q::PrefixSamplingKey prefix_key{2, 0, false};
  retry([&] { model.prepare_prefix_sampling(2); },
      [&] { return !model.prefix_sampling_programs.contains(prefix_key); });
  auto *const prefix = model.prefix_sampling_programs.at(prefix_key).get();
  require(prefix != nullptr && prefix->carrier.initialized(),
      "complete prefix program lost its original prepared Invocation");
  auto *const prefix_run = q::take(prefix->carrier.borrow());
  q::check(prefix->io.check_program(prefix->programs.child(0).compiled));

  // Prime only the original KV tuple. Its permitted retention and ready head
  // do not mean canonical state/hidden pools or a Session have been admitted.
  model.prepare_cache();
  const auto kv_bytes = model.known_cache_backing_bytes;
  require(kv_bytes && !model.hidden_pool.valid() && idle(),
      "auxiliary fixture unexpectedly prepared canonical state storage");
  const auto draft_key = std::tuple{std::size_t{1}, true, std::size_t{0}};
  retry([&] { model.prepare_draft_program(1, true); },
      [&] { return !model.draft_programs.contains(draft_key); });
  auto *const draft = model.draft_programs.at(draft_key).get();
  require(draft != nullptr && draft->carrier.initialized() && draft->guard.members.empty(),
      "complete draft program replaced its original empty guard/Invocation boundary");
  auto *const draft_run = q::take(draft->carrier.borrow());
  q::check(draft->io.check_program(draft->programs.child(0).compiled));

  const auto warm = q::take(model.execution_budget.usage(resource));
  {
    auto pressure = q::take(model.execution_budget.reserve(resource,
        warm.limit_bytes - warm.live_bytes));
    model.prepare_sampling_frontier();
    model.prepare_prefix_sampling(2);
    model.prepare_draft_program(1, true);
    require(model.sampling_configuration().frontier.get() == frontier &&
        model.prefix_sampling_programs.at(prefix_key).get() == prefix &&
        model.draft_programs.at(draft_key).get() == draft &&
        q::take(prefix->carrier.borrow()) == prefix_run &&
        q::take(draft->carrier.borrow()) == draft_run &&
        !frontier->carrier.initialized() && idle() &&
        q::take(model.execution_budget.usage(resource)).live_bytes == warm.limit_bytes,
        "warm auxiliary lookup allocated metadata or replaced a peer program/Invocation");
  }
  require(q::take(model.execution_budget.usage(resource)).live_bytes == warm.live_bytes &&
      model.known_cache_backing_bytes == kv_bytes && !model.hidden_pool.valid() && idle(),
      "auxiliary failure/retry/warm checks changed tracked ownership or canonical state");
  std::cout << "{\"status\":\"PASS\",\"case\":\"auxiliary-preparation\",\"backend\":\"cpu\","
      "\"cold_budget_refusals\":3,\"same_key_retries\":3,\"warm_original_programs\":true,"
      "\"tracked_execution_live_restored\":true,\"submitted_model_graphs\":0,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_auxiliary_preparation_case
