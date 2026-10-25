#pragma once

#include "hyle/compute/detail/paged_arena_access.h"

// Included after the actual Model implementation. The prepared AA fixture has
// nonzero CT weights and two different original K/V calibration pairs; it is
// deliberately a wiring/state check, not a trained-model quality certificate.
namespace qwen_e4_cache_case {
namespace q = qwen35;
inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
using BankBytes = std::vector<std::vector<std::byte>>;
inline BankBytes read_pages(q::Model &model, const q::cache::Snapshot &head) {
  BankBytes result;
  for (const auto *banks : {&head.keys, &head.values})
    for (const auto &bank : *banks) {
      const auto pages = q::c::detail::PagedArenaAccess::pages(bank);
      require(pages.size() == 1, "finite E4 case requires one ready physical page per bank");
      const auto view = q::take(pages.front().view());
      // Only the committed rows are compared: an appendable old reader never
      // gains authority to inspect a subsequently written suffix.
      const auto bytes = head.length * model.cache_plan.row_bytes.key;
      require(bytes <= view.size(), "ready E4 frame does not cover its committed rows");
      std::vector<std::byte> values(bytes);
      q::check(model.execution.read(q::take(view.subview(0, bytes)), values));
      result.push_back(std::move(values));
    }
  return result;
}
inline void retire(std::shared_ptr<q::Session> &session) {
  q::check(session->stop());
  const auto observed = session->try_retire(); q::check(observed.status);
  require(observed.retired, "E4 Session did not actually retire"); session.reset();
}
inline int run(const std::filesystem::path &input,
    q::BackendSelection backend = {.primary = q::BackendKind::kCpu}) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.layers == "AA" && geometry.hidden == 64 && geometry.key_depth == 64 &&
      geometry.value_depth == 64 && geometry.kv_heads == 1 && geometry.mtp_layers == 0,
      "E4 case requires prepared ordinary AA/H64/D64 fixture");
  require(source.roles.cache_policy == q::prepared::CachePolicy::kFloat8OriginalCalibration &&
      source.cache_storage.size() == 2 && source.roles.cache_calibration.size() == 4,
      "E4 case lacks complete original calibration roles");
  const auto storage = source.cache_storage;
  require(storage[0][0] != storage[0][1] && storage[0] != storage[1] &&
      storage[1][0] != storage[1][1], "fixture must distinguish per-layer and K/V storage");
  require(source.cache_row_bytes == q::cache::KvRowBytes{64, 64}, "E4 plan inferred 16-bit rows");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 4, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry));
  const auto plain_plan = q::cache::plan(geometry, {128, 128}, 16, 4, 2, 0);
  require(plan.backing_bytes - 4 == (plain_plan.backing_bytes - 4) / 2,
      "physical E4 bank payload was not halved independently of arena sentinels");
  q::Model model(std::move(source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20, {false, {}, 0}, 0, backend);
  const std::array rows{std::size_t{1}, std::size_t{2}};
  model.prepare_family(rows);
  for (const auto count : rows) model.plan_replay_uses(count, 1, 8);
  model.prepare_execution(rows);
  require(model.precision.activation_type() == q::Type::kBFloat16 &&
      model.attention_schemas.size() == 2 && !model.omitted_cache_roles && !model.cache_plan.draft_cache,
      "E4 cache changed activation/MTP policy or lost selected scales");
  require(model.known_cache_backing_bytes == plan.backing_bytes,
      "actual cache backing differs from physical carrier plan");
  for (std::size_t layer = 0; layer < 2; ++layer) {
    const auto &schema = model.target_attention_schema(layer);
    require(schema.capacity().key_storage == storage[layer][0] &&
        schema.capacity().value_storage == storage[layer][1],
        "target layer lost the original shared topology or its own calibration");
    require(model.page_inventory.keys[layer].page_bytes() == 16 * 64 &&
        model.page_inventory.values[layer].page_bytes() == 16 * 64,
        "actual E4 arenas retained the old 16-bit extent");
  }
  for (const auto &[key, family] : model.families) {
    (void)key; std::vector<q::c::CacheStorage> encoders;
    for (std::size_t index = 0; index < family->graph.node_count(); ++index) {
      const auto node = q::take(family->graph.node_id(index));
      const auto operation = family->graph.operation(node);
      if (!operation) continue; // A symbolic view has no concrete operation.
      if (const auto encoder = operation->custom_as<q::c::model_ops::CacheEncode>())
        encoders.push_back(encoder->storage);
    }
    require(encoders.size() == 4 && encoders[0] == storage[0][0] && encoders[1] == storage[0][1] &&
        encoders[2] == storage[1][0] && encoders[3] == storage[1][1],
        "model cold graph did not encode each actual K/V bank with its own scale");
  }
  const std::array chunks{std::size_t{2}, std::size_t{1}};
  auto first = model.create_session(chunks);
  const std::array<std::int32_t, 2> prompt{1, 2};
  const auto pending = q::evaluate(model, *first, prompt, "e4-prefill", {}, false);
  require(first->length() == 2 && pending >= 0 && pending < geometry.vocabulary,
      "E4 prefill did not publish a complete ordinary boundary");
  auto old = first->head;
  for (std::size_t layer = 0; layer < 2; ++layer)
    require(model.target_attention_schema(layer).owns(old->keys[layer].layout()) &&
        model.attention_schema.owns(old->values[layer].layout()) &&
        old->keys[layer].layout() == old->values[layer].layout(),
        "E4 banks did not retain the same original metadata domain/version");
  const auto old_bytes = read_pages(model, *old);
  const std::array<std::size_t, 1> continuation{1};
  auto fork = model.create_session(continuation, old);
  require(fork->head.get() == old.get(), "E4 prefix fork copied/replaced the immutable state");
  const std::array<std::int32_t, 1> token{pending};
  (void)q::evaluate(model, *first, token, "e4-decode", {}, false);
  require(first->length() == 3 && old->length == 2 && read_pages(model, *old) == old_bytes,
      "E4 continuation widened or changed the retained old reader");
  const auto advanced = read_pages(model, *first->head);
  (void)q::evaluate(model, *fork, token, "e4-fork", {}, false);
  require(fork->length() == 3 && fork->head->next_id == first->head->next_id &&
      read_pages(model, *fork->head) == advanced && read_pages(model, *old) == old_bytes,
      "equal E4 prefix forks did not preserve the same complete bank bytes/result");
  retire(first); retire(fork); old.reset();
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      model.draft_programs.empty(), "E4 ordinary consumer retained state or created draft resources");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << model.execution.name()
      << "\",\"generated_AA_fixture\":true,\"original_scale_roles\":4,\"physical_row_bytes\":64,"
         "\"activation\":\"BF16\",\"prefill_rows\":2,\"decode_rows\":1,\"fork_equal\":true,"
         "\"old_readers_unchanged\":true,\"final_versions\":0,\"model_quality\":false,\"HTTP\":0}\n";
  return 0;
}
} // namespace qwen_e4_cache_case
