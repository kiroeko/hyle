#pragma once

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>

#include "../../engine/inference/output_delivery.h"

// Included after the real Model implementation. This is an ordinary CPU
// numerical-error fixture, not a native fault injection or model-quality test.
namespace qwen_participant_failure_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

// Derive only the existing 98,944-byte tiny Text inventory. Keep an independent
// finite copy of the original tied head BEFORE changing one embedding row;
// otherwise that tied NaN weight would invalidate good B's logits as well.
inline std::filesystem::path prepare_fixture(const std::filesystem::path &input,
    const std::filesystem::path &directory) {
  const auto original = q::read_weight_roles(input / "weights.tsv", false);
  const auto &g = original.geometry;
  require(g.hidden == 64 && g.intermediate == 128 && g.vocabulary == 64 &&
      g.layers == "A" && g.tied_head && original.selected.size() == 13 && original.bytes == 98944,
      "participant failure requires the cached untrained tied A/H64/FFN128/V64 fixture");
  std::filesystem::create_directories(directory);
  const auto payload = std::filesystem::absolute(directory / "participant-bf16.bin");
  const auto manifest = directory / "weights.tsv";
  std::ofstream body(payload, std::ios::binary), output(manifest);
  require(bool(body) && bool(output), "cannot create the bounded participant fixture");
  output << q::prepared::role_signature << '\n' << std::string(64, '4') << '\n' << "geometry";
  for (auto axis : {g.hidden, g.intermediate, g.vocabulary, g.query_heads, g.kv_heads,
      g.key_depth, g.value_depth, g.rotary, g.gdn_key_heads, g.gdn_value_heads,
      g.gdn_key_depth, g.gdn_value_depth, g.conv_kernel, g.context_limit}) output << ' ' << axis;
  output << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10)
      << g.rope_theta << ' ' << g.epsilon << " 0 " << g.mtp_layers << ' '
      << (g.gdn_layout == q::GdnHeadLayout::kGrouped ? "grouped" : "cyclic")
      << ' ' << std::quoted(g.layers) << '\n';
  std::vector<char> finite_head;
  std::size_t offset = 0;
  for (const auto &role : original.selected) {
    require(role.dtype == "BF16" && role.bytes <= 16384,
        "participant fixture must only copy the bounded original BF16 roles");
    std::ifstream source(role.source, std::ios::binary);
    require(bool(source), "cannot open original tiny role bytes");
    source.seekg(static_cast<std::streamoff>(role.offset));
    std::vector<char> bytes(role.bytes);
    source.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(bool(source), "original tiny role is truncated");
    if (role.name == "model.language_model.embed_tokens.weight") {
      finite_head = bytes;
      const std::uint16_t quiet_nan = 0x7fc1;
      for (std::size_t column = 0; column < 64; ++column)
        std::memcpy(bytes.data() + (63 * 64 + column) * 2, &quiet_nan, 2);
    }
    output << role.name << " BF16 " << role.shape.size();
    for (auto axis : role.shape) output << ' ' << axis;
    output << ' ' << std::quoted(payload.generic_string()) << ' ' << offset << ' ' << bytes.size() << '\n';
    body.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    offset += bytes.size();
  }
  require(finite_head.size() == 8192 && offset == 98944,
      "participant fixture did not retain the exact independent finite head");
  output << "lm_head.weight BF16 2 64 64 " << std::quoted(payload.generic_string())
      << ' ' << offset << ' ' << finite_head.size() << '\n';
  body.write(finite_head.data(), static_cast<std::streamsize>(finite_head.size()));
  body.close(); output.close();
  require(bool(body) && bool(output), "cannot finish the bounded participant fixture");
  require(std::filesystem::file_size(payload) == 107136,
      "participant fixture exceeded its exact 107,136-byte bound");
  return manifest;
}

inline void retire(q::GenerationRequest &request) {
  const auto observed = q::retire_generation_request(request);
  if (observed.retired) request.session.reset();
  q::check(observed.status);
  require(observed.retired, "participant owner must retain its key until actual Session retirement");
}

inline int run(const std::filesystem::path &input, std::filesystem::path fixture = {}) {
  // Synthetic Status transport checks only: a native/read error must preserve
  // its code/domain/number/message and never acquire a participant qualifier.
  const auto native = hyle::Status::FromNative(hyle::StatusCode::Unavailable,
      {hyle::NativeErrorDomain::CudaDriver, 700}, "synthetic read observation");
  for (bool through_result : {false, true}) {
    bool observed = false;
    try {
      if (through_result) {
        hyle::Result<int> value(std::unexpect, native);
        (void)q::take(std::move(value));
      } else q::check(native);
    } catch (const i::EvaluationFailure &failure) {
      observed = failure.status() == native && !failure.participant() &&
          !failure.retirement_observed();
    }
    require(observed, "native/read Status was erased or misclassified as a local participant failure");
  }
  if (fixture.empty()) fixture = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
      / ".scratch/validation/qwen-participant-failure";
  const auto manifest = prepare_fixture(input, fixture);
  const auto geometry = q::read_weight_geometry(manifest);
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, cache_rows, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(manifest, geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      16u << 20, {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu});
  require(model.execution.host_only() && !model.cache_plan.draft_cache && !model.physical_mtp_roles,
      "participant isolation must remain ordinary pure CPU without draft resources");
  const std::array rows{std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(rows, false, width);
    model.plan_replay_uses(2, width, 8);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(rows, width);
  const std::array a_turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{63}, 1}};
  const std::array b_turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{3}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  // Deliberately opposite slot order: member 0 is bad A in Window slot 1.
  requests.emplace_back("B", b_turns[0].prompt, std::filesystem::path{}, 2);
  requests.emplace_back("A", a_turns[0].prompt, std::filesystem::path{}, 2);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto &request = requests[index];
    const auto turns = index ? std::span<const i::GenerationTurn>(a_turns)
                             : std::span<const i::GenerationTurn>(b_turns);
    request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 10 + index, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.round_emitted.resize(1); request.round_emitted[0].reserve(1);
    request.turn_observations.resize(turns.size());
    for (std::size_t turn = 0; turn < turns.size(); ++turn) {
      auto owner = std::make_shared<std::vector<std::int32_t>>(turns[turn].prompt.size() + (turn ? 1 : 0));
      std::copy(turns[turn].prompt.begin(), turns[turn].prompt.end(), owner->begin() + (turn ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
  }
  auto &a = requests[1], &b = requests[0];
  i::RequestWindow window; window.prepare(2, 2);
  const auto key_a = window.enqueue(1), key_b = window.enqueue(0);
  a.request_key = key_a; b.request_key = key_b;
  std::array<i::OutputDelivery, 2> delivery{i::OutputDelivery(1), i::OutputDelivery(1)};
  const auto begin = [](q::GenerationRequest &request) {
    request.prompt = request.progress.begin_turn(*request.warm_inputs[request.progress.turn_index],
        request.progress.turn_index ? std::optional<std::int32_t>(request.session->head->next_id) : std::nullopt);
  };
  const auto prepare = [&](q::GenerationRequest &request) {
    return q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr,
        requests, delivery[request.request_key.slot].available(request.progress.generated.size()));
  };
  begin(a); begin(b); window.admitted(key_a); window.admitted(key_b);
  std::array seed{prepare(a), prepare(b)};
  q::execute_generation_calls(model, seed, false, nullptr, requests);
  require(a.progress.turn_done && b.progress.turn_done && a.progress.generated.size() == 1 &&
      b.progress.generated.size() == 1 && a.session->head->sampling.draw == 1 &&
      b.session->head->sampling.draw == 1 && a.session->head->next_id != 63 && b.session->head->next_id != 63,
      "fixture qualification failed: a healthy seed sampled the deliberately invalid future input row");
  for (auto *request : {&a, &b}) {
    window.release_turn(request->request_key, true);
    delivery[request->request_key.slot].acknowledge_prefix(1, 1);
    request->progress.advance_turn(); begin(*request);
    window.admitted(request->request_key);
  }
  const auto *const a_head = a.session->head.get(), *const b_head = b.session->head.get();
  const auto a_before = q::generation_boundary(*a_head), b_before = q::generation_boundary(*b_head);
  const auto a_ids = a.progress.generated, b_ids = b.progress.generated;
  const auto *const b_input = b.prompt.data();
  std::array calls{prepare(a), prepare(b)};
  bool local_failure = false;
  try { q::execute_generation_calls(model, calls, false, nullptr, requests); }
  catch (const i::EvaluationFailure &failure) {
    // The index is relative to these original calls, NEVER RequestKey::slot.
    require(failure.participant() == 0 && failure.status().IsInvalidArgument() &&
        !failure.status().has_native_error() && failure.retirement_observed() &&
        failure.retirement().retired && failure.retirement().status.ok(),
        "invalid A was not individually qualified after a successful original batch retirement");
    const auto bad_key = calls[*failure.participant()].request->request_key;
    require(bad_key == key_a && bad_key.slot == 1 && key_b.slot == 0,
        "participant index was confused with the request Window slot");
    require(a.session->head.get() == a_head && b.session->head.get() == b_head &&
        q::generation_boundary(*a.session->head) == a_before &&
        q::generation_boundary(*b.session->head) == b_before &&
        a.progress.generated == a_ids && b.progress.generated == b_ids &&
        a.ledger->committed_count == 1 && b.ledger->committed_count == 1 &&
        !a.ledger->records[1].committed && !b.ledger->records[1].committed &&
        a.progress.observed_commits() == 1 && b.progress.observed_commits() == 1 &&
        b.progress.turn_started() && !b.progress.prefilled && b.prompt.data() == b_input &&
        a.session->retirement_status.ok() && b.session->retirement_status.ok() &&
        !b.session->stopping && !b.session->failed &&
        !model.active_step && !model.carrier_borrowed && !a.session->pending && !b.session->pending,
        "failed batch changed the original committed Head/ledger/pending/RNG or held its execution carrier");
    // Only clean, proven retirement permits these existing owner transitions.
    window.retire_active(bad_key); retire(a);
    require(delivery[bad_key.slot].pending(a.progress.generated).empty(),
        "bad A still owns undelivered committed output");
    window.release(bad_key);
    window.requeue(key_b);
    local_failure = true;
  }
  require(local_failure && !window.matches(key_a) && window.matches(key_b) &&
      window.waiting_front() == key_b && b.request_key == key_b,
      "finite owner did not retain/requeue the surviving original request key");
  window.admitted(key_b);
  std::array retry{prepare(b)};
  q::execute_generation_calls(model, retry, false, nullptr, requests);
  require(b.progress.request_done() && b.progress.generated.size() == 2 && b.session->length() == 4 &&
      b.session->head->sampling.draw == b_before.sampling.draw + 1 && b.ledger->committed_count == 2 &&
      b.progress.observed_commits() == 2 && b.ledger->records[1].before == b_before &&
      b.ledger->records[1].ids().size() == 2 && b.ledger->records[1].ids()[0] == b_before.next_id &&
      b.ledger->records[1].ids()[1] == 3 && b.session->retirement_status.ok(),
      "same-key survivor did not consume its original pending once and publish exactly one continuation");
  const auto final_b_ids = b.progress.generated;
  window.release_turn(key_b, false);
  require(window.matches(key_b) && delivery[key_b.slot].pending(b.progress.generated).size() == 1,
      "survivor slot escaped before its final output drain");
  delivery[key_b.slot].acknowledge_prefix(2, 2); retire(b); window.release(key_b);
  require(!window.matches(key_b) && window.active().empty() && !window.waiting_front() &&
      !model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "participant failure owner retained a final key, complete version or execution carrier");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"ordinary\":true,"
      "\"fixture\":\"derived_original_tiny_with_one_NaN_embedding_row_and_finite_independent_head\","
      "\"fixture_bytes\":107136,\"member_index\":0,\"bad_window_slot\":1,"
      "\"original_batch_retired\":true,\"survivor_same_key\":true,"
      "\"survivor_committed_records\":2,\"survivor_draw\":2,\"survivor_ids\":["
      << final_b_ids[0] << ',' << final_b_ids[1]
      << "],\"native_status_transport\":true,\"native_fault_injected\":false,"
      "\"multiple_invalid_members\":\"source_only_fail_closed\",\"MTP\":\"unchanged_not_run\","
      "\"final_versions\":0,\"GPU\":false,\"HTTP\":false}\n";
  return 0;
}
} // namespace qwen_participant_failure_case
