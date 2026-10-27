#include "../../engine/models/qwen35/main.cc"
#include "qwen_cpu_ahead_case.h"
#include "qwen_host_image_case.h"
#include "qwen_mtp_batch_state_case.h"

// Prefix lengths are fixture control. All target/draft calculations, selected
// state, repair and publication use the original Model; this is no speed test.
namespace {
namespace q = qwen35;
namespace i = hyle::inference;
namespace state_case = qwen_mtp_batch_state_case;
using qwen_host_image_case::require;
using Image = state_case::StateImage;

struct Arm final {
  std::array<std::vector<Image>, 2> boundaries;
  std::array<std::vector<std::uint64_t>, 2> epochs;
  std::array<std::vector<std::int32_t>, 2> ids;
  q::MtpRoundResult tail;
  std::size_t calls = 0, prepared = 0, hits = 0, discarded = 0, selected_bytes = 0;
};

Arm run_arm(q::Model &model, bool consume) {
  model.discard_decode_ahead();
  const auto prepared_before = model.cpu_ahead_prepared;
  const auto hits_before = model.cpu_ahead_hits;
  const auto discarded_before = model.cpu_ahead_discarded;
  const std::array rows{std::size_t{3}};
  std::array sessions{
      model.create_session(rows, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 7, 11}),
      model.create_session(rows, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 8, 13})};
  const std::array<std::array<std::int32_t, 3>, 2> prompt{{{1, 2, 3}, {3, 2, 1}}};
  const std::array seed{q::EvaluationRequest{sessions[0], prompt[0]},
      q::EvaluationRequest{sessions[1], prompt[1]}};
  const std::array seed_outputs{q::EvaluationOutput{.session = sessions[0]},
      q::EvaluationOutput{.session = sessions[1]}};
  const auto no_observer = [](std::size_t, const q::CompletedEvaluationView &) {};
  model.prepare_evaluations(seed, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluations();
  std::array<std::int32_t, 2> seed_next{};
  model.finish_evaluations(seed_outputs, seed_next, no_observer);
  Arm result; ++result.calls;
  std::array original{sessions[0]->checkpoint(), sessions[1]->checkpoint()};
  const std::array original_images{state_case::image(model, *original[0]), state_case::image(model, *original[1])};
  std::array<q::cache::SnapshotRef, 2> first;
  std::array<Image, 2> first_images;
  const auto save_boundary = [&](std::size_t index) {
    result.boundaries[index].push_back(state_case::image(model, *sessions[index]->head));
    result.epochs[index].push_back(sessions[index]->head.epoch());
    result.ids[index].push_back(sessions[index]->head->next_id);
  };
  for (std::size_t index = 0; index < sessions.size(); ++index) save_boundary(index);

  const auto verify = [&](std::span<const std::size_t> order,
      std::span<const std::size_t> accepted, bool prepare_next, std::size_t expected_hits) {
    require(order.size() == accepted.size() && !order.empty(), "MTP ahead fixture has invalid member geometry");
    std::vector<std::array<std::int32_t, 3>> inputs(order.size());
    std::vector<q::EvaluationRequest> requests;
    std::vector<q::EvaluationOutput> outputs;
    std::vector<Image> before;
    std::vector<std::uint64_t> epochs;
    requests.reserve(order.size()); outputs.reserve(order.size());
    for (std::size_t member = 0; member < order.size(); ++member) {
      const auto index = order[member];
      require(index < sessions.size() && accepted[member] >= 1 && accepted[member] <= 3,
          "MTP ahead fixture chose an invalid actual prefix");
      inputs[member] = {sessions[index]->head->next_id, static_cast<std::int32_t>(4 + 2 * index),
          static_cast<std::int32_t>(5 + 2 * index)};
      requests.push_back({sessions[index], inputs[member]});
      outputs.push_back({.session = sessions[index]});
      before.push_back(state_case::image(model, *sessions[index]->head));
      epochs.push_back(sessions[index]->head.epoch());
    }
    model.prepare_evaluations(requests, nullptr, q::GraphRole::kVerify);
    require(model.active_step && model.active_step->members.size() == order.size() &&
        model.active_step->input_tokens == 3 && model.active_step->phase->prefix_rows == 3,
        "MTP ahead fixture did not prepare its actual K3 verifier");
    require(model.cpu_ahead_hits - hits_before == (consume ? expected_hits : 0),
        "MTP ahead did not consume the actual reordered/subset member entries");
    model.submit_evaluations(); ++result.calls;
    if (prepare_next) {
      const auto old_count = model.cpu_ahead_prepared;
      q::check(model.prepare_decode_ahead());
      require(model.cpu_ahead_prepared == old_count + 1 && model.decode_ahead && model.decode_ahead->ready,
          "submitted K3 verifier did not prepare its finite host metadata");
      for (std::size_t member = 0; member < order.size(); ++member)
        require(sessions[order[member]]->head.epoch() == epochs[member] &&
            q::generation_boundary(*sessions[order[member]]->head) ==
                q::GenerationBoundary{before[member].length, before[member].frontier,
                    before[member].lag, before[member].pending, before[member].catchup, before[member].sampling},
            "host metadata preparation published an epoch, accepted prefix or RNG draw");
      if (!consume) model.discard_decode_ahead();
    }
    std::vector<state_case::Bank> banks;
    std::vector<std::int32_t> next(order.size(), -1);
    model.finish_selected_evaluations(outputs, next,
        [&](std::span<const std::int32_t> candidates, q::StepState &step, auto selected) {
          require(candidates.size() == order.size() * 3, "MTP ahead verifier lost real target rows");
          if (order.size() == 2)
            for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
              if (model.geometry.is_gdn(layer))
                for (const bool recurrent : {false, true})
                  banks.push_back(state_case::inspect(model, step, layer, recurrent));
          for (std::size_t member = 0; member < order.size(); ++member) {
            step.members[member].accepted_tokens = accepted[member];
            // Actual selected target result; only the prefix length is controlled.
            selected[member] = candidates[member * 3 + accepted[member] - 1];
          }
        }, [&](q::StepState &step) { model.repair_recursive_draft(step); }, no_observer);
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
        "MTP ahead verifier retained original work after finish");
    for (std::size_t member = 0; member < order.size(); ++member) {
      const auto index = order[member]; const auto &head = *sessions[index]->head;
      const auto actual = state_case::image(model, head);
      require(sessions[index]->head.epoch() == epochs[member] + 1 &&
          actual.length == before[member].length + accepted[member] && actual.pending == next[member] &&
          actual.sampling == before[member].sampling && actual.lag == (accepted[member] == 1 ? 0u : 1u) &&
          actual.frontier + actual.lag == actual.length &&
          actual.catchup == (actual.lag ? inputs[member][accepted[member] - 1] : -1),
          "MTP ahead changed actual acceptance, epoch, pending, repair frontier or controlled RNG");
      for (const auto &bank : banks) {
        const auto chosen = member * 3 + accepted[member] - 1;
        const auto &lease = bank.recurrent ? head.state[bank.layer].recurrent : head.state[bank.layer].history;
        require(state_case::read(model, lease) == bank.bytes[chosen],
            "MTP ahead committed another member's raw GDN/conv prefix");
        result.selected_bytes += bank.bytes[chosen].size();
      }
      save_boundary(index);
    }
    for (std::size_t index = 0; index < sessions.size(); ++index) {
      require(state_case::image(model, *original[index]) == original_images[index],
          "MTP ahead changed an immutable initial checkpoint");
      if (first[index]) require(state_case::image(model, *first[index]) == first_images[index],
          "MTP ahead changed the frozen first accepted boundary");
    }
  };
  const std::array normal{std::size_t{0}, std::size_t{1}};
  const std::array reordered{std::size_t{1}, std::size_t{0}};
  const std::array prefixes{std::size_t{1}, std::size_t{3}};
  verify(normal, prefixes, true, 0); // Original A/B accept [1,3].
  for (std::size_t index = 0; index < sessions.size(); ++index) {
    first[index] = sessions[index]->checkpoint(); first_images[index] = state_case::image(model, *first[index]);
  }
  verify(reordered, prefixes, true, 1); // Original A/B now accept [3,1].

  // A retained three rows and needs lag1 catch-up; B retained one and is stopped.
  // The actual next singleton is T2, not the previous verifier's T3.
  const auto stopped_image = state_case::image(model, *sessions[1]->head);
  const auto stopped_epoch = sessions[1]->head.epoch();
  q::check(sessions[1]->stop());
  const std::array stopped_ids{sessions[1]->head->next_id, std::int32_t{6}, std::int32_t{7}};
  bool stopped_rejected = false;
  try { model.prepare_evaluation(*sessions[1], stopped_ids, nullptr, q::GraphRole::kVerify); }
  catch (const std::runtime_error &error) {
    stopped_rejected = std::string_view(error.what()).find("stopped") != std::string_view::npos;
  }
  require(stopped_rejected && !model.active_step && !model.carrier_borrowed &&
      sessions[1]->head.epoch() == stopped_epoch && state_case::image(model, *sessions[1]->head) == stopped_image,
      "stopped member preparation was accepted or advanced its original state/RNG");
  const auto &survivor = sessions[0];
  const auto before_tail = state_case::image(model, *survivor->head);
  const auto tail_epoch = survivor->head.epoch();
  require(before_tail.lag == 1, "MTP ahead catch-up input lost its actual repaired lag1");
  const auto pending = survivor->head->next_id;
  model.prepare_evaluation(*survivor, std::span(&pending, 1), nullptr, q::GraphRole::kMtpCatchupRound);
  require(model.active_step && model.active_step->members.size() == 1 && model.active_step->input_tokens == 2 &&
      model.cpu_ahead_hits - hits_before == (consume ? 2u : 0u),
      "MTP ahead did not consume the real singleton T2 catch-up geometry");
  model.submit_evaluations(); ++result.calls;
  const std::array tail_selection{q::MtpGenerationSelection{survivor, pending, 2}};
  const std::array tail_output{q::EvaluationOutput{.session = survivor}};
  std::array<q::MtpRoundResult, 1> tail_results;
  model.finish_mtp_rounds(tail_selection, std::span<const std::int32_t>{}, tail_output, tail_results, no_observer);
  result.tail = std::move(tail_results[0]);
  const auto actual_tail = state_case::image(model, *survivor->head);
  require(!model.active_step && !model.carrier_borrowed && survivor->head.epoch() == tail_epoch + 1 &&
      result.tail.processed_target >= 1 && result.tail.processed_target <= 2 &&
      actual_tail.length == before_tail.length + result.tail.processed_target &&
      actual_tail.pending == result.tail.next_id && actual_tail.sampling == before_tail.sampling.advanced(3),
      "MTP ahead changed the real T2 accepted boundary or sampling cursor");
  save_boundary(0);
  for (std::size_t index = 0; index < sessions.size(); ++index)
    require(state_case::image(model, *original[index]) == original_images[index] &&
        state_case::image(model, *first[index]) == first_images[index],
        "MTP ahead catch-up mutated an immutable earlier checkpoint");
  require(sessions[1]->head.epoch() == stopped_epoch && state_case::image(model, *sessions[1]->head) == stopped_image,
      "surviving metadata subset advanced the stopped peer");
  result.prepared = model.cpu_ahead_prepared - prepared_before;
  result.hits = model.cpu_ahead_hits - hits_before;
  result.discarded = model.cpu_ahead_discarded - discarded_before;
  require(result.calls == 4 && result.prepared == 2 && result.hits == (consume ? 2u : 0u) &&
      result.discarded == (consume ? 1u : 2u),
      "MTP ahead did not consume the reordered/subset entries or discard the stopped member");
  require(!model.decode_ahead || !model.decode_ahead->ready,
      "MTP ahead retained stale cancelled-member preparation");
  first = {}; original = {};
  for (auto &session : sessions) qwen_host_image_case::retire(session);
  require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "MTP ahead weak entries retained a completed Session/version or carrier");
  return result;
}

int run(const std::filesystem::path &input, i::BackendKind kind) {
  const auto ordinary = qwen_cpu_ahead_case::exercise(input, kind);
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() > 0 && geometry.attention_layers() > 0,
      "MTP ahead requires the cached C16/K3 fixture with at least16 tokens");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
  const std::array rows{std::size_t{3}};
  model.prepare_family(rows, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 2, q::GraphRole::kMtpPrefill);
  model.prepare_family(rows, false, 2, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 2, 2, q::GraphRole::kMtpPrefill);
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(rows, false, width, q::GraphRole::kVerify);
    model.plan_replay_uses(3, width, 8, q::GraphRole::kVerify);
  }
  const std::array pair{std::size_t{2}};
  model.prepare_family(pair, false, 1, q::GraphRole::kMtpCatchupRound);
  model.plan_replay_uses(2, 1, 2, q::GraphRole::kMtpCatchupRound);
  for (std::size_t count = 1; count <= 3; ++count) model.prepare_draft_program(count, false);
  for (std::size_t count = 1; count <= 2; ++count) model.prepare_draft_program(count, true);
  model.freeze_shape_plan();
  model.prepare_execution(rows, 2, q::GraphRole::kMtpPrefill);
  for (const auto width : {std::size_t{1}, std::size_t{2}})
    model.prepare_execution(rows, width, q::GraphRole::kVerify);
  model.prepare_execution(pair, 1, q::GraphRole::kMtpCatchupRound);
  const auto control = run_arm(model, false), candidate = run_arm(model, true);
  require(control.boundaries == candidate.boundaries && control.epochs == candidate.epochs && control.ids == candidate.ids,
      "MTP ahead changed complete initialized KV/GDN/conv/hidden/draft bytes, logical boundaries, epoch or RNG");
  require(control.selected_bytes == candidate.selected_bytes && candidate.selected_bytes > 0,
      "MTP ahead lost its real selected-state byte oracle");
  qwen_host_image_case::same_round(control.tail, candidate.tail);
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << qwen_host_image_case::backend_name(kind) << "\",\"ordinary\":";
  qwen_cpu_ahead_case::write_json(ordinary, kind);
  std::cout << ",\"mtp\":{\"target_graph_calls\":8,\"prefixes_by_original_member\":[[1,3],[3,1]],"
      "\"reorder_consumed\":true,\"survivor_subset_consumed\":true,\"stopped_peer_rejected\":true,"
      "\"actual_t3_to_t2_lag1\":true,"
      "\"candidate_prepared\":" << candidate.prepared << ",\"candidate_hits\":" << candidate.hits
      << ",\"candidate_discarded\":" << candidate.discarded << ",\"control_hits\":" << control.hits
      << ",\"selected_bytes\":" << candidate.selected_bytes
      << ",\"complete_initialized_state_equal\":true,\"old_checkpoints_unchanged\":true,\"final_versions\":0,"
      "\"controlled_k3_prefix_selection\":true,\"k3_rng_unchanged\":true,\"tail_t2_rng_advanced\":3,"
      "\"natural_recursive_acceptance_claim\":false,"
      "\"gpu_successor_submissions\":0,\"performance_claim\":false}}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cpu|--cuda prepared local inputs\n"; return 77; }
    if (argc != 3 || (std::string_view(argv[1]) != "--cpu" && std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_mtp_ahead_test --cpu|--cuda PREPARED_LOCAL_INPUTS");
    return run(argv[2], std::string_view(argv[1]) == "--cpu" ? i::BackendKind::kCpu : i::BackendKind::kCuda);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
