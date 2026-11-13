// A real model/Graph consumer that links the model core without the CLI or
// request driver. Pass a prepared tiny local input; no weights are downloaded.
#include "../../engine/models/qwen35/model_core.h"
#include "../../engine/models/qwen35/token_ids.h"
#include "../../engine/models/qwen35/request_installation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace q = qwen35;
namespace i = hyle::inference;

const void *qwen_model_core_probe_names() noexcept;

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

std::string fixture_from_environment() {
#if defined(_MSC_VER)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_QWEN_CORE_INPUTS"))
    throw std::runtime_error("cannot read HYLE_QWEN_CORE_INPUTS");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto *value = std::getenv("HYLE_QWEN_CORE_INPUTS");
  return value ? value : "";
#endif
}
} // namespace

int main(int argc, char **argv) try {
  {
    q::VisionGeometry geometry;geometry.merge=2;geometry.image_id=99;geometry.start_id=98;geometry.end_id=97;
    const std::array<std::int32_t,10> ids{5,98,99,99,99,99,99,99,97,6};
    i::PromptAttachment image{"qwen35.image.patches.f32.v1",2,6,{1,4,6},{}};
    const auto plan=q::image_position_plan(geometry,ids,std::span(&image,1),0,10);
    const std::vector<std::array<std::int32_t,3>> expected{{10,10,10},{11,11,11},{12,12,12},
        {12,12,13},{12,12,14},{12,13,12},{12,13,13},{12,13,14},{15,15,15},{16,16,16}};
    require(plan.positions==expected,"non-square image T/H/W positions or following text are incorrect");
    image.begin=1;
    require(q::image_position_plan(geometry,ids,std::span(&image,1),1,10).positions==expected,
        "pending token changed the image's exact prompt interval");
    image.rows=5;bool rejected=false;
    try{(void)q::image_position_plan(geometry,ids,std::span(&image,1),1,10);}
    catch(const std::invalid_argument &){rejected=true;}
    require(rejected,"image grid/placeholder mismatch was accepted");
    q::cache::State state;state.length=20;state.rotary_offset=-3;
    const auto before=q::generation_boundary(state);state.rotary_offset=-2;
    require(before.rotary_offset==-3 && before!=q::generation_boundary(state),
        "complete generation boundary dropped the image rotary offset");
    std::cout<<"PASS Qwen image positions: non-square grid, pending, next text and complete boundary\n";
  }
  if (argc > 4) throw std::runtime_error("usage: qwen_model_core_test [PREPARED_INPUT] [--cuda] [--shared]");
  const auto input = argc > 1 ? std::string(argv[1]) : fixture_from_environment();
  if (input.empty()) {
    std::cerr << "SKIP: pass prepared input or set HYLE_QWEN_CORE_INPUTS\n";
    return 77;
  }
  auto backend = i::BackendKind::kCpu;
  bool shared = false;
  for (int j = 2; j < argc; ++j) {
    if (std::string_view(argv[j]) == "--cuda") backend = i::BackendKind::kCuda;
    else if (std::string_view(argv[j]) == "--shared") shared = true;
    else throw std::runtime_error("unknown core option");
  }
  if (std::ranges::find(i::Backend::compiled_backends(), backend) == i::Backend::compiled_backends().end()) {
    std::cerr << "SKIP: requested backend was not compiled\n";
    return 77;
  }
  const auto directory = std::filesystem::path(input);
  require(qwen_model_core_probe_names() == q::layer0_probe_names.data(),
      "model core inline constants differ between translation units");
  auto source = q::prepare_weight_source(directory / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto prompt = q::read_ids(directory / "prompt.ids", geometry.vocabulary, 16);
  require(prompt.size() >= 2, "core consumer needs at least two input IDs");
  constexpr std::size_t capacity = 16;
  const auto precision = q::PrecisionProfile::from_roles(source.roles);
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, capacity, precision), geometry),
      1, 1, false, false, 1, 0, precision);
  auto endpoint = q::create_execution(geometry, plan, 0, {.primary = backend});
  std::shared_ptr<q::WeightStorage> assets;
  if (shared) {
    const std::array sources{&source};
    auto roles = q::shared_weight_roles(sources);
    require(roles.bytes == source.roles.bytes && roles.selected == source.roles.selected,
        "single source union changed its physical roles");
    auto extended = source;
    auto extra = source.roles.selected.front(); extra.name += ".test-extra";
    extended.roles.selected.push_back(extra);
    const std::array extended_sources{&source, &extended};
    const auto union_roles = q::shared_weight_roles(extended_sources);
    require(union_roles.selected.size() == source.roles.selected.size() + 1 &&
        union_roles.bytes == q::cache::add(source.roles.bytes, extra.bytes),
        "profile union omitted a role or loaded duplicate physical bytes");
    extended.roles.selected.front().offset += 1;
    bool conflict = false;
    try { (void)q::shared_weight_roles(extended_sources); } catch (const std::invalid_argument &) { conflict = true; }
    require(conflict, "matching source identity hid a conflicting physical role");
    assets = q::prepare_shared_weights(endpoint, std::move(roles));
  }
  auto original = std::make_unique<q::Model>(std::move(endpoint), hyle::compute::ResourceBudget{},
      std::move(source), geometry, capacity, plan, std::filesystem::path{}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, q::Model::Sampling{}, 0,
      std::span<const q::SamplingConfiguration>{}, assets);
  std::unique_ptr<q::Model> alias;
  if (shared) {
    auto second_source = q::prepare_weight_source(directory / "weights.tsv", false);
    auto same = q::take(original->execution.share_endpoint());
    alias = std::make_unique<q::Model>(std::move(same), hyle::compute::ResourceBudget{},
        std::move(second_source), geometry, capacity, plan, std::filesystem::path{}, false,
        q::ReplayPolicy::kCommands, SIZE_MAX, q::Model::Sampling{}, 0,
        std::span<const q::SamplingConfiguration>{}, assets);
    require(&original->weights == &alias->weights && &original->weight_pools == &alias->weight_pools &&
        original->execution.same_endpoint(alias->execution), "prepared model copied weights or replaced the original endpoint");
    require(original->resources_ != alias->resources_, "prepared profiles share mutable state resources");
    original.reset(); assets.reset(); // Neither another Model nor the assembler keeps the assets alive.
  }
  auto &model = shared ? *alias : *original;
  {
    i::GenerationSessionSchedule schedule{.id = "installation", .turns = {{{1, 2}, 4}, {{3, 4}, 4}},
        .mode = i::GenerationSessionMode::kTurns, .sampling_seed = 42, .max_draft_tokens = 0,
        .cache_sharing_domain = 19, .sampling_stream = 7};
    const auto visible = std::span(schedule.turns).first(1);
    const auto arrival = q::Clock::now();
    auto initial = q::install_generation_request({.schedule = schedule, .visible_turns = visible, .model = model,
        .output_capacity = 8, .turn_capacity = 8, .record_capacity = 32, .open_turns = true, .arrival = arrival});
    auto admitted = q::install_generation_request({.schedule = schedule, .visible_turns = visible, .model = model,
        .prepared_model = &model, .output_capacity = 8, .turn_capacity = 8, .record_capacity = 32,
        .open_turns = true, .arrival = arrival});
    require(initial.progress.initial_sampling == admitted.progress.initial_sampling &&
        initial.progress.initial_sampling == q::cache::SamplingCursor{42, 7, 0} &&
        initial.cache_sharing_domain == 19 && initial.max_draft_tokens == 0 && !initial.prepared_model &&
        admitted.prepared_model == &model && initial.warm_inputs.size() == 1 && initial.turn_observations.size() == 1 &&
        initial.turn_observations.front().arrival == arrival,
        "request installation changed RNG, profile, visibility or initial arrival");
    schedule.turns[0].prompt[0] = 5;
    require(initial.prompt[0] == 1 && admitted.prompt[0] == 1,
        "installed request borrows mutable ingress input instead of its retained payload");
    auto diagnostic = q::install_generation_request({.schedule = schedule, .visible_turns = schedule.turns, .model = model,
        .output_capacity = 8, .turn_capacity = 2, .record_capacity = 32});
    require(diagnostic.warm_inputs.size() == 2 && diagnostic.warm_inputs[1]->size() == 3 &&
        (*diagnostic.warm_inputs[1])[1] == 3 && diagnostic.turn_observations[0].arrival == q::Clock::time_point{},
        "diagnostic installation lost a visible follow-up or its pending-token prefix");
    schedule.mode = i::GenerationSessionMode::kLive;
    auto live = q::install_generation_request({.schedule = schedule, .visible_turns = visible, .model = model,
        .output_capacity = 8, .turn_capacity = 8, .record_capacity = 32, .open_turns = true});
    auto moved = std::move(live);
    require(moved.live_turns.size() == 1 && moved.progress.initial_sampling == initial.progress.initial_sampling &&
        moved.prompt[0] == 5 && moved.warm_inputs.size() == 1,
        "moving an installed live request invalidated its payload or exposed a future turn");
  }
  bool missing_verifier_rejected = false, missing_timer_rejected = false;
  try { model.bind_device_verification_inputs(q::GraphRole::kVerify, {}); }
  catch (const std::logic_error &) { missing_verifier_rejected = true; }
  try { model.start_device_verification_timing(); }
  catch (const std::logic_error &) { missing_timer_rejected = true; }
  require(missing_verifier_rejected && missing_timer_rejected && !model.active_step,
      "device verification must not invent a prepared target episode");
  const std::array<std::size_t, 2> chunks{2, 1};
  require(model.prepare_family(chunks), "core consumer could not prepare its model family");
  model.plan_execution(chunks);
  model.prepare_execution(chunks);
  auto session = model.create_session(chunks);
  std::int32_t next = -1;
  const auto evaluate = [&](std::span<const std::int32_t> ids) {
    model.prepare_evaluation(*session, ids);
    model.submit_evaluation(*session);
    const std::array outputs{q::EvaluationOutput{.session = session}};
    model.finish_evaluations(outputs, std::span(&next, 1),
        [](std::size_t, const q::CompletedEvaluationView &) {});
  };
  evaluate(std::span(prompt).first(2));
  require(session->length() == 2 && session->head->next_id == next && next >= 0,
      "prefill did not publish its complete state and next ID");
  const auto previous = next;
  evaluate(std::span(&previous, 1));
  require(session->length() == 3 && session->head->next_id == next && next >= 0,
      "decode did not publish its complete state and next ID");
  require(session->head->sampling == q::cache::SamplingCursor{}, "greedy core advanced RNG");
  q::check(session->stop());
  const auto retired = session->drain();
  q::check(retired.status);
  require(retired.retired, "core consumer did not retire its original session");
  session.reset();
  require(!model.active_step && !model.successor_step && !model.snapshot_versions.live,
      "core consumer retained a step or committed version");
  std::cout << "PASS Qwen model core: prefill, decode, greedy RNG and actual session retirement shared=" << shared << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "qwen model core: " << error.what() << '\n';
  return 1;
}
