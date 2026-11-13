// Explicit diagnostic amalgamation. The product compiles generation_runtime.cc
// directly and never includes this historical workload/CLI consumer.
#include "generation_runtime.cc"
#include "generation_report.cc"
#include "driver.h"

int qwen35::run(int argc, char **argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Hyle Qwen diagnostic driver\n"
        "Usage: hyle_qwen35_diagnostic --inputs DIR [options]\n"
        "  --inputs DIR                Locally prepared checkpoint and token IDs\n"
        "  --backend cpu|cuda|apple|vulkan  Primary backend; required when multiple GPU routes are compiled\n"
        "  --mtp auto|off|on           Auto: prepared/explicit assets or eligible CUDA GGUF full-head K1\n"
        "  --mtp-max-drafts N          Maximum proposals; 0 disables drafting (default: artifact recommendation or 1)\n"
        "  --mtp-draft-vocabulary auto|full|FILE  Prepared draft IDs (default: auto); greedy only\n"
        "  --mtp-draft-head auto|original|FILE  Prepared proposal head (default: auto); greedy only\n"
        "  --sampling greedy|random    Token selection (default: greedy)\n"
        "  --temperature FLOAT         Random policy temperature > 0 (default: 1)\n"
        "  --top-k N                    Random policy top-k; 0 disables (default: 0)\n"
        "  --top-p FLOAT                Random policy nucleus in (0,1] (default: 1)\n"
        "  --seed N                     Unsigned 64-bit random seed (default: 0)\n"
        "  --steps N                   Maximum output tokens (default: 4)\n"
        "  --capacity N                Total context capacity (default: 128)\n"
        "  --prefill-chunk-tokens N    Maximum real rows per natural prefill episode\n"
        "  --cpu-ffn-layers N          First N target FFNs on CPU; needs CPU+GPU build (default: 0)\n"
        "  --schedule FILE             One session's prompt/generate turns\n"
        "  --session-schedules FILE    Multiple named session schedules\n"
        "  --max-sessions auto|N       Bounded retained session states\n"
        "  --active-sequences auto|N   Maximum actual execution batch width\n"
        "  --cache-bytes auto|N        Session cache backing budget\n"
        "  --prefix-cache off|auto     Reuse completed prompt/turn checkpoints\n"
        "Advanced execution and diagnostics:\n"
        "  --warmup-shapes lazy|planned  Prepare known carriers before ready (default: lazy)\n"
        "  --progress direct|scheduler  Request progress implementation (default: direct)\n"
        "  --replay auto|commands|prefer-capture\n"
        "  --require-bounded-generation on|off  Reject routes that need future request shapes (default: off)\n"
        "  --decode-overlap auto|off   Automatic route or disable overlap\n"
        "  --trace DIR                 Write generation trace to a local directory\n"
        "  --trace-level full|decisions  State and decision trace, or decisions only\n"
        "  --preparation-diagnostics 0|1  Report preparation diagnostics for fixed replay schedules\n"
        "  --layer0-probe DIR          Write layer-zero probe results\n"
        "  --continuation IDS          Append fixed token IDs before fixed replay\n"
        "  --decode-ids IDS            Fixed replay token IDs\n"
        "Prepare inputs with engine/models/qwen35/prepare.py; no model is downloaded at runtime.\n";
    return 0;
  }
  const auto startup = qwen35::Clock::now();
  try {
    std::filesystem::path input, continuation, trace, layer0_probe, decode_ids, schedule_file, session_file;
    std::filesystem::path draft_vocabulary = "auto";
    std::filesystem::path draft_head="auto";
    std::size_t steps = 4, capacity = 128, max_sessions = 0, active_sequences = 0, cache_bytes = 0, max_drafts = 1, cpu_ffn_layers = 0;
    std::size_t max_prefill_rows = SIZE_MAX;
    bool warmup_planned_shapes = false, decode_overlap = true, require_bounded_generation = false;
    bool progress = false, trace_states = true, explicit_steps = false;
    bool mtp_enabled = false;
    std::string mtp_policy = "auto";
    bool explicit_mtp_drafts = false;
    qwen35::Model::Sampling sampler;
    bool explicit_sampling_parameters = false;
    bool single_schedule = false, multiple_schedules = false;
    std::string prefix_policy;
    bool explicit_continuation = false, explicit_decode_ids = false, preparation_diagnostics = false;
    auto replay = qwen35::ReplayPolicy::kAuto;
    qwen35::BackendSelection backend_selection;
    const auto positive_size = [](const std::string &option, const std::string &value) {
      std::size_t result = 0;
      const auto converted = std::from_chars(value.data(), value.data() + value.size(), result);
      if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(option + " requires a positive integer");
      return result;
    };
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("option needs a value");
      const std::string value = argv[++i];
      if (option == "--inputs") input = value;
      else if (option == "--prefill-chunk-tokens") max_prefill_rows = positive_size(option, value);
      else if (option == "--require-bounded-generation") {
        if (value != "on" && value != "off")
          throw std::runtime_error("--require-bounded-generation requires on or off");
        require_bounded_generation = value == "on";
      }
      else if (option == "--decode-overlap") {
        // Internal benchmark/diagnostic override. Normal use selects the
        // automatic policy and falls back using actual execution facts.
        if (value != "off" && value != "auto") throw std::runtime_error("--decode-overlap requires off or auto");
        decode_overlap = value == "auto";
      }
      else if (option == "--warmup-shapes") {
        if (value != "lazy" && value != "planned")
          throw std::runtime_error("--warmup-shapes requires lazy or planned");
        warmup_planned_shapes = value == "planned";
      }
      else if (option == "--backend") {
        if (value == "cpu") backend_selection.primary = qwen35::BackendKind::kCpu;
        else if (value == "cuda") backend_selection.primary = qwen35::BackendKind::kCuda;
        else if (value == "apple") backend_selection.primary = qwen35::BackendKind::kApple;
        else if (value == "vulkan") backend_selection.primary = qwen35::BackendKind::kVulkan;
        else throw std::runtime_error("--backend requires cpu, cuda, apple or vulkan");
      }
      else if (option == "--sampling") {
        if (value != "greedy" && value != "random") throw std::runtime_error("--sampling requires greedy or random");
        sampler.enabled = value == "random";
      }
      else if (option == "--temperature" || option == "--top-p") {
        float number = 0;
        std::istringstream parsed(value); parsed.imbue(std::locale::classic());
        parsed >> std::noskipws >> number;
        if (!parsed || parsed.peek() != std::char_traits<char>::eof() || !std::isfinite(number) ||
            number <= 0 || (option == "--top-p" && number > 1))
          throw std::runtime_error(option + " requires a finite positive number" + (option == "--top-p" ? " at most one" : ""));
        if (option == "--temperature") sampler.policy.temperature = number; else sampler.policy.top_p = number;
        explicit_sampling_parameters = true;
      }
      else if (option == "--top-k" || option == "--seed") {
        std::uint64_t number = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            (option == "--top-k" && number > INT32_MAX))
          throw std::runtime_error(option + " requires an unsigned integer" + (option == "--top-k" ? " no greater than the vocabulary" : ""));
        if (option == "--seed") sampler.seed = number; else sampler.policy.top_k = static_cast<std::size_t>(number);
        explicit_sampling_parameters = true;
      }
      else if (option == "--mtp") {
        if (value != "auto" && value != "on" && value != "off") throw std::runtime_error("--mtp requires auto, on or off");
        mtp_policy = value;
      }
      else if (option == "--mtp-max-drafts") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), max_drafts);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--mtp-max-drafts requires a nonnegative integer within the supported size range");
        explicit_mtp_drafts = true;
      }
      else if (option == "--mtp-draft-vocabulary") {
        if (value.empty()) throw std::runtime_error("--mtp-draft-vocabulary requires a nonempty file path");
        draft_vocabulary = value;
      }
      else if(option=="--mtp-draft-head"){
        if(value.empty())throw std::runtime_error("--mtp-draft-head requires a nonempty file path");
        draft_head=value;
      }
      else if (option == "--cpu-ffn-layers") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), cpu_ffn_layers);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--cpu-ffn-layers requires a nonnegative integer within the supported size range");
      }
      else if (option == "--continuation") { continuation = value; explicit_continuation = true; }
      else if (option == "--schedule") { schedule_file = value; single_schedule = true; }
      else if (option == "--session-schedules") { session_file = value; multiple_schedules = true; }
      else if (option == "--max-sessions") max_sessions = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--active-sequences") active_sequences = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--cache-bytes") cache_bytes = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--prefix-cache") {
        if (value != "off" && value != "auto") throw std::runtime_error("--prefix-cache requires off or auto");
        prefix_policy = value;
      }
      else if (option == "--trace") trace = value;
      else if (option == "--preparation-diagnostics") {
        if (value != "0" && value != "1") throw std::runtime_error("--preparation-diagnostics requires 0 or 1");
        preparation_diagnostics = value == "1";
      }
      else if (option == "--trace-level") {
        if (value != "full" && value != "decisions")
          throw std::runtime_error("--trace-level requires full or decisions");
        trace_states = value == "full";
      }
      else if (option == "--decode-ids") { decode_ids = value; explicit_decode_ids = true; }
      else if (option == "--layer0-probe") layer0_probe = value;
      else if (option == "--steps") { steps = positive_size(option, value); explicit_steps = true; }
      else if (option == "--capacity") capacity = positive_size(option, value);
      else if (option == "--progress") {
        if (value != "direct" && value != "scheduler")
          throw std::runtime_error("--progress requires direct or scheduler");
        progress = value == "scheduler";
      }
      else if (option == "--replay") {
        if (value != "auto" && value != "commands" && value != "prefer-capture")
          throw std::runtime_error("--replay requires auto, commands or prefer-capture");
        replay = value == "auto" ? qwen35::ReplayPolicy::kAuto
            : value == "prefer-capture" ? qwen35::ReplayPolicy::kPreferCapture
                                         : qwen35::ReplayPolicy::kCommands;
      }
      else throw std::runtime_error("unknown option: " + option);
    }
    const bool scheduled = single_schedule || multiple_schedules;
    if (input.empty()) throw std::runtime_error("--inputs DIR is required");
    backend_selection = qwen35::select_execution(backend_selection, cpu_ffn_layers);
    const bool prefix_enabled = prefix_policy == "auto" || (prefix_policy.empty() && multiple_schedules);
    using Grammar = qwen35::benchmark::ScheduleGrammar;
    const auto grammar = single_schedule ? qwen35::benchmark::schedule_grammar(schedule_file)
        : multiple_schedules ? qwen35::benchmark::session_schedule_grammar(session_file,
            std::vector<qwen35::benchmark::SessionManifestEntry>{}.max_size()) : Grammar::kPromptOnly;
    const bool natural_schedule = scheduled && grammar != Grammar::kFixedInput;
    const bool natural_single = !scheduled && !explicit_continuation && !explicit_decode_ids &&
        layer0_probe.empty() && !preparation_diagnostics;
    const bool natural_generation = natural_schedule || natural_single;
    const bool unmodified_greedy = !sampler.enabled && sampler.modifiers.empty();
    const hyle::inference::ColdWorkloadFacts cold_workload{
        .natural_generation = natural_generation, .multiple_schedules = multiple_schedules,
        .prefix_enabled = prefix_enabled, .scheduler_progress = progress,
        .planned_shapes = warmup_planned_shapes, .cpu_assist = cpu_ffn_layers != 0,
        .greedy = unmodified_greedy,
        .state_trace = !trace.empty() && trace_states,
        .max_sessions = max_sessions, .active_sequences = active_sequences, .capacity = capacity};
    // A source-bound vocabulary opts a prepared directory into drafting.
    // The policy inspects current configuration, never future requests.
    namespace mtp = hyle::inference::mtp;
    const qwen35::MtpDefaultRequest default_request{
        .mode = mtp_policy == "off" ? mtp::RequestedMode::kOff :
            mtp_policy == "on" ? mtp::RequestedMode::kOn : mtp::RequestedMode::kAuto,
        .workload = cold_workload,
        .automatic_backend_eligible = backend_selection.primary == qwen35::BackendKind::kCuda,
        .max_drafts = max_drafts, .automatic_vocabulary = draft_vocabulary == "auto"};
    // The legacy CLI supports policy keywords; resolve them here so the
    // model's asset selection receives an unambiguous file or policy.
    auto vocabulary_request = draft_vocabulary == "auto" ? qwen35::DraftAssetRequest::automatic() :
        draft_vocabulary == "full" ? qwen35::DraftAssetRequest::original() :
        qwen35::DraftAssetRequest::from_file(std::move(draft_vocabulary));
    auto head_request = draft_head == "auto" ? qwen35::DraftAssetRequest::automatic() :
        draft_head == "original" ? qwen35::DraftAssetRequest::original() :
        qwen35::DraftAssetRequest::from_file(std::move(draft_head));
    auto assets = qwen35::select_draft_assets(input, default_request,
        std::move(vocabulary_request), std::move(head_request),
        qwen35::Backend::supports_encoded_row_gather(backend_selection));
    const auto default_selection = assets.policy;
    mtp_enabled = default_selection.enabled;
    auto weight_source = qwen35::prepare_weight_source(input / "weights.tsv", mtp_enabled,
        assets.vocabulary, assets.head, assets.auto_head);
    const auto depth_selection = mtp::select_draft_depth(default_selection, explicit_mtp_drafts,
        max_drafts, weight_source.draft_preferences.recommended_drafts);
    max_drafts = depth_selection.max_drafts;
    const auto geometry = weight_source.roles.geometry;
    qwen35::validate_mtp_source(geometry, mtp_enabled);
    qwen35::FfnPlacement::validate(cpu_ffn_layers, geometry.layers.size());
    if (sampler.policy.top_k > static_cast<std::size_t>(geometry.vocabulary))
      throw std::runtime_error("--top-k exceeds the source vocabulary");
    if (explicit_mtp_drafts && max_drafts != 0 && !mtp_enabled)
      throw std::runtime_error("--mtp-max-drafts requires --mtp on or prepared automatic MTP");
    if (explicit_sampling_parameters && !sampler.enabled)
      throw std::runtime_error("--temperature, --top-k, --top-p and --seed require --sampling random");
    if (sampler.enabled && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("random sampling does not support layer or preparation diagnostics");
    if (mtp_enabled) {
      if (explicit_continuation || explicit_decode_ids || !layer0_probe.empty())
        throw std::runtime_error("--mtp on uses natural generation; --continuation, --decode-ids and --layer0-probe are not supported");
      if (preparation_diagnostics)
        throw std::runtime_error("--preparation-diagnostics is not supported with --mtp on");

    }
    if (prefix_enabled && !scheduled) throw std::runtime_error("--prefix-cache auto requires a schedule");
    if (single_schedule && multiple_schedules)
      throw std::runtime_error("--schedule and --session-schedules are exclusive");
    // The default single-session route admits later turns only after the
    // previous published wave retires. Auto N/R are configuration bounds of
    // one here; future schedule lengths and quotas cannot select this policy.
    // Other backends, sampling, multi-session/prefix and explicit
    // planned workloads retain their finite route and invalid-performance label.
    const auto bounded_plan = qwen35::select_bounded_generation({
        .workload = cold_workload,
        .supported_backend = backend_selection.primary == qwen35::BackendKind::kCuda,
        .mtp_enabled = mtp_enabled, .max_drafts = max_drafts});
    const bool bounded_generation = bounded_plan.enabled;
    if (require_bounded_generation && !bounded_generation)
      throw std::runtime_error("bounded natural generation unavailable: " + std::string(inference::route_reason_name(bounded_plan.reason)));
    if (max_prefill_rows != SIZE_MAX && !natural_generation)
      throw std::runtime_error("--prefill-chunk-tokens requires natural generation; fixed replay is not supported");
    if (mtp_enabled && grammar == Grammar::kFixedInput)
      throw std::runtime_error("--mtp on requires natural prompt/generate requests; decode IDSFILE is fixed replay");
    if (natural_schedule && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("natural requests do not support layer or preparation diagnostics");
    if (scheduled && (explicit_continuation || explicit_decode_ids || (explicit_steps && !natural_schedule)))
      throw std::runtime_error("schedule inputs are exclusive with continuation, decode IDs and fixed-replay --steps");
    if ((single_schedule && schedule_file.empty()) || (multiple_schedules && session_file.empty()))
      throw std::runtime_error("schedule requires a nonempty manifest path");
    if (preparation_diagnostics && !scheduled)
      throw std::runtime_error("--preparation-diagnostics requires a fixed replay schedule");
    std::vector<qwen35::benchmark::SessionSchedule> schedules;
    std::vector<hyle::inference::GenerationSessionSchedule> generation_schedules;
    if (natural_schedule && single_schedule)
      generation_schedules.push_back({"default", qwen35::benchmark::read_generation_schedule(schedule_file,
          geometry.vocabulary, capacity, steps, explicit_steps)});
    else if (natural_schedule && multiple_schedules)
      generation_schedules = qwen35::benchmark::read_generation_session_schedules(session_file, geometry.vocabulary,
          capacity, std::vector<hyle::inference::GenerationSessionSchedule>{}.max_size(), steps, explicit_steps);
    else if (single_schedule)
      schedules.push_back({"default", qwen35::benchmark::read_schedule(schedule_file, geometry.vocabulary, capacity, true)});
    else if (multiple_schedules)
      schedules = qwen35::benchmark::read_session_schedules(session_file, geometry.vocabulary, capacity,
          std::vector<qwen35::benchmark::SessionSchedule>{}.max_size(), true);
    if (bounded_generation && natural_schedule && generation_schedules.size() != 1)
      throw std::runtime_error("bounded workload driver accepts exactly one session");
    const auto input_sessions = bounded_generation ? std::size_t{1}
        : natural_schedule ? generation_schedules.size() : scheduled ? schedules.size() : std::size_t{1};
    if (bounded_generation && active_sequences == 0) active_sequences = 1;
    const auto source_precision = qwen35::PrecisionProfile::from_roles(weight_source.roles);
    const auto recurrent_payload = qwen35::recurrent_bytes(qwen35::state_schema_for(geometry, capacity, source_precision), geometry);
    // Configuration describes a maximum, never permission to overrun context.
    // Actual calls additionally clip against output quota and remaining rows.
    max_drafts = (std::min)(max_drafts, capacity - 1);
    if (max_sessions == 0) max_sessions = input_sessions;
    const auto cached_slots = (std::min)(max_sessions, input_sessions);
    if (active_sequences > max_sessions)
      throw std::runtime_error("--active-sequences cannot exceed --max-sessions");
    const auto forced = decode_ids.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(decode_ids, geometry.vocabulary, capacity);
    if (!decode_ids.empty()) {
      if (forced.empty()) throw std::runtime_error("--decode-ids requires a nonempty token tape");
      const auto forced_steps = qwen35::cache::add(forced.size(), 1);
      if (explicit_steps && steps != forced_steps)
        throw std::runtime_error("--steps must equal decode input count plus one");
      steps = forced_steps;
    }
    if (input.empty())
      throw std::runtime_error("usage: hyle --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp auto|off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
    const auto prompt = scheduled ? std::vector<std::int32_t>{} : qwen35::read_ids(input / "prompt.ids", geometry.vocabulary, capacity);
    const auto extra = continuation.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(continuation, geometry.vocabulary, capacity);
    if (!scheduled) {
      if (prompt.empty()) throw std::runtime_error("generation requires a nonempty prompt before loading weights");
      if (prompt.size() > capacity || extra.size() > capacity - prompt.size() ||
          steps - 1 > capacity - prompt.size() - extra.size())
        throw std::runtime_error("prompt, continuation and requested generation exceed --capacity");
    }
    if (natural_single) generation_schedules.push_back({"default", {{prompt, steps}}});
    const inference::PrefillPolicy prefill(max_prefill_rows);
    std::vector<std::size_t> planned_chunks;
    std::size_t mtp_schema_rows = 0;
    bool mtp_has_decode = !scheduled && steps > 1;
    if (bounded_generation) {
      const auto &first = generation_schedules.front().turns.front();
      planned_chunks.push_back(prefill.next(first.prompt.size()).rows);
      mtp_schema_rows = capacity;
      mtp_has_decode = mtp_enabled && capacity > 1;
    } else if (natural_generation) {
      for (const auto &request : generation_schedules) for (std::size_t i = 0; i < request.turns.size(); ++i) {
        const auto tokens = qwen35::cache::add(request.turns[i].prompt.size(), i ? 1 : 0);
        const auto chunk_rows = prefill.next(tokens).rows;
        planned_chunks.push_back(chunk_rows);
        // The first nonempty prefix leaves at most tokens-1 new rows. Later
        // turns can consume the full carrier plus their historical catch-up.
        if (mtp_enabled) mtp_schema_rows = (std::max)(mtp_schema_rows,
            qwen35::cache::add(chunk_rows, i || (prefix_enabled && tokens > chunk_rows) ? 1 : 0));
        mtp_has_decode = mtp_has_decode || request.turns[i].max_outputs > 1;
      }
    } else if (scheduled) {
      for (const auto &request : schedules) for (const auto &phase : request.phases) {
        const auto tokens = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? phase.ids.size() : std::size_t{1};
        if (std::find(planned_chunks.begin(), planned_chunks.end(), tokens) == planned_chunks.end())
          planned_chunks.push_back(tokens);
      }
    } else {
      planned_chunks.push_back(prompt.size());
      if (!extra.empty()) planned_chunks.push_back(extra.size());
      planned_chunks.insert(planned_chunks.end(), steps - 1, 1);
    }
    // Immutable host token owners are bounded by the validated input batch.
    // Only suffixes at real prompt/turn boundaries enter the cold shape union.
    std::vector<qwen35::cache::TokenPlanRef> token_plans;
    if (prefix_enabled) {
      token_plans.reserve(input_sessions);
      if (natural_generation) for (const auto &request : generation_schedules) {
        token_plans.push_back(qwen35::cache::token_plan(request.turns.front().prompt, capacity));
      } else for (const auto &request : schedules) {
        std::vector<std::span<const std::int32_t>> chunks;
        chunks.reserve(request.phases.size());
        for (const auto &phase : request.phases) chunks.push_back(phase.ids);
        token_plans.push_back(qwen35::cache::token_plan(chunks, capacity));
      }
    }
    const auto max_query_rows = bounded_generation ? capacity : (std::max)({*std::max_element(planned_chunks.begin(), planned_chunks.end()), mtp_schema_rows,
        mtp_enabled && mtp_has_decode ? max_drafts + 1 : std::size_t{1}});
    const auto requested_active = (std::min)(active_sequences, input_sessions);
    const bool stored_draft_head = qwen35::supports_stored_draft_projection(weight_source.roles);
    const qwen35::DeviceRelayCapabilities device_relay{
        backend_selection.primary, cpu_ffn_layers != 0, progress,
        sampler.enabled && mtp_enabled, !sampler.modifiers.empty()};
    const qwen35::DecodeRouteFacts route_facts{
        .workload = cold_workload, .overlap_requested = decode_overlap,
        .mtp_enabled = mtp_enabled, .max_drafts = max_drafts,
        .max_device_drafts = qwen35::kMaximumDeviceMtpDrafts,
        .stored_draft_head = stored_draft_head,
        .relay_eligible = device_relay.eligible(), .relay_refusal = device_relay.refusal_reason(),
        .commands_replay = replay == qwen35::ReplayPolicy::kCommands,
        .has_decode = bounded_generation ? capacity > 1 : std::ranges::any_of(generation_schedules, [](const auto &request) {
          return std::ranges::any_of(request.turns, [](const auto &turn) { return turn.max_outputs > 1; });
        }), .resolved_active_sequences = active_sequences == 0 ? cached_slots : requested_active};
    // A continuous conditional successor retains its completed producer until
    // the consumer really retires. Count that third original batch in the
    // existing page/state bound, independently of simultaneous GPU work.
    auto [overlap_plan, verification_plan] = qwen35::plan_decode_routes(route_facts);
    bool overlap_selected = overlap_plan.enabled;
    bool device_verification_selected = verification_plan.enabled;
    // Ask the actual backend about the provisional maximum batch and every
    // target cache storage. Unsupported implementations keep the original
    // two-batch finite route. The created endpoint is reused by Model below.
    auto continuous = qwen35::prepare_mtp_continuation({.routes = route_facts, .geometry = geometry,
        .source = weight_source, .precision = source_precision, .backend = backend_selection,
        .context = capacity, .residents = cached_slots,
        .active_sequences = active_sequences == 0 ? cached_slots : requested_active,
        .query_rows = mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows,
        .recurrent_bytes = recurrent_payload, .draft_depth = max_drafts, .overlap = overlap_selected,
        .verification = device_verification_selected, .random = sampler.enabled});
    auto overlap_depth_reason = continuous.reason;
    auto overlap_depth = continuous.enabled ? std::size_t{3} : std::size_t{2};
    const auto plan_for_routes = [&](std::size_t residents, bool overlap, std::size_t depth, bool verification) {
      return qwen35::cache::plan(geometry, weight_source.cache_row_bytes, capacity, residents,
          mtp_enabled && mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? residents : requested_active,
          mtp_enabled ? max_drafts + 1 : 1, mtp_enabled, sampler.enabled, overlap ? depth : 1,
          verification && (active_sequences == 0 ? residents : requested_active) == 1 ? 2 : 0, source_precision,
          verification ? max_drafts : 0);
    };
    const auto minimum_residents = active_sequences == 0 ? std::size_t{1} : requested_active;
    const auto budget_routes = qwen35::select_decode_budget_routes(
        {overlap_plan, verification_plan, overlap_depth, overlap_depth_reason},
        [&](const auto &routes) {
          return !cache_bytes || plan_for_routes(minimum_residents, routes.overlap.enabled,
              routes.overlap_depth, routes.verification.enabled).backing_bytes <= cache_bytes;
        });
    overlap_plan = budget_routes.overlap;
    verification_plan = budget_routes.verification;
    overlap_selected = overlap_plan.enabled;
    device_verification_selected = verification_plan.enabled;
    overlap_depth = budget_routes.overlap_depth;
    overlap_depth_reason = budget_routes.overlap_depth_reason;
    const auto plan_for = [&](std::size_t residents) {
      return plan_for_routes(residents, overlap_selected, overlap_depth, device_verification_selected);
    };
    auto selected_cache = plan_for(cached_slots);
    // Caller-written pools have an exact cold formula. Native generated
    // results instead enter the same byte scope after real completion, before
    // any append is taken or head published. Do not estimate native N from the
    // CUDA pool formula or pretend auto bounds opaque execution allocations.
    // CPU cache metadata shares the payload's actual host resource. The GPU
    // payload formula is not an upper bound for that combined byte scope.
    // Auto keeps the configured version/batch bounds; an explicit byte cap
    // still accounts for and limits every cache allocation, including metadata.
    const bool native_generated_storage = backend_selection.primary == qwen35::BackendKind::kApple ||
        backend_selection.primary == qwen35::BackendKind::kVulkan;
    if (cache_bytes == 0) cache_bytes = (native_generated_storage || backend_selection.primary == qwen35::BackendKind::kCpu)
        ? SIZE_MAX : selected_cache.backing_bytes;
    else if (!native_generated_storage && selected_cache.backing_bytes > cache_bytes) {
      const auto minimum = plan_for(minimum_residents);
      if (minimum.backing_bytes > cache_bytes)
        throw std::runtime_error("--cache-bytes cannot hold the configured active sequences: requires " +
            std::to_string(minimum.backing_bytes) + " shared KV/RNN backing bytes");
      std::size_t low = minimum_residents, high = cached_slots;
      while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (plan_for(middle).backing_bytes <= cache_bytes) low = middle;
        else high = middle - 1;
      }
      selected_cache = plan_for(low);
    }
    const auto cache_slots = selected_cache.resident_sessions;
    const auto active_slots = selected_cache.active_sequences;
    const auto metadata_prepare_seconds = weight_source.metadata_prepare_seconds;
    const auto model_start = qwen35::Clock::now();
    auto execution = continuous.endpoint ? std::move(*continuous.endpoint)
        : qwen35::create_execution(geometry, selected_cache, cpu_ffn_layers, backend_selection);
    qwen35::Model model(std::move(execution), {}, std::move(weight_source), geometry, capacity,
        selected_cache, layer0_probe, progress, replay, cache_bytes, sampler, cpu_ffn_layers);
    const qwen35::RouteDecisionFacts routes{
        .mtp_selection_reason = default_selection.reason, .mtp_depth_source = depth_selection.source,
        .draft_vocabulary_selection_reason = assets.vocabulary_reason,
        .bounded_generation_plan_reason = bounded_plan.reason, .decode_overlap_plan_reason = overlap_plan.reason,
        .device_verification_plan_reason = verification_plan.reason, .overlap_depth_plan_reason = overlap_depth_reason};
    model.configure_draft_policy({max_drafts});
    const auto constructed = qwen35::Clock::now();
    if (!trace.empty()) model.trace_routes = trace / "routes";
    if (natural_generation) {
      if (mtp_enabled) return qwen35::run_generation_sessions<qwen35::GenerationMode::kMtp>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected, {}, nullptr, bounded_generation, routes);
      return qwen35::run_generation_sessions<qwen35::GenerationMode::kText>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected, {}, nullptr, bounded_generation, routes);
    }

    // These owners are declared after Model, including on exception paths.
    // The schedule is consumer policy; every call uses the same Model/Graph chain.
    struct PhaseSummary {
      std::size_t context_before = 0, context_after = 0;
      double seconds = 0, step_sum_seconds = 0;
      std::string name, memory_name;
      qwen35::Clock::time_point start{};
    };
    struct Call { std::size_t phase, local; };
    // Count/idle policy only. Actual shared arena and state-pool backings own
    // the byte reservations; charging this directory per Session would double
    // count shared pages and immutable snapshots.
    hyle::inference::SessionCache resident_cache({cache_slots, (std::numeric_limits<std::size_t>::max)()});
    // Destroy runs (and their real slots) before this directory and Model.
    auto prefix_cache = prefix_enabled ? std::make_unique<qwen35::cache::PrefixCache>(cache_slots) : nullptr;
    struct SessionRun {
      struct RecomputedDecision { std::size_t chunk; std::int32_t previous, actual; };
      std::shared_ptr<qwen35::Session> session;
      hyle::inference::SessionCache::Pin pin;
      std::vector<std::size_t> chunks;
      std::shared_ptr<qwen35::cache::ReplayHistory> history;
      std::vector<PhaseSummary> summaries;
      std::vector<Call> calls;
      std::vector<std::int32_t> generated, consumed;
      std::vector<inference::StepObservation> observations;
      std::vector<inference::PreparationObservation> preparation;
      qwen35::cache::SamplingCursor saved_sampling;
      std::int32_t saved_pending = -1;
      std::filesystem::path trace;
      std::string trace_subdirectory;
      qwen35::Clock::time_point start{}, endpoint{};
      std::size_t visible_tokens = 0, cache_hits = 0, cache_misses = 0;
      std::size_t restored_calls = 0, restored_tokens = 0;
      std::vector<RecomputedDecision> recomputed_decisions;
      std::size_t graph_participations = 0, actual_graph_tokens = 0;
      std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
      std::size_t prefix_sampling_calls = 0;
      double step_sum_seconds = 0, first_token_elapsed_seconds = 0, cache_prepare_seconds = 0;
    };
    std::vector<SessionRun> runs(schedules.size());
    std::size_t peak_committed_versions = 0;
    const auto note_committed_versions = [&] {
      peak_committed_versions = (std::max)(peak_committed_versions, std::size_t(model.snapshot_versions.live));
    };
    std::size_t all_calls = 0;
    for (std::size_t s = 0; s < schedules.size(); ++s) {
      const auto &request = schedules[s];
      auto &run = runs[s];
      run.saved_sampling = {model.sampling_seed, static_cast<std::uint64_t>(s), 0};
      run.summaries.resize(request.phases.size());
      if (prefix_enabled) run.history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
      run.trace_subdirectory = "session" + std::to_string(s) + "_" + request.id;
      run.trace = trace.empty() || !multiple_schedules ? trace : trace / run.trace_subdirectory;
      for (std::size_t i = 0; i < request.phases.size(); ++i) {
        const auto &phase = request.phases[i];
        auto &summary = run.summaries[i];
        summary.name = "round" + std::to_string(phase.turn) +
            (phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? "_prefill" : "_decode");
        summary.memory_name = multiple_schedules ? request.id + "_" + summary.name : summary.name;
        const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
        const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
        for (std::size_t local = 0; local < count; ++local) {
          run.calls.push_back({i, local});
          run.chunks.push_back(is_prompt ? phase.ids.size() : std::size_t{1});
        }
      }
      if (run.calls.size() > (std::numeric_limits<std::size_t>::max)() - all_calls)
        throw std::runtime_error("total schedule call count overflows");
      all_calls += run.calls.size();
      run.generated.reserve(run.calls.size()); run.consumed.reserve(run.calls.size());
      run.observations.resize(run.calls.size());
      if (preparation_diagnostics) run.preparation.resize(run.calls.size());
      for (std::size_t i = 0; i < run.calls.size(); ++i) {
        const auto call = run.calls[i];
        run.observations[i].name = qwen35::benchmark::phase_name(request.phases[call.phase], call.local);
      }
    }
    const auto waves = scheduled ? qwen35::benchmark::make_schedule_waves(schedules, active_slots)
        : std::vector<qwen35::benchmark::ScheduleWave>{};
    std::vector<std::pair<std::size_t, std::size_t>> dispatch;
    dispatch.reserve(all_calls);
    std::vector<std::vector<std::size_t>> query_options(all_calls);
    std::vector<std::map<std::size_t, std::size_t>> published_at(schedules.size());
    std::vector<std::size_t> prior_tokens(schedules.size());
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      dispatch.emplace_back(call.session, call.call);
      prior_tokens[call.session] = qwen35::cache::add(prior_tokens[call.session], call.query_rows);
      const auto &phase = schedules[call.session].phases[call.phase];
      if (phase.kind == qwen35::benchmark::PhaseKind::kPrompt || call.local + 1 == phase.ids.size())
        published_at[call.session].emplace(prior_tokens[call.session], w);
    }
    std::fill(prior_tokens.begin(), prior_tokens.end(), 0);
    std::size_t restore_call_bound = 0;
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      auto &before = prior_tokens[call.session];
      const auto target = qwen35::cache::add(before, call.query_rows);
      if (call.kind == qwen35::benchmark::PhaseKind::kPrompt) {
        if (schedules.size() > cache_slots) restore_call_bound = qwen35::cache::add(restore_call_bound, before);
        if (prefix_enabled) for (std::size_t candidate = 0; candidate < token_plans.size(); ++candidate)
          for (const auto boundary : token_plans[candidate]->checkpoints) {
            const auto producer = published_at[candidate].find(boundary);
            if (boundary < before || boundary > target || producer == published_at[candidate].end() || producer->second >= w ||
                !qwen35::cache::same_prefix(*token_plans[candidate], *token_plans[call.session], boundary)) continue;
            query_options[call.ordinal].push_back(target - boundary);
          }
      }
      before = target;
    }
    const auto shape_uses = qwen35::benchmark::plan_batch_shape_uses(waves, query_options);
    if (!scheduled) {
      model.prepare_family(planned_chunks);
      if (!layer0_probe.empty()) model.prepare_family(planned_chunks, true);
      model.plan_execution(planned_chunks);
    } else {
      std::map<std::size_t, std::vector<std::size_t>> chunks_by_width;
      for (const auto &shape : shape_uses) chunks_by_width[shape.requests].push_back(shape.query_rows);
      for (const auto &[width, chunks] : chunks_by_width) {
        model.prepare_family(chunks, false, width);
        if (!layer0_probe.empty()) model.prepare_family(chunks, true, width);
      }
      if (!multiple_schedules && !prefix_enabled) model.plan_execution(runs.front().chunks);
      else for (const auto &shape : shape_uses)
        model.plan_replay_uses(shape.query_rows, shape.requests,
            qwen35::cache::add(shape.max_wave_calls, shape.requests == 1 ? restore_call_bound : 0));
    }
    struct BatchObservation {
      qwen35::benchmark::PhaseKind kind;
      std::size_t tokens;
      bool restore;
      double seconds;
      hyle::container::SmallVector<std::size_t, 4> sessions;
    };
    struct WaveObservation {
      int kind;
      std::size_t outputs = 0, logical_tokens = 0, actual_tokens = 0, graph_calls = 0;
      double seconds = 0;
    };
    std::vector<BatchObservation> batches;
    batches.reserve(all_calls);
    std::vector<WaveObservation> wave_observations;
    wave_observations.reserve(waves.size());
    std::size_t executed_graph_calls = 0, executed_graph_tokens = 0, max_batch_requests = 0;
    const auto record_batch = [&](qwen35::benchmark::PhaseKind kind, std::size_t tokens,
                                   std::span<const std::size_t> members, bool restore, double seconds) {
      BatchObservation value{kind, tokens, restore, seconds, {}};
      if (!value.sessions.try_reserve(members.size())) throw std::bad_alloc{};
      for (auto s : members) value.sessions.push_back(s);
      batches.push_back(std::move(value));
      executed_graph_calls = qwen35::cache::add(executed_graph_calls, 1);
      executed_graph_tokens = qwen35::cache::add(executed_graph_tokens, qwen35::cache::multiply(tokens, members.size()));
      max_batch_requests = (std::max)(max_batch_requests, members.size());
    };
    std::vector<std::size_t> evicted_sessions;
    evicted_sessions.reserve(all_calls);
    const auto make_initial_room = [&] {
      if (model.active_step) throw std::runtime_error("initial admission requires retired sessions");
      const std::array selected{qwen35::version_holder(nullptr)};
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache ? prefix_cache->size() : 0,
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache && prefix_cache->retains(head); }).fits};
      };
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache && prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("prefix/session heads retain the committed version budget before initial admission");
    };
    const auto make_batch_room = [&](std::span<qwen35::Session *const> owners) {
      if (owners.empty() || model.active_step || model.snapshot_versions.live > cache_slots)
        throw std::runtime_error("batch advance requires retired sessions within the committed version budget");
      for (auto *owner : owners) if (!owner || !owner->head || owner->pending || owner->failed || owner->stopping)
        throw std::runtime_error("batch advance contains a stopped, failed or pending session");
      // With prefix reuse disabled, each resident Session has its own sole
      // committed head. Advancing it replaces one version; no directory scan.
      if (!prefix_cache) return;
      const auto selected = owners | std::views::transform([](const auto *owner) {
        return qwen35::version_holder(owner);
      });
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache->size(),
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache->retains(head); }).fits};
      };
      // Dropping a directory alias need not free a version. Re-read both live
      // inventory and external retainers after EVERY eviction, including full hits.
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("batch commit exceeds N retained snapshots; release a checkpoint/session or increase the cache");
    };
    const auto make_advance_room = [&](qwen35::Session &owner) {
      const std::array owners{&owner}; make_batch_room(owners);
    };
    const auto ensure_resident = [&](std::size_t s, bool defer_restore) {
      auto &run = runs[s];
      if (run.pin) return;
      const auto begin = qwen35::Clock::now();
      const auto admission = resident_cache.admit(s, 0, [&](std::size_t victim) {
        auto &old = runs[victim];
        if (old.pin) return hyle::inference::RetirementObservation{};
        hyle::inference::RetirementObservation retired{true, {}};
        if (old.session) {
          (void)old.session->stop();
          retired = old.session->try_retire();
          if (!retired.retired) return retired;
        }
        old.session.reset(); // Actual state leases retire before quota returns.
        evicted_sessions.push_back(victim);
        return retired;
      });
      check(admission.retirement_status);
      if (!admission.accepted())
        throw std::runtime_error(admission.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven
            ? "session cache cannot prove retirement; the original owner remains charged"
            : "session cache admission exhausted; reduce --active-sequences or increase --max-sessions/--cache-bytes");
      run.pin = resident_cache.pin(s);
      if (!run.pin) throw std::runtime_error("session cache could not pin the admitted owner");
      if (run.session) ++run.cache_hits;
      else {
        ++run.cache_misses;
        if (!defer_restore) {
          make_initial_room();
          run.session = model.create_session(run.chunks, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute the original token/chunk tape; never resample or replace
          // already returned tokens. A different legal batch reduction may
          // change logits/greedy decisions. The rebuilt real head is authority
          // for new work; old decisions remain diagnostics, not state payload.
          for (std::size_t i = 0; i < run.generated.size(); ++i) {
            const auto call = run.calls[i];
            const auto &phase = schedules[s].phases[call.phase];
            const auto ids = phase.kind == qwen35::benchmark::PhaseKind::kPrompt
                ? std::span<const std::int32_t>(phase.ids)
                : std::span<const std::int32_t>(&phase.ids[call.local], 1);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_call" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != run.generated[i]) run.recomputed_decisions.push_back({i, run.generated[i], next});
            ++run.restored_calls; run.restored_tokens += ids.size();
            ++run.graph_participations; run.actual_graph_tokens += ids.size();
            const std::array members{s};
            record_batch(phase.kind, ids.size(), members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
            note_committed_versions();
          }
          if (run.session->length() != run.visible_tokens)
            throw std::runtime_error("restored cache context differs from retained consumer history");
          if (run.saved_pending >= 0) model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
        }
      }
      (void)model.cache_usage();
      run.cache_prepare_seconds += std::chrono::duration<double>(qwen35::Clock::now() - begin).count();
    };
    model.freeze_shape_plan();
    if (warmup_planned_shapes) {
      const auto begin = qwen35::Clock::now();
      const auto shapes = model.prepare_planned_execution();
      std::cerr << "shape_warmup policy=planned shapes=" << shapes
                << " seconds=" << std::chrono::duration<double>(qwen35::Clock::now() - begin).count() << '\n';
    }
    // Prefix mode admits at the timed request boundary, before any initial
    // state allocation. A full hit must not create or evaluate an empty state.
    if (scheduled && !prefix_enabled) {
      // Exact nominal buckets are known before admission. Suffix alternatives
      // stay cold metadata until an actual prefix hit selects one.
      if (!multiple_schedules) model.prepare_execution(runs.front().chunks);
      else for (const auto &wave : waves) {
        std::vector<std::size_t> rows;
        rows.reserve(wave.calls.size());
        for (const auto &call : wave.calls) rows.push_back(call.query_rows);
        for (const auto &bucket : qwen35::benchmark::bucket_schedule_wave(wave, rows)) {
          const std::array chunks{bucket.query_rows};
          model.prepare_execution(chunks, bucket.calls.size());
        }
      }
      if (schedules.size() > cache_slots) for (const auto &run : runs)
        model.prepare_execution(run.chunks);
    }
    if (!prefix_enabled) for (std::size_t s = 0; s < (std::min)(active_slots, runs.size()); ++s) ensure_resident(s, false);
    auto session = scheduled ? std::shared_ptr<qwen35::Session>{} : model.create_session(planned_chunks);
    if (session) model.prepare_execution(planned_chunks);
    inference::BatchIndices batch_indices;
    if (scheduled) batch_indices.prepare(active_slots, active_slots);
    std::vector<std::int32_t> generated, consumed;
    if (!scheduled) { generated.reserve(steps); consumed.reserve(steps - 1); }
    const auto ready = qwen35::Clock::now();
    std::cerr << "initialized model_seconds=" << metadata_prepare_seconds +
                  std::chrono::duration<double>(constructed - model_start).count()
              << " shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
              << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
    std::cerr << "cache_limits context=" << capacity << " max_sessions=" << max_sessions
              << " active_sequences=" << active_slots << " effective_cache_sessions=" << cache_slots
              << " recurrent_payload_bytes_per_snapshot=" << recurrent_payload
              << " backend=" << model.execution.name()
              << " shared_cache_backing_bytes=" << model.known_cache_backing_bytes
              << " native_execution_private_bytes_unknown=" << model.execution.native_generated_storage()
              << " cache_budget_scope=shared_pools_and_retained_output_backing";
    if (cache_bytes == SIZE_MAX) std::cerr << " cache_requested_byte_limit=none";
    else std::cerr << " cache_requested_byte_limit=" << cache_bytes;
    std::cerr << " physical_vram_limit_known=false\n";
    if (scheduled) {
      qwen35::Clock::time_point journey_start{}, endpoint{};
      struct PreparedCall {
        std::span<const std::int32_t> ids, logical_ids;
        std::size_t before = 0, reused = 0;
        bool used_prefix = false;
        qwen35::Clock::time_point call_start{};
        double complete_seconds = 0;
        std::int32_t next = -1;
      };
      std::vector<PreparedCall> prepared;
      std::vector<std::size_t> actual_rows, members;
      std::vector<qwen35::Session *> owners;
      std::vector<qwen35::EvaluationRequest> requests;
      std::vector<qwen35::EvaluationOutput> outputs;
      std::vector<std::int32_t> next_ids;
      prepared.reserve(active_slots); actual_rows.reserve(active_slots); members.reserve(active_slots);
      owners.reserve(active_slots); requests.reserve(active_slots); outputs.reserve(active_slots); next_ids.reserve(active_slots);
      for (const auto &wave : waves) {
        const auto wave_start = qwen35::Clock::now();
        if (journey_start == qwen35::Clock::time_point{}) journey_start = wave_start;
        const auto graphs_before = executed_graph_calls, tokens_before = executed_graph_tokens;
        WaveObservation wave_observation{wave.calls.front().kind == qwen35::benchmark::PhaseKind::kPrompt ? 0 : 1};
        prepared.clear(); prepared.resize(wave.calls.size()); actual_rows.clear();
        // Resolve the whole wave against PREVIOUS publications. A full hit
        // stays pinned and charged while the other members are evaluated.
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s];
          const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase];
          auto &summary = run.summaries[call.phase];
          if (call.local == 0) {
            summary.context_before = run.visible_tokens;
            std::cerr << "memory_phase=" << summary.memory_name << "_begin\n";
            summary.start = qwen35::Clock::now();
            if (index == 0) run.start = summary.start;
          }
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j];
          work.call_start = qwen35::Clock::now();
          if (is_prompt) ensure_resident(s, prefix_enabled);
          const auto logical_ids = is_prompt ? std::span<const std::int32_t>(phase.ids)
              : std::span<const std::int32_t>(&phase.ids[call.local], 1);
          auto ids = logical_ids;
          const auto before = run.visible_tokens;
          const auto target = before + logical_ids.size();
          std::size_t reused = 0;
          bool used_prefix = false;
        if (prefix_enabled && is_prompt) {
          // An equal-length checkpoint is useful after eviction, but must not
          // replace a resident same-boundary Session and lose its recordings.
          auto hit = prefix_cache->lookup(*token_plans[s], target,
              before + (run.session ? std::size_t{1} : std::size_t{0}), &model, model.math_identity());
          if (hit) {
            if (hit.payload.snapshot->next_id >= geometry.vocabulary)
              throw std::runtime_error("prefix next_id is outside the model vocabulary");
            reused = hit.payload.snapshot->length - before;
            const auto suffix = target - hit.payload.snapshot->length;
            std::vector<std::size_t> remaining;
            remaining.reserve(run.chunks.size() - index);
            if (suffix) remaining.push_back(suffix);
            remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index + 1), run.chunks.end());
            // All fallible host/IO preparation precedes withdrawal of the old
            // Session. These are real original Model slots, not shadow state.
            auto history = qwen35::cache::copy_history(hit.payload.replay, capacity);
            const auto cursor = run.session ? run.session->head->sampling : run.saved_sampling;
            auto adopted = model.create_session(remaining, hit.payload.snapshot, qwen35::GraphRole::kText, cursor, suffix == 0);
            // This fixed-input CLI tape has one default sampling configuration.
            if (model.default_sampler.enabled && !suffix) {
              history->chunks.back().next_id = adopted->head->next_id; ++run.prefix_sampling_calls;
            }
            if (run.session) {
              (void)run.session->stop();
              const auto retired = run.session->try_retire();
              if (retired.retired) run.session.reset();
              check(retired.status);
              if (!retired.retired)
                throw std::runtime_error("prefix adoption cannot retire the original session");
            }
            run.session = std::move(adopted);
            run.history = std::move(history);
            ids = logical_ids.subspan(reused);
            hit.reset(); // No temporary Snapshot owner may hide from advance admission.
            used_prefix = true; ++run.prefix_hits; run.reused_tokens += reused;
            if (ids.empty()) ++run.prefix_full_hits;
          } else ++run.prefix_misses;
        }
        if (prefix_enabled && !run.session) {
          make_initial_room();
          std::vector<std::size_t> remaining;
          remaining.reserve(run.history->chunks.size() + run.chunks.size() - index);
          for (const auto &part : run.history->chunks) remaining.push_back(part.count);
          remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index), run.chunks.end());
          run.session = model.create_session(remaining, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute actual evaluated chunks, including imported prefixes.
          // Keep any old shared history immutable: new checkpoint metadata
          // must describe the REAL rebuilt head and its recomputed decisions.
          auto rebuilt_history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
          for (std::size_t i = 0; i < run.history->chunks.size(); ++i) {
            const auto &part = run.history->chunks[i];
            const auto restored_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_chunk" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, restored_ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != part.next_id) run.recomputed_decisions.push_back({i, part.next_id, next});
            rebuilt_history->append(part.input, part.first, part.count, next);
            ++run.restored_calls; run.restored_tokens += restored_ids.size();
            ++run.graph_participations; run.actual_graph_tokens += restored_ids.size();
            const std::array restored_members{s};
            record_batch(phase.kind, restored_ids.size(), restored_members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
          }
          if (run.session->length() != before)
            throw std::runtime_error("actual-chunk recompute differs from the retained logical boundary");
          if (run.saved_pending >= 0) {
            model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
            // The fixed-input tape retains that same CLI-wide configuration.
            if (model.default_sampler.enabled && !rebuilt_history->chunks.empty())
              rebuilt_history->chunks.back().next_id = run.saved_pending;
          }
          run.history = std::move(rebuilt_history);
        }
          work.ids = ids; work.logical_ids = logical_ids;
          work.before = before; work.reused = reused; work.used_prefix = used_prefix;
          actual_rows.push_back(ids.size());
          wave_observation.logical_tokens += logical_ids.size();
          if (wave_observation.kind != (is_prompt ? 0 : 1)) wave_observation.kind = -1;
          if (ids.empty()) {
            work.next = run.session->head->next_id;
            run.endpoint = qwen35::Clock::now();
            auto &observation = run.observations[index];
            observation.graph_executed = false; observation.graph_input_tokens = 0;
            observation.visible = run.session->length(); observation.next = work.next;
            observation.scheduled_progress = progress;
          }
        }
        // Validate the entire resolved wave before any bucket can submit.
        // A later incompatible member must not turn an old preflight error
        // into a partially committed wave.
        if (actual_rows.size() != wave.calls.size())
          throw std::runtime_error("batch wave query lengths do not match its calls");
        for (std::size_t j = 0; j < wave.calls.size(); ++j)
          if (actual_rows[j] > wave.calls[j].query_rows)
            throw std::runtime_error("batch wave actual query exceeds its logical input");
        inference::dispatch_batches(batch_indices, wave.calls.size(),
            [&](std::size_t index) { return actual_rows[index] != 0; },
            [&](std::size_t) -> std::size_t { return active_slots; },
            [&](std::size_t first, std::size_t j) {
              return wave.calls[j].kind == wave.calls[first].kind && actual_rows[j] == actual_rows[first];
            }, [&](std::span<const std::size_t> selected) -> std::size_t {
              const auto kind = wave.calls[selected.front()].kind;
              const auto query_rows = actual_rows[selected.front()];
              requests.clear(); outputs.clear(); owners.clear(); members.clear(); next_ids.resize(selected.size());
              for (const auto j : selected) {
                const auto &call = wave.calls[j]; auto &run = runs[call.session]; auto &work = prepared[j];
                requests.push_back({run.session, work.ids, nullptr, {}});
                owners.push_back(run.session.get()); members.push_back(call.session);
                outputs.push_back(qwen35::evaluation_output(model, run.session,
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}},
                    &work.complete_seconds, &run.endpoint, &run.observations[call.call],
                    preparation_diagnostics ? &run.preparation[call.call] : nullptr));
              }
              const auto batch_start = qwen35::Clock::now();
              make_batch_room(owners);
              inference::PreparationObservation preparation;
              model.prepare_evaluations(requests, preparation_diagnostics ? &preparation : nullptr);
              if (preparation_diagnostics) for (const auto j : selected)
                runs[wave.calls[j].session].preparation[wave.calls[j].call] = preparation;
              model.submit_evaluations();
              model.finish_evaluations(outputs, next_ids, [&](std::size_t at, const qwen35::CompletedEvaluationView &view) {
                const auto &call = wave.calls[selected[at]]; const auto &run = runs[call.session];
                qwen35::observe_completed_evaluation(model, outputs[at],
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}}, view);
              });
              const auto batch_endpoint = runs[members.front()].endpoint;
              record_batch(kind, query_rows, members, false,
                  std::chrono::duration<double>(batch_endpoint - batch_start).count());
              for (std::size_t at = 0; at < selected.size(); ++at) {
                const auto j = selected[at]; const auto &call = wave.calls[j];
                auto &run = runs[call.session]; auto &work = prepared[j]; work.next = next_ids[at];
                ++run.graph_participations; run.actual_graph_tokens += work.ids.size();
                if (prefix_enabled) run.history->append(token_plans[call.session], work.before + work.reused, work.ids.size(), work.next);
              }
              // No temporary strong Session owner may retain an old head across
              // next-wave adoption or eviction outside the charged directory.
              requests.clear(); outputs.clear();
              if (model.snapshot_versions.live > cache_slots)
                throw std::runtime_error("committed batch versions exceed the finite cache budget");
              note_committed_versions();
              return selected.size();
            });
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s]; const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase]; auto &summary = run.summaries[call.phase];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j]; auto &observation = run.observations[index];
          const auto logical_ids = work.logical_ids;
          const auto call_start = work.call_start;
          const auto next = work.next; const auto reused = work.reused; const auto used_prefix = work.used_prefix;
          // Per-session latency includes time waiting for its batch. Aggregate
          // throughput uses the wave wall once, never the sum of these spans.
          auto complete_seconds = std::chrono::duration<double>(run.endpoint - call_start).count();
          observation.step_seconds = complete_seconds;
          if (prefix_enabled) {
            observation.input_tokens = logical_ids.size();
            observation.prefix_hit = used_prefix; observation.reused_tokens = reused;
            if (model.snapshot_versions.live > cache_slots)
              throw std::runtime_error("committed prefix/session versions exceed the finite cache budget");
            if (!observation.graph_executed) {
              observation.seconds = complete_seconds;
              const auto write_hit_trace = [&](const std::filesystem::path &base) {
                if (base.empty()) return;
                const auto directory = base / observation.name;
                std::filesystem::create_directories(directory);
                std::ofstream output(directory / "cache-hit.json");
                output << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << (sampler.enabled ? 1 : 0)
                       << ",\"graph_calls\":" << (sampler.enabled ? 1 : 0) << ",\"tensor_payload_available\":false,\"visible_tokens\":"
                       << run.session->length() << ",\"next_id\":" << next << "}\n";
                if (!output) throw std::runtime_error("cannot write prefix-hit metadata trace");
              };
              write_hit_trace(run.trace);
              if (!layer0_probe.empty()) write_hit_trace(multiple_schedules ? layer0_probe / run.trace_subdirectory : layer0_probe);
            }
          }
          note_committed_versions();
          endpoint = (std::max)(endpoint, run.endpoint);
          run.visible_tokens = run.session->length();
          summary.step_sum_seconds += complete_seconds; run.step_sum_seconds += complete_seconds;
          run.generated.push_back(next);
          run.saved_sampling = run.session->head->sampling; run.saved_pending = next;
          if (!is_prompt) run.consumed.push_back(phase.ids[call.local]);
          if (index == 0) run.first_token_elapsed_seconds = std::chrono::duration<double>(run.endpoint - startup).count();
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 == count) {
            summary.context_after = run.visible_tokens;
            summary.seconds = std::chrono::duration<double>(run.endpoint - summary.start).count();
            std::cerr << "memory_phase=" << summary.memory_name << "_end\n";
            // Terminal prompts are real completed requests too.
            if (!is_prompt || call.phase + 1 == schedules[s].phases.size()) run.pin.reset();
          }
        }
        // Publication is a wave boundary. Same-wave calls never consume each
        // other's new checkpoint, matching the dependency and cold shape plan.
        if (prefix_enabled) for (const auto &call : wave.calls) {
          const auto &phase = schedules[call.session].phases[call.phase];
          const auto count = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 != count) continue;
          auto &run = runs[call.session];
          const qwen35::cache::ReplayPrefix history{run.history, run.history->chunks.size(), run.history->length};
          (void)prefix_cache->publish(token_plans[call.session], run.session->checkpoint(), history);
        }
        wave_observation.outputs = wave.calls.size();
        wave_observation.actual_tokens = executed_graph_tokens - tokens_before;
        wave_observation.graph_calls = executed_graph_calls - graphs_before;
        endpoint = qwen35::Clock::now(); // Include all result/checkpoint publication once.
        wave_observation.seconds = std::chrono::duration<double>(endpoint - wave_start).count();
        wave_observations.push_back(wave_observation);
      }
      for (std::size_t s = 0; s < runs.size(); ++s) {
        const auto &run = runs[s];
        if (multiple_schedules) std::cerr << "session_steps id=" << schedules[s].id << '\n';
        for (const auto &observation : run.observations) hyle::engine::reporting::write_step_observation(std::cerr, observation);
        if (preparation_diagnostics) for (std::size_t i = 0; i < run.observations.size(); ++i)
          if (run.observations[i].graph_executed)
            hyle::engine::reporting::write_preparation_observation(std::cerr, run.observations[i], run.preparation[i]);
      }
      // Define after the execution observations. Call once in the single
      // schedule's top-level object, or once in the multi-schedule top-level
      // object; do not call for each nested Session. Existing output sets
      // setprecision(17). Session IDs were validated as ASCII [A-Za-z0-9_-].
      const auto print_execution = [&] {
        std::cout << ",\"actual_graph_calls\":" << executed_graph_calls + model.frontier_sampling_graph_calls
                  << ",\"actual_graph_tokens\":" << executed_graph_tokens
                  << ",\"max_batch_requests\":" << max_batch_requests
                  << ",\"batches\":[";
        for (std::size_t i = 0; i < batches.size(); ++i) {
          const auto &batch = batches[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (batch.kind == qwen35::benchmark::PhaseKind::kPrompt ? "prefill" : "decode")
                    << "\",\"restore\":" << (batch.restore ? "true" : "false")
                    << ",\"query_rows\":" << batch.tokens
                    << ",\"requests\":" << batch.sessions.size()
                    << ",\"seconds\":" << batch.seconds << ",\"session_ids\":[";
          for (std::size_t j = 0; j < batch.sessions.size(); ++j)
            std::cout << (j ? "," : "") << '"' << schedules[batch.sessions[j]].id << '"';
          std::cout << "]}";
        }
        std::cout << "],\"waves\":[";
        for (std::size_t i = 0; i < wave_observations.size(); ++i) {
          const auto &wave = wave_observations[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (wave.kind == 0 ? "prefill" : wave.kind == 1 ? "decode" : "mixed")
                    << "\",\"outputs\":" << wave.outputs
                    << ",\"logical_tokens\":" << wave.logical_tokens
                    << ",\"actual_tokens\":" << wave.actual_tokens
                    << ",\"graph_calls\":" << wave.graph_calls
                    << ",\"seconds\":" << wave.seconds << '}';
        }
        std::cout << ']';
      };
      const auto print_result = [&](std::size_t s) {
        const auto &run = runs[s];
        const auto &phases = schedules[s].phases;
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << ",\"visible_tokens\":"
                  << run.visible_tokens << ",\"generated_ids\":[";
        for (std::size_t i = 0; i < run.generated.size(); ++i) std::cout << (i ? "," : "") << run.generated[i];
        std::cout << "],\"fixed_decode\":true,\"decode_input_ids\":[";
        for (std::size_t i = 0; i < run.consumed.size(); ++i) std::cout << (i ? "," : "") << run.consumed[i];
        std::cout << "],\"schedule\":true,\"phase_count\":" << phases.size()
                  << ",\"turn_count\":" << phases.back().turn
                  << ",\"processed_tokens\":" << run.visible_tokens
                  << ",\"generated_tokens\":" << run.generated.size()
                  << ",\"session_seconds\":" << std::chrono::duration<double>(run.endpoint - run.start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(run.endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << run.step_sum_seconds
                  << ",\"cache_hits\":" << run.cache_hits << ",\"cache_misses\":" << run.cache_misses
                  << ",\"restored_calls\":" << run.restored_calls << ",\"restored_tokens\":" << run.restored_tokens
                  << ",\"restore_contract\":\"recompute_original_inputs\""
                  << ",\"cache_prepare_seconds\":" << run.cache_prepare_seconds
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_hits\":" << run.prefix_hits << ",\"prefix_misses\":" << run.prefix_misses
                  << ",\"prefix_full_hits\":" << run.prefix_full_hits << ",\"reused_tokens\":" << run.reused_tokens
                  << ",\"graph_participations\":" << run.graph_participations + run.prefix_sampling_calls
                  << ",\"prefix_sampling_graph_calls\":" << run.prefix_sampling_calls
                  << ",\"sampling_stream\":" << run.saved_sampling.stream
                  << ",\"sampling_draw_cursor\":" << run.saved_sampling.draw
                  << ",\"graph_input_tokens\":" << run.actual_graph_tokens
                  << hyle::engine::reporting::qwen35_cache_report(model)
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false");
        std::cout << ",\"recomputed_decision_changes\":[";
        for (std::size_t i = 0; i < run.recomputed_decisions.size(); ++i) {
          const auto &change = run.recomputed_decisions[i];
          std::cout << (i ? "," : "") << "{\"chunk\":" << change.chunk
                    << ",\"previous\":" << change.previous << ",\"actual\":" << change.actual << '}';
        }
        std::cout << ']';
        if (multiple_schedules) std::cout << ",\"session_id\":\"" << schedules[s].id
            << "\",\"trace_subdirectory\":\"" << run.trace_subdirectory << '"';
        if (!multiple_schedules) print_execution();
        std::cout << ",\"phases\":[";
        for (std::size_t i = 0; i < phases.size(); ++i) {
          const auto &phase = phases[i]; const auto &summary = run.summaries[i];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          std::cout << (i ? "," : "") << "{\"turn\":" << phase.turn
                    << ",\"kind\":\"" << (is_prompt ? "prefill" : "decode") << "\",\"name\":\"" << summary.name
                    << "\",\"input_tokens\":" << phase.ids.size() << ",\"calls\":" << count << ",\"generated_tokens\":" << count
                    << ",\"context_before\":" << summary.context_before << ",\"context_after\":" << summary.context_after
                    << ",\"seconds\":" << summary.seconds << ",\"step_sum_seconds\":" << summary.step_sum_seconds << '}';
        }
        std::cout << "]}";
      };
      std::cerr << "first_token_elapsed_seconds=" << runs.front().first_token_elapsed_seconds << '\n';
      std::cout << std::setprecision(17);
      if (!multiple_schedules) print_result(0);
      else {
        std::size_t processed = 0, graph_participations = 0;
        std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
        double step_sum = 0;
        for (const auto &run : runs) {
          processed += run.visible_tokens; step_sum += run.step_sum_seconds;
          graph_participations += run.graph_participations + run.prefix_sampling_calls;
          prefix_hits += run.prefix_hits; prefix_misses += run.prefix_misses;
          prefix_full_hits += run.prefix_full_hits; reused_tokens += run.reused_tokens;
        }
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << ",\"session_count\":" << runs.size()
                  << ",\"execution\":\"equal_T_batched\",\"processed_tokens\":" << processed
                  << ",\"generated_tokens\":" << all_calls
                  << ",\"graph_participations\":" << graph_participations
                  << ",\"prefix_hits\":" << prefix_hits << ",\"prefix_misses\":" << prefix_misses
                  << ",\"prefix_full_hits\":" << prefix_full_hits << ",\"reused_tokens\":" << reused_tokens
                  << ",\"interleaved_seconds\":" << std::chrono::duration<double>(endpoint - journey_start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << step_sum
                  << ",\"max_cached_sessions\":" << max_sessions << ",\"active_sequences\":" << active_slots
                  << ",\"effective_cache_sessions\":" << cache_slots
                  << ",\"cache_recurrent_payload_bytes_per_snapshot\":" << recurrent_payload
                  << hyle::engine::reporting::qwen35_cache_report(model)
                  << ",\"cache_page_rows\":" << qwen35::cache::Plan::page_rows
                  << ",\"cache_max_physical_pages\":" << selected_cache.max_pages
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_retained_sessions\":" << resident_cache.retained_sessions()
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_entry_capacity\":" << (prefix_enabled ? cache_slots : 0)
                  << ",\"prefix_entries\":" << (prefix_cache ? prefix_cache->size() : 0)
                  << ",\"prefix_evictions\":" << (prefix_cache ? prefix_cache->evictions() : 0)
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false")
                  << ",\"logical_dispatch_session_ids\":[";
        for (std::size_t i = 0; i < dispatch.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[dispatch[i].first].id << '"';
        std::cout << "],\"evicted_session_ids\":[";
        for (std::size_t i = 0; i < evicted_sessions.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[evicted_sessions[i]].id << '"';
        std::cout << ']';
        print_execution();
        std::cout << ",\"sessions\":[";
        for (std::size_t s = 0; s < runs.size(); ++s) { if (s) std::cout << ','; print_result(s); }
        std::cout << "]}";
      }
      std::cout << '\n';
      return 0;
    }
    std::cerr << "measure_memory name=ready_before_first\nmemory_phase=generation_prefill_begin\n";
    double prefill_step_seconds = 0, continuation_step_seconds = 0, continuation_seconds = 0, decode_step_sum_seconds = 0;
    qwen35::Clock::time_point prefill_endpoint{};
    const auto prefill_start = qwen35::Clock::now();
    auto next = qwen35::evaluate(model, *session, prompt, "prefill", trace, trace_states,
        &prefill_step_seconds, &prefill_endpoint);
    auto endpoint = prefill_endpoint;
    std::cerr << "first_token_elapsed_seconds=" << std::chrono::duration<double>(prefill_endpoint - startup).count() << '\n';
    std::cerr << "memory_phase=generation_prefill_end\nmemory_phase=generation_decode_begin\n";
    if (!extra.empty()) {
      const auto continuation_start = qwen35::Clock::now();
      next = qwen35::evaluate(model, *session, extra, "continuation", trace, trace_states,
          &continuation_step_seconds, &endpoint);
      continuation_seconds = std::chrono::duration<double>(endpoint - continuation_start).count();
    }
    generated.push_back(next);
    for (std::size_t step = 1; step < steps && (!forced.empty() || !qwen35::is_end_of_generation(next)); ++step) {
      const auto previous = forced.empty() ? next : forced[step - 1];
      double step_seconds = 0;
      next = qwen35::evaluate(model, *session, std::span(&previous, 1), "decode" + std::to_string(step), trace, trace_states,
          &step_seconds, &endpoint);
      decode_step_sum_seconds += step_seconds;
      consumed.push_back(previous); generated.push_back(next);
    }
    std::cerr << "memory_phase=generation_decode_end\n";
    const auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
    const auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
    const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
    const auto graph_calls = 1 + std::size_t(!extra.empty()) + consumed.size();
    std::cout << std::setprecision(17)
              << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
              << "\",\"mtp\":false" << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << hyle::engine::reporting::qwen35_cache_report(model)
              << ",\"sampling_stream\":" << session->head->sampling.stream
              << ",\"sampling_draw_cursor\":" << session->head->sampling.draw
              << ",\"visible_tokens\":" << session->length() << ",\"generated_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
    std::cout << "],\"fixed_decode\":" << (forced.empty() ? "false" : "true") << ",\"decode_input_ids\":[";
    for (std::size_t i = 0; i < consumed.size(); ++i) std::cout << (i ? "," : "") << consumed[i];
    std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
              << ",\"session_seconds\":" << session_seconds
              << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
              << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
              << ",\"prefill_step_seconds\":" << prefill_step_seconds
              << ",\"continuation_seconds\":" << continuation_seconds << ",\"continuation_step_seconds\":" << continuation_step_seconds
              << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
              << ",\"step_sum_seconds\":" << prefill_step_seconds + continuation_step_seconds + decode_step_sum_seconds
              << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? prompt.size() / prefill_seconds : 0)
              << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? consumed.size() / decode_seconds : 0)
              << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
              << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
              << ",\"generated_decode_tokens\":" << consumed.size() << ",\"actual_graph_calls\":" << graph_calls
              << ",\"prefill_graph_calls\":1,\"decode_graph_calls\":" << consumed.size()
              << ",\"continuation_graph_calls\":" << std::size_t(!extra.empty())
              << ",\"prefill_target_rows\":" << prompt.size() << ",\"prefill_draft_rows\":0"
              << ",\"continuation_target_rows\":" << extra.size() << ",\"verified_target_rows\":" << consumed.size()
              << ",\"actual_target_rows\":" << prompt.size() + extra.size() + consumed.size() << ",\"actual_draft_rows\":0"
              << ",\"committed_target_rows\":" << session->length()
              << ",\"committed_target_decode_rows\":" << consumed.size()
              << ",\"draft_rows\":0,\"draft_rounds\":0,\"agreements\":0,\"consumed_drafts\":0"
              << ",\"decode_seconds_includes_continuation\":" << (extra.empty() ? "false" : "true")
              << ",\"diagnostic\":" << (!trace.empty() || !layer0_probe.empty() ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen3.5 text consumer: " << error.what() << '\n'; return 1;
  }
}
