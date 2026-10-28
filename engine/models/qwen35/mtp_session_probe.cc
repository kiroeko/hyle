// Explicit single-Model MTP isolation consumer; no throughput/quality claim.
#include "main.cc"
#include "probe_geometry.h"

#include <bit>
#include <condition_variable>
#include <cstring>
#include <mutex>

namespace {
namespace q = qwen35;
namespace c = q::c;
using Bytes = std::vector<std::byte>;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
std::string quoted(std::string_view value) {
  std::ostringstream out; out << '"';
  for (const auto ch : value) {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch == '\n') out << "\\n";
    else if (static_cast<unsigned char>(ch) < 32) out << '?';
    else out << ch;
  }
  out << '"'; return out.str();
}
Bytes read(c::ReadLease const &lease) {
  if (!lease.valid()) return {};
  Bytes bytes(lease.size()); q::check(q::take(lease.view()).read(bytes)); return bytes;
}
Bytes read(q::Tensor &tensor) {
  Bytes bytes(tensor.buffer.size()); q::check(tensor.buffer.view().read(bytes)); return bytes;
}
void write(const std::filesystem::path &path, const Bytes &bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(out), "cannot write MTP Session probe vector");
}
Bytes read_file(const std::filesystem::path &path, std::size_t size) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(in && in.tellg() == static_cast<std::streamoff>(size), "baseline vector extent differs");
  in.seekg(0); Bytes bytes(size);
  in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
  require(static_cast<bool>(in), "cannot read MTP Session baseline"); return bytes;
}
std::string fingerprint(const Bytes &bytes) {
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto value : bytes) { hash ^= std::to_integer<unsigned char>(value); hash *= 1099511628211ull; }
  std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}
struct Blob final { std::string name; bool bf16 = false; Bytes bytes; };

// Read exactly the committed logical rows, independent of physical page IDs.
// No capacity/whole-bank view is used as permission for an unused tail.
Bytes logical_rows(const c::PagedArena &arena, const c::PagedArenaFrame &frame, std::size_t rows) {
  const auto view = q::take(frame.view()); const auto host = view.layout().host();
  const auto count = (rows + q::cache::Plan::page_rows - 1) / q::cache::Plan::page_rows;
  require(frame.valid() && host.page_indices.size() == count && frame.page_count() == count &&
          host.last_page_lengths.size() == 1, "MTP frame has the wrong committed page footprint");
  require(!count || (count - 1) * q::cache::Plan::page_rows +
          static_cast<std::size_t>(host.last_page_lengths[0]) == rows, "MTP frame frontier differs from its snapshot");
  require(arena.owns(view), "MTP frame belongs to another arena");
  Bytes result(q::cache::multiply(rows, std::size_t{1024}));
  std::size_t offset = 0;
  for (std::size_t logical = 0; logical < count; ++logical) {
    const auto selected = q::take(view.page_prefix(static_cast<std::uint32_t>(host.page_indices[logical])));
    const auto bytes = (std::min)(std::size_t{16 * 1024}, result.size() - offset);
    q::check(q::take(q::take(selected.view()).subview(0, bytes)).read(std::span(result).subspan(offset, bytes)));
    offset += bytes;
  }
  return result;
}
struct HeadImage final {
  std::string metadata;
  std::vector<std::uint32_t> target_pages, draft_pages;
  std::vector<Blob> blobs;
};
HeadImage image(q::Model &model, const q::cache::State &head) {
  model.validate_mtp_snapshot(head);
  require(head.draft && head.carry_hidden.valid(), "incomplete MTP head");
  HeadImage result;
  std::ostringstream metadata;
  metadata << "{\"length\":" << head.length << ",\"next_id\":" << head.next_id
    << ",\"draft_frontier\":" << head.draft->frontier << ",\"lag\":" << head.draft->lag
    << ",\"catchup_token\":" << head.draft->catchup_token << ",\"carry_bytes\":" << head.carry_hidden.size()
    << ",\"prior_bytes\":" << head.prior_hidden.size() << '}';
  result.metadata = metadata.str();
  for (const auto &page : head.pages) result.target_pages.push_back(page.page());
  for (const auto &page : head.draft->pages) result.draft_pages.push_back(page.page());
  for (std::size_t layer = 0; layer < head.state.size(); ++layer) if (layer % 4 != 3) {
    result.blobs.push_back({"layer" + std::to_string(layer) + ".conv.bf16", true, read(head.state[layer].history)});
    result.blobs.push_back({"layer" + std::to_string(layer) + ".gdn.f32", false, read(head.state[layer].recurrent)});
  }
  for (std::size_t layer = 0; layer < head.keys.size(); ++layer) {
    result.blobs.push_back({"target" + std::to_string(layer) + ".key.bf16", true,
        logical_rows(model.page_inventory.keys[layer], head.keys[layer], head.length)});
    result.blobs.push_back({"target" + std::to_string(layer) + ".value.bf16", true,
        logical_rows(model.page_inventory.values[layer], head.values[layer], head.length)});
  }
  result.blobs.push_back({"draft.key.bf16", true, logical_rows(model.draft_inventory.keys[0], head.draft->key, head.draft->frontier)});
  result.blobs.push_back({"draft.value.bf16", true, logical_rows(model.draft_inventory.values[0], head.draft->value, head.draft->frontier)});
  result.blobs.push_back({"carry.bf16", true, read(head.carry_hidden)});
  result.blobs.push_back({"prior.bf16", true, read(head.prior_hidden)});
  return result;
}
bool same(const HeadImage &a, const HeadImage &b) {
  if (a.metadata != b.metadata || a.target_pages != b.target_pages || a.draft_pages != b.draft_pages || a.blobs.size() != b.blobs.size()) return false;
  for (std::size_t i = 0; i < a.blobs.size(); ++i)
    if (a.blobs[i].name != b.blobs[i].name || a.blobs[i].bytes != b.blobs[i].bytes) return false;
  return true;
}

struct VectorRecord final { std::string name, path, hash; bool bf16 = false; std::size_t bytes = 0; };
struct SavedStep final { std::string metadata; std::vector<VectorRecord> vectors; };
struct Metric final {
  std::string name, baseline, actual, dtype, baseline_hash, actual_hash;
  std::size_t bytes = 0, nonfinite_baseline = 0, nonfinite_actual = 0;
  bool byte_equal = false;
  double max_abs = 0, squared_error = 0, reference_squared = 0;
};
Metric compare(const VectorRecord &reference, const Blob &actual, const std::filesystem::path &path) {
  require(reference.name == actual.name && reference.bf16 == actual.bf16 && reference.bytes == actual.bytes.size(), "corresponding MTP vector shape/type differs");
  const auto baseline = read_file(reference.path, reference.bytes);
  Metric result{.name = actual.name, .baseline = reference.path, .dtype = actual.bf16 ? "BF16" : "F32",
      .baseline_hash = reference.hash, .actual_hash = fingerprint(actual.bytes), .bytes = actual.bytes.size(), .byte_equal = baseline == actual.bytes};
  // Store just one baseline payload on byte equality; differences retain the
  // actual raw vector. The observation records the exact reuse relation.
  if (!result.byte_equal) { write(path, actual.bytes); result.actual = path.string(); }
  else result.actual = reference.path;
  const auto width = actual.bf16 ? sizeof(std::uint16_t) : sizeof(float);
  require(reference.bytes % width == 0, "MTP vector has a non-scalar byte extent");
  const auto scalar = [&](const Bytes &data, std::size_t index) {
    if (actual.bf16) { std::uint16_t word = 0; std::memcpy(&word, data.data() + index * width, sizeof(word)); return std::bit_cast<float>(std::uint32_t(word) << 16); }
    float value = 0; std::memcpy(&value, data.data() + index * width, sizeof(value)); return value;
  };
  for (std::size_t i = 0; i < reference.bytes / width; ++i) {
    const double x = scalar(baseline, i), y = scalar(actual.bytes, i);
    result.nonfinite_baseline += !std::isfinite(x); result.nonfinite_actual += !std::isfinite(y);
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    const auto error = y - x; result.max_abs = (std::max)(result.max_abs, std::abs(error));
    result.squared_error += error * error; result.reference_squared += x * x;
  }
  return result;
}
struct Run final {
  std::string name;
  std::span<const std::int32_t> prompt;
  std::size_t quota = 0, step = 0;
  std::shared_ptr<q::Session> session;
  std::vector<std::int32_t> generated;
  bool done() const { return !generated.empty() && (generated.size() >= quota || q::is_end_of_generation(generated.back())); }
};
struct Probe final {
  std::filesystem::path output;
  std::array<std::vector<SavedStep>, 2> baseline;
  std::vector<Metric> metrics;
  std::vector<std::string> episodes;
  std::size_t peak_versions = 0, idle_checks = 0, after_stop_calls = 0;
  std::size_t actual_backing = 0, phases = 0, slots = 0;
  bool all_bytes = true, all_finite = true, stopped_b = false;
  void observe(q::Model &model) {
    peak_versions = (std::max)(peak_versions, std::size_t(model.snapshot_versions.live));
    require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots, "MTP Session probe exceeded finite snapshot capacity");
    actual_backing = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;
    require(actual_backing == model.cache_plan.backing_bytes, "MTP state backing is not charged exactly once");
  }
  void save_or_compare(std::size_t which, Run &run, HeadImage captured, bool reference) {
    auto &saved = baseline[which];
    const auto directory = output / ((reference ? "baseline_" : "interleaved_") + run.name) / ("step" + std::to_string(run.step));
    if (reference) {
      SavedStep step; step.metadata = captured.metadata;
      for (const auto &blob : captured.blobs) {
        const auto path = directory / blob.name; write(path, blob.bytes);
        step.vectors.push_back({blob.name, path.string(), fingerprint(blob.bytes), blob.bf16, blob.bytes.size()});
      }
      saved.push_back(std::move(step));
    } else {
      require(run.step < saved.size(), "interleaved MTP used more natural episodes than its baseline");
      const auto &step = saved[run.step];
      require(captured.metadata == step.metadata && captured.blobs.size() == step.vectors.size(), "interleaved natural MTP transition differs from its baseline");
      for (std::size_t i = 0; i < captured.blobs.size(); ++i) {
        auto metric = compare(step.vectors[i], captured.blobs[i], directory / captured.blobs[i].name);
        metric.name = run.name + "/step" + std::to_string(run.step) + "/" + metric.name;
        all_bytes = all_bytes && metric.byte_equal;
        all_finite = all_finite && !metric.nonfinite_baseline && !metric.nonfinite_actual;
        metrics.push_back(std::move(metric));
      }
    }
    ++run.step;
  }
  void report(std::string_view status, q::Model &model, std::string_view error = {}) {
    phases = model.phases.size(); slots = 0;
    for (const auto &[key, phase] : model.phases) {
      (void)key;
      for (std::size_t slot = 0; slot != phase->slot_count; ++slot) slots += phase->carriers[slot].initialized();
    }
    std::ofstream out(output / "observation.json"); require(static_cast<bool>(out), "cannot write MTP Session observation");
    out << std::setprecision(17) << "{\"status\":" << quoted(status) << ",\"error\":" << quoted(error)
      << ",\"model_objects\":1,\"weight_tensors\":335,\"capacity\":64,\"resident_limit\":2,\"active_sequences\":1"
      << ",\"baseline_prompt_rows\":[15,23],\"output_quotas\":[8,16],\"all_vectors_byte_equal\":" << (all_bytes ? "true" : "false")
      << ",\"all_finite\":" << (all_finite ? "true" : "false") << ",\"idle_head_full_byte_checks\":" << idle_checks
      << ",\"stopped_EOS_session_B\":" << (stopped_b ? "true" : "false") << ",\"A_calls_after_B_stop\":" << after_stop_calls
      << ",\"peak_live_snapshot_versions\":" << peak_versions << ",\"snapshot_slots\":" << model.cache_plan.snapshot_slots
      << ",\"state_slots\":" << model.cache_plan.state_slots << ",\"hidden_slots\":" << model.cache_plan.hidden_slots
      << ",\"managed_backing_bytes\":" << actual_backing << ",\"planned_backing_bytes\":" << model.cache_plan.backing_bytes
      << ",\"final_live_snapshot_versions\":" << model.snapshot_versions.live << ",\"actual_phases\":" << phases << ",\"actual_slots\":" << slots
      << ",\"fingerprint\":\"FNV1a64 diagnostic identity; byte equality uses complete arrays\",\"timing_comparison\":false,\"numerical_tolerance_policy\":null"
      << ",\"scope\":\"Same Model/GraphRole/partitions/N2 plan/native math and auto capture policy. Serial active R1; no MTP simultaneous R2, server isolation, or cross-engine quality claim. Physical page IDs may differ between sequential baselines and interleaved allocations; KV vectors contain exact logical committed rows. Equal actual payloads reference one saved baseline raw file.\""
      << ",\"episodes\":[";
    for (std::size_t i = 0; i < episodes.size(); ++i) out << (i ? "," : "") << episodes[i];
    out << "],\"baseline\":[";
    for (std::size_t r = 0; r < baseline.size(); ++r) {
      out << (r ? "," : "") << '[';
      for (std::size_t s = 0; s < baseline[r].size(); ++s) {
        const auto &step = baseline[r][s]; out << (s ? "," : "") << "{\"metadata\":" << step.metadata << ",\"vectors\":[";
        for (std::size_t i = 0; i < step.vectors.size(); ++i) { const auto &v = step.vectors[i];
          out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"path\":" << quoted(v.path) << ",\"bytes\":" << v.bytes
              << ",\"dtype\":" << quoted(v.bf16 ? "BF16" : "F32") << ",\"fnv1a64\":" << quoted(v.hash) << '}'; }
        out << "]}";
      }
      out << ']';
    }
    out << "],\"vectors\":[";
    for (std::size_t i = 0; i < metrics.size(); ++i) { const auto &v = metrics[i];
      out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"baseline\":" << quoted(v.baseline)
        << ",\"actual\":" << quoted(v.actual) << ",\"dtype\":" << quoted(v.dtype) << ",\"bytes\":" << v.bytes
        << ",\"byte_equal\":" << (v.byte_equal ? "true" : "false") << ",\"actual_reuses_baseline_payload\":" << (v.byte_equal ? "true" : "false")
        << ",\"baseline_fnv1a64\":" << quoted(v.baseline_hash) << ",\"actual_fnv1a64\":" << quoted(v.actual_hash)
        << ",\"nonfinite_baseline\":" << v.nonfinite_baseline << ",\"nonfinite_actual\":" << v.nonfinite_actual
        << ",\"max_abs\":" << v.max_abs << ",\"squared_error\":" << v.squared_error << ",\"reference_squared\":" << v.reference_squared << '}';
    }
    out << "]}\n"; require(static_cast<bool>(out), "MTP Session observation write failed");
  }
};

void retire(Run &run) {
  q::check(run.session->stop()); const auto observed = run.session->try_retire(); q::check(observed.status);
  require(observed.retired, "completed MTP Session failed actual retirement");
  run.session.reset();
}
void advance(q::Model &model, Probe &probe, Run &run, std::size_t which, bool reference) {
  const bool prefill = run.generated.empty();
  require(prefill || !run.done(), "MTP probe attempted to advance an ended Session");
  const auto before = run.session->length();
  const auto quota = run.quota - run.generated.size();
  const auto lag = run.session->head->draft->lag;
  const bool single = !prefill && (quota == 1 || model.capacity - before == 1);
  const auto role = prefill ? q::GraphRole::kMtpPrefill : single
      ? (lag ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpSingle)
      : (lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound);
  const auto pending = run.session->head->next_id;
  const auto inputs = prefill ? run.prompt : std::span(&pending, 1);
  model.prepare_evaluation(*run.session, inputs, nullptr, role); probe.observe(model);
  require(model.active_step && run.session->pending, "original MTP episode was not admitted");
  auto *phase = run.session->pending->phase;
  hyle::inference::StepObservation observation;
  model.submit_evaluation(*run.session);
  q::MtpRoundResult round;
  const auto name = run.name + (prefill ? "_prefill" : "_round" + std::to_string(run.step));
  if (prefill) run.generated.push_back(q::finish_evaluation(model, *run.session, name, {}, false, nullptr, nullptr, &observation));
  else {
    const q::EvaluationIO output_io{.name = name, .trace_states = false};
    const auto output = q::evaluation_output(model, run.session, output_io, nullptr, nullptr, &observation);
    round = q::finish_mtp_round(model, *run.session, pending, quota, q::model_eos_tokens, &output, output_io);
    run.generated.insert(run.generated.end(), round.emitted.begin(), round.emitted.end());
  }
  probe.observe(model);
  require(!model.active_step && !model.carrier_borrowed && !run.session->pending &&
          model.snapshot_versions.live <= model.cache_plan.resident_sessions, "MTP episode retained a candidate/carrier after finish");
  auto captured = image(model, *run.session->head);
  captured.blobs.push_back({"output.logits.f32", false, read(phase->io[q::Model::slot_index].logits)});
  captured.blobs.push_back({"output.hidden.bf16", true, read(phase->io[q::Model::slot_index].final_hidden)});
  std::ostringstream meta;
  meta << "{\"head\":" << captured.metadata << ",\"role\":" << static_cast<int>(role)
    << ",\"input_rows\":" << inputs.size() << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < run.generated.size(); ++i) meta << (i ? "," : "") << run.generated[i];
  meta << "],\"pending\":" << round.pending << ",\"proposal\":" << round.draft << ",\"target_first\":" << round.target_first
    << ",\"target_second\":" << round.target_second << ",\"agreed\":" << (round.agreed ? "true" : "false")
    << ",\"accepted\":" << (round.accepted_draft ? "true" : "false") << ",\"processed_target\":" << round.processed_target
    << ",\"processed_draft\":" << round.processed_draft << '}'; captured.metadata = meta.str();
  std::ostringstream episode;
  episode << "{\"reference\":" << (reference ? "true" : "false") << ",\"session\":" << quoted(run.name) << ",\"step\":" << run.step
    << ",\"metadata\":" << captured.metadata << ",\"capture_recorded\":" << observation.capture_recorded
    << ",\"capture_reused\":" << observation.capture_reused << ",\"slot_index_after\":" << model.slot_index << '}';
  probe.episodes.push_back(episode.str());
  probe.save_or_compare(which, run, std::move(captured), reference);
}

void run(const std::filesystem::path &inputs, const std::filesystem::path &output, q::BackendSelection selection) {
  auto source = q::read_ids(inputs / "prompt.ids"); require(source.size() >= 23, "probe needs original prompt first23 IDs");
  require(!std::filesystem::exists(output), "probe output directory must be fresh"); std::filesystem::create_directories(output);
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, 64, 2, 23, q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry), 1, 2, true);
  q::Model model(std::move(weight_source), geometry, 64, plan, {}, false,
      q::ReplayPolicy::kAuto, SIZE_MAX, {false, {}, 0}, 0, selection);
  require(model.weights.size() == 335, "MTP Session probe requires complete335 weight inventory");
  Probe probe{.output = output};
  try {
    const std::array prefills{std::size_t{15},std::size_t{23}};
    const std::array pair{std::size_t{2}};
    const std::array single{std::size_t{1}};
    model.prepare_family(prefills, false, 1, q::GraphRole::kMtpPrefill);
    // Each shape/role uses the same original auto-capture policy in both
    // stages, conservatively planned before preparing the serial Model slot.
    for (const auto size : prefills) model.plan_replay_uses(size, 1, 4, q::GraphRole::kMtpPrefill);
    for (const auto role : {q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound,q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(q::drafts_token(role) ? std::span(pair) : std::span(single), false, 1, role);
      model.plan_replay_uses(q::drafts_token(role) ? 2u : 1u, 1, 48, role);
    }
    probe.observe(model);
    const auto make_run = [&](std::size_t which) {
      const auto size = prefills[which]; const std::array calls{size};
      return Run{.name = which ? "B" : "A", .prompt = std::span(source).first(size), .quota = which ? 16u : 8u,
          .session = model.create_session(calls, {}, q::GraphRole::kMtpPrefill)};
    };
    for (std::size_t which = 0; which < 2; ++which) {
      auto baseline = make_run(which);
      do { advance(model, probe, baseline, which, true); } while (!baseline.done());
      retire(baseline); require(model.snapshot_versions.live == 0, "sequential baseline retained device state");
    }
    std::array runs{make_run(0), make_run(1)};
    for (;;) {
      bool progress = false;
      for (std::size_t which = 0; which < runs.size(); ++which) {
        auto &current = runs[which]; auto &other = runs[1 - which];
        if (!current.session || current.done()) continue;
        const auto idle = other.session ? std::optional(image(model, *other.session->head)) : std::nullopt;
        advance(model, probe, current, which, false); progress = true;
        if (idle) { require(same(*idle, image(model, *other.session->head)), "advancing one MTP Session changed another complete head"); ++probe.idle_checks; }
        if (probe.stopped_b && which == 0) ++probe.after_stop_calls;
        if (current.done() && which == 1) {
          require(q::is_end_of_generation(current.generated.back()), "fixed P23 Session did not end at its natural EOS");
          require(runs[0].session && !runs[0].done(), "probe did not retain a live Session across early stop");
          retire(current); probe.stopped_b = true; probe.observe(model);
          require(!model.carrier_borrowed && !model.active_step, "stopping idle Session closed shared admission");
        }
      }
      if (!progress) break;
    }
    for (std::size_t which = 0; which < runs.size(); ++which) {
      require(runs[which].step == probe.baseline[which].size(), "interleaved MTP episode count differs from baseline");
      if (runs[which].session) retire(runs[which]);
    }
    probe.observe(model);
    require(probe.stopped_b && probe.after_stop_calls && probe.idle_checks && probe.all_finite &&
            model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed,
            "MTP Session isolation/finite-state/early-stop cleanup did not complete");
    probe.report(probe.all_bytes ? "PASS_BYTE_EXACT" : "PASS_STRUCTURE_NUMERICAL_DIFFERENCES_RECORDED", model);
    std::cout << "{\"status\":\"MTP_SESSION_PROBE_COMPLETED\",\"all_vectors_byte_equal\":" << (probe.all_bytes ? "true" : "false")
              << ",\"vectors\":" << probe.metrics.size() << ",\"idle_head_checks\":" << probe.idle_checks
              << ",\"after_stop_calls\":" << probe.after_stop_calls << ",\"timing_comparison\":false}\n";
  } catch (const std::exception &error) { probe.report("FAIL", model, error.what()); throw; }
}

struct PrefixCase final {
  std::string name, seed_metadata;
  std::vector<SavedStep> baseline;
};
struct PrefixObservation final {
  std::filesystem::path output;
  std::array<PrefixCase, 3> cases{{{"whole", {}, {}}, {"T1", {}, {}}, {"T8", {}, {}}}};
  std::vector<Metric> metrics;
  std::vector<std::string> episodes;
  std::size_t graph_calls = 0, peak_versions = 0, full_hits = 0, full_hit_vectors = 0, old_checks = 0;
  std::size_t peak_retained_output_bytes = 0;
  std::size_t negative_bound_banks = 0, negative_versions_before = 0, negative_versions_pending = 0, negative_versions_after = 0;
  bool all_bytes = true, all_finite = true, negative_rejected = false, same_retry_slot = false;
  std::string negative_error;
  void observe(q::Model &model) {
    peak_versions = (std::max)(peak_versions, std::size_t(model.snapshot_versions.live));
    const auto usage = model.cache_usage();
    require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots && usage.live_bytes <= usage.limit_bytes,
            "MTP prefix probe exceeded its finite snapshot or actual cache budget");
    require(model.execution.native_generated_storage() || usage.live_bytes == model.cache_plan.backing_bytes,
            "CUDA prefix state backing differs from its exact caller-pool plan");
    peak_retained_output_bytes = (std::max)(peak_retained_output_bytes,
        usage.live_bytes - model.known_cache_backing_bytes);
  }
  void compare_vectors(const SavedStep &reference, const HeadImage &actual, const std::string &label) {
    require(reference.vectors.size() == actual.blobs.size(), "prefix vector inventory differs");
    for (std::size_t i = 0; i < actual.blobs.size(); ++i) {
      auto metric = compare(reference.vectors[i], actual.blobs[i], output / "differences" / label / actual.blobs[i].name);
      metric.name = label + "/" + metric.name; all_bytes = all_bytes && metric.byte_equal;
      all_finite = all_finite && !metric.nonfinite_baseline && !metric.nonfinite_actual; metrics.push_back(std::move(metric));
    }
  }
  void record(std::size_t which, std::size_t index, bool reference, HeadImage actual, std::string metadata) {
    auto &selected = cases[which];
    const auto label = selected.name + "/step" + std::to_string(index);
    if (reference) {
      require(index == selected.baseline.size(), "prefix baseline episode index differs");
      SavedStep saved; saved.metadata = std::move(metadata);
      if (index == 0) selected.seed_metadata = actual.metadata;
      if (which && index == 0) {
        require(saved.metadata == cases[0].baseline[0].metadata, "freshP15 baseline partitions differ");
        compare_vectors(cases[0].baseline[0], actual, "baseline_" + label);
        saved.vectors = cases[0].baseline[0].vectors;
      } else for (const auto &blob : actual.blobs) {
        const auto path = output / ("baseline_" + selected.name) / ("step" + std::to_string(index)) / blob.name;
        write(path, blob.bytes); saved.vectors.push_back({blob.name,path.string(),fingerprint(blob.bytes),blob.bf16,blob.bytes.size()});
      }
      selected.baseline.push_back(std::move(saved));
    } else {
      require(index < selected.baseline.size(), "prefix branch has no matching actual partition baseline");
      if (metadata != selected.baseline[index].metadata) {
        const auto directory = output / "transition_differences" / label;
        for (const auto &blob : actual.blobs) write(directory / blob.name,blob.bytes);
        std::ofstream description(directory / "metadata.json"); description << metadata << '\n';
        require(static_cast<bool>(description), "cannot retain differing prefix transition");
      }
      require(metadata == selected.baseline[index].metadata, "prefix natural transition differs from its same-partition baseline");
      compare_vectors(selected.baseline[index], actual, label);
    }
  }
  void report(std::string_view status, q::Model &model, std::string_view error = {}) const {
    std::ofstream out(output / "observation.json"); require(static_cast<bool>(out), "cannot write prefix probe observation");
    out << std::setprecision(17) << "{\"status\":" << quoted(status) << ",\"scenario\":\"prefix\",\"error\":" << quoted(error)
      << ",\"model_objects\":1,\"weight_tensors\":335,\"capacity\":64,\"resident_limit\":2,\"active_sequences\":1"
      << ",\"actual_graph_calls\":" << graph_calls << ",\"full_hits\":" << full_hits << ",\"full_hit_head_vectors\":" << full_hit_vectors
      << ",\"old_prefix_full_byte_checks\":" << old_checks << ",\"peak_live_snapshot_versions\":" << peak_versions
      << ",\"final_live_snapshot_versions\":" << model.snapshot_versions.live
      << model.cache_report() << ",\"peak_retained_output_backing_bytes\":" << peak_retained_output_bytes
      << ",\"all_vectors_byte_equal\":" << (all_bytes ? "true" : "false") << ",\"all_finite\":" << (all_finite ? "true" : "false")
      << ",\"negative_teacher_forced_input\":248045,\"cached_pending\":198,\"negative_rejected\":" << (negative_rejected ? "true" : "false")
      << ",\"negative_error\":" << quoted(negative_error) << ",\"negative_bound_banks\":" << negative_bound_banks
      << ",\"negative_versions_before\":" << negative_versions_before << ",\"negative_versions_pending\":" << negative_versions_pending
      << ",\"negative_versions_after\":" << negative_versions_after << ",\"retry_same_carrier_slot\":" << (same_retry_slot ? "true" : "false")
      << ",\"timing_comparison\":false,\"numerical_tolerance_policy\":null"
      << ",\"scope\":\"Complete52-component prompt checkpoint, no fake Graph outputs on fullhit. Actual commits54-components. T1=P15+[source index16=248045], T8=P15+source[15:23]; each is compared with its own identical input partition, never monolithicP23. Directory and one current Session consume at most N2 distinct committed versions.\""
      << ",\"episodes\":[";
    for (std::size_t i = 0; i < episodes.size(); ++i) out << (i ? "," : "") << episodes[i];
    out << "],\"baselines\":[";
    for (std::size_t cindex = 0; cindex < cases.size(); ++cindex) {
      const auto &selected = cases[cindex]; out << (cindex ? "," : "") << "{\"case\":" << quoted(selected.name) << ",\"steps\":[";
      for (std::size_t index = 0; index < selected.baseline.size(); ++index) {
        const auto &step = selected.baseline[index]; out << (index ? "," : "") << "{\"metadata\":" << step.metadata << ",\"vectors\":[";
        for (std::size_t i = 0; i < step.vectors.size(); ++i) { const auto &v = step.vectors[i];
          out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"path\":" << quoted(v.path)
            << ",\"dtype\":" << quoted(v.bf16 ? "BF16" : "F32") << ",\"bytes\":" << v.bytes << ",\"fnv1a64\":" << quoted(v.hash) << '}'; }
        out << "]}";
      }
      out << "]}";
    }
    out << "],\"vectors\":[";
    for (std::size_t i = 0; i < metrics.size(); ++i) { const auto &v = metrics[i];
      out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"baseline\":" << quoted(v.baseline)
        << ",\"actual\":" << quoted(v.actual) << ",\"dtype\":" << quoted(v.dtype) << ",\"bytes\":" << v.bytes
        << ",\"byte_equal\":" << (v.byte_equal ? "true" : "false") << ",\"actual_reuses_baseline_payload\":" << (v.byte_equal ? "true" : "false")
        << ",\"baseline_fnv1a64\":" << quoted(v.baseline_hash) << ",\"actual_fnv1a64\":" << quoted(v.actual_hash)
        << ",\"nonfinite_baseline\":" << v.nonfinite_baseline << ",\"nonfinite_actual\":" << v.nonfinite_actual
        << ",\"max_abs\":" << v.max_abs << ",\"squared_error\":" << v.squared_error << ",\"reference_squared\":" << v.reference_squared << '}';
    }
    out << "]}\n"; require(static_cast<bool>(out), "prefix probe report failed");
  }
};

void retire_prefix_session(std::shared_ptr<q::Session> &session) {
  q::check(session->stop()); const auto observed = session->try_retire(); q::check(observed.status);
  require(observed.retired, "prefix Session did not really retire"); session.reset();
}
void prefix_commit(q::Model &model, PrefixObservation &observation, std::size_t which, std::size_t index,
                   const std::shared_ptr<q::Session> &session, std::span<const std::int32_t> inputs,
                   q::GraphRole role, bool natural, bool reference, q::g::Invocation *retry_slot = nullptr) {
  model.prepare_evaluation(*session, inputs, nullptr, role); observation.observe(model);
  auto *phase = session->pending->phase;
  if (retry_slot) { observation.same_retry_slot = session->pending->execution.run == retry_slot; require(observation.same_retry_slot, "teacher-forced retry replaced its carrier slot"); }
  model.submit_evaluation(*session); ++observation.graph_calls;
  // Completion proves the original Graph results ready. Export only actual
  // generated owners before finish takes page transactions and unbinds ports;
  // no GraphOutput handle survives to pin the Invocation across that release.
  q::check(session->pending->execution.completion.wait());
  const auto retain_output = [&](q::V port, q::Tensor &caller) -> c::ReadLease {
    if (caller.buffer) return {};
    auto result = q::take(session->pending->execution.run->result(port));
    require(q::take(result.description()) == caller.description,
            "prefix generated output description differs from its compiled port");
    auto owner = q::take(result.storage_lease());
    require(owner.size() == q::take(caller.description.required_bytes()),
            "prefix generated output lease differs from its real logical extent");
    return owner;
  };
  const auto generated_logits = retain_output(phase->logits_port, phase->io[q::Model::slot_index].logits);
  const auto generated_hidden = retain_output(phase->hidden_port, phase->io[q::Model::slot_index].final_hidden);
  hyle::inference::StepObservation step_observation;
  q::MtpRoundResult round;
  std::int32_t next = -1;
  const auto name = observation.cases[which].name + "_step" + std::to_string(index);
  if (natural) {
    const q::EvaluationIO output_io{.name = name, .trace_states = false};
    const auto output = q::evaluation_output(model, session, output_io, nullptr, nullptr, &step_observation);
    round = q::finish_mtp_round(model, *session, inputs.front(), 2, q::model_eos_tokens, &output, output_io); next = round.next_id;
  } else next = q::finish_evaluation(model, *session, name, {}, false, nullptr, nullptr, &step_observation);
  observation.observe(model);
  require(!model.active_step && !model.carrier_borrowed && !session->pending && model.snapshot_versions.live <= 2,
          "prefix commit retained a candidate or shared carrier");
  auto captured = image(model, *session->head);
  captured.blobs.push_back({"output.logits.f32",false,generated_logits ? read(generated_logits) : read(phase->io[q::Model::slot_index].logits)});
  captured.blobs.push_back({"output.hidden.bf16",true,generated_hidden ? read(generated_hidden) : read(phase->io[q::Model::slot_index].final_hidden)});
  std::ostringstream meta;
  meta << "{\"head\":" << captured.metadata << ",\"role\":" << static_cast<int>(role) << ",\"inputs\":[";
  for (std::size_t i = 0; i < inputs.size(); ++i) meta << (i ? "," : "") << inputs[i];
  meta << "],\"natural_round\":" << (natural ? "true" : "false") << ",\"next_id\":" << next
    << ",\"pending\":" << round.pending << ",\"proposal\":" << round.draft << ",\"target_first\":" << round.target_first
    << ",\"target_second\":" << round.target_second << ",\"accepted\":" << (round.accepted_draft ? "true" : "false")
    << ",\"processed_target\":" << round.processed_target << ",\"processed_draft\":" << round.processed_draft << '}';
  std::ostringstream episode;
  episode << "{\"reference\":" << (reference ? "true" : "false") << ",\"case\":" << quoted(observation.cases[which].name)
    << ",\"step\":" << index << ",\"metadata\":" << meta.str() << ",\"capture_recorded\":" << step_observation.capture_recorded
    << ",\"capture_reused\":" << step_observation.capture_reused << '}'; observation.episodes.push_back(episode.str());
  observation.record(which,index,reference,std::move(captured),meta.str());
}

void run_prefix(const std::filesystem::path &inputs, const std::filesystem::path &output, q::BackendSelection selection) {
  const auto source = q::read_ids(inputs / "prompt.ids"); require(source.size() >= 23 && source[16] == 248045, "prefix probe needs original first23 IDs");
  require(!std::filesystem::exists(output), "prefix probe output must be fresh"); std::filesystem::create_directories(output);
  const auto prompt = std::span(source).first(15), suffix8 = std::span(source).subspan(15,8), suffix1 = std::span(source).subspan(16,1);
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, 64,2,15,q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry),1,2,true);
  constexpr std::size_t cache_byte_limit = 256u * 1024u * 1024u;
  q::Model model(std::move(weight_source), geometry, 64,plan,{},false,q::ReplayPolicy::kAuto,cache_byte_limit,
      {false, {}, 0}, 0, selection);
  require(model.weights.size() == 335, "prefix probe requires335 weights");
  q::cache::PrefixCache directory(1);
  PrefixObservation observation{.output=output};
  try {
    const std::array prefills{std::size_t{15},std::size_t{8}};
    const std::array single{std::size_t{1}};
    const std::array pair{std::size_t{2}};
    model.prepare_family(prefills,false,1,q::GraphRole::kMtpPrefill);
    for (auto tokens : prefills) model.plan_replay_uses(tokens,1,24,q::GraphRole::kMtpPrefill);
    for (auto role : {q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound,q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(q::drafts_token(role) ? std::span(pair) : std::span(single),false,1,role);
      model.plan_replay_uses(q::drafts_token(role) ? 2u : 1u,1,24,role);
    }
    const std::array calls{std::size_t{15}};
    auto producer = model.create_session(calls,{},q::GraphRole::kMtpPrefill);
    model.prepare_evaluation(*producer,prompt,nullptr,q::GraphRole::kMtpPrefill); observation.observe(model);
    model.submit_evaluation(*producer); ++observation.graph_calls;
    const auto cached_next = q::finish_evaluation(model, *producer,"prefix_seed",{},false);
    require(cached_next == 198 && producer->head->draft->lag == 0 && suffix1[0] != cached_next, "fixed lag0 prefix/teacher-forced witness changed");
    const auto checkpoint = image(model,*producer->head); // CPU only, never an extra SnapshotRef.
    const auto token_plan = q::cache::token_plan(prompt,64);
    auto history = std::make_shared<q::cache::ReplayHistory>(64); history->append(token_plan,0,15,cached_next);
    require(directory.publish(token_plan,producer->checkpoint(),{history,1,15}), "real P15 checkpoint publish failed");
    retire_prefix_session(producer); require(model.snapshot_versions.live == 1, "source Session did not leave exactly one directory version");
    const auto check_old = [&]() {
      auto hit = directory.lookup(*token_plan,15,15,&model,model.math_identity());
      require(hit && same(checkpoint,image(model,*hit.payload.snapshot)), "prefix branch changed the directory's complete old head"); ++observation.old_checks;
    };
    // Same partition baselines, each independently initialized; no baseline
    // Snapshot survives retirement beside the one cache checkpoint.
    for (std::size_t which = 0; which < observation.cases.size(); ++which) {
      auto session = model.create_session(calls,{},q::GraphRole::kMtpPrefill);
      prefix_commit(model,observation,which,0,session,prompt,q::GraphRole::kMtpPrefill,false,true);
      std::size_t index = 1;
      if (which) prefix_commit(model,observation,which,index++,session,which == 1 ? suffix1 : suffix8,
          which == 1 ? q::GraphRole::kMtpSingle : q::GraphRole::kMtpPrefill,false,true);
      const auto pending = session->head->next_id;
      require(!q::is_end_of_generation(pending), "fixed partial baseline ended before its natural round");
      prefix_commit(model,observation,which,index,session,std::span(&pending,1),q::GraphRole::kMtpRound,true,true);
      retire_prefix_session(session); observation.observe(model); check_old();
      require(model.snapshot_versions.live == 1, "baseline retained device state outside its directory");
    }
    for (std::size_t which = 0; which < observation.cases.size(); ++which) {
      const auto graph_before = observation.graph_calls;
      auto hit = directory.lookup(*token_plan,15,15,&model,model.math_identity());
      require(hit && hit.payload.snapshot->next_id == cached_next, "exact whole-prompt hit unavailable");
      auto session = model.create_session({},hit.payload.snapshot,q::GraphRole::kMtpPrefill); hit.reset();
      require(std::all_of(session->tails.begin(),session->tails.end(),[](const auto &tail){return !tail.valid();}) &&
              std::all_of(session->draft_tails.begin(),session->draft_tails.end(),[](const auto &tail){return !tail.valid();}),
              "prefix adoption imported exclusive tail authority");
      const auto adopted = image(model,*session->head);
      require(adopted.metadata == observation.cases[which].seed_metadata && same(checkpoint,adopted), "full hit did not preserve complete original state");
      SavedStep reference_head = observation.cases[which].baseline[0]; reference_head.vectors.resize(52);
      observation.compare_vectors(reference_head,adopted,observation.cases[which].name+"/full_hit");
      require(observation.graph_calls == graph_before && !model.active_step && !model.carrier_borrowed,
              "whole prompt hit executed or manufactured a Graph output");
      ++observation.full_hits; observation.full_hit_vectors += adopted.blobs.size();
      std::size_t index = 1;
      q::g::Invocation *retry_slot = nullptr;
      if (which == 1) {
        const auto before = image(model,*session->head); const auto *original = session->head.get();
        const auto target_owners = model.page_inventory.owners, draft_owners = model.draft_inventory.owners;
        observation.negative_versions_before = model.snapshot_versions.live;
        model.prepare_evaluation(*session,suffix1,nullptr,q::GraphRole::kMtpSingle); observation.observe(model);
        retry_slot = session->pending->execution.run; observation.negative_versions_pending = model.snapshot_versions.live;
        observation.negative_bound_banks = static_cast<std::size_t>(std::count_if(session->pending->execution.bound.begin(),session->pending->execution.bound.end(),[](auto value) { return static_cast<bool>(value); }));
        require(observation.negative_bound_banks == 14, "teacher-forced test did not bind all14 original banks");
        model.submit_evaluation(*session); ++observation.graph_calls;
        try { (void)q::finish_mtp_round(model, *session,cached_next,2); }
        catch (const std::runtime_error &error) { observation.negative_rejected = true; observation.negative_error = error.what(); }
        observation.negative_versions_after = model.snapshot_versions.live;
        require(observation.negative_rejected && !model.active_step && !model.carrier_borrowed && !session->pending &&
                session->head.get() == original &&
                model.snapshot_versions.live == observation.negative_versions_before &&
                model.page_inventory.owners == target_owners && model.draft_inventory.owners == draft_owners &&
                same(before,image(model,*session->head)), "terminal selector rejection did not reclaim its original transaction");
        check_old();
      }
      if (which) prefix_commit(model,observation,which,index++,session,which == 1 ? suffix1 : suffix8,
          which == 1 ? q::GraphRole::kMtpSingle : q::GraphRole::kMtpPrefill,false,false,retry_slot);
      const auto pending = session->head->next_id;
      prefix_commit(model,observation,which,index,session,std::span(&pending,1),q::GraphRole::kMtpRound,true,false);
      check_old(); retire_prefix_session(session); observation.observe(model);
      require(model.snapshot_versions.live == 1, "completed cached branch retained another committed version");
    }
    require(directory.evict_one() && !directory.size(), "prefix directory did not release its final entry");
    observation.observe(model);
    require(model.snapshot_versions.live == 0 && observation.full_hits == 3 && observation.full_hit_vectors == 156 &&
            observation.graph_calls == 15 && observation.negative_rejected && observation.same_retry_slot && observation.all_finite,
            "finite prefix/selection/reclamation probe did not complete");
    require(!model.execution.native_generated_storage() || observation.peak_retained_output_bytes != 0,
            "native prefix probe did not observe actual charged generated state backing");
    observation.report(observation.all_bytes ? "PASS_BYTE_EXACT" : "PASS_STRUCTURE_NUMERICAL_DIFFERENCES_RECORDED",model);
    std::cout << "{\"status\":\"MTP_PREFIX_PROBE_COMPLETED\",\"actual_graph_calls\":" << observation.graph_calls
      << ",\"full_hits\":" << observation.full_hits << ",\"vectors\":" << observation.metrics.size()
      << ",\"all_vectors_byte_equal\":" << (observation.all_bytes ? "true" : "false") << ",\"timing_comparison\":false}\n";
  } catch (const std::exception &error) { observation.report("FAIL",model,error.what()); throw; }
}
// Recovery keeps only this CPU ledger and raw oracle. It never retains a
// Snapshot, arena permission or hidden tensor from the retired GPU head.
std::string resume_record_json(const q::GenerationCommitRecord &record) {
  const auto boundary = [](const q::GenerationBoundary &value) {
    std::ostringstream out;
    out << "{\"length\":" << value.length << ",\"frontier\":" << value.frontier
      << ",\"lag\":" << value.lag << ",\"next_id\":" << value.next_id
      << ",\"catchup_token\":" << value.catchup_token << '}'; return out.str();
  };
  std::ostringstream out;
  out << "{\"role\":" << static_cast<int>(record.role) << ",\"generation\":" << (record.generation ? "true" : "false")
    << ",\"committed\":" << (record.committed ? "true" : "false") << ",\"quota\":" << record.quota
    << ",\"turn\":" << record.turn << ",\"inputs\":[";
  const auto ids = record.ids();
  for (std::size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
  out << "],\"before\":" << boundary(record.before) << ",\"after\":" << boundary(record.after)
    << ",\"accepted_tokens\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows
    << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first
    << ",\"target_second\":" << record.target_second << ",\"agreed\":" << (record.agreed ? "true" : "false")
    << ",\"accepted_draft\":" << (record.accepted_draft ? "true" : "false")
    << ",\"stopped\":" << (record.stopped ? "true" : "false") << ",\"emitted\":[";
  for (std::size_t i = 0; i < record.emitted_count; ++i) out << (i ? "," : "") << record.emitted[i];
  out << "]}"; return out.str();
}
q::GenerationLedger resume_prefix(const q::GenerationLedger &source, std::size_t count) {
  require(count && count <= source.committed_count, "resume ledger prefix is not committed");
  q::GenerationLedger result(source.records.size());
  std::copy_n(source.records.begin(),count,result.records.begin()); result.committed_count = count; return result;
}
HeadImage resume_image(q::Model &model, const q::Session &session, const q::GenerationCommitRecord &record) {
  auto captured = image(model,*session.head);
  const auto tokens = q::drafts_token(record.role) ? std::size_t{2} : record.ids().size();
  auto &phase = *model.phases.at(q::ExecutionShape{1,tokens,record.role});
  captured.blobs.push_back({"output.logits.f32",false,read(phase.io[q::Model::slot_index].logits)});
  captured.blobs.push_back({"output.hidden.bf16",true,read(phase.io[q::Model::slot_index].final_hidden)});
  return captured;
}
struct ResumeObservation final {
  std::filesystem::path output;
  std::array<SavedStep,4> baseline;
  std::vector<Metric> metrics;
  std::vector<std::string> episodes;
  std::size_t graph_calls = 0, positive_graph_calls = 0, peak_versions = 0, actual_backing = 0;
  std::size_t zero_version_retirements = 0, mismatch_graph_calls = 0;
  bool all_bytes = true, all_finite = true, mismatch_rejected = false, trace_rejected = false;
  bool trace_commit_witness = false, correct_retry = false;
  std::string mismatch_error, trace_error, bad_expected, actual_first;
  void observe(q::Model &model) {
    peak_versions = (std::max)(peak_versions,std::size_t(model.snapshot_versions.live));
    actual_backing = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;
    require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots && actual_backing == model.cache_plan.backing_bytes,
            "resume probe exceeded its finite snapshot/backing plan");
  }
  void record(std::size_t index, HeadImage captured, bool reference, const std::string &label) {
    require(index < baseline.size() && captured.blobs.size() == 54, "resume output is not its complete54-component image");
    if (reference) {
      auto &saved = baseline[index]; saved.metadata = captured.metadata;
      for (const auto &blob : captured.blobs) {
        const auto path = output / "baseline" / ("step" + std::to_string(index)) / blob.name;
        write(path,blob.bytes); saved.vectors.push_back({blob.name,path.string(),fingerprint(blob.bytes),blob.bf16,blob.bytes.size()});
      }
      return;
    }
    const auto &saved = baseline[index];
    if (saved.metadata != captured.metadata) {
      const auto directory = output / "transition_differences" / label;
      for (const auto &blob : captured.blobs) write(directory / blob.name,blob.bytes);
      std::ofstream description(directory / "metadata.json"); description << captured.metadata << '\n';
      require(static_cast<bool>(description), "cannot retain differing resume transition");
    }
    require(saved.metadata == captured.metadata && saved.vectors.size() == captured.blobs.size(),
            "recovered MTP head differs from the same-partition continuity boundary");
    for (std::size_t i = 0; i < captured.blobs.size(); ++i) {
      auto metric = compare(saved.vectors[i],captured.blobs[i],output / "differences" / label / captured.blobs[i].name);
      metric.name = label + "/" + metric.name; all_bytes = all_bytes && metric.byte_equal;
      all_finite = all_finite && !metric.nonfinite_baseline && !metric.nonfinite_actual; metrics.push_back(std::move(metric));
    }
  }
  void report(std::string_view status, q::Model &model, const q::GenerationLedger &ledger, std::string_view error = {}) const {
    std::ofstream out(output / "observation.json"); require(static_cast<bool>(out), "cannot write resume probe observation");
    out << std::setprecision(17) << "{\"status\":" << quoted(status) << ",\"scenario\":\"resume\",\"error\":" << quoted(error)
      << ",\"model_objects\":1,\"weight_tensors\":335,\"capacity\":64,\"resident_limit\":1,\"active_sequences\":1"
      << ",\"actual_graph_calls\":" << graph_calls << ",\"positive_graph_calls\":" << positive_graph_calls
      << ",\"peak_live_snapshot_versions\":" << peak_versions << ",\"snapshot_slots\":" << model.cache_plan.snapshot_slots
      << ",\"final_live_snapshot_versions\":" << model.snapshot_versions.live << ",\"zero_version_retirements\":" << zero_version_retirements
      << ",\"managed_backing_bytes\":" << actual_backing << ",\"planned_backing_bytes\":" << model.cache_plan.backing_bytes
      << ",\"all_vectors_byte_equal\":" << (all_bytes ? "true" : "false") << ",\"all_finite\":" << (all_finite ? "true" : "false")
      << ",\"wrong_expected_rejected\":" << (mismatch_rejected ? "true" : "false") << ",\"wrong_expected_error\":" << quoted(mismatch_error)
      << ",\"wrong_expected_actual_graph_calls\":" << mismatch_graph_calls << ",\"wrong_expected\":" << (bad_expected.empty() ? "null" : bad_expected)
      << ",\"original_first_record\":" << (actual_first.empty() ? "null" : actual_first)
      << ",\"postcommit_trace_failed\":" << (trace_rejected ? "true" : "false") << ",\"postcommit_trace_error\":" << quoted(trace_error)
      << ",\"postcommit_witness_installed\":" << (trace_commit_witness ? "true" : "false") << ",\"original_ledger_retry_succeeded\":" << (correct_retry ? "true" : "false")
      << ",\"timing_comparison\":false,\"numerical_tolerance_policy\":null"
      << ",\"scope\":\"N1 complete GPU heads are actually retired before product restore. Only original CPU token/decision ledger and raw52/54 oracle survive. Replay uses original roles, input partitions, quota and freshly recomputed proposals. Physical page IDs are not compared across allocations. Final replay Graph outputs54 and final head52 are checked; intermediate replay commits are validated by the product ledger, not claimed as independently read54 images. Equal actual payloads reference one saved baseline raw file.\""
      << ",\"ledger\":[";
    for (std::size_t i = 0; i < ledger.committed_count; ++i) out << (i ? "," : "") << resume_record_json(ledger.records[i]);
    out << "],\"episodes\":[";
    for (std::size_t i = 0; i < episodes.size(); ++i) out << (i ? "," : "") << episodes[i];
    out << "],\"baseline\":[";
    for (std::size_t s = 0; s < baseline.size(); ++s) {
      const auto &step = baseline[s]; out << (s ? "," : "") << "{\"metadata\":" << (step.metadata.empty() ? "null" : step.metadata) << ",\"vectors\":[";
      for (std::size_t i = 0; i < step.vectors.size(); ++i) { const auto &v = step.vectors[i];
        out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"path\":" << quoted(v.path)
          << ",\"dtype\":" << quoted(v.bf16 ? "BF16" : "F32") << ",\"bytes\":" << v.bytes << ",\"fnv1a64\":" << quoted(v.hash) << '}'; }
      out << "]}";
    }
    out << "],\"vectors\":[";
    for (std::size_t i = 0; i < metrics.size(); ++i) { const auto &v = metrics[i];
      out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"baseline\":" << quoted(v.baseline) << ",\"actual\":" << quoted(v.actual)
        << ",\"dtype\":" << quoted(v.dtype) << ",\"bytes\":" << v.bytes << ",\"byte_equal\":" << (v.byte_equal ? "true" : "false")
        << ",\"actual_reuses_baseline_payload\":" << (v.byte_equal ? "true" : "false") << ",\"baseline_fnv1a64\":" << quoted(v.baseline_hash)
        << ",\"actual_fnv1a64\":" << quoted(v.actual_hash) << ",\"nonfinite_baseline\":" << v.nonfinite_baseline
        << ",\"nonfinite_actual\":" << v.nonfinite_actual << ",\"max_abs\":" << v.max_abs << ",\"squared_error\":" << v.squared_error
        << ",\"reference_squared\":" << v.reference_squared << '}';
    }
    out << "]}\n"; require(static_cast<bool>(out), "resume observation write failed");
  }
};
void retire_resume_session(q::Model &model, ResumeObservation &observation, std::shared_ptr<q::Session> &session) {
  retire_prefix_session(session); observation.observe(model);
  require(model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed,
          "resume retained a complete GPU head, candidate or borrowed carrier after retirement");
  ++observation.zero_version_retirements;
}
void resume_commit(q::Model &model, ResumeObservation &observation, const std::shared_ptr<q::Session> &session,
                   q::GenerationLedger &ledger, std::span<const std::int32_t> inputs, q::GraphRole role,
                   bool generation, std::size_t quota, const std::string &name, const std::filesystem::path &trace = {}) {
  q::prepare_generation_record(ledger,*session,role,inputs,generation,quota,0);
  model.prepare_evaluation(*session,inputs,nullptr,role); observation.observe(model);
  require(session->pending && std::count_if(session->pending->execution.bound.begin(),session->pending->execution.bound.end(),[](auto value) { return static_cast<bool>(value); }) == 14,
          "resume episode did not use the original14-bank transaction");
  model.submit_evaluation(*session); ++observation.graph_calls; ++observation.positive_graph_calls;
  hyle::inference::StepObservation step;
  const q::EvaluationIO selected_io{.name = name, .trace = &trace, .trace_states = false};
  const auto selected = q::evaluation_output(model, session, selected_io, nullptr, nullptr, &step, nullptr, &ledger);
  if (generation) (void)q::finish_mtp_round(model, *session,inputs.front(),quota,q::model_eos_tokens,&selected, selected_io);
  else (void)q::finish_evaluation(model, *session,name,trace,false,nullptr,nullptr,&step,nullptr,{},&ledger);
  observation.observe(model);
  require(!model.active_step && !model.carrier_borrowed && !session->pending && ledger.committed_count &&
          ledger.records[ledger.committed_count - 1].committed, "resume commit lacks its original retirement/ledger witness");
  std::ostringstream episode;
  episode << "{\"name\":" << quoted(name) << ",\"record\":" << resume_record_json(ledger.records[ledger.committed_count - 1])
    << ",\"capture_recorded\":" << step.capture_recorded << ",\"capture_reused\":" << step.capture_reused << '}';
  observation.episodes.push_back(episode.str());
}
void run_resume(const std::filesystem::path &inputs, const std::filesystem::path &output, q::BackendSelection selection) {
  const auto source = q::read_ids(inputs / "prompt.ids"); require(source.size() >= 15, "resume probe needs the original P15 prompt");
  require(!std::filesystem::exists(output), "resume output must be fresh"); std::filesystem::create_directories(output);
  const auto prompt = std::span(source).first(15);
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, 64,1,15,q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry),1,2,true);
  q::Model model(std::move(weight_source), geometry, 64,plan,{},false,
      q::ReplayPolicy::kAuto,SIZE_MAX,{false,{},0},0,selection);
  require(model.weights.size() == 335, "resume needs the complete335 Model");
  ResumeObservation observation{.output=output}; q::GenerationLedger ledger(4);
  try {
    const std::array prefills{std::size_t{15}};
    const std::array pair{std::size_t{2}};
    const std::array single{std::size_t{1}};
    model.prepare_family(prefills,false,1,q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(15,1,24,q::GraphRole::kMtpPrefill);
    for (auto role : {q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound,q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(q::drafts_token(role) ? std::span(pair) : std::span(single),false,1,role);
      model.plan_replay_uses(q::drafts_token(role) ? 2u : 1u,1,24,role);
    }
    auto session = model.create_session(prefills,{},q::GraphRole::kMtpPrefill);
    const std::array roles{q::GraphRole::kMtpPrefill,q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound,q::GraphRole::kMtpSingle};
    for (std::size_t i = 0; i < roles.size(); ++i) {
      const auto pending = session->head->next_id;
      resume_commit(model,observation,session,ledger,i ? std::span(&pending,1) : prompt,roles[i],i != 0,i == 3 ? 1u : 2u,"continuity" + std::to_string(i));
      observation.record(i,resume_image(model,*session,ledger.records[i]),true,{});
    }
    require(ledger.records[0].after == q::GenerationBoundary{15,15,0,198,-1} &&
            ledger.records[1].after == q::GenerationBoundary{17,16,1,74455,248045} && ledger.records[1].accepted_draft &&
            ledger.records[2].after == q::GenerationBoundary{18,18,0,271,-1} && !ledger.records[2].accepted_draft &&
            ledger.records[3].after.length == 19 && ledger.records[3].after.frontier == 19 && ledger.records[3].after.lag == 0,
            "fixed natural accept/lag1/reject/Single continuity changed");
    observation.actual_first = resume_record_json(ledger.records[0]);
    retire_resume_session(model,observation,session);

    auto lag1_ledger = resume_prefix(ledger,2); q::GenerationRestoreStats lag1_stats;
    session = q::restore_mtp_session(model,lag1_ledger,{},false,&lag1_stats);
    observation.graph_calls += lag1_stats.graph_calls; observation.positive_graph_calls += lag1_stats.graph_calls;
    require(lag1_stats.graph_calls == 2 && q::generation_boundary(*session->head) == ledger.records[1].after,
            "product restore did not reconstruct the lag1 boundary");
    observation.observe(model); observation.record(1,resume_image(model,*session,ledger.records[1]),false,"restore_lag1");
    const auto trace_file = output / "trace-blocker";
    { std::ofstream file(trace_file); file << "ordinary file: postcommit trace failure\n"; require(static_cast<bool>(file),"cannot prepare trace failure"); }
    const auto pending = session->head->next_id;
    try { resume_commit(model,observation,session,lag1_ledger,std::span(&pending,1),q::GraphRole::kMtpCatchupRound,true,2,"postcommit",trace_file); }
    catch (const std::filesystem::filesystem_error &error) { observation.trace_rejected = true; observation.trace_error = error.what(); }
    observation.observe(model);
    observation.trace_commit_witness = observation.trace_rejected && lag1_ledger.committed_count == 3 &&
        lag1_ledger.records[2].committed && lag1_ledger.records[2].same_result(ledger.records[2]) &&
        q::generation_boundary(*session->head) == ledger.records[2].after && !model.active_step && !model.carrier_borrowed && !session->pending;
    require(observation.trace_commit_witness,"postcommit trace failure lost or duplicated the authoritative ledger/head");
    observation.episodes.push_back("{\"name\":\"postcommit_trace_failure\",\"record\":" + resume_record_json(lag1_ledger.records[2]) + "}");
    observation.record(2,resume_image(model,*session,ledger.records[2]),false,"postcommit_trace_failure");
    retire_resume_session(model,observation,session);

    auto expected_bad = resume_prefix(lag1_ledger,3);
    expected_bad.records[0].target_first = static_cast<std::int32_t>((expected_bad.records[0].target_first + 1) % model.geometry.vocabulary);
    observation.bad_expected = resume_record_json(expected_bad.records[0]); q::GenerationRestoreStats bad_stats;
    std::shared_ptr<q::Session> unexpected;
    try { unexpected = q::restore_mtp_session(model,expected_bad,{},false,&bad_stats); }
    catch (const std::runtime_error &error) { observation.mismatch_rejected = true; observation.mismatch_error = error.what(); }
    observation.graph_calls += bad_stats.graph_calls; observation.mismatch_graph_calls = bad_stats.graph_calls;
    observation.observe(model);
    require(observation.mismatch_rejected && observation.mismatch_error.find("natural decision or state boundary diverged") != std::string::npos &&
            bad_stats.graph_calls == 1 && !unexpected && !model.active_step && !model.carrier_borrowed && model.snapshot_versions.live == 0 &&
            resume_record_json(ledger.records[0]) == observation.actual_first && !ledger.records[0].generation,
            "wrong expected metadata was not rejected before publishing a recovered head with actual retirement");

    q::GenerationRestoreStats lag0_stats;
    session = q::restore_mtp_session(model,lag1_ledger,{},false,&lag0_stats);
    observation.graph_calls += lag0_stats.graph_calls; observation.positive_graph_calls += lag0_stats.graph_calls;
    require(lag0_stats.graph_calls == 3 && q::generation_boundary(*session->head) == ledger.records[2].after,
            "correct postcommit ledger did not retry naturally to lag0");
    observation.correct_retry = true; observation.observe(model);
    observation.record(2,resume_image(model,*session,ledger.records[2]),false,"restore_lag0");
    const auto final_pending = session->head->next_id;
    resume_commit(model,observation,session,lag1_ledger,std::span(&final_pending,1),q::GraphRole::kMtpSingle,true,1,"resumed_single");
    require(lag1_ledger.committed_count == 4 && lag1_ledger.records[3].same_result(ledger.records[3]),"quota1 continuation changed its natural record");
    observation.record(3,resume_image(model,*session,ledger.records[3]),false,"resumed_single");
    retire_resume_session(model,observation,session);
    require(observation.positive_graph_calls == 11 && observation.graph_calls == 12 && observation.metrics.size() == 216 && observation.all_finite,
            "resume positive/negative graph or complete-vector work did not match the finite probe");
    observation.report(observation.all_bytes ? "PASS_BYTE_EXACT" : "PASS_STRUCTURE_NUMERICAL_DIFFERENCES_RECORDED",model,ledger);
    std::cout << "{\"status\":\"MTP_RESUME_PROBE_COMPLETED\",\"actual_graph_calls\":" << observation.graph_calls
      << ",\"positive_graph_calls\":" << observation.positive_graph_calls << ",\"vectors\":" << observation.metrics.size()
      << ",\"all_vectors_byte_equal\":" << (observation.all_bytes ? "true" : "false") << ",\"timing_comparison\":false}\n";
  } catch (const std::exception &error) { observation.observe(model); observation.report("FAIL",model,ledger,error.what()); throw; }
}
#if defined(HYLE_COMPUTE_CUDA)
struct BatchNativeGate final {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false, open = false;
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<BatchNativeGate *>(opaque);
    std::unique_lock lock(gate.mutex); gate.entered = true; gate.condition.notify_all();
    gate.condition.wait(lock,[&] { return gate.open; }); // No CUDA call on this callback.
  }
  void wait_until_entered() {
    std::unique_lock lock(mutex);
    require(condition.wait_for(lock,std::chrono::seconds(30),[&] { return entered; }),"MTP batch did not enter its real native gate");
  }
  void release() noexcept { { std::lock_guard lock(mutex); open = true; } condition.notify_all(); }
  ~BatchNativeGate() { release(); }
};
struct BatchReleaseAndDrain final {
  BatchNativeGate &gate;
  c::Context &context;
  ~BatchReleaseAndDrain() { gate.release(); context.drain().IgnoreError(); }
};
#endif
HeadImage batch_image(q::Model &model, const q::Session &session, q::Phase &phase, std::size_t member, std::size_t width) {
  require(width && member < width,"invalid MTP output member");
  auto captured = image(model,*session.head);
  const auto part = [&](q::Tensor &tensor) {
    require(tensor.buffer.size() % width == 0,"packed MTP output has unequal member extents");
    const auto size = tensor.buffer.size() / width; Bytes result(size);
    q::check(q::take(tensor.buffer.view().subview(member * size,size)).read(result)); return result;
  };
  captured.blobs.push_back({"output.logits.f32",false,part(phase.io[q::Model::slot_index].logits)});
  captured.blobs.push_back({"output.hidden.bf16",true,part(phase.io[q::Model::slot_index].final_hidden)}); return captured;
}
#if defined(HYLE_COMPUTE_CUDA)
bool batch_disjoint(const c::ReadLease &a, const c::ReadLease &b) {
  const auto x = q::take(a.view()), y = q::take(b.view());
  return x.resource() != y.resource() || x.native_handle() != y.native_handle() ||
      q::cache::add(x.offset(),x.size()) <= y.offset() || q::cache::add(y.offset(),y.size()) <= x.offset();
}
#endif
std::size_t batch_nonfinite(const Blob &blob) {
  const auto width = blob.bf16 ? sizeof(std::uint16_t) : sizeof(float);
  require(blob.bytes.size() % width == 0,"batch vector is not scalar aligned"); std::size_t count = 0;
  for (std::size_t i = 0; i < blob.bytes.size(); i += width) {
    float value = 0;
    if (blob.bf16) { std::uint16_t word = 0; std::memcpy(&word,blob.bytes.data() + i,sizeof(word)); value = std::bit_cast<float>(std::uint32_t(word) << 16); }
    else std::memcpy(&value,blob.bytes.data() + i,sizeof(value));
    count += !std::isfinite(value);
  }
  return count;
}
struct BatchImage final { std::string label; SavedStep saved; };
struct MtpBatchObservation final {
  std::filesystem::path output;
  std::vector<BatchImage> images;
  std::vector<Metric> metrics;
  std::vector<std::string> episodes, extent_changes;
  std::vector<q::GenerationRestoreStats::DecisionChange> rebuild_changes;
  std::size_t graph_calls = 0, peak_versions = 0, old_checks = 0, idle_checks = 0, backing = 0;
  std::size_t versions_before_cancel = 0, versions_blocked = 0, versions_after_cancel = 0;
  std::size_t rebuild_graphs = 0, rebuild_target_rows = 0, rebuild_draft_rows = 0, normalized_lag_rows = 0;
  std::int32_t historical_pending = -1, rebuilt_prediction = -1, external_pending = -1, external_original_prediction = -1;
  bool finite = true, cancelled = false, blocked_retained = false, dropped_completion = false, same_slot_retry = false;
  bool mixed_acceptance = false, independent_frames = false, pending_policy = false, rebuilt_exact = false;
  std::array<std::size_t,2> accepted{};
  void observe(q::Model &model) {
    peak_versions = (std::max)(peak_versions,std::size_t(model.snapshot_versions.live));
    backing = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;
    require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots && backing == model.cache_plan.backing_bytes,
            "MTP batch exceeded its finite version/backing plan");
  }
  SavedStep record(const std::string &label, HeadImage captured, const SavedStep *reference = nullptr, bool exact = false) {
    require(captured.blobs.size() == 54,"batch capture omitted a complete head or member output");
    SavedStep saved; saved.metadata = captured.metadata;
    if (exact) require(reference && reference->metadata == captured.metadata,"canonical rebuild boundary differs");
    for (const auto &blob : captured.blobs) {
      const auto count = batch_nonfinite(blob); finite = finite && !count;
      const auto path = output / label / blob.name;
      const auto found = reference ? std::find_if(reference->vectors.begin(),reference->vectors.end(),
          [&](const auto &v) { return v.name == blob.name; }) : std::vector<VectorRecord>::const_iterator{};
      if (reference && found != reference->vectors.end() && found->bf16 == blob.bf16 && found->bytes == blob.bytes.size()) {
        auto metric = compare(*found,blob,path); metric.name = label + "/" + metric.name;
        finite = finite && !metric.nonfinite_baseline && !metric.nonfinite_actual;
        if (exact) require(metric.byte_equal,"same canonical teacher-forced reconstruction differs bytewise");
        saved.vectors.push_back({blob.name,metric.actual,metric.actual_hash,blob.bf16,blob.bytes.size()}); metrics.push_back(std::move(metric));
      } else {
        require(!exact,"canonical reconstruction changed a vector extent"); write(path,blob.bytes);
        saved.vectors.push_back({blob.name,path.string(),fingerprint(blob.bytes),blob.bf16,blob.bytes.size()});
        if (reference) extent_changes.push_back(label + "/" + blob.name);
      }
    }
    images.push_back({label,saved}); return saved;
  }
  void episode(const std::string &name, const q::GenerationCommitRecord &record) {
    std::ostringstream out;
    out << "{\"name\":" << quoted(name) << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
      << ",\"record\":" << resume_record_json(record) << ",\"committed_inputs\":[";
    const auto ids = record.committed_ids();
    for (std::size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
    out << "]}"; episodes.push_back(out.str());
  }
  void report(std::string_view status, q::Model &model, std::string_view error = {}) const {
    std::ofstream out(output / "observation.json"); require(static_cast<bool>(out),"cannot write MTP batch observation");
    out << std::setprecision(17) << "{\"status\":" << quoted(status) << ",\"scenario\":\"batch\",\"error\":" << quoted(error)
      << ",\"model_objects\":1,\"weight_tensors\":335,\"capacity\":64,\"resident_limit\":4,\"active_sequences\":2"
      << ",\"actual_graph_calls\":" << graph_calls << ",\"peak_live_snapshot_versions\":" << peak_versions
      << ",\"final_live_snapshot_versions\":" << model.snapshot_versions.live << ",\"managed_backing_bytes\":" << backing
      << ",\"planned_backing_bytes\":" << model.cache_plan.backing_bytes << ",\"all_finite\":" << (finite ? "true" : "false")
      << ",\"old_head_full_byte_checks\":" << old_checks << ",\"idle_B_full_byte_checks\":" << idle_checks
      << ",\"different_member_acceptance_observed\":" << (mixed_acceptance ? "true" : "false")
      << ",\"accepted_tokens\":[" << accepted[0] << ',' << accepted[1] << "]"
      << ",\"independent_ready_frames_and_states\":" << (independent_frames ? "true" : "false")
      << ",\"cancelled_whole_batch\":" << (cancelled ? "true" : "false") << ",\"blocked_candidates_retained\":" << (blocked_retained ? "true" : "false")
      << ",\"dropped_live_completion\":" << (dropped_completion ? "true" : "false") << ",\"same_R2_carrier_slot_retry\":" << (same_slot_retry ? "true" : "false")
      << ",\"versions_before_cancel\":" << versions_before_cancel << ",\"versions_blocked\":" << versions_blocked
      << ",\"versions_after_cancel\":" << versions_after_cancel << ",\"blocked_retirement_polls\":1"
      << ",\"normal_rebuild_graph_calls\":" << rebuild_graphs << ",\"normal_rebuild_target_rows\":" << rebuild_target_rows
      << ",\"normal_rebuild_draft_rows\":" << rebuild_draft_rows << ",\"normalized_lag_rows\":" << normalized_lag_rows
      << ",\"canonical_teacherforced_rebuild_byte_exact\":" << (rebuilt_exact ? "true" : "false")
      << ",\"historical_unconsumed_pending\":" << historical_pending << ",\"rebuilt_actual_prediction\":" << rebuilt_prediction
      << ",\"recomputed_decision_changed\":" << (historical_pending != rebuilt_prediction ? "true" : "false")
      << ",\"external_sampling_policy\":\"explicit legal non-argmax token published by this probe, not a natural-greedy disagreement\""
      << ",\"external_policy_original_prediction\":" << external_original_prediction << ",\"external_policy_published_pending\":" << external_pending
      << ",\"external_pending_used_without_overwriting_head\":" << (pending_policy ? "true" : "false")
      << ",\"timing_comparison\":false,\"numerical_tolerance_policy\":null"
      << ",\"scope\":\"Real R2 round with different histories, per-member quota2/1 and fresh natural proposals. R1-versus-R2 natural metrics are diagnostic and may have different accepted inputs or extents; no cross-B byte/ID or quality assertion. Canonical same-R1 teacher-forced committed-token rebuild alone has a byte oracle. Rebuild drops original GPU owners, normalizes draft lag0, emits no client output and consumes historical pending once before new user input. One real blocked cancellation uses original Graph/Completion/permissions; no device reads inside gate.\""
      << ",\"recomputed_decision_changes\":[";
    for (std::size_t i = 0; i < rebuild_changes.size(); ++i) { const auto &v = rebuild_changes[i];
      out << (i ? "," : "") << "{\"episode\":" << v.episode << ",\"historical\":" << v.historical << ",\"fresh\":" << v.fresh << '}'; }
    out << "],\"episodes\":[";
    for (std::size_t i = 0; i < episodes.size(); ++i) out << (i ? "," : "") << episodes[i];
    out << "],\"cross_B_extent_changes\":[";
    for (std::size_t i = 0; i < extent_changes.size(); ++i) out << (i ? "," : "") << quoted(extent_changes[i]);
    out << "],\"images\":[";
    for (std::size_t k = 0; k < images.size(); ++k) { const auto &entry = images[k];
      out << (k ? "," : "") << "{\"label\":" << quoted(entry.label) << ",\"metadata\":" << entry.saved.metadata << ",\"vectors\":[";
      for (std::size_t i = 0; i < entry.saved.vectors.size(); ++i) { const auto &v = entry.saved.vectors[i];
        out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"path\":" << quoted(v.path) << ",\"bytes\":" << v.bytes
          << ",\"dtype\":" << quoted(v.bf16 ? "BF16" : "F32") << ",\"fnv1a64\":" << quoted(v.hash) << '}'; }
      out << "]}";
    }
    out << "],\"vectors\":[";
    for (std::size_t i = 0; i < metrics.size(); ++i) { const auto &v = metrics[i];
      out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"baseline\":" << quoted(v.baseline) << ",\"actual\":" << quoted(v.actual)
        << ",\"dtype\":" << quoted(v.dtype) << ",\"bytes\":" << v.bytes << ",\"byte_equal\":" << (v.byte_equal ? "true" : "false")
        << ",\"actual_reuses_baseline_payload\":" << (v.byte_equal ? "true" : "false") << ",\"nonfinite_baseline\":" << v.nonfinite_baseline
        << ",\"nonfinite_actual\":" << v.nonfinite_actual << ",\"max_abs\":" << v.max_abs << ",\"squared_error\":" << v.squared_error
        << ",\"reference_squared\":" << v.reference_squared << '}';
    }
    out << "]}\n"; require(static_cast<bool>(out),"cannot finish MTP batch observation");
  }
};
void batch_single_commit(q::Model &model, MtpBatchObservation &observation, const std::shared_ptr<q::Session> &session,
                         q::GenerationLedger &ledger, std::span<const std::int32_t> inputs, q::GraphRole role,
                         bool generation, std::size_t quota, const std::string &name, std::int32_t expected_pending = -1,
                         std::size_t turn = 0) {
  q::prepare_generation_record(ledger,*session,role,inputs,generation,quota,turn);
  model.prepare_evaluation(*session,inputs,nullptr,role); observation.observe(model); model.submit_evaluation(*session); ++observation.graph_calls;
  const q::EvaluationIO output_io{.name = name, .trace_states = false};
  const auto output = q::evaluation_output(model, session, output_io, nullptr, nullptr, nullptr, nullptr, &ledger);
  if (generation) (void)q::finish_mtp_round(model, *session,expected_pending,quota,q::model_eos_tokens,&output, output_io);
  else (void)q::finish_evaluation(model, *session,name,{},false,nullptr,nullptr,nullptr,nullptr,{},&ledger);
  observation.observe(model); observation.episode(name,ledger.records[ledger.committed_count - 1]);
}
#if defined(HYLE_COMPUTE_CUDA)
void run_batch(const std::filesystem::path &inputs, const std::filesystem::path &output, q::BackendSelection selection) {
  const auto source = q::read_ids(inputs / "prompt.ids"); require(source.size() >= 23,"MTP batch needs original P15/P23 inputs");
  require(!std::filesystem::exists(output),"MTP batch output must be fresh"); std::filesystem::create_directories(output);
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, 64,4,23,q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry),2,2,true);
  q::Model model(std::move(weight_source), geometry, 64,plan,{},false,
      q::ReplayPolicy::kAuto,SIZE_MAX,{false,{},0},0,
      selection);
  require(model.weights.size() == 335,"MTP batch needs335 weights");
  MtpBatchObservation observation{.output=output};
  try {
    const std::array prefills{std::size_t{15},std::size_t{23},std::size_t{2}};
    const std::array pair{std::size_t{2}};
    const std::array single{std::size_t{1}};
    model.prepare_family(prefills,false,1,q::GraphRole::kMtpPrefill);
    for (auto count : prefills) model.plan_replay_uses(count,1,16,q::GraphRole::kMtpPrefill);
    for (auto role : {q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound,q::GraphRole::kMtpSingle,q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(q::drafts_token(role) ? std::span(pair) : std::span(single),false,1,role);
      model.plan_replay_uses(q::drafts_token(role) ? 2u : 1u,1,16,role);
    }
    model.prepare_family(pair,false,2,q::GraphRole::kMtpRound); model.plan_replay_uses(2,2,8,q::GraphRole::kMtpRound);
    std::array<SavedStep,2> references;
    for (std::size_t j = 0; j < 2; ++j) {
      const auto count = j ? 23u : 15u; const std::array calls{std::size_t(count)};
      auto session = model.create_session(calls,{},q::GraphRole::kMtpPrefill); q::GenerationLedger ledger(4);
      batch_single_commit(model,observation,session,ledger,std::span(source).first(count),q::GraphRole::kMtpPrefill,false,2,"reference_prefill" + std::to_string(j));
      const auto pending = session->head->next_id;
      batch_single_commit(model,observation,session,ledger,std::span(&pending,1),q::GraphRole::kMtpRound,true,j ? 1u : 2u,"reference_round" + std::to_string(j),pending);
      auto &phase = *model.phases.at(q::ExecutionShape{1,2,q::GraphRole::kMtpRound});
      references[j] = observation.record("reference_member" + std::to_string(j),batch_image(model,*session,phase,0,1));
      retire_prefix_session(session); require(model.snapshot_versions.live == 0,"R1 numeric reference retained a GPU head");
    }
    std::array<std::shared_ptr<q::Session>,2> sessions;
    std::array<q::GenerationLedger,2> ledgers{q::GenerationLedger(8),q::GenerationLedger(8)};
    for (std::size_t j = 0; j < 2; ++j) {
      const auto count = j ? 23u : 15u; const std::array calls{std::size_t(count)};
      sessions[j] = model.create_session(calls,{},q::GraphRole::kMtpPrefill);
      batch_single_commit(model,observation,sessions[j],ledgers[j],std::span(source).first(count),q::GraphRole::kMtpPrefill,false,2,"batch_prefill" + std::to_string(j));
    }
    std::array old_heads{sessions[0]->checkpoint(),sessions[1]->checkpoint()};
    const std::array old_images{image(model,*old_heads[0]),image(model,*old_heads[1])};
    std::array pending{sessions[0]->head->next_id,sessions[1]->head->next_id};
    std::array requests{q::EvaluationRequest{sessions[0],std::span(&pending[0],1)},q::EvaluationRequest{sessions[1],std::span(&pending[1],1)}};
    for (std::size_t j = 0; j < 2; ++j)
      q::prepare_generation_record(ledgers[j],*sessions[j],q::GraphRole::kMtpRound,requests[j].ids,true,j ? 1u : 2u,0);
    observation.versions_before_cancel = model.snapshot_versions.live;
    model.prepare_evaluations(requests,nullptr,q::GraphRole::kMtpRound); observation.observe(model);
    auto *const episode = model.active_step.get(); auto *const phase = episode->phase; auto *const run = episode->execution.run;
    const auto slot = model.slot_index;
    require(phase->slot_count == 1 && slot == 0 && run == q::take(phase->carriers[slot].borrow()) && run->valid() &&
            episode->members.size() == 2 && phase->family->requests == 2 && phase->io[slot].attention_layout.facts().requests == 2 &&
            phase->io[slot].attention_layout.facts().query_rows == 4 && std::count_if(episode->execution.bound.begin(),episode->execution.bound.end(),[](auto value) { return static_cast<bool>(value); }) == 14,
            "MTP batch did not prepare one real R2/T2 fourteen-bank Graph");
    const std::array candidate_ids{episode->members[0].next.get(),episode->members[1].next.get()};
    require(candidate_ids[0] && candidate_ids[1],"MTP batch did not retain both unique unpublished candidates");
    observation.versions_blocked = model.snapshot_versions.live;
    require(observation.versions_blocked == observation.versions_before_cancel + 2,"MTP batch did not charge both candidates");
    {
      BatchNativeGate gate; BatchReleaseAndDrain release{gate,*q::take(model.execution.common_context())};
      auto blocked = q::take(c::cuda::submit(*q::take(model.execution.common_queue()),[&](cudaStream_t stream) noexcept { return cudaLaunchHostFunc(stream,&BatchNativeGate::wait,&gate); }));
      gate.wait_until_entered(); model.submit_evaluations(); ++observation.graph_calls;
      q::check(sessions[0]->stop()); observation.dropped_completion = episode->execution.completion.valid();
      require(observation.dropped_completion,"MTP batch had no actual Completion to drop"); episode->execution.completion = {};
      const auto blocked_retirement = sessions[0]->try_retire();
      if (!blocked_retirement.status.IsCancelled() || blocked_retirement.status.has_native_error()) q::check(blocked_retirement.status);
      require(!blocked_retirement.retired,"MTP batch retired while its native queue was blocked");
      observation.blocked_retained = model.active_step.get() == episode && model.carrier_borrowed &&
          sessions[0]->pending == episode && sessions[1]->pending == episode &&
          episode->members[0].next.get() == candidate_ids[0] && episode->members[1].next.get() == candidate_ids[1] &&
          sessions[0]->head.get() == old_heads[0].get() && sessions[1]->head.get() == old_heads[1].get() && episode->cancelled &&
          model.snapshot_versions.live == observation.versions_blocked &&
          ledgers[0].committed_count == 1 && ledgers[1].committed_count == 1 &&
          q::take(model.cache_budget.usage(model.execution.device())).live_bytes == observation.backing;
      require(observation.blocked_retained,"blocked MTP cancellation lost candidates, oldheads, versions, ledger or carrier");
      // Never read device bytes while the real callback blocks this queue.
      gate.release(); q::check(blocked.wait()); blocked = {};
      const auto drained = q::take(model.execution.common_context())->drain(); if (!drained.IsCancelled() || drained.has_native_error()) q::check(drained);
      const auto run_drained = run->drain(); if (!run_drained.IsCancelled() || run_drained.has_native_error()) q::check(run_drained);
      const auto cancelled_retirement = sessions[0]->try_retire();
      if (!cancelled_retirement.status.IsCancelled() || cancelled_retirement.status.has_native_error()) q::check(cancelled_retirement.status);
      require(cancelled_retirement.retired,"released MTP batch failed its original retirement gates");
    }
    observation.versions_after_cancel = model.snapshot_versions.live;
    observation.cancelled = !model.active_step && !model.carrier_borrowed && !sessions[0]->head && !sessions[0]->pending &&
        !sessions[1]->pending && sessions[1]->head.get() == old_heads[1].get() &&
        observation.versions_after_cancel == observation.versions_before_cancel && !run->valid();
    require(observation.cancelled,"retired batch partially published a head or retained a candidate");
    for (std::size_t j = 0; j < 2; ++j) { require(same(old_images[j],image(model,*old_heads[j])),"cancel changed an old complete MTP head"); ++observation.old_checks; }
    sessions[0] = model.create_session({},old_heads[0],q::GraphRole::kMtpPrefill); requests[0].session = sessions[0];
    for (std::size_t j = 0; j < 2; ++j)
      q::prepare_generation_record(ledgers[j],*sessions[j],q::GraphRole::kMtpRound,requests[j].ids,true,j ? 1u : 2u,0);
    model.prepare_evaluations(requests,nullptr,q::GraphRole::kMtpRound); observation.observe(model);
    observation.same_slot_retry = model.active_step->phase == phase && model.active_step->execution.run == run &&
        run == q::take(phase->carriers[slot].borrow()) && run->valid(); require(observation.same_slot_retry,"MTP retry replaced its original finite R2 carrier slot");
    const std::array selections{q::MtpGenerationSelection{sessions[0],pending[0],2},q::MtpGenerationSelection{sessions[1],pending[1],1}};
    const std::array outputs_io{q::EvaluationIO{.name = "batch_member0", .trace_states = false},
        q::EvaluationIO{.name = "batch_member1", .trace_states = false}};
    const std::array outputs{q::evaluation_output(model, sessions[0], outputs_io[0], nullptr, nullptr, nullptr, nullptr, &ledgers[0]),
        q::evaluation_output(model, sessions[1], outputs_io[1], nullptr, nullptr, nullptr, nullptr, &ledgers[1])};
    std::array<q::MtpRoundResult,2> results;
    model.submit_evaluations(); ++observation.graph_calls;
    model.finish_mtp_rounds(selections,q::model_eos_tokens,outputs,results, [&](std::size_t j, const q::CompletedEvaluationView &view) {
          q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
        }); observation.observe(model);
    for (std::size_t j = 0; j < 2; ++j) {
      const auto &record = ledgers[j].records[1]; observation.accepted[j] = record.accepted;
      require(record.committed && record.requests == 2 && record.member_index == j && record.accepted >= 1 && record.accepted <= 2 &&
              record.before.length == (j ? 23u : 15u) &&
              record.after.length == record.before.length + record.accepted && record.accepted == results[j].processed_target &&
              record.after.next_id == results[j].next_id && record.pending_input == pending[j],"MTP B2 decision/position was routed to another member");
      require(same(old_images[j],image(model,*old_heads[j])),"B2 commit changed an old complete prefix"); ++observation.old_checks;
      observation.episode("batch_member" + std::to_string(j),record);
      auto captured = batch_image(model,*sessions[j],*phase,j,2);
      const auto &hidden = captured.blobs.back().bytes, &logits = captured.blobs[captured.blobs.size() - 2].bytes;
      require(hidden.size() == 8192 && logits.size() == 2 * 248320 * sizeof(float),"R2 member output contains another member's rows");
      const auto carry = read(sessions[j]->head->carry_hidden), prior = read(sessions[j]->head->prior_hidden);
      require(carry.size() == 4096 && std::equal(carry.begin(),carry.end(),hidden.begin() + static_cast<std::ptrdiff_t>((record.accepted - 1) * 4096)) &&
              (record.accepted == 2 ? prior.size() == 4096 && std::equal(prior.begin(),prior.end(),hidden.begin()) : prior.empty()),
              "B2 accepted carry/prior does not match that member's actual target row");
      std::array<std::int32_t,2> raw_argmax{-1,-1};
      for (std::size_t row = 0; row < 2; ++row) {
        float best = -(std::numeric_limits<float>::infinity)();
        for (std::int32_t token = 0; token < model.geometry.vocabulary; ++token) {
          float value = 0; std::memcpy(&value,logits.data() + (row * 248320 + static_cast<std::size_t>(token)) * sizeof(float),sizeof(value));
          require(std::isfinite(value),"MTP target logits are nonfinite");
          if (raw_argmax[row] < 0 || value > best) { best = value; raw_argmax[row] = token; }
        }
      }
      require(record.target_first == raw_argmax[0] && record.target_second == raw_argmax[1],"B2 sampled IDs do not match their own member/row logits");
      (void)observation.record("batch_member" + std::to_string(j),std::move(captured),&references[j]);
      const auto &head = *sessions[j]->head;
      require(head.draft->frontier == record.before.frontier + 1 && head.draft->lag == record.accepted - 1 &&
              head.draft->catchup_token == (record.accepted == 2 ? record.draft : -1) &&
              head.draft->key.layout().facts().requests == 1 && head.draft->value.layout().facts().requests == 1,
              "public draft frame exposes another member");
      for (const auto &frame : head.keys) require(frame.layout().facts().requests == 1,"public target frame exposes packed members");
    }
    observation.mixed_acceptance = observation.accepted[0] != observation.accepted[1];
    require(observation.accepted[1] == 1 && sessions[0]->head->pages.back().page() != sessions[1]->head->pages.back().page() &&
            sessions[0]->head->draft->pages.back().page() != sessions[1]->head->draft->pages.back().page(),"quota1 or per-member dirty page isolation failed");
    for (std::size_t layer = 0; layer < 24; ++layer) if (layer % 4 != 3)
      require(batch_disjoint(sessions[0]->head->state[layer].history,sessions[1]->head->state[layer].history) &&
              batch_disjoint(sessions[0]->head->state[layer].recurrent,sessions[1]->head->state[layer].recurrent),"MTP batch recurrent results overlap");
    require(batch_disjoint(sessions[0]->head->carry_hidden,sessions[1]->head->carry_hidden),"MTP batch carry rows overlap");
    observation.independent_frames = true; old_heads = {};

    const auto idle_b = image(model,*sessions[1]->head);
    observation.historical_pending = results[0].next_id;
    retire_prefix_session(sessions[0]); require(model.snapshot_versions.live == 1,"evicted MTP A still owns a GPU version");
    // An independent teacher-forced reference uses only the immutable accepted
    // chunks. It is not an alternative MTP executor or a forced new proposal.
    const std::array reference_calls{std::size_t{15}};
    auto canonical = model.create_session(reference_calls,{},q::GraphRole::kMtpPrefill);
    q::Phase *canonical_phase = nullptr;
    for (std::size_t i = 0; i < ledgers[0].committed_count; ++i) {
      const auto ids = ledgers[0].records[i].committed_ids();
      const auto role = ids.size() == 1 ? q::GraphRole::kMtpSingle : q::GraphRole::kMtpPrefill;
      model.prepare_evaluation(*canonical,ids,nullptr,role); observation.observe(model); canonical_phase = canonical->pending->phase;
      model.submit_evaluation(*canonical); ++observation.graph_calls;
      (void)q::finish_evaluation(model, *canonical,"canonical_rebuild" + std::to_string(i),{},false);
    }
    require(canonical_phase && canonical->head->draft->lag == 0 && canonical->head->draft->frontier == canonical->length(),"canonical reconstruction has an incomplete draft state");
    const auto canonical_saved = observation.record("canonical_committed_rebuild",batch_image(model,*canonical,*canonical_phase,0,1));
    retire_prefix_session(canonical); require(model.snapshot_versions.live == 1,"canonical rebuild reference retained a GPU head");
    q::GenerationRestoreStats stats;
    sessions[0] = q::rebuild_generation_session(model,ledgers[0],{},false,&stats);
    observation.graph_calls += stats.graph_calls; observation.rebuild_graphs = stats.graph_calls;
    observation.rebuild_target_rows = stats.target_rows; observation.rebuild_draft_rows = stats.draft_rows; observation.observe(model);
    observation.rebuild_changes = stats.decision_changes; observation.normalized_lag_rows = stats.normalized_lag_rows;
    require(stats.graph_calls == 2 && stats.target_rows == ledgers[0].records[1].after.length && stats.draft_rows == stats.target_rows,
            "normal rebuild changed accepted chunks or miscounted normalized target/draft work");
    require(stats.normalized_lag_rows == ledgers[0].records[1].after.lag,"normal rebuild did not report its actual draft lag normalization");
    const auto last_count = ledgers[0].records[ledgers[0].committed_count - 1].committed_ids().size();
    const auto rebuilt_role = last_count == 1 ? q::GraphRole::kMtpSingle : q::GraphRole::kMtpPrefill;
    auto &rebuilt_phase = *model.phases.at(q::ExecutionShape{1,last_count,rebuilt_role});
    (void)observation.record("normal_rebuild",batch_image(model,*sessions[0],rebuilt_phase,0,1),&canonical_saved,true);
    observation.rebuilt_exact = true; observation.rebuilt_prediction = sessions[0]->head->next_id;
    require(sessions[0]->length() == ledgers[0].records[1].after.length && sessions[0]->head->draft->lag == 0 &&
            sessions[0]->head->draft->frontier == sessions[0]->length(),"normal reconstruction did not normalize the actual committed transcript");
    require(same(idle_b,image(model,*sessions[1]->head)),"rebuilding A changed idle B"); ++observation.idle_checks;
    // The factual emitted pending is consumed once, even if the rebuilt
    // prediction differs. A genuine new user token follows in the same warm Graph.
    const std::array warm{observation.historical_pending,source[16]};
    const auto before_warm = sessions[0]->length();
    batch_single_commit(model,observation,sessions[0],ledgers[0],warm,q::GraphRole::kMtpPrefill,false,2,"warm_actual_pending_plus_user",-1,1);
    require(sessions[0]->length() == before_warm + 2 && ledgers[0].records[2].committed_ids()[0] == observation.historical_pending,
            "warm turn did not consume the historical pending exactly once before user input");
    auto &warm_phase = *model.phases.at(q::ExecutionShape{1,2,q::GraphRole::kMtpPrefill});
    (void)observation.record("warm_actual_pending_plus_user",batch_image(model,*sessions[0],warm_phase,0,1));
    observation.external_original_prediction = sessions[0]->head->next_id;
    observation.external_pending = source[16];
    if (observation.external_pending == observation.external_original_prediction || q::is_end_of_generation(observation.external_pending))
      observation.external_pending = observation.external_original_prediction == 0 ? 1 : 0;
    // This is an explicitly declared external sampling policy, not a claim
    // that natural argmax changed. Publishing this legal token is engine policy.
    const auto *const original_head = sessions[0]->head.get();
    q::prepare_generation_record(ledgers[0],*sessions[0],q::GraphRole::kMtpRound,std::span(&observation.external_pending,1),true,2,1);
    model.prepare_evaluation(*sessions[0],std::span(&observation.external_pending,1),nullptr,q::GraphRole::kMtpRound); observation.observe(model);
    require(sessions[0]->head.get() == original_head && sessions[0]->head->next_id == observation.external_original_prediction &&
            observation.external_pending != observation.external_original_prediction,"expected pending overwrote the real head prediction");
    auto *const policy_phase = sessions[0]->pending->phase;
    const q::EvaluationIO policy_output_io{.name = "external_sampled_pending", .trace_states = false};
    const auto policy_output = q::evaluation_output(model, sessions[0], policy_output_io, nullptr, nullptr, nullptr, nullptr, &ledgers[0]);
    model.submit_evaluation(*sessions[0]); ++observation.graph_calls;
    const auto policy_result = q::finish_mtp_round(model, *sessions[0],observation.external_pending,2,q::model_eos_tokens,&policy_output, policy_output_io);
    observation.pending_policy = policy_result.pending == observation.external_pending && ledgers[0].records[3].pending_input == observation.external_pending;
    require(observation.pending_policy,"generation ignored its original explicit external pending token");
    observation.episode("external_sampled_pending",ledgers[0].records[3]);
    (void)observation.record("external_sampled_pending",batch_image(model,*sessions[0],*policy_phase,0,1));
    require(same(idle_b,image(model,*sessions[1]->head)),"A warm/policy continuation changed idle B"); ++observation.idle_checks;
    retire_prefix_session(sessions[0]); retire_prefix_session(sessions[1]); observation.observe(model);
    require(observation.graph_calls == 14 && observation.images.size() == 8 && observation.finite && observation.cancelled &&
            observation.blocked_retained && observation.same_slot_retry && observation.rebuilt_exact &&
            observation.pending_policy && !model.active_step && !model.carrier_borrowed && model.snapshot_versions.live == 0,
            "MTP batch/reconstruction/retirement did not complete");
    observation.report("PASS_STRUCTURE_NUMERICAL_METRICS_RECORDED",model);
    std::cout << "{\"status\":\"MTP_BATCH_PROBE_COMPLETED\",\"actual_graph_calls\":" << observation.graph_calls
      << ",\"compared_vectors\":" << observation.metrics.size() << ",\"mixed_acceptance_observed\":" << (observation.mixed_acceptance ? "true" : "false")
      << ",\"all_finite\":true,\"timing_comparison\":false}\n";
  } catch (const std::exception &error) { observation.observe(model); observation.report("FAIL",model,error.what()); throw; }
}
#endif
struct MtpNumericsObservation final {
  MtpBatchObservation data;
  std::vector<std::string> controls, skipped;
  std::size_t seed_unchanged_checks = 0, control_unchanged_checks = 0;
  void report(std::string_view status, q::Model &model, std::string_view error = {}) const {
    std::ofstream out(data.output / "observation.json"); require(static_cast<bool>(out),"cannot write fixed-state numerical observation");
    out << std::setprecision(17) << "{\"status\":" << quoted(status) << ",\"scenario\":\"numerics\",\"error\":" << quoted(error)
      << ",\"model_objects\":1,\"weight_tensors\":335,\"capacity\":64,\"resident_limit\":6,\"active_sequences\":2"
      << ",\"actual_graph_calls\":" << data.graph_calls << ",\"image_count\":" << data.images.size()
      << ",\"peak_live_snapshot_versions\":" << data.peak_versions << ",\"final_live_snapshot_versions\":" << model.snapshot_versions.live
      << ",\"managed_backing_bytes\":" << data.backing << ",\"planned_backing_bytes\":" << model.cache_plan.backing_bytes
      << ",\"all_finite\":" << (data.finite ? "true" : "false") << ",\"seed_full52_unchanged_checks\":" << seed_unchanged_checks
      << ",\"control_full52_unchanged_checks\":" << control_unchanged_checks << ",\"timing_comparison\":false,\"numerical_tolerance_policy\":null"
      << ",\"scope\":\"One R1 P15 and a real two-sequence P15 Graph form two complete old-state sources. Each source advances only through two natural R1 MTP rounds. Fixed-state controls borrow the exact same immutable Snapshot pointer through independent mutable Sessions. R1 and R2 fresh proposal IDs are observed, never injected; different proposals mean target row1 inputs differ. Intra-B equal-prefix comparisons and cross-B comparisons are raw numerical diagnostics, not a full-model error budget. No replay/padding, state backfill or second executor.\""
      << ",\"controls\":[";
    for (std::size_t i = 0; i < controls.size(); ++i) out << (i ? "," : "") << controls[i];
    out << "],\"skipped_controls\":[";
    for (std::size_t i = 0; i < skipped.size(); ++i) out << (i ? "," : "") << skipped[i];
    out << "],\"episodes\":[";
    for (std::size_t i = 0; i < data.episodes.size(); ++i) out << (i ? "," : "") << data.episodes[i];
    out << "],\"extent_changes\":[";
    for (std::size_t i = 0; i < data.extent_changes.size(); ++i) out << (i ? "," : "") << quoted(data.extent_changes[i]);
    out << "],\"images\":[";
    for (std::size_t k = 0; k < data.images.size(); ++k) { const auto &entry = data.images[k];
      out << (k ? "," : "") << "{\"label\":" << quoted(entry.label) << ",\"metadata\":" << entry.saved.metadata << ",\"vectors\":[";
      for (std::size_t i = 0; i < entry.saved.vectors.size(); ++i) { const auto &v = entry.saved.vectors[i];
        out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"path\":" << quoted(v.path) << ",\"bytes\":" << v.bytes
          << ",\"dtype\":" << quoted(v.bf16 ? "BF16" : "F32") << ",\"fnv1a64\":" << quoted(v.hash) << '}'; }
      out << "]}";
    }
    out << "],\"vectors\":[";
    for (std::size_t i = 0; i < data.metrics.size(); ++i) { const auto &v = data.metrics[i];
      out << (i ? "," : "") << "{\"name\":" << quoted(v.name) << ",\"baseline\":" << quoted(v.baseline) << ",\"actual\":" << quoted(v.actual)
        << ",\"dtype\":" << quoted(v.dtype) << ",\"bytes\":" << v.bytes << ",\"byte_equal\":" << (v.byte_equal ? "true" : "false")
        << ",\"actual_reuses_baseline_payload\":" << (v.byte_equal ? "true" : "false") << ",\"nonfinite_baseline\":" << v.nonfinite_baseline
        << ",\"nonfinite_actual\":" << v.nonfinite_actual << ",\"max_abs\":" << v.max_abs << ",\"squared_error\":" << v.squared_error
        << ",\"reference_squared\":" << v.reference_squared << '}';
    }
    out << "]}\n"; require(static_cast<bool>(out),"cannot finish fixed-state numerical observation");
  }
};
void run_numerics(const std::filesystem::path &inputs, const std::filesystem::path &output, q::BackendSelection selection) {
  const auto tokens = q::read_ids(inputs / "prompt.ids"); require(tokens.size() >= 15,"numerical control needs original P15 input");
  require(!std::filesystem::exists(output),"numerical control output must be fresh"); std::filesystem::create_directories(output);
  const auto prompt = std::span(tokens).first(15);
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, 64,6,15,q::recurrent_bytes(q::state_schema_for(geometry, 64), geometry),2,2,true);
  q::Model model(std::move(weight_source), geometry, 64,plan,{},false,
      q::ReplayPolicy::kAuto,SIZE_MAX,{false,{},0},0,selection);
  require(model.weights.size() == 335,"numerical control requires335 weights");
  model.trace_routes = output / "routes";
  MtpNumericsObservation observation{.data={.output=output}};
  try {
    const std::array prefills{std::size_t{15}};
    const std::array pair{std::size_t{2}};
    model.prepare_family(prefills,false,1,q::GraphRole::kMtpPrefill); model.plan_replay_uses(15,1,8,q::GraphRole::kMtpPrefill);
    model.prepare_family(prefills,false,2,q::GraphRole::kMtpPrefill); model.plan_replay_uses(15,2,8,q::GraphRole::kMtpPrefill);
    for (auto role : {q::GraphRole::kMtpRound,q::GraphRole::kMtpCatchupRound}) {
      model.prepare_family(pair,false,1,role); model.plan_replay_uses(2,1,16,role);
    }
    model.prepare_family(pair,false,2,q::GraphRole::kMtpRound); model.plan_replay_uses(2,2,8,q::GraphRole::kMtpRound);
    std::array<q::cache::SnapshotRef,2> seed_heads, old_heads;
    auto seed = model.create_session(prefills,{},q::GraphRole::kMtpPrefill); q::GenerationLedger seed_record(1);
    batch_single_commit(model,observation.data,seed,seed_record,prompt,q::GraphRole::kMtpPrefill,false,0,"seed_R1");
    auto &prefill1 = *model.phases.at(q::ExecutionShape{1,15,q::GraphRole::kMtpPrefill});
    const auto seed_r1 = observation.data.record("seed_R1",batch_image(model,*seed,prefill1,0,1));
    seed_heads[0] = seed->checkpoint(); retire_prefix_session(seed);

    std::array seed_sessions{model.create_session(prefills,{},q::GraphRole::kMtpPrefill),model.create_session(prefills,{},q::GraphRole::kMtpPrefill)};
    std::array<q::GenerationLedger,2> seed_ledgers{q::GenerationLedger(1),q::GenerationLedger(1)};
    std::array seed_requests{q::EvaluationRequest{seed_sessions[0],prompt},q::EvaluationRequest{seed_sessions[1],prompt}};
    for (std::size_t j = 0; j < 2; ++j)
      q::prepare_generation_record(seed_ledgers[j],*seed_sessions[j],q::GraphRole::kMtpPrefill,prompt,false,0,0);
    model.prepare_evaluations(seed_requests,nullptr,q::GraphRole::kMtpPrefill); observation.data.observe(model);
    auto *const prefill2 = model.active_step->phase;
    const std::array seed_outputs_io{q::EvaluationIO{.name = "seed_R2_0", .trace_states = false},
        q::EvaluationIO{.name = "seed_R2_1", .trace_states = false}};
    const std::array seed_outputs{q::evaluation_output(model, seed_sessions[0], seed_outputs_io[0], nullptr, nullptr, nullptr, nullptr, &seed_ledgers[0]),
        q::evaluation_output(model, seed_sessions[1], seed_outputs_io[1], nullptr, nullptr, nullptr, nullptr, &seed_ledgers[1])};
    std::array<std::int32_t,2> seed_ids{-1,-1}; model.submit_evaluations(); ++observation.data.graph_calls;
    model.finish_evaluations(seed_outputs,seed_ids, [&](std::size_t j, const q::CompletedEvaluationView &view) {
          q::observe_completed_evaluation(model, seed_outputs[j], seed_outputs_io[j], view);
        }); observation.data.observe(model);
    const auto seed_r2 = observation.data.record("seed_R2_0",batch_image(model,*seed_sessions[0],*prefill2,0,2),&seed_r1);
    (void)observation.data.record("seed_R2_1",batch_image(model,*seed_sessions[1],*prefill2,1,2),&seed_r2);
    for (std::size_t j = 0; j < 2; ++j) observation.data.episode("seed_R2_" + std::to_string(j),seed_ledgers[j].records[0]);
    seed_heads[1] = seed_sessions[0]->checkpoint();
    for (auto &session : seed_sessions) retire_prefix_session(session);

    std::array<std::array<SavedStep,2>,2> source_steps;
    for (std::size_t source = 0; source < 2; ++source) {
      const auto original_seed = image(model,*seed_heads[source]);
      auto session = model.create_session({},seed_heads[source],q::GraphRole::kMtpPrefill); q::GenerationLedger ledger(2);
      for (std::size_t step = 0; step < 2; ++step) {
        const auto pending = session->head->next_id;
        if (q::is_end_of_generation(pending)) {
          observation.skipped.push_back("{\"source\":" + std::to_string(source) + ",\"reason\":\"natural EOS before the planned L18 control\"}"); break;
        }
        const auto role = session->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
        const auto name = "source" + std::to_string(source) + "_R1_step" + std::to_string(step);
        batch_single_commit(model,observation.data,session,ledger,std::span(&pending,1),role,true,2,name,pending);
        auto &phase = *model.phases.at(q::ExecutionShape{1,2,role});
        source_steps[source][step] = observation.data.record(name,batch_image(model,*session,phase,0,1),source ? &source_steps[0][step] : nullptr);
      }
      require(same(original_seed,image(model,*seed_heads[source])),"natural R1 derivation changed its old seed state"); ++observation.seed_unchanged_checks;
      old_heads[source] = session->checkpoint(); retire_prefix_session(session);
    }
    seed_heads = {}; // Only the two real old L18 sources remain on the device.
    for (std::size_t source = 0; source < 2; ++source) {
      const auto &old = old_heads[source]; const auto &boundary = *old;
      if (boundary.length != 18 || boundary.draft->frontier != 18 || boundary.draft->lag || boundary.next_id != 271) {
        std::ostringstream reason;
        reason << "{\"source\":" << source << ",\"actual_boundary\":" << image(model,boundary).metadata
          << ",\"reason\":\"natural oldstate is not the requested L18/D18/lag0/pending271; no token or acceptance was forced\"}";
        observation.skipped.push_back(reason.str()); continue;
      }
      const auto original = image(model,*old);
      const auto pending = boundary.next_id;
      auto single = model.create_session({},old,q::GraphRole::kMtpPrefill); q::GenerationLedger single_ledger(1);
      require(single->head.get() == old.get(),"R1 numerical control recomputed instead of borrowing its old Snapshot");
      const auto label = "control_source" + std::to_string(source);
      batch_single_commit(model,observation.data,single,single_ledger,std::span(&pending,1),q::GraphRole::kMtpRound,true,2,label + "_R1",pending);
      auto &round1 = *model.phases.at(q::ExecutionShape{1,2,q::GraphRole::kMtpRound});
      const auto reference = observation.data.record(label + "_R1",batch_image(model,*single,round1,0,1));
      require(same(original,image(model,*old)),"R1 fixed-state branch changed the source52 components"); ++observation.control_unchanged_checks;
      retire_prefix_session(single);

      std::array twins{model.create_session({},old,q::GraphRole::kMtpPrefill),model.create_session({},old,q::GraphRole::kMtpPrefill)};
      std::array<q::GenerationLedger,2> twin_ledgers{q::GenerationLedger(1),q::GenerationLedger(1)};
      require(twins[0] != twins[1] && twins[0]->head.get() == old.get() && twins[1]->head.get() == old.get(),
              "R2 control requires two independent Sessions sharing the exact same immutable old Snapshot");
      const std::array twin_requests{q::EvaluationRequest{twins[0],std::span(&pending,1)},q::EvaluationRequest{twins[1],std::span(&pending,1)}};
      for (std::size_t j = 0; j < 2; ++j)
        q::prepare_generation_record(twin_ledgers[j],*twins[j],q::GraphRole::kMtpRound,twin_requests[j].ids,true,2,0);
      model.prepare_evaluations(twin_requests,nullptr,q::GraphRole::kMtpRound); observation.data.observe(model);
      auto *const phase = model.active_step->phase;
      require(phase->family->requests == 2 && model.active_step->members[0].base == old.get() && model.active_step->members[1].base == old.get(),
              "R2 numerical control is not one true Graph with the exact retained old state");
      const std::array selections{q::MtpGenerationSelection{twins[0],pending,2},q::MtpGenerationSelection{twins[1],pending,2}};
      const std::array outputs_io{q::EvaluationIO{.name = label, .trace_states = false},
          q::EvaluationIO{.name = label, .trace_states = false}};
      const std::array outputs{q::evaluation_output(model, twins[0], outputs_io[0], nullptr, nullptr, nullptr, nullptr, &twin_ledgers[0]),
          q::evaluation_output(model, twins[1], outputs_io[1], nullptr, nullptr, nullptr, nullptr, &twin_ledgers[1])};
      std::array<q::MtpRoundResult,2> results; model.submit_evaluations(); ++observation.data.graph_calls;
      model.finish_mtp_rounds(selections,q::model_eos_tokens,outputs,results, [&](std::size_t j, const q::CompletedEvaluationView &view) {
            q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
          }); observation.data.observe(model);
      const auto member0 = observation.data.record(label + "_R2_member0",batch_image(model,*twins[0],*phase,0,2),&reference);
      (void)observation.data.record(label + "_R2_member1",batch_image(model,*twins[1],*phase,1,2),&member0);
      for (std::size_t j = 0; j < 2; ++j) observation.data.episode(label + "_R2_member" + std::to_string(j),twin_ledgers[j].records[0]);
      require(same(original,image(model,*old)),"R2 fixed-state branches changed source52 components"); ++observation.control_unchanged_checks;
      const auto &r1 = single_ledger.records[0];
      std::ostringstream control;
      control << "{\"source\":" << source << ",\"same_old_snapshot_pointer\":true,\"independent_R2_sessions\":true"
        << ",\"old_boundary\":" << original.metadata << ",\"pending\":" << pending << ",\"R1_draft\":" << r1.draft
        << ",\"R2_drafts\":[" << results[0].draft << ',' << results[1].draft << "]"
        << ",\"R1_target_ids\":[" << r1.target_first << ',' << r1.target_second << "]"
        << ",\"R2_target_ids\":[[" << results[0].target_first << ',' << results[0].target_second << "],[" << results[1].target_first << ',' << results[1].target_second << "]]"
        << ",\"target_two_rows_use_identical_inputs\":" << (r1.draft == results[0].draft && r1.draft == results[1].draft ? "true" : "false")
        << ",\"old_full52_bytes_unchanged\":true}";
      observation.controls.push_back(control.str());
      for (auto &session : twins) retire_prefix_session(session);
    }
    old_heads = {}; observation.data.observe(model);
    require(!model.active_step && !model.carrier_borrowed && model.snapshot_versions.live == 0 && observation.data.finite,
            "numerical control retained native state or produced nonfinite data");
    if (observation.skipped.empty()) require(observation.data.graph_calls == 10 && observation.data.images.size() == 13 && observation.controls.size() == 2,
            "finite numerical control did not execute its requested10 Graphs/13images");
    observation.report(observation.skipped.empty() ? "DIAGNOSTIC_COMPLETE_NUMERICAL_METRICS_ONLY" : "DIAGNOSTIC_NATURAL_PRECONDITIONS_NOT_MET",model);
    std::cout << "{\"status\":\"MTP_NUMERICS_PROBE_COMPLETED\",\"actual_graph_calls\":" << observation.data.graph_calls
      << ",\"images\":" << observation.data.images.size() << ",\"controls\":" << observation.controls.size()
      << ",\"skipped_controls\":" << observation.skipped.size() << ",\"all_finite\":true,\"timing_comparison\":false}\n";
  } catch (const std::exception &error) { observation.data.observe(model); observation.report("FAIL",model,error.what()); throw; }
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path inputs, output;
    std::string scenario = "sessions";
    q::BackendSelection selection;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("probe option lacks its value");
      const std::string_view option(argv[i]);
      if (option == "--inputs") inputs = argv[i + 1];
      else if (option == "--output") output = argv[i + 1];
      else if (option == "--scenario") scenario = argv[i + 1];
      else if (option == "--backend") {
        const std::string_view value(argv[i + 1]);
        if (value == "cpu") selection.primary = q::BackendKind::kCpu;
        else if (value == "cuda") selection.primary = q::BackendKind::kCuda;
        else if (value == "apple") selection.primary = q::BackendKind::kApple;
        else if (value == "vulkan") selection.primary = q::BackendKind::kVulkan;
        else throw std::runtime_error("--backend requires cpu, cuda, apple or vulkan");
      }
      else throw std::runtime_error("unknown MTP Session probe option");
    }
    if (inputs.empty() || output.empty() || (scenario != "sessions" && scenario != "prefix" && scenario != "resume" && scenario != "batch" && scenario != "numerics"))
      throw std::runtime_error("usage: hyle_qwen35_mtp_session_probe --inputs DIR --output FRESH_DIR [--backend cpu|cuda|apple|vulkan] [--scenario sessions|prefix|resume|batch|numerics]");
    selection = q::select_execution(selection, 0);
    if (selection.primary != q::BackendKind::kCuda && scenario != "prefix")
      throw std::runtime_error("non-CUDA MTP Session probe currently supports --scenario prefix only");
    if (scenario == "numerics") run_numerics(inputs,output,selection);
    else if (scenario == "batch") {
#if defined(HYLE_COMPUTE_CUDA)
      run_batch(inputs,output,selection);
#endif
    }
    else if (scenario == "resume") run_resume(inputs,output,selection);
    else if (scenario == "prefix") run_prefix(inputs,output,selection); else run(inputs, output,selection); return 0;
  } catch (const std::exception &error) { std::cerr << "Qwen MTP Session probe: " << error.what() << '\n'; return 1; }
}
