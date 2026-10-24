// Explicit native correctness consumer of the original unified Model path.
// No throughput result, alternative executor or oracle feedback into inputs.
#include "main.cc"
#include "probe_geometry.h"

#include <condition_variable>
#include <cstring>
#include <mutex>
#include "hyle/compute/detail/paged_arena_access.h"

namespace {
namespace q = qwen35;
namespace c = q::c;
using Bytes = std::vector<std::byte>;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

struct NativeGate final {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false, open = false;
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<NativeGate *>(opaque);
    std::unique_lock lock(gate.mutex); gate.entered = true; gate.condition.notify_all();
    gate.condition.wait(lock, [&] { return gate.open; }); // No CUDA calls on this callback.
  }
  void wait_until_entered() {
    std::unique_lock lock(mutex);
    require(condition.wait_for(lock, std::chrono::seconds(30), [&] { return entered; }), "batch Queue did not enter its native gate");
  }
  void release() noexcept {
    { std::lock_guard lock(mutex); open = true; }
    condition.notify_all();
  }
  ~NativeGate() { release(); }
};

struct ReleaseAndDrain final {
  NativeGate &gate;
  c::Context &context;
  ~ReleaseAndDrain() { gate.release(); context.drain().IgnoreError(); }
};

std::string json_string(std::string_view text) {
  std::ostringstream out; out << '"';
  for (const auto value : text) {
    const auto ch = static_cast<unsigned char>(value);
    if (ch == '"' || ch == '\\') out << '\\' << value;
    else if (ch == '\n') out << "\\n";
    else if (ch == '\r') out << "\\r";
    else if (ch == '\t') out << "\\t";
    else if (ch < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(ch) << std::dec;
    else out << value;
  }
  out << '"'; return out.str();
}

Bytes read_lease(const c::ReadLease &lease) {
  Bytes result(lease.size()); q::check(q::take(lease.view()).read(result)); return result;
}

Bytes read_file(const std::filesystem::path &path, std::size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(stream), "missing actual output vector");
  require(stream.tellg() == static_cast<std::streamoff>(expected), "per-Session trace has an unexpected byte extent");
  stream.seekg(0); Bytes bytes(expected);
  stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(stream), "cannot read complete actual output vector"); return bytes;
}

struct Metric final {
  std::string name, baseline_path, batch_path, dtype;
  std::size_t bytes = 0, elements = 0, nonfinite_baseline = 0, nonfinite_batch = 0, zero_reference_mismatches = 0;
  double max_abs = 0, squared_error = 0, reference_squared = 0, batch_squared = 0, dot = 0, max_relative_nonzero_reference = 0;
  bool byte_equal = false;
};

Metric compare(const std::filesystem::path &baseline, const std::filesystem::path &batch,
               std::string name, std::size_t bytes, bool bf16) {
  const auto a = read_file(baseline, bytes), b = read_file(batch, bytes);
  Metric result; result.name = std::move(name); result.baseline_path = baseline.string(); result.batch_path = batch.string();
  result.dtype = bf16 ? "BF16" : "F32"; result.bytes = bytes;
  const auto width = bf16 ? sizeof(std::uint16_t) : sizeof(float);
  require(bytes % width == 0, "output extent is not scalar aligned"); result.elements = bytes / width;
  result.byte_equal = a == b;
  const auto scalar = [&](const Bytes &data, std::size_t index) {
    if (bf16) {
      std::uint16_t word = 0; std::memcpy(&word, data.data() + index * width, sizeof(word));
      return std::bit_cast<float>(std::uint32_t(word) << 16);
    }
    float value = 0; std::memcpy(&value, data.data() + index * width, sizeof(value)); return value;
  };
  for (std::size_t i = 0; i < result.elements; ++i) {
    const double x = scalar(a, i), y = scalar(b, i);
    result.nonfinite_baseline += !std::isfinite(x); result.nonfinite_batch += !std::isfinite(y);
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    const auto error = std::abs(y - x); result.max_abs = (std::max)(result.max_abs, error);
    result.squared_error += error * error; result.reference_squared += x * x; result.batch_squared += y * y; result.dot += x * y;
    if (x != 0) result.max_relative_nonzero_reference = (std::max)(result.max_relative_nonzero_reference, error / std::abs(x));
    else result.zero_reference_mismatches += y != 0;
  }
  return result;
}

struct Observation final {
  std::filesystem::path output;
  std::vector<Metric> metrics;
  std::array<std::int32_t, 4> baseline_ids{}, batch_ids{};
  std::array<std::int32_t, 7> teacher_forced{};
  bool ids_equal = true, finite = true, old_unchanged = false, independent_frames = false, independent_states = false;
  bool one_batch_episode = false, continued_independently = false;
  bool cancel_tested = false, dropped_live_completion = false, pending_retained = false, cancelled_heads_unchanged = false, batch_retry_same_slot = false;
  std::size_t cancel_versions_before = 0, cancel_versions_blocked = 0, cancel_versions_retired = 0, cancel_backing_bytes = 0;
  std::array<std::int32_t, 2> retry_ids{-1,-1};
  void write(std::string_view status, std::string_view error = {}) const {
    std::ofstream stream(output / "observation.json"); require(static_cast<bool>(stream), "cannot write batch probe observation");
    stream << std::setprecision(17) << "{\"status\":" << json_string(status) << ",\"error\":" << json_string(error)
      << ",\"capacity\":256,\"resident_limit\":8,\"max_active\":2,\"batch_requests\":2,\"tokens_per_request\":1"
      << ",\"one_batch_episode\":" << (one_batch_episode ? "true" : "false")
      << ",\"ids_equal\":" << (ids_equal ? "true" : "false") << ",\"all_finite\":" << (finite ? "true" : "false")
      << ",\"old_snapshots_byte_unchanged\":" << (old_unchanged ? "true" : "false")
      << ",\"independent_ready_frames\":" << (independent_frames ? "true" : "false")
      << ",\"nonoverlapping_recurrent_outputs\":" << (independent_states ? "true" : "false")
      << ",\"independent_B1_continuation\":" << (continued_independently ? "true" : "false")
      << ",\"timing_comparison\":false,\"cancel_tested\":" << (cancel_tested ? "true" : "false")
      << ",\"dropped_live_completion\":" << (dropped_live_completion ? "true" : "false")
      << ",\"blocked_candidates_pending_versions_backing_retained\":" << (pending_retained ? "true" : "false")
      << ",\"cancelled_heads_byte_unchanged\":" << (cancelled_heads_unchanged ? "true" : "false")
      << ",\"R2_retry_same_carrier_slot\":" << (batch_retry_same_slot ? "true" : "false")
      << ",\"blocked_retirement_checks\":" << (cancel_tested ? 1 : 0)
      << ",\"cancel_versions_before\":" << cancel_versions_before << ",\"cancel_versions_blocked\":" << cancel_versions_blocked
      << ",\"cancel_versions_retired\":" << cancel_versions_retired << ",\"managed_backing_bytes\":" << cancel_backing_bytes
      << ",\"retirement_version_scope\":\"Before/retired counts include two explicitly retained old snapshot oracles. Shared physical arena/pool backing remains owned and charged.\""
      << ",\"cancel_means_native_preemption\":false,\"native_error_injected\":false,\"numerical_tolerance_policy\":null"
      << ",\"numerical_scope\":\"No full-model B1-versus-B2 error budget exists here. Metrics and raw38vectors per pair are reported; no byte-equivalence or bounded-error claim is inferred from finite values and equal greedyIDs.\"";
    const auto ids = [&](const char *name, const auto &values) {
      stream << ",\"" << name << "\":[";
      for (std::size_t i = 0; i < values.size(); ++i) stream << (i ? "," : "") << values[i];
      stream << ']';
    };
    ids("teacher_forced_ids", teacher_forced); ids("baseline_ids", baseline_ids); ids("batch_and_continuation_ids", batch_ids); ids("retired_R2_retry_ids", retry_ids);
    stream << ",\"vectors\":[";
    for (std::size_t i = 0; i < metrics.size(); ++i) {
      const auto &value = metrics[i];
      stream << (i ? "," : "") << "{\"name\":" << json_string(value.name) << ",\"baseline\":" << json_string(value.baseline_path)
        << ",\"batch\":" << json_string(value.batch_path) << ",\"dtype\":" << json_string(value.dtype)
        << ",\"bytes\":" << value.bytes << ",\"elements\":" << value.elements
        << ",\"byte_equal\":" << (value.byte_equal ? "true" : "false") << ",\"nonfinite_baseline\":" << value.nonfinite_baseline
        << ",\"nonfinite_batch\":" << value.nonfinite_batch << ",\"max_abs\":" << value.max_abs
        << ",\"rmse\":" << std::sqrt(value.squared_error / static_cast<double>(value.elements))
        << ",\"max_relative_nonzero_reference\":" << value.max_relative_nonzero_reference
        << ",\"zero_reference_mismatches\":" << value.zero_reference_mismatches << ",\"cosine\":";
      if (value.reference_squared && value.batch_squared) stream << value.dot / std::sqrt(value.reference_squared * value.batch_squared);
      else stream << "null";
      stream << '}';
    }
    stream << "]}\n"; require(static_cast<bool>(stream), "cannot finish batch probe observation");
  }
};

struct SavedSnapshot final {
  q::cache::SnapshotRef head;
  std::vector<Bytes> kv, recurrent;
};

SavedSnapshot save(q::cache::SnapshotRef head) {
  require(static_cast<bool>(head), "missing committed snapshot"); SavedSnapshot result; result.head = std::move(head);
  for (std::size_t i = 0; i < result.head->keys.size(); ++i)
    for (const auto *frame : {&result.head->keys[i], &result.head->values[i]})
      for (const auto &guard : c::detail::PagedArenaAccess::pages(*frame)) result.kv.push_back(read_lease(guard));
  for (const auto &state : result.head->state)
    for (const auto *lease : {&state.history, &state.recurrent}) if (lease->valid()) result.recurrent.push_back(read_lease(*lease));
  require(result.recurrent.size() == 36, "snapshot is missing a conv/GDN resource"); return result;
}

bool unchanged(const SavedSnapshot &old) {
  const auto current = save(old.head); return current.kv == old.kv && current.recurrent == old.recurrent;
}

bool disjoint(const c::ReadLease &a, const c::ReadLease &b) {
  const auto x = q::take(a.view()), y = q::take(b.view());
  return x.resource() != y.resource() || x.native_handle() != y.native_handle() ||
      q::cache::add(x.offset(), x.size()) <= y.offset() || q::cache::add(y.offset(), y.size()) <= x.offset();
}

void check_frames(const q::cache::Snapshot &a, const q::cache::Snapshot &b) {
  require(a.length == 131 && b.length == 132, "B2 publication used a common position instead of individual histories");
  require(a.pages.back().page() != b.pages.back().page(), "batch requests share a dirty physical tail");
  for (const auto *head : {&a, &b}) {
    for (std::size_t bank = 0; bank < head->keys.size(); ++bank)
      for (const auto *frame : {&head->keys[bank], &head->values[bank]}) {
        const auto layout = frame->layout(); const auto host = layout.host();
        require(layout.facts().requests == 1 && layout.facts().query_rows == 1, "public Session frame still exposes the packed batch");
        require(host.page_indices.size() == head->pages.size(), "projected frame has another Session's footprint");
        for (std::size_t i = 0; i < head->pages.size(); ++i)
          require(host.page_indices[i] == static_cast<std::int32_t>(head->pages[i].page()), "projected frame mapping differs from its Session inventory");
      }
  }
}

void compare_case(Observation &observation, q::Model &model, std::string_view baseline, std::string_view batch) {
  const auto root = observation.output / "trace";
  const auto add = [&](std::string filename, std::size_t bytes, bool bf16) {
    auto value = compare(root / baseline / filename, root / batch / filename,
        std::string(baseline) + "_vs_" + std::string(batch) + "/" + filename, bytes, bf16);
    observation.finite = observation.finite && value.nonfinite_baseline == 0 && value.nonfinite_batch == 0;
    observation.metrics.push_back(std::move(value));
  };
  add("hidden.bf16", static_cast<std::size_t>(model.geometry.hidden) * sizeof(std::uint16_t), true);
  add("logits.f32", static_cast<std::size_t>(model.geometry.vocabulary) * sizeof(float), false);
  for (std::size_t layer = 0; layer < model.state_schema.size(); ++layer) if (layer % 4 != 3) {
    const auto prefix = "layer" + std::to_string(layer);
    add(prefix + ".conv.bf16", q::take(model.state_schema[layer].history.required_bytes()), true);
    add(prefix + ".gdn.f32", q::take(model.state_schema[layer].recurrent.required_bytes()), false);
  }
  observation.write("RUNNING");
}

void cancel_batch(Observation &observation, q::Model &model,
                  std::shared_ptr<q::Session> &a, const std::shared_ptr<q::Session> &b,
                  std::int32_t id_a, std::int32_t id_b) {
  const auto old_a = save(a->head), old_b = save(b->head); // Read only before inserting the gate.
  observation.cancel_versions_before = model.snapshot_versions.live;
  const auto slot = model.slot_index;
  std::array requests{q::EvaluationRequest{a, std::span(&id_a, 1)}, q::EvaluationRequest{b, std::span(&id_b, 1)}};
  model.prepare_evaluations(requests); // All uploads, planning and capture precede the blocker.
  auto *const episode = model.active_step.get();
  require(episode && episode->members.size() == 2 && a->pending == episode && b->pending == episode,
          "cancel case did not prepare one genuine two-member episode");
  auto *const phase = episode->phase; auto *const run = episode->run;
  require(phase && phase->slots.size() == 1 && slot == 0 && run &&
          run == &phase->slots[slot] && run->valid(), "cancel case lost its original serial R2 carrier slot");
  std::array<std::weak_ptr<const q::cache::Snapshot>, 2> candidates{episode->members[0].next, episode->members[1].next};
  const std::array candidate_ids{episode->members[0].next.get(), episode->members[1].next.get()};
  observation.cancel_versions_blocked = model.snapshot_versions.live;
  require(observation.cancel_versions_blocked == observation.cancel_versions_before + 2,
          "two admitted candidates are not both charged");
  observation.cancel_backing_bytes = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;
  require(observation.cancel_backing_bytes == model.cache_plan.backing_bytes, "managed batch backing differs from its explicit physical plan");
  NativeGate gate;
  ReleaseAndDrain cleanup{gate, *q::take(model.execution.common_context())};
  auto blocked = q::take(c::cuda::submit(*q::take(model.execution.common_queue()), [&](cudaStream_t stream) noexcept {
    return cudaLaunchHostFunc(stream, &NativeGate::wait, &gate);
  }));
  gate.wait_until_entered(); model.submit_evaluations();
  require(a->head == old_a.head && b->head == old_b.head, "batch submission prematurely changed a head");
  q::check(a->stop());
  require(a->stopping && !b->stopping && episode->cancelled, "stopping one participant did not cancel its whole episode");
  observation.dropped_live_completion = episode->completion.valid();
  require(observation.dropped_live_completion, "batch had no original live public Completion to abandon");
  episode->completion = {}; // Original Graph/native uses still own both candidates.
  require(!q::take(a->try_retire()), "native-blocked batch retired after public Completion was dropped"); // Exactly one blocked check.
  require(model.active_step.get() == episode && a->pending == episode && b->pending == episode &&
          !candidates[0].expired() && !candidates[1].expired() &&
          episode->members[0].next.get() == candidate_ids[0] && episode->members[1].next.get() == candidate_ids[1] &&
          a->head == old_a.head && b->head == old_b.head && model.carrier_borrowed &&
          model.snapshot_versions.live == observation.cancel_versions_blocked &&
          q::take(model.cache_budget.usage(model.execution.device())).live_bytes == observation.cancel_backing_bytes,
          "blocked batch returned a candidate, head, version charge, backing or carrier early");
  observation.pending_retained = true;
  // No device reads while the callback blocks its Queue. Only real native
  // retirement, not cancellation/terminal flags or sleep, permits collection.
  gate.release(); q::check(blocked.wait()); blocked = {};
  const auto native_drained = q::take(model.execution.common_context())->drain();
  if (!native_drained.ok() && !native_drained.IsCancelled()) q::check(native_drained);
  const auto drained = run->drain(); if (!drained.ok() && !drained.IsCancelled()) q::check(drained);
  require(q::take(a->try_retire()), "released original batch did not pass its actual typed retirement gates");
  require(!a->pending && !b->pending && !model.active_step && !model.carrier_borrowed && !a->head &&
          b->head == old_b.head && candidates[0].expired() && candidates[1].expired() && !run->valid(),
          "whole-batch abort retained candidates, published B or kept stopped A's head");
  observation.cancel_versions_retired = model.snapshot_versions.live;
  require(observation.cancel_versions_retired == observation.cancel_versions_before,
          "retired candidate versions did not return to the explicit old-head count");
  require(q::take(model.cache_budget.usage(model.execution.device())).live_bytes == observation.cancel_backing_bytes,
          "retirement incorrectly returned the still-owned shared physical backing budget");
  observation.cancelled_heads_unchanged = unchanged(old_a) && unchanged(old_b);
  require(observation.cancelled_heads_unchanged, "batch cancellation modified a committed old page/state byte");
  const std::array<std::size_t, 1> calls{1}; auto retry_a = model.create_session(calls, old_a.head);
  requests[0].session = retry_a;
  model.prepare_evaluations(requests);
  require(model.active_step && model.active_step->phase == phase && model.active_step->run == run &&
          run == &phase->slots[slot] && run->valid(), "R2 retry did not rebuild the original aborted carrier slot");
  model.submit_evaluations();
  const std::array outputs_io{q::EvaluationIO{.name = "retired_retry_a"},
      q::EvaluationIO{.name = "retired_retry_b"}};
  const std::array outputs{q::evaluation_output(model, retry_a, outputs_io[0]),
      q::evaluation_output(model, b, outputs_io[1])};
  model.finish_evaluations(outputs, observation.retry_ids, [&](std::size_t j, const q::CompletedEvaluationView &view) {
        q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
      });
  require(retry_a->length() == old_a.head->length + 1 && b->length() == old_b.head->length + 1 &&
          !retry_a->pending && !b->pending && !model.active_step && !model.carrier_borrowed,
          "retired R2 retry failed to publish its two independent new heads");
  require(unchanged(old_a) && unchanged(old_b), "retired R2 retry changed its retained old prefixes");
  require(model.snapshot_versions.live <= model.cache_plan.resident_sessions, "retired R2 retry exceeded N8 committed versions");
  a = std::move(retry_a);
  observation.batch_retry_same_slot = true; observation.cancel_tested = true;
  std::cout << "Qwen real B2 native-blocked whole-batch cancel/drop-Completion, actual retirement and same-slot R2 retry PASS\n";
}

void run(const std::filesystem::path &inputs, const std::filesystem::path &output) {
  std::filesystem::create_directories(output);
  require(!std::filesystem::exists(output / "observation.json") && !std::filesystem::exists(output / "trace"),
          "batch probe output already contains results");
  Observation observation; observation.output = output;
  try {
    const auto prompt = q::read_ids(inputs / "prompt.ids");
    require(prompt.size() == 128, "batch probe needs original P128 input");
    for (std::size_t i = 0; i < observation.teacher_forced.size(); ++i) observation.teacher_forced[i] = prompt[i];
    const auto &ids = observation.teacher_forced;
    constexpr std::size_t capacity = 256, residents = 8, requests = 2;
    const auto geometry = q::probe::read_two_b_geometry(inputs);
    const auto plan = q::cache::plan(geometry, capacity, residents, prompt.size(), q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), requests);
    q::Model model(inputs / "weights.tsv", geometry, capacity, plan, {}, false,
        q::ReplayPolicy::kAuto, SIZE_MAX, {false, {}, 0}, 0,
        {.primary=hyle::inference::BackendKind::kCuda});
    const std::array<std::size_t, 2> chunks{128, 1}; const std::array<std::size_t, 1> singleton{1};
    model.prepare_family(chunks, false, 1); model.prepare_family(singleton, false, requests);
    const std::array<std::size_t, 11> b1_calls{128,1,1,1,1,1,1,1,1,1,1};
    const std::array<std::size_t, 3> b2_calls{1,1,1}; // Initial math, cancelled episode, retired retry.
    model.plan_execution(b1_calls, 1); model.plan_execution(b2_calls, requests);
    const auto trace = output / "trace";
    auto seed = model.create_session(chunks);
    (void)q::evaluate(model, *seed, prompt, "seed_prefill");
    (void)q::evaluate(model, *seed, std::span(&ids[0], 1), "seed129");
    require(seed->length() == 129, "shared seed is not P128 plus one T1");
    auto origin_a = model.create_session(singleton, seed->head);
    const std::array<std::size_t, 2> twice{1,1}; auto origin_b = model.create_session(twice, seed->head);
    (void)q::evaluate(model, *origin_a, std::span(&ids[1], 1), "origin_a130");
    (void)q::evaluate(model, *origin_b, std::span(&ids[2], 1), "origin_b130");
    (void)q::evaluate(model, *origin_b, std::span(&ids[3], 1), "origin_b131");
    std::array old{save(seed->head), save(origin_a->head), save(origin_b->head)};
    auto baseline_a = model.create_session(twice, origin_a->head), baseline_b = model.create_session(twice, origin_b->head);
    auto batch_a = model.create_session(twice, origin_a->head), batch_b = model.create_session(twice, origin_b->head);
    observation.baseline_ids[0] = q::evaluate(model, *baseline_a, std::span(&ids[4], 1), "b1_a", trace);
    observation.baseline_ids[1] = q::evaluate(model, *baseline_b, std::span(&ids[5], 1), "b1_b", trace);
    std::array evaluations{q::EvaluationRequest{batch_a, std::span(&ids[4], 1)}, q::EvaluationRequest{batch_b, std::span(&ids[5], 1)}};
    model.prepare_evaluations(evaluations);
    require(model.active_step && model.active_step->members.size() == requests && batch_a->pending == model.active_step.get() && batch_b->pending == model.active_step.get(),
        "B2 prepare did not retain one original shared episode");
    require(model.active_step->phase && model.active_step->phase->family->requests == requests &&
        model.active_step->phase->attention_layout.facts().requests == requests && model.active_step->phase->attention_layout.facts().query_rows == requests,
        "prepared native Graph is not actual B2/T1");
    require(batch_a->head == origin_a->head && batch_b->head == origin_b->head, "prepare prematurely published a candidate head");
    observation.one_batch_episode = true;
    model.submit_evaluations();
    const std::array outputs_io{q::EvaluationIO{.name = "b2_a", .trace = &trace},
        q::EvaluationIO{.name = "b2_b", .trace = &trace}};
    std::array outputs{q::evaluation_output(model, batch_a, outputs_io[0]),
        q::evaluation_output(model, batch_b, outputs_io[1])};
    std::array<std::int32_t, 2> batch_ids{-1,-1}; model.finish_evaluations(outputs, batch_ids, [&](std::size_t j, const q::CompletedEvaluationView &view) {
      q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
    });
    observation.batch_ids[0] = batch_ids[0]; observation.batch_ids[1] = batch_ids[1];
    require(!model.active_step && !batch_a->pending && !batch_b->pending, "finished batch did not return its original execution loan");
    check_frames(*batch_a->head, *batch_b->head); observation.independent_frames = true;
    for (std::size_t layer = 0; layer < model.state_schema.size(); ++layer) if (layer % 4 != 3) {
      require(disjoint(batch_a->head->state[layer].history, batch_b->head->state[layer].history) &&
          disjoint(batch_a->head->state[layer].recurrent, batch_b->head->state[layer].recurrent), "two Sessions share overlapping writable recurrent results");
    }
    observation.independent_states = true;
    compare_case(observation, model, "b1_a", "b2_a"); compare_case(observation, model, "b1_b", "b2_b");
    observation.baseline_ids[2] = q::evaluate(model, *baseline_a, std::span(&ids[6], 1), "b1_a_continue", trace);
    observation.batch_ids[2] = q::evaluate(model, *batch_a, std::span(&ids[6], 1), "b2_a_continue", trace);
    observation.baseline_ids[3] = q::evaluate(model, *baseline_b, std::span(&ids[6], 1), "b1_b_continue", trace);
    observation.batch_ids[3] = q::evaluate(model, *batch_b, std::span(&ids[6], 1), "b2_b_continue", trace);
    compare_case(observation, model, "b1_a_continue", "b2_a_continue"); compare_case(observation, model, "b1_b_continue", "b2_b_continue");
    observation.continued_independently = batch_a->length() == 132 && batch_b->length() == 133;
    observation.old_unchanged = std::all_of(old.begin(), old.end(), [](const auto &value) { return unchanged(value); });
    observation.ids_equal = observation.baseline_ids == observation.batch_ids;
    observation.write("COMPLETE");
    require(model.snapshot_versions.live <= residents, "retained committed snapshots exceed the explicit N8 budget");
    require(observation.old_unchanged && observation.continued_independently, "batch/continuation changed an old prefix or another Session's position");
    require(observation.ids_equal && observation.finite, "B1/B2 greedy IDs differ or raw model/state output is nonfinite; see unchanged raw vectors and metrics");
    require(observation.metrics.size() == 152, "did not compare hidden/logits and all36states for four same-partition pairs");
    // The numeric comparisons and original three old-byte oracles are now
    // complete. Release these no-longer-needed real owners before the one
    // extra lifetime case; do not hide their cost with a larger/unbounded N.
    old = {}; seed.reset(); origin_a.reset(); origin_b.reset(); baseline_a.reset(); baseline_b.reset();
    cancel_batch(observation, model, batch_a, batch_b, ids[4], ids[5]);
    observation.write("PASS structural isolation, finite outputs and greedy IDs; numeric metrics only");
    std::cout << "Qwen real B2/T1 finite/ID/isolation and independent B1 continuation PASS; numerical error metrics only, no approved cross-batch tolerance\n";
  } catch (const std::exception &error) {
    try { observation.write("FAIL", error.what()); } catch (...) {}
    throw;
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path inputs, output;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("batch probe option lacks its value");
      const std::string_view option(argv[i]);
      if (option == "--inputs") inputs = argv[i + 1];
      else if (option == "--output") output = argv[i + 1];
      else throw std::runtime_error("unknown batch probe option");
    }
    if (inputs.empty() || output.empty()) throw std::runtime_error("usage: hyle_qwen35_batch_probe --inputs DIR --output DIR");
    run(inputs, output); return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen batch probe: " << error.what() << '\n'; return 1;
  }
}
