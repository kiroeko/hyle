// Actual native MTP retained-state quota rejection and original-path retry.
// This consumer intentionally runs no timing or model-quality comparison.
#include "main.cc"
#include "probe_geometry.h"


namespace {
namespace q = qwen35;
namespace c = q::c;
using Bytes = std::vector<std::byte>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
std::string quoted(std::string_view value) {
  std::ostringstream out; out << '"';
  for (const auto ch : value) {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch == '\n') out << "\\n";
    else if (ch == '\r') out << "\\r";
    else if (ch == '\t') out << "\\t";
    else if (static_cast<unsigned char>(ch) < 32) out << '?';
    else out << ch;
  }
  out << '"'; return out.str();
}
std::string fingerprint(const Bytes &bytes) {
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto value : bytes) {
    hash ^= std::to_integer<unsigned char>(value); hash *= 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
Bytes read(const c::ReadLease &lease) {
  if (!lease.valid()) return {};
  Bytes bytes(lease.size());
  q::check(q::take(lease.view()).read(bytes));
  return bytes;
}

// Inspect only the committed logical rows authorized by the original frame.
// Physical tails outside that read range are not part of this comparison.
Bytes logical_rows(const c::PagedArena &arena, const c::PagedArenaFrame &frame,
                   std::size_t rows) {
  const auto view = q::take(frame.view()); const auto layout = view.layout().host();
  const auto count = (rows + q::cache::Plan::page_rows - 1) / q::cache::Plan::page_rows;
  require(frame.valid() && layout.page_indices.size() == count &&
          frame.page_count() == count && layout.last_page_lengths.size() == 1,
          "committed KV frame has an invalid page footprint");
  require(!count || (count - 1) * q::cache::Plan::page_rows +
          static_cast<std::size_t>(layout.last_page_lengths[0]) == rows,
          "committed KV frame frontier differs from its complete Snapshot");
  require(arena.owns(view), "committed KV frame belongs to another arena");
  Bytes bytes(q::cache::multiply(rows, std::size_t{1024}));
  std::size_t offset = 0;
  for (std::size_t logical = 0; logical < count; ++logical) {
    const auto selected = q::take(view.page_prefix(static_cast<std::uint32_t>(layout.page_indices[logical])));
    const auto length = (std::min)(std::size_t{16 * 1024}, bytes.size() - offset);
    q::check(q::take(q::take(selected.view()).subview(0, length)).read(
        std::span(bytes).subspan(offset, length)));
    offset += length;
  }
  return bytes;
}
struct Blob final { std::string name; Bytes bytes; };
struct HeadImage final {
  q::GenerationBoundary boundary;
  const void *model = nullptr, *math = nullptr;
  std::array<c::BatchLayoutVersion, 14> layouts;
  std::vector<std::uint32_t> target_pages, draft_pages;
  std::vector<Blob> blobs;
};
HeadImage image(q::Model &model, const q::cache::State &head) {
  model.validate_mtp_snapshot(head);
  require(head.draft && head.carry_hidden.valid(), "native MTP Snapshot is incomplete");
  HeadImage result;
  result.boundary = q::generation_boundary(head); result.model = head.model; result.math = head.math;
  for (const auto &page : head.pages) result.target_pages.push_back(page.page());
  for (const auto &page : head.draft->pages) result.draft_pages.push_back(page.page());
  for (std::size_t layer = 0; layer < head.state.size(); ++layer) if (layer % 4 != 3) {
    result.blobs.push_back({"layer" + std::to_string(layer) + ".conv.bf16", read(head.state[layer].history)});
    result.blobs.push_back({"layer" + std::to_string(layer) + ".gdn.f32", read(head.state[layer].recurrent)});
  }
  for (std::size_t layer = 0; layer < head.keys.size(); ++layer) {
    result.layouts[layer * 2] = head.keys[layer].layout();
    result.layouts[layer * 2 + 1] = head.values[layer].layout();
    result.blobs.push_back({"target" + std::to_string(layer) + ".key.bf16",
        logical_rows(model.page_inventory.keys[layer], head.keys[layer], head.length)});
    result.blobs.push_back({"target" + std::to_string(layer) + ".value.bf16",
        logical_rows(model.page_inventory.values[layer], head.values[layer], head.length)});
  }
  result.layouts[12] = head.draft->key.layout(); result.layouts[13] = head.draft->value.layout();
  result.blobs.push_back({"draft.key.bf16", logical_rows(model.draft_inventory.keys[0], head.draft->key, head.draft->frontier)});
  result.blobs.push_back({"draft.value.bf16", logical_rows(model.draft_inventory.values[0], head.draft->value, head.draft->frontier)});
  result.blobs.push_back({"carry.bf16", read(head.carry_hidden)});
  result.blobs.push_back({"prior.bf16", read(head.prior_hidden)});
  require(result.blobs.size() == 52, "complete MTP comparison must contain 36 states, 14 KV ranges and two hidden rows");
  return result;
}
bool same_payload(const HeadImage &a, const HeadImage &b) {
  if (a.boundary != b.boundary || a.model != b.model || a.math != b.math || a.blobs.size() != b.blobs.size()) return false;
  for (std::size_t i = 0; i < a.blobs.size(); ++i)
    if (a.blobs[i].name != b.blobs[i].name || a.blobs[i].bytes != b.blobs[i].bytes) return false;
  return true;
}
bool unchanged(const HeadImage &a, const HeadImage &b) {
  return same_payload(a, b) && a.layouts == b.layouts &&
      a.target_pages == b.target_pages && a.draft_pages == b.draft_pages;
}
void retire(std::shared_ptr<q::Session> &session) {
  q::check(session->stop());
  const auto observed = session->try_retire(); q::check(observed.status);
  require(observed.retired, "completed native Session did not retire");
  session.reset();
}
q::GraphRole round_role(const q::Session &session) {
  return session.head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
}
q::MtpRoundResult finish(q::Model &model, const std::shared_ptr<q::Session> &session,
                       std::int32_t pending, q::GenerationLedger &ledger,
                       std::string_view name, const q::GenerationCommitRecord *expected = nullptr) {
  const q::EvaluationIO output_io{.name = name, .trace_states = false};
  const auto output = q::evaluation_output(model, session, output_io, nullptr, nullptr, nullptr, nullptr, &ledger, expected);
  return q::finish_mtp_round(model, *session, pending, 2,
      q::model_eos_tokens, &output, output_io);
}

void run(const std::filesystem::path &inputs, const std::filesystem::path &output) {
  hyle::inference::BackendSelection selection;
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
  selection.primary = hyle::inference::BackendKind::kApple;
#elif defined(HYLE_COMPUTE_VULKAN)
  selection.primary = hyle::inference::BackendKind::kVulkan;
#else
  throw std::runtime_error("retained-budget probe requires an Apple/Vulkan backend");
#endif
  auto prompt = q::read_ids(inputs / "prompt.ids");
  require(prompt.size() >= 15, "retained-budget probe needs the original first15 prompt IDs");
  prompt.resize(15);
  require(!std::filesystem::exists(output), "retained-budget probe output directory must be fresh");
  std::filesystem::create_directories(output);
  constexpr std::size_t capacity = 64, limit = 256u * 1024u * 1024u;
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, capacity, 1, prompt.size(),
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 1, 2, true);
  // This is an explicit finite admission limit, not the CUDA pool formula or
  // a claim that native execution/private peak allocations fit in 256 MiB.
  q::Model model(std::move(weight_source), geometry, capacity, plan, {}, false,
      q::ReplayPolicy::kAuto, limit, {false, {}, 0}, 0, selection);
  require(model.execution.native_generated_storage(),
          "retained-budget probe requires a native-generated-storage backend");
  require(model.weights.size() == 335, "retained-budget probe needs all335 real target/MTP weights");
  const std::array prefill{std::size_t{15}}, pair{std::size_t{2}}, single{std::size_t{1}};
  model.prepare_family(prefill, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(15, 1, 1, q::GraphRole::kMtpPrefill);
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound,
                         q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
    model.prepare_family(q::drafts_token(role) ? std::span(pair) : std::span(single), false, 1, role);
    model.plan_replay_uses(q::drafts_token(role) ? 2u : 1u, 1, 8, role);
  }
  auto session = model.create_session(prefill, {}, q::GraphRole::kMtpPrefill);
  q::GenerationLedger ledger(8);
  q::prepare_generation_record(ledger, *session, q::GraphRole::kMtpPrefill, prompt, false, 0, 0);
  model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluation(*session);
  (void)q::finish_evaluation(model, *session, "retained_budget_prefill", {}, false,
      nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  std::size_t seed_rounds = 0;
  // The real first15 prompt reaches lag1 after its first accepted K1 round.
  // Keep a small finite fallback without imposing hard-coded token IDs.
  while (!session->head->draft->lag && seed_rounds < 3) {
    const auto pending = session->head->next_id;
    require(!q::is_end_of_generation(pending), "native seed stopped before a natural accepted draft");
    const auto role = round_role(*session);
    q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 2, 0);
    model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role);
    model.submit_evaluation(*session);
    (void)finish(model, session, pending, ledger, "retained_budget_seed"); ++seed_rounds;
  }
  require(session->head->draft->lag == 1 && session->head->prior_hidden.valid(),
          "finite native seed did not produce lag1 with both carry and prior");
  auto old = session->checkpoint();
  const auto saved = image(model, *old);
  const auto pending = old->next_id;
  require(!q::is_end_of_generation(pending), "lag1 seed pending token is EOS");
  const auto role = round_role(*session);

  // Establish an actual same-Snapshot, same-role reference before pressure.
  auto reference = model.create_session(pair, old, role);
  q::GenerationLedger reference_ledger(1);
  q::prepare_generation_record(reference_ledger, *reference, role, std::span(&pending, 1), true, 2, 0);
  model.prepare_evaluation(*reference, std::span(&pending, 1), nullptr, role);
  model.submit_evaluation(*reference);
  const auto expected_round = finish(model, reference, pending, reference_ledger, "retained_budget_reference");
  const auto expected = image(model, *reference->head);
  require(reference_ledger.committed_count == 1 && reference_ledger.records[0].committed,
          "unpressured original-path reference did not commit its ledger");
  require(unchanged(saved, image(model, *old)), "reference branch changed old complete Snapshot bytes or metadata");
  retire(reference);
  require(model.snapshot_versions.live == 1, "reference retained a second Snapshot version after retirement");

  const auto committed_before = ledger.committed_count;
  const auto committed_records = ledger.records;
  const auto slot = model.slot_index;
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 2, 0);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role);
  require(session->pending && model.active_step.get() == session->pending && model.carrier_borrowed,
          "pressure episode did not borrow the original Model carrier");
  auto *const phase = session->pending->phase;
  auto *const invocation = session->pending->run;
  const auto *const candidate = session->pending->members[0].next.get();
  const auto *const episode = session->pending;
  require(candidate && model.active_step.get() == episode && phase->slot_count == 1 && slot == 0 && invocation == q::take(phase->carriers[slot].borrow()) && model.snapshot_versions.live == 2 &&
          std::count_if(session->pending->bound.begin(), session->pending->bound.end(), [](auto value) { return static_cast<bool>(value); }) == 14,
          "pressure episode did not retain its original slot/candidate/all14 append permissions");
  model.submit_evaluation(*session);
  // Prove native completion separately. The original finish still owns and
  // waits its same Completion, and performs all export/selection/publication.
  q::check(session->pending->completion.wait());
  require(model.active_step.get() == episode && session->pending == episode &&
          episode->members[0].next.get() == candidate && model.snapshot_versions.live == 2,
          "native completion discarded the original unpublished candidate before retention admission");
  const auto before_pressure = model.cache_usage();
  require(before_pressure.limit_bytes == limit && before_pressure.live_bytes < limit,
          "finite retained budget is unexpectedly full before synthetic quota pressure");
  const auto pressure_bytes = limit - before_pressure.live_bytes;
  auto pressure = q::take(model.cache_budget.reserve(model.execution.device(), pressure_bytes));
  require(model.cache_usage().live_bytes == limit, "quota-only pressure did not fill the exact native resource limit");
  std::string rejection;
  try { (void)finish(model, session, pending, ledger, "retained_budget_rejected"); }
  catch (const std::exception &error) { rejection = error.what(); }
  require(rejection.find("cache retained-state admission failed before snapshot publication:") != std::string::npos,
          "completed native episode did not reject specifically at retained-state export admission");
  require(session->head.get() == old.get() && !session->pending && !session->failed && !session->stopping &&
          !model.active_step && !model.carrier_borrowed &&
          model.snapshot_versions.live == 1 &&
          !invocation->valid() && !invocation->in_flight(),
          "retention rejection published a head or retained the candidate/slot/admission");
  require(ledger.committed_count == committed_before && !ledger.records[committed_before].committed,
          "retention rejection published the reserved replay ledger entry");
  for (std::size_t i = 0; i < committed_before; ++i)
    require(ledger.records[i].committed && ledger.records[i].same_result(committed_records[i]),
            "retention rejection changed an earlier committed replay ledger record");
  require(unchanged(saved, image(model, *old)), "quota rejection changed old whole Snapshot bytes or metadata");
  const auto after_rejection_with_pressure = model.cache_usage().live_bytes;
  pressure.reset();
  const auto after_release = model.cache_usage().live_bytes;
  require(after_release == before_pressure.live_bytes,
          "retired rejected exports leaked a native retained-state budget charge");

  // Retry the same original Session, Snapshot, role and Model slot. This
  // prepares the retired failed Invocation in the existing Model slot.
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 2, 0);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role);
  require(session->pending->phase == phase && session->pending->run == invocation && invocation->valid() &&
          session->pending->members[0].base == old.get(),
          "unpressured retry did not rebuild the same slot against the original complete Snapshot");
  model.submit_evaluation(*session);
  const auto retried_round = finish(model, session, pending, ledger, "retained_budget_retry", &reference_ledger.records[0]);
  require(retried_round.emitted == expected_round.emitted &&
          ledger.committed_count == committed_before + 1 && ledger.records[committed_before].committed &&
          ledger.records[committed_before].same_result(reference_ledger.records[0]) &&
          same_payload(expected, image(model, *session->head)),
          "unpressured retry differs from the original-path same-Snapshot reference");
  require(unchanged(saved, image(model, *old)) && session->head.get() != old.get() &&
          !session->pending && !model.active_step && !model.carrier_borrowed,
          "successful retry changed old Snapshot or retained a Model execution loan");
  const auto successful_usage = model.cache_usage();
  require(successful_usage.live_bytes > model.known_cache_backing_bytes,
          "successful native heads have no actual retained-output backing charge");
  const auto retained_output_bytes = successful_usage.live_bytes - model.known_cache_backing_bytes;
  const auto old_target_rows = old->length, old_draft_rows = old->draft->frontier;
  const auto old_lag = old->draft->lag;
  retire(session); old.reset();
  require(!model.active_step && !model.carrier_borrowed && model.snapshot_versions.live == 0,
          "final native consumer retirement retained a Snapshot or execution loan");

  std::ofstream report(output / "observation.json");
  require(static_cast<bool>(report), "cannot write retained-budget observation");
  report << "{\"status\":\"PASS_LIMITED_NATIVE_RETENTION_REJECTION_RETRY\",\"backend\":" << quoted(model.execution.name())
      << ",\"capacity\":64,\"resident_limit\":1,\"active_sequences\":1,\"cache_limit_bytes\":" << limit
      << ",\"seed_rounds\":" << seed_rounds << ",\"old_target_rows\":" << old_target_rows
      << ",\"old_draft_rows\":" << old_draft_rows << ",\"old_lag\":" << old_lag
      << ",\"complete_old_components\":52,\"old_whole_snapshot_unchanged_checks\":3"
      << ",\"native_completed_before_quota_pressure\":true,\"pressure_allocates_device_memory\":false"
      << ",\"quota_pressure_bytes\":" << pressure_bytes << ",\"bytes_before_pressure\":" << before_pressure.live_bytes
      << ",\"bytes_after_rejection_with_pressure\":" << after_rejection_with_pressure
      << ",\"bytes_after_pressure_release\":" << after_release
      << ",\"rejection\":" << quoted(rejection)
      << ",\"rejected_head_and_ledger_unpublished\":true,\"rejected_candidate_and_slot_retired\":true"
      << ",\"retry_same_model_phase_slot_base\":true,\"retry_matches_reference_whole_payload\":true"
      << ",\"cache_shared_backing_bytes\":" << model.known_cache_backing_bytes
      << ",\"actual_retained_output_backing_bytes\":" << retained_output_bytes
      << ",\"cache_live_requested_bytes\":" << successful_usage.live_bytes
      << ",\"cache_peak_requested_bytes\":" << successful_usage.peak_bytes
      << ",\"native_private_bytes_unknown\":true,\"physical_memory_cap_known\":false,\"timing_comparison\":false"
      << ",\"fingerprint\":\"FNV1a64 diagnostic; equality compares full byte arrays\",\"old_components\":[";
  for (std::size_t i = 0; i < saved.blobs.size(); ++i) {
    const auto &blob = saved.blobs[i];
    report << (i ? "," : "") << "{\"name\":" << quoted(blob.name) << ",\"bytes\":" << blob.bytes.size()
        << ",\"fnv1a64\":" << quoted(fingerprint(blob.bytes)) << '}';
  }
  report << "],\"reference_and_retry_emitted\":[";
  for (std::size_t i = 0; i < expected_round.emitted.size(); ++i)
    report << (i ? "," : "") << expected_round.emitted[i];
  report << "]}\n";
  require(static_cast<bool>(report), "retained-budget observation write failed");
  report.close();
  std::cout << "{\"status\":\"PASS_LIMITED_NATIVE_RETENTION_REJECTION_RETRY\",\"old_components\":52,"
      << "\"actual_retained_output_backing_bytes\":" << retained_output_bytes << ",\"timing_comparison\":false}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path inputs, output;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("retained-budget probe option lacks a value");
      const std::string_view option(argv[i]);
      if (option == "--inputs") inputs = argv[i + 1];
      else if (option == "--out") output = argv[i + 1];
      else throw std::runtime_error("unknown retained-budget probe option");
    }
    if (inputs.empty() || output.empty())
      throw std::runtime_error("usage: hyle_qwen35_retained_budget_probe --inputs DIR --out FRESH_DIR");
    run(inputs, output); return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen native retained-budget probe: " << error.what() << '\n'; return 1;
  }
}
