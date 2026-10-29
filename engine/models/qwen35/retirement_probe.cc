// Explicit, non-timing consumer of the same Model/Session implementation.
#include "main.cc"
#include "probe_geometry.h"

#include "probe_page_prefixes.h"
#include "native_gate.h"
#include <thread>

namespace {
namespace q = qwen35;
namespace c = q::c;
namespace cb = q::benchmark;
using NativeGate = q::probe::NativeGate;
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
constexpr auto native_backend = hyle::inference::BackendKind::kApple;
#elif defined(HYLE_COMPUTE_CUDA)
constexpr auto native_backend = hyle::inference::BackendKind::kCuda;
#elif defined(HYLE_COMPUTE_VULKAN)
constexpr auto native_backend = hyle::inference::BackendKind::kVulkan;
#else
#error "This native gate probe requires an Apple, CUDA or Vulkan endpoint"
#endif

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "retirement_probe_assertion_failed reason=" << message << '\n';
    throw std::runtime_error(message);
  }
}

// This guard dies before the gate, Session and Model on every exit path.
// Release the actual queue wait before original Graph/Model retirement. Closing
// a native Context would prevent the same Model retry, so drain the episode.
struct ReleaseAndDrain final {
  NativeGate &gate;
  q::Model &model;
  ~ReleaseAndDrain() {
    std::cerr << "native_gate_cleanup stage=release_begin\n";
    gate.release();
    std::cerr << "native_gate_cleanup stage=model_abort_begin\n";
    const auto observed = model.abort_episode();
    observed.status.IgnoreError(); // Cleanup never substitutes for a retirement proof.
    std::cerr << "native_gate_cleanup stage=gate_wait_begin\n";
    gate.wait().IgnoreError();
    std::cerr << "native_gate_cleanup stage=done\n";
  }
};

std::vector<std::byte> sample(const c::ReadLease &lease) {
  const auto bytes = (std::min)(lease.size(), std::size_t{64});
  std::vector<std::byte> result(bytes);
  auto view = q::take(lease.view());
  q::check(q::take(view.subview(0, bytes)).read(result));
  return result;
}

// Actual immutable guards, including their bounded views. This probe inspects
// internal permissions to verify retirement; ordinary consumers use the frame.
std::vector<std::vector<std::byte>> committed_pages(const q::cache::State &head) {
  std::vector<std::vector<std::byte>> result;
  const auto append = [&](const c::PagedArenaFrame &frame) {
    const auto view = q::take(frame.view());
    const auto guards = q::take(q::probe::unique_page_prefixes(view));
    for (const auto &lease : guards) {
      auto &bytes = result.emplace_back(lease.size());
      q::check(q::take(lease.view()).read(bytes));
    }
  };
  for (std::size_t i = 0; i < head.keys.size(); ++i) { append(head.keys[i]); append(head.values[i]); }
  if (head.draft) { append(head.draft->key); append(head.draft->value); }
  return result;
}

std::vector<std::vector<std::byte>> committed_recurrent(const q::cache::State &head) {
  std::vector<std::vector<std::byte>> result;
  for (const auto &state : head.state)
    for (const auto *lease : {&state.history, &state.recurrent}) if (lease->valid()) {
      auto &bytes = result.emplace_back(lease->size());
      q::check(q::take(lease->view()).read(bytes));
    }
  return result;
}

std::size_t head_lease_bytes(const q::cache::State &head, const q::Model &model) {
  std::size_t bytes = 0;
  for (const auto &state : head.state) {
    bytes = q::cache::add(bytes, state.history.size());
    bytes = q::cache::add(bytes, state.recurrent.size());
  }
  for (std::size_t i = 0; i < head.keys.size(); ++i) {
    bytes = q::cache::add(bytes, q::cache::multiply(head.keys[i].page_count(),
                                                model.page_inventory.keys[i].page_bytes()));
    bytes = q::cache::add(bytes, q::cache::multiply(head.values[i].page_count(),
                                                model.page_inventory.values[i].page_bytes()));
  }
  if (head.draft) {
    bytes = q::cache::add(bytes, q::cache::multiply(head.draft->key.page_count(), model.draft_inventory.keys[0].page_bytes()));
    bytes = q::cache::add(bytes, q::cache::multiply(head.draft->value.page_count(), model.draft_inventory.values[0].page_bytes()));
    bytes = q::cache::add(bytes, head.carry_hidden.size());
    bytes = q::cache::add(bytes, head.prior_hidden.size());
  }
  return bytes;
}

struct MtpSaved final {
  std::vector<std::vector<std::byte>> pages, recurrent, hidden;
  std::array<c::BatchLayoutVersion, 14> layouts;
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t next = -1, catchup = -1;
};

MtpSaved save_mtp(const q::cache::State &head) {
  require(head.draft && head.carry_hidden.valid(), "MTP probe needs a complete target/draft head");
  MtpSaved saved;
  saved.pages = committed_pages(head); saved.recurrent = committed_recurrent(head);
  for (std::size_t i = 0; i < head.keys.size(); ++i) {
    saved.layouts[i * 2] = head.keys[i].layout(); saved.layouts[i * 2 + 1] = head.values[i].layout();
  }
  saved.layouts[12] = head.draft->key.layout(); saved.layouts[13] = head.draft->value.layout();
  for (const auto *lease : {&head.carry_hidden, &head.prior_hidden}) if (lease->valid()) {
    auto &bytes = saved.hidden.emplace_back(lease->size());
    q::check(q::take(lease->view()).read(bytes));
  }
  saved.length = head.length; saved.next = head.next_id;
  saved.frontier = head.draft->frontier; saved.lag = head.draft->lag;
  saved.catchup = head.draft->catchup_token;
  return saved;
}

void unchanged_mtp(const q::cache::State &head, const MtpSaved &saved) {
  const auto now = save_mtp(head);
  require(now.pages == saved.pages && now.recurrent == saved.recurrent && now.hidden == saved.hidden && now.layouts == saved.layouts &&
          now.length == saved.length && now.next == saved.next && now.frontier == saved.frontier &&
          now.lag == saved.lag && now.catchup == saved.catchup,
          "retirement or selection failure changed an older complete MTP head");
}

void complete_mtp(q::Model &model, const q::Session &session) {
  require(session.head && session.head->draft, "MTP session lost its complete head");
  model.validate_mtp_snapshot(*session.head);
  const auto &head = *session.head;
  const auto exact = [](const c::PagedArenaFrame &frame, std::size_t rows,
                        const std::vector<q::cache::PageRef> &pages) {
    const auto layout = frame.layout().host();
    const auto count = (rows + q::cache::Plan::page_rows - 1) / q::cache::Plan::page_rows;
    require(count && frame.page_count() == count && pages.size() == count &&
            layout.page_indices.size() == count && layout.last_page_lengths.size() == 1 &&
            (count - 1) * q::cache::Plan::page_rows +
                static_cast<std::size_t>(layout.last_page_lengths[0]) == rows,
            "MTP ready frame exposes the wrong committed frontier");
    for (std::size_t i = 0; i < count; ++i)
      require(layout.page_indices[i] == static_cast<std::int32_t>(pages[i].page()),
              "MTP ready frame has another page inventory");
  };
  for (std::size_t i = 0; i < head.keys.size(); ++i) {
    exact(head.keys[i], head.length, head.pages); exact(head.values[i], head.length, head.pages);
  }
  exact(head.draft->key, head.draft->frontier, head.draft->pages);
  exact(head.draft->value, head.draft->frontier, head.draft->pages);
}

void run(const std::filesystem::path &inputs, const std::filesystem::path &tape_path) {
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  const auto prompt = q::read_ids(inputs / "prompt.ids");
  const auto tape = q::read_ids(tape_path);
  require(prompt.size() == 128 && tape.size() >= 2, "probe needs original P128 and at least2decode inputs");
  for (const auto id : prompt) require(id >= 0 && id < geometry.vocabulary, "prompt token outside vocabulary");
  for (std::size_t i = 0; i < 2; ++i)
    require(tape[i] >= 0 && tape[i] < geometry.vocabulary, "decode token outside vocabulary");

  constexpr std::size_t capacity = 256;
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", false);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, capacity, 1, prompt.size(),
                                  q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry));
  const bool progress = native_backend == hyle::inference::BackendKind::kApple ||
      native_backend == hyle::inference::BackendKind::kVulkan;
  q::Model model(std::move(weight_source), geometry, capacity, plan, {}, progress,
      q::ReplayPolicy::kAuto, SIZE_MAX, {false, {}, 0}, 0, {.primary=native_backend});
  const std::array<std::size_t, 3> calls{128, 1, 1};
  model.prepare_family(calls);
  model.plan_execution(calls);
  hyle::inference::SessionCache directory({1, (std::numeric_limits<std::size_t>::max)()});
  auto session = model.create_session(calls);
  auto admission = directory.admit(0, 0, [](std::size_t) { return hyle::inference::RetirementObservation{}; });
  q::check(admission.retirement_status);
  require(admission.accepted(), "initial owner was not admitted");
  auto pin = directory.pin(0);
  require(static_cast<bool>(pin), "initial owner could not pin");
  (void)q::evaluate(model, *session, prompt, "retirement_prefill");
  (void)q::evaluate(model, *session, std::span(&tape[0], 1), "retirement_decode1");
  auto old_prefix = session->checkpoint();
  require(old_prefix && old_prefix->length == 129, "committed prefix is not P128+D1");
  const auto old_conv = sample(old_prefix->state[0].history);
  const auto old_gdn = sample(old_prefix->state[0].recurrent);
  const auto old_kv = committed_pages(*old_prefix);
  const auto old_recurrent = committed_recurrent(*old_prefix);
  require(old_recurrent.size() == 36, "probe did not retain all36conv/GDN state resources");
  const auto old_pages = old_prefix->pages.size();
  const auto retained_logical_bytes = head_lease_bytes(*old_prefix, model);

  // These idle logical owners alias one committed version; they allocate no
  // additional snapshot. Only the original active owner is in the directory.
  const std::array<std::size_t, 1> resume{1};
  auto idle = model.create_session(resume, old_prefix);
  auto waiting = model.create_session(resume, old_prefix);

  // All recording/materialization happens before placing the native blocker.
  const auto borrowed_slot = model.slot_index;
  model.prepare_evaluation(*session, std::span(&tape[1], 1));
  require(session->pending && session->pending == model.active_step.get() &&
          session->pending->members.size() == 1 && session->pending->members[0].session == session &&
          session->pending->members[0].next && session->pending->members[0].base == old_prefix.get(),
          "prepared candidate lost its original Model episode/base");
  auto *const shared_phase = session->pending->phase;
  auto *const shared_run = session->pending->execution.run;
  require(shared_phase && shared_run && shared_run->valid() &&
          shared_phase->slots.size() == 1 && borrowed_slot == 0 &&
          shared_run == q::take(shared_phase->slots[borrowed_slot].carrier.borrow()),
          "prepared candidate did not borrow an actual Model execution slot");
  q::check(idle->stop());
  const auto idle_retirement = idle->try_retire(); q::check(idle_retirement.status);
  require(idle_retirement.retired, "idle logical Session did not retire");
  idle.reset();
  require(session->pending->phase == shared_phase && session->pending->execution.run == shared_run &&
          shared_run->valid() && model.carrier_borrowed,
          "stopping an idle Session disturbed another Session's shared execution loan");
  const auto *candidate_identity = session->pending->members[0].next.get();
  require(candidate_identity, "prepared episode has no unique unpublished candidate");
  const auto dirty_page = session->pending->members[0].next->pages.back().page();
  require(dirty_page == old_prefix->pages.back().page(),
          "probe did not enter actual same-page exclusive suffix continuation");
  require(model.snapshot_versions.live == 2 && model.carrier_borrowed,
          "prepared N+1candidate is not actually retained or admitted serially");
  const auto pending_bytes = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;

  NativeGate gate(model.execution);
  ReleaseAndDrain cleanup{gate, model};
  q::check(gate.arm()); q::check(gate.wait_until_entered());
  model.submit_evaluation(*session);
  q::check(gate.wait_for_model_submission());
  require(session->head.get() == old_prefix.get() && session->length() == 129,
          "submission prematurely published candidate state");
  std::size_t rejected_prepares = 0;
  const auto expect_borrow_busy = [&](const char *stage) {
    bool rejected = false;
    try { model.prepare_evaluation(*waiting, std::span(&tape[1], 1)); }
    catch (const std::exception &error) {
      rejected = true;
      std::cerr << "shared_execution_prepare_rejected stage=" << stage
                << " reason=" << error.what() << '\n';
    }
    require(rejected && !waiting->pending && !waiting->failed && !waiting->stopping &&
            waiting->head.get() == old_prefix.get() &&
            session->pending && session->pending == model.active_step.get() &&
            session->pending->phase == shared_phase && session->pending->execution.run == shared_run &&
            model.carrier_borrowed,
            "another Session acquired or changed a native-pending shared execution loan");
    ++rejected_prepares;
  };
  expect_borrow_busy("native_pending");
  require(directory.begin_retire(0), "owner could not enter retiring state");
  require(!directory.contains(0) && directory.retiring(0) && !directory.pin(0),
          "retiring owner remained visible or pinnable");
  pin.reset();
  q::check(session->stop()); // Unimplemented cancellation is not GPU preemption.
  expect_borrow_busy("stopped_native_pending");
  std::size_t collector_calls = 0;
  const auto collect = [&] {
    const auto observed = directory.collect_retired(0, [&](std::size_t owner) {
      require(owner == 0 && session, "collector lost the actual Session owner");
      ++collector_calls;
      const auto retired = session->try_retire();
      if (!retired.retired) return retired;
      require(!session->pending && !model.active_step,
              "retired Model episode left its borrowed Session pending link alive");
      session.reset();
      return retired;
    });
    if (!observed.status.IsCancelled() || observed.status.has_native_error()) q::check(observed.status);
    return observed;
  };
  require(!collect().retired, "collect reclaimed a native-blocked owner");
  const bool dropped_public_completion = session->pending->execution.completion.valid();
  // One externally serialized owner abandons the public observation. The
  // original Graph/native reaper must retain all actual resource uses.
  session->pending->execution.completion = {};
  expect_borrow_busy("dropped_completion_native_pending");
  require(!collect().retired, "dropping public Completion reclaimed native-blocked work");
  require(directory.retained_sessions() == 1 &&
          session->pending && session->pending->members[0].next.get() == candidate_identity &&
          session->head.get() == old_prefix.get() && model.snapshot_versions.live == 2 &&
          model.carrier_borrowed,
          "pending retirement released candidate, visible head or finite version charge");
  require(q::take(model.cache_budget.usage(model.execution.device())).live_bytes == pending_bytes,
          "blocked owner returned managed bytes before native retirement");
  const auto same_id = directory.admit(0, 0, [](std::size_t) { return hyle::inference::RetirementObservation{.retired = true}; });
  const auto other_id = directory.admit(1, 0, [](std::size_t) { return hyle::inference::RetirementObservation{.retired = true}; });
  q::check(same_id.retirement_status); q::check(other_id.retirement_status);
  require(same_id.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven &&
          other_id.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven,
          "resident limit1 readmitted a still-retiring owner or another owner");
  auto forbidden = model.page_inventory.keys[0].acquire(dirty_page);
  require(!forbidden && forbidden.error().IsUnavailable(),
          "pending dirty page was writable before real retirement");

  gate.release();
  q::check(gate.wait()); require(!gate.timed_out(), "native gate watchdog opened during assertions");
  const auto deadline = q::Clock::now() + std::chrono::seconds(30);
  while (!collect().retired) {
    require(q::Clock::now() < deadline, "native retirement did not become collectible");
    std::this_thread::yield();
  }
  require(!session && !directory.retiring(0) && directory.retained_sessions() == 0 &&
          directory.retained_bytes() == 0 &&
          model.snapshot_versions.live == 1 && !model.carrier_borrowed && !model.active_step &&
          !shared_run->valid(),
          "actual retirement did not release the candidate/owner");
  require(old_prefix->length == 129 && old_prefix->pages.size() == old_pages &&
          sample(old_prefix->state[0].history) == old_conv &&
          sample(old_prefix->state[0].recurrent) == old_gdn &&
          committed_pages(*old_prefix) == old_kv && committed_recurrent(*old_prefix) == old_recurrent,
          "cancellation changed the retained committed prefix");
  for (std::size_t i = 0; i < model.page_inventory.keys.size(); ++i) {
    auto key = model.page_inventory.keys[i].acquire(dirty_page);
    auto value = model.page_inventory.values[i].acquire(dirty_page);
    require(!key && key.error().IsUnavailable() && !value && value.error().IsUnavailable(),
            "retained old prefix lost its original page after suffix retirement");
  }
  admission = directory.admit(1, 0, [](std::size_t) { return hyle::inference::RetirementObservation{}; });
  q::check(admission.retirement_status);
  require(admission.accepted(), "retired directory slot was not reusable");
  auto next_pin = directory.pin(1);
  require(static_cast<bool>(next_pin), "replacement owner could not pin");
  auto recovered = std::move(waiting);
  std::weak_ptr<const q::cache::State> retired_prefix = old_prefix;
  old_prefix.reset(); // N1 keeps only the new Session's original committed head.
  model.prepare_evaluation(*recovered, std::span(&tape[1], 1));
  require(recovered->pending && recovered->pending->phase == shared_phase &&
          recovered->pending->execution.run == shared_run && shared_run->valid(),
          "the waiting Session did not rebuild the retired aborted slot in the original Model bundle");
  model.submit_evaluation(*recovered);
  const auto impossible_trace = inputs / "weights.tsv";
  require(std::filesystem::is_regular_file(impossible_trace), "probe trace rejection needs the original manifest file");
  bool post_commit_trace_rejected = false;
  try { (void)q::finish_evaluation(model, *recovered, "retirement_recovered", impossible_trace, false); }
  catch (const std::filesystem::filesystem_error &error) {
    post_commit_trace_rejected = true;
    std::cerr << "shared_execution_post_commit_trace_rejected reason=" << error.what() << '\n';
  }
  require(post_commit_trace_rejected && recovered->length() == 130 && !recovered->pending &&
          !recovered->failed && !model.carrier_borrowed && !model.active_step && retired_prefix.expired() &&
          model.snapshot_versions.live == 1,
          "post-commit diagnostic failure rolled back the head or retained its execution loan");

  // The reported IO failure follows the real head commit. Another logical owner
  // must be able to borrow the same Model carrier without resetting that head.
  auto after_report = model.create_session(resume, recovered->checkpoint());
  std::weak_ptr<const q::cache::State> report_head = recovered->checkpoint();
  q::check(recovered->stop());
  const auto report_retirement = recovered->try_retire(); q::check(report_retirement.status);
  require(report_retirement.retired, "reported-error Session did not retire its committed head alias");
  recovered.reset();
  const auto after_report_slot = model.slot_index;
  model.prepare_evaluation(*after_report, std::span(&tape[1], 1));
  require(after_report->pending && after_report->pending->phase == shared_phase &&
          shared_phase->slots.size() == 1 && after_report_slot == 0 &&
          after_report->pending->execution.run == q::take(shared_phase->slots[after_report_slot].carrier.borrow()),
          "a different Session did not recover the original serial Model slot");
  model.submit_evaluation(*after_report);
  (void)q::finish_evaluation(model, *after_report, "retirement_after_report_error");
  recovered = std::move(after_report);
  require(recovered->length() == 131 && !recovered->pending && !model.carrier_borrowed && !model.active_step &&
          report_head.expired() && model.snapshot_versions.live == 1,
          "a different Session could not continue after a committed diagnostic failure");
  require(retired_prefix.expired() &&
          model.snapshot_versions.live == 1,
          "retained old prefix was not a usable independent continuation");
  next_pin.reset();
  const auto live_drain = recovered->drain(); q::check(live_drain.status);
  require(!live_drain.retired && recovered->head, "draining an unstopped Session retired its committed head");
  recovered.reset();
  for (std::size_t i = 0; i < model.page_inventory.keys.size(); ++i) {
    auto key = q::take(model.page_inventory.keys[i].acquire(dirty_page));
    auto value = q::take(model.page_inventory.values[i].acquire(dirty_page));
    require(key.initialized_rows() == 0 && value.initialized_rows() == 0,
            "retired original page could not be reclaimed after all old readers released");
  }
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << model.execution.name()
            << "\",\"native_gate\":\"" << NativeGate::name()
            << "\",\"scheduled_progress\":" << (model.scheduler ? "true" : "false")
            << ",\"model_native_submission_observed\":true,\"gate_watchdog_opened\":false"
            << ",\"capacity\":256,\"resident_limit\":1,"
            << "\"old_prefix_tokens\":129,\"cancelled_candidate_tokens\":130,"
            << "\"directory_reported_bytes\":0,\"old_prefix_logical_lease_bytes\":" << retained_logical_bytes
            << ",\"managed_bytes_while_native_pending\":" << pending_bytes
            << ",\"managed_backing_bytes\":" << model.cache_plan.backing_bytes
            << ",\"collector_calls\":" << collector_calls
            << ",\"dropped_live_public_completion\":" << (dropped_public_completion ? "true" : "false")
            << ",\"same_page_exclusive_suffix\":true,\"old_kv_guards_and_bytes_unchanged\":true"
            << ",\"idle_stop_preserved_other_execution_loan\":true"
            << ",\"busy_prepare_rejections\":" << rejected_prepares
            << ",\"retired_aborted_slot_rebuilt_in_same_model_bundle\":true"
            << ",\"retired_model_episode_detached_all_pending_links\":true"
            << ",\"slot_index_before_cancel\":" << borrowed_slot
            << ",\"slot_index_before_report_reuse\":" << after_report_slot
            << ",\"post_commit_trace_error_preserved_head_and_returned_loan\":true"
            << ",\"different_session_continued_after_report_error\":true,\"final_tokens\":" << 131
            << ",\"old_recurrent_resources_byte_unchanged\":" << old_recurrent.size()
            << ",\"head_alias_sessions_added_distinct_versions\":0,\"native_error_injected\":false"
            << ",\"cancel_means_native_preemption\":false,\"timing_comparison\":false}\n";
}
void run_mtp(const std::filesystem::path &inputs, std::size_t verify_rows) {
  require(verify_rows == 1 || verify_rows == 2, "MTP verify rows must be one or two");
  auto prompt = q::read_ids(inputs / "prompt.ids");
  require(prompt.size() >= 15, "MTP retirement probe needs the original first15 prompt IDs");
  prompt.resize(15);
  constexpr std::size_t capacity = 64;
  // N2 permits the deliberately retained lag1 prefix beside one new committed
  // continuation. N+1 is still only one actual pending Snapshot candidate.
  const auto geometry = q::probe::read_two_b_geometry(inputs);
  auto weight_source = q::prepare_weight_source(inputs / "weights.tsv", true);
  const auto plan = q::cache::plan(geometry, weight_source.cache_row_bytes, capacity, 2, prompt.size(),
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 1, 2, true);
  const bool progress = native_backend == hyle::inference::BackendKind::kApple ||
      native_backend == hyle::inference::BackendKind::kVulkan;
  q::Model model(std::move(weight_source), geometry, capacity, plan, {}, progress,
      q::ReplayPolicy::kAuto, SIZE_MAX, {false, {}, 0}, 0, {.primary=native_backend});
  require(model.weights.size() == 335, "MTP probe did not load the real text+15MTP inventory");
  const auto catchup_role = verify_rows == 1 ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpCatchupRound;
  const std::array prefill{std::size_t{15}}, pair{std::size_t{2}}, catchup_calls{verify_rows};
  model.prepare_family(prefill, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(15, 1, 1, q::GraphRole::kMtpPrefill);
  model.prepare_family(pair, false, 1, q::GraphRole::kMtpRound);
  model.plan_replay_uses(2, 1, 8, q::GraphRole::kMtpRound);
  model.prepare_family(catchup_calls, false, 1, catchup_role);
  model.plan_replay_uses(verify_rows, 1, 8, catchup_role);
  auto session = model.create_session(prefill, {}, q::GraphRole::kMtpPrefill);
  model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluation(*session);
  const auto prefill_next = q::finish_evaluation(model, *session, "mtp_retirement_prefill15");
  require(prefill_next == 198, "fixed MTP seed no longer has its previously observed pending ID");
  model.prepare_evaluation(*session, std::span(&prefill_next, 1), nullptr, q::GraphRole::kMtpRound);
  model.submit_evaluation(*session);
  const auto seed = q::finish_mtp_round(model, *session, prefill_next, 2);
  complete_mtp(model, *session);
  require(seed.agreed && seed.accepted_draft && seed.pending == 198 && seed.draft == 248045 &&
          seed.target_first == 248045 && seed.next_id == 74455 && session->length() == 17 &&
          session->head->draft->frontier == 16 && session->head->draft->lag == 1 &&
          session->head->draft->catchup_token == seed.draft,
          "real first MTP round did not naturally create the fixed lag1 boundary");
  auto old_prefix = session->checkpoint();
  const auto saved = save_mtp(*old_prefix);
  require(saved.recurrent.size() == 36 && saved.hidden.size() == 2 &&
          old_prefix->carry_hidden.size() == 4096 && old_prefix->prior_hidden.size() == 4096,
          "lag1 prefix does not own all36states and both target hidden rows");
  const auto logical_bytes = head_lease_bytes(*old_prefix, model);
  auto waiting = model.create_session(catchup_calls, old_prefix, catchup_role);
  auto idle = model.create_session(catchup_calls, old_prefix, catchup_role);
  hyle::inference::SessionCache directory({1, (std::numeric_limits<std::size_t>::max)()});
  const auto admission = directory.admit(0, 0, [](std::size_t) { return hyle::inference::RetirementObservation{}; });
  q::check(admission.retirement_status); require(admission.accepted(), "MTP owner admission failed");
  auto pin = directory.pin(0); require(static_cast<bool>(pin), "MTP owner pin failed");
  const auto pending = old_prefix->next_id;
  const auto slot = model.slot_index;
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, catchup_role);
  auto *const phase = session->pending->phase;
  auto *const run = session->pending->execution.run;
  require(phase && phase->slots.size() == 1 && slot == 0 && run == q::take(phase->slots[slot].carrier.borrow()) && run->valid() &&
          session->pending == model.active_step.get() && session->pending->members.size() == 1 &&
          session->pending->members[0].base == old_prefix.get(),
          "MTP catchup did not borrow its original Model carrier and complete base");
  const auto all_bound = [&] {
    return session && session->pending &&
        std::count_if(session->pending->execution.bound.begin(), session->pending->execution.bound.end(), [](auto value) { return static_cast<bool>(value); }) == 14;
  };
  require(all_bound() && session->pending->members[0].next->length == 17 + verify_rows &&
          session->pending->members[0].next->draft->frontier == 18,
          "catchup did not prepare all14 real append permissions and both query frontiers");
  q::check(idle->stop()); const auto idle_retirement = idle->try_retire(); q::check(idle_retirement.status);
  require(idle_retirement.retired, "idle MTP alias could not retire"); idle.reset();
  require(run->valid() && model.carrier_borrowed && all_bound(), "idle stop closed another MTP carrier");
  const auto *candidate_identity = session->pending->members[0].next.get();
  require(candidate_identity, "prepared MTP episode has no unique unpublished candidate");
  const auto target_page = session->pending->members[0].next->pages.back().page();
  const auto draft_page = session->pending->members[0].next->draft->pages.back().page();
  const auto native_pending_bytes = q::take(model.cache_budget.usage(model.execution.device())).live_bytes;
  require(model.snapshot_versions.live == 2 && model.carrier_borrowed,
          "MTP pending candidate did not retain its finite version/admission");

  NativeGate gate(model.execution);
  ReleaseAndDrain cleanup{gate, model};
  q::check(gate.arm()); q::check(gate.wait_until_entered());
  model.submit_evaluation(*session);
  q::check(gate.wait_for_model_submission());
  std::size_t busy_rejections = 0;
  const auto busy = [&] {
    bool rejected = false;
    try { model.prepare_evaluation(*waiting, std::span(&pending, 1), nullptr, catchup_role); }
    catch (const std::exception &) { rejected = true; }
    require(rejected && !waiting->pending && !waiting->failed && !waiting->stopping &&
            waiting->head.get() == old_prefix.get() && model.carrier_borrowed && all_bound() &&
            session->pending->execution.run == run,
            "another MTP Session disturbed a native-pending shared loan");
    ++busy_rejections;
  };
  busy();
  require(directory.begin_retire(0), "MTP owner could not enter retiring state"); pin.reset();
  std::cerr << "native_gate_probe stage=mtp_stop_begin\n";
  q::check(session->stop());
  std::cerr << "native_gate_probe stage=mtp_stop_done\n";
  busy();
  std::size_t collector_calls = 0;
  const auto collect = [&] {
    const auto observed = directory.collect_retired(0, [&](std::size_t id) {
      require(id == 0 && session, "MTP collector lost its original owner"); ++collector_calls;
      const auto retired = session->try_retire();
      if (!retired.retired) return retired;
      require(!session->pending && !model.active_step, "MTP retired episode retained a pending link");
      session.reset(); return retired;
    });
    if (!observed.status.IsCancelled() || observed.status.has_native_error()) q::check(observed.status);
    return observed;
  };
  std::cerr << "native_gate_probe stage=mtp_first_collect_begin\n";
  const auto collected_before_release = collect();
  std::cerr << "native_gate_probe stage=mtp_first_collect_done retired=" << collected_before_release.retired << '\n';
  require(!collected_before_release.retired, "MTP blocked work retired before its original native owner");
  const bool dropped = session->pending->execution.completion.valid();
  session->pending->execution.completion = {};
  busy();
  require(!collect().retired && session->head.get() == old_prefix.get() && all_bound() &&
          session->pending->members[0].next.get() == candidate_identity &&
          model.snapshot_versions.live == 2 && model.carrier_borrowed && directory.retained_sessions() == 1 &&
          q::take(model.cache_budget.usage(model.execution.device())).live_bytes == native_pending_bytes,
          "dropping MTP Completion returned candidate, permissions or budget before retirement");
  const auto same_id = directory.admit(0, 0, [](std::size_t) { return hyle::inference::RetirementObservation{.retired = true}; });
  const auto other_id = directory.admit(1, 0, [](std::size_t) { return hyle::inference::RetirementObservation{.retired = true}; });
  q::check(same_id.retirement_status); q::check(other_id.retirement_status);
  require(same_id.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven &&
          other_id.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven,
          "MTP retiring directory slot was readmitted");
  for (std::size_t i = 0; i < model.page_inventory.keys.size(); ++i)
    require(!model.page_inventory.keys[i].acquire(target_page) && !model.page_inventory.values[i].acquire(target_page),
            "native-pending target dirty authority was reacquired");
  require(!model.draft_inventory.keys[0].acquire(draft_page) && !model.draft_inventory.values[0].acquire(draft_page),
          "native-pending draft dirty authority was reacquired");
  // No device readback while the original Queue is blocked. Compare every
  // immutable byte only after native retirement has actually become observable.
  gate.release(); q::check(gate.wait()); require(!gate.timed_out(), "native gate watchdog opened during MTP assertions");
  const auto deadline = q::Clock::now() + std::chrono::seconds(30);
  while (!collect().retired) { require(q::Clock::now() < deadline, "MTP native retirement timed out"); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
  require(!session && !model.active_step && !model.carrier_borrowed &&
          model.snapshot_versions.live == 1 && !run->valid() &&
          directory.retained_sessions() == 0 && directory.retained_bytes() == 0,
          "MTP retirement did not reclaim all14 candidates/loan/finite owner charges");
  unchanged_mtp(*old_prefix, saved);
  auto recovered = std::move(waiting);
  model.prepare_evaluation(*recovered, std::span(&pending, 1), nullptr, catchup_role);
  require(recovered->pending->phase == phase && recovered->pending->execution.run == run && run->valid() &&
          std::count_if(recovered->pending->execution.bound.begin(), recovered->pending->execution.bound.end(), [](auto value) { return static_cast<bool>(value); }) == 14,
          "MTP retry did not rebuild its same retired slot and all14 bindings");
  model.submit_evaluation(*recovered);
  const auto retried = q::finish_mtp_round(model, *recovered, pending, verify_rows);
  complete_mtp(model, *recovered); unchanged_mtp(*old_prefix, saved);
  require(recovered->length() == 17 + retried.processed_target && recovered->head->draft->frontier == 18 &&
          recovered->head->draft->lag == recovered->length() - 18 && retried.processed_draft == 2,
          "retried MTP catchup committed inconsistent target/draft frontiers");
  const auto retry_length = recovered->length();
  q::check(recovered->stop()); const auto retry_retirement = recovered->try_retire(); q::check(retry_retirement.status);
  require(retry_retirement.retired, "successful MTP retry could not retire"); recovered.reset();

  // A host selector failure happens after the real successful terminal but
  // before any of the 14 append outputs or the complete new head is published.
  auto policy_session = model.create_session(catchup_calls, old_prefix, catchup_role);
  model.prepare_evaluation(*policy_session, std::span(&pending, 1), nullptr, catchup_role);
  auto *const policy_run = policy_session->pending->execution.run;
  const auto policy_target_page = policy_session->pending->members[0].next->pages.back().page();
  const auto policy_draft_page = policy_session->pending->members[0].next->draft->pages.back().page();
  require(policy_target_page != old_prefix->pages.back().page() &&
          policy_draft_page != old_prefix->draft->pages.back().page(),
          "policy rejection probe did not reserve fresh target and draft dirty pages");
  const auto *const policy_candidate = policy_session->pending->members[0].next.get();
  const auto policy_epoch = policy_session->head.epoch();
  require(policy_candidate && model.snapshot_versions.live == 2,
          "host policy episode did not retain its unique candidate version");
  model.submit_evaluation(*policy_session);
  bool selector_called = false, selector_rejected = false;
  {
    const std::array outputs_io{q::EvaluationIO{.name = "mtp_policy_reject"}};
    const std::array outputs{q::evaluation_output(model, policy_session, outputs_io[0])};
    std::int32_t ignored = -1;
    try {
      model.finish_selected_evaluations(outputs, std::span(&ignored, 1),
          [&](std::span<const std::int32_t> ids, q::StepState &step, std::span<std::int32_t>) {
            require(ids.size() == verify_rows && (verify_rows == 1 ? step.members[0].draft_token == -1 : step.members[0].draft_token >= 0) && step.members[0].base == old_prefix.get() &&
                    std::count_if(step.execution.bound.begin(), step.execution.bound.end(), [](auto value) { return static_cast<bool>(value); }) == 14,
                    "host policy was not called after the actual complete MTP terminal");
            selector_called = true;
            throw std::logic_error("intentional host MTP accepted-boundary selector rejection");
          }, [&](std::size_t j, const q::CompletedEvaluationView &view) {
            q::observe_completed_evaluation(model, outputs[j], outputs_io[j], view);
          });
    } catch (const std::logic_error &error) {
      selector_rejected = true; std::cerr << "mtp_selector_rejected reason=" << error.what() << '\n';
    }
  }
  require(selector_called && selector_rejected && policy_session->head.get() == old_prefix.get() &&
          policy_session->head.epoch() == policy_epoch &&
          !policy_session->pending && !model.active_step && !model.carrier_borrowed &&
          model.snapshot_versions.live == 1 && !policy_run->valid(),
          "host selector failure published a head or retained an MTP candidate/carrier");
  unchanged_mtp(*old_prefix, saved);
  for (std::size_t i = 0; i < model.page_inventory.keys.size(); ++i) {
    auto k = q::take(model.page_inventory.keys[i].acquire(policy_target_page));
    auto v = q::take(model.page_inventory.values[i].acquire(policy_target_page));
    require(k.lease().valid() && v.lease().valid(), "selector rejection retained target dirty permissions");
  }
  {
    auto k = q::take(model.draft_inventory.keys[0].acquire(policy_draft_page));
    auto v = q::take(model.draft_inventory.values[0].acquire(policy_draft_page));
    require(k.lease().valid() && v.lease().valid(), "selector rejection retained draft dirty permissions");
  }
  model.prepare_evaluation(*policy_session, std::span(&pending, 1), nullptr, catchup_role);
  require(policy_session->pending->execution.run == policy_run && policy_run->valid(), "host policy abort did not rebuild its original MTP slot");
  model.submit_evaluation(*policy_session);
  const auto final = q::finish_mtp_round(model, *policy_session, pending, verify_rows);
  complete_mtp(model, *policy_session); unchanged_mtp(*old_prefix, saved);
  require(policy_session->head->draft->frontier == 18 && policy_session->length() == 17 + final.processed_target &&
          final.next_id == retried.next_id && final.processed_target == retried.processed_target,
          "old lag1 head could not continue after host policy abort");
  const auto final_length = policy_session->length(), final_frontier = policy_session->head->draft->frontier;
  std::weak_ptr<const q::cache::State> prefix_weak = old_prefix;
  old_prefix.reset();
  require(prefix_weak.expired() && model.snapshot_versions.live == 1, "MTP probe leaked an extra committed prefix owner");
  q::check(policy_session->stop()); const auto final_retirement = policy_session->try_retire(); q::check(final_retirement.status);
  require(final_retirement.retired, "final MTP Session did not retire"); policy_session.reset();
  require(model.snapshot_versions.live == 0 && !model.active_step && !model.carrier_borrowed,
          "MTP probe left committed or pending owners behind");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << model.execution.name()
            << "\",\"native_gate\":\"" << NativeGate::name()
            << "\",\"scheduled_progress\":" << (model.scheduler ? "true" : "false")
            << ",\"model_native_submission_observed\":true,\"gate_watchdog_opened\":false"
            << ",\"mtp\":true,\"weight_tensors\":335,\"capacity\":64,\"resident_snapshot_limit\":2"
            << ",\"candidate_capacity\":3,\"actual_graph_submissions\":6,\"natural_lag1_seed\":true,\"old_target_tokens\":17,\"old_draft_frontier\":16"
            << ",\"pending_target_query_rows\":" << verify_rows
            << ",\"writer_only_draft\":" << (verify_rows == 1 ? "true" : "false")
            << ",\"pending_draft_query_rows\":2,\"bound_append_banks\":14"
            << ",\"cancelled_candidate_target_tokens\":" << 17 + verify_rows
            << ",\"cancelled_candidate_draft_frontier\":18"
            << ",\"busy_prepare_rejections\":" << busy_rejections << ",\"collector_calls\":" << collector_calls
            << ",\"dropped_live_public_completion\":" << (dropped ? "true" : "false")
            << ",\"managed_bytes_while_native_pending\":" << native_pending_bytes
            << ",\"managed_backing_bytes\":" << model.cache_plan.backing_bytes
            << ",\"old_prefix_logical_lease_bytes\":" << logical_bytes
            << ",\"all14_native_permissions_retained_until_actual_retirement\":true,\"same_slot_rebuilt_after_close\":true"
            << ",\"old_target_and_draft_kv_bytes_unchanged\":true,\"old_recurrent_resources_byte_unchanged\":36"
            << ",\"old_carry_and_prior_bytes_unchanged\":true,\"selector_failed_after_successful_terminal\":true"
            << ",\"selector_failure_published_head\":false,\"selector_abort_reclaimed_all14_dirty_banks\":true"
            << ",\"retry_target_tokens\":" << retry_length << ",\"final_target_tokens\":" << final_length
            << ",\"final_draft_frontier\":" << final_frontier
            << ",\"native_error_injected\":false,\"host_policy_error_injected\":true"
            << ",\"cancel_means_native_preemption\":false,\"timing_comparison\":false}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path inputs, tape;
    bool mtp = false;
    std::size_t verify_rows = 2;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("probe option lacks its value");
      const std::string_view option(argv[i]);
      if (option == "--inputs") inputs = argv[i + 1];
      else if (option == "--decode-ids") tape = argv[i + 1];
      else if (option == "--verify-rows") {
        const std::string_view value(argv[i + 1]);
        if (value != "1" && value != "2") throw std::runtime_error("--verify-rows requires 1 or 2");
        verify_rows = value == "1" ? 1u : 2u;
      }
      else if (option == "--mtp") {
        const std::string_view value(argv[i + 1]);
        if (value != "on" && value != "off") throw std::runtime_error("--mtp requires on or off");
        mtp = value == "on";
      }
      else throw std::runtime_error("unknown retirement probe option");
    }
    if (inputs.empty() || (!mtp && tape.empty()) || (mtp && !tape.empty()) || (!mtp && verify_rows != 2))
      throw std::runtime_error("usage: hyle_qwen35_retirement_probe --inputs DIR [--decode-ids IDSFILE | --mtp on [--verify-rows 1|2]]");
    if (mtp) run_mtp(inputs, verify_rows); else run(inputs, tape);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen retirement probe: " << error.what() << '\n';
    return 1;
  }
}
