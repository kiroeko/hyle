#pragma once

#include "qwen_generation_state_case.h"

// Included after the original Model. Every carrier, admission, candidate and
// retirement below belongs to that Model; no hook or Completion is fabricated.
namespace qwen_cold_shape_fifo_case {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_generation_state_case::require;
using qwen_generation_state_case::state_bytes;
using qwen_generation_state_case::StateBytes;

struct Result final {
  std::array<std::int32_t, 2> ids{};
  std::array<StateBytes, 2> states;
  std::array<std::uint64_t, 2> epochs{};
  std::vector<q::GenerationCommitRecord> records;
  std::size_t submissions = 0;
  bool external_source_refusal = false;
};

struct ColdInventory final {
  std::vector<std::pair<q::FamilyKey, const q::Family *>> families;
  std::vector<std::pair<q::ExecutionShape, const q::Phase *>> phases;
};

inline ColdInventory cold_inventory(const q::Model &model) {
  ColdInventory result;
  for (const auto &[key, family] : model.families) result.families.emplace_back(key, family.get());
  for (const auto &[key, phase] : model.phases) result.phases.emplace_back(key, phase.get());
  return result;
}

struct PreparedOwners final {
  const q::Family *family = nullptr;
  const i::PreparedProgram *child = nullptr;
  const q::Phase *phase = nullptr;
  const void *invocation = nullptr;
  friend bool operator==(const PreparedOwners &, const PreparedOwners &) = default;
};

inline PreparedOwners owners(q::Model &model, std::size_t rows) {
  const auto selected = model.find_child(1, rows, q::GraphRole::kText);
  const auto phase = model.phases.find({1, rows});
  require(selected && phase != model.phases.end() && phase->second->slots.size() == 1,
      "FIFO key lost its actual ordinary Family/Phase");
  auto &carrier = phase->second->slots[0].carrier;
  require(carrier.initialized() && !carrier.closing() && carrier.retired(),
      "FIFO witness did not prepare and retire its real ordinary carrier");
  const auto *run = q::take(carrier.borrow());
  return {&selected->family, &selected->program, phase->second.get(), run->object_identity()};
}

inline void require_imported(q::Model &model) {
  const auto found = model.phases.find({1, 1, q::GraphRole::kText,
      q::TextCompletion::kEmitNext, 0, true});
  require(found != model.phases.end() && found->second->slots.size() == q::execution_slot_count,
      "FIFO victim did not prepare its original three imported carriers");
  for (const auto &slot : found->second->slots.slots())
    require(slot.carrier.initialized() && !slot.carrier.closing() && slot.carrier.retired(),
        "FIFO imported Phase has an unprepared or busy original carrier");
}

inline bool same_records(const std::vector<q::GenerationCommitRecord> &a,
    const std::vector<q::GenerationCommitRecord> &b, bool exact_time = false) {
  if (a.size() != b.size()) return false;
  for (std::size_t at = 0; at != a.size(); ++at) {
    const auto &x = a[at], &y = b[at];
    if (!x.same_result(y) || x.committed != y.committed || x.user_output != y.user_output ||
        x.imported != y.imported || x.turn != y.turn || !std::ranges::equal(x.ids(), y.ids()) ||
        (exact_time && (x.input_owner != y.input_owner || x.step_seconds != y.step_seconds || x.endpoint != y.endpoint)))
      return false;
  }
  return true;
}

inline Result arm(const std::filesystem::path &input, q::BackendKind kind, bool fifo) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 32, 2, 7,
      q::recurrent_bytes(q::state_schema_for(geometry, 32), geometry), 1, 1, false, false, 2);
  // The real CUDA Source seam requires captured ranges. CPU has no owned
  // StorageOwner for that seam and keeps the original command-only route.
  const auto replay = kind == q::BackendKind::kCuda
      ? q::ReplayPolicy::kPreferCapture : q::ReplayPolicy::kCommands;
  q::Model model(std::move(source), geometry, 32, plan, {}, false,
      replay, SIZE_MAX, {}, 0, {.primary = kind});
  const std::array finite{std::size_t{3}, fifo ? std::size_t{2} : std::size_t{1}};
  require(model.prepare_family(finite), "FIFO fixture did not prepare its original finite family");
  model.plan_execution(finite); model.freeze_shape_plan(); model.prepare_state_storage();
  require(model.shape_plan.capacity() == 2 && model.shape_plan.size() == 2 &&
      !model.shape_plan.admissions() && !model.shape_plan.evictions(),
      "FIFO fixture lost its frozen two-extra-key bound");
  const std::array prompt_rows{std::size_t{3}};
  model.prepare_execution(prompt_rows);
  const auto finite_family = model.find_child(1, 3, q::GraphRole::kText);
  require(finite_family.has_value(), "FIFO fixture lost the initial finite family");
  const auto *const finite_owner = &finite_family->family;
  const auto *const finite_prompt = &finite_family->program;
  const auto *const math = model.math_identity();
  const auto finite_unchanged = [&] {
    const auto a = model.find_child(1, 3, q::GraphRole::kText);
    const auto b = model.find_child(1, fifo ? 2u : 1u, q::GraphRole::kText);
    require(a && b && &a->family == finite_owner && &b->family == finite_owner &&
        &a->program == finite_prompt && model.math_identity() == math,
        "exact FIFO admission changed the original finite Family or model math identity");
  };
  PreparedOwners five, seven;
  if (fifo) {
    require(!model.find_child(1, 1, q::GraphRole::kText), "FIFO T1 was unexpectedly finite");
    model.admit_exact_shape(1, 1, q::GraphRole::kText);
    const std::array one{std::size_t{1}};
    model.prepare_execution(one); model.prepare_relay_execution(); require_imported(model);
    model.admit_exact_shape(5, 1, q::GraphRole::kText);
    const std::array rows{std::size_t{5}}; model.prepare_execution(rows); five = owners(model, 5);
    require(model.shape_plan.admissions() == 2 && !model.shape_plan.evictions() &&
        model.shape_plan.retained() == 2 && model.shape_plan.contains({1, 1}) && model.shape_plan.contains({1, 5}),
        "FIFO did not retain the two actual exact keys T1 then T5");
  } else {
    const std::array one{std::size_t{1}}; model.prepare_execution(one);
  }
  auto prompt_owner = std::make_shared<std::vector<std::int32_t>>(
      q::read_ids(input / "prompt.ids", geometry.vocabulary, model.capacity));
  require(*prompt_owner == std::vector<std::int32_t>{1, 2, 3}, "FIFO needs the original three-token prompt");
  const auto prompt = std::span<const std::int32_t>(*prompt_owner);
  auto session = model.create_session(prompt_rows);
  q::GenerationLedger ledger(2, 1);
  std::shared_ptr<q::StepSources> sources;
  q::cache::SnapshotRef checkpoint;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (sources && sources->submission.valid()) sources->submission.wait().IgnoreError();
    sources.reset(); (void)model.abort_episode(); checkpoint.reset();
    if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
  }};
  Result result;
  const auto finish = [&](const char *name) {
    return q::finish_evaluation(model, *session, name, {}, false,
        nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  };
  q::prepare_generation_record(ledger, *session, q::GraphRole::kText, prompt, false, 1, 0, prompt_owner);
  model.prepare_evaluation(*session, prompt); model.submit_evaluations(); ++result.submissions;
  result.ids[0] = finish("cold-fifo-prefill");
  result.states[0] = state_bytes(model, *session->head); result.epochs[0] = session->head.epoch();
  checkpoint = session->checkpoint();
  const auto pending = session->head->next_id;
  const auto *const head = session->head.get();
  const auto epoch = session->head.epoch();
  const auto versions = model.snapshot_versions.live;
  const auto free_pages = model.page_inventory.free_ids.size();
  const auto page_owners = model.page_inventory.owners;
  const auto unchanged_head = [&] {
    require(session->head.get() == head && session->head.epoch() == epoch &&
        state_bytes(model, *session->head) == result.states[0] && state_bytes(model, *checkpoint) == result.states[0],
        "FIFO admission/recovery changed the complete committed state, checkpoint, epoch or cursor");
  };
  if (fifo) {
    q::prepare_generation_record(ledger, *session, q::GraphRole::kText, std::span(&pending, 1), true, 1, 0);
    const auto before_records = ledger.records;
    const auto before_commits = ledger.committed_count;
    const auto unchanged_ledger = [&] {
      require(ledger.committed_count == before_commits && same_records(ledger.records, before_records, true),
          "FIFO refusal/recovery changed an original committed or uncommitted ledger record");
    };
    model.prepare_evaluation(*session, std::span(&pending, 1));
    auto *const step = model.active_step.get();
    require(step && step->prepared && !step->submitted && model.carrier_borrowed == 1 &&
        model.snapshot_versions.live == versions + 1 && session->pending == step,
        "FIFO did not borrow its original T1 candidate and state version");
    auto *const transactions = step->execution.state_transactions;
    require(transactions && transactions->bound(), "FIFO T1 did not bind its original state transactions");
    const auto entries = model.shape_plan.entries();
    const auto inventory = cold_inventory(model);
    const auto prepared_free = model.page_inventory.free_ids.size();
    const auto prepared_owners = model.page_inventory.owners;
    const auto refuse_seven = [&] {
      require(!model.find_child(1, 7, q::GraphRole::kText), "busy FIFO admission already has T7");
      bool rejected = false;
      try { model.admit_exact_shape(7, 1, q::GraphRole::kText); }
      catch (const std::runtime_error &error) {
        rejected = std::string_view(error.what()) == "actual suffix admission exceeds the idle model's original query/state bounds";
      }
      const auto after = cold_inventory(model);
      require(rejected && model.shape_plan.entries() == entries && model.shape_plan.capacity() == 2 &&
          model.shape_plan.retained() == 2 && model.shape_plan.admissions() == 2 && !model.shape_plan.evictions() &&
          after.families == inventory.families && after.phases == inventory.phases &&
          model.page_inventory.free_ids.size() == prepared_free && model.page_inventory.owners == prepared_owners &&
          model.active_step.get() == step && model.carrier_borrowed == 1,
          "busy exact admission changed FIFO metadata, native owners, pages or the original loan");
      unchanged_head(); unchanged_ledger(); finite_unchanged();
    };
    refuse_seven();
    if (kind == q::BackendKind::kCuda) {
      sources = model.prepare_submission_sources(*step);
      require(!sources->states.empty() && sources->states.size() == geometry.recurrent_layers() * 2 &&
          sources->pages.size() == geometry.attention_layers() * 2,
          "FIFO Source check omitted original state or KV banks");
    }
    model.submit_evaluations(*step); ++result.submissions;
    if (sources) {
      require(sources->submission.valid() && sources->primary_output.valid() && step->execution.sources == sources,
          "FIFO Source check did not export its real accepted submission");
      for (const auto &store : sources->states) require(store.valid(), "FIFO Store Source did not capture");
      for (const auto &page : sources->pages) require(page.valid(), "FIFO Page Source did not capture");
      q::check(sources->submission.wait());
      const auto blocked = model.abort_episode(); q::check(blocked.status);
      require(!blocked.retired && model.active_step.get() == step && session->pending == step &&
          model.carrier_borrowed == 1 && transactions->bound() && model.snapshot_versions.live == versions + 1,
          "external Source did not retain its candidate after actual native completion");
      refuse_seven(); result.external_source_refusal = true;
    } else {
      q::check(step->execution.wait());
      require(kind == q::BackendKind::kCpu, "FIFO omitted the supported CUDA Source ownership branch");
    }
    const auto weak_sources = std::weak_ptr(sources);
    const auto weak_step = std::weak_ptr(model.active_step);
    sources.reset();
    const auto retired = model.abort_episode(); q::check(retired.status);
    require(retired.retired && weak_sources.expired() && weak_step.expired() && !model.active_step &&
        !model.successor_step && !model.carrier_borrowed && !session->pending && !transactions->bound() &&
        model.snapshot_versions.live == versions && model.page_inventory.free_ids.size() == free_pages &&
        model.page_inventory.owners == page_owners && !session->failed && !session->stopping && session->retirement_status.ok(),
        "actual FIFO candidate recovery did not restore original state/page/loan ownership");
    unchanged_head(); unchanged_ledger(); finite_unchanged();

    model.admit_exact_shape(7, 1, q::GraphRole::kText);
    require(model.shape_plan.admissions() == 3 && model.shape_plan.evictions() == 1 && model.shape_plan.retained() == 2 &&
        !model.shape_plan.contains({1, 1}) && model.shape_plan.contains({1, 5}) && model.shape_plan.contains({1, 7}) &&
        !model.families.contains(q::FamilyKey::exact(1, q::GraphRole::kText, 1)) &&
        !model.phases.contains({1, 1}) && !model.phases.contains({1, 1, q::GraphRole::kText,
            q::TextCompletion::kEmitNext, 0, true}) && owners(model, 5) == five,
        "FIFO did not evict oldest T1 ordinary/imported Phase and Family while preserving T5");
    const std::array seven_rows{std::size_t{7}}; model.prepare_execution(seven_rows); seven = owners(model, 7);
    unchanged_head(); unchanged_ledger(); finite_unchanged();
    model.admit_exact_shape(1, 1, q::GraphRole::kText);
    require(model.shape_plan.admissions() == 4 && model.shape_plan.evictions() == 2 && model.shape_plan.retained() == 2 &&
        !model.shape_plan.contains({1, 5}) && model.shape_plan.contains({1, 7}) && model.shape_plan.contains({1, 1}) &&
        !model.families.contains(q::FamilyKey::exact(1, q::GraphRole::kText, 5)) && !model.phases.contains({1, 5}) &&
        owners(model, 7) == seven,
        "FIFO did not evict oldest T5 while preserving the actually prepared T7");
    const std::array one{std::size_t{1}};
    model.prepare_execution(one); model.prepare_relay_execution(); require_imported(model);
    (void)owners(model, 1); unchanged_head(); unchanged_ledger(); finite_unchanged();
  }
  q::prepare_generation_record(ledger, *session, q::GraphRole::kText, std::span(&pending, 1), true, 1, 0);
  model.prepare_evaluation(*session, std::span(&pending, 1)); model.submit_evaluations(); ++result.submissions;
  result.ids[1] = finish("cold-fifo-retry");
  result.states[1] = state_bytes(model, *session->head); result.epochs[1] = session->head.epoch();
  require(ledger.committed_count == 2 && session->head.epoch() == epoch + 1 &&
      state_bytes(model, *checkpoint) == result.states[0] && result.submissions == (fifo ? 3u : 2u),
      "FIFO rebuild retry changed the old checkpoint or omitted/repeated the only final publication");
  result.records.assign(ledger.records.begin(), ledger.records.begin() + ledger.committed_count);
  checkpoint.reset();
  q::check(session->stop()); const auto drained = session->drain(); q::check(drained.status);
  const auto retired = session->try_retire(); q::check(retired.status);
  require(retired.retired, "FIFO original Session did not actually retire"); session.reset();
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      std::ranges::all_of(model.page_inventory.owners, [](auto owners) { return owners == 0; }),
      "FIFO check retained a state version, candidate, native loan or original page owner");
  cleanup.dismiss(); return result;
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  const auto source = q::prepared::read_roles(input / "weights.tsv", false);
  const auto &g = source.geometry;
  if ((kind != q::BackendKind::kCpu && kind != q::BackendKind::kCuda) || g.hidden != 64 ||
      g.intermediate != 128 || g.vocabulary != 64 || g.layers != "GA" || g.context_limit != 32) {
    std::cerr << "SKIP: cold-shape FIFO requires the cached untrained GA/H64/I128/V64/C32 ordinary fixture" << '\n';
    return 77;
  }
  const auto control = arm(input, kind, false);
  const auto candidate = arm(input, kind, true);
  require(control.ids == candidate.ids && control.states == candidate.states && control.epochs == candidate.epochs &&
      same_records(control.records, candidate.records) && control.submissions + candidate.submissions == 5,
      "FIFO rebuild differs from same-width finite control output, complete state, epoch, cursor or ledger");
  require(candidate.external_source_refusal == (kind == q::BackendKind::kCuda),
      "FIFO external-Source coverage differs from the actual backend seam");
  std::cout << R"({"status":"PASS","backend":")" << qwen_host_image_case::backend_name(kind)
      << R"(","capacity":2,"accepted_graph_submissions":5,"admissions":4,"evictions":2,)"
         R"("T1_ordinary_and_imported_evicted_rebuilt":true,"T5_and_T7_prepared":true,)"
         R"("finite_and_surviving_owners_unchanged":true,"loan_refusal":true,"external_source_refusal":)"
      << (candidate.external_source_refusal ? "true" : "false")
      << R"(,"complete_state_epoch_cursor_ledger_equal":true,"final_versions_pages_loans":0,)"
         R"("native_fault_injection":false,"performance_claim":false})" << '\n';
  return 0;
}
} // namespace qwen_cold_shape_fifo_case
