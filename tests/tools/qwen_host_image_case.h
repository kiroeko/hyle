#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

// Included after the product Model. Uses only already prepared local weights;
// ordinary and MTP are finite state/ownership checks, not quality or speed tests.
namespace qwen_host_image_case {
namespace q = qwen35;
namespace i = hyle::inference;
inline void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

struct ImageBytes final {
  q::cache::HostImageIdentity identity;
  q::cache::HostImageBoundary boundary;
  std::vector<q::cache::HostPayloadDesc> layout;
  std::vector<std::vector<std::byte>> payloads;
  std::size_t total_bytes = 0;
};
inline ImageBytes read_image(const q::cache::HostImage &image) {
  require(image.ready() && image.payload_count() == image.layout.size(), "image payload is not completely published");
  ImageBytes result{image.identity, image.boundary, image.layout, {}, image.total_bytes};
  result.payloads.reserve(image.payload_count());
  for (std::size_t index = 0; index < image.payload_count(); ++index) {
    result.payloads.emplace_back(image.layout[index].bytes);
    q::check(q::take(image.payload(index).view()).read(result.payloads.back()));
  }
  return result;
}
inline bool same_boundary(const q::cache::HostImageBoundary &a, const q::cache::HostImageBoundary &b) {
  return std::tie(a.length, a.evaluated_next_id, a.next_id, a.evaluated_sampling, a.sampling,
             a.sampling_wrapper, a.has_carry, a.has_prior, a.draft) ==
      std::tie(b.length, b.evaluated_next_id, b.next_id, b.evaluated_sampling, b.sampling,
             b.sampling_wrapper, b.has_carry, b.has_prior, b.draft);
}
inline void same_image(const ImageBytes &a, const ImageBytes &b) {
  require(a.identity == b.identity && same_boundary(a.boundary, b.boundary) && a.layout == b.layout &&
      a.total_bytes == b.total_bytes && a.payloads == b.payloads,
      "full host image metadata or initialized payload bytes differ");
}
inline void retire(std::shared_ptr<q::Session> &session) {
  if (!session) return;
  q::check(session->stop());
  const auto drained = session->drain(); q::check(drained.status);
  const auto observed = session->try_retire(); q::check(observed.status);
  require(observed.retired, "host image case did not observe original Session retirement");
  session.reset();
}
inline q::MtpRoundResult round(q::Model &model, const std::shared_ptr<q::Session> &session) {
  const auto pending = session->head->next_id;
  const auto role = session->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role);
  model.submit_evaluation(*session);
  return q::finish_mtp_round(model, *session, pending, 2, q::model_eos_tokens, nullptr,
      {.name = "host-image-round", .trace_states = false});
}
inline void same_round(const q::MtpRoundResult &a, const q::MtpRoundResult &b) {
  require(std::tie(a.pending, a.draft, a.target_first, a.target_second, a.next_id, a.agreed, a.accepted_draft,
              a.stopped, a.processed_target, a.processed_draft, a.proposal_count, a.consumed_drafts,
              a.verified_rows, a.auxiliary_graph_calls, a.emitted) ==
          std::tie(b.pending, b.draft, b.target_first, b.target_second, b.next_id, b.agreed, b.accepted_draft,
              b.stopped, b.processed_target, b.processed_draft, b.proposal_count, b.consumed_drafts,
              b.verified_rows, b.auxiliary_graph_calls, b.emitted),
      "baseline and restored actual MTP rounds differ");
}
struct RoleBytes final { std::size_t payloads = 0, bytes = 0, nonzero_bytes = 0; };
struct ModeReport final {
  std::size_t hidden = 0, target_layers = 0, total_bytes = 0, payloads = 0;
  std::array<RoleBytes, 8> roles{};
  bool byte_limit_at_factory = false;
  bool adapter_parking = false;
  bool allocation_failure_applicable = false, pre_submit_failure_preserved = false;
  std::size_t continuation_lag = 0;
};

inline ModeReport run_mode(const std::filesystem::path &input, q::BackendKind kind, bool draft) {
  constexpr std::size_t host_limit = 256u << 20;
  std::shared_ptr<const q::cache::HostImage> escaped;
  std::shared_ptr<i::HostStateStorage> storage_after_model;
  std::weak_ptr<const void> cold;
  ImageBytes retained_expected;
  ModeReport report;
  {
    auto source = q::prepare_weight_source(input / "weights.tsv", draft);
    const auto geometry = source.roles.geometry;
    require(geometry.context_limit >= 16 && geometry.vocabulary >= 4 && (!draft || geometry.mtp_layers == 1),
        "host image case requires C16, four tokens and the selected original MTP roles");
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 4, 3,
        q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, draft ? 2 : 1, draft, true);
    q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
        {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
    cold = model.resources_;
    report.hidden = static_cast<std::size_t>(geometry.hidden); report.target_layers = geometry.layers.size();
    const auto prompt_role = draft ? q::GraphRole::kMtpPrefill : q::GraphRole::kText;
    const std::array prefill{std::size_t{3}};
    const std::array ordinary{std::size_t{3}, std::size_t{1}};
    const std::array pair{std::size_t{2}};
    if (draft) {
      model.prepare_family(prefill, false, 1, prompt_role); model.plan_replay_uses(3, 1, 12, prompt_role);
      for (auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
        model.prepare_family(pair, false, 1, role); model.plan_replay_uses(2, 1, 12, role);
      }
      model.prepare_execution(prefill, 1, prompt_role);
      for (auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) model.prepare_execution(pair, 1, role);
    } else {
      model.prepare_family(ordinary);
      for (auto count : ordinary) model.plan_replay_uses(count, 1, 12);
      model.prepare_execution(ordinary);
    }
    auto session = model.create_session(prefill, {}, prompt_role, q::cache::SamplingCursor{123, 7, 0});
    const std::array<std::int32_t, 3> prompt{1, 2, 3};
    model.prepare_evaluation(*session, prompt, nullptr, prompt_role); model.submit_evaluation(*session);
    (void)q::finish_evaluation(model, *session, "host-image-prefill", {}, false);
    const auto evaluated_next = session->head->state_identity()->next_id;
    const auto evaluated_sampling = session->head->state_identity()->sampling;
    require(evaluated_next >= 0 && evaluated_next < geometry.vocabulary, "prefill did not produce a complete evaluated frontier");
    const auto historical = static_cast<std::int32_t>((std::int64_t(evaluated_next) + 1) % geometry.vocabulary);
    const q::cache::SamplingCursor historical_sampling{123, 7, 7};
    model.restore_pending_boundary(*session, historical_sampling, historical);
    auto old_alias = session->checkpoint();
    require(old_alias->state_identity()->next_id == evaluated_next && old_alias->next_id == historical &&
        old_alias->state_identity()->sampling == evaluated_sampling && old_alias->sampling == historical_sampling,
        "historical wrapper did not preserve its canonical evaluated decision/cursor");
    auto storage = q::take(i::HostStateStorage::create(model.execution, 2, host_limit));
    auto first = model.export_session(*session, storage);
    auto original = read_image(*first);
    require(original.boundary.sampling_wrapper && original.boundary.evaluated_next_id == evaluated_next &&
        original.boundary.next_id == historical && original.boundary.evaluated_sampling == evaluated_sampling &&
        original.boundary.sampling == historical_sampling, "host image lost its evaluated/historical boundary distinction");
    report.total_bytes = original.total_bytes; report.payloads = original.layout.size();
    for (std::size_t index = 0; index < original.layout.size(); ++index) {
      auto &role = report.roles[static_cast<std::size_t>(original.layout[index].role)];
      ++role.payloads; role.bytes += original.payloads[index].size();
      role.nonzero_bytes += static_cast<std::size_t>(std::count_if(original.payloads[index].begin(), original.payloads[index].end(),
          [](std::byte value) { return value != std::byte{}; }));
    }
    // Occupy only the first real history pool. Import has already consumed its
    // earlier target page writers when this exact pre-submit allocation fails.
    const auto reject_import_allocation = [&] {
      const auto part = std::find_if(original.layout.begin(), original.layout.end(), [](const auto &item) {
        return item.role == q::cache::HostPayloadRole::kConvHistory;
      });
      if (part == original.layout.end()) return;
      report.allocation_failure_applicable = true;
      auto &pool = model.state_pools[part->layer].history;
      std::vector<hyle::compute::WriteLease> blockers;
      blockers.reserve(pool.stats().max_blocks);
      bool exhausted = false;
      const auto bound = std::size_t(pool.stats().max_blocks) + 1;
      for (std::size_t attempt = 0; attempt < bound; ++attempt) {
        auto writer = pool.allocate(part->bytes, q::cache::state_pool_alignment);
        if (!writer) {
          require(writer.error().IsResourceExhausted(), "history-pool injection failed for an unrelated reason");
          exhausted = true; break;
        }
        blockers.push_back(std::move(*writer));
      }
      require(exhausted, "finite history-pool fill did not exhaust its original allocation bound");
      const auto target_free = model.page_inventory.free_ids.size();
      const auto draft_free = model.draft_inventory.free_ids.size();
      const auto versions_before = model.snapshot_versions.live;
      bool rejected = false;
      try { (void)model.restore_session(first); }
      catch (const i::EvaluationFailure &failure) {
        if (!failure.status().IsResourceExhausted()) throw;
        rejected = true;
      }
      require(rejected && model.snapshot_versions.live == versions_before &&
          model.page_inventory.free_ids.size() == target_free && model.draft_inventory.free_ids.size() == draft_free,
          "pre-submit restore allocation failure leaked a version or physical page reservation");
      same_image(original, read_image(*first));
      require(old_alias->state_identity()->next_id == evaluated_next && old_alias->next_id == historical &&
          old_alias->sampling == historical_sampling, "failed import changed its retained original boundary");
      blockers.clear();
      report.pre_submit_failure_preserved = true;
    };
    std::shared_ptr<q::Session> imported;
    std::size_t versions = 0;
    if (draft) {
      retire(session); versions = model.snapshot_versions.live;
      reject_import_allocation();
      imported = model.restore_session(first);
    } else {
      std::vector<q::GenerationRequest> parked; parked.reserve(1);
      parked.emplace_back("host-image-policy", prompt, std::filesystem::path{}, 1);
      parked[0].session = std::move(session); (void)parked[0].progress.begin_turn();
      bool executed = false;
      const auto no_execution = [&](std::span<const i::RequestKey>) { executed = true; };
      q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(parked),
          false, static_cast<std::vector<std::size_t> *>(nullptr), no_execution, storage};
      i::RequestKey key; key.slot = 0; key.generation = 1;
      const auto retired = adapter.retire(key, i::RequestRetirement::kEvicted); q::check(retired.status);
      require(retired.retired && !parked[0].session && parked[0].host_image && storage->live_images() == 2,
          "ordinary request adapter did not park its complete state before retiring the old Session");
      versions = model.snapshot_versions.live;
      reject_import_allocation();
      adapter.admit(key, {});
      require(parked[0].session && !parked[0].host_image && !executed && storage->live_images() == 1,
          "ordinary request adapter did not restore its parked complete state without model execution");
      imported = std::move(parked[0].session); report.adapter_parking = true;
    }
    require(model.snapshot_versions.live == versions + 1 && imported->head->state_identity() != old_alias->state_identity() &&
        imported->head->pages.size() == old_alias->pages.size(), "restore did not acquire one fresh complete model-state version");
    for (std::size_t page = 0; page < old_alias->pages.size(); ++page)
      require(imported->head->pages[page].page() != old_alias->pages[page].page(), "restored target page aliases the retained old permission");
    if (draft) for (std::size_t page = 0; page < old_alias->draft->pages.size(); ++page)
      require(imported->head->draft->pages[page].page() != old_alias->draft->pages[page].page(),
          "restored draft page aliases the retained old permission");
    auto second = model.export_session(*imported, storage);
    same_image(original, read_image(*second));
    require(storage->live_images() == 2, "roundtrip did not charge two actual host images");
    const auto *before = imported->head.get(); const auto epoch = imported->head.epoch();
    const auto boundary_before = q::generation_boundary(*imported->head);
    bool count_rejected = false;
    try { (void)model.export_session(*imported, storage); }
    catch (const i::EvaluationFailure &failure) {
      if (!failure.status().IsResourceExhausted()) throw;
      count_rejected = true;
    }
    require(count_rejected && storage->live_images() == 2 && imported->head.get() == before &&
        imported->head.epoch() == epoch && q::generation_boundary(*imported->head) == boundary_before,
        "image-count rejection changed a committed head/epoch or failed to preserve its count");
    auto small = i::HostStateStorage::create(model.execution, 1, 1);
    if (!small) {
      require(small.error().IsResourceExhausted(), "small host budget failed for an unrelated reason");
      report.byte_limit_at_factory = true;
    } else {
      bool bytes_rejected = false;
      try { (void)model.export_session(*imported, *small); }
      catch (const i::EvaluationFailure &failure) {
        if (!failure.status().IsResourceExhausted()) throw;
        bytes_rejected = true;
      }
      require(bytes_rejected && (*small)->live_images() == 0, "small byte budget admitted a full image or retained a failed admission");
    }
    require(imported->head.get() == before && imported->head.epoch() == epoch &&
        q::generation_boundary(*imported->head) == boundary_before, "byte-budget rejection changed the original complete head");
    first.reset(); second.reset(); require(storage->live_images() == 0, "released roundtrip images retained count reservations");
    auto baseline = model.create_session({}, old_alias, prompt_role);
    require(q::generation_boundary(*baseline->head) == q::generation_boundary(*imported->head),
        "baseline and restored continuation did not begin at the same historical boundary");
    if (draft) {
      const auto expected = round(model, baseline);
      const auto actual = round(model, imported);
      same_round(expected, actual);
    }
    else {
      const std::array pending{baseline->head->next_id};
      const auto expected = q::evaluate(model, *baseline, pending, "host-image-baseline", {}, false);
      const auto actual = q::evaluate(model, *imported, pending, "host-image-restored", {}, false);
      require(expected == actual, "ordinary restored next-token calculation differs from the same-shape baseline");
    }
    auto continued_baseline = model.export_session(*baseline, storage);
    auto continued_import = model.export_session(*imported, storage);
    const auto continued = read_image(*continued_baseline);
    same_image(continued, read_image(*continued_import));
    report.continuation_lag = continued.boundary.draft ? continued.boundary.draft->lag : 0;
    continued_baseline.reset(); continued_import.reset(); require(!storage->live_images(), "continuation image reservations leaked");
    auto old_reader = model.create_session({}, old_alias, prompt_role);
    auto unchanged = model.export_session(*old_reader, storage);
    same_image(original, read_image(*unchanged));
    retire(old_reader); retire(baseline); retire(imported); old_alias.reset();
    require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
        "image roundtrip retained a model version, native episode or original execution carrier");
    escaped = std::move(unchanged); storage_after_model = std::move(storage);
    retained_expected = std::move(original);
    require(storage_after_model->live_images() == 1, "escaped image lost its independent admission");
  }
  // No old Snapshot/Session survives this scope. The escaped image and host
  // endpoint may retain a Context, but must not retain ColdResources/weights.
  require(cold.expired(), "host image retained ColdResources after Model destruction");
  require(escaped && escaped->ready() && !escaped->identity.model_lifetime.lock(), "escaped image lifetime identity did not expire");
  const auto retained = read_image(*escaped);
  same_image(retained_expected, retained);
  escaped.reset();
  require(storage_after_model->live_images() == 0, "last host image release did not return its count reservation");
  return report;
}

inline void write_report(const ModeReport &report) {
  constexpr std::array names{"target_key", "target_value", "conv", "gdn", "carry", "prior", "draft_key", "draft_value"};
  std::cout << "{\"hidden\":" << report.hidden << ",\"target_layers\":" << report.target_layers
      << ",\"payloads\":" << report.payloads << ",\"bytes\":" << report.total_bytes
      << ",\"byte_limit_at_factory\":" << (report.byte_limit_at_factory ? "true" : "false")
      << ",\"adapter_parking\":" << (report.adapter_parking ? "true" : "false")
      << ",\"allocation_failure_applicable\":" << (report.allocation_failure_applicable ? "true" : "false")
      << ",\"pre_submit_failure_preserved\":" << (report.pre_submit_failure_preserved ? "true" : "false")
      << ",\"continuation_lag\":" << report.continuation_lag << ",\"roles\":{";
  for (std::size_t index = 0; index < names.size(); ++index) {
    const auto &role = report.roles[index];
    std::cout << (index ? "," : "") << '"' << names[index] << "\":{\"payloads\":" << role.payloads
        << ",\"bytes\":" << role.bytes << ",\"nonzero_bytes\":" << role.nonzero_bytes << '}';
  }
  std::cout << "}}";
}
inline int run(const std::filesystem::path &inputs, qwen35::BackendKind kind) {
  const auto ordinary = run_mode(inputs, kind, false);
  const auto mtp = run_mode(inputs, kind, true);
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << (kind == q::BackendKind::kCuda ? "cuda" : "cpu")
      << "\",\"ordinary\":";
  write_report(ordinary); std::cout << ",\"mtp\":"; write_report(mtp);
  std::cout << ",\"all_initialized_bytes_equal\":true,\"historical_wrapper_preserved\":true,"
      "\"same_shape_continuation_equal\":true,\"old_alias_unchanged\":true,\"cold_resources_expired\":true,"
      "\"final_images\":0,\"final_versions\":0,\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_host_image_case
