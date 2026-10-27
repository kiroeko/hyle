#pragma once

#include <bit>
#include <cmath>
#include <cstring>

// Included after the actual Model and the existing complete host-image helpers.
// These untrained GA weights check state/row semantics, not model quality/TPS.
namespace qwen_packed_prefill_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
using h::require;
// Same absolute state budget as the independent FP64 recurrence reference in
// tests/compute_models/packed_sequence_case.h::gdn_math. This is a cross-
// implementation comparison, not a claim that this complete model ran in FP64.
inline constexpr double gdn_state_absolute_budget = 0.000003;

struct PayloadDifference final {
  q::cache::HostPayloadDesc layout;
  std::size_t first_byte = 0, different_bytes = 0, nonfinite_differences = 0;
  double maximum_absolute = 0, maximum_relative = 0;
};
struct StepReport final {
  std::size_t sequence = 0, prefix = 0, active = 0, physical = 0, epoch = 0;
  std::int32_t exact_next = -1, bucket_next = -1;
  bool prefill = false, boundary_equal = false, payloads_equal = false, numerically_equivalent = false;
  std::vector<PayloadDifference> differences;
};

inline q::Type payload_type(q::Model &model, const q::cache::HostPayloadDesc &part) {
  switch (part.role) {
  case q::cache::HostPayloadRole::kGdnState: return q::Type::kFloat32;
  case q::cache::HostPayloadRole::kConvHistory: return model.state_schema.at(part.layer).history.data_type();
  case q::cache::HostPayloadRole::kTargetKey:
  case q::cache::HostPayloadRole::kTargetValue: {
    const auto point = std::ranges::find(model.attention_layer_ids, part.layer);
    require(point != model.attention_layer_ids.end(), "prefill image names an unknown attention layer");
    const auto slot = static_cast<std::size_t>(point - model.attention_layer_ids.begin());
    return q::take(q::c::batch_cache_description(model.target_attention_schema(slot),
        part.role == q::cache::HostPayloadRole::kTargetKey ? q::c::BatchCacheRole::kKey
                                                        : q::c::BatchCacheRole::kValue)).data_type();
  }
  default: return model.precision.activation_type();
  }
}

inline PayloadDifference compare_payload(q::Model &model, const q::cache::HostPayloadDesc &part,
    const std::vector<std::byte> &a, const std::vector<std::byte> &b) {
  require(a.size() == b.size(), "prefill logical payload extents differ");
  PayloadDifference result{part, a.size()};
  for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) {
    result.first_byte = (std::min)(result.first_byte, i); ++result.different_bytes;
  }
  const auto type = payload_type(model, part);
  require(type == q::Type::kFloat32 || type == q::Type::kBFloat16,
      "prefill numeric diagnostic expects the selected original BF16/F32 fixture");
  const auto width = type == q::Type::kFloat32 ? 4u : 2u;
  require(a.size() % width == 0, "prefill floating payload has a partial element");
  const auto number = [&](const auto &bytes, std::size_t offset) {
    std::uint32_t bits = 0;
    if (width == 4) std::memcpy(&bits, bytes.data() + offset, 4);
    else { std::uint16_t small; std::memcpy(&small, bytes.data() + offset, 2); bits = std::uint32_t(small) << 16; }
    return double(std::bit_cast<float>(bits));
  };
  for (std::size_t i = 0; i < a.size(); i += width) {
    const auto x = number(a, i), y = number(b, i);
    if (!std::isfinite(x) || !std::isfinite(y)) {
      result.nonfinite_differences += std::memcmp(a.data() + i, b.data() + i, width) != 0; continue;
    }
    const auto delta = std::abs(x - y);
    result.maximum_absolute = (std::max)(result.maximum_absolute, delta);
    const auto denominator = (std::max)(std::abs(x), std::abs(y));
    if (denominator) result.maximum_relative = (std::max)(result.maximum_relative, delta / denominator);
  }
  return result;
}

inline h::ImageBytes capture(q::Model &model, q::Session &session,
    const std::shared_ptr<hyle::inference::HostStateStorage> &storage) {
  auto image = model.export_session(session, storage);
  auto result = h::read_image(*image);
  require(result.identity == model.host_image_identity(session.sampling_configuration().index),
      "prefill export changed its original Model or sampling identity");
  image.reset(); require(!storage->live_images(), "prefill image reservation did not retire");
  return result;
}

inline void compare(q::Model &model, const h::ImageBytes &a, const h::ImageBytes &b, StepReport &report) {
  require(a.layout == b.layout && a.total_bytes == b.total_bytes && a.payloads.size() == b.payloads.size(),
      "prefill complete logical state layout differs");
  report.boundary_equal = h::same_boundary(a.boundary, b.boundary);
  report.payloads_equal = a.payloads == b.payloads;
  report.numerically_equivalent = report.boundary_equal;
  for (std::size_t i = 0; i < a.payloads.size(); ++i) if (a.payloads[i] != b.payloads[i]) {
    auto difference = compare_payload(model, a.layout[i], a.payloads[i], b.payloads[i]);
    // Only newly computed F32 recurrence admits arithmetic-rounding error.
    // Cache/history/hidden bytes and all immutable checkpoints stay exact.
    const bool allowed = a.layout[i].role == q::cache::HostPayloadRole::kGdnState &&
        payload_type(model, a.layout[i]) == q::Type::kFloat32 &&
        difference.nonfinite_differences == 0 &&
        difference.maximum_absolute <= gdn_state_absolute_budget;
    report.numerically_equivalent = report.numerically_equivalent && allowed;
    report.differences.push_back(std::move(difference));
  }
}

inline void check_old(q::Model &model, q::cache::SnapshotRef prefix, const h::ImageBytes &before,
    const std::shared_ptr<hyle::inference::HostStateStorage> &storage) {
  // An empty or state-only prefix has no next token yet. Admit a normal
  // one-row future plan without executing it; do not ask the complete-prefix
  // shortcut (which deliberately requires a valid pending output) to adopt it.
  const std::array cold{std::size_t{1}};
  const bool planned = !prefix->length || prefix->next_id < 0;
  auto reader = model.create_session(planned ? std::span(cold) : std::span<const std::size_t>{}, prefix);
  h::same_image(before, capture(model, *reader, storage));
  h::retire(reader);
}

inline void write_step(const StepReport &report) {
  std::cout << "{\"sequence\":" << report.sequence << ",\"prefix\":" << report.prefix
      << ",\"active_rows\":" << report.active << ",\"physical_rows\":" << report.physical
      << ",\"prefill\":" << (report.prefill ? "true" : "false") << ",\"epoch\":" << report.epoch
      << ",\"exact_next\":" << report.exact_next << ",\"bucket_next\":" << report.bucket_next
      << ",\"boundary_equal\":" << (report.boundary_equal ? "true" : "false")
      << ",\"payloads_byte_equal\":" << (report.payloads_equal ? "true" : "false")
      << ",\"numerically_equivalent\":" << (report.numerically_equivalent ? "true" : "false") << ",\"differences\":[";
  for (std::size_t i = 0; i < report.differences.size(); ++i) {
    const auto &part = report.differences[i];
    std::cout << (i ? "," : "") << "{\"role\":" << static_cast<int>(part.layout.role)
        << ",\"layer\":" << part.layout.layer << ",\"page\":" << part.layout.page
        << ",\"bytes\":" << part.layout.bytes << ",\"first_byte\":" << part.first_byte
        << ",\"different_bytes\":" << part.different_bytes << ",\"maximum_absolute\":" << part.maximum_absolute
        << ",\"maximum_relative\":" << part.maximum_relative << ",\"nonfinite_differences\":" << part.nonfinite_differences << '}';
  }
  std::cout << "]}";
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto exact_source = q::prepare_weight_source(input / "weights.tsv", false);
  auto bucket_source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = exact_source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.recurrent_layers() &&
      geometry.attention_layers(), "packed prefill case needs the local C16 GA fixture with at least sixteen tokens");
  const auto plan = q::cache::plan(geometry, exact_source.cache_row_bytes, 16, 4, 8,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model exact(std::move(exact_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model bucket(std::move(bucket_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array exact_chunks{std::size_t{1}, std::size_t{3}, std::size_t{8}};
  const std::array bucket_chunks{std::size_t{8}};
  exact.prepare_family(exact_chunks);
  require(bucket.prepare_prefill_family(bucket_chunks), "selected CPU/CUDA route did not prepare its packed prefill math");
  for (const auto rows : exact_chunks) exact.plan_replay_uses(rows, 1, 8);
  bucket.plan_replay_uses(8, 1, 8, q::GraphRole::kText, q::TextCompletion::kEmitNext, 0, true);
  exact.prepare_execution(exact_chunks); bucket.prepare_prefill_execution(bucket_chunks);
  auto exact_storage = q::take(hyle::inference::HostStateStorage::create(exact.execution, 1, 4u << 20));
  auto bucket_storage = q::take(hyle::inference::HostStateStorage::create(bucket.execution, 1, 4u << 20));
  std::array<std::shared_ptr<q::Session>, 2> a, b;
  for (std::size_t j = 0; j < a.size(); ++j) {
    const q::cache::SamplingCursor cursor{123, j + 7, 0};
    a[j] = exact.create_session(exact_chunks, {}, q::GraphRole::kText, cursor);
    b[j] = bucket.create_session(bucket_chunks, {}, q::GraphRole::kText, cursor);
  }
  const q::Phase *carrier = nullptr;
  std::vector<StepReport> reports;
  const auto advance = [&](std::size_t sequence, std::span<const std::int32_t> ids, bool prefill) {
    auto &control = *a[sequence]; auto &candidate = *b[sequence];
    const auto prefix = control.length(), old_epoch = control.head.epoch();
    require(candidate.length() == prefix && candidate.head.epoch() == old_epoch &&
        control.head->sampling == candidate.head->sampling, "prefill inputs do not share the same logical boundary");
    const auto cursor = control.head->sampling;
    const auto old_a = control.checkpoint(), old_b = candidate.checkpoint();
    const auto before_a = capture(exact, control, exact_storage), before_b = capture(bucket, candidate, bucket_storage);
    StepReport report{sequence, prefix, ids.size(), ids.size(), static_cast<std::size_t>(old_epoch + 1)};
    report.prefill = prefill;
    report.exact_next = q::evaluate(exact, control, ids, "packed-exact", {}, false);
    if (prefill) {
      const std::array request{q::EvaluationRequest{b[sequence], ids}};
      bucket.prepare_prefill_evaluations(request);
      require(bucket.active_step && bucket.active_step->input_tokens == ids.size() &&
          bucket.active_step->phase->query_rows == 8 && bucket.active_step->phase->family->packed_prefill,
          "packed prefill did not retain L rows in the original packed R8 carrier");
      if (carrier) require(carrier == bucket.active_step->phase, "packed prefill silently compiled an L-specific carrier");
      else carrier = bucket.active_step->phase;
      report.physical = bucket.active_step->phase->query_rows;
      bucket.submit_evaluations();
      report.bucket_next = q::finish_evaluation(bucket, candidate, "packed-bucket", {}, false);
    } else report.bucket_next = q::evaluate(bucket, candidate, ids, "packed-next", {}, false);
    require(control.length() == prefix + ids.size() && candidate.length() == control.length() &&
        control.head.epoch() == old_epoch + 1 && candidate.head.epoch() == control.head.epoch() &&
        control.head->sampling == cursor.advanced(1) && candidate.head->sampling == control.head->sampling,
        "packed prefill consumed padding, advanced the wrong epoch or changed the RNG draw count");
    compare(exact, capture(exact, control, exact_storage), capture(bucket, candidate, bucket_storage), report);
    check_old(exact, old_a, before_a, exact_storage); check_old(bucket, old_b, before_b, bucket_storage);
    reports.push_back(std::move(report));
  };
  constexpr std::array<std::array<std::size_t, 3>, 2> lengths{{{1, 3, 8}, {8, 3, 1}}};
  for (std::size_t j = 0; j < lengths.size(); ++j) for (const auto length : lengths[j]) {
    std::array<std::int32_t, 8> ids;
    for (std::size_t row = 0; row < length; ++row)
      ids[row] = static_cast<std::int32_t>(1 + (a[j]->length() + row + 7 * j) % 15);
    advance(j, std::span(ids).first(length), true);
  }
  // Only now admit T1 exact decode, so no new exact family can change the
  // minimal R choice during the six real bucket reuse calls above.
  for (std::size_t j = 0; j < a.size(); ++j) {
    const auto id = a[j]->head->next_id; advance(j, std::span(&id, 1), false);
  }
  for (auto &session : a) h::retire(session);
  for (auto &session : b) h::retire(session);
  require(!exact.snapshot_versions.live && !bucket.snapshot_versions.live && !exact.active_step &&
      !bucket.active_step && !exact.carrier_borrowed && !bucket.carrier_borrowed &&
      !exact_storage->live_images() && !bucket_storage->live_images() &&
      exact.page_inventory.free_ids.size() == plan.max_pages && bucket.page_inventory.free_ids.size() == plan.max_pages,
      "packed prefill retained a version, image, page reservation or original execution carrier");
  bool equal = true, byte_equal = true;
  std::cout << "{\"backend\":\"" << h::backend_name(kind)
      << "\",\"mode\":\"packed_prefill\",\"synthetic_untrained\":true,\"cold_rows\":8"
      << ",\"active_prefill_rows\":24,\"physical_prefill_rows\":48,\"next_decode_rows\":2,\"steps\":[";
  for (std::size_t at = 0; at < reports.size(); ++at) {
    const auto &report = reports[at]; equal = equal && report.numerically_equivalent;
    byte_equal = byte_equal && report.payloads_equal;
    std::cout << (at ? "," : ""); write_step(report);
  }
  std::cout << "],\"old_checkpoints_byte_unchanged\":true,\"rng_and_epochs_equal\":true,\"final_versions\":0"
      << ",\"payloads_byte_equal\":" << (byte_equal ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equal ? "true" : "false")
      << ",\"numerical_budget\":{\"role\":\"newly_computed_f32_gdn_state\",\"absolute\":" << gdn_state_absolute_budget
      << ",\"nonfinite_differences_allowed\":false,\"other_payloads\":\"byte_exact\""
      << ",\"basis\":\"packed_sequence_case.h gdn_math independent FP64 state oracle\",\"complete_model_fp64_oracle\":false}"
      << ",\"performance_claim\":false,\"status\":\"" << (equal ? "PASS" : "VALUE_OR_CONTRACT_DIFFERENCE") << "\"}\n";
  require(equal, "packed versus exact state exceeds the inherited F32 GDN budget or an exact boundary/payload differs; complete byte and numeric diagnostics were retained");
  return 0;
}
} // namespace qwen_packed_prefill_case
