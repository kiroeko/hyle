#pragma once

#include "qwen_geometry_fixture.h"

// Included after the original Model. This oracle copies initialized logical
// payloads and keeps no physical leases or native work alive.
namespace qwen_generation_state_case {
namespace q = qwen35;
namespace c = hyle::compute;
inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

struct StateBytes final {
  q::GenerationBoundary boundary, evaluated;
  std::vector<q::cache::HostPayloadDesc> layout;
  std::vector<std::vector<std::byte>> payloads;
  friend bool operator==(const StateBytes &, const StateBytes &) = default;
};

// Compare initialized logical payloads, never physical page IDs or padding.
inline StateBytes state_bytes(q::Model &model, const q::cache::State &state) {
  const auto &evaluated = *state.state_identity();
  const auto boundary = q::take(q::cache::host_image_boundary(state));
  StateBytes result{q::generation_boundary(state), q::generation_boundary(evaluated),
      q::take(q::cache::make_host_image_layout(model.host_image_schema(), boundary)), {}};
  result.payloads.reserve(result.layout.size());
  for (const auto &part : result.layout) {
    c::ReadLease lease;
    switch (part.role) {
    case q::cache::HostPayloadRole::kTargetKey:
    case q::cache::HostPayloadRole::kTargetValue: {
      const auto found = std::find(model.attention_layer_ids.begin(), model.attention_layer_ids.end(), part.layer);
      require(found != model.attention_layer_ids.end() && part.page < evaluated.pages.size(),
          "generation oracle names an absent attention layer/page");
      const auto bank = static_cast<std::size_t>(found - model.attention_layer_ids.begin());
      const auto &frame = part.role == q::cache::HostPayloadRole::kTargetKey
          ? evaluated.keys[bank] : evaluated.values[bank];
      lease = q::take(q::take(frame.view()).page_prefix(evaluated.pages[part.page].page()));
      break;
    }
    case q::cache::HostPayloadRole::kConvHistory:
      require(part.layer < evaluated.state.size(), "generation oracle names an absent history layer");
      lease = evaluated.state[part.layer].history;
      break;
    case q::cache::HostPayloadRole::kGdnState:
      require(part.layer < evaluated.state.size(), "generation oracle names an absent recurrent layer");
      lease = evaluated.state[part.layer].recurrent;
      break;
    default:
      throw std::runtime_error("ordinary generation oracle unexpectedly contains draft state");
    }
    require(lease.valid() && lease.size() == part.bytes,
        "generation oracle differs from its actual initialized lease extent");
    result.payloads.emplace_back(part.bytes);
    q::check(model.execution.read(q::take(lease.view()), result.payloads.back()));
  }
  return result;
}

// A small, untrained but complete GA model with the real Qwen EOS vocabulary.
// All original state-producing branches still execute. Zero residual output
// projections make x equal to the input embedding, so untied one-hot head
// rows express a finite greedy token chain through the original kernels.
// No selected ID, EOS policy, Graph permission or runtime flag is injected.
inline std::filesystem::path model_eos_fixture(const std::filesystem::path &tiny,
    const std::filesystem::path &folder, std::size_t eos_output) {
  const auto original = q::prepared::read_roles(tiny / "weights.tsv", false);
  auto geometry = original.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.layers == "GA" &&
      eos_output >= 9 && eos_output <= 10, "EOS fixture requires the original bounded tiny GA source");
  const auto norm = std::ranges::find(original.selected, std::string("model.language_model.norm.weight"),
      &q::prepared::Role::name);
  require(norm != original.selected.end() && norm->dtype == "BF16" && norm->bytes == 128,
      "EOS fixture requires the actual tiny final norm");
  std::array<std::uint16_t, 64> gamma{};
  std::ifstream norm_payload(norm->source, std::ios::binary);
  norm_payload.seekg(static_cast<std::streamoff>(norm->offset));
  norm_payload.read(reinterpret_cast<char *>(gamma.data()), static_cast<std::streamsize>(sizeof(gamma)));
  // HF raw gamma is incremented by one by the original Router::norm_absolute.
  require(bool(norm_payload) && std::ranges::all_of(gamma, [](auto value) { return value == 0; }),
      "EOS fixture requires raw zero final gamma, hence original effective gamma +1");
  geometry.vocabulary = q::model_eos_tokens.back() + 1;
  geometry.tied_head = false; geometry.mtp_layers = 0;
  geometry.context_limit = 32; geometry.validate();
  std::filesystem::create_directories(folder);
  require(!std::filesystem::exists(folder / "weights.tsv") &&
      !std::filesystem::exists(folder / "boundary-weights.bin"), "EOS fixture output already exists");
  const auto body = std::filesystem::absolute(folder / "boundary-weights.bin");
  std::ofstream payload(body, std::ios::binary), manifest(folder / "weights.tsv");
  require(bool(payload) && bool(manifest), "cannot create finite model EOS fixture");
  manifest << q::prepared::role_signature << '\n' << std::string(64, eos_output == 9 ? '9' : 'a')
      << '\n' << qwen_test::geometry_line(geometry) << '\n';
  std::size_t offset = 0;
  for (const auto &[name, parameter] : q::prepared::parameter_inventory(false, geometry)) {
    const bool embedding = name == "model.language_model.embed_tokens.weight";
    const bool head = name == "lm_head.weight";
    const bool residual = name.ends_with(".mlp.down_proj.weight") ||
        name.ends_with(".linear_attn.out_proj.weight") || name.ends_with(".self_attn.o_proj.weight");
    if (!embedding && !head && !residual) {
      const auto found = std::ranges::find(original.selected, name, &q::prepared::Role::name);
      require(found != original.selected.end() && found->shape == parameter.shape,
          "EOS fixture cannot reuse an original tiny state-producing role");
      manifest << name << ' ' << found->dtype << ' ' << parameter.shape.size();
      for (auto dimension : parameter.shape) manifest << ' ' << dimension;
      manifest << ' ' << std::quoted(std::filesystem::absolute(found->source).generic_string())
          << ' ' << found->offset << ' ' << found->bytes << '\n';
      continue;
    }
    std::size_t elements = 1;
    for (auto dimension : parameter.shape) elements = q::cache::multiply(elements, static_cast<std::size_t>(dimension));
    std::vector<std::uint16_t> values(elements, 0);
    if (embedding) {
      for (std::size_t token = 0; token < 64; ++token) values[token * 64 + token] = 0x3f80;
      values[static_cast<std::size_t>(q::model_eos_tokens.front()) * 64 + 63] = 0x3f80;
    } else if (head) {
      // Prompt A ends at 3; B swaps its final two genuine prompt IDs and
      // ends at 2. The peer's disjoint 32..45 chain never meets model EOS.
      values[32 * 64 + 2] = 0x3f80;
      for (std::size_t token = 3; token < 3 + eos_output - 1; ++token)
        values[(token + 1) * 64 + token] = 0x3f80;
      values[static_cast<std::size_t>(q::model_eos_tokens.front()) * 64 + 3 + eos_output - 1] = 0x3f80;
      for (std::size_t token = 32; token < 47; ++token) values[(token + 1) * 64 + token] = 0x3f80;
      // A speculative successor may consume EOS, but it must never publish.
      values[63 * 64 + 63] = 0x3f80;
    }
    const auto bytes = q::cache::multiply(values.size(), sizeof(std::uint16_t));
    payload.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(bytes));
    manifest << name << " BF16 " << parameter.shape.size();
    for (auto dimension : parameter.shape) manifest << ' ' << dimension;
    manifest << ' ' << std::quoted(body.generic_string()) << ' ' << offset << ' ' << bytes << '\n';
    offset = q::cache::add(offset, bytes);
  }
  require(bool(payload) && bool(manifest) && offset < (65u << 20), "EOS fixture payload write failed or exceeded its bound");
  payload.close(); manifest.close();
  std::ofstream prompt(folder / "prompt.ids"); prompt << "1 2 3\n";
  require(bool(prompt), "cannot write EOS fixture prompt");
  return folder;
}

// Replace a real, retired output owner of the *next* imported carrier. Keeping
// the old owner prevents allocation-address reuse. The ordinary Graph bind
// revokes its old capture; pending B must reject rather than record a new one.
inline q::PreparedReadback replace_successor_output(q::Model &model,
    const q::GenerationContinuation &continuation, std::size_t width) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
      continuation.ready && continuation.ready_steps >= 6,
      "cold tuple replacement requires an actually warmed, idle owner boundary");
  const auto found = model.phases.find({width, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, 0, true});
  require(found != model.phases.end(), "cold tuple replacement lost its actual imported family");
  auto &phase = *found->second;
  const auto slot = (continuation.next_slot + 1) % q::execution_slot_count;
  require(continuation.direction_warm[slot] && (!continuation.ready_slot || *continuation.ready_slot != slot),
      "cold tuple replacement selected a live source or unwarmed direction");
  auto &run = phase.slots[slot]; auto &readback = phase.io[slot].next_id;
  require(phase.preparation_uses[slot] > 0,
      "cold tuple fixture did not replace an actually used imported carrier");
  // Model finish has already withdrawn appends, so output observation is no
  // longer available for reclaim(). Ordinary retired unbind returns the
  // original Graph-owned lease; its pool still retains the old backing.
  q::check(run.unbind(phase.id_port));
  auto retained = std::move(readback);
  readback = q::take(q::PreparedReadback::prepare(phase.child->compiled, phase.id_port,
      model.execution, model.execution_budget));
  q::check(readback.bind(run));
  return retained;
}

} // namespace qwen_generation_state_case
