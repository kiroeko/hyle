#pragma once

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

} // namespace qwen_generation_state_case
