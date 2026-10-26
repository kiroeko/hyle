#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Included after qwen_host_image_case.h and the product Model. This finite
// ordinary Text consumer records identities without retaining extra readers.
namespace qwen_state_reuse_case {
namespace q = qwen35;
using qwen_host_image_case::require;

struct BlockIdentity final {
  std::uint64_t generation = 0;
  std::uintptr_t handle = 0;
  std::size_t offset = 0;
  friend bool operator==(const BlockIdentity &, const BlockIdentity &) = default;
};
inline BlockIdentity identity(const hyle::compute::ReadLease &lease) {
  const auto view = q::take(lease.view());
  return {lease.generation(), reinterpret_cast<std::uintptr_t>(view.native_handle()), view.offset()};
}
struct OutputTrace final {
  std::size_t layer = 0;
  bool recurrent = false, external_store = false;
  std::size_t pool_blocks = 0, cache_peak = 0;
  std::array<BlockIdentity, 6> blocks{};
};
inline std::vector<std::vector<std::byte>> state_bytes(q::Model &model, const q::cache::State &state) {
  std::vector<std::vector<std::byte>> result;
  for (std::size_t layer = 0; layer < model.state_schema.size(); ++layer) if (model.geometry.is_gdn(layer)) {
    for (const auto *lease : {&state.state[layer].history, &state.state[layer].recurrent}) {
      result.emplace_back(lease->size());
      q::check(model.execution.read(q::take(lease->view()), result.back()));
    }
  }
  return result;
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 4,
      "state reuse case requires C16 and four tokens in its prepared local fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  require(plan.state_slots == 3, "ordinary N2/R1 state capacity must remain N+R");
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
  const std::array calls{std::size_t{3}, std::size_t{1}};
  model.prepare_family(calls);
  model.plan_replay_uses(3, 1, 1);
  model.plan_replay_uses(1, 1, 8);
  model.prepare_execution(calls);
  auto session = model.create_session(calls, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 7, 0});
  auto &decode = model.prepare_child(1, false);
  std::vector<OutputTrace> traces;
  for (std::size_t layer = 0; layer < model.state_schema.size(); ++layer) if (geometry.is_gdn(layer)) {
    const auto &ports = decode.layers[layer];
    for (const bool recurrent : {false, true}) {
      const auto &port = recurrent ? ports.recurrent : ports.history;
      const auto &pools = model.state_pools[layer];
      const auto &pool = recurrent ? pools.recurrent : pools.history;
      const auto &layout = recurrent ? pools.recurrent_layout : pools.history_layout;
      require(port.store.valid() && port.gather.valid() && layout.valid(),
          "state reuse case requires the original external Store and slot layout for every state role");
      require(pool.stats().max_blocks == plan.state_slots && layout.slot_count() == plan.state_slots,
          "external state Store pool grew beyond its original N+R capacity");
      traces.push_back({layer, recurrent, true, pool.stats().max_blocks});
    }
  }
  require(!traces.empty(), "state reuse case requires at least one recurrent layer and its two external Stores");
  const auto inspect_caches = [&] {
    for (auto &trace : traces) {
      const auto &cache = trace.recurrent ? model.state_pools[trace.layer].recurrent_reuse
          : model.state_pools[trace.layer].history_reuse;
      require(cache.size() <= plan.active_sequences && cache.capacity() <= plan.active_sequences,
          "state writer cache exceeded the shared R inventory");
      trace.cache_peak = (std::max)(trace.cache_peak, cache.size());
    }
  };
  const auto cached_roles = [&] {
    return static_cast<std::size_t>(std::ranges::count_if(traces, [&](const auto &trace) {
      const auto &reuse = trace.recurrent ? model.state_pools[trace.layer].recurrent_reuse
          : model.state_pools[trace.layer].history_reuse;
      return reuse.size() != 0;
    }));
  };
  // Leave cached writers available, but occupy every raw free slot in each
  // such original pool. Admission must consume those cached permissions.
  const auto occupy_raw_free = [&] {
    std::vector<hyle::compute::WriteLease> blockers;
    for (const auto &trace : traces) {
      auto &pools = model.state_pools[trace.layer];
      const auto &reuse = trace.recurrent ? pools.recurrent_reuse : pools.history_reuse;
      if (!reuse.size()) continue;
      auto &pool = trace.recurrent ? pools.recurrent : pools.history;
      bool exhausted = false;
      for (std::size_t attempt = 0; attempt <= pool.stats().max_blocks; ++attempt) {
        auto writer = pool.allocate(reuse.bytes(), q::cache::state_pool_alignment);
        if (!writer) {
          require(writer.error().IsResourceExhausted(), "state-pool pressure failed for an unrelated reason");
          exhausted = true; break;
        }
        blockers.push_back(std::move(*writer));
      }
      require(exhausted, "bounded raw state-pool pressure did not exhaust its free slots");
    }
    return blockers;
  };
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  (void)q::evaluate(model, *session, prompt, "state-reuse-prefill", {}, false);
  inspect_caches();
  const auto cold_pressure_roles = cached_roles();
  {
    const auto *head = session->head.get(); const auto epoch = session->head.epoch();
    const auto boundary = q::generation_boundary(*head);
    const auto bytes = state_bytes(model, *head);
    auto blockers = occupy_raw_free();
    auto cold = model.create_session(calls, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 8, 0});
    require(!cached_roles() && cold->length() == 0 && model.snapshot_versions.live == 2 &&
        session->head.get() == head && session->head.epoch() == epoch &&
        q::generation_boundary(*session->head) == boundary && state_bytes(model, *session->head) == bytes,
        "cold Session admission did not consume cached state or changed another committed head");
    for (const auto &part : state_bytes(model, *cold->head))
      require(std::ranges::all_of(part, [](auto byte) { return byte == std::byte{}; }),
          "cold Session did not zero its recovered original state writer");
    qwen_host_image_case::retire(cold);
    require(model.snapshot_versions.live == 1, "cold admission check retained its temporary version");
  }
  for (std::size_t step = 0; step < 6; ++step) {
    const auto pending = session->head->next_id;
    const auto sampling = session->head->sampling;
    const auto epoch = session->head.epoch();
    (void)q::evaluate(model, *session, std::span(&pending, 1), "state-reuse-decode", {}, false);
    require(session->length() == 4 + step && session->head->sampling == sampling.advanced(1) &&
        session->head.epoch() == epoch + 1, "ordinary decode lost its committed length/cursor/epoch");
    for (auto &trace : traces) {
      const auto &state = session->head->state[trace.layer];
      trace.blocks[step] = identity(trace.recurrent ? state.recurrent : state.history);
      if (step >= 2) require(trace.blocks[step] == trace.blocks[step - 2] &&
          (trace.blocks[step].handle != trace.blocks[step - 1].handle ||
           trace.blocks[step].offset != trace.blocks[step - 1].offset),
          "ordinary external state Store did not alternate its same two retired original blocks");
    }
    inspect_caches();
  }
  const auto checkpoint_epoch = session->head.epoch();
  auto checkpoint = session->checkpoint();
  const auto *checkpoint_identity = checkpoint.get();
  const auto checkpoint_length = checkpoint->length;
  const auto checkpoint_sampling = checkpoint->sampling;
  const auto checkpoint_next = checkpoint->next_id;
  const auto frozen_bytes = state_bytes(model, *checkpoint);
  require(checkpoint_length == 9 && session->head.epoch() == checkpoint_epoch,
      "checkpoint changed the successful working-state boundary");
  const auto restore_pressure_roles = cached_roles();
  {
    auto storage = q::take(hyle::inference::HostStateStorage::create(model.execution, 1, 256u << 20));
    auto image = model.export_session(*session, storage);
    auto blockers = occupy_raw_free();
    auto restored = model.restore_session(image);
    require(!cached_roles() && model.snapshot_versions.live == 2 &&
        q::generation_boundary(*restored->head) == q::generation_boundary(*checkpoint) &&
        state_bytes(model, *restored->head) == frozen_bytes &&
        state_bytes(model, *checkpoint) == frozen_bytes && session->head.epoch() == checkpoint_epoch,
        "host restore did not consume cached state or preserve its complete source boundary");
    qwen_host_image_case::retire(restored);
    image.reset();
    require(!storage->live_images() && model.snapshot_versions.live == 1,
        "cached-state restore check retained an image or temporary state version");
  }
  for (std::size_t step = 0; step < 2; ++step) {
    const auto pending = session->head->next_id;
    (void)q::evaluate(model, *session, std::span(&pending, 1), "state-reuse-frozen", {}, false);
    require(checkpoint.get() == checkpoint_identity && checkpoint->length == checkpoint_length &&
        checkpoint->sampling == checkpoint_sampling && checkpoint->next_id == checkpoint_next &&
        state_bytes(model, *checkpoint) == frozen_bytes &&
        session->head.epoch() == checkpoint_epoch + step + 1 && session->length() == 10 + step,
        "later decode changed the retained immutable checkpoint or its original state bytes");
    inspect_caches();
  }
  require(!model.active_step && !model.carrier_borrowed && model.snapshot_versions.live == 2,
      "retained checkpoint and current head must own exactly two successful versions");
  qwen_host_image_case::retire(session);
  require(model.snapshot_versions.live == 1 && state_bytes(model, *checkpoint) == frozen_bytes,
      "Session retirement changed or released its explicitly retained checkpoint");
  checkpoint.reset();
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "state reuse case retained a version or original native episode");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"mode\":\"state_reuse\",\"hidden\":" << geometry.hidden
      << ",\"layers\":" << geometry.layers.size() << ",\"applicable\":true"
      << ",\"resident_sessions\":2,\"active_sequences\":1,\"state_slots\":" << plan.state_slots
      << ",\"prefill_tokens\":3,\"decode_steps\":8,\"final_length\":11,\"unfrozen_decode_steps\":6"
      << ",\"cold_acquire_pressure_roles\":" << cold_pressure_roles
      << ",\"host_restore_pressure_roles\":" << restore_pressure_roles
      << ",\"checkpoint_epoch\":" << checkpoint_epoch << ",\"outputs\":[";
  for (std::size_t at = 0; at < traces.size(); ++at) {
    const auto &trace = traces[at];
    std::cout << (at ? "," : "") << "{\"layer\":" << trace.layer
        << ",\"role\":\"" << (trace.recurrent ? "gdn" : "conv")
        << "\",\"external_store\":" << (trace.external_store ? "true" : "false")
        << ",\"pool_max_blocks\":" << trace.pool_blocks << ",\"cache_peak\":" << trace.cache_peak << ",\"blocks\":[";
    for (std::size_t step = 0; step < trace.blocks.size(); ++step) {
      const auto &block = trace.blocks[step];
      std::cout << (step ? "," : "") << "{\"generation\":" << block.generation
          << ",\"native_handle\":" << block.handle << ",\"offset\":" << block.offset << '}';
    }
    std::cout << "]}";
  }
  std::cout << "],\"checkpoint_bytes_unchanged\":true,\"final_versions\":0,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_state_reuse_case
