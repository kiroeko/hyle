#pragma once

#include <array>
#include <cstring>
#include <span>
#include <vector>

// A real MemoryResource range seam, followed by the original Model/Backend.
// Faults below are injected CPU writes, not CUDA driver-failure coverage.
namespace qwen_input_packet_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
using h::require;

class CountingResource final : public q::mem::MemoryResource {
public:
  std::string_view name() const noexcept override { return "test.packet.delegate"; }
  q::mem::MemoryProperties properties() const noexcept override {
    auto result = q::mem::system_resource().properties();
    result.domain = {this, 1}; return result;
  }
  hyle::Result<q::mem::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    auto result = q::mem::system_resource().resource().allocate(bytes, alignment);
    if (result) ++live;
    return result;
  }
  void deallocate(q::mem::ResourceAllocation allocation) noexcept override {
    q::mem::system_resource().resource().deallocate(allocation); --live;
  }
  hyle::Status read(void *handle, std::size_t offset, std::span<std::byte> bytes) noexcept override {
    return q::mem::system_resource().resource().read(handle, offset, bytes);
  }
  hyle::Result<q::mem::ResourceHostMapping> map_host(void *handle, std::size_t offset,
      std::size_t bytes, q::mem::HostAccess access) noexcept override {
    return q::mem::system_resource().resource().map_host(handle, offset, bytes, access);
  }
  hyle::Status unmap_host(void *handle, void *token, std::size_t offset, std::size_t bytes,
      q::mem::HostAccess access) noexcept override {
    return q::mem::system_resource().resource().unmap_host(handle, token, offset, bytes, access);
  }
  hyle::Status flush(void *handle, std::size_t offset, std::size_t bytes) noexcept override {
    return q::mem::system_resource().resource().flush(handle, offset, bytes);
  }
  hyle::Status invalidate(void *handle, std::size_t offset, std::size_t bytes) noexcept override {
    return q::mem::system_resource().resource().invalidate(handle, offset, bytes);
  }
  hyle::Status write(void *handle, std::size_t offset, std::span<const std::byte> bytes) noexcept override {
    ++writes; last_bytes = bytes.size();
    const auto count = failure.ok() ? bytes.size() : (std::min)(std::size_t{5}, bytes.size());
    auto result = q::mem::system_resource().resource().write(handle, offset, bytes.first(count));
    return result.ok() ? failure : result;
  }
  hyle::Status failure;
  std::size_t live = 0, writes = 0, last_bytes = 0;
};

inline q::c::ResourceBudget budget_for(std::span<const q::mem::MemoryResourceRef> resources) {
  std::vector<q::c::ResourceBudgetLimit> limits;
  for (const auto resource : resources)
    if (std::ranges::none_of(limits, [&](const auto &limit) { return limit.resource == resource; }))
      limits.push_back({resource, 64u << 20});
  return q::take(q::c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
}

inline std::vector<std::byte> payload(q::InputPacket &packet) {
  std::vector<std::byte> bytes(packet.bytes());
  const auto token = packet.tokens().storage();
  q::check(token.resource().resource().read(token.native_handle(), token.offset(), bytes));
  return bytes;
}

inline void count_and_faults(bool budgeted) {
  CountingResource resource;
  const q::mem::MemoryResourceRef device{resource};
  const std::array resources{device, q::mem::system_resource()};
  auto budget = budgeted ? budget_for(resources) : q::c::ResourceBudget{};
  {
    q::InputPacket invalid;
    const auto status = invalid.prepare(q::dense(q::Type::kFloat32, {3, 1}), q::dense(q::Type::kInt32, {3}),
        device, budget, budget);
    require(status.IsInvalidArgument() && !invalid.prepared() && !invalid.allocation_refused() &&
        !resource.live && !resource.writes, "invalid packet descriptor allocated or published storage");
  }
  if (budgeted) {
    // Allocate the real device payload first, then refuse host staging. Both
    // unpublished owners must disappear; the resource sees actual frees.
    const std::array tight_limits{q::c::ResourceBudgetLimit{q::mem::system_resource(), 1u << 20}};
    auto tight = q::take(q::c::ResourceBudget::create({.limits = tight_limits, .allow_unattributed = true}));
    const auto host_before = q::take(tight.usage(q::mem::system_resource()));
    auto pressure = q::take(tight.reserve(q::mem::system_resource(), host_before.limit_bytes - host_before.live_bytes));
    const auto before = q::take(budget.usage(device)).live_bytes;
    q::InputPacket refused;
    const auto status = refused.prepare(q::dense(q::Type::kInt32, {3, 1}), q::dense(q::Type::kInt32, {3}),
        device, budget, tight);
    require(status.IsResourceExhausted() && !status.has_native_error() && refused.allocation_refused() &&
        !refused.prepared() && resource.live == 0 && q::take(budget.usage(device)).live_bytes == before &&
        q::take(tight.usage(q::mem::system_resource())).live_bytes == host_before.limit_bytes,
        "cold staging refusal retained its unpublished real device allocation");
  }
  {
    q::InputPacket packet;
    q::check(packet.prepare(q::dense(q::Type::kInt32, {3, 1}), q::dense(q::Type::kInt32, {3}),
        device, budget, budget));
    const auto token = packet.tokens().storage(), position = packet.positions().storage();
    const auto live = resource.live;
    require(packet.prepare(q::dense(q::Type::kInt32, {3, 1}), q::dense(q::Type::kInt32, {3}),
        device, budget, budget).IsFailedPrecondition() && resource.live == live && !resource.writes &&
        packet.tokens().storage().native_handle() == token.native_handle(),
        "repeated packet preparation changed its original fixed owner");
    require(token.resource() == device && position.resource() == device &&
        token.native_handle() == position.native_handle() && token.offset() == 0 && position.offset() == 256 &&
        token.size() == 12 && position.size() == 12 && packet.bytes() == 268,
        "packet ports lost their same root, exact disjoint extents or 256-byte position alignment");
    const std::array<std::int32_t, 3> ids{4, 7, 9}, positions{0, 1, 2};
    q::check(packet.write(ids, positions));
    std::vector<std::byte> expected(268);
    std::memcpy(expected.data(), ids.data(), 12); std::memcpy(expected.data() + 256, positions.data(), 12);
    require(resource.writes == 1 && resource.last_bytes == expected.size() && payload(packet) == expected,
        "packet did not perform one complete write with initialized padding");
    require(packet.write(std::span(ids).first(2), positions).IsInvalidArgument() &&
        packet.write(ids, std::span(positions).first(2)).IsInvalidArgument() && resource.writes == 1 &&
        payload(packet) == expected, "invalid packet extent changed payload or reached the resource");
    const std::array<std::int32_t, 3> replacement{31, 32, 33}, next_positions{3, 4, 5};
    std::vector<std::byte> next(268);
    std::memcpy(next.data(), replacement.data(), 12); std::memcpy(next.data() + 256, next_positions.data(), 12);
    const std::array failures{
        hyle::Status::Unavailable("injected partial packet write"),
        hyle::Status::FromNative(hyle::StatusCode::Unavailable, {hyle::NativeErrorDomain::Unknown, 73},
            "injected native-bearing partial packet write")};
    for (const auto &failure : failures) {
      resource.failure = failure;
      const auto before = payload(packet); const auto writes = resource.writes;
      const auto refused = packet.write(replacement, next_positions);
      auto partial = before; std::copy_n(next.begin(), 5, partial.begin());
      require(refused == failure && resource.writes == writes + 1 && payload(packet) == partial &&
          packet.tokens().storage().native_handle() == token.native_handle(),
          "partial upload changed its error, owner or unmodified suffix");
      resource.failure = {};
      q::check(packet.write(replacement, next_positions));
      require(resource.writes == writes + 2 && payload(packet) == next,
          "permitted retry did not overwrite the entire original packet");
      q::check(packet.write(ids, positions));
    }
  }
  require(resource.live == 0, "packet retained a delegated allocation after its owner was destroyed");
}

inline void real_backend(const std::filesystem::path &input, q::BackendKind kind, bool budgeted) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "input packet needs the cached GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, false, 2);
  auto execution = q::create_execution(geometry, plan, 0, {.primary = kind});
  const auto device = execution.device();
  const std::array resources{device, execution.staging(), q::mem::system_resource()};
  auto budget = budgeted ? budget_for(resources) : q::c::ResourceBudget{};
  q::Model model(std::move(execution), budget, std::move(source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20, {.enabled = false}, 0);
  const std::array rows{std::size_t{1}, std::size_t{3}};
  require(model.prepare_family(rows), "packet fixture did not prepare the original finite Family");
  if (!budgeted) {
    require(model.prepare_family(rows, false, 2), "packet fixture did not prepare its original R2 family");
    require(model.prepare_prefill_family(rows), "packet fixture did not prepare the original packed family");
    require(model.prepare_family(rows, true), "packet fixture did not prepare the original diagnostic family");
  }
  // All original use plans precede the first physical carrier. Diagnostic
  // uses the same non-diagnostic key; imported relay the non-imported key.
  model.plan_execution(rows);
  if (!budgeted) {
    model.plan_replay_uses(3, 2, 1);
    model.plan_replay_uses(3, 1, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, 0, q::RowPacking::kPrefill);
  }
  const auto original = model.find_child(1, 1, q::GraphRole::kText);
  require(bool(original), "packet fixture lost its original finite child");
  const auto *const family = &original->family;
  if (budgeted) {
    const auto before = q::take(budget.usage(device));
    {
      auto pressure = q::take(budget.reserve(device, before.limit_bytes - before.live_bytes));
      bool exhausted = false;
      try { (void)model.prepare_child(1, false); }
      catch (const hyle::inference::EvaluationFailure &failure) {
        exhausted = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
      }
      require(exhausted && model.phases.empty() && model.probe_phases.empty() &&
          !model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
          q::take(budget.usage(device)).live_bytes == before.limit_bytes,
          "mandatory physical refusal published a Phase or leaked original tracked storage");
    }
    require(q::take(budget.usage(device)).live_bytes == before.live_bytes,
        "failed physical candidate retained its requested-byte charge");
  }
  if (!budgeted) {
    // Only the default allocation path has an independent execution scope.
    // Explicit shared-budget Models intentionally use that same original
    // scope for execution and old tensors, so pressure there is mandatory.
    const auto host = q::mem::system_resource();
    const auto execution_before = q::take(model.execution_budget.usage(host));
    {
      auto pressure = q::take(model.execution_budget.reserve(host,
          execution_before.limit_bytes - execution_before.live_bytes));
      auto &fallback = model.prepare_child(1, false);
      const auto &old_io = fallback.slots[0].data;
      require(!old_io.input_packet.prepared() && old_io.input_packet.allocation_refused() &&
          old_io.input_packet_preparation.IsResourceExhausted() && !old_io.input_packet_preparation.has_native_error() &&
          old_io.tokens.caller_owned() && old_io.positions.caller_owned() && fallback.family == family &&
          q::take(model.execution_budget.usage(host)).live_bytes == execution_before.limit_bytes,
          "cold packet capacity fallback masked its refusal, changed Family or retained unpublished staging");
    }
    require(q::take(model.execution_budget.usage(host)).live_bytes == execution_before.live_bytes,
        "cold fallback retained its failed staging charge");
    require(!model.prepare_child(1, false).slots[0].data.input_packet.prepared(),
        "warm fallback Phase silently replaced its fixed Tensor owners");
  }
  auto &phase = model.prepare_child(3, false);
  auto &io = phase.slots[0].data;
  require(io.input_packet.prepared() && io.input_packet_preparation.ok() && !io.tokens.caller_owned() &&
      !io.positions.caller_owned() && phase.slots.size() == 1,
      "real CPU/CUDA ordinary Phase did not select one packet instead of its old two allocations");
  const std::array<std::int32_t, 3> ids{4, 7, 9}, positions{0, 1, 2};
  q::check(io.input_packet.write(ids, positions));
  std::array<std::int32_t, 3> read_ids{}, read_positions{};
  q::check(io.input_packet.tokens().storage().read(std::as_writable_bytes(std::span(read_ids))));
  q::check(io.input_packet.positions().storage().read(std::as_writable_bytes(std::span(read_positions))));
  require(read_ids == ids && read_positions == positions, "real Backend packet ranges differ from original IDs/positions");
  model.prepare_state_storage(); model.initialize_slot(phase, 0);
  const auto *const run = q::take(phase.slots[0].carrier.borrow());
  require(&model.prepare_child(3, false) == &phase, "warm lookup replaced the original packet Phase");
  model.initialize_slot(phase, 0);
  require(q::take(phase.slots[0].carrier.borrow()) == run && !model.active_step && !model.carrier_borrowed &&
      !model.snapshot_versions.live, "warm packet initialization replaced its Invocation or published state");
  if (!budgeted) {
    auto &pair = model.prepare_child(3, false, 2);
    auto &packet = pair.slots[0].data.input_packet;
    const std::array<std::int32_t, 6> pair_ids{4, 7, 9, 10, 11, 12}, pair_positions{0, 1, 2, 3, 4, 5};
    q::check(packet.write(pair_ids, pair_positions));
    std::array<std::int32_t, 6> read_pair{};
    q::check(packet.tokens().storage().read(std::as_writable_bytes(std::span(read_pair))));
    require(read_pair == pair_ids && packet.tokens().storage().size() == 24 &&
        packet.positions().storage().size() == 24 && packet.positions().storage().offset() == 256 &&
        packet.tokens().storage().native_handle() != io.input_packet.tokens().storage().native_handle(),
        "R2 exact Phase packet lost its row extents or shared the R1 mutable owner");
    const auto old_inputs = [](const q::Phase &excluded) {
      return std::ranges::all_of(excluded.slots.slots(), [](const auto &slot) {
        return !slot.data.input_packet.prepared() && slot.data.tokens.caller_owned() && slot.data.positions.caller_owned();
      });
    };
    auto &relay = model.prepare_child(1, false, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, 0, true);
    require(relay.slots.size() == q::execution_slot_count && old_inputs(relay),
        "packet gate replaced any of the original three imported relay carriers");
    auto &packed = model.prepare_child(3, false, 1, q::GraphRole::kText,
        q::TextCompletion::kEmitNext, 0, false, q::RowPacking::kPrefill);
    require(old_inputs(packed), "packet gate replaced original packed-prefill inputs");
    require(old_inputs(model.prepare_child(1, true)), "packet gate replaced original diagnostic inputs");
  }
}

inline void mtp_gate(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 1, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 2, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      16u << 20, {.enabled = false}, 0, {.primary = kind});
  const std::array rows{std::size_t{3}};
  require(model.prepare_family(rows, false, 1, q::GraphRole::kMtpPrefill),
      "packet fixture did not prepare its original MTP prefill family");
  model.plan_execution(rows, 1, q::GraphRole::kMtpPrefill);
  const auto &phase = model.prepare_child(3, false, 1, q::GraphRole::kMtpPrefill);
  require(std::ranges::all_of(phase.slots.slots(), [](const auto &slot) {
        return !slot.data.input_packet.prepared() && slot.data.tokens.caller_owned() && slot.data.positions.caller_owned();
      }), "packet gate replaced original MTP inputs");
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda, "input packet case supports CPU/CUDA");
  count_and_faults(false); count_and_faults(true);
  real_backend(input, kind, false); real_backend(input, kind, true);
  mtp_gate(input, kind);
  std::cout << "{\"status\":\"PASS\",\"case\":\"input-packet\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"allocation_modes\":2,\"single_write_and_padding\":true,\"exact_same_root_views\":true,"
         "\"invalid_extent_no_write\":true,\"injected_partial_write_retry\":true,\"injected_native_status_preserved\":true,"
         "\"cold_capacity_fallback\":true,\"mandatory_refusal_unpublished\":true,\"warm_original_invocation\":true,"
         "\"cold_negative_gates\":\"relay3slots,packed,diagnostic,MTP\","
         "\"submitted_graphs\":0,\"real_cuda_fault_injection\":false}\n";
  return 0;
}
} // namespace qwen_input_packet_case
