#include <gtest/gtest.h>

#include "dspark_accepted_prefix_fixture.h"

namespace {
namespace p = dspark_accepted_prefix_fixture;
namespace s = deepseek_sparse_prefix_fixture;
namespace c = hyle::compute;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;

// Full required target state at the existing projected-attention boundary,
// plus all three DSpark main-KV rings/carry/protocol in one immutable version.
// The finite Sink->main_x bridge is a lease/Graph dependency fixture, not the
// checkpoint main_proj or a complete target decoder/proposal/acceptance driver.
// accepted_rows=0 is an abort; rows1..6 consume old pending plus0..5 matched
// proposals. These are injected selector outcomes, not measured acceptance.
// Token projections are supplied at the existing fixture boundary. No
// proposal-local KV is accepted as a persistent main-KV row.
struct Step final {
  std::vector<c::ReadLease> target, caught;
  std::int32_t pending;
};
Result<Step> execute_single(s::Route &route, const s::Case &mode,
    const s::PreparedInputs &input, p::CompleteState &state,
    const c::StateVersion &before, int token,
    s::GraphProgram &target, s::GraphProgram &bridge, s::GraphProgram &catchup) {
  HYLE_ASSIGN_OR_RETURN(auto members, state.target(before));
  HYLE_ASSIGN_OR_RETURN(auto arguments, input.inputs(mode, members, token, 1));
  HYLE_ASSIGN_OR_RETURN(auto result, route.execute(target, arguments));
  const auto attention = state.target_members + (mode.indexed() ? 1u : 0u);
  const std::array bridge_input{result[attention]};
  HYLE_ASSIGN_OR_RETURN(auto main, route.execute(bridge, bridge_input));
  HYLE_ASSIGN_OR_RETURN(auto rings, state.rings(before));
  std::vector<c::ReadLease> catchup_input{main[0]};
  catchup_input.insert(catchup_input.end(), rings.begin(), rings.end());
  HYLE_ASSIGN_OR_RETURN(auto caught, route.execute(catchup, catchup_input));
  HYLE_ASSIGN_OR_RETURN(auto ids, route.read<std::int32_t>(main[1]));
  HYLE_RETURN_IF_ERROR(s::require(ids.size() == 1 && ids[0] >= 0 && ids[0] < 17,
                                "fresh projected target pending"));
  result.resize(state.target_members);
  return Step{std::move(result), std::move(caught), ids[0]};
}

Status check_ring_oracle(s::Route &route, const s::Case &mode,
    const std::array<std::vector<float>, p::kDraftStages> &initial,
    const c::ReadLease &main, std::span<const c::ReadLease> caught) {
  HYLE_ASSIGN_OR_RETURN(auto values, route.read<float>(main));
  auto expected = initial;
  const s::ds::fp8::ModelLinear projection{{}, s::kDepth, p::kMain, 443, true};
  const auto scale = s::sem::oracle::scale(s::kDepth);
  for (int t = 0; t < p::kVerifyRows; ++t) {
    const auto row = std::span(values).subspan(std::size_t(t) * p::kMain, p::kMain);
    const auto projected = projection.reference(row);
    const auto normalized = s::ds::norm_reference(projected, scale);
    const auto rotated = s::ds::rope_reference(normalized, s::kDepth,
        std::int32_t(mode.base_position + t), false);
    const auto kv = s::ds::cache_reference(rotated, s::kDepth, false);
    const auto offset = std::size_t((mode.base_position + t) % p::kWindow) * s::kDepth;
    for (std::size_t stage = 0; stage < expected.size(); ++stage) {
      std::copy(kv.begin(), kv.end(), expected[stage].begin() + std::ptrdiff_t(offset));
      HYLE_ASSIGN_OR_RETURN(auto observed, route.read<float>(caught[std::size_t(t) * p::kCatchupStride + stage]));
      HYLE_RETURN_IF_ERROR(s::close(observed, expected[stage], "independent stage main-KV projection/position/wrap",
          s::ds::sem::q::kExpertAbsoluteTolerance, s::ds::sem::q::kExpertRelativeTolerance));
      // Every other ring row must preserve the old bits exactly, including
      // rows overwritten only by the rejected verify suffix at larger t.
      for (int slot = 0; slot < p::kWindow; ++slot) {
        bool touched = false;
        for (int written = 0; written <= t; ++written)
          touched |= slot == (mode.base_position + written) % p::kWindow;
        if (!touched) {
          const auto first = std::size_t(slot) * s::kDepth;
          HYLE_RETURN_IF_ERROR(s::require(std::equal(observed.begin() + std::ptrdiff_t(first),
              observed.begin() + std::ptrdiff_t(first + s::kDepth),
              initial[stage].begin() + std::ptrdiff_t(first)), "unselected main-KV ring row changed"));
        }
      }
    }
  }
  return Status::Ok();
}

Status run_prefix(const s::Case &mode) {
  HYLE_ASSIGN_OR_RETURN(auto context, s::g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(s::m::system_resource(), 256u << 20,
      {.max_blocks = 2048}));
  s::Route route{std::move(context), std::move(queue), std::move(pool)};
  HYLE_ASSIGN_OR_RETURN(auto input, s::prepare_inputs(route, mode));
  std::array<std::vector<float>, p::kDraftStages> initial_rings;
  std::array<c::ReadLease, p::kDraftStages> ring_leases;
  for (std::size_t stage = 0; stage < initial_rings.size(); ++stage) {
    auto &ring = initial_rings[stage]; ring.resize(p::kWindow * s::kDepth);
    for (std::size_t i = 0; i < ring.size(); ++i)
      ring[i] = float(int((i * 11 + stage * 7) % 31) - 15) / 32;
    ring = s::ds::cache_reference(std::move(ring), s::kDepth, false);
    HYLE_ASSIGN_OR_RETURN(ring_leases[stage], route.load(std::as_bytes(std::span(ring))));
  }
  const std::vector<float> initial_carry(p::kMain, .25F);
  HYLE_ASSIGN_OR_RETURN(auto carry, route.load(std::as_bytes(std::span(initial_carry))));
  HYLE_ASSIGN_OR_RETURN(auto state, p::CompleteState::create(mode));
  HYLE_ASSIGN_OR_RETURN(auto base, state.import(input.base_values, ring_leases, carry, 3));
  HYLE_ASSIGN_OR_RETURN(auto old, p::image(route, state, base));

  HYLE_ASSIGN_OR_RETURN(auto verify, route.prepare(s::make_fragment(mode, p::kVerifyRows, s::cursor(mode, 0))));
  HYLE_ASSIGN_OR_RETURN(auto bridge, route.prepare(p::main_bridge(p::kVerifyRows)));
  HYLE_ASSIGN_OR_RETURN(auto catchup, route.prepare(p::main_catchup(p::kVerifyRows, mode.base_position)));
  HYLE_ASSIGN_OR_RETURN(auto arguments, input.inputs(mode, input.base_values, 0, p::kVerifyRows));
  HYLE_ASSIGN_OR_RETURN(auto verified, route.execute(verify, arguments));
  const auto attention = std::size_t(p::kVerifyRows) * state.target_members + (mode.indexed() ? 1u : 0u);
  const std::array bridge_input{verified[attention]};
  HYLE_ASSIGN_OR_RETURN(auto mains, route.execute(bridge, bridge_input));
  std::vector<c::ReadLease> catchup_input{mains[0]};
  catchup_input.insert(catchup_input.end(), ring_leases.begin(), ring_leases.end());
  HYLE_ASSIGN_OR_RETURN(auto caught, route.execute(catchup, catchup_input));
  HYLE_ASSIGN_OR_RETURN(auto fresh, route.read<std::int32_t>(mains[1]));
  HYLE_RETURN_IF_ERROR(s::require(fresh.size() == p::kVerifyRows, "verify fresh/bonus row count"));
  for (const auto id : fresh)
    HYLE_RETURN_IF_ERROR(s::require(id >= 0 && id < 17, "verify fresh target ID range"));
  HYLE_RETURN_IF_ERROR(check_ring_oracle(route, mode, initial_rings, mains[0], caught));

  HYLE_ASSIGN_OR_RETURN(auto single_bridge, route.prepare(p::main_bridge(1)));
  std::vector<s::GraphProgram> single, single_catchup;
  for (int t = 0; t <= p::kVerifyRows; ++t) {
    HYLE_ASSIGN_OR_RETURN(auto program, route.prepare(s::make_fragment(mode, 1, s::cursor(mode, t))));
    single.push_back(std::move(program));
    HYLE_ASSIGN_OR_RETURN(auto update, route.prepare(p::main_catchup(1, mode.base_position + t)));
    single_catchup.push_back(std::move(update));
  }
  std::vector<c::StateVersion> reference{base};
  for (int t = 0; t < p::kVerifyRows; ++t) {
    HYLE_ASSIGN_OR_RETURN(auto next, execute_single(route, mode, input, state, reference.back(), t,
        single[t], single_bridge, single_catchup[t]));
    HYLE_ASSIGN_OR_RETURN(auto version, state.advance(reference.back(), next.target, next.caught, 1, next.pending));
    reference.push_back(std::move(version));
  }

  c::StateVersion full;
  for (int accepted_rows = 0; accepted_rows <= p::kVerifyRows; ++accepted_rows) {
    HYLE_ASSIGN_OR_RETURN(auto head, c::StateHead::create(state.store, base));
    std::span<const c::ReadLease> selected_target, selected_catchup;
    const auto pending = accepted_rows ? fresh[accepted_rows - 1] : 3;
    if (accepted_rows) {
      selected_target = std::span(verified).subspan(std::size_t(accepted_rows - 1) * state.target_members, state.target_members);
      selected_catchup = std::span(caught).subspan(std::size_t(accepted_rows - 1) * p::kCatchupStride, p::kCatchupStride);
    }
    HYLE_ASSIGN_OR_RETURN(auto candidate, state.advance(base, selected_target, selected_catchup, accepted_rows, pending));
    if (accepted_rows) HYLE_RETURN_IF_ERROR(head.publish(base.identity(), candidate));
    HYLE_ASSIGN_OR_RETURN(auto selected, head.current());
    HYLE_RETURN_IF_ERROR(p::compare(route, state, selected, reference[accepted_rows]));
    HYLE_ASSIGN_OR_RETURN(auto preserved, p::image(route, state, base));
    HYLE_RETURN_IF_ERROR(s::require(preserved == old, "accept/reject changed old complete snapshot"));
    HYLE_ASSIGN_OR_RETURN(auto protocol_lease, selected.member(state.entries[state.boundary_entry()]));
    HYLE_ASSIGN_OR_RETURN(auto protocol, route.read<p::Boundary>(protocol_lease));
    HYLE_RETURN_IF_ERROR(s::require(protocol.size() == 1 &&
        protocol[0].target_frontier == std::uint32_t(mode.base_position + accepted_rows) &&
        protocol[0].draft_frontier == protocol[0].target_frontier &&
        protocol[0].accepted_rows == std::uint32_t(accepted_rows) && protocol[0].pending == pending,
        "accepted input/frontier/pending accounting"));
    if (accepted_rows == p::kVerifyRows) {
      full = selected;
      const auto stale = head.publish(base.identity(), base);
      HYLE_RETURN_IF_ERROR(s::require(stale.IsUnavailable(), "stale complete-head publication accepted"));
      HYLE_ASSIGN_OR_RETURN(auto after_stale, head.current());
      HYLE_RETURN_IF_ERROR(s::require(after_stale.identity() == selected.identity(), "stale publication changed head"));
    }
    std::printf("DSpark projected ratio%d accepted_inputs=%d matched_proposals=%d complete_state PASS\n",
        mode.ratio, accepted_rows, accepted_rows ? accepted_rows - 1 : -1);
  }

  // A real failed member replacement after other members were staged must
  // not publish even the target portion. The wrong-size lease remains valid;
  // failure does not consume old/new permissions. No fake Completion is used.
  HYLE_ASSIGN_OR_RETURN(auto failure_head, c::StateHead::create(state.store, base));
  HYLE_ASSIGN_OR_RETURN(auto partial, state.store.begin(base));
  for (std::size_t i = 0; i < state.target_members; ++i)
    HYLE_RETURN_IF_ERROR(partial.replace(state.entries[i], verified[i]));
  const auto bad = partial.replace(state.entries[state.carry_entry()], ring_leases[0]);
  HYLE_RETURN_IF_ERROR(s::require(bad.IsInvalidArgument(), "wrong carry extent was accepted"));
  HYLE_RETURN_IF_ERROR(partial.cancel());
  HYLE_ASSIGN_OR_RETURN(auto unchanged, failure_head.current());
  HYLE_RETURN_IF_ERROR(s::require(unchanged.identity() == base.identity(), "failed draft partially published"));
  HYLE_ASSIGN_OR_RETURN(auto failure_bytes, p::image(route, state, unchanged));
  HYLE_RETURN_IF_ERROR(s::require(failure_bytes == old, "failed draft changed old target/rings/carry"));

  // All-five-accepted includes six target input rows and a fresh bonus. Both
  // the block-selected and one-input reference then evaluate the next supplied
  // projected row via the same Graph route and publish every state member.
  // This is a state continuation check, not a claim that the finite bridge
  // implements original target embeddings or a natural greedy token ledger.
  std::array<c::StateVersion, 2> continued;
  const std::array from{full, reference.back()};
  for (std::size_t arm = 0; arm < from.size(); ++arm) {
    HYLE_ASSIGN_OR_RETURN(auto next, execute_single(route, mode, input, state, from[arm], p::kVerifyRows,
        single[p::kVerifyRows], single_bridge, single_catchup[p::kVerifyRows]));
    HYLE_ASSIGN_OR_RETURN(continued[arm], state.advance(from[arm], next.target, next.caught,
        1, next.pending));
  }
  HYLE_RETURN_IF_ERROR(p::compare(route, state, continued[0], continued[1]));
  HYLE_ASSIGN_OR_RETURN(auto preserved, p::image(route, state, base));
  HYLE_RETURN_IF_ERROR(s::require(preserved == old, "continuation changed old complete snapshot"));
  HYLE_RETURN_IF_ERROR(state.store.collect());
  std::printf("DSpark projected ratio%d target_members=%zu stages=3 verify6/accept0..6, stagedfailure/stalehead/continuation PASS; projected-boundary only\n",
      mode.ratio, state.target_members);
  return Status::Ok();
}

TEST(DSparkAcceptedPrefix, Ratio4TargetIndexAndThreeMainRingsPublishAtomically) {
  // Cross both a compression publication and physical main-KV slot127->0.
  const s::Case mode{4, 4, 1024, 516, 511, 2047};
  const auto status = run_prefix(mode);
  ASSERT_TRUE(status.ok()) << status.message();
}
TEST(DSparkAcceptedPrefix, Ratio128HalfWindowsAndThreeMainRingsPublishAtomically) {
  const auto status = run_prefix(s::kWide);
  ASSERT_TRUE(status.ok()) << status.message();
}
} // namespace
