#pragma once

#include "deepseek_sparse_prefix_fixture.h"

#include <limits>

// Projected-boundary composition, not the complete 43-layer target decoder.
// The target Sink output is an actual Graph result. The finite bridge below
// repeats its first head into a 4096-wide main_x boundary; it is deliberately
// not described as the checkpoint's three-selected-layer main_proj.
// Three DSpark main-KV rings then use the existing C4 weight/Norm/RoPE/QAT
// branch. Proposal-only KV has no path into this immutable state schema.
namespace dspark_accepted_prefix_fixture {
using namespace deepseek_sparse_prefix_fixture;
inline constexpr int kDraftStages = 3, kWindow = 128, kMain = 4096;
inline constexpr int kProposalRows = 5, kVerifyRows = kProposalRows + 1;
inline constexpr int kCatchupStride = kDraftStages + 1; // three rings + carry

struct Boundary final {
  Cursor target;
  std::uint32_t target_frontier, draft_frontier;
  std::int32_t pending;
  std::uint32_t accepted_rows;
};
static_assert(sizeof(Boundary) == 7 * sizeof(std::uint32_t));

inline sem::Fragment main_bridge(int count) {
  sem::Fragment f{.profile = nullptr, .region = 0x4453504252494447};
  auto sink = f.input(dense({1, count, kHeads, kDepth}));
  auto first = ds::slice(f, ds::reshape(f, sink, {count, kHeads, kDepth}), 1, 0, 1);
  auto row = ds::reshape(f, first, {count, kDepth});
  auto main = row;
  for (int repeat = 1; repeat < kMain / kDepth; ++repeat)
    main = ds::concat(f, main, row, 1);
  // A finite actual target score head. These are fresh/bonus IDs from target
  // results, never copied from a proposal tape or from host metadata.
  constexpr int vocabulary = 17;
  std::vector<float> weights(kMain * vocabulary, 0);
  for (int id = 0; id < vocabulary; ++id) {
    weights[id] = float(id - 8) / 8;
    weights[vocabulary + id] = float((id * 7) % 13 - 6) / 8;
  }
  const auto weight = f.constant(dense({kMain, vocabulary}), sem::bytes(weights));
  const auto scores = f.one(c::Operation(c::MatMul{}), {main, weight});
  const auto selected = f.append(c::Operation(c::TopK{.count = 1}), {scores}, 2)[1];
  f.outputs = {copy(f, main), copy(f, ds::reshape(f, selected, {count}))};
  return f;
}

inline sem::Fragment main_catchup(int count, int first_position) {
  sem::Fragment f{.profile = nullptr, .region = 0x4453504341544348, .quantized = true};
  const auto main = f.input(dense({count, kMain}));
  std::array<ValueId, kDraftStages> rings;
  for (auto &ring : rings) ring = f.input(dense({kWindow, kDepth}));
  // Match cpu_c4_model_branches::draft_attention: shared synthetic seed443,
  // BF16 visible activation, dynamic FP8 projection, BF16 norm, RoPE, and
  // FP8 non-RoPE cache boundary. Each stage has an independently owned ring.
  const auto projection = ds::fp8::model_linear(f, main, kDepth, kMain, 443, true);
  auto scale = sem::oracle::scale(kDepth);
  const auto normalized = ds::norm(f, projection.output, scale);
  for (int t = 0; t < count; ++t) {
    const auto position = f.constant(dense({1}, c::DataType::kInt32),
        sem::bytes(std::vector<std::int32_t>{std::int32_t(first_position + t)}));
    const auto single = ds::slice(f, normalized, 0, t, 1);
    const auto rotated = ds::rope(f, ds::reshape(f, single, {1, 1, 1, kDepth}), position, false);
    const auto kv = ds::cache_quant(f, ds::reshape(f, rotated, {1, kDepth}), false);
    for (auto &ring : rings) {
      ring = replace_cache(f, ring, kv, (first_position + t) % kWindow, kWindow);
      f.outputs.push_back(copy(f, ring));
    }
    f.outputs.push_back(copy(f, ds::slice(f, main, 0, t, 1)));
  }
  return f;
}

struct CompleteState final {
  Case mode;
  c::StateSchema schema;
  c::StateStore store;
  std::vector<c::StateEntryId> entries;
  std::size_t target_members = 0;

  std::size_t ring_entry() const { return target_members; }
  std::size_t carry_entry() const { return target_members + kDraftStages; }
  std::size_t boundary_entry() const { return carry_entry() + 1; }

  static Result<CompleteState> create(const Case &mode) {
    CompleteState result{.mode = mode};
    const auto descriptions = state_descriptions(mode);
    std::vector<c::StateEntrySpec> specs;
    for (const auto role : mode.states()) specs.push_back({.description = descriptions[role]});
    result.target_members = specs.size();
    for (int stage = 0; stage < kDraftStages; ++stage)
      specs.push_back({.description = dense({kWindow, kDepth})});
    specs.push_back({.description = dense({1, kMain})});
    specs.push_back({.kind = c::StateEntryKind::kHostMetadata, .metadata_bytes = sizeof(Boundary)});
    HYLE_ASSIGN_OR_RETURN(result.schema, c::StateSchema::create(specs));
    HYLE_ASSIGN_OR_RETURN(result.store, c::StateStore::create(result.schema,
        {.max_versions = 32, .max_drafts = 4, .max_retained_blocks = 1024,
         .max_backing_bytes = (256u << 20) + (128u << 10)}));
    for (std::size_t i = 0; i < specs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto entry, result.schema.entry(i)); result.entries.push_back(entry);
    }
    return result;
  }
  Result<c::StateVersion> import(std::span<const c::ReadLease> target,
      std::span<const c::ReadLease> rings, const c::ReadLease &carry, std::int32_t pending) {
    HYLE_RETURN_IF_ERROR(require(target.size() == target_members && rings.size() == kDraftStages,
                                "complete base member count"));
    std::vector<c::StateMember> members;
    for (std::size_t i = 0; i < target.size(); ++i) members.push_back({entries[i], target[i]});
    for (std::size_t i = 0; i < rings.size(); ++i) members.push_back({entries[ring_entry() + i], rings[i]});
    members.push_back({entries[carry_entry()], carry});
    const auto current = cursor(mode, 0);
    const Boundary facts{current, current.position, current.position, pending, 0};
    members.push_back({entries[boundary_entry()], {}, std::as_bytes(std::span(&facts, 1))});
    return store.import(members);
  }
  Result<std::vector<c::ReadLease>> target(const c::StateVersion &version) const {
    std::vector<c::ReadLease> result;
    for (std::size_t i = 0; i < target_members; ++i) {
      HYLE_ASSIGN_OR_RETURN(auto member, version.member(entries[i])); result.push_back(std::move(member));
    }
    return result;
  }
  Result<std::array<c::ReadLease, kDraftStages>> rings(const c::StateVersion &version) const {
    std::array<c::ReadLease, kDraftStages> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(result[i], version.member(entries[ring_entry() + i]));
    }
    return result;
  }
  Result<Boundary> boundary(const c::StateVersion &version) const {
    HYLE_ASSIGN_OR_RETURN(auto member, version.member(entries[boundary_entry()]));
    HYLE_ASSIGN_OR_RETURN(auto view, member.view());
    Boundary result{};
    HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(&result, 1))));
    HYLE_RETURN_IF_ERROR(require(mode.ratio > 0 && mode.chunk > 0 &&
        result.target.fill < std::uint32_t(mode.chunk) &&
        result.target.fill == result.target.position % std::uint32_t(mode.chunk) &&
        result.target.visible <= std::uint32_t(mode.capacity) && result.pending >= 0 &&
        result.target_frontier == result.target.position &&
        result.draft_frontier == result.target_frontier,
        "complete boundary is not fully caught up"));
    return result;
  }
  // consumed_rows counts this episode's target INPUTS, including old pending.
  // k=0 aborts; k=1..6 corresponds to matching a=k-1 of five proposals.
  // Caller selects target prefix k and catchup prefix k before finalization.
  // Both frontiers include all k verified main rows; pending is target row k-1.
  // The new cursor starts at the actual committed version, never mode.base_position.
  Result<c::StateVersion> advance(const c::StateVersion &from,
      std::span<const c::ReadLease> target, std::span<const c::ReadLease> catchup,
      int consumed_rows, std::int32_t pending) {
    HYLE_RETURN_IF_ERROR(require(consumed_rows >= 0 && consumed_rows <= kVerifyRows,
                                "complete accepted input count"));
    HYLE_ASSIGN_OR_RETURN(auto facts, boundary(from));
    if (!consumed_rows) return from;
    HYLE_RETURN_IF_ERROR(require(pending >= 0, "complete pending ID must be valid"));
    HYLE_RETURN_IF_ERROR(require(target.size() == target_members && catchup.size() == kCatchupStride,
                                "complete replacement member count"));
    const auto rows = std::uint32_t(consumed_rows);
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    HYLE_RETURN_IF_ERROR(require(facts.target.position <= maximum - rows &&
        facts.accepted_rows <= maximum - rows, "complete frontier overflow"));
    const auto position = facts.target.position + rows;
    const auto completed = position / std::uint32_t(mode.ratio) -
        facts.target.position / std::uint32_t(mode.ratio);
    HYLE_RETURN_IF_ERROR(require(completed <= std::uint32_t(mode.capacity) - facts.target.visible,
                                "complete compressed capacity exhausted"));
    facts.target.position = position;
    facts.target.visible += completed;
    facts.target.fill = (facts.target.fill + rows) % std::uint32_t(mode.chunk);
    facts.target_frontier = facts.draft_frontier = position;
    facts.pending = pending;
    facts.accepted_rows += rows;
    HYLE_ASSIGN_OR_RETURN(auto draft, store.begin(from));
    for (std::size_t i = 0; i < target.size(); ++i)
      HYLE_RETURN_IF_ERROR(draft.replace(entries[i], target[i]));
    for (std::size_t i = 0; i < kCatchupStride; ++i)
      HYLE_RETURN_IF_ERROR(draft.replace(entries[ring_entry() + i], catchup[i]));
    HYLE_RETURN_IF_ERROR(draft.replace_metadata(entries[boundary_entry()], std::as_bytes(std::span(&facts, 1))));
    return draft.finalize();
  }
};

using Image = std::vector<std::vector<std::byte>>;
inline Result<Image> image(Route &route, const CompleteState &state, const c::StateVersion &version) {
  Image result;
  for (const auto &entry : state.entries) {
    HYLE_ASSIGN_OR_RETURN(auto member, version.member(entry));
    HYLE_ASSIGN_OR_RETURN(auto bytes, route.read<std::byte>(member)); result.push_back(std::move(bytes));
  }
  return result;
}
inline Status compare(Route &route, const CompleteState &state,
    const c::StateVersion &a, const c::StateVersion &b) {
  for (std::size_t i = 0; i < state.entries.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto x, a.member(state.entries[i]));
    HYLE_ASSIGN_OR_RETURN(auto y, b.member(state.entries[i]));
    if (i == state.boundary_entry()) {
      HYLE_ASSIGN_OR_RETURN(auto lhs, route.read<std::byte>(x));
      HYLE_ASSIGN_OR_RETURN(auto rhs, route.read<std::byte>(y));
      HYLE_RETURN_IF_ERROR(require(lhs == rhs, "complete prefix protocol metadata"));
    } else {
      HYLE_ASSIGN_OR_RETURN(auto lhs, route.read<float>(x));
      HYLE_ASSIGN_OR_RETURN(auto rhs, route.read<float>(y));
      HYLE_RETURN_IF_ERROR(close(lhs, rhs, "complete target/rings/carry"));
    }
  }
  return Status::Ok();
}
} // namespace dspark_accepted_prefix_fixture
