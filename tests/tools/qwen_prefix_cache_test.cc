// Host capacity/directory checks. Metadata-only Snapshots below do not assert
// native state permission; pool capacity uses real leases without model execution.
#include "../../engine/models/qwen35/prefix_cache.h"
#include "../../engine/models/qwen35/batch_schedule.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "../../engine/inference/reusable_output_storage.h"
#include "qwen_geometry_fixture.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

#if !defined(HYLE_PREFIX_CACHE_STANDALONE)
#include <gtest/gtest.h>
#endif

#if defined(HYLE_PREFIX_CACHE_STANDALONE)
namespace { bool reject_allocations = false; }
void *operator new(std::size_t bytes) {
  if (reject_allocations) throw std::bad_alloc();
  if (auto *value = std::malloc(bytes ? bytes : 1)) return value;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *value) noexcept { std::free(value); }
void operator delete[](void *value) noexcept { std::free(value); }
void operator delete(void *value, std::size_t) noexcept { std::free(value); }
void operator delete[](void *value, std::size_t) noexcept { std::free(value); }
#endif

namespace {
namespace p = qwen35::cache;
const auto geometry = qwen_test::q35();
namespace b = qwen35::benchmark;
namespace mtp = hyle::inference::mtp;
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }

// A real, empty permission footprint: metadata tests do not claim initialized
// model state, but checkpoint() must still freeze valid working frames.
p::c::PagedArenaFrame empty_working_frame() {
  namespace c = hyle::compute;
  c::BatchCapacity capacity; capacity.page_size = 1;
  auto schema = c::BatchSchema::create(capacity).value();
  auto arena = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
      hyle::mem::system_resource(), {.max_backing_bytes = 1024}).value();
  const std::array<std::int32_t, 1> zero{0};
  auto layout = schema.version({zero, zero, {}, {}}).value();
  auto read = arena.read(layout, {}).value();
  return c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(read).value()).value();
}
p::c::PagedArenaFrame frozen_alias(const p::c::PagedArenaFrame &frame) {
  return p::c::PagedArenaFrame::from_frozen(frame.snapshot().value()).value();
}

void mtp_prefix_rejection_and_bonus() {
  const std::array<std::int32_t, 1> eos{19};
  std::vector<std::int32_t> emitted;
  for (std::size_t k = 1; k <= 6; ++k) {
    std::vector<std::int32_t> inputs(k + 1, 2); inputs[0] = 7;
    for (std::size_t accepted = 0; accepted <= k; ++accepted) {
      std::vector<std::int32_t> targets(k + 1, 2), finals(k, 3);
      std::vector<std::uint8_t> decisions(k, 1);
      targets[accepted] = 3;
      if (accepted < k) decisions[accepted] = 0;
      const auto greedy = mtp::select_greedy_prefix(inputs, targets, k + 1, eos, 20, emitted);
      require(greedy.consumed == accepted + 1 && greedy.consumed_drafts == accepted &&
          emitted.size() == accepted + 1 && emitted.back() == 3 && !greedy.stopped,
          "greedy first/middle rejection or full-prefix bonus selected another consumed boundary");
      const auto random = mtp::select_sampled_prefix(inputs, decisions, finals, k + 1, eos, 20, emitted);
      require(random.consumed == accepted + 1 && random.consumed_drafts == accepted &&
          emitted.size() == accepted + 1 && emitted.back() == 3 && !random.stopped,
          "sampled first/middle rejection must use that row's residual, all accepted must use the last bonus");
    }
  }
  const std::array<std::int32_t, 1> input{7}, target{3};
  const auto disabled = mtp::select_greedy_prefix(input, target, 1, eos, 20, emitted);
  require(disabled.consumed == 1 && !disabled.consumed_drafts && emitted == std::vector<std::int32_t>{3},
      "K0 must consume only the pending input and emit one new pending token");
  const auto no_draft = mtp::select_sampled_prefix(input, {}, target, 1, eos, 20, emitted);
  require(no_draft.consumed == 1 && !no_draft.consumed_drafts && emitted == std::vector<std::int32_t>{3},
      "confidence-truncated m0 sampling must use one target draw and retain its output pending");
}

void mtp_prefix_eos_and_quota() {
  const std::array<std::int32_t, 1> eos{19};
  std::vector<std::int32_t> emitted;
  for (std::size_t k = 1; k <= 6; ++k) {
    std::vector<std::int32_t> inputs(k + 1, 2), targets(k + 1, 2), finals(k, 4);
    std::vector<std::uint8_t> decisions(k, 1); inputs[0] = 7; inputs.back() = targets[k - 1] = 19;
    auto selected = mtp::select_greedy_prefix(inputs, targets, k + 1, eos, 20, emitted);
    require(selected.consumed == k && selected.consumed_drafts == k - 1 && selected.stopped &&
        emitted.size() == k && emitted.back() == 19,
        "matching EOS must be emitted without consuming its target row");
    selected = mtp::select_sampled_prefix(inputs, decisions, finals, k + 1, eos, 20, emitted);
    require(selected.consumed == k && selected.consumed_drafts == k - 1 && selected.stopped &&
        emitted.size() == k && emitted.back() == 19,
        "sampled accepted EOS must remain pending instead of entering committed KV");
    inputs.back() = targets[k - 1] = 2; targets.back() = finals.back() = 19;
    selected = mtp::select_sampled_prefix(inputs, decisions, finals, k + 1, eos, 20, emitted);
    require(selected.consumed == k + 1 && selected.consumed_drafts == k && selected.stopped && emitted.back() == 19,
        "EOS bonus must consume all nonterminal verifier inputs and remain pending itself");
    for (std::size_t quota = 1; quota <= k + 1; ++quota) {
      selected = mtp::select_sampled_prefix(inputs, decisions, finals, quota, eos, 20, emitted);
      require(selected.consumed == quota && selected.consumed_drafts == quota - 1 && emitted.size() == quota,
          "quota tail must leave the last emitted output pending");
      const auto greedy = mtp::select_greedy_prefix(inputs, targets, quota, eos, 20, emitted);
      require(greedy.consumed == quota && greedy.consumed_drafts == quota - 1 && emitted.size() == quota,
          "greedy quota tail must select the same consumed-input geometry");
    }
  }
}

void mtp_prefix_invalid_inputs_and_rng() {
  const std::array<std::int32_t, 1> eos{19};
  std::vector<std::int32_t> emitted;
  const std::array<std::int32_t, 2> inputs{7,2}, targets{2,3};
  const std::array<std::int32_t, 1> final{3};
  std::size_t rejected = 0;
  const auto must_reject = [&](auto call) { try { call(); } catch (const std::runtime_error &) { ++rejected; } };
  must_reject([&] { (void)mtp::select_greedy_prefix({}, {}, 1, eos, 20, emitted); });
  must_reject([&] { (void)mtp::select_greedy_prefix(inputs, targets, 0, eos, 20, emitted); });
  must_reject([&] { (void)mtp::select_greedy_prefix(inputs, final, 2, eos, 20, emitted); });
  must_reject([&] { (void)mtp::select_greedy_prefix(inputs, targets, 2, eos, 2, emitted); });
  must_reject([&] { (void)mtp::select_greedy_prefix(eos, eos, 1, eos, 20, emitted); });
  const std::array<std::uint8_t, 1> invalid{2};
  must_reject([&] { (void)mtp::select_sampled_prefix(inputs, invalid, final, 2, eos, 20, emitted); });
  require(rejected == 6, "malformed MTP selections must reject before state/RNG publication");
  for (std::uint64_t actual = 0; actual <= 6; ++actual) {
    const p::SamplingCursor base{123,4,11};
    for (std::uint64_t i = 0; i < actual; ++i)
      require(base.identity(2*i) != base.identity(2*i+1), "proposal and acceptance need independent addresses");
    const auto next = base.advanced(2*actual+1);
    require(next.draw == 12+2*actual && base.draw == 11 && base.identity(2*actual) != next.identity(),
        "one actual truncated proposal block must reserve exactly 2m+1 draws without mutating the old cursor");
  }
}

void mtp_request_stop_policy() {
  namespace inference = hyle::inference;
  const std::array<std::int32_t, 1> eos{19}, first_stop{2}, middle_stop{3}, bonus_stop{4}, extra_eos{19};
  const inference::RequestStopPolicy first(first_stop), middle(middle_stop), bonus(bonus_stop),
      rejected(extra_eos), ignore({}, true), ignored_but_extra(extra_eos, true);
  const std::array<std::int32_t, 3> inputs{7,2,3}, targets{2,3,4};
  const std::array<std::uint8_t, 2> accepts{1,1};
  const std::array<std::int32_t, 2> finals{3,4};
  std::vector<std::int32_t> emitted;
  for (const bool sampled : {false, true}) {
    const auto choose = [&](const inference::RequestStopPolicy &policy) {
      return sampled ? mtp::select_sampled_prefix(inputs,accepts,finals,3,eos,20,emitted,&policy)
                     : mtp::select_greedy_prefix(inputs,targets,3,eos,20,emitted,&policy);
    };
    auto selected = choose(first);
    require(selected.consumed == 1 && selected.consumed_drafts == 0 && selected.next_id == 2 &&
        selected.stopped && emitted == std::vector<std::int32_t>{2},
        "accepted first extra-stop was consumed instead of remaining pending");
    selected = choose(middle);
    require(selected.consumed == 2 && selected.consumed_drafts == 1 && selected.next_id == 3 &&
        selected.stopped && emitted == std::vector<std::int32_t>({2,3}),
        "accepted middle extra-stop selected a later consumed prefix");
    selected = choose(bonus);
    require(selected.consumed == 3 && selected.consumed_drafts == 2 && selected.next_id == 4 &&
        selected.stopped && emitted == std::vector<std::int32_t>({2,3,4}),
        "extra-stop bonus did not stay pending after the complete consumed verifier prefix");
    const std::array<std::int32_t, 3> bad_draft{7,19,3};
    const std::array<std::uint8_t, 2> rejected_flags{0,1};
    const std::array<std::int32_t, 2> rejected_finals{2,4};
    selected = sampled ? mtp::select_sampled_prefix(bad_draft,rejected_flags,rejected_finals,3,eos,20,emitted,&rejected)
                       : mtp::select_greedy_prefix(bad_draft,targets,3,eos,20,emitted,&rejected);
    require(selected.consumed == 1 && !selected.consumed_drafts && selected.next_id == 2 &&
        !selected.stopped && emitted == std::vector<std::int32_t>{2},
        "a rejected stop proposal terminated the request");
    const std::array<std::int32_t, 2> eos_pending{19,2}, eos_target{2,19};
    const std::array<std::uint8_t, 1> agree{1};
    const std::array<std::int32_t, 1> eos_final{19};
    selected = sampled ? mtp::select_sampled_prefix(eos_pending,agree,eos_final,2,eos,20,emitted,&ignore)
                       : mtp::select_greedy_prefix(eos_pending,eos_target,2,eos,20,emitted,&ignore);
    require(selected.consumed == 2 && selected.consumed_drafts == 1 && selected.next_id == 19 &&
        !selected.stopped && emitted == std::vector<std::int32_t>({2,19}),
        "ignoreEOS failed to admit a pending model EOS or incorrectly stopped its bonus");
    const std::array<std::int32_t, 2> ordinary_pending{7,2};
    selected = sampled ? mtp::select_sampled_prefix(ordinary_pending,agree,eos_final,2,eos,20,emitted,&ignored_but_extra)
                       : mtp::select_greedy_prefix(ordinary_pending,eos_target,2,eos,20,emitted,&ignored_but_extra);
    require(selected.consumed == 2 && selected.stopped && selected.next_id == 19,
        "ignoreEOS incorrectly disabled the same ID's explicit extra-stop policy");
  }
  std::array<std::int32_t, 4> bounded{-77,-77,-77,-77};
  const auto selected = mtp::select_greedy_prefix(inputs,targets,3,eos,20,std::span(bounded),&first);
  require(selected.consumed == 1 && bounded == std::array<std::int32_t,4>{2,-77,-77,-77},
      "bounded stop selection wrote outside its actual emitted prefix");
  bool refused = false;
  const std::array<std::int32_t, 1> terminal{19};
  try { (void)mtp::select_greedy_prefix(terminal,terminal,1,eos,20,emitted); }
  catch (const std::runtime_error &) { refused = true; }
  require(refused,"default MTP policy consumed a terminal pending model EOS");
}

void schedule_waves_keep_dependencies() {
  const std::array sessions{
    b::SessionSchedule{"a", {{b::PhaseKind::kPrompt, {1,2,3}, 1},
        {b::PhaseKind::kDecode, {4,5}, 1}, {b::PhaseKind::kPrompt, {6,7}, 2},
        {b::PhaseKind::kDecode, {8}, 2}}},
    b::SessionSchedule{"b", {{b::PhaseKind::kPrompt, {9}, 1}, {b::PhaseKind::kDecode, {10}, 1}}},
    b::SessionSchedule{"c", {{b::PhaseKind::kPrompt, {11,12}, 1}}}
  };
  const auto waves = b::make_schedule_waves(sessions, 2);
  require(waves.size() == 6 && waves[0].calls.size() == 2 && waves[1].calls.size() == 2 &&
      waves[2].calls.size() == 1 && waves[3].cohort == 1 && waves[3].calls[0].session == 2 &&
      waves[4].turn == 2 && waves[4].calls[0].session == 0,
      "turn/cohort waves must not refill across a still active cohort");
  std::array<std::size_t, 3> next{};
  std::size_t ordinal = 0;
  for (const auto &wave : waves) {
    std::array<bool, 3> seen{};
    for (const auto &call : wave.calls) {
      require(!seen[call.session] && call.call == next[call.session]++ && call.ordinal == ordinal++,
          "a wave repeated or reordered a Session dependency");
      seen[call.session] = true;
      require(sessions[call.session].phases[call.phase].turn == wave.turn,
          "a later turn crossed an earlier cohort");
    }
  }
  require(next == std::array<std::size_t, 3>{5,2,1} && waves[2].calls[0].local == 1 &&
      waves[4].calls[0].phase == 2, "unequal decode lengths lost original call indices");
  const auto single = b::make_schedule_waves(sessions, 1);
  require(single.size() == 8 && single[0].calls[0].session == 0 &&
      single[2].calls[0].session == 0 && single[3].calls[0].session == 1 &&
      single[5].calls[0].session == 2 && single[6].turn == 2,
      "capacity1 must use the same turn/cohort plan without interleaving a later user");
  bool rejected = false;
  try { (void)b::make_schedule_waves(sessions, 0); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "zero active capacity must reject before execution");
}

void schedule_runtime_buckets_use_actual_rows() {
  const b::ScheduleWave wave{1, 0, {
    {0,0,0,0,0,b::PhaseKind::kPrompt,4}, {1,0,0,0,1,b::PhaseKind::kPrompt,2},
    {2,1,1,0,2,b::PhaseKind::kDecode,1}, {3,0,0,0,3,b::PhaseKind::kPrompt,7}}};
  const std::array<std::size_t, 4> actual{2,2,1,0};
  const auto buckets = b::bucket_schedule_wave(wave, actual);
  require(wave.calls.size() == 4 && buckets.size() == 2 && buckets[0].query_rows == 2 &&
      buckets[0].kind == b::PhaseKind::kPrompt && buckets[0].calls == std::vector<std::size_t>{0,1} &&
      buckets[1].kind == b::PhaseKind::kDecode && buckets[1].calls == std::vector<std::size_t>{2},
      "only effective equal kind/T calls may batch; full hit retains its logical wave entry");
  const std::array<std::size_t, 4> all_one{1,1,1,0};
  const auto one = b::bucket_schedule_wave(wave, all_one);
  require(one.size() == 2 && one[0].calls.size() == 2 && one[1].calls.size() == 1,
      "prompt1 must not merge with decode1 just because query rows match");
  bool rejected = false;
  try { (void)b::bucket_schedule_wave(wave, std::array<std::size_t, 4>{5,2,1,0}); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "actual query expansion cannot be hidden by padding");
}

void schedule_shape_domain_is_wave_local() {
  const std::array sessions{
    b::SessionSchedule{"a", {{b::PhaseKind::kPrompt, {1,2,3,4}, 1}, {b::PhaseKind::kDecode, {5}, 1}}},
    b::SessionSchedule{"b", {{b::PhaseKind::kPrompt, {1,2}, 1}, {b::PhaseKind::kDecode, {6}, 1}}},
    b::SessionSchedule{"c", {{b::PhaseKind::kPrompt, {1,2,3}, 1}, {b::PhaseKind::kDecode, {7}, 1}}}
  };
  const auto waves = b::make_schedule_waves(sessions, 2);
  const auto find = [](const auto &shapes, std::size_t requests, std::size_t rows) {
    return std::find_if(shapes.begin(), shapes.end(), [&](const auto &shape) {
      return shape.requests == requests && shape.query_rows == rows;
    });
  };
  const auto nominal = b::plan_batch_shape_uses(waves);
  require(find(nominal, 2, 1) != nominal.end() && find(nominal, 2, 2) == nominal.end() &&
      find(nominal, 2, 3) == nominal.end() && find(nominal, 2, 4) == nominal.end(),
      "different prompt shapes or different cohorts cannot form a cold shape Cartesian product");
  // Only the first wave's two prompts can both become T2. C's possible T1
  // lives in a later cohort and must not enlarge any earlier decode bucket.
  std::vector<std::vector<std::size_t>> options(6);
  options[0] = {0,2,4,2}; options[1] = {0,2}; options[4] = {0,1,3};
  const auto dynamic = b::plan_batch_shape_uses(waves, options);
  const auto r2t2 = find(dynamic, 2, 2), r2t1 = find(dynamic, 2, 1), r1t2 = find(dynamic, 1, 2);
  require(r2t2 != dynamic.end() && r2t2->max_wave_calls == 1 &&
      r2t1 != dynamic.end() && r2t1->max_wave_calls == 1 &&
      r1t2 != dynamic.end() && r1t2->max_wave_calls == 2 &&
      find(dynamic, 2, 3) == dynamic.end() && find(dynamic, 2, 4) == dynamic.end(),
      "reachable same-wave alternatives must bound actual width and replay uses");
  for (auto &rows : options) rows = {0};
  const auto hits = b::plan_batch_shape_uses(waves, options);
  require(hits.size() == nominal.size() && std::all_of(hits.begin(), hits.end(), [&](const auto &shape) {
      const auto original = find(nominal, shape.requests, shape.query_rows);
      return original != nominal.end() && original->max_wave_calls == shape.max_wave_calls;
    }), "full-hit alternatives must not remove the real cache-miss/restore shape domain");
}

p::TokenPlanRef tokens(std::initializer_list<std::int32_t> ids) {
  return p::token_plan(std::span(ids.begin(), ids.size()), ids.size());
}
p::SnapshotRef snapshot(std::size_t length, const void *model, const void *math, std::int32_t next = 71) {
  auto value = std::make_shared<p::State>();
  value->length = length; value->model = model; value->math = math; value->next_id = next; return value;
}
p::ReplayPrefix history(const p::TokenPlanRef &input, const p::SnapshotRef &head) {
  auto value = std::make_shared<p::ReplayHistory>(input->ids.size());
  value->append(input, 0, head->length, head->next_id);
  return {value, 1, head->length};
}
bool publish(p::PrefixCache &cache, const p::TokenPlanRef &input, const p::SnapshotRef &head) {
  return cache.publish(input, head, history(input, head));
}

void sampling_cursor_addresses_are_transactional() {
  const p::SamplingCursor base{0x1111111122222222ULL, 0x3333333344444444ULL, 0xffffffffULL};
  require(base.identity() == std::array<std::uint32_t,4>{0x66666666u,0x22222222u,0xffffffffu,0u},
      "sampling key and draw words must preserve the explicit 64-bit address");
  require(base.identity(1) == std::array<std::uint32_t,4>{0x66666666u,0x22222222u,0u,1u},
      "draw offsets must carry into the high address word");
  const auto candidate = base.advanced(3);
  require(candidate.seed == base.seed && candidate.stream == base.stream && candidate.draw == 0x100000002ULL,
      "advancing a candidate must preserve its seed/stream");
  require(base.draw == 0xffffffffULL && base.identity() != candidate.identity(),
      "preparing or abandoning a candidate must not publish its cursor");
  const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
  const p::SamplingCursor exhausted{7,13,maximum};
  require(exhausted.advanced(0) == exhausted && exhausted.identity()[2] == UINT32_MAX &&
      exhausted.identity()[3] == UINT32_MAX, "the final exact draw address remains representable");
  bool identity_rejected = false, advance_rejected = false;
  try { (void)exhausted.identity(1); } catch (const std::runtime_error &) { identity_rejected = true; }
  try { (void)exhausted.advanced(1); } catch (const std::runtime_error &) { advance_rejected = true; }
  require(identity_rejected && advance_rejected && exhausted.draw == maximum,
      "address/commit overflow must reject without mutating the original cursor");
}

void sampling_wrappers_share_one_state_version() {
  int model = 0, math = 0, other_model = 0, other_math = 0;
  p::VersionCount versions{0,1};
  auto canonical = std::make_shared<p::State>();
  canonical->version = p::VersionRef::acquire(versions).value();
  canonical->model = &model; canonical->math = &math; canonical->length = 3;
  canonical->next_id = 71; canonical->sampling = {1,2,3};
  std::weak_ptr<const p::State> lifetime = canonical;
  auto first = std::make_shared<p::State>();
  first->state_origin = canonical;
  first->model = &model; first->math = &math; first->length = 3;
  first->next_id = 81; first->sampling = {11,12,1};
  auto second = std::make_shared<p::State>();
  second->state_origin = first->state_origin;
  second->model = &model; second->math = &math; second->length = 3;
  second->next_id = 91; second->sampling = {21,22,1};
  require(first->state_identity() == canonical.get() && second->state_identity() == canonical.get() &&
      !second->state_origin->state_origin && versions.live == 1 && !p::VersionRef::acquire(versions),
      "sampling forks must retain one canonical evaluated-state version, without origin chains");

  p::PrefixCache cache(1);
  const auto input = tokens({4,5,6});
  require(publish(cache,input,first) && cache.retains(canonical.get()) && cache.retains(second.get()),
      "directory retention must recognize every wrapper of the same canonical state");
  second->sampling = second->sampling.advanced(4); second->next_id = 92;
  auto hit = cache.lookup(*input,3,0,&model,&math);
  require(hit && hit.payload.snapshot == first && hit.payload.snapshot->state_identity() == second->state_identity() &&
      hit.payload.snapshot->sampling == p::SamplingCursor{11,12,1} && hit.payload.snapshot->next_id == 81 &&
      canonical->sampling == p::SamplingCursor{1,2,3} && canonical->next_id == 71,
      "same-token model-state reuse must not share request cursors or pending tokens");
  require(!cache.lookup(*input,3,0,&other_model,&math) && !cache.lookup(*input,3,0,&model,&other_math),
      "sampler metadata independence must not weaken actual model/math isolation");
  p::SnapshotRef escaped = hit.payload.snapshot;
  hit.reset(); canonical.reset(); first.reset(); second.reset();
  require(cache.evict_one() && !lifetime.expired() && versions.live == 1,
      "an escaped request fork must retain its canonical version after directory eviction");
  escaped.reset();
  require(lifetime.expired() && versions.live == 0,
      "the canonical finite version must retire exactly once after its last fork");
}

void working_state_reuses_metadata_and_freezes_explicitly() {
  static_assert(!std::is_copy_constructible_v<p::WorkingState>);
  p::VersionCount versions{0,2}; p::WorkingState working;
  const auto next = [&](std::size_t length) {
    auto candidate = working.acquire_candidate();
    candidate->version = p::VersionRef::acquire(versions).value();
    candidate->pages.reserve(8);
    if (!candidate->keys.try_resize(3) || !candidate->values.try_resize(3) ||
        !candidate->state.try_resize(2)) throw std::bad_alloc();
    if (!working)
      for (std::size_t i = 0; i < candidate->keys.size(); ++i) {
        candidate->keys[i] = empty_working_frame(); candidate->values[i] = empty_working_frame();
      }
    candidate->length = length; candidate->next_id = 17; candidate->sampling = {1,2,length};
    working.commit(std::move(candidate), [](p::State *previous, p::State &next) noexcept {
      if (previous)
        for (std::size_t i = 0; i < next.keys.size(); ++i) {
          next.keys[i] = std::move(previous->keys[i]); next.values[i] = std::move(previous->values[i]);
        }
    }, [](p::State &) noexcept {});
  };
  next(1); const auto *first = working.get();
  next(2); const auto *second = working.get();
  bool stable = first != second;
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = true;
#endif
  try {
    for (std::size_t i = 0; i < 128; ++i) {
      next(i + 3);
      stable = stable && working.get() == (i % 2 ? second : first) && versions.live == 1;
    }
  } catch (...) {
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
    reject_allocations = false;
#endif
    throw;
  }
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = false;
#endif
  require(stable, "ordinary commits allocated/replaced warm metadata or kept an old version");
  const auto *identity = working.get(); const auto epoch = working.epoch();
  std::array<p::c::PagedArenaFootprint, 6> footprints;
  for (std::size_t i = 0; i < working->keys.size(); ++i) {
    footprints[i * 2] = working->keys[i].view().value();
    footprints[i * 2 + 1] = working->values[i].view().value();
  }
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  bool rejected = false; reject_allocations = true;
  try { (void)working.checkpoint(); }
  catch (const std::bad_alloc &) { rejected = true; }
  catch (...) { reject_allocations = false; throw; }
  reject_allocations = false;
  require(rejected && working.get() == identity && working.epoch() == epoch && versions.live == 1 &&
      working->length == 130 && working->next_id == 17 && working->sampling == p::SamplingCursor{1,2,130},
      "checkpoint allocation failure changed the unique committed boundary or version");
  for (std::size_t i = 0; i < working->keys.size(); ++i)
    require(!working->keys[i].sealed() && !working->values[i].sealed() &&
        footprints[i * 2].valid() && footprints[i * 2 + 1].valid(),
        "checkpoint allocation failure sealed or invalidated a working frame");
#endif
  auto frozen = working.checkpoint();
  require(frozen.get() == identity && working.get() == identity && working.epoch() == epoch && versions.live == 1,
      "checkpoint copied State, changed epoch or counted a second version");
  for (std::size_t i = 0; i < frozen->keys.size(); ++i)
    require(frozen->keys[i].sealed() && frozen->values[i].sealed() &&
        footprints[i * 2].valid() && footprints[i * 2 + 1].valid(),
        "checkpoint failed to permanently seal the original frame metadata");
  require(working.checkpoint() == frozen, "repeated checkpoint replaced an immutable committed identity");
  working.reset(); require(versions.live == 1 && frozen->length == 130, "reset revoked an escaped checkpoint");
  frozen.reset(); require(!versions.live, "final checkpoint retained a model-state version");
}

void working_state_adopts_sampling_origin_without_another_version() {
  p::VersionCount versions{0,2}; p::WorkingState source, fork;
  auto initial = source.acquire_candidate(); initial->version = p::VersionRef::acquire(versions).value();
  initial->length = 3; initial->next_id = 71; initial->sampling = {1,2,3};
  initial->draft.emplace(); initial->draft->frontier = 2; initial->draft->lag = 1; initial->draft->catchup_token = 5;
  initial->draft->key = empty_working_frame(); initial->draft->value = empty_working_frame();
  source.commit(std::move(initial)); auto canonical = source.checkpoint();
  fork.adopt(canonical); source.reset();
  require(fork.get() == canonical.get() && versions.live == 1, "checkpoint adoption copied its version");
  auto wrapper = fork.acquire_candidate();
  wrapper->state_origin = canonical; wrapper->length = canonical->length; wrapper->draft.emplace();
  wrapper->draft->pages = canonical->draft->pages;
  wrapper->draft->key = frozen_alias(canonical->draft->key);
  wrapper->draft->value = frozen_alias(canonical->draft->value);
  wrapper->draft->frontier = canonical->draft->frontier; wrapper->draft->lag = canonical->draft->lag;
  wrapper->draft->catchup_token = canonical->draft->catchup_token;
  wrapper->next_id = 81; wrapper->sampling = {11,12,13};
  fork.commit(std::move(wrapper)); auto sampled = fork.checkpoint(); canonical.reset();
  require(versions.live == 1 && sampled->state_identity() == sampled->state_origin.get() &&
      !sampled->state_origin->state_origin && sampled->state_identity()->next_id == 71 && sampled->next_id == 81 &&
      sampled->draft->frontier == 2 && sampled->draft->lag == 1 && sampled->draft->catchup_token == 5,
      "sampling wrapper changed canonical pending, complete draft metadata or version count");
  auto evaluated = fork.acquire_candidate(); evaluated->version = p::VersionRef::acquire(versions).value();
  evaluated->length = 4; evaluated->next_id = 91; fork.commit(std::move(evaluated));
  require(versions.live == 2 && sampled->next_id == 81 && fork->next_id == 91,
      "new unique state overwrote its frozen sampling boundary or lost a version");
  fork.reset(); require(versions.live == 1, "working owner retained a retired evaluated version");
  sampled.reset(); require(!versions.live, "sampling origin retained a canonical version after final release");
}

void recycled_candidate_releases_permissions_and_owners() {
  namespace c = hyle::compute;
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 257, {.max_blocks = 1}).value();
  p::VersionCount versions{0,1}, origins{0,1}; p::WorkingState working;
  auto owner = std::make_shared<int>(7); std::weak_ptr<const void> lifetime = owner;
  auto canonical = std::make_shared<p::State>(); canonical->version = p::VersionRef::acquire(origins).value();
  auto candidate = working.acquire_candidate(); const auto *metadata = candidate.get();
  candidate->version = p::VersionRef::acquire(versions).value(); candidate->resource_owner = owner;
  candidate->state_origin = canonical; candidate->pages.reserve(8);
  require(candidate->state.try_resize(1), "candidate state metadata could not be prepared");
  auto writer = pool.allocate(128).value();
  candidate->state[0].history = writer.publish().value(); owner.reset(); canonical.reset();
  require(pool.stats().live_blocks == 1 && versions.live == 1 && origins.live == 1 && !lifetime.expired(),
      "candidate did not own its actual lease, resource owner and canonical versions");
  working.recycle_candidate(std::move(candidate));
  require(!versions.live && !origins.live && lifetime.expired() && !pool.stats().live_blocks && !working,
      "empty candidate metadata retained an old lease, owner, origin or version");
  auto reused = working.acquire_candidate();
  require(reused.get() == metadata && reused->pages.capacity() >= 8 && reused->pages.empty() && reused->state.empty() &&
      !reused->state_origin && !reused->resource_owner && !reused->carry_hidden && !reused->prior_hidden && !reused->draft,
      "candidate reuse lost metadata capacity or revived old permissions");
}

struct ReusePayloadResource final : hyle::mem::MemoryResource {
  std::size_t allocations = 0, mappings = 0;
  std::string_view name() const noexcept override { return "reusable-output-test"; }
  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::system_resource().resource().properties();
  }
  hyle::Result<hyle::mem::ResourceAllocation> allocate(
      std::size_t bytes, std::size_t alignment) noexcept override {
    ++allocations;
    return hyle::mem::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    hyle::mem::system_resource().resource().deallocate(allocation);
  }
  hyle::Result<hyle::mem::ResourceHostMapping> map_host(void *handle,
      std::size_t offset, std::size_t bytes, hyle::mem::HostAccess access) noexcept override {
    ++mappings;
    return hyle::mem::system_resource().resource().map_host(handle, offset, bytes, access);
  }
  hyle::Status unmap_host(void *handle, void *token, std::size_t offset,
      std::size_t bytes, hyle::mem::HostAccess access) noexcept override {
    return hyle::mem::system_resource().resource().unmap_host(handle, token, offset, bytes, access);
  }
};

void reusable_output_storage_is_bounded_and_preserves_rejected_readers() {
  namespace c = hyle::compute;
  hyle::inference::ReusableOutputStorage cache;
  require(cache.prepare(1, 128).ok() && cache.capacity() == 1 && cache.bytes() == 128 && !cache.take(),
      "cold reusable output storage must begin empty with its explicit limit");
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 769, {.max_blocks = 3}).value();
  const auto ready = [&](std::size_t bytes) {
    auto writer = pool.allocate(bytes).value();
    require(writer.view()->fill(std::byte{17}).ok(), "reusable output fixture fill failed");
    return writer.publish().value();
  };
  auto wrong_size = ready(64), first = ready(128), second = ready(128);
  const auto wrong_generation = wrong_size.generation(), second_generation = second.generation();
  require(!cache.recycle(wrong_size) && wrong_size.valid() && wrong_size.generation() == wrong_generation && !cache.size(),
      "a different output extent must remain caller-owned");
  auto shared = first;
  require(!cache.recycle(first) && first.valid() && shared.valid() && !cache.size(),
      "a retained snapshot reader must prevent writer recovery without consumption");
  shared.reset();
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto admitted = uses.begin().value();
  require(admitted.read(first).ok(), "original output read admission failed");
  require(!cache.recycle(first) && first.valid() && !cache.size(),
      "an original admitted native reader must prevent recycling");
  admitted = {};
  require(cache.recycle(first) && !first.valid() && cache.size() == 1,
      "the same output should promote only after its final reader withdraws");
  require(!cache.recycle(second) && second.valid() && second.generation() == second_generation && cache.size() == 1,
      "a full reusable output cache must leave a rejected lease unchanged");
  require(!cache.prepare(2, 128).ok() && cache.capacity() == 1 && cache.size() == 1,
      "cold reconfiguration must not discard cached writer permissions");
  cache.clear();
  require(!cache.size() && !cache.take() && pool.stats().live_blocks == 2,
      "clearing cached writers must release their original blocks without replacing them");
  require(cache.prepare(0, 128).ok() && !cache.capacity() && !cache.recycle(second) && second.valid(),
      "zero capacity must disable recycling without consuming the offered output");
  require(!cache.prepare(1, 0).ok() && !cache.capacity() && cache.bytes() == 128,
      "invalid cold preparation must preserve the previous configuration");
  hyle::inference::ReusableOutputStorage ordered;
  require(ordered.prepare(2, 128).ok(), "ordered state cache preparation failed");
  auto third = ready(128);
  const auto third_generation = third.generation();
  require(ordered.recycle(second) && ordered.recycle(third), "ordered state rows did not recycle");
  auto older = ordered.take(), newer = ordered.take();
  require(older && newer && older->generation() == second_generation && newer->generation() == third_generation,
      "state reuse must preserve stable batch member order instead of reversing its address tuple");
}

void reusable_output_storage_reuses_generation_and_address_without_payload_work() {
  namespace c = hyle::compute;
  ReusePayloadResource resource;
  auto pool = c::LeasePool::create(hyle::mem::MemoryResourceRef(resource), 257, {.max_blocks = 1}).value();
  hyle::inference::ReusableOutputStorage cache;
  require(cache.prepare(1, 128).ok(), "reusable output metadata preparation failed");
  auto writer = pool.allocate(128).value();
  require(writer.view()->fill(std::byte{29}).ok(), "original output fill failed");
  const auto generation = writer.generation();
  const auto original = writer.view().value();
  auto ready = writer.publish().value();
  const auto allocations = resource.allocations, mappings = resource.mappings;
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = true;
#endif
  try {
    for (std::size_t step = 0; step < 64; ++step) {
      require(cache.recycle(ready) && !ready.valid(), "successful recycle did not transfer its original output");
      auto recovered = cache.take();
      require(recovered && !cache.size() && recovered->generation() == generation,
          "reusable output changed generation or retained duplicate write permission");
      const auto storage = recovered->view().value();
      require(storage.native_handle() == original.native_handle() && storage.offset() == original.offset() &&
          storage.size() == original.size(), "reusable output moved or resized its physical storage");
      ready = recovered->publish().value();
    }
    require(cache.recycle(ready), "final original output recycle failed");
    hyle::inference::ReusableOutputStorage moved(std::move(cache));
    require(!cache.size() && !cache.take() && moved.size() == 1,
        "moving the cache retained a live source counter or lost a writer");
    cache = std::move(moved);
    require(!moved.size() && !moved.take() && cache.size() == 1,
        "move assignment duplicated cached write permission");
    auto recovered = cache.take();
    ready = recovered->publish().value();
    require(resource.allocations == allocations && resource.mappings == mappings &&
        pool.stats().live_blocks == 1 && !pool.stats().reclaimed_blocks,
        "recycle/take must not allocate, map, copy or reallocate original payload storage");
  } catch (...) {
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
    reject_allocations = false;
#endif
    throw;
  }
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = false;
#endif
  std::array<std::byte, 128> bytes{};
  require(ready.view()->read(bytes).ok() && std::ranges::all_of(bytes, [](auto byte) { return byte == std::byte{29}; }),
      "writer reuse modified the original successfully retired output bytes");
}

void working_state_recycles_only_successfully_superseded_unique_heads() {
  namespace c = hyle::compute;
  auto pool = c::LeasePool::create(hyle::mem::system_resource(), 769, {.max_blocks = 3}).value();
  hyle::inference::ReusableOutputStorage cache;
  require(cache.prepare(1, 128).ok(), "commit-hook cache metadata preparation failed");
  p::VersionCount versions{0, 3}; p::PageInventory pages; pages.prepare(1, 0);
  p::WorkingState working;
  auto token = std::make_shared<int>(23); std::weak_ptr<const void> old_owner = token;
  const auto candidate = [&](std::size_t length) {
    auto next = working.acquire_candidate();
    next->version = p::VersionRef::acquire(versions).value(); next->length = length;
    require(next->state.try_resize(1), "commit-hook state metadata preparation failed");
    auto writer = pool.allocate(128).value();
    require(writer.view()->fill(std::byte(length)).ok(), "commit-hook original output fill failed");
    next->state[0].history = writer.publish().value();
    return next;
  };
  auto first = candidate(1);
  const auto generation = first->state[0].history.generation();
  auto page = pages.acquire().value(); const auto page_id = page.reference.page();
  first->pages.push_back(std::move(page.reference)); first->resource_owner = token; token.reset();
  std::size_t callbacks = 0; bool hook_valid = true;
  const auto recycle = [&](p::State &previous) noexcept {
    ++callbacks;
    hook_valid = hook_valid && previous.length == 1 && versions.live == 2 &&
        previous.pages.size() == 1 && pages.owners[page_id] == 1 && !old_owner.expired();
    hook_valid = hook_valid && cache.recycle(previous.state[0].history);
  };
  working.commit(std::move(first), recycle);
  require(!callbacks && !cache.size() && versions.live == 1 && pages.owners[page_id] == 1,
      "initial commit must not recycle a nonexistent previous head");
  working.commit(candidate(2), recycle);
  require(callbacks == 1 && hook_valid && cache.size() == 1 && versions.live == 1 &&
      old_owner.expired() && pages.owners[page_id] == 0 && pages.free_ids.size() == 1,
      "successful old-head recycling must not retain its VersionRef, page policy or resource owner");
  auto frozen = working.checkpoint();
  const auto frozen_generation = frozen->state[0].history.generation();
  auto failed = candidate(99);
  working.recycle_candidate(std::move(failed));
  require(callbacks == 1 && cache.size() == 1 && versions.live == 1 && working.get() == frozen.get(),
      "discarding a failed candidate must not recycle it or disturb a committed frozen head");
  working.commit(candidate(3), recycle);
  require(callbacks == 1 && cache.size() == 1 && versions.live == 2 && frozen->length == 2 &&
      frozen->state[0].history.generation() == frozen_generation && frozen->state[0].history.view(),
      "a frozen previous boundary must not reach the unique-old-head hook");
  auto recovered = cache.take();
  require(recovered && recovered->generation() == generation, "the hook did not retain the original successful old output");
  recovered.reset(); working.reset();
  require(callbacks == 1 && versions.live == 1 && pool.stats().live_blocks == 1,
      "cache and working reset must leave only the explicitly escaped frozen version");
  frozen.reset();
  require(!versions.live && !pool.stats().live_blocks, "the output cache retained a frozen version after final release");
}

void completed_consumed_checkpoints() {
  int model=0,math=0;
  p::VersionCount versions{0,2};
  auto head=std::make_shared<p::State>();
  head->version=p::VersionRef::acquire(versions).value();
  head->model=&model;head->math=&math;head->length=8;head->next_id=248044;
  head->sampling={17,29,13};
  head->draft.emplace();head->draft->frontier=7;head->draft->lag=1;head->draft->catchup_token=8;
  // Metadata-only frames here still make no native state-permission claim.
  // The directory must retain this exact complete Snapshot, not reconstruct
  // a target-only/lag0/RNG-default replacement from its token key.
  std::array<std::int32_t,3> prompt{1,2,3};
  const std::array<std::int32_t,1> accept1{4};
  const std::array<std::int32_t,2> accept2{5,6},next_turn{7,8};
  const std::array<p::ConsumedChunk,4> chunks{{{prompt,4},{accept1,5},{accept2,7},{next_turn,248044}}};
  auto checkpoint=p::consumed_checkpoint(chunks,*head,12);
  require(checkpoint.input->ids==std::vector<std::int32_t>{1,2,3,4,5,6,7,8} &&
      checkpoint.input->checkpoints==std::vector<std::size_t>{8} &&
      checkpoint.input->hashes.size()==9 && checkpoint.replay.length==8 && checkpoint.replay.chunks==4,
      "completed key must contain exactly consumed inputs, excluding pending EOS and hypothetical source boundaries");
  prompt[0]=99;
  require(checkpoint.input->ids[0]==1 && checkpoint.replay.history->chunks[0].input==checkpoint.input,
      "completed key must own immutable tokens after the caller releases or changes its borrowed chunks");
  const auto rebuilt=p::copy_history(checkpoint.replay,12);
  require(rebuilt->chunks.size()==4 && rebuilt->chunks[0].count==3 && rebuilt->chunks[1].count==1 &&
      rebuilt->chunks[2].count==2 && rebuilt->chunks[3].count==2 &&
      rebuilt->chunks[2].first==4 && rebuilt->chunks[3].next_id==248044,
      "rebuild must retain actual accept1/accept2 and later-turn consumed chunk boundaries");
  p::PrefixCache cache(1);
  require(cache.publish(checkpoint.input,head,checkpoint.replay),"completed generated checkpoint publication");
  auto full=cache.lookup(*tokens({1,2,3,4,5,6,7,8}),8,0,&model,&math);
  require(full.payload.snapshot==head && full.payload.snapshot->sampling==p::SamplingCursor{17,29,13} &&
      full.payload.snapshot->draft->frontier==7 && full.payload.snapshot->draft->lag==1 &&
      full.payload.snapshot->draft->catchup_token==8 && full.payload.snapshot->next_id==248044,
      "complete hit must retain original request metadata and lag1 Snapshot without fabricating fresh state");
  auto partial=cache.lookup(*tokens({1,2,3,4,5,6,7,8,248044,10}),10,0,&model,&math);
  require(partial.payload.snapshot==head && partial.payload.snapshot->length==8 &&
      !cache.lookup(*tokens({1,2,3,4,5,6,7,9}),8,0,&model,&math),
      "later requests may reuse the generated prefix but cannot claim the unconsumed pending token was processed");
  require(!cache.evict_one() && versions.live==1,
      "multiple complete/partial hits must share the finite version and pin its single directory slot");
  auto escaped=partial.payload.snapshot;full.reset();partial.reset();head.reset();
  require(cache.evict_one() && escaped->length==8 && versions.live==1,
      "eviction drops only directory ownership, not an escaped complete checkpoint");
  escaped.reset();require(versions.live==0,"last checkpoint owner must retire the canonical version once");
}

void completed_checkpoint_validation() {
  int model=0,math=0;
  const auto head=snapshot(3,&model,&math,4);
  const std::array<std::int32_t,3> actual{1,2,3};
  const std::array<p::ConsumedChunk,1> valid{{{actual,4}}};
  const auto rejects=[&](std::span<const p::ConsumedChunk> chunks,const p::SnapshotRef &state,std::size_t capacity) {
    bool rejected=false;
    try {(void)p::consumed_checkpoint(chunks,*state,capacity);}
    catch(const std::runtime_error &) {rejected=true;}
    require(rejected,"incomplete/unbounded consumed checkpoint must reject before publication");
  };
  const std::array<std::int32_t,4> includes_pending{1,2,3,4};
  const std::array<p::ConsumedChunk,1> extra{{{includes_pending,4}}};
  const std::array<p::ConsumedChunk,1> missing{{{std::span(actual).first(2),4}}};
  const std::array<p::ConsumedChunk,1> wrong_pending{{{actual,5}}};
  const std::array<std::int32_t,3> negative{1,-1,3};
  const std::array<p::ConsumedChunk,1> invalid_ids{{{negative,4}}};
  const std::array<p::ConsumedChunk,2> empty_chunk{{{{},1},{actual,4}}};
  rejects(extra,head,8);rejects(missing,head,8);rejects(wrong_pending,head,8);
  rejects(invalid_ids,head,8);rejects(empty_chunk,head,8);rejects({},head,8);
  rejects(valid,head,2);rejects(valid,snapshot(3,&model,&math,-1),8);
  // Repeating an ID is legal; exclusion is by the actual committed length,
  // never by dropping all IDs equal to the pending token.
  const std::array<std::int32_t,2> repeated{42,42};
  const std::array<p::ConsumedChunk,1> repeat{{{repeated,42}}};
  const auto allowed=p::consumed_checkpoint(repeat,*snapshot(2,&model,&math,42),2);
  require(allowed.input->ids==std::vector<std::int32_t>{42,42},"repeated consumed/pending token IDs remain legal");
}

void aligned_state_pool_capacity() {
  namespace c = hyle::compute;
  // A 128-byte BF16 hidden row exposes the missing gap: the second row starts
  // at 256, so a payload-only 257-byte backing cannot hold both live rows.
  auto old = c::LeasePool::create(hyle::mem::system_resource(), 257,
      {.max_blocks=2, .backing_alignment=p::state_pool_alignment}).value();
  auto first = old.allocate(128, p::state_pool_alignment).value();
  auto old_read = first.publish().value();
  const auto rejected = old.allocate(128, p::state_pool_alignment);
  require(!rejected && rejected.error().IsResourceExhausted(),
      "payload-only capacity must reproduce the second aligned allocation failure");
  const auto fills = [](std::size_t payload, std::uint32_t slots) {
    const auto capacity = p::state_pool_backing_bytes(payload, slots);
    auto pool = c::LeasePool::create(hyle::mem::system_resource(), capacity,
        {.max_blocks=slots, .backing_alignment=p::state_pool_alignment}).value();
    std::vector<c::ReadLease> reads; reads.reserve(slots);
    for (int reuse = 0; reuse != 2; ++reuse) {
      for (std::uint32_t slot = 0; slot != slots; ++slot) {
        auto write = pool.allocate(payload, p::state_pool_alignment).value();
        require(write.view()->fill(std::byte(slot + 1)).ok(), "state row fill failed");
        reads.push_back(write.publish().value());
      }
      require(pool.stats().live_blocks == slots && pool.stats().capacity_bytes == capacity,
          "aligned backing must hold the configured number of simultaneous state rows");
      std::vector<std::byte> bytes(payload);
      for (std::uint32_t slot = 0; slot != slots; ++slot) {
        require(reads[slot].view()->read(bytes).ok(), "state row read failed");
        for (const auto value : bytes)
          require(value == std::byte(slot + 1), "aligned live state rows must remain independent");
      }
      reads.clear();
    }
  };
  fills(128, 2);   // Small hidden rows.
  fills(1152, 6);  // Small GDN convolution history, independent pool.
  fills(4096, 3);  // Already aligned official hidden rows need no extra gaps.
  const auto rejects = [](std::size_t payload, std::size_t slots) {
    bool rejected_capacity = false;
    try { (void)p::state_pool_backing_bytes(payload, slots); }
    catch (const std::runtime_error &) { rejected_capacity = true; }
    require(rejected_capacity, "invalid or overflowing pool capacity must reject");
  };
  rejects(0, 1); rejects(128, 0);
  rejects((std::numeric_limits<std::size_t>::max)(), 1);
  rejects(128, (std::numeric_limits<std::size_t>::max)());
}

p::CacheRowBytes bf16_cache_rows(const qwen35::ModelGeometry &g) {
  // These host capacity fixtures explicitly describe both available banks.
  const p::KvRowBytes rows{p::multiply(p::multiply(static_cast<std::size_t>(g.kv_heads), static_cast<std::size_t>(g.key_depth)), 2),
      p::multiply(p::multiply(static_cast<std::size_t>(g.kv_heads), static_cast<std::size_t>(g.value_depth)), 2)};
  return {.target = rows, .draft = rows};
}

void batch_state_capacity() {
  constexpr std::size_t payload = 19537920; // Current 36 Qwen conv/GDN resources.
  const auto single = p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, payload);
  const auto explicit_single = p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, payload, 1, 1, false, false, 1);
  require(single.in_flight_batches == 1 && explicit_single.in_flight_batches == 1 &&
      single.active_sequences == 1 && single.max_requests == 1 &&
      single.max_query_rows == 16 && single.max_kv_rows == 256 &&
      single.max_pages == 65 && single.max_page_refs == 16 && single.state_slots == 5 &&
      single.key_bank_bytes == 1064961 && single.backing_bytes == 110469168 &&
      explicit_single.backing_bytes == single.backing_bytes,
      "R1 must preserve the original real Qwen state/page budget");
  const auto batch = p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, payload, 2);
  require(batch.active_sequences == 2 && batch.max_requests == 2 &&
      batch.max_query_rows == 32 && batch.max_kv_rows == 512 && batch.max_page_refs == 32 &&
      batch.state_slots == 6 && batch.max_pages == 66 && batch.key_bank_bytes == 1081345 &&
      batch.backing_bytes == 130203696 &&
      batch.backing_bytes - single.backing_bytes == payload + 12 * 16 * 1024,
      "R2 must reserve two candidates, packed request extents and both possible COW tails");
  const auto serial_frontier = p::plan(geometry, bf16_cache_rows(geometry), 17, 1, 1, payload, 1, 1, false, true, 1);
  const auto overlapped = p::plan(geometry, bf16_cache_rows(geometry), 17, 1, 1, payload, 1, 1, false, true, 2);
  require(serial_frontier.state_slots == 2 && serial_frontier.snapshot_slots == 2 && serial_frontier.max_pages == 3 &&
      overlapped.in_flight_batches == 2 && overlapped.state_slots == 3 && overlapped.snapshot_slots == 3 &&
      overlapped.max_pages == 4 && overlapped.hidden_slots == 3 && overlapped.candidate_boundaries == 1 &&
      overlapped.max_requests == 1 && overlapped.max_query_rows == 1 && overlapped.max_kv_rows == 17 &&
      overlapped.max_page_refs == 2 && overlapped.active_sequences == 1 && overlapped.resident_sessions == 1 &&
      overlapped.backing_bytes - serial_frontier.backing_bytes == payload + 12 * 16 * 1024 + 4096,
      "lag1 needs committed/A/B state versions, two COW tails and one extra hidden row without widening the batch");
  const auto bf16_rows = bf16_cache_rows(geometry);
  const p::KvRowBytes e4_rows{bf16_rows.target.key / 2, bf16_rows.target.value / 2};
  const auto e4 = p::plan(geometry, {.target = e4_rows}, 256, 4, 16, payload, 2);
  const auto mixed_widths = p::plan(geometry, {.target = {e4_rows.key, bf16_rows.target.value}}, 256, 4, 16, payload, 2);
  require(batch.row_bytes == bf16_rows && e4.row_bytes.target == e4_rows &&
      !batch.draft_key_bank_bytes && !batch.draft_value_bank_bytes &&
      !e4.draft_key_bank_bytes && !e4.draft_value_bank_bytes &&
      e4.key_bank_bytes == (batch.key_bank_bytes - 1) / 2 + 1 &&
      e4.value_bank_bytes == (batch.value_bank_bytes - 1) / 2 + 1 &&
      batch.backing_bytes - e4.backing_bytes ==
          ((batch.key_bank_bytes - e4.key_bank_bytes) + (batch.value_bank_bytes - e4.value_bank_bytes)) * batch.attention_layers &&
      mixed_widths.key_bank_bytes == e4.key_bank_bytes && mixed_widths.value_bank_bytes == batch.value_bank_bytes &&
      e4.recurrent_payload_bytes == batch.recurrent_payload_bytes && e4.state_slots == batch.state_slots &&
      e4.hidden_row_bytes == batch.hidden_row_bytes && e4.hidden_bytes == batch.hidden_bytes,
      "actual K/V row extents must change only physical KV backing, independently for each bank");
  const auto three=p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,3,true);
  const auto five=p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,5,true);
  require(three.candidate_boundaries==3 && three.state_slots==10 && three.hidden_slots==14 &&
      five.candidate_boundaries==5 && five.state_slots==14 && five.hidden_slots==18 &&
      three.snapshot_slots==6 && five.snapshot_slots==6 &&
      three.max_pages==batch.max_pages && five.max_pages==batch.max_pages,
      "K2/K4 candidate material rows must grow without adding committed versions or whole-context KV copies");
  const auto deep = p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, payload, 2, 3, true, false, 3);
  require(deep.in_flight_batches == 3 && deep.candidate_boundaries == 3 && deep.state_slots == 22 &&
      deep.snapshot_slots == 10 && deep.max_pages == 70 && deep.hidden_slots == 26 &&
      deep.max_requests == three.max_requests && deep.max_query_rows == three.max_query_rows &&
      deep.max_kv_rows == three.max_kv_rows && deep.max_page_refs == three.max_page_refs &&
      deep.resident_sessions == three.resident_sessions && deep.active_sequences == three.active_sequences,
      "three retained batches must scale candidate rows/versions/tails, independently of per-batch R and K");
  const auto two=p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,2,true);
  require(two.state_slots==8 && two.hidden_slots==12 &&
      two.hidden_bytes==49153 && two.backing_bytes==171491379,
      "existing two-boundary MTP state/hidden/backing numbers must remain unchanged");
  const auto e4_mtp = p::plan(geometry, {.target = e4_rows, .draft = bf16_rows.draft}, 256, 4, 16, payload, 2, 2, true);
  require(e4_mtp.key_bank_bytes == 540673 && e4_mtp.value_bank_bytes == 540673 &&
      e4_mtp.draft_key_bank_bytes == 1081345 && e4_mtp.draft_value_bank_bytes == 1081345 &&
      // Six target attention layers save 6*2*540672 bytes; NextN does not.
      e4_mtp.backing_bytes == 165003315 && e4_mtp.hidden_bytes == two.hidden_bytes &&
      e4_mtp.state_slots == two.state_slots && e4_mtp.max_pages == two.max_pages,
      "E4 target banks must not halve the independent BF16 NextN bank, states or hidden storage");
  const auto no_draft=p::plan(geometry, bf16_cache_rows(geometry), 1,1,1,0,1,1,true);
  require(no_draft.candidate_boundaries==1 && no_draft.state_slots==2 &&
      no_draft.hidden_slots==3 && no_draft.hidden_bytes==12289 && no_draft.backing_bytes==471091,
      "target-only MTP/context1 still needs finite committed/candidate carry storage");
  const auto larger = qwen_test::q27();
  // Independent original Q27 geometry: 48 states of 3*10240 BF16 history
  // and 48*128*128 F32 recurrent entries; 16 target KV layers plus NextN.
  const auto larger_plan = p::plan(larger, bf16_cache_rows(larger), 64, 1, 15, 153944064, 1, 2, true);
  require(larger_plan.attention_layers == 16 && larger_plan.recurrent_layers == 48 &&
      larger_plan.hidden_row_bytes == 10240 && larger_plan.key_bank_bytes == 163841 &&
      larger_plan.value_bank_bytes == 163841 && larger_plan.hidden_bytes == 40961 &&
      larger_plan.backing_bytes == 467443843,
      "Q27 independent cache counts and widths must replace every Q35 fixed bank/hidden assumption");
  const auto rejects = [](auto make_plan) {
    bool rejected = false;
    try { (void)make_plan(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid or overflowing batch state capacity must reject");
  };
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, payload, 0); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 1, 16, payload, 2); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 0, 4, 1, payload, 1); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 0, 16, payload, 1); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 0, payload, 1); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 257, payload, 1); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,0,true); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,257,true); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,1,false,false,0); });
  rejects([&] { return p::plan(geometry, {.target = {0, 1024}}, 256, 4, 16, payload, 1); });
  rejects([&] { return p::plan(geometry, {.target = {1024, 0}}, 256, 4, 16, payload, 1); });
  rejects([&] { return p::plan(geometry, {.target = e4_rows}, 256, 4, 16, payload, 1, 2, true); });
  rejects([&] { return p::plan(geometry, {.target = e4_rows, .draft = {1024, 0}}, 256, 4, 16, payload, 1, 2, true); });
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  rejects([&] { return p::plan(geometry, {.target = {maximum, 1024}}, 1, 1, 1, 0); });
  rejects([&] { return p::plan(geometry, {.target = {1024, maximum}}, 1, 1, 1, 0); });
  rejects([&] { return p::plan(geometry, {.target = e4_rows, .draft = {maximum, 1024}}, 1, 1, 1, 0, 1, 1, true); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), maximum, 1, 1, payload, 1); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,maximum,true); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256,4,16,payload,2,1,false,false,maximum); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 256, 4, 16, maximum, 2); });
  const auto max_index = std::size_t((std::numeric_limits<std::int32_t>::max)());
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 1,1,1,0,1,1,false,false,max_index); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), max_index, 2, 1, 0, 2); });
  rejects([&] { return p::plan(geometry, bf16_cache_rows(geometry), 1, max_index, 1, 0, 1); });
}

void exact_boundaries_and_domains() {
  int model = 0, other_model = 0, math = 0, other_math = 0;
  p::PrefixCache cache(8);
  std::array<std::int32_t, 2> first{1, 2};
  std::array<std::int32_t, 4> rest{3, 4, 5, 6};
  const std::array input_chunks{std::span<const std::int32_t>(first), std::span<const std::int32_t>(rest)};
  const auto a = p::token_plan(input_chunks, 6), b = tokens({1, 2, 3, 4, 8, 9});
  rest[0] = 99; // Caller storage can change after the stable key was prepared.
  const auto flat = tokens({1, 2, 3, 4, 5, 6});
  require(a->ids == flat->ids && a->hashes == flat->hashes &&
      a->checkpoints == std::vector<std::size_t>{2, 6},
      "chunked inputs must retain their own IDs and exact boundaries independently of parser types");
  const auto c = tokens({1, 2, 3, 4, 7, 10});
  const auto a6 = snapshot(6, &model, &math), b6 = snapshot(6, &model, &math);
  require(publish(cache, a, a6) && publish(cache, b, b6), "divergent compressed edges publish");
  require(!cache.lookup(*c, 6, 0, &model, &math), "structural split at four must not fabricate recurrent state");
  const auto a2 = snapshot(2, &model, &math);
  require(publish(cache, a, a2), "publish actual ancestor checkpoint");
  auto hit = cache.lookup(*c, 6, 0, &model, &math);
  require(hit.payload.snapshot == a2, "fallback must choose actual ancestor instead of longest structural prefix"); hit.reset();
  const auto a4 = snapshot(4, &model, &math, 93);
  auto chunks = std::make_shared<p::ReplayHistory>(6);
  chunks->append(a, 0, 2, 31); chunks->append(a, 2, 2, a4->next_id);
  require(cache.publish(a, a4, {chunks, 2, 4}), "real state may attach at an existing split");
  hit = cache.lookup(*c, 5, 3, &model, &math);
  require(hit.payload.snapshot == a4 && hit.payload.replay.chunks == 2, "longest actual boundary must preserve source chunks");
  const auto replay = p::copy_history(hit.payload.replay, 6);
  require(replay->length == 4 && replay->chunks[0].count == 2 && replay->chunks[1].next_id == 93,
      "rebuild must preserve actual chunk boundaries and next_id"); hit.reset();
  require(!cache.lookup(*a, 3, 3, &model, &math), "target inside compressed edge cannot synthesize a checkpoint");
  require(cache.lookup(*a, 3, 0, &model, &math).payload.snapshot == a2, "target may return a shorter real ancestor");
  const auto other = snapshot(4, &other_model, &math), other_policy = snapshot(4, &model, &other_math);
  require(publish(cache, a, other) && publish(cache, a, other_policy), "separate model/math namespaces publish");
  require(cache.lookup(*a, 4, 0, &other_model, &math).payload.snapshot == other, "model identity must isolate roots");
  require(cache.lookup(*a, 4, 0, &model, &other_math).payload.snapshot == other_policy, "math identity must isolate roots");
  auto collision = std::make_shared<p::TokenPlan>(*a);
  collision->ids[0] = 99; // Deliberately retain all original hashes.
  require(!cache.lookup(*collision, 6, 0, &model, &math), "equal hashes never replace exact tokens");
  auto invalid = history(a, a6); invalid.length = 5;
  const auto before = cache.size(); bool rejected = false;
  try { (void)cache.publish(a, a6, invalid); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && cache.size() == before && cache.lookup(*a, 6, 0, &model, &math).payload.snapshot == a6,
      "invalid replay must not mutate the old index");
}

void shared_pins_and_lru() {
  int model = 0, math = 0;
  const auto a = tokens({10, 20, 30}), b = tokens({11, 20}), c = tokens({12, 20}), d = tokens({13, 20});
  const auto sa = snapshot(3, &model, &math), sb = snapshot(2, &model, &math);
  const auto sc = snapshot(2, &model, &math), sd = snapshot(2, &model, &math);
  p::PrefixCache cache(2);
  require(publish(cache, a, sa) && publish(cache, b, sb), "fill bounded cache");
  auto first = cache.lookup(*a, 3, 0, &model, &math);
  auto second = cache.lookup(*a, 3, 0, &model, &math);
  require(first.payload.snapshot == sa && second.payload.snapshot == sa, "multiple read-only hits may pin one payload");
  first.reset();
  require(cache.evict_one() && cache.retains(sa.get()) && !cache.retains(sb.get()),
      "releasing one hit must not unpin another; LRU skips pinned payload");
  require(publish(cache, c, sc), "reuse free payload slot");
  auto pinned_c = cache.lookup(*c, 2, 0, &model, &math);
  require(!publish(cache, d, sd) && !publish(cache, a, snapshot(3, &model, &math)),
      "all pinned capacity and pinned replacement reject without mutation");
  require(cache.size() == 2 && second.payload.snapshot == sa && pinned_c.payload.snapshot == sc, "failed admission preserves hits");
  pinned_c.reset();
  require(publish(cache, d, sd) && cache.evictions() == 2 && !cache.retains(sc.get()), "only unpinned victim may be replaced");
  auto escaped = second.payload.snapshot; second.reset();
  require(cache.evict_one() && !cache.retains(sa.get()) && escaped->length == 3,
      "directory eviction does not destroy an escaped snapshot owner");
  auto moved = cache.lookup(*d, 2, 0, &model, &math);
  auto another = std::move(moved);
  require(!moved && another.payload.snapshot == sd && !cache.evict_one(), "moved hit retains exactly one pin");
  another.reset(); require(cache.evict_one() && cache.size() == 0, "last reset permits final eviction");
}

// An independent small linear oracle checks changing compressed shapes and
// eviction of payloads with children. It intentionally has no radix topology.
void bounded_tree_reuse() {
  struct Saved { p::TokenPlanRef input; p::SnapshotRef head; std::size_t age; };
  int models[3]{}, math = 0;
  for (const std::size_t limit : {std::size_t{1}, std::size_t{5}}) {
    p::PrefixCache cache(limit); std::vector<Saved> saved; std::size_t clock = 0, evictions = 0;
    for (std::size_t step = 0; step < 400; ++step) {
      const auto model = &models[(step / 11) % 3];
      const auto input = tokens({1, 2, static_cast<std::int32_t>((step / 3) % 4), 4,
          static_cast<std::int32_t>((step / 5) % 5), 6});
      const auto length = std::size_t{1} + (step * 7) % 6;
      const auto head = snapshot(length, model, &math, static_cast<std::int32_t>(step));
      auto found = std::find_if(saved.begin(), saved.end(), [&](const Saved &entry) {
        return entry.head->model == model && entry.head->length == length &&
            std::equal(input->ids.begin(), input->ids.begin() + static_cast<std::ptrdiff_t>(length), entry.input->ids.begin());
      });
      if (found != saved.end()) *found = {input, head, ++clock};
      else {
        if (saved.size() == limit) {
          saved.erase(std::min_element(saved.begin(), saved.end(), [](const Saved &a, const Saved &b) { return a.age < b.age; }));
          ++evictions;
        }
        saved.push_back({input, head, ++clock});
      }
      require(publish(cache, input, head), "bounded insert/prune must always find its cold slots");
      for (std::size_t target = 1; target <= 6; ++target) {
        Saved *best = nullptr;
        for (auto &entry : saved) if (entry.head->model == model && entry.head->length <= target &&
            (!best || best->head->length < entry.head->length) &&
            std::equal(entry.input->ids.begin(), entry.input->ids.begin() + static_cast<std::ptrdiff_t>(entry.head->length), input->ids.begin()))
          best = &entry;
        auto hit = cache.lookup(*input, target, 0, model, &math);
        require(hit.payload.snapshot == (best ? best->head : p::SnapshotRef{}), "radix result differs from exact linear boundary oracle");
        if (best) best->age = ++clock;
      }
      require(cache.size() == saved.size() && cache.evictions() == evictions, "payload bound/LRU diverged under split/prune churn");
    }
  }
}

void typed_directory_presence_is_independent_of_payload_value() {
  struct Payload final {
    std::size_t marker = 0;
    Payload() noexcept = default;
    explicit Payload(std::size_t value) noexcept : marker(value) {}
    Payload(const Payload &) noexcept = default;
    Payload &operator=(const Payload &) noexcept = default;
    Payload(Payload &&other) noexcept : marker(std::exchange(other.marker, 0)) {}
    Payload &operator=(Payload &&other) noexcept { marker = std::exchange(other.marker, 0); return *this; }
  };
  hyle::inference::PrefixDirectory<Payload> directory(1);
  const auto input = tokens({3, 4}); int model = 0, math = 0;
  const hyle::inference::CacheDomain domain{&model, &math};
  require(directory.publish(domain, input, 2, Payload{}), "typed directory stores a caller-validated default-valued payload");
  auto hit = directory.lookup(domain, *input, 2, 1);
  require(hit && hit.payload.marker == 0 && hit.complete_length() == 2 &&
      directory.retains([](const Payload &value) { return value.marker == 0; }),
      "occupied node, not payload truthiness, determines a typed directory hit");
  Payload pending{7};
  require(!directory.evict_one() && !directory.publish(domain, input, 2, std::move(pending)) && pending.marker == 7,
      "typed payload replacement must preserve both the pinned entry and rejected rvalue");
  auto moved = std::move(hit);
  require(!hit && hit.complete_length() == 0 && moved && moved.complete_length() == 2 && moved.payload.marker == 0,
      "moving a typed hit must transfer exactly its original pin and payload");
  moved.reset();
  require(directory.evict_one() && directory.size() == 0 && directory.evictions == 1 &&
      !directory.retains([](const Payload &) { return true; }),
      "free default-valued payload slots must not retain or manufacture entries");
  require(directory.publish(domain, input, 2, std::move(pending)) && pending.marker == 0 &&
      directory.lookup(domain, *input, 2, 1).payload.marker == 7,
      "successful typed payload publication must move the original source only at commit");
}

void allocation_free_commit() {
  int model = 0, math = 0, other_model = 0;
  p::PrefixCache cache(2);
  const auto a = tokens({1, 2, 3, 4}), b = tokens({1, 2, 9, 8}), c = tokens({1, 2});
  const auto sa = snapshot(4, &model, &math), sb = snapshot(4, &model, &math);
  const auto sc = snapshot(2, &model, &math), other = snapshot(4, &other_model, &math);
  const auto ra = history(a, sa), rb = history(b, sb), rc = history(c, sc), ro = history(a, other);
  require(cache.publish(a, sa, ra), "initial boundary");
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = true;
#endif
  bool ok = false;
  try {
    // Split, full-capacity insertion at an internal node, domain insertion and
    // pruning are all exercised with every ordinary heap allocation rejected.
    ok = cache.publish(b, sb, rb) && cache.publish(c, sc, rc) && cache.publish(a, other, ro);
    auto hit = cache.lookup(*a, 4, 0, &other_model, &math);
    ok = ok && hit.payload.snapshot == other && cache.size() == 2;
    hit.reset(); ok = ok && cache.evict_one() && cache.evict_one() && !cache.size();
  } catch (...) {
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
    reject_allocations = false;
#endif
    throw;
  }
#if defined(HYLE_PREFIX_CACHE_STANDALONE)
  reject_allocations = false;
#endif
  require(ok, "post-construction commits must not allocate or exhaust structural scratch slots");
}
} // namespace

#if defined(HYLE_PREFIX_CACHE_STANDALONE)
int main() {
  try {
    mtp_prefix_rejection_and_bonus(); std::puts("PASS MTP first/middle rejection and all-prefix bonus");
    mtp_prefix_eos_and_quota(); std::puts("PASS MTP accepted/bonus EOS and quota pending boundaries");
    mtp_prefix_invalid_inputs_and_rng(); std::puts("PASS MTP malformed selections and actual-width RNG blocks");
    mtp_request_stop_policy(); std::puts("PASS MTP per-request accepted/rejected/bonus stops and ignoreEOS");
    schedule_waves_keep_dependencies(); std::puts("PASS turn/cohort waves and independent Session call order");
    schedule_runtime_buckets_use_actual_rows(); std::puts("PASS actual suffix rows, full hits and kind-separated buckets");
    schedule_shape_domain_is_wave_local(); std::puts("PASS finite wave-local batch shapes and conservative replay counts");
    batch_state_capacity(); std::puts("PASS R1/R2 state/page capacity and rejected invalid/overflow limits");
    aligned_state_pool_capacity(); std::puts("PASS aligned hidden/history capacity, independent live rows and reuse");
    exact_boundaries_and_domains(); std::puts("PASS exact boundaries, replay chunks and model/math domains");
    shared_pins_and_lru(); std::puts("PASS shared read-only pins, bounded admission and LRU");
    sampling_cursor_addresses_are_transactional(); std::puts("PASS transactional sampling addresses and overflow");
    sampling_wrappers_share_one_state_version(); std::puts("PASS independent sampler metadata and canonical state-version retirement");
    working_state_reuses_metadata_and_freezes_explicitly(); std::puts("PASS working-state metadata reuse and explicit checkpoint allocation failure");
    working_state_adopts_sampling_origin_without_another_version(); std::puts("PASS working-state adoption, canonical sampling origin and shared version accounting");
    recycled_candidate_releases_permissions_and_owners(); std::puts("PASS recycled candidate releases actual lease, owner and versions");
    reusable_output_storage_is_bounded_and_preserves_rejected_readers(); std::puts("PASS bounded output reuse preserves pinned readers and rejected leases");
    reusable_output_storage_reuses_generation_and_address_without_payload_work(); std::puts("PASS original output generation/address reuse without payload work or hot allocation");
    working_state_recycles_only_successfully_superseded_unique_heads(); std::puts("PASS successful unique-old-head recycling preserves frozen states and releases versions/pages");
    completed_consumed_checkpoints(); std::puts("PASS complete generated/multiturn consumed keys, lag/RNG retention, pins and retirement");
    completed_checkpoint_validation(); std::puts("PASS rejected unconsumed pending, missing inputs and invalid completed checkpoints");
    bounded_tree_reuse(); std::puts("PASS split/prune/internal-boundary churn against linear oracle");
    typed_directory_presence_is_independent_of_payload_value(); std::puts("PASS typed payload node presence, move pins and reset");
    allocation_free_commit(); std::puts("PASS allocation-free commit and bounded structural scratch");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
#else
TEST(QwenMtpPrefix, RejectionAndBonus) { mtp_prefix_rejection_and_bonus(); }
TEST(QwenMtpPrefix, EosAndQuotaLeaveFinalOutputPending) { mtp_prefix_eos_and_quota(); }
TEST(QwenMtpPrefix, InvalidSelectionsAndActualWidthRng) { mtp_prefix_invalid_inputs_and_rng(); }
TEST(QwenMtpPrefix, RequestStopPolicies) { mtp_request_stop_policy(); }
TEST(QwenBatchSchedule, TurnCohortsPreserveSessionDependencies) { schedule_waves_keep_dependencies(); }
TEST(QwenBatchSchedule, RuntimeBucketsUseActualSuffixLengths) { schedule_runtime_buckets_use_actual_rows(); }
TEST(QwenBatchSchedule, ColdShapesUseOnlyReachableWaveMembers) { schedule_shape_domain_is_wave_local(); }
TEST(QwenSessionState, BatchCapacityPreservesSingleAndBoundsAllCandidates) { batch_state_capacity(); }
TEST(QwenSessionState, AlignedStatePoolsHoldEveryConfiguredLiveRow) { aligned_state_pool_capacity(); }
TEST(QwenPrefixCache, ExactSnapshotBoundariesAndDomains) { exact_boundaries_and_domains(); }
TEST(QwenPrefixCache, SharedPinsAndLru) { shared_pins_and_lru(); }
TEST(QwenSessionState, SamplingCursorAddressesAreTransactional) { sampling_cursor_addresses_are_transactional(); }
TEST(QwenPrefixCache, SamplingWrappersShareOneStateVersion) { sampling_wrappers_share_one_state_version(); }
TEST(QwenSessionState, WorkingMetadataReuseAndExplicitCheckpoint) { working_state_reuses_metadata_and_freezes_explicitly(); }
TEST(QwenSessionState, WorkingStateSharesCanonicalSamplingVersion) { working_state_adopts_sampling_origin_without_another_version(); }
TEST(QwenSessionState, RecycledCandidateReleasesPermissionsAndOwners) { recycled_candidate_releases_permissions_and_owners(); }
TEST(InferenceReusableOutputStorage, BoundedAndPreservesRejectedReaders) { reusable_output_storage_is_bounded_and_preserves_rejected_readers(); }
TEST(InferenceReusableOutputStorage, OriginalGenerationAndAddressWithoutPayloadWork) { reusable_output_storage_reuses_generation_and_address_without_payload_work(); }
TEST(QwenSessionState, RecyclesOnlySuccessfulUniqueOldHeads) { working_state_recycles_only_successfully_superseded_unique_heads(); }
TEST(QwenPrefixCache, CompletedConsumedCheckpoints) { completed_consumed_checkpoints(); }
TEST(QwenPrefixCache, CompletedCheckpointValidation) { completed_checkpoint_validation(); }
TEST(QwenPrefixCache, BoundedTreeReuse) { bounded_tree_reuse(); }
TEST(InferencePrefixDirectory, NodePresenceAndTypedPayloadPins) { typed_directory_presence_is_independent_of_payload_value(); }
TEST(QwenPrefixCache, AllocationFreeCommit) { allocation_free_commit(); }
#endif
