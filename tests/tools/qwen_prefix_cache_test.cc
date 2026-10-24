// Host capacity/directory checks only. Metadata-only Snapshots below do not assert any
// native state permission; actual KV/GDN reuse stays in the Qwen consumer tests.
#include "../../tools/qwen35_text/prefix_cache.h"
#include "../../tools/qwen35_text/batch_schedule.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

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
namespace b = qwen35::benchmark;
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }

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
  const qwen35::benchmark::Phase phase{qwen35::benchmark::PhaseKind::kPrompt, ids, 1};
  return p::token_plan(std::span(&phase, 1), ids.size());
}
p::SnapshotRef snapshot(std::size_t length, const void *model, const void *math, std::int32_t next = 71) {
  auto value = std::make_shared<p::Snapshot>();
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

void batch_state_capacity() {
  constexpr std::size_t payload = 19537920; // Current 36 Qwen conv/GDN resources.
  const auto single = p::plan(256, 4, 16, payload);
  const auto explicit_single = p::plan(256, 4, 16, payload, 1);
  require(single.active_sequences == 1 && single.max_requests == 1 &&
      single.max_query_rows == 16 && single.max_kv_rows == 256 &&
      single.max_pages == 65 && single.max_page_refs == 16 && single.state_slots == 5 &&
      single.bank_bytes == 1064961 && single.backing_bytes == 110469168 &&
      explicit_single.backing_bytes == single.backing_bytes,
      "R1 must preserve the original real Qwen state/page budget");
  const auto batch = p::plan(256, 4, 16, payload, 2);
  require(batch.active_sequences == 2 && batch.max_requests == 2 &&
      batch.max_query_rows == 32 && batch.max_kv_rows == 512 && batch.max_page_refs == 32 &&
      batch.state_slots == 6 && batch.max_pages == 66 && batch.bank_bytes == 1081345 &&
      batch.backing_bytes == 130203696 &&
      batch.backing_bytes - single.backing_bytes == payload + 12 * 16 * 1024,
      "R2 must reserve two candidates, packed request extents and both possible COW tails");
  const auto rejects = [](auto make_plan) {
    bool rejected = false;
    try { (void)make_plan(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid or overflowing batch state capacity must reject");
  };
  rejects([&] { return p::plan(256, 4, 16, payload, 0); });
  rejects([&] { return p::plan(256, 1, 16, payload, 2); });
  rejects([&] { return p::plan(0, 4, 1, payload, 1); });
  rejects([&] { return p::plan(256, 0, 16, payload, 1); });
  rejects([&] { return p::plan(256, 4, 0, payload, 1); });
  rejects([&] { return p::plan(256, 4, 257, payload, 1); });
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  rejects([&] { return p::plan(maximum, 1, 1, payload, 1); });
  rejects([&] { return p::plan(256, 4, 16, maximum, 2); });
  const auto max_index = std::size_t((std::numeric_limits<std::int32_t>::max)());
  rejects([&] { return p::plan(max_index, 2, 1, 0, 2); });
  rejects([&] { return p::plan(1, max_index, 1, 0, 1); });
}

void exact_boundaries_and_domains() {
  int model = 0, other_model = 0, math = 0, other_math = 0;
  p::PrefixCache cache(8);
  const auto a = tokens({1, 2, 3, 4, 5, 6}), b = tokens({1, 2, 3, 4, 8, 9});
  const auto c = tokens({1, 2, 3, 4, 7, 10});
  const auto a6 = snapshot(6, &model, &math), b6 = snapshot(6, &model, &math);
  require(publish(cache, a, a6) && publish(cache, b, b6), "divergent compressed edges publish");
  require(!cache.lookup(*c, 6, 0, &model, &math), "structural split at four must not fabricate recurrent state");
  const auto a2 = snapshot(2, &model, &math);
  require(publish(cache, a, a2), "publish actual ancestor checkpoint");
  auto hit = cache.lookup(*c, 6, 0, &model, &math);
  require(hit.snapshot == a2, "fallback must choose actual ancestor instead of longest structural prefix"); hit.reset();
  const auto a4 = snapshot(4, &model, &math, 93);
  auto chunks = std::make_shared<p::ReplayHistory>(6);
  chunks->append(a, 0, 2, 31); chunks->append(a, 2, 2, a4->next_id);
  require(cache.publish(a, a4, {chunks, 2, 4}), "real state may attach at an existing split");
  hit = cache.lookup(*c, 5, 3, &model, &math);
  require(hit.snapshot == a4 && hit.replay.chunks == 2, "longest actual boundary must preserve source chunks");
  const auto replay = p::copy_history(hit.replay, 6);
  require(replay->length == 4 && replay->chunks[0].count == 2 && replay->chunks[1].next_id == 93,
      "rebuild must preserve actual chunk boundaries and next_id"); hit.reset();
  require(!cache.lookup(*a, 3, 3, &model, &math), "target inside compressed edge cannot synthesize a checkpoint");
  require(cache.lookup(*a, 3, 0, &model, &math).snapshot == a2, "target may return a shorter real ancestor");
  const auto other = snapshot(4, &other_model, &math), other_policy = snapshot(4, &model, &other_math);
  require(publish(cache, a, other) && publish(cache, a, other_policy), "separate model/math namespaces publish");
  require(cache.lookup(*a, 4, 0, &other_model, &math).snapshot == other, "model identity must isolate roots");
  require(cache.lookup(*a, 4, 0, &model, &other_math).snapshot == other_policy, "math identity must isolate roots");
  auto collision = std::make_shared<p::TokenPlan>(*a);
  collision->ids[0] = 99; // Deliberately retain all original hashes.
  require(!cache.lookup(*collision, 6, 0, &model, &math), "equal hashes never replace exact tokens");
  auto invalid = history(a, a6); invalid.length = 5;
  const auto before = cache.size(); bool rejected = false;
  try { (void)cache.publish(a, a6, invalid); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && cache.size() == before && cache.lookup(*a, 6, 0, &model, &math).snapshot == a6,
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
  require(first.snapshot == sa && second.snapshot == sa, "multiple read-only hits may pin one payload");
  first.reset();
  require(cache.evict_one() && cache.retains(sa.get()) && !cache.retains(sb.get()),
      "releasing one hit must not unpin another; LRU skips pinned payload");
  require(publish(cache, c, sc), "reuse free payload slot");
  auto pinned_c = cache.lookup(*c, 2, 0, &model, &math);
  require(!publish(cache, d, sd) && !publish(cache, a, snapshot(3, &model, &math)),
      "all pinned capacity and pinned replacement reject without mutation");
  require(cache.size() == 2 && second.snapshot == sa && pinned_c.snapshot == sc, "failed admission preserves hits");
  pinned_c.reset();
  require(publish(cache, d, sd) && cache.evictions == 2 && !cache.retains(sc.get()), "only unpinned victim may be replaced");
  auto escaped = second.snapshot; second.reset();
  require(cache.evict_one() && !cache.retains(sa.get()) && escaped->length == 3,
      "directory eviction does not destroy an escaped snapshot owner");
  auto moved = cache.lookup(*d, 2, 0, &model, &math);
  auto another = std::move(moved);
  require(!moved && another.snapshot == sd && !cache.evict_one(), "moved hit retains exactly one pin");
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
        require(hit.snapshot == (best ? best->head : p::SnapshotRef{}), "radix result differs from exact linear boundary oracle");
        if (best) best->age = ++clock;
      }
      require(cache.size() == saved.size() && cache.evictions == evictions, "payload bound/LRU diverged under split/prune churn");
    }
  }
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
    ok = ok && hit.snapshot == other && cache.size() == 2;
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
    schedule_waves_keep_dependencies(); std::puts("PASS turn/cohort waves and independent Session call order");
    schedule_runtime_buckets_use_actual_rows(); std::puts("PASS actual suffix rows, full hits and kind-separated buckets");
    schedule_shape_domain_is_wave_local(); std::puts("PASS finite wave-local batch shapes and conservative replay counts");
    batch_state_capacity(); std::puts("PASS R1/R2 state/page capacity and rejected invalid/overflow limits");
    exact_boundaries_and_domains(); std::puts("PASS exact boundaries, replay chunks and model/math domains");
    shared_pins_and_lru(); std::puts("PASS shared read-only pins, bounded admission and LRU");
    bounded_tree_reuse(); std::puts("PASS split/prune/internal-boundary churn against linear oracle");
    allocation_free_commit(); std::puts("PASS allocation-free commit and bounded structural scratch");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
#else
TEST(QwenBatchSchedule, TurnCohortsPreserveSessionDependencies) { schedule_waves_keep_dependencies(); }
TEST(QwenBatchSchedule, RuntimeBucketsUseActualSuffixLengths) { schedule_runtime_buckets_use_actual_rows(); }
TEST(QwenBatchSchedule, ColdShapesUseOnlyReachableWaveMembers) { schedule_shape_domain_is_wave_local(); }
TEST(QwenSessionState, BatchCapacityPreservesSingleAndBoundsAllCandidates) { batch_state_capacity(); }
TEST(QwenPrefixCache, ExactSnapshotBoundariesAndDomains) { exact_boundaries_and_domains(); }
TEST(QwenPrefixCache, SharedPinsAndLru) { shared_pins_and_lru(); }
TEST(QwenPrefixCache, BoundedTreeReuse) { bounded_tree_reuse(); }
TEST(QwenPrefixCache, AllocationFreeCommit) { allocation_free_commit(); }
#endif
