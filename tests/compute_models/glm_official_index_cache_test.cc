#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/state_versions.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_GLM_CACHE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_GLM_CACHE_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "glm_official_index_cache.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace codec = glm_official_index_cache;
namespace sem = codec::sem;
using Bytes = std::vector<std::byte>;
using Leases = std::vector<c::ReadLease>;
template<class T> T take(hyle::Result<T> result, const char *where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, const char *where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> bytes) {
  auto backing = take(m::Buffer::create(owner.ref(), bytes.size(), 256), "explicit input backing");
  check(backing.view().write(bytes), "explicit initial/parameter input");
  return take(owner.adopt_buffer(std::move(backing)), "initial readonly input");
}
template<class T> std::vector<T> typed(const c::ReadLease &lease) {
  if (lease.size() % sizeof(T)) throw std::runtime_error("typed cache extent");
  std::vector<T> values(lease.size() / sizeof(T));
  check(take(lease.view(), "actual cache view").read(std::as_writable_bytes(std::span(values))), "read actual cache");
  return values;
}
Bytes read(const c::ReadLease &lease) { return typed<std::byte>(lease); }
struct Program final {
  sem::Fragment fragment;
  sem::GraphCase logical;
  Leases parameters;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool pool;
  static Program create(sem::Fragment fragment, c::Context &context,
                        const c::StorageOwner &owner, bool cuda, std::uint32_t generations) {
    Program p; p.fragment = std::move(fragment); p.logical = sem::make_graph(p.fragment);
    std::vector<g::ConstantBinding> constants;
    for (std::size_t i = 0; i < p.logical.constants.size(); ++i) {
      p.parameters.push_back(upload(owner, p.fragment.constant_bytes[i]));
      constants.push_back({p.logical.constants[i], take(p.parameters.back().tensor(
          p.fragment.description(p.fragment.constants[i])), "typed cache parameter")});
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*p.fragment.profile), "official GLM reader profile"),
        c::WorkloadPhase::kDecode, {0x48594c454d4f444cULL, p.fragment.region, 1}};
    const g::CompileOptions options{.constants = constants, .workload = workload};
    const auto compile = [&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_GLM_CACHE_CPU)
      if (!cuda) return g::cpu::compile(p.logical.graph, context, options);
#endif
#if defined(HYLE_TEST_GLM_CACHE_CUDA)
      if (cuda) return g::cuda::compile(p.logical.graph, context, owner, options);
#endif
      throw std::runtime_error("GLM official cache backend not enabled");
    };
    p.compiled = take(compile(), "ordinary official cache Graph");
    p.invocation = take(p.compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "cache invocation");
    std::size_t footprint = 0;
    for (std::size_t i = 0; i < p.logical.outputs.size(); ++i) {
      if (take(p.compiled.executable.output_ownership(i), "cache result ownership") != g::OutputOwnership::kCaller)
        throw std::runtime_error("cache journey needs caller-owned result");
      const auto extent = take(p.fragment.description(p.fragment.outputs[i]).required_bytes(), "cache output extent");
      footprint += (extent + 255u) / 256u * 256u;
    }
    p.pool = take(c::LeasePool::create(owner.ref(), footprint * generations,
        {.max_blocks = std::uint32_t(p.logical.outputs.size() * generations), .backing_alignment = 256}), "bounded cache output pool");
    return p;
  }
  Leases execute(c::Queue &queue, std::span<const c::ReadLease> input) {
    if (input.size() != logical.inputs.size()) throw std::runtime_error("cache Graph input count");
    for (std::size_t i = 0; i < input.size(); ++i)
      check(invocation.bind(logical.inputs[i], fragment.description(fragment.inputs[i]), input[i]), "bind actual cache input lease");
    for (std::size_t i = 0; i < logical.outputs.size(); ++i) {
      const auto &desc = fragment.description(fragment.outputs[i]);
      check(invocation.bind(logical.outputs[i], desc, take(pool.allocate(take(desc.required_bytes(), "cache result extent")),
          "cache result writer")), "bind cache result");
    }
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    check(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}), "prepare actual cache Graph");
    { auto completed = take(invocation.submit(queues), "cache submit"); check(completed.wait(), "cache producer complete"); }
    Leases result;
    for (auto port : logical.outputs)
      result.push_back(take(take(invocation.result(port), "actual cache result").storage_lease(), "retained actual result lease"));
    check(invocation.reset(), "retire invocation permissions");
    if (invocation.submit(queues)) throw std::runtime_error("reset cache invocation retained stale replay");
    return result;
  }
};
enum Entry : std::size_t { kCodes, kScales, kCursor, kIds, kAttention, kEntries };
struct State final {
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId, kEntries> entries;
  static State create(const c::StorageOwner &owner) {
    const std::array desc{sem::dense({codec::kCapacity, codec::kDepth}, c::DataType::kFloat8E4M3FN),
        sem::dense({codec::kCapacity, 1}), sem::dense({1}, c::DataType::kInt32),
        sem::dense({codec::kSelected}, c::DataType::kInt32), sem::dense({1, 1, codec::kMlaHeads, codec::kLatent})};
    std::array<c::StateEntrySpec, kEntries> specs;
    for (std::size_t i = 0; i < specs.size(); ++i) specs[i] = {.description = desc[i], .resource = owner.ref()};
    State s; s.schema = take(c::StateSchema::create(specs), "E4/scales/IDs/MLA schema");
    s.store = take(c::StateStore::create(s.schema, {.max_versions = 10, .max_drafts = 2,
        .max_retained_blocks = 64, .max_backing_bytes = 8u << 20}), "bounded official cache state");
    for (std::size_t i = 0; i < specs.size(); ++i) s.entries[i] = take(s.schema.entry(i), "cache state entry");
    return s;
  }
  c::ReadLease member(const c::StateVersion &v, Entry entry) const { return take(v.member(entries[entry]), "actual cache state member"); }
  c::StateVersion stage(const c::StateVersion &base, const Leases &producer, const Leases &consumer) {
    const std::array<c::ReadLease, kEntries> leases{producer[codec::Producer::codes], producer[codec::Producer::scales],
        producer[codec::Producer::cursor], consumer[codec::Consumer::ids], consumer[codec::Consumer::attention]};
    if (!base.valid()) {
      std::array<c::StateMember, kEntries> initial;
      for (std::size_t i = 0; i < initial.size(); ++i) initial[i] = {entries[i], leases[i]};
      return take(store.import(initial), "import actual base Graph outputs");
    }
    auto draft = take(store.begin(base), "one complete encoded cache/selection/MLA draft");
    for (std::size_t i = 0; i < leases.size(); ++i)
      check(draft.replace(entries[i], leases[i]), "stage actual producer/consumer lease together");
    return take(draft.finalize(), "finalize one coherent cache/IDs/MLA checkpoint");
  }
};
using Image = std::array<Bytes, kEntries>;
Image image(const State &state, const c::StateVersion &v) {
  Image result; for (std::size_t i = 0; i < result.size(); ++i) result[i] = read(state.member(v, Entry(i))); return result;
}
struct Raw final { std::vector<std::uint16_t> keys; std::vector<float> gates; };
Raw raw(std::int32_t begin, std::int32_t pools, bool zero = false) {
  Raw r; r.keys.resize(std::size_t(pools * codec::kPool * codec::kDepth)); r.gates.resize(r.keys.size());
  const auto bias = codec::ape();
  for (std::int32_t p = 0; p < pools; ++p)
    for (std::int64_t slot = 0; slot < codec::kPool; ++slot)
      for (std::int64_t d = 0; d < codec::kDepth; ++d) {
        const auto i = std::size_t((p * codec::kPool + slot) * codec::kDepth + d);
        const float amplitude = 1.0F + float(begin + p) / 16.0F;
        const float value = zero || begin + p == 0 ? 0.0F :
            amplitude * (1.0F + float(d * 13 % 17) / 8.0F) + float(slot * 2 - 3) / 32.0F;
        r.keys[i] = codec::bfloat(value);
        // Nonuniform keys, all four slots used. Dyadic gate+APE yields an
        // exact .25 softmax and isolates codec/layout from exp approximation.
        r.gates[i] = -bias[std::size_t(slot * codec::kDepth + d)] + float(d % 3) / 8.0F;
      }
  return r;
}
codec::Encoded exact_quant(std::span<const float> source, std::int64_t rows,
                 const Leases &output, std::size_t codes, std::size_t scales,
                 std::size_t rotated, std::size_t transformed, std::size_t transform_input) {
  const auto raw = typed<float>(output[transformed]);
  const auto bf16 = typed<float>(output[rotated]);
  const auto actual_input = typed<float>(output[transform_input]);
  if (source.size() != std::size_t(rows * codec::kDepth) || raw.size() != source.size() || bf16.size() != raw.size() || actual_input.size() != raw.size())
    throw std::runtime_error("independent Hadamard/quantizer boundary extent");
  // Public Hadamard promises F32 accumulation, not a fixed butterfly bit
  // order. Independently form the exact Sylvester signs in F64. gamma129
  // bounds a128-term F32 dot including coefficient rounding, and also the
  // shorter butterfly path. It is fixed before observing any native result.
  constexpr double unit_roundoff = 0x1p-24;
  constexpr double gamma129 = 129 * unit_roundoff / (1 - 129 * unit_roundoff);
  const double normalization = 1.0 / std::sqrt(double(codec::kDepth));
  std::size_t bad_math = 0, bad_bfloat = 0, bad_input = 0;
  double max_error = 0, max_ratio = 0;
  for (std::int64_t row = 0; row < rows; ++row) {
    std::array<float, codec::kDepth> input;
    double sumabs = 0;
    for (std::int64_t d = 0; d < codec::kDepth; ++d) {
      input[std::size_t(d)] = codec::wide(codec::bfloat(source[std::size_t(row * codec::kDepth + d)]));
      if (std::bit_cast<std::uint32_t>(actual_input[std::size_t(row * codec::kDepth + d)]) !=
          std::bit_cast<std::uint32_t>(input[std::size_t(d)])) ++bad_input;
      sumabs += std::abs(double(input[std::size_t(d)]));
    }
    const double bound = gamma129 * sumabs * normalization;
    for (std::int64_t d = 0; d < codec::kDepth; ++d) {
      double sum = 0;
      for (std::int64_t k = 0; k < codec::kDepth; ++k)
        sum += (std::popcount(unsigned(d & k)) & 1 ? -1.0 : 1.0) * input[std::size_t(k)];
      const auto i = std::size_t(row * codec::kDepth + d);
      const double difference = std::abs(double(raw[i]) - sum * normalization);
      if (!std::isfinite(raw[i]) || difference > bound) ++bad_math;
      max_error = std::max(max_error, difference);
      max_ratio = std::max(max_ratio, bound ? difference / bound : difference);
      if (std::bit_cast<std::uint32_t>(bf16[i]) !=
          std::bit_cast<std::uint32_t>(codec::wide(codec::bfloat(raw[i])))) ++bad_bfloat;
    }
  }
  EXPECT_EQ(bad_input, 0u) << "original typed query / independently pooled raw BF16 keys -> exact BF16 -> F32 Hadamard input";
  EXPECT_EQ(bad_math, 0u) << "Hadamard F64 mathematical reference: maxAbs=" << max_error << " max/gamma129Bound=" << max_ratio;
  EXPECT_EQ(bad_bfloat, 0u) << "exact BF16 RNE of actual independently checked F32 transform";
  const auto ref = codec::encode_rotated(bf16, rows);
  EXPECT_EQ(typed<std::uint8_t>(output[codes]), ref.codes);
  EXPECT_EQ(typed<float>(output[scales]), ref.scales);
  return ref;
}
void run(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  auto queue = take(context.create_queue(), "ordinary cache queue");
  auto baseline = Program::create(codec::producer(512).fragment, context, owner, cuda, 1);
  auto append = Program::create(codec::producer(1).fragment, context, owner, cuda, 5);
  auto consume_base = Program::create(codec::consumer(512).fragment, context, owner, cuda, 1);
  auto consume_a = Program::create(codec::consumer(513).fragment, context, owner, cuda, 2);
  auto consume_bc = Program::create(codec::consumer(514).fragment, context, owner, cuda, 3);
  std::vector<std::uint16_t> query(std::size_t(codec::kHeads * codec::kDepth));
  std::vector<float> query_f32(query.size()), weights(codec::kHeads);
  for (std::size_t h = 0; h < weights.size(); ++h) {
    weights[h] = h % 7 == 0 ? -1.0F / 64.0F : 1.0F / 8.0F + float(h % 3) / 32.0F;
    for (std::int64_t d = 0; d < codec::kDepth; ++d) {
      const auto i = h * codec::kDepth + std::size_t(d);
      query[i] = codec::bfloat((1.0F + float(d * 13 % 17) / 8.0F) * (1.0F + float(h % 5) / 8.0F));
      query_f32[i] = codec::wide(query[i]);
    }
  }
  std::vector<float> latent(std::size_t(codec::kCapacity * codec::kPool * codec::kLatent));
  std::vector<float> mla_query(std::size_t(codec::kMlaHeads * codec::kLatent));
  for (std::size_t i = 0; i < latent.size(); ++i)
    latent[i] = float(int((i / codec::kLatent) % 97) - 48) / 64.0F + float(int(i % codec::kLatent) % 11 - 5) / 128.0F;
  for (std::size_t i = 0; i < mla_query.size(); ++i) mla_query[i] = float(int(i * 7 % 17) - 8) / 256.0F;
  const std::array consumer_inputs{upload(owner, sem::bytes(query)), upload(owner, sem::bytes(weights)),
      upload(owner, sem::bytes(latent)), upload(owner, sem::bytes(mla_query))};
  std::size_t submissions = 0, peak_backing = 0;
  {
    auto state = State::create(owner);
    const auto compute = [&](const c::StateVersion &from, Raw raw_input, std::int64_t visible,
                             Program &producer_program, Program &consumer_program) {
      const auto prior_codes = from.valid() ? state.member(from, kCodes) : upload(owner,
          sem::bytes(std::vector<std::uint8_t>(std::size_t(codec::kCapacity * codec::kDepth), 0)));
      const auto prior_scales = from.valid() ? state.member(from, kScales) : upload(owner,
          sem::bytes(std::vector<float>(codec::kCapacity, std::numeric_limits<float>::quiet_NaN())));
      const auto cursor = from.valid() ? state.member(from, kCursor) : upload(owner, sem::bytes(std::vector<std::int32_t>{0}));
      const auto start = typed<std::int32_t>(cursor).at(0);
      const auto pools = visible - start;
      if (pools <= 0 || raw_input.keys.size() != std::size_t(pools * codec::kPool * codec::kDepth))
        throw std::runtime_error("cache journey pool/cursor mismatch");
      const std::array producer_inputs{prior_codes, prior_scales, upload(owner, sem::bytes(raw_input.keys)),
          upload(owner, sem::bytes(raw_input.gates)), cursor};
      auto encoded = producer_program.execute(queue, producer_inputs); ++submissions;
      const auto ref = codec::pool_reference(raw_input.keys, raw_input.gates, pools);
      const auto actual_pooled = typed<float>(encoded[codec::Producer::pooled]);
      sem::oracle::expect_close(actual_pooled, ref.pooled, 8e-5F, 4e-5F);
      (void)exact_quant(ref.pooled, pools, encoded, codec::Producer::new_codes,
          codec::Producer::new_scales, codec::Producer::rotated, codec::Producer::transformed, codec::Producer::transform_input);
      auto expected_codes = read(prior_codes), expected_scales = read(prior_scales);
      const auto code_update = read(encoded[codec::Producer::new_codes]), scale_update = read(encoded[codec::Producer::new_scales]);
      std::copy(code_update.begin(), code_update.end(), expected_codes.begin() + start * codec::kDepth);
      std::copy(scale_update.begin(), scale_update.end(), expected_scales.begin() + start * sizeof(float));
      EXPECT_EQ(read(encoded[codec::Producer::codes]), expected_codes);
      EXPECT_EQ(read(encoded[codec::Producer::scales]), expected_scales);
      EXPECT_EQ(typed<std::int32_t>(encoded[codec::Producer::cursor]), std::vector<std::int32_t>{std::int32_t(visible)});
      const std::array consume_inputs{encoded[codec::Producer::codes], encoded[codec::Producer::scales],
          consumer_inputs[0], consumer_inputs[1], consumer_inputs[2], consumer_inputs[3]};
      auto selected = consumer_program.execute(queue, consume_inputs); ++submissions;
      const auto query_ref = exact_quant(query_f32, codec::kHeads, selected, codec::Consumer::query_codes,
          codec::Consumer::query_scales, codec::Consumer::query_rotated, codec::Consumer::query_transformed, codec::Consumer::query_transform_input);
      const auto codes = typed<std::uint8_t>(encoded[codec::Producer::codes]);
      const auto scales = typed<float>(encoded[codec::Producer::scales]);
      const auto scores = codec::score_reference(codes, scales, query_ref, weights, visible);
      sem::oracle::expect_close(typed<float>(selected[codec::Consumer::scores]), scores, 8e-5F, 4e-5F);
      const auto pools_selected = typed<std::int32_t>(selected[codec::Consumer::pool_ids]);
      const auto ids = typed<std::int32_t>(selected[codec::Consumer::ids]);
      std::vector<std::int32_t> expected(std::size_t(visible), 0);
      std::iota(expected.begin(), expected.end(), 0);
      std::stable_sort(expected.begin(), expected.end(), [&](auto a, auto b) { return scores[std::size_t(a)] > scores[std::size_t(b)]; });
      expected.resize(codec::kTopPools); std::sort(expected.begin(), expected.end());
      auto actual_set = pools_selected; std::sort(actual_set.begin(), actual_set.end());
      EXPECT_EQ(actual_set, expected) << "independent score-derived TopK membership";
      if (pools_selected.size() != codec::kTopPools || ids.size() != codec::kSelected)
        throw std::runtime_error("real TopK ID extent");
      for (std::size_t i = 0; i < pools_selected.size(); ++i) {
        if (pools_selected[i] < 0 || pools_selected[i] >= visible) throw std::runtime_error("TopK selected invalid pool");
        for (std::size_t j = 0; j < codec::kPool; ++j) EXPECT_EQ(ids[i * codec::kPool + j], pools_selected[i] * codec::kPool + std::int32_t(j));
      }
      sem::oracle::expect_close(typed<float>(selected[codec::Consumer::attention]),
          codec::mla_reference(ids, latent, mla_query), 8e-5F, 4e-5F);
      auto next = state.stage(from, encoded, selected);
      EXPECT_FALSE(state.member(next, kCodes).try_write());
      EXPECT_FALSE(state.member(next, kIds).try_write());
      peak_backing = std::max(peak_backing, state.store.stats().backing_bytes);
      return next;
    };
    {
      const auto base = compute({}, raw(0, 512), 512, baseline, consume_base);
      const auto base_image = image(state, base);
      auto head = take(c::StateHead::create(state.store, base), "one encoded cache/IDs/MLA accepted head");
      const auto a = compute(base, raw(512, 1), 513, append, consume_a);
      const auto a_image = image(state, a);
      const auto rejected_b = compute(a, raw(513, 1), 514, append, consume_bc);
      const auto rejected_image = image(state, rejected_b);
      check(head.publish(base.identity(), a), "accept the complete first K4 cache checkpoint");
      EXPECT_FALSE(head.publish(base.identity(), rejected_b).ok());
      auto cancelled = take(state.store.begin(a), "deliberately incomplete cache splice");
      check(cancelled.replace(state.entries[kCodes], state.member(rejected_b, kCodes)), "stage rejected code candidate only");
      EXPECT_FALSE(cancelled.replace(state.entries[kScales], state.member(a, kCursor)).ok());
      check(cancelled.cancel(), "cancel incoherent code/scale candidate");
      EXPECT_EQ(image(state, take(head.current(), "head after cancelled splice")), a_image);
      // Continue from accepted A: C replaces the rejected pool at the SAME
      // logical cursor and is allzero, exercising floor and selected-ID change.
      const auto c_next = compute(take(head.current(), "accepted actual cache input"), raw(513, 1, true), 514, append, consume_bc);
      check(head.publish(a.identity(), c_next), "publish whole continued code/scale/IDs/MLA state");
      EXPECT_NE(read(state.member(c_next, kCodes)), read(state.member(rejected_b, kCodes)));
      auto rejected_set = typed<std::int32_t>(state.member(rejected_b, kIds));
      auto continued_set = typed<std::int32_t>(state.member(c_next, kIds));
      std::sort(rejected_set.begin(), rejected_set.end()); std::sort(continued_set.begin(), continued_set.end());
      EXPECT_NE(rejected_set, continued_set);
      EXPECT_NE(read(state.member(c_next, kAttention)), read(state.member(rejected_b, kAttention)));
      const auto independent_a = compute(base, raw(512, 1), 513, append, consume_a);
      const auto independent_c = compute(independent_a, raw(513, 1, true), 514, append, consume_bc);
      EXPECT_EQ(image(state, independent_c), image(state, c_next));
      EXPECT_EQ(image(state, base), base_image); EXPECT_EQ(image(state, a), a_image);
      EXPECT_EQ(image(state, rejected_b), rejected_image);
    }
    check(state.store.collect(), "retire encoded cache state versions");
    EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  }
  for (const auto *p : {&baseline, &append, &consume_base, &consume_a, &consume_bc}) {
    EXPECT_EQ(p->pool.stats().live_blocks, 0u); EXPECT_EQ(p->pool.stats().active_uses, 0u);
    for (std::size_t i = 0; i < p->parameters.size(); ++i) EXPECT_EQ(read(p->parameters[i]), p->fragment.constant_bytes[i]);
  }
  EXPECT_EQ(submissions, 12u);
  ::testing::Test::RecordProperty("actual_submissions", int(submissions));
  ::testing::Test::RecordProperty("persistent_index_bytes", std::to_string(codec::kCapacity * (codec::kDepth + 4)));
  ::testing::Test::RecordProperty("f32_index_comparison_bytes", std::to_string(codec::kCapacity * codec::kDepth * 4));
  ::testing::Test::RecordProperty("peak_state_backing_bytes", std::to_string(peak_backing));
  ::testing::Test::RecordProperty("reader", "fixed SGLang06992 default ue8m0; K4 pooledF32->BF16->normalizedHadamard128->BF16; explicit Float8BlockQuantize PowerOfTwo/floor1e-4; F32 reciprocal448 raw-scale then ceil-power-of-two, divide normalization; real E4 RNE codes plus F32 scales; matching rounded query and TileLang scorer logical math");
  ::testing::Test::RecordProperty("journey", "actual encoded lease->scorer->TopK512 I32->2048-token MLA; one StateHead code/scale/cursor/IDs/MLA; acceptA,rejectB,continueC,independent base recomputation,stale CAS,cancel,old readonly versions");
  ::testing::Test::RecordProperty("scope", "generated raw K4/query/headweights/latent boundary; complete closed-pool chunks; logical E4+F32scale storage,not vendor pagefooter ABI; partial raw tails covered separately; no original/full model,natural token acceptance,or throughput claim");
}
#if defined(HYLE_TEST_GLM_CACHE_CUDA)
TEST(GlmOfficialIndexCache, CudaEncodedLeaseSelectsAndPublishesMlaPrefix) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice), "CUDA cache memory")), "retained CUDA allocation domain");
  run(context, owner, true);
}
#endif
#if defined(HYLE_TEST_GLM_CACHE_CPU)
struct SystemResource final { m::MemoryResourceRef ref() noexcept { return m::system_resource(); } };
TEST(GlmOfficialIndexCache, CpuEncodedLeaseSelectsAndPublishesMlaPrefix) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run(context, owner, false);
}
#endif
} // namespace
