#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/weight_import.h"
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "d5_glm_remaining_fragments.h"
#include "original_role_input.h"
#include "glm_original_mtp_roles.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace glm = d5_glm_remaining_fragments;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace original = original_role_input;
using Bytes = std::vector<std::byte>;
namespace prefix=glm_original_mtp_roles;
using prefix::kHidden;using prefix::kJoined;using prefix::kQuery;
using prefix::kEpsilon;using prefix::kQaSha;using prefix::kScaleSha;
using prefix::kInputNormSha;using prefix::kQueryNormSha;
using prefix::bf16;using prefix::multiply;using prefix::norm_reference;
using prefix::eh_reference;using prefix::qa_reference;using prefix::close_bfloat;
using Roles=prefix::Data;
constexpr std::int64_t kIndexHeads=32,kIndexDepth=128,kIndexWidth=4096;

// Original GLM-5.3-Flash layer45, zai-org@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a.
// Seven original roles: enorm/hnorm/EH/input_layernorm/q_a E4/F32 scale/qnorm.
// Source order is NextN norm(E,H)->EH->decoder input RMS->q_a->q_a RMS.
// SGLang@06992c92ad4494fe9f9636e75be3e35040512447: deepseek_nextn.py,
// glm5_next.py and attention/mla/forward_mla.py. The q_a output is BF16;
// dynamic activation groups are128, Divide, floor1e-10, F32 accumulation.
// Divide is this consumer's existing explicit contract. Fixed SGLang has
// other CUDA/Triton quantization branches with different F32 reciprocal
// boundaries; this test makes no cross-kernel bitwise-equivalence claim.
// weight_scale_inv is the stored POSITIVE MULTIPLIER, never inverted by name.
// A second ordinary Graph consumes the ACTUAL qnorm ReadLease at the existing
// indexer wq_b boundary. Its BF16 sparse fixture weight (seed239) is generated.
// A grouped E1/slot1 Graph independently consumes the ACTUAL input-norm lease
// with the same original q_a bank, retaining the dense code-domain dot regression.
// This is a connected input/query component, not a complete indexer, decoder,
// cache/accepted-prefix state, target hidden provenance or natural acceptance.
template<class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
std::string directory(const char *name) {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  if (_dupenv_s(&value, &size, name)) throw std::runtime_error("cannot read original GLM role directory");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto value = std::getenv(name); return value ? value : "";
#endif
}
template<class T> Bytes bytes(std::span<const T> values) {
  const auto raw = std::as_bytes(values); return {raw.begin(), raw.end()};
}
c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions,
                    c::DataType type = c::DataType::kBFloat16) {
  return take(c::TensorDesc::dense(type, {dimensions.begin(), dimensions.size()}), "typed original description");
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> values) {
  auto buffer = take(m::Buffer::create(owner.ref(), values.size(), 256), "actual original backing");
  check(buffer.view().write(values), "actual immutable input upload");
  return take(owner.adopt_buffer(std::move(buffer)), "qualified original input lease");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(), "actual readonly result").read(result), "actual retained result bytes");
  return result;
}
std::vector<std::uint16_t> bfloat_values(const Bytes &values) {
  if (values.size() % 2) throw std::runtime_error("BF16 output extent");
  std::vector<std::uint16_t> result(values.size() / 2);
  std::memcpy(result.data(), values.data(), values.size()); return result;
}

sem::Fragment input_query(const Roles &roles) {
  sem::Fragment f{.profile = &glm::core::profiles::glm53_flash_official, .region = 0x474c4d51414348ULL};
  const auto row = dense({1, kHidden});
  const auto hidden = f.input(row), embedding = f.input(row);
  const auto p=prefix::append(f,hidden,embedding,roles);
  f.outputs={p.hn,p.en,p.joined,p.eh,p.nx,p.qa,p.qr};
  return f;
}
sem::Fragment indexer_query() {
  sem::Fragment f{.profile = &glm::core::profiles::glm53_flash_official, .region = 0x474c4d575142ULL};
  const auto qr = f.input(dense({1, kQuery}));
  // Reuse the existing Flash indexer wq_b recipe, dtype and seed at full shape.
  // Sparse generated parameters are explicit, not mislabeled original weights.
  const auto projected = glm::linear::projection(f, qr, kIndexWidth, kQuery,
      glm::linear::Storage::kBFloat16, 239);
  // The existing recipe exposes its F32 accumulator. Restore the declared
  // BF16 wq_b store before its [T,heads,depth] query reshape.
  const auto stored = f.one(c::Operation(c::Cast{.output_type = c::DataType::kBFloat16}), {projected.output});
  const auto query = glm::reshape(f, stored, {1, kIndexHeads, kIndexDepth});
  f.outputs = {f.one(c::Operation(c::Copy{}), {query})};
  return f;
}
sem::Fragment grouped_query(const Roles &roles) {
  sem::Fragment f{.profile = &glm::core::profiles::glm53_flash_official, .region = 0x474c4d51414752ULL};
  const auto input = f.input(dense({1, 1, kHidden}));
  const auto ids = f.input(dense({1, 1}, c::DataType::kInt32));
  const auto weight = f.constant(take(c::float8_grouped_weight_description(1, kQuery, kHidden),
      "original q_a grouped E1 packed type"), roles.canonical);
  f.outputs = {f.one(c::model_ops::grouped_float8_linear(
      c::model_ops::Float8ScalePolicy::kDivide, 1.0e-10F), {weight, input, ids})};
  return f;
}

// Small test-only holder for ordinary consumers. No new product executor,
// global cache, provider forcing or host reconstruction of an upstream result.
struct Program final {
  sem::Fragment fragment;
  sem::GraphCase logical;
  std::vector<c::ReadLease> parameters;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool outputs;
  static Program create(sem::Fragment fragment, c::Context &context,
                        const c::StorageOwner &owner, bool cuda) {
    Program result;
    result.fragment = std::move(fragment);
    result.logical = sem::make_graph(result.fragment);
    std::vector<g::ConstantBinding> constants;
    for (std::size_t i = 0; i < result.fragment.constants.size(); ++i) {
      result.parameters.push_back(upload(owner, result.fragment.constant_bytes[i]));
      constants.push_back({result.logical.constants[i], take(result.parameters.back().tensor(
          result.fragment.description(result.fragment.constants[i])), "typed original parameter")});
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(
        *result.fragment.profile), "fixed official GLM artifact profile"), c::WorkloadPhase::kDecode,
        {0x48594c454d4f444cULL, result.fragment.region, 1}};
    const g::CompileOptions options{.constants = constants, .workload = workload};
    const auto compile = [&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CPU)
      if (!cuda) return g::cpu::compile(result.logical.graph, context, options);
#endif
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CUDA)
      if (cuda) return g::cuda::compile(result.logical.graph, context, owner, options);
#endif
      throw std::runtime_error("original GLM chain backend not enabled");
    };
    result.compiled = take(compile(), "ordinary original GLM component compile");
    result.invocation = take(result.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{}}), "ordinary reusable Invocation");
    std::size_t footprint = 0;
    for (std::size_t i = 0; i < result.fragment.outputs.size(); ++i) {
      if (take(result.compiled.executable.output_ownership(i), "original component ownership") != g::OutputOwnership::kCaller)
        throw std::runtime_error("original chain requires existing caller-owned outputs");
      const auto extent = take(result.fragment.description(result.fragment.outputs[i]).required_bytes(), "output extent");
      footprint += (extent + 255u) / 256u * 256u;
    }
    result.outputs = take(c::LeasePool::create(owner.ref(), 2 * footprint,
        {.max_blocks = std::uint32_t(2 * result.fragment.outputs.size()), .backing_alignment = 256}), "bounded two output generations");
    return result;
  }
  std::vector<c::ReadLease> execute(c::Queue &queue, std::span<const c::ReadLease> inputs) {
    if (inputs.size() != fragment.inputs.size()) throw std::runtime_error("original component input count");
    for (std::size_t i = 0; i < inputs.size(); ++i)
      check(invocation.bind(logical.inputs[i], fragment.description(fragment.inputs[i]), inputs[i]), "bind actual original input lease");
    for (std::size_t i = 0; i < fragment.outputs.size(); ++i) {
      const auto &desc = fragment.description(fragment.outputs[i]);
      check(invocation.bind(logical.outputs[i], desc, take(outputs.allocate(
          take(desc.required_bytes(), "actual output bytes")), "new bounded output writer")), "bind caller output writer");
    }
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    check(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands,
        .host_wait = g::HostWaitPolicy::kAllow}), "prepare actual complete component");
    { auto done = take(invocation.submit(queues), "strict original component submit"); check(done.wait(), "actual component Completion"); }
    std::vector<c::ReadLease> result;
    for (const auto output : logical.outputs) {
      auto completed = take(invocation.result(output), "actual completed output");
      result.push_back(take(completed.storage_lease(), "independently retained actual permission"));
    }
    check(invocation.reset(), "withdraw completed invocation permissions");
    if (invocation.submit(queues)) throw std::runtime_error("reset original chain retained stale preparation");
    return result;
  }
};

std::vector<std::uint16_t> input(unsigned generation, bool embedding) {
  std::vector<std::uint16_t> result(kHidden);
  if (generation == 2) return result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const int value = int((i * (embedding ? 37u : 19u) + (embedding ? 11u : 3u) + generation * 23u) % 97u) - 48;
    result[i] = bf16(float(value) / (embedding ? 64.0F : 32.0F));
  }
  return result;
}
// Explicit, failure-only test diagnostics. Never change the assertion, input,
// provider choice or native work. The caller must choose a repository scratch
// directory; ordinary users and successful runs write no files.
void diagnose_query_failure(bool cuda, unsigned generation,
                            const Bytes &nx, const Bytes &qa,
                            std::span<const float> reference) {
  const auto selected = directory("HYLE_GLM_MTP_CHAIN_DIAGNOSTICS_DIR");
  if (selected.empty()) return;
  const auto source = std::filesystem::absolute(std::filesystem::path(__FILE__));
  const auto scratch = std::filesystem::weakly_canonical(
      source.parent_path().parent_path().parent_path() / ".scratch");
  auto target = std::filesystem::weakly_canonical(std::filesystem::absolute(selected));
  const auto require_scratch = [&]() {
    const auto relative = target.lexically_relative(scratch);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == "..")
      throw std::runtime_error("original GLM diagnostics must stay under this repository .scratch");
  };
  require_scratch();
  std::filesystem::create_directories(target);
  target = std::filesystem::canonical(target);
  require_scratch();
  const auto prefix = std::string(cuda ? "cuda" : "cpu") + "-generation-" + std::to_string(generation);
  const auto write = [&](const char *role, std::span<const std::byte> values) {
    const auto path = target / (prefix + role + ".bf16");
    if (std::filesystem::exists(std::filesystem::symlink_status(path)))
      throw std::runtime_error("original GLM diagnostic file already exists; choose a fresh scratch directory");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size()));
    if (!output) throw std::runtime_error("cannot write optional original GLM failure diagnostic");
  };
  std::vector<std::uint16_t> expected;
  expected.reserve(reference.size());
  for (auto value : reference) expected.push_back(bf16(value));
  write("-nx", nx); write("-qa", qa); write("-reference", std::as_bytes(std::span(expected)));
}

void run(const Roles &roles, c::Context &context, const c::StorageOwner &owner, bool cuda) {
  auto queue = take(context.create_queue(), "ordinary original-chain queue");
  auto first = Program::create(input_query(roles), context, owner, cuda);
  auto next = Program::create(indexer_query(), context, owner, cuda);
  auto grouped = Program::create(grouped_query(roles), context, owner, cuda);
  const std::array<std::int32_t, 1> expert{0};
  const auto grouped_ids = upload(owner, std::as_bytes(std::span(expert)));
  std::vector<c::ReadLease> oldest, old_query, old_grouped;
  std::vector<Bytes> oldest_bytes;
  Bytes old_query_bytes, old_grouped_bytes;
  bool distinct_order = false, changed_query = false;
  for (unsigned generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    const auto hidden = input(generation, false), embedding = input(generation, true);
    const std::array inputs{upload(owner, bytes<std::uint16_t>(hidden)), upload(owner, bytes<std::uint16_t>(embedding))};
    auto current = first.execute(queue, inputs);
    // No host reconstruction/copy: the exact result storage permission is
    // the next Graph's input, even after the producer Invocation reset.
    const std::array actual_query_input{current[6]};
    auto query = next.execute(queue, actual_query_input);
    // E1 grouped bytes match the original dense canonical bank. Only the
    // logical activation shape changes; the same actual nx storage is bound.
    const std::array actual_grouped_input{current[4], grouped_ids};
    auto grouped_result = grouped.execute(queue, actual_grouped_input);
    std::array<std::vector<std::uint16_t>, 7> actual;
    std::vector<Bytes> raw;
    for (std::size_t i = 0; i < current.size(); ++i) { raw.push_back(read(current[i])); actual[i] = bfloat_values(raw.back()); }
    const auto query_raw = read(query[0]);
    const auto query_values = bfloat_values(query_raw);
    const auto grouped_raw = read(grouped_result[0]);
    const auto grouped_values = bfloat_values(grouped_raw);
    ASSERT_TRUE(close_bfloat(actual[0], norm_reference(hidden, roles.hnorm), "hnorm"));
    ASSERT_TRUE(close_bfloat(actual[1], norm_reference(embedding, roles.enorm), "enorm"));
    auto joined_bytes = raw[1]; joined_bytes.insert(joined_bytes.end(), raw[0].begin(), raw[0].end());
    EXPECT_EQ(raw[2], joined_bytes) << "embedding-before-hidden original EH order";
    distinct_order |= raw[0] != raw[1];
    ASSERT_TRUE(close_bfloat(actual[3], eh_reference(actual[2], roles), "original_EH"));
    ASSERT_TRUE(close_bfloat(actual[4], norm_reference(actual[3], roles.input_norm), "original_decoder_input_norm"));
    const auto expected_qa = qa_reference(actual[4], roles, cuda);
    const bool query_ok = close_bfloat(actual[5], expected_qa, "original_dynamic128_q_a");
    if (!query_ok) diagnose_query_failure(cuda, generation, raw[4], raw[5], expected_qa);
    ASSERT_TRUE(query_ok);
    ASSERT_TRUE(close_bfloat(grouped_values, expected_qa, "original_grouped_dynamic128_q_a"));
    ASSERT_TRUE(close_bfloat(actual[6], norm_reference(actual[5], roles.query_norm), "original_q_a_norm"));
    std::vector<float> expected_query(kIndexWidth);
    for (std::size_t row = 0; row < expected_query.size(); ++row) {
      const auto col = (17u * row + 239u) % std::size_t(kQuery);
      const auto coefficient = original::widen(bf16(0.5F + float((row + 239u) % 13u) / 32.0F));
      expected_query[row] = original::widen(bf16(multiply(original::widen(actual[6][col]), coefficient)));
    }
    ASSERT_TRUE(close_bfloat(query_values, expected_query, "actual_lease_indexer_wq_b"));
    if (generation == 2) {
      for (const auto &values : actual) for (auto value : values) EXPECT_EQ(original::widen(value), 0.0F);
      for (auto value : query_values) EXPECT_EQ(original::widen(value), 0.0F);
      for (auto value : grouped_values) EXPECT_EQ(original::widen(value), 0.0F);
    }
    if (generation == 0) {
      oldest = std::move(current); old_query = std::move(query);
      old_grouped = std::move(grouped_result);
      oldest_bytes = raw; old_query_bytes = query_raw; old_grouped_bytes = grouped_raw;
    } else {
      changed_query |= query_raw != old_query_bytes;
      for (std::size_t i = 0; i < oldest.size(); ++i) EXPECT_EQ(read(oldest[i]), oldest_bytes[i]);
      EXPECT_EQ(read(old_query[0]), old_query_bytes);
      EXPECT_EQ(read(old_grouped[0]), old_grouped_bytes);
    }
  }
  EXPECT_TRUE(distinct_order); EXPECT_TRUE(changed_query);
  for (std::size_t i = 0; i < oldest.size(); ++i) EXPECT_EQ(read(oldest[i]), oldest_bytes[i]);
  EXPECT_EQ(read(old_query[0]), old_query_bytes);
  EXPECT_EQ(read(old_grouped[0]), old_grouped_bytes);
  EXPECT_EQ(read(grouped_ids), bytes<std::int32_t>(expert));
  oldest.clear(); old_query.clear(); old_grouped.clear();
  EXPECT_EQ(first.outputs.stats().live_blocks, 0u); EXPECT_EQ(first.outputs.stats().active_uses, 0u);
  EXPECT_EQ(next.outputs.stats().live_blocks, 0u); EXPECT_EQ(next.outputs.stats().active_uses, 0u);
  EXPECT_EQ(grouped.outputs.stats().live_blocks, 0u); EXPECT_EQ(grouped.outputs.stats().active_uses, 0u);
  for (const auto *program : {&first, &next, &grouped})
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->fragment.constant_bytes[i]) << "immutable original/generated parameter changed";
  ::testing::Test::RecordProperty("scope", "original layer45 NextN input->inputnorm->dynamic128 q_a->qnorm; actual qnorm lease->generated BF16 indexer wq_b; actual inputnorm lease->grouped E1/slot1 original q_a; no full-model/indexer/cache/natural-acceptance claim");
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("actual_submissions", 9);
  ::testing::Test::RecordProperty("grouped_submissions", 3);
  ::testing::Test::RecordProperty("original_role_count", 7);
  ::testing::Test::RecordProperty("original_role_bytes", 73429504);
  ::testing::Test::RecordProperty("q_a_sha256", kQaSha);
  ::testing::Test::RecordProperty("q_a_scale_sha256", kScaleSha);
  ::testing::Test::RecordProperty("input_norm_sha256", kInputNormSha);
  ::testing::Test::RecordProperty("q_a_norm_sha256", kQueryNormSha);
  ::testing::Test::RecordProperty("producer_output_pool", "two generations;14 blocks;110592 bytes");
  ::testing::Test::RecordProperty("successor_output_pool", "two generations;2 blocks;16384 bytes");
  ::testing::Test::RecordProperty("grouped_output_pool", "two generations;2 blocks;6144 bytes");
}

class GlmOriginalMtpChain : public ::testing::Test {
protected:
  static inline std::unique_ptr<const Roles> roles;
  static void SetUpTestSuite() {
    const auto input = directory("HYLE_GLM_MTP_INPUT_DIR"), query = directory("HYLE_GLM_MTP_QUERY_DIR");
    if (!input.empty() && !query.empty()) roles = std::make_unique<const Roles>(Roles::load(input, query));
  }
  static void TearDownTestSuite() { roles.reset(); }
  void SetUp() override {
    if (!roles) GTEST_SKIP() << "Set HYLE_GLM_MTP_INPUT_DIR and HYLE_GLM_MTP_QUERY_DIR to the authenticated seven original layer45 roles";
  }
};
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CUDA)
TEST_F(GlmOriginalMtpChain, CudaOriginalInputFeedsFp8QueryAndActualIndexerLease) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "actual CUDA device resource")), "original CUDA allocation owner");
  run(*roles, context, owner, true);
}
#endif
#if defined(HYLE_TEST_GLM_ORIGINAL_CHAIN_CPU)
struct SystemResource final { m::MemoryResourceRef ref() noexcept { return m::system_resource(); } };
TEST_F(GlmOriginalMtpChain, CpuOriginalInputFeedsFp8QueryAndActualIndexerLease) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "original system allocation owner");
  run(*roles, context, owner, false);
}
#endif
} // namespace
