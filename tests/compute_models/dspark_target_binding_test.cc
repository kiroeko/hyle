#include <gtest/gtest.h>
#include "../../engine/inference/output_pool_capacity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dspark_target_binding.h"
#include "cpu_c4_flash_sidecars.h"
#include "original_role_input.h"
#if defined(HYLE_TEST_DSPARK_TARGET_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_DSPARK_TARGET_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

namespace {
namespace b = dspark_target_binding;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace sidecar = cpu_c4_flash_sidecars;
namespace sem = d5_qwen38_24t_semantic_fragments;
using Bytes = std::vector<std::byte>;
template<class T> T take(hyle::Result<T> result, const char *where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, const char *where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
template<class T> Bytes bytes(std::span<const T> values) {
  const auto source = std::as_bytes(values); return {source.begin(), source.end()};
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> values) {
  auto buffer = take(m::Buffer::create(owner.ref(), values.size(), 256), "bounded fixture input");
  check(buffer.view().write(values), "one-time fixture upload");
  return take(owner.adopt_buffer(std::move(buffer)), "retained fixture allocation owner");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(), "actual readonly storage").read(result), "read actual result"); return result;
}
std::vector<float> floats(const c::ReadLease &lease) {
  const auto raw = read(lease);
  if (raw.size() % 4) throw std::runtime_error("F32 result extent");
  std::vector<float> result(raw.size() / 4); std::memcpy(result.data(), raw.data(), raw.size()); return result;
}
b::ReadonlyRole role(std::string name, c::TensorDesc description, const c::StorageOwner &owner,
                     std::span<const std::byte> source) {
  return {std::move(name), description, owner, upload(owner, source)};
}
b::TargetIdentity target() {
  return {b::kModelRepository, b::kModelRevision, b::kSidecarRepository, b::kSidecarRevision,
          "UD-Q4_K_XL"};
}
b::SidecarIdentity selected_sidecar() { return {b::kSidecarRepository, b::kSidecarRevision, b::kSidecarFile}; }

constexpr auto kFcSha = "4516df35cd05ecd36b93e73ac2ae74c524ec6edda13b634a7895b7eabe0c8c04";
constexpr auto kMainNormSha = "0f99d64b4fa46264c9bd2efdffcb66d3ec0cdd0ca61109f929ed958bcb843f0b";
constexpr auto kWkvSha = "a52e49dc69f4eff1c8287b3887123dd7a583bdaee7d42efe55310f035bafd35f";
constexpr auto kKvNormSha = "edbbd17729cfb3c0852c8b6cdb8c1044eb9f501d8864c7ef4444c568e56fc966";
struct Parameters final {
  sem::Weight fc, wkv;
  std::vector<float> main_scale, kv_scale;
  bool original = false;
};
Parameters generated_parameters() {
  // Existing selected-sidecar complete(main_proj seed673), attention(stage0
  // wkv seed613), and norm helpers; these values are explicitly generated.
  return {sem::weight(b::kHidden, 3 * b::kHidden, c::GgufQuantizedType::kQ8_0, false, 673),
      sem::weight(b::kKv, b::kHidden, c::GgufQuantizedType::kQ8_0, false, 613),
      sem::oracle::scale(b::kHidden), sem::oracle::scale(b::kKv), false};
}
Parameters original_parameters(const std::filesystem::path &directory) {
  Parameters result;
  result.original = true;
  result.fc.description = take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32, 3 * b::kHidden, b::kHidden), "original fc Q8 type");
  result.wkv.description = take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32, b::kHidden, b::kKv), "original stage0 wkv Q8 type");
  result.fc.storage = original_role_input::read_bytes(directory / "fc.weight.q8_0", 53477376, kFcSha);
  result.main_scale = original_role_input::read_float(directory / "enc.output_norm.weight.f32", b::kHidden, kMainNormSha);
  result.wkv.storage = original_role_input::read_bytes(directory / "blk.0.attn_kv.weight.q8_0", 2228224, kWkvSha);
  result.kv_scale = original_role_input::read_float(directory / "blk.0.attn_kv_a_norm.weight.f32", b::kKv, kKvNormSha);
  return result;
}
std::string original_directory() {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_DSPARK_TARGET_ROLES_DIR"))
    throw std::runtime_error("cannot read DSpark original role directory");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto value = std::getenv("HYLE_DSPARK_TARGET_ROLES_DIR");
  return value ? value : "";
#endif
}

// Decode only one original Q8 row at a time with the stock byte codec, then
// independently sum F32 inputs x exact decoded weights in compensated FP64.
// No native vec-dot, native activation quantizer, dense shadow W or provider
// output supplies this equation. The established quantized-chain gate allows
// the native Q8_0/Q8_1 activation quantization, as in original Q27 EH checks.
std::vector<float> project(const Parameters &parameters, const sem::Weight &weight,
                           std::span<const float> input) {
  if (!parameters.original) return sem::project(input, weight);
  const auto k = weight.description.dimensions()[0], n = weight.description.dimensions()[1];
  const auto stride = std::size_t(ggml_row_size(GGML_TYPE_Q8_0, k));
  const auto *codec = ggml_get_type_traits(GGML_TYPE_Q8_0);
  if (!codec || !codec->to_float || input.size() != std::size_t(k) || weight.storage.size() != std::size_t(n) * stride)
    throw std::runtime_error("original DSpark Q8 reference shape/codec differs");
  std::vector<float> row(std::size_t(k), 0.0F), result(std::size_t(n), 0.0F);
  for (std::int64_t output = 0; output < n; ++output) {
    codec->to_float(weight.storage.data() + std::size_t(output) * stride, row.data(), k);
    double sum = 0, correction = 0;
    for (std::int64_t column = 0; column < k; ++column) {
      if (!std::isfinite(row[std::size_t(column)])) throw std::runtime_error("original DSpark decoded weight is nonfinite");
      const double term = double(input[std::size_t(column)]) * row[std::size_t(column)];
      const double next = sum + term;
      correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum;
      sum = next;
    }
    result[std::size_t(output)] = float(sum + correction);
  }
  return result;
}

// Only ordinary public compile/Invocation calls for this finite consumer.
// The reusable binding header contains no submit loop or executor.
struct Run final {
  g::Graph graph;
  std::vector<c::TensorDesc> inputs, outputs;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool pool;
};
Run prepare(g::Graph graph, std::vector<c::TensorDesc> inputs, std::vector<c::TensorDesc> outputs,
            std::span<const g::ConstantBinding> constants, c::Context &context,
            const c::StorageOwner &owner, bool cuda) {
  Run result{.graph = std::move(graph), .inputs = std::move(inputs), .outputs = std::move(outputs)};
  const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(sidecar::q8_profile),
      "fixed formal Q8 sidecar identity"), c::WorkloadPhase::kDecode,
      {0x48594c454d4f444cULL, 0x445350544150ULL, 1}};
  const g::CompileOptions options{.constants = constants, .workload = workload};
  const auto compile = [&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_DSPARK_TARGET_CPU)
    if (!cuda) return g::cpu::compile(result.graph, context, options);
#endif
#if defined(HYLE_TEST_DSPARK_TARGET_CUDA)
    if (cuda) return g::cuda::compile(result.graph, context, owner, options);
#endif
    throw std::runtime_error("DSpark target backend was not built");
  };
  result.compiled = take(compile(), "ordinary target component compile");
  result.invocation = take(result.compiled.executable.create_invocation(
      {.resources = g::ResourceOptions{}}), "original reusable Invocation");
  std::vector<std::size_t> output_sizes;
  output_sizes.reserve(result.outputs.size());
  for (std::size_t i = 0; i < result.outputs.size(); ++i) {
    if (take(result.compiled.executable.output_ownership(i), "component output ownership") != g::OutputOwnership::kCaller)
      throw std::runtime_error("DSpark component requires caller outputs");
    output_sizes.push_back(take(result.outputs[i].required_bytes(), "actual output bytes"));
  }
  if (output_sizes.empty()) throw std::runtime_error("caller output inventory must be nonempty");
  const auto output_capacity = take(hyle::inference::checked_output_pool_capacity(output_sizes, 2, 256),
      "component output pool capacity");
  result.pool = take(c::LeasePool::create(owner.ref(), output_capacity.backing_bytes,
      {.max_blocks = output_capacity.max_blocks, .backing_alignment = 256}), "two bounded component generations");
  return result;
}
std::vector<c::ReadLease> execute(Run &run, c::Queue &queue, std::span<const c::ReadLease> inputs) {
  if (inputs.size() != run.inputs.size()) throw std::runtime_error("DSpark input count");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    check(run.invocation.bind(take(run.graph.input_id(i), "actual input ID"), run.inputs[i], inputs[i]), "bind actual producer lease");
  for (std::size_t i = 0; i < run.outputs.size(); ++i)
    check(run.invocation.bind(take(run.graph.output_id(i), "actual output ID"), run.outputs[i], take(run.pool.allocate(
        take(run.outputs[i].required_bytes(), "result extent")), "new result writer")), "bind caller writer");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  check(run.invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands,
      .host_wait = g::HostWaitPolicy::kAllow}), "prepare complete ordinary component");
  {auto done = take(run.invocation.submit(queues), "strict ordinary submit"); check(done.wait(), "actual native retirement");}
  std::vector<c::ReadLease> result;
  for (std::size_t i = 0; i < run.outputs.size(); ++i) {
    auto value = take(run.invocation.result(take(run.graph.output_id(i), "published output")), "successful Graph result");
    result.push_back(take(value.storage_lease(), "retained actual producer permission"));
  }
  check(run.invocation.reset(), "withdraw Invocation permissions");
  EXPECT_FALSE(run.invocation.submit(queues));
  return result;
}
g::Graph producer(c::TensorDesc capture) {
  auto builder = take(g::GraphBuilder::create(), "capture producer builder");
  for (int i = 0; i < 3; ++i) {
    auto input = take(builder.input(capture), "raw HC input");
    auto output = take(builder.add(c::Unary{c::UnaryKind::kNeg}, input), "actual HC producer arithmetic");
    check(builder.publish(output), "publish actual HC result");
  }
  return take(std::move(builder).finish(), "capture producer graph");
}
void expect_quantized(std::span<const float> actual, std::span<const float> expected) {
  // The existing selected-sidecar quantized-chain gate, without new tolerances.
  d5_qwen38_24t_moe_profiles::expect_output(&sidecar::q8_profile, actual, expected);
}

void exercise(const Parameters &parameters, c::Context &context, const c::StorageOwner &owner, bool cuda) {
  const auto &fc = parameters.fc, &wkv = parameters.wkv;
  const auto &main_scale = parameters.main_scale, &kv_scale = parameters.kv_scale;
  const auto norm_h = take(b::plain(std::array{b::kHidden}), "main norm type");
  const auto norm_kv = take(b::plain(std::array{b::kKv}), "stage norm type");
  b::ProjectionRoles roles{target(), selected_sidecar(), role("fc.weight", fc.description, owner, fc.storage),
      role("enc.output_norm.weight", norm_h, owner, bytes<float>(main_scale))};
  b::StageKvRoles stage{target(), selected_sidecar(), 0, role("blk.0.attn_kv.weight", wkv.description, owner, wkv.storage),
      role("blk.0.attn_kv_a_norm.weight", norm_kv, owner, bytes<float>(kv_scale))};
  auto projection = take(b::make_projection(roles, 1), "typed supplied target projection");
  auto kv = take(b::make_main_kv(stage, 1), "same-stage supplied main KV boundary");
  const auto constant_projection = take(projection.constants(), "retained target constants");
  const auto constant_kv = take(kv.constants(), "retained same-stage constants");
  auto source = prepare(producer(projection.capture), std::vector(3, projection.capture), std::vector(3, projection.capture),
      {}, context, owner, cuda);
  auto main = prepare(projection.graph, std::vector(3, projection.capture), {projection.output},
      constant_projection, context, owner, cuda);
  auto downstream = prepare(kv.graph, {kv.input}, {kv.output}, constant_kv, context, owner, cuda);
  auto queue = take(context.create_queue(), "ordinary source/projection/KV queue");
  const std::array<const b::ReadonlyRole *, 4> immutable{&roles.projection, &roles.norm, &stage.projection, &stage.norm};
  std::array<Bytes, 4> original;
  for (std::size_t i = 0; i < immutable.size(); ++i) original[i] = read(immutable[i]->lease);
  std::vector<c::ReadLease> first_taps, first_main, first_kv;
  std::array<Bytes, 3> first_tap_bytes;
  Bytes first_main_bytes, first_kv_bytes;
  for (std::uint64_t generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    std::array<std::vector<float>, 3> raw;
    std::array<c::ReadLease, 3> input;
    std::vector<float> joined(3 * b::kHidden);
    for (std::size_t layer = 0; layer < raw.size(); ++layer) {
      raw[layer].resize(4 * b::kHidden);
      for (std::int64_t h = 0; h < b::kHidden; ++h) {
        double mean = 0;
        for (int stream = 0; stream < 4; ++stream) {
          const float value = generation == 2 ? 0.0F : float(int((h * 7 + layer * 23 + stream * 11 + generation * 37) % 127) - 63) / 32.0F;
          raw[layer][std::size_t(stream * b::kHidden + h)] = value; mean -= value / 4.0;
        }
        joined[layer * b::kHidden + std::size_t(h)] = float(mean);
      }
      input[layer] = upload(owner, bytes<float>(raw[layer]));
    }
    auto taps = execute(source, queue, input);
    std::vector<float> actual_joined(3 * b::kHidden);
    for (std::size_t layer = 0; layer < taps.size(); ++layer) {
      const auto actual_capture = floats(taps[layer]);
      for (std::int64_t h = 0; h < b::kHidden; ++h) {
        double sum = 0;
        for (int stream = 0; stream < 4; ++stream) {
          const auto index = std::size_t(stream * b::kHidden + h);
          EXPECT_EQ(actual_capture[index], -raw[layer][index]);
          sum += actual_capture[index];
        }
        actual_joined[layer * b::kHidden + std::size_t(h)] = float(sum / 4.0);
      }
    }
    b::CaptureFrame frame{target(), 5, 7, generation + 1, std::int64_t(127 + generation), 1};
    // Deliberately shuffled, with both explicit indexing conventions.
    std::array<b::TargetTap, 3> claims{
      b::TargetTap{frame, b::LayerConvention::kGgufHiddenStateOneBased, 43, projection.capture, taps[2]},
      b::TargetTap{frame, b::LayerConvention::kOfficialBlockZeroBased, 40, projection.capture, taps[0]},
      b::TargetTap{frame, b::LayerConvention::kGgufHiddenStateOneBased, 42, projection.capture, taps[1]}};
    auto stale = claims; ++stale[0].frame.epoch;
    EXPECT_FALSE(projection.qualify(frame, stale));
    auto wrong_position = claims; ++wrong_position[1].frame.first_position;
    EXPECT_FALSE(projection.qualify(frame, wrong_position));
    auto duplicate = claims; duplicate[0].layer = 42;
    EXPECT_FALSE(projection.qualify(frame, duplicate));
    auto foreign = claims; foreign[2].frame.target.model_revision = "different target";
    EXPECT_FALSE(projection.qualify(frame, foreign));
    const auto accepted = take(projection.qualify(frame, claims), "qualify actual target capture tuple");
    for (std::size_t i = 0; i < accepted.size(); ++i) {
      const auto a = take(accepted[i].view(), "qualified actual capture"), z = take(taps[i].view(), "actual producer capture");
      EXPECT_EQ(a.resource(), z.resource()); EXPECT_EQ(a.native_handle(), z.native_handle());
      EXPECT_EQ(a.offset(), z.offset()); EXPECT_EQ(accepted[i].generation(), taps[i].generation());
    }
    auto main_result = execute(main, queue, accepted);
    const std::array real_main{main_result[0]};
    auto kv_result = execute(downstream, queue, real_main);
    const auto main_values = floats(main_result[0]), kv_values = floats(kv_result[0]);
    if (generation == 2) {
      for (auto value : main_values) EXPECT_EQ(value, 0.0F);
      for (auto value : kv_values) EXPECT_EQ(value, 0.0F);
    } else {
      const auto expected_main = sidecar::norm_ref(project(parameters, fc, actual_joined), main_scale);
      expect_quantized(main_values, expected_main);
      const auto whole_main = sidecar::norm_ref(project(parameters, fc, joined), main_scale);
      expect_quantized(main_values, whole_main);
      // Use the actual upstream result as the independent next-dot input;
      // it is never reconstructed or re-uploaded as an execution argument.
      const auto expected_kv = sidecar::norm_ref(project(parameters, wkv, main_values), kv_scale);
      expect_quantized(kv_values, expected_kv);
      const auto whole_kv = sidecar::norm_ref(project(parameters, wkv, whole_main), kv_scale);
      expect_quantized(kv_values, whole_kv);
    }
    if (generation == 0) {
      first_taps = taps; first_main = main_result; first_kv = kv_result;
      for (std::size_t i = 0; i < taps.size(); ++i) first_tap_bytes[i] = read(taps[i]);
      first_main_bytes = read(main_result[0]); first_kv_bytes = read(kv_result[0]);
    } else {
      for (std::size_t i = 0; i < first_taps.size(); ++i) EXPECT_EQ(read(first_taps[i]), first_tap_bytes[i]);
      EXPECT_EQ(read(first_main[0]), first_main_bytes); EXPECT_EQ(read(first_kv[0]), first_kv_bytes);
    }
  }
  first_taps.clear(); first_main.clear(); first_kv.clear();
  for (const auto *run : {&source, &main, &downstream}) {
    EXPECT_EQ(run->pool.stats().live_blocks, 0u); EXPECT_EQ(run->pool.stats().active_uses, 0u);
  }
  for (std::size_t i = 0; i < immutable.size(); ++i) EXPECT_EQ(read(immutable[i]->lease), original[i]);
  auto bad = roles; bad.projection.name = "other role";
  EXPECT_FALSE(b::make_projection(std::move(bad), 1));
  ::testing::Test::RecordProperty("actual_submissions", 9);
  ::testing::Test::RecordProperty("scope", parameters.original
      ? "formal Flash0731 four original Q8/F32 roles; generated raw HC producer leases -> original main projection/norm -> original stage0 wkv/norm 512-wide pre-RoPE; no original target captures/physical ring/full-model claim"
      : "formal Flash0731 selected Q8 role types, generated parameters; actual raw HC producer leases -> mean/concat/main projection/norm -> same stage0 wkv/norm 512-wide pre-RoPE; no trained payload/physical ring/full-model claim");
  if (parameters.original) {
    ::testing::Test::RecordProperty("original_role_count", 4);
    ::testing::Test::RecordProperty("original_role_bytes", 55724032);
    ::testing::Test::RecordProperty("fc_sha256", kFcSha);
    ::testing::Test::RecordProperty("main_norm_sha256", kMainNormSha);
    ::testing::Test::RecordProperty("wkv_sha256", kWkvSha);
    ::testing::Test::RecordProperty("kv_norm_sha256", kKvNormSha);
  }
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
}
class DSparkOriginalTargetBinding : public ::testing::Test {
protected:
  static inline std::unique_ptr<const Parameters> parameters;
  static void SetUpTestSuite() {
    const auto directory = original_directory();
    if (!directory.empty()) parameters = std::make_unique<const Parameters>(original_parameters(directory));
  }
  static void TearDownTestSuite() { parameters.reset(); }
  void SetUp() override {
    if (!parameters) GTEST_SKIP() << "Set HYLE_DSPARK_TARGET_ROLES_DIR to the authenticated four original Flash0731 sidecar roles";
  }
};
#if defined(HYLE_TEST_DSPARK_TARGET_CPU)
struct System final {m::MemoryResourceRef ref() noexcept {return m::system_resource();}};
TEST(DSparkTargetBinding, CpuActualCaptureLeasesReachSameStageMainKv) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  auto owner = take(c::StorageOwner::adopt(context, System{}), "retained CPU resource");
  exercise(generated_parameters(), context, owner, false);
}
TEST_F(DSparkOriginalTargetBinding, CpuOriginalCaptureProjectionFeedsSameStageMainKv) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  auto owner = take(c::StorageOwner::adopt(context, System{}), "retained CPU resource");
  exercise(*parameters, context, owner, false);
}
#endif
#if defined(HYLE_TEST_DSPARK_TARGET_CUDA)
TEST(DSparkTargetBinding, CudaActualCaptureLeasesReachSameStageMainKv) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "actual CUDA device resource")), "retained CUDA owner");
  exercise(generated_parameters(), context, owner, true);
}
TEST_F(DSparkOriginalTargetBinding, CudaOriginalCaptureProjectionFeedsSameStageMainKv) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "actual CUDA device resource")), "retained CUDA owner");
  exercise(*parameters, context, owner, true);
}
#endif
} // namespace
