#include <gtest/gtest.h>

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

#include "dspark_main_history.h"
#include "cpu_c4_flash_sidecars.h"
#include "original_role_input.h"
#include "../../engine/inference/output_pool_capacity.h"
#include "row_cache_fixture.h"
#if defined(HYLE_TEST_DSPARK_HISTORY_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_DSPARK_HISTORY_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

namespace {
namespace h = dspark_main_history;
namespace b = dspark_target_binding;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace s = cpu_c4_flash_sidecars;
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
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size()); check(take(lease.view(), "actual readonly view").read(result), "read actual history"); return result;
}
std::vector<float> floats(const c::ReadLease &lease) {
  const auto raw = read(lease); if (raw.size() % 4) throw std::runtime_error("history F32 extent");
  std::vector<float> result(raw.size() / 4); std::memcpy(result.data(), raw.data(), raw.size()); return result;
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> source) {
  auto backing = take(m::Buffer::create(owner.ref(), source.size(), 256), "history fixture backing");
  check(backing.view().write(source), "one-time explicit upload");
  return take(owner.adopt_buffer(std::move(backing)), "retained original owner");
}
b::ReadonlyRole role(std::string name, const c::TensorDesc &description, const c::StorageOwner &owner,
                     std::span<const std::byte> source) {
  return {std::move(name), description, owner, upload(owner, source)};
}
b::TargetIdentity target() {return {b::kModelRepository, b::kModelRevision, b::kSidecarRepository, b::kSidecarRevision, "UD-Q4_K_XL"};}
b::SidecarIdentity sidecar() {return {b::kSidecarRepository, b::kSidecarRevision, b::kSidecarFile};}
std::string directory(const char *name) {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  if (_dupenv_s(&value, &size, name)) throw std::runtime_error("history environment lookup failed");
  const std::unique_ptr<char, decltype(&std::free)> cleanup(value, &std::free); return value ? value : "";
#else
  const auto value = std::getenv(name); return value ? value : "";
#endif
}
constexpr std::array kWkvSha{
  "a52e49dc69f4eff1c8287b3887123dd7a583bdaee7d42efe55310f035bafd35f",
  "0a707c0a1eb00673838914bb12b070fe45d70c851efdfe8931e6b914979eb09d",
  "8624943fb84a7d82a235d65d262e79ccf93823b6af67afbcc9e2ff83c1856138"};
constexpr std::array kNormSha{
  "edbbd17729cfb3c0852c8b6cdb8c1044eb9f501d8864c7ef4444c568e56fc966",
  "733bdaab61d5b429b3a0caa6f74d082d34d7e993f1cb27d508e403eff4baf569",
  "1e49befb7879c1a6fd05311bc7c23f5436b1818238b33bcf272f4f54cc53b556"};
struct Data final {
  Bytes fc;
  std::vector<float> main_norm;
  std::array<Bytes, 3> wkv;
  std::array<std::vector<float>, 3> norm;
  static Data load(const std::filesystem::path &base, const std::filesystem::path &extra) {
    Data result;
    result.fc = original_role_input::read_bytes(base / "fc.weight.q8_0", 53477376,
        "4516df35cd05ecd36b93e73ac2ae74c524ec6edda13b634a7895b7eabe0c8c04");
    result.main_norm = original_role_input::read_float(base / "enc.output_norm.weight.f32", b::kHidden,
        "0f99d64b4fa46264c9bd2efdffcb66d3ec0cdd0ca61109f929ed958bcb843f0b");
    for (std::size_t stage = 0; stage < 3; ++stage) {
      const auto root = stage == 0 ? base : extra;
      const auto prefix = "blk." + std::to_string(stage);
      result.wkv[stage] = original_role_input::read_bytes(root / (prefix + ".attn_kv.weight.q8_0"), 2228224, kWkvSha[stage]);
      result.norm[stage] = original_role_input::read_float(root / (prefix + ".attn_kv_a_norm.weight.f32"), b::kKv, kNormSha[stage]);
    }
    return result;
  }
};

// The finite test invokes the original public Graph lifecycle. The model
// header only builds a DAG and StateDraft; no product executor is introduced.
struct Run final {
  g::Graph graph;
  std::vector<c::TensorDesc> inputs, outputs;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool pool;
};
Run prepare(g::Graph graph, std::vector<c::TensorDesc> inputs, std::vector<c::TensorDesc> outputs,
            std::span<const g::ConstantBinding> constants, c::Context &context,
            const c::StorageOwner &owner, bool cuda, std::uint32_t generations = 2) {
  Run result{.graph = std::move(graph), .inputs = std::move(inputs), .outputs = std::move(outputs)};
  const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(s::q8_profile), "formal Q8 sidecar profile"),
      c::WorkloadPhase::kDecode, {0x48594c454d4f444cULL, 0x44535048495354ULL, 1}};
  const g::CompileOptions options{.constants = constants, .workload = workload};
  const auto compile = [&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_DSPARK_HISTORY_CPU)
    if (!cuda) return g::cpu::compile(result.graph, context, options);
#endif
#if defined(HYLE_TEST_DSPARK_HISTORY_CUDA)
    if (cuda) return g::cuda::compile(result.graph, context, owner, options);
#endif
    throw std::runtime_error("DSpark history backend unavailable");
  };
  result.compiled = take(compile(), "ordinary history compile");
  result.invocation = take(result.compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "ordinary Invocation");
  std::vector<std::size_t> output_sizes;
  output_sizes.reserve(result.outputs.size());
  for (std::size_t i = 0; i < result.outputs.size(); ++i) {
    if (take(result.compiled.executable.output_ownership(i), "history ownership") != g::OutputOwnership::kCaller)
      throw std::runtime_error("history caller outputs required");
    output_sizes.push_back(take(result.outputs[i].required_bytes(), "history output bytes"));
  }
  if (output_sizes.empty()) throw std::runtime_error("caller output inventory must be nonempty");
  const auto output_capacity = take(hyle::inference::checked_output_pool_capacity(output_sizes, generations, 256),
      "history output pool capacity");
  result.pool = take(c::LeasePool::create(owner.ref(), output_capacity.backing_bytes,
      {.max_blocks = output_capacity.max_blocks, .backing_alignment = 256}), "finite history output pool");
  return result;
}
std::vector<c::ReadLease> execute(Run &run, c::Queue &queue, std::span<const c::ReadLease> input) {
  if (input.size() != run.inputs.size()) throw std::runtime_error("history input count");
  for (std::size_t i = 0; i < input.size(); ++i)
    check(run.invocation.bind(take(run.graph.input_id(i), "history input ID"), run.inputs[i], input[i]), "bind actual input lease");
  for (std::size_t i = 0; i < run.outputs.size(); ++i)
    check(run.invocation.bind(take(run.graph.output_id(i), "history output ID"), run.outputs[i], take(run.pool.allocate(
        take(run.outputs[i].required_bytes(), "history result bytes")), "new history writer")), "bind result permission");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  check(run.invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands,
      .host_wait = g::HostWaitPolicy::kAllow}), "prepare complete history DAG");
  {auto done = take(run.invocation.submit(queues), "strict history submit"); check(done.wait(), "actual history retirement");}
  std::vector<c::ReadLease> result;
  for (std::size_t i = 0; i < run.outputs.size(); ++i) {
    auto value = take(run.invocation.result(take(run.graph.output_id(i), "history published output")), "completed history result");
    result.push_back(take(value.storage_lease(), "retained history result lease"));
  }
  check(run.invocation.reset(), "withdraw history Invocation permissions");
  EXPECT_FALSE(run.invocation.submit(queues)); return result;
}
g::Graph producer(const c::TensorDesc &capture) {
  auto builder = take(g::GraphBuilder::create(), "HC producer builder");
  for (int i = 0; i < 3; ++i) {
    const auto input = take(builder.input(capture), "HC source input");
    check(builder.publish(take(builder.add(c::Unary{c::UnaryKind::kNeg}, input), "actual HC source arithmetic")), "publish HC result");
  }
  return take(std::move(builder).finish(), "HC source graph");
}
std::vector<c::TensorDesc> history_outputs(const h::History &history) {
  std::vector<c::TensorDesc> result;
  for (std::int64_t prefix = 0; prefix < history.rows; ++prefix) {
    for (int stage = 0; stage < 3; ++stage) result.push_back(history.ring_description);
    result.push_back(history.carry_description);
  }
  for (std::size_t stage = 0; stage < 3; ++stage) {
    result.push_back(history.key_description);
    result.push_back(history.key_description);
    result.push_back(history.entry_description);
  }
  return result;
}
std::vector<c::TensorDesc> history_inputs(const h::History &history) {
  return {history.main_description, history.index_description, history.index_description,
          history.ring_description, history.ring_description, history.ring_description};
}
struct Consumer final {
  g::Graph graph;
  std::vector<c::TensorDesc> inputs, outputs;
};
Consumer consumer(const h::History &history) {
  Consumer result;
  const auto indices = take(c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{2}), "visible history slots type");
  const auto output = take(b::plain(std::array<std::int64_t, 4>{1, 1, 1, b::kKv}), "history attention output type");
  result.inputs = {history.carry_description, indices, history.ring_description, history.ring_description, history.ring_description};
  result.outputs = std::vector(3, output);
  auto builder = take(g::GraphBuilder::create(), "encoded history successor builder");
  std::array<g::ValueId, 5> ports;
  for (std::size_t i = 0; i < ports.size(); ++i) ports[i] = take(builder.input(result.inputs[i]), "successor input");
  const auto first = take(c::View::slice(1, 0, b::kKv), "actual carry query slice");
  const auto sliced = take(builder.add(first, ports[0]), "actual main carry query");
  const auto qshape = take(c::View::reshape(take(c::TensorShape::create(
      std::array<std::int64_t, 4>{1, 1, 1, b::kKv}), "query dimensions")), "query shape");
  const auto query = take(builder.add(qshape, sliced), "query from actual selected carry lease");
  for (std::size_t stage = 0; stage < 3; ++stage) {
    const auto gather = take(builder.add_node(c::model_ops::row_cache_gather_decode(h::kEncoding),
        std::array{ports[stage + 2], ports[1]}), "actual encoded stage gather");
    const auto decoded = take(builder.result(gather), "actual BF16 history decode");
    const auto fp32 = take(builder.add(c::Cast{c::DataType::kFloat32}, decoded), "attention BF16 reader");
    const auto shape = take(c::View::reshape(take(c::TensorShape::create(
        std::array<std::int64_t, 4>{1, 2, 1, b::kKv}), "visible KV dimensions")), "visible KV shape");
    const auto kv = take(builder.add(shape, fp32), "physical ring selected KV");
    const auto attention = take(builder.add(c::ScaledDotProductAttention{.scale = 1.0F / std::sqrt(float(b::kKv))}, query, kv, kv), "actual history attention successor");
    check(builder.publish(attention), "publish successor arithmetic");
  }
  result.graph = take(std::move(builder).finish(), "encoded history successor Graph");
  return result;
}
// Use the established definition oracle in bounded five-row chunks, without
// replacing runtime results with oracle uploads. Only initial cache is imported.
Bytes encoded_reference(std::span<const std::uint16_t> values) {
  if (values.size() % b::kKv) throw std::runtime_error("reference cache row extent");
  const auto rows = values.size() / b::kKv;
  const auto width = h::kEncoding.row_bytes();
  Bytes result(rows * width);
  std::vector<std::uint16_t> chunk(row_cache_test::kRows * b::kKv);
  for (std::size_t first = 0; first < rows; first += row_cache_test::kRows) {
    const auto count = (std::min)(row_cache_test::kRows, rows - first);
    std::fill(chunk.begin(), chunk.end(), 0);
    std::copy_n(values.begin() + first * b::kKv, count * b::kKv, chunk.begin());
    const auto oracle = row_cache_test::reference(h::kEncoding, chunk);
    std::memcpy(result.data() + first * width, oracle.bytes.data(), count * width);
  }
  return result;
}
std::vector<float> decode_row(std::span<const std::byte> carrier) {
  if (carrier.size() != h::kEncoding.row_bytes()) throw std::runtime_error("encoded physical row extent");
  std::vector<float> result(b::kKv);
  for (std::size_t d = 0; d < 448; ++d) {
    const auto code = std::to_integer<std::uint8_t>(carrier[d]);
    EXPECT_NE(code & 127, 127);
    const int exponent = int(std::to_integer<std::uint8_t>(carrier[576 + d / 64])) - 127;
    const double value = std::ldexp(row_cache_test::value(code & 127, false), exponent) * (code & 128 ? -1 : 1);
    result[d] = float(row_cache_test::decode(row_cache_test::bfloat(float(value))));
  }
  for (std::size_t d = 448; d < std::size_t(b::kKv); ++d) {
    const auto offset = 448 + (d - 448) * 2;
    const auto bits = std::uint16_t(std::uint16_t(std::to_integer<std::uint8_t>(carrier[offset])) |
        std::uint16_t(std::to_integer<std::uint8_t>(carrier[offset + 1])) << 8);
    result[d] = float(row_cache_test::decode(bits));
  }
  EXPECT_EQ(carrier[583], std::byte{0}); return result;
}
using Image = std::array<Bytes, 5>;
Image image(const h::HistoryState &state, const c::StateVersion &version) {
  Image result;
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = read(take(version.member(state.entries[i]), "complete history member"));
  return result;
}
std::vector<float> norm_reference(const Data &data, std::size_t stage, std::span<const float> main,
                                  const b::CaptureFrame &frame) {
  const auto stride = std::size_t(ggml_row_size(GGML_TYPE_Q8_0, b::kHidden));
  const auto *codec = ggml_get_type_traits(GGML_TYPE_Q8_0);
  // Independent mathematical dot over original decoded weights and actual F32
  // predecessor. Native activation quantization is judged by the established
  // profile gate here, before discontinuous BF16/E4 boundaries.
  std::vector<float> row(b::kHidden), projected(std::size_t(frame.rows * b::kKv));
  for (std::size_t output = 0; output < std::size_t(b::kKv); ++output) {
    codec->to_float(data.wkv[stage].data() + output * stride, row.data(), b::kHidden);
    for (std::int64_t token = 0; token < frame.rows; ++token) {
      double sum = 0, correction = 0;
      for (std::size_t column = 0; column < row.size(); ++column) {
        const double term = double(main[std::size_t(token * b::kHidden) + column]) * row[column];
        const double next = sum + term;
        correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum; sum = next;
      }
      projected[std::size_t(token * b::kKv) + output] = float(sum + correction);
    }
  }
  return s::norm_ref(projected, data.norm[stage]);
}
std::array<Bytes, 3> check_boundaries(const Data &data, const h::History &history,
                                    std::span<const c::ReadLease> results, std::span<const float> main,
                                    const b::CaptureFrame &frame) {
  const auto prefix_count = std::size_t(history.rows * 4);
  if (results.size() != prefix_count + 9) throw std::runtime_error("complete prefix and arithmetic checkpoints required");
  std::array<Bytes, 3> expected_carriers;
  const auto positions = take(h::placement(frame), "reference positions").positions;
  for (std::size_t stage = 0; stage < 3; ++stage) {
    SCOPED_TRACE(::testing::Message() << "stage=" << stage << " first_position=" << frame.first_position);
    const auto normalized = floats(results[prefix_count + stage * 3]);
    const auto joined = floats(results[prefix_count + stage * 3 + 1]);
    const auto raw_entry = read(results[prefix_count + stage * 3 + 2]);
    if (normalized.size() != std::size_t(frame.rows * b::kKv) || joined.size() != normalized.size() || raw_entry.size() != normalized.size() * 2)
      throw std::runtime_error("actual checkpoint shape differs from finite history");
    const auto expected_norm = norm_reference(data, stage, main, frame);
    d5_qwen38_24t_moe_profiles::expect_output(&s::q8_profile, normalized, expected_norm);
    std::vector<float> rounded(normalized.size());
    std::vector<std::uint16_t> entry(normalized.size());
    std::memcpy(entry.data(), raw_entry.data(), raw_entry.size());
    for (std::size_t i = 0; i < rounded.size(); ++i) {
      EXPECT_TRUE(std::isfinite(normalized[i]));
      rounded[i] = float(row_cache_test::decode(row_cache_test::bfloat(normalized[i])));
    }
    const auto expected_joined = s::rope_ref(rounded, 1, positions);
    for (std::size_t i = 0; i < joined.size(); ++i) {
      EXPECT_TRUE(std::isfinite(joined[i]));
      if (i % b::kKv < 448) EXPECT_EQ(joined[i], rounded[i]) << i;
      else EXPECT_NEAR(joined[i], expected_joined[i], 3.e-6 + 3.e-6 * std::abs(expected_joined[i])) << i;
      // Exact Cast is checked against its actual F32 input; the independent
      // trig/rotation gate above does not dictate a backend's last F32 bit.
      EXPECT_EQ(entry[i], row_cache_test::bfloat(joined[i])) << i;
    }
    expected_carriers[stage] = encoded_reference(entry);
  }
  return expected_carriers;
}
void check_prefix(const std::array<Bytes, 3> &expected_carriers, const h::HistoryState &state, const c::StateVersion &from,
                  const c::StateVersion &selected, std::span<const float> main,
                  const b::CaptureFrame &frame, std::uint32_t accepted) {
  const auto slots = take(h::placement(frame), "verified slots").slots;
  for (std::size_t stage = 0; stage < 3; ++stage) {
    const auto old = read(take(from.member(state.entries[stage]), "old ring"));
    const auto actual = read(take(selected.member(state.entries[stage]), "selected physical ring"));
    for (std::int64_t row = 0; row < h::kWindow; ++row) {
      std::int64_t token = -1;
      for (std::uint32_t t = 0; t < accepted; ++t) if (slots[t] == row) token = t;
      const auto width = h::kEncoding.row_bytes();
      const auto offset = std::size_t(row) * width;
      if (token < 0) {
        EXPECT_TRUE(std::equal(old.begin() + offset, old.begin() + offset + width, actual.begin() + offset));
      } else {
        SCOPED_TRACE(::testing::Message() << "stage=" << stage << " accepted=" << accepted
            << " first_position=" << frame.first_position << " token=" << token << " physical_row=" << row);
        const auto expected = std::span(expected_carriers[stage]).subspan(std::size_t(token) * width, width);
        for (std::size_t i = 0; i < width; ++i) EXPECT_EQ(actual[offset + i], expected[i]) << "carrier byte=" << i;
      }
    }
  }
  if (accepted) {
    const auto carry = read(take(selected.member(state.entries[3]), "last accepted main carry"));
    EXPECT_EQ(carry, bytes<float>(main.subspan(std::size_t(accepted - 1) * b::kHidden, b::kHidden)));
  }
}

void exercise(const Data &data, c::Context &context, const c::StorageOwner &owner, bool cuda) {
  const auto fc_description = take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32,
      3 * b::kHidden, b::kHidden), "original main projection type");
  const auto wkv_description = take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32,
      b::kHidden, b::kKv), "original stage projection type");
  b::ProjectionRoles projection_roles{target(), sidecar(), role("fc.weight", fc_description, owner, data.fc),
      role("enc.output_norm.weight", take(b::plain(std::array{b::kHidden}), "main norm type"), owner, bytes<float>(data.main_norm))};
  std::array<b::StageKvRoles, 3> roles;
  for (std::size_t stage = 0; stage < roles.size(); ++stage) {
    const auto prefix = "blk." + std::to_string(stage);
    roles[stage] = {target(), sidecar(), std::int32_t(stage), role(prefix + ".attn_kv.weight", wkv_description, owner, data.wkv[stage]),
      role(prefix + ".attn_kv_a_norm.weight", take(b::plain(std::array{b::kKv}), "stage norm type"), owner, bytes<float>(data.norm[stage]))};
  }
  EXPECT_NE(data.wkv[0], data.wkv[1]); EXPECT_NE(data.wkv[1], data.wkv[2]);
  const auto frequencies = s::ds::rope_frequencies(false, false);
  auto frequency = role("dflash.rope.frequencies", take(b::plain(std::array<std::int64_t, 1>{32}), "frequency type"), owner, bytes<float>(frequencies));
  auto history2 = take(h::make_history(roles, frequency, 2), "three-stage verify2 history");
  auto history1 = take(h::make_history(roles, frequency, 1), "three-stage continuation history");
  auto projection2 = take(b::make_projection(projection_roles, 2), "actual target projection2");
  auto projection1 = take(b::make_projection(projection_roles, 1), "actual target projection1");
  const auto constants2 = take(history2.constants(), "history2 constants"), constants1 = take(history1.constants(), "history1 constants");
  const auto pc2 = take(projection2.constants(), "projection2 constants"), pc1 = take(projection1.constants(), "projection1 constants");
  auto source2 = prepare(producer(projection2.capture), std::vector(3, projection2.capture), std::vector(3, projection2.capture), {}, context, owner, cuda);
  auto source1 = prepare(producer(projection1.capture), std::vector(3, projection1.capture), std::vector(3, projection1.capture), {}, context, owner, cuda);
  auto main2 = prepare(projection2.graph, std::vector(3, projection2.capture), {projection2.output}, pc2, context, owner, cuda);
  auto main1 = prepare(projection1.graph, std::vector(3, projection1.capture), {projection1.output}, pc1, context, owner, cuda);
  auto block = prepare(history2.graph, history_inputs(history2), history_outputs(history2), constants2, context, owner, cuda);
  auto single = prepare(history1.graph, history_inputs(history1), history_outputs(history1), constants1, context, owner, cuda, 4);
  auto successor = consumer(history1);
  auto attention = prepare(successor.graph, successor.inputs, successor.outputs, {}, context, owner, cuda);
  auto queue = take(context.create_queue(), "ordinary history queue");
  auto state = take(h::HistoryState::create(target()), "one model history schema/store");
  int calls = 0;
  const auto produce_main = [&](const b::CaptureFrame &frame, Run &source, Run &main, const b::Projection &logical, int seed) {
    std::array<c::ReadLease, 3> input;
    for (std::size_t layer = 0; layer < input.size(); ++layer) {
      std::vector<float> raw(std::size_t(frame.rows * 4 * b::kHidden));
      for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = float(int((i * 7 + layer * 31 + seed * 17) % 127) - 63) / 32.0F;
      input[layer] = upload(owner, bytes<float>(raw));
    }
    auto taps = execute(source, queue, input); ++calls;
    const std::array<b::TargetTap, 3> qualified{
      b::TargetTap{frame, b::LayerConvention::kGgufHiddenStateOneBased, 41, logical.capture, taps[0]},
      b::TargetTap{frame, b::LayerConvention::kGgufHiddenStateOneBased, 42, logical.capture, taps[1]},
      b::TargetTap{frame, b::LayerConvention::kGgufHiddenStateOneBased, 43, logical.capture, taps[2]}};
    const auto ordered = take(logical.qualify(frame, qualified), "actual target tap tuple");
    auto result = execute(main, queue, ordered); ++calls; return result[0];
  };
  const auto run_history = [&](const h::History &logical, Run &run, const c::StateVersion &from,
                               const b::CaptureFrame &frame, const c::ReadLease &main) {
    auto stale = frame; ++stale.epoch; EXPECT_FALSE(logical.qualify(stale, frame, main));
    const auto actual = take(logical.qualify(frame, frame, main), "actual qualified main projection result");
    const auto positions = take(h::placement(frame), "explicit verified positions");
    std::vector<c::ReadLease> input{actual, upload(owner, bytes<std::int32_t>(positions.positions)), upload(owner, bytes<std::int32_t>(positions.slots))};
    for (std::size_t stage = 0; stage < 3; ++stage) input.push_back(take(from.member(state.entries[stage]), "old stage physical ring"));
    auto result = execute(run, queue, input); ++calls; return result;
  };
  const auto consume = [&](const c::StateVersion &version) {
    const auto facts = take(state.boundary(version), "selected consumer frontier");
    const std::array<std::int32_t, 2> ids{std::int32_t((facts.next_position - 2) % h::kWindow),
        std::int32_t((facts.next_position - 1) % h::kWindow)};
    std::vector<c::ReadLease> inputs{take(version.member(state.entries[3]), "actual selected main carry"),
        upload(owner, bytes<std::int32_t>(ids))};
    for (std::size_t stage = 0; stage < 3; ++stage) inputs.push_back(take(version.member(state.entries[stage]), "actual selected encoded ring"));
    const auto before_consumer = image(state, version);
    auto results = execute(attention, queue, inputs); ++calls;
    const auto query = floats(inputs[0]);
    const float factor = 1.0F / std::sqrt(float(b::kKv));
    for (std::size_t stage = 0; stage < 3; ++stage) {
      std::array<std::vector<float>, 2> keys;
      for (std::size_t row = 0; row < 2; ++row) keys[row] = decode_row(std::span(before_consumer[stage]).subspan(
          std::size_t(ids[row]) * h::kEncoding.row_bytes(), h::kEncoding.row_bytes()));
      std::array<double, 2> scores{}, weights{};
      for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t d = 0; d < std::size_t(b::kKv); ++d) scores[row] += double(query[d]) * keys[row][d];
        scores[row] *= factor;
      }
      const auto maximum = (std::max)(scores[0], scores[1]);
      for (std::size_t row = 0; row < 2; ++row) weights[row] = std::exp(scores[row] - maximum);
      const auto denominator = weights[0] + weights[1];
      const auto actual = floats(results[stage]);
      ASSERT_EQ(actual.size(), std::size_t(b::kKv));
      for (std::size_t d = 0; d < actual.size(); ++d) {
        const auto expected = (weights[0] * keys[0][d] + weights[1] * keys[1][d]) / denominator;
        ASSERT_TRUE(std::isfinite(actual[d]));
        EXPECT_NEAR(actual[d], expected, 2e-5 + 2e-5 * std::abs(expected)) << "stage=" << stage << " d=" << d;
      }
    }
    EXPECT_EQ(image(state, version), before_consumer);
  };
  {
    std::vector<c::StateMember> members;
    for (std::size_t stage = 0; stage < 3; ++stage) {
      std::vector<std::uint16_t> ring(std::size_t(h::kWindow * b::kKv));
      for (std::size_t i = 0; i < ring.size(); ++i) ring[i] = row_cache_test::bfloat(float(stage * 100 + 10) + float(i % 127) / 128);
      members.push_back({state.entries[stage], upload(owner, encoded_reference(ring))});
    }
    const std::vector<float> initial_carry(b::kHidden, .25F);
    members.push_back({state.entries[3], upload(owner, bytes<float>(initial_carry))});
    const h::Boundary initial{5, 7, 1, 127, 3, 0};
    members.push_back({state.entries[4], {}, std::as_bytes(std::span(&initial, 1))});
    auto base = take(state.store.import(members), "declared initial three-stage history boundary"); members.clear();
    const auto base_image = image(state, base);
    const b::CaptureFrame frame{target(), 5, 7, 2, 127, 2};
    auto actual_main = produce_main(frame, source2, main2, projection2, 3);
    const auto main_values = floats(actual_main);
    auto results = run_history(history2, block, base, frame, actual_main);
    const auto expected = check_boundaries(data, history2, results, main_values, frame);
    const auto prefix_results = std::span<const c::ReadLease>(results).first(std::size_t(frame.rows * 4));
    std::array<c::StateHead, 3> heads;
    std::array<c::StateVersion, 3> before;
    std::array<Image, 3> snapshots;
    for (std::uint32_t accepted = 0; accepted <= 2; ++accepted) {
      const auto index = accepted == 0 ? 2u : accepted - 1;
      heads[index] = take(c::StateHead::create(state.store, base), "independent accepted/old branch head");
      auto candidate = take(state.select(base, prefix_results, frame, accepted, 17 + std::int32_t(accepted)), "select whole three-stage prefix");
      if (accepted) check(heads[index].publish(base.identity(), candidate), "single whole-prefix publication");
      else EXPECT_EQ(candidate.identity(), base.identity());
      before[index] = take(heads[index].current(), "actual published history");
      snapshots[index] = image(state, before[index]);
      check_prefix(expected, state, base, before[index], main_values, frame, accepted);
      EXPECT_EQ(image(state, base), base_image);
    }
    consume(before[0]); // Real prefix1 selected carrier, before any continuation.
    EXPECT_TRUE(heads[0].publish(base.identity(), base).IsUnavailable());
    auto failed = take(state.store.begin(base), "staged history failure");
    check(failed.replace(state.entries[0], results[0]), "stage one valid ring");
    EXPECT_TRUE(failed.replace(state.entries[3], results[0]).IsInvalidArgument());
    check(failed.cancel(), "cancel whole failed history draft"); EXPECT_FALSE(failed.finalize());
    for (std::size_t branch = 0; branch < heads.size(); ++branch) {
      const auto prior = take(heads[branch].current(), "accepted/old live head");
      const auto facts = take(state.boundary(prior), "actual head frontier");
      const b::CaptureFrame fresh{target(), 5, 7, 3 + branch, facts.next_position, 1};
      auto next_main = produce_main(fresh, source1, main1, projection1, 19 + int(branch) * 13);
      const auto next_values = floats(next_main);
      auto continued = run_history(history1, single, prior, fresh, next_main);
      const auto next_expected = check_boundaries(data, history1, continued, next_values, fresh);
      const auto continued_prefix = std::span<const c::ReadLease>(continued).first(std::size_t(fresh.rows * 4));
      auto candidate = take(state.select(prior, continued_prefix, fresh, 1, 31 + std::int32_t(branch)), "actual own-head continuation");
      check(heads[branch].publish(prior.identity(), candidate), "atomic next verified-main publication");
      const auto current = take(heads[branch].current(), "continued complete history");
      check_prefix(next_expected, state, prior, current, next_values, fresh, 1);
      const auto after = take(state.boundary(current), "continued metadata");
      EXPECT_EQ(after.next_position, facts.next_position + 1); EXPECT_EQ(after.pending, 31 + std::int32_t(branch));
      EXPECT_EQ(after.consumed, facts.consumed + 1); EXPECT_EQ(after.epoch, fresh.epoch);
      EXPECT_EQ(image(state, before[branch]), snapshots[branch]); EXPECT_EQ(image(state, base), base_image);
      consume(current); // Includes the retained old branch's actual fresh result.
    }
    for (auto &value : std::span(results).first(std::size_t(frame.rows * 4))) EXPECT_FALSE(value.try_write());
  }
  check(state.store.collect(), "collect retired history versions");
  EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  for (const auto *run : {&source2, &source1, &main2, &main1, &block, &single, &attention}) {
    EXPECT_EQ(run->pool.stats().live_blocks, 0u); EXPECT_EQ(run->pool.stats().active_uses, 0u);
  }
  EXPECT_EQ(read(projection_roles.projection.lease), data.fc);
  EXPECT_EQ(read(projection_roles.norm.lease), bytes<float>(data.main_norm));
  for (std::size_t stage = 0; stage < 3; ++stage) {
    EXPECT_EQ(read(roles[stage].projection.lease), data.wkv[stage]);
    EXPECT_EQ(read(roles[stage].norm.lease), bytes<float>(data.norm[stage]));
  }
  EXPECT_EQ(calls, 16);
  ::testing::Test::RecordProperty("actual_submissions", calls);
  ::testing::Test::RecordProperty("original_role_count", 8);
  ::testing::Test::RecordProperty("original_role_bytes", 60184576);
  ::testing::Test::RecordProperty("physical_ring_row_bytes", h::kEncoding.row_bytes());
  ::testing::Test::RecordProperty("numerical_checks", "original decoded Q8/F32 independent dot/norm with established profile gate; actual postnorm -> BF16/last64 RoPE F32 gate; actual joined F32 -> exact BF16 bits; actual BF16 entry -> exact584B row and selected ring; exact old-ring preservation; actual carrier -> independent attention oracle");
  ::testing::Test::RecordProperty("scope", "original Flash0731 Q8 fc/norm + three distinct stage wkv/norm; generated HC -> original main projection lease -> F32 norm -> BF16 -> last64 RoPE -> BF16 -> group64 QDQ physical128x584 three-rings -> actual gather-decode/attention; artificial prefix0/1/2 and fresh own-head/old-branch continuation; not original target, full official BF16 arithmetic or full DSpark MTP");
}
class DSparkMainHistory : public ::testing::Test {
protected:
  static inline std::unique_ptr<const Data> data;
  static void SetUpTestSuite() {
    const auto base = directory("HYLE_DSPARK_TARGET_ROLES_DIR"), extra = directory("HYLE_DSPARK_HISTORY_ROLES_DIR");
    if (!base.empty() && !extra.empty()) data = std::make_unique<const Data>(Data::load(base, extra));
  }
  static void TearDownTestSuite() {data.reset();}
  void SetUp() override {if (!data) GTEST_SKIP() << "Set authenticated HYLE_DSPARK_TARGET_ROLES_DIR and HYLE_DSPARK_HISTORY_ROLES_DIR";}
};
#if defined(HYLE_TEST_DSPARK_HISTORY_CPU)
struct System final {m::MemoryResourceRef ref() noexcept {return m::system_resource();}};
TEST_F(DSparkMainHistory, CpuThreeOriginalStagesPublishPhysicalAcceptedHistoryAndContinue) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  auto owner = take(c::StorageOwner::adopt(context, System{}), "retained CPU owner"); exercise(*data, context, owner, false);
}
#endif
#if defined(HYLE_TEST_DSPARK_HISTORY_CUDA)
TEST_F(DSparkMainHistory, CudaThreeOriginalStagesPublishPhysicalAcceptedHistoryAndContinue) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice), "actual CUDA resource")), "retained CUDA owner"); exercise(*data, context, owner, true);
}
#endif
} // namespace
