#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/original_role_input.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace original = original_role_input;
constexpr std::size_t kHidden = 4096, kWidth = 2 * kHidden;
constexpr float kEpsilon = 1.0e-5F;
constexpr auto kProjectionSha = "17af8402a6b674a462f2847612b2477e76c0de7dfc5ff9fba50ead1e5c88d649";
constexpr auto kEmbeddingNormSha = "187f2b4efe991b8c69a1a0365ded195b7fd463d50a7facd0d7196402a66dcc54";
constexpr auto kHiddenNormSha = "3fbc01762fea938ae24e94ece075b03dd5cb44cbe218884712b35d865b820501";

// zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a,
// original layer45 eh_proj/enorm/hnorm, authenticated safetensors ranges.
// Source boundary: SGLang@06992c92ad4494fe9f9636e75be3e35040512447
// deepseek_nextn.py:198-212 and fused_eh_norm impl/norm.cuh:69-106.
// BF16 activations/weights, F32 normalization and multiply, ONE final BF16
// store; concatenate embedding before hidden. This is an original parameter
// input component, not the NextN attention/MoE/cache/head or complete model.
template<class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
std::string input_directory() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_GLM_MTP_INPUT_DIR"))
    throw std::runtime_error("cannot read GLM original role directory");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto value = std::getenv("HYLE_GLM_MTP_INPUT_DIR");
  return value ? value : "";
#endif
}
c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions) {
  return take(c::TensorDesc::dense(c::DataType::kBFloat16,
      {dimensions.begin(), dimensions.size()}), "BF16 description");
}
std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::uint16_t> values) {
  auto buffer = take(m::Buffer::create(owner.ref(), values.size_bytes(), 256), "device input allocation");
  check(buffer.view().write(std::as_bytes(values)), "immutable actual role/input upload");
  return take(c::ReadLease::adopt(std::move(buffer)), "independent input permission");
}
std::vector<std::uint16_t> read(const c::ReadLease &lease) {
  if (lease.size() % 2) throw std::runtime_error("BF16 result extent");
  std::vector<std::uint16_t> output(lease.size() / 2);
  check(take(lease.view(), "readonly result view").read(std::as_writable_bytes(std::span(output))), "actual result read");
  return output;
}
std::vector<std::uint16_t> input(int generation, bool embedding) {
  std::vector<std::uint16_t> result(kHidden);
  if (generation == 2) return result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const int value = int((i * (embedding ? 37u : 19u) +
        std::size_t(embedding ? 11 : 3) + std::size_t(generation) * 23u) % 97u) - 48;
    result[i] = bf16(float(value) / (embedding ? 64.0F : 32.0F));
  }
  return result;
}
std::vector<float> norm_reference(std::span<const std::uint16_t> input,
                                 std::span<const std::uint16_t> weight) {
  double square = 0;
  for (auto value : input) {
    const auto f = double(original::widen(value));
    square += f * f;
  }
  const auto factor = 1.0 / std::sqrt(square / double(input.size()) + double(kEpsilon));
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < output.size(); ++i)
    output[i] = original::widen(bf16(float(double(original::widen(input[i])) *
        factor * double(original::widen(weight[i])))));
  return output;
}
bool close_bfloat(std::span<const std::uint16_t> actual,
                  std::span<const float> expected, std::string_view role) {
  if (actual.size() != expected.size()) { ADD_FAILURE() << role << " extent"; return false; }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto value = original::widen(actual[i]);
    // Existing BF16 cuBLAS regression gate, not adjusted from these inputs.
    const double tolerance = 0.008 * std::abs(double(expected[i])) + 0.0001;
    if (!std::isfinite(value) || std::abs(double(value) - expected[i]) > tolerance) {
      ADD_FAILURE() << role << '[' << i << "] actual=" << value << " independent=" << expected[i]
                    << " tolerance=" << tolerance;
      return false;
    }
  }
  return true;
}

struct Dot final { double sum = 0, correction = 0; };
void add(Dot &dot, double product) {
  const auto next = dot.sum + product;
  dot.correction += std::abs(dot.sum) >= std::abs(product)
      ? (dot.sum - next) + product : (product - next) + dot.sum;
  dot.sum = next;
}

TEST(CudaGlmOriginalMtpInput, Layer45OriginalNormAndProjectionPreserveOrderAndOldResults) {
  const auto directory = input_directory();
  if (directory.empty()) GTEST_SKIP() << "Set HYLE_GLM_MTP_INPUT_DIR to pinned layer45 BF16 roles";
  const auto root = std::filesystem::path(directory);
  const auto projection = original::read_bfloat(root / "eh_proj.weight.bf16", kHidden * kWidth, kProjectionSha);
  const auto enorm = original::read_bfloat(root / "enorm.weight.bf16", kHidden, kEmbeddingNormSha);
  const auto hnorm = original::read_bfloat(root / "hnorm.weight.bf16", kHidden, kHiddenNormSha);
  auto context = take(g::cuda::create_context(), "default CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "original device resource")), "allocation domain");
  auto queue = take(context.create_queue(), "original compute queue");
  auto builder = take(g::GraphBuilder::create(), "ordinary Graph builder");
  const auto row = dense({1, kHidden}), joined_desc = dense({1, kWidth});
  const auto hidden = take(builder.input(row), "previous collapsed hidden");
  const auto embedding = take(builder.input(row), "token embedding");
  const auto en_weight = take(builder.constant(dense({kHidden})), "enorm role");
  const auto hn_weight = take(builder.constant(dense({kHidden})), "hnorm role");
  const auto eh_weight = take(builder.constant(dense({kHidden, kWidth})), "original EH row-major role");
  const auto hn = take(builder.add(c::RmsNorm{.epsilon = kEpsilon}, hidden, hn_weight), "hidden RMSNorm");
  const auto en = take(builder.add(c::RmsNorm{.epsilon = kEpsilon}, embedding, en_weight), "embedding RMSNorm");
  const auto joined = take(builder.add(c::Concat{.axis = 1}, en, hn), "embedding-before-hidden concat");
  const std::array<std::int32_t, 2> axes{1, 0};
  const auto weight = take(builder.add(take(c::View::permute(axes), "EH transpose description"), eh_weight), "EH logical transpose");
  const auto output = take(builder.add(c::MatMul{}, joined, weight), "original BF16 projection");
  const std::array outputs{hn, en, joined, output};
  const std::array descriptions{row, row, joined_desc, row};
  for (auto value : outputs) check(builder.publish(value), "public result");
  const auto graph = take(std::move(builder).finish(), "finish original component Graph");
  const std::array parameter_leases{upload(owner, enorm), upload(owner, hnorm), upload(owner, projection)};
  const std::array parameter_ids{en_weight, hn_weight, eh_weight};
  const std::array parameter_desc{dense({kHidden}), dense({kHidden}), dense({kHidden, kWidth})};
  std::array<g::ConstantBinding, 3> constants;
  for (std::size_t i = 0; i < constants.size(); ++i)
    constants[i] = {parameter_ids[i], take(c::ConstTensorView::create(parameter_desc[i],
        take(parameter_leases[i].view(), "immutable original parameter")), "typed original binding")};
  auto compiled = take(g::cuda::compile(graph, context, owner, {.constants = constants}), "cold original input component");
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < descriptions.size(); ++i) {
    bytes += take(descriptions[i].required_bytes(), "output bytes");
    ASSERT_EQ(take(compiled.executable.output_ownership(i), "caller output"), g::OutputOwnership::kCaller);
  }
  auto pool = take(c::LeasePool::create(owner.ref(), 2 * bytes,
      {.max_blocks = 8, .backing_alignment = 256}), "two generations");
  std::array<c::ReadLease, 4> oldest;
  std::array<std::vector<std::uint16_t>, 4> oldest_values;
  std::size_t order_witnesses = 0;
  {
    auto invocation = take(compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{}}), "one reusable Invocation");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    for (int generation = 0; generation < 3; ++generation) {
      SCOPED_TRACE(generation);
      const auto h = input(generation, false), e = input(generation, true);
      check(invocation.bind(hidden, row, upload(owner, h)), "bind current hidden");
      check(invocation.bind(embedding, row, upload(owner, e)), "bind current embedding");
      for (std::size_t i = 0; i < outputs.size(); ++i)
        check(invocation.bind(outputs[i], descriptions[i], take(pool.allocate(
            take(descriptions[i].required_bytes(), "output extent")), "new result writer")), "bind caller result");
      check(g::cuda::prepare_replay(invocation, queue,
          {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}), "prepare actual component");
      { auto done = take(invocation.submit(queues), "strict submit"); check(done.wait(), "whole component completion"); }
      std::array<c::ReadLease, 4> current;
      std::array<std::vector<std::uint16_t>, 4> actual;
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        auto result = take(invocation.result(outputs[i]), "completed result");
        current[i] = take(result.storage_lease(), "independent old-result owner");
        actual[i] = read(current[i]);
      }
      ASSERT_TRUE(close_bfloat(actual[0], norm_reference(h, hnorm), "original hidden norm"));
      ASSERT_TRUE(close_bfloat(actual[1], norm_reference(e, enorm), "original embedding norm"));
      auto expected_join = actual[1];
      expected_join.insert(expected_join.end(), actual[0].begin(), actual[0].end());
      EXPECT_EQ(actual[2], expected_join);
      std::vector<float> expected(kHidden);
      for (std::size_t out = 0; out < kHidden; ++out) {
        Dot correct, reversed;
        for (std::size_t col = 0; col < kWidth; ++col) {
          const auto w = double(original::widen(projection[out * kWidth + col]));
          add(correct, double(original::widen(actual[2][col])) * w);
          add(reversed, double(original::widen(actual[2][(col + kHidden) % kWidth])) * w);
        }
        expected[out] = original::widen(bf16(float(correct.sum + correct.correction)));
        const auto wrong = original::widen(bf16(float(reversed.sum + reversed.correction)));
        order_witnesses += std::abs(double(wrong) - expected[out]) >
            0.008 * std::abs(double(expected[out])) + 0.0001;
      }
      ASSERT_TRUE(close_bfloat(actual[3], expected, "original EH projection"));
      if (generation == 2)
        for (const auto &values : actual)
          EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](auto value) { return value == 0; }));
      check(invocation.reset(), "withdraw successful permissions");
      EXPECT_FALSE(invocation.submit(queues).has_value());
      if (generation == 0) { oldest = std::move(current); oldest_values = actual; }
      else for (std::size_t i = 0; i < oldest.size(); ++i) EXPECT_EQ(read(oldest[i]), oldest_values[i]);
    }
  }
  EXPECT_GT(order_witnesses, 100u);
  for (std::size_t i = 0; i < oldest.size(); ++i) EXPECT_EQ(read(oldest[i]), oldest_values[i]);
  oldest = {};
  EXPECT_EQ(pool.stats().live_blocks, 0u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  RecordProperty("scope", "original GLM-5.3-Flash layer45 BF16 norm/EH input component only; no attention/MoE/state/head/full-model claim");
  RecordProperty("actual_submissions", 3);
  RecordProperty("original_role_bytes", 67125248);
  RecordProperty("eh_proj_sha256", kProjectionSha);
  RecordProperty("enorm_sha256", kEmbeddingNormSha);
  RecordProperty("hnorm_sha256", kHiddenNormSha);
}
} // namespace
