#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
using Bytes = std::vector<std::byte>;
constexpr std::size_t kHidden = 4096, kRank = 256, kWidth = kHidden + kRank;
constexpr char kPayloadSha256[] =
    "15c5d09c7842d0f947ce0afdc154691ead377e6263af71bd2f7cb34bd592f30f";

// One original trained role, not a whole-model or MTP-quality test:
// unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768
// dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf, conf_proj.weight BF16[1,4352].
// Exact source range10731388192..10731396895 (8704B); no network access here.
// GA inference-model.py:807-815 publishes the raw F32 projection. The pinned
// Unsloth dflash.cpp:365-375 additionally applies sigmoid; publish that policy
// separately so a probability can never stand in for the original raw score.

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result)
    throw std::runtime_error(std::string(where) + ": " +
                             std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok())
    throw std::runtime_error(std::string(where) + ": " +
                             std::string(status.message()));
}
c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> dimensions) {
  return take(c::TensorDesc::dense(type, {dimensions.begin(), dimensions.size()}),
              "tensor description");
}
float widen(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t{value} << 16u);
}
std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}


std::array<float, kWidth> load_original(const char *path) {
  const auto payload = original_role_input::read_bfloat(path, kWidth, kPayloadSha256);
  std::array<float, kWidth> result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = original_role_input::widen(payload[i]);
  return result;
}

using Input = std::array<std::uint16_t, kWidth>;
Input input(const std::array<float, kWidth> &weight, int generation) {
  Input result{};
  if (generation != 2) {
    const int seed = generation == 3 ? 0 : generation;
    for (std::size_t i = 0; i < result.size(); ++i) {
      const auto value = int((i * 37 + std::size_t(seed + 1) * 11) % 61) - 30;
      result[i] = bf16(float(value) / 16.0F);
    }
    return result;
  }
  // Cross the hidden/Markov boundary using two ORIGINAL weight coefficients.
  // Swapped BF16 operands form equal/opposite exact F32 products, so this is
  // strong cancellation without requiring a particular vendor reduction tree.
  const auto largest = [&](std::size_t first, std::size_t end) {
    auto selected = first;
    for (auto i = first + 1; i < end; ++i)
      if (std::abs(weight[i]) > std::abs(weight[selected])) selected = i;
    if (weight[selected] == 0)
      throw std::runtime_error("confidence role has no nonzero coefficient in one input part");
    return selected;
  };
  const auto h = largest(0, kHidden), r = largest(kHidden, kWidth);
  result[h] = bf16(weight[r] * 16.0F);
  result[r] = bf16(-weight[h] * 16.0F);
  return result;
}
struct Reference final { double raw = 0, absolute_products = 0; };
Reference reference(const Input &value, const std::array<float, kWidth> &weight) {
  Reference result;
  double correction = 0;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const double product = double(widen(value[i])) * double(weight[i]);
    const double next = result.raw + product;
    correction += std::abs(result.raw) >= std::abs(product)
        ? (result.raw - next) + product : (product - next) + result.raw;
    result.raw = next;
    result.absolute_products += std::abs(product);
  }
  result.raw += correction; // Independent FP64 Neumaier dot, not a GPU tree.
  return result;
}
double sigmoid(double value) {
  if (value >= 0) return 1.0 / (1.0 + std::exp(-value));
  const auto e = std::exp(value);
  return e / (1.0 + e);
}
c::ReadLease upload(const c::StorageOwner &owner,
                    std::span<const std::byte> payload) {
  auto buffer = take(m::Buffer::create(owner.ref(), payload.size(), 256), "input backing");
  check(buffer.view().write(payload), "upload actual input bytes");
  return take(c::ReadLease::adopt(std::move(buffer)), "ready immutable input");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(), "independent result view").read(result), "read original backing");
  return result;
}
float scalar(const Bytes &payload) {
  if (payload.size() != sizeof(float))
    throw std::runtime_error("confidence output lost its scalar extent");
  float result;
  std::memcpy(&result, payload.data(), sizeof(result));
  return result;
}
} // namespace

TEST(CudaDSparkOriginalConfidence, FullWidthRawAndSidecarPolicyKeepPriorLeases) {
#if defined(_WIN32)
  char *environment = nullptr;
  std::size_t environment_size = 0;
  if (_dupenv_s(&environment, &environment_size, "HYLE_DSPARK_CONFIDENCE_BF16") != 0)
    throw std::runtime_error("cannot read original confidence payload environment");
  const std::unique_ptr<char, decltype(&std::free)> environment_owner(environment, &std::free);
  const std::string path = environment ? environment : "";
#else
  const char *environment = std::getenv("HYLE_DSPARK_CONFIDENCE_BF16");
  const std::string path = environment ? environment : "";
#endif
  if (path.empty())
    GTEST_SKIP() << "Set HYLE_DSPARK_CONFIDENCE_BF16 to the authenticated8704B original role";
  const auto weight = load_original(path.c_str()); // Before native context/device creation.
  RecordProperty("original_payload_sha256", kPayloadSha256);
  RecordProperty("original_payload_role", "conf_proj.weight BF16[1,4352], fixed fbbb5b93fb787c21338159b0af3318bb3f4d9768");
  auto context = take(g::cuda::create_context(), "default CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context,
      take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice), "device resource")), "storage owner");
  auto queue = take(context.create_queue(), "original queue");
  auto weight_buffer = take(m::Buffer::create(owner.ref(), sizeof(weight), 256), "original widened parameter");
  check(weight_buffer.view().write(std::as_bytes(std::span(weight))), "cold BF16-source exact widening");
  const auto hidden_desc = dense(c::DataType::kBFloat16, {1, kHidden});
  const auto markov_desc = dense(c::DataType::kBFloat16, {1, kRank});
  const auto weight_desc = dense(c::DataType::kFloat32, {1, kWidth});
  const auto output_desc = dense(c::DataType::kFloat32, {1, 1});
  auto b = take(g::GraphBuilder::create(), "confidence Graph builder");
  const auto hidden = take(b.input(hidden_desc), "full hidden port");
  const auto markov = take(b.input(markov_desc), "full Markov port");
  const auto parameter = take(b.constant(weight_desc), "immutable original confidence parameter");
  const std::array pair{hidden, markov};
  const auto concatenated = take(b.result(take(b.add_node(c::Operation(c::Concat{1}), pair), "source Concat")), "joined BF16 features");
  const auto features = take(b.add(c::Cast{c::DataType::kFloat32}, concatenated), "source F32 operand boundary");
  const auto rhs = take(b.add(take(c::View::permute(std::array<std::int32_t, 2>{1, 0}), "weight transpose"), parameter), "original parameter view");
  const auto raw = take(b.add(c::MatMul{}, features, rhs), "raw F32 confidence head");
  const auto probability = take(b.add(c::Unary{c::UnaryKind::kSigmoid}, raw), "separate sidecar probability policy");
  check(b.publish(raw), "publish original raw score");
  check(b.publish(probability), "publish separate probability");
  const auto logical = take(std::move(b).finish(), "finish confidence Graph");
  const std::array constants{g::ConstantBinding{parameter,
      take(c::ConstTensorView::create(weight_desc, weight_buffer.view()), "constant parameter view")}};
  auto compiled = take(g::cuda::compile(logical, context, owner, {.constants = constants}), "ordinary confidence compile");
  const std::array values{raw, probability};
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(take(logical.value_description(values[i]), "published output dtype"), output_desc);
    ASSERT_EQ(take(compiled.executable.output_ownership(i), "output ownership"), g::OutputOwnership::kCaller);
  }
  auto pool = take(c::LeasePool::create(owner.ref(), 4 * 256,
      {.max_blocks = 4, .backing_alignment = 256}), "bounded two-generation output pool");
  std::array<c::ReadLease, 2> oldest;
  std::array<Bytes, 2> first_bytes;
  bool f32_rounding_witness = false;
  {
    auto run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "one original Invocation");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    for (int generation = 0; generation < 4; ++generation) {
      SCOPED_TRACE(generation);
      const auto source = input(weight, generation);
      const auto expected = reference(source, weight);
      const double projection_tolerance = 1.0e-6 * std::abs(expected.raw) + 1.0e-6;
      f32_rounding_witness |= std::abs(double(widen(bf16(float(expected.raw)))) - expected.raw) > projection_tolerance;
      if (generation == 2) {
        ASSERT_EQ(expected.raw, 0.0);
        ASSERT_GT(expected.absolute_products, 0.0);
      }
      check(run.bind(hidden, hidden_desc, upload(owner,
          std::as_bytes(std::span(source).first(kHidden)))), "bind actual hidden permission");
      check(run.bind(markov, markov_desc, upload(owner,
          std::as_bytes(std::span(source).subspan(kHidden)))), "bind actual Markov permission");
      for (auto value : values)
        check(run.bind(value, output_desc, take(pool.allocate(sizeof(float)), "fresh output writer")), "bind output permission");
      check(g::cuda::prepare_replay(run, queue,
          {.mode = g::ReplayMode::kPreferCapture, .host_wait = g::HostWaitPolicy::kReject}), "original prepare and seal");
      {
        auto completion = take(run.submit(queues), "strict prepared Graph submit");
        check(completion.wait(), "actual terminal retirement");
      }
      EXPECT_FALSE(run.in_flight());
      std::array<c::ReadLease, 2> current;
      std::array<Bytes, 2> actual;
      for (std::size_t i = 0; i < values.size(); ++i) {
        auto output = take(run.result(values[i]), "completed confidence result");
        current[i] = take(output.storage_lease(), "independent result backing");
        actual[i] = read(current[i]);
      } // No GraphOutput or public Completion remains before reset/reuse.
      const auto actual_raw = scalar(actual[0]), actual_probability = scalar(actual[1]);
      EXPECT_TRUE(std::isfinite(actual_raw));
      EXPECT_TRUE(std::isfinite(actual_probability));
      // Same F32 projection criterion as cublas_provider_test.cc:151; never
      // replace its absolute/relative budget with the looser BF16 criterion.
      EXPECT_NEAR(actual_raw, expected.raw, projection_tolerance);
      // Additional existing forward-error contract (stable_silu_test.cc:949):
      // BF16*widened-BF16 products are exact F32, independent of reduction tree.
      constexpr double u = 0x1p-24;
      constexpr double gamma = (kWidth + 64) * u / (1 - (kWidth + 64) * u);
      const auto rounded = float(expected.raw);
      const double ulp = std::abs(double(std::nextafter(rounded,
          std::numeric_limits<float>::infinity())) - double(rounded));
      EXPECT_LE(std::abs(double(actual_raw) - expected.raw),
          gamma * expected.absolute_products + 2 * ulp);
      const auto expected_probability = sigmoid(expected.raw);
      EXPECT_NEAR(actual_probability, expected_probability,
                  1.0e-6 * std::abs(expected_probability) + 1.0e-6);
      if (generation == 2) {
        EXPECT_EQ(actual_raw, 0.F);
        EXPECT_EQ(actual_probability, .5F);
      }
      check(run.reset(), "withdraw completed permissions");
      EXPECT_EQ(pool.stats().active_uses, 0u);
      EXPECT_FALSE(run.submit(queues).has_value()); // No implicit reprepare.
      if (generation == 0) {
        oldest = std::move(current);
        first_bytes = actual;
      } else {
        for (std::size_t i = 0; i < oldest.size(); ++i)
          EXPECT_EQ(read(oldest[i]), first_bytes[i]);
        EXPECT_EQ(pool.stats().live_blocks, 4u);
        EXPECT_FALSE(pool.allocate(1).has_value());
        if (generation == 1) EXPECT_NE(actual[0], first_bytes[0]);
        if (generation == 3) EXPECT_EQ(actual, first_bytes);
      }
      std::array<float, kWidth> unchanged{};
      check(weight_buffer.view().read(std::as_writable_bytes(std::span(unchanged))), "read immutable original parameter");
      EXPECT_EQ(unchanged, weight);
    }
  }
  EXPECT_TRUE(f32_rounding_witness) << "original head must distinguish raw F32 from BF16-rounded output";
  for (std::size_t i = 0; i < oldest.size(); ++i)
    EXPECT_EQ(read(oldest[i]), first_bytes[i]); // Original Invocation is gone.
  oldest = {};
  EXPECT_EQ(pool.stats().live_blocks, 0u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  RecordProperty("actual_submissions", 4);
  RecordProperty("scope", "one original8704B confidence role/full4096+256width/rawF32+separateSigmoid; not completeMTP/quality");
}
