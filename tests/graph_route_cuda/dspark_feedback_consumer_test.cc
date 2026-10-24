#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Bytes = std::vector<std::byte>;

// Fixed GA Flash source: inference-model.py:719-740,795-815,860-874,939-943
// under .scratch/model-scope-2026-10-19/flash-0731 (revision7872f01b...).
// This begins at the HC-collapsed hidden AND its already-normalized BF16
// counterpart. It tests the actual five-row head feedback, not HC/RMSNorm,
// the three draft stages, KV, verification, accepted state or original payload.
// The finite vocabulary is complete, never an independently sampled shard.
constexpr std::size_t kRows = 5, kHidden = 8, kRank = 4, kVocab = 17;
enum Input : std::size_t {
  kNormalized, kHiddenInput, kFirst, kTeacher, kEmbedding,
  kLmWeight, kMarkovWeight, kConfidenceWeight, kInputs
};
enum Output : std::size_t {
  kIds, kMarkov, kLogits, kConfidence,
  kTeacherIds, kTeacherMarkov, kTeacherLogits, kTeacherConfidence, kOutputs
};

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
                    std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(type, {dims.begin(), dims.size()}), "shape");
}
template <class T> Bytes bytes(std::span<const T> source) {
  const auto view = std::as_bytes(source);
  return {view.begin(), view.end()};
}
template <class T, std::size_t N> Bytes bytes(const std::array<T, N> &source) {
  return bytes<T>(std::span<const T>(source));
}
// Independent scalar BF16 RNE; no provider/test numerical helpers are used.
std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}
float widen(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t{value} << 16u);
}

struct Data final {
  std::array<std::uint16_t, kRows * kHidden> normalized{}, hidden{};
  std::array<std::int32_t, 1> first{};
  std::array<std::int32_t, kRows> teacher{};
  std::array<std::uint16_t, kVocab * kRank> embedding{};
  std::array<float, kVocab * kHidden> lm{};
  std::array<float, kVocab * kRank> markov{};
  std::array<float, kHidden + kRank> confidence{};
  std::array<Bytes, kInputs> payloads() const {
    return {bytes(normalized), bytes(hidden), bytes(first), bytes(teacher),
            bytes(embedding), bytes(lm), bytes(markov), bytes(confidence)};
  }
};
Data data(std::int32_t first, int variant, std::int32_t teacher, bool tie) {
  Data result;
  result.first[0] = first;
  result.teacher.fill(teacher);
  for (std::size_t row = 0; row < kRows; ++row) {
    result.normalized[row * kHidden] = bf16(1.0F);
    result.normalized[row * kHidden + 1] = bf16(float(row) / 4.0F);
    result.normalized[row * kHidden + 3] = bf16(tie && row == 0 ? 1.0F : 0.0F);
    for (std::size_t col = 0; col < kHidden; ++col)
      result.hidden[row * kHidden + col] =
          bf16(float(row + col + static_cast<std::size_t>(variant)) / 8.0F);
  }
  for (std::size_t token = 0; token < kVocab; ++token)
    result.embedding[token * kRank + token % kRank] = bf16(1.0F);
  // Embedding0 -> token2 -> token3 -> token1 -> token0; LM alone chooses4.
  constexpr std::array<std::size_t, kRank> next{2, 0, 3, 1};
  for (std::size_t channel = 0; channel < kRank; ++channel)
    result.markov[next[channel] * kRank + channel] = widen(bf16(8.0F));
  result.lm[4 * kHidden] = widen(bf16(2.0F));
  // One explicit exact top tie; stable greedy must select6 rather than7.
  result.lm[6 * kHidden + 3] = widen(bf16(16.0F));
  result.lm[7 * kHidden + 3] = widen(bf16(16.0F));
  for (std::size_t col = 0; col < kHidden; ++col)
    result.confidence[col] = widen(bf16(float(col + 1) / 16.0F));
  for (std::size_t col = 0; col < kRank; ++col)
    result.confidence[kHidden + col] = widen(bf16(float(1u << col)));
  return result;
}

// Source-order CPU oracle. Float ports are quantized before use; scalar dots
// accumulate independently in double, then round to F32. Coefficients here are
// small dyadic values with sparse products, so the expected sums are exact.
struct Expected final {
  std::array<std::int32_t, kRows + 1> ids{};
  std::array<std::uint16_t, kRows * kRank> markov{};
  std::array<float, kRows * kVocab> logits{};
  std::array<float, kRows> confidence{};
};
Expected reference(const Data &input, bool teacher_forced) {
  Expected result;
  result.ids[0] = input.first[0];
  for (std::size_t row = 0; row < kRows; ++row) {
    const auto token = teacher_forced ? input.teacher[row] : result.ids[row];
    if (token < 0 || static_cast<std::size_t>(token) >= kVocab)
      throw std::runtime_error("oracle received an invalid token");
    for (std::size_t col = 0; col < kRank; ++col)
      result.markov[row * kRank + col] =
          input.embedding[static_cast<std::size_t>(token) * kRank + col];
    for (std::size_t candidate = 0; candidate < kVocab; ++candidate) {
      double lm = 0, bias = 0;
      for (std::size_t col = 0; col < kHidden; ++col)
        lm += double(widen(input.normalized[row * kHidden + col])) *
              double(input.lm[candidate * kHidden + col]);
      for (std::size_t col = 0; col < kRank; ++col)
        bias += double(widen(result.markov[row * kRank + col])) *
                double(input.markov[candidate * kRank + col]);
      result.logits[row * kVocab + candidate] = float(lm) + float(bias);
    }
    std::int32_t best = 0;
    for (std::size_t candidate = 1; candidate < kVocab; ++candidate)
      if (result.logits[row * kVocab + candidate] >
          result.logits[row * kVocab + static_cast<std::size_t>(best)])
        best = static_cast<std::int32_t>(candidate);
    result.ids[row + 1] = best;
    double confidence = 0;
    for (std::size_t col = 0; col < kHidden; ++col)
      confidence += double(widen(input.hidden[row * kHidden + col])) *
                    double(input.confidence[col]);
    for (std::size_t col = 0; col < kRank; ++col)
      confidence += double(widen(result.markov[row * kRank + col])) *
                    double(input.confidence[kHidden + col]);
    result.confidence[row] = float(confidence);
  }
  return result;
}

struct Logical final {
  g::Graph graph;
  std::array<g::ValueId, kInputs> inputs;
  std::array<g::ValueId, kOutputs> outputs;
  std::array<c::TensorDesc, kInputs> descriptions;
};
Logical build() {
  Logical result;
  result.descriptions = {
      dense(c::DataType::kBFloat16, {kRows, kHidden}),
      dense(c::DataType::kBFloat16, {kRows, kHidden}),
      dense(c::DataType::kInt32, {1}), dense(c::DataType::kInt32, {kRows}),
      dense(c::DataType::kBFloat16, {kVocab, kRank}),
      dense(c::DataType::kFloat32, {kVocab, kHidden}),
      dense(c::DataType::kFloat32, {kVocab, kRank}),
      dense(c::DataType::kFloat32, {1, kHidden + kRank})};
  auto b = take(g::GraphBuilder::create(), "feedback builder");
  for (std::size_t i = 0; i < kInputs; ++i)
    result.inputs[i] = take(b.input(result.descriptions[i]), "input");
  const auto slice = [&](g::ValueId value, std::size_t row) {
    return take(b.add(take(c::View::slice(0, static_cast<std::int64_t>(row), 1),
                            "row slice"), value), "slice node");
  };
  const auto transpose = [&](g::ValueId value) {
    return take(b.add(take(c::View::permute(std::array<std::int32_t, 2>{1, 0}),
                            "weight transpose"), value), "weight view");
  };
  const auto as_float = [&](g::ValueId value) {
    return take(b.add(c::Cast{c::DataType::kFloat32}, value), "F32 source boundary");
  };
  const auto concatenate = [&](std::span<const g::ValueId> values, int axis) {
    const auto node = take(b.add_node(c::Operation(c::Concat{axis}), values), "concat");
    return take(b.result(node), "concat output");
  };
  const auto lm = transpose(result.inputs[kLmWeight]);
  const auto mw = transpose(result.inputs[kMarkovWeight]);
  const auto cw = transpose(result.inputs[kConfidenceWeight]);
  const auto base = take(b.add(c::MatMul{}, as_float(result.inputs[kNormalized]), lm), "LM head");
  for (const bool forced : {false, true}) {
    std::array<g::ValueId, kRows + 1> ids;
    std::array<g::ValueId, kRows> embeddings, logits, confidence;
    ids[0] = result.inputs[kFirst];
    for (std::size_t row = 0; row < kRows; ++row) {
      // The natural path's row>0 ID is the actual preceding TopK output.
      const auto current = forced ? slice(result.inputs[kTeacher], row) : ids[row];
      embeddings[row] = take(b.add(c::Gather{}, result.inputs[kEmbedding], current), "Markov Gather");
      const auto bias = take(b.add(c::MatMul{}, as_float(embeddings[row]), mw), "Markov head");
      logits[row] = take(b.add(c::Binary{c::BinaryKind::kAdd}, slice(base, row), bias), "adjusted logits");
      const auto top = take(b.add_node(c::Operation(c::TopK{.count = 1, .stable = true}),
                                      std::array{logits[row]}), "greedy TopK1");
      const auto selected = take(b.result(top, 1), "actual I32 selected ID");
      const auto shape = take(c::TensorShape::create(std::array<std::int64_t, 1>{1}), "ID shape");
      ids[row + 1] = take(b.add(take(c::View::reshape(shape), "ID reshape"), selected), "ID vector");
      const std::array pair{slice(result.inputs[kHiddenInput], row), embeddings[row]};
      // Confidence uses h_i and embedding(token_i), not norm(h_i) or token_(i+1).
      confidence[row] = take(b.add(c::MatMul{}, as_float(concatenate(pair, 1)), cw), "paired confidence");
    }
    const auto first_output = forced ? kTeacherIds : kIds;
    result.outputs[first_output] = concatenate(std::span(ids).subspan(forced ? 1u : 0u), 0);
    result.outputs[first_output + 1] = concatenate(embeddings, 0);
    result.outputs[first_output + 2] = concatenate(logits, 0);
    result.outputs[first_output + 3] = concatenate(confidence, 0);
  }
  for (const auto value : result.outputs) check(b.publish(value), "publish head output");
  result.graph = take(std::move(b).finish(), "finish feedback graph");
  return result;
}

c::ReadLease upload(const c::StorageOwner &owner, const Bytes &payload) {
  auto buffer = take(m::Buffer::create(owner.ref(), payload.size(), 256), "input backing");
  check(buffer.view().write(payload), "upload ready input");
  return take(c::ReadLease::adopt(std::move(buffer)), "ready input owner");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(), "retained result view").read(result), "read independent output");
  return result;
}
std::array<Bytes, kOutputs> expected_bytes(const Data &input) {
  const auto actual = reference(input, false), teacher = reference(input, true);
  return {bytes(actual.ids), bytes(actual.markov), bytes(actual.logits), bytes(actual.confidence),
          bytes<std::int32_t>(std::span(teacher.ids).subspan(1)), bytes(teacher.markov),
          bytes(teacher.logits), bytes(teacher.confidence)};
}

TEST(CudaDSparkFeedback, FiveGreedyRowsFeedActualMarkovIdsAndKeepPriorResults) {
  auto context = take(g::cuda::create_context(), "default CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context,
      take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice), "device resource")), "storage owner");
  auto queue = take(context.create_queue(), "original queue");
  const auto logical = build();
  auto compiled = take(g::cuda::compile(logical.graph, context, owner), "ordinary Graph compile");
  auto run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "original Invocation");
  std::array<c::TensorDesc, kOutputs> descriptions;
  std::size_t generation_bytes = 0;
  for (std::size_t i = 0; i < kOutputs; ++i) {
    descriptions[i] = take(logical.graph.value_description(logical.outputs[i]), "output description");
    const auto size = take(descriptions[i].required_bytes(), "output extent");
    generation_bytes += (size + 255u) / 256u * 256u;
    ASSERT_EQ(take(compiled.executable.output_ownership(i), "caller output policy"), g::OutputOwnership::kCaller);
  }
  EXPECT_EQ(descriptions[kIds].data_type(), c::DataType::kInt32);
  EXPECT_EQ(descriptions[kTeacherIds].data_type(), c::DataType::kInt32);
  auto pool = take(c::LeasePool::create(owner.ref(), generation_bytes * 2,
      {.max_blocks = static_cast<std::uint32_t>(kOutputs * 2), .backing_alignment = 256}), "bounded two-generation output pool");
  std::array<c::ReadLease, kOutputs> first;
  std::array<Bytes, kOutputs> first_bytes, previous;
  for (int repeat = 0; repeat < 4; ++repeat) {
    SCOPED_TRACE(repeat);
    const auto input = data(repeat == 1 || repeat == 2 ? 1 : 0,
                            repeat == 1 || repeat == 2 ? 1 : 0,
                            repeat == 2 ? 1 : 0, repeat == 3);
    const auto oracle = reference(input, false), wrong = reference(input, true);
    const auto expected = expected_bytes(input);
    EXPECT_EQ(oracle.ids.front(), input.first[0]);
    EXPECT_NE(oracle.markov, wrong.markov);
    EXPECT_NE(oracle.confidence, wrong.confidence);
    EXPECT_NE(oracle.ids, wrong.ids);
    if (repeat == 0) {
      EXPECT_EQ(oracle.ids, (std::array<std::int32_t, 6>{0, 2, 3, 1, 0, 2}));
      EXPECT_NE(oracle.ids[1], 4); // LM-alone winner; actual Markov changes sampling.
      EXPECT_NE(oracle.ids[1], oracle.ids[2]);
      EXPECT_NE(oracle.markov[kRank], oracle.markov[0]);
    }
    if (repeat == 3) {
      EXPECT_EQ(oracle.logits[6], 16.0F);
      EXPECT_EQ(oracle.logits[7], 16.0F);
      EXPECT_EQ(oracle.ids[1], 6); // Exact tie, lowest vocabulary index.
    }
    const auto payloads = input.payloads();
    for (std::size_t i = 0; i < kInputs; ++i)
      check(run.bind(logical.inputs[i], logical.descriptions[i], upload(owner, payloads[i])), "bind real input permission");
    for (std::size_t i = 0; i < kOutputs; ++i)
      check(run.bind(logical.outputs[i], descriptions[i],
          take(pool.allocate(take(descriptions[i].required_bytes(), "writer extent")), "fresh output writer")), "bind output writer");
    check(g::cuda::prepare_replay(run, queue,
        {.mode = g::ReplayMode::kPreferCapture, .host_wait = g::HostWaitPolicy::kReject}), "ordinary prepare and seal");
    {
      auto done = take(run.submit(queue), "five-row actual Graph submit");
      check(done.wait(), "real terminal retirement");
    }
    EXPECT_FALSE(run.in_flight());
    std::array<c::ReadLease, kOutputs> current;
    std::array<Bytes, kOutputs> actual;
    for (std::size_t i = 0; i < kOutputs; ++i) {
      auto result = take(run.result(logical.outputs[i]), "successful Graph output");
      if (i == 0) EXPECT_TRUE(run.reset().IsUnavailable()); // Original result owns this slot.
      current[i] = take(result.storage_lease(), "independent output backing");
      actual[i] = read(current[i]);
      EXPECT_EQ(actual[i], expected[i]) << "output=" << i;
      if (i != kIds && i != kMarkov && i != kTeacherIds && i != kTeacherMarkov) {
        for (std::size_t byte = 0; byte < actual[i].size(); byte += sizeof(float)) {
          std::array<std::byte, sizeof(float)> bits{};
          for (std::size_t j = 0; j < bits.size(); ++j) bits[j] = actual[i][byte + j];
          EXPECT_TRUE(std::isfinite(std::bit_cast<float>(bits)));
        }
      }
    }
    check(run.reset(), "withdraw completed input/output permissions");
    EXPECT_EQ(pool.stats().active_uses, 0u);
    EXPECT_FALSE(run.submit(queue).has_value()); // Reset is not a prepared route.
    if (repeat == 0) {
      first = std::move(current); first_bytes = actual;
    } else {
      for (std::size_t i = 0; i < kOutputs; ++i) EXPECT_EQ(read(first[i]), first_bytes[i]);
      EXPECT_EQ(pool.stats().live_blocks, kOutputs * 2);
      EXPECT_FALSE(pool.allocate(1).has_value()); // Two real generations exhaust this pool.
    }
    if (repeat == 1) EXPECT_NE(actual[kIds], first_bytes[kIds]);
    if (repeat == 2) {
      for (std::size_t i = kIds; i <= kConfidence; ++i) EXPECT_EQ(actual[i], previous[i]);
      EXPECT_NE(actual[kTeacherMarkov], previous[kTeacherMarkov]);
      EXPECT_NE(actual[kTeacherConfidence], previous[kTeacherConfidence]);
      EXPECT_NE(actual[kTeacherIds], previous[kTeacherIds]);
    }
    previous = std::move(actual);
  }
  for (std::size_t i = 0; i < kOutputs; ++i) EXPECT_EQ(read(first[i]), first_bytes[i]);
  first = {};
  EXPECT_EQ(pool.stats().live_blocks, 0u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  ::testing::Test::RecordProperty("scope", "finite17-vocabulary/8-hidden/4-Markov, sourceBF16+F32 head boundary only");
  ::testing::Test::RecordProperty("actual_submissions", 4);
  ::testing::Test::RecordProperty("proposal_rows_per_submission", 5);
}
} // namespace
