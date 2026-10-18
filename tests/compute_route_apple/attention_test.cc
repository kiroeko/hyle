#include "tensor_test_support.h"

#include <array>
#include <cmath>
#include <limits>

namespace {
using namespace apple_tensor_test;

void attention_case(int depth, mx::Dtype type, c::AttentionLayout layout,
                    int queries, int keys, float scale, bool causal,
                    c::MaskKind mask_kind) {
  SCOPED_TRACE(::testing::Message() << "depth=" << depth << " dtype=" << type
      << " layout=" << int(layout) << " Q=" << queries << " KV=" << keys
      << " scale=" << scale << " causal=" << causal
      << " mask=" << int(mask_kind));
  constexpr int heads = 4, kv_heads = 2;
  std::vector<double> q(heads * queries * depth), k(kv_heads * keys * depth),
      v(kv_heads * keys * depth), mask(queries * keys);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t)
      q[(h * queries + t) * depth] = double((t + h) % 7 - 3) / 8;
  for (int h = 0; h < kv_heads; ++h)
    for (int s = 0; s < keys; ++s) {
      k[(h * keys + s) * depth] = double((s + h * 3) % 9 - 4) / 8;
      for (int d = 0; d < depth; ++d)
        v[(h * keys + s) * depth + d] = double((s * 5 + h * 3 + d) % 17 - 8) / 16;
    }
  for (int t = 0; t < queries; ++t)
    for (int s = 0; s < keys; ++s)
      mask[t * keys + s] = mask_kind == c::MaskKind::kBoolean
          ? double(t % 64 != 0 && s % 3 != 0) : double((t + s) % 3 - 1) / 4;

  auto query = make(q, {1, heads, queries, depth}, type);
  auto key = make(k, {1, kv_heads, keys, depth}, type);
  auto value = make(v, {1, kv_heads, keys, depth}, type);
  if (layout == c::AttentionLayout::kBatchSequenceHeadDepth) {
    query = mx::transpose(query, {0, 2, 1, 3}, stream());
    key = mx::transpose(key, {0, 2, 1, 3}, stream());
    value = mx::transpose(value, {0, 2, 1, 3}, stream());
  }
  std::vector<mx::array> inputs{ready(query), ready(key), ready(value)};
  if (mask_kind == c::MaskKind::kBoolean) {
    std::vector<std::uint8_t> bytes(mask.begin(), mask.end());
    inputs.push_back(ready(mx::astype(
        mx::array(bytes.data(), {queries, keys}, mx::uint8), mx::bool_, stream())));
  } else if (mask_kind == c::MaskKind::kAdditive) {
    inputs.push_back(make(mask, {queries, keys}, type));
  }
  const c::ScaledDotProductAttention operation{
      .layout = layout, .mask = mask_kind, .scale = scale, .causal = causal};
  auto context = a::Context::create().value();
  auto output = execute(context, operation, inputs);
  ASSERT_TRUE(output) << output.error().message();
  if (layout == c::AttentionLayout::kBatchSequenceHeadDepth)
    *output = ready(mx::contiguous(
        mx::transpose(*output, {0, 2, 1, 3}, stream()), false, stream()));

  std::vector<double> expected(q.size()), scores(keys);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t) {
      const int kh = h / (heads / kv_heads);
      bool any = false;
      double maximum = -std::numeric_limits<double>::infinity();
      for (int s = 0; s < keys; ++s) {
        const bool visible = (!causal || s <= keys - queries + t) &&
            (mask_kind != c::MaskKind::kBoolean || mask[t * keys + s] != 0);
        scores[s] = visible ? q[(h * queries + t) * depth] *
            k[(kh * keys + s) * depth] * scale +
            (mask_kind == c::MaskKind::kAdditive ? mask[t * keys + s] : 0)
            : -std::numeric_limits<double>::infinity();
        any |= visible;
        maximum = std::max(maximum, scores[s]);
      }
      // The existing MLX boolean/causal fallback uses a finite minimum for
      // masked scores, hence a fully masked row retains its uniform result.
      double denominator = 0;
      for (auto &score : scores) {
        score = any ? std::exp(score - maximum) : 1;
        denominator += score;
      }
      // V repeats every 17 channels; compute those FP64 oracle sums once.
      std::array<double, 17> sums{};
      for (int d = 0; d < std::min(depth, 17); ++d)
        for (int s = 0; s < keys; ++s)
          sums[d] += scores[s] * v[(kh * keys + s) * depth + d];
      for (int d = 0; d < depth; ++d)
        expected[(h * queries + t) * depth + d] = sums[d % 17] / denominator;
    }
  expect(*output, expected);
}

TEST(AppleAttention, SupportedLargeDepthsRetainLayoutsMasksAndScale) {
  for (int depth : {192, 256}) {
    SCOPED_TRACE(depth);
    attention_case(depth, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                   17, 31, 0.125F, true, c::MaskKind::kNone);
    attention_case(depth, mx::float16, c::AttentionLayout::kBatchHeadSequenceDepth,
                   19, 13, -0.25F, false, c::MaskKind::kNone);
    attention_case(depth, mx::bfloat16, c::AttentionLayout::kBatchSequenceHeadDepth,
                   17, 31, 0, true, c::MaskKind::kNone);
    attention_case(depth, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                   17, 31, 0.125F, false, c::MaskKind::kAdditive);
    attention_case(depth, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                   17, 31, 0.125F, false, c::MaskKind::kBoolean);
  }
}

TEST(AppleAttention, UnsupportedFusedDomainsKeepTheirExistingFallback) {
  attention_case(40, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 17, 31, 0.125F, true, c::MaskKind::kNone);
  attention_case(192, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                 1, 31, 0.125F, true, c::MaskKind::kNone);
  attention_case(256, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 17, 13, 0.125F, true, c::MaskKind::kNone);
}

TEST(AppleAttention, NarrowBooleanMasksRetainFullyMaskedRowSemantics) {
  constexpr int queries = 1025, keys = 769, heads = 4, kv_heads = 2;
  std::vector<std::uint8_t> mask(queries * keys);
  for (int t = 0; t < queries; ++t)
    for (int s = 0; s < keys; ++s)
      mask[t * keys + s] = t % 64 != 0 && s % 3 != 0;
  for (int depth : {192, 256})
    for (const auto type : {mx::float16, mx::bfloat16}) {
      SCOPED_TRACE(::testing::Message() << depth << " " << type);
      const std::array inputs{
          make(std::vector<double>(heads * queries * depth),
               {1, heads, queries, depth}, type),
          make(std::vector<double>(kv_heads * keys * depth),
               {1, kv_heads, keys, depth}, type),
          make(std::vector<double>(kv_heads * keys * depth, 1.25),
               {1, kv_heads, keys, depth}, type),
          ready(mx::astype(mx::array(mask.data(), {queries, keys}, mx::uint8),
                           mx::bool_, stream()))};
      auto context = a::Context::create().value();
      auto output = execute(context, c::ScaledDotProductAttention{
          .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
          .mask = c::MaskKind::kBoolean, .scale = 0.125F}, inputs);
      ASSERT_TRUE(output) << output.error().message();
      expect(*output, std::vector<double>(heads * queries * depth, 1.25));
    }
}

TEST(AppleAttention, MathQueryBlocksKeepMasksLayoutsAndGlobalCausalOffset) {
  attention_case(192, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                 1025, 1281, 0.125F, true, c::MaskKind::kNone);
  attention_case(256, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 193, 769, 0, false, c::MaskKind::kBoolean);
  attention_case(192, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                 193, 769, -0.25F, false, c::MaskKind::kAdditive);
  // The first 256 global rows are fully masked, across many block boundaries.
  attention_case(40, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 1025, 769, -0.25F, true, c::MaskKind::kNone);
}

TEST(AppleAttention, LongCausalD256RetainsFp32Reference) {
  attention_case(256, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 1025, 1281, 0.125F, true, c::MaskKind::kNone);
}

TEST(AppleAttention, MathQueryBlocksRetainBroadcastMasksAndReboundOwners) {
  constexpr int batches = 2, heads = 2, queries = 1025, keys = 769, depth = 192;
  const auto q = make(std::vector<double>(batches * heads * queries * depth),
                       {batches, heads, queries, depth}, mx::float32);
  const auto k = make(std::vector<double>(batches * keys * depth),
                       {batches, 1, keys, depth}, mx::float32);
  std::vector<double> v(batches * keys * depth);
  for (int b = 0; b < batches; ++b)
    for (int s = 0; s < keys; ++s)
      std::fill_n(v.begin() + (b * keys + s) * depth, depth,
                   double(2 * (s % 2)) + double(b) / 4);
  const auto value = make(v, {batches, 1, keys, depth}, mx::float32);
  std::vector<std::uint8_t> bits(heads * keys);
  for (int s = 0; s < keys; ++s)
    bits[keys + s] = s % 2 == 0;
  const auto boolean_mask = ready(mx::astype(
      mx::array(bits.data(), {1, heads, 1, keys}, mx::uint8), mx::bool_, stream()));
  auto context = a::Context::create().value();
  const std::array inputs{q, k, value, boolean_mask};
  auto program = context.prepare(c::Operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
      .mask = c::MaskKind::kBoolean, .scale = -0.25F}), inputs).value();
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  auto expected = [&](bool all_visible) {
    std::vector<double> result(batches * heads * queries * depth);
    for (int b = 0; b < batches; ++b)
      for (int h = 0; h < heads; ++h) {
        const double mean = (h == 0 || all_visible ? double(keys - 1) / keys : 0)
            + double(b) / 4;
        std::fill_n(result.begin() + (b * heads + h) * queries * depth,
                     queries * depth, mean);
      }
    return result;
  };
  {
    // bind retains the real arrays independently for each lazy invocation.
    const std::array alternate{q, k, value,
        ready(mx::ones({1, heads, 1, keys}, mx::bool_, stream()))};
    ASSERT_TRUE(first.bind(inputs).ok());
    ASSERT_TRUE(second.bind(alternate).ok());
    auto first_work = first.submit().value();
    auto second_work = second.submit().value();
    auto first_result = first_work.wait().value();
    auto second_result = second_work.wait().value();
    expect(first_result.get()->arrays()[0], expected(false));
    expect(second_result.get()->arrays()[0], expected(true));
  }
  ASSERT_TRUE(first.bind(inputs).ok());
  auto rebound_work = first.submit().value();
  auto rebound_result = rebound_work.wait().value();
  expect(rebound_result.get()->arrays()[0], expected(false));

  // Unlike boolean masking, an all -Inf additive row retains native NaNs.
  std::vector<double> additive(heads * keys, -std::numeric_limits<double>::infinity());
  for (int s = 0; s < keys; s += 2)
    additive[keys + s] = 0;
  const std::array additive_inputs{q, k, value,
      make(additive, {1, heads, 1, keys}, mx::float32)};
  auto output = execute(context, c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
      .mask = c::MaskKind::kAdditive, .scale = 0}, additive_inputs);
  ASSERT_TRUE(output) << output.error().message();
  auto additive_expected = expected(false);
  for (int b = 0; b < batches; ++b)
    std::fill_n(additive_expected.begin() + b * heads * queries * depth,
                 queries * depth, std::numeric_limits<double>::quiet_NaN());
  expect(*output, additive_expected);

  auto combined_mask = context.prepare(c::Operation(c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
      .mask = c::MaskKind::kBoolean, .causal = true}), inputs);
  ASSERT_FALSE(combined_mask);
  EXPECT_EQ(combined_mask.error().code(), hyle::StatusCode::Unimplemented);
}

TEST(AppleAttention, FusedFloat32KeepsGqaLayoutsAndTailReference) {
  attention_case(256, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                 17, 31, 0.125F, true, c::MaskKind::kNone);
  attention_case(256, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 33, 65, -0.25F, true, c::MaskKind::kNone);
}

TEST(AppleAttention, FusedFloat32KeepsWeakProbabilities) {
  constexpr int queries = 17, keys = 257, heads = 4, kv_heads = 2, depth = 256;
  std::vector<double> q(heads * queries * depth), k(kv_heads * keys * depth),
      v(kv_heads * keys * depth, 60000);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t)
      q[(h * queries + t) * depth] = 1;
  for (int h = 0; h < kv_heads; ++h)
    for (int s = 0; s < keys; ++s) {
      k[(h * keys + s) * depth] = s == 0 ? 0 : -256;
      if (s == 0)
        std::fill_n(v.begin() + h * keys * depth, depth, 0);
    }
  const std::array inputs{make(q, {1, heads, queries, depth}, mx::float32),
      make(k, {1, kv_heads, keys, depth}, mx::float32),
      make(v, {1, kv_heads, keys, depth}, mx::float32)};
  auto context = a::Context::create().value();
  auto output = execute(context, c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
      .scale = 0.0625F, .causal = true}, inputs);
  ASSERT_TRUE(output) << output.error().message();
  std::vector<double> expected(heads * queries * depth);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t) {
      const double weak = double(keys - queries + t) * std::exp(-16.0);
      std::fill_n(expected.begin() + (h * queries + t) * depth, depth,
                   60000 * weak / (1 + weak));
    }
  expect(*output, expected);
}

TEST(AppleAttention, D192CausalGqaKeepsBhSdTailReference) {
  attention_case(192, mx::float32, c::AttentionLayout::kBatchHeadSequenceDepth,
                 17, 31, 0.125F, true, c::MaskKind::kNone);
}

TEST(AppleAttention, D192NoncausalGqaKeepsBsHdNegativeScaleReference) {
  attention_case(192, mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                 33, 17, -0.25F, false, c::MaskKind::kNone);
}

TEST(AppleAttention, D192Float32KeepsWeakProbabilities) {
  constexpr int queries = 17, keys = 257, heads = 4, kv_heads = 2, depth = 192;
  std::vector<double> q(heads * queries * depth), k(kv_heads * keys * depth),
      v(kv_heads * keys * depth, 60000);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t)
      q[(h * queries + t) * depth] = 1;
  for (int h = 0; h < kv_heads; ++h)
    for (int s = 0; s < keys; ++s) {
      k[(h * keys + s) * depth] = s == 0 ? 0 : -256;
      if (s == 0)
        std::fill_n(v.begin() + h * keys * depth, depth, 0);
    }
  const std::array inputs{make(q, {1, heads, queries, depth}, mx::float32),
      make(k, {1, kv_heads, keys, depth}, mx::float32),
      make(v, {1, kv_heads, keys, depth}, mx::float32)};
  auto context = a::Context::create().value();
  auto output = execute(context, c::ScaledDotProductAttention{
      .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
      .scale = 0.0625F, .causal = true}, inputs);
  ASSERT_TRUE(output) << output.error().message();
  std::vector<double> expected(heads * queries * depth);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t) {
      const double weak = double(keys - queries + t) * std::exp(-16.0);
      std::fill_n(expected.begin() + (h * queries + t) * depth, depth,
                   60000 * weak / (1 + weak));
    }
  expect(*output, expected);
}

TEST(AppleAttention, HalfPrefillRetainsWeakProbabilitiesBeforeLargeValues) {
  constexpr int queries = 17, keys = 257, heads = 4, kv_heads = 2;
  for (int depth : {192, 256}) {
    SCOPED_TRACE(depth);
    std::vector<double> q(heads * queries * depth), k(kv_heads * keys * depth),
        v(kv_heads * keys * depth, 60000);
    for (int h = 0; h < heads; ++h)
      for (int t = 0; t < queries; ++t)
        q[(h * queries + t) * depth] = 1;
    for (int h = 0; h < kv_heads; ++h)
      for (int s = 0; s < keys; ++s) {
        k[(h * keys + s) * depth] = s == 0 ? 0 : -256;
        if (s == 0)
          std::fill_n(v.begin() + h * keys * depth, depth, 0);
      }
    const std::array inputs{make(q, {1, heads, queries, depth}, mx::float16),
        make(k, {1, kv_heads, keys, depth}, mx::float16),
        make(v, {1, kv_heads, keys, depth}, mx::float16)};
    auto context = a::Context::create().value();
    auto result = execute(context, c::ScaledDotProductAttention{
        .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
        .scale = 0.0625F, .causal = true}, inputs);
    ASSERT_TRUE(result) << result.error().message();
    const auto observed = values(*result);
    for (int h = 0; h < heads; ++h)
      for (int t = 0; t < queries; ++t) {
        const double weak = double(keys - queries + t) * std::exp(-16.0);
        const double expected = 60000 * weak / (1 + weak);
        for (int d = 0; d < depth; ++d)
          EXPECT_NEAR(observed[(h * queries + t) * depth + d], expected, 2e-3);
      }
  }
}
TEST(AppleAttention, ExtremeFiniteFloat32ScaleRetainsZeroScoreReference) {
  constexpr int heads = 2, kv_heads = 1, queries = 9, keys = 17;
  auto context = a::Context::create().value();
  for (int depth : {192, 256}) {
    const bool causal = depth == 256;
    std::vector<double> v(kv_heads * keys * depth);
    for (int s = 0; s < keys; ++s)
      for (int d = 0; d < depth; ++d)
        v[s * depth + d] = double((s * 7 + d * 3) % 19 - 9) / 8;
    const std::array inputs{
        make(std::vector<double>(heads * queries * depth),
             {1, heads, queries, depth}, mx::float32),
        make(std::vector<double>(kv_heads * keys * depth),
             {1, kv_heads, keys, depth}, mx::float32),
        make(v, {1, kv_heads, keys, depth}, mx::float32)};

    // Exact zero Q/K scores give a uniform distribution over visible keys.
    // V's binary fractions are exactly representable as F32. Compute each
    // nontrivial channel/prefix mean in FP64, independently of native SDPA.
    std::vector<double> expected(heads * queries * depth);
    for (int h = 0; h < heads; ++h)
      for (int t = 0; t < queries; ++t) {
        const int visible = causal ? keys - queries + t + 1 : keys;
        for (int d = 0; d < depth; ++d) {
          double sum = 0;
          for (int s = 0; s < visible; ++s)
            sum += v[s * depth + d];
          expected[(h * queries + t) * depth + d] = sum / visible;
        }
      }
    for (float scale : {std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max()}) {
      SCOPED_TRACE(::testing::Message() << "depth=" << depth
          << " causal=" << causal << " scale=" << scale);
      auto output = execute(context, c::ScaledDotProductAttention{
          .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
          .scale = scale, .causal = causal}, inputs);
      ASSERT_TRUE(output) << output.error().message();
      ASSERT_EQ(output->dtype(), mx::float32);
      // Reuse the existing F32 rel6e-6 + abs2e-6 comparison unchanged.
      expect(*output, expected);
    }
  }
}

TEST(AppleAttention, CenteredShortQueryMathKeepsMasksTailAndReboundOwners) {
  namespace sq = ordinary_short_query_centered;
  for (const auto layout : {c::AttentionLayout::kBatchHeadSequenceDepth,
                            c::AttentionLayout::kBatchSequenceHeadDepth}) {
    SCOPED_TRACE(static_cast<int>(layout));
    const auto expected_a = sq::reference(layout, 0);
    const auto expected_b = sq::reference(layout, 1);
    auto inputs_a = sq::ports(layout, 0), inputs_b = sq::ports(layout, 1);
    const c::Operation operation(c::ScaledDotProductAttention{
        .layout = layout, .mask = c::MaskKind::kBoolean, .scale = sq::scale});
    auto context = a::Context::create().value();
    // Exercise both typed prepare entries with this selected >64MiB shape.
    auto program = context.prepare(operation, inputs_a).value();
    std::array<a::InputSpec, 4> specs;
    for (std::size_t i = 0; i < specs.size(); ++i)
      specs[i] = {inputs_a[i].shape(), inputs_a[i].dtype()};
    auto from_specs = context.prepare(operation, specs).value();
    auto first = program.create_invocation().value();
    auto second = from_specs.create_invocation().value();
    ASSERT_TRUE(first.bind(inputs_a).ok());
    ASSERT_TRUE(second.bind(inputs_b).ok());
    auto work_a = first.submit().value(), work_b = second.submit().value();
    auto result_a = work_a.wait().value(), result_b = work_b.wait().value();
    auto old_output = result_a.get()->arrays()[0];
    expect(old_output, expected_a);
    expect(result_b.get()->arrays()[0], expected_b);
    ASSERT_FALSE(::testing::Test::HasFailure());
    const auto old_bits = bits(old_output);
    // Keep only an independent native copy for this slot before four-port rebind.
    // The other invocation's result can remain held without blocking this slot.
    result_a = {};
    work_a = {};
    ASSERT_TRUE(first.bind(inputs_b).ok());
    auto rebound = first.submit().value();
    inputs_a = inputs_b;
    first = {};
    second = {};
    program = {};
    from_specs = {};
    context = {};
    auto final_result = rebound.wait().value();
    expect(final_result.get()->arrays()[0], expected_b);
    EXPECT_EQ(bits(old_output), old_bits);
  }
}

} // namespace
