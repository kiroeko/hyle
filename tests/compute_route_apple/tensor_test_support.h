#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/apple.h"

namespace apple_tensor_test {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
using F16 = hyle::simd::float16_t;
using B16 = hyle::simd::bfloat16_t;
inline mx::Stream stream() { return mx::default_stream(mx::Device::gpu); }
inline mx::array ready(mx::array value) {
  mx::async_eval({value});
  mx::synchronize(stream());
  value.wait();
  return value;
}
inline c::DataType logical(mx::Dtype type) {
  if (type == mx::float16)
    return c::DataType::kFloat16;
  if (type == mx::bfloat16)
    return c::DataType::kBFloat16;
  if (type == mx::int32)
    return c::DataType::kInt32;
  if (type == mx::int64)
    return c::DataType::kInt64;
  if (type == mx::bool_)
    return c::DataType::kBool;
  return c::DataType::kFloat32;
}
inline double rounded(double value, mx::Dtype type) {
  if (type == mx::float16)
    return half_test::decode(half_test::encode<F16>(value));
  if (type == mx::bfloat16)
    return half_test::decode(half_test::encode<B16>(value));
  return static_cast<float>(value);
}
inline mx::array make(std::span<const double> values, mx::Shape shape,
                      mx::Dtype type) {
  if (type == mx::float32) {
    std::vector<float> data(values.begin(), values.end());
    return mx::array(data.data(), std::move(shape), mx::float32);
  }
  std::vector<std::uint16_t> data(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    data[i] = type == mx::float16 ? half_test::encode<F16>(values[i]).bits
                                  : half_test::encode<B16>(values[i]).bits;
  return ready(mx::view(mx::array(data.data(), std::move(shape), mx::uint16),
                        type, stream()));
}
inline std::vector<std::uint64_t> bits(const mx::array &array) {
  std::vector<std::uint64_t> result(array.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    std::size_t remaining = index, offset = 0;
    for (std::size_t axis = array.ndim(); axis-- > 0;) {
      offset += (remaining % array.shape()[axis]) * array.strides()[axis];
      remaining /= array.shape()[axis];
    }
    std::memcpy(&result[index], array.data<char>() + offset * array.itemsize(),
                array.itemsize());
  }
  return result;
}
inline std::vector<double> values(const mx::array &array) {
  const auto raw = bits(array);
  std::vector<double> result(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (array.dtype() == mx::float32)
      result[i] = std::bit_cast<float>(static_cast<std::uint32_t>(raw[i]));
    else if (array.dtype() == mx::float16)
      result[i] =
          half_test::decode_bits<F16>(static_cast<std::uint16_t>(raw[i]));
    else if (array.dtype() == mx::bfloat16)
      result[i] =
          half_test::decode_bits<B16>(static_cast<std::uint16_t>(raw[i]));
    else if (array.dtype() == mx::int32)
      result[i] =
          std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(raw[i]));
    else if (array.dtype() == mx::int64)
      result[i] = static_cast<double>(std::bit_cast<std::int64_t>(raw[i]));
    else
      result[i] = raw[i] != 0;
  }
  return result;
}
inline hyle::Result<mx::array> execute(a::Context &context,
                                       c::Operation operation,
                                       std::span<const mx::array> inputs) {
  HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(operation, inputs));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  return result.get()->arrays()[0];
}
template <class Descriptor>
inline hyle::Result<mx::array> execute(a::Context &context,
                                       Descriptor operation,
                                       std::span<const mx::array> inputs) {
  return execute(context, c::Operation(std::move(operation)), inputs);
}
inline void expect(const mx::array &actual, std::span<const double> expected) {
  const auto observed = values(actual);
  ASSERT_EQ(observed.size(), expected.size());
  const auto type = actual.dtype();
  const double relative = type == mx::bfloat16  ? 0.008
                          : type == mx::float16 ? 0.001
                                                : 6e-6;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto value = rounded(expected[i], type);
    if (std::isnan(value))
      EXPECT_TRUE(std::isnan(observed[i])) << i;
    else if (std::isinf(value))
      EXPECT_EQ(observed[i], value) << i;
    else
      EXPECT_NEAR(observed[i], value,
                  relative * std::abs(value) +
                      (type == mx::float32 ? 2e-6 : 2e-7))
          << i;
  }
}
inline double activation(c::UnaryKind kind, double x) {
  switch (kind) {
  case c::UnaryKind::kAbs:
    return std::abs(x);
  case c::UnaryKind::kNeg:
    return -x;
  case c::UnaryKind::kExp:
    return std::exp(x);
  case c::UnaryKind::kLog:
    return std::log(x);
  case c::UnaryKind::kSqrt:
    return std::sqrt(x);
  case c::UnaryKind::kRsqrt:
    return 1 / std::sqrt(x);
  case c::UnaryKind::kTanh:
    return std::tanh(x);
  case c::UnaryKind::kSigmoid:
    return 1 / (1 + std::exp(-x));
  case c::UnaryKind::kRelu:
    return std::isnan(x) ? x : (x > 0 ? x : 0);
  case c::UnaryKind::kSoftplus:
    return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
  case c::UnaryKind::kSilu:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : x / (1 + std::exp(-x));
  case c::UnaryKind::kGelu:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
  case c::UnaryKind::kGeluTanh:
    return x == -std::numeric_limits<double>::infinity()
               ? 0
               : 0.5 * x *
                     (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * x *
                                    (1 + 0.044715 * x * x)));
  }
  return std::numeric_limits<double>::quiet_NaN();
}
inline std::vector<double> pattern(std::size_t count, int phase = 0) {
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = (static_cast<int>((i * 7 + phase * 3) % 29) - 14) * 0.17;
  return result;
}

// Additional centered functional input for wiring/ownership regressions.
// The original offset stress and its failed FP64 budget remain unchanged in
// the frozen candidate. This fixture neither replaces it nor a performance point.
namespace ordinary_short_query_centered {
inline constexpr int queries = 513, keys = 8193, heads = 8, kv_heads = 2, depth = 256;
inline constexpr float scale = 0.0625F;
static_assert(1ULL * heads * queries * keys * sizeof(float) >= 64ULL * 1024 * 1024);

inline bool visible(int t, int s, int phase) {
  return t != 0 && t != 127 && t != 128 && t != queries - 1 &&
      s <= keys - queries + t && (phase == 0 || s % 5 != 0);
}
inline double q0(int t, int phase) { return double((t + phase) % 7 - 3) / 8; }
inline double q1(int h, int phase) { return double((h + phase) % 5 - 2) / 16; }
inline double k0(int s, int phase) { return double((s + phase * 3) % 11 - 5) / 16; }
inline double k1(int s, int phase) { return double((s + phase) % 3 - 1) / 8; }
inline double a(int s, int phase) {
  return 0.25 * ((s + phase) % 2 == 0 ? 1.25 : -1.25) +
      2 * k0(s, phase) + k1(s, phase);
}
inline double channel(int h, int d, int phase) {
  return double((d + phase) % 17 - 8) / 64 + double(h + phase) / 32;
}

inline std::array<mx::array, 4> ports(c::AttentionLayout layout, int phase) {
  std::vector<float> q(heads * queries * depth), k(kv_heads * keys * depth),
      v(kv_heads * keys * depth);
  std::vector<std::uint8_t> mask(queries * keys);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t) {
      q[(h * queries + t) * depth] = static_cast<float>(q0(t, phase));
      q[(h * queries + t) * depth + 1] = static_cast<float>(q1(h, phase));
    }
  for (int h = 0; h < kv_heads; ++h)
    for (int s = 0; s < keys; ++s) {
      k[(h * keys + s) * depth] = static_cast<float>(k0(s, phase));
      k[(h * keys + s) * depth + 1] = static_cast<float>(k1(s, phase));
      for (int d = 0; d < depth; ++d)
        v[(h * keys + s) * depth + d] =
            static_cast<float>(a(s, phase) * (1 + channel(h, d, phase)));
    }
  for (int t = 0; t < queries; ++t)
    for (int s = 0; s < keys; ++s)
      mask[t * keys + s] = visible(t, s, phase);
  std::array result{
      mx::array(q.data(), {1, heads, queries, depth}, mx::float32),
      mx::array(k.data(), {1, kv_heads, keys, depth}, mx::float32),
      mx::array(v.data(), {1, kv_heads, keys, depth}, mx::float32),
      mx::astype(mx::array(mask.data(), {queries, keys}, mx::uint8),
                 mx::bool_, stream())};
  for (std::size_t i = 0; i < 3; ++i)
    if (layout == c::AttentionLayout::kBatchSequenceHeadDepth)
      result[i] = mx::transpose(result[i], {0, 2, 1, 3}, stream());
  for (auto &input : result) input = ready(std::move(input));
  return result;
}

inline std::vector<double> reference(c::AttentionLayout layout, int phase) {
  std::vector<double> result(heads * queries * depth);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < queries; ++t) {
      // Independent FP64 two-channel QK and separable V oracle. All inputs
      // are exactly representable dyadic F32; no native attention is called.
      const bool all_masked = t == 0 || t == 127 || t == 128 || t == queries - 1;
      double mass = 0, weighted = 0;
      for (int s = 0; s < keys; ++s) {
        const double weight = all_masked ? 1.0 : visible(t, s, phase)
            ? std::exp(double(scale) *
                (q0(t, phase) * k0(s, phase) + q1(h, phase) * k1(s, phase)))
            : 0.0;
        mass += weight;
        weighted += weight * a(s, phase);
      }
      for (int d = 0; d < depth; ++d) {
        const auto index = layout == c::AttentionLayout::kBatchSequenceHeadDepth
            ? (t * heads + h) * depth + d : (h * queries + t) * depth + d;
        result[index] = (weighted / mass) * (1 + channel(h / (heads / kv_heads), d, phase));
      }
    }
  return result;
}

// A few additional wiring cases, not a precision-domain sweep. Keep the
// original centered fixture above unchanged, including its two-phase data.
struct MathCase {
  const char *name;
  int queries, keys, query_depth, value_depth;
  mx::Dtype type;
  c::AttentionLayout layout;
  c::MaskKind mask;
  bool causal = false;
};
inline MathCase wide_bool_case() {
  return {"wide Bool head-one broadcast", 257, 8193, 512, 17, mx::float32,
          c::AttentionLayout::kBatchHeadSequenceDepth, c::MaskKind::kBoolean};
}
inline std::array<MathCase, 4> extended_cases() {
  return {wide_bool_case(),
          MathCase{"F16 unequal depths without mask", 513, 8193, 512, 17,
                   mx::float16, c::AttentionLayout::kBatchSequenceHeadDepth,
                   c::MaskKind::kNone},
          MathCase{"BF16 per-query additive broadcast", 513, 8193, 40, 33,
                   mx::bfloat16, c::AttentionLayout::kBatchHeadSequenceDepth,
                   c::MaskKind::kAdditive},
          MathCase{"long causal Q greater than KV", 2049, 1025, 64, 64,
                   mx::float32, c::AttentionLayout::kBatchSequenceHeadDepth,
                   c::MaskKind::kNone, true}};
}
inline bool empty_row(const MathCase &test, int t) {
  return t == 0 || t == 127 || t == 128 || t == test.queries - 1;
}
inline bool visible(const MathCase &test, int t, int s, int phase) {
  return !empty_row(test, t) && s <= test.keys - test.queries + t &&
      (phase == 0 || s % 5 != 0);
}
inline double query_bias(const MathCase &test, int t, int phase) {
  return empty_row(test, t) ? -std::numeric_limits<double>::infinity()
                           : double((t + phase) % 3 - 1) / 4096;
}
inline double value_at(const MathCase &test, int h, int s, int d, int phase) {
  // Only the new low-dtype cases carry this fixed DC component. Their oracle
  // rounds the actual input V, rather than treating BF16 values as exact reals.
  const double offset = test.type == mx::float32 ? 0 : 0.5;
  return offset + a(s, phase) * (1 + channel(h, d, phase));
}
inline std::vector<mx::array> ports(const MathCase &test, int phase) {
  std::vector<double> q(heads * test.queries * test.query_depth),
      k(kv_heads * test.keys * test.query_depth),
      v(kv_heads * test.keys * test.value_depth);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < test.queries; ++t) {
      q[(h * test.queries + t) * test.query_depth] = q0(t, phase);
      q[(h * test.queries + t) * test.query_depth + 1] = q1(h, phase);
    }
  for (int h = 0; h < kv_heads; ++h)
    for (int s = 0; s < test.keys; ++s) {
      k[(h * test.keys + s) * test.query_depth] = k0(s, phase);
      k[(h * test.keys + s) * test.query_depth + 1] = k1(s, phase);
      for (int d = 0; d < test.value_depth; ++d)
        v[(h * test.keys + s) * test.value_depth + d] =
            value_at(test, h, s, d, phase);
    }
  std::vector<mx::array> result{
      make(q, {1, heads, test.queries, test.query_depth}, test.type),
      make(k, {1, kv_heads, test.keys, test.query_depth}, test.type),
      make(v, {1, kv_heads, test.keys, test.value_depth}, test.type)};
  if (test.mask == c::MaskKind::kBoolean) {
    std::vector<std::uint8_t> mask(test.queries * test.keys);
    for (int t = 0; t < test.queries; ++t)
      for (int s = 0; s < test.keys; ++s)
        mask[t * test.keys + s] = visible(test, t, s, phase);
    result.push_back(mx::astype(
        mx::array(mask.data(), {1, 1, test.queries, test.keys}, mx::uint8),
        mx::bool_, stream()));
  } else if (test.mask == c::MaskKind::kAdditive) {
    std::vector<double> mask(test.queries);
    for (int t = 0; t < test.queries; ++t)
      mask[t] = query_bias(test, t, phase);
    // Keep a real zero KV stride, including across the Q127/Q128 block seam.
    result.push_back(mx::broadcast_to(
        make(mask, {1, 1, test.queries, 1}, test.type),
        {1, 1, test.queries, test.keys}, stream()));
  }
  for (std::size_t i = 0; i < 3; ++i)
    if (test.layout == c::AttentionLayout::kBatchSequenceHeadDepth)
      result[i] = mx::transpose(result[i], {0, 2, 1, 3}, stream());
  for (auto &input : result) input = ready(std::move(input));
  return result;
}
inline std::vector<double> reference(const MathCase &test, int phase) {
  std::vector<double> result(heads * test.queries * test.value_depth);
  std::vector<double> key0(test.keys), key1(test.keys);
  for (int s = 0; s < test.keys; ++s) {
    key0[s] = rounded(k0(s, phase), test.type);
    key1[s] = rounded(k1(s, phase), test.type);
  }
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < test.queries; ++t) {
      const bool nan_row = test.mask == c::MaskKind::kAdditive && empty_row(test, t);
      // Boolean/causal all-false rows retain finite-min => uniform semantics;
      // an explicitly additive -Inf row instead preserves the NaN result.
      const bool uniform = (test.mask == c::MaskKind::kBoolean && empty_row(test, t)) ||
          (test.causal && test.keys - test.queries + t < 0);
      double mass = 0, weighted = 0;
      std::array<double, 66> residue_mass{}; // a(s, phase) has period lcm(2,11,3).
      const double query0 = rounded(q0(t, phase), test.type);
      const double query1 = rounded(q1(h, phase), test.type);
      const double bias = test.mask == c::MaskKind::kAdditive
          ? rounded(query_bias(test, t, phase), test.type) : 0;
      if (!nan_row)
        for (int s = 0; s < test.keys; ++s) {
          const bool allowed = test.mask == c::MaskKind::kBoolean
              ? visible(test, t, s, phase)
              : !test.causal || s <= test.keys - test.queries + t;
          const double score = double(scale) *
              (query0 * key0[s] + query1 * key1[s]) + bias;
          const double weight = uniform ? 1 : allowed ? std::exp(score) : 0;
          mass += weight;
          weighted += weight * a(s, phase);
          residue_mass[s % residue_mass.size()] += weight;
        }
      for (int d = 0; d < test.value_depth; ++d) {
        double value = std::numeric_limits<double>::quiet_NaN();
        if (!nan_row) {
          const int kv_h = h / (heads / kv_heads);
          if (test.type == mx::float32)
            value = (weighted / mass) * (1 + channel(kv_h, d, phase));
          else {
            // Independent FP64 weighting of actual F16/BF16 input values.
            // Period aggregation avoids a keys-by-output-depth oracle loop.
            double sum = 0;
            for (std::size_t r = 0; r < residue_mass.size(); ++r)
              sum += residue_mass[r] * rounded(
                  value_at(test, kv_h, static_cast<int>(r), d, phase), test.type);
            value = sum / mass;
          }
        }
        const auto index = test.layout == c::AttentionLayout::kBatchSequenceHeadDepth
            ? (t * heads + h) * test.value_depth + d
            : (h * test.queries + t) * test.value_depth + d;
        result[index] = value;
      }
    }
  return result;
}
} // namespace ordinary_short_query_centered
} // namespace apple_tensor_test
