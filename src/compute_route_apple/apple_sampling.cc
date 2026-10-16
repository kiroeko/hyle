#include "apple_tensor.h"
#include "apple_checks.h"
#include "apple_sampling_math.h"

#include <climits>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "hyle/compute/sampling_ops.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
mx::array random_bits(const sampling_ops::RandomBits &operation,
                        const mx::array &identities, mx::Stream stream) {
  const auto rows = identities.shape(0);
  const auto width = static_cast<int>(operation.word_count);
  if (rows == 0 || width == 0)
    return mx::zeros({rows, width}, mx::uint32, stream);
  static const auto kernel = mx::fast::metal_kernel("hyle_counter_random_bits",
      {"identities"}, {"output"}, R"METAL(
    uint blocks = (uint(WORDS) + 3u) / 4u;
    uint index = thread_position_in_grid.x;
    if (index >= uint(identities_shape[0]) * blocks) return;
    uint row = index / blocks, column = index % blocks;
    uint4 words = hyle_philox(uint4(column, 0, identities[row*4+2], identities[row*4+3]),
                             uint2(identities[row*4], identities[row*4+1]));
    for (uint lane = 0; lane < 4 && column*4+lane < uint(WORDS); ++lane)
      output[row*uint(WORDS)+column*4+lane] = words[lane];
  )METAL", kSamplingMetalHeader, true, false, mx::CompileOptions{});
  const auto work = rows * ((std::uint64_t(width) + 3) / 4);
  return kernel({identities}, {{rows, width}}, {mx::uint32},
      {static_cast<int>((work + 127) / 128 * 128), 1, 1}, {128, 1, 1},
      {{"WORDS", width}}, std::nullopt, false, stream)[0];
}
mx::array sampled_scores(const mx::array &logits, const mx::array &identities,
                           const mx::array &maxima, mx::Stream stream) {
  static const auto kernel = mx::fast::metal_kernel("hyle_categorical_scores",
      {"logits", "identities", "maxima"}, {"output"}, R"METAL(
    uint index = thread_position_in_grid.x;
    uint width = uint(logits_shape[1]);
    if (index >= uint(logits_shape[0]) * width) return;
    uint row = index / width, column = index % width;
    uint4 words = hyle_philox(uint4(column/2, 0, identities[row*4+2], identities[row*4+3]),
                             uint2(identities[row*4], identities[row*4+1]));
    uint lane = (column % 2) * 2;
    float value = float(logits[index]);
    if (isnan(value) || value == as_type<float>(0x7f800000u))
      output[index] = as_type<float>(0xff800000u);
    else
      output[index] = (value - maxima[row]) + hyle_gumbel(words[lane], words[lane+1]);
  )METAL", kSamplingMetalHeader, true, false, mx::CompileOptions{});
  const auto work = logits.size();
  return kernel({logits, identities, maxima}, {logits.shape()}, {mx::float32},
      {static_cast<int>((work + 127) / 128 * 128), 1, 1}, {128, 1, 1},
      {}, std::nullopt, false, stream)[0];
}
} // namespace

Result<std::vector<mx::array>> lower_sampling(const Operation &operation,
    std::span<const mx::array> inputs, mx::Stream stream) {
  const auto bits = operation.custom_as<sampling_ops::RandomBits>();
  const auto categorical = operation.custom_as<sampling_ops::Categorical>();
  if (!bits && !categorical)
    return std::unexpected(Status::Unimplemented("Apple sampling schema is not bound"));
  for (const auto &input : inputs)
    if (input.size() > INT32_MAX - 127)
      return std::unexpected(Status::OutOfRange("Apple sampling exceeds native flat index range"));
  if (bits) {
    const auto elements = std::uint64_t(inputs[0].shape(0)) * bits->word_count;
    if (elements > INT32_MAX - 127)
      return std::unexpected(Status::OutOfRange("Apple random output exceeds native flat index range"));
    return std::vector{random_bits(*bits, inputs[0], stream)};
  }
  const auto &logits = inputs[0];
  const auto rows = logits.shape(0);
  if (rows == 0) {
    std::vector output{mx::zeros({0}, mx::int32, stream)};
    if (categorical->return_log_probability)
      output.push_back(mx::zeros({0}, mx::float32, stream));
    return output;
  }
  const auto infinity = mx::array(std::numeric_limits<float>::infinity());
  const auto bad = mx::logical_or(mx::isnan(logits, stream),
                                    mx::equal(logits, infinity, stream), stream);
  auto maxima = mx::astype(mx::max(logits, -1, true, stream), mx::float32, stream);
  const auto finite_max = mx::isfinite(maxima, stream);
  const auto invalid = mx::logical_or(mx::any(bad, stream),
      mx::any(mx::logical_not(finite_max, stream), stream), stream);
  auto status = mx::where(invalid, mx::array(kInvalidNumericInput), mx::array(0), stream);
  maxima = mx::where(finite_max, maxima, mx::array(0.F), stream);
  const auto scores = sampled_scores(logits, inputs[1], maxima, stream);
  auto indices = mx::astype(mx::argmax(scores, -1, false, stream), mx::int32, stream);
  std::vector output{indices};
  if (categorical->return_log_probability) {
    const auto shifted = mx::subtract(mx::astype(logits, mx::float32, stream), maxima, stream);
    auto selected = mx::take_along_axis(shifted, mx::expand_dims(indices, -1, stream), -1, stream);
    auto probability = mx::subtract(selected, mx::logsumexp(shifted, -1, true, stream), stream);
    output.push_back(mx::reshape(probability, {rows}, stream));
  }
  for (auto &value : output) {
    HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(value), status, stream));
    value = std::move(checked);
  }
  return output;
}
} // namespace hyle::compute::apple::detail
