#pragma once

// Diagnostic replay of already observed, successful original Model outputs.
// It neither changes the Model graph nor supplies an acceptance exception.
#if defined(HYLE_COMPUTE_VULKAN)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <vector>
#include "../../src/compute_route_vulkan/vulkan_composition_access.h"
#include "../../src/compute_route_vulkan/vulkan_device_lengths.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "hyle/simd/dtype.h"

namespace qwen_attention_reader_pair_case {
namespace q = qwen35;
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
using Bytes = std::vector<std::byte>;
using BF16 = hyle::simd::bfloat16_t;
using qwen_host_image_case::require;

inline std::vector<BF16> values(std::span<const std::byte> bytes) {
  require(bytes.size() % sizeof(BF16) == 0, "attention observation has partial BF16 elements");
  std::vector<BF16> result(bytes.size() / sizeof(BF16));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

inline void replay(std::string_view label, const q::ModelGeometry &geometry,
    std::uint32_t rows, std::uint32_t history, const c::TensorDesc &query_description,
    std::span<const std::byte> query_bytes, std::span<const std::byte> original_output,
    std::span<const std::byte> key_bytes, std::span<const std::byte> value_bytes,
    bool original_device_reader) {
  constexpr std::uint32_t capacity_rows = 32768, page_size = 16;
  const auto heads = static_cast<std::uint32_t>(geometry.query_heads),
      kv_heads = static_cast<std::uint32_t>(geometry.kv_heads),
      depth = static_cast<std::uint32_t>(geometry.key_depth),
      value_depth = static_cast<std::uint32_t>(geometry.value_depth);
  require(rows && history >= rows && history <= page_size && heads && kv_heads &&
      heads % kv_heads == 0 && depth && value_depth,
      "attention observation exceeds the fixed short-prefix diagnostic");
  require(query_description.data_type() == c::DataType::kBFloat16 &&
      query_description.is_dense() && query_bytes.size() == std::size_t(rows) * heads * depth * sizeof(BF16) &&
      original_output.size() == std::size_t(rows) * heads * value_depth * sizeof(BF16) &&
      key_bytes.size() == std::size_t(history) * kv_heads * depth * sizeof(BF16) &&
      value_bytes.size() == std::size_t(history) * kv_heads * value_depth * sizeof(BF16),
      "attention observation is not its exact live Q/K/V/output prefix");
  const float scale = 1.F / std::sqrt(static_cast<float>(depth));
  auto context = q::take(v::Context::create());
  const auto storage = q::take(c::CacheStorage::plain(c::DataType::kBFloat16));
  const c::BatchCapacity capacity{.query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16, .key_storage = storage, .value_storage = storage,
      .max_requests = 1, .max_query_rows = rows, .max_kv_rows = capacity_rows,
      .max_pages = capacity_rows / page_size, .max_page_refs = capacity_rows / page_size,
      .query_heads = heads, .kv_heads = kv_heads, .qk_head_dim = depth,
      .value_head_dim = value_depth, .page_size = page_size};
  const auto schema = q::take(c::BatchSchema::create(capacity));
  const std::array<std::int32_t, 2> queries{0, static_cast<std::int32_t>(rows)}, page_offsets{0, 1};
  const std::array<std::int32_t, 1> pages{0}, live{static_cast<std::int32_t>(history)}, tail{page_size};
  const auto host_layout = q::take(schema.version({queries, page_offsets, pages, live}));
  const auto envelope = q::take(schema.version({queries, page_offsets, pages, tail}));
  const auto recipe = q::take(c::NativePagedBatchProgram::prepare(schema,
      {.max_host_metadata_bytes = capacity.max_page_refs * sizeof(std::int32_t),
       .scale = scale, .causal = true, .window_left = -1}));
  const auto host_program = q::take(d::PagedBatchAccess::prepare_paged_graph(
      d::Access::runtime(context), recipe, host_layout, rows, {}));
  const auto device_program = q::take(d::prepare_device_lengths_paged_graph(
      d::Access::runtime(context), schema, rows, scale, true, -1));
  require(host_program.input_descriptions()[0] == query_description &&
      device_program.input_descriptions()[0] == query_description,
      "attention replay would reinterpret the original query descriptor");
  const auto query = values(query_bytes), original = values(original_output),
      live_keys = values(key_bytes), live_values = values(value_bytes);
  const auto nan = hyle::simd::f32_to_bf16(std::numeric_limits<float>::quiet_NaN());
  std::vector<BF16> keys(std::size_t(capacity_rows) * kv_heads * depth, nan),
      vals(std::size_t(capacity_rows) * kv_heads * value_depth, nan);
  std::copy(live_keys.begin(), live_keys.end(), keys.begin());
  std::copy(live_values.begin(), live_values.end(), vals.begin());
  const auto qt = q::take(context.upload(query_description, query_bytes));
  const auto kt = q::take(context.upload(host_program.input_descriptions()[1], std::as_bytes(std::span(keys))));
  const auto vt = q::take(context.upload(host_program.input_descriptions()[2], std::as_bytes(std::span(vals))));
  const auto host_indices = q::take(q::take(recipe.version(host_layout)).indices());
  std::vector<std::int32_t> indices(*host_program.input_descriptions()[3].element_count());
  q::check(q::take(host_indices.view()).read(std::as_writable_bytes(std::span(indices))));
  const auto it = q::take(context.upload(host_program.input_descriptions()[3], std::as_bytes(std::span(indices))));
  const std::array<std::int32_t, 1> qlen{static_cast<std::int32_t>(rows)};
  const auto qlt = q::take(context.upload(device_program.input_descriptions()[3], std::as_bytes(std::span(qlen))));
  const auto klt = q::take(context.upload(device_program.input_descriptions()[4], std::as_bytes(std::span(live))));
  std::vector<std::int32_t> metadata(*device_program.input_descriptions()[5].element_count());
  q::check(q::take(q::take(envelope.metadata()).view()).read(std::as_writable_bytes(std::span(metadata))));
  const auto mt = q::take(context.upload(device_program.input_descriptions()[5], std::as_bytes(std::span(metadata))));
  const auto execute = [&](const v::Program &program, std::span<const v::Tensor> inputs) {
    auto invocation = q::take(program.create_invocation());
    q::check(invocation.bind(inputs));
    auto submitted = q::take(invocation.submit());
    const auto completed = q::take(submitted.wait());
    std::vector<BF16> output(original.size());
    q::check(context.download(completed->tensors()[0], std::as_writable_bytes(std::span(output))));
    return output;
  };
  const auto host = execute(host_program, std::array{qt, kt, vt, it});
  const auto device = execute(device_program, std::array{qt, kt, vt, qlt, klt, mt});
  std::vector<double> oracle(original.size());
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t head = 0; head < heads; ++head) {
      const auto kh = head / (heads / kv_heads), end = history - rows + row + 1;
      std::array<double, page_size> weights{};
      double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
      for (std::uint32_t token = 0; token < end; ++token) {
        double dot = 0;
        for (std::uint32_t channel = 0; channel < depth; ++channel)
          dot += double(hyle::simd::bf16_to_f32(query[(row * heads + head) * depth + channel])) *
              double(hyle::simd::bf16_to_f32(live_keys[(token * kv_heads + kh) * depth + channel]));
        weights[token] = dot * double(scale); maximum = (std::max)(maximum, weights[token]);
      }
      for (std::uint32_t token = 0; token < end; ++token) {
        weights[token] = std::exp(weights[token] - maximum); denominator += weights[token];
      }
      for (std::uint32_t channel = 0; channel < value_depth; ++channel) {
        double sum = 0;
        for (std::uint32_t token = 0; token < end; ++token)
          sum += weights[token] * double(hyle::simd::bf16_to_f32(
              live_values[(token * kv_heads + kh) * value_depth + channel]));
        oracle[(row * heads + head) * value_depth + channel] = sum / denominator;
      }
    }
  for (const auto which : {0, 1, 2}) {
    const auto &actual = which == 0 ? host : which == 1 ? device : original;
    double raw_error = 0, rounded_error = 0;
    std::size_t wrong_rounded = 0, nonfinite = 0, different_model = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto x = double(hyle::simd::bf16_to_f32(actual[index]));
      const auto rounded = hyle::simd::f32_to_bf16(static_cast<float>(oracle[index]));
      nonfinite += !std::isfinite(x);
      raw_error = (std::max)(raw_error, std::isfinite(x) ? std::abs(x - oracle[index]) : INFINITY);
      rounded_error = (std::max)(rounded_error,
          std::isfinite(x) ? std::abs(x - double(hyle::simd::bf16_to_f32(rounded))) : INFINITY);
      wrong_rounded += actual[index].bits != rounded.bits;
      different_model += actual[index].bits != original[index].bits;
    }
    std::cerr << "actual_attention_reader_replay label=" << label
              << " reader=" << (which == 0 ? "host" : which == 1 ? "device" : "original")
              << " original_reader=" << (original_device_reader ? "device" : "host")
              << " Q=" << rows << " KV=" << history << " query0=" << history - rows
              << " elements=" << actual.size() << " raw_double_maxabs=" << raw_error
              << " rounded_double_maxabs=" << rounded_error << " non_nearest_bf16=" << wrong_rounded
              << " different_from_original=" << different_model << " nonfinite=" << nonfinite << '\n';
    require(!nonfinite, "actual attention replay produced nonfinite values from a finite live prefix");
    // Exactly the pre-existing short-reader component gates. This does not
    // assign a cross-recipe tolerance to downstream Model state or outputs.
    if (which == 0) require(raw_error <= .008, "original host reader exceeds its existing double budget");
    if (which == 1) require(rounded_error <= .003, "device reader exceeds its existing BF16 oracle budget");
  }
  q::check(context.close());
}
} // namespace qwen_attention_reader_pair_case
#endif
