#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "../../third_party/random123_counter/philox.h"
#include "hyle/compute/sampling_ops.h"
#include "hyle/simd/math.h"

namespace hyle::compute::native::internal {
namespace {
namespace random = hyle::borrowed::random123_counter;

struct Mappings final {
  std::array<mem::HostMapping, 2> inputs;
  std::array<mem::HostMapping, 2> outputs;
  mem::HostMapping scratch;
  std::size_t writable = 0;
  bool wrote = false;

  Status finish(Status terminal, std::span<const TensorView> views) noexcept {
    for (std::size_t i = 0; i < writable; ++i) {
      if (wrote && !views[i].storage().resource().properties().has(
                       mem::MemoryProperty::kHostDeviceCoherent))
        preserve_first_error(terminal, outputs[i].flush());
      preserve_first_error(terminal, outputs[i].close());
    }
    preserve_first_error(terminal, scratch.close());
    for (auto &input : inputs)
      preserve_first_error(terminal, input.close());
    return terminal;
  }
};

Status map_inputs(Mappings &maps,
                   std::span<const ConstTensorView> inputs) noexcept {
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto storage = inputs[i].storage();
    if (!storage.resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent))
      HYLE_RETURN_IF_ERROR(storage.invalidate());
    HYLE_ASSIGN_OR_RETURN(auto mapping, storage.map_host());
    maps.inputs[i] = std::move(mapping);
    if (!aligned_for_storage(maps.inputs[i].bytes().data(),
                              inputs[i].description().required_alignment()))
      return Status::FailedPrecondition("CPU sampling input alignment differs");
  }
  return Status::Ok();
}

Status map_outputs(Mappings &maps, std::span<const TensorView> outputs,
                    std::array<std::byte *, 2> &data) noexcept {
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto mapping,
                          outputs[i].storage().map_host(mem::HostAccess::kWrite));
    maps.outputs[i] = std::move(mapping);
    ++maps.writable;
    HYLE_ASSIGN_OR_RETURN(auto bytes, maps.outputs[i].writable_bytes());
    data[i] = bytes.data();
    if (!aligned_for_storage(data[i], outputs[i].description().required_alignment()))
      return Status::FailedPrecondition("CPU sampling output alignment differs");
  }
  return Status::Ok();
}

struct BitsEpisode final {
  const std::uint32_t *identities;
  std::uint32_t *output;
  std::size_t width;
};
void bits_range(void *state, std::size_t, std::size_t begin,
                 std::size_t end) noexcept {
  auto &episode = *static_cast<BitsEpisode *>(state);
  for (auto row = begin; row < end; ++row) {
    const auto *identity = episode.identities + row * 4;
    for (std::size_t column = 0; column < episode.width; column += 4) {
      const auto words = random::block(identity, column / 4);
      for (unsigned lane = 0; lane < 4 && column + lane < episode.width; ++lane)
        episode.output[row * episode.width + column + lane] = random::lane(words, lane);
    }
  }
}
detail::EnqueueResult submit_bits(const NativePlan &plan,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (plan.parameters[0] == 0 || plan.parameters[1] == 0)
    return completed(Status::Ok());
  Mappings maps;
  const auto finish = [&](Status status) { return completed(maps.finish(std::move(status), outputs)); };
  auto status = map_inputs(maps, inputs);
  if (!status.ok()) return finish(std::move(status));
  std::array<std::byte *, 2> data{};
  status = map_outputs(maps, outputs, data);
  if (!status.ok()) return finish(std::move(status));
  BitsEpisode episode{reinterpret_cast<const std::uint32_t *>(maps.inputs[0].bytes().data()),
                       reinterpret_cast<std::uint32_t *>(data[0]), plan.parameters[1]};
  maps.wrote = true;
  parallel_for_ranges(plan, plan.parameters[0], &bits_range, &episode);
  return finish(Status::Ok());
}

template<class Value> struct CategoricalEpisode final {
  const NativePlan *plan;
  const Value *logits;
  const std::uint32_t *identities;
  std::int32_t *indices = nullptr;
  float *log_probabilities = nullptr;
  float *maxima;
  float *tiles;
  std::atomic<bool> invalid{false};
};
template<class Value>
void validate_rows(void *state, std::size_t, std::size_t begin,
                     std::size_t end) noexcept {
  auto &episode = *static_cast<CategoricalEpisode<Value> *>(state);
  const auto width = episode.plan->parameters[1];
  for (auto row = begin; row < end; ++row) {
    auto maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t column = 0; column < width; ++column) {
      const auto value = numeric::decode(episode.logits[row * width + column]);
      if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
        episode.invalid.store(true, std::memory_order_relaxed);
      maximum = (std::max)(maximum, value);
    }
    if (!std::isfinite(maximum))
      episode.invalid.store(true, std::memory_order_relaxed);
    episode.maxima[row] = maximum;
  }
}
template<class Value>
void categorical_rows(void *state, std::size_t task, std::size_t begin,
                         std::size_t end) noexcept {
  auto &episode = *static_cast<CategoricalEpisode<Value> *>(state);
  const auto width = episode.plan->parameters[1], capacity = episode.plan->parameters[2];
  auto *tile = episode.tiles + task * capacity;
  for (auto row = begin; row < end; ++row) {
    const auto *input = episode.logits + row * width;
    const auto maximum = episode.maxima[row];
    const auto weights = [&](std::size_t start) noexcept {
      const auto count = (std::min)(capacity, width - start);
      numeric::load_tile<Value>(input, start, tile, count);
      for (std::size_t i = 0; i < count; ++i) tile[i] -= maximum;
      simd::exp_f32_batch(tile, tile, count);
      return count;
    };
    double total = 0;
    for (std::size_t start = 0; start < width; start += capacity) {
      const auto count = weights(start);
      for (std::size_t i = 0; i < count; ++i) total += tile[i];
    }
    const auto random_words = random::block(episode.identities + row * 4, 0);
    // The 52-bit midpoint is exactly representable in F64 at both ends.
    const auto uniform = (double(random_words.x0) * 0x1p20 +
                           double(random_words.x1 >> 12) + .5) * 0x1p-52;
    const auto threshold = total * uniform;
    double cumulative = 0;
    std::size_t selected = width;
    for (std::size_t start = 0; start < width && selected == width; start += capacity) {
      const auto count = weights(start);
      for (std::size_t i = 0; i < count; ++i) {
        cumulative += tile[i];
        if (cumulative > threshold) { selected = start + i; break; }
      }
    }
    // Repeated SIMD exp uses the same recipe. Rounding a target at the last
    // cumulative boundary must still select positive mass, never a masked tail.
    if (selected == width) {
      for (selected = width; selected > 0; --selected)
        if (numeric::decode(input[selected - 1]) == maximum) { --selected; break; }
    }
    episode.indices[row] = static_cast<std::int32_t>(selected);
    if (episode.log_probabilities)
      episode.log_probabilities[row] = static_cast<float>(
          double(numeric::decode(input[selected])) - double(maximum) - std::log(total));
  }
}
template<class Value>
detail::EnqueueResult submit_categorical(const NativePlan &plan,
    std::span<const ConstTensorView> inputs, std::span<const TensorView> outputs,
    mem::BufferView workspace) noexcept {
  if (plan.parameters[0] == 0) return completed(Status::Ok());
  Mappings maps;
  const auto finish = [&](Status status) { return completed(maps.finish(std::move(status), outputs)); };
  auto status = map_inputs(maps, inputs);
  if (!status.ok()) return finish(std::move(status));
  auto scratch = workspace.map_host(mem::HostAccess::kWrite);
  if (!scratch) return finish(std::move(scratch).error());
  maps.scratch = std::move(*scratch);
  auto bytes = maps.scratch.writable_bytes();
  if (!bytes || bytes->size() < plan.workspace.size ||
      !aligned_for_storage(bytes->data(), alignof(float)))
    return finish(Status::FailedPrecondition("CPU sampling workspace mapping differs"));
  auto *maxima = reinterpret_cast<float *>(bytes->data());
  CategoricalEpisode<Value> episode{&plan,
      reinterpret_cast<const Value *>(maps.inputs[0].bytes().data()),
      reinterpret_cast<const std::uint32_t *>(maps.inputs[1].bytes().data()),
      nullptr, nullptr, maxima, maxima + plan.parameters[0]};
  parallel_for_ranges(plan, plan.parameters[0], &validate_rows<Value>, &episode);
  if (episode.invalid.load(std::memory_order_relaxed))
    return finish(Status::InvalidArgument("Categorical requires valid nonzero row mass"));
  std::array<std::byte *, 2> data{};
  status = map_outputs(maps, outputs, data);
  if (!status.ok()) return finish(std::move(status));
  episode.indices = reinterpret_cast<std::int32_t *>(data[0]);
  episode.log_probabilities = outputs.size() == 2 ? reinterpret_cast<float *>(data[1]) : nullptr;
  maps.wrote = true;
  parallel_for_ranges(plan, plan.parameters[0], &categorical_rows<Value>, &episode);
  return finish(Status::Ok());
}
} // namespace

Status prepare_sampling(Operation operation, std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs,
                          NativePlan &plan) noexcept {
  const auto bits = operation.custom_as<sampling_ops::RandomBits>();
  const auto categorical = operation.custom_as<sampling_ops::Categorical>();
  if (!bits && !categorical)
    return Status::Unimplemented("operation is not a sampling numerical schema");
  std::array<TensorDesc, 2> inferred;
  if (outputs.size() > inferred.size())
    return Status::InvalidArgument("sampling output count differs");
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs,
                          std::span(inferred).first(outputs.size())));
  if (!std::ranges::equal(outputs, std::span(inferred).first(outputs.size())))
    return Status::InvalidArgument("sampling output descriptions differ");
  plan.parameters = {static_cast<std::size_t>(inputs[0].dimensions()[0]),
      bits ? bits->word_count : static_cast<std::size_t>(inputs[0].dimensions()[1]),
      categorical ? 1024u : 0u};
  if (bits) plan.submit_route = &submit_bits;
  else if (inputs[0].data_type() == DataType::kFloat32)
    plan.workspace_submit_route = &submit_categorical<float>;
  else if (inputs[0].data_type() == DataType::kFloat16)
    plan.workspace_submit_route = &submit_categorical<simd::float16_t>;
  else plan.workspace_submit_route = &submit_categorical<simd::bfloat16_t>;
  return Status::Ok();
}
} // namespace hyle::compute::native::internal
