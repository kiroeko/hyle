#include "native_plan.h"
#include "numeric.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "hyle/compute/sequence_ops.h"
#include "hyle/core/compiler.h"
#include "hyle/platform/cpu.h"

#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
#include <immintrin.h>
#endif
#include "hyle/mem/resource.h"

namespace hyle::compute::native::internal {
namespace {

struct Mappings final {
  std::array<mem::HostMapping, 7> inputs;
  std::array<mem::HostMapping, 2> outputs;
  std::array<const std::byte *, 7> source{};
  std::array<std::byte *, 2> target{};

  Status read(std::span<const ConstTensorView> views) noexcept {
    for (std::size_t i = 0; i != views.size(); ++i) {
      const auto storage = views[i].storage();
      if (storage.empty()) continue;
      if (!storage.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
        HYLE_RETURN_IF_ERROR(storage.invalidate());
      HYLE_ASSIGN_OR_RETURN(inputs[i], storage.map_host());
      source[i] = inputs[i].bytes().data();
      if (!aligned_for_storage(source[i], views[i].description().required_alignment()))
        return Status::FailedPrecondition("CPU packed sequence input mapping is misaligned");
    }
    return Status::Ok();
  }

  Status write(std::span<const TensorView> views) noexcept {
    for (std::size_t i = 0; i != views.size(); ++i) {
      if (views[i].storage().empty()) continue;
      HYLE_ASSIGN_OR_RETURN(outputs[i], views[i].storage().map_host(mem::HostAccess::kWrite));
      HYLE_ASSIGN_OR_RETURN(auto bytes, outputs[i].writable_bytes());
      target[i] = bytes.data();
      if (!aligned_for_storage(target[i], views[i].description().required_alignment()))
        return Status::FailedPrecondition("CPU packed sequence output mapping is misaligned");
    }
    return Status::Ok();
  }

  Status close(Status terminal, std::span<const TensorView> views) noexcept {
    for (std::size_t i = 0; i != outputs.size(); ++i) {
      if (outputs[i].valid() &&
          !views[i].storage().resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
        preserve_first_error(terminal, outputs[i].flush());
      preserve_first_error(terminal, outputs[i].close());
    }
    for (auto &mapping : inputs) preserve_first_error(terminal, mapping.close());
    return terminal;
  }
};

std::size_t offset_at(const std::byte *offsets, std::size_t index) noexcept {
  std::int32_t value;
  std::memcpy(&value, offsets + index * sizeof(value), sizeof(value));
  return static_cast<std::size_t>(value);
}

template <bool Prefixes = false>
Status validate_offsets(const std::byte *offsets, std::size_t batches,
                        std::size_t rows, std::size_t snapshots) noexcept {
  std::int32_t previous;
  std::memcpy(&previous, offsets, sizeof(previous));
  if (previous != 0)
    return Status::InvalidArgument("packed sequence offsets or per-sequence snapshot lengths are invalid");
  for (std::size_t i = 1; i <= batches; ++i) {
    std::int32_t next;
    std::memcpy(&next, offsets + i * sizeof(next), sizeof(next));
    if (next < previous || next < 0 || static_cast<std::size_t>(next) > rows)
      return Status::InvalidArgument("packed sequence offsets or per-sequence snapshot lengths are invalid");
    if constexpr (Prefixes) {
      if (static_cast<std::size_t>(next - previous) > snapshots)
        return Status::InvalidArgument("packed GDN sequence length exceeds prefix capacity");
    } else {
      if (snapshots > 1 && static_cast<std::size_t>(next - previous) < snapshots)
        return Status::InvalidArgument("packed sequence offsets or per-sequence snapshot lengths are invalid");
    }
    previous = next;
  }
  return Status::Ok();
}

template <class Value> struct GdnEpisode final {
  const NativePlan *plan;
  const Mappings *maps;
  std::size_t key_heads, value_heads, key_depth, value_depth, gate_depth;
};

template <bool HardwareFma>
HYLE_FORCE_INLINE float recurrent_fma(float a, float b, float c) noexcept {
#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
  if constexpr (HardwareFma)
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
#endif
  return std::fma(a, b, c);
}

template <class Value, bool Prefixes, bool HardwareFma>
void run_gdn(void *context, std::size_t, std::size_t begin, std::size_t end) noexcept {
  const auto &episode = *static_cast<GdnEpisode<Value> *>(context);
  const auto &maps = *episode.maps;
  const auto snapshots = static_cast<std::size_t>(episode.plan->scalar_bits);
  const auto heads = episode.value_heads, dv = episode.value_depth, dk = episode.key_depth;
  const auto state_stride = heads * dv * dk;
  auto *state_output = reinterpret_cast<float *>(maps.target[1]);
  for (auto unit = begin; unit != end; ++unit) {
    const auto batch = unit / (heads * dv), head = unit / dv % heads, channel = unit % dv;
    const auto first = offset_at(maps.source[6], batch), last = offset_at(maps.source[6], batch + 1);
    const auto length = last - first;
    if constexpr (Prefixes) {
      // No valid prefix exists for an empty member; its slots remain +0.
      if (length == 0) continue;
    }
    const auto live_snapshot = Prefixes ? length - 1 : snapshots - 1;
    auto *state = state_output + (batch * snapshots + live_snapshot) * state_stride +
                  (head * dv + channel) * dk;
    std::memcpy(state, maps.source[5] + unit * dk * sizeof(float), dk * sizeof(float));
    for (auto row = first; row != last; ++row) {
      const auto key_base = (row * episode.key_heads + head % episode.key_heads) * dk;
      const auto value_index = (row * heads + head) * dv + channel;
      const auto gate_index = (row * heads + head) * episode.gate_depth +
                              (episode.gate_depth == 1 ? 0 : channel);
      const float decay = std::exp(numeric::load<float>(maps.source[3] + gate_index * sizeof(float)));
      const float beta = numeric::load<float>(maps.source[4] + (row * heads + head) * sizeof(float));
      float prediction = 0.0F;
      for (std::size_t key = 0; key != dk; ++key) {
        state[key] *= decay;
        prediction = recurrent_fma<HardwareFma>(state[key], numeric::load<Value>(maps.source[1] +
            (key_base + key) * sizeof(Value)), prediction);
      }
      const float delta = beta * (numeric::load<Value>(maps.source[2] + value_index * sizeof(Value)) - prediction);
      float attended = 0.0F;
      for (std::size_t key = 0; key != dk; ++key) {
        state[key] = recurrent_fma<HardwareFma>(delta, numeric::load<Value>(maps.source[1] +
            (key_base + key) * sizeof(Value)), state[key]);
        attended = recurrent_fma<HardwareFma>(state[key], numeric::load<Value>(maps.source[0] +
            (key_base + key) * sizeof(Value)), attended);
      }
      numeric::store<Value>(maps.target[0] + value_index * sizeof(Value), attended * episode.plan->scalar);
      if constexpr (Prefixes) {
        const auto snapshot = row - first;
        // The last valid prefix slot is the live state, not a padding slot.
        if (snapshot + 1 < length)
          std::memcpy(state_output + (batch * snapshots + snapshot) * state_stride +
                          (head * dv + channel) * dk,
                      state, dk * sizeof(float));
      } else {
        if (snapshots > 1 && row - first >= length - snapshots) {
          const auto snapshot = row - first - (length - snapshots);
          // The last slot remains the live recurrent state until the final token.
          if (snapshot + 1 < snapshots)
            std::memcpy(state_output + (batch * snapshots + snapshot) * state_stride +
                            (head * dv + channel) * dk,
                        state, dk * sizeof(float));
        }
      }
    }
  }
}


#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
inline constexpr std::size_t kChannelTile = 8;
inline constexpr std::size_t kKeyTileCapacity = 256;

// Each lane owns one value channel. The F32 accumulators traverse keys in
// the original order and the tile is private to this GDN range invocation.
void run_gdn_channel_tile(void *context, std::size_t, std::size_t begin,
                          std::size_t end) noexcept {
  const auto &episode = *static_cast<GdnEpisode<float> *>(context);
  const auto &maps = *episode.maps;
  const auto heads = episode.value_heads, dv = episode.value_depth, dk = episode.key_depth;
  const auto q = reinterpret_cast<const float *>(maps.source[0]);
  const auto k = reinterpret_cast<const float *>(maps.source[1]);
  const auto v = reinterpret_cast<const float *>(maps.source[2]);
  const auto gate = reinterpret_cast<const float *>(maps.source[3]);
  const auto beta = reinterpret_cast<const float *>(maps.source[4]);
  const auto initial_state = reinterpret_cast<const float *>(maps.source[5]);
  auto *values = reinterpret_cast<float *>(maps.target[0]);
  auto *final_state = reinterpret_cast<float *>(maps.target[1]);
  const auto scale = _mm256_set1_ps(episode.plan->scalar);
  const auto strides = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                         _mm256_set1_epi32(static_cast<int>(dk)));
  alignas(32) __m256 state[kKeyTileCapacity];
  for (auto tile = begin; tile != end; ++tile) {
    const auto unit = tile * kChannelTile;
    const auto batch = unit / (heads * dv), head = unit / dv % heads, channel = unit % dv;
    const auto first = offset_at(maps.source[6], batch), last = offset_at(maps.source[6], batch + 1);
    const auto initial = initial_state + unit * dk;
    auto *result_state = final_state + unit * dk;
    if (first == last) {
      std::memcpy(result_state, initial, kChannelTile * dk * sizeof(float));
      continue;
    }
    for (std::size_t key = 0; key != dk; ++key)
      state[key] = _mm256_i32gather_ps(initial + key, strides, 4);
    for (auto row = first; row != last; ++row) {
      const auto key_base = (row * episode.key_heads + head % episode.key_heads) * dk;
      const auto gate_index = row * heads + head;
      const auto value_index = gate_index * dv + channel;
      const auto decay = _mm256_set1_ps(std::exp(gate[gate_index]));
      auto prediction = _mm256_setzero_ps();
      for (std::size_t key = 0; key != dk; ++key) {
        const auto decayed = _mm256_mul_ps(state[key], decay);
        prediction = _mm256_fmadd_ps(decayed, _mm256_set1_ps(k[key_base + key]), prediction);
        state[key] = decayed;
      }
      const auto remaining = _mm256_sub_ps(_mm256_loadu_ps(v + value_index), prediction);
      const auto delta = _mm256_mul_ps(_mm256_set1_ps(beta[gate_index]), remaining);
      auto attended = _mm256_setzero_ps();
      for (std::size_t key = 0; key != dk; ++key) {
        const auto updated = _mm256_fmadd_ps(delta, _mm256_set1_ps(k[key_base + key]), state[key]);
        attended = _mm256_fmadd_ps(updated, _mm256_set1_ps(q[key_base + key]), attended);
        state[key] = updated;
      }
      _mm256_storeu_ps(values + value_index, _mm256_mul_ps(attended, scale));
    }
    alignas(32) float column[kChannelTile];
    for (std::size_t key = 0; key != dk; ++key) {
      _mm256_store_ps(column, state[key]);
      for (std::size_t lane = 0; lane != kChannelTile; ++lane)
        result_state[lane * dk + key] = column[lane];
    }
  }
}
#endif

template <class Value, bool Prefixes = false, bool HardwareFma = false, bool ChannelTile = false>
detail::EnqueueResult submit_gdn(const NativePlan &plan, std::span<const ConstTensorView> inputs,
                                std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 7 || outputs.size() != 2)
    return completed(Status::Internal("CPU packed GDN binding count differs"));
  Mappings maps;
  auto status = maps.read(inputs);
  const auto state = inputs[5].description().dimensions();
  const auto rows = static_cast<std::size_t>(inputs[0].description().dimensions()[0]);
  if (status.ok()) status = validate_offsets<Prefixes>(maps.source[6], static_cast<std::size_t>(state[0]),
                                            rows, static_cast<std::size_t>(plan.scalar_bits));
  if (status.ok()) status = maps.write(outputs);
  if (status.ok()) {
    if (plan.parameters[2] != 0)
      std::memset(maps.target[0], 0, plan.parameters[2]);
    if constexpr (Prefixes) {
      const auto state_bytes = *outputs[1].description().required_bytes();
      if (state_bytes != 0) std::memset(maps.target[1], 0, state_bytes);
    }
    GdnEpisode<Value> episode{&plan, &maps,
        static_cast<std::size_t>(inputs[0].description().dimensions()[1]),
        static_cast<std::size_t>(state[1]), static_cast<std::size_t>(state[3]),
        static_cast<std::size_t>(state[2]), static_cast<std::size_t>(inputs[3].description().dimensions()[2])};
#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
    if constexpr (ChannelTile) {
      parallel_for_ranges(plan, plan.parameters[0] / kChannelTile, &run_gdn_channel_tile, &episode);
    } else
#endif
    parallel_for_ranges(plan, plan.parameters[0], &run_gdn<Value, Prefixes, HardwareFma>, &episode);
  }
  return completed(maps.close(std::move(status), outputs));
}

template <bool Prefixes = false>
SubmitRoute select_gdn_submit(DataType type) noexcept {
  if (type == DataType::kFloat32) {
#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
    // The baseline x64 build otherwise calls the CRT for each FMA.
    // Select the identical ordered hardware operation during preparation.
    using enum platform::CpuFeature;
    if (platform::cpu_features().has_all(platform::CpuFeatureSet{kX86Avx, kX86Fma}))
      return &submit_gdn<float, Prefixes, true>;
#endif
    return &submit_gdn<float, Prefixes>;
  }
  return type == DataType::kFloat16 ? &submit_gdn<simd::float16_t, Prefixes>
                                   : &submit_gdn<simd::bfloat16_t, Prefixes>;
}

template <class Value> struct ConvEpisode final {
  const Mappings *maps;
  std::size_t channels, taps;
};

template <class Value>
void run_conv(void *context, std::size_t, std::size_t begin, std::size_t end) noexcept {
  const auto &episode = *static_cast<ConvEpisode<Value> *>(context);
  const auto &maps = *episode.maps;
  const auto channels = episode.channels, taps = episode.taps, history_rows = taps - 1;
  for (auto unit = begin; unit != end; ++unit) {
    const auto batch = unit / channels, channel = unit % channels;
    const auto first = offset_at(maps.source[3], batch), last = offset_at(maps.source[3], batch + 1);
    const auto length = last - first;
    const auto sample = [&](std::size_t position) noexcept {
      return position < history_rows
          ? maps.source[2] + ((batch * history_rows + position) * channels + channel) * sizeof(Value)
          : maps.source[0] + ((first + position - history_rows) * channels + channel) * sizeof(Value);
    };
    for (std::size_t row = 0; row != length; ++row) {
      float sum = 0.0F;
      for (std::size_t tap = 0; tap != taps; ++tap)
        sum = std::fma(numeric::load<Value>(sample(row + tap)),
                       numeric::load<Value>(maps.source[1] + (channel * taps + tap) * sizeof(Value)), sum);
      numeric::store<Value>(maps.target[0] + ((first + row) * channels + channel) * sizeof(Value), sum);
    }
    for (std::size_t row = 0; row != history_rows; ++row)
      std::memcpy(maps.target[1] + ((batch * history_rows + row) * channels + channel) * sizeof(Value),
                  sample(length + row), sizeof(Value));
  }
}

template <class Value>
detail::EnqueueResult submit_conv(const NativePlan &plan, std::span<const ConstTensorView> inputs,
                                 std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 4 || outputs.size() != 2)
    return completed(Status::Internal("CPU packed convolution binding count differs"));
  Mappings maps;
  auto status = maps.read(inputs);
  const auto history = inputs[2].description().dimensions();
  if (status.ok()) status = validate_offsets(maps.source[3], static_cast<std::size_t>(history[0]),
      static_cast<std::size_t>(inputs[0].description().dimensions()[0]), 1);
  if (status.ok()) status = maps.write(outputs);
  if (status.ok()) {
    if (plan.parameters[2] != 0)
      std::memset(maps.target[0], 0, plan.parameters[2]);
    ConvEpisode<Value> episode{&maps, static_cast<std::size_t>(history[2]),
        static_cast<std::size_t>(inputs[1].description().dimensions()[1])};
    parallel_for_ranges(plan, plan.parameters[0], &run_conv<Value>, &episode);
  }
  return completed(maps.close(std::move(status), outputs));
}

} // namespace

Status prepare_sequence(Operation operation, std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs, NativePlan &plan) noexcept {
  std::array<TensorDesc, 2> expected;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, expected));
  if (outputs.size() != expected.size() || outputs[0] != expected[0] || outputs[1] != expected[1])
    return Status::InvalidArgument("CPU packed sequence output descriptions differ");
  const auto type = inputs[0].data_type();
  if (const auto gdn = operation.custom_as<sequence_ops::PackedGatedDeltaNet>()) {
    const auto state = inputs[5].dimensions();
    plan.parameters = {*inputs[5].element_count() / static_cast<std::size_t>(state[3]),
                        static_cast<std::size_t>(state[3]), *outputs[0].required_bytes()};
    plan.scalar_bits = gdn->snapshot_count;
    plan.scalar = gdn->output_scale.value_or(1.0F / std::sqrt(static_cast<float>(state[3])));
    plan.submit_route = select_gdn_submit<>(type);
#if HYLE_ARCH_X64 && HYLE_COMPILER_MSVC
    // The copy into a channel tile is amortized by more than one row.
    // Each sequence has one final state and one gate per value head.
    using enum platform::CpuFeature;
    if (type == DataType::kFloat32 && gdn->snapshot_count == 1 &&
        inputs[0].dimensions()[0] > 1 && inputs[3].dimensions()[2] == 1 &&
        state[2] % kChannelTile == 0 && state[3] <= kKeyTileCapacity &&
        platform::cpu_features().has_all(platform::CpuFeatureSet{kX86Avx2, kX86Fma}))
      plan.submit_route = &submit_gdn<float, false, true, true>;
#endif

  } else if (const auto prefixes = operation.custom_as<sequence_ops::PackedGatedDeltaNetPrefixes>()) {
    const auto state = inputs[5].dimensions();
    plan.parameters = {*inputs[5].element_count() / static_cast<std::size_t>(state[3]),
                        static_cast<std::size_t>(state[3]), *outputs[0].required_bytes()};
    plan.scalar_bits = prefixes->max_prefixes;
    plan.scalar = prefixes->output_scale.value_or(1.0F / std::sqrt(static_cast<float>(state[3])));
    plan.submit_route = select_gdn_submit<true>(type);
  } else if (operation.custom_as<sequence_ops::PackedCausalConv1d>()) {
    const auto history = inputs[2].dimensions();
    // A K=1 empty history still needs one independent work unit for every
    // sequence/channel; unlike a nonempty tensor it cannot prove B*C fits.
    const auto batches = static_cast<std::size_t>(history[0]);
    const auto channels = static_cast<std::size_t>(history[2]);
    if (batches > SIZE_MAX / channels)
      return Status::OutOfRange("CPU packed convolution task count overflows");
    plan.parameters = {batches * channels, static_cast<std::size_t>(inputs[1].dimensions()[1]), *outputs[0].required_bytes()};
    plan.submit_route = type == DataType::kFloat32 ? &submit_conv<float>
        : type == DataType::kFloat16 ? &submit_conv<simd::float16_t> : &submit_conv<simd::bfloat16_t>;
  } else {
    return Status::Unimplemented("CPU packed sequence operation is unsupported");
  }
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
