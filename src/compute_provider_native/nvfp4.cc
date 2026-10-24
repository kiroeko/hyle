#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/model_ops.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {
enum class Math { kModelOpt, kLocalFloat32, kLocalBFloat16, kMxfp4 };
// The pinned non-fast ModelOpt table deliberately maps code8 to positive zero.
constexpr std::array<float, 16> kValues{
    0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F,
    0.F, -.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};

float nearest_e2m1(float magnitude) noexcept {
  if (magnitude <= .25F) return 0.F;
  if (magnitude < .75F) return .5F;
  if (magnitude <= 1.25F) return 1.F;
  if (magnitude < 1.75F) return 1.5F;
  if (magnitude <= 2.5F) return 2.F;
  if (magnitude < 3.5F) return 3.F;
  return magnitude <= 5.F ? 4.F : 6.F;
}

bool bounded(const void *pointer, std::size_t bytes) noexcept {
  return pointer && bytes <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(pointer);
}
bool overlaps(const void *a, std::size_t na, const void *b,
               std::size_t nb) noexcept {
  const auto x = reinterpret_cast<std::uintptr_t>(a);
  const auto y = reinterpret_cast<std::uintptr_t>(b);
  return x < y + nb && y < x + na;
}

struct Mappings final {
  std::array<mem::HostMapping, 6> input;
  mem::HostMapping scratch, output;
  bool wrote = false;

  Status finish(Status status, mem::BufferView destination) noexcept {
    if (wrote && !destination.resource().properties().has(
                     mem::MemoryProperty::kHostDeviceCoherent))
      preserve_first_error(status, output.flush());
    preserve_first_error(status, output.close());
    preserve_first_error(status, scratch.close());
    for (auto &mapping : input)
      preserve_first_error(status, mapping.close());
    return status;
  }
};

struct QuantizeEpisode final {
  const simd::bfloat16_t *input;
  simd::bfloat16_t *quantized;
  const std::byte *globals;
  const std::int32_t *ids;
  std::size_t slots, groups_per_row, token_stride, slot_stride;
  std::atomic<bool> invalid{false};
};

template <Math Mode, bool WeightOnly = false>
void quantize_groups(void *state, std::size_t, std::size_t begin,
                     std::size_t end) noexcept {
  constexpr std::size_t kGroup = Mode == Math::kMxfp4 ? 32 : 16;
  auto &episode = *static_cast<QuantizeEpisode *>(state);
  std::array<float, kGroup> values{};
  for (auto group = begin; group < end; ++group) {
    const auto row = group / episode.groups_per_row;
    const auto within_row = group % episode.groups_per_row;
    float global = 0.F;
    if constexpr (Mode != Math::kMxfp4) {
      const auto expert = episode.ids ? static_cast<std::size_t>(episode.ids[row]) : 0;
      global = numeric::load<float>(episode.globals + expert * sizeof(float));
    }
    const auto input_offset = (row / episode.slots) * episode.token_stride +
        (row % episode.slots) * episode.slot_stride + within_row * kGroup;
    simd::convert(episode.input + input_offset, values.data(), kGroup);
    float maximum = 0.F;
    bool finite = true;
    for (const auto value : values) {
      finite &= std::isfinite(value);
      maximum = (std::max)(maximum, std::abs(value));
    }
    if (!finite) {
      episode.invalid.store(true, std::memory_order_relaxed);
      continue;
    }
    if constexpr (WeightOnly) continue;
    if constexpr (Mode == Math::kMxfp4) {
      const auto rounded = ((std::bit_cast<std::uint32_t>(maximum) >> 16) + 0x20u) & 0x7f80u;
      const int exponent = rounded == 0 ? -127 : rounded == 0x7f80u ? 127
          : std::clamp(static_cast<int>(rounded >> 7) - 129, -127, 127);
      const float scale = std::ldexp(1.F, exponent);
      for (std::size_t lane = 0; lane < kGroup; ++lane) {
        volatile float divided = values[lane] / scale;
        const float normalized = simd::bf16_to_f32(simd::f32_to_bf16(divided));
        const float code = std::copysign(nearest_e2m1(std::abs(normalized)), normalized);
        volatile float restored = code * scale;
        episode.quantized[group * kGroup + lane] = simd::f32_to_bf16(restored);
      }
    } else if constexpr (Mode == Math::kModelOpt) {
      // Keep each source F32 operation and the original tiny-scale guard.
      const float denominator = 6.F * global;
      const float normalized = (std::min)(maximum / denominator, 448.F);
      const float block = simd::f8e4m3_to_f32(simd::f32_to_f8e4m3(normalized));
      const float scaled = block * global;
      const float scale = scaled >= 1.e-5F ? scaled : 1.F;
      for (std::size_t lane = 0; lane < kGroup; ++lane) {
        const float magnitude = nearest_e2m1(std::abs(values[lane]) / scale);
        const float restored = magnitude * scale;
        // The original comparison maps input -0 to +0; negative tiny values
        // which quantize to zero retain their negative sign.
        episode.quantized[group * kGroup + lane] = simd::f32_to_bf16(
            values[lane] >= 0.F ? restored : -restored);
      }
    } else {
      // CT's stored global is a divisor. The BF16 division before multiplying
      // that F32 tensor is part of its local scale calculation.
      volatile float denominator = 6.F;
      volatile float divided = maximum / denominator;
      const float base = simd::bf16_to_f32(simd::f32_to_bf16(divided));
      volatile float normalized = base * global;
      float block = simd::f8e4m3_to_f32(simd::f32_to_f8e4m3(normalized));
      if (block == 0.F) block = .125F;
      volatile float scale = block / global;
      for (std::size_t lane = 0; lane < kGroup; ++lane) {
        volatile float divided_value = values[lane] / scale;
        float scaled = std::clamp(static_cast<float>(divided_value), -6.F, 6.F);
        if constexpr (Mode == Math::kLocalBFloat16)
          scaled = simd::bf16_to_f32(simd::f32_to_bf16(scaled));
        const float code = nearest_e2m1(std::abs(scaled));
        volatile float restored = code * scale;
        // Portable Torch sign: explicit -0 (including pre-round underflow)
        // becomes +0; a negative nonzero rounding to code0 retains -0.
        episode.quantized[group * kGroup + lane] = simd::f32_to_bf16(
            scaled >= 0.F ? restored : -restored);
      }
    }
  }
}

struct DotEpisode final {
  const NativePlan *plan;
  const simd::bfloat16_t *activation;
  const std::uint8_t *weights;
  const std::uint8_t *scales;
  const std::byte *globals;
  const std::int32_t *ids;
  simd::bfloat16_t *output;
  std::size_t output_slots, quant_slots, token_stride, slot_stride;
};

template <Math Mode, bool WeightOnly = false>
void dot_rows(void *state, std::size_t, std::size_t begin,
               std::size_t end) noexcept {
  constexpr std::size_t kGroup = Mode == Math::kMxfp4 ? 32 : 16;
  const auto &episode = *static_cast<const DotEpisode *>(state);
  const auto n = episode.plan->parameters[1];
  const auto k = episode.plan->parameters[2];
  const auto groups = k / kGroup;
  std::array<simd::bfloat16_t, kGroup> decoded{};
  for (auto index = begin; index < end; ++index) {
    const auto row = index / n, column = index % n;
    const auto expert = episode.ids ? static_cast<std::size_t>(episode.ids[row]) : 0;
    float global = 0.F;
    if constexpr (Mode != Math::kMxfp4)
      global = numeric::load<float>(episode.globals + expert * sizeof(float));
    const auto quant_row = (row / episode.output_slots) * episode.quant_slots +
        (episode.quant_slots == 1 ? 0 : row % episode.output_slots);
    const auto activation_offset = WeightOnly
        ? (row / episode.output_slots) * episode.token_stride +
              (row % episode.output_slots) * episode.slot_stride
        : quant_row * k;
    const auto *activation = episode.activation + activation_offset;
    const auto *weights = episode.weights + (expert * n + column) * (k / 2);
    const auto *scales = episode.scales + (expert * n + column) * groups;
    float sum = 0.F;
    for (std::size_t group = 0; group < groups; ++group) {
      float scale;
      if constexpr (Mode == Math::kMxfp4) {
        scale = std::ldexp(1.F, static_cast<int>(scales[group]) - 127);
      } else {
        const auto block = simd::f8e4m3_to_f32(
            std::bit_cast<simd::float8_e4m3_t>(scales[group]));
        if constexpr (Mode == Math::kModelOpt) {
          scale = block * global;
        } else {
          volatile float divided = block / global;
          scale = divided;
        }
      }
      for (std::size_t lane = 0; lane < kGroup; ++lane) {
        const auto packed = weights[group * (kGroup / 2) + lane / 2];
        const auto code = (packed >> (4 * (lane % 2))) & 15U;
        // Do not fold either zero code or move the scale after the dot:
        // even valid finite source scales may overflow this F32 product.
        if constexpr (Mode == Math::kModelOpt) {
          decoded[lane] = simd::f32_to_bf16(kValues[code] * scale);
        } else {
          const float value = code == 8 ? -0.F : kValues[code];
          volatile float product = value * scale;
          decoded[lane] = simd::f32_to_bf16(product);
        }
      }
      sum += simd::dot(activation + group * kGroup, decoded.data(), kGroup);
    }
    episode.output[index] = simd::f32_to_bf16(sum);
  }
}

template <Math Mode, bool WeightOnly = false>
detail::EnqueueResult submit_nvfp4(const NativePlan &plan,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, mem::BufferView workspace) noexcept {
  constexpr bool microscaled = Mode == Math::kMxfp4;
  constexpr std::size_t kGroup = microscaled ? 32 : 16;
  constexpr std::size_t dense_ports = microscaled ? 3 : 5;
  if ((inputs.size() != dense_ports && inputs.size() != dense_ports + 1) || outputs.size() != 1)
    return completed(Status::Internal("native NVFP4 binding count differs"));
  const bool grouped = inputs.size() == dense_ports + 1;
  Mappings maps;
  const auto finish = [&](Status status) {
    return completed(maps.finish(std::move(status), outputs[0].storage()));
  };
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto storage = inputs[i].storage();
    if (!storage.resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent)) {
      auto status = storage.invalidate();
      if (!status.ok()) return finish(std::move(status));
    }
    auto mapped = storage.map_host();
    if (!mapped) return finish(std::move(mapped).error());
    maps.input[i] = std::move(*mapped);
    if (!aligned_for_storage(maps.input[i].bytes().data(),
                              inputs[i].description().required_alignment()))
      return finish(Status::FailedPrecondition("native NVFP4 input mapping alignment differs"));
  }
  const auto *ids = grouped
      ? reinterpret_cast<const std::int32_t *>(maps.input[dense_ports].bytes().data()) : nullptr;
  const auto rows = plan.parameters[0], n = plan.parameters[1], k = plan.parameters[2];
  const auto experts = grouped ? static_cast<std::size_t>(inputs[1].description().dimensions()[0]) : 1;
  // Reject every ID before the first expert offset or selected metadata load.
  // In particular a later bad ID cannot reach an out-of-bounds global/scale.
  if (grouped) for (std::size_t row = 0; row < rows; ++row) {
    if (ids[row] < 0 || static_cast<std::size_t>(ids[row]) >= experts)
      return finish(Status::OutOfRange("grouped NVFP4 expert id is outside the bank"));
  }
  const auto *weight_globals = microscaled ? nullptr : maps.input[3].bytes().data();
  const auto *input_globals = microscaled ? nullptr : maps.input[4].bytes().data();
  const auto *scales = reinterpret_cast<const std::uint8_t *>(
      maps.input[2].bytes().data());
  const auto scale_count = n * (k / kGroup);
  // Dense validates its sole metadata once. Grouped touches only actually
  // selected experts, preserving invalid/unreferenced expert storage.
  const auto metadata_rows = grouped ? rows : 1;
  for (std::size_t row = 0; row < metadata_rows; ++row) {
    const auto expert = grouped ? static_cast<std::size_t>(ids[row]) : 0;
    if constexpr (!microscaled) {
      const float weight_global = numeric::load<float>(weight_globals + expert * sizeof(float));
      const float input_global = numeric::load<float>(input_globals + expert * sizeof(float));
      if (!std::isfinite(weight_global) || !(weight_global > 0.F) ||
          !std::isfinite(input_global) || !(input_global > 0.F))
        return finish(Status::InvalidArgument("NVFP4 globals must be positive finite F32"));
    }
    const auto *selected = scales + expert * scale_count;
    for (std::size_t i = 0; i < scale_count; ++i) {
      if constexpr (microscaled) {
        if (selected[i] == 255)
          return finish(Status::InvalidArgument("MXFP4 scale255 is outside the finite scale domain"));
      } else {
        const float scale = simd::f8e4m3_to_f32(
            std::bit_cast<simd::float8_e4m3_t>(selected[i]));
        if (!std::isfinite(scale) || scale < 0.F)
          return finish(Status::InvalidArgument("NVFP4 block scales must be nonnegative finite E4M3"));
      }
    }
  }
  std::byte *scratch_data = nullptr;
  if constexpr (!WeightOnly) {
    auto scratch = workspace.map_host(mem::HostAccess::kWrite);
    if (!scratch) return finish(std::move(scratch).error());
    maps.scratch = std::move(*scratch);
    auto bytes = maps.scratch.writable_bytes();
    if (!bytes || bytes->size() < plan.workspace.size ||
        !aligned_for_storage(bytes->data(), alignof(simd::bfloat16_t)))
      return finish(Status::FailedPrecondition("native NVFP4 workspace mapping differs"));
    scratch_data = bytes->data();
  }
  auto mapped = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!mapped) return finish(std::move(mapped).error());
  maps.output = std::move(*mapped);
  auto output = maps.output.writable_bytes();
  const auto output_bytes = *outputs[0].description().required_bytes();
  if (!output || output->size() < output_bytes ||
      !aligned_for_storage(output->data(), alignof(simd::bfloat16_t)))
    return finish(Status::FailedPrecondition("native NVFP4 output mapping differs"));
  if (!bounded(output->data(), output_bytes) ||
      (!WeightOnly && (!bounded(scratch_data, plan.workspace.size) ||
       overlaps(output->data(), output_bytes, scratch_data, plan.workspace.size))))
    return finish(Status::InvalidArgument("native NVFP4 output/workspace overlaps"));
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    const auto data = maps.input[port].bytes();
    if (!bounded(data.data(), data.size()) ||
        overlaps(data.data(), data.size(), output->data(), output_bytes) ||
        (!WeightOnly && overlaps(data.data(), data.size(), scratch_data, plan.workspace.size)))
      return finish(Status::InvalidArgument("native NVFP4 input/output/workspace overlaps"));
  }
  auto *activation = reinterpret_cast<simd::bfloat16_t *>(scratch_data);
  const auto *original_activation = reinterpret_cast<const simd::bfloat16_t *>(
      maps.input[0].bytes().data());
  const auto slots = grouped ? static_cast<std::size_t>(inputs.back().description().dimensions()[1]) : 1;
  const auto quant_slots = static_cast<std::size_t>(plan.scalar_bits);
  const auto quant_rows = (rows / slots) * quant_slots;
  const auto strides = inputs[0].description().byte_strides();
  const auto slot_stride = grouped && inputs[0].description().dimensions()[1] != 1
      ? static_cast<std::size_t>(strides[1]) / sizeof(simd::bfloat16_t) : 0;
  QuantizeEpisode quantize{
      original_activation,
      activation, input_globals, microscaled ? nullptr : ids, quant_slots, k / kGroup,
      static_cast<std::size_t>(strides[0]) / sizeof(simd::bfloat16_t), slot_stride};
  parallel_for_ranges(plan, quant_rows * (k / kGroup),
                       &quantize_groups<Mode, WeightOnly>, &quantize);
  if (quantize.invalid.load(std::memory_order_relaxed))
    return finish(Status::InvalidArgument("NVFP4 requires finite BF16 activation"));
  DotEpisode episode{&plan, WeightOnly ? original_activation : activation,
      reinterpret_cast<const std::uint8_t *>(maps.input[1].bytes().data()),
      scales, weight_globals, ids, reinterpret_cast<simd::bfloat16_t *>(output->data()),
      slots, quant_slots,
      static_cast<std::size_t>(strides[0]) / sizeof(simd::bfloat16_t), slot_stride};
  maps.wrote = true;
  parallel_for_ranges(plan, *outputs[0].description().element_count(),
                      &dot_rows<Mode, WeightOnly>, &episode);
  return finish(Status::Ok());
}
} // namespace

Status prepare_nvfp4(Operation operation, std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs,
                      NativePlan &plan) noexcept {
  const auto nvfp4 = operation.custom_as<model_ops::Nvfp4Linear>();
  const auto grouped = operation.custom_as<model_ops::GroupedNvfp4Linear>();
  const auto local = operation.custom_as<model_ops::LocalNvfp4Linear>();
  const auto local_grouped = operation.custom_as<model_ops::GroupedLocalNvfp4Linear>();
  const auto mx = operation.custom_as<model_ops::Mxfp4Linear>();
  const auto mx_grouped = operation.custom_as<model_ops::GroupedMxfp4Linear>();
  if (!nvfp4 && !grouped && !local && !local_grouped && !mx && !mx_grouped)
    return Status::Unimplemented("operation is not NVFP4 reference linear");
  const bool is_grouped = grouped || local_grouped || mx_grouped;
  const bool microscaled = mx || mx_grouped;
  const bool weight_only = microscaled &&
      (mx ? mx->activation_mode : mx_grouped->activation_mode) ==
          model_ops::Mxfp4ActivationMode::kBFloat16;
  std::array<TensorDesc, 1> inferred;
  const auto status = mx_grouped ? mx_grouped->infer_output_descriptions(inputs, inferred)
      : mx ? mx->infer_output_descriptions(inputs, inferred)
      : local_grouped ? local_grouped->infer_output_descriptions(inputs, inferred)
      : local ? local->infer_output_descriptions(inputs, inferred)
      : grouped ? grouped->infer_output_descriptions(inputs, inferred)
                : nvfp4->infer_output_descriptions(inputs, inferred);
  HYLE_RETURN_IF_ERROR(status);
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("native NVFP4 output description differs");
  const auto tokens = static_cast<std::size_t>(inputs[0].dimensions()[0]);
  const auto slots = is_grouped ? static_cast<std::size_t>(inputs.back().dimensions()[1]) : 1;
  const auto n = static_cast<std::size_t>(inputs[1].dimensions()[is_grouped ? 1 : 0]);
  const auto k = static_cast<std::size_t>(inputs[0].dimensions()[is_grouped ? 2 : 1]);
  const auto quant_slots = microscaled &&
      (!is_grouped || inputs[0].dimensions()[1] == 1 || inputs[0].byte_strides()[1] == 0)
      ? 1 : slots;
  if (tokens > SIZE_MAX / slots || (!weight_only &&
      tokens > SIZE_MAX / quant_slots / sizeof(simd::bfloat16_t) / k))
    return Status::OutOfRange("native NVFP4 selected activation workspace overflows");
  const auto rows = tokens * slots;
  plan.parameters = {rows, n, k};
  // Cold family metadata: NVFP4 depends on each selected IG; MXFP4 shared A
  // instead has one quantized row per token. No new common-plan field needed.
  plan.scalar_bits = quant_slots;
  plan.workspace = {.size = weight_only ? 0 : tokens * quant_slots * k * sizeof(simd::bfloat16_t),
                    .alignment = alignof(simd::bfloat16_t)};
  if (microscaled) {
    plan.workspace_submit_route = weight_only ? &submit_nvfp4<Math::kMxfp4, true>
                                              : &submit_nvfp4<Math::kMxfp4>;
  } else if (local || local_grouped) {
    const auto rounding = local ? local->rounding : local_grouped->rounding;
    plan.workspace_submit_route = rounding == model_ops::Fp4InputRounding::kFloat32
        ? &submit_nvfp4<Math::kLocalFloat32>
        : &submit_nvfp4<Math::kLocalBFloat16>;
  } else {
    plan.workspace_submit_route = &submit_nvfp4<Math::kModelOpt>;
  }
  return Status::Ok();
}
} // namespace hyle::compute::native::internal
