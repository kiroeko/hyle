#include "native_plan.h"
#include "numeric.h"

#include "../compute_packed_shared/artifact_packed_linear.h"
#include "hyle/simd/ops.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

namespace hyle::compute::native::internal {
namespace {
constexpr std::size_t kTile = 64;
using ArtifactPackedLinear = packed::internal::ArtifactPackedLinear;

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
  std::array<mem::HostMapping, 7> inputs;
  mem::HostMapping output;
  bool wrote = false;
  Status finish(Status status, mem::BufferView destination) noexcept {
    if (wrote && !destination.resource().properties().has(
                     mem::MemoryProperty::kHostDeviceCoherent))
      preserve_first_error(status, output.flush());
    preserve_first_error(status, output.close());
    for (auto &mapping : inputs)
      preserve_first_error(status, mapping.close());
    return status;
  }
};

template <class Scale, class Decoded, class Operand, bool Grouped, bool Nf4 = false>
struct Episode final {
  using Activation = std::conditional_t<Grouped, float, Operand>;
  using Output = std::conditional_t<Grouped, float, Operand>;
  const NativePlan *plan;
  const Activation *activation;
  const std::uint32_t *codes, *zeros;
  const std::int32_t *indices;
  const Scale *scales;
  Output *output;
  std::size_t slots;
  const std::uint8_t *nf4_codes, *nf4_absmax;
  const float *nf4_nested_scale, *nf4_nested_map, *nf4_offset, *nf4_map;
};

template <class Scale, class Decoded, class Operand, bool Grouped, bool Nf4 = false>
void dot_rows(void *state, std::size_t, std::size_t begin,
              std::size_t end) noexcept {
  const auto &episode = *static_cast<const Episode<Scale, Decoded, Operand, Grouped, Nf4> *>(state);
  const auto n = episode.plan->parameters[1], k = episode.plan->parameters[2];
  const auto group_size = episode.plan->axis, groups = k / group_size;
  std::array<float, kTile> activation{}, weight{};
  for (auto index = begin; index < end; ++index) {
    const auto row = index / n, column = index % n;
    const auto weight_row = Grouped
        ? static_cast<std::size_t>(episode.indices[row]) * n + column : column;
    float sum = 0.F;
    for (std::size_t group = 0; group < groups; ++group) {
      const bool gptq = episode.plan->scalar_bits == 2;
      float group_scale;
      if constexpr (Nf4) {
        const auto original_group = weight_row * groups + group;
        // Original nested dequantization has two separate F32 roundings.
        volatile float nested_product = episode.nf4_nested_map[episode.nf4_absmax[original_group]] *
            episode.nf4_nested_scale[original_group / 256];
        volatile float nested_sum = nested_product + *episode.nf4_offset;
        group_scale = nested_sum;
      } else {
        group_scale = gptq ? 0.F : numeric::decode(episode.scales[weight_row * groups + group]);
      }
      const auto group_zero = gptq ? 0U : episode.zeros
          ? (episode.zeros[(weight_row / 8) * groups + group] >> (4 * (weight_row % 8))) & 15U
          : 8U;
      const auto group_begin = group * group_size;
      for (std::size_t offset = 0; offset < group_size; offset += kTile) {
        const auto count = (std::min)(kTile, group_size - offset);
        const auto first = group_begin + offset;
        if constexpr (Grouped) {
          const auto *source = reinterpret_cast<const std::byte *>(episode.activation) +
              (row / episode.slots) * episode.plan->row_first_stride +
              (row % episode.slots) * episode.plan->row_second_stride;
          for (std::size_t lane = 0; lane < count; ++lane)
            std::memcpy(&activation[lane],
                source + (first + lane) * episode.plan->row_output_stride,
                sizeof(float));
        } else {
          numeric::load_tile<Operand>(episode.activation, row * k + first,
                                     activation.data(), count);
        }
        for (std::size_t lane = 0; lane < count; ++lane) {
          const auto depth = first + lane;
          const auto selected = gptq ? static_cast<std::size_t>(episode.indices[depth]) : group;
          const float scale = gptq ? numeric::decode(episode.scales[selected * n + column]) : group_scale;
          const auto zero = gptq
              ? ((episode.zeros[selected * (n / 8) + column / 8] >> (4 * (column % 8))) & 15U) + 1U
              : group_zero;
          unsigned code;
          if constexpr (Nf4) {
            const auto original = weight_row * k + depth;
            code = (episode.nf4_codes[original / 2] >> (original % 2 ? 0 : 4)) & 15U;
          } else {
            code = (episode.codes[gptq ? (depth / 8) * n + column
                                      : weight_row * (k / 8) + depth / 8] >>
                    (4 * (depth % 8))) & 15U;
          }
          // The centered integer is exact in every admitted scale dtype.
          // Keep the F32 product boundary even when all following casts are
          // F32; it must not become part of the dot's fused multiply-add.
          volatile float product = (Nf4 ? episode.nf4_map[code]
                                       : float(int(code) - int(zero))) * scale;
          const auto source = numeric::encode<Scale>(product);
          const auto decoded = numeric::encode<Decoded>(numeric::decode(source));
          const auto operand = numeric::encode<Operand>(numeric::decode(decoded));
          weight[lane] = numeric::decode(operand);
        }
        sum += simd::dot(activation.data(), weight.data(), count);
      }
    }
    episode.output[index] = numeric::encode<typename Episode<Scale, Decoded, Operand, Grouped, Nf4>::Output>(sum);
  }
}

template <class Scale, class Decoded, class Operand, bool Grouped, bool Nf4 = false>
detail::EnqueueResult submit_ct_packed(const NativePlan &plan,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  const bool gptq = plan.scalar_bits == 2;
  const bool asymmetric = plan.scalar_bits == 1;
  if (inputs.size() != (Nf4 ? 7U : gptq ? 5U : asymmetric ? 4U : 3U) + Grouped || outputs.size() != 1)
    return completed(Status::Internal("native CT packed binding count differs"));
  Mappings maps;
  const auto finish = [&](Status status) {
    return completed(maps.finish(std::move(status), outputs[0].storage()));
  };
  std::array<std::size_t, 7> sizes{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto storage = inputs[i].storage();
    if (!storage.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
      auto status = storage.invalidate();
      if (!status.ok()) return finish(std::move(status));
    }
    auto mapped = storage.map_host();
    if (!mapped) return finish(std::move(mapped).error());
    maps.inputs[i] = std::move(*mapped);
    sizes[i] = *inputs[i].description().required_bytes();
    const auto data = maps.inputs[i].bytes();
    if (data.size() < sizes[i] || !bounded(data.data(), sizes[i]) ||
        !aligned_for_storage(data.data(), inputs[i].description().required_alignment()))
      return finish(Status::FailedPrecondition("native CT packed input mapping differs"));
  }
  const auto *indices = gptq || Grouped
      ? reinterpret_cast<const std::int32_t *>(maps.inputs[gptq ? 4 : inputs.size() - 1].bytes().data()) : nullptr;
  if (gptq || Grouped) {
    const auto limit = Grouped
        ? static_cast<std::size_t>(inputs[1].description().dimensions()[0]) / plan.parameters[1]
        : plan.parameters[2] / plan.axis;
    const auto count = Grouped ? plan.parameters[0] : plan.parameters[2];
    // Validate the complete current index payload before any derived SF/Z
    // address, not just the tile which happens to execute first.
    for (std::size_t i = 0; i < count; ++i)
      if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= limit)
        return finish(Status::OutOfRange("packed artifact index is outside the original bank"));
  }
  auto mapped = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!mapped) return finish(std::move(mapped).error());
  maps.output = std::move(*mapped);
  auto output = maps.output.writable_bytes();
  const auto bytes = *outputs[0].description().required_bytes();
  if (!output || output->size() < bytes || !bounded(output->data(), bytes) ||
      !aligned_for_storage(output->data(), alignof(typename Episode<Scale, Decoded, Operand, Grouped, Nf4>::Output)))
    return finish(Status::FailedPrecondition("native CT packed output mapping differs"));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (overlaps(maps.inputs[i].bytes().data(), sizes[i], output->data(), bytes))
      return finish(Status::InvalidArgument("native CT packed input/output overlaps"));
  using CurrentEpisode = Episode<Scale, Decoded, Operand, Grouped, Nf4>;
  CurrentEpisode episode{&plan,
      reinterpret_cast<const typename CurrentEpisode::Activation *>(maps.inputs[0].bytes().data()),
      reinterpret_cast<const std::uint32_t *>(maps.inputs[1].bytes().data()),
      gptq || asymmetric ? reinterpret_cast<const std::uint32_t *>(maps.inputs[2].bytes().data()) : nullptr,
      indices,
      reinterpret_cast<const Scale *>(maps.inputs[gptq ? 3 : inputs.size() - 1 - Grouped].bytes().data()),
      reinterpret_cast<typename CurrentEpisode::Output *>(output->data()),
      Grouped ? static_cast<std::size_t>(inputs.back().description().dimensions()[1]) : 1,
      Nf4 ? reinterpret_cast<const std::uint8_t *>(maps.inputs[1].bytes().data()) : nullptr,
      Nf4 ? reinterpret_cast<const std::uint8_t *>(maps.inputs[2].bytes().data()) : nullptr,
      Nf4 ? reinterpret_cast<const float *>(maps.inputs[3].bytes().data()) : nullptr,
      Nf4 ? reinterpret_cast<const float *>(maps.inputs[4].bytes().data()) : nullptr,
      Nf4 ? reinterpret_cast<const float *>(maps.inputs[5].bytes().data()) : nullptr,
      Nf4 ? reinterpret_cast<const float *>(maps.inputs[6].bytes().data()) : nullptr};
  maps.wrote = true;
  parallel_for_ranges(plan, *outputs[0].description().element_count(),
                       &dot_rows<Scale, Decoded, Operand, Grouped, Nf4>, &episode);
  return finish(Status::Ok());
}

template <class Scale, class Decoded, bool Grouped, bool Nf4 = false>
SubmitRoute operand_route(DataType type) noexcept {
  if (type == DataType::kFloat16)
    return &submit_ct_packed<Scale, Decoded, simd::float16_t, Grouped, Nf4>;
  if (type == DataType::kBFloat16)
    return &submit_ct_packed<Scale, Decoded, simd::bfloat16_t, Grouped, Nf4>;
  return &submit_ct_packed<Scale, Decoded, float, Grouped, Nf4>;
}
template <class Scale, bool Grouped = false, bool Nf4 = false>
SubmitRoute decode_route(const ArtifactPackedLinear &operation) noexcept {
  if (operation.decode_type == DataType::kFloat16)
    return operand_route<Scale, simd::float16_t, Grouped, Nf4>(operation.operand_type);
  if (operation.decode_type == DataType::kBFloat16)
    return operand_route<Scale, simd::bfloat16_t, Grouped, Nf4>(operation.operand_type);
  return operand_route<Scale, float, Grouped, Nf4>(operation.operand_type);
}
} // namespace

Status prepare_ct_packed(Operation operation, std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept {
  const auto ct = operation.custom_as<ArtifactPackedLinear>();
  if (!ct) return Status::Unimplemented("operation is not CT packed linear");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(ct->infer_output_descriptions(inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("native CT packed output description differs");
  const bool grouped = ct->expert_count != 0;
  if (!inputs[0].is_plain() || (!grouped && (!inputs[0].is_dense() ||
      inputs[0].rank() != 2 || inputs[0].dimensions()[0] <= 0)))
    return Status::Unimplemented("native CT packed linear requires positive dense rank2 activation");
  const bool nf4 = ct->format == model_ops::WeightArtifactFormat::kNf4DoubleQuant;
  const auto columns = grouped ? static_cast<std::size_t>(outputs[0].dimensions()[2])
      : static_cast<std::size_t>(outputs[0].dimensions()[1]);
  plan.parameters = {grouped ? *outputs[0].element_count() / columns
                            : static_cast<std::size_t>(inputs[0].dimensions()[0]),
      columns, static_cast<std::size_t>(inputs[0].dimensions().back())};
  plan.axis = ct->group_size;
  const bool gptq = ct->format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  plan.scalar_bits = nf4 ? 3 : gptq ? 2 : ct->format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4;
  const auto scale = inputs[gptq ? 3 : inputs.size() - 1 - grouped].data_type();
  if (nf4) {
    plan.submit_route = decode_route<float, false, true>(*ct);
  } else if (grouped) {
    plan.row_first_stride = static_cast<std::size_t>(inputs[0].byte_strides()[0]);
    plan.row_second_stride = inputs[0].dimensions()[1] == 1 ? 0
        : static_cast<std::size_t>(inputs[0].byte_strides()[1]);
    plan.row_output_stride = static_cast<std::size_t>(inputs[0].byte_strides()[2]);
    plan.submit_route = scale == DataType::kFloat16 ? decode_route<simd::float16_t, true>(*ct)
        : scale == DataType::kBFloat16 ? decode_route<simd::bfloat16_t, true>(*ct)
                                     : decode_route<float, true>(*ct);
  } else {
    plan.submit_route = scale == DataType::kFloat16 ? decode_route<simd::float16_t>(*ct)
      : scale == DataType::kBFloat16 ? decode_route<simd::bfloat16_t>(*ct)
                                   : decode_route<float>(*ct);
  }
  return Status::Ok();
}
} // namespace hyle::compute::native::internal
