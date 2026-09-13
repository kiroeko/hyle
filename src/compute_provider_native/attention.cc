#include "native_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {

Status require_plain_f32(const TensorDesc &description,
                         const char *message) noexcept {
  if (description.data_type() != DataType::kFloat32 ||
      !description.is_plain()) {
    return Status::Unimplemented(message);
  }
  return Status::Ok();
}

Status invalidate_for_read(mem::ConstBufferView storage) noexcept {
  if (storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    return Status::Ok();
  }
  return storage.invalidate();
}

float load_f32(const std::byte *data, std::size_t offset) noexcept {
  float value = 0.0F;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

inline constexpr std::size_t kAttentionScoreTile = 64;

std::size_t attention_linear_index(AttentionLayout layout, std::size_t batch,
                                   std::size_t sequence, std::size_t head,
                                   std::size_t depth, std::size_t sequences,
                                   std::size_t heads,
                                   std::size_t row_depth) noexcept {
  if (layout == AttentionLayout::kBatchSequenceHeadDepth) {
    return ((batch * sequences + sequence) * heads + head) * row_depth + depth;
  }
  return ((batch * heads + head) * sequences + sequence) * row_depth + depth;
}

bool causal_key_visible(std::size_t query, std::size_t key, std::size_t queries,
                        std::size_t keys) noexcept {
  // Bottom-right alignment is the natural autoregressive contract: a one-row
  // decode query sees the complete existing KV prefix. More general position
  // policies remain expressible through the explicit mask input.
  if (queries <= keys)
    return key <= keys - queries + query;
  const std::size_t empty_prefix = queries - keys;
  return query >= empty_prefix && key <= query - empty_prefix;
}

struct AttentionEpisode final {
  const NativePlan *plan = nullptr;
  std::array<const TensorDesc *, 4> descriptions{};
  std::array<const std::byte *, 4> inputs{};
  std::array<std::int64_t, 4> score_shape{};
  float *output = nullptr;
  std::byte *workspace = nullptr;
};

float attention_dot(const AttentionEpisode &episode, std::size_t batch,
                    std::size_t head, std::size_t query,
                    std::size_t key) noexcept {
  const NativePlan &plan = *episode.plan;
  if (plan.attention_query_depth == 0)
    return 0.0F;
  const TensorDesc &query_desc = *episode.descriptions[0];
  const TensorDesc &key_desc = *episode.descriptions[1];
  const std::size_t query_base = attention_linear_index(
      plan.attention_layout, batch, query, head, 0, plan.attention_queries,
      plan.attention_heads, plan.attention_query_depth);
  const std::size_t key_base = attention_linear_index(
      plan.attention_layout, batch, key, head, 0, plan.attention_keys,
      plan.attention_heads, plan.attention_query_depth);
  if (query_desc.is_dense() && key_desc.is_dense()) {
    const auto *query_row =
        reinterpret_cast<const float *>(episode.inputs[0]) + query_base;
    const auto *key_row =
        reinterpret_cast<const float *>(episode.inputs[1]) + key_base;
    return simd::dot(query_row, key_row, plan.attention_query_depth);
  }
  float score = 0.0F;
  for (std::size_t depth = 0; depth < plan.attention_query_depth; ++depth) {
    score += load_f32(episode.inputs[0],
                      logical_byte_offset(query_desc, query_base + depth)) *
             load_f32(episode.inputs[1],
                      logical_byte_offset(key_desc, key_base + depth));
  }
  return score;
}

float attention_mask_value(const AttentionEpisode &episode,
                           std::size_t score_linear) noexcept {
  const NativePlan &plan = *episode.plan;
  if (plan.mask_kind == MaskKind::kNone)
    return 0.0F;
  const TensorDesc &mask_desc = *episode.descriptions[3];
  const std::size_t offset = logical_broadcast_byte_offset(
      mask_desc, episode.score_shape, score_linear);
  if (plan.mask_kind == MaskKind::kBoolean) {
    return episode.inputs[3][offset] == std::byte{0}
               ? -std::numeric_limits<float>::infinity()
               : 0.0F;
  }
  return load_f32(episode.inputs[3], offset);
}

void accumulate_attention_value(const AttentionEpisode &episode,
                                std::size_t batch, std::size_t head,
                                std::size_t key, float weight,
                                float *accumulator) noexcept {
  const NativePlan &plan = *episode.plan;
  const TensorDesc &value_desc = *episode.descriptions[2];
  const std::size_t value_base = attention_linear_index(
      plan.attention_layout, batch, key, head, 0, plan.attention_keys,
      plan.attention_heads, plan.attention_value_depth);
  if (value_desc.is_dense()) {
    const auto *value_row =
        reinterpret_cast<const float *>(episode.inputs[2]) + value_base;
    simd::saxpy(weight, value_row, accumulator, plan.attention_value_depth);
    return;
  }
  for (std::size_t depth = 0; depth < plan.attention_value_depth; ++depth) {
    accumulator[depth] +=
        weight * load_f32(episode.inputs[2],
                          logical_byte_offset(value_desc, value_base + depth));
  }
}

void write_invalid_attention_row(const NativePlan &plan, float *output,
                                 std::size_t batch, std::size_t head,
                                 std::size_t query) noexcept {
  const std::size_t output_base = attention_linear_index(
      plan.attention_layout, batch, query, head, 0, plan.attention_queries,
      plan.attention_heads, plan.attention_value_depth);
  std::fill_n(output + output_base, plan.attention_value_depth,
              std::numeric_limits<float>::quiet_NaN());
}

void run_attention_range(void *context, std::size_t task_index,
                         std::size_t begin, std::size_t end) noexcept {
  auto &episode = *static_cast<AttentionEpisode *>(context);
  const NativePlan &plan = *episode.plan;
  std::byte *scratch =
      episode.workspace + task_index * plan.attention_workspace_stride;
  auto *accumulator = reinterpret_cast<float *>(scratch);
  auto *scores = reinterpret_cast<float *>(
      scratch + plan.attention_value_depth * sizeof(float));

  for (std::size_t unit = begin; unit < end; ++unit) {
    const std::size_t query = unit % plan.attention_queries;
    const std::size_t head_block = unit / plan.attention_queries;
    const std::size_t head = head_block % plan.attention_heads;
    const std::size_t batch = head_block / plan.attention_heads;
    std::fill_n(accumulator, plan.attention_value_depth, 0.0F);
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0F;
    bool valid = true;

    for (std::size_t tile_begin = 0; tile_begin < plan.attention_keys;
         tile_begin += kAttentionScoreTile) {
      const std::size_t tile_size =
          (std::min)(kAttentionScoreTile, plan.attention_keys - tile_begin);
      float tile_max = -std::numeric_limits<float>::infinity();
      for (std::size_t lane = 0; lane < tile_size; ++lane) {
        const std::size_t key = tile_begin + lane;
        float score = -std::numeric_limits<float>::infinity();
        if (!plan.causal ||
            causal_key_visible(query, key, plan.attention_queries,
                               plan.attention_keys)) {
          score =
              attention_dot(episode, batch, head, query, key) * plan.scalar +
              attention_mask_value(episode,
                                   ((batch * plan.attention_heads + head) *
                                        plan.attention_queries +
                                    query) *
                                           plan.attention_keys +
                                       key);
        }
        scores[lane] = score;
        if (std::isnan(score) ||
            score == std::numeric_limits<float>::infinity()) {
          valid = false;
          break;
        }
        if (std::isfinite(score))
          tile_max = (std::max)(tile_max, score);
      }
      if (!valid)
        break;
      if (!std::isfinite(tile_max))
        continue;

      const float next_max = (std::max)(running_max, tile_max);
      const float old_scale =
          running_sum == 0.0F ? 0.0F : std::exp(running_max - next_max);
      if (plan.attention_value_depth != 0) {
        if (old_scale == 0.0F)
          std::fill_n(accumulator, plan.attention_value_depth, 0.0F);
        else
          simd::scale(accumulator, old_scale, accumulator,
                      plan.attention_value_depth);
      }
      simd::add_scalar(scores, -next_max, scores, tile_size);
      simd::exp_f32_batch(scores, scores, tile_size);
      const float tile_sum = simd::sum(scores, tile_size);
      for (std::size_t lane = 0; lane < tile_size; ++lane) {
        if (scores[lane] != 0.0F) {
          accumulate_attention_value(episode, batch, head, tile_begin + lane,
                                     scores[lane], accumulator);
        }
      }
      running_sum = running_sum * old_scale + tile_sum;
      running_max = next_max;
    }

    if (!valid || !(running_sum > 0.0F) || !std::isfinite(running_sum)) {
      write_invalid_attention_row(plan, episode.output, batch, head, query);
      continue;
    }
    const std::size_t output_base = attention_linear_index(
        plan.attention_layout, batch, query, head, 0, plan.attention_queries,
        plan.attention_heads, plan.attention_value_depth);
    simd::scale(accumulator, 1.0F / running_sum, episode.output + output_base,
                plan.attention_value_depth);
  }
}

detail::EnqueueResult submit_attention(const NativePlan &plan,
                                       std::span<const ConstTensorView> inputs,
                                       std::span<const TensorView> outputs,
                                       mem::BufferView workspace) noexcept {
  if ((inputs.size() != 3 && inputs.size() != 4) || outputs.size() != 1 ||
      plan.attention_workspace_stride == 0) {
    return completed(Status::Internal("native SDPA plan is invalid"));
  }
  auto output_elements = outputs[0].description().element_count();
  if (!output_elements)
    return completed(std::move(output_elements).error());
  if (*output_elements == 0)
    return completed(Status::Ok());

  std::array<mem::HostMapping, 4> input_mappings{};
  std::array<const std::byte *, 4> input_data{};
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    Status status = invalidate_for_read(inputs[index].storage());
    if (!status.ok()) {
      for (mem::HostMapping &mapping : input_mappings)
        mapping.close().IgnoreError();
      return completed(std::move(status));
    }
    auto mapping = inputs[index].storage().map_host();
    if (!mapping) {
      for (mem::HostMapping &opened : input_mappings)
        opened.close().IgnoreError();
      return completed(std::move(mapping).error());
    }
    input_mappings[index] = std::move(*mapping);
    input_data[index] = input_mappings[index].bytes().data();
  }

  const mem::BufferView output_storage = outputs[0].storage();
  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    for (mem::HostMapping &mapping : input_mappings)
      mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto workspace_result = workspace.map_host(mem::HostAccess::kWrite);
  if (!workspace_result) {
    output_mapping.close().IgnoreError();
    for (mem::HostMapping &mapping : input_mappings)
      mapping.close().IgnoreError();
    return completed(std::move(workspace_result).error());
  }
  mem::HostMapping workspace_mapping = std::move(*workspace_result);
  auto output_bytes = output_mapping.writable_bytes();
  auto scratch_bytes = workspace_mapping.writable_bytes();
  if (!output_bytes || !scratch_bytes) {
    workspace_mapping.close().IgnoreError();
    output_mapping.close().IgnoreError();
    for (mem::HostMapping &mapping : input_mappings)
      mapping.close().IgnoreError();
    return completed(
        Status::FailedPrecondition("CPU SDPA writable mapping is unavailable"));
  }
  auto *output = reinterpret_cast<float *>(output_bytes->data());
  std::byte *scratch = scratch_bytes->data();
  bool aligned = aligned_for_storage(
      output, outputs[0].description().required_alignment());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    aligned = aligned && aligned_for_storage(
                             input_data[index],
                             inputs[index].description().required_alignment());
  }
  aligned = aligned && aligned_for_storage(scratch, plan.workspace.alignment);
  if (!aligned) {
    workspace_mapping.close().IgnoreError();
    output_mapping.close().IgnoreError();
    for (mem::HostMapping &mapping : input_mappings)
      mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU SDPA mapping does not satisfy tensor alignment"));
  }

  AttentionEpisode episode{
      .plan = &plan,
      .descriptions = {&inputs[0].description(), &inputs[1].description(),
                       &inputs[2].description(),
                       inputs.size() == 4 ? &inputs[3].description() : nullptr},
      .inputs = input_data,
      .score_shape = {static_cast<std::int64_t>(plan.attention_batches),
                      static_cast<std::int64_t>(plan.attention_heads),
                      static_cast<std::int64_t>(plan.attention_queries),
                      static_cast<std::int64_t>(plan.attention_keys)},
      .output = output,
      .workspace = scratch,
  };
  parallel_for_ranges(plan, plan.parameters[0], &run_attention_range, &episode);

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, workspace_mapping.close());
  preserve_first_error(terminal, output_mapping.close());
  for (mem::HostMapping &mapping : input_mappings)
    preserve_first_error(terminal, mapping.close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_attention(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept {
  switch (operation.kind()) {
  case OperationKind::kScaledDotProductAttention: {
    if ((inputs.size() != 3 && inputs.size() != 4) || outputs.size() != 1)
      return Status::Internal("native SDPA arity is invalid");
    for (std::size_t index = 0; index < 3; ++index) {
      Status status = require_plain_f32(
          inputs[index],
          "native SDPA currently requires plain float32 Q/K/V tensors");
      if (!status.ok())
        return status;
    }
    Status status = require_plain_f32(
        outputs[0], "native SDPA currently requires float32 output");
    if (!status.ok() || !outputs[0].is_dense()) {
      return status.ok()
                 ? Status::Unimplemented("native SDPA requires dense output")
                 : status;
    }
    const ScaledDotProductAttention *attention =
        operation.get_if<ScaledDotProductAttention>();
    if (attention == nullptr)
      return Status::Internal("native SDPA descriptor is missing");
    if (attention->accumulation_type != DataType::kUnknown &&
        attention->accumulation_type != DataType::kFloat32) {
      return Status::Unimplemented(
          "native SDPA currently supports float32 accumulation");
    }
    if (attention->mask != MaskKind::kNone) {
      if (inputs.size() != 4 || !inputs[3].is_plain() ||
          (attention->mask == MaskKind::kBoolean &&
           inputs[3].data_type() != DataType::kBool) ||
          (attention->mask == MaskKind::kAdditive &&
           inputs[3].data_type() != DataType::kFloat32)) {
        return Status::Unimplemented(
            "native SDPA requires a plain Bool or float32 mask");
      }
    }

    const std::size_t sequence_axis =
        attention->layout == AttentionLayout::kBatchSequenceHeadDepth ? 1 : 2;
    const std::size_t head_axis =
        attention->layout == AttentionLayout::kBatchSequenceHeadDepth ? 2 : 1;
    auto batches =
        math::try_integer_cast<std::size_t>(inputs[0].dimensions()[0]);
    auto heads =
        math::try_integer_cast<std::size_t>(inputs[0].dimensions()[head_axis]);
    auto queries = math::try_integer_cast<std::size_t>(
        inputs[0].dimensions()[sequence_axis]);
    auto keys = math::try_integer_cast<std::size_t>(
        inputs[1].dimensions()[sequence_axis]);
    auto query_depth =
        math::try_integer_cast<std::size_t>(inputs[0].dimensions()[3]);
    auto value_depth =
        math::try_integer_cast<std::size_t>(inputs[2].dimensions()[3]);
    if (!batches || !heads || !queries || !keys || !query_depth ||
        !value_depth) {
      return Status::OutOfRange("native SDPA dimension exceeds size_t");
    }
    auto batch_heads = math::try_multiply(*batches, *heads);
    auto units = batch_heads ? math::try_multiply(*batch_heads, *queries)
                             : std::optional<std::size_t>{};
    auto score_elements = units ? math::try_multiply(*units, *keys)
                                : std::optional<std::size_t>{};
    auto accumulator_bytes = math::try_multiply(*value_depth, sizeof(float));
    constexpr std::size_t score_bytes = kAttentionScoreTile * sizeof(float);
    auto scratch_bytes = accumulator_bytes
                             ? math::try_add(*accumulator_bytes, score_bytes)
                             : std::optional<std::size_t>{};
    auto padded_bytes = scratch_bytes
                            ? math::try_add(*scratch_bytes, std::size_t{63})
                            : std::optional<std::size_t>{};
    if (!units || !score_elements || !padded_bytes)
      return Status::OutOfRange("native SDPA workspace size overflows");
    const std::size_t workspace_stride = *padded_bytes & ~std::size_t{63};
    plan.parameters[0] = *units;
    plan.scalar = attention->scale;
    plan.mask_kind = attention->mask;
    plan.attention_layout = attention->layout;
    plan.causal = attention->causal;
    plan.attention_batches = *batches;
    plan.attention_heads = *heads;
    plan.attention_queries = *queries;
    plan.attention_keys = *keys;
    plan.attention_query_depth = *query_depth;
    plan.attention_value_depth = *value_depth;
    plan.attention_workspace_stride = workspace_stride;
    plan.workspace = WorkspaceRequirement{
        .size = workspace_stride,
        .alignment = 64,
    };
    plan.workspace_submit_route = &submit_attention;
    return Status::Ok();
  }
  default:
    return Status::Unimplemented("operation is not native attention");
  }
}

} // namespace hyle::compute::native::internal
