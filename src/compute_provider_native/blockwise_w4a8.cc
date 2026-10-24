#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hyle/compute/model_ops.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/ops.h"

namespace hyle::compute::native::internal {
namespace {
using BFloat = simd::bfloat16_t;
constexpr std::size_t kGroup = 512;
float bf(float v) noexcept { return simd::bf16_to_f32(simd::f32_to_bf16(v)); }
bool bounded(const void *p, std::size_t n) noexcept {
  return p && n <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(p);
}
bool overlaps(const void *a, std::size_t na, const void *b, std::size_t nb) noexcept {
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  return x < y + nb && y < x + na;
}
struct Mappings final {
  std::array<mem::HostMapping, 4> input;
  mem::HostMapping scratch, output;
  bool wrote = false;
  Status finish(Status s, mem::BufferView destination) noexcept {
    if (wrote && !destination.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      preserve_first_error(s, output.flush());
    preserve_first_error(s, output.close());
    preserve_first_error(s, scratch.close());
    for (auto &m : input) preserve_first_error(s, m.close());
    return s;
  }
};
struct Quantize final {
  const BFloat *input;
  std::int8_t *codes;
  BFloat *scales;
  std::size_t groups, slots, token_stride, slot_stride;
  std::atomic<bool> invalid{false};
};
void quantize(void *state, std::size_t, std::size_t begin, std::size_t end) noexcept {
  auto &p = *static_cast<Quantize *>(state);
  for (auto block = begin; block < end; ++block) {
    const auto row = block / p.groups, g = block % p.groups;
    const auto *a = p.input + (row / p.slots) * p.token_stride +
        (row % p.slots) * p.slot_stride + g * kGroup;
    float maximum = 0.F;
    bool valid = true;
    for (std::size_t i = 0; i < kGroup; ++i) {
      const auto v = simd::bf16_to_f32(a[i]);
      valid &= std::isfinite(v);
      maximum = (std::max)(maximum, std::abs(v));
    }
    // Separate operations keep all source BF16 rounding points, including
    // the multiplication by the already-rounded reciprocal constant.
    volatile float product = maximum * 0.00787353515625F;
    const float scale = bf(product);
    volatile float reciprocal = scale == 0.F ? 0.F : 1.F / scale;
    const float inv = bf(reciprocal);
    p.scales[block] = simd::f32_to_bf16(scale);
    for (std::size_t i = 0; i < kGroup; ++i) {
      volatile float normalized = simd::bf16_to_f32(a[i]) * inv;
      const float value = bf(normalized);
      // Check the truncated value, so e.g. -128.5 remains a legal -128.
      // Never perform an undefined out-of-range C++ float-to-I8 cast.
      if (!std::isfinite(value) || !(value > -129.F && value < 128.F)) {
        valid = false;
        p.codes[block * kGroup + i] = 0;
      } else {
        p.codes[block * kGroup + i] = static_cast<std::int8_t>(value);
      }
    }
    if (!valid) p.invalid.store(true, std::memory_order_relaxed);
  }
}
struct Dot final {
  const std::int8_t *a;
  const BFloat *activation_scales;
  const std::uint32_t *w;
  const float *scales;
  const std::int32_t *ids;
  BFloat *output;
  std::size_t columns, depth, slots, quant_slots;
};
void dot(void *state, std::size_t, std::size_t begin, std::size_t end) noexcept {
  auto &p = *static_cast<Dot *>(state);
  const auto groups = p.depth / kGroup;
  for (auto index = begin; index < end; ++index) {
    const auto row = index / p.columns, column = index % p.columns;
    const auto expert = p.ids ? static_cast<std::size_t>(p.ids[row]) : 0;
    const auto qrow = (row / p.slots) * p.quant_slots +
        (p.quant_slots == 1 ? 0 : row % p.slots);
    const auto *a = p.a + qrow * p.depth;
    const auto *w = p.w + (expert * p.columns + column) * (p.depth / 8);
    const auto *sf = p.scales + (expert * p.columns + column) * groups;
    float sum = 0.F;
    for (std::size_t g = 0; g < groups; ++g) {
      std::int32_t total = 0;
      for (std::size_t i = 0; i < kGroup; ++i) {
        const auto k = g * kGroup + i;
        const auto code = static_cast<std::int32_t>((w[k / 8] >> ((k % 8) * 4)) & 15u) - 8;
        total += static_cast<std::int32_t>(a[k]) * code;
      }
      const float d = bf(static_cast<float>(total));
      volatile float first = d * simd::bf16_to_f32(p.activation_scales[qrow * groups + g]);
      const float partial = bf(first);
      volatile float second = partial * bf(sf[g]);
      const float value = bf(second);
      volatile float added = sum + value;
      sum = bf(added);
    }
    p.output[index] = simd::f32_to_bf16(sum);
  }
}

detail::EnqueueResult submit(const NativePlan &plan,
    std::span<const ConstTensorView> in, std::span<const TensorView> out,
    mem::BufferView workspace) noexcept {
  if ((in.size() != 3 && in.size() != 4) || out.size() != 1)
    return completed(Status::Internal("blockwise W4A8 binding counts differ"));
  Mappings maps;
  const auto finish = [&](Status s) { return completed(maps.finish(std::move(s), out[0].storage())); };
  for (std::size_t i = 0; i < in.size(); ++i) {
    const auto storage = in[i].storage();
    if (!storage.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
      const auto s = storage.invalidate();
      if (!s.ok()) return finish(s);
    }
    auto mapped = storage.map_host();
    if (!mapped) return finish(mapped.error());
    maps.input[i] = std::move(*mapped);
    if (!aligned_for_storage(maps.input[i].bytes().data(), in[i].description().required_alignment()))
      return finish(Status::FailedPrecondition("blockwise W4A8 input mapping alignment differs"));
  }
  const bool grouped = in.size() == 4;
  const auto rows = plan.parameters[0], n = plan.parameters[1], k = plan.parameters[2];
  const auto slots = grouped ? static_cast<std::size_t>(in[3].description().dimensions()[1]) : 1;
  const auto experts = grouped ? static_cast<std::size_t>(in[1].description().dimensions()[0]) : 1;
  const auto *ids = grouped ? reinterpret_cast<const std::int32_t *>(maps.input[3].bytes().data()) : nullptr;
  // No expert-derived address is formed until the entire ID tensor is valid.
  if (ids) for (std::size_t i = 0; i < rows; ++i) {
    if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= experts)
      return finish(Status::OutOfRange("blockwise W4A8 expert ID is outside the bank"));
  }
  const auto *sf = reinterpret_cast<const float *>(maps.input[2].bytes().data());
  for (std::size_t r = 0; r < (grouped ? rows : 1); ++r) {
    const auto e = grouped ? static_cast<std::size_t>(ids[r]) : 0;
    const auto *selected = sf + e * n * (k / kGroup);
    for (std::size_t i = 0; i < n * (k / kGroup); ++i)
      if (!std::isfinite(selected[i]) || !(selected[i] > 0.F))
        return finish(Status::InvalidArgument("blockwise W4A8 selected scales must be positive finite F32"));
  }
  auto scratch = workspace.map_host(mem::HostAccess::kWrite);
  if (!scratch) return finish(scratch.error());
  maps.scratch = std::move(*scratch);
  auto bytes = maps.scratch.writable_bytes();
  if (!bytes || bytes->size() < plan.workspace.size || !aligned_for_storage(bytes->data(), 2))
    return finish(Status::FailedPrecondition("blockwise W4A8 workspace mapping differs"));
  auto output = out[0].storage().map_host(mem::HostAccess::kWrite);
  if (!output) return finish(output.error());
  maps.output = std::move(*output);
  auto destination = maps.output.writable_bytes();
  const auto output_bytes = *out[0].description().required_bytes();
  if (!destination || destination->size() < output_bytes || !aligned_for_storage(destination->data(), 2))
    return finish(Status::FailedPrecondition("blockwise W4A8 output mapping differs"));
  if (!bounded(destination->data(), output_bytes) || !bounded(bytes->data(), plan.workspace.size) ||
      overlaps(destination->data(), output_bytes, bytes->data(), plan.workspace.size))
    return finish(Status::InvalidArgument("blockwise W4A8 output/workspace overlaps"));
  for (std::size_t i = 0; i < in.size(); ++i) {
    const auto data = maps.input[i].bytes();
    if (!bounded(data.data(), data.size()) || overlaps(data.data(), data.size(), destination->data(), output_bytes) ||
        overlaps(data.data(), data.size(), bytes->data(), plan.workspace.size))
      return finish(Status::InvalidArgument("blockwise W4A8 input/output/workspace overlaps"));
  }
  const auto quant_slots = static_cast<std::size_t>(plan.scalar_bits);
  const auto quant_rows = (rows / slots) * quant_slots;
  auto *codes = reinterpret_cast<std::int8_t *>(bytes->data());
  auto *as = reinterpret_cast<BFloat *>(bytes->data() + quant_rows * k);
  const auto strides = in[0].description().byte_strides();
  Quantize q{reinterpret_cast<const BFloat *>(maps.input[0].bytes().data()), codes, as,
      k / kGroup, quant_slots, static_cast<std::size_t>(strides[0] / 2),
      grouped && in[0].description().dimensions()[1] != 1 ? static_cast<std::size_t>(strides[1] / 2) : 0};
  parallel_for_ranges(plan, quant_rows * (k / kGroup), &quantize, &q);
  if (q.invalid.load(std::memory_order_relaxed))
    return finish(Status::InvalidArgument("blockwise W4A8 activation or derived I8 code is outside the defined domain"));
  Dot d{codes, as, reinterpret_cast<const std::uint32_t *>(maps.input[1].bytes().data()), sf, ids,
      reinterpret_cast<BFloat *>(destination->data()), n, k, slots, quant_slots};
  maps.wrote = true;
  parallel_for_ranges(plan, *out[0].description().element_count(), &dot, &d);
  return finish(Status::Ok());
}
} // namespace

Status prepare_blockwise_w4a8(Operation operation, std::span<const TensorDesc> in,
    std::span<const TensorDesc> out, NativePlan &plan) noexcept {
  const bool grouped = operation.custom_as<model_ops::GroupedBlockwiseW4A8Linear>().has_value();
  if (!grouped && !operation.custom_as<model_ops::BlockwiseW4A8Linear>())
    return Status::Unimplemented("operation is not blockwise W4A8 linear");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, in, inferred));
  if (out.size() != 1 || out[0] != inferred[0])
    return Status::InvalidArgument("blockwise W4A8 output description differs");
  const auto tokens = static_cast<std::size_t>(in[0].dimensions()[0]);
  const auto slots = grouped ? static_cast<std::size_t>(in[3].dimensions()[1]) : 1;
  const auto n = static_cast<std::size_t>(in[1].dimensions()[grouped ? 1 : 0]);
  const auto k = static_cast<std::size_t>(in[0].dimensions().back());
  const auto quant_slots = !grouped || in[0].dimensions()[1] == 1 || in[0].byte_strides()[1] == 0 ? 1 : slots;
  if (tokens > SIZE_MAX / slots || k > SIZE_MAX - (k / kGroup) * 2 ||
      tokens > SIZE_MAX / quant_slots / (k + (k / kGroup) * 2))
    return Status::OutOfRange("blockwise W4A8 activation workspace overflows");
  plan.parameters = {tokens * slots, n, k};
  plan.scalar_bits = quant_slots;
  plan.workspace = {.size = tokens * quant_slots * (k + (k / kGroup) * 2), .alignment = 2};
  plan.workspace_submit_route = &submit;
  return Status::Ok();
}
} // namespace hyle::compute::native::internal
