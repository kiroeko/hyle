#include "native_plan.h"
#include "numeric.h"
#include "../compute_packed_shared/static_fp8_linear.h"
#include "hyle/simd/ops.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace hyle::compute::native::internal {
namespace {
constexpr std::size_t kTile = 64;
bool bounded(const void *p, std::size_t n) noexcept {
  return p && n <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(p);
}
bool overlaps(const void *a, std::size_t na, const void *b, std::size_t nb) noexcept {
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  return x < y + nb && y < x + na;
}
struct Mappings final {
  std::array<mem::HostMapping, 4> inputs;
  mem::HostMapping scratch, output;
  bool wrote = false;
  Status finish(Status s, mem::BufferView destination) noexcept {
    if (wrote && !destination.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      preserve_first_error(s, output.flush());
    preserve_first_error(s, output.close());
    preserve_first_error(s, scratch.close());
    for (auto &m : inputs) preserve_first_error(s, m.close());
    return s;
  }
};
template<class Input> struct Quantize final {
  const Input *a;
  simd::float8_e4m3_t *q;
  const std::byte *scale;
  std::size_t depth, stride;
  float scalar_reciprocal;
};
template<class Input>
void quantize(void *state, std::size_t, std::size_t begin, std::size_t end) noexcept {
  const auto &e = *static_cast<const Quantize<Input> *>(state);
  for (auto row = begin; row < end; ++row) {
    // Compute each row's reciprocal once; original scalar mode retains its
    // single reciprocal. Scale addresses/strides were proved during prepare.
    volatile float reciprocal = e.stride == 0 ? e.scalar_reciprocal
        : 1.F / numeric::load<float>(e.scale + row * e.stride);
    for (std::size_t lane = 0; lane < e.depth; ++lane) {
      const auto i = row * e.depth + lane;
      volatile float normalized = numeric::decode(e.a[i]) * reciprocal;
      e.q[i] = simd::f32_to_f8e4m3(normalized);
    }
  }
}
template<class Output, bool WeightFirst> struct Dot final {
  const NativePlan *plan;
  const simd::float8_e4m3_t *a, *w;
  const std::byte *sa, *sw;
  Output *output;
};
template<class Output, bool WeightFirst>
void dot(void *state, std::size_t, std::size_t begin, std::size_t end) noexcept {
  const auto &e = *static_cast<const Dot<Output, WeightFirst> *>(state);
  const auto n = e.plan->parameters[1], k = e.plan->parameters[2];
  std::array<float, kTile> a{}, w{};
  for (auto index = begin; index < end; ++index) {
    const auto row = index / n, column = index % n;
    float sum = 0.F;
    for (std::size_t first = 0; first < k; first += kTile) {
      const auto count = (std::min)(kTile, k - first);
      for (std::size_t lane = 0; lane < count; ++lane) {
        a[lane] = simd::f8e4m3_to_f32(e.a[row * k + first + lane]);
        w[lane] = simd::f8e4m3_to_f32(e.w[column * k + first + lane]);
      }
      sum += simd::dot(a.data(), w.data(), count);
    }
    // Keep both RNEs and their order, including 0*Inf and finite overflow.
    const float sa = numeric::load<float>(e.sa + row * e.plan->row_first_stride);
    const float sw = numeric::load<float>(e.sw + column * e.plan->row_second_stride);
    volatile float scaled = sum * (WeightFirst ? sw : sa);
    volatile float restored = scaled * (WeightFirst ? sa : sw);
    e.output[index] = numeric::encode<Output>(restored);
  }
}
template<class Input, class Output, bool WeightFirst>
detail::EnqueueResult submit(const NativePlan &plan,
    std::span<const ConstTensorView> inputs, std::span<const TensorView> outputs,
    mem::BufferView workspace) noexcept {
  if (inputs.size() != 4 || outputs.size() != 1)
    return completed(Status::Internal("native static FP8 binding count differs"));
  Mappings maps;
  const auto finish = [&](Status s) { return completed(maps.finish(std::move(s), outputs[0].storage())); };
  std::array<std::size_t, 4> sizes{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto view = inputs[i].storage();
    if (!view.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
      auto status = view.invalidate(); if (!status.ok()) return finish(std::move(status));
    }
    auto mapped = view.map_host(); if (!mapped) return finish(std::move(mapped).error());
    maps.inputs[i] = std::move(*mapped); sizes[i] = *inputs[i].description().required_bytes();
    const auto data = maps.inputs[i].bytes();
    if (data.size() < sizes[i] || !bounded(data.data(), sizes[i]) ||
        !aligned_for_storage(data.data(), inputs[i].description().required_alignment()))
      return finish(Status::FailedPrecondition("native static FP8 input mapping differs"));
  }
  auto mapped = workspace.map_host(mem::HostAccess::kWrite);
  if (!mapped) return finish(std::move(mapped).error());
  maps.scratch = std::move(*mapped); auto scratch = maps.scratch.writable_bytes();
  if (!scratch || scratch->size() < plan.workspace.size || !bounded(scratch->data(), plan.workspace.size))
    return finish(Status::FailedPrecondition("native static FP8 workspace mapping differs"));
  mapped = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!mapped) return finish(std::move(mapped).error());
  maps.output = std::move(*mapped); auto destination = maps.output.writable_bytes();
  const auto bytes = *outputs[0].description().required_bytes();
  if (!destination || destination->size() < bytes || !bounded(destination->data(), bytes) ||
      !aligned_for_storage(destination->data(), alignof(Output)))
    return finish(Status::FailedPrecondition("native static FP8 output mapping differs"));
  if (overlaps(destination->data(), bytes, scratch->data(), plan.workspace.size))
    return finish(Status::InvalidArgument("native static FP8 workspace/output overlaps"));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (overlaps(maps.inputs[i].bytes().data(), sizes[i], destination->data(), bytes) ||
        overlaps(maps.inputs[i].bytes().data(), sizes[i], scratch->data(), plan.workspace.size))
      return finish(Status::InvalidArgument("native static FP8 input/output/workspace overlaps"));
  const auto *sa = maps.inputs[2].bytes().data();
  const auto *sw = maps.inputs[3].bytes().data();
  const float scalar_sa = numeric::load<float>(sa);
  volatile float reciprocal = 1.F / scalar_sa;
  auto *q = reinterpret_cast<simd::float8_e4m3_t *>(scratch->data());
  Quantize<Input> qepisode{reinterpret_cast<const Input *>(maps.inputs[0].bytes().data()), q,
      sa, plan.parameters[2], plan.row_first_stride, reciprocal};
  parallel_for_ranges(plan, plan.parameters[0], &quantize<Input>, &qepisode);
  Dot<Output, WeightFirst> episode{&plan, q,
      reinterpret_cast<const simd::float8_e4m3_t *>(maps.inputs[1].bytes().data()), sa, sw,
      reinterpret_cast<Output *>(destination->data())};
  maps.wrote = true;
  parallel_for_ranges(plan, *outputs[0].description().element_count(), &dot<Output, WeightFirst>, &episode);
  return finish(Status::Ok());
}
template<class Input, bool WeightFirst> WorkspaceSubmitRoute output_route(DataType t) noexcept {
  if (t == DataType::kFloat16) return &submit<Input, simd::float16_t, WeightFirst>;
  if (t == DataType::kBFloat16) return &submit<Input, simd::bfloat16_t, WeightFirst>;
  return &submit<Input, float, WeightFirst>;
}
template<bool WeightFirst> WorkspaceSubmitRoute input_route(DataType in, DataType out) noexcept {
  if (in == DataType::kFloat16) return output_route<simd::float16_t, WeightFirst>(out);
  if (in == DataType::kBFloat16) return output_route<simd::bfloat16_t, WeightFirst>(out);
  return output_route<float, WeightFirst>(out);
}
} // namespace
Status prepare_static_fp8(Operation operation, std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs, NativePlan &plan) noexcept {
  const auto op = operation.custom_as<packed::internal::StaticFloat8Linear>();
  if (!op) return Status::Unimplemented("operation is not static FP8 linear");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(op->infer_output_descriptions(inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("native static FP8 output description differs");
  plan.parameters = {static_cast<std::size_t>(inputs[0].dimensions()[0]),
      static_cast<std::size_t>(inputs[1].dimensions()[0]),
      static_cast<std::size_t>(inputs[0].dimensions()[1])};
  // Existing family-local slots hold byte strides, not mutable scale values.
  plan.row_first_stride = inputs[2].rank() == 2
      ? static_cast<std::size_t>(inputs[2].byte_strides()[0]) : 0;
  plan.row_second_stride = *inputs[3].element_count() == 1 ? 0 : sizeof(float);
  const auto count = inputs[0].element_count();
  if (!count) return count.error();
  plan.workspace = {.size = *count, .alignment = alignof(simd::float8_e4m3_t)};
  plan.workspace_submit_route = op->scale_order == packed::internal::StaticFloat8ScaleOrder::kWeightThenActivation
      ? input_route<true>(inputs[0].data_type(), op->output_type)
      : input_route<false>(inputs[0].data_type(), op->output_type);
  return Status::Ok();
}
} // namespace hyle::compute::native::internal
