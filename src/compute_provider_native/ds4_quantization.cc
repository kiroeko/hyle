#include "native_plan.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>

#include "ds4_roundtrip.h"
#include "hyle/compute/model_ops.h"
#include "hyle/mem/resource.h"
#include "numeric.h"

namespace hyle::compute::native::internal {
namespace {

using Format = model_ops::BlockQuantizationFormat;

template <bool Float4>
Status convert_groups(const std::byte *source, std::byte *destination,
                      std::size_t elements) noexcept {
  constexpr std::size_t group = Float4 ? 32 : 64;
  std::array<float, group> values{};
  for (std::size_t base = 0; base < elements; base += group) {
    for (std::size_t index = 0; index < group; ++index) {
      values[index] = numeric::load<simd::bfloat16_t>(
          source + (base + index) * sizeof(simd::bfloat16_t));
      if (!std::isfinite(values[index]))
        return Status::InvalidArgument("ds4 roundtrip requires finite BF16 input");
    }
    if constexpr (Float4)
      ds4_quantization::dsv4_fp4_act_quantize_row_inplace_cpu(values.data(), group);
    else
      ds4_quantization::dsv4_fp8_kv_quantize_row_inplace_cpu(values.data(), group, 0);
    for (std::size_t index = 0; index < group; ++index)
      numeric::store<simd::bfloat16_t>(
          destination + (base + index) * sizeof(simd::bfloat16_t), values[index]);
  }
  return Status::Ok();
}

template <bool Float4>
detail::EnqueueResult submit_roundtrip(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("ds4 roundtrip binding count is invalid"));
  const auto input = inputs[0].storage();
  const auto output = outputs[0].storage();
  if (!input.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
    auto status = input.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }
  auto mapped_input = input.map_host();
  if (!mapped_input)
    return completed(std::move(mapped_input).error());
  auto mapped_output = output.map_host(mem::HostAccess::kWrite);
  if (!mapped_output) {
    mapped_input->close().IgnoreError();
    return completed(std::move(mapped_output).error());
  }
  auto writable = mapped_output->writable_bytes();
  if (!writable) {
    mapped_output->close().IgnoreError();
    mapped_input->close().IgnoreError();
    return completed(std::move(writable).error());
  }
  Status terminal = convert_groups<Float4>(mapped_input->bytes().data(),
                                           writable->data(), plan.parameters[0]);
  if (!output.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, mapped_output->flush());
  preserve_first_error(terminal, mapped_output->close());
  preserve_first_error(terminal, mapped_input->close());
  return completed(std::move(terminal));
}

} // namespace

Status prepare_ds4_quantization(Operation operation,
                                std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs,
                                NativePlan &plan) noexcept {
  const auto roundtrip = operation.custom_as<model_ops::BlockQuantizationRoundtrip>();
  if (!roundtrip || inputs.size() != 1 || outputs.size() != 1)
    return Status::Unimplemented("CPU ds4 supports exact quantization roundtrip only");
  if (roundtrip->format != Format::kFloat4E2M1Block32 &&
      roundtrip->format != Format::kFloat8E4M3Block64)
    return Status::Unimplemented("ds4 has no original group128 roundtrip entry");
  std::array<TensorDesc, 1> inferred;
  const auto status = roundtrip->infer_output_descriptions(inputs, inferred);
  if (!status.ok())
    return status;
  if (outputs[0] != inferred[0])
    return Status::InvalidArgument("ds4 roundtrip output description differs");
  auto count = inputs[0].element_count();
  if (!count)
    return std::move(count).error();
  plan.parameters[0] = *count;
  plan.submit_route = roundtrip->format == Format::kFloat4E2M1Block32
                          ? &submit_roundtrip<true>
                          : &submit_roundtrip<false>;
  return Status::Ok();
}

} // namespace hyle::compute::native::internal
