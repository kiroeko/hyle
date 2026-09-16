#include "hyle/compute/providers/marlin.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <cuda_fp16.h>

#include "backend_internal.h"
#include "marlin_internal.h"

namespace hyle::compute::marlin {

Result<QuantizedWeights> QuantizedWeights::pack(
    const Context &context, mem::MemoryResourceRef resource,
    std::int64_t output_channels, std::int64_t input_channels,
    std::span<const std::int8_t> values, std::span<const float> scales,
    mem::MemoryResourceRef host_scratch) noexcept {
  HYLE_RETURN_IF_ERROR(internal::weight_shape(output_channels, input_channels));
  if (!context.is_open() || context.device().backend_id() != cuda::kBackendId)
    return std::unexpected(Status::FailedPrecondition(
        "Marlin packing needs an open CUDA Context"));
  const auto n = static_cast<std::size_t>(output_channels);
  const auto k = static_cast<std::size_t>(input_channels);
  if (n > std::numeric_limits<std::size_t>::max() / k)
    return std::unexpected(
        Status::OutOfRange("Marlin weight extent overflows"));
  const auto groups = k / 128;
  if (values.size() != n * k || scales.size() != n * groups)
    return std::unexpected(
        Status::InvalidArgument("Marlin packing input extents do not agree"));
  const auto *backend = detail::BackendAccess::vtable(cuda::backend());
  HYLE_RETURN_IF_ERROR(backend->validate_resource(context.device(), resource));
  if (!resource.properties().has(mem::MemoryProperty::kDeviceAccessible) ||
      !host_scratch.valid() ||
      !host_scratch.properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument(
        "Marlin packing resource accessibility is invalid"));
  const std::size_t packed_bytes = n * k / 2;
  const std::size_t scale_bytes = n * groups * sizeof(__half);
  HYLE_ASSIGN_OR_RETURN(auto primary, cuda::primary_context(context));
  HYLE_ASSIGN_OR_RETURN(auto hardware,
                        internal::hardware(context.device(), primary));
  cuda::detail::ScopedPrimaryContext scope(primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  HYLE_ASSIGN_OR_RETURN(
      auto staging,
      mem::Buffer::create(host_scratch, packed_bytes + scale_bytes, 16));
  HYLE_ASSIGN_OR_RETURN(auto mapping,
                        staging.view().map_host(mem::HostAccess::kReadWrite));
  HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
  auto *gptq = reinterpret_cast<std::uint32_t *>(bytes.data());
  auto *packed_scales = reinterpret_cast<__half *>(bytes.data() + packed_bytes);
  std::memset(gptq, 0, packed_bytes);
  for (std::size_t channel = 0; channel < n; ++channel) {
    for (std::size_t inner = 0; inner < k; ++inner) {
      const int value = values[channel * k + inner];
      if (value < -8 || value > 7)
        return std::unexpected(
            Status::InvalidArgument("Marlin Q4 value is outside [-8,7]"));
      gptq[(inner / 8) * n + channel] |= static_cast<std::uint32_t>(value + 8)
                                         << (4 * (inner % 8));
    }
    for (std::size_t group = 0; group < groups; ++group) {
      const float value = scales[channel * groups + group];
      if (!std::isfinite(value) || value < 0 || value > 65504.0f)
        return std::unexpected(Status::InvalidArgument(
            "Marlin scale is outside finite nonnegative FP16 range"));
      // Pinned marlin_utils.py grouped scale permutation after [N,G]->[G,N].
      const std::size_t index = group * n + channel;
      const std::size_t target =
          (index / 64) * 64 + (index % 8) * 8 + (index % 64) / 8;
      packed_scales[target] = __float2half_rn(value);
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto temporary,
                        mem::Buffer::create(resource, packed_bytes, 16));
  QuantizedWeights result;
  HYLE_ASSIGN_OR_RETURN(result.weight_storage_,
                        mem::Buffer::create(resource, packed_bytes, 16));
  HYLE_ASSIGN_OR_RETURN(result.scale_storage_,
                        mem::Buffer::create(resource, scale_bytes, 16));
  HYLE_RETURN_IF_ERROR(temporary.view().write(bytes.first(packed_bytes)));
  HYLE_RETURN_IF_ERROR(result.scale_storage_.view().write(
      bytes.subspan(packed_bytes, scale_bytes)));
  HYLE_ASSIGN_OR_RETURN(auto source,
                        cuda::pointer<std::uint32_t>(temporary.view()));
  HYLE_ASSIGN_OR_RETURN(auto destination, cuda::pointer<std::byte>(
                                              result.weight_storage_.view()));
  HYLE_RETURN_IF_ERROR(
      cuda::status(native::prepare_repack(hardware.shared_bytes)));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(
      auto completion, cuda::submit(queue, [&](cudaStream_t stream) noexcept {
        return native::repack(
            source, destination, static_cast<int>(input_channels),
            static_cast<int>(output_channels), hardware.multiprocessors,
            hardware.shared_bytes, stream);
      }));
  // Repack may have accepted a prefix before an error. Keep every temporary
  // and the final payload alive until this synchronous cold operation ends.
  HYLE_RETURN_IF_ERROR(completion.wait());
  HYLE_RETURN_IF_ERROR(mapping.close());
  HYLE_ASSIGN_OR_RETURN(auto weight_format,
                        StorageFormat::opaque(kWeightFormat, 16));
  HYLE_ASSIGN_OR_RETURN(auto scale_format,
                        StorageFormat::opaque(kScaleFormat, 16));
  HYLE_ASSIGN_OR_RETURN(
      auto weight_desc,
      TensorDesc::opaque(DataType::kFloat16,
                         std::array{output_channels, input_channels},
                         weight_format, packed_bytes));
  HYLE_ASSIGN_OR_RETURN(
      auto scale_desc,
      TensorDesc::opaque(DataType::kFloat16,
                         std::array{output_channels, input_channels / 128},
                         scale_format, scale_bytes));
  HYLE_ASSIGN_OR_RETURN(
      result.weights_,
      ConstTensorView::create(weight_desc, result.weight_storage_.view()));
  HYLE_ASSIGN_OR_RETURN(
      result.scales_,
      ConstTensorView::create(scale_desc, result.scale_storage_.view()));
  return result;
}

} // namespace hyle::compute::marlin
