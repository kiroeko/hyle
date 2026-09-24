#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>

#include <hyle/compute/backends/cpu.h>

namespace example {
namespace {
using namespace hyle::compute;
using hyle::Result;
using hyle::Status;

constexpr ProviderRuntimeProperties runtime{
    .replay_class = ProviderReplayClass::kPrepared, .blocks_host = true};

struct AffineKernel final {
  // The owning factory stays alive through Context -> prepared -> Invocation.
  const std::size_t *chunk_elements;
  float scale;
  float bias;
  struct Invocation {};

  Result<Invocation> create_invocation(hyle::mem::BufferView,
                                       hyle::mem::AllocatorRef) const noexcept {
    return Invocation{};
  }
  AliasSupport alias_support(std::size_t output,
                             std::size_t input) const noexcept {
    return {output == 0 && input == 0, 0};
  }
  KernelCommandResult
  enqueue(Invocation &, Queue &, std::span<const ConstTensorView> inputs,
          std::span<const TensorView> outputs) const noexcept {
    auto input = inputs[0].storage().map_host();
    if (!input)
      return KernelCommandResult::rejected(std::move(input).error());
    auto output = outputs[0].storage().map_host(hyle::mem::HostAccess::kWrite);
    if (!output)
      return KernelCommandResult::rejected(std::move(output).error());
    auto writable = output->writable_bytes();
    if (!writable)
      return KernelCommandResult::rejected(std::move(writable).error());
    const auto *source = reinterpret_cast<const float *>(input->bytes().data());
    auto *destination = reinterpret_cast<float *>(writable->data());
    const auto count = inputs[0].description().element_count().value();
    for (std::size_t first = 0; first < count;) {
      const auto end = first + (std::min)(count - first, *chunk_elements);
      for (auto index = first; index < end; ++index)
        destination[index] = std::fma(source[index], scale, bias);
      first = end;
    }
    auto status = output->close();
    if (status.ok())
      status = input->close();
    return KernelCommandResult::accepted_command(std::move(status));
  }
};

struct AffineFactory final {
  std::unique_ptr<std::size_t> chunk_elements;

  Result<KernelProposal>
  propose(const KernelRequest &request, std::span<ResourceRequirement> inputs,
          std::span<ResourceRequirement> outputs) const noexcept {
    if (!request.operation.custom_as<Affine>() || request.inputs.size() != 1 ||
        request.outputs.size() != 1 || !request.inputs[0].is_dense() ||
        request.inputs[0].data_type() != DataType::kFloat32 ||
        request.inputs[0] != request.outputs[0])
      return std::unexpected(
          Status::Unimplemented("requires dense f32 affine"));
    // Mapping is the actual requirement; this kernel does not depend on one
    // allocator instance (which may differ between statically linked images).
    const ResourceRequirement storage{
        .required_properties = hyle::mem::MemoryProperty::kHostAccessible,
        .minimum_alignment = alignof(float)};
    inputs[0] = storage;
    outputs[0] = storage;
    return KernelProposal{.runtime = runtime,
                          .cost = {.workspace = WorkspaceRequirement{},
                                   .persistent_bytes = 0,
                                   .copy_bytes = 0,
                                   .repack_bytes = 0,
                                   .synchronization_count = 0,
                                   .dispatch_count = 1},
                          .implementation = ProviderImplementationId{1}};
  }
  KernelPreparation prepare(const KernelPrepareRequest &request,
                            hyle::mem::AllocatorRef allocator) const noexcept {
    const auto operation = request.operation.custom_as<Affine>();
    if (!operation)
      return KernelPreparation::error(
          Status::Internal("selected schema changed"));
    auto kernel = make_custom_kernel(
        AffineKernel{chunk_elements.get(), operation->scale, operation->bias},
        {.runtime = runtime, .persistent_bytes = 0}, allocator);
    if (!kernel)
      return KernelPreparation::error(std::move(kernel).error());
    return KernelPreparation::prepared(std::move(*kernel));
  }
};
} // namespace

Result<Provider> make_provider(std::size_t chunk_elements) noexcept {
  if (chunk_elements == 0)
    return std::unexpected(
        Status::InvalidArgument("chunk size must be positive"));
  std::unique_ptr<std::size_t> owned(new (std::nothrow)
                                         std::size_t(chunk_elements));
  if (!owned)
    return std::unexpected(
        Status::ResourceExhausted("example configuration allocation failed"));
  return make_custom_provider(
      {{0x4558414d504c45ULL, 2}, "example-affine-cpu", "1"}, cpu::kBackendId,
      AffineFactory{std::move(owned)});
}
} // namespace example
