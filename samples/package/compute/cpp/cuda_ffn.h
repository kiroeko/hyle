#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <cuda_fp16.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace sample_cuda_direct {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

struct Tensor {
  mem::Buffer storage;
  compute::TensorView view;
};

inline hyle::Result<Tensor> tensor(mem::MemoryResourceRef resource,
                                   std::int64_t rows, std::int64_t columns) {
  HYLE_ASSIGN_OR_RETURN(auto desc,
                        compute::TensorDesc::dense(compute::DataType::kFloat16,
                                                   std::array{rows, columns}));
  HYLE_ASSIGN_OR_RETURN(auto bytes, desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto storage,
                        mem::Buffer::create(resource, bytes, 256));
  HYLE_ASSIGN_OR_RETURN(auto view,
                        compute::TensorView::create(desc, storage.view()));
  return Tensor{std::move(storage), view};
}

// Q4 gate/up -> SwiGLU -> dense down. All three native implementations are
// selected by the route; the consumer supplies no Provider list or profile.
inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto resource, cuda::MemoryResource::create(
                                           context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_RETURN_IF_ERROR(queue.reserve_completions(3));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(resource.ref(), 1, 128));
  HYLE_ASSIGN_OR_RETURN(auto projected, tensor(resource.ref(), 1, 256));
  HYLE_ASSIGN_OR_RETURN(auto activated, tensor(resource.ref(), 1, 128));
  HYLE_ASSIGN_OR_RETURN(auto down, tensor(resource.ref(), 128, 64));
  HYLE_ASSIGN_OR_RETURN(auto output, tensor(resource.ref(), 1, 64));

  std::vector<std::int8_t> q(256 * 128);
  for (std::size_t n = 0; n < 256; ++n)
    for (std::size_t k = 0; k < 128; ++k)
      q[n * 128 + k] = static_cast<std::int8_t>(n < 128 ? 1 : 2);
  const std::vector<float> scales(256, 1.0f / 32);
  HYLE_ASSIGN_OR_RETURN(auto weights,
                        cuda::QuantizedWeights::pack(context, resource.ref(),
                                                     256, 128, q, scales));
  const std::vector<__half> down_values(128 * 64, __float2half(1.0f / 64));
  HYLE_RETURN_IF_ERROR(
      down.storage.view().write(std::as_bytes(std::span(down_values))));
  const std::array first_inputs{input.view.as_const(), weights.weights(),
                                weights.scales()};
  HYLE_ASSIGN_OR_RETURN(
      auto first,
      compute::prepare(context,
                       compute::Operation(cuda::QuantizedWeights::operation()),
                       first_inputs, std::array{projected.view}));
  HYLE_ASSIGN_OR_RETURN(auto middle,
                        compute::prepare(context, compute::GatedActivation{},
                                         projected.view.description(),
                                         resource.ref()));
  HYLE_ASSIGN_OR_RETURN(
      auto last,
      compute::prepare(context, compute::MatMul{}, activated.view.description(),
                       down.view.description(), resource.ref()));
  std::array prepared{std::move(first), std::move(middle), std::move(last)};
  std::array<mem::Buffer, 3> workspace;
  std::array<compute::OperationInvocation, 3> slots;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const auto needed = prepared[i].workspace_requirement();
    HYLE_ASSIGN_OR_RETURN(
        workspace[i],
        mem::Buffer::create(resource.ref(), needed.size, needed.alignment));
    HYLE_ASSIGN_OR_RETURN(slots[i],
                          prepared[i].create_invocation(workspace[i].view()));
  }

  for (int iteration = 1; iteration <= 3; ++iteration) {
    std::array<__half, 128> values;
    values.fill(__float2half(static_cast<float>(iteration) / 64));
    HYLE_RETURN_IF_ERROR(
        input.storage.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(
        auto done_first,
        slots[0].submit(queue, first_inputs, std::array{projected.view}));
    HYLE_ASSIGN_OR_RETURN(
        auto done_middle,
        slots[1].submit(queue, projected.view.as_const(), activated.view));
    HYLE_ASSIGN_OR_RETURN(auto done_last,
                          slots[2].submit(queue, activated.view.as_const(),
                                          down.view.as_const(), output.view));
    HYLE_RETURN_IF_ERROR(done_last.wait());
    HYLE_RETURN_IF_ERROR(done_middle.wait());
    HYLE_RETURN_IF_ERROR(done_first.wait());
    std::array<__half, 64> result;
    HYLE_RETURN_IF_ERROR(
        output.storage.view().read(std::as_writable_bytes(std::span(result))));
    const float gate = static_cast<float>(iteration) / 16;
    const float intermediate =
        __half2float(__float2half(gate / (1 + std::exp(-gate)) * 2 * gate));
    const float expected = 2 * intermediate;
    for (const auto value : result)
      if (std::abs(__half2float(value) - expected) > 0.00005f)
        return hyle::Status::Internal("CUDA Direct FFN numerical check failed");
  }
  return hyle::Status::Ok();
}

} // namespace sample_cuda_direct
