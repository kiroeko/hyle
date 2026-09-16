#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <cuda_fp16.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/mem/buffer.h"

namespace sample_cuda_graph {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace graph = hyle::graph;
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

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, graph::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto resource, cuda::MemoryResource::create(
                                           context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(resource.ref(), 1, 128));
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

  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(input.view.description()));
  HYLE_ASSIGN_OR_RETURN(auto w,
                        builder.constant(weights.weights().description()));
  HYLE_ASSIGN_OR_RETURN(auto s,
                        builder.constant(weights.scales().description()));
  HYLE_ASSIGN_OR_RETURN(auto d, builder.constant(down.view.description()));
  HYLE_ASSIGN_OR_RETURN(
      auto gate_up, builder.add(cuda::QuantizedWeights::operation(), x, w, s));
  HYLE_ASSIGN_OR_RETURN(auto activated,
                        builder.add(compute::GatedActivation{}, gate_up));
  HYLE_ASSIGN_OR_RETURN(auto y, builder.add(compute::MatMul{}, activated, d));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  const std::array constants{graph::ConstantBinding{w, weights.weights()},
                             graph::ConstantBinding{s, weights.scales()},
                             graph::ConstantBinding{d, down.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,
                        graph::cuda::compile(logical, context, resource.ref(),
                                             {.constants = constants}));
  HYLE_ASSIGN_OR_RETURN(auto slot, compiled.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(x, input.view.as_const()));
  HYLE_RETURN_IF_ERROR(slot.bind(y, output.view));
  HYLE_RETURN_IF_ERROR(graph::cuda::prepare_replay(slot, queue));
  if (slot.replay_route() != graph::ReplayRoute::kCaptured)
    return hyle::Status::FailedPrecondition(
        "this FFN requires native CUDA capture");

  // Stable addresses and a fixed native program; only input contents change.
  for (int iteration = 1; iteration <= 3; ++iteration) {
    std::array<__half, 128> values;
    values.fill(__float2half(static_cast<float>(iteration) / 64));
    HYLE_RETURN_IF_ERROR(
        input.storage.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto completion, slot.submit(queue));
    HYLE_RETURN_IF_ERROR(completion.wait());
    std::array<__half, 64> result;
    HYLE_RETURN_IF_ERROR(
        output.storage.view().read(std::as_writable_bytes(std::span(result))));
    const float gate = static_cast<float>(iteration) / 16;
    const float intermediate =
        __half2float(__float2half(gate / (1 + std::exp(-gate)) * 2 * gate));
    for (const auto value : result)
      if (std::abs(__half2float(value) - 2 * intermediate) > 0.00005f)
        return hyle::Status::Internal("CUDA Graph FFN numerical check failed");
  }
  return hyle::Status::Ok();
}

} // namespace sample_cuda_graph
