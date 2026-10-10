#include <hyle/graph/cpu.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>
#include <cuda_fp16.h>
#include "../../compute/cpp/workload.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <utility>

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace bridge = hyle::graph::cpu_cuda;
namespace mem = hyle::mem;
using hyle::Status;
template <class T> using Result = hyle::Result<T>;

struct Tensor final {
  mem::Buffer buffer;
  compute::TensorView view;
};
Result<Tensor> tensor(mem::MemoryResourceRef resource,
                      compute::TensorDesc description) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(resource, bytes, 256));
  HYLE_ASSIGN_OR_RETURN(auto view,
                       compute::TensorView::create(description, buffer.view()));
  return Tensor{std::move(buffer), view};
}
Status require(bool condition, const char *message) noexcept {
  return condition ? Status::Ok() : Status::Internal(message);
}

// Explicitly drain before this scope's borrowed buffers/Queues are destroyed,
// including an error return after a different slot accepted work.
struct Drain final {
  hyle::sched::Scheduler &scheduler;
  ~Drain() { scheduler.drain(); }
};

#include "inference.h"

Status run() noexcept {
  HYLE_ASSIGN_OR_RETURN(auto cpu, graph::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto cuda, graph::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto host, compute::cuda::MemoryResource::create(
                                      cuda, compute::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, compute::cuda::MemoryResource::create(
                                        cuda, compute::cuda::MemoryKind::kDevice));
  hyle::sched::Scheduler scheduler(2, mem::system_allocator(), 64);
  HYLE_ASSIGN_OR_RETURN(auto row,
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto scale_desc,
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix,
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(mem::system_resource(), matrix));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(host.ref(), row));
  HYLE_ASSIGN_OR_RETURN(auto scale, tensor(host.ref(), scale_desc));
  const std::array catalog{bridge::WeightSpec{7, matrix},
                           bridge::WeightSpec{11, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto weights,
      bridge::WeightSlots::create(cuda, host, device, catalog, 16384));
  HYLE_ASSIGN_OR_RETURN(auto transfer, cuda.create_queue());
  HYLE_RETURN_IF_ERROR(transfer.reserve_completions(8));
  std::array<float, 4096> matrix_values{};
  for (std::size_t i = 0; i < 64; ++i)
    matrix_values[i * 64 + i] = -2.0f;
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(matrix_values))));
  const std::array selected{bridge::WeightFragment{7, source.view.as_const()}};
  HYLE_ASSIGN_OR_RETURN(auto batch0, weights.upload(0, selected, transfer));
  HYLE_ASSIGN_OR_RETURN(auto batch1, weights.upload(1, selected, transfer));
  HYLE_RETURN_IF_ERROR(require(!weights.upload(0, selected, transfer),
                               "retained weight batch was overwritten"));
  const std::array unknown{bridge::WeightFragment{99, source.view.as_const()}};
  // A live batch also guards bad requests; test ID validation after release.
  batch1 = {};
  HYLE_RETURN_IF_ERROR(weights.close());
  weights = {};
  HYLE_ASSIGN_OR_RETURN(weights,
      bridge::WeightSlots::create(cuda, host, device, catalog, 16384));
  HYLE_RETURN_IF_ERROR(require(!weights.upload(1, unknown, transfer),
                               "unknown expert ID was accepted"));
  const std::array duplicate{selected[0], selected[0]};
  HYLE_RETURN_IF_ERROR(require(!weights.upload(1, duplicate, transfer),
                               "duplicate expert ID was accepted"));
  HYLE_ASSIGN_OR_RETURN(batch1, weights.upload(1, selected, transfer));

  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(row));
  HYLE_ASSIGN_OR_RETURN(auto s, builder.input(scale_desc));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(matrix));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(compute::RmsNorm{}, x, s));
  HYLE_ASSIGN_OR_RETURN(auto projected, builder.add(compute::MatMul{}, normalized, w));
  HYLE_ASSIGN_OR_RETURN(auto y,
                       builder.add(compute::Unary{compute::UnaryKind::kAbs}, projected));
  HYLE_RETURN_IF_ERROR(builder.publish(y));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto n0, logical.node_id(0));
  HYLE_ASSIGN_OR_RETURN(auto n1, logical.node_id(1));
  HYLE_ASSIGN_OR_RETURN(auto n2, logical.node_id(2));
  const std::array placement{
      graph::NodePlacement{n0, bridge::kCpuTarget, graph::PlacementStrength::kRequired},
      graph::NodePlacement{n1, bridge::kCudaTarget, graph::PlacementStrength::kRequired},
      graph::NodePlacement{n2, bridge::kCpuTarget, graph::PlacementStrength::kRequired}};
  const std::array external{
      graph::ExternalResourceContract{x, bridge::kHostStorage},
      graph::ExternalResourceContract{s, bridge::kHostStorage},
      graph::ExternalResourceContract{w, bridge::kDeviceStorage},
      graph::ExternalResourceContract{y, bridge::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,
      bridge::compile(logical, {&cpu, &cuda, &host, &device},
                      {.node_placements = placement, .external_resources = external}));
  std::array<bridge::Invocation, 2> runs;
  std::array<Tensor, 2> outputs;
  std::array<compute::Queue, 2> gpu_queues, cpu_queues;
  std::array<compute::OwnedResult<bridge::Results>, 2> results;
  for (std::size_t i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(gpu_queues[i], cuda.create_queue());
    HYLE_ASSIGN_OR_RETURN(cpu_queues[i], cpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(outputs[i], tensor(host.ref(), row));
    HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
    const std::array queues{
        graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu_queues[i]},
        graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queues[i]}};
    HYLE_ASSIGN_OR_RETURN(runs[i], bridge::Invocation::create(
                                      std::move(native), scheduler, queues));
    HYLE_RETURN_IF_ERROR(runs[i].bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(runs[i].bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(runs[i].bind(w, i == 0 ? batch0 : batch1, 7));
    HYLE_RETURN_IF_ERROR(runs[i].bind(y, outputs[i].view));
    HYLE_RETURN_IF_ERROR(runs[i].prepare_replay());
  }
  std::array<float, 64> values, scales;
  scales.fill(1.0f);
  float squares = 0;
  for (std::size_t i = 0; i < 64; ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 9) - 4);
    squares += values[i] * values[i];
  }
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(scale.buffer.view().write(std::as_bytes(std::span(scales))));
  Drain drain{scheduler};
  for (std::size_t i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(results[i], runs[i].submit());
  }
  for (std::size_t i = 0; i < 2; ++i) {
    HYLE_ASSIGN_OR_RETURN(auto lease, results[i].wait());
    HYLE_ASSIGN_OR_RETURN(auto view, lease->output(y));
    std::array<float, 64> actual;
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(actual))));
    const float inverse = 1.0f / std::sqrt(squares / 64 + compute::RmsNorm{}.epsilon);
    for (std::size_t j = 0; j < 64; ++j)
      HYLE_RETURN_IF_ERROR(require(std::abs(actual[j] - std::abs(2 * values[j] * inverse)) < 0.0001f,
                                   "CPU CUDA CPU numeric result differs"));
    results[i] = {};
    HYLE_RETURN_IF_ERROR(require(!runs[i].reset().ok(), "old result lease allowed reset"));
    lease = {};
    // Stable payload/addresses can repeat without selecting Providers again.
    HYLE_ASSIGN_OR_RETURN(auto repeated, runs[i].run());
    repeated = {};
  }
  batch0 = {};
  batch1 = {};
  HYLE_RETURN_IF_ERROR(require(!weights.upload(1, selected, transfer),
                               "bound graph did not retain weight batch"));
  for (auto &invocation : runs) {
    HYLE_RETURN_IF_ERROR(invocation.reset());
    HYLE_RETURN_IF_ERROR(invocation.close());
  }
  HYLE_ASSIGN_OR_RETURN(auto reused, weights.upload(1, selected, transfer));
  HYLE_RETURN_IF_ERROR(reused.wait());
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &gpu_queues[0]},
      graph::QueueBinding{graph::ExecutionLaneId{1}, &cpu_queues[0]}};
  auto prepare = [&](hyle::sched::Scheduler &executor) noexcept -> Result<bridge::Invocation> {
    HYLE_ASSIGN_OR_RETURN(auto native, compiled.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto invocation,
        bridge::Invocation::create(std::move(native), executor, queues));
    HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.bind(s, scale.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.bind(w, reused, 7));
    HYLE_RETURN_IF_ERROR(invocation.bind(y, outputs[0].view));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
    return invocation;
  };
  {
    HYLE_ASSIGN_OR_RETURN(auto cancelled, prepare(scheduler));
    HYLE_ASSIGN_OR_RETURN(auto pending, cancelled.submit());
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    auto settled = pending.wait();
    if (!settled && !settled.error().IsCancelled())
      return settled.error();
    const auto drained = cancelled.close();
    if (!drained.ok() && !drained.IsCancelled())
      return drained;
    HYLE_RETURN_IF_ERROR(require(!cancelled.submit(), "closed slot accepted work"));
  }
  HYLE_RETURN_IF_ERROR(outputs[0].buffer.view().fill(std::byte{}));
  {
    // Scheduler drain is explicit and occurs while every borrowed native owner
    // and external buffer remains alive. Public handle destruction adds no wait.
    hyle::sched::Scheduler temporary(1, mem::system_allocator(), 16);
    HYLE_ASSIGN_OR_RETURN(auto abandoned, prepare(temporary));
    HYLE_ASSIGN_OR_RETURN(auto pending, abandoned.submit());
    pending = {};
    abandoned = {};
  }
  std::array<float, 64> abandoned_output;
  HYLE_RETURN_IF_ERROR(outputs[0].buffer.view().read(
      std::as_writable_bytes(std::span(abandoned_output))));
  HYLE_RETURN_IF_ERROR(require(abandoned_output[0] > 0.0f,
                               "abandoned work lost its execution owner"));
  HYLE_ASSIGN_OR_RETURN(auto independent, cuda.create_queue());
  HYLE_RETURN_IF_ERROR(weights.close());
  HYLE_RETURN_IF_ERROR(inference_case(cpu, cuda, host, device, scheduler));
  std::printf("CPU+CUDA: two real math slots, fixed weight IDs, native DMA dependencies, leases and replay passed\n");
  return Status::Ok();
}
} // namespace

int main() {
  const auto result = run();
  if (!result.ok()) {
    std::fprintf(stderr, "CPU+CUDA: %.*s\n", static_cast<int>(result.message().size()),
                 result.message().data());
    return 1;
  }
  return 0;
}
