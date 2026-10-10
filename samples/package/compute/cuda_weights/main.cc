#include <hyle/compute/cuda_weights.h>
#include <hyle/compute/prepared_operation.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

namespace {
namespace compute = hyle::compute;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;
hyle::Status run() noexcept {
  HYLE_ASSIGN_OR_RETURN(auto context, cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto host, cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device, cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto transfer, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_RETURN_IF_ERROR(transfer.reserve_completions(8));
  HYLE_RETURN_IF_ERROR(queue.reserve_completions(8));
  HYLE_ASSIGN_OR_RETURN(auto row, compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto matrix, compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto source, mem::Buffer::create(mem::system_resource(), 16384, 256));
  HYLE_ASSIGN_OR_RETURN(auto x_buffer, mem::Buffer::create(device.ref(), 256, 256));
  HYLE_ASSIGN_OR_RETURN(auto y_buffer, mem::Buffer::create(device.ref(), 256, 256));
  HYLE_ASSIGN_OR_RETURN(auto source_view, compute::ConstTensorView::create(matrix, source.view()));
  HYLE_ASSIGN_OR_RETURN(auto x, compute::ConstTensorView::create(row, x_buffer.view()));
  HYLE_ASSIGN_OR_RETURN(auto y, compute::TensorView::create(row, y_buffer.view()));
  std::array<float, 4096> weight_values{};
  for (std::size_t i = 0; i < 64; ++i)
    weight_values[i * 64 + i] = 2.0f;
  std::array<float, 64> values;
  values.fill(3.0f);
  HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(weight_values))));
  HYLE_RETURN_IF_ERROR(x_buffer.view().write(std::as_bytes(std::span(values))));
  const std::array catalog{cuda::WeightSpec{7, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto slots, cuda::WeightSlots::create(context, host, device, catalog, 16384));
  HYLE_ASSIGN_OR_RETURN(auto empty, slots.upload(1, {}, transfer));
  HYLE_RETURN_IF_ERROR(empty.wait());
  if (empty.binding(7))
    return hyle::Status::Internal("empty selection exposed a binding");
  empty = {};
  const std::array selection{cuda::WeightFragment{7, source_view}};
  HYLE_ASSIGN_OR_RETURN(auto batch, slots.upload(0, selection, transfer));
  HYLE_ASSIGN_OR_RETURN(auto weight, batch.binding(7));
  HYLE_ASSIGN_OR_RETURN(auto program,
      compute::prepare(context, compute::MatMul{}, x.description(),
                       weight.description(), device.ref()));
  const auto needed = program.workspace_requirement();
  HYLE_ASSIGN_OR_RETURN(auto workspace,
      mem::Buffer::create(device.ref(), needed.size, needed.alignment));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(workspace.view()));
  HYLE_ASSIGN_OR_RETURN(auto dependency, batch.order(queue));
  HYLE_ASSIGN_OR_RETURN(auto completed, invocation.submit(queue, std::array{x, weight}, std::array{y}));
  HYLE_RETURN_IF_ERROR(completed.wait());
  HYLE_RETURN_IF_ERROR(dependency.wait());
  HYLE_RETURN_IF_ERROR(y_buffer.view().read(std::as_writable_bytes(std::span(values))));
  for (auto value : values)
    if (std::abs(value - 6.0f) > 0.00001f)
      return hyle::Status::Internal("Direct CUDA weight-backed MatMul differs");
  HYLE_RETURN_IF_ERROR(slots.close());
  std::printf("Direct CUDA: selected weight DMA, native dependency and mature MatMul passed\n");
  return hyle::Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) {
    std::fprintf(stderr, "%.*s\n", static_cast<int>(status.message().size()), status.message().data());
    return 1;
  }
  return 0;
}
