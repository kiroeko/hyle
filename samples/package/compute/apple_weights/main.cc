#include <hyle/compute/apple_weights.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace m = hyle::mem;

hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto matrix, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 64}));
  HYLE_ASSIGN_OR_RETURN(auto source, m::Buffer::create(m::system_resource(), 16384));
  std::array<float, 4096> identity{};
  for (std::size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(identity))));
  HYLE_ASSIGN_OR_RETURN(auto source_tensor, c::ConstTensorView::create(matrix, source.view()));
  const std::array catalog{a::WeightSpec{7, matrix}};
  HYLE_ASSIGN_OR_RETURN(auto weights, a::WeightSlots::create(context, memory, catalog, 16384));
  HYLE_ASSIGN_OR_RETURN(auto empty, weights.upload(1, {}));
  HYLE_RETURN_IF_ERROR(empty.wait());
  if (empty.binding(7)) return hyle::Status::Internal("empty Apple weight selection exposed data");
  empty = {};
  const std::array selection{a::WeightFragment{7, source_tensor}};
  HYLE_ASSIGN_OR_RETURN(auto batch, weights.upload(0, selection));
  HYLE_ASSIGN_OR_RETURN(auto completion, batch.completion());
  HYLE_RETURN_IF_ERROR(completion.wait());
  HYLE_ASSIGN_OR_RETURN(auto native_weight, batch.binding(7));
  HYLE_ASSIGN_OR_RETURN(auto input, memory.allocate(row));
  HYLE_ASSIGN_OR_RETURN(auto input_buffer, m::Buffer::create(m::system_resource(), 256));
  std::array<float, 64> values;
  values.fill(3.0f);
  HYLE_RETURN_IF_ERROR(input_buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto input_view, c::ConstTensorView::create(row, input_buffer.view()));
  HYLE_RETURN_IF_ERROR(memory.copy_from_host(input, input_view));
  const std::array specs{*a::input_spec(row), *a::input_spec(matrix)};
  HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(c::MatMul{}, specs));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
  const std::array inputs{input, native_weight};
  HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto pending, invocation.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, pending.wait());
  HYLE_ASSIGN_OR_RETURN(auto output, c::TensorView::create(row, input_buffer.view()));
  HYLE_RETURN_IF_ERROR(memory.copy_to_host(result->arrays()[0], output));
  HYLE_RETURN_IF_ERROR(input_buffer.view().read(std::as_writable_bytes(std::span(values))));
  for (auto value : values)
    if (std::abs(value - 6.0f) > 1e-5f) return hyle::Status::Internal("Apple weight-backed MatMul differs");
  batch = {};
  if (weights.upload(0, selection))
    return hyle::Status::Internal("Apple array alias did not retain its weight slot");
  HYLE_RETURN_IF_ERROR(weights.close());
  std::printf("Apple Direct: real Metal weight transfer, ready owner and mature MatMul passed\n");
  return hyle::Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
