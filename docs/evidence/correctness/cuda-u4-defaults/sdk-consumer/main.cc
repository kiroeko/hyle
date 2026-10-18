#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>

#include "hyle/compute/cuda.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/graph.h"
#include "hyle/mem/buffer.h"

namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

template <class T> T take(hyle::Result<T> result) {
  if (!result) {
    std::fprintf(stderr, "%s\n", result.error().message().data());
    std::exit(2);
  }
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) {
    std::fprintf(stderr, "%s\n", status.message().data());
    std::exit(3);
  }
}
int main() {
  const auto desc = take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 64}));
  auto context = take(g::cuda::create_context());
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  auto allocate = [&]() { return take(m::Buffer::create(memory.ref(), 128 * sizeof(float), 256)); };
  auto input = allocate(), output = allocate();
  auto in = take(c::ConstTensorView::create(desc, input.view()));
  auto out = take(c::TensorView::create(desc, output.view()));
  auto queue = take(context.create_queue());
  auto direct = take(c::prepare(context, c::Operation(c::Unary{c::UnaryKind::kExp}),
      std::span(&in, 1), std::span(&out, 1)));
  if (direct.route_policy().namespace_id != 0x48594c4543554444ULL)
    return 4;
  auto call = take(direct.create_invocation());
  auto builder = take(g::GraphBuilder::create());
  auto x = take(builder.input(desc));
  auto y = take(builder.add(c::Unary{c::UnaryKind::kExp}, x));
  auto z = take(builder.add(c::Unary{c::UnaryKind::kTanh}, y));
  require(builder.publish(z));
  auto logical = take(std::move(builder).finish());
  auto compiled = take(g::cuda::compile(logical, context, memory.ref()));
  if (compiled.report.step_count() != 1 || compiled.report.region_count() != 1)
    return 5;
  auto graph = take(compiled.executable.create_invocation());
  require(graph.bind(x, in)); require(graph.bind(z, out));
  require(g::cuda::prepare_replay(graph, queue));
  for (float value : {-1.0F, 0.25F}) {
    std::array<float, 128> values{}; values.fill(value);
    require(input.view().write(std::as_bytes(std::span(values))));
    auto done = take(call.submit(queue, std::span(&in, 1), std::span(&out, 1)));
    require(done.wait()); require(output.view().read(std::as_writable_bytes(std::span(values))));
    for (float actual : values) if (std::abs(actual - std::exp(value)) > 1.0e-5F) return 6;
    auto completed = take(graph.submit(queue)); require(completed.wait());
    require(output.view().read(std::as_writable_bytes(std::span(values))));
    for (float actual : values) if (std::abs(actual - std::tanh(std::exp(value))) > 1.0e-5F) return 7;
  }
  std::puts("PASS ordinary cold policy, complete ggml region, changing payload replay");
}
