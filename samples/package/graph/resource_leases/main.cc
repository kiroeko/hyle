#include <hyle/compute/leased_operation.h>
#include <hyle/graph/leased_invocation.h>
#include <hyle/mem/stats.h>

#if defined(HYLE_COMPUTE_CPU)
#include <hyle/graph/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/compute/backends/cuda.h>
#include <hyle/graph/cuda.h>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/vulkan.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using hyle::Status;
template <class T> using Result = hyle::Result<T>;
constexpr std::array<float, 4> kLeft{1, 2, 3, 4}, kRight{5, 6, 7, 8};
// Independent row-major scalar product, shared by all native routes.
constexpr std::array<float, 4> kExpected{19, 22, 43, 50};

Status check(std::span<const float> actual,
             std::span<const float> expected = kExpected) {
  if (actual.size() != expected.size())
    return Status::Internal("lease consumer result size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1.e-4f)
      return Status::Internal("lease consumer mathematical result");
  return Status::Ok();
}

Result<c::TensorDesc> description() {
  return c::TensorDesc::dense(c::DataType::kFloat32,
                              std::array<std::int64_t, 2>{2, 2});
}

struct Definition final {
  g::Graph graph;
  g::ValueId left, right, product, alias;
};
Result<Definition> definition(c::TensorDesc desc) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto left, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto right, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, left, right));
  HYLE_RETURN_IF_ERROR(builder.publish(product));
  HYLE_ASSIGN_OR_RETURN(auto slice, c::View::slice(0, 0, 2));
  HYLE_ASSIGN_OR_RETURN(auto alias, builder.add(slice, left));
  HYLE_RETURN_IF_ERROR(builder.publish(alias));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return Definition{std::move(graph), left, right, product, alias};
}

Status read(const c::ReadLease &lease,
            std::span<const float> expected = kExpected) {
  HYLE_ASSIGN_OR_RETURN(auto view, lease.view());
  std::array<float, 4> actual{};
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
  return check(actual, expected);
}

#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA)
template <class Compile>
Status common(c::Context &context, m::MemoryResourceRef resource,
              Compile compile) {
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto stats,
                        m::StatsAllocator::create(m::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(
      auto pool,
      c::LeasePool::create(resource, 1024,
                           {.max_blocks = 3, .allocator = stats.get()}));
  HYLE_ASSIGN_OR_RETURN(auto desc, description());
  HYLE_ASSIGN_OR_RETURN(auto a_write, pool.allocate(sizeof(kLeft)));
  HYLE_ASSIGN_OR_RETURN(auto b_write, pool.allocate(sizeof(kRight)));
  HYLE_ASSIGN_OR_RETURN(auto output, pool.allocate(sizeof(kExpected)));
  HYLE_ASSIGN_OR_RETURN(auto a_view, a_write.view());
  HYLE_ASSIGN_OR_RETURN(auto b_view, b_write.view());
  HYLE_RETURN_IF_ERROR(a_view.write(std::as_bytes(std::span(kLeft))));
  HYLE_RETURN_IF_ERROR(b_view.write(std::as_bytes(std::span(kRight))));
  HYLE_ASSIGN_OR_RETURN(auto left, a_write.publish());
  HYLE_ASSIGN_OR_RETURN(auto right, b_write.publish());
  HYLE_ASSIGN_OR_RETURN(auto a_tensor, left.tensor(desc));
  HYLE_ASSIGN_OR_RETURN(auto b_tensor, right.tensor(desc));
  HYLE_ASSIGN_OR_RETURN(auto out_tensor, output.tensor(desc));
  HYLE_ASSIGN_OR_RETURN(auto prepared,
                        c::prepare(context, c::Operation(c::MatMul{}),
                                   std::array{a_tensor, b_tensor},
                                   std::array{out_tensor}));
  c::LeasePool workspace_pool;
  c::WriteLease workspace;
  const auto required = prepared.workspace_requirement();
  if (required.size) {
    const auto alignment = (std::max)(required.alignment, std::size_t{256});
    HYLE_ASSIGN_OR_RETURN(workspace_pool,
                          c::LeasePool::create(resource, required.size,
                                               {.max_blocks = 1,
                                                .backing_alignment = alignment,
                                                .allocator = stats.get()}));
    HYLE_ASSIGN_OR_RETURN(workspace,
                          workspace_pool.allocate(required.size, alignment));
  }
  HYLE_ASSIGN_OR_RETURN(auto direct, c::LeasedOperationInvocation::create(
                                         prepared, std::move(workspace),
                                         {.allocator = stats.get()}));
  std::array inputs{c::LeasedInput(desc, left), c::LeasedInput(desc, right)};
  const std::array outputs{c::LeasedOutput(desc, output)};
  // Warm the original mature provider before counting lease metadata.
  HYLE_ASSIGN_OR_RETURN(auto warm, direct.submit(queue, inputs, outputs));
  HYLE_RETURN_IF_ERROR(warm.wait());
  const auto cold = stats.snapshot().total_allocs;
  for (int i = 0; i < 8; ++i) {
    HYLE_ASSIGN_OR_RETURN(auto done, direct.submit(queue, inputs, outputs));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto ready, output.publish());
    auto second_reader = ready;
    if (ready.try_write())
      return Status::Internal("shared readers allowed a writer");
    HYLE_RETURN_IF_ERROR(read(ready));
    second_reader.reset();
    HYLE_ASSIGN_OR_RETURN(output, ready.try_write());
  }
  if (stats.snapshot().total_allocs != cold)
    return Status::Internal("direct lease metadata grew during replay");
  HYLE_ASSIGN_OR_RETURN(auto held_direct,
                        direct.submit(queue, inputs, outputs));
  HYLE_RETURN_IF_ERROR(direct.drain());
  HYLE_RETURN_IF_ERROR(held_direct.wait());
  if (direct.submit(queue, inputs, outputs))
    return Status::Internal("closed direct accepted work");
  HYLE_ASSIGN_OR_RETURN(auto graph, definition(desc));
  HYLE_ASSIGN_OR_RETURN(auto compiled, compile(graph.graph));
  HYLE_ASSIGN_OR_RETURN(
      auto run, g::LeasedInvocation::create(compiled.executable,
                                            {.allocator = stats.get()}));
  HYLE_RETURN_IF_ERROR(run.bind(graph.left, desc, left));
  HYLE_RETURN_IF_ERROR(run.bind(graph.right, desc, right));
  HYLE_RETURN_IF_ERROR(run.bind(graph.product, desc, std::move(output)));
  for (auto &input : inputs)
    input.read.reset();
  left.reset();
  right.reset();
  direct = {};
  prepared = {};
  held_direct = {};
  warm = {};
  const auto graph_cold = stats.snapshot().total_allocs;
  for (int i = 0; i < 8; ++i) {
    HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queue));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, run.output(graph.product));
    HYLE_ASSIGN_OR_RETURN(auto lease, result.storage_lease());
    HYLE_RETURN_IF_ERROR(read(lease));
    if (run.submit(queue) || run.reset().ok())
      return Status::Internal("graph overwrote a retained result");
  }
  if (stats.snapshot().total_allocs != graph_cold)
    return Status::Internal("graph lease metadata grew during replay");
  HYLE_ASSIGN_OR_RETURN(auto held, run.submit(queue));
  HYLE_RETURN_IF_ERROR(run.drain());
  HYLE_RETURN_IF_ERROR(held.wait());
  HYLE_ASSIGN_OR_RETURN(auto alias, run.output(graph.alias));
  HYLE_ASSIGN_OR_RETURN(auto retained, alias.storage_lease());
  alias.reset();
  run = {};
  HYLE_RETURN_IF_ERROR(read(retained, kLeft));
  if (pool.stats().active_uses != 0 || pool.stats().live_blocks != 1)
    return Status::Internal("common source alias retention or retirement");
  retained.reset();
  if (pool.stats().live_blocks != 0)
    return Status::Internal(
        "common source alias did not return the last block");
  return Status::Ok();
}
#endif

#if defined(HYLE_COMPUTE_APPLE)
Status apple() {
  namespace a = c::apple;
  namespace mx = mlx::core;
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(
      auto pool, c::LeasePool::create(memory.ref(), 1024, {.max_blocks = 2}));
  HYLE_ASSIGN_OR_RETURN(auto desc, description());
  HYLE_ASSIGN_OR_RETURN(auto aw, pool.allocate(sizeof(kLeft)));
  HYLE_ASSIGN_OR_RETURN(auto bw, pool.allocate(sizeof(kRight)));
  HYLE_ASSIGN_OR_RETURN(auto av, aw.view());
  HYLE_ASSIGN_OR_RETURN(auto bv, bw.view());
  HYLE_RETURN_IF_ERROR(av.write(std::as_bytes(std::span(kLeft))));
  HYLE_RETURN_IF_ERROR(bv.write(std::as_bytes(std::span(kRight))));
  HYLE_ASSIGN_OR_RETURN(auto left, aw.publish());
  HYLE_ASSIGN_OR_RETURN(auto right, bw.publish());
  HYLE_ASSIGN_OR_RETURN(auto native_left, memory.binding(left, desc));
  HYLE_ASSIGN_OR_RETURN(auto native_right, memory.binding(right, desc));
  std::array native_inputs{native_left, native_right};
  HYLE_ASSIGN_OR_RETURN(
      auto program,
      context.prepare(c::MatMul{}, std::span<const mx::array>(native_inputs)));
  HYLE_ASSIGN_OR_RETURN(auto direct,
                        a::LeasedInvocation::create(program, memory));
  std::array inputs{a::LeasedInput{desc, left}, a::LeasedInput{desc, right}};
  HYLE_RETURN_IF_ERROR(direct.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto direct_done, direct.submit());
  HYLE_RETURN_IF_ERROR(direct.drain());
  HYLE_ASSIGN_OR_RETURN(auto direct_result, direct_done.wait());
  HYLE_RETURN_IF_ERROR(check({direct_result->arrays()[0].data<float>(), 4}));
  HYLE_ASSIGN_OR_RETURN(auto graph, definition(desc));
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::apple::compile(context, graph.graph));
  HYLE_ASSIGN_OR_RETURN(auto run,
                        a::LeasedInvocation::create(compiled, memory));
  HYLE_RETURN_IF_ERROR(run.bind(inputs));
  left.reset();
  right.reset();
  for (auto &input : inputs)
    input.source.reset();
  HYLE_ASSIGN_OR_RETURN(auto done, run.submit());
  HYLE_RETURN_IF_ERROR(run.drain());
  HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
  HYLE_RETURN_IF_ERROR(check({result->arrays()[0].data<float>(), 4}));
  HYLE_ASSIGN_OR_RETURN(auto alias,
                        memory.import_readonly(result->arrays()[1], desc));
  if (alias.try_write() || run.submit())
    return Status::Internal("native Apple readonly/close contract");
  result = {};
  done = {};
  run = {};
  compiled = {};
  direct_result = {};
  direct_done = {};
  direct = {};
  program = {};
  native_left = mx::array(0);
  native_right = mx::array(0);
  for (auto &input : native_inputs)
    input = mx::array(0);
  HYLE_RETURN_IF_ERROR(read(alias, kLeft));
  if (pool.stats().live_blocks != 1 || pool.stats().active_uses != 0)
    return Status::Internal("Apple source alias retention or retirement");
  alias.reset();
  if (pool.stats().live_blocks != 0)
    return Status::Internal("Apple source alias did not return the last block");
  return Status::Ok();
}
#endif

#if defined(HYLE_COMPUTE_VULKAN)
Status vulkan() {
  namespace v = c::vulkan;
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(
      auto memory, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto desc, description());
  HYLE_ASSIGN_OR_RETURN(auto native_left,
                        context.upload(desc, std::as_bytes(std::span(kLeft))));
  HYLE_ASSIGN_OR_RETURN(auto native_right,
                        context.upload(desc, std::as_bytes(std::span(kRight))));
  HYLE_ASSIGN_OR_RETURN(auto left, memory.import_readonly(native_left));
  HYLE_ASSIGN_OR_RETURN(auto right, memory.import_readonly(native_right));
  HYLE_ASSIGN_OR_RETURN(auto program,
                        v::prepare(context, c::MatMul{}, desc, desc));
  HYLE_ASSIGN_OR_RETURN(auto direct,
                        v::LeasedInvocation::create(program, memory));
  std::array inputs{v::LeasedInput{desc, left}, v::LeasedInput{desc, right}};
  HYLE_RETURN_IF_ERROR(direct.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto direct_done, direct.submit());
  HYLE_RETURN_IF_ERROR(direct.drain());
  HYLE_ASSIGN_OR_RETURN(auto direct_result, direct_done.wait());
  std::array<float, 4> actual{};
  HYLE_RETURN_IF_ERROR(context.download(
      direct_result->tensors()[0], std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(check(actual));
  HYLE_ASSIGN_OR_RETURN(auto graph, definition(desc));
  HYLE_ASSIGN_OR_RETURN(auto compiled,
                        g::vulkan::compile(context, graph.graph));
  HYLE_ASSIGN_OR_RETURN(auto run,
                        v::LeasedInvocation::create(compiled, memory));
  HYLE_RETURN_IF_ERROR(run.bind(inputs));
  left.reset();
  right.reset();
  native_left = {};
  native_right = {};
  for (auto &input : inputs)
    input.source.reset();
  HYLE_ASSIGN_OR_RETURN(auto done, run.submit());
  HYLE_RETURN_IF_ERROR(run.drain());
  HYLE_ASSIGN_OR_RETURN(auto result, done.wait());
  HYLE_RETURN_IF_ERROR(context.download(
      result->tensors()[0], std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(check(actual));
  HYLE_ASSIGN_OR_RETURN(auto alias,
                        memory.import_readonly(result->tensors()[1]));
  HYLE_ASSIGN_OR_RETURN(auto retained, memory.binding(alias, desc));
  if (alias.try_write() || run.submit())
    return Status::Internal("native Vulkan readonly/close contract");
  result = {};
  done = {};
  run = {};
  compiled = {};
  alias.reset();
  direct_result = {};
  direct_done = {};
  direct = {};
  program = {};
  HYLE_RETURN_IF_ERROR(
      context.download(retained, std::as_writable_bytes(std::span(actual))));
  return check(actual, kLeft);
}
#endif

Status run() {
#if defined(HYLE_COMPUTE_CPU)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_RETURN_IF_ERROR(
      common(cpu, m::system_resource(),
             [&](const g::Graph &graph) { return g::compile(graph, cpu); }));
  std::puts(
      "CPU leased Direct+Graph math, eight replays and held drain passed");
#endif
#if defined(HYLE_COMPUTE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto cuda, g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto memory, c::cuda::MemoryResource::create(
                                         cuda, c::cuda::MemoryKind::kDevice));
  HYLE_RETURN_IF_ERROR(common(cuda, memory.ref(), [&](const g::Graph &graph) {
    return g::cuda::compile(graph, cuda, memory.ref());
  }));
  std::puts(
      "CUDA leased Direct+Graph math, eight replays and held drain passed");
#endif
#if defined(HYLE_COMPUTE_APPLE)
  HYLE_RETURN_IF_ERROR(apple());
  std::puts("Apple native leased Direct+Graph math, held drain and source "
            "alias passed");
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_RETURN_IF_ERROR(vulkan());
  std::puts("Vulkan native leased Direct+Graph math, held drain and source "
            "alias passed");
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok())
    return 0;
  const auto message = status.message();
  std::fprintf(stderr, "resource leases: %.*s\n",
               static_cast<int>(message.size()), message.data());
  return 1;
}
