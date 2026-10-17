#include <hyle/graph/vulkan.h>
#if defined(HYLE_FOUNDATION_A_HYBRID)
#include <hyle/graph/cpu.h>
#include <hyle/graph/cpu_vulkan.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/blocking.h>
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace v = c::vulkan;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

struct Symbolic {
  g::Graph graph;
  g::DimensionId rows;
  g::ValueId input, output;
};
Result<Symbolic> symbolic_graph() {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows, builder.dimension({1, 4}));
  HYLE_ASSIGN_OR_RETURN(auto spec, g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto first, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input}, std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto middle, builder.result(first));
  HYLE_ASSIGN_OR_RETURN(auto last, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{middle}, std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(last));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return Symbolic{std::move(graph), rows, input, output};
}

Status variants(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto f, symbolic_graph());
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  HYLE_ASSIGN_OR_RETURN(auto family, g::vulkan::compile_variants(context, f.graph, table));
  HYLE_RETURN_IF_ERROR(require(family.variant_count() == 2, "native variant count differs"));
  auto absent = family.select(std::array{g::ShapeBinding{f.rows, 2}});
  HYLE_RETURN_IF_ERROR(require(!absent && absent.error().IsUnimplemented(), "native family covered undeclared shape"));
  HYLE_ASSIGN_OR_RETURN(auto program, family.select(four));
  HYLE_ASSIGN_OR_RETURN(auto first, program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto second, program.create_invocation());
  const auto desc = program.input_descriptions()[0];
  std::array<float, 32> values, changed;
  for (std::size_t i = 0; i < values.size(); ++i) { values[i] = float(i) - 16; changed[i] = 2 * values[i]; }
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(desc, std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto other, context.upload(desc, std::as_bytes(std::span(changed))));
  HYLE_RETURN_IF_ERROR(first.bind(input));
  HYLE_ASSIGN_OR_RETURN(auto submitted, first.submit());
  HYLE_ASSIGN_OR_RETURN(auto old, submitted.wait());
  HYLE_RETURN_IF_ERROR(second.bind(other));
  HYLE_ASSIGN_OR_RETURN(auto next, second.submit());
  HYLE_ASSIGN_OR_RETURN(auto fresh, next.wait());
  std::array<float, 32> actual, original;
  HYLE_RETURN_IF_ERROR(context.download(fresh->tensors()[0], std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(context.download(old->tensors()[0], std::as_writable_bytes(std::span(original))));
  for (std::size_t i = 0; i < values.size(); ++i)
    HYLE_RETURN_IF_ERROR(require(actual[i] == -std::abs(changed[i]) && original[i] == -std::abs(values[i]),
                                 "native slots shared output or math differs"));
  return Status::Ok();
}

struct Control {
  g::Graph graph;
  g::ValueId table, index, input, output;
};
Result<Control> control_graph() {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto table_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 8}));
  HYLE_ASSIGN_OR_RETURN(auto index_desc, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto input_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{8}));
  HYLE_ASSIGN_OR_RETURN(auto table, builder.input(table_desc));
  HYLE_ASSIGN_OR_RETURN(auto index, builder.input(index_desc));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(input_desc));
  HYLE_ASSIGN_OR_RETURN(auto prerequisite, builder.add_node(c::Operation(c::Gather{0, c::IndexBoundsPolicy::kChecked}),
      std::array{table, index}));
  HYLE_ASSIGN_OR_RETURN(auto successor, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{input}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(successor, prerequisite));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(successor));
  HYLE_RETURN_IF_ERROR(builder.publish(output)); // Gather has no data consumer.
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return Control{std::move(graph), table, index, input, output};
}

Status control(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto f, control_graph());
  g::vulkan::CompileReport facts;
  HYLE_ASSIGN_OR_RETURN(auto program, g::vulkan::compile(context, f.graph, {}, {.report = &facts}));
  HYLE_RETURN_IF_ERROR(require(facts.nodes.size() == 2, "native compiler erased control-only predecessor"));
  std::array<float, 16> table{};
  const std::array<float, 8> values{1,2,3,4,5,6,7,8};
  const std::array<std::int32_t, 1> bad{2}, good{1};
  HYLE_ASSIGN_OR_RETURN(auto data, context.upload(*f.graph.input_description(0), std::as_bytes(std::span(table))));
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(*f.graph.input_description(2), std::as_bytes(std::span(values))));
  for (const auto &indices : {bad, good}) {
    HYLE_ASSIGN_OR_RETURN(auto index, context.upload(*f.graph.input_description(1), std::as_bytes(std::span(indices))));
    HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
    HYLE_RETURN_IF_ERROR(invocation.bind(std::array{data, index, input}));
    HYLE_ASSIGN_OR_RETURN(auto submitted, invocation.submit());
    auto result = submitted.wait();
    if (indices == bad) {
      HYLE_RETURN_IF_ERROR(require(!result && result.error().IsOutOfRange(), "control predecessor failure was erased"));
    } else {
      if (!result) return result.error();
      std::array<float, 8> actual;
      HYLE_RETURN_IF_ERROR(context.download((*result)->tensors()[0], std::as_writable_bytes(std::span(actual))));
      for (std::size_t i=0;i<actual.size();++i)
        HYLE_RETURN_IF_ERROR(require(actual[i] == -values[i], "controlled native math differs"));
    }
  }
  return Status::Ok();
}

#if defined(HYLE_FOUNDATION_A_HYBRID)
Status hybrid(const v::Context &context) {
  namespace b = g::cpu_vulkan;
  namespace m = hyle::mem;
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  const b::Environment environment{.cpu=&cpu, .vulkan=&context, .device=&device, .staging=&staging};
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(), 1);
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain(){scheduler.drain();} } drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto f, symbolic_graph());
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  const std::array placement{g::NodePlacement{*f.graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto family, b::compile_variants(f.graph, environment, table, {.node_placements=placement}));
  HYLE_RETURN_IF_ERROR(require(!family.select(std::array{g::ShapeBinding{f.rows,2}}), "hybrid covered undeclared shape"));
  for (const auto &shape : {one, four}) {
    HYLE_ASSIGN_OR_RETURN(auto point, g::specialize_exact(f.graph, shape));
    HYLE_ASSIGN_OR_RETURN(auto desc, point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(auto storage, m::Buffer::create(m::system_resource(), *desc.required_bytes(), 64));
    HYLE_ASSIGN_OR_RETURN(auto input, c::TensorView::create(desc, storage.view()));
    std::vector<float> values(*desc.element_count(), -3.0f), actual(values.size());
    HYLE_RETURN_IF_ERROR(storage.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto program, family.select(shape));
    HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(scheduler, queue, completion_waiters));
    HYLE_RETURN_IF_ERROR(invocation.bind(f.input, input.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.run());
    HYLE_ASSIGN_OR_RETURN(auto native, result->tensor(f.output));
    HYLE_RETURN_IF_ERROR(context.download(native, std::as_writable_bytes(std::span(actual))));
    for (auto value : actual) HYLE_RETURN_IF_ERROR(require(value == -3.0f, "hybrid exact math differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old hybrid result failed to guard slot"));
  }
  HYLE_ASSIGN_OR_RETURN(auto controlled, control_graph());
  const std::array last_cpu{g::NodePlacement{*controlled.graph.node_id(1), b::kCpuTarget, g::PlacementStrength::kRequired}};
  b::CompileReport facts;
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(controlled.graph, environment, {.node_placements=last_cpu}, &facts));
  HYLE_RETURN_IF_ERROR(require(facts.fragments.size() == 2, "hybrid erased control-only fragment"));
  std::array<m::Buffer, 3> backing;
  std::array<c::ConstTensorView, 3> inputs;
  for (std::size_t i=0;i<inputs.size();++i) {
    HYLE_ASSIGN_OR_RETURN(auto desc, controlled.graph.input_description(i));
    HYLE_ASSIGN_OR_RETURN(backing[i], m::Buffer::create(m::system_resource(), *desc.required_bytes(), 64));
    HYLE_ASSIGN_OR_RETURN(inputs[i], c::ConstTensorView::create(desc, backing[i].view()));
    std::vector<std::byte> zero(*desc.required_bytes());
    HYLE_RETURN_IF_ERROR(backing[i].view().write(zero));
  }
  const std::array<std::int32_t, 1> bad{2};
  HYLE_RETURN_IF_ERROR(backing[1].view().write(std::as_bytes(std::span(bad))));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(scheduler, queue, completion_waiters));
  for (std::size_t i=0;i<inputs.size();++i) HYLE_RETURN_IF_ERROR(invocation.bind(*controlled.graph.input_id(i), inputs[i]));
  auto reject_wait = invocation.prepare_replay({.host_wait=g::HostWaitPolicy::kReject});
  HYLE_RETURN_IF_ERROR(require(!reject_wait.ok(), "hybrid silently accepted forbidden host wait"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
  auto result = invocation.run();
  HYLE_RETURN_IF_ERROR(require(!result && result.error().IsOutOfRange(), "hybrid erased failed control predecessor"));
  return Status::Ok();
}
#endif

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(variants(context));
  HYLE_RETURN_IF_ERROR(control(context));
#if defined(HYLE_FOUNDATION_A_HYBRID)
  HYLE_RETURN_IF_ERROR(hybrid(context));
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (!status.ok()) { std::fprintf(stderr,"foundation A Vulkan: %.*s\n",int(status.message().size()),status.message().data()); return 1; }
  std::puts("foundation A Vulkan exact/control/slot journey PASS");
}
