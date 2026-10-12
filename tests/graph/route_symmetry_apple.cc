#include <hyle/graph/apple.h>
#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
#include <hyle/graph/cpu.h>
#include <hyle/graph/cpu_apple.h>
#include <hyle/sched/scheduler.h>
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <mlx/ops.h>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;
#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
namespace b = g::cpu_apple;
#endif
namespace mx = mlx::core;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  const b::Environment environment{.cpu=&cpu, .apple=&context, .native=&memory};
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 32);
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain(){scheduler.drain();} } drain{scheduler};
#endif
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows, builder.dimension({1,4}));
  HYLE_ASSIGN_OR_RETURN(auto spec, g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}));
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto first, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{x}, std::array{spec}));
  // Independent data branch; the first node is live only through control.
  HYLE_ASSIGN_OR_RETURN(auto last, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{x}, std::array{spec}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(last, first));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(last));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array one{g::ShapeBinding{rows,1}}, four{g::ShapeBinding{rows,4}};
  const std::array points{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  HYLE_ASSIGN_OR_RETURN(auto family, g::apple::compile_variants(context, graph, points));
  HYLE_RETURN_IF_ERROR(require(!family.select(std::array{g::ShapeBinding{rows,2}}), "Apple covered undeclared shape"));
  HYLE_ASSIGN_OR_RETURN(auto native, family.select(four));
  HYLE_ASSIGN_OR_RETURN(auto first_slot, native.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto second_slot, native.create_invocation());
  std::array<float,32> values, changed;
  for (std::size_t i=0;i<values.size();++i) { values[i]=float(i)-16; changed[i]=2*values[i]; }
  auto input=mx::array(values.data(), {4,8});
  auto other=mx::array(changed.data(), {4,8});
  HYLE_RETURN_IF_ERROR(first_slot.bind(std::array{input}));
  HYLE_ASSIGN_OR_RETURN(auto submitted, first_slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto old, submitted.wait());
  HYLE_RETURN_IF_ERROR(second_slot.bind(std::array{other}));
  HYLE_ASSIGN_OR_RETURN(auto next, second_slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto fresh, next.wait());
  const auto *before=old->arrays()[0].data<float>();
  const auto *after=fresh->arrays()[0].data<float>();
  for (std::size_t i=0;i<values.size();++i)
    HYLE_RETURN_IF_ERROR(require(before[i]==-values[i] && after[i]==-changed[i], "Apple slots or controlled math differ"));

#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
  const std::array placement{g::NodePlacement{first,b::kCpuTarget,g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto hybrids, b::compile_variants(graph,environment,points,{.node_placements=placement}));
  for (const auto &shape : {one,four}) {
    HYLE_ASSIGN_OR_RETURN(auto point,g::specialize_exact(graph,shape));
    HYLE_ASSIGN_OR_RETURN(auto desc,point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(auto host,hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    std::vector<float> data(*desc.element_count(),3.0f);
    HYLE_RETURN_IF_ERROR(host.view().write(std::as_bytes(std::span(data))));
    HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(desc,host.view()));
    HYLE_ASSIGN_OR_RETURN(auto program,hybrids.select(shape));
    HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation(scheduler,queue));
    HYLE_RETURN_IF_ERROR(slot.bind(x,tensor));
    auto forbidden=slot.prepare_replay({.host_wait=g::HostWaitPolicy::kReject});
    HYLE_RETURN_IF_ERROR(require(!forbidden.ok(),"Apple hybrid hid control completion waits"));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto result,slot.run());
    HYLE_ASSIGN_OR_RETURN(auto array,result->array(output));
    for (std::size_t i=0;i<data.size();++i)
      HYLE_RETURN_IF_ERROR(require(array.data<float>()[i]==-3.0f,"Apple hybrid exact/control math differs"));
    HYLE_RETURN_IF_ERROR(require(!slot.reset().ok(),"Apple hybrid old result failed to guard slot"));
  }
#endif

  // The checked predecessor has no tensor consumer. Its accepted failure must
  // still prevent an otherwise valid, independent native successor result.
  HYLE_ASSIGN_OR_RETURN(auto checked, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto table_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{2,8}));
  HYLE_ASSIGN_OR_RETURN(auto index_desc, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t,1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto values_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,1>{8}));
  HYLE_ASSIGN_OR_RETURN(auto table_input, checked.input(table_desc));
  HYLE_ASSIGN_OR_RETURN(auto index_input, checked.input(index_desc));
  HYLE_ASSIGN_OR_RETURN(auto values_input, checked.input(values_desc));
  HYLE_ASSIGN_OR_RETURN(auto predecessor, checked.add_node(
      c::Operation(c::Gather{0,c::IndexBoundsPolicy::kChecked}),
      std::array{table_input,index_input}));
  HYLE_ASSIGN_OR_RETURN(auto successor, checked.add_node(
      c::Operation(c::Unary{c::UnaryKind::kNeg}),std::array{values_input}));
  HYLE_RETURN_IF_ERROR(checked.depends_on(successor,predecessor));
  HYLE_ASSIGN_OR_RETURN(auto controlled_output, checked.result(successor));
  HYLE_RETURN_IF_ERROR(checked.publish(controlled_output));
  HYLE_ASSIGN_OR_RETURN(auto controlled_graph, std::move(checked).finish());
  HYLE_ASSIGN_OR_RETURN(auto controlled, g::apple::compile(context,controlled_graph));
  const std::array<float,16> table_values{};
  const std::array<float,8> independent{1,2,3,4,5,6,7,8};
  const std::array<std::int32_t,1> bad{2},good{1};
  const auto native_table=mx::array(table_values.data(),{2,8});
  const auto native_values=mx::array(independent.data(),{8});
  HYLE_ASSIGN_OR_RETURN(auto failing, controlled.create_invocation());
  HYLE_RETURN_IF_ERROR(failing.bind(std::array{native_table,mx::array(bad.data(),{1}),native_values}));
  HYLE_ASSIGN_OR_RETURN(auto accepted, failing.submit());
  auto failure=accepted.wait();
  HYLE_RETURN_IF_ERROR(require(!failure && failure.error().IsOutOfRange(),
      "Apple erased a failing control-only predecessor"));
  HYLE_ASSIGN_OR_RETURN(auto succeeding, controlled.create_invocation());
  HYLE_RETURN_IF_ERROR(succeeding.bind(std::array{native_table,mx::array(good.data(),{1}),native_values}));
  HYLE_ASSIGN_OR_RETURN(auto valid, succeeding.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, valid.wait());
  const auto *actual=result->arrays()[0].data<float>();
  for(std::size_t i=0;i<independent.size();++i)
    HYLE_RETURN_IF_ERROR(require(actual[i]==-independent[i],"Apple checked control success differs"));
  return Status::Ok();
}
} // namespace

int main() {
  const auto status=run();
  if(!status.ok()){std::fprintf(stderr,"foundation A Apple: %.*s\n",int(status.message().size()),status.message().data());return 1;}
  std::puts("foundation A Apple exact/control/slot journey PASS");
}
