#include <hyle/graph/cpu.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>

#include <array>
#include <cmath>
#include <cstdio>

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace b=g::cpu_cuda;
using hyle::Status;
Status require(bool condition,const char *message) {
  return condition?Status::Ok():Status::Internal(message);
}
Status run() {
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto cuda,g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(auto staging,c::cuda::MemoryResource::create(cuda,c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(cuda,c::cuda::MemoryKind::kDevice));
  const b::Environment environment{.cpu=&cpu,.cuda=&cuda,.staging=&staging,.device=&device};
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows,builder.dimension({1,4}));
  HYLE_ASSIGN_OR_RETURN(auto spec,g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows),g::Dimension::fixed(8)}));
  HYLE_ASSIGN_OR_RETURN(auto input,builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto first,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input},std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto middle,builder.result(first));
  HYLE_ASSIGN_OR_RETURN(auto last,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{middle},std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.result(last));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array one{g::ShapeBinding{rows,1}},four{g::ShapeBinding{rows,4}};
  const std::array points{g::ExactShapeVariant{one},g::ExactShapeVariant{four}};
  const std::array placements{
      g::NodePlacement{first,b::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{last,b::kCudaTarget,g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{input,b::kHostStorage},
                            g::ExternalResourceContract{output,b::kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto family,b::compile_variants(graph,environment,points,
      {.node_placements=placements,.external_resources=external}));
  auto missing=family.select(std::array{g::ShapeBinding{rows,2}});
  HYLE_RETURN_IF_ERROR(require(!missing&&missing.error().IsUnimplemented(),"CUDA finite family covered undeclared point"));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu_queue,cuda.create_queue());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&gpu_queue},g::QueueBinding{g::ExecutionLaneId{1},&cpu_queue}};
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  struct Drain {hyle::sched::Scheduler &scheduler;~Drain(){scheduler.drain();}} drain{scheduler};
  for(const auto &shape:{one,four}) {
    HYLE_ASSIGN_OR_RETURN(auto point,g::specialize_exact(graph,shape));
    HYLE_ASSIGN_OR_RETURN(auto desc,point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(auto host,hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    std::array<float,32> values{};
    for(std::size_t i=0;i<*desc.element_count();++i)values[i]=float(i)-4;
    HYLE_RETURN_IF_ERROR(host.view().write(std::as_bytes(std::span(values).first(*desc.element_count()))));
    HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(desc,host.view()));
    HYLE_ASSIGN_OR_RETURN(auto destination,hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    HYLE_ASSIGN_OR_RETURN(auto output_tensor,c::TensorView::create(desc,destination.view()));
    HYLE_ASSIGN_OR_RETURN(auto program,family.select(shape));
    HYLE_ASSIGN_OR_RETURN(auto native,program.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto invocation,b::Invocation::create(std::move(native),scheduler,queues));
    HYLE_RETURN_IF_ERROR(invocation.bind(input,tensor));
    HYLE_RETURN_IF_ERROR(invocation.bind(output,output_tensor));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
    HYLE_ASSIGN_OR_RETURN(auto result,invocation.run());
    HYLE_ASSIGN_OR_RETURN(auto actual,result->output(output));
    std::array<float,32> observed{};
    HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(observed).first(*desc.element_count()))));
    for(std::size_t i=0;i<*desc.element_count();++i)
      HYLE_RETURN_IF_ERROR(require(observed[i]==-std::abs(values[i]),"CPU+CUDA exact math differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(),"CPU+CUDA old result did not guard slot"));
  }
  auto quota=b::compile_variants(graph,environment,points,{.compiled_storage={.max_bytes=1}});
  HYLE_RETURN_IF_ERROR(require(!quota&&quota.error().IsUnimplemented(),"CPU+CUDA family silently split a global quota"));
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"Foundation A CPU+CUDA: %.*s\n",int(status.message().size()),status.message().data());return 1;}std::puts("Foundation A CPU+CUDA finite family PASS");}
