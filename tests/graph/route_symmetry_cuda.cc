#include <hyle/graph/cpu.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/blocking.h>

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
  HYLE_ASSIGN_OR_RETURN(auto family,b::compile(graph,environment,
      {.shape_policy=g::CompileShapePolicy::kFiniteExact,.exact_variants=points,
       .node_placements=placements,.external_resources=external}));
  HYLE_RETURN_IF_ERROR(require(!family.executable.has_execution_route() &&
      family.executable.exact_variant_count()==points.size(),"CPU+CUDA finite parent invented a general route"));
  const std::array absent{g::ShapeBinding{rows,2}};
  auto missing=family.executable.create_invocation({.shape=absent});
  HYLE_RETURN_IF_ERROR(require(!missing&&missing.error().IsUnimplemented(),"CUDA finite family covered undeclared point"));
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu_queue,cuda.create_queue());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&gpu_queue},g::QueueBinding{g::ExecutionLaneId{1},&cpu_queue}};
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool pool(hyle::mem::system_allocator(),1);
  struct Drain {hyle::sched::Scheduler &scheduler;~Drain(){scheduler.drain();}} drain{scheduler};
  for(const auto &shape:{one,four}) {
    HYLE_ASSIGN_OR_RETURN(auto native,family.executable.create_invocation({.shape=shape}));
    HYLE_ASSIGN_OR_RETURN(auto actual_rows,native.specialization().binding(0));
    HYLE_RETURN_IF_ERROR(require(actual_rows.dimension==rows && actual_rows.value==shape[0].value,
        "CPU+CUDA selected signature differs from its requested shape"));
    HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t,2>{actual_rows.value,8}));
    HYLE_ASSIGN_OR_RETURN(auto host,hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    std::array<float,32> values{};
    for(std::size_t i=0;i<*desc.element_count();++i)values[i]=float(i)-4;
    HYLE_RETURN_IF_ERROR(host.view().write(std::as_bytes(std::span(values).first(*desc.element_count()))));
    HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(desc,host.view()));
    HYLE_ASSIGN_OR_RETURN(auto destination,hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    HYLE_ASSIGN_OR_RETURN(auto output_tensor,c::TensorView::create(desc,destination.view()));
    auto invocation=std::move(native);
    HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation,scheduler,pool));
    HYLE_RETURN_IF_ERROR(invocation.bind(input,tensor));
    HYLE_RETURN_IF_ERROR(invocation.bind(output,output_tensor));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto pending,invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(pending.wait());
    HYLE_ASSIGN_OR_RETURN(auto result,invocation.result(output));
    HYLE_ASSIGN_OR_RETURN(auto actual,result.tensor());
    std::array<float,32> observed{};
    HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(observed).first(*desc.element_count()))));
    for(std::size_t i=0;i<*desc.element_count();++i)
      HYLE_RETURN_IF_ERROR(require(observed[i]==-std::abs(values[i]),"CPU+CUDA exact math differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(),"CPU+CUDA old result did not guard slot"));
  }
  auto quota=b::compile(graph,environment,
      {.shape_policy=g::CompileShapePolicy::kFiniteExact,.exact_variants=points,
       .compiled_storage={.max_bytes=1},.node_placements=placements,.external_resources=external});
  HYLE_RETURN_IF_ERROR(require(!quota&&quota.error().IsResourceExhausted(),
      "CPU+CUDA finite family did not enforce its cumulative compiled storage quota"));
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"Foundation A CPU+CUDA: %.*s\n",int(status.message().size()),status.message().data());return 1;}std::puts("Foundation A CPU+CUDA finite family PASS");}
