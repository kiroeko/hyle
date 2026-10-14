#include <hyle/compute/batch_execution.h>
#include <hyle/compute/backends/cuda.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/batch.h>
#include <array>
#include <cstdio>

#if !defined(_WIN32) || !defined(HYLE_COMPUTE_CUDA)
#error This consumer qualifies the declared Windows CUDA native-batch absence
#endif
namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace cu=c::cuda;
using hyle::Status;
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto context,g::cuda::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu,cu::MemoryResource::create(context,cu::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost));
  struct Drain{c::Context&context;~Drain(){context.drain().IgnoreError();}}drain{context};
  HYLE_ASSIGN_OR_RETURN(auto environment,cu::environment(context));
  c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=1,.max_query_rows=1,.max_kv_rows=1,.max_pages=0,.max_page_refs=0,.query_heads=2,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));auto direct=c::BatchProgram::prepare(context,schema,{.resource=gpu.ref(),.host_workspace=staging.ref(),.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10,.scale=.125f});
  if(direct || !direct.error().IsUnimplemented())return Status::Internal("Windows CUDA native batch absence was not explicitly diagnosed");
  char message[512]{};direct.error().ToString(message,sizeof(message));std::printf("public Windows CUDA actual GPU %s native BatchPlanning unsupported: %s PASS\n",environment.device_name.data(),message);
  HYLE_ASSIGN_OR_RETURN(auto qdesc,c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{1,2,64}));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());std::array<g::ValueId,4> ports;const std::array descriptions{qdesc,qdesc,qdesc,schema.metadata_description().value()};
  for(unsigned i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(ports[i],builder.input(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::RaggedAttention{cap,.125f}),ports));HYLE_ASSIGN_OR_RETURN(auto output,builder.result(node));HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  auto compiled=g::cuda::compile(graph,context,gpu.ref(),{.operation_fallback=g::OperationFallbackPolicy::kReject});
  if(compiled)return Status::Internal("Windows CUDA accepted unqualified native ragged Graph");
  compiled.error().ToString(message,sizeof(message));std::printf("public Windows CUDA typed ragged Graph rejected before execution: %s PASS\n",message);return Status::Ok();
}
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);const auto status=run();if(status.ok())return 0;char message[768]{};status.ToString(message,sizeof(message));std::fprintf(stderr,"public Windows CUDA absence: %s\n",message);return 1;}
