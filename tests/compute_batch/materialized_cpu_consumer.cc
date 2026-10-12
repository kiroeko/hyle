#include <hyle/graph/batch.h>
#include <hyle/graph/cpu.h>
#include <hyle/compute/leased_operation.h>
#include <hyle/simd/dtype.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
using hyle::Status;template<class T>using Result=hyle::Result<T>;
float decode(std::uint16_t bits,bool bf16){if(bf16)return std::bit_cast<float>(std::uint32_t(bits)<<16);const auto e=(bits>>10)&31u,m=bits&1023u;const float sign=bits&32768?-1.f:1.f;if(e==31)return m?NAN:sign*INFINITY;return sign*float(e?std::ldexp(1.+double(m)/1024.,int(e)-15):std::ldexp(double(m),-24));}
std::uint16_t encode(float value,bool bf16){return bf16?hyle::simd::f32_to_bf16(value).bits:hyle::simd::f32_to_f16(value).bits;}
Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
template<class T>Result<c::ReadLease> load(c::LeasePool&pool,const std::vector<T>&values){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(values.size()*sizeof(T)));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(values))));return writer.publish();}
Status oracle(const std::vector<std::uint16_t>&q,const std::vector<std::uint16_t>&k,const std::vector<std::uint16_t>&v,const c::BatchLayoutVersion&layout,const c::ReadLease&output,bool bf16,bool causal){
  const auto cap=layout.schema().capacity();const auto host=layout.host();std::vector<std::uint16_t> values(q.size());HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(values))));double high_error=0;
  for(unsigned r=0;r<layout.facts().requests;++r){const int qlength=host.query_indptr[r+1]-host.query_indptr[r],count=host.kv_indptr[r+1]-host.kv_indptr[r];const int length=cap.storage==c::BatchStorage::kPaged?(count-1)*int(cap.page_size)+host.last_page_lengths[r]:count;
    const auto physical=[&](int i){return cap.storage==c::BatchStorage::kPaged?host.page_indices[host.kv_indptr[r]+i/cap.page_size]*int(cap.page_size)+i%cap.page_size:host.kv_indptr[r]+i;};
    for(int qi=0;qi<qlength;++qi)for(unsigned h=0;h<cap.query_heads;++h){const auto row=host.query_indptr[r]+qi;const int valid=causal?length-qlength+qi+1:length;std::vector<double> scores(valid);double maximum=-INFINITY,sum=0;
      for(int i=0;i<valid;++i){double dot=0;for(unsigned d=0;d<cap.head_dim;++d)dot+=double(decode(q[(row*cap.query_heads+h)*cap.head_dim+d],bf16))*decode(k[(physical(i)*cap.kv_heads+h/(cap.query_heads/cap.kv_heads))*cap.head_dim+d],bf16);scores[i]=dot/std::sqrt(double(cap.head_dim));maximum=std::max(maximum,scores[i]);}
      for(auto&score:scores){score=std::exp(score-maximum);sum+=score;}for(unsigned d=0;d<cap.head_dim;++d){double expected=0;for(int i=0;i<valid;++i)expected+=scores[i]/sum*decode(v[(physical(i)*cap.kv_heads+h/(cap.query_heads/cap.kv_heads))*cap.head_dim+d],bf16);const double error=std::abs(decode(values[(row*cap.query_heads+h)*cap.head_dim+d],bf16)-expected);if(!std::isfinite(error)||error>.007)return Status::Internal("materialized reference independent oracle differs");high_error=std::max(high_error,error);}
    }
  }
  std::printf("materialized independent oracle maxError=%.9g PASS\n",high_error);return Status::Ok();
}
Status exercise(c::Context&context,c::Queue&queue,bool paged,bool bf16,bool causal){
  c::BatchCapacity cap{.storage=paged?c::BatchStorage::kPaged:c::BatchStorage::kRagged,.data_type=bf16?c::DataType::kBFloat16:c::DataType::kFloat16,.max_requests=2,.max_query_rows=4,.max_kv_rows=32,.max_pages=paged?5u:0u,.max_page_refs=paged?3u:0u,.query_heads=4,.kv_heads=2,.head_dim=64,.page_size=paged?16u:1u};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,3> qptr{0,1,3},kvptr{0,paged?2:19,paged?3:29},pages{2,0,3};const std::array<std::int32_t,2> tails{3,10};
  HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,paged?std::span(pages):std::span<const std::int32_t>{},paged?std::span(tails):std::span<const std::int32_t>{}}));
  HYLE_ASSIGN_OR_RETURN(auto program,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=24,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=16u<<20,.scale=.125f,.causal=causal}));HYLE_ASSIGN_OR_RETURN(auto version,program.version(layout));
  const auto descriptions=program.descriptions();std::vector<std::uint16_t> q(4*4*64),k((paged?5*16:32)*2*64,encode(NAN,bf16)),v(k.size(),encode(NAN,bf16));
  for(unsigned i=0;i<q.size();++i)q[i]=encode(.2f*std::sin(float(i)*.017f),bf16);
  for(unsigned r=0;r<2;++r){const int length=r?10:19;for(int i=0;i<length;++i){const auto row=paged?pages[kvptr[r]+i/16]*16+i%16:kvptr[r]+i;for(unsigned h=0;h<2;++h)for(unsigned d=0;d<64;++d){const auto index=(row*2+h)*64+d;k[index]=encode(.3f*std::cos(float(index)*.019f),bf16);v[index]=encode(std::sin(float(index)*.013f),bf16);}}}
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),64u<<20,{.max_blocks=64}));std::array<c::ReadLease,24> values;
  HYLE_ASSIGN_OR_RETURN(values[0],load(pool,q));HYLE_ASSIGN_OR_RETURN(values[1],load(pool,k));HYLE_ASSIGN_OR_RETURN(values[2],load(pool,v));HYLE_ASSIGN_OR_RETURN(values[3],version.indices());HYLE_ASSIGN_OR_RETURN(values[4],version.mask());
  // Direct cold prepares every actual operation before executing any math.
  std::array<c::LeasedOperationInvocation,19> direct;
  for(std::size_t i=0;i<program.steps().size();++i){const auto&step=program.steps()[i];if(step.operation.kind()==c::OperationKind::kView)continue;std::array<c::TensorDesc,2> inputs;for(unsigned j=0;j<step.input_count;++j)inputs[j]=descriptions[step.inputs[j]];
    HYLE_ASSIGN_OR_RETURN(auto prepared,c::prepare(context,step.operation,std::span(inputs).first(step.input_count),std::span(&descriptions[step.output],1),hyle::mem::system_resource()));c::WriteLease workspace;auto required=prepared.workspace_requirement();if(required.size){HYLE_ASSIGN_OR_RETURN(workspace,pool.allocate(required.size,std::max(required.alignment,std::size_t{256})));}HYLE_ASSIGN_OR_RETURN(direct[i],c::LeasedOperationInvocation::create(prepared,std::move(workspace)));
  }
  for(std::size_t i=0;i<program.steps().size();++i){const auto&step=program.steps()[i];if(step.operation.kind()==c::OperationKind::kView){values[step.output]=values[step.inputs[0]];continue;}
    HYLE_ASSIGN_OR_RETURN(auto bytes,descriptions[step.output].required_bytes());HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));std::vector<c::LeasedInput> inputs;for(unsigned j=0;j<step.input_count;++j)inputs.emplace_back(descriptions[step.inputs[j]],values[step.inputs[j]]);const std::array outputs{c::LeasedOutput(descriptions[step.output],writer)};HYLE_ASSIGN_OR_RETURN(auto done,direct[i].submit(queue,inputs,outputs));HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(values[step.output],writer.publish());
  }
  HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,values[program.output_value()],bf16,causal));
  HYLE_ASSIGN_OR_RETURN(auto logical,g::batch::materialized_graph(program));const std::array domains{g::StorageDomain{hyle::mem::system_resource()}};const std::array targets{g::ExecutionTarget{&context,g::StorageDomainId{0}}};HYLE_ASSIGN_OR_RETURN(auto compiled,g::compile(logical.graph,{domains,targets,{}}));HYLE_ASSIGN_OR_RETURN(auto invocation,g::LeasedInvocation::create(compiled.executable));
  for(unsigned i=0;i<5;++i)HYLE_RETURN_IF_ERROR(invocation.bind(logical.inputs[i],descriptions[i],values[i]));HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(q.size()*sizeof(std::uint16_t)));HYLE_RETURN_IF_ERROR(invocation.bind(logical.output,descriptions[0],std::move(writer)));HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue));HYLE_RETURN_IF_ERROR(done.wait());HYLE_ASSIGN_OR_RETURN(auto output,invocation.output(logical.output));HYLE_ASSIGN_OR_RETURN(auto lease,output.storage_lease());HYLE_RETURN_IF_ERROR(oracle(q,k,v,layout,lease,bf16,causal));
  std::printf("CPU explicit materialized %s %s causal=%d Direct/Graph NaN-padding PASS; host=%zu gather=%zu logical=%zu\n",paged?"paged":"ragged",bf16?"BF16":"F16",int(causal),program.costs().index_bytes+program.costs().mask_bytes,program.costs().gathered_kv_bytes,program.costs().logical_intermediate_bytes);return Status::Ok();
}
Status run(){HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());for(bool paged:{true,false})for(bool bf16:{false,true})for(bool causal:{false,true})HYLE_RETURN_IF_ERROR(exercise(context,queue,paged,bf16,causal));return context.drain();}
} // namespace
int main(){const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F5 materialized CPU: %s\n",text);return 1;}
