#include "hyle/compute/state_clone.h"
#include "hyle/compute/leased_operation.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/executable.h"
#if defined(HYLE_ROW_CACHE_CUDA)
#include "hyle/graph/cuda.h"
#include "hyle/compute/backends/cuda.h"
#else
#include "hyle/graph/cpu.h"
#endif
#include "../compute_models/row_cache_fixture.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;
using hyle::Status;template<class T>using Result=hyle::Result<T>;
Status require(bool condition,const char *message){return condition?Status::Ok():Status::DataLoss(message);}
Result<c::TensorDesc> dense(c::DataType type,std::initializer_list<std::int64_t> shape){return c::TensorDesc::dense(type,{shape.begin(),shape.size()});}
template<class T>Result<std::vector<T>> read(const c::ReadLease &lease){
  std::vector<T> values(lease.size()/sizeof(T));HYLE_ASSIGN_OR_RETURN(auto view,lease.view());
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(values))));return values;
}
template<class Compile>struct Route {
  c::Context &context;c::Queue &queue;m::MemoryResourceRef resource;c::LeasePool &pool;Compile compile;
  Result<c::ReadLease> load(std::span<const std::byte> bytes){
    HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto view,writer.view());HYLE_RETURN_IF_ERROR(view.write(bytes));return writer.publish();
  }
  Result<c::ReadLease> math(c::Operation operation,std::span<const c::TensorDesc> descriptions,
      std::span<const c::ReadLease> inputs,c::TensorDesc output){
    HYLE_ASSIGN_OR_RETURN(auto prepared,c::prepare(context,operation,descriptions,std::span(&output,1),resource));
    HYLE_ASSIGN_OR_RETURN(auto bytes,output.required_bytes());HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));
    c::WriteLease workspace;if(prepared.workspace_requirement().size){HYLE_ASSIGN_OR_RETURN(workspace,pool.allocate(prepared.workspace_requirement().size));}
    HYLE_ASSIGN_OR_RETURN(auto invocation,c::LeasedOperationInvocation::create(prepared,std::move(workspace)));
    std::vector<c::LeasedInput> leases;for(std::size_t i=0;i<inputs.size();++i)leases.emplace_back(descriptions[i],inputs[i]);
    const std::array outputs{c::LeasedOutput{output,writer}};
    {HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue,leases,outputs));HYLE_RETURN_IF_ERROR(done.wait());}
    return writer.publish();
  }
  Result<c::WriteLease> copy_fragment(c::ReadLease source,c::WriteLease destination,std::size_t offset){
    const auto generation=destination.generation();HYLE_ASSIGN_OR_RETURN(auto original,destination.view());
    const std::array specs{c::CopyResourceSpec{resource,source.size()},c::CopyResourceSpec{resource,destination.size()}};
    HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
    HYLE_ASSIGN_OR_RETURN(auto slot,c::CopyBatchInvocation::create(program,queue));
    const auto bytes=source.size();HYLE_RETURN_IF_ERROR(slot.bind_source({0},source));
    HYLE_RETURN_IF_ERROR(slot.bind_mutable({1},std::move(destination)));
    const std::array<c::CopyPhase,1> phases{{{}}};
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,offset,bytes}}};
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(phases,ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    HYLE_RETURN_IF_ERROR(require(!slot.unbind_mutable({1}),"retained copy submission returned writer"));
    submitted={}; // Return the whole original writer, never publish a subrange as a whole chunk.
    HYLE_ASSIGN_OR_RETURN(auto returned,slot.unbind_mutable({1}));HYLE_ASSIGN_OR_RETURN(auto view,returned.view());
    HYLE_RETURN_IF_ERROR(require(returned.generation()==generation && view.native_handle()==original.native_handle() &&
        view.offset()==original.offset() && view.size()==original.size(),"range copy changed writer root/generation/extent"));
    const auto report=slot.report();HYLE_RETURN_IF_ERROR(require(report.logical_bytes==bytes,"range copy charged wrong active bytes"));
    std::printf("F2 append offset=%zu bytes=%zu retained chunk=%zu same root/generation\n",offset,bytes,returned.size());
    return returned;
  }
};

std::vector<std::uint16_t> bounded(c::RowCacheEncoding encoding,int phase){
  std::vector<std::uint16_t> values(5*encoding.depth());
  for(std::size_t row=0;row<5;++row)for(std::size_t d=0;d<encoding.depth();++d)
    values[row*encoding.depth()+d]=row_cache_test::bfloat(double(int((row*7+d*3+phase*11)%33)-16)/8);
  return values;
}

template<class Route>Result<c::ReadLease> encode(Route &route,c::RowCacheEncoding encoding,
    std::span<const std::uint16_t> values){
  const auto rows=static_cast<std::int64_t>(values.size()/encoding.depth());
  HYLE_ASSIGN_OR_RETURN(auto input,dense(c::DataType::kBFloat16,{rows,encoding.depth()}));
  HYLE_ASSIGN_OR_RETURN(auto output,c::row_cache_carrier_description(encoding,rows));
  HYLE_ASSIGN_OR_RETURN(auto source,route.load(std::as_bytes(values)));
  return route.math(c::model_ops::row_cache_encode(encoding),std::array{input},std::array{source},output);
}

template<class Route>Status selected(Route &route,c::RowCacheEncoding encoding,const c::ReadLease &bank,
    std::span<const std::uint16_t> expected,std::uint64_t visible){
  HYLE_ASSIGN_OR_RETURN(auto bd,c::row_cache_carrier_description(encoding,5));
  HYLE_ASSIGN_OR_RETURN(auto id,dense(c::DataType::kInt32,{2,3}));
  const std::array<std::int32_t,6> ids{std::int32_t(visible-1),1,std::int32_t(visible-1),0,2,1};
  HYLE_ASSIGN_OR_RETURN(auto index,route.load(std::as_bytes(std::span(ids))));
  const std::array descriptions{bd,id};const auto op=c::model_ops::row_cache_gather_decode(encoding);
  HYLE_ASSIGN_OR_RETURN(auto output,c::infer_single_output(op,descriptions));
  HYLE_ASSIGN_OR_RETURN(auto result,route.math(op,descriptions,std::array{bank,index},output));
  HYLE_ASSIGN_OR_RETURN(auto actual,read<std::uint16_t>(result));
  for(std::size_t i=0;i<ids.size();++i)for(std::size_t d=0;d<encoding.depth();++d)
    HYLE_RETURN_IF_ERROR(require(actual[i*encoding.depth()+d]==expected[std::size_t(ids[i])*encoding.depth()+d],"StateStore selected BF16 oracle"));
  return Status::Ok();
}

template<class Route>Status sink(Route &route,const c::ReadLease &bank,std::span<const std::uint16_t> decoded,
    std::uint64_t visible,int phase){
  constexpr std::size_t B=2,H=64,D=512,S=3;const float scale=1.F/std::sqrt(float(D));
  const auto encoding=c::RowCacheEncoding::float8_block64_tail64();
  HYLE_ASSIGN_OR_RETURN(auto bd,c::row_cache_carrier_description(encoding,5));
  HYLE_ASSIGN_OR_RETURN(auto id,dense(c::DataType::kInt32,{B,S}));
  HYLE_ASSIGN_OR_RETURN(auto qd,dense(c::DataType::kFloat32,{B,1,H,D}));
  HYLE_ASSIGN_OR_RETURN(auto md,dense(c::DataType::kFloat32,{B,1,S}));
  HYLE_ASSIGN_OR_RETURN(auto sd,dense(c::DataType::kFloat32,{H}));
  const std::array descriptions{bd,id,qd,md,sd};
  const std::array<std::int32_t,B*S> ids{std::int32_t(visible-1),1,std::int32_t(visible-1),0,2,1};
  std::vector<float> query(B*H*D),mask{0,-INFINITY,-.25F,-.125F,0,-INFINITY},sinks(H);
  for(std::size_t i=0;i<query.size();++i)query[i]=float(int((i*13+phase*7)%43)-21)/64;
  for(std::size_t h=0;h<H;++h)sinks[h]=float(int((h*7+phase*3)%17)-8)/8;
  HYLE_ASSIGN_OR_RETURN(auto indices,route.load(std::as_bytes(std::span(ids))));
  HYLE_ASSIGN_OR_RETURN(auto q,route.load(std::as_bytes(std::span(query))));
  HYLE_ASSIGN_OR_RETURN(auto m,route.load(std::as_bytes(std::span(mask))));
  HYLE_ASSIGN_OR_RETURN(auto sk,route.load(std::as_bytes(std::span(sinks))));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());std::array<g::ValueId,5> ports;
  for(std::size_t i=0;i<ports.size();++i){HYLE_ASSIGN_OR_RETURN(ports[i],builder.input(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto gathered_node,builder.add_node(c::model_ops::row_cache_gather_decode(encoding),std::array{ports[0],ports[1]}));
  HYLE_ASSIGN_OR_RETURN(auto gathered,builder.result(gathered_node));
  HYLE_ASSIGN_OR_RETURN(auto floats,builder.add(c::Cast{c::DataType::kFloat32},gathered));
  HYLE_ASSIGN_OR_RETURN(auto shape,c::TensorShape::create(std::array<std::int64_t,4>{B,S,1,D}));
  HYLE_ASSIGN_OR_RETURN(auto reshape,c::View::reshape(shape));HYLE_ASSIGN_OR_RETURN(auto kv,builder.add(reshape,floats));
  HYLE_ASSIGN_OR_RETURN(auto sink_node,builder.add_node(c::model_ops::sink_attention(scale),std::array{ports[2],kv,kv,ports[3],ports[4]}));
  HYLE_ASSIGN_OR_RETURN(auto output,builder.result(sink_node));HYLE_ASSIGN_OR_RETURN(auto negative,builder.add(c::Unary{c::UnaryKind::kNeg},output));
  HYLE_RETURN_IF_ERROR(builder.publish(output));HYLE_RETURN_IF_ERROR(builder.publish(negative));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());HYLE_ASSIGN_OR_RETURN(auto compiled,route.compile(graph));
  HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  const std::array<c::ReadLease,5> inputs{bank,indices,q,m,sk};
  for(std::size_t i=0;i<inputs.size();++i)HYLE_RETURN_IF_ERROR(invocation.bind(ports[i],descriptions[i],inputs[i]));
  for(auto value:{output,negative}){HYLE_ASSIGN_OR_RETURN(auto writer,route.pool.allocate(B*H*D*sizeof(float)));HYLE_RETURN_IF_ERROR(invocation.bind(value,qd,std::move(writer)));}
  {HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(route.queue));HYLE_RETURN_IF_ERROR(done.wait());}
  HYLE_ASSIGN_OR_RETURN(auto result,invocation.result(output));HYLE_ASSIGN_OR_RETURN(auto result_lease,result.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto neg_result,invocation.result(negative));HYLE_ASSIGN_OR_RETURN(auto neg_lease,neg_result.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto actual,read<float>(result_lease));HYLE_ASSIGN_OR_RETURN(auto neg,read<float>(neg_lease));
  double maximum_error=0;
  for(std::size_t b=0;b<B;++b)for(std::size_t h=0;h<H;++h){
    std::array<double,S> scores,weights;double maximum=sinks[h];
    for(std::size_t s=0;s<S;++s){double dot=0;for(std::size_t d=0;d<D;++d)dot+=double(query[(b*H+h)*D+d])*row_cache_test::decode(decoded[std::size_t(ids[b*S+s])*D+d]);scores[s]=dot*scale+mask[b*S+s];maximum=(std::max)(maximum,scores[s]);}
    double denominator=std::exp(double(sinks[h])-maximum);for(std::size_t s=0;s<S;++s){weights[s]=std::exp(scores[s]-maximum);denominator+=weights[s];}
    for(std::size_t d=0;d<D;++d){double expected=0;for(std::size_t s=0;s<S;++s)expected+=weights[s]*row_cache_test::decode(decoded[std::size_t(ids[b*S+s])*D+d]);expected/=denominator;const auto at=(b*H+h)*D+d;const double error=std::abs(double(actual[at])-expected);maximum_error=(std::max)(maximum_error,error);HYLE_RETURN_IF_ERROR(require(std::isfinite(actual[at])&&error<=2e-5+2e-5*std::abs(expected),"packed selected Sink FP64 oracle"));HYLE_RETURN_IF_ERROR(require(neg[at]==-actual[at],"Sink Neg dual publish"));}
  }
  std::printf("Graph packed selected BF16->F32->View->one Sink H64/B2/Q1/S3 +Neg visible=%llu maxabs=%.9g PASS\n",static_cast<unsigned long long>(visible),maximum_error);
  return Status::Ok();
}

template<class Route>Status exercise(Route &route,c::RowCacheEncoding encoding){
  const auto width=encoding.row_bytes(),depth=encoding.depth();const std::size_t bytes=5*width;
  HYLE_ASSIGN_OR_RETURN(auto description,c::row_cache_carrier_description(encoding,5));
  HYLE_ASSIGN_OR_RETURN(auto schema,c::StateSchema::create(std::array{c::StateEntrySpec{.description=description},c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=8}}));
  HYLE_ASSIGN_OR_RETURN(auto page_id,schema.entry(0));HYLE_ASSIGN_OR_RETURN(auto metadata_id,schema.entry(1));HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(schema));
  const auto original=bounded(encoding,0);const auto base_oracle=row_cache_test::reference(encoding,original);
  HYLE_ASSIGN_OR_RETURN(auto prefix,encode(route,encoding,std::span(original).first(3*depth)));
  HYLE_ASSIGN_OR_RETURN(auto initial,route.pool.allocate(bytes));HYLE_ASSIGN_OR_RETURN(auto blank,initial.view());HYLE_RETURN_IF_ERROR(blank.fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(auto filled,route.copy_fragment(prefix,std::move(initial),0));HYLE_ASSIGN_OR_RETURN(auto initial_lease,filled.publish());
  const std::uint64_t visible=3;HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{page_id,initial_lease},c::StateMember{metadata_id,{},std::as_bytes(std::span(&visible,1))}}));
  HYLE_ASSIGN_OR_RETURN(auto base_bytes,read<std::uint8_t>(initial_lease));
  std::vector<std::uint8_t> expected_base(bytes);std::copy_n(base_oracle.bytes.begin(),3*width,expected_base.begin());HYLE_RETURN_IF_ERROR(require(base_bytes==expected_base,"initial visible3 chunk bytes"));
  const std::array clone_specs{c::CopyResourceSpec{route.resource,bytes},c::CopyResourceSpec{route.resource,bytes}};
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchProgram::prepare(route.context,clone_specs));
  HYLE_ASSIGN_OR_RETURN(auto clones,c::StateCloneRoute::prepare(store,{.resource=route.resource,.capacity_bytes=32u*1024u,.slots=2},copy,[&](const c::CopyBatchProgram &p)noexcept{return c::CopyBatchInvocation::create(p,route.queue);}));
  {
    HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));HYLE_ASSIGN_OR_RETURN(auto rejected,store.begin(base));
    const std::uint64_t speculative_visible=5;
    HYLE_RETURN_IF_ERROR(rejected.replace_metadata(metadata_id,std::as_bytes(std::span(&speculative_visible,1))));
    HYLE_RETURN_IF_ERROR(rejected.cancel());HYLE_RETURN_IF_ERROR(require(!rejected.finalize(),"accept0 draft finalized"));
    HYLE_ASSIGN_OR_RETURN(auto unchanged,head.current());HYLE_RETURN_IF_ERROR(require(unchanged.identity()==base.identity(),"accept0 changed base head"));
  }
  for(int accepted:{1,2}){
    HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
    HYLE_ASSIGN_OR_RETURN(auto ticket,draft.clone_for_write(page_id,clones));HYLE_ASSIGN_OR_RETURN(auto writer,ticket.wait());
    auto current=original;const auto incoming=bounded(encoding,accepted);std::copy_n(incoming.begin()+3*depth,accepted*depth,current.begin()+3*depth);
    HYLE_ASSIGN_OR_RETURN(auto encoded,encode(route,encoding,std::span(current).subspan(3*depth,accepted*depth)));
    HYLE_ASSIGN_OR_RETURN(auto returned,route.copy_fragment(encoded,std::move(writer.lease()),3*width));
    HYLE_ASSIGN_OR_RETURN(auto ready,returned.publish());HYLE_RETURN_IF_ERROR(writer.complete(ready));
    const std::uint64_t next_visible=3+accepted;HYLE_RETURN_IF_ERROR(draft.replace_metadata(metadata_id,std::as_bytes(std::span(&next_visible,1))));
    HYLE_ASSIGN_OR_RETURN(auto candidate,draft.finalize());HYLE_RETURN_IF_ERROR(head.publish(base.identity(),candidate));
    HYLE_ASSIGN_OR_RETURN(auto published,head.current());HYLE_ASSIGN_OR_RETURN(auto page,published.member(page_id));HYLE_ASSIGN_OR_RETURN(auto actual,read<std::uint8_t>(page));
    const auto oracle=row_cache_test::reference(encoding,current);auto expected=expected_base;std::copy_n(oracle.bytes.begin()+3*width,accepted*width,expected.begin()+3*width);
    HYLE_RETURN_IF_ERROR(require(actual==expected,"append changed untouched prefix or suffix"));
    HYLE_ASSIGN_OR_RETURN(auto metadata,published.member(metadata_id));HYLE_ASSIGN_OR_RETURN(auto logical,read<std::uint64_t>(metadata));HYLE_RETURN_IF_ERROR(require(logical.size()==1&&logical[0]==next_visible,"joint metadata publication"));
    HYLE_ASSIGN_OR_RETURN(auto old,base.member(page_id));HYLE_ASSIGN_OR_RETURN(auto old_bytes,read<std::uint8_t>(old));HYLE_RETURN_IF_ERROR(require(old_bytes==expected_base,"base readonly bytes changed"));
    HYLE_ASSIGN_OR_RETURN(auto old_metadata,base.member(metadata_id));HYLE_ASSIGN_OR_RETURN(auto old_visible,read<std::uint64_t>(old_metadata));HYLE_RETURN_IF_ERROR(require(old_visible.size()==1&&old_visible[0]==3,"base metadata changed"));
    HYLE_RETURN_IF_ERROR(selected(route,encoding,page,oracle.decoded,next_visible));
    if(depth==512)HYLE_RETURN_IF_ERROR(sink(route,page,oracle.decoded,next_visible,accepted));
    HYLE_ASSIGN_OR_RETURN(auto failed,store.begin(published));auto invalid=std::vector<std::uint16_t>(depth,0x7fc0);auto rejected=encode(route,encoding,invalid);HYLE_RETURN_IF_ERROR(require(!rejected&&rejected.error().IsInvalidArgument(),"nonfinite append producer accepted"));HYLE_RETURN_IF_ERROR(failed.cancel());HYLE_RETURN_IF_ERROR(require(!failed.finalize(),"failed draft finalized"));
    HYLE_ASSIGN_OR_RETURN(auto unchanged,head.current());HYLE_RETURN_IF_ERROR(require(unchanged.identity()==published.identity(),"failed draft changed head"));
    HYLE_ASSIGN_OR_RETURN(auto report,clones.report());HYLE_RETURN_IF_ERROR(require(report.logical_bytes==bytes,"clone did not copy exactly fixed chunk"));
    std::printf("State depth=%u visible3->%llu fixed5-row COW=%zuB append=%uB prefix/base/metadata/failure PASS\n",depth,static_cast<unsigned long long>(next_visible),bytes,accepted*width);
  }
  HYLE_RETURN_IF_ERROR(store.drain());return Status::Ok();
}
Status run(){
#if defined(HYLE_ROW_CACHE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto context,g::cuda::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());HYLE_ASSIGN_OR_RETURN(auto memory,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));const auto resource=memory.ref();
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,2u*1024u*1024u));Route route{context,queue,resource,pool,[&](const g::Graph &graph){return g::cuda::compile(graph,context,resource);}};
#else
  HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());const auto resource=m::system_resource();
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(resource,2u*1024u*1024u));Route route{context,queue,resource,pool,[&](const g::Graph &graph){return g::compile(graph,context);}};
#endif
  for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()})HYLE_RETURN_IF_ERROR(exercise(route,encoding));return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"row cache StateStore consumer: %s\n",text);return 1;}
