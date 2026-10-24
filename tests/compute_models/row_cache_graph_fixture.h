#pragma once
#include "row_cache_paged_fixture.h"
#include "hyle/graph/batch.h"
#include "hyle/compute/model_ops.h"
#include <cmath>
#include <numeric>
namespace row_cache_graph_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;namespace f=row_cache_paged_test;
inline c::TensorDesc dense(c::DataType type,std::initializer_list<std::int64_t>shape){return c::TensorDesc::dense(type,{shape.begin(),shape.size()}).value();}
template<class T>std::vector<T> read(const g::GraphOutput&result){auto tensor=result.tensor().value();std::vector<T> values(*tensor.description().element_count());
  EXPECT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(values))).ok());return values;}
inline row_cache_test::Reference data(int phase){const auto encoding=c::RowCacheEncoding::float8_block64_tail64();row_cache_test::Reference result;
  for(int part:{0,1}){std::vector<std::uint16_t> values(5*512);
    for(std::size_t i=0;i<values.size();++i)values[i]=row_cache_test::bfloat(double(int((i*7+phase*11+part*13)%33)-16)/8);
    const auto next=row_cache_test::reference(encoding,values);result.bytes.insert(result.bytes.end(),next.bytes.begin(),next.bytes.end());result.decoded.insert(result.decoded.end(),next.decoded.begin(),next.decoded.end());}return result;}
inline std::vector<std::int32_t> topk(std::span<const float>scores){std::vector<std::int32_t> result;
  for(std::size_t row=0;row<4;++row){std::array<std::int32_t,5> order{0,1,2,3,4};std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[row*5+a]>scores[row*5+b];});result.insert(result.end(),order.begin(),order.begin()+2);}return result;}
inline std::vector<std::uint16_t> selected(const row_cache_test::Reference&source,std::span<const std::int32_t>ids,int phase){std::vector<std::uint16_t> result;
  for(std::size_t i=0;i<ids.size();++i){const auto id=ids[i];const auto page=i/2>=2?(phase?2:0):(id<2?(phase?1:4):(phase?4:1));
    const auto first=source.decoded.begin()+(page*2+id%2)*512;result.insert(result.end(),first,first+512);}return result;}
inline std::vector<double> sink(std::span<const float>query,std::span<const std::uint16_t>kv,std::span<const float>mask,std::span<const float>sinks){
  std::vector<double> result(query.size());const double scale=.0625;
  for(std::size_t b=0;b<4;++b)for(std::size_t h=0;h<64;++h){std::array<double,2> scores{};double maximum=sinks[h];
    for(std::size_t s=0;s<2;++s){double dot=0;for(std::size_t d=0;d<512;++d)dot+=double(query[(b*64+h)*512+d])*row_cache_test::decode(kv[(b*2+s)*512+d]);scores[s]=dot*scale+mask[b*2+s];maximum=std::max(maximum,scores[s]);}
    const double a=std::exp(scores[0]-maximum),z=std::exp(scores[1]-maximum),den=std::exp(double(sinks[h])-maximum)+a+z;
    for(std::size_t d=0;d<512;++d)result[(b*64+h)*512+d]=(a*row_cache_test::decode(kv[b*2*512+d])+z*row_cache_test::decode(kv[(b*2+1)*512+d]))/den;
  }return result;
}
template<class Compile>void exercise(c::Context&context,m::MemoryResourceRef resource,Compile compile){
  const auto encoding=c::RowCacheEncoding::float8_block64_tail64();auto schema=c::RowCacheBatchSchema::create(f::capacity(encoding)).value();
  auto builder=g::GraphBuilder::create().value();const auto scores_desc=dense(c::DataType::kFloat32,{4,5}),qdesc=dense(c::DataType::kFloat32,{4,1,64,512}),mdesc=dense(c::DataType::kFloat32,{4,1,2}),sdesc=dense(c::DataType::kFloat32,{64});
  const auto scores=builder.input(scores_desc).value(),q=builder.input(qdesc).value(),mask=builder.input(mdesc).value(),sk=builder.input(sdesc).value();
  const auto top=f::take(builder.add_node(c::Operation(c::TopK{.count=2}),std::array{scores}),"TopK node");const auto indices=builder.result(top,1).value();
  const auto macro=f::take(g::batch::add_row_cache_gather(builder,schema,indices),"row Graph macro");
  const auto floats=f::take(builder.add(c::Cast{c::DataType::kFloat32},macro.output()),"selected Cast");const auto shape=c::TensorShape::create(std::array<std::int64_t,4>{4,2,1,512}).value();
  const auto kv=f::take(builder.add(c::View::reshape(shape).value(),floats),"selected View");const auto attended=f::take(builder.add_node(c::model_ops::sink_attention(.0625F),std::array{q,kv,kv,mask,sk}),"selected Sink");
  const auto output=builder.result(attended).value(),negative=builder.add(c::Unary{c::UnaryKind::kNeg},output).value();
  for(auto value:{indices,macro.output(),output,negative})ASSERT_TRUE(builder.publish(value).ok());
  const auto logical=std::move(builder).finish().value();auto compiled=f::take(compile(logical),"row Graph compile");auto queue=context.create_queue().value();
  const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  auto slot=f::take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),"row Graph create");
  auto pool=c::LeasePool::create(resource,4u<<20).value();std::vector<float> old;
  for(int phase:{0,1}){const auto reference=data(phase);auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=10*encoding.row_bytes()+1}).value();auto ready=f::pages(arena,encoding,reference);auto version=f::layout(schema,phase);
    ASSERT_TRUE(g::batch::prepare(slot, logical, lanes, {}, std::array{g::batch::RowCacheBinding{macro,version}}).ok());
    auto owned=f::frame(arena,version,ready,phase);EXPECT_TRUE(g::batch::bind(slot, macro.bank_input(), f::frame(arena,f::layout(schema,phase),ready,phase)).IsInvalidArgument());
    ASSERT_TRUE(g::batch::bind(slot, macro.bank_input(), owned).ok());
    std::vector<float> score_values(20,-1000.F),query(4*64*512),mask_values(8),sink_values(64);
    for(std::size_t row=0;row<4;++row){score_values[row*5]=float(row+phase);score_values[row*5+1]=float(5-row-phase);if(row<2)score_values[row*5+2]=float(7+phase-row);mask_values[row*2]=-.125F*float(row+phase);mask_values[row*2+1]=-.25F;}
    for(std::size_t i=0;i<query.size();++i)query[i]=float(int((i*13+phase*7)%43)-21)/64;
    for(std::size_t h=0;h<64;++h)sink_values[h]=float(int((h*7+phase*3)%17)-8)/8;
    const std::array descriptions{scores_desc,qdesc,mdesc,sdesc};const std::array ports{scores,q,mask,sk};const std::array payloads{std::span(score_values),std::span(query),std::span(mask_values),std::span(sink_values)};
    for(std::size_t i=0;i<ports.size();++i)ASSERT_TRUE(slot.bind(ports[i], descriptions[i], f::load(pool,std::as_bytes(payloads[i]))).ok());
    for(auto value:{indices,macro.output(),output,negative}){const auto desc=logical.value_description(value).value();auto writer=pool.allocate(*desc.required_bytes()).value();ASSERT_TRUE(slot.bind(value, desc, std::move(writer)).ok());}
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    auto actual_ids=slot.result(indices).value(),actual_selected=slot.result(macro.output()).value(),actual_output=slot.result(output).value(),actual_neg=slot.result(negative).value();
    const auto expected_ids=topk(score_values);const auto expected_kv=selected(reference,expected_ids,phase);EXPECT_EQ(read<std::int32_t>(actual_ids),expected_ids);EXPECT_EQ(read<std::uint16_t>(actual_selected),expected_kv);
    const auto expected=sink(query,expected_kv,mask_values,sink_values);const auto actual=read<float>(actual_output),neg=read<float>(actual_neg);
    for(std::size_t i=0;i<actual.size();++i){ASSERT_LE(std::abs(double(actual[i])-expected[i]),2e-5+2e-5*std::abs(expected[i]))<<i;ASSERT_EQ(neg[i],-actual[i]);}
    if(phase)EXPECT_NE(actual,old);else old=actual;
    EXPECT_TRUE(g::batch::prepare(slot, logical, lanes, {}, std::array{g::batch::RowCacheBinding{macro,version}}).IsUnavailable());
  }
  ASSERT_TRUE(slot.drain().ok());
}
template<class Compile>void empty_and_private(c::Context&context,m::MemoryResourceRef resource,Compile compile){
  const auto encoding=c::RowCacheEncoding::float4_block32();auto schema=c::RowCacheBatchSchema::create(f::capacity(encoding)).value();auto queue=context.create_queue().value();
  const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  for(int mode:{0,1,2,3}){SCOPED_TRACE(mode);auto builder=g::GraphBuilder::create().value();
    const auto ids_desc=dense(c::DataType::kInt32,{4,mode?2:0});const auto ids=builder.input(ids_desc).value();auto macro=g::batch::add_row_cache_gather(builder,schema,ids).value();
    ASSERT_TRUE(builder.publish(macro.output()).ok());
    if(mode==1)ASSERT_TRUE(builder.publish(macro.bank_input()).ok());
    if(mode==2){const auto shape=c::TensorShape::create(std::array<std::int64_t,2>{10,encoding.row_bytes()}).value();
      const auto alias=builder.add(c::View::reshape(shape).value(),macro.bank_input()).value();ASSERT_TRUE(builder.publish(alias).ok());}
    const auto graph=std::move(builder).finish().value();auto compiled=f::take(compile(graph),"empty/private row Graph compile");
    auto slot=compiled.executable.create_invocation({.resources=g::ResourceOptions{}}).value();
    const std::array<std::int32_t,4> q{0,2,2,4},kv{};const std::array<std::int32_t,3> tails{};auto version=schema.version({q,kv,{},tails}).value();
    const auto prepared=g::batch::prepare(slot,graph,lanes,{},std::array{g::batch::RowCacheBinding{macro,version}});
    if(mode==1 || mode==2){EXPECT_TRUE(prepared.IsInvalidArgument());continue;}
    if(mode==3){EXPECT_TRUE(prepared.IsOutOfRange());continue;}
    ASSERT_TRUE(prepared.ok())<<prepared.message();
    auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=10*encoding.row_bytes()+1}).value();ASSERT_TRUE(g::batch::bind(slot, macro.bank_input(), arena.read(version,{}).value()).ok());
    auto pool=c::LeasePool::create(resource,1024).value();ASSERT_TRUE(slot.bind(ids, ids_desc, f::load(pool,{})).ok());
    const auto desc=graph.value_description(macro.output()).value();auto writer=pool.allocate(4).value();ASSERT_TRUE(slot.bind(macro.output(), desc, std::move(writer)).ok());
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());EXPECT_TRUE(read<std::uint16_t>(slot.result(macro.output()).value()).empty());
  }
}
} // namespace row_cache_graph_test
