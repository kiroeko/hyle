#pragma once
#include "row_cache_graph_fixture.h"
#include "row_cache_indexer_fixture.h"
namespace row_cache_paged_indexer_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;namespace gb=g::batch;
namespace f=row_cache_paged_test;using row_cache_graph_test::dense;using row_cache_graph_test::read;
inline c::RowCacheBatchCapacity capacity(c::RowCacheEncoding e){return {.encoding=e,.max_requests=2,.max_query_rows=3,.max_kv_rows=8,.max_pages=4,.max_page_refs=4,.page_size=2};}
inline c::RowCacheLayoutVersion layout(const c::RowCacheBatchSchema&s,bool empty){
  const std::array<std::int32_t,3> q{0,2,3},kv=empty?std::array<std::int32_t,3>{0,0,0}:std::array<std::int32_t,3>{0,2,3};
  const std::array<std::int32_t,3> pages{3,1,0};const std::array<std::int32_t,2> tails=empty?std::array<std::int32_t,2>{0,0}:std::array<std::int32_t,2>{1,1};
  return s.version({q,kv,empty?std::span<const std::int32_t>{}:std::span(pages),tails}).value();
}
inline std::int32_t physical(std::size_t q,std::int32_t logical){return q<2?(logical<2?6+logical:2):logical;}
inline std::vector<c::ReadLease> upload(c::PagedArena&a,std::span<const std::uint8_t>bytes,std::size_t width){
  std::vector<c::ReadLease> pages;for(unsigned p=0;p<4;++p){auto writer=a.acquire(p).value();
    EXPECT_TRUE(writer.lease().view()->write(std::as_bytes(bytes.subspan(p*2*width,2*width))).ok());pages.push_back(writer.publish(2).value());}return pages;
}
inline c::PagedRowCacheRead frame(c::PagedArena&a,c::RowCacheLayoutVersion v,const std::vector<c::ReadLease>&p,bool empty){
  if(empty)return a.read(v,{}).value();const std::array exact{p[1],p[0],p[3]};return a.read(v,exact).value();
}
template<class Compile>void exercise(c::Context&context,m::MemoryResourceRef resource,Compile compile){
  auto scorer_schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  auto value_schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float8_block64_tail64())).value();
  row_cache_indexer_test::Fixture values(8,3);values.ranges={0,8,0,8,0,8};const auto dense_scores=values.expected();
  std::vector<std::uint16_t> attention_source(10*512);for(std::size_t i=0;i<attention_source.size();++i)attention_source[i]=row_cache_test::bfloat(double(int((i*7)%31)-15)/8);
  auto attention=row_cache_test::reference(value_schema.capacity().encoding,std::span(attention_source).first(5*512));
  const auto tail=row_cache_test::reference(value_schema.capacity().encoding,std::span(attention_source).subspan(5*512));
  attention.bytes.insert(attention.bytes.end(),tail.bytes.begin(),tail.bytes.end());attention.decoded.insert(attention.decoded.end(),tail.decoded.begin(),tail.decoded.end());
  attention.bytes.resize(8*584);attention.decoded.resize(8*512);
  auto builder=g::GraphBuilder::create().value();
  const auto q=builder.input(values.descriptions[0]).value(),w=builder.input(values.descriptions[2]).value();
  const auto rd=dense(c::DataType::kInt32,{3,2}),rawid_desc=dense(c::DataType::kInt32,{3,3});
  const auto ranges=builder.input(rd).value(),raw_ranges=builder.input(rd).value(),raw_ids=builder.input(rawid_desc).value();
  const auto sqdesc=dense(c::DataType::kFloat32,{3,1,64,512}),sdesc=dense(c::DataType::kFloat32,{64});
  const auto sq=builder.input(sqdesc).value(),sink=builder.input(sdesc).value();
  const auto scorer=f::take(gb::add_row_cache_indexer(builder,scorer_schema,q,w,ranges,4),"paged indexer macro");
  const auto top=builder.add_node(c::Operation(c::TopK{.count=3}),std::array{scorer.output()}).value();const auto ids=builder.result(top,1).value();
  const auto selected=f::take(gb::add_row_cache_gather_padded(builder,value_schema,ids,ranges,4),"compressed padding macro");
  const auto raw=f::take(gb::add_row_cache_gather_padded(builder,value_schema,raw_ids,raw_ranges,4),"raw window padding macro");
  const auto joined=builder.add_node(c::Operation(c::Concat{.axis=1}),std::array{raw.output(),selected.output()}).value();
  const auto floats=builder.add(c::Cast{c::DataType::kFloat32},builder.result(joined).value()).value();
  const auto key_view=builder.add(c::View::reshape(c::TensorShape::create(std::array<std::int64_t,4>{3,6,1,512}).value()).value(),floats).value();
  const auto both_masks=builder.add_node(c::Operation(c::Concat{.axis=1}),std::array{raw.mask(),selected.mask()}).value();
  const auto mask=builder.add(c::View::reshape(c::TensorShape::create(std::array<std::int64_t,3>{3,1,6}).value()).value(),builder.result(both_masks).value()).value();
  const auto attn=builder.add_node(c::model_ops::sink_attention(.0625F),std::array{sq,key_view,key_view,mask,sink}).value();
  const auto out=builder.result(attn).value(),neg=builder.add(c::Unary{c::UnaryKind::kNeg},out).value();
  const std::array published{scorer.output(),ids,raw.output(),selected.output(),raw.mask(),selected.mask(),out,neg};
  for(auto v:published)ASSERT_TRUE(builder.publish(v).ok());
  const auto graph=std::move(builder).finish().value();auto compiled=f::take(compile(graph),"paged score/TopK/padded/raw/Sink compile");
  auto queue=context.create_queue().value();const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  auto slot=f::take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),"typed scorer/gather coordinator");
  auto score_arena=c::PagedArena::create(scorer_schema,resource,{.max_backing_bytes=8*68+1}).value();
  auto value_arena=c::PagedArena::create(value_schema,resource,{.max_backing_bytes=8*584+1}).value();
  const auto score_pages=upload(score_arena,values.bank,68),value_pages=upload(value_arena,attention.bytes,584);
  auto pool=c::LeasePool::create(resource,4u<<20).value();
  for(bool empty:{false,true}){SCOPED_TRACE(empty);const auto sv=layout(scorer_schema,empty),vv=layout(value_schema,empty);
    const std::array layouts{gb::RowCacheBinding{scorer.reader(),sv},gb::RowCacheBinding{selected.reader(),vv},gb::RowCacheBinding{raw.reader(),vv}};
    ASSERT_TRUE(gb::prepare(slot, graph, lanes, {}, layouts).ok());
    ASSERT_TRUE(gb::bind(slot, scorer.bank_input(), frame(score_arena,sv,score_pages,empty)).ok());
    ASSERT_TRUE(gb::bind(slot, selected.bank_input(), frame(value_arena,vv,value_pages,empty)).ok());
    ASSERT_TRUE(gb::bind(slot, raw.bank_input(), frame(value_arena,vv,value_pages,empty)).ok());
    // First compressed prefill: query0 has no compressed history, query1/2
    // have one visible row, yet TopK count is 3. Raw windows also include
    // future entries and an explicit -1; neither may read an unleased row0.
    const std::vector<std::int32_t> r=empty?std::vector<std::int32_t>(6):std::vector<std::int32_t>{0,0,0,1,0,1};
    const std::vector<std::int32_t> rr=empty?std::vector<std::int32_t>(6):std::vector<std::int32_t>{0,1,0,2,0,1};
    const std::vector<std::int32_t> ri{-1,0,1,0,1,2,0,1,2};
    ASSERT_TRUE(slot.bind(q, values.descriptions[0], f::load(pool,std::as_bytes(std::span(values.query)))).ok());
    ASSERT_TRUE(slot.bind(w, values.descriptions[2], f::load(pool,std::as_bytes(std::span(values.weights)))).ok());
    ASSERT_TRUE(slot.bind(ranges, rd, f::load(pool,std::as_bytes(std::span(r)))).ok());
    ASSERT_TRUE(slot.bind(raw_ranges, rd, f::load(pool,std::as_bytes(std::span(rr)))).ok());
    ASSERT_TRUE(slot.bind(raw_ids, rawid_desc, f::load(pool,std::as_bytes(std::span(ri)))).ok());
    const std::vector<float> query(3*64*512,0.F),sinks(64,.25F);
    ASSERT_TRUE(slot.bind(sq, sqdesc, f::load(pool,std::as_bytes(std::span(query)))).ok());
    ASSERT_TRUE(slot.bind(sink, sdesc, f::load(pool,std::as_bytes(std::span(sinks)))).ok());
    for(auto v:published){const auto desc=graph.value_description(v).value();auto writer=pool.allocate((std::max)(std::size_t{4},*desc.required_bytes())).value();ASSERT_TRUE(slot.bind(v, desc, std::move(writer)).ok());}
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    const auto actual_scores=read<float>(slot.result(scorer.output()).value());
    for(std::size_t row=0;row<3;++row)for(int key=0;key<4;++key){const auto expected=key>=r[row*2]&&key<r[row*2+1]?dense_scores[row*8+physical(row,key)]:-std::numeric_limits<float>::infinity();EXPECT_FLOAT_EQ(actual_scores[row*4+key],expected);}
    const auto selected_ids=read<std::int32_t>(slot.result(ids).value());
    const auto raw_values=read<std::uint16_t>(slot.result(raw.output()).value()),selected_values=read<std::uint16_t>(slot.result(selected.output()).value());
    const auto rm=read<float>(slot.result(raw.mask()).value()),sm=read<float>(slot.result(selected.mask()).value());
    std::vector<double> expected(3*64*512);
    for(std::size_t row=0;row<3;++row){std::size_t valid_rows=0;
      for(int kind=0;kind<2;++kind)for(std::size_t k=0;k<3;++k){const auto id=kind?selected_ids[row*3+k]:ri[row*3+k];const auto begin=kind?r[row*2]:rr[row*2],end=kind?r[row*2+1]:rr[row*2+1];const bool valid=id>=begin&&id<end;
        ASSERT_GE(id,-1);ASSERT_LT(id,4);EXPECT_EQ((kind?sm:rm)[row*3+k],valid?0.F:-std::numeric_limits<float>::infinity());
        if(valid)++valid_rows;
        for(std::size_t d=0;d<512;++d){const auto wanted=valid?attention.decoded[physical(row,id)*512+d]:std::uint16_t{0};ASSERT_EQ((kind?selected_values:raw_values)[(row*3+k)*512+d],wanted)<<"row="<<row<<" kind="<<kind<<" k="<<k<<" d="<<d;
          if(valid)for(std::size_t h=0;h<64;++h)expected[(row*64+h)*512+d]+=row_cache_test::decode(wanted);}}
      const auto denominator=std::exp(.25)+double(valid_rows);for(std::size_t h=0;h<64;++h)for(std::size_t d=0;d<512;++d)expected[(row*64+h)*512+d]/=denominator;
    }
    const auto actual=read<float>(slot.result(out).value()),negative=read<float>(slot.result(neg).value());
    for(std::size_t i=0;i<actual.size();++i){ASSERT_LE(std::abs(double(actual[i])-expected[i]),2e-5+2e-5*std::abs(expected[i]))<<i;ASSERT_EQ(negative[i],-actual[i]);}
    {auto retained=slot.result(out).value();EXPECT_TRUE(gb::prepare(slot, graph, lanes, {}, layouts).IsUnavailable());EXPECT_EQ(read<float>(retained),actual);}
  }
  ASSERT_TRUE(slot.drain().ok());
}

template<class Compile>void rejected_private_readers(c::Context&context,Compile compile){
  const auto schema=c::RowCacheBatchSchema::create(capacity(c::RowCacheEncoding::float4_block32())).value();
  auto queue=context.create_queue().value();const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  for(bool alias:{false,true}){auto b=g::GraphBuilder::create().value();
    const auto q=b.input(dense(c::DataType::kBFloat16,{3,64,128})).value(),w=b.input(dense(c::DataType::kBFloat16,{3,64})).value(),r=b.input(dense(c::DataType::kInt32,{3,2})).value();
    const auto scorer=gb::add_row_cache_indexer(b,schema,q,w,r,4).value();ASSERT_TRUE(b.publish(scorer.output()).ok());
    auto leaked=scorer.bank_input();if(alias)leaked=b.add(c::View::reshape(c::TensorShape::create(std::array<std::int64_t,2>{8,68}).value()).value(),leaked).value();
    ASSERT_TRUE(b.publish(leaked).ok());const auto graph=std::move(b).finish().value();auto compiled=f::take(compile(graph),"private-capacity negative compile");
    auto slot=compiled.executable.create_invocation({.resources=g::ResourceOptions{}}).value();
    const auto denied=gb::prepare(slot,graph,lanes,{},std::array{gb::RowCacheBinding{scorer.reader(),layout(schema,false)}});EXPECT_TRUE(denied.IsInvalidArgument());
  }
}
template<class Compile>void invalid_padded(c::Context&context,m::MemoryResourceRef resource,Compile compile){
  const auto encoding=c::RowCacheEncoding::float8_block64_tail64();const auto schema=c::RowCacheBatchSchema::create(capacity(encoding)).value();
  for(int failure:{0,1,2,3}){SCOPED_TRACE(failure);auto builder=g::GraphBuilder::create().value();const auto id_desc=dense(c::DataType::kInt32,{3,3}),range_desc=dense(c::DataType::kInt32,{3,2});
    const auto ids=builder.input(id_desc).value(),ranges=builder.input(range_desc).value();const auto macro=gb::add_row_cache_gather_padded(builder,schema,ids,ranges,4).value();
    ASSERT_TRUE(builder.publish(macro.output()).ok());ASSERT_TRUE(builder.publish(macro.mask()).ok());const auto graph=std::move(builder).finish().value();auto compiled=f::take(compile(graph),"invalid padded compile");
    auto queue=context.create_queue().value();const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
    auto slot=compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=8*584+1}).value();
    // No bank contents need reading: all other IDs are padding. The true
    // page frame still carries exactly the visible layout's required pages.
    std::vector<std::uint8_t> bank(8*584);auto pages=upload(arena,bank,584);auto version=layout(schema,false);
    ASSERT_TRUE(gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{macro.reader(),version}}).ok());ASSERT_TRUE(gb::bind(slot, macro.bank_input(), frame(arena,version,pages,false)).ok());
    auto pool=c::LeasePool::create(resource,32768).value();std::vector<std::int32_t> selected(9,-1),r{0,0,0,1,0,1};
    if(failure<2)selected.back()=failure?-2:4;else{r[4]=failure==2?-1:0;r[5]=failure==2?1:2;}
    ASSERT_TRUE(slot.bind(ids, id_desc, f::load(pool,std::as_bytes(std::span(selected)))).ok());ASSERT_TRUE(slot.bind(ranges, range_desc, f::load(pool,std::as_bytes(std::span(r)))).ok());
    const auto desc=graph.value_description(macro.output()).value();auto writer=pool.allocate(*desc.required_bytes()).value();const auto raw=writer.view().value();const std::vector<std::uint16_t> sentinel(9*512,0x5a5a);
    ASSERT_TRUE(raw.write(std::as_bytes(std::span(sentinel))).ok());ASSERT_TRUE(slot.bind(macro.output(), desc, std::move(writer)).ok());
    const auto md=graph.value_description(macro.mask()).value();auto mw=pool.allocate(*md.required_bytes()).value();ASSERT_TRUE(slot.bind(macro.mask(), md, std::move(mw)).ok());
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done);EXPECT_TRUE(done->wait().IsOutOfRange());EXPECT_FALSE(slot.result(macro.output()));
    std::vector<std::uint16_t> actual(sentinel.size());ASSERT_TRUE(raw.read(std::as_writable_bytes(std::span(actual))).ok());EXPECT_EQ(actual,sentinel);
    // The Map's mask is an accepted prefix write; Graph failure does not
    // promise to roll back arbitrary earlier caller buffers.
  }
}

template<class Compile>void crosses_tile(c::Context&context,m::MemoryResourceRef resource,Compile compile){
  const auto encoding=c::RowCacheEncoding::float4_block32();const auto schema=c::RowCacheBatchSchema::create({.encoding=encoding,.max_requests=2,.max_query_rows=3,.max_kv_rows=274,.max_pages=4,.max_page_refs=4,.page_size=128}).value();
  const std::array<std::int32_t,3> ki{0,3,4};const std::array<std::int32_t,4> pi{3,1,2,0};const std::array<std::int32_t,2> tail{1,17};
  row_cache_indexer_test::Fixture fdata(512,3);fdata.ranges={0,512,0,512,0,512};const auto reference=fdata.expected();
  auto builder=g::GraphBuilder::create().value();const auto q=builder.input(fdata.descriptions[0]).value(),w=builder.input(fdata.descriptions[2]).value(),r=builder.input(fdata.descriptions[3]).value();
  const auto scorer=gb::add_row_cache_indexer(builder,schema,q,w,r,257).value();ASSERT_TRUE(builder.publish(scorer.output()).ok());const auto graph=std::move(builder).finish().value();auto compiled=f::take(compile(graph),"cross-tile paged scorer compile");
  auto queue=context.create_queue().value();const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  std::array slots{compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
    compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
  auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=512*68+1}).value();std::vector<c::ReadLease> pages;
  for(std::uint32_t page=0;page<4;++page){auto writer=arena.acquire(page).value();ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(fdata.bank).subspan(page*128*68,128*68))).ok());pages.push_back(writer.publish(128).value());}
  auto pool=c::LeasePool::create(resource,131072).value();const auto desc=graph.value_description(scorer.output()).value();
  g::GraphOutput old_result;std::vector<float> old_values;
  for(int phase:{0,1}){SCOPED_TRACE(phase);auto&slot=slots[phase];
    const std::array<std::int32_t,3> qi=phase?std::array<std::int32_t,3>{0,1,3}:std::array<std::int32_t,3>{0,2,3};
    const auto version=schema.version({qi,ki,pi,tail}).value();
    const std::array<std::int32_t,6> ranges=phase?std::array<std::int32_t,6>{0,257,0,17,1,17}:std::array<std::int32_t,6>{0,257,128,257,1,17};
    ASSERT_TRUE(gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{scorer.reader(),version}}).ok());ASSERT_TRUE(gb::bind(slot, scorer.bank_input(), arena.read(version,pages).value()).ok());
    ASSERT_TRUE(slot.bind(q, fdata.descriptions[0], f::load(pool,std::as_bytes(std::span(fdata.query)))).ok());ASSERT_TRUE(slot.bind(w, fdata.descriptions[2], f::load(pool,std::as_bytes(std::span(fdata.weights)))).ok());ASSERT_TRUE(slot.bind(r, fdata.descriptions[3], f::load(pool,std::as_bytes(std::span(ranges)))).ok());
    auto writer=pool.allocate(*desc.required_bytes()).value();ASSERT_TRUE(slot.bind(scorer.output(), desc, std::move(writer)).ok());
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok());const auto actual=read<float>(slot.result(scorer.output()).value());
    for(std::size_t query=0;query<3;++query)for(int key=0;key<257;++key){
      const auto request=query<std::size_t(qi[1])?0u:1u;const bool visible=key>=ranges[query*2]&&key<ranges[query*2+1];
      const auto physical_row=visible?pi[std::size_t(ki[request])+std::size_t(key)/128]*128+key%128:0;
      const auto expected=visible?reference[query*512+physical_row]:-std::numeric_limits<float>::infinity();
      EXPECT_FLOAT_EQ(actual[query*257+key],expected)<<query<<':'<<key;
    }
    if(!phase){old_result=slot.result(scorer.output()).value();old_values=actual;
      EXPECT_TRUE(gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{scorer.reader(),version}}).IsUnavailable());
    }else{EXPECT_EQ(read<float>(old_result),old_values);EXPECT_NE(actual,old_values);}
    // The two slots have different grouping; the second also repeats cold
    // preparation before a final-query failure. The first output stays ready.
    if(phase){*done=c::Completion{};const std::array<std::int32_t,6> invalid{0,257,0,17,1,18};
      ASSERT_TRUE(gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{scorer.reader(),version}}).ok());ASSERT_TRUE(gb::bind(slot, scorer.bank_input(), arena.read(version,pages).value()).ok());
      ASSERT_TRUE(slot.bind(q, fdata.descriptions[0], f::load(pool,std::as_bytes(std::span(fdata.query)))).ok());ASSERT_TRUE(slot.bind(w, fdata.descriptions[2], f::load(pool,std::as_bytes(std::span(fdata.weights)))).ok());ASSERT_TRUE(slot.bind(r, fdata.descriptions[3], f::load(pool,std::as_bytes(std::span(invalid)))).ok());
      auto bad_writer=pool.allocate(*desc.required_bytes()).value();const auto raw=bad_writer.view().value();const std::vector<float> sentinel(3*257,91.F);ASSERT_TRUE(raw.write(std::as_bytes(std::span(sentinel))).ok());ASSERT_TRUE(slot.bind(scorer.output(), desc, std::move(bad_writer)).ok());
      ASSERT_TRUE(slot.prepare_replay(lanes).ok());
      auto failed=slot.submit(lanes);ASSERT_TRUE(failed)<<failed.error().message();EXPECT_TRUE(failed->wait().IsOutOfRange());EXPECT_FALSE(slot.result(scorer.output()));
      std::vector<float> preserved(sentinel.size());ASSERT_TRUE(raw.read(std::as_writable_bytes(std::span(preserved))).ok());EXPECT_EQ(preserved,sentinel);EXPECT_EQ(read<float>(old_result),old_values);
    }
  }
}
} // namespace row_cache_paged_indexer_test
