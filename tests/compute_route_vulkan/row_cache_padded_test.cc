#include "../compute_models/row_cache_native_fixture.h"
#include "hyle/compute/model_ops.h"
#include <cstring>
#include <optional>
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
#include "hyle/compute/apple_row_cache.h"
#include "hyle/graph/apple.h"
#include <mlx/ops.h>
namespace n=hyle::compute::apple;
namespace ng=hyle::graph::apple;
#else
#include "hyle/compute/vulkan_row_cache.h"
#include "hyle/graph/vulkan.h"
namespace n=hyle::compute::vulkan;
namespace ng=hyle::graph::vulkan;
#endif
namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace f=row_cache_native_test;
using f::take;
c::TensorDesc dense(c::DataType type,std::initializer_list<std::int64_t> shape){return c::TensorDesc::dense(type,std::span(shape.begin(),shape.size())).value();}
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
namespace mx=mlx::core;using Input=mx::array;
#else
using Input=n::Tensor;
#endif
Input upload(const n::Context&context,c::TensorDesc desc,std::span<const std::byte>data,int prefix=0){
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
  (void)context;const auto count=desc.element_count().value();mx::array base(std::uint8_t{0});
  if(desc.data_type()==c::DataType::kUInt8){std::vector<std::uint8_t>v(count+prefix);if(!data.empty())std::memcpy(v.data()+prefix,data.data(),data.size());base=mx::array(v.data(),{int(v.size())},mx::uint8);}
  else if(desc.data_type()==c::DataType::kInt32){std::vector<std::int32_t>v(count+prefix);if(!data.empty())std::memcpy(v.data()+prefix,data.data(),data.size());base=mx::array(v.data(),{int(v.size())},mx::int32);}
  else if(desc.data_type()==c::DataType::kFloat32){std::vector<float>v(count+prefix);if(!data.empty())std::memcpy(v.data()+prefix,data.data(),data.size());base=mx::array(v.data(),{int(v.size())},mx::float32);}
  else{std::vector<std::uint16_t>v(count+prefix);if(!data.empty())std::memcpy(v.data()+prefix,data.data(),data.size());base=mx::view(mx::array(v.data(),{int(v.size())},mx::uint16),mx::bfloat16);}
  mx::Shape shape;for(auto d:desc.dimensions())shape.push_back(int(d));auto result=mx::reshape(mx::slice(base,{prefix},{prefix+int(count)}),shape);result.eval();result.wait();return result;
#else
  if(data.empty())return context.upload(desc,data).value();const auto count=desc.element_count().value(),item=c::data_type_size(desc.data_type());
  std::vector<std::byte>bytes((count+prefix)*item);std::copy(data.begin(),data.end(),bytes.begin()+prefix*item);
  auto base=context.upload(dense(desc.data_type(),{std::int64_t(count+prefix)}),bytes).value();
  return base.view(c::View::slice(0,prefix,std::int64_t(count)).value()).value().view(c::View::reshape(c::TensorShape::create(desc.dimensions()).value()).value()).value();
#endif
}
template<class T>std::vector<T> read(const n::Context&context,const Input&input){
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
  (void)context;if(!input.size())return{};return {input.data<T>(),input.data<T>()+input.size()};
#else
  std::vector<T>result(input.description().element_count().value());EXPECT_TRUE(context.download(input,std::as_writable_bytes(std::span(result))).ok());return result;
#endif
}
std::span<const Input> outputs(const n::Results&r){
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
 return r.arrays();
#else
 return r.tensors();
#endif
}
n::Program direct(const n::Context&context,c::Operation operation,std::span<const c::TensorDesc>descs){
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
  std::vector<n::InputSpec>specs;for(const auto&d:descs)specs.push_back(n::input_spec(d).value());return take(context.prepare(operation,specs),"Apple Direct padded");
#else
  std::vector<n::Argument>specs;for(const auto&d:descs)specs.emplace_back(d);return take(n::prepare(context,operation,specs),"Vulkan Direct padded");
#endif
}
std::vector<std::uint16_t> selected(c::RowCacheEncoding encoding,const row_cache_test::Reference&source,std::span<const std::int32_t>ids){
  std::vector<std::uint16_t>result(ids.size()*encoding.depth());
  for(std::size_t r=0;r<ids.size();++r){if(ids[r]>=0){
    std::copy_n(source.decoded.begin()+std::size_t(ids[r])*encoding.depth(),encoding.depth(),result.begin()+r*encoding.depth());}}
  return result;
}
void ordinary(bool graph){auto context=n::Context::create().value();
  for(const auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}){
    SCOPED_TRACE(encoding.depth());const auto bank_desc=c::row_cache_carrier_description(encoding,5).value(),ids_desc=dense(c::DataType::kInt32,{2,3});
    const auto operation=c::model_ops::row_cache_gather_decode_padded(encoding);auto program=direct(context,operation,std::array{bank_desc,ids_desc});
    if(graph){auto builder=g::GraphBuilder::create().value();auto a=builder.input(bank_desc).value(),i=builder.input(ids_desc).value();
      auto decoded=builder.result(builder.add_node(operation,std::array{a,i}).value()).value();
      auto reshaped=builder.add(c::View::reshape(c::TensorShape::create(std::array<std::int64_t,2>{6,encoding.depth()}).value()).value(),decoded).value();
      ASSERT_TRUE(builder.publish(decoded).ok());ASSERT_TRUE(builder.publish(reshaped).ok());program=take(ng::compile(context,std::move(builder).finish().value()),"native padded Graph");}
    auto slots=std::array{program.create_invocation().value(),program.create_invocation().value()};std::optional<Input>old;std::vector<std::uint16_t>old_bits;
    const std::array<std::int32_t,6>ids{-1,4,0,2,-1,1};
    for(int phase:{0,1}){const auto source=row_cache_test::reference(encoding,row_cache_test::input_values(encoding,phase));
      auto bank=upload(context,bank_desc,std::as_bytes(std::span(source.bytes)),1),index=upload(context,ids_desc,std::as_bytes(std::span(ids)),phase+1);
      ASSERT_TRUE(slots[phase].bind(std::array{bank,index}).ok());auto done=take(slots[phase].submit(),"ordinary padded submit");auto result=take(done.wait(),"ordinary padded wait");
      const auto values=outputs(*result.get());const auto wanted=selected(encoding,source,ids);EXPECT_EQ(read<std::uint16_t>(context,values[0]),wanted);
      if(graph)EXPECT_EQ(read<std::uint16_t>(context,values[1]),wanted);
      if(!phase){old=values[0];old_bits=wanted;}else EXPECT_EQ(read<std::uint16_t>(context,*old),old_bits);
      for(std::int32_t bad:{-2,5}){auto corrupt=ids;corrupt.back()=bad;auto failed=program.create_invocation().value();
        ASSERT_TRUE(failed.bind(std::array{bank,upload(context,ids_desc,std::as_bytes(std::span(corrupt)))}).ok());
        auto work=take(failed.submit(),"bad padded submit");auto error=work.wait();ASSERT_FALSE(error);EXPECT_TRUE(error.error().IsOutOfRange());EXPECT_EQ(read<std::uint16_t>(context,*old),old_bits);}
    }
    const auto empty_desc=c::row_cache_carrier_description(encoding,0).value();auto empty=direct(context,operation,std::array{empty_desc,ids_desc}).create_invocation().value();
    const std::array<std::int32_t,6>padding{-1,-1,-1,-1,-1,-1};ASSERT_TRUE(empty.bind(std::array{upload(context,empty_desc,{}),upload(context,ids_desc,std::as_bytes(std::span(padding)))}).ok());
    auto done=take(empty.submit(),"empty physical padded submit");auto result=take(done.wait(),"empty physical padded wait");EXPECT_EQ(read<std::uint16_t>(context,outputs(*result.get())[0]),std::vector<std::uint16_t>(6*encoding.depth(),0));
  }
}
struct Route final{
  n::Context context=n::Context::create().value();
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
  n::MemoryResource device=n::MemoryResource::create(context).value(),host_resource=n::MemoryResource::create(context).value();
#else
  n::MemoryResource device=n::MemoryResource::create(context,n::MemoryKind::kDevice).value(),host_resource=n::MemoryResource::create(context,n::MemoryKind::kStaging).value();
#endif
  c::LeasePool host=c::LeasePool::create(host_resource.ref(),65536).value(),pool=c::LeasePool::create(device.ref(),65536).value();
  n::CopyBatchProgram copier=n::CopyBatchProgram::prepare(context,std::array{c::CopyResourceSpec{host_resource.ref(),65536},c::CopyResourceSpec{device.ref(),65536}}).value();
  n::CopyBatchInvocation copy=n::CopyBatchInvocation::create(copier).value();
  c::WriteLease write(c::WriteLease destination,std::span<const std::byte>bytes){auto hw=host.allocate(bytes.size()).value();EXPECT_TRUE(hw.view()->write(bytes).ok());auto source=hw.publish().value();
    EXPECT_TRUE(copy.reset().ok());EXPECT_TRUE(copy.bind_source({0},source).ok());EXPECT_TRUE(copy.bind_mutable({1},std::move(destination)).ok());
    const std::array<c::CopyPhase,1>phases{{{}}};const std::array<c::CopyRange,1>ranges{{{{0},{1},{0},0,0,bytes.size()}}};EXPECT_TRUE(copy.prepare_replay(phases,ranges).ok());
    {auto done=take(copy.submit(),"typed upload");EXPECT_TRUE(done.wait().ok());}auto result=copy.unbind_mutable({1}).value();EXPECT_TRUE(copy.reset().ok());return result;}
  c::ReadLease load(std::span<const std::byte>bytes){return write(pool.allocate(bytes.size()).value(),bytes).publish().value();}
};
n::Program sink_program(const n::Context&context){auto builder=g::GraphBuilder::create().value();
  const auto decoded=builder.input(dense(c::DataType::kBFloat16,{3,3,512})).value(),mask=builder.input(dense(c::DataType::kFloat32,{3,3})).value();
  const auto query=builder.input(dense(c::DataType::kFloat32,{3,1,64,512})).value(),sinks=builder.input(dense(c::DataType::kFloat32,{64})).value();
  const auto wide=builder.add(c::Cast{c::DataType::kFloat32},decoded).value();
  const auto reshape=[&](g::ValueId value,std::initializer_list<std::int64_t>dims){return builder.add(c::View::reshape(c::TensorShape::create(std::span(dims.begin(),dims.size())).value()).value(),value).value();};
  const auto kv=reshape(wide,{3,3,1,512}),bias=reshape(mask,{3,1,3});
  auto out=builder.result(builder.add_node(c::model_ops::sink_attention(1.F/16),std::array{query,kv,kv,bias,sinks}).value()).value();
  auto neg=builder.add(c::Unary{c::UnaryKind::kNeg},out).value();EXPECT_TRUE(builder.publish(out).ok());EXPECT_TRUE(builder.publish(neg).ok());
  return take(ng::compile(context,std::move(builder).finish().value()),"selected rows and actual mask to Sink Graph");
}
void check_sink(Route&r,n::Program&program,const std::span<const Input>args,std::span<const std::uint16_t>bits,std::span<const float>mask){
  const auto q=dense(c::DataType::kFloat32,{3,1,64,512}),s=dense(c::DataType::kFloat32,{64});std::vector<float>zero(3*64*512),sinks(64,.25F);
  auto slot=program.create_invocation().value();ASSERT_TRUE(slot.bind(std::array{args[0],args[1],upload(r.context,q,std::as_bytes(std::span(zero))),upload(r.context,s,std::as_bytes(std::span(sinks)))}).ok());
  auto done=take(slot.submit(),"first prefill Sink submit");auto result=take(done.wait(),"first prefill Sink wait");auto out=outputs(*result.get());
  for(unsigned port=0;port<2;++port){const auto actual=read<float>(r.context,out[port]);ASSERT_EQ(actual.size(),3u*64*512);
    for(unsigned t=0;t<3;++t){double denominator=std::exp(.25);for(unsigned k=0;k<3;++k)if(mask[t*3+k]==0)denominator+=1;
      for(unsigned d=0;d<512;++d){double expected=0;for(unsigned k=0;k<3;++k)if(mask[t*3+k]==0)expected+=row_cache_test::decode(bits[(t*3+k)*512+d]);expected/=denominator;if(port)expected=-expected;
        for(unsigned h=0;h<64;++h)ASSERT_NEAR(actual[(t*64+h)*512+d],expected,2e-5+2e-5*std::abs(expected))<<t<<'/'<<h<<'/'<<d;}}
  }
}
void typed(){Route r;
 for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}){SCOPED_TRACE(encoding.depth());
  const c::RowCacheBatchCapacity cap{.encoding=encoding,.max_requests=2,.max_query_rows=3,.max_kv_rows=8,.max_pages=4,.max_page_refs=4,.page_size=2};
  auto schema=c::RowCacheBatchSchema::create(cap).value();auto arena=c::PagedArena::create(schema,r.device.ref(),{.max_backing_bytes=8*encoding.row_bytes()+1}).value();
  // Keep the independent five-row codec oracle's actual row count, then take
  // eight physical rows from two separately encoded inputs.
  auto make=[&](int phase){std::vector<std::uint16_t>v(5*encoding.depth());for(unsigned row=0;row<5;++row)for(unsigned d=0;d<encoding.depth();++d)v[row*encoding.depth()+d]=row_cache_test::bfloat((int((row*7+d*3+phase)%17)-8)/8.);return row_cache_test::reference(encoding,v);};
  auto data=make(0),next=make(1);data.bytes.insert(data.bytes.end(),next.bytes.begin(),next.bytes.end());data.decoded.insert(data.decoded.end(),next.decoded.begin(),next.decoded.end());
  std::array<c::ReadLease,4>pages;for(unsigned p:{0u,1u,3u}){auto writer=arena.acquire(p).value();writer.lease()=r.write(std::move(writer.lease()),std::as_bytes(std::span(data.bytes).subspan(p*2*encoding.row_bytes(),2*encoding.row_bytes())));pages[p]=writer.publish(2).value();}
  const std::array<std::int32_t,3>qi{0,2,3},ki{0,2,3},pi{3,1,0};const std::array<std::int32_t,2>tail{1,1};auto version=schema.version({qi,ki,pi,tail}).value();
  const auto desc=dense(c::DataType::kInt32,{3,3});auto owner=take(n::PagedRowCacheGatherPaddedInvocation::create(r.context,r.device,r.host_resource,schema,desc,4),"typed padded reader");
  auto other=take(n::PagedRowCacheGatherPaddedInvocation::create(r.context,r.device,r.host_resource,schema,desc,4),"independent typed padded reader");
  auto sink=encoding.depth()==512?std::optional<n::Program>(sink_program(r.context)):std::nullopt;
  std::optional<Input>old;std::vector<std::uint16_t>old_bits;
  const std::array<std::int32_t,9>ids{-1,0,1,0,1,2,0,1,2};
  for(int phase:{0,1}){const std::array<std::int32_t,6>ranges=phase?std::array<std::int32_t,6>{0,1,0,2,0,1}:std::array<std::int32_t,6>{0,0,0,1,0,1};
    auto&slot=phase?other:owner;ASSERT_TRUE(slot.prepare_layout(version).ok());auto frame=arena.read(version,std::array{pages[3],pages[1],pages[0]}).value();
    auto done=take(slot.submit(frame,r.load(std::as_bytes(std::span(ids))),r.load(std::as_bytes(std::span(ranges)))),"typed first prefill submit");auto result=take(done.wait(),"typed first prefill wait");auto out=outputs(*result.get());
    std::array<std::int32_t,9>physical;std::vector<float>mask(9,-std::numeric_limits<float>::infinity());physical.fill(-1);
    for(unsigned i=0;i<9;++i){const auto t=i/3;const auto id=ids[i];if(id>=ranges[t*2]&&id<ranges[t*2+1]){physical[i]=t<2?(id<2?6+id:2):0;mask[i]=0;}}
    const auto wanted=selected(encoding,data,physical);EXPECT_EQ(read<std::uint16_t>(r.context,out[0]),wanted);EXPECT_EQ(read<float>(r.context,out[1]),mask);
    EXPECT_FALSE(slot.prepare_layout(version).ok());if(sink)check_sink(r,*sink,out,wanted,mask);
    if(!phase){old=out[0];old_bits=wanted;}else EXPECT_EQ(read<std::uint16_t>(r.context,*old),old_bits);
  }
  // Fresh empty history has no ReadLease pages; owning every page writer
  // simultaneously proves padding cannot borrow an arbitrary arena row0.
  auto empty_arena=c::PagedArena::create(schema,r.device.ref(),{.max_backing_bytes=8*encoding.row_bytes()+1}).value();
  std::array<c::PagedPageWriter,4>writers;for(unsigned p=0;p<4;++p)writers[p]=empty_arena.acquire(p).value();
  const std::array<std::int32_t,3>zero{};const std::array<std::int32_t,2>tails{};auto empty_version=schema.version({qi,zero,{},tails}).value();
  auto empty=take(n::PagedRowCacheGatherPaddedInvocation::create(r.context,r.device,r.host_resource,schema,desc,4),"empty first prefill reader");ASSERT_TRUE(empty.prepare_layout(empty_version).ok());
  const std::array<std::int32_t,6>empty_ranges{};auto empty_frame=empty_arena.read(empty_version,{}).value();
  {auto done=take(empty.submit(empty_frame,r.load(std::as_bytes(std::span(ids))),r.load(std::as_bytes(std::span(empty_ranges)))),"zero footprint first prefill");auto result=take(done.wait(),"zero footprint wait");auto out=outputs(*result.get());
    const std::vector<std::uint16_t>wanted(9*encoding.depth(),0);const std::vector<float>mask(9,-std::numeric_limits<float>::infinity());EXPECT_EQ(read<std::uint16_t>(r.context,out[0]),wanted);EXPECT_EQ(read<float>(r.context,out[1]),mask);if(sink)check_sink(r,*sink,out,wanted,mask);}
  for(unsigned p=0;p<4;++p)EXPECT_TRUE(writers[p].valid());
  for(int kind:{0,1,2}){auto badids=ids;std::array<std::int32_t,6>ranges{0,1,0,2,0,1};if(kind<2)badids.back()=kind?-2:4;else ranges.back()=2;
    auto failed=take(n::PagedRowCacheGatherPaddedInvocation::create(r.context,r.device,r.host_resource,schema,desc,4),"invalid typed reader");ASSERT_TRUE(failed.prepare_layout(version).ok());
    auto done=take(failed.submit(arena.read(version,std::array{pages[0],pages[1],pages[3]}).value(),r.load(std::as_bytes(std::span(badids))),r.load(std::as_bytes(std::span(ranges)))),"invalid typed submit");auto result=done.wait();ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange());EXPECT_EQ(read<std::uint16_t>(r.context,*old),old_bits);}
 }
}
}
#if defined(HYLE_NATIVE_ROW_PADDED_APPLE)
TEST(AppleRowCachePadded,DirectPreservesOnlyMinusOnePaddingAcrossSlots){ordinary(false);}
TEST(AppleRowCachePadded,GraphPreservesCheckedOutputAndOddBankOrigins){ordinary(true);}
TEST(AppleRowCachePadded,TypedFirstPrefillConsumesMaskAndEmptyHistoryHoldsAllWriters){typed();}
#else
TEST(VulkanRowCachePadded,DirectPreservesOnlyMinusOnePaddingAcrossSlots){ordinary(false);}
TEST(VulkanRowCachePadded,GraphPreservesCheckedOutputAndOddBankOrigins){ordinary(true);}
TEST(VulkanRowCachePadded,TypedFirstPrefillConsumesMaskAndEmptyHistoryHoldsAllWriters){typed();}
#endif
