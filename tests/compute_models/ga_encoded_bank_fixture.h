#pragma once

#include "expert_working_set_fixture.h"

// A real original-FP4 selected bank, not NVFP4 or an expanded floating bank.
// This isolates encoded-fragment transport before the complete GA FFN journey.
namespace ga_encoded_bank_test {
namespace base=expert_working_set_test;
namespace c=base::c;
namespace g=base::g;
namespace m=base::m;
namespace nv=base::nv;
using base::take;
using base::check;
using base::dense;
using base::Type;
constexpr std::int64_t experts=7,global_experts=256,tokens=2,slots_count=6,n=2048,k=4096;
constexpr std::uint64_t weight_id=10,map_id=50;
inline std::array<std::int32_t,7> selection(unsigned epoch) {
  if(epoch==0)return {17,251,3,127,63,192,224};
  if(epoch==1)return {9,31,255,0,82,173,234};
  return {224,17,192,3,251,63,127};
}
inline std::array<std::int32_t,12> selected_ids(unsigned epoch) {
  const auto ids=selection(epoch);
  return {ids[2],ids[0],ids[6],ids[4],ids[1],ids[3],
      ids[5],ids[1],ids[3],ids[0],ids[6],ids[2]};
}
struct Case {
  const c::TensorDesc a_desc=dense(Type::kBFloat16,{tokens,1,k});
  const c::TensorDesc ids_desc=dense(Type::kInt32,{tokens,slots_count});
  const c::TensorDesc output_desc=dense(Type::kBFloat16,{tokens,slots_count,n});
  std::vector<base::BankSpec> banks{
      {weight_id,take(c::float4_grouped_weight_description(experts,n,k),"original FP4 bank")},
      {map_id,dense(Type::kInt32,{global_experts})}};
  std::vector<g::WeightBinding> bindings;
  g::Graph graph;
  g::ValueId a,ids,output;
  Case(){
    auto b=take(g::GraphBuilder::create(),"encoded bank builder");
    a=take(b.input(a_desc),"original BF16 activation");
    ids=take(b.input(ids_desc),"global selected IDs");
    const auto w=take(b.input(banks[0].description),"original typed encoded bank");
    const auto map=take(b.input(banks[1].description),"same-generation remap");
    bindings={{w,weight_id},{map,map_id}};
    const auto local=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},map,ids),"global to local IDs");
    output=take(b.add(c::model_ops::grouped_float4_linear(),w,a,local),"original FP4/A128 program");
    check(b.publish(output),"publish encoded result");
    graph=take(std::move(b).finish(),"encoded bank graph");
  }
  std::size_t logical_bytes()const{return banks[0].description.required_bytes().value()+global_experts*4;}
  std::size_t capacity()const{return (logical_bytes()+4096+255)/256*256;}
};
struct Sources {
  base::Sources original;
  Sources()=default;
  explicit Sources(const Case &f,unsigned epoch){
    const auto globals=selection(epoch);
    const auto routed=selected_ids(epoch);
    const auto code_plane=std::size_t(experts*n*k/2),code_bytes=std::size_t(n*k/2);
    const auto scale_bytes=std::size_t(n*(k/32)*4);
    for(std::size_t local=0;local<globals.size();++local){
      const auto global=globals[local];
      std::vector<std::uint8_t> codes(code_bytes);
      for(std::int64_t row=0;row<n;++row)for(std::int64_t pair=0;pair<k/2;++pair)
        codes[std::size_t(row*(k/2)+pair)]=std::uint8_t(
            base::Sources::code(global,row,pair*2,epoch)|
            (base::Sources::code(global,row,pair*2+1,epoch)<<4));
      const auto raw=std::as_bytes(std::span(codes));
      // Exercise a byte split that is deliberately not a logical F32 boundary.
      if(local==0){
        original.add(weight_id,dense(Type::kUInt8,{1}),raw.first(1),0);
        original.add(weight_id,dense(Type::kUInt8,{std::int64_t(code_bytes-1)}),raw.subspan(1),1);
      }else original.add(weight_id,dense(Type::kUInt8,{std::int64_t(code_bytes)}),raw,local*code_bytes);
      std::vector<float> scales(std::size_t(n*(k/32)));
      for(std::int64_t row=0;row<n;++row)for(std::int64_t group=0;group<k/32;++group)
        scales[std::size_t(row*(k/32)+group)]=std::ldexp(1.F,int((global+row+group+epoch)%3)-4);
      original.add(weight_id,dense(Type::kUInt8,{std::int64_t(scale_bytes)}),
          std::as_bytes(std::span(scales)),code_plane+local*scale_bytes);
    }
    std::vector<std::int32_t> map(std::size_t(global_experts),-1);
    for(std::size_t local=0;local<globals.size();++local)map[std::size_t(globals[local])]=std::int32_t(local);
    original.add(map_id,dense(Type::kInt32,{global_experts}),std::as_bytes(std::span(map)),0);
    std::vector<std::uint16_t> expected(std::size_t(tokens*slots_count*n));
    for(std::int64_t token=0;token<tokens;++token)for(std::int64_t slot=0;slot<slots_count;++slot)
      for(std::int64_t row=0;row<n;++row){
        const auto global=routed[std::size_t(token*slots_count+slot)],hot=token==0?0:128;
        // One-hot6 survives the original dynamic A128 power-of-two E4 QDQ
        // exactly. Every other W byte remains real, dense and transported.
        const auto scale=std::ldexp(1.F,int((global+row+hot/32+epoch)%3)-4);
        expected[std::size_t((token*slots_count+slot)*n+row)]=nv::bf16(
            6.F*nv::e2m1[base::Sources::code(global,row,hot,epoch)]*scale);
      }
    original.expected=base::raw(expected);
    std::reverse(original.slices.begin(),original.slices.end());
    std::size_t bytes=0;
    for(const auto &slice:original.slices)bytes+=slice.source.description().required_bytes().value();
    EXPECT_EQ(bytes,f.logical_bytes());
    for(const auto &buffer:original.storage){
      original.original_bytes.emplace_back(buffer.size());
      check(buffer.view().read(original.original_bytes.back()),"snapshot encoded source");
    }
  }
};
template<class Api> void expect_output(Api &api,const base::Episode &episode){
  const auto tensor=take(episode.output.tensor(),"actual encoded result");
  std::vector<std::byte> bytes(episode.expected.size());
  check(api.read(tensor.storage(),bytes),"read encoded result");EXPECT_EQ(bytes,episode.expected);
}
template<class Api,class Batch> base::Episode execute(Api &api,const Case &f,const g::Executable &executable,
    const Batch &batch,unsigned epoch,std::vector<std::byte> expected,c::LeasePool &pool,bool invalid=false){
  base::Episode result;
  result.invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"original encoded invocation");
  std::vector<std::uint16_t> activation(std::size_t(tokens*k),0);
  activation[0]=nv::bf16(6.F);activation[std::size_t(k+128)]=nv::bf16(6.F);
  auto ids=selected_ids(epoch);if(invalid)ids.back()=254;
  const std::array payload{base::raw(activation),base::raw(std::span<const std::int32_t>(ids))};
  const std::array desc{f.a_desc,f.ids_desc};const std::array ports{f.a,f.ids};
  for(std::size_t i=0;i<2;++i){
    result.inputs[i]=take(m::Buffer::create(api.resource,payload[i].size(),256),"encoded dynamic input");
    check(api.write(result.inputs[i].view(),payload[i]),"write encoded dynamic input");
    check(result.invocation.bind(ports[i],take(c::ConstTensorView::create(desc[i],result.inputs[i].view()),"encoded input tensor")),"bind original input");
  }
  check(api.bind(result.invocation,f.bindings,batch),"atomic original FP4/map bundle");
  if(pool.valid())check(result.invocation.bind(f.output,f.output_desc,
      take(pool.allocate(f.output_desc.required_bytes().value()),"encoded output writer")),"bind encoded writer");
  check(api.prepare(result.invocation),"prepare original encoded graph");
  auto done=take(result.invocation.submit(api.queues),"submit original encoded graph");
  const auto status=done.wait();
  if(invalid){EXPECT_FALSE(status.ok());EXPECT_FALSE(result.invocation.result(f.output));return result;}
  check(status,"encoded completion");
  result.output=take(result.invocation.result(f.output),"retain original encoded output");
  result.lease=take(result.output.storage_lease(),"independent encoded result backing");
  result.expected=std::move(expected);expect_output(api,result);return result;
}
template<class Api> void run(Api &api){
  Case f;auto compiled=take(api.compile(f.graph),"compile encoded working set");
  auto slots=take(api.create_slots(f.banks,f.capacity()),"original two encoded bank slots");
  c::LeasePool pool;
  if(take(compiled.executable.output_ownership(),"encoded ownership")==g::OutputOwnership::kCaller)
    pool=take(c::LeasePool::create(api.resource,4*f.output_desc.required_bytes().value(),{.max_blocks=4}),"encoded result arena");
  std::array<base::Episode,3> episodes;
  std::array<typename Api::Batch,2> batches;
  for(unsigned epoch=0;epoch<2;++epoch){
    Sources source(f,epoch);auto expected=source.original.expected;
    batches[epoch]=take(api.upload(slots,epoch,source.original.slices),"upload actual encoded byte fragments");
    source.original.expect_unchanged();source=Sources{};
    base::expect_report(batches[epoch],f.logical_bytes(),f.capacity());
    episodes[epoch]=execute(api,f,compiled.executable,batches[epoch],epoch,std::move(expected),pool);
  }
  EXPECT_NE(episodes[0].expected,episodes[1].expected);expect_output(api,episodes[0]);
  Sources third(f,2);auto expected=third.original.expected;
  EXPECT_FALSE(api.upload(slots,0,third.original.slices));batches[0]={};
  EXPECT_FALSE(api.upload(slots,0,third.original.slices));episodes[0]=base::Episode{};
  batches[0]=take(api.upload(slots,0,third.original.slices),"reuse encoded slot after actual readers retire");
  third.original.expect_unchanged();third=Sources{};
  base::expect_report(batches[0],f.logical_bytes(),f.capacity());
  {auto failed=execute(api,f,compiled.executable,batches[0],2,{},pool,true);}
  episodes[2]=execute(api,f,compiled.executable,batches[0],2,std::move(expected),pool);
  expect_output(api,episodes[1]);
  batches={};episodes[1].invocation={};episodes[2].invocation={};compiled={};slots={};pool={};
  expect_output(api,episodes[1]);expect_output(api,episodes[2]);
  ::testing::Test::RecordProperty("successful_submissions",3);
  ::testing::Test::RecordProperty("expected_failed_submissions",1);
  ::testing::Test::RecordProperty("selected_bank","7 of256,original FP4 N2048/K4096,Top6,T2; encoded W+F32 scale tail and plain I32 map");
  ::testing::Test::RecordProperty("byte_fragments","16,one-byte split,out-of-order,original host sources destroyed before submit");
  ::testing::Test::RecordProperty("scope","encoded resource bridge and existing original FP4 math; complete routed/shared FFN remains separate");
}
} // namespace ga_encoded_bank_test
