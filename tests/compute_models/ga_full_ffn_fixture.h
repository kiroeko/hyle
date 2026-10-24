#pragma once

#include "d5_deepseek_official_fragments.h"
#include "ga_indexer_state_fixture.h"
#include "expert_working_set_fixture.h"

#include <iomanip>
#include <sstream>

// One source-bound consumer of the existing complete ordinary FFN recipe.
// Physical bank import differs between host and native WeightSlots; graph,
// finite payloads, independent oracle and execution checks are shared.
namespace ga_full_ffn_test {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace recipe=d5_deepseek_official_fragments;
namespace moe=d5_moe_execution_fragments;
namespace base=expert_working_set_test;
namespace f=ga_indexer_state_test;
namespace c=base::c;
namespace g=base::g;
namespace m=base::m;
using base::take;
using base::check;
using base::dense;
using base::Type;
constexpr std::array<std::int32_t,7> global_ids{11,37,83,127,191,223,251};
constexpr std::uint64_t map_id=1000;
inline std::string precise(double value){std::ostringstream out;out<<std::setprecision(9)<<value;return out.str();}

struct Case {
  bool hash;
  sem::Fragment fixture;
  std::vector<base::BankSpec> banks;
  std::vector<g::WeightBinding> bindings;
  std::vector<g::ValueId> inputs,outputs;
  std::vector<std::vector<std::byte>> bank_bytes;
  g::Graph graph;
  explicit Case(bool use_hash):hash(use_hash){
    auto model=moe::cases::kCases[4];
    // Formal Gate explicitly calls linear(x.float(), weight.float()). These
    // synthetic F32 coefficients do not claim a checkpoint storage dtype.
    model.router_bf16=false;
    fixture=recipe::moe_chain(model,hash,nullptr,false);
    bank_bytes=std::move(fixture.constant_bytes);
    const auto old_router=bank_bytes[0];
    const auto row_bytes=std::size_t(model.hidden)*sizeof(float);
    std::fill(bank_bytes[0].begin(),bank_bytes[0].end(),std::byte{0});
    for(std::size_t local=0;local<global_ids.size();++local)
      std::memcpy(bank_bytes[0].data()+std::size_t(global_ids[local])*row_bytes,
          old_router.data()+local*row_bytes,row_bytes);
    if(!hash){
      const auto old_bias=bank_bytes[1];std::vector<float> bias(std::size_t(model.experts),-4.F);
      for(std::size_t local=0;local<global_ids.size();++local)
        std::memcpy(&bias[std::size_t(global_ids[local])],old_bias.data()+local*4,4);
      bank_bytes[1]=base::raw(bias);
    }
    for(auto &replay:fixture.replays){
      for(auto &tag:replay.outputs[0])tag=float(global_ids.at(std::size_t(tag)));
      if(hash)for(std::size_t byte=0;byte<replay.inputs[2].size();byte+=4){
        std::int32_t local;std::memcpy(&local,replay.inputs[2].data()+byte,4);
        const auto global=global_ids.at(std::size_t(local));
        std::memcpy(replay.inputs[2].data()+byte,&global,4);
      }
    }
    auto b=take(g::GraphBuilder::create(),"complete ordinary GA FFN");
    std::vector<g::ValueId> values(fixture.descriptions.size());
    for(auto value:fixture.inputs){
      const auto input=take(b.input(fixture.description(value)),"original FFN input");
      values[value.ordinal()]=input;inputs.push_back(input);
    }
    for(std::size_t i=0;i<fixture.constants.size();++i){
      const auto old=fixture.constants[i];auto description=fixture.description(old);
      // The original F32 router RHS is a transpose view. Register physical
      // dense [E,D], then restore the same logical [D,E] view in the graph.
      if(i==0)description=take(description.transposed(),"physical F32 router bank");
      const auto input=take(b.input(description),"original typed weight bank");
      banks.push_back({10+i,description});bindings.push_back({input,10+i});
      values[old.ordinal()]=i==0?take(b.add(take(c::View::permute(
          std::array<std::int32_t,2>{1,0}),"router transpose"),input),"router metadata view"):input;
    }
    const auto map_desc=dense(Type::kInt32,{256});
    const auto map=take(b.input(map_desc),"explicit I32 global-to-local map");
    banks.push_back({map_id,map_desc});bindings.push_back({map,map_id});
    std::vector<std::int32_t> mapping(256,-1);
    for(std::size_t i=0;i<global_ids.size();++i)mapping[std::size_t(global_ids[i])]=std::int32_t(i);
    bank_bytes.push_back(base::raw(mapping));
    g::ValueId local_ids;bool mapped=false;unsigned fp4_nodes=0,fp8_nodes=0;
    for(const auto &step:fixture.nodes){
      std::vector<g::ValueId> args;
      for(auto value:step.inputs)args.push_back(values[value.ordinal()]);
      if(step.operation.custom_as<c::model_ops::GroupedFloat4Linear>()){
        if(!mapped){local_ids=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},map,args[2]),"checked original expert remap");mapped=true;}
        args[2]=local_ids;++fp4_nodes;
      }
      if(step.operation.custom_as<c::model_ops::Float8BlockLinear>())++fp8_nodes;
      const auto node=take(b.add_node(step.operation,args),"existing FFN recipe node");
      for(std::size_t i=0;i<step.outputs.size();++i)
        values[step.outputs[i].ordinal()]=take(b.result(node,i),"ordinary FFN result");
    }
    EXPECT_EQ(fp4_nodes,3U);EXPECT_EQ(fp8_nodes,3U);
    for(auto value:fixture.outputs){
      const auto output=values[value.ordinal()];check(b.publish(output),"publish FFN source boundary");outputs.push_back(output);
    }
    graph=take(std::move(b).finish(),"complete GA FFN graph");
  }
  std::size_t logical_bytes()const{std::size_t bytes=0;for(const auto &bank:banks)bytes+=bank.description.required_bytes().value();return bytes;}
  std::size_t capacity()const{std::size_t bytes=0;for(const auto &bank:banks)bytes+=((bank.description.required_bytes().value()+255)/256)*256;return bytes;}
  std::size_t output_arena()const{
    std::size_t bytes=0;for(auto output:fixture.outputs)bytes+=((fixture.description(output).required_bytes().value()+255)/256)*256;
    return fixture.replays.size()*bytes;
  }
};

struct Episode {
  g::Invocation invocation;
  std::vector<c::ReadLease> inputs;
  std::vector<g::GraphOutput> outputs;
  std::vector<float> final_words;
};
struct Metrics {double maximum_peak_error=0,maximum_relative_l2=0;int maximum_first_projection_bf16_ulp=0;
  std::array<double,12> peak_error{},relative_l2{};};
inline void compare(std::size_t port,std::span<const float> actual,
    std::span<const float> expected,Metrics &metrics){
  SCOPED_TRACE(::testing::Message()<<"source boundary "<<port);
  ASSERT_EQ(actual.size(),expected.size());
  if(port==0){EXPECT_EQ(std::vector<float>(actual.begin(),actual.end()),std::vector<float>(expected.begin(),expected.end()));return;}
  double peak=0,error=0,square_error=0,square_reference=0;
  for(std::size_t i=0;i<actual.size();++i){
    ASSERT_TRUE(std::isfinite(actual[i]))<<i;
    peak=std::max(peak,double(std::abs(expected[i])));
    error=std::max(error,double(std::abs(actual[i]-expected[i])));
    square_error+=double(actual[i]-expected[i])*double(actual[i]-expected[i]);
    square_reference+=double(expected[i])*expected[i];
  }
  ASSERT_GT(peak,0.);ASSERT_GT(square_reference,0.);
  metrics.maximum_peak_error=std::max(metrics.maximum_peak_error,error/peak);
  metrics.maximum_relative_l2=std::max(metrics.maximum_relative_l2,std::sqrt(square_error/square_reference));
  metrics.peak_error[port]=std::max(metrics.peak_error[port],error/peak);
  metrics.relative_l2[port]=std::max(metrics.relative_l2[port],std::sqrt(square_error/square_reference));
  // Declared before backend execution: finite dyadic first projections are
  // exact. F32 routing/SiLU admits its existing F64-assisted oracle error.
  // Later BF16->A8 midpoint propagation uses a bounded scale-relative gate;
  // it cannot hide zero results or the independent wrong-postdown witness.
  const bool arithmetic=port==1||port==8||port==11;
  EXPECT_LE(std::sqrt(square_error/square_reference),arithmetic?2.e-5:5.e-3);
  EXPECT_LE(error/peak,arithmetic?2.e-5:3.e-2);
  if(port==1)for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected[i],2.e-6F);
  if(port==6||port==7||port==9||port==10){
    const auto order=[](std::uint16_t bits){return bits&0x8000?0x8000-int(bits&0x7fff):0x8000+int(bits);};
    int maximum=0;std::size_t non_bfloat=0;
    for(std::size_t i=0;i<actual.size();++i){
      non_bfloat+=actual[i]!=recipe::fp8::bfloat(actual[i]);
      const auto distance=std::abs(order(base::nv::bf16(actual[i]))-order(base::nv::bf16(expected[i])));
      maximum=std::max(maximum,distance);
    }
    metrics.maximum_first_projection_bf16_ulp=std::max(metrics.maximum_first_projection_bf16_ulp,maximum);
    EXPECT_EQ(non_bfloat,0U);EXPECT_EQ(maximum,0);
  }
}
template<class Api,class Bind> Episode execute(Api &api,const Case &test,
    const g::Executable &executable,Bind bind_weights,unsigned replay,c::LeasePool &pool,Metrics &metrics){
  SCOPED_TRACE(::testing::Message()<<"hash="<<test.hash<<"/replay="<<replay);
  Episode episode;episode.invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"independent full FFN invocation");
  check(bind_weights(episode.invocation),"bind original complete weight generation");
  for(std::size_t i=0;i<test.inputs.size();++i){
    const auto &bytes=test.fixture.replays[replay].inputs[i];
    episode.inputs.push_back(f::upload(api.resource,bytes,[&](auto target,auto source){return api.write(target,source);}));
    check(episode.invocation.bind(test.inputs[i],test.fixture.description(test.fixture.inputs[i]),episode.inputs.back()),"bind original FFN dynamic input");
  }
  for(std::size_t i=0;i<test.outputs.size();++i)
    if(take(executable.output_ownership(i),"FFN output ownership")==g::OutputOwnership::kCaller){
      const auto description=test.fixture.description(test.fixture.outputs[i]);
      check(episode.invocation.bind(test.outputs[i],description,take(pool.allocate(description.required_bytes().value()),"FFN original output writer")),"bind owned FFN output");
    }
  check(api.prepare(episode.invocation),"prepare full FFN");
  auto done=take(episode.invocation.submit(api.queues),"submit complete original FFN");check(done.wait(),"full FFN completion");
  for(std::size_t i=0;i<test.outputs.size();++i){
    episode.outputs.push_back(take(episode.invocation.result(test.outputs[i]),"retained FFN source boundary"));
    auto actual=f::read_result<float>(episode.outputs.back(),[&](m::ConstBufferView source,std::span<std::byte> destination){return api.read(source,destination);});
    compare(i,actual,test.fixture.replays[replay].outputs[i],metrics);
    if(i==5)episode.final_words=std::move(actual);
  }
  EXPECT_TRUE(episode.invocation.reset().IsUnavailable());
  return episode;
}
template<class Api> void expect_old(Api &api,const Episode &episode){
  EXPECT_EQ(f::read_result<float>(episode.outputs[5],[&](m::ConstBufferView source,std::span<std::byte> destination){return api.read(source,destination);}),episode.final_words);
}
inline void record(const Metrics &metrics){
  ::testing::Test::RecordProperty("successful_submissions",6);
  ::testing::Test::RecordProperty("geometry","FlashD4096/H2048,E256,Top6,resident7;score/hash three payloads each");
  ::testing::Test::RecordProperty("maximum_peak_normalized_error",precise(metrics.maximum_peak_error));
  ::testing::Test::RecordProperty("maximum_relative_l2_error",precise(metrics.maximum_relative_l2));
  ::testing::Test::RecordProperty("maximum_first_projection_bf16_ulp",metrics.maximum_first_projection_bf16_ulp);
  for(std::size_t i=1;i<12;++i){
    ::testing::Test::RecordProperty("stage"+std::to_string(i)+"_peak_error",precise(metrics.peak_error[i]));
    ::testing::Test::RecordProperty("stage"+std::to_string(i)+"_relative_l2",precise(metrics.relative_l2[i]));
  }
  ::testing::Test::RecordProperty("scope","complete finite FFN and original bank/ID/lifetime composition;synthetic payloads,not whole model or throughput");
}
template<class Api> void run_gpu(Api &api){
  Metrics metrics;
  for(bool hash:{false,true}){
    Case test(hash);auto compiled=take(api.compile(test.graph),"compile complete GA FFN");
    auto slots=take(api.create_slots(test.banks,test.capacity()),"mixed encoded/plain original bank catalog");
    base::Sources sources;
    for(std::size_t i=0;i<test.banks.size();++i){
      const auto &bank=test.banks[i];const auto &bytes=test.bank_bytes[i];
      const auto source_desc=bank.description.is_plain()?bank.description:dense(Type::kUInt8,{std::int64_t(bytes.size())});
      sources.add(bank.id,source_desc,bytes,0);
    }
    auto batch=take(api.upload(slots,0,sources.slices),"one real mixed FFN weight generation");
    base::expect_report(batch,test.logical_bytes(),test.capacity());
    sources=base::Sources{};test.bank_bytes={};
    auto pool=take(c::LeasePool::create(api.resource,test.output_arena(),{.max_blocks=64}),"finite full FFN output arena");
    std::vector<Episode> episodes;
    for(unsigned replay=0;replay<3;++replay){
      episodes.push_back(execute(api,test,compiled.executable,[&](auto &invocation){return api.bind(invocation,test.bindings,batch);},replay,pool,metrics));
      expect_old(api,episodes.front());
    }
    EXPECT_NE(episodes[0].final_words,episodes[1].final_words);
    batch={};compiled={};slots={};pool={};for(auto &episode:episodes)episode.invocation={};
    for(const auto &episode:episodes)expect_old(api,episode);
  }
  record(metrics);
}
template<class Api> void run_cpu(Api &api){
  Metrics metrics;
  for(bool hash:{false,true}){
    Case test(hash);auto compiled=take(api.compile(test.graph),"compile complete CPU GA FFN");
    std::vector<c::ReadLease> weights;
    for(const auto &bytes:test.bank_bytes)weights.push_back(f::upload(api.resource,bytes,[&](auto target,auto source){return api.write(target,source);}));
    test.bank_bytes={};
    auto pool=take(c::LeasePool::create(api.resource,test.output_arena(),{.max_blocks=64}),"finite host FFN output arena");
    std::vector<Episode> episodes;
    for(unsigned replay=0;replay<3;++replay){
      episodes.push_back(execute(api,test,compiled.executable,[&](auto &invocation){
        for(std::size_t i=0;i<weights.size();++i){
          const auto status=invocation.bind(test.bindings[i].value,test.banks[i].description,weights[i]);
          if(!status.ok())return status;
        }return hyle::Status::Ok();
      },replay,pool,metrics));
      expect_old(api,episodes.front());
    }
    EXPECT_NE(episodes[0].final_words,episodes[1].final_words);
    weights={};compiled={};pool={};for(auto &episode:episodes)episode.invocation={};
    for(const auto &episode:episodes)expect_old(api,episode);
  }
  record(metrics);
}
} // namespace ga_full_ffn_test
