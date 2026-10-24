#pragma once

#include "expert_working_set_fixture.h"
#include "ga_indexer_state_fixture.h"
#include "mlx_affine_bitstream.h"
#include <iomanip>
#include <numeric>
#include <sstream>

// Fixed default MLX-VLM GLM5-Next ordinary MoE reference. This uses the
// checkpoint's role geometry/types, but synthetic original-format bytes.
// The backend adapter is the existing ga_full_ffn Backend API. GPU adapters
// must create original WeightSlots with max_fragments=128 (92 real slices).
namespace glm_flash_mixed_mlx_test {
namespace base=expert_working_set_test;
namespace f=ga_indexer_state_test;
namespace nv=nvfp4_reference_test;
namespace c=base::c;
namespace g=base::g;
namespace m=base::m;
using base::take;
using base::check;
using base::dense;
using base::Type;
inline constexpr std::int64_t tokens=2,hidden=4096,intermediate=2048,
    global_experts=288,resident=9,slots=8,group=64;
inline constexpr std::array<std::int32_t,9> globals{11,37,83,127,191,223,251,271,287};
inline constexpr std::uint64_t map_id=1000,bias_id=1001;
inline constexpr std::size_t output_count=8;

inline float rn(double x){return nv::f32(x);}
inline float mul(float x,float y){return rn(double(x)*double(y));}
inline float add(float x,float y){return rn(double(x)+double(y));}
// Metadata is generated as finite normal binary16 words, independently widened.
inline float half(std::uint16_t x){
  const auto magnitude=std::ldexp(float(1024U+(x&1023U)),int((x>>10)&31U)-25);
  return x&0x8000U?-magnitude:magnitude;
}
inline std::int64_t rows(unsigned stage){return stage%3==2?hidden:intermediate;}
inline std::int64_t depth(unsigned stage){return stage%3==2?intermediate:hidden;}
inline unsigned width(unsigned stage){return stage<2?4U:stage==2?5U:6U;}
inline std::uint64_t bank_id(unsigned stage,unsigned port){return 10U+stage*3U+port;}
inline c::TensorDesc fragment_desc(unsigned stage,unsigned port){
  return dense(port?Type::kFloat16:Type::kUInt32,
      {rows(stage),port?depth(stage)/group:depth(stage)/32*std::int64_t(width(stage))});
}
inline std::uint16_t scale_bits(unsigned stage,std::int32_t expert,
    std::int64_t n,std::int64_t block,unsigned epoch){
  const std::uint16_t exponent=stage%3==2?0x2400U:0x3800U;
  const auto fraction=std::uint16_t(3U+unsigned((expert*17+n*11+block*29+std::int64_t(epoch)*47+stage*31)%797));
  const auto sign=(expert+n+block+stage)%5==0?0x8000U:0U;
  return std::uint16_t(exponent|fraction|sign);
}
inline std::array<std::int64_t,2> positions(unsigned stage,std::int32_t expert,
    std::int64_t n,unsigned epoch){
  const auto k=depth(stage);
  return {(n*7+std::int64_t(expert)*13+stage*19+std::int64_t(epoch)*23)%(k/2),
      k/2+(n*11+std::int64_t(expert)*5+stage*31+std::int64_t(epoch)*17)%(k/2)};
}
inline unsigned code(unsigned stage,std::int32_t expert,std::int64_t n,
    unsigned lane,unsigned epoch){
  return 2U+unsigned((expert*3+n*5+stage*7+lane*11+epoch*13)%((1U<<width(stage))-2U));
}
inline void put(std::span<std::uint32_t> words,std::size_t column,unsigned bits,unsigned value){
  // Independent single-bit scatter, including cross-word affine5/6 codes.
  for(unsigned bit=0;bit<bits;++bit){
    const auto p=column*bits+bit;const auto mask=std::uint32_t{1}<<(p%32);
    words[p/32]=(words[p/32]&~mask)|(((value>>bit)&1U)?mask:0U);
  }
}
inline std::array<std::vector<std::byte>,3> fragment(unsigned stage,
    std::int32_t expert,unsigned epoch){
  const auto n=rows(stage),k=depth(stage),words=k/32*std::int64_t(width(stage));
  const std::vector<std::uint32_t> ones(std::size_t(k),1U);
  const auto template_row=mlx_affine_test::pack(ones,1,std::size_t(k),width(stage));
  std::vector<std::uint32_t> packed(std::size_t(n*words));
  std::vector<std::uint16_t> scales(std::size_t(n*k/group)),biases(scales.size());
  for(std::int64_t row=0;row<n;++row){
    auto target=std::span(packed).subspan(std::size_t(row*words),std::size_t(words));
    std::copy(template_row.begin(),template_row.end(),target.begin());
    const auto p=positions(stage,expert,row,epoch);
    for(unsigned lane=0;lane<2;++lane)put(target,std::size_t(p[lane]),width(stage),code(stage,expert,row,lane,epoch));
    for(std::int64_t block=0;block<k/group;++block){
      const auto i=std::size_t(row*(k/group)+block);
      scales[i]=scale_bits(stage,expert,row,block,epoch);
      biases[i]=std::uint16_t(scales[i]^0x8000U); // Baseline code1 decodes to exact zero.
    }
  }
  return {base::raw(packed),base::raw(scales),base::raw(biases)};
}
inline std::vector<float> project(unsigned stage,std::int32_t expert,
    std::span<const float> a,unsigned epoch,bool wrong_metadata=false){
  std::vector<float> out(std::size_t(rows(stage)),0.F);
  for(std::int64_t n=0;n<rows(stage);++n){
    const auto p=positions(stage,expert,n,epoch);float sum=0;
    for(unsigned lane=0;lane<2;++lane){
      float sf=half(scale_bits(stage,expert,n,p[lane]/group,epoch));
      if(wrong_metadata)sf=nv::rounded_bf16(sf);
      const float bias=-sf;
      // F32 product then F32 add. Never narrow decoded W to BF16.
      const float w=add(mul(float(code(stage,expert,n,lane,epoch)),sf),bias);
      sum=add(sum,mul(a[std::size_t(p[lane])],w));
    }
    out[std::size_t(n)]=sum;
  }
  return out;
}
inline std::vector<float> activate(std::span<const float> gate,std::span<const float> up){
  std::vector<float> out(gate.size());
  for(std::size_t i=0;i<out.size();++i){
    const float a=std::min(gate[i],10.F),b=std::clamp(up[i],-10.F,10.F);
    const float silu=rn(double(a)/(1.+std::exp(-double(a))));out[i]=mul(silu,b);
  }
  return out;
}
inline std::vector<float> correction(){
  std::vector<float> out(std::size_t(global_experts),-2.F);
  for(std::size_t i=0;i<globals.size();++i)out[std::size_t(globals[i])]=float(int(i%3)-1)/32.F;
  out[286]=0.F; // Normal logit-8 stays out; failed-episode logit20 must select it.
  return out;
}
inline std::vector<float> logits(unsigned phase,bool absent=false){
  std::vector<float> out(std::size_t(tokens*global_experts),-8.F);
  for(std::int64_t t=0;t<tokens;++t)for(std::size_t local=0;local<globals.size();++local){
    const auto rank=(local+std::size_t(t)*2+phase*3)%globals.size();
    out[std::size_t(t*global_experts+globals[local])]=float(12-int(rank)*3)/16.F;
  }
  // Global286 is valid for the original router, absent from this resident set.
  if(absent)out[std::size_t((tokens-1)*global_experts+286)]=20.F;
  return out;
}
inline std::vector<std::uint16_t> activation(unsigned phase){
  constexpr std::array<float,5> values{.75F,-.5F,1.F,1.25F,-1.5F};
  std::vector<std::uint16_t> out(std::size_t(tokens*hidden));
  for(std::size_t i=0;i<out.size();++i)out[i]=nv::bf16(values[(i*3+i/std::size_t(hidden)+phase*7)%values.size()]);
  return out;
}
struct Reference {
  std::vector<std::int32_t> ids;
  // Ports1..7: normalized weights, gate, up, activation, down, shared, final.
  std::array<std::vector<float>,7> values;
};
inline Reference oracle(unsigned epoch,unsigned phase,bool wrong_metadata=false,
    bool premature_bf16=false,bool wrong_softmax=false){
  Reference out;const auto a_bits=activation(phase);const auto logit=logits(phase),bias=correction();
  for(std::int64_t t=0;t<tokens;++t){
    std::vector<float> a(std::size_t(hidden),0.F);
    for(std::int64_t k=0;k<hidden;++k)a[std::size_t(k)]=nv::wide(a_bits[std::size_t(t*hidden+k)]);
    std::vector<float> score(std::size_t(global_experts),0.F);std::vector<std::int32_t> order(std::size_t(global_experts),0);
    std::iota(order.begin(),order.end(),0);
    for(std::int64_t e=0;e<global_experts;++e)score[std::size_t(e)]=rn(1./(1.+std::exp(-double(logit[std::size_t(t*global_experts+e)]))));
    std::stable_sort(order.begin(),order.end(),[&](auto l,auto r){return add(score[std::size_t(l)],bias[std::size_t(l)])>add(score[std::size_t(r)],bias[std::size_t(r)]);});
    order.resize(std::size_t(slots));out.ids.insert(out.ids.end(),order.begin(),order.end());
    std::array<float,std::size_t(slots)> selected{};float total=0;
    for(std::size_t s=0;s<selected.size();++s){selected[s]=wrong_softmax?rn(std::exp(double(logit[std::size_t(t*global_experts+order[s])]))):score[std::size_t(order[s])];total=add(total,selected[s]);}
    total=add(total,1.e-20F);std::vector<float> combined(std::size_t(hidden),0.F);
    for(std::size_t s=0;s<selected.size();++s){
      const float weight=mul(rn(double(selected[s])/double(total)),2.5F);out.values[0].push_back(weight);
      auto gate=project(0,order[s],a,epoch,wrong_metadata),up=project(1,order[s],a,epoch,wrong_metadata);
      if(premature_bf16){for(auto &v:gate)v=nv::rounded_bf16(v);for(auto &v:up)v=nv::rounded_bf16(v);}
      const auto act=activate(gate,up);auto down=project(2,order[s],act,epoch,wrong_metadata);
      for(std::size_t k=0;k<down.size();++k)combined[k]=add(combined[k],mul(down[k],weight));
      for(unsigned i=0;i<4;++i){const auto &v=i==0?gate:i==1?up:i==2?act:down;out.values[i+1].insert(out.values[i+1].end(),v.begin(),v.end());}
    }
    auto sg=project(3,0,a,epoch,wrong_metadata),su=project(4,0,a,epoch,wrong_metadata);
    const auto shared=project(5,0,activate(sg,su),epoch,wrong_metadata);
    out.values[5].insert(out.values[5].end(),shared.begin(),shared.end());
    for(std::size_t k=0;k<shared.size();++k)out.values[6].push_back(add(combined[k],shared[k]));
  }
  return out;
}
struct Case {
  g::Graph graph;
  std::vector<base::BankSpec> banks;
  std::vector<g::WeightBinding> bindings;
  std::array<g::ValueId,2> inputs;
  std::array<c::TensorDesc,2> input_desc{dense(Type::kBFloat16,{tokens,1,hidden}),dense(Type::kFloat32,{tokens,global_experts})};
  std::vector<g::ValueId> outputs;
  Case(){
    auto b=take(g::GraphBuilder::create(),"GLM mixed affine full consumer");
    for(unsigned i=0;i<inputs.size();++i)inputs[i]=take(b.input(input_desc[i]),"GLM explicit A/logits boundary");
    std::array<std::array<g::ValueId,3>,6> weights;
    for(unsigned stage=0;stage<6;++stage)for(unsigned port=0;port<3;++port){
      auto desc=fragment_desc(stage,port);
      if(stage<3)desc=dense(desc.data_type(),{resident,desc.dimensions()[0],desc.dimensions()[1]});
      weights[stage][port]=take(b.input(desc),"original F16 metadata bank");
      banks.push_back({bank_id(stage,port),desc});bindings.push_back({weights[stage][port],bank_id(stage,port)});
    }
    const auto map=take(b.input(dense(Type::kInt32,{global_experts})),"original immutable local map");
    banks.push_back({map_id,dense(Type::kInt32,{global_experts})});bindings.push_back({map,map_id});
    const auto bias=take(b.input(dense(Type::kFloat32,{global_experts})),"selection-only correction");
    banks.push_back({bias_id,dense(Type::kFloat32,{global_experts})});bindings.push_back({bias,bias_id});
    const auto binary=[&](c::BinaryKind kind,g::ValueId a,g::ValueId z){return take(b.add(c::Binary{kind},a,z),"GLM F32 pointwise");};
    const auto number=[&](float v){return take(b.add(c::Fill{.output_shape=take(c::TensorShape::create(std::array<std::int64_t,1>{1}),"scalar shape"),.output_type=Type::kFloat32,.value=c::ScalarValue(v)}),"GLM source literal");};
    const auto sigmoid=take(b.add(c::Unary{c::UnaryKind::kSigmoid},inputs[1]),"unbiased sigmoid scores");
    const auto choice=binary(c::BinaryKind::kAdd,sigmoid,bias);
    const auto top=take(b.add_node(c::Operation(c::TopK{.count=std::size_t(slots)}),std::array{choice}),"GLM Top8 choice");
    const auto ids=take(b.result(top,1),"original global selected IDs");
    const auto selected=take(b.add(c::model_ops::gather_elements(),sigmoid,ids),"gather unbiased probabilities");
    const auto sum=take(b.add(c::Reduce{.kind=c::ReductionKind::kSum,.axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"route sum axis"),.keep_dimensions=true,.accumulation_type=Type::kFloat32},selected),"route probability sum");
    const auto route=binary(c::BinaryKind::kMultiply,binary(c::BinaryKind::kDivide,selected,binary(c::BinaryKind::kAdd,sum,number(1.e-20F))),number(2.5F));
    const auto local=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},map,ids),"same-generation global-to-local");
    const auto projection=[&](unsigned stage,g::ValueId a){
      return stage<3?take(b.add(c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine,width(stage),64),a,weights[stage][0],weights[stage][1],weights[stage][2],local),"GLM original grouped projection"):
          take(b.add(c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine,width(stage),64),a,weights[stage][0],weights[stage][1],weights[stage][2]),"GLM original shared projection");
    };
    const auto limited=[&](g::ValueId gate,g::ValueId up){
      gate=binary(c::BinaryKind::kMinimum,gate,number(10.F));
      up=binary(c::BinaryKind::kMaximum,binary(c::BinaryKind::kMinimum,up,number(10.F)),number(-10.F));
      return binary(c::BinaryKind::kMultiply,take(b.add(c::Unary{c::UnaryKind::kSilu},gate),"source F32 SiLU"),up);
    };
    const auto gate=projection(0,inputs[0]),up=projection(1,inputs[0]),act=limited(gate,up),down=projection(2,act);
    const auto route3=take(b.add(f::reshape({tokens,slots,1}),route),"post-down weights view");
    const auto weighted=binary(c::BinaryKind::kMultiply,down,route3);
    const auto combined=take(b.add(c::Reduce{.kind=c::ReductionKind::kSum,.axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"expert sum axis"),.accumulation_type=Type::kFloat32},weighted),"F32 expert sum");
    const auto a2=take(b.add(f::reshape({tokens,hidden}),inputs[0]),"shared A rows");
    const auto shared=projection(5,limited(projection(3,a2),projection(4,a2)));
    const auto final=binary(c::BinaryKind::kAdd,combined,shared);
    outputs={ids,route,gate,up,act,down,shared,final};
    for(auto output:outputs)check(b.publish(output),"publish GLM source boundary");
    graph=take(std::move(b).finish(),"ordinary GLM mixed full consumer");
    for(std::size_t i=1;i<outputs.size();++i)EXPECT_EQ(take(graph.value_description(outputs[i]),"mixed output inference").data_type(),Type::kFloat32);
  }
  std::size_t logical_bytes()const{std::size_t n=0;for(const auto &x:banks)n+=x.description.required_bytes().value();return n;}
  std::size_t capacity()const{std::size_t n=0;for(const auto &x:banks)n+=(x.description.required_bytes().value()+255)/256*256;return n;}
  std::size_t arena()const{std::size_t n=0;for(auto x:outputs)n+=(take(graph.value_description(x),"output extent").required_bytes().value()+255)/256*256;return n*4;}
};
inline base::Sources make_sources(unsigned epoch){
  base::Sources out;
  for(unsigned stage=0;stage<6;++stage)for(std::int64_t e=0;e<(stage<3?resident:1);++e){
    const auto payload=fragment(stage,stage<3?globals[std::size_t(e)]:0,epoch);
    for(unsigned port=0;port<3;++port)out.add(bank_id(stage,port),fragment_desc(stage,port),payload[port],std::size_t(e)*payload[port].size());
  }
  std::vector<std::int32_t> map(std::size_t(global_experts),-1);
  for(std::size_t e=0;e<globals.size();++e)map[std::size_t(globals[e])]=std::int32_t(e);
  out.add(map_id,dense(Type::kInt32,{global_experts}),base::raw(map),0);
  out.add(bias_id,dense(Type::kFloat32,{global_experts}),base::raw(correction()),0);
  std::reverse(out.slices.begin(),out.slices.end());
  EXPECT_EQ(out.slices.size(),92U);return out;
}
struct Episode {g::Invocation invocation;std::vector<c::ReadLease> inputs;std::vector<g::GraphOutput> outputs;std::vector<float> final;};
struct Metrics {double peak=0,l2=0;std::size_t metadata_counter=0,bf16_counter=0,softmax_counter=0;
  std::array<double,3> counter_peak{},counter_l2{};};
inline void compare(std::span<const float> actual,std::span<const float> expected,Metrics &mtr){
  ASSERT_EQ(actual.size(),expected.size());double peak=0,error=0,se=0,sr=0;
  for(std::size_t i=0;i<actual.size();++i){ASSERT_TRUE(std::isfinite(actual[i]));const double d=double(actual[i])-double(expected[i]);peak=std::max(peak,std::abs(double(expected[i])));error=std::max(error,std::abs(d));se+=d*d;sr+=double(expected[i])*double(expected[i]);}
  ASSERT_GT(peak,0.);ASSERT_GT(sr,0.);mtr.peak=std::max(mtr.peak,error/peak);mtr.l2=std::max(mtr.l2,std::sqrt(se/sr));
  EXPECT_LE(error/peak,2.e-5);EXPECT_LE(std::sqrt(se/sr),2.e-5);
}
inline std::size_t different(std::span<const float> a,std::span<const float> b){std::size_t n=0;for(std::size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;}
template<class Api> void expect_old(Api &api,const Episode &episode){
  EXPECT_EQ(f::read_result<float>(episode.outputs.back(),[&](m::ConstBufferView s,std::span<std::byte> d){return api.read(s,d);}),episode.final);
}
template<class Api,class Bind> Episode execute(Api &api,const Case &test,const g::Executable &executable,
    Bind bind,unsigned epoch,unsigned phase,c::LeasePool &pool,Metrics &metrics,bool absent=false){
  Episode out;out.invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"original GLM invocation");
  check(bind(out.invocation),"same-generation original banks and map");
  const std::array payload{base::raw(activation(phase)),base::raw(logits(phase,absent))};
  for(unsigned i=0;i<2;++i){out.inputs.push_back(f::upload(api.resource,payload[i],[&](auto d,auto s){return api.write(d,s);}));check(out.invocation.bind(test.inputs[i],test.input_desc[i],out.inputs.back()),"bind BF16 A/F32 logits");}
  for(std::size_t i=0;i<test.outputs.size();++i)if(take(executable.output_ownership(i),"result ownership")==g::OutputOwnership::kCaller){const auto desc=take(test.graph.value_description(test.outputs[i]),"published shape");check(out.invocation.bind(test.outputs[i],desc,take(pool.allocate(desc.required_bytes().value()),"original output writer")),"bind GLM output");}
  check(api.prepare(out.invocation),"prepare original GLM graph");auto done=take(out.invocation.submit(api.queues),"submit ordinary GLM MoE");const auto status=done.wait();
  if(absent){EXPECT_FALSE(status.ok());for(auto value:test.outputs)EXPECT_FALSE(out.invocation.result(value));return out;}
  check(status,"GLM source-bound completion");const auto expected=oracle(epoch,phase);
  for(auto value:test.outputs)out.outputs.push_back(take(out.invocation.result(value),"retain GLM boundary"));
  EXPECT_EQ(f::read_result<std::int32_t>(out.outputs[0],[&](m::ConstBufferView s,std::span<std::byte> d){return api.read(s,d);}),expected.ids);
  for(std::size_t i=1;i<out.outputs.size();++i){auto actual=f::read_result<float>(out.outputs[i],[&](m::ConstBufferView s,std::span<std::byte> d){return api.read(s,d);});compare(actual,expected.values[i-1],metrics);if(i+1==out.outputs.size())out.final=std::move(actual);}
  EXPECT_TRUE(out.invocation.reset().IsUnavailable());return out;
}
inline void counters(Metrics &metrics){
  const auto exact=oracle(0,0);
  const std::array wrong{oracle(0,0,true),oracle(0,0,false,true),oracle(0,0,false,false,true)};
  metrics.metadata_counter=different(exact.values.back(),wrong[0].values.back());
  metrics.bf16_counter=different(exact.values.back(),wrong[1].values.back());
  metrics.softmax_counter=different(exact.values.back(),wrong[2].values.back());
  EXPECT_GT(metrics.metadata_counter,0U);EXPECT_GT(metrics.bf16_counter,0U);EXPECT_GT(metrics.softmax_counter,0U);
  for(std::size_t mode=0;mode<wrong.size();++mode){
    double peak=0,error=0,se=0,sr=0;
    for(std::size_t i=0;i<exact.values.back().size();++i){
      const double a=exact.values.back()[i],d=double(wrong[mode].values.back()[i])-a;
      peak=std::max(peak,std::abs(a));error=std::max(error,std::abs(d));se+=d*d;sr+=a*a;
    }
    metrics.counter_peak[mode]=error/peak;metrics.counter_l2[mode]=std::sqrt(se/sr);
    EXPECT_GT(std::max(metrics.counter_peak[mode],metrics.counter_l2[mode]),2.e-5);
  }
}
inline void record(const Metrics &metrics,bool gpu){
  ::testing::Test::RecordProperty("successful_submissions",4);::testing::Test::RecordProperty("expected_failed_submissions",1);
  ::testing::Test::RecordProperty("successful_weight_generations",3);::testing::Test::RecordProperty("native_weight_slots",gpu?2:0);
  ::testing::Test::RecordProperty("original_fragments_per_generation",92);
  ::testing::Test::RecordProperty("metadata_bf16_counterfactual_words",std::to_string(metrics.metadata_counter));
  ::testing::Test::RecordProperty("early_bf16_counterfactual_words",std::to_string(metrics.bf16_counter));
  ::testing::Test::RecordProperty("softmax_counterfactual_words",std::to_string(metrics.softmax_counter));
  const auto precise=[](double x){std::ostringstream s;s<<std::setprecision(9)<<x;return s.str();};
  ::testing::Test::RecordProperty("maximum_peak_normalized_error",precise(metrics.peak));
  ::testing::Test::RecordProperty("maximum_relative_l2_error",precise(metrics.l2));
  for(std::size_t i=0;i<metrics.counter_peak.size();++i){
    ::testing::Test::RecordProperty("wrong_policy"+std::to_string(i)+"_peak_error",precise(metrics.counter_peak[i]));
    ::testing::Test::RecordProperty("wrong_policy"+std::to_string(i)+"_relative_l2",precise(metrics.counter_l2[i]));
  }
  ::testing::Test::RecordProperty("scope","GLM32 ordinary BF16A/F16metadata g64 routed4/4/5+shared6;D4096/H2048,E288Top8,resident9;synthetic stored-format bytes,no checkpoint/model/quality/performance claim");
}
template<class Api> void run_gpu(Api &api){
  Case test;Metrics metrics;counters(metrics);auto compiled=take(api.compile(test.graph),"compile complete mixed GLM consumer");
  auto slots_owner=take(api.create_slots(test.banks,test.capacity()),"two original slots,max_fragments128");
  auto pool=take(c::LeasePool::create(api.resource,test.arena(),{.max_blocks=40}),"finite GLM outputs");
  std::array<typename Api::Batch,2> batches;std::array<Episode,3> episodes;
  for(unsigned epoch=0;epoch<2;++epoch){auto source=make_sources(epoch);batches[epoch]=take(api.upload(slots_owner,epoch,source.slices),"original independent MLX fragment upload");base::expect_report(batches[epoch],test.logical_bytes(),test.capacity());source=base::Sources{};episodes[epoch]=execute(api,test,compiled.executable,[&](auto &inv){return api.bind(inv,test.bindings,batches[epoch]);},epoch,epoch,pool,metrics);}
  expect_old(api,episodes[0]);expect_old(api,episodes[1]);
  auto third=make_sources(2);EXPECT_FALSE(api.upload(slots_owner,0,third.slices));batches[0]={};EXPECT_FALSE(api.upload(slots_owner,0,third.slices));episodes[0]=Episode{};
  batches[0]=take(api.upload(slots_owner,0,third.slices),"reuse after original owner release");base::expect_report(batches[0],test.logical_bytes(),test.capacity());third=base::Sources{};
  episodes[2]=execute(api,test,compiled.executable,[&](auto &inv){return api.bind(inv,test.bindings,batches[0]);},2,2,pool,metrics);expect_old(api,episodes[1]);
  auto failed=execute(api,test,compiled.executable,[&](auto &inv){return api.bind(inv,test.bindings,batches[0]);},2,2,pool,metrics,true);failed=Episode{};
  auto recovered=execute(api,test,compiled.executable,[&](auto &inv){return api.bind(inv,test.bindings,batches[0]);},2,3,pool,metrics);
  EXPECT_NE(episodes[1].final,episodes[2].final);batches={};compiled={};for(auto &e:episodes)e.invocation={};recovered.invocation={};pool={};check(slots_owner.close(),"close retired original uploads");slots_owner={};
  expect_old(api,episodes[1]);expect_old(api,episodes[2]);expect_old(api,recovered);record(metrics,true);
}
template<class Api> void run_cpu(Api &api){
  Case test;Metrics metrics;counters(metrics);auto compiled=take(api.compile(test.graph),"compile same CPU GLM graph");
  auto pool=take(c::LeasePool::create(api.resource,test.arena(),{.max_blocks=40}),"host original output pool");
  std::array<Episode,3> episodes;std::vector<c::ReadLease> weights;
  const auto bind=[&](auto &inv){for(std::size_t i=0;i<weights.size();++i){const auto s=inv.bind(test.bindings[i].value,test.banks[i].description,weights[i]);if(!s.ok())return s;}return hyle::Status::Ok();};
  for(unsigned epoch=0;epoch<3;++epoch){
    auto source=make_sources(epoch);weights.clear();
    for(const auto &bank:test.banks){std::vector<std::byte> bytes(bank.description.required_bytes().value());for(const auto &slice:source.slices)if(slice.id==bank.id){const auto n=slice.source.description().required_bytes().value();auto view=take(slice.source.storage().subview(0,n),"original host fragment span");check(view.read(std::span(bytes).subspan(slice.destination_offset,n)),"explicit cold CPU expert assembly");}weights.push_back(f::upload(api.resource,bytes,[&](auto d,auto s){return api.write(d,s);}));}
    source=base::Sources{};episodes[epoch]=execute(api,test,compiled.executable,bind,epoch,epoch,pool,metrics);if(epoch)expect_old(api,episodes[epoch-1]);
  }
  auto failed=execute(api,test,compiled.executable,bind,2,2,pool,metrics,true);failed=Episode{};
  auto recovered=execute(api,test,compiled.executable,bind,2,3,pool,metrics);
  weights.clear();compiled={};pool={};for(auto &e:episodes)e.invocation={};recovered.invocation={};for(const auto &e:episodes)expect_old(api,e);expect_old(api,recovered);record(metrics,false);
}
} // namespace glm_flash_mixed_mlx_test
