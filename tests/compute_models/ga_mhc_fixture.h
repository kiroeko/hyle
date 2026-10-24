#pragma once

#include "ga_indexer_state_fixture.h"
#include "ga_mhc_reference_fixture.h"

// Formal GA HC4 typed roles. Attention/FFN computation is an explicit BF16
// branch boundary, not an invented miniature attention or expert engine.
namespace ga_mhc_test {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace f=ga_indexer_state_test;
namespace ref=ga_mhc_reference_test;
namespace scalar=nvfp4_reference_test;
using f::take;
using f::check;
using f::dense;
using f::reshape;
using f::Type;

struct Program {
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation slot;
  std::vector<c::TensorDesc> inputs_desc,constants_desc;
  std::vector<g::ValueId> inputs,constants,outputs;
  bool audit=false;
};
struct Formula {
  g::GraphBuilder &builder;
  template<class Op,class... Args> g::ValueId one(Op op,Args... inputs) {
    return take(builder.add(op,inputs...),"ordinary mHC formula node");
  }
  g::ValueId number(float value) {
    return one(c::Fill{.output_shape=take(c::TensorShape::create(std::array<std::int64_t,1>{1}),"literal dimensions"),
        .output_type=Type::kFloat32,.value=c::ScalarValue(value)});
  }
  g::ValueId binary(c::BinaryKind kind,g::ValueId a,g::ValueId b) {return one(c::Binary{kind},a,b);}
  g::ValueId slice(g::ValueId value,int axis,std::int64_t first,std::int64_t count) {
    return one(take(c::View::slice(axis,first,count),"mHC slice"),value);
  }
  g::ValueId inverse_rms(g::ValueId flat) {
    const auto square=binary(c::BinaryKind::kMultiply,flat,flat);
    const auto mean=one(c::Reduce{.kind=c::ReductionKind::kMean,
        .axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"mHC norm axis"),
        .keep_dimensions=true,.accumulation_type=Type::kFloat32},square);
    return one(c::Unary{c::UnaryKind::kRsqrt},binary(c::BinaryKind::kAdd,mean,number(1.e-6F)));
  }
  g::ValueId linear(g::ValueId flat,g::ValueId fn) {
    const auto transpose=one(take(c::View::permute(std::array<std::int32_t,2>{1,0}),"F32 model projection transpose"),fn);
    return one(c::MatMul{.accumulation_type=Type::kFloat32},flat,transpose);
  }
  g::ValueId gate(g::ValueId mixes,g::ValueId scale,g::ValueId base,int offset,bool post) {
    const auto affine=binary(c::BinaryKind::kAdd,
        binary(c::BinaryKind::kMultiply,slice(mixes,1,offset,4),slice(scale,0,offset/4,1)),
        slice(base,0,offset,4));
    const auto sigmoid=one(c::Unary{c::UnaryKind::kSigmoid},affine);
    return post?binary(c::BinaryKind::kMultiply,sigmoid,number(2.F))
               :binary(c::BinaryKind::kAdd,sigmoid,number(1.e-6F));
  }
};

inline Program build_pre(std::int64_t tokens,bool audit) {
  constexpr std::int64_t depth=128;const auto rows=2*tokens;
  Program p;p.audit=audit;
  p.inputs_desc={dense(Type::kBFloat16,{2,tokens,4,depth})};
  p.constants_desc={dense(Type::kFloat32,{24,4*depth}),dense(Type::kFloat32,{3}),dense(Type::kFloat32,{24})};
  auto b=take(g::GraphBuilder::create(),"GA mHC pre builder");Formula x{b};
  for(const auto &d:p.inputs_desc)p.inputs.push_back(take(b.input(d),"BF16 stream boundary"));
  for(const auto &d:p.constants_desc)p.constants.push_back(take(b.constant(d),"original F32 HC parameter"));
  const auto wide=x.one(c::Cast{Type::kFloat32},p.inputs[0]);
  const auto flat=x.one(reshape({rows,4*depth}),wide);
  const auto factor=x.inverse_rms(flat),linear=x.linear(flat,p.constants[0]);
  // Preserve F.linear(raw F32 x, fn) * rsqrt. Moving normalization before
  // the projection changes the source's finite F32 arithmetic expression.
  const auto mixes=x.binary(c::BinaryKind::kMultiply,linear,factor);
  const auto pre=x.gate(mixes,p.constants[1],p.constants[2],0,false);
  const auto post=x.gate(mixes,p.constants[1],p.constants[2],4,true);
  const auto comb=x.one(c::model_ops::hyper_connection_combine(1.e-6F,20),mixes,p.constants[1],p.constants[2]);
  const auto collapsed_raw=x.one(c::model_ops::hyper_connection_pre(),x.one(reshape({rows,4,depth}),wide),pre);
  // Publish the material BF16 result. A final View over private workspace is
  // readable but deliberately cannot export an independent storage lease.
  const auto collapsed=x.one(c::Cast{Type::kBFloat16},x.one(reshape({2,tokens,depth}),collapsed_raw));
  p.outputs={collapsed,post,comb};
  if(audit){p.outputs.push_back(factor);p.outputs.push_back(linear);p.outputs.push_back(mixes);p.outputs.push_back(pre);p.outputs.push_back(collapsed_raw);}
  for(auto output:p.outputs)check(b.publish(output),"publish typed pre output");
  p.graph=take(std::move(b).finish(),"typed HC pre graph");return p;
}
inline Program build_post_head(std::int64_t tokens,bool audit) {
  constexpr std::int64_t depth=128;const auto rows=2*tokens;
  Program p;p.audit=audit;
  p.inputs_desc={dense(Type::kBFloat16,{2,tokens,depth}),dense(Type::kBFloat16,{2,tokens,4,depth}),
      dense(Type::kFloat32,{rows,4}),dense(Type::kFloat32,{rows,4,4})};
  p.constants_desc={dense(Type::kFloat32,{4,4*depth}),dense(Type::kFloat32,{1}),dense(Type::kFloat32,{4})};
  auto b=take(g::GraphBuilder::create(),"GA mHC post/head builder");Formula x{b};
  for(const auto &d:p.inputs_desc)p.inputs.push_back(take(b.input(d),"typed post/head input"));
  for(const auto &d:p.constants_desc)p.constants.push_back(take(b.constant(d),"original F32 head parameter"));
  const auto branch=x.one(reshape({rows,depth}),x.one(c::Cast{Type::kFloat32},p.inputs[0]));
  const auto residual=x.one(reshape({rows,4,depth}),x.one(c::Cast{Type::kFloat32},p.inputs[1]));
  const auto expanded_raw=x.one(c::model_ops::hyper_connection_post(),branch,residual,p.inputs[2],p.inputs[3]);
  const auto expanded=x.one(c::Cast{Type::kBFloat16},x.one(reshape({2,tokens,4,depth}),expanded_raw));
  // hc_head consumes the BF16 post result, not the unrounded F32 temporary.
  const auto head_state=x.one(c::Cast{Type::kFloat32},x.one(reshape({rows,4,depth}),expanded));
  const auto flat=x.one(reshape({rows,4*depth}),head_state);
  const auto factor=x.inverse_rms(flat),linear=x.linear(flat,p.constants[0]);
  const auto mixes=x.binary(c::BinaryKind::kMultiply,linear,factor);
  const auto affine=x.binary(c::BinaryKind::kAdd,x.binary(c::BinaryKind::kMultiply,mixes,p.constants[1]),p.constants[2]);
  const auto pre=x.binary(c::BinaryKind::kAdd,x.one(c::Unary{c::UnaryKind::kSigmoid},affine),x.number(1.e-6F));
  const auto head_raw=x.one(c::model_ops::hyper_connection_pre(),head_state,pre);
  const auto head=x.one(c::Cast{Type::kBFloat16},x.one(reshape({2,tokens,depth}),head_raw));
  p.outputs={expanded,head};
  if(audit)for(auto v:std::array{expanded_raw,factor,linear,mixes,pre,head_raw})p.outputs.push_back(v);
  for(auto output:p.outputs)check(b.publish(output),"publish typed post/head output");
  p.graph=take(std::move(b).finish(),"typed post/head graph");return p;
}

inline void bind_outputs(Program &p,c::LeasePool &pool) {
  for(std::size_t port=0;port<p.outputs.size();++port)
    if(take(p.compiled.executable.output_ownership(port),"HC output ownership")==g::OutputOwnership::kCaller){
      const auto desc=take(p.graph.value_description(p.outputs[port]),"HC output description");
      auto writer=take(pool.allocate(take(desc.required_bytes(),"HC output extent")),"independent output permission");
      check(p.slot.bind(p.outputs[port],desc,std::move(writer)),"bind original owned HC output");
    }
}
inline float near_float(std::span<const float> actual,std::span<const float> expected,float absolute,float relative){
  if(actual.size()!=expected.size())throw std::runtime_error("mHC F32 reference extent differs");
  float maximum=0;
  for(std::size_t i=0;i<actual.size();++i){
    const auto difference=std::abs(actual[i]-expected[i]);maximum=std::max(maximum,difference);
    EXPECT_NEAR(actual[i],expected[i],absolute+relative*std::abs(expected[i]))<<i;
  }
  return maximum;
}
inline int near_bfloat(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> expected){
  if(actual.size()!=expected.size())throw std::runtime_error("mHC BF16 reference extent differs");
  const auto order=[](std::uint16_t x){return x&0x8000?0x8000-int(x&0x7fff):0x8000+int(x);};
  int maximum=0;
  for(std::size_t i=0;i<actual.size();++i){
    const auto distance=std::abs(order(actual[i])-order(expected[i]));maximum=std::max(maximum,distance);
    EXPECT_LE(distance,ref::bfloat_ulp_tolerance)<<i<<": "<<scalar::wide(actual[i])<<" vs "<<scalar::wide(expected[i]);
  }
  return maximum;
}

template<class Compile,class Write,class Read,class Prepare>
void run_mhc(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  int submissions=0,max_bfloat_ulp=0;
  std::array<std::size_t,3> counterfactual_words{};
  float max_f32_error=0;
  // A caller-borrowed constant must outlive even the programs held by old
  // GraphOutputs. Owned constants additionally retain the original owner/lease.
  std::vector<c::ReadLease> retained_parameters;
  g::GraphOutput retained_collapsed,retained_head;
  std::vector<std::uint16_t> old_collapsed,old_head;
  c::ReadLease escaped_head;
  const auto upload=[&]<class T>(std::span<const T> data){return f::upload(resource,std::as_bytes(data),write,owner);};
  for(const std::int64_t tokens:{3,1}){
    SCOPED_TRACE(::testing::Message()<<"B2/S"<<tokens<<"/HC4/D128");
    const auto input0=ref::make_inputs(tokens,0),input1=ref::make_inputs(tokens,1);
    const auto expected0=ref::reference(input0);
    const std::array expected{expected0,ref::reference(input1,expected0.expanded_bf16,input1.branch_bf16)};
    const std::array witnesses{ref::counterfactuals(input0,expected[0]),ref::counterfactuals(input1,expected[1],expected[0].expanded_bf16,input1.branch_bf16)};
    for(const auto &w:witnesses){
      EXPECT_GT(w.transposed_comb_words,0U);EXPECT_GT(w.early_bfloat_pre_words,0U);EXPECT_GT(w.omitted_post_boundary_head_words,0U);
      counterfactual_words[0]+=w.transposed_comb_words;counterfactual_words[1]+=w.early_bfloat_pre_words;counterfactual_words[2]+=w.omitted_post_boundary_head_words;
    }
    const auto parameters=[&](bool head){
      return head?std::array{upload(std::span<const float>(input0.head_fn)),upload(std::span<const float>(input0.head_scale)),upload(std::span<const float>(input0.head_base))}
                 :std::array{upload(std::span<const float>(input0.fn)),upload(std::span<const float>(input0.scale)),upload(std::span<const float>(input0.base))};
    };
    const auto pre_parameters=parameters(false),head_parameters=parameters(true);
    retained_parameters.insert(retained_parameters.end(),pre_parameters.begin(),pre_parameters.end());
    retained_parameters.insert(retained_parameters.end(),head_parameters.begin(),head_parameters.end());
    std::array programs{build_pre(tokens,true),build_pre(tokens,false),build_post_head(tokens,true),build_post_head(tokens,false)};
    for(std::size_t i=0;i<programs.size();++i){
      auto &p=programs[i];const auto &leases=i<2?pre_parameters:head_parameters;
      std::vector<g::ConstantBinding> constants;
      for(unsigned j=0;j<3;++j)constants.push_back({p.constants[j],take(leases[j].tensor(p.constants_desc[j]),"constant F32 parameter"),g::ConstantPolicy::kBorrow,owner,owner?leases[j]:c::ReadLease{}});
      p.compiled=take(compile(p.graph,g::CompileOptions{.constants=constants}),"compile source typed HC role");
      p.slot=take(p.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"independent original HC generation");
    }
    // StateStore charges the complete retained arena, not just its subviews.
    // Reserve this finite journey's published outputs, including alignment.
    std::size_t arena_bytes=0;
    for(const auto &p:programs)for(auto output:p.outputs){
      const auto desc=take(p.graph.value_description(output),"HC arena output");
      const auto bytes=take(desc.required_bytes(),"HC arena extent");
      arena_bytes+=(bytes+255U)&~std::size_t(255U);
    }
    auto pool=take(c::LeasePool::create(resource,arena_bytes,{.max_blocks=128}),"HC owned output arena");
    const auto stream_desc=dense(Type::kBFloat16,{2,tokens,4,128}),collapse_desc=dense(Type::kBFloat16,{2,tokens,128});
    struct Geometry{std::uint32_t batches,tokens,depth,generation;};
    const std::array specs{c::StateEntrySpec{.description=stream_desc},c::StateEntrySpec{.description=collapse_desc},
        c::StateEntrySpec{.description=dense(Type::kFloat32,{2*tokens,4})},c::StateEntrySpec{.description=dense(Type::kFloat32,{2*tokens,4,4})},
        c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Geometry)}};
    auto schema=take(c::StateSchema::create(specs),"HC residual/coefficient schema");
    auto store=take(c::StateStore::create(schema,{.max_versions=8,.max_drafts=2,.max_retained_blocks=32,.max_backing_bytes=2U<<20}),"HC version store");
    std::array<c::StateEntryId,5> ids;
    for(unsigned i=0;i<ids.size();++i)ids[i]=take(schema.entry(i),"HC state identity");
    std::array<c::StateVersion,2> versions;
    c::StateHead head;
    auto stream=upload(std::span<const std::uint16_t>(input0.stream_bf16));
    for(unsigned generation=0;generation<2;++generation){
      const auto &e=expected[generation];auto &pre=programs[generation];auto &post=programs[2+generation];
      const auto branch=upload(std::span<const std::uint16_t>(generation?input1.branch_bf16:input0.branch_bf16));
      check(pre.slot.bind(pre.inputs[0],pre.inputs_desc[0],stream),"bind original BF16 stream");
      bind_outputs(pre,pool);check(prepare(pre.slot,queues),"prepare source HC pre");
      auto pre_done=take(pre.slot.submit(queues),"submit source HC pre");check(pre_done.wait(),"HC pre original completion");++submissions;
      std::vector<g::GraphOutput> pre_results;
      for(auto output:pre.outputs)pre_results.push_back(take(pre.slot.result(output),"owned HC pre result"));
      max_bfloat_ulp=std::max(max_bfloat_ulp,near_bfloat(f::read_result<std::uint16_t>(pre_results[0],read),e.collapsed_bf16));
      max_f32_error=std::max(max_f32_error,near_float(f::read_result<float>(pre_results[1],read),e.post,ref::absolute_tolerance,ref::relative_tolerance));
      max_f32_error=std::max(max_f32_error,near_float(f::read_result<float>(pre_results[2],read),e.comb,ref::absolute_tolerance,ref::relative_tolerance));
      if(pre.audit){
        const std::array<std::span<const float>,5> audit{e.norm_factor,e.linear_mixes,e.mixes,e.pre,e.collapsed_f32};
        for(unsigned i=0;i<audit.size();++i)max_f32_error=std::max(max_f32_error,near_float(f::read_result<float>(pre_results[3+i],read),audit[i],ref::absolute_tolerance,ref::relative_tolerance));
      }
      std::array<c::ReadLease,3> produced;
      for(unsigned i=0;i<3;++i)produced[i]=take(pre_results[i].storage_lease(),"actual HC pre storage lease");
      const Geometry geometry{2,std::uint32_t(tokens),128,generation};
      if(!generation){
        const std::array members{c::StateMember{ids[0],stream},c::StateMember{ids[1],produced[0]},c::StateMember{ids[2],produced[1]},
            c::StateMember{ids[3],produced[2]},c::StateMember{ids[4],{},std::as_bytes(std::span(&geometry,1))}};
        versions[0]=take(store.import(members),"atomic original residual/pre/post/comb state");
        head=take(c::StateHead::create(store,versions[0]),"HC state head");
      }else{
        auto draft=take(store.begin(versions[0]),"next HC coefficient generation");
        check(draft.replace(ids[0],stream),"replace original next residual");
        for(unsigned i=0;i<3;++i)check(draft.replace(ids[1+i],produced[i]),"replace actual next HC coefficients");
        check(draft.replace_metadata(ids[4],std::as_bytes(std::span(&geometry,1))),"replace HC host geometry");
        versions[1]=take(draft.finalize(),"finalize same-generation HC state");
        check(head.publish(versions[0].identity(),versions[1]),"publish only completed coefficient generation");
        EXPECT_FALSE(head.publish(versions[0].identity(),versions[0]).ok());
      }
      const auto &version=versions[generation];
      Geometry actual_geometry{};
      auto metadata=take(take(version.member(ids[4]),"same-generation host geometry").view(),"owned host geometry bytes");
      check(metadata.read(std::as_writable_bytes(std::span(&actual_geometry,1))),"read exact host snapshot");
      EXPECT_EQ(actual_geometry.batches,2U);EXPECT_EQ(actual_geometry.tokens,std::uint32_t(tokens));
      EXPECT_EQ(actual_geometry.depth,128U);EXPECT_EQ(actual_geometry.generation,generation);
      const std::array post_inputs{branch,take(version.member(ids[0]),"versioned residual"),take(version.member(ids[2]),"versioned post weights"),take(version.member(ids[3]),"versioned combination")};
      for(unsigned i=0;i<post_inputs.size();++i)check(post.slot.bind(post.inputs[i],post.inputs_desc[i],post_inputs[i]),"bind same-generation HC post role");
      bind_outputs(post,pool);check(prepare(post.slot,queues),"prepare source HC post/head");
      auto post_done=take(post.slot.submit(queues),"submit source HC post/head");check(post_done.wait(),"HC post original completion");++submissions;
      std::vector<g::GraphOutput> post_results;
      for(auto output:post.outputs)post_results.push_back(take(post.slot.result(output),"owned HC post/head result"));
      max_bfloat_ulp=std::max(max_bfloat_ulp,near_bfloat(f::read_result<std::uint16_t>(post_results[0],read),e.expanded_bf16));
      max_bfloat_ulp=std::max(max_bfloat_ulp,near_bfloat(f::read_result<std::uint16_t>(post_results[1],read),e.head_bf16));
      if(post.audit){
        const std::array<std::span<const float>,6> audit{e.post_raw_f32,e.head_norm_factor,e.head_linear_mixes,e.head_mixes,e.head_pre,e.head_raw_f32};
        for(unsigned i=0;i<audit.size();++i)max_f32_error=std::max(max_f32_error,near_float(f::read_result<float>(post_results[2+i],read),audit[i],ref::absolute_tolerance,ref::relative_tolerance));
      }
      stream=take(post_results[0].storage_lease(),"original BF16 post feeds next HC pre");
      if(!retained_head.valid()){
        retained_collapsed=pre_results[0];retained_head=post_results[1];escaped_head=take(retained_head.storage_lease(),"independent original HC head backing");
        old_collapsed=f::read_result<std::uint16_t>(retained_collapsed,read);old_head=f::read_result<std::uint16_t>(retained_head,read);
        EXPECT_TRUE(pre.slot.reset().IsUnavailable());EXPECT_TRUE(post.slot.reset().IsUnavailable());
      }
      EXPECT_EQ(f::read_result<std::uint16_t>(retained_collapsed,read),old_collapsed);
      EXPECT_EQ(f::read_result<std::uint16_t>(retained_head,read),old_head);
    }
    // The source's independent result owner, not a reconstructed host vector,
    // keeps the data alive after the original public producer handles close.
    stream={};versions={};head={};store={};schema={};programs={};pool={};
    EXPECT_EQ(f::read_result<std::uint16_t>(retained_head,read),old_head);
    EXPECT_EQ(f::read_result<std::uint16_t>(retained_collapsed,read),old_collapsed);
  }
  EXPECT_EQ(submissions,8);
  retained_collapsed.reset();retained_head.reset();
  auto lease_view=take(escaped_head.view(),"escaped actual head bytes");
  std::vector<std::uint16_t> escaped(old_head.size());
  check(read(lease_view,std::as_writable_bytes(std::span(escaped))),"read escaped HC head without original owners");
  EXPECT_EQ(escaped,old_head);
  ::testing::Test::RecordProperty("transposed_comb_counterfactual_words",std::to_string(counterfactual_words[0]));
  ::testing::Test::RecordProperty("early_bf16_pre_counterfactual_words",std::to_string(counterfactual_words[1]));
  ::testing::Test::RecordProperty("omitted_post_bf16_head_counterfactual_words",std::to_string(counterfactual_words[2]));
  ::testing::Test::RecordProperty("successful_submissions",submissions);
  ::testing::Test::RecordProperty("geometries","B2/S3 and S1/HC4/D128; two generations each");
  ::testing::Test::RecordProperty("maximum_bf16_ulp",max_bfloat_ulp);
  ::testing::Test::RecordProperty("maximum_f32_error",std::to_string(max_f32_error));
  ::testing::Test::RecordProperty("state_feedback","actual BF16 post GraphOutput.storage_lease; original residual/post/comb atomically versioned");
  ::testing::Test::RecordProperty("private_generations",2);
  ::testing::Test::RecordProperty("independent_head_lease","read after final GraphOutputs and original public producer handles were destroyed");
  ::testing::Test::RecordProperty("scope","source HC typed composition and original resources; projected BF16 branch boundary, not whole attention/expert/model");
}
} // namespace ga_mhc_test
