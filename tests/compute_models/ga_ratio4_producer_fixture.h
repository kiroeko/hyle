#pragma once

#include "ga_indexer_state_fixture.h"
#include "ga_ratio4_reference_fixture.h"
#include "batched_rope_fixture.h"

// Test consumer of ordinary public operations. Projected inputs and model
// parameters are explicit; no host readback participates in state feedback.
namespace ga_ratio4_producer_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace f = ga_indexer_state_test;
namespace ref = ga_ratio4_reference_test;
namespace scalar = nvfp4_reference_test;
using f::take;
using f::check;
using f::dense;
using f::Type;
using f::reshape;

struct Producer {
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation slot;
  std::vector<g::Invocation> generations;
  std::size_t next_generation=0;
  std::vector<c::TensorDesc> descriptions;
  std::vector<g::ValueId> inputs, outputs;
  bool emitted = false, used = false;
};
inline Producer build_producer(std::int64_t count, bool emit, bool expose_stages=true) {
  Producer p; p.emitted = emit;
  p.descriptions = {dense(Type::kFloat32,{2,8,256}), dense(Type::kFloat32,{2,8,256}),
      dense(Type::kFloat32,{2,count,256}), dense(Type::kFloat32,{2,count,256}),
      dense(Type::kFloat32,{4,256}), dense(Type::kInt32,{count}), dense(Type::kInt32,{count})};
  if (emit) {
    const std::array extra{dense(Type::kBFloat16,{2,3,128}), dense(Type::kFloat32,{128}),
        dense(Type::kInt32,{1}), dense(Type::kFloat32,{32}), dense(Type::kInt32,{1})};
    p.descriptions.insert(p.descriptions.end(), extra.begin(), extra.end());
  }
  auto b = take(g::GraphBuilder::create(), "ratio4 builder");
  for (const auto &desc : p.descriptions) p.inputs.push_back(take(b.input(desc), "ratio4 input"));
  const auto one = [&](auto operation, auto... args) {
    return take(b.add(operation,args...), "ratio4 ordinary operation");
  };
  const auto slice = [&](g::ValueId value, std::int32_t axis, std::int64_t start, std::int64_t length) {
    return one(take(c::View::slice(axis,start,length), "ratio4 slice descriptor"),value);
  };
  const auto selected_ape = one(c::Gather{.axis=0},p.inputs[4],p.inputs[5]);
  const auto score = one(c::Binary{c::BinaryKind::kAdd},p.inputs[3],selected_ape);
  const auto kv = one(c::Scatter{.axis=1},p.inputs[0],p.inputs[6],p.inputs[2]);
  const auto scores = one(c::Scatter{.axis=1},p.inputs[1],p.inputs[6],score);
  if (!emit) p.outputs = {kv,scores};
  else {
    const auto window = [&](g::ValueId value) {
      const auto previous = slice(slice(value,1,0,4),2,0,128);
      const auto current = slice(slice(value,1,4,4),2,128,128);
      return one(c::Concat{.axis=1},previous,current);
    };
    const auto values = window(kv), logits = window(scores);
    const auto probabilities = one(c::Softmax{.axis=1},logits);
    const auto product = one(c::Binary{c::BinaryKind::kMultiply},values,probabilities);
    const auto pooled = one(c::Reduce{.kind=c::ReductionKind::kSum,
        .axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"ratio4 reduction axis"),
        .accumulation_type=Type::kFloat32},product);
    const auto narrow = one(c::Cast{Type::kBFloat16},pooled);
    const auto norm = one(c::RmsNorm{.epsilon=1.e-6F},narrow,p.inputs[8]);
    const auto prefix = slice(norm,1,0,64);
    // Official complex rotation has a genuine F32 intermediate domain. A
    // BF16 RoPE operation would be a different low-precision expression.
    const auto tail_f32 = one(c::Cast{Type::kFloat32},slice(norm,1,64,64));
    const auto tail = one(reshape({2,1,64}),tail_f32);
    const auto rotated = one(c::RoPE{.rotary_dimensions=64,.sequence_axis=1,
        .layout=c::RopeLayout::kInterleaved},tail,p.inputs[9],p.inputs[10]);
    const auto tail_bf16 = one(reshape({2,64}),one(c::Cast{Type::kBFloat16},rotated));
    const auto rope = one(c::Concat{.axis=1},prefix,tail_bf16);
    const auto hadamard = one(c::Hadamard{},rope);
    const auto qat = one(c::model_ops::block_quantization_roundtrip(
        c::model_ops::BlockQuantizationFormat::kFloat4E2M1Block32),hadamard);
    const auto cache = one(c::Scatter{.axis=1},p.inputs[7],p.inputs[11],one(reshape({2,1,128}),qat));
    const auto current_kv = slice(kv,1,4,4), current_scores = slice(scores,1,4,4);
    // Roll all256 channels, retaining the already-APE-added scores exactly.
    p.outputs = {one(c::Concat{.axis=1},current_kv,current_kv),
        one(c::Concat{.axis=1},current_scores,current_scores),cache,
        pooled,narrow,norm,rope,hadamard,qat};
  }
  if(emit && !expose_stages)p.outputs.resize(3);
  for (auto value : p.outputs) check(b.publish(value),"publish ratio4 result");
  p.graph = take(std::move(b).finish(),"ratio4 graph");
  return p;
}

inline Producer build_consumer(int active) {
  Producer p;
  if(!active){
    p.descriptions={dense(Type::kFloat32,{2,1,0})};
    auto b=take(g::GraphBuilder::create(),"first-empty indexer builder");
    p.inputs={take(b.input(p.descriptions[0]),"first-empty score input")};
    const auto node=take(b.add_node(c::Operation(c::TopK{.count=0}),p.inputs),"first-empty TopK");
    p.outputs={take(b.result(node,0),"first-empty selected values"),take(b.result(node,1),"first-empty ids")};
    for(auto output:p.outputs)check(b.publish(output),"publish first-empty selection");
    p.graph=take(std::move(b).finish(),"first-empty indexer graph");
    return p;
  }
  p.descriptions={dense(Type::kBFloat16,{2,1,64,128}),
      take(c::TensorDesc::strided_bytes(Type::kBFloat16,std::array<std::int64_t,3>{2,active,128},
          std::array<std::int64_t,3>{3*128*2,128*2,2}),"produced active cache prefix"),
      dense(Type::kBFloat16,{2,1,64}),
      take(c::TensorDesc::strided_bytes(Type::kFloat16,std::array<std::int64_t,3>{2,1,active},
          std::array<std::int64_t,3>{3*2,3*2,2}),"produced causal prefix"),
      dense(Type::kInt32,{2,3})};
  auto b=take(g::GraphBuilder::create(),"produced-cache score builder");
  for(const auto &desc:p.descriptions)p.inputs.push_back(take(b.input(desc),"produced-cache original input"));
  const auto one=[&](auto op,auto... inputs){return take(b.add(op,inputs...),"produced-cache operation");};
  const auto kt=one(take(c::View::permute(std::array<std::int32_t,4>{0,1,3,2}),"produced key permutation"),
      one(reshape({2,1,active,128}),p.inputs[1]));
  const auto dot=one(c::MatMul{.accumulation_type=Type::kFloat32},p.inputs[0],kt);
  const auto positive=one(c::Unary{c::UnaryKind::kRelu},dot);
  const auto weighted=one(c::Binary{c::BinaryKind::kMultiply},positive,one(reshape({2,1,64,1}),p.inputs[2]));
  const auto sum=one(c::Reduce{.kind=c::ReductionKind::kSum,
      .axes=take(c::Axes::from(std::array<std::int32_t,1>{2}),"produced head axis"),
      .accumulation_type=Type::kFloat32},weighted);
  const auto scores=one(c::Binary{c::BinaryKind::kAdd},sum,one(c::Cast{Type::kBFloat16},p.inputs[3]));
  const auto selection=take(b.add_node(c::Operation(c::TopK{.count=std::size_t(active)}),
      std::array{one(c::Cast{Type::kFloat32},scores)}),"produced cache selection");
  const auto values=take(b.result(selection,0),"produced selected scores"),local=take(b.result(selection,1),"produced local ids");
  const auto mapped=one(c::model_ops::gather_elements(),p.inputs[4],one(reshape({2,active}),local));
  p.outputs={scores,values,local,mapped};
  for(auto v:p.outputs)check(b.publish(v),"publish produced indexer output");
  p.graph=take(std::move(b).finish(),"actual produced-cache consumer");
  return p;
}

inline void bind_owned_outputs(const g::CompiledGraph &compiled, const g::Graph &graph,
    std::span<const g::ValueId> outputs, g::Invocation &slot, c::LeasePool &pool) {
  for (std::size_t port=0; port<outputs.size(); ++port)
    if (take(compiled.executable.output_ownership(port),"ratio4 output ownership")==g::OutputOwnership::kCaller) {
      const auto desc=take(graph.value_description(outputs[port]),"ratio4 result description");
      auto writer=take(pool.allocate(take(desc.required_bytes(),"ratio4 output bytes")),"fresh result writer");
      check(slot.bind(outputs[port],desc,std::move(writer)),"bind owned ratio4 output");
    }
}
template<class T,class Read>
std::vector<T> read_lease(const c::ReadLease &lease,const c::TensorDesc &desc,Read read) {
  auto tensor=take(lease.tensor(desc),"retained state tensor");
  std::vector<T> values(take(desc.element_count(),"retained state elements"));
  check(read(tensor.storage(),std::as_writable_bytes(std::span(values))),"read dense state for assertion only");
  return values;
}
inline void equal_f32(std::span<const float> actual,std::span<const float> expected) {
  ASSERT_EQ(actual.size(),expected.size());
  for (std::size_t i=0;i<actual.size();++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),std::bit_cast<std::uint32_t>(expected[i])) << i;
}
inline void near_f32(std::span<const float> actual,std::span<const float> expected,float absolute,float relative) {
  ASSERT_EQ(actual.size(),expected.size());
  for (std::size_t i=0;i<actual.size();++i)
    EXPECT_NEAR(actual[i],expected[i],absolute+relative*std::abs(expected[i])) << i;
}
inline int near_bf16(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> expected) {
  if(actual.size()!=expected.size())throw std::runtime_error("typed producer result has wrong length");
  const auto ordered=[](std::uint16_t word){return word&0x8000?0x8000-int(word&0x7fff):0x8000+int(word);};
  int maximum=0;
  for(std::size_t i=0;i<actual.size();++i){
    const auto distance=std::abs(ordered(actual[i])-ordered(expected[i]));
    maximum=std::max(maximum,distance);
    EXPECT_LE(distance,ref::intermediate_bfloat_ulp_tolerance) << "BF16 ULP at " << i << ": " << scalar::wide(actual[i]) << " vs " << scalar::wide(expected[i]);
  }
  return maximum;
}

// The production change for natural BSD data must preserve the existing
// multihead BSHD route, including a partial rotary depth and negative axis.
template<class Compile,class Write,class Read,class Prepare>
void run_rope_neighbor(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  const std::array descriptions{dense(Type::kFloat32,{2,3,2,8}),dense(Type::kInt32,{2,3}),dense(Type::kFloat32,{3,2})};
  auto b=take(g::GraphBuilder::create(),"existing multihead RoPE builder");
  std::array<g::ValueId,3> inputs;
  for(unsigned i=0;i<inputs.size();++i)inputs[i]=take(b.input(descriptions[i]),"multihead input");
  const auto output=take(b.add(c::RoPE{.rotary_dimensions=4,.sequence_axis=-3,
      .layout=c::RopeLayout::kInterleaved},inputs[0],inputs[1],inputs[2]),"original multihead RoPE");
  check(b.publish(output),"publish multihead output");
  auto graph=take(std::move(b).finish(),"existing multihead graph");
  auto compiled=take(compile(graph,g::CompileOptions{}),"compile existing multihead neighbor");
  auto slot=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"multihead invocation");
  const auto data=batched_rope_test::data(7);
  std::vector<float> source;source.reserve(data.size());
  for(const auto value:data)source.push_back(static_cast<float>(value));
  const auto expected=batched_rope_test::reference(data,batched_rope_test::kPositions,true,true,c::RopeLayout::kInterleaved);
  const std::array payloads{scalar::bytes<float>(source),scalar::bytes<std::int32_t>(batched_rope_test::kPositions),
      scalar::bytes<float>(batched_rope_test::kFrequencies)};
  for(unsigned i=0;i<inputs.size();++i)check(slot.bind(inputs[i],descriptions[i],f::upload(resource,payloads[i],write,owner)),"bind original multihead payload");
  auto pool=take(c::LeasePool::create(resource,4096),"multihead output backing");
  bind_owned_outputs(compiled,graph,std::array{output},slot,pool);
  check(prepare(slot,queues),"prepare multihead neighbor");
  auto done=take(slot.submit(queues),"submit multihead neighbor");check(done.wait(),"multihead completion");
  auto result=take(slot.result(output),"retained multihead result");
  slot={};compiled={};graph={};pool={};
  const auto actual=f::read_result<float>(result,read);
  ASSERT_EQ(actual.size(),expected.size());
  for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected[i],3.e-6+3.e-6*std::abs(expected[i]))<<i;
  ::testing::Test::RecordProperty("successful_submissions",1);
  ::testing::Test::RecordProperty("geometry","old B2/S3/H2/D8, rotary4, sequence_axis-3, batch positions and token frequencies");
}

template<class Compile,class Write,class Read,class Prepare>
void run_producer(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  const auto data=ref::make_inputs();
  ref::Reference oracle(data);
  std::array<ref::Snapshot,6> expected;
  expected[0]=oracle.advance(0,3);
  for (int i=1;i<6;++i) expected[i]=oracle.advance(i+2,1);
  EXPECT_GT(expected[1].minimum_fp4_midpoint_distance,.2F);
  EXPECT_GT(expected[5].minimum_fp4_midpoint_distance,.2F);
  EXPECT_GT(expected[1].norm_early_rounding_differences+expected[5].norm_early_rounding_differences,0U);
  std::array programs{build_producer(3,false),build_producer(1,false),build_producer(1,true),build_producer(1,true,false)};
  // Every retained StateVersion needs independent output backing. Some native
  // programs keep that backing fixed for one Invocation, while other sources
  // can accept fresh writers. Reserve this finite journey's generations cold
  // with the same public API on every backend; never overwrite an old reader.
  constexpr std::array<unsigned,4> generation_counts{1,6,2,1};
  for (std::size_t i=0;i<programs.size();++i) {
    auto &p=programs[i];
    p.compiled=take(compile(p.graph,g::CompileOptions{}),"compile original ratio4 producer");
    p.generations.reserve(generation_counts[i]);
    for(unsigned generation=0;generation<generation_counts[i];++generation)
      p.generations.push_back(take(p.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"cold original producer generation"));
  }
  std::array consumers{build_consumer(0),build_consumer(1),build_consumer(2)};
  for(auto &p:consumers){
    p.compiled=take(compile(p.graph,g::CompileOptions{}),"cold compile produced cache consumer");
    p.slot=take(p.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"produced cache consumer slot");
  }
  auto output_pool=take(c::LeasePool::create(resource,2U<<20,{.max_blocks=128}),"ratio4 output backing");
  const auto upload=[&]<class T>(std::span<const T> values) {
    return f::upload(resource,std::as_bytes(values),write,owner);
  };
  const auto kv_desc=dense(Type::kFloat32,{2,8,256}),cache_desc=dense(Type::kBFloat16,{2,3,128});
  const auto mask_desc=dense(Type::kFloat16,{2,1,3}),map_desc=dense(Type::kInt32,{2,3});
  struct Geometry { std::uint32_t tokens,active,capacity; };
  const std::array specs{c::StateEntrySpec{.description=kv_desc},c::StateEntrySpec{.description=kv_desc},
      c::StateEntrySpec{.description=cache_desc},c::StateEntrySpec{.description=mask_desc},
      c::StateEntrySpec{.description=map_desc},c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Geometry)}};
  auto schema=take(c::StateSchema::create(specs),"ratio4 state schema");
  auto store=take(c::StateStore::create(schema,{.max_versions=32,.max_drafts=2,.max_retained_blocks=128,
      .max_backing_bytes=8U<<20}),"ratio4 state store");
  std::array<c::StateEntryId,6> ids;
  for (unsigned i=0;i<ids.size();++i) ids[i]=take(schema.entry(i),"ratio4 state entry");
  std::vector<float> zeros(2*8*256),invalid(zeros.size(),-std::numeric_limits<float>::infinity());
  std::vector<std::uint16_t> initial_cache(2*3*128,scalar::bf16(-77.F)),initial_mask(6,0xfc00);
  std::vector<std::int32_t> initial_map(6,-1);
  const Geometry initial_geometry{0,0,3};
  const std::array members{c::StateMember{ids[0],upload(std::span<const float>(zeros))},
      c::StateMember{ids[1],upload(std::span<const float>(invalid))},
      c::StateMember{ids[2],upload(std::span<const std::uint16_t>(initial_cache))},
      c::StateMember{ids[3],upload(std::span<const std::uint16_t>(initial_mask))},
      c::StateMember{ids[4],upload(std::span<const std::int32_t>(initial_map))},
      c::StateMember{ids[5],{},std::as_bytes(std::span(&initial_geometry,1))}};
  auto version=take(store.import(members),"import original F32 window and BF16 cache");
  auto head=take(c::StateHead::create(store,version),"ratio4 head");
  const auto ape=upload(std::span<const float>(data.ape));
  const auto gamma=upload(std::span<const float>(data.gamma));
  const auto frequencies=upload(std::span<const float>(data.frequencies));
  std::array<c::StateVersion,6> main_versions;
  g::GraphOutput old_cache_result;
  c::ReadLease old_cache_lease,old_window_lease;
  std::vector<std::uint16_t> old_cache_values;
  std::vector<float> old_window_values;
  int successful_submissions=0,emit_count=0;
  bool accepted_continuation=false;
  std::array<int,4> maximum_bf16_ulp{};

  const auto consume=[&](const c::StateVersion &current,int active) {
    auto &p=consumers[std::size_t(active)];auto &slot=p.slot;
    if(p.used)check(slot.reset(),"reuse immutable produced cache consumer");
    p.used=true;
    if(!active){
      // Geometry tells us no compressed row exists. This ordinary empty
      // selection never builds an invalid zero-row QAT/MatMul branch.
      auto source=take(m::Buffer::create(resource,0,256),"first-empty source");
      check(slot.bind(p.inputs[0],take(c::ConstTensorView::create(p.descriptions[0],source.view()),"first-empty source view")),"bind first-empty score");
      std::array<m::Buffer,2> external;
      for(unsigned i=0;i<2;++i)if(take(p.compiled.executable.output_ownership(i),"first-empty ownership")==g::OutputOwnership::kCaller){
        external[i]=take(m::Buffer::create(resource,0,256),"first-empty output");
        check(slot.bind(p.outputs[i],take(c::TensorView::create(take(p.graph.value_description(p.outputs[i]),"first-empty description"),external[i].view()),"first-empty tensor")),"bind first-empty output");
      }
      check(prepare(slot,queues),"prepare first-empty indexer");
      auto done=take(slot.submit(queues),"submit first-empty indexer");check(done.wait(),"first-empty successful completion");
      ++successful_submissions;
      for(auto output:p.outputs)EXPECT_EQ(take(take(slot.result(output),"first-empty result").description(),"first-empty owned description").dimensions().back(),0);
      return;
    }
    std::vector<std::uint16_t> query(2*64*128);
    std::vector<std::uint16_t> weights(2*64);
    for(int batch=0;batch<2;++batch){
      query[std::size_t(batch*64*128)]=scalar::bf16(1.F);
      query[std::size_t(batch*64*128+32)]=scalar::bf16(.5F);
      query[std::size_t((batch*64+1)*128)]=scalar::bf16(-.5F);
      weights[std::size_t(batch*64)]=scalar::bf16(1.0078125F);
      weights[std::size_t(batch*64+1)]=scalar::bf16(.5F);
    }
    const std::array leases{upload(std::span<const std::uint16_t>(query)),
        take(current.member(ids[2]),"original produced cache lease"),
        upload(std::span<const std::uint16_t>(weights)),take(current.member(ids[3]),"original produced mask lease"),
        take(current.member(ids[4]),"original produced map lease")};
    for(unsigned i=0;i<leases.size();++i)check(slot.bind(p.inputs[i],p.descriptions[i],leases[i]),"bind produced original state resource");
    bind_owned_outputs(p.compiled,p.graph,p.outputs,slot,output_pool);
    check(prepare(slot,queues),"prepare actual produced cache consumer");
    auto done=take(slot.submit(queues),"submit actual produced cache consumer");check(done.wait(),"produced cache consumer completion");
    ++successful_submissions;
    const auto actual_scores=f::read_result<std::uint16_t>(take(slot.result(p.outputs[0]),"produced scores"),read);
    const auto actual_values=f::read_result<float>(take(slot.result(p.outputs[1]),"produced values"),read);
    const auto actual_ids=f::read_result<std::int32_t>(take(slot.result(p.outputs[2]),"produced local positions"),read);
    const auto actual_map=f::read_result<std::int32_t>(take(slot.result(p.outputs[3]),"produced original ids"),read);
    const auto &cache=expected[active==1?1:5].cache;
    for(int batch=0;batch<2;++batch) {
      std::vector<float> sorted;
      for(int row=0;row<active;++row) {
        const auto at=std::size_t((batch*3+row)*128);
        const auto d0=scalar::rounded_bf16(scalar::f32(double(scalar::wide(cache[at]))+.5*scalar::wide(cache[at+32])));
        const auto d1=scalar::rounded_bf16(scalar::multiply(-.5F,scalar::wide(cache[at])));
        const auto value=scalar::rounded_bf16(scalar::f32(double(scalar::rounded_bf16(
            scalar::multiply(std::max(0.F,d0),1.0078125F)))+scalar::rounded_bf16(scalar::multiply(std::max(0.F,d1),.5F))));
        EXPECT_EQ(actual_scores[std::size_t(batch*active+row)],scalar::bf16(value));
        sorted.push_back(value);
      }
      std::sort(sorted.begin(),sorted.end(),std::greater<float>{});
      std::vector<bool> seen(std::size_t(active),false);
      for(int i=0;i<active;++i) {
        const auto at=std::size_t(batch*active+i);const auto id=actual_ids[at];
        ASSERT_GE(id,0);ASSERT_LT(id,active);ASSERT_FALSE(seen[std::size_t(id)]);seen[std::size_t(id)]=true;
        EXPECT_EQ(actual_values[at],sorted[std::size_t(i)]);
        EXPECT_EQ(actual_values[at],scalar::wide(actual_scores[std::size_t(batch*active+id)]));
        EXPECT_EQ(actual_map[at],16777217+batch*1000+id);
      }
    }
  };

  const auto step=[&](int index,const c::StateVersion &base) {
    const int start=index?index+2:0,count=index?1:3;
    const auto &snapshot=expected[std::size_t(index)];
    auto &p=programs[snapshot.emitted?(accepted_continuation?3:2):index?1:0];
    if(p.next_generation>=p.generations.size())throw std::runtime_error("finite producer generation reservation exhausted");
    auto &slot=p.generations[p.next_generation++];
    const auto new_kv=ref::projected_slice(data.kv,start,count),new_gate=ref::projected_slice(data.gate,start,count);
    std::vector<std::int32_t> ape_rows(std::size_t(count),0),window_rows(std::size_t(count),0);
    for(int i=0;i<count;++i){ape_rows[std::size_t(i)]=(start+i)%4;window_rows[std::size_t(i)]=4+(start+i)%4;}
    std::vector<c::ReadLease> inputs{take(base.member(ids[0]),"previous original KV window"),
        take(base.member(ids[1]),"previous original score window"),upload(std::span<const float>(new_kv)),
        upload(std::span<const float>(new_gate)),ape,upload(std::span<const std::int32_t>(ape_rows)),
        upload(std::span<const std::int32_t>(window_rows))};
    if(p.emitted){
      const std::array<std::int32_t,1> position{start+1-4},cache_row{start/4};
      inputs.push_back(take(base.member(ids[2]),"previous original BF16 cache"));inputs.push_back(gamma);
      inputs.push_back(upload(std::span<const std::int32_t>(position)));inputs.push_back(frequencies);
      inputs.push_back(upload(std::span<const std::int32_t>(cache_row)));
    }
    for(std::size_t i=0;i<inputs.size();++i)
      check(slot.bind(p.inputs[i],p.descriptions[i],inputs[i]),"bind actual producer resource");
    bind_owned_outputs(p.compiled,p.graph,p.outputs,slot,output_pool);
    check(prepare(slot,queues),"prepare original producer invocation");
    auto done=take(slot.submit(queues),"submit original F32 overlap producer");check(done.wait(),"producer successful completion");
    ++successful_submissions;
    std::vector<g::GraphOutput> results;
    for(auto output:p.outputs)results.push_back(take(slot.result(output),"owned producer result"));
    equal_f32(f::read_result<float>(results[0],read),snapshot.kv_state);
    equal_f32(f::read_result<float>(results[1],read),snapshot.score_state);
    if(p.emitted) {
      EXPECT_EQ(f::read_result<std::uint16_t>(results[2],read),snapshot.cache);
      if(p.outputs.size()>3){
      near_f32(f::read_result<float>(results[3],read),snapshot.pooled_f32,ref::pool_absolute_tolerance,ref::pool_relative_tolerance);
      // Explicit producer stages expose the real typed boundaries. The
      // independent oracle does not prescribe a vendor reduction/trig tree.
      maximum_bf16_ulp[0]=std::max(maximum_bf16_ulp[0],near_bf16(f::read_result<std::uint16_t>(results[4],read),snapshot.pooled_bf16));
      maximum_bf16_ulp[1]=std::max(maximum_bf16_ulp[1],near_bf16(f::read_result<std::uint16_t>(results[5],read),snapshot.norm_bf16));
      maximum_bf16_ulp[2]=std::max(maximum_bf16_ulp[2],near_bf16(f::read_result<std::uint16_t>(results[6],read),snapshot.rope_bf16));
      maximum_bf16_ulp[3]=std::max(maximum_bf16_ulp[3],near_bf16(f::read_result<std::uint16_t>(results[7],read),snapshot.hadamard_bf16));
      EXPECT_EQ(f::read_result<std::uint16_t>(results[8],read),snapshot.qat_bf16);
      }
      ++emit_count;
    }
    auto draft=take(store.begin(base),"begin completed producer generation");
    for(unsigned i=0;i<2;++i)check(draft.replace(ids[i],take(results[i].storage_lease(),"actual F32 producer output lease")),"replace F32 state resource");
    if(p.emitted)check(draft.replace(ids[2],take(results[2].storage_lease(),"actual BF16 producer cache lease")),"replace actual BF16 cache");
    std::vector<std::uint16_t> mask(6,0xfc00);std::vector<std::int32_t> map(6,-1);
    for(int batch=0;batch<2;++batch)for(int row=0;row<int(snapshot.active_rows);++row){
      mask[std::size_t(batch*3+row)]=0;map[std::size_t(batch*3+row)]=16777217+batch*1000+row;
    }
    check(draft.replace(ids[3],upload(std::span<const std::uint16_t>(mask))),"replace same generation mask");
    check(draft.replace(ids[4],upload(std::span<const std::int32_t>(map))),"replace same generation integer map");
    const Geometry geometry{std::uint32_t(start+count),std::uint32_t(snapshot.active_rows),3};
    check(draft.replace_metadata(ids[5],std::as_bytes(std::span(&geometry,1))),"replace producer host geometry");
    auto next=take(draft.finalize(),"finalize actual producer generation");
    check(head.publish(base.identity(),next),"publish only successful actual producer state");
    EXPECT_FALSE(head.publish(base.identity(),base).ok());
    EXPECT_EQ(read_lease<std::uint16_t>(take(next.member(ids[2]),"completed producer cache"),cache_desc,read),snapshot.cache);
    if(p.emitted && !old_cache_result.valid()) {
      old_cache_result=results[2];old_cache_lease=take(results[2].storage_lease(),"retain first cache backing");
      old_cache_values=snapshot.cache;old_window_lease=take(results[0].storage_lease(),"retain first rolled F32 window");
      old_window_values=snapshot.kv_state;
      EXPECT_TRUE(slot.reset().IsUnavailable());
    }
    if(old_cache_result.valid()) {
      EXPECT_EQ(f::read_result<std::uint16_t>(old_cache_result,read),old_cache_values);
      equal_f32(read_lease<float>(old_window_lease,kv_desc,read),old_window_values);
    }
    return next;
  };
  for(int i=0;i<6;++i){version=step(i,version);main_versions[std::size_t(i)]=version;
    if(!i || expected[std::size_t(i)].emitted)consume(version,int(expected[std::size_t(i)].active_rows));}
  auto full=version;
  // Accept the first four tokens, restoring the complete original generation,
  // and continue with the same immutable plans. No state payload is reconstructed
  // on the host and no MTP acceptance policy is put into the library.
  check(head.publish(version.identity(),main_versions[1]),"publish accepted-prefix generation");
  version=main_versions[1];accepted_continuation=true;
  for(int i=2;i<6;++i)version=step(i,version);
  consume(version,2);
  EXPECT_FALSE(head.publish(full.identity(),full).ok());
  EXPECT_EQ(read_lease<std::uint16_t>(take(full.member(ids[2]),"retained main cache"),cache_desc,read),expected[5].cache);
  EXPECT_EQ(successful_submissions,14);EXPECT_EQ(emit_count,3);
  programs={};consumers={};main_versions={};version={};full={};head={};store={};schema={};output_pool={};
  EXPECT_EQ(f::read_result<std::uint16_t>(old_cache_result,read),old_cache_values);
  EXPECT_EQ(read_lease<std::uint16_t>(old_cache_lease,cache_desc,read),old_cache_values);
  equal_f32(read_lease<float>(old_window_lease,kv_desc,read),old_window_values);
  ::testing::Test::RecordProperty("minimum_fp4_midpoint_distance",std::to_string(std::min(expected[1].minimum_fp4_midpoint_distance,expected[5].minimum_fp4_midpoint_distance)));
  ::testing::Test::RecordProperty("norm_early_rounding_differences",int(expected[1].norm_early_rounding_differences+expected[5].norm_early_rounding_differences));
  ::testing::Test::RecordProperty("successful_submissions",successful_submissions);
  for(unsigned i=0;i<maximum_bf16_ulp.size();++i)
    ::testing::Test::RecordProperty("maximum_bf16_ulp_stage"+std::to_string(i),maximum_bf16_ulp[i]);
  ::testing::Test::RecordProperty("producer_submissions",10);
  ::testing::Test::RecordProperty("cold_producer_generations",10);
  ::testing::Test::RecordProperty("consumer_submissions",4);
  ::testing::Test::RecordProperty("compression_emissions",emit_count);
  ::testing::Test::RecordProperty("private_intermediate_emissions",1);
  ::testing::Test::RecordProperty("geometry","B2/D128/RD64; F32 window8x256; BF16 cache capacity3 active0->1->2");
  ::testing::Test::RecordProperty("state_feedback","original GraphOutput.storage_lease; no host payload feedback; accepted4 continuation");
  ::testing::Test::RecordProperty("scope","projected F32 KV/gate producer and typed BF16 score/selection consumer; not full model/projection/prefill8/throughput");
}
} // namespace ga_ratio4_producer_test
