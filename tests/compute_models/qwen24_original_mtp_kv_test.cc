#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/state_versions.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_Q24_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_Q24_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "d5_qwen35_gguf_fragment_support.h"
#include "d5_qwen38_24t_moe_profiles.h"
#include "qwen_original_mtp_roles.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace q24=qwen_original_mtp_roles::q24;
namespace numeric=d5_qwen35_gguf_fragment;
using Bytes=std::vector<std::byte>;
constexpr auto kHidden=q24::kHidden,kHeads=q24::kHeads,kDepth=q24::kDepth,kWidth=q24::kWidth;
constexpr std::int64_t kCapacity=6;
constexpr auto kProjectionBytes=q24::kProjectionBytes;
constexpr std::size_t kRowBytes=std::size_t(kWidth)*sizeof(float);
constexpr auto kEpsilon=q24::kEpsilon;
constexpr auto kKeySha=q24::kKeySha;
constexpr auto kValueSha=q24::kValueSha;
constexpr auto kNormSha=q24::kNormSha;

// Original blk92 K/V Q5_K and K-norm F32 from the selected MAIN UD-IQ4_XS
// shard00029-of00029, unsloth/Qwen3.8-2.4T-A95B-GGUF@567d3e6ac26c5474b18311e619c04350fb9a5556.
// Input is already-attention-normalized F32 H, not original embeddings or the
// 92-layer target/NextN decoder. The text-only MRoPE axes have equal positions:
// HalfSplit64, sections[11,11,10,0], theta1e7; K-norm epsilon1e-6, V unnormalized.
// The accepted count below is an explicit one-layer prefix selector, never a
// natural draft/target token comparison or a complete-model MTP claim.
template<class T> T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status,std::string_view where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
std::string directory() {
#if defined(_WIN32)
  char *value=nullptr;std::size_t size=0;
  if(_dupenv_s(&value,&size,"HYLE_QWEN24_MTP_KV_DIR")) throw std::runtime_error("cannot read Q24 original KV directory");
  const std::unique_ptr<char,decltype(&std::free)> owner(value,&std::free);
  return value?value:"";
#else
  const auto value=std::getenv("HYLE_QWEN24_MTP_KV_DIR");return value?value:"";
#endif
}
template<class T> Bytes bytes(std::span<const T> values) {
  const auto view=std::as_bytes(values);return {view.begin(),view.end()};
}
Bytes read(const c::ReadLease &lease) {
  Bytes output(lease.size());check(take(lease.view(),"readonly result view").read(output),"read actual retained bytes");return output;
}
std::vector<float> floats(const Bytes &source) {
  if(source.size()%sizeof(float)) throw std::runtime_error("F32 result extent");
  std::vector<float> result(source.size()/sizeof(float));std::memcpy(result.data(),source.data(),source.size());return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> values) {
  auto backing=take(m::Buffer::create(owner.ref(),values.size(),256),"real parameter/input backing");
  check(backing.view().write(values),"upload exact original or declared boundary bytes");
  return take(owner.adopt_buffer(std::move(backing)),"immutable original backing owner");
}
c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions,c::DataType type=c::DataType::kFloat32) {
  return take(c::TensorDesc::dense(type,{dimensions.begin(),dimensions.size()}),"logical tensor description");
}
c::TensorDesc cache_desc() {return dense({1,kCapacity,kHeads,kDepth});}
using Roles=q24::Data;
using q24::frequencies;
using q24::project;
using q24::rotate;
struct Logical final {
  g::Graph graph;
  int count,first_slot;
  std::array<g::ValueId,4> inputs,parameters;
  std::array<c::TensorDesc,4> input_desc,parameter_desc;
  // raw K, raw V, normalized K, rotated K, then one full K/V pair per prefix.
  std::vector<g::ValueId> outputs;
  std::vector<c::TensorDesc> output_desc;
};
Logical build(int count,int first_slot) {
  if(count<=0 || first_slot<0 || first_slot+count>kCapacity) throw std::runtime_error("finite KV append interval");
  Logical result{.count=count,.first_slot=first_slot};
  result.input_desc={dense({count,kHidden}),dense({count},c::DataType::kInt32),cache_desc(),cache_desc()};
  const auto packed=take(c::gguf_weight_description(c::GgufQuantizedType::kQ5K,
      c::DataType::kFloat32,kHidden,kWidth),"original packed Q5_K geometry");
  if(take(packed.required_bytes(),"original packed extent")!=kProjectionBytes) throw std::runtime_error("Q24 role byte extent");
  result.parameter_desc={packed,packed,dense({kDepth}),dense({32})};
  auto builder=take(g::GraphBuilder::create(),"ordinary original K/V Graph");
  for(std::size_t i=0;i<result.inputs.size();++i) result.inputs[i]=take(builder.input(result.input_desc[i]),"actual H/position/old KV port");
  for(std::size_t i=0;i<result.parameters.size();++i) result.parameters[i]=take(builder.constant(result.parameter_desc[i]),"immutable original parameter");
  const auto key=take(builder.add(c::MatMul{},result.inputs[0],result.parameters[0]),"original packed K projection");
  const auto value=take(builder.add(c::MatMul{},result.inputs[0],result.parameters[1]),"original packed V projection");
  const auto shape=take(c::TensorShape::create(std::array<std::int64_t,4>{1,count,kHeads,kDepth}),"K/V head shape");
  const auto heads=take(c::View::reshape(shape),"K/V head reshape");
  const auto kh=take(builder.add(heads,key),"actual K heads");
  const auto vh=take(builder.add(heads,value),"actual V heads");
  const auto normalized=take(builder.add(c::RmsNorm{.epsilon=kEpsilon},kh,result.parameters[2]),"original per-head K norm");
  const auto rotated=take(builder.add(c::RoPE{.rotary_dimensions=64,.sequence_axis=1,
      .layout=c::RopeLayout::kHalfSplit},normalized,result.inputs[1],result.parameters[3]),"text-position HalfSplit64 RoPE");
  result.outputs={key,value,normalized,rotated};
  result.output_desc={dense({count,kWidth}),dense({count,kWidth}),dense({1,count,kHeads,kDepth}),dense({1,count,kHeads,kDepth})};
  const auto slice=[&](g::ValueId source,int first,int rows) {
    return take(builder.add(take(c::View::slice(1,first,rows),"KV row slice"),source),"KV slice");
  };
  const auto concat=[&](g::ValueId a,g::ValueId b) {return take(builder.add(c::Concat{.axis=1},a,b),"actual KV concat");};
  const auto replace=[&](g::ValueId before,g::ValueId row,int slot) {
    auto updated=row;
    if(slot) updated=concat(slice(before,0,slot),updated);
    if(slot+1<kCapacity) updated=concat(updated,slice(before,slot+1,int(kCapacity)-slot-1));
    return updated;
  };
  auto keys=result.inputs[2],values=result.inputs[3];
  for(int row=0;row<count;++row) {
    keys=replace(keys,slice(rotated,row,1),first_slot+row);
    values=replace(values,slice(vh,row,1),first_slot+row);
    result.outputs.push_back(keys);result.outputs.push_back(values);
    result.output_desc.push_back(cache_desc());result.output_desc.push_back(cache_desc());
  }
  for(const auto output:result.outputs) check(builder.publish(output),"publish real component/prefix output");
  result.graph=take(std::move(builder).finish(),"finish original KV prefix Graph");return result;
}
struct Prepared final {Logical logical;g::CompiledGraph compiled;g::Invocation invocation;};
Prepared prepare(Logical logical,c::Context &context,const c::StorageOwner &owner,bool cuda,
                 const q24::Roles &parameters) {
  std::array<g::ConstantBinding,4> bindings;
  for(std::size_t i=0;i<bindings.size();++i)
    bindings[i]=take(parameters[i].binding(logical.parameters[i]),"share original immutable parameter owner");
  const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(
      d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile),"selected primary main-GGUF identity"),
      c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,0x5132344f5249474bULL,1}};
  const g::CompileOptions options{.constants=bindings,.workload=workload};
  const auto compile=[&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_Q24_CPU)
    if(!cuda) return g::cpu::compile(logical.graph,context,options);
#endif
#if defined(HYLE_TEST_Q24_CUDA)
    if(cuda) return g::cuda::compile(logical.graph,context,owner,options);
#endif
    throw std::runtime_error("requested original KV backend was not enabled");
  };
  auto compiled=take(compile(),"ordinary original K/V compile");
  for(std::size_t i=0;i<logical.outputs.size();++i)
    if(take(compiled.executable.output_ownership(i),"prefix output ownership")!=g::OutputOwnership::kCaller)
      throw std::runtime_error("KV prefix fixture requires caller outputs");
  auto invocation=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"original KV Invocation");
  return {std::move(logical),std::move(compiled),std::move(invocation)};
}
struct Boundary final {std::uint32_t valid_rows;std::int32_t next_position;};
static_assert(sizeof(Boundary)==8);
struct KvState final {
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId,3> entries;
  static KvState create(const c::StorageOwner &owner) {
    const std::array specs{c::StateEntrySpec{.description=cache_desc(),.resource=owner.ref()},
        c::StateEntrySpec{.description=cache_desc(),.resource=owner.ref()},
        c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Boundary)}};
    KvState result;
    result.schema=take(c::StateSchema::create(specs),"bounded complete KV schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=16,.max_drafts=2,
        .max_retained_blocks=64,.max_backing_bytes=512u<<10}),"KV immutable version store");
    for(std::size_t i=0;i<result.entries.size();++i) result.entries[i]=take(result.schema.entry(i),"real KV schema member");
    return result;
  }
  Boundary boundary(const c::StateVersion &version) const {
    const auto member=take(version.member(entries[2]),"actual committed KV boundary");
    Boundary result{};check(take(member.view(),"host protocol bytes").read(std::as_writable_bytes(std::span(&result,1))),"read committed boundary");
    if(result.valid_rows>kCapacity || result.next_position<0) throw std::runtime_error("invalid committed KV boundary");
    return result;
  }
  c::StateVersion import(const c::ReadLease &key,const c::ReadLease &value,Boundary boundary) {
    const std::array members{c::StateMember{entries[0],key},c::StateMember{entries[1],value},
        c::StateMember{entries[2],{},std::as_bytes(std::span(&boundary,1))}};
    return take(store.import(members),"import actual complete KV boundary");
  }
  c::StateVersion prefix(const c::StateVersion &from,const c::ReadLease &key,const c::ReadLease &value,int consumed) {
    auto facts=boundary(from);
    if(consumed<=0 || facts.valid_rows+std::uint32_t(consumed)>kCapacity ||
        facts.next_position>std::numeric_limits<std::int32_t>::max()-consumed)
      throw std::runtime_error("finite accepted KV prefix");
    facts.valid_rows+=std::uint32_t(consumed);facts.next_position+=consumed;
    auto draft=take(store.begin(from),"original atomic KV draft");
    check(draft.replace(entries[0],key),"stage complete K");check(draft.replace(entries[1],value),"stage complete V");
    check(draft.replace_metadata(entries[2],std::as_bytes(std::span(&facts,1))),"stage matching absolute position");
    return take(draft.finalize(),"finalize complete KV candidate");
  }
};
using Image=std::array<Bytes,3>;
Image image(const KvState &state,const c::StateVersion &version) {
  Image result;for(std::size_t i=0;i<result.size();++i) result[i]=read(take(version.member(state.entries[i]),"retained whole state member"));return result;
}
std::vector<float> hidden(int seed) {
  std::vector<float> result(kHidden);
  for(std::size_t i=0;i<result.size();++i) result[i]=float(int((i*37+std::size_t(seed)*19)%97)-48)/32.F+float(i%7)/32768.F;
  return result;
}
std::vector<c::ReadLease> execute(Prepared &program,c::Queue &queue,const c::StorageOwner &owner,
    c::LeasePool &pool,const KvState &state,const c::StateVersion &from,std::span<const float> activation) {
  const auto facts=state.boundary(from);
  if(facts.valid_rows!=std::uint32_t(program.logical.first_slot) || activation.size()!=std::size_t(program.logical.count*kHidden))
    throw std::runtime_error("actual state/normalized-H specialization mismatch");
  std::vector<std::int32_t> positions(std::size_t(program.logical.count),0);
  for(std::size_t i=0;i<positions.size();++i) positions[i]=facts.next_position+std::int32_t(i);
  const std::array inputs{upload(owner,bytes<float>(activation)),upload(owner,bytes<std::int32_t>(positions)),
      take(from.member(state.entries[0]),"actual old K permission"),take(from.member(state.entries[1]),"actual old V permission")};
  for(std::size_t i=0;i<inputs.size();++i) check(program.invocation.bind(program.logical.inputs[i],program.logical.input_desc[i],inputs[i]),"bind actual state and input");
  for(std::size_t i=0;i<program.logical.outputs.size();++i)
    check(program.invocation.bind(program.logical.outputs[i],program.logical.output_desc[i],take(pool.allocate(
        take(program.logical.output_desc[i].required_bytes(),"real prefix output extent")),"new result permission")),"bind actual output");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  check(program.invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare whole K/V Graph");
  {auto done=take(program.invocation.submit(queues),"strict K/V Graph submit");check(done.wait(),"actual whole K/V retirement");}
  std::vector<c::ReadLease> output;
  for(const auto value:program.logical.outputs) {auto result=take(program.invocation.result(value),"successful prefix result");output.push_back(take(result.storage_lease(),"independent prefix backing"));}
  check(program.invocation.reset(),"withdraw completed permissions");
  if(program.invocation.submit(queues)) throw std::runtime_error("reset KV graph remained prepared");
  return output;
}
void quantized_gate(std::span<const float> actual,std::span<const float> expected,const char *label) {
  SCOPED_TRACE(label);ASSERT_EQ(actual.size(),expected.size());
  double square=0,maximum=0;
  for(std::size_t i=0;i<actual.size();++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));ASSERT_TRUE(std::isfinite(expected[i]));
    const double error=double(actual[i])-expected[i];square+=error*error;maximum=std::max(maximum,std::abs(error));
  }
  std::printf("Q24 original Q5_K %s maxAbs=%.17g rmse=%.17g\n",label,maximum,std::sqrt(square/double(actual.size())));
  // Reuse AY's established peak-normalized quantized-chain contract. CPU
  // Q8_K and CUDA Q8_1 activation codecs are distinct; neither is replaced by
  // an invented BF16 activation/weight equation or a cross-device byte gate.
  d5_qwen38_24t_moe_profiles::expect_output(&d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,actual,expected);
}
void check_results(const Roles &roles,const Prepared &program,const KvState &state,
    const c::StateVersion &from,std::span<const float> activation,std::span<const c::ReadLease> results) {
  const auto facts=state.boundary(from);const auto count=program.logical.count;
  std::vector<Bytes> actual;for(const auto &result:results) actual.push_back(read(result));
  const auto key=floats(actual[0]),value=floats(actual[1]),norm=floats(actual[2]),rotated=floats(actual[3]);
  const auto ref_key=project(roles.key,activation,count),ref_value=project(roles.value,activation,count);
  quantized_gate(key,ref_key,"original K projection");quantized_gate(value,ref_value,"original V projection");
  const auto ref_norm=numeric::rms_norm_rows(key,roles.norm,kDepth,kEpsilon);
  numeric::expect_close(norm,ref_norm,3e-5F,2e-5F);
  numeric::expect_close(rotated,rotate(norm,count,facts.next_position),8e-5F,4e-5F);
  const auto whole_norm=numeric::rms_norm_rows(ref_key,roles.norm,kDepth,kEpsilon);
  quantized_gate(rotated,rotate(whole_norm,count,facts.next_position),"independent original K-to-norm-to-RoPE chain");
  for(int row=0;row<count;++row) for(int head=0;head<kHeads;++head) {
    const auto first=std::size_t(row*kWidth+head*kDepth+64)*sizeof(float);
    EXPECT_TRUE(std::equal(actual[2].begin()+std::ptrdiff_t(first),
        actual[2].begin()+std::ptrdiff_t(first+(kDepth-64)*sizeof(float)),actual[3].begin()+std::ptrdiff_t(first)))<<"RoPE changed unrotated192-value tail";
  }
  auto expected_k=read(take(from.member(state.entries[0]),"old full K"));
  auto expected_v=read(take(from.member(state.entries[1]),"old full V"));
  for(int row=0;row<count;++row) {
    const auto destination=(std::size_t(facts.valid_rows)+std::size_t(row))*kRowBytes;
    std::copy_n(actual[3].begin()+std::ptrdiff_t(std::size_t(row)*kRowBytes),kRowBytes,expected_k.begin()+std::ptrdiff_t(destination));
    std::copy_n(actual[1].begin()+std::ptrdiff_t(std::size_t(row)*kRowBytes),kRowBytes,expected_v.begin()+std::ptrdiff_t(destination));
    EXPECT_TRUE(actual[std::size_t(4+2*row)]==expected_k)<<"full prefix K/untouched tail bytes";
    EXPECT_TRUE(actual[std::size_t(5+2*row)]==expected_v)<<"full prefix V/untouched tail bytes";
  }
}

void run(const Roles &roles,c::Context &context,const c::StorageOwner &owner,bool cuda) {
  const auto freq=frequencies();
  const std::array<std::span<const std::byte>,4> payloads{roles.key,roles.value,std::as_bytes(std::span(roles.norm)),std::as_bytes(std::span(freq))};
  const auto parameters=q24::bind(roles,owner);
  auto block=prepare(build(2,2),context,owner,cuda,parameters);
  auto single=prepare(build(1,3),context,owner,cuda,parameters);
  auto queue=take(context.create_queue(),"one original KV queue");
  const auto footprint=[](const Logical &logical) {
    std::size_t total=0;
    for(const auto &desc:logical.output_desc) total+=(take(desc.required_bytes(),"complete output footprint")+255u)/256u*256u;
    return total;
  };
  // Both T2 generations (8 outputs each) and T1 (6 outputs) are deliberately
  // retained together:22 real blocks /320KiB, including10 full-capacity KV
  // banks. StateVersion shares those readers instead of allocating copies.
  const auto output_bytes=2*footprint(block.logical)+footprint(single.logical);
  const auto output_blocks=std::uint32_t(2*block.logical.outputs.size()+single.logical.outputs.size());
  auto pool=take(c::LeasePool::create(owner.ref(),output_bytes,{.max_blocks=output_blocks}),"bounded overlapping KV output permissions");
  auto state=KvState::create(owner);
  {
    std::array<std::vector<float>,2> initial;
    for(std::size_t role=0;role<initial.size();++role) {
      initial[role].resize(std::size_t(kCapacity*kWidth));
      for(std::size_t i=0;i<initial[role].size();++i) initial[role][i]=i<2*std::size_t(kWidth)
          ?float(int((i*7+role*13)%31)-15)/32.F:std::bit_cast<float>(std::uint32_t(0x7fc00001u+role));
    }
    const auto old_k=upload(owner,bytes<float>(initial[0])),old_v=upload(owner,bytes<float>(initial[1]));
    auto base=state.import(old_k,old_v,{2,37});auto head=take(c::StateHead::create(state.store,base),"complete KV head");
    const auto base_bytes=image(state,base);
    const auto h0=hidden(3),h1=hidden(17),h2=hidden(39);
    auto verify_h=h0;verify_h.insert(verify_h.end(),h1.begin(),h1.end());
    auto verified=execute(block,queue,owner,pool,state,base,verify_h);
    check_results(roles,block,state,base,verify_h,verified);ASSERT_FALSE(::testing::Test::HasFailure());
    std::vector<Bytes> verified_bytes;for(const auto &value:verified) verified_bytes.push_back(read(value));
    // Failed staging after a real K replacement cannot partially publish V
    // or position. The wrong-size member is another genuine readonly view.
    {
      auto failed=take(state.store.begin(base),"failure-path original draft");
      check(failed.replace(state.entries[0],verified[4]),"stage genuine candidate K");
      const auto bad=take(verified[5].subview(0,sizeof(float)),"wrong-extent V view");
      EXPECT_TRUE(failed.replace(state.entries[1],bad).IsInvalidArgument());
      check(failed.cancel(),"cancel unpublishable complete KV draft");
      EXPECT_TRUE(take(head.current(),"unchanged head").identity()==base.identity());
    }
    auto selected=state.prefix(base,verified[4],verified[5],1);
    auto rejected_suffix=state.prefix(base,verified[6],verified[7],2);
    EXPECT_TRUE(take(head.current(),"still unpublished").identity()==base.identity());
    check(head.publish(base.identity(),selected),"publish K/V/position together");
    EXPECT_TRUE(head.publish(base.identity(),rejected_suffix).IsUnavailable());
    const auto selected_bytes=image(state,selected);
    const auto facts=state.boundary(take(head.current(),"selected real head"));
    ASSERT_EQ(facts.valid_rows,3u);ASSERT_EQ(facts.next_position,38);
    // Continue from the ACTUAL head with fresh H2, not the rejected H1 row.
    auto before=take(head.current(),"actual next-round complete version");
    auto continued_outputs=execute(single,queue,owner,pool,state,before,h2);
    check_results(roles,single,state,before,h2,continued_outputs);ASSERT_FALSE(::testing::Test::HasFailure());
    auto continued=state.prefix(before,continued_outputs[4],continued_outputs[5],1);
    check(head.publish(before.identity(),continued),"publish actual next round");
    const auto continued_bytes=image(state,continued);
    ASSERT_EQ(state.boundary(continued).valid_rows,4u);ASSERT_EQ(state.boundary(continued).next_position,39);
    EXPECT_TRUE(image(state,base)==base_bytes);EXPECT_TRUE(image(state,selected)==selected_bytes);
    // Reuse the ORIGINAL T2 Invocation from retained base: H0,H2 must agree
    // with accepted-prefix H0 then T1 H2, independently of rejected H1.
    auto branch_h=h0;branch_h.insert(branch_h.end(),h2.begin(),h2.end());
    auto branch_outputs=execute(block,queue,owner,pool,state,base,branch_h);
    check_results(roles,block,state,base,branch_h,branch_outputs);ASSERT_FALSE(::testing::Test::HasFailure());
    EXPECT_TRUE(read(branch_outputs[4])==verified_bytes[4]);EXPECT_TRUE(read(branch_outputs[5])==verified_bytes[5]);
    auto branch=state.prefix(base,branch_outputs[6],branch_outputs[7],2);
    auto branch_head=take(c::StateHead::create(state.store,base),"old-version independent branch head");
    check(branch_head.publish(base.identity(),branch),"publish retained old-version continuation");
    const auto branch_bytes=image(state,branch);
    EXPECT_TRUE(branch_bytes[2]==continued_bytes[2]);
    for(std::size_t role=0;role<2;++role) {
      // Only the newly written H2 row crosses T1/T2 carrier shapes; earlier
      // rows and unused NaN tails must preserve their actual bits exactly.
      const auto first=3*kRowBytes,last=4*kRowBytes;
      EXPECT_TRUE(std::equal(branch_bytes[role].begin(),branch_bytes[role].begin()+std::ptrdiff_t(first),continued_bytes[role].begin()));
      EXPECT_TRUE(std::equal(branch_bytes[role].begin()+std::ptrdiff_t(last),branch_bytes[role].end(),continued_bytes[role].begin()+std::ptrdiff_t(last)));
      const auto lhs=floats(Bytes(branch_bytes[role].begin()+std::ptrdiff_t(first),branch_bytes[role].begin()+std::ptrdiff_t(last)));
      const auto rhs=floats(Bytes(continued_bytes[role].begin()+std::ptrdiff_t(first),continued_bytes[role].begin()+std::ptrdiff_t(last)));
      numeric::expect_close(lhs,rhs,8e-5F,4e-5F);
    }
    EXPECT_TRUE(take(head.current(),"main head after old branch").identity()==continued.identity());
    EXPECT_TRUE(image(state,base)==base_bytes);EXPECT_TRUE(image(state,selected)==selected_bytes);EXPECT_TRUE(image(state,continued)==continued_bytes);
    for(std::size_t i=0;i<verified.size();++i) EXPECT_TRUE(read(verified[i])==verified_bytes[i]);
    // The rejected H1 row was never published and fresh H2 was really used.
    EXPECT_TRUE(read(continued_outputs[4])!=read(verified[6]) || read(continued_outputs[5])!=read(verified[7]));
    auto immutable=take(selected.member(state.entries[0]),"retained readonly K member");EXPECT_FALSE(immutable.try_write());
  }
  check(state.store.collect(),"retire complete KV versions");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
  EXPECT_EQ(pool.stats().live_blocks,0u);EXPECT_EQ(pool.stats().active_uses,0u);
  for(std::size_t i=0;i<parameters.size();++i) {const auto observed=read(parameters[i].lease);EXPECT_TRUE(std::equal(observed.begin(),observed.end(),payloads[i].begin(),payloads[i].end()));}
  ::testing::Test::RecordProperty("scope","original Q24 primary blk92 K/V Q5_K and K-norm F32; normalized-H one-layer KV prefix boundary only, not complete MTP model");
  ::testing::Test::RecordProperty("backend",cuda?"ordinary CUDA Graph":"ordinary CPU Graph");
  ::testing::Test::RecordProperty("actual_submissions",3);
  ::testing::Test::RecordProperty("prefix_protocol","cap6/valid2/pos37; T2 selector1 -> actualhead T1 freshH2 -> oldbase originalT2(H0,H2); K/V/position atomic");
  ::testing::Test::RecordProperty("original_role_bytes",11535360);
  ::testing::Test::RecordProperty("key_sha256",kKeySha);::testing::Test::RecordProperty("value_sha256",kValueSha);::testing::Test::RecordProperty("knorm_sha256",kNormSha);
}
class Qwen24OriginalMtpKv:public ::testing::Test {
protected:
  static inline std::unique_ptr<const Roles> parameters;
  static void SetUpTestSuite() {const auto path=directory();if(!path.empty()) parameters=std::make_unique<const Roles>(Roles::load(path));}
  static void TearDownTestSuite() {parameters.reset();}
  void SetUp() override {if(!parameters) GTEST_SKIP()<<"Set HYLE_QWEN24_MTP_KV_DIR to authenticated original K/V and K-norm roles";}
};
#if defined(HYLE_TEST_Q24_CUDA)
TEST_F(Qwen24OriginalMtpKv,CudaOriginalPackedKvSelectsWholePrefixAndContinues) {
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"real CUDA resource")),"original CUDA storage domain");run(*parameters,context,owner,true);
}
#endif
#if defined(HYLE_TEST_Q24_CPU)
struct SystemResource final {m::MemoryResourceRef ref() noexcept {return m::system_resource();}};
TEST_F(Qwen24OriginalMtpKv,CpuOriginalPackedKvSelectsWholePrefixAndContinues) {
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"system resource owner");run(*parameters,context,owner,false);
}
#endif
} // namespace
