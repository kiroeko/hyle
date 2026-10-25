// Ordinary heterogeneous StateStore integration, using generated parameters
// at the fixed Kimi-K3 geometry. This is not original-weight model execution.
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "d5_kimi_remaining_fragments.h"
#include "kimi_original_attention_roles.h"
#include "kimi_original_attention_input.h"
#include "kimi_original_attention_reference.h"
#include "model_graph_run.h"
#include "hyle/compute/state_versions.h"
#if defined(HYLE_TEST_KIMI_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

namespace {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace kda=d5_kda_full_fragments;
namespace kimi=d5_kimi_remaining_fragments;
namespace c=hyle::compute;
using model_graph_run::detail::take;
using model_graph_run::detail::check;
constexpr std::int64_t kCapacity=4, kChannels=3*96*128;
constexpr auto kProfile=d5_moe_semantic_fragments::kKdaProfiles[0];
struct Epoch final {std::uint64_t step;std::int32_t position,session;};

TEST(KimiHeterogeneousStatePlan,RejectsInvalidColdHistoryBeforeParameterConstruction) {
  for(const auto history:std::array{kimi::OrdinaryHistory{0,0},kimi::OrdinaryHistory{4,-1},
      kimi::OrdinaryHistory{4,4},kimi::OrdinaryHistory{std::int64_t(INT32_MAX)+1,0}}) {
    EXPECT_THROW(kimi::mla(kProfile,false,history),std::runtime_error);
  }
}

TEST(KimiHeterogeneousStatePlan,EmptyHistoryUsesRealPositiveVisiblePrefixAndFullCapacitySuccessor) {
  kimi::MlaReference reference;
  const auto f=kimi::mla(kProfile,false,kimi::OrdinaryHistory{kCapacity,0},&reference);
  ASSERT_EQ(f.inputs.size(),2u);ASSERT_EQ(f.outputs.size(),9u);
  EXPECT_EQ(f.description(f.inputs[1]),sem::dense({kCapacity,576}));
  EXPECT_EQ(f.description(f.outputs[0]),sem::dense({kCapacity,576}));
  EXPECT_EQ(f.description(f.outputs[3]),sem::dense({1,1,96,512}));
  std::size_t scatter=0,attention=0;
  for(const auto &node:f.nodes) {
    if(node.operation.get_if<c::Scatter>())++scatter;
    if(node.operation.get_if<c::ScaledDotProductAttention>()) {
      ++attention;EXPECT_EQ(f.description(node.inputs[1]),sem::dense({1,1,1,576}));
      EXPECT_EQ(f.description(node.inputs[2]),sem::dense({1,1,1,512}));
    }
  }
  EXPECT_EQ(scatter,1u);EXPECT_EQ(attention,1u);
  const auto poison=std::bit_cast<float>(std::uint32_t{0x7fc12345});
  const auto expected=reference(d5_gguf_test::activation(7168,3),std::vector<float>(kCapacity*576,poison));
  ASSERT_EQ(expected[0].size(),std::size_t(kCapacity*576));
  EXPECT_TRUE(std::all_of(expected[0].begin(),expected[0].begin()+576,[](float v){return std::isfinite(v);}));
  for(std::size_t i=576;i<expected[0].size();++i)
    ASSERT_EQ(std::bit_cast<std::uint32_t>(expected[0][i]),std::uint32_t{0x7fc12345});
  for(std::size_t output=1;output<expected.size();++output)
    EXPECT_TRUE(std::all_of(expected[output].begin(),expected[output].end(),[](float v){return std::isfinite(v);}));
}

TEST(KimiHeterogeneousStatePlan,SuppliedParameterInputsComposeKdaAndNopeMlaWithoutGeneratedWeights) {
  sem::Fragment f{.profile=kProfile,.region=0x535550504c494544ULL};
  const auto input=[&](std::initializer_list<std::int64_t> shape,c::DataType type=c::DataType::kFloat32) {
    return f.input(sem::dense(shape,type));
  };
  const auto weight=[&](std::int64_t n,std::int64_t k) {return input({n,k},c::DataType::kBFloat16);};
  const auto x=input({1,7168}),history=input({3,kChannels}),state=input({1,96,128,128});
  const kda::KdaParameters recurrent_parameters{
      weight(12288,7168),weight(12288,7168),weight(12288,7168),weight(128,7168),weight(12288,128),
      f.input(sem::dense({96,7168}).transposed().value()),weight(12288,7168),weight(7168,12288),input({kChannels,4}),input({128,1}),
      input({1,1,96,128}),input({128}),{},kda::DecayEncoding::kLog};
  const auto past=input({kCapacity,576});
  const kimi::MlaParameters attention_parameters{
      input({7168}),f.input(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,7168,1536).value()),
      input({1536}),weight(18432,1536),weight(576,7168),input({512}),
      input({96,512,128},c::DataType::kBFloat16),input({96,128,512},c::DataType::kBFloat16),
      weight(12288,7168),weight(7168,12288)};
  const auto input_count=f.inputs.size();
  const auto recurrent=kda::append_kda(f,x,history,state,recurrent_parameters,{7168,96,128,1.e-5F,1.e-5F});
  const auto attention=kimi::append_mla(f,recurrent.output,past,attention_parameters,
      kimi::OrdinaryHistory{kCapacity,0});
  EXPECT_EQ(f.inputs.size(),input_count);EXPECT_TRUE(f.outputs.empty());EXPECT_TRUE(f.state_ports.empty());
  EXPECT_EQ(f.description(recurrent.history),sem::dense({3,kChannels}));
  EXPECT_EQ(f.description(recurrent.state),sem::dense({1,96,128,128}));
  EXPECT_EQ(f.description(attention.cache),sem::dense({kCapacity,576}));
  EXPECT_EQ(f.description(attention.output),sem::dense({1,7168}));
  const auto linear_uses=[&](sem::ValueId supplied,bool grouped=false) {
    return std::count_if(f.nodes.begin(),f.nodes.end(),[&](const auto &node) {
      const auto operation=grouped?node.operation.template custom_as<c::model_ops::GroupedMatMul>().has_value():
          (node.operation.template custom_as<c::model_ops::DenseLinearF32>().has_value()||
           node.operation.template get_if<c::MatMul>()!=nullptr);
      return operation&&node.inputs[grouped?0:1].ordinal()==supplied.ordinal();
    });
  };
  for(auto supplied:{recurrent_parameters.q,recurrent_parameters.k,recurrent_parameters.v,recurrent_parameters.f_a,
      recurrent_parameters.f_b,recurrent_parameters.beta,recurrent_parameters.gate,recurrent_parameters.output,
      attention_parameters.q_a,attention_parameters.q_b,attention_parameters.kv_a,attention_parameters.gate,attention_parameters.output})
    EXPECT_EQ(linear_uses(supplied),1);
  EXPECT_EQ(linear_uses(attention_parameters.absorbed_key,true),1);
  EXPECT_EQ(linear_uses(attention_parameters.absorbed_value,true),1);
  for(auto constant:f.constants)
    EXPECT_NE(f.description(constant).data_type(),c::DataType::kBFloat16);
  std::size_t rotary=0,scatter=0,attentions=0,recurrences=0;
  for(const auto &node:f.nodes) {
    rotary+=node.operation.get_if<c::RoPE>()!=nullptr;
    scatter+=node.operation.get_if<c::Scatter>()!=nullptr;
    attentions+=node.operation.get_if<c::ScaledDotProductAttention>()!=nullptr;
    recurrences+=node.operation.get_if<c::GatedDeltaNet>()!=nullptr;
  }
  EXPECT_EQ(rotary,0u);EXPECT_EQ(scatter,1u);EXPECT_EQ(attentions,1u);EXPECT_EQ(recurrences,1u);
  const auto nodes=f.nodes.size(),constants=f.constants.size(),values=f.descriptions.size();
  auto invalid=attention_parameters;invalid.output=attention_parameters.q_a;
  EXPECT_THROW(kimi::append_mla(f,recurrent.output,past,invalid,kimi::OrdinaryHistory{kCapacity,0}),std::runtime_error);
  EXPECT_EQ(f.nodes.size(),nodes);EXPECT_EQ(f.constants.size(),constants);EXPECT_EQ(f.descriptions.size(),values);
  const auto packed_format=f.description(attention_parameters.q_a).storage_format();
  const auto blocked_x=f.input(c::TensorDesc::blocked_dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{1,7168},packed_format).value());
  const auto blocked_past=f.input(c::TensorDesc::blocked_dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{kCapacity,576},packed_format).value());
  const auto blocked_values=f.descriptions.size();
  EXPECT_THROW(kda::append_kda(f,blocked_x,history,state,recurrent_parameters,{7168,96,128,1.e-5F,1.e-5F}),std::runtime_error);
  EXPECT_THROW(kimi::append_mla(f,recurrent.output,blocked_past,attention_parameters,kimi::OrdinaryHistory{kCapacity,0}),std::runtime_error);
  EXPECT_EQ(f.nodes.size(),nodes);EXPECT_EQ(f.constants.size(),constants);EXPECT_EQ(f.descriptions.size(),blocked_values);
  ::testing::Test::RecordProperty("supplied_parameter_scope",
      "Original-geometry BF16, non-square F32 and GGUF parameter inputs are descriptors only; no numeric weight buffers, source payload or native work. Both appends consume these actual Values; no seed factory.");
}

TEST(KimiHeterogeneousStatePlan,EpochMetadataRoundTripsThePublishedHeadAndPreservesOldReader) {
  const std::array specs{c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Epoch)}};
  const auto schema=take(c::StateSchema::create(specs),"Kimi host Epoch schema");
  const auto entry=take(schema.entry(0),"Kimi host Epoch entry");
  auto store=take(c::StateStore::create(schema),"Kimi host Epoch store");
  {
    const Epoch initial{0,0,7},next{1,1,7};
    const std::array members{c::StateMember{entry,{},std::as_bytes(std::span(&initial,1))}};
    const auto base=take(store.import(members),"Kimi host Epoch initial");
    auto head=take(c::StateHead::create(store,base),"Kimi host Epoch Head");
    auto draft=take(store.begin(base),"Kimi host Epoch candidate");
    check(draft.replace_metadata(entry,std::as_bytes(std::span(&next,1))),"Kimi host Epoch replace");
    const auto candidate=take(draft.finalize(),"Kimi host Epoch finalized");
    check(head.publish(base.identity(),candidate),"Kimi host Epoch publish");
    const auto observe=[&](const c::StateVersion &v) {
      Epoch result{};const auto reader=take(v.member(entry),"Kimi host Epoch reader");
      check(take(reader.view(),"Kimi host Epoch view").read(std::as_writable_bytes(std::span(&result,1))),"Kimi host Epoch bytes");return result;
    };
    const auto actual=observe(take(head.current(),"Kimi host Epoch current"));
    EXPECT_EQ(actual.step,next.step);EXPECT_EQ(actual.position,next.position);EXPECT_EQ(actual.session,next.session);
    const auto old=observe(base);EXPECT_EQ(old.step,initial.step);EXPECT_EQ(old.position,initial.position);EXPECT_EQ(old.session,initial.session);
    EXPECT_FALSE(head.publish(base.identity(),candidate).ok());
  }
  check(store.collect(),"Kimi host Epoch collection");EXPECT_EQ(store.stats().versions,0u);
}
} // namespace

#if defined(HYLE_TEST_KIMI_STATE_CUDA)
namespace {
namespace g=hyle::graph;
namespace m=hyle::mem;
using Bytes=std::vector<std::byte>;

Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());check(take(lease.view(),"Kimi readonly view").read(result),"Kimi readonly bytes");return result;
}
template<class T> std::vector<T> values(const c::ReadLease &lease) {
  const auto raw=read(lease);if(raw.size()%sizeof(T))throw std::runtime_error("Kimi typed extent");
  std::vector<T> result(raw.size()/sizeof(T));std::memcpy(result.data(),raw.data(),raw.size());return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> raw) {
  auto buffer=take(m::Buffer::create(owner.ref(),raw.size(),256),"Kimi immutable backing");
  check(buffer.view().write(raw),"Kimi one-time upload");return take(owner.adopt_buffer(std::move(buffer)),"Kimi original readonly permission");
}
void verified() {if(::testing::Test::HasFailure())throw std::runtime_error("Kimi independent reference failed before publication");}

struct Literal final {
  c::TensorDesc description;Bytes raw;c::ReadLease lease;
};
// Only cold generated parameters are shared by the three concrete MLA phases.
// Exact bytes/descriptor equality is checked, not inferred from a role's shape.
struct LiteralBank final {
  std::vector<std::vector<Literal>> slots;
  c::ReadLease bind(std::size_t slot,const c::TensorDesc &description,Bytes raw,const c::StorageOwner &owner) {
    if(slots.size()<=slot)slots.resize(slot+1);
    for(const auto &literal:slots[slot])if(literal.description==description&&literal.raw==raw)return literal.lease;
    auto lease=upload(owner,raw);slots[slot].push_back({description,std::move(raw),lease});return lease;
  }
  void release_host_bytes() {for(auto &slot:slots)for(auto &literal:slot)Bytes{}.swap(literal.raw);}
};
struct Run final {
  sem::Fragment facts;sem::GraphCase logical;
  std::vector<c::ReadLease> literals;
  g::CompiledGraph compiled;g::Invocation invocation;c::LeasePool pool;
  std::vector<model_graph_run::Port> inputs;
  std::vector<model_graph_run::OutputPort> outputs;
  std::size_t submissions=0;
  static sem::ValueId source_constant(sem::Fragment &f,std::vector<c::ReadLease> &supplied,
      const original_named_role_source::ReadonlyRole &role,const c::StorageOwner &owner) {
    if(f.constants.size()!=supplied.size())throw std::runtime_error("original Kimi learned constants must be the entire cold prefix");
    if(role.owner.ref()!=owner.ref())throw std::runtime_error("original Kimi constant owner differs from its cold Graph owner");
    const auto value=f.constant(role.description,{});supplied.push_back(role.lease);return value;
  }
  static Run prepare(sem::Fragment f,c::Context &context,const c::StorageOwner &owner,LiteralBank *shared=nullptr,
      std::vector<c::ReadLease> supplied_literals={}) {
    Run result;result.literals=std::move(supplied_literals);
    const auto supplied=result.literals.size();
    if(supplied>f.constants.size())throw std::runtime_error("original Kimi constant lease count differs");
    for(std::size_t i=0;i<supplied;++i) {
      const auto &description=f.description(f.constants[i]);
      if(!f.constant_bytes[i].empty()||result.literals[i].size()!=take(description.required_bytes(),"original Kimi supplied constant extent"))
        throw std::runtime_error("original Kimi constant must have its actual lease and no generated payload");
      take(result.literals[i].tensor(description),"original Kimi typed source lease");
    }
    // Explicit caller outputs keep every generated value independent of the
    // phase's mutable storage and of source/input aliases across later steps.
    for(auto &output:f.outputs)output=f.one(c::Operation(c::Copy{}),{output});
    result.logical=sem::make_graph(f);std::vector<g::ConstantBinding> constants;
    for(std::size_t i=0;i<f.constants.size();++i) {
      const auto &description=f.description(f.constants[i]);
      if(i>=supplied) {
        if(f.constant_bytes[i].empty()&&take(description.required_bytes(),"Kimi literal extent"))
          throw std::runtime_error("original Kimi constant source lease is absent");
        result.literals.push_back(shared?shared->bind(i,description,std::move(f.constant_bytes[i]),owner):upload(owner,f.constant_bytes[i]));
      }
      constants.push_back({.value=result.logical.constants[i],.tensor=take(result.literals[i].tensor(description),"Kimi typed literal"),
          .owner=owner,.lease=result.literals[i]});
    }
    f.constant_bytes.clear();
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*f.profile),"Kimi geometry profile"),f.phase,{0x48594c454d4f444cULL,f.region,1}};
    result.compiled=take(g::cuda::compile(result.logical.graph,context,owner,{.constants=constants,.workload=workload}),"Kimi ordinary Graph compile");
    std::size_t bytes=0;
    for(std::size_t i=0;i<f.inputs.size();++i)result.inputs.push_back({result.logical.inputs[i],f.description(f.inputs[i])});
    for(std::size_t i=0;i<f.outputs.size();++i) {
      result.outputs.push_back(model_graph_run::prepare_output(result.compiled,{result.logical.outputs[i],f.description(f.outputs[i])}));
      bytes+=(take(f.description(f.outputs[i]).required_bytes(),"Kimi output extent")+255)/256*256;
    }
    constexpr std::uint32_t generations=8;
    result.pool=take(c::LeasePool::create(owner.ref(),bytes*generations,
        {.max_blocks=std::uint32_t(f.outputs.size())*generations,.backing_alignment=256}),"bounded Kimi output generations");
    result.invocation=take(result.compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"Kimi ordinary Invocation");
    result.facts=std::move(f);return result;
  }
  std::vector<c::ReadLease> execute(c::Queue &queue,std::span<const c::ReadLease> input) {
    return model_graph_run::execute(invocation,pool,queue,inputs,input,outputs,submissions);
  }
};

// Cold-only original parameter assembly. ReadLeases are moved into the
// existing Run::literals vector; no second table owns the native constants.
inline Run prepare_original_kda(const kimi_original_attention_roles::Kda &original,
    c::Context &context,const c::StorageOwner &owner) {
  sem::Fragment f{.profile=kProfile,.region=0x030602};
  const auto input=f.input(sem::dense({1,7168}));
  const auto history=f.input(sem::dense({3,kChannels}));
  const auto state=f.input(sem::dense({1,96,128,128}));
  std::vector<c::ReadLease> literals;literals.reserve(12);
  const auto constant=[&](const original_named_role_source::ReadonlyRole &role) {
    return Run::source_constant(f,literals,role,owner);
  };
  const auto parameters=kimi_original_attention_values::declare_kda(original.view(),constant);
  const auto result=kda::append_kda(f,input,history,state,parameters,{7168,96,128,1.e-5F,1.e-5F});
  const auto outputs=result.outputs();f.outputs.assign(outputs.begin(),outputs.end());
  f.state_ports={{1,0},{2,2}};
  return Run::prepare(std::move(f),context,owner,nullptr,std::move(literals));
}
inline Run prepare_original_mla(std::int32_t position,const kimi_original_attention_roles::Mla &original,
    c::Context &context,const c::StorageOwner &owner,LiteralBank &small_literals) {
  sem::Fragment f{.profile=kProfile,.region=0x070701};
  const auto input=f.input(sem::dense({1,7168}));
  const auto past=f.input(sem::dense({kCapacity,576}));
  std::vector<c::ReadLease> literals;literals.reserve(10);
  const auto constant=[&](const original_named_role_source::ReadonlyRole &role) {
    return Run::source_constant(f,literals,role,owner);
  };
  const auto parameters=kimi_original_attention_values::declare_mla(original.view(),constant);
  const auto result=kimi::append_mla(f,input,past,parameters,kimi::OrdinaryHistory{kCapacity,position});
  const auto outputs=result.outputs();f.outputs.assign(outputs.begin(),outputs.end());
  f.bfloat_chain_outputs={0,1,2,3,4,5,6,8};f.state_ports={{1,0}};
  // Same real integer cursor successor as the synthetic ordinary consumer.
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32));
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto joined=f.one(c::Operation(c::Concat{.axis=0}),{cursor,one});
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),{joined});
  f.outputs.push_back(f.one(c::Operation(c::Gather{}),{scanned,one}));
  return Run::prepare(std::move(f),context,owner,&small_literals,std::move(literals));
}

enum Entry:std::size_t {kHistory,kRecurrent,kMla,kCursor,kFeature,kEpoch,kEntries};
struct State final {
  c::StateSchema schema;c::StateStore store;std::array<c::StateEntryId,kEntries> entries;
  c::ReadLease member(const c::StateVersion &v,Entry entry) const {return take(v.member(entries[entry]),"Kimi complete member");}
  Epoch epoch(const c::StateVersion &version) const {
    const auto raw=read(member(version,kEpoch));
    if(raw.size()!=sizeof(Epoch))throw std::runtime_error("Kimi Epoch metadata extent");
    Epoch result{};std::memcpy(&result,raw.data(),sizeof(result));return result;
  }
  static State create(const c::StorageOwner &owner,std::size_t budget) {
    const std::array descriptions{sem::dense({3,kChannels}),sem::dense({1,96,128,128}),sem::dense({kCapacity,576}),
        sem::dense({1},c::DataType::kInt32),sem::dense({1,7168})};
    std::vector<c::StateEntrySpec> specs;
    for(const auto &description:descriptions)specs.push_back({.description=description,.resource=owner.ref()});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Epoch)});
    State result;result.schema=take(c::StateSchema::create(specs),"Kimi heterogeneous schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=12,.max_drafts=3,.max_retained_blocks=128,
        .max_backing_bytes=budget,.max_host_snapshot_bytes=4096}),"one bounded heterogeneous StateStore");
    for(std::size_t i=0;i<kEntries;++i)result.entries[i]=take(result.schema.entry(i),"Kimi state entry");return result;
  }
  c::StateVersion initial(const c::StorageOwner &owner,std::int32_t session) {
    std::vector<c::StateMember> members;
    for(std::size_t i=0;i<kEpoch;++i) {
      const auto spec=take(schema.specification(entries[i]),"Kimi initial descriptor");
      Bytes raw(take(spec.description.required_bytes(),"Kimi initial bytes"));
      if(i==kMla)raw=sem::bytes(std::vector<float>(std::size_t(kCapacity*576),std::bit_cast<float>(std::uint32_t{0x7fc12345})));
      members.push_back({entries[i],upload(owner,raw)});
    }
    const Epoch epoch{0,0,session};members.push_back({entries[kEpoch],{},std::as_bytes(std::span(&epoch,1))});
    return take(store.import(members),"Kimi empty complete state");
  }
};

struct Mla final {Run run;kimi::MlaReference reference;};
Mla prepare_mla(std::int32_t position,c::Context &context,const c::StorageOwner &owner,LiteralBank &literals) {
  Mla result;auto f=kimi::mla(kProfile,false,kimi::OrdinaryHistory{kCapacity,position},&result.reference);
  // The actual stored cursor is advanced by the original public integer graph
  // operations. Cold phase admission checks its value before executing.
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32));
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto joined=f.one(c::Operation(c::Concat{.axis=0}),{cursor,one});
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),{joined});
  f.outputs.push_back(f.one(c::Operation(c::Gather{}),{scanned,one}));
  result.run=Run::prepare(std::move(f),context,owner,&literals);return result;
}

void compare(Run &run,std::span<const c::ReadLease> output,const std::vector<std::vector<float>> &expected,
             std::size_t visible=0) {
  ASSERT_GE(output.size(),expected.size());
  for(std::size_t i=0;i<expected.size();++i) {
    SCOPED_TRACE("actual output "+std::to_string(i));const auto actual=values<float>(output[i]);
    ASSERT_EQ(actual.size(),expected[i].size());
    if(visible&&i==0) {
      const auto count=visible*576;
      sem::expect_output(run.facts,std::span(actual).first(count),std::span(expected[i]).first(count),{},false,nullptr,true,i);
      for(std::size_t at=count;at<actual.size();++at)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[at]),std::bit_cast<std::uint32_t>(expected[i][at]));
    } else {
      for(const auto v:actual)ASSERT_TRUE(std::isfinite(v));
      sem::expect_output(run.facts,actual,expected[i],{},false,nullptr,run.facts.bfloat_chain_output(i),i);
    }
  }
}
std::array<Bytes,kEpoch> snapshot(const State &state,const c::StateVersion &version) {
  std::array<Bytes,kEpoch> result;
  for(std::size_t i=0;i<kEpoch;++i)result[i]=read(state.member(version,static_cast<Entry>(i)));return result;
}
struct SharedDevice final {std::shared_ptr<c::cuda::MemoryResource> device;m::MemoryResourceRef ref() noexcept{return device->ref();}};

TEST(KimiHeterogeneousState,CudaBorrowedLiteralPrefixUsesActualLeaseAndRejectsMisalignment) {
  auto context=take(g::cuda::create_context(),"literal-prefix CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"literal-prefix actual device"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"literal-prefix original native owner");
  auto queue=take(context.create_queue(),"literal-prefix original queue");
  const std::vector<float> weights{1.F,2.F,3.F,4.F},input_values{5.F,6.F,7.F,8.F};
  const auto description=sem::dense({4});
  auto lease=upload(owner,sem::bytes(weights));
  const auto original_view=take(lease.view(),"literal-prefix original view");
  original_named_role_source::ReadonlyRole role{"synthetic F32 literal",description,owner,lease,nullptr};
  sem::Fragment bad{.profile=kProfile,.region=0x4b494d494c495431ULL};
  bad.constant(description,sem::bytes(weights));
  std::vector<c::ReadLease> empty_prefix;
  const auto bad_constants=bad.constants.size(),bad_values=bad.descriptions.size();
  EXPECT_THROW(Run::source_constant(bad,empty_prefix,role,owner),std::runtime_error);
  EXPECT_EQ(bad.constants.size(),bad_constants);EXPECT_EQ(bad.descriptions.size(),bad_values);EXPECT_TRUE(empty_prefix.empty());
  const auto reject=[&](bool real_bytes,bool excessive) {
    sem::Fragment f{.profile=kProfile,.region=0x4b494d494c495432ULL};
    const auto input=f.input(description);
    const auto constant=f.constant(description,real_bytes?sem::bytes(weights):Bytes{});
    f.outputs={f.one(c::Operation(c::Binary{c::BinaryKind::kAdd}),{input,constant})};
    std::vector<c::ReadLease> supplied{lease};if(excessive)supplied.push_back(lease);
    EXPECT_THROW(Run::prepare(std::move(f),context,owner,nullptr,std::move(supplied)),std::runtime_error);
  };
  reject(true,false);reject(false,true);
  c::ReadLease escaped;
  {
    sem::Fragment f{.profile=kProfile,.region=0x4b494d494c495433ULL};
    const auto input=f.input(description);std::vector<c::ReadLease> supplied;
    const auto constant=Run::source_constant(f,supplied,role,owner);
    ASSERT_EQ(f.constant_bytes.size(),1u);EXPECT_TRUE(f.constant_bytes[0].empty());
    f.outputs={f.one(c::Operation(c::Binary{c::BinaryKind::kAdd}),{input,constant})};
    auto run=Run::prepare(std::move(f),context,owner,nullptr,std::move(supplied));
    ASSERT_EQ(run.literals.size(),1u);
    const auto actual_view=take(run.literals[0].view(),"literal-prefix retained original view");
    EXPECT_EQ(actual_view.native_handle(),original_view.native_handle());EXPECT_EQ(actual_view.offset(),original_view.offset());
    EXPECT_EQ(actual_view.resource(),original_view.resource());
    role.lease={};lease={}; // The original Run owns the same immutable lease.
    const std::array inputs{upload(owner,sem::bytes(input_values))};
    const auto output=run.execute(queue,inputs);
    ASSERT_EQ(output.size(),1u);EXPECT_EQ(values<float>(output[0]),(std::vector<float>{6.F,8.F,10.F,12.F}));
    EXPECT_EQ(run.submissions,1u);escaped=run.literals[0];
  }
  EXPECT_EQ(values<float>(escaped),weights);escaped={};
  ::testing::Test::RecordProperty("source_scope","Synthetic F32[4] lease only; no original Kimi payload read or numeric qualification");
  ::testing::Test::RecordProperty("binding_scope","Actual original native handle/offset/resource reused through existing Run literals and GraphConstantBinding; no numeric constant byte carrier");
}

TEST(KimiHeterogeneousState,CudaOriginalGeometrySyntheticKdaMlaPublishesWholeStateAndRestoresForks) {
  auto context=take(g::cuda::create_context(),"Kimi CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Kimi actual device"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"Kimi original native owner");
  auto queue=take(context.create_queue(),"Kimi original queue");
  kda::Reference kda_reference;auto kda_run=Run::prepare(kda::complete(kProfile,&kda_reference),context,owner);
  LiteralBank literals;std::array<Mla,3> mla;
  for(std::int32_t p=0;p<3;++p)mla[std::size_t(p)]=prepare_mla(p,context,owner,literals);
  literals.release_host_bytes();
  auto budget=kda_run.pool.stats().capacity_bytes+32u*1024u*1024u;
  for(const auto &phase:mla)budget+=phase.run.pool.stats().capacity_bytes;
  auto state=State::create(owner,budget);std::size_t committed=0,cancelled=0,stale=0;
  {
    auto initial_a=state.initial(owner,0),initial_b=state.initial(owner,1);
    auto a=take(c::StateHead::create(state.store,initial_a),"Kimi ordinary Head A");
    auto b=take(c::StateHead::create(state.store,initial_b),"Kimi ordinary Head B");
    const auto empty=snapshot(state,initial_a);
    const auto execute=[&](c::StateHead &head,std::int32_t position,int seed,std::int32_t session,bool cancel) {
      auto before=take(head.current(),"Kimi expected base");const auto old=snapshot(state,before);
      ASSERT_EQ(values<std::int32_t>(state.member(before,kCursor)),std::vector<std::int32_t>{position});
      const auto before_epoch=state.epoch(before);ASSERT_EQ(before_epoch.step,std::uint64_t(position));ASSERT_EQ(before_epoch.position,position);
      const auto input=d5_gguf_test::activation(7168,seed);
      const std::array args{upload(owner,sem::bytes(input)),state.member(before,kHistory),state.member(before,kRecurrent)};
      auto actual_kda=kda_run.execute(queue,args);
      compare(kda_run,actual_kda,kda_reference(input,values<float>(args[1]),values<float>(args[2])));verified();
      auto draft=take(state.store.begin(before),"Kimi unpublished heterogeneous candidate");
      check(draft.replace(state.entries[kHistory],actual_kda[0]),"Kimi candidate convolution history");
      check(draft.replace(state.entries[kRecurrent],actual_kda[2]),"Kimi candidate KDA recurrent state");
      if(cancel) {
        const auto submissions=mla[std::size_t(position)].run.submissions;
        check(draft.cancel(),"cancel unpublished Kimi candidate after KDA");++cancelled;
        EXPECT_TRUE(take(head.current(),"cancelled Kimi Head").identity()==before.identity());
        EXPECT_EQ(state.epoch(take(head.current(),"cancelled Kimi Epoch")).step,before_epoch.step);
        EXPECT_EQ(snapshot(state,before),old);EXPECT_EQ(mla[std::size_t(position)].run.submissions,submissions);return;
      }
      auto &phase=mla[std::size_t(position)];
      const std::array mla_args{actual_kda[5],state.member(before,kMla),state.member(before,kCursor)};
      const auto actual_mla=phase.run.execute(queue,mla_args);
      // The second graph receives the actual retained KDA output lease, never
      // an oracle-generated bridge. Its independent evaluator reads that same
      // actual predecessor and the actual immutable old MLA state.
      compare(phase.run,actual_mla,phase.reference(values<float>(actual_kda[5]),values<float>(mla_args[1])),std::size_t(position+1));
      EXPECT_EQ(values<std::int32_t>(actual_mla[9]),std::vector<std::int32_t>{position+1});verified();
      check(draft.replace(state.entries[kMla],actual_mla[0]),"Kimi complete MLA cache");
      check(draft.replace(state.entries[kCursor],actual_mla[9]),"Kimi actual next cursor");
      check(draft.replace(state.entries[kFeature],actual_mla[6]),"Kimi actual final feature");
      const Epoch epoch{std::uint64_t(position+1),position+1,session};
      check(draft.replace_metadata(state.entries[kEpoch],std::as_bytes(std::span(&epoch,1))),"Kimi common epoch witness");
      const auto candidate=take(draft.finalize(),"Kimi finalized complete tuple");
      const auto actual_epoch=state.epoch(candidate);
      EXPECT_EQ(actual_epoch.step,epoch.step);EXPECT_EQ(actual_epoch.position,epoch.position);EXPECT_EQ(actual_epoch.session,epoch.session);verified();
      check(head.publish(before.identity(),candidate),"atomic Kimi complete Head publication");++committed;
      EXPECT_FALSE(head.publish(before.identity(),candidate).ok());++stale;
      EXPECT_EQ(snapshot(state,before),old);
      std::printf("Kimi synthetic ordinary session=%d position=%d visible=%d epoch=%llu completeResources=5\n",session,position,position+1,
          static_cast<unsigned long long>(epoch.step));
    };
    execute(a,0,3,0,false);execute(b,0,17,1,false);
    auto a1=take(a.current(),"Kimi retained A1 state");auto b1=take(b.current(),"Kimi independent B1 state");
    EXPECT_NE(read(state.member(a1,kFeature)),read(state.member(b1,kFeature)));
    const auto old_a=snapshot(state,a1),old_b=snapshot(state,b1);
    auto fork=take(c::StateHead::create(state.store,a1),"Kimi immutable restored fork");
    execute(a,1,29,0,true);execute(a,1,29,0,false);
    EXPECT_TRUE(take(fork.current(),"unmodified Kimi fork").identity()==a1.identity());
    execute(fork,1,41,2,false);execute(a,2,53,0,false);execute(b,1,67,1,false);
    EXPECT_EQ(snapshot(state,a1),old_a);EXPECT_EQ(snapshot(state,b1),old_b);EXPECT_EQ(snapshot(state,initial_a),empty);
    EXPECT_EQ(values<std::int32_t>(state.member(take(a.current(),"Kimi final A"),kCursor)),std::vector<std::int32_t>{3});
    EXPECT_EQ(values<std::int32_t>(state.member(take(b.current(),"Kimi final B"),kCursor)),std::vector<std::int32_t>{2});
    EXPECT_NE(read(state.member(take(fork.current(),"Kimi final fork"),kFeature)),read(state.member(take(a.current(),"Kimi final main"),kFeature)));
  }
  check(state.store.collect(),"Kimi final retired-state collection");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);EXPECT_EQ(state.store.stats().backing_bytes,0u);
  EXPECT_EQ(committed,6u);EXPECT_EQ(cancelled,1u);EXPECT_EQ(stale,6u);
  EXPECT_EQ(kda_run.submissions,7u);EXPECT_EQ(mla[0].run.submissions,2u);EXPECT_EQ(mla[1].run.submissions,3u);EXPECT_EQ(mla[2].run.submissions,1u);
  ::testing::Test::RecordProperty("weight_scope","Generated sparse BF16/F32 weights at fixed Kimi7168/96x128 geometry; NOT original artifact payload or quality");
  ::testing::Test::RecordProperty("composition_scope","Actual KDA hidden lease feeds actual absorbed MLA. Intervening full model layers/residuals and LM head are not executed; no generated token or MTP claim.");
  ::testing::Test::RecordProperty("state_scope","One StateStore, two independent Heads and one immutable restored fork; convolution history, recurrent KDA, full finite MLA cache, actual Graph I32 cursor, actual final feature, Epoch metadata. AttnRes is frame-only and absent.");
  ::testing::Test::RecordProperty("ordinary_commits",std::to_string(committed));
  ::testing::Test::RecordProperty("cancelled_unpublished_candidates",std::to_string(cancelled));
  ::testing::Test::RecordProperty("stale_publish_rejections",std::to_string(stale));
  ::testing::Test::RecordProperty("native_cancellation_scope","Cancellation is of the unpublished StateDraft after KDA has actually retired; no GPU preemption claim");
  ::testing::Test::RecordProperty("state_backing_budget",std::to_string(budget));
}
// Original artifact operands, bounded ordinary state continuation. The KDA0
// input is a supplied normalized-upstream fixture; KDA0 -> MLA3 skips actual
// intervening layers and therefore is NOT a full model/quality/token claim.
TEST(KimiHeterogeneousState,CudaOriginalParametersKdaMlaKeepCompleteStateAcrossContinuation) {
  namespace original=kimi_original_attention_roles;
  namespace source_input=kimi_original_attention_input;
  namespace reference=kimi_original_attention_reference;
  const auto directory=source_input::environment();
  if(directory.empty())GTEST_SKIP()<<"Set HYLE_KIMI_ORIGINAL_ATTENTION_DIR to the completed exact 23-role batch; no implicit download";
  const auto input=source_input::load(directory);
  const auto source_kda=original::Data::load(input.directory,original::Component::kKda,input.pins());
  const auto source_mla=original::Data::load(input.directory,original::Component::kMla,input.pins());
  auto context=take(g::cuda::create_context(),"original-parameter Kimi CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original-parameter Kimi device"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"original-parameter Kimi owner");
  auto queue=take(context.create_queue(),"original-parameter Kimi queue");
  const auto kda_operands=original::prepare_kda(source_kda,owner);
  const auto mla_operands=original::prepare_mla(source_mla,owner);
  auto recurrent=prepare_original_kda(kda_operands,context,owner);
  LiteralBank small_literals;
  std::array<decltype(recurrent),2> attention;
  for(std::int32_t position=0;position<2;++position)
    attention[std::size_t(position)]=prepare_original_mla(position,mla_operands,context,owner,small_literals);
  small_literals.release_host_bytes();
  std::size_t state_budget=recurrent.pool.stats().capacity_bytes+32u*1024u*1024u;
  for(const auto &run:attention)state_budget+=run.pool.stats().capacity_bytes;
  auto state=State::create(owner,state_budget);
  std::size_t committed=0,stale=0,cancelled=0;
  const auto expect_dot=[&](float actual,const reference::Dot &expected,std::string_view name,std::size_t row) {
    const auto error=reference::scalar::distance(double(actual),expected.center);
    ASSERT_TRUE(std::isfinite(actual))<<name<<" row="<<row;
    ASSERT_LE(error,expected.bound)<<name<<" row="<<row<<" actual="<<actual
        <<" original F64="<<expected.center<<" sumabs="<<expected.sumabs;
  };
  {
    auto initial=state.initial(owner,0);
    auto head=take(c::StateHead::create(state.store,initial),"original-parameter complete Head");
    const auto old_initial=snapshot(state,initial);
    c::StateVersion first;
    for(std::int32_t position=0;position<2;++position) {
      auto before=take(head.current(),"original-parameter expected base");
      const auto old=snapshot(state,before);
      ASSERT_EQ(values<std::int32_t>(state.member(before,kCursor)),std::vector<std::int32_t>{position});
      const auto old_epoch=state.epoch(before);
      ASSERT_EQ(old_epoch.step,std::uint64_t(position));ASSERT_EQ(old_epoch.position,position);
      const auto generated_input=d5_gguf_test::activation(7168,3+26*position);
      // Explicit upstream fixture with unit norm scale, not an alleged
      // original layer0 input-norm/residual computation outside this fragment.
      const auto activation=sem::oracle::rms_norm_rows(generated_input,
          std::vector<float>(7168,1.F),7168,1.e-5F);
      const std::array args{upload(owner,sem::bytes(activation)),
          state.member(before,kHistory),state.member(before,kRecurrent)};
      auto actual_kda=recurrent.execute(queue,args);
      const auto prior_history=values<float>(args[1]),prior_state=values<float>(args[2]);
      compare(recurrent,actual_kda,reference::kda(*source_kda,activation,prior_history,prior_state));verified();
      // Local raw-row certificates start at ACTUAL native producer values.
      // Current Q/K/V are also the last row of the actual next history.
      const auto actual_history=values<float>(actual_kda[0]);
      const auto current=std::span(actual_history).last(std::size_t(kChannels));
      constexpr std::array projections{original::Role::kKdaQ,original::Role::kKdaK,original::Role::kKdaV};
      for(std::size_t part=0;part<projections.size();++part)
        for(const auto row:std::array<std::size_t,3>{0,reference::width/2,reference::width-1})
          expect_dot(current[part*reference::width+row],reference::original_row(*source_kda,projections[part],row,activation),
              original::specs[std::size_t(projections[part])].name,row);
      const auto actual_gated=values<float>(actual_kda[4]),actual_output=values<float>(actual_kda[5]);
      for(const auto row:std::array<std::size_t,3>{0,reference::hidden/2,reference::hidden-1})
        expect_dot(actual_output[row],reference::original_row(*source_kda,original::Role::kKdaOutput,row,actual_gated),"KDA original output",row);
      verified();
      auto draft=take(state.store.begin(before),"original-parameter complete draft");
      check(draft.replace(state.entries[kHistory],actual_kda[0]),"original convolution history");
      check(draft.replace(state.entries[kRecurrent],actual_kda[2]),"original recurrent state");
      auto &run=attention[std::size_t(position)];
      const std::array mla_args{actual_kda[5],state.member(before,kMla),state.member(before,kCursor)};
      auto actual_mla=run.execute(queue,mla_args);
      // Actual native KDA output is a labelled upstream fixture for layer3.
      // Neither a host oracle feature nor an ideal old state feeds the graph.
      const auto actual_input=values<float>(mla_args[0]),old_cache=values<float>(mla_args[1]);
      compare(run,actual_mla,reference::mla(*source_mla,actual_input,old_cache,
          std::size_t(kCapacity),std::size_t(position)),std::size_t(position+1));
      ASSERT_EQ(values<std::int32_t>(actual_mla[9]),std::vector<std::int32_t>{position+1});
      const auto query=values<float>(actual_mla[1]),absorbed=values<float>(actual_mla[2]);
      const auto attended=values<float>(actual_mla[3]),projected=values<float>(actual_mla[4]);
      for(const auto h:std::array<std::size_t,3>{0,reference::heads/2,reference::heads-1}) {
        for(const auto row:std::array<std::size_t,3>{0,256,511})
          expect_dot(absorbed[h*512+row],reference::original_absorbed_row(*source_mla,h,row,
              std::span(query).subspan(h*192,128),true),"original KV-b absorbed K",h*512+row);
        for(const auto row:std::array<std::size_t,3>{0,64,127})
          expect_dot(projected[h*128+row],reference::original_absorbed_row(*source_mla,h,row,
              std::span(attended).subspan(h*512,512),false),"original KV-b V",h*128+row);
      }
      const auto normalized=values<float>(actual_mla[7]),gate=values<float>(actual_mla[8]);
      const auto gated=values<float>(actual_mla[5]),output=values<float>(actual_mla[6]);
      for(const auto row:std::array<std::size_t,3>{0,reference::width/2,reference::width-1})
        expect_dot(gate[row],reference::original_row(*source_mla,original::Role::kMlaGate,row,normalized),"MLA original gate",row);
      for(const auto row:std::array<std::size_t,3>{0,reference::hidden/2,reference::hidden-1})
        expect_dot(output[row],reference::original_row(*source_mla,original::Role::kMlaOutput,row,gated),"MLA original output",row);
      verified();
      check(draft.replace(state.entries[kMla],actual_mla[0]),"original complete MLA cache");
      check(draft.replace(state.entries[kCursor],actual_mla[9]),"original actual cursor");
      check(draft.replace(state.entries[kFeature],actual_mla[6]),"original final feature");
      const Epoch epoch{std::uint64_t(position+1),position+1,0};
      check(draft.replace_metadata(state.entries[kEpoch],std::as_bytes(std::span(&epoch,1))),"original complete Epoch");
      const auto candidate=take(draft.finalize(),"original complete candidate");
      const auto observed_epoch=state.epoch(candidate);
      ASSERT_EQ(observed_epoch.step,epoch.step);ASSERT_EQ(observed_epoch.position,epoch.position);ASSERT_EQ(observed_epoch.session,epoch.session);
      check(head.publish(before.identity(),candidate),"original whole-state publication");++committed;
      ASSERT_FALSE(head.publish(before.identity(),candidate).ok());++stale;
      EXPECT_EQ(snapshot(state,before),old);
      if(position==0)first=candidate;
      std::printf("Kimi original-role ordinary position=%d visible=%d epoch=%llu completeResources=5\n",
          position,position+1,static_cast<unsigned long long>(epoch.step));
    }
    const auto retained_first=snapshot(state,first);
    auto fork=take(c::StateHead::create(state.store,first),"original complete old-version fork");
    const auto fork_base=take(fork.current(),"original old-version fork base");
    EXPECT_TRUE(fork_base.identity()==first.identity());
    EXPECT_EQ(values<std::int32_t>(state.member(fork_base,kCursor)),std::vector<std::int32_t>{1});
    const auto current_head=take(head.current(),"original current Head after continuation");
    EXPECT_EQ(values<std::int32_t>(state.member(current_head,kCursor)),std::vector<std::int32_t>{2});
    // No third native forward: cancel an unpublished candidate against a real
    // retained complete version, and require both current Heads unchanged.
    auto abandoned=take(state.store.begin(first),"original unpublished old-version candidate");
    check(abandoned.replace(state.entries[kFeature],state.member(current_head,kFeature)),"original unpublished replacement reader");
    check(abandoned.cancel(),"original unpublished candidate cancellation");++cancelled;
    EXPECT_TRUE(take(head.current(),"original Head after cancel").identity()==current_head.identity());
    EXPECT_TRUE(take(fork.current(),"original fork after cancel").identity()==first.identity());
    EXPECT_EQ(snapshot(state,first),retained_first);EXPECT_EQ(snapshot(state,initial),old_initial);
    const auto first_epoch=state.epoch(first),initial_epoch=state.epoch(initial);
    EXPECT_EQ(first_epoch.step,1u);EXPECT_EQ(first_epoch.position,1);EXPECT_EQ(first_epoch.session,0);
    EXPECT_EQ(initial_epoch.step,0u);EXPECT_EQ(initial_epoch.position,0);EXPECT_EQ(initial_epoch.session,0);
  }
  check(state.store.collect(),"original final state collection");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);EXPECT_EQ(state.store.stats().backing_bytes,0u);
  EXPECT_EQ(committed,2u);EXPECT_EQ(stale,2u);EXPECT_EQ(cancelled,1u);
  EXPECT_EQ(recurrent.submissions,2u);EXPECT_EQ(attention[0].submissions,1u);EXPECT_EQ(attention[1].submissions,1u);
  ::testing::Test::RecordProperty("source_scope","Original fixed-revision KDA0 14 roles + MLA3 9 roles; per-union/per-role SHA checked, original raw KV-b is not uploaded");
  ::testing::Test::RecordProperty("upstream_scope","Supplied normalized KDA0 input; KDA0 output is an explicit upstream fixture for MLA3, not intervening layers or complete model execution");
  ::testing::Test::RecordProperty("oracle_scope","Original raw row/tap/KV-b indexing; no prepared K/V or expanded F32 weight bank. Existing six/nine-output accuracy gates plus selected actual-producer F64 projection certificates");
  ::testing::Test::RecordProperty("state_scope","Two real complete Head commits, actual prior native state feedback, immutable old readers, restored old-version fork, unpublished cancellation, stale expected-base rejection; no native preemption claim");
}

} // namespace
#endif
