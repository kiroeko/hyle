// Ordinary original target77 and its genuine layer74 shared-index boundary.
// These cold description checks require neither weights nor a native device.
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "glm_pro_target_fragment.h"
#include "glm_pro_target_reference.h"
#include "glm_original_moe_resident_weights.h"
#include "glm_original_moe_reference.h"
#include "model_graph_run.h"
#include "model_resident_fragment.h"
#include "../../engine/inference/graph_bindings.h"
#include "hyle/compute/state_versions.h"
#if defined(HYLE_TEST_GLM_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

namespace {
namespace target=glm_pro_target_fragment;
namespace model=glm_pro_mtp_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;

struct DescriptionFixture final {
  sem::Fragment fragment{.profile=&d5_moe_semantic_fragments::profiles::glm53_official,
      .region=0x474c4d373453484fULL,.quantized=true};
  target::Parameters parameters;
  model::moe::Parameters experts;
  Value hidden,residual,main,cursor;
  DescriptionFixture() {
    const auto input=[&](std::initializer_list<std::int64_t> shape,c::DataType type=c::DataType::kFloat32) {
      return fragment.input(sem::dense(shape,type));
    };
    const auto fp8=[&](std::int64_t n,std::int64_t k) {
      return fragment.input(glm_shared_vocabulary::take(c::float8_weight_description(k,n,
          c::Float8ScaleMode::kFloat32Block128),"ordinary target description"));
    };
    hidden=input({1,6144},c::DataType::kBFloat16);residual=input({1,6144},c::DataType::kBFloat16);
    main=input({2054,576});cursor=input({1},c::DataType::kInt32);
    parameters.input_norm=input({6144},c::DataType::kBFloat16);
    parameters.query_a=fp8(2048,6144);parameters.query_norm=input({2048},c::DataType::kBFloat16);
    parameters.target_norm=input({6144},c::DataType::kBFloat16);
    parameters.attention={input({512},c::DataType::kBFloat16),fp8(576,6144),fp8(6144,16384),
        input({6144},c::DataType::kBFloat16),fp8(16384,2048),
        input({64,512,192},c::DataType::kBFloat16),input({64,256,512},c::DataType::kBFloat16)};
    constexpr auto geometry=glm_moe_geometry::kPro;
    experts.router=input({geometry.experts,geometry.hidden},c::DataType::kBFloat16);
    experts.bias=input({geometry.experts});experts.map=input({geometry.experts},c::DataType::kInt32);
    for(std::size_t leg=0;leg<3;++leg) {
      const auto n=leg==2?geometry.hidden:geometry.intermediate,k=leg==2?geometry.intermediate:geometry.hidden;
      experts.shared[leg]=fragment.input(glm_moe_storage::packed_description(n,k,1,
          glm_moe_storage::Kind::kFloat8Block128,false));
      experts.banks[leg]=fragment.input(glm_moe_storage::packed_description(n,k,geometry.slots,
          glm_moe_storage::Kind::kFloat8Block128,true));
    }
  }
  target::Result append(Value selection) {
    return target::append(fragment,hidden,residual,main,selection,cursor,parameters,experts);
  }
};

TEST(GlmProOrdinaryEarlyHistoryPlan,UsesRealPositiveSelectionWidthsWithoutPadding) {
  std::optional<std::size_t> nodes;
  for(const auto width:std::array<std::int64_t,4>{1,2,4,2048}) {
    DescriptionFixture fixture;auto &f=fixture.fragment;
    const auto ids=f.input(sem::dense({width},c::DataType::kInt32));
    const auto result=fixture.append(ids);
    EXPECT_EQ(result.ports.selection.ordinal(),ids.ordinal());
    EXPECT_EQ(f.description(result.ports.attention.selected_full),sem::dense({width,576}));
    EXPECT_EQ(f.description(result.ports.attention.selected_value),sem::dense({width,512}));
    EXPECT_EQ(f.description(result.main),sem::dense({2054,576}));
    EXPECT_EQ(f.description(result.cursor),sem::dense({1},c::DataType::kInt32));
    if(nodes)EXPECT_EQ(f.nodes.size(),*nodes);else nodes=f.nodes.size();
    f.outputs={result.main,result.cursor,result.normalized_tap};
    const auto graph=sem::make_graph(f);
    EXPECT_EQ(graph.outputs.size(),3u);
  }
}

TEST(GlmProOrdinaryEarlyHistoryPlan,RejectsInvalidSelectionBeforeAddingModelNodes) {
  const std::array descriptions{sem::dense({2049},c::DataType::kInt32),
      sem::dense({1,1},c::DataType::kInt32),sem::dense({1},c::DataType::kFloat32)};
  for(const auto &description:descriptions) {
    DescriptionFixture fixture;const auto ids=fixture.fragment.input(description);
    const auto nodes=fixture.fragment.nodes.size();
    EXPECT_THROW(fixture.append(ids),std::runtime_error);
    EXPECT_EQ(fixture.fragment.nodes.size(),nodes);
  }
}
} // namespace

#if defined(HYLE_TEST_GLM_STATE_CUDA)
namespace {
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace shared=glm_shared_vocabulary;
namespace original=glm_pro_target_roles;
namespace expert=glm_original_moe_roles;
namespace supply=glm_original_moe_resident_weights;
namespace ref=glm_pro_mtp_reference;
namespace d=ref::detail;
namespace lm=glm_shared_vocabulary_reference;
namespace index_ref=glm_original_indexer_reference;
using shared::take;using shared::check;
using Bytes=std::vector<std::byte>;
constexpr std::int64_t kEarlyCapacity=8;
constexpr auto kGeometry=glm_moe_geometry::kPro;

Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());check(take(lease.view(),"ordinary readonly view").read(result),"ordinary readonly bytes");return result;
}
template<class T> std::vector<T> values(const c::ReadLease &lease) {
  const auto raw=read(lease);if(raw.size()%sizeof(T))throw std::runtime_error("ordinary typed result extent");
  std::vector<T> result(raw.size()/sizeof(T));std::memcpy(result.data(),raw.data(),raw.size());return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> bytes) {
  auto bank=take(m::Buffer::create(owner.ref(),bytes.size(),256),"ordinary immutable backing");
  check(bank.view().write(bytes),"ordinary one-time upload");return take(owner.adopt_buffer(std::move(bank)),"ordinary readonly permission");
}
std::string directory(const char *name) {
#if defined(_WIN32)
  char *value=nullptr;std::size_t size=0;if(_dupenv_s(&value,&size,name))throw std::runtime_error("ordinary source environment");
  const std::unique_ptr<char,decltype(&std::free)> guard(value,&std::free);return value?value:"";
#else
  const auto value=std::getenv(name);return value?value:"";
#endif
}
void verified() {if(::testing::Test::HasFailure())throw std::runtime_error("ordinary local reference failed before publication");}
std::vector<std::uint16_t> fixture(std::size_t size,std::int32_t token,std::int32_t position,std::size_t salt) {
  std::vector<std::uint16_t> result(size);
  for(std::size_t i=0;i<size;++i)result[i]=index_ref::detail::bfloat(float(int((i*17+std::size_t(token)*3+std::size_t(position)*5+salt)%41)-20)/32.F);
  return result;
}
struct RawIndexRole final {std::size_t offset,bytes;const char *sha;};
// Exact slices of the single original layer74 bundle, authenticated at fixed
// aca966 revision. These are not NextN78 parameters with matching shapes.
constexpr std::array<RawIndexRole,7> kIndexRaw{{
    {2240,256,"085afd0cfb3aea6a036734f480cc9ea94c02ef982a8a330fdd60e87d2f8fa343"},
    {2496,256,"7cd7e04706211a3590d72f5f593eb228fc712d38e15eb2b098ac4d1c5420d8d0"},
    {2752,393216,"cf643d901737dc057b6f0c99c73c992942c15f465c3871a3812311ef3127955b"},
    {395968,786432,"f72d19751b3c825990684f03d1953a32d60f6e3d37f91dec2588858dcb05b957"},
    {0,192,"43e6d7e71e9308fa6baec88cd299bfb909b6d0cbbfa473e9c7f3404e0909dc35"},
    {1182400,8388608,"1e2b4f3b88d7bfaf93317d5ccee92f2da455c4119fd6a814f6ed73e8b7169c06"},
    {192,2048,"6c73a0a1b5098aaf8dfb92e2d2a19e2276a1488cfffe3499d70b0bb1dd4620a3"}}};
struct IndexSource final {
  m::FileMapping mapping;
  std::array<m::Buffer,2> canonical;
  std::array<shared::ReadonlyRole,5> roles; // affine bias/norm, weights, key, query
  std::span<const std::byte> raw(std::size_t role) const {
    const auto &s=kIndexRaw.at(role);return take(mapping.bytes(),"original74 mapped slice").subspan(s.offset,s.bytes);
  }
  static IndexSource load(const std::filesystem::path &path,const c::StorageOwner &owner) {
    IndexSource result;result.mapping=take(m::FileMapping::open(path.string().c_str()),"original74 mmap");
    const auto whole=take(result.mapping.bytes(),"original74 bundle");
    if(whole.size()!=9571008||shared::detail::mapped_sha256(whole)!="3354c6e1b603b5d211cd32084d71f94ef5289142f2c7f2490f89c11ddbbaf444")
      throw std::runtime_error("original74 bundle identity differs");
    for(std::size_t i=0;i<kIndexRaw.size();++i)if(shared::detail::mapped_sha256(result.raw(i))!=kIndexRaw[i].sha)
      throw std::runtime_error("original74 individual raw role identity differs");
    for(std::size_t i=0;i<2;++i) {
      const auto widened=d::widen(d::bfloats(result.raw(i)));
      result.roles[i]={i?"original74 k_norm.weight":"original74 k_norm.bias",sem::dense({128}),owner,upload(owner,std::as_bytes(std::span(widened)))};
    }
    result.roles[2]={"original74 weights_proj BF16",sem::dense({32,6144},c::DataType::kBFloat16),owner,upload(owner,result.raw(2))};
    for(std::size_t i=0;i<2;++i) {
      const auto n=i?4096:128,k=i?2048:6144;const auto code=3+i*2,scale=code+1;
      const auto desc=take(c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128),"original74 E4/SF carrier");
      result.canonical[i]=take(m::Buffer::create(m::system_resource(),take(desc.required_bytes(),"original74 carrier bytes"),64),"original74 unchanged carrier");
      auto mapped=take(result.mapping.view(),"original74 raw mapping");
      const auto source=[&](std::size_t slot,const c::TensorDesc &type) {
        const auto &range=kIndexRaw[slot];return take(c::ConstTensorView::create(type,
            take(mapped.subview(range.offset,range.bytes),"original74 raw range")),"original74 raw tensor");
      };
      auto output=take(c::TensorView::create(desc,result.canonical[i].view()),"original74 import target");
      check(c::import_float8_block128_weight(source(code,sem::dense({n,k},c::DataType::kFloat8E4M3FN)),
          source(scale,sem::dense({(n+127)/128,k/128})),output),"original74 exact E4/SF import");
      const auto bytes=take(std::as_const(result.canonical[i]).view().map_host(),"original74 carrier readonly map");
      const auto payload=bytes.bytes();
      if(!std::equal(result.raw(code).begin(),result.raw(code).end(),payload.begin())||
          !std::equal(result.raw(scale).begin(),result.raw(scale).end(),payload.begin()+std::ptrdiff_t(result.raw(code).size())))
        throw std::runtime_error("original74 import changed E4/SF bytes");
      result.roles[3+i]={i?"original74 wq_b E4/SF":"original74 wk E4/SF",desc,owner,upload(owner,payload)};
    }
    return result;
  }
};

struct External final {Value value;shared::ReadonlyRole role;};
struct Recipe final {
  sem::Fragment fragment{.profile=&d5_moe_semantic_fragments::profiles::glm53_official,
      .region=0x474c4d4541524c59ULL,.quantized=true};
  std::vector<External> constants;
  std::vector<std::pair<Value,std::size_t>> observations;
  Value bind(const shared::ReadonlyRole &role) {
    const auto value=fragment.constant(role.description,{});constants.push_back({value,role});return value;
  }
  std::size_t expose(Value value) {
    const auto found=std::find_if(observations.begin(),observations.end(),[&](const auto &v){return v.first.ordinal()==value.ordinal();});
    if(found!=observations.end())return found->second;
    const auto port=fragment.outputs.size();fragment.outputs.push_back(fragment.one(c::Operation(c::Copy{}),{value}));
    observations.push_back({value,port});return port;
  }
  void observe(Value &value) {
    if(std::none_of(fragment.inputs.begin(),fragment.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();}))expose(value);
  }
};
struct Run final {
  Recipe recipe;sem::GraphCase logical;
  std::unique_ptr<model_graph_run::ColdProgram> program;
  g::Invocation invocation;c::LeasePool pool;
  std::vector<c::ReadLease> literals;
  std::optional<model_graph_run::InputPlan> resident_inputs;
  std::array<g::WeightBinding,4> resident_bindings{};
  std::size_t submissions=0;
  static Run prepare(Recipe recipe,c::Context &context,const c::StorageOwner &owner,std::uint32_t generations=4) {
    Run result;result.recipe=std::move(recipe);auto &f=result.recipe.fragment;result.logical=sem::make_graph(f);
    const std::array<g::StorageDomain,1> domains{{{.resource=owner.ref(),.owner=owner}}};
    hyle::inference::GraphBindings bindings(domains);
    for(std::size_t i=0;i<f.constants.size();++i) {
      const auto found=std::find_if(result.recipe.constants.begin(),result.recipe.constants.end(),[&](const External &e){return e.value.ordinal()==f.constants[i].ordinal();});
      if(found!=result.recipe.constants.end())check(bindings.bind_constant(take(found->role.binding(result.logical.constants[i]),"original ready parameter")),"ordinary original constant binding");
      else {
        result.literals.push_back(upload(owner,f.constant_bytes[i]));
        check(bindings.bind_constant({.value=result.logical.constants[i],.tensor=take(result.literals.back().tensor(f.description(f.constants[i])),"cold literal"),.owner=owner,.lease=result.literals.back()}),"ordinary literal binding");
      }
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*f.profile),"ordinary original profile"),f.phase,{0x48594c454d4f444cULL,f.region,1}};
    auto compiled=take(g::cuda::compile(result.logical.graph,context,owner,bindings.options({.workload=workload})),"ordinary74/77 Graph compile");
    std::size_t bytes=0;
    std::vector<model_graph_run::Port> inputs,outputs;
    for(std::size_t i=0;i<f.inputs.size();++i)inputs.push_back({result.logical.inputs[i],f.description(f.inputs[i])});
    for(std::size_t i=0;i<f.outputs.size();++i) {
      outputs.push_back({result.logical.outputs[i],f.description(f.outputs[i])});
      bytes+=(take(f.description(f.outputs[i]).required_bytes(),"ordinary output extent")+255)/256*256;
    }
    result.program=model_graph_run::ColdProgram::create(std::move(compiled),inputs,outputs);
    result.pool=take(c::LeasePool::create(owner.ref(),bytes*generations,
        {.max_blocks=std::uint32_t(f.outputs.size())*generations,.backing_alignment=256}),"finite ordinary result generations");
    result.invocation=take(result.program->compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"ordinary Invocation");return result;
  }
  std::vector<c::ReadLease> execute(c::Queue &queue,std::span<const c::ReadLease> input,
      const model_graph_run::InputPlan *input_plan=nullptr,const std::function<void(g::Invocation&)> &binder={}) {
    return model_graph_run::execute(*program,invocation,pool,queue,input,submissions,input_plan,binder);
  }
};
c::ReadLease observed(const Recipe &r,std::span<const c::ReadLease> inputs,std::span<const c::ReadLease> outputs,Value value) {
  const auto &f=r.fragment;const auto out=std::find_if(r.observations.begin(),r.observations.end(),[&](const auto &v){return v.first.ordinal()==value.ordinal();});
  if(out!=r.observations.end())return outputs[out->second];
  const auto in=std::find_if(f.inputs.begin(),f.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
  if(in==f.inputs.end()||std::size_t(in-f.inputs.begin())>=inputs.size())throw std::runtime_error("ordinary observer lost real producer");
  return inputs[std::size_t(in-f.inputs.begin())];
}
struct IndexProgram final {Recipe recipe;model::IndexPorts ports;Value cursor;Run run;};
IndexProgram prepare_index(const IndexSource &source,c::Context &context,const c::StorageOwner &owner,std::int32_t position) {
  IndexProgram result;auto &r=result.recipe;auto &f=r.fragment;
  const auto input=f.input(sem::dense({1,6144},c::DataType::kBFloat16)),query=f.input(sem::dense({1,2048},c::DataType::kBFloat16));
  const auto codes=f.input(sem::dense({kEarlyCapacity,128},c::DataType::kFloat8E4M3FN));
  const auto scales=f.input(sem::dense({kEarlyCapacity,1})),cursor=f.input(sem::dense({1},c::DataType::kInt32));
  model::Parameters p;p.index_bias=r.bind(source.roles[0]);p.index_norm=r.bind(source.roles[1]);p.index_weights=r.bind(source.roles[2]);
  p.index_key=r.bind(source.roles[3]);p.index_query=r.bind(source.roles[4]);
  auto &ports=result.ports;ports.history=model::append_index_history(f,input,cursor,codes,scales,p);
  ports.query=model::fp8(f,query,p.index_query);
  ports.rope=model::rotate(f,model::reshape(f,ports.query.stored,{1,32,128}),cursor);
  ports.encoded=model::codec::quantize(f,model::reshape(f,model::cast(f,ports.rope.stored,c::DataType::kFloat32),{32,128}),32);
  ports.weights_input=input;ports.weights=f.one(c::model_ops::dense_linear_f32(),{input,p.index_weights});
  ports.scores=model::codec::score_encoded(f,ports.history.codes,ports.history.scales,ports.encoded,ports.weights,position+1);
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(position+1),.stable=true}),{ports.scores},2);
  ports.ids=model::contiguous(f,model::reshape(f,top[1],{position+1}));
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),{f.one(c::Operation(c::Concat{.axis=0}),{cursor,one})});
  result.cursor=f.one(c::Operation(c::Gather{}),{scanned,one});
  r.expose(ports.history.codes);r.expose(ports.history.scales);r.expose(ports.ids);r.expose(result.cursor);
  ports.visit_values([&](Value &v){r.observe(v);});result.run=Run::prepare(r,context,owner);return result;
}

struct TargetSource final {
  original::Bound parameters;expert::Roles experts;
  std::array<shared::ReadonlyRole,2> prepared;c::LeasePool prepared_pool;
  supply::CudaSupply cuda_supply;
  m::FileMapping head_mapping;
  shared::ReadonlyRole head;
  std::vector<expert::Selection> selections;
  bool moe_full_checked=false;
};
TargetSource prepare_target(c::Context &context,const c::StorageOwner &owner,c::Queue &queue,
    const c::cuda::MemoryResource &device) {
  const auto static_dir=directory("HYLE_GLM_PRO_TARGET_DIR"),expert_dir=directory("HYLE_GLM_PRO_TARGET_MOE_DIR"),vocab_dir=directory("HYLE_GLM_PRO_VOCAB_DIR");
  if(static_dir.empty()||expert_dir.empty()||vocab_dir.empty())throw std::runtime_error("ordinary Pro requires its cached original77 roles, experts and actual untied head");
  TargetSource result;
  result.parameters=original::bind(std::make_shared<const original::Data>(original::Data::load(static_dir)),owner);
  result.experts=expert::bind(std::make_shared<const expert::Data>(expert::Data::load(expert_dir,original::kMoeIndexSha256,kGeometry)),owner);
  {
    Recipe recipe;const auto kv=model::append_prepare_kv(recipe.fragment,recipe.bind(result.parameters.at(original::Role::kKvB)));
    recipe.expose(kv.key);recipe.expose(kv.value);auto cold=Run::prepare(std::move(recipe),context,owner,1);
    const auto output=cold.execute(queue,{});
    EXPECT_TRUE(ref::expect_prepared_kv_matrix(result.parameters.source->matrix(original::Role::kKvB),
        values<std::uint16_t>(output[0]),values<std::uint16_t>(output[1])));verified();
    result.prepared[0]={"original77 prepared key",sem::dense({64,512,192},c::DataType::kBFloat16),owner,output[0]};
    result.prepared[1]={"original77 prepared value",sem::dense({64,256,512},c::DataType::kBFloat16),owner,output[1]};
    result.prepared_pool=std::move(cold.pool);
  }
  result.parameters.roles[original::index(original::Role::kKvB)].lease={};
  result.cuda_supply=take(supply::CudaSupply::create(context,owner,device,kGeometry),"ordinary original77 double expert generation");
  result.head_mapping=take(m::FileMapping::open((std::filesystem::path(vocab_dir)/"head.bf16").string().c_str()),"original untied Pro head mmap");
  const auto head=take(result.head_mapping.bytes(),"original Pro head bytes");
  if(head.size()!=std::size_t(154880)*6144*2||shared::detail::mapped_sha256(head)!=glm_pro_original_roles::kVocabularyPins.sha256[1])
    throw std::runtime_error("ordinary untied Pro head identity differs");
  result.head={"original untied154880 Pro head",sem::dense({154880,6144},c::DataType::kBFloat16),owner,upload(owner,head)};
  return result;
}
struct TargetProgram final {
  Recipe recipe;target::Result result;
  model_resident_fragment::ResidentDraft<model::moe::Boundary,1> plan;
  std::vector<Run> phases;
  std::vector<c::ReadLease> execute(TargetSource &source,c::Queue &queue,std::span<const c::ReadLease> input) {
    auto &f=recipe.fragment;if(input.size()+4!=f.inputs.size())throw std::runtime_error("ordinary77 actual input extent");
    std::vector<c::ReadLease> live(f.descriptions.size());
    for(std::size_t i=0;i<input.size();++i)live[f.inputs[i].ordinal()]=input[i];
    std::array<std::vector<c::ReadLease>,2> completed;
    for(std::size_t i=0;i<phases.size();++i) {
      const auto &phase=plan.phases[i];auto &run=phases[i];
      std::optional<c::cuda::WeightBatch> resident;
      const model_graph_run::InputPlan *input_plan=nullptr;
      std::function<void(g::Invocation&)> binder;
      if(phase.resident_stage) {
        const auto &frontier=plan.routers.at(*phase.resident_stage);
        source.selections.push_back(expert::select(values<std::int32_t>(live.at(frontier.ids.ordinal())),kGeometry));
        resident.emplace(take(source.cuda_supply.upload((source.selections.size()-1)%2,source.selections.back(),*source.experts.source,queue),"actual ordinary77 selected expert generation"));
        input_plan=&*run.resident_inputs;
        binder=[&](g::Invocation &active){check(g::cuda::bind(active,run.resident_bindings,*resident),"same original expert/map generation");};
      }
      std::vector<c::ReadLease> arguments;for(const auto value:phase.original_inputs)arguments.push_back(live[value.ordinal()]);
      completed[i]=run.execute(queue,arguments,input_plan,binder);
      for(std::size_t j=0;j<phase.original_outputs.size();++j)live[phase.original_outputs[j].ordinal()]=completed[i][j];
      if(resident) {
        const auto report=take(resident->report(),"actual expert transfer report");
        EXPECT_EQ(report.logical_bytes,source.cuda_supply.layout().logical_bytes);EXPECT_EQ(report.completed_bytes,report.submitted_bytes);
        EXPECT_FALSE(report.copied_bytes_unknown);
      }
    }
    std::vector<c::ReadLease> output;for(const auto where:plan.full_outputs)output.push_back(completed[where.phase][where.port]);return output;
  }
};
TargetProgram prepare_forward(TargetSource &source,c::Context &context,const c::StorageOwner &owner,std::int32_t position) {
  TargetProgram result;auto &r=result.recipe;auto &f=r.fragment;
  const auto hidden=f.input(sem::dense({1,6144},c::DataType::kBFloat16)),residual=f.input(sem::dense({1,6144},c::DataType::kBFloat16));
  const auto main=f.input(sem::dense({kEarlyCapacity,576})),cursor=f.input(sem::dense({1},c::DataType::kInt32));
  const auto ids=f.input(sem::dense({position+1},c::DataType::kInt32));
  const auto bank=[&](original::Role role){return r.bind(source.parameters.at(role));};
  target::Parameters p;p.input_norm=bank(original::Role::kInputNorm);p.query_a=bank(original::Role::kQueryA);
  p.query_norm=bank(original::Role::kQueryNorm);p.target_norm=bank(original::Role::kTargetNorm);
  p.attention={bank(original::Role::kKvNorm),bank(original::Role::kKvA),bank(original::Role::kOutput),bank(original::Role::kPostNorm),
      bank(original::Role::kQueryB),r.bind(source.prepared[0]),r.bind(source.prepared[1])};
  model::moe::Parameters moe;moe.router=r.bind(source.experts.router);moe.bias=r.bind(source.experts.bias);
  for(std::size_t leg=0;leg<3;++leg) {
    moe.shared[leg]=r.bind(source.experts.shared[leg]);moe.banks[leg]=f.input(expert::resident_description(static_cast<expert::Leg>(leg),kGeometry));
  }
  moe.map=f.input(sem::dense({256},c::DataType::kInt32));
  result.result=target::append(f,hidden,residual,main,ids,cursor,p,moe);
  r.expose(result.result.main);r.expose(result.result.cursor);r.expose(result.result.normalized_tap);
  // The resident map is a real late-bound input, not one of target_args.
  // Retain its actual device output for the same-generation mapping oracle.
  r.expose(moe.map);
  result.result.ports.visit_values([&](Value &v){r.observe(v);});result.result.feed_forward.ports.visit_values([&](Value &v){r.observe(v);});
  result.plan=model_resident_fragment::split(f,std::array{result.result.feed_forward.boundary},0x474c4d4541525030ULL);
  std::vector<shared::ReadonlyRole> constants;
  for(std::size_t i=0;i<f.constants.size();++i) {
    const auto found=std::find_if(r.constants.begin(),r.constants.end(),[&](const External &e){return e.value.ordinal()==f.constants[i].ordinal();});
    if(found!=r.constants.end())constants.push_back(found->role);
    else constants.push_back({"ordinary77 cold literal",f.description(f.constants[i]),owner,upload(owner,f.constant_bytes[i])});
  }
  for(const auto &phase:result.plan.phases) {
    Recipe local;local.fragment=phase.fragment;
    for(std::size_t i=0;i<phase.original_constants.size();++i) {
      const auto original_value=phase.original_constants[i];const auto found=std::find_if(f.constants.begin(),f.constants.end(),[&](Value v){return v.ordinal()==original_value.ordinal();});
      if(found==f.constants.end())throw std::runtime_error("ordinary77 cut lost constant");
      local.constants.push_back({local.fragment.constants[i],constants[std::size_t(found-f.constants.begin())]});
    }
    result.phases.push_back(Run::prepare(std::move(local),context,owner));
    if(phase.resident_stage) {
      auto &run=result.phases.back();
      const auto &frontier=result.plan.routers.at(*phase.resident_stage);
      const std::array banks{frontier.banks[0],frontier.banks[1],frontier.banks[2],frontier.map};
      std::array<std::size_t,4> dynamic{};
      for(std::size_t plane=0;plane<banks.size();++plane) {
        const auto found=std::find_if(phase.original_inputs.begin(),phase.original_inputs.end(),
            [&](Value v){return v.ordinal()==banks[plane].ordinal();});
        if(found==phase.original_inputs.end())throw std::runtime_error("ordinary77 cut lost expert input");
        dynamic[plane]=std::size_t(found-phase.original_inputs.begin());
        run.resident_bindings[plane]={run.program->input_ports[dynamic[plane]].id,supply::kBankIds[plane]};
      }
      run.resident_inputs=run.program->input_plan(dynamic);
    }
  }
  return result;
}

template<class Actual>
void observe_index(const IndexSource &source,const model::IndexPorts &ports,Actual actual,std::span<const std::byte> old_codes,std::span<const std::byte> old_scales,std::int32_t position) {
  const auto fp=[&](Value v){return d::floats(actual(v));};const auto bf=[&](Value v){return d::bfloats(actual(v));};
  const auto projection=[&](const model::ProjectionPorts &p,std::size_t raw,std::size_t n,std::size_t k) {
    EXPECT_TRUE(glm_original_attention_reference::expect_fp8_matrix(fp(p.raw),bf(p.input),source.raw(raw),source.raw(raw+1),n,k,"original74 actual FP8 projection",true));
    EXPECT_TRUE(index_ref::expect_bfloat_store(bf(p.stored),fp(p.raw),"original74 BF16 projection store"));
  };
  projection(ports.history.key,3,128,6144);projection(ports.query,5,4096,2048);
  EXPECT_TRUE(lm::expect_widen(fp(ports.history.key_norm.input),bf(ports.history.key.stored)));
  EXPECT_TRUE(d::local(fp(ports.history.key_norm.raw),index_ref::layer_norm(d::widen(bf(ports.history.key.stored)),
      d::bfloats(source.raw(1)),d::bfloats(source.raw(0))),"original74 real affine norm"));
  EXPECT_TRUE(index_ref::expect_bfloat_store(bf(ports.history.key_norm.stored),fp(ports.history.key_norm.raw),"original74 key norm store"));
  EXPECT_TRUE(d::same(bf(ports.history.rope.input),bf(ports.history.key_norm.stored),"original74 key rotary source"));
  EXPECT_TRUE(d::rope(ports.history.rope,position,128,actual));
  EXPECT_TRUE(d::quantized(ports.history.encoded,bf(ports.history.rope.stored),1,actual));
  EXPECT_TRUE(d::same(bf(ports.rope.input),bf(ports.query.stored),"original74 query rotary source"));
  EXPECT_TRUE(d::rope(ports.rope,position,128,actual));EXPECT_TRUE(d::quantized(ports.encoded,bf(ports.rope.stored),32,actual));
  EXPECT_TRUE(lm::expect_logits(fp(ports.weights),bf(ports.weights_input),source.raw(2),32,6144,true));
  auto expected_codes=Bytes(old_codes.begin(),old_codes.end()),expected_scales=Bytes(old_scales.begin(),old_scales.end());
  const auto row=actual(ports.history.encoded.codes),scale=actual(ports.history.encoded.scale);
  std::copy(row.begin(),row.end(),expected_codes.begin()+std::ptrdiff_t(position*128));
  std::copy(scale.begin(),scale.end(),expected_scales.begin()+std::ptrdiff_t(position*4));
  EXPECT_EQ(actual(ports.history.codes),expected_codes);EXPECT_EQ(actual(ports.history.scales),expected_scales);
  model::codec::Encoded encoded;encoded.codes=d::codes(actual(ports.encoded.codes));encoded.scales=fp(ports.encoded.scale);
  const auto scores=fp(ports.scores);
  EXPECT_TRUE(d::local(scores,model::codec::score_reference(d::codes(actual(ports.history.codes)),fp(ports.history.scales),encoded,fp(ports.weights),position+1),"original74 actual cache scores"));
  EXPECT_EQ(scores.size(),std::size_t(position+1));EXPECT_TRUE(d::selection(scores,d::integers(actual(ports.ids))));verified();
}

// Reuse the existing independent original-Pro MoE observation wiring; no
// model execution, routing policy or MTP controller is copied here.
template<class Owner,class ReadValue>
bool observe_moe(Owner &source,const glm_original_moe_fragment::Result &result,
                 ReadValue actual) {
  namespace ref=glm_original_moe_reference;
  namespace boundary=glm_original_indexer_reference;
  namespace lm=glm_shared_vocabulary_reference;
  namespace bytes=glm_pro_mtp_reference::detail;
  constexpr auto geometry=glm_moe_geometry::kPro;
  if(!source.experts.source||source.experts.source->model()!=geometry||source.selections.empty()) {
    ADD_FAILURE()<<"original Pro MoE observer requires the actual Pro source and resident selection";
    return false;
  }
  const auto &data=*source.experts.source;
  const auto &ports=result.ports;
  const auto fp=[&](Value value){return bytes::floats(actual(value));};
  const auto bf=[&](Value value){return bytes::bfloats(actual(value));};
  const auto integer=[&](Value value){return bytes::integers(actual(value));};
  const auto input=bf(ports.input);
  const auto global=integer(ports.ids),local=integer(ports.local_ids),map=integer(ports.map);
  const auto logits=fp(ports.logits),scores=fp(ports.scores),selection=fp(ports.selection);
  const auto selected=fp(ports.selected_scores),sum=fp(ports.sum),denominator=fp(ports.denominator);
  const auto normalized=fp(ports.normalized),weights=fp(ports.weights);
  if(!ref::expect_router(logits,input,data.router_bytes(),geometry)||
      !ref::expect_routing({logits,scores,selection,selected,sum,denominator,normalized,weights,global},
          data.bias_bytes(),geometry))return false;
  const auto &resident=source.selections.back();
  if(map!=resident.map) {
    ADD_FAILURE()<<"original Pro MoE actual map differs from the same resident generation";
    return false;
  }
  if(!ref::expect_mapping(global,map,local,resident.globals,geometry))return false;
  const bool full=!source.moe_full_checked;
  const auto matrix=[&](expert::Leg leg,std::optional<std::int32_t> selected_expert=std::nullopt) {
    const auto [outputs,width]=expert::geometry(leg,geometry);
    return ref::Fp8Matrix{
        selected_expert?data.expert_bytes(leg,*selected_expert,false):data.shared_bytes(leg,false),
        selected_expert?data.expert_bytes(leg,*selected_expert,true):data.shared_bytes(leg,true),
        std::size_t(outputs),std::size_t(width)};
  };
  const auto branch=[&](const glm_original_moe_fragment::ProjectionPorts &projection,bool routed) {
    if(!projection.gate_raw||!projection.up_raw||!projection.down_raw||projection.down_stored)
      throw std::runtime_error("original Pro E4 observer requires real F32 projection outputs");
    const auto gate=fp(*projection.gate_raw),up=fp(*projection.up_raw),down=fp(*projection.down_raw);
    const auto stored_gate=bf(projection.gate_stored),stored_up=bf(projection.up_stored);
    const auto activated=bf(projection.activation_stored);
    if(routed) {
      if(!ref::expect_selected_projection(gate,input,global,
              [&](std::int32_t id){return matrix(expert::Leg::kGate,id);},
              "original Pro selected gate",full,geometry)||
          !ref::expect_selected_projection(up,input,global,
              [&](std::int32_t id){return matrix(expert::Leg::kUp,id);},
              "original Pro selected up",full,geometry)||
          !ref::expect_selected_projection(down,activated,global,
              [&](std::int32_t id){return matrix(expert::Leg::kDown,id);},
              "original Pro selected down",full,geometry))return false;
    } else {
      if(!ref::expect_fp8_projection(gate,input,matrix(expert::Leg::kGate),"original Pro shared gate",full)||
          !ref::expect_fp8_projection(up,input,matrix(expert::Leg::kUp),"original Pro shared up",full)||
          !ref::expect_fp8_projection(down,activated,matrix(expert::Leg::kDown),"original Pro shared down",full))return false;
    }
    const auto activation=fp(projection.activation_raw);
    // kPro has no SwiGLU limit: these ports must retain the un-clipped actual
    // BF16 predecessors. The same helper explicitly checks that profile.
    return boundary::expect_bfloat_store(stored_gate,gate,"Pro actual gate BF16 store")&&
        boundary::expect_bfloat_store(stored_up,up,"Pro actual up BF16 store")&&
        ref::expect_activation(fp(projection.gate_clipped),fp(projection.up_clipped),
            activation,stored_gate,stored_up,geometry)&&
        boundary::expect_bfloat_store(activated,activation,"Pro actual SwiGLU BF16 store");
  };
  if(!branch(ports.routed,true)||!branch(ports.shared,false))return false;
  const auto weighted=fp(ports.weighted_raw);
  const auto weighted_stored=bf(ports.weighted_stored);
  const auto routed_sum=fp(ports.routed_sum_raw);
  const auto routed_stored=bf(ports.routed_sum_stored);
  const auto shared_stored=bf(ports.shared_stored);
  const auto merged=fp(ports.merge_raw);
  const auto merged_stored=bf(ports.merge_stored);
  if(!ref::expect_weighted_down(weighted,fp(ports.routed.down_raw.value()),weights,geometry)||
      !boundary::expect_bfloat_store(weighted_stored,weighted,"Pro actual weighted down BF16 store")||
      !ref::expect_routed_sum(routed_sum,weighted_stored,geometry)||
      !boundary::expect_bfloat_store(routed_stored,routed_sum,"Pro actual routed sum BF16 store")||
      !boundary::expect_bfloat_store(shared_stored,fp(ports.shared.down_raw.value()),"Pro actual shared down BF16 store")||
      !lm::expect_sum(merged,bytes::widen(routed_stored),bytes::widen(shared_stored))||
      !boundary::expect_bfloat_store(merged_stored,merged,"Pro actual shared/routed merge BF16 store")||
      !bytes::same(bf(result.output),merged_stored,"Pro actual MoE returned BF16 merge"))return false;
  source.moe_full_checked=true;
  return true;
}

struct SharedSelection final {c::ReadLease ids;std::int32_t position;std::uint64_t epoch;};
void validate_selection(std::span<const std::int32_t> ids,std::int32_t position,std::uint64_t epoch,std::uint64_t expected_epoch) {
  if(position<0||position>=kEarlyCapacity||epoch!=expected_epoch||ids.size()!=std::size_t(position+1))
    throw std::runtime_error("ordinary shared index epoch/visible extent differs");
  std::array<bool,kEarlyCapacity> seen{};
  for(const auto id:ids) {
    if(id<0||id>position||seen[std::size_t(id)])throw std::runtime_error("ordinary shared index future/duplicate ID");
    seen[std::size_t(id)]=true;
  }
}
const c::ReadLease &carry_shared_index(const SharedSelection &selection,std::uint64_t expected_epoch,std::int32_t layer) {
  if(layer!=75&&layer!=76)throw std::runtime_error("ordinary shared index layer boundary");
  validate_selection(values<std::int32_t>(selection.ids),selection.position,selection.epoch,expected_epoch);return selection.ids;
}
enum StateEntry:std::size_t {kCodes,kScales,kMain,kCursor,kIndexCursor,kTap,kPending,kFrontier,kStateEntries};
struct Frontier final {std::int32_t position,index_position,pending,selection_rows;std::uint64_t epoch;};
struct State final {
  c::StateSchema schema;c::StateStore store;std::array<c::StateEntryId,kStateEntries> entries;
  c::ReadLease member(const c::StateVersion &version,StateEntry entry) const {return take(version.member(entries[entry]),"ordinary complete state member");}
  Frontier frontier(const c::StateVersion &version) const {
    const auto raw=read(member(version,kFrontier));if(raw.size()!=sizeof(Frontier))throw std::runtime_error("ordinary frontier metadata extent");
    Frontier result{};std::memcpy(&result,raw.data(),sizeof(result));return result;
  }
  static State create(const c::StorageOwner &owner,std::size_t backing_budget) {
    const std::array descriptions{sem::dense({kEarlyCapacity,128},c::DataType::kFloat8E4M3FN),sem::dense({kEarlyCapacity,1}),
        sem::dense({kEarlyCapacity,576}),sem::dense({1},c::DataType::kInt32),sem::dense({1},c::DataType::kInt32),
        sem::dense({1,6144},c::DataType::kBFloat16),sem::dense({1},c::DataType::kInt32)};
    State result;std::vector<c::StateEntrySpec> specs;
    for(const auto &description:descriptions)specs.push_back({.description=description,.resource=owner.ref()});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Frontier)});
    result.schema=take(c::StateSchema::create(specs),"ordinary74/77 complete fixed state schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=6,.max_drafts=2,.max_retained_blocks=128,
        .max_backing_bytes=backing_budget,.max_host_snapshot_bytes=4096}),"bounded ordinary74/77 StateStore");
    for(std::size_t i=0;i<entries_count();++i)result.entries[i]=take(result.schema.entry(i),"ordinary state ID");return result;
  }
  static constexpr std::size_t entries_count(){return kStateEntries;}
  c::StateVersion initial(const c::StorageOwner &owner) {
    std::vector<c::StateMember> members;
    for(std::size_t i=0;i<kFrontier;++i) {
      const auto spec=take(schema.specification(entries[i]),"initial ordinary state description");
      Bytes raw(take(spec.description.required_bytes(),"initial ordinary state bytes"));
      if(i==kCodes)std::fill(raw.begin(),raw.end(),std::byte{0x7f});
      else if(i==kScales||i==kMain) {
        std::vector<float> poison(raw.size()/4,std::bit_cast<float>(std::uint32_t{0x7fc12345}));raw=sem::bytes(poison);
      } else if(i==kPending)raw=sem::bytes(std::vector<std::int32_t>{2});
      members.push_back({entries[i],upload(owner,raw)});
    }
    const Frontier first{0,0,2,0,0};members.push_back({entries[kFrontier],{},std::as_bytes(std::span(&first,1))});
    return take(store.import(members),"complete empty ordinary74/77 Head");
  }
};
struct SharedDevice final {std::shared_ptr<c::cuda::MemoryResource> device;m::MemoryResourceRef ref() noexcept{return device->ref();}};

// The ordinary adapter forwards only the actual shared selection result. It
// does not execute or claim the omitted layer75/76 body. Their input74 and
// target76 hidden/residual boundaries below are explicit finite fixtures.
TEST(GlmProOrdinaryEarlyHistory,CudaActualLayer74IndexSuppliesShortTarget77) {
  const auto bundle=directory("HYLE_GLM_PRO_INDEX74_BUNDLE");ASSERT_FALSE(bundle.empty());
  auto context=take(g::cuda::create_context(),"ordinary Pro CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"ordinary Pro actual device"));
  auto owner=take(c::StorageOwner::adopt(context,SharedDevice{device}),"ordinary Pro original native owner");
  auto queue=take(context.create_queue(),"ordinary Pro original queue");
  const auto index_source=IndexSource::load(bundle,owner);auto target_source=prepare_target(context,owner,queue,*device);
  std::array<IndexProgram,4> index;std::array<TargetProgram,4> target;
  for(std::int32_t p=0;p<4;++p) {index[std::size_t(p)]=prepare_index(index_source,context,owner,p);target[std::size_t(p)]=prepare_forward(target_source,context,owner,p);}
  Recipe head_recipe;const auto tap=head_recipe.fragment.input(sem::dense({1,6144},c::DataType::kBFloat16));
  const auto logits=head_recipe.fragment.one(c::model_ops::dense_linear_f32(),{tap,head_recipe.bind(target_source.head)});
  const auto top=head_recipe.fragment.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
  head_recipe.expose(top[1]);head_recipe.expose(logits);head_recipe.expose(top[0]);auto head_program=Run::prepare(std::move(head_recipe),context,owner);
  std::size_t budget=head_program.pool.stats().capacity_bytes+1024*1024;
  for(std::size_t i=0;i<index.size();++i) {
    budget+=index[i].run.pool.stats().capacity_bytes;
    for(const auto &phase:target[i].phases)budget+=phase.pool.stats().capacity_bytes;
  }
  auto state=State::create(owner,budget);std::size_t stale=0,invalid=0;
  {
    auto initial=state.initial(owner);auto head=take(c::StateHead::create(state.store,initial),"ordinary atomic Head");initial={};
    for(std::int32_t p=0;p<4;++p) {
      const auto before=take(head.current(),"ordinary expected base");const auto boundary=state.frontier(before);
      ASSERT_EQ(boundary.position,p);ASSERT_EQ(boundary.index_position,p);
      const auto token=values<std::int32_t>(state.member(before,kPending));ASSERT_EQ(token,std::vector<std::int32_t>{boundary.pending});
      ASSERT_EQ(values<std::int32_t>(state.member(before,kCursor)),std::vector<std::int32_t>{p});
      ASSERT_EQ(values<std::int32_t>(state.member(before,kIndexCursor)),std::vector<std::int32_t>{p});
      std::array<Bytes,kFrontier> old;for(std::size_t i=0;i<kFrontier;++i)old[i]=read(state.member(before,static_cast<StateEntry>(i)));
      const auto index_input=fixture(6144,token[0],p,11),query_input=fixture(2048,token[0],p,17);
      const std::array index_args{upload(owner,sem::bytes(index_input)),upload(owner,sem::bytes(query_input)),
          state.member(before,kCodes),state.member(before,kScales),state.member(before,kIndexCursor)};
      auto &ip=index[std::size_t(p)];const auto io=ip.run.execute(queue,index_args);
      const auto get_index=[&](Value v){return read(observed(ip.recipe,index_args,io,v));};
      observe_index(index_source,ip.ports,get_index,old[kCodes],old[kScales],p);
      const SharedSelection selected{io[2],p,boundary.epoch};
      const auto &after75=carry_shared_index(selected,boundary.epoch,75);
      const auto &after76=carry_shared_index(selected,boundary.epoch,76);EXPECT_EQ(&after75,&after76);
      const auto ids=values<std::int32_t>(after76);EXPECT_EQ(ids.size(),std::size_t(p+1));
      EXPECT_THROW(validate_selection(ids,p,boundary.epoch+1,boundary.epoch),std::runtime_error);++invalid;
      auto future=ids;future[0]=p+1;EXPECT_THROW(validate_selection(future,p,boundary.epoch,boundary.epoch),std::runtime_error);++invalid;
      if(ids.size()>1) {auto duplicate=ids;duplicate[1]=duplicate[0];EXPECT_THROW(validate_selection(duplicate,p,boundary.epoch,boundary.epoch),std::runtime_error);++invalid;}
      EXPECT_TRUE(take(head.current(),"unchanged Head after index rejection").identity()==before.identity());
      const std::array target_args{upload(owner,sem::bytes(fixture(6144,token[0],p,23))),upload(owner,sem::bytes(fixture(6144,token[0],p,29))),
          state.member(before,kMain),state.member(before,kCursor),after76};
      auto &tp=target[std::size_t(p)];const auto output=tp.execute(target_source,queue,target_args);
      const auto get_target=[&](Value v){return read(observed(tp.recipe,target_args,output,v));};
      EXPECT_TRUE(glm_pro_target_reference::observe(*target_source.parameters.source,tp.result,get_target,p==0));
      EXPECT_TRUE(observe_moe(target_source,tp.result.feed_forward,get_target));verified();
      const std::array head_args{output[2]};const auto prediction=head_program.execute(queue,head_args);
      const auto pending=values<std::int32_t>(prediction[0]);
      const auto all_logits=values<float>(prediction[1]),maxima=values<float>(prediction[2]);
      EXPECT_TRUE(lm::expect_logits(all_logits,values<std::uint16_t>(output[2]),take(target_source.head_mapping.bytes(),"original Pro head reference"),154880,6144,p==0,pending));
      EXPECT_TRUE(lm::expect_greedy(all_logits,pending,154880,maxima));verified();
      auto draft=take(state.store.begin(before),"ordinary unpublished complete candidate");
      const std::array resources{io[0],io[1],output[0],output[1],io[3],output[2],prediction[0]};
      for(std::size_t i=0;i<resources.size();++i)check(draft.replace(state.entries[i],resources[i]),"ordinary complete successor resource");
      const Frontier next{p+1,p+1,pending.at(0),std::int32_t(ids.size()),boundary.epoch+1};
      check(draft.replace_metadata(state.entries[kFrontier],std::as_bytes(std::span(&next,1))),"ordinary common epoch witness");
      const auto candidate=take(draft.finalize(),"ordinary finalized complete state");
      EXPECT_EQ(values<std::int32_t>(state.member(candidate,kIndexCursor)),std::vector<std::int32_t>{p+1});
      EXPECT_EQ(values<std::int32_t>(state.member(candidate,kCursor)),std::vector<std::int32_t>{p+1});
      check(head.publish(before.identity(),candidate),"atomic ordinary state/pending/epoch publication");
      EXPECT_FALSE(head.publish(before.identity(),candidate).ok());++stale;
      for(std::size_t i=0;i<kFrontier;++i)EXPECT_EQ(read(state.member(before,static_cast<StateEntry>(i))),old[i]);
      EXPECT_EQ(values<std::int32_t>(selected.ids),ids); // transient same-epoch reader remains valid through publication
      std::printf("GLM Pro ordinary p=%d selected=%zu epoch=%llu next=%d indexCursor=%d targetCursor=%d\n",p,ids.size(),
          static_cast<unsigned long long>(next.epoch),next.pending,next.index_position,next.position);
    }
    const auto final=take(head.current(),"ordinary final state");EXPECT_EQ(state.frontier(final).position,4);
    EXPECT_EQ(state.frontier(final).epoch,4u);EXPECT_EQ(state.frontier(final).selection_rows,4);
  }
  check(state.store.collect(),"ordinary final retired-state collection");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);EXPECT_EQ(state.store.stats().backing_bytes,0u);
  ::testing::Test::RecordProperty("actual_positions","0,1,2,3");::testing::Test::RecordProperty("actual_selected_widths","1,2,3,4");
  ::testing::Test::RecordProperty("index74_original_roles",7);::testing::Test::RecordProperty("index74_original_bytes","9571008");
  ::testing::Test::RecordProperty("stale_publish_rejections",std::to_string(stale));::testing::Test::RecordProperty("invalid_shared_index_rejections",std::to_string(invalid));
  ::testing::Test::RecordProperty("ordinary_persistent_schema","7 actual resources: index E4 codes/F32 scales, target main F32 cache, separate target/index I32 cursors, target BF16 tap, pending I32; Frontier24 metadata");
  ::testing::Test::RecordProperty("shared_index_scope","Original layer74 real query/key/weights/QAT/score/TopK; same original I32 lease and epoch forwarded across explicit75/76 sharing boundary to target77. No layer75/76 body execution claim. Variable IDs are same-forward Frame data, not padded persistent state.");
  ::testing::Test::RecordProperty("upstream_fixture_scope","Layer74 normalized input/query-rank and layer76 target hidden/residual are finite BF16 fixtures conditioned on actual previous pending ID. Not full78-layer generation, quality, TPS, MTP or a complete prefix2048 warm run.");
  ::testing::Test::RecordProperty("state_backing_budget",std::to_string(budget));
}
} // namespace
#endif
