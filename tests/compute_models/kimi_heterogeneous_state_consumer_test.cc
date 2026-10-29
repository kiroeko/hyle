// Ordinary heterogeneous StateStore integration, using generated parameters
// at the fixed Kimi-K3 geometry. This is not original-weight model execution.
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "d5_kimi_remaining_fragments.h"
#include "kimi_original_attention_roles.h"
#include "kimi_original_attention_input.h"
#include "kimi_original_attention_reference.h"
#include "model_graph_run.h"
#include "../../engine/inference/backend.h"
#include "../../engine/inference/execution_episode.h"
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
namespace inference=hyle::inference;
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
struct Run;
using Episode=inference::ExecutionEpisode<Run,inference::SubmissionViews,0>;
struct Pending final {
  std::shared_ptr<Run> owner;
  std::shared_ptr<Episode> execution;
  // Host call ordering only; retirement is always queried from the original
  // Completion/Invocation and carrier, never inferred from this observation.
  bool observed=false;
  Pending()=default;
  Pending(std::shared_ptr<Run> run,std::shared_ptr<Episode> episode)
      :owner(std::move(run)),execution(std::move(episode)) {}
  Pending(const Pending&)=delete;
  Pending &operator=(const Pending&)=delete;
  Pending(Pending&&)=default;
  Pending &operator=(Pending&&)=delete;
  ~Pending();
};
struct Run final : std::enable_shared_from_this<Run> {
  // The accepted original Completion retains this actual Run, including its
  // stable carrier node and the Backend's real Context/Queue/storage endpoint.
  std::shared_ptr<inference::Backend> backend;
  sem::Fragment facts;sem::GraphCase logical;
  std::vector<c::ReadLease> literals;
  std::unique_ptr<model_graph_run::ColdProgram> cold_program;
  c::LeasePool pool;
  std::shared_ptr<Episode> pending;
  bool submit_attempted=false;
  inference::ProgramCarrier<Run> carrier;
  std::size_t submissions=0,source_submissions_before_host_observation=0;
  std::size_t recorded_capture_ranges=0,reused_capture_ranges=0,host_waits=0;
  std::size_t cold_capture_preparations=0,cold_recorded_ranges=0,cold_reused_ranges=0;
  double cold_capture_seconds=0;
  g::ReplayReport last_replay{};
  static sem::ValueId source_constant(sem::Fragment &f,std::vector<c::ReadLease> &supplied,
      const original_named_role_source::ReadonlyRole &role,const c::StorageOwner &owner) {
    if(f.constants.size()!=supplied.size())throw std::runtime_error("original Kimi learned constants must be the entire cold prefix");
    if(role.owner.ref()!=owner.ref())throw std::runtime_error("original Kimi constant owner differs from its cold Graph owner");
    const auto value=f.constant(role.description,{});supplied.push_back(role.lease);return value;
  }
  static std::shared_ptr<Run> prepare(sem::Fragment f,std::shared_ptr<inference::Backend> selected_backend,LiteralBank *shared=nullptr,
      std::vector<c::ReadLease> supplied_literals={}) {
    auto result=std::make_shared<Run>();result->backend=std::move(selected_backend);
    auto &context=*take(result->backend->common_context(),"Kimi original Backend Context");
    const auto &owner=result->backend->storage_owner();
    result->literals=std::move(supplied_literals);
    const auto supplied=result->literals.size();
    if(supplied>f.constants.size())throw std::runtime_error("original Kimi constant lease count differs");
    for(std::size_t i=0;i<supplied;++i) {
      const auto &description=f.description(f.constants[i]);
      if(!f.constant_bytes[i].empty()||result->literals[i].size()!=take(description.required_bytes(),"original Kimi supplied constant extent"))
        throw std::runtime_error("original Kimi constant must have its actual lease and no generated payload");
      take(result->literals[i].tensor(description),"original Kimi typed source lease");
    }
    // Explicit caller outputs keep every generated value independent of the
    // phase's mutable storage and of source/input aliases across later steps.
    for(auto &output:f.outputs)output=f.one(c::Operation(c::Copy{}),{output});
    result->logical=sem::make_graph(f);std::vector<g::ConstantBinding> constants;
    for(std::size_t i=0;i<f.constants.size();++i) {
      const auto &description=f.description(f.constants[i]);
      if(i>=supplied) {
        if(f.constant_bytes[i].empty()&&take(description.required_bytes(),"Kimi literal extent"))
          throw std::runtime_error("original Kimi constant source lease is absent");
        result->literals.push_back(shared?shared->bind(i,description,std::move(f.constant_bytes[i]),owner):upload(owner,f.constant_bytes[i]));
      }
      constants.push_back({.value=result->logical.constants[i],.tensor=take(result->literals[i].tensor(description),"Kimi typed literal"),
          .owner=owner,.lease=result->literals[i]});
    }
    f.constant_bytes.clear();
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*f.profile),"Kimi geometry profile"),f.phase,{0x48594c454d4f444cULL,f.region,1}};
    std::vector<model_graph_run::Port> inputs,outputs;
    for(std::size_t i=0;i<f.inputs.size();++i)inputs.push_back({result->logical.inputs[i],f.description(f.inputs[i])});
    for(std::size_t i=0;i<f.outputs.size();++i)outputs.push_back({result->logical.outputs[i],f.description(f.outputs[i])});
    result->cold_program=model_graph_run::ColdProgram::create(
        take(g::cuda::compile(result->logical.graph,context,owner,{.constants=constants,.workload=workload}),"Kimi ordinary Graph compile"),inputs,outputs);
    std::size_t bytes=0;
    for(std::size_t i=0;i<result->cold_program->output_count();++i)
      bytes+=(take(result->cold_program->output(i).description.required_bytes(),"Kimi output extent")+255)/256*256;
    constexpr std::uint32_t generations=8;
    result->pool=take(c::LeasePool::create(owner.ref(),bytes*generations,
        {.max_blocks=std::uint32_t(f.outputs.size())*generations,.backing_alignment=256}),"bounded Kimi output generations");
    check(result->carrier.prepare(result->cold_program->compiled,result->backend->queue_bindings()),"Kimi original carrier preparation");
    check(result->carrier.initialize({.io=result->cold_program->io,.invocation={.resources=g::ResourceOptions{}}}),"Kimi original Invocation installation");
    result->facts=std::move(f);return result;
  }

  inference::RetirementObservation recover_pending() noexcept {
    if(!pending)return {true,{}};
    auto episode=pending;
    // A plain bind refusal has not touched Source/export or submission. Use
    // the original reset only with the carrier's actual retirement proof.
    if(!submit_attempted&&!episode->incoming&&!episode->sources&&episode->run&&carrier.retired()) {
      const auto reset=episode->run->reset();
      if(reset.ok()) {
        episode->run=nullptr;episode->carrier=nullptr;pending.reset();return {true,{}};
      }
    }
    episode->cleanup_pending=true;
    auto observed=episode->retire(*backend);
    auto recovered=episode->recover([]() noexcept {});
    // An unsubmitted Source consumer may retain the original predecessor
    // retirement gate. Drain only that original episode, leaving its exports
    // intact; then recover this consumer before producer cleanup can proceed.
    if(recovered&&!*recovered&&episode->predecessor_episode) {
      const auto predecessor=episode->predecessor_episode->retire(*backend);
      inference::record_retirement_status(observed.status,predecessor.status);
      recovered=episode->recover([]() noexcept {});
    }
    if(!recovered)inference::record_retirement_status(observed.status,recovered.error());
    observed.retired=carrier.retired()&&observed.retired&&recovered&&*recovered;
    if(observed.retired)pending.reset(); // A partial return retains this exact original episode.
    return observed;
  }

  Pending prepare_inputs(std::span<const c::ReadLease> input,c::WriteLease *hidden_writer=nullptr) {
    if(input.size()!=cold_program->input_count())throw std::runtime_error("Kimi original input count");
    if(pending) {
      const auto observed=recover_pending();
      if(!observed.retired)throw inference::EvaluationFailure(hyle::Status::Unavailable("Kimi original episode is still retiring"));
      check(observed.status,"Kimi previous original retirement");
    }
    if(!carrier.initialized())
      check(carrier.initialize({.io=cold_program->io,.invocation={.resources=g::ResourceOptions{}}}),"Kimi original Invocation retry");
    auto episode=std::make_shared<Episode>();pending=episode;submit_attempted=false;
    try {
      auto &run=*take(carrier.borrow(),"Kimi original carrier admission");
      episode->run=&run;episode->carrier=&carrier;
      // All ordinary binding and capture preparation precede producer submit.
      // A real input refusal retains the original Invocation for a legal retry.
      for(std::size_t i=0;i<input.size();++i)
        check(cold_program->bind(run,i,input[i]),"Kimi ordinary dynamic input");
      for(std::size_t i=0;i<cold_program->output_count();++i) {
        if(i==5&&hidden_writer) {
          const auto &output=cold_program->output(i);
          check(run.bind(output.value,output.description,std::move(*hidden_writer)),"Kimi original reclaimed hidden writer");
        } else check(cold_program->bind_output(run,i,pool),"Kimi original output writer");
      }
      return {shared_from_this(),std::move(episode)};
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

  static const char *route_name(g::ReplayRoute route) noexcept {
    switch(route) {
      case g::ReplayRoute::kCommands:return "commands";
      case g::ReplayRoute::kCaptured:return "captured";
      case g::ReplayRoute::kHybrid:return "hybrid";
      default:return "unprepared";
    }
  }
  g::ReplayReport replay_report(const Episode &episode,const char *stage) const {
    const auto report=take(episode.run->replay_report(),"Kimi actual replay report");
    std::printf("Kimi region=%llu stage=%s route=%s recorded=%zu reused=%zu hostWait=%zu deviceWait=%zu\n",
        static_cast<unsigned long long>(facts.region),stage,route_name(report.route),
        report.recorded_capture_range_count,report.reused_capture_range_count,
        report.host_wait_dependency_count,report.device_wait_dependency_count);
    return report;
  }
  c::WriteLease prepare_source_reuse(Pending &prepared,c::ReadLease &hidden_ready) {
    if(prepared.owner.get()!=this||prepared.execution!=pending||submit_attempted)
      throw std::runtime_error("Kimi cold capture requires its original unsubmitted episode");
    auto &episode=*prepared.execution;
    const auto start=std::chrono::steady_clock::now();
    try {
      const auto original=take(hidden_ready.tensor(cold_program->input(0).description),"Kimi cold hidden tensor");
      const auto generation=hidden_ready.generation();
      check(episode.prepare_replay({.mode=g::ReplayMode::kPreferCapture,.host_wait=g::HostWaitPolicy::kReject}),
          "Kimi complete MLA cold capture");
      const auto report=replay_report(episode,"consumer-cold-prepare");
      if((report.route!=g::ReplayRoute::kCaptured&&report.route!=g::ReplayRoute::kHybrid)||
          report.host_wait_dependency_count||(!report.recorded_capture_range_count&&!report.reused_capture_range_count))
        throw std::runtime_error("Kimi MLA cold preparation did not establish actual capture coverage");
      ++cold_capture_preparations;cold_recorded_ranges+=report.recorded_capture_range_count;
      cold_reused_ranges+=report.reused_capture_range_count;
      // Preserve every other original input, output writer and captured private
      // address. Only this complete ready input hands its one block to KDA.
      check(episode.run->unbind(cold_program->input(0).value),"Kimi cold hidden input withdrawal");
      auto writer=take(hidden_ready.try_write(),"Kimi original hidden writer handoff");
      const auto reclaimed=take(writer.view(),"Kimi reclaimed hidden backing");
      const auto storage=original.storage();
      if(writer.generation()!=generation||reclaimed.resource()!=storage.resource()||
          reclaimed.native_handle()!=storage.native_handle()||reclaimed.offset()!=storage.offset()||
          reclaimed.size()!=storage.size())
        throw std::runtime_error("Kimi cold hidden handoff changed its original physical writer");
      cold_capture_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      return writer;
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

  void submit(Pending &prepared,Pending *previous=nullptr,bool capture_hidden=false) {
    if(prepared.owner.get()!=this||prepared.execution!=pending||submit_attempted)
      throw std::runtime_error("Kimi submit requires its original prepared episode");
    auto &episode=*prepared.execution;auto &run=*episode.run;
    try {
      if(previous) {
        if(!previous->execution||!previous->execution->sources||previous->observed)
          throw std::runtime_error("Kimi MLA requires its unobserved original KDA submission");
        episode.incoming=std::make_unique<inference::IncomingExecutionSources<inference::SubmissionViews>>();
        episode.incoming->owner=previous->execution->sources;
        episode.incoming->bindings.push_back(episode.incoming->owner->primary_output.reader_binding(cold_program->input(0).value));
        episode.predecessor_episode=previous->execution;
      }
      if(capture_hidden) {
        episode.sources=std::make_shared<inference::SubmissionViews>();
        check(episode.sources->prepare(run,std::span<const inference::SubmissionViews::StoreCapture>{},std::span<const inference::SubmissionViews::PageCapture>{},{}),"Kimi original hidden Source preparation");
      }
      check(episode.bind_consumers(logical.graph),"Kimi original consumer bundle");
      check(episode.prepare_replay({.mode=capture_hidden?g::ReplayMode::kPreferCapture:g::ReplayMode::kCommands,
          .host_wait=(previous||capture_hidden)?g::HostWaitPolicy::kReject:g::HostWaitPolicy::kAllow}),"Kimi original replay preparation");
      last_replay=replay_report(episode,previous?"consumer-source-reuse":capture_hidden?"producer":"standalone");
      if(previous) {
        const auto source=take(run.input_source_report(),"Kimi actual dependency report");
        std::printf("Kimi region=%llu sources=%zu sourceDeviceWait=%zu sourceHostWait=%zu\n",
            static_cast<unsigned long long>(facts.region),source.source_count,source.device_wait_count,source.host_wait_count);
        if(last_replay.recorded_capture_range_count||last_replay.host_wait_dependency_count||
            !last_replay.reused_capture_range_count||source.source_count!=1||source.device_wait_count!=1||source.host_wait_count)
          throw std::runtime_error("Kimi Source submit did not reuse complete device-ordered cold capture");
      }
      recorded_capture_ranges+=last_replay.recorded_capture_range_count;
      reused_capture_ranges+=last_replay.reused_capture_range_count;host_waits+=last_replay.host_wait_dependency_count;
      submit_attempted=true;
      check(episode.submit(shared_from_this()),"Kimi original Graph submit");
      if(previous)++source_submissions_before_host_observation;
      if(capture_hidden)
        take(episode.sources->capture(run,episode.completion,cold_program->output(5).value,
            backend->storage_owner(),pool),"Kimi actual hidden OutputSource");
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

  std::vector<c::ReadLease> finish(Pending &submitted) {
    if(submitted.owner.get()!=this||submitted.execution!=pending||!submit_attempted)
      throw std::runtime_error("Kimi finish requires its original pending episode");
    auto &episode=*submitted.execution;
    submitted.observed=true;
    try {
      check(episode.wait_success(),"Kimi original whole Graph retirement");++submissions;
      if(episode.sources)check(episode.sources->detach_outputs(),"Kimi original hidden writer publication");
      std::vector<c::ReadLease> result;result.reserve(cold_program->output_count());
      for(std::size_t i=0;i<cold_program->output_count();++i)
        result.push_back(take(cold_program->retain(*episode.run,i),"Kimi retained original result"));
      check(episode.release_source_bindings(),"Kimi original consumer Source withdrawal");
      episode.incoming.reset();episode.predecessor_episode.reset();
      if(episode.sources) {
        check(episode.sources->submission.detach(),"Kimi original producer Source withdrawal");
        episode.sources.reset();
      }
      // Successful serial reuse keeps the same Invocation. Recovery alone
      // uses reset_retired(), after the original transactions have returned.
      check(episode.run->reset(),"Kimi original Invocation reset");
      if(episode.run->submit(carrier.queues()))throw std::runtime_error("Kimi reset retained stale replay");
      episode.run=nullptr;episode.carrier=nullptr;pending.reset();
      submitted.execution.reset();submitted.owner.reset();return result;
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }
  std::vector<c::ReadLease> execute(std::span<const c::ReadLease> input) {
    auto prepared=prepare_inputs(input);submit(prepared);return finish(prepared);
  }
  void record_properties(const std::string &prefix) const {
    ::testing::Test::RecordProperty(prefix+"_replay_route",route_name(last_replay.route));
    ::testing::Test::RecordProperty(prefix+"_recorded_capture_ranges",std::to_string(recorded_capture_ranges));
    ::testing::Test::RecordProperty(prefix+"_reused_capture_ranges",std::to_string(reused_capture_ranges));
    ::testing::Test::RecordProperty(prefix+"_host_waits",std::to_string(host_waits));
    ::testing::Test::RecordProperty(prefix+"_cold_capture_preparations",std::to_string(cold_capture_preparations));
    ::testing::Test::RecordProperty(prefix+"_cold_recorded_capture_ranges",std::to_string(cold_recorded_ranges));
    ::testing::Test::RecordProperty(prefix+"_cold_reused_capture_ranges",std::to_string(cold_reused_ranges));
    ::testing::Test::RecordProperty(prefix+"_cold_capture_seconds",std::to_string(cold_capture_seconds));
  }
};
Pending::~Pending() {if(owner&&execution&&owner->pending==execution)(void)owner->recover_pending();}

// Cold-only original parameter assembly. ReadLeases are moved into the
// existing Run::literals vector; no second table owns the native constants.
inline std::shared_ptr<Run> prepare_original_kda(const kimi_original_attention_roles::Kda &original,
    const std::shared_ptr<inference::Backend> &backend) {
  const auto &owner=backend->storage_owner();
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
  return Run::prepare(std::move(f),backend,nullptr,std::move(literals));
}
inline std::shared_ptr<Run> prepare_original_mla(std::int32_t position,const kimi_original_attention_roles::Mla &original,
    const std::shared_ptr<inference::Backend> &backend,LiteralBank &small_literals) {
  const auto &owner=backend->storage_owner();
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
  return Run::prepare(std::move(f),backend,&small_literals,std::move(literals));
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

struct Mla final {std::shared_ptr<Run> run;kimi::MlaReference reference;};
Mla prepare_mla(std::int32_t position,const std::shared_ptr<inference::Backend> &backend,LiteralBank &literals) {
  Mla result;auto f=kimi::mla(kProfile,false,kimi::OrdinaryHistory{kCapacity,position},&result.reference);
  // The actual stored cursor is advanced by the original public integer graph
  // operations. Cold phase admission checks its value before executing.
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32));
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto joined=f.one(c::Operation(c::Concat{.axis=0}),{cursor,one});
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),{joined});
  f.outputs.push_back(f.one(c::Operation(c::Gather{}),{scanned,one}));
  result.run=Run::prepare(std::move(f),backend,&literals);return result;
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

TEST(KimiHeterogeneousState,CudaBorrowedLiteralPrefixUsesActualLeaseAndRejectsMisalignment) {
  auto backend=std::make_shared<inference::Backend>(take(inference::Backend::create({.primary=inference::BackendKind::kCuda}),"literal-prefix original Backend"));
  const auto &owner=backend->storage_owner();
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
    EXPECT_THROW(Run::prepare(std::move(f),backend,nullptr,std::move(supplied)),std::runtime_error);
  };
  reject(true,false);reject(false,true);
  c::ReadLease escaped;
  {
    sem::Fragment f{.profile=kProfile,.region=0x4b494d494c495433ULL};
    const auto input=f.input(description);std::vector<c::ReadLease> supplied;
    const auto constant=Run::source_constant(f,supplied,role,owner);
    ASSERT_EQ(f.constant_bytes.size(),1u);EXPECT_TRUE(f.constant_bytes[0].empty());
    f.outputs={f.one(c::Operation(c::Binary{c::BinaryKind::kAdd}),{input,constant})};
    auto run=Run::prepare(std::move(f),backend,nullptr,std::move(supplied));
    ASSERT_EQ(run->literals.size(),1u);
    const auto actual_view=take(run->literals[0].view(),"literal-prefix retained original view");
    EXPECT_EQ(actual_view.native_handle(),original_view.native_handle());EXPECT_EQ(actual_view.offset(),original_view.offset());
    EXPECT_EQ(actual_view.resource(),original_view.resource());
    role.lease={};lease={}; // The original Run owns the same immutable lease.
    const std::array inputs{upload(owner,sem::bytes(input_values))};
    const auto output=run->execute(inputs);
    ASSERT_EQ(output.size(),1u);EXPECT_EQ(values<float>(output[0]),(std::vector<float>{6.F,8.F,10.F,12.F}));
    EXPECT_EQ(run->submissions,1u);escaped=run->literals[0];
  }
  EXPECT_EQ(values<float>(escaped),weights);escaped={};
  ::testing::Test::RecordProperty("source_scope","Synthetic F32[4] lease only; no original Kimi payload read or numeric qualification");
  ::testing::Test::RecordProperty("binding_scope","Actual original native handle/offset/resource reused through existing Run literals and GraphConstantBinding; no numeric constant byte carrier");
}

TEST(KimiHeterogeneousState,CudaOriginalGeometrySyntheticKdaMlaPublishesWholeStateAndRestoresForks) {
  auto backend=std::make_shared<inference::Backend>(take(inference::Backend::create({.primary=inference::BackendKind::kCuda}),"Kimi original Backend"));
  const auto &owner=backend->storage_owner();
  kda::Reference kda_reference;auto kda_run=Run::prepare(kda::complete(kProfile,&kda_reference),backend);
  LiteralBank literals;std::array<Mla,3> mla;
  for(std::int32_t p=0;p<3;++p)mla[std::size_t(p)]=prepare_mla(p,backend,literals);
  literals.release_host_bytes();
  auto budget=kda_run->pool.stats().capacity_bytes+32u*1024u*1024u;
  for(const auto &phase:mla)budget+=phase.run->pool.stats().capacity_bytes;
  auto state=State::create(owner,budget);std::size_t committed=0,cancelled=0,stale=0,binding_rejections=0;
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
      auto &phase=mla[std::size_t(position)];
      std::array mla_args{c::ReadLease{},state.member(before,kMla),state.member(before,kCursor)};
      // Declare the producer guard first so original consumer recovery always
      // precedes producer withdrawal during exception unwinding.
      std::optional<Pending> submitted_kda;
      std::optional<Pending> submitted_mla;
      std::vector<c::ReadLease> actual_mla;
      c::WriteLease hidden_writer;
      const bool rejected_binding=!cancel&&binding_rejections==0;
      if(!cancel) {
        hidden_writer=take(kda_run->pool.allocate(take(kda_run->cold_program->output(5).description.required_bytes(),
            "Kimi future hidden extent")),"Kimi original hidden pool writer");
        check(take(hidden_writer.view(),"Kimi cold hidden initialization view").fill(std::byte{}),"Kimi cold hidden initialization");
        mla_args[0]=take(hidden_writer.publish(),"Kimi initialized future hidden lease");
        if(rejected_binding) {
          auto *original_consumer=take(phase.run->carrier.borrow(),"Kimi retry original MLA Invocation");
          const auto submissions=phase.run->submissions;
          // Actual I32 cursor storage is too short for the F32 MLA cache.
          // The ordinary preparation refuses before producer acceptance or any
          // Source binding, then legally retries the same original Invocation.
          const std::array refused_args{mla_args[0],mla_args[2],mla_args[2]};
          bool refused=false;
          try { (void)phase.run->prepare_inputs(refused_args); }
          catch(const inference::EvaluationFailure &failure) {
            refused=true;EXPECT_TRUE(failure.status().IsOutOfRange());
            EXPECT_TRUE(failure.retirement_observed()&&failure.retirement().retired);
          }
          ASSERT_TRUE(refused);
          EXPECT_EQ(take(phase.run->carrier.borrow(),"Kimi same Invocation after bind refusal"),original_consumer);
          EXPECT_EQ(phase.run->submissions,submissions);
          EXPECT_FALSE(kda_run->pending);
          EXPECT_TRUE(take(head.current(),"Kimi Head after MLA binding refusal").identity()==before.identity());
          EXPECT_EQ(snapshot(state,before),old);
          const auto unchanged=state.epoch(before);
          EXPECT_EQ(unchanged.step,before_epoch.step);EXPECT_EQ(unchanged.position,before_epoch.position);
          EXPECT_EQ(unchanged.session,before_epoch.session);
          ++binding_rejections;
        }
        submitted_mla.emplace(phase.run->prepare_inputs(mla_args));
        hidden_writer=phase.run->prepare_source_reuse(*submitted_mla,mla_args[0]);
      }
      submitted_kda.emplace(kda_run->prepare_inputs(args,cancel?nullptr:&hidden_writer));
      kda_run->submit(*submitted_kda,nullptr,true);
      if(!cancel) {
        phase.run->submit(*submitted_mla,&*submitted_kda);
        EXPECT_FALSE(submitted_kda->observed);
        // Both original submissions were accepted before any host result
        // comparison/read/wait. MLA imports KDA's complete mathematical verdict.
        actual_mla=phase.run->finish(*submitted_mla);
      }
      auto actual_kda=kda_run->finish(*submitted_kda);
      compare(*kda_run,actual_kda,kda_reference(input,values<float>(args[1]),values<float>(args[2])));verified();
      auto draft=take(state.store.begin(before),"Kimi unpublished heterogeneous candidate");
      check(draft.replace(state.entries[kHistory],actual_kda[0]),"Kimi candidate convolution history");
      check(draft.replace(state.entries[kRecurrent],actual_kda[2]),"Kimi candidate KDA recurrent state");
      if(cancel) {
        const auto submissions=phase.run->submissions;
        check(draft.cancel(),"cancel unpublished Kimi candidate after KDA");++cancelled;
        EXPECT_TRUE(take(head.current(),"cancelled Kimi Head").identity()==before.identity());
        EXPECT_EQ(state.epoch(take(head.current(),"cancelled Kimi Epoch")).step,before_epoch.step);
        EXPECT_EQ(snapshot(state,before),old);EXPECT_EQ(phase.run->submissions,submissions);return;
      }
      // The evaluator reads the same actual KDA result after native dependency
      // consumption, alongside the original immutable old MLA state.
      compare(*phase.run,actual_mla,phase.reference(values<float>(actual_kda[5]),values<float>(mla_args[1])),std::size_t(position+1));
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
  EXPECT_EQ(kda_run->submissions,7u);EXPECT_EQ(mla[0].run->submissions,2u);EXPECT_EQ(mla[1].run->submissions,3u);EXPECT_EQ(mla[2].run->submissions,1u);
  EXPECT_EQ(binding_rejections,1u);
  const auto ordered_submissions=mla[0].run->source_submissions_before_host_observation+
      mla[1].run->source_submissions_before_host_observation+mla[2].run->source_submissions_before_host_observation;
  EXPECT_EQ(ordered_submissions,6u);
  EXPECT_EQ(mla[0].run->cold_capture_preparations+mla[1].run->cold_capture_preparations+
      mla[2].run->cold_capture_preparations,6u);
  EXPECT_EQ(mla[0].run->recorded_capture_ranges+mla[1].run->recorded_capture_ranges+mla[2].run->recorded_capture_ranges,0u);
  EXPECT_EQ(mla[0].run->host_waits+mla[1].run->host_waits+mla[2].run->host_waits,0u);
  kda_run->record_properties("kda");
  for(std::size_t i=0;i<mla.size();++i)mla[i].run->record_properties("mla"+std::to_string(i));
  ::testing::Test::RecordProperty("weight_scope","Generated sparse BF16/F32 weights at fixed Kimi7168/96x128 geometry; NOT original artifact payload or quality");
  ::testing::Test::RecordProperty("composition_scope","Actual KDA hidden OutputSource feeds actual absorbed MLA on the original Queue before producer host observation. Intervening full model layers/residuals and LM head are not executed; no generated token or MTP claim.");
  ::testing::Test::RecordProperty("second_graph_submitted_before_producer_host_observation",std::to_string(ordered_submissions));
  ::testing::Test::RecordProperty("second_graph_binding_rejections_same_invocation_retry",std::to_string(binding_rejections));
  ::testing::Test::RecordProperty("binding_retry_scope","Actual short cache lease refuses ordinary MLA preparation before producer acceptance; same Invocation retry preserves Head, complete device state and all Epoch fields");
  ::testing::Test::RecordProperty("cold_capture_scope","Before KDA acceptance, initialize its original hidden pool writer, prepare MLA capture with the exact old cache/cursor and future output writers, withdraw only input0 and reclaim that same hidden writer for KDA; no extra Graph submit or pool");
  ::testing::Test::RecordProperty("retirement_scope","Original Run Scope retains Backend, cold program, pool, literals and stable ProgramCarrier node; Source bundle withdrawal precedes producer detach; successful reuse preserves Invocation.reset");
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
  auto backend=std::make_shared<inference::Backend>(take(inference::Backend::create({.primary=inference::BackendKind::kCuda}),"original-parameter Kimi Backend"));
  const auto &owner=backend->storage_owner();
  const auto kda_operands=original::prepare_kda(source_kda,owner);
  const auto mla_operands=original::prepare_mla(source_mla,owner);
  auto recurrent=prepare_original_kda(kda_operands,backend);
  LiteralBank small_literals;
  std::array<decltype(recurrent),2> attention;
  for(std::int32_t position=0;position<2;++position)
    attention[std::size_t(position)]=prepare_original_mla(position,mla_operands,backend,small_literals);
  small_literals.release_host_bytes();
  std::size_t state_budget=recurrent->pool.stats().capacity_bytes+32u*1024u*1024u;
  for(const auto &run:attention)state_budget+=run->pool.stats().capacity_bytes;
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
      auto &run=attention[std::size_t(position)];
      auto hidden_writer=take(recurrent->pool.allocate(take(recurrent->cold_program->output(5).description.required_bytes(),
          "original Kimi future hidden extent")),"original Kimi hidden pool writer");
      check(take(hidden_writer.view(),"original Kimi cold hidden initialization view").fill(std::byte{}),"original Kimi cold hidden initialization");
      std::array mla_args{take(hidden_writer.publish(),"original Kimi initialized future hidden lease"),
          state.member(before,kMla),state.member(before,kCursor)};
      std::optional<Pending> submitted_kda;
      auto submitted_mla=run->prepare_inputs(mla_args);
      hidden_writer=run->prepare_source_reuse(submitted_mla,mla_args[0]);
      submitted_kda.emplace(recurrent->prepare_inputs(args,&hidden_writer));
      recurrent->submit(*submitted_kda,nullptr,true);
      run->submit(submitted_mla,&*submitted_kda);
      EXPECT_FALSE(submitted_kda->observed);
      auto actual_mla=run->finish(submitted_mla);
      auto actual_kda=recurrent->finish(*submitted_kda);
      const auto prior_history=values<float>(args[1]),prior_state=values<float>(args[2]);
      compare(*recurrent,actual_kda,reference::kda(*source_kda,activation,prior_history,prior_state));verified();
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
      // Actual native KDA output is a labelled upstream fixture for layer3.
      // Neither a host oracle feature nor an ideal old state feeds the graph.
      const auto actual_input=values<float>(actual_kda[5]),old_cache=values<float>(mla_args[1]);
      compare(*run,actual_mla,reference::mla(*source_mla,actual_input,old_cache,
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
  EXPECT_EQ(recurrent->submissions,2u);EXPECT_EQ(attention[0]->submissions,1u);EXPECT_EQ(attention[1]->submissions,1u);
  const auto ordered_submissions=attention[0]->source_submissions_before_host_observation+
      attention[1]->source_submissions_before_host_observation;
  EXPECT_EQ(ordered_submissions,2u);
  EXPECT_EQ(attention[0]->cold_capture_preparations+attention[1]->cold_capture_preparations,2u);
  EXPECT_EQ(attention[0]->recorded_capture_ranges+attention[1]->recorded_capture_ranges,0u);
  EXPECT_EQ(attention[0]->host_waits+attention[1]->host_waits,0u);
  recurrent->record_properties("kda");
  for(std::size_t i=0;i<attention.size();++i)attention[i]->record_properties("mla"+std::to_string(i));
  ::testing::Test::RecordProperty("second_graph_submitted_before_producer_host_observation",std::to_string(ordered_submissions));
  ::testing::Test::RecordProperty("cold_capture_scope","Each exact physical MLA input/output tuple is captured before KDA acceptance using its initialized original hidden writer; the same writer is reclaimed for KDA and MLA Source admission reuses capture without recording or host wait");
  ::testing::Test::RecordProperty("submission_dependency_scope","Original KDA hidden OutputSource and complete producer verdict feed MLA input0 on the same Backend Queue; both graphs submit before host result observation or oracle reads");
  ::testing::Test::RecordProperty("retirement_scope","Actual original Run Scope pins Backend/cold program/pool/literals/carrier node; consumer Source withdrawal precedes producer detach, and successful Invocation.reset reuse is preserved");
  ::testing::Test::RecordProperty("source_scope","Original fixed-revision KDA0 14 roles + MLA3 9 roles; per-union/per-role SHA checked, original raw KV-b is not uploaded");
  ::testing::Test::RecordProperty("upstream_scope","Supplied normalized KDA0 input; KDA0 output is an explicit upstream fixture for MLA3, not intervening layers or complete model execution");
  ::testing::Test::RecordProperty("oracle_scope","Original raw row/tap/KV-b indexing; no prepared K/V or expanded F32 weight bank. Existing six/nine-output accuracy gates plus selected actual-producer F64 projection certificates");
  ::testing::Test::RecordProperty("state_scope","Two real complete Head commits, actual prior native state feedback, immutable old readers, restored old-version fork, unpublished cancellation, stale expected-base rejection; no native preemption claim");
}

} // namespace
#endif
