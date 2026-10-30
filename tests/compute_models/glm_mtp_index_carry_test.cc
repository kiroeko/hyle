#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "../../engine/inference/output_pool_capacity.h"
#include "hyle/compute/state_versions.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_GLM_CARRY_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_GLM_CARRY_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "d5_glm_remaining_fragments.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace glm=d5_glm_remaining_fragments;
namespace sem=d5_qwen38_24t_semantic_fragments;
using Bytes=std::vector<std::byte>;
using hyle::Status;
template<class T> using Result=hyle::Result<T>;
constexpr std::int64_t kTail=1,kCarried=2051,kSelected=2048+kTail;
constexpr std::int64_t kTokens=2054+kTail,kWidth=glm::kLatent,kIndexDepth=128;
constexpr auto kModel=glm::kCases[4]; // Generated Q8_0 Flash indexer, not original weights.

// Source: SGLang@06992c92ad4494fe9f9636e75be3e35040512447,
// layers/attention/index_topk_share.py:mtp_iteration/update/publish.
// This fixture binds actual producer I32 storage to a skip-indexer Graph.
// It covers one DSA carry and its index-key/gate/provenance, not a complete
// GLM MTP decoder, original payload, natural acceptance or global cache.
template<class T> T take(Result<T> result,const char *where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
void check(Status status,const char *where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> bytes) {
  auto buffer=take(m::Buffer::create(owner.ref(),bytes.size(),256),"ordinary input allocation");
  check(buffer.view().write(bytes),"actual input bytes");
  return take(c::ReadLease::adopt(std::move(buffer)),"qualified readonly input");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(),"readonly actual result").read(result),"actual result read");
  return result;
}
template<class T> std::vector<T> typed(const c::ReadLease &lease) {
  if(lease.size()%sizeof(T)) throw std::runtime_error("typed carry extent");
  std::vector<T> result(lease.size()/sizeof(T));
  check(take(lease.view(),"typed readonly view").read(std::as_writable_bytes(std::span(result))),"typed actual bytes");
  return result;
}

// Ordinary Graph/Invocation helper only: the same original compile, bind,
// prepare, submit, Completion and reset path for both native storage domains.
struct Program final {
  sem::Fragment fragment;
  sem::GraphCase logical;
  std::vector<c::ReadLease> parameters;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool outputs;

  static Program create(sem::Fragment fragment,c::Context &context,
      const c::StorageOwner &owner,bool cuda,std::uint32_t generations) {
    Program result;
    result.fragment=std::move(fragment);
    result.fragment.replays.clear();
    if(!result.fragment.prepared_constants.empty())
      throw std::runtime_error("carry fixture requires the existing direct indexer constants");
    result.logical=sem::make_graph(result.fragment);
    std::vector<g::ConstantBinding> constants;
    for(std::size_t i=0;i<result.fragment.constants.size();++i) {
      result.parameters.push_back(upload(owner,result.fragment.constant_bytes[i]));
      constants.push_back({result.logical.constants[i],take(result.parameters.back().tensor(
          result.fragment.description(result.fragment.constants[i])),"indexer constant view")});
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*kModel.profile),"fixed indexer profile"),
        c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,0x474c4d4341525259ULL,1}};
    const g::CompileOptions options{.constants=constants,.workload=workload};
    const auto compile=[&]() -> Result<g::CompiledGraph> {
#if defined(HYLE_TEST_GLM_CARRY_CPU)
      if(!cuda) return g::cpu::compile(result.logical.graph,context,options);
#endif
#if defined(HYLE_TEST_GLM_CARRY_CUDA)
      if(cuda) return g::cuda::compile(result.logical.graph,context,owner,options);
#endif
      throw std::runtime_error("index carry backend not enabled");
    };
    result.compiled=take(compile(),"ordinary index carry compile");
    result.invocation=take(result.compiled.executable.create_invocation(
        {.resources=g::ResourceOptions{}}),"one original Invocation");
    std::vector<std::size_t> output_sizes;
    output_sizes.reserve(result.fragment.outputs.size());
    for(std::size_t i=0;i<result.fragment.outputs.size();++i) {
      if(take(result.compiled.executable.output_ownership(i),"output ownership")!=g::OutputOwnership::kCaller)
        throw std::runtime_error("carry fixture requires original caller outputs");
      output_sizes.push_back(take(result.fragment.description(result.fragment.outputs[i]).required_bytes(),"actual output extent"));
    }
    if (output_sizes.empty()) throw std::runtime_error("caller output inventory must be nonempty");
    const auto output_capacity=take(hyle::inference::checked_output_pool_capacity(output_sizes,generations,256),
        "carry output pool capacity");
    result.outputs=take(c::LeasePool::create(owner.ref(),output_capacity.backing_bytes,
        {.max_blocks=output_capacity.max_blocks,.backing_alignment=256}),"bounded original output generations");
    return result;
  }
  std::vector<c::ReadLease> execute(c::Queue &queue,std::span<const c::ReadLease> inputs) {
    if(inputs.size()!=fragment.inputs.size()) throw std::runtime_error("actual carry input count");
    for(std::size_t i=0;i<inputs.size();++i)
      check(invocation.bind(logical.inputs[i],fragment.description(fragment.inputs[i]),inputs[i]),"bind actual input permission");
    for(std::size_t i=0;i<fragment.outputs.size();++i) {
      const auto &desc=fragment.description(fragment.outputs[i]);
      check(invocation.bind(logical.outputs[i],desc,take(outputs.allocate(
          take(desc.required_bytes(),"actual result bytes")),"new original result writer")),"bind original result writer");
    }
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare complete ordinary Graph");
    {auto completion=take(invocation.submit(queues),"actual Graph submit");check(completion.wait(),"actual Graph retirement");}
    std::vector<c::ReadLease> result;
    for(const auto output:logical.outputs) {
      auto value=take(invocation.result(output),"successful Graph result");
      result.push_back(take(value.storage_lease(),"independently retained result permission"));
    }
    check(invocation.reset(),"withdraw completed permissions");
    if(invocation.submit(queues)) throw std::runtime_error("reset graph retained stale preparation");
    return result;
  }
};

sem::Fragment skip_indexer() {
  // Exactly the ordinary Gather/SDPA program in check_index_carry:968-987.
  // It contains no TopK or projection, so reuse never executes an indexer.
  sem::Fragment f{.profile=kModel.profile,.region=0x474c4d534b4950};
  const auto ids=f.input(sem::dense({kCarried},c::DataType::kInt32));
  const auto query=f.input(sem::dense({1,1,glm::kHeads,kWidth}));
  const auto cache=f.input(sem::dense({kTokens,kWidth}));
  std::vector<std::int32_t> identity(kCarried);
  std::iota(identity.begin(),identity.end(),0);
  const auto all=f.constant(sem::dense({kCarried},c::DataType::kInt32),sem::bytes(identity));
  const auto carried=f.one(c::Operation(c::Gather{}),{ids,all});
  const auto valid=glm::reshape(f,glm::core::rows(f,glm::reshape(f,carried,{kCarried,1}),0,kSelected),{kSelected});
  const auto chosen=f.one(c::Operation(c::Gather{}),{cache,valid});
  const auto values=glm::kimi::slice_heads(f,chosen,kSelected,kWidth,0,glm::kLatent);
  const auto attention=f.one(c::Operation(c::ScaledDotProductAttention{.scale=1.F/16.F,
      .causal=true,.accumulation_type=c::DataType::kFloat32}),
      {query,glm::reshape(f,chosen,{1,kSelected,1,kWidth}),glm::reshape(f,values,{1,kSelected,1,glm::kLatent})});
  f.outputs={carried,attention};
  for(const auto &node:f.nodes)
    if(node.operation.get_if<c::TopK>() || node.operation.get_if<c::MatMul>())
      throw std::runtime_error("skip-indexer graph accidentally recomputes selection");
  return f;
}

// Epoch qualification is engine policy, not revoked historical F1 access.
// Old Versions remain genuinely readable; the current iteration refuses to
// nominate them for a new reuse submission after reset/finally.
struct Provenance final {std::uint64_t epoch,producer,uses;};
static_assert(sizeof(Provenance)==24);
struct Carry final {
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId,4> entries; // IDs, original index key/gate, metadata.
  c::StateHead head;
  std::uint64_t epoch=0;
  bool enabled=false;

  static Carry create(const c::StorageOwner &owner) {
    const std::array specs{c::StateEntrySpec{.description=sem::dense({kCarried},c::DataType::kInt32),.resource=owner.ref()},
        c::StateEntrySpec{.description=sem::dense({kTokens-1,kIndexDepth}),.resource=owner.ref()},
        c::StateEntrySpec{.description=sem::dense({kTokens-1,kIndexDepth}),.resource=owner.ref()},
        c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Provenance)}};
    Carry result;
    result.schema=take(c::StateSchema::create(specs),"original generic carry schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=16,.max_drafts=2,
        .max_retained_blocks=96,.max_backing_bytes=32u<<20}),"bounded original carry state");
    for(std::size_t i=0;i<result.entries.size();++i) result.entries[i]=take(result.schema.entry(i),"carry entry");
    return result;
  }
  Provenance facts(const c::StateVersion &version) const {
    const auto member=take(version.member(entries[3]),"qualified provenance member");
    Provenance result{};
    check(take(member.view(),"actual provenance bytes").read(std::as_writable_bytes(std::span(&result,1))),"read real provenance");
    return result;
  }
  c::StateVersion producer(std::span<const c::ReadLease> result,std::uint64_t serial) {
    if(result.size()!=8 || !serial) throw std::runtime_error("actual Flash producer result shape");
    const Provenance metadata{0,serial,0};
    const std::array members{c::StateMember{entries[0],result[6]},c::StateMember{entries[1],result[0]},
        c::StateMember{entries[2],result[7]},c::StateMember{entries[3],{},std::as_bytes(std::span(&metadata,1))}};
    return take(store.import(members),"retain actual producer leases without rebuilding I32");
  }
  void begin(bool keep_seed,const c::StateVersion *seed=nullptr) {
    enabled=false;head={};
    if(epoch==std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("epoch overflow");
    const auto next=epoch+1;
    c::StateHead selected;
    if(keep_seed && seed && seed->valid()) {
      auto metadata=facts(*seed);metadata.epoch=next;metadata.uses=0;
      auto draft=take(store.begin(*seed),"qualify actual seed for this epoch");
      check(draft.replace_metadata(entries[3],std::as_bytes(std::span(&metadata,1))),"matching current epoch");
      selected=take(c::StateHead::create(store,take(draft.finalize(),"complete seed version")),"seeded iteration head");
    }
    epoch=next;head=std::move(selected);enabled=true;
  }
  void end() noexcept {enabled=false;head={};}
  Result<c::StateVersion> current() const {
    if(!enabled || !head.valid()) return std::unexpected(Status::Unavailable("no same-epoch index carry"));
    return head.current();
  }
  Result<c::ReadLease> qualify(const c::StateVersion &version) const {
    if(!enabled || !head.valid()) return std::unexpected(Status::Unavailable("index carry reuse is disabled"));
    const auto current=head.current();if(!current) return std::unexpected(current.error());
    if(current->identity()!=version.identity() || facts(version).epoch!=epoch)
      return std::unexpected(Status::Unavailable("index carry belongs to another epoch/head"));
    return version.member(entries[0]);
  }
  void fresh(const c::StateVersion &seed) {
    if(!enabled) throw std::runtime_error("fresh indexer outside an iteration");
    auto metadata=facts(seed);metadata.epoch=epoch;metadata.uses=0;
    auto draft=take(store.begin(seed),"stage actual fresh index carry");
    check(draft.replace_metadata(entries[3],std::as_bytes(std::span(&metadata,1))),"fresh producer epoch");
    auto version=take(draft.finalize(),"finalize actual fresh carry");
    if(head.valid()) {const auto old=take(head.current(),"previous actual carry head");check(head.publish(old.identity(),version),"publish fresh full carry");}
    else head=take(c::StateHead::create(store,version),"publish fresh carry head");
  }
  c::StateVersion publish(const c::StateVersion &from,const c::ReadLease &actual_output) {
    (void)take(qualify(from),"validate same-epoch provenance before publish");
    auto metadata=facts(from);
    if(metadata.uses==std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("reuse count overflow");
    ++metadata.uses;
    auto draft=take(store.begin(from),"original state draft");
    check(draft.replace(entries[0],actual_output),"stage actual I32 Graph result");
    check(draft.replace_metadata(entries[3],std::as_bytes(std::span(&metadata,1))),"stage matching provenance");
    auto version=take(draft.finalize(),"finalize complete single-layer carry");
    check(head.publish(from.identity(),version),"atomically publish carried ID/provenance");
    return version;
  }
};
struct Iteration final {
  Carry *carry;
  Iteration(Carry &value,bool keep_seed,const c::StateVersion *seed=nullptr):carry(&value) {carry->begin(keep_seed,seed);}
  ~Iteration() {carry->end();}
};
using Image=std::array<Bytes,4>;
Image image(const Carry &carry,const c::StateVersion &version) {
  Image result;for(std::size_t i=0;i<result.size();++i) result[i]=read(take(version.member(carry.entries[i]),"actual old state member"));return result;
}
void same_storage(const c::ReadLease &a,const c::ReadLease &b) {
  const auto x=take(a.view(),"producer readonly identity"),y=take(b.view(),"consumer readonly identity");
  EXPECT_EQ(x.resource(),y.resource());EXPECT_EQ(x.native_handle(),y.native_handle());
  EXPECT_EQ(x.offset(),y.offset());EXPECT_EQ(x.size(),y.size());EXPECT_EQ(a.generation(),b.generation());
}
void verify_producer(const sem::Fragment &fragment,std::span<const c::ReadLease> result) {
  if(fragment.replays.empty() || result.size()!=fragment.outputs.size()) throw std::runtime_error("missing producer oracle");
  for(std::size_t port=0;port<result.size();++port) {
    auto values=typed<float>(result[port]);
    values=sem::decode_output(fragment,port,std::move(values));
    const auto *ids=port==6 ? &*fragment.unordered_ids : nullptr;
    sem::expect_output(fragment,values,fragment.replays[0].outputs[port],{},false,ids);
  }
}
std::vector<c::ReadLease> producer_inputs(const sem::Fragment &fragment,const c::StorageOwner &owner) {
  std::vector<c::ReadLease> result;
  for(const auto &bytes:fragment.replays.at(0).inputs) result.push_back(upload(owner,bytes));
  return result;
}
c::StateVersion consume(Carry &carry,Program &skip,c::Queue &queue,const c::StorageOwner &owner,
    const c::StateVersion &from,int seed) {
  // This gate happens before bind/prepare/submit, not after executing stale IDs.
  const auto input=take(carry.qualify(from),"same-epoch nomination before Graph preparation");
  same_storage(input,take(from.member(carry.entries[0]),"actual version ID member"));
  auto query=d5_gguf_test::activation(glm::kHeads*kWidth,101+seed*14);
  auto cache=d5_gguf_test::activation(kTokens*kWidth,113+seed*14);
  for(auto &value:query) value/=4;
  for(auto &value:cache) value*=4;
  const std::array arguments{input,upload(owner,sem::bytes(query)),upload(owner,sem::bytes(cache))};
  const auto before=read(input);
  auto outputs=skip.execute(queue,arguments);
  EXPECT_TRUE(read(outputs[0])==before)<<"actual I32 producer order/padding changed";
  const auto selected=typed<std::int32_t>(input); // Oracle only; never rebound.
  std::vector<float> oracle_ids(selected.size());
  std::transform(selected.begin(),selected.end(),oracle_ids.begin(),
      [](std::int32_t value) {return static_cast<float>(value);});
  const auto expected=glm::attention_reference(query,cache,oracle_ids,kSelected,kWidth);
  const auto actual=typed<float>(outputs[1]);
  sem::oracle::expect_close(actual,expected,skip.fragment.absolute_tolerance,4e-5F);
  const auto version=carry.publish(from,outputs[0]);
  same_storage(take(version.member(carry.entries[0]),"published actual carry"),outputs[0]);
  EXPECT_TRUE(read(input)==before);
  return version;
}

void run(c::Context &context,const c::StorageOwner &owner,bool cuda) {
  auto normal=glm::indexer(kModel,kTail,false,false);normal.replays.resize(1);
  auto reversed=glm::indexer(kModel,kTail,false,true);reversed.replays.resize(1);
  auto indexer=Program::create(normal,context,owner,cuda,3);
  auto skip=Program::create(skip_indexer(),context,owner,cuda,4);
  auto queue=take(context.create_queue(),"original queue");
  auto carry=Carry::create(owner);
  int producer_calls=0,reuse_calls=0;
  {
    auto first=indexer.execute(queue,producer_inputs(normal,owner));++producer_calls;
    verify_producer(normal,first);ASSERT_FALSE(::testing::Test::HasFailure());
    auto seed=carry.producer(first,std::uint64_t(producer_calls));
    same_storage(take(seed.member(carry.entries[0]),"original producer I32"),first[6]);
    first.clear();
    const auto seed_bytes=image(carry,seed);
    c::StateVersion old_epoch;
    Image old_bytes;
    {
      Iteration epoch(carry,false,&seed);
      EXPECT_FALSE(carry.current())<<"default begin must clear even a real seed";
      EXPECT_FALSE(carry.qualify(seed));
      auto recomputed=indexer.execute(queue,producer_inputs(normal,owner));++producer_calls;
      verify_producer(normal,recomputed);ASSERT_FALSE(::testing::Test::HasFailure());
      const auto actual=carry.producer(recomputed,std::uint64_t(producer_calls));
      carry.fresh(actual);recomputed.clear();
      auto before=take(carry.current(),"fresh current iteration");
      same_storage(take(before.member(carry.entries[0]),"fresh actual ID permission"),take(actual.member(carry.entries[0]),"producer result permission"));
      const auto initial=image(carry,before);
      auto once=consume(carry,skip,queue,owner,before,0);++reuse_calls;
      auto twice=consume(carry,skip,queue,owner,take(carry.current(),"actual first skip head"),1);++reuse_calls;
      EXPECT_EQ(carry.facts(twice).uses,2u);EXPECT_EQ(carry.facts(twice).producer,2u);
      EXPECT_TRUE(image(carry,before)==initial);EXPECT_TRUE(image(carry,seed)==seed_bytes);
      old_epoch=once;old_bytes=image(carry,once);
    }
    EXPECT_FALSE(carry.current());EXPECT_FALSE(carry.qualify(old_epoch));
    EXPECT_TRUE(image(carry,old_epoch)==old_bytes)<<"finally cannot invalidate old readonly ownership";
    struct ExpectedExit final {};
    try {
      Iteration epoch(carry,true,&seed);
      auto current=take(carry.current(),"explicit keep_carry_seed");
      same_storage(take(current.member(carry.entries[0]),"seeded same-epoch ID lease"),take(seed.member(carry.entries[0]),"actual pre-iteration seed"));
      EXPECT_FALSE(carry.qualify(old_epoch))<<"another active epoch cannot authorize old carry";
      const auto result=consume(carry,skip,queue,owner,current,2);++reuse_calls;
      EXPECT_EQ(carry.facts(result).producer,1u);EXPECT_EQ(carry.facts(result).uses,1u);
      throw ExpectedExit{};
    } catch(const ExpectedExit &) {}
    EXPECT_FALSE(carry.current());EXPECT_FALSE(carry.qualify(old_epoch));
    EXPECT_TRUE(image(carry,seed)==seed_bytes);EXPECT_TRUE(image(carry,old_epoch)==old_bytes);
    {
      Iteration epoch(carry,true); // keep=true without a seed still clears.
      EXPECT_FALSE(carry.current());EXPECT_FALSE(carry.qualify(seed));
      auto output=indexer.execute(queue,producer_inputs(reversed,owner));++producer_calls;
      verify_producer(reversed,output);ASSERT_FALSE(::testing::Test::HasFailure());
      const auto fresh=carry.producer(output,std::uint64_t(producer_calls));
      auto a=typed<std::int32_t>(take(seed.member(carry.entries[0]),"first producer IDs"));
      auto b=typed<std::int32_t>(output[6]);std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
      EXPECT_NE(a,b)<<"different actual history must change selected membership";
      carry.fresh(fresh);output.clear();
      const auto result=consume(carry,skip,queue,owner,take(carry.current(),"fresh reversed-history epoch"),3);++reuse_calls;
      EXPECT_EQ(carry.facts(result).producer,3u);EXPECT_EQ(carry.facts(result).uses,1u);
      EXPECT_FALSE(carry.qualify(old_epoch));
    }
    EXPECT_FALSE(carry.current());EXPECT_FALSE(carry.qualify(seed));
    EXPECT_TRUE(image(carry,seed)==seed_bytes);EXPECT_TRUE(image(carry,old_epoch)==old_bytes);
    auto old=take(old_epoch.member(carry.entries[0]),"retained historical actual output");
    EXPECT_FALSE(old.try_write());
  }
  check(carry.store.collect(),"collect original retired carry versions");
  EXPECT_EQ(carry.store.stats().versions,0u);EXPECT_EQ(carry.store.stats().drafts,0u);
  EXPECT_EQ(indexer.outputs.stats().active_uses,0u);EXPECT_EQ(indexer.outputs.stats().live_blocks,0u);
  EXPECT_EQ(skip.outputs.stats().active_uses,0u);EXPECT_EQ(skip.outputs.stats().live_blocks,0u);
  EXPECT_EQ(producer_calls,3);EXPECT_EQ(reuse_calls,4);
  ::testing::Test::RecordProperty("actual_submissions",producer_calls+reuse_calls);
  ::testing::Test::RecordProperty("indexer_submissions",producer_calls);
  ::testing::Test::RecordProperty("skip_indexer_submissions",reuse_calls);
  ::testing::Test::RecordProperty("scope","single-layer GLM Flash generated-weight index carry; actual I32 ReadLease -> skip Gather/SDPA -> StateHead; no full model/original parameter claim");
  ::testing::Test::RecordProperty("epoch_policy","default-clear, keep-only-real-seed, finally-clear-and-disable, old-version-readonly-but-not-reusable");
  ::testing::Test::RecordProperty("backend",cuda?"ordinary CUDA Graph":"ordinary CPU Graph");
}
#if defined(HYLE_TEST_GLM_CARRY_CUDA)
TEST(GlmMtpIndexCarry,CudaActualI32LeaseSurvivesSkipAndObeysEpoch) {
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original CUDA memory")),"retained original storage domain");
  run(context,owner,true);
}
#endif
#if defined(HYLE_TEST_GLM_CARRY_CPU)
struct SystemResource final {m::MemoryResourceRef ref() noexcept {return m::system_resource();}};
TEST(GlmMtpIndexCarry,CpuActualI32LeaseSurvivesSkipAndObeysEpoch) {
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"system storage owner");
  run(context,owner,false);
}
#endif
} // namespace
