#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#if defined(HYLE_TEST_Q27_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_Q27_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "d5_gguf_test_support.h"
#include "d5_qwen35_gguf_fragment_support.h"
#include "d5_qwen38_gdn_fixture.h"
#include "qwen_original_mtp_roles.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace q27=qwen_original_mtp_roles::q27;
using Bytes=std::vector<std::byte>;
constexpr auto kHidden=q27::kHidden,kWidth=q27::kWidth;
constexpr auto kProjectionBytes=q27::kProjectionBytes;
constexpr auto kEpsilon=q27::kEpsilon;
constexpr auto kProjectionSha=q27::kProjectionSha;
constexpr auto kEmbeddingNormSha=q27::kEmbeddingNormSha;
constexpr auto kHiddenNormSha=q27::kHiddenNormSha;

// Fixed MAIN Qwen3.8-27B-UD-Q4_K_M.gguf, not the separate C4 MTP sidecar:
// unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502.
// blk.64.nextn EH Q6_K[10240,5120] and F32 enorm/hnorm[5120]. Metadata
// qwen35.attention.layer_norm_rms_epsilon is exactly 1e-6F. Fixed llama
// qwen35 graph_mtp normalizes F32 e/h, concatenates e BEFORE h, then EH.
// Only this input component is tested: no decoder/KV/head/accepted state.
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
  if(_dupenv_s(&value,&size,"HYLE_QWEN27_MTP_INPUT_DIR")) throw std::runtime_error("cannot read Q27 original-role directory");
  const std::unique_ptr<char,decltype(&std::free)> owner(value,&std::free);
  return value?value:"";
#else
  const auto value=std::getenv("HYLE_QWEN27_MTP_INPUT_DIR");return value?value:"";
#endif
}
template<class T> Bytes bytes(std::span<const T> values) {
  const auto view=std::as_bytes(values);return {view.begin(),view.end()};
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(),"readonly original result").read(result),"read retained actual bytes");return result;
}
std::vector<float> floats(const Bytes &bytes) {
  if(bytes.size()%sizeof(float)) throw std::runtime_error("F32 result extent");
  std::vector<float> result(bytes.size()/sizeof(float));
  std::memcpy(result.data(),bytes.data(),bytes.size());return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> payload) {
  auto storage=take(m::Buffer::create(owner.ref(),payload.size(),256),"actual parameter/input backing");
  check(storage.view().write(payload),"upload exact original or declared input bytes");
  return take(owner.adopt_buffer(std::move(storage)),"immutable real backing permission");
}
c::TensorDesc dense(std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(c::DataType::kFloat32,{dims.begin(),dims.size()}),"F32 logical description");
}
using Roles=q27::Data;
using q27::project;
struct Logical final {
  g::Graph graph;
  std::array<g::ValueId,2> inputs;
  std::array<g::ValueId,3> parameters;
  std::array<g::ValueId,4> outputs;
  std::array<c::TensorDesc,3> parameter_desc;
  std::array<c::TensorDesc,4> output_desc;
};
Logical build() {
  Logical result;
  const auto row=dense({1,kHidden}),joined_desc=dense({1,kWidth});
  result.parameter_desc={dense({kHidden}),dense({kHidden}),take(c::gguf_weight_description(
      c::GgufQuantizedType::kQ6K,c::DataType::kFloat32,kWidth,kHidden),"original packed Q6_K descriptor")};
  if(take(result.parameter_desc[2].required_bytes(),"original packed extent")!=kProjectionBytes)
    throw std::runtime_error("Q27 original packed EH geometry differs");
  auto builder=take(g::GraphBuilder::create(),"ordinary Q27 input Graph");
  for(auto &input:result.inputs) input=take(builder.input(row),"F32 hidden/embedding boundary");
  for(std::size_t i=0;i<result.parameters.size();++i)
    result.parameters[i]=take(builder.constant(result.parameter_desc[i]),"typed original parameter role");
  const auto hn=take(builder.add(c::RmsNorm{.epsilon=kEpsilon},result.inputs[0],result.parameters[1]),"original hidden norm");
  const auto en=take(builder.add(c::RmsNorm{.epsilon=kEpsilon},result.inputs[1],result.parameters[0]),"original embedding norm");
  const auto joined=take(builder.add(c::Concat{.axis=1},en,hn),"embedding-before-hidden concat");
  const auto projected=take(builder.add(c::MatMul{},joined,result.parameters[2]),"original Q6_K EH projection");
  result.outputs={hn,en,joined,projected};result.output_desc={row,row,joined_desc,row};
  for(const auto output:result.outputs) check(builder.publish(output),"publish input component result");
  result.graph=take(std::move(builder).finish(),"finish Q27 input Graph");return result;
}
std::vector<float> input(int generation,bool embedding) {
  std::vector<float> result(kHidden);
  if(generation==2) return result;
  for(std::size_t i=0;i<result.size();++i) {
    const auto value=int((i*(embedding?37u:19u)+std::size_t(embedding?11:3)+std::size_t(generation)*23u)%97u)-48;
    result[i]=float(value)/(embedding?64.F:32.F)+float(i%7)/32768.F;
  }
  return result;
}
void quantized_gate(std::span<const float> actual,std::span<const float> expected,const char *label) {
  SCOPED_TRACE(label);
  ASSERT_EQ(actual.size(),expected.size());
  float scale=0;double square=0,max_error=0;
  for(std::size_t i=0;i<expected.size();++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));ASSERT_TRUE(std::isfinite(expected[i]));
    scale=std::max(scale,std::abs(expected[i]));
    const double error=double(actual[i])-expected[i];square+=error*error;max_error=std::max(max_error,std::abs(error));
  }
  std::printf("Q27 original Q6_K %s maxAbs=%.17g rmse=%.17g referencePeak=%.17g\n",
      label,max_error,std::sqrt(square/double(expected.size())),double(scale));
  if(scale==0) {for(const auto value:actual) EXPECT_EQ(value,0.F);return;}
  std::vector<float> a(actual.size()),e(expected.size());
  for(std::size_t i=0;i<expected.size();++i) {a[i]=actual[i]/scale;e[i]=expected[i]/scale;}
  // Exactly the established quantized-chain gate and scale normalization in
  // d5_qwen38_24t_moe_profiles.h:59-83, not a new Q27-specific tolerance.
  // Original CPU/GPU dots may quantize F32 activations to Q8_K/Q8_1; this
  // independently reduced decoded-weight equation retains the F32 input.
  d5_gguf_test::expect_quantized_chain_matmul_close(a,e);
}

void run(const Roles &roles,c::Context &context,const c::StorageOwner &owner,bool cuda) {
  const auto logical=build();
  const std::array<std::span<const std::byte>,3> payloads{
      std::as_bytes(std::span(roles.embedding_norm)),std::as_bytes(std::span(roles.hidden_norm)),roles.projection};
  const auto bound=q27::bind(roles,owner);
  std::array<c::ReadLease,3> parameters;
  std::array<g::ConstantBinding,3> constants;
  for(std::size_t i=0;i<parameters.size();++i) {
    parameters[i]=bound[i].lease;
    constants[i]=take(bound[i].binding(logical.parameters[i]),"share original immutable parameter owner");
  }
  const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(
      d5_qwen38_gdn_fixture::kPrimaryGgufProfile),"fixed primary main-GGUF profile"),c::WorkloadPhase::kDecode,
      {0x48594c454d4f444cULL,0x5132374d5450494eULL,1}};
  const g::CompileOptions options{.constants=constants,.workload=workload};
  const auto compile=[&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_Q27_CPU)
    if(!cuda) return g::cpu::compile(logical.graph,context,options);
#endif
#if defined(HYLE_TEST_Q27_CUDA)
    if(cuda) return g::cuda::compile(logical.graph,context,owner,options);
#endif
    throw std::runtime_error("requested original-role consumer backend was not enabled");
  };
  auto compiled=take(compile(),"ordinary original Q27 component compile");
  std::size_t generation_bytes=0;
  for(std::size_t i=0;i<logical.outputs.size();++i) {
    ASSERT_EQ(take(compiled.executable.output_ownership(i),"output ownership"),g::OutputOwnership::kCaller);
    generation_bytes+=(take(logical.output_desc[i].required_bytes(),"output extent")+255u)/256u*256u;
  }
  auto pool=take(c::LeasePool::create(owner.ref(),2*generation_bytes,
      {.max_blocks=8,.backing_alignment=256}),"bounded two-generation component outputs");
  auto queue=take(context.create_queue(),"original queue");
  std::array<c::ReadLease,4> oldest;
  std::array<Bytes,4> old_bytes;
  bool distinct_order=false;
  {
    auto invocation=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"one original component Invocation");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    for(int generation=0;generation<3;++generation) {
      SCOPED_TRACE(generation);
      const auto h=input(generation,false),e=input(generation,true);
      check(invocation.bind(logical.inputs[0],logical.output_desc[0],upload(owner,bytes<float>(h))),"current hidden permission");
      check(invocation.bind(logical.inputs[1],logical.output_desc[1],upload(owner,bytes<float>(e))),"current embedding permission");
      for(std::size_t i=0;i<logical.outputs.size();++i)
        check(invocation.bind(logical.outputs[i],logical.output_desc[i],take(pool.allocate(
            take(logical.output_desc[i].required_bytes(),"new output extent")),"fresh output writer")),"bind current output");
      check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare ordinary component");
      {auto done=take(invocation.submit(queues),"strict prepared component submit");check(done.wait(),"actual component retirement");}
      std::array<c::ReadLease,4> current;
      std::array<Bytes,4> actual_bytes;
      std::array<std::vector<float>,4> actual;
      for(std::size_t i=0;i<logical.outputs.size();++i) {
        auto result=take(invocation.result(logical.outputs[i]),"successful actual result");
        current[i]=take(result.storage_lease(),"independent result permission");
        actual_bytes[i]=read(current[i]);actual[i]=floats(actual_bytes[i]);
      }
      const auto hn=d5_qwen35_gguf_fragment::rms_norm(h,roles.hidden_norm,kEpsilon);
      const auto en=d5_qwen35_gguf_fragment::rms_norm(e,roles.embedding_norm,kEpsilon);
      d5_qwen35_gguf_fragment::expect_close(actual[0],hn,3e-5F,2e-5F);
      d5_qwen35_gguf_fragment::expect_close(actual[1],en,3e-5F,2e-5F);
      auto joined_bytes=actual_bytes[1];joined_bytes.insert(joined_bytes.end(),actual_bytes[0].begin(),actual_bytes[0].end());
      EXPECT_TRUE(actual_bytes[2]==joined_bytes)<<"embedding-before-hidden byte order";
      distinct_order|=actual_bytes[0]!=actual_bytes[1];
      auto joined=en;joined.insert(joined.end(),hn.begin(),hn.end());
      const auto expected=project(roles.projection,actual[2],joined);
      quantized_gate(actual[3],expected.local,"actual-joined local equation");
      quantized_gate(actual[3],expected.whole,"independent norm-to-EH chain equation");
      ASSERT_FALSE(::testing::Test::HasFailure());
      if(generation==2) for(const auto &values:actual) for(const auto value:values) EXPECT_EQ(value,0.F);
      if(generation==1) EXPECT_NE(actual_bytes[3],old_bytes[3]);
      check(invocation.reset(),"withdraw completed permissions");
      EXPECT_FALSE(invocation.submit(queues));
      if(generation==0) {oldest=std::move(current);old_bytes=actual_bytes;}
      else for(std::size_t i=0;i<oldest.size();++i) EXPECT_TRUE(read(oldest[i])==old_bytes[i]);
    }
  }
  EXPECT_TRUE(distinct_order);
  for(std::size_t i=0;i<oldest.size();++i) EXPECT_TRUE(read(oldest[i])==old_bytes[i]);
  oldest={};
  EXPECT_EQ(pool.stats().live_blocks,0u);EXPECT_EQ(pool.stats().active_uses,0u);
  for(std::size_t i=0;i<parameters.size();++i) {
    const auto observed=read(parameters[i]);
    EXPECT_TRUE(std::equal(observed.begin(),observed.end(),payloads[i].begin(),payloads[i].end()))<<"original immutable parameter bytes changed";
  }
  ::testing::Test::RecordProperty("scope","original Q27 primary main-GGUF blk64 Q6_K EH plus two F32 norms; generated F32 input boundary only, not complete MTP model/state");
  ::testing::Test::RecordProperty("backend",cuda?"ordinary CUDA Graph":"ordinary CPU Graph");
  ::testing::Test::RecordProperty("actual_submissions",3);
  ::testing::Test::RecordProperty("original_role_bytes",43048960);
  ::testing::Test::RecordProperty("eh_proj_sha256",kProjectionSha);
  ::testing::Test::RecordProperty("enorm_sha256",kEmbeddingNormSha);
  ::testing::Test::RecordProperty("hnorm_sha256",kHiddenNormSha);
}

class Qwen27OriginalMtpInput:public ::testing::Test {
protected:
  static inline std::unique_ptr<const Roles> parameters;
  static void SetUpTestSuite() {
    const auto path=directory();
    if(!path.empty()) parameters=std::make_unique<const Roles>(Roles::load(path));
  }
  static void TearDownTestSuite() {parameters.reset();}
  void SetUp() override {
    if(!parameters) GTEST_SKIP()<<"Set HYLE_QWEN27_MTP_INPUT_DIR to the authenticated three original main-GGUF roles";
  }
};
#if defined(HYLE_TEST_Q27_CUDA)
TEST_F(Qwen27OriginalMtpInput,CudaPrimaryQ6KRolesKeepOrderAndPriorResults) {
  auto context=take(g::cuda::create_context(),"ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"actual device resource")),"CUDA allocation owner");
  run(*parameters,context,owner,true);
}
#endif
#if defined(HYLE_TEST_Q27_CPU)
struct SystemResource final {m::MemoryResourceRef ref() noexcept {return m::system_resource();}};
TEST_F(Qwen27OriginalMtpInput,CpuPrimaryQ6KRolesKeepOrderAndPriorResults) {
  auto context=take(g::cpu::create_context(),"ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original system allocation owner");
  run(*parameters,context,owner,false);
}
#endif
} // namespace
