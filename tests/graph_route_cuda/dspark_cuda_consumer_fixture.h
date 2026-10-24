#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/dspark_complete_graph_fixture.h"
#include "../compute_models/dspark_prepare_constants.h"

// The existing DSpark CUDA consumer's cold preparation and public lease IO.
// This helper owns no scheduler, model state, sampling policy or hidden copy.
namespace dspark_cuda_fixture {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace fixture = dspark_complete_graph;
namespace sem = d5_qwen38_24t_semantic_fragments;
using Bytes = std::vector<std::byte>;

template<class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
inline c::WorkloadContext workload(const sem::Fragment &fragment) {
  return {take(c::ModelArtifactProfile::create(*fragment.profile), "actual DSpark profile"),
          fragment.phase, {0x48594c454d4f444cULL, fragment.region, 1}};
}
inline std::string environment(const char *name) {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0) throw std::runtime_error("cannot read original-role environment");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto value = std::getenv(name); return value ? value : "";
#endif
}
inline m::Buffer buffer(const c::StorageOwner &owner, std::size_t size) {
  return take(m::Buffer::create(owner.ref(),size,256),"real device allocation");
}
inline c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> bytes) {
  auto output = buffer(owner,bytes.size());
  check(output.view().write(bytes),"upload actual immutable input");
  return take(c::ReadLease::adopt(std::move(output)),"own input permission");
}
inline Bytes read(const c::ReadLease &lease) {
  Bytes output(lease.size());
  check(take(lease.view(),"readonly independent view").read(output),"read actual result");
  return output;
}
template<class T> std::vector<T> decode(const Bytes &bytes) {
  if (bytes.size()%sizeof(T)) throw std::runtime_error("typed result byte extent");
  std::vector<T> output(bytes.size()/sizeof(T));
  std::memcpy(output.data(),bytes.data(),bytes.size()); return output;
}

// One cold Graph prepares every original stage output parameter and losslessly
// widens the shared BF16 LM/Markov/confidence sources. Nothing is rebuilt per
// replay. Check its outputs in bounded chunks, including the complete LM bank.
inline std::vector<m::Buffer> prepare_constants(fixture::CompleteProgram &program,
    c::Context &context, const c::StorageOwner &owner, c::Queue &queue) {
  return dspark_prepare_constants::prepare(program,owner,queue,workload(*program.fragment.preparation),
      [&](const g::Graph &graph,g::CompileOptions options) {
        return g::cuda::compile(graph,context,owner,options);
      });
}


// A fixed full-Flash proposal Graph. Shared immutable parameters are prepared
// exactly once; every execute uses this same Invocation and real permissions.
struct Draft final {
  sem::Fragment fragment;
  sem::GraphCase logical;
  std::vector<m::Buffer> parameters;
  g::CompiledGraph compiled;
  g::Invocation invocation;
  c::LeasePool outputs;
  c::Queue *queue = nullptr;

  static Draft create(c::Context &context, const c::StorageOwner &owner,
                      c::Queue &queue, fixture::HeadParameters heads) {
    auto program=fixture::build_flash(heads);
    auto prepared=prepare_constants(program,context,owner,queue);
    Draft result;
    result.fragment=std::move(program.fragment);
    result.fragment.preparation.reset();
    result.logical=sem::make_graph(result.fragment);
    result.parameters.resize(result.fragment.constants.size());
    std::vector<g::ConstantBinding> constants;
    for (std::size_t i=0;i<result.parameters.size();++i) {
      const auto found=std::find(result.fragment.prepared_constants.begin(),
          result.fragment.prepared_constants.end(),i);
      if (found!=result.fragment.prepared_constants.end())
        result.parameters[i]=std::move(prepared[std::size_t(found-result.fragment.prepared_constants.begin())]);
      else {
        result.parameters[i]=buffer(owner,result.fragment.constant_bytes[i].size());
        check(result.parameters[i].view().write(result.fragment.constant_bytes[i]),"one-time proposal constant upload");
      }
      constants.push_back({result.logical.constants[i],take(c::ConstTensorView::create(
          result.fragment.description(result.fragment.constants[i]),result.parameters[i].view()),"fixed proposal parameter")});
    }
    result.fragment.constant_bytes.clear();
    result.compiled=take(g::cuda::compile(result.logical.graph,context,owner,
        {.constants=constants,.workload=workload(result.fragment)}),"full proposal compile");
    result.invocation=take(result.compiled.executable.create_invocation(
        {.resources=g::ResourceOptions{}}),"single proposal Invocation");
    std::size_t bytes=0;
    for (std::size_t i=0;i<result.fragment.outputs.size();++i) {
      if (take(result.compiled.executable.output_ownership(i),"proposal output ownership")!=g::OutputOwnership::kCaller)
        throw std::runtime_error("proposal output must use a caller permission");
      const auto size=take(result.fragment.description(result.fragment.outputs[i]).required_bytes(),"proposal result bytes");
      bytes+=(size+255u)/256u*256u;
    }
    result.outputs=take(c::LeasePool::create(owner.ref(),bytes*2,
        {.max_blocks=fixture::kOutputs*2,.backing_alignment=256}),"two real proposal generations");
    result.queue=&queue;
    return result;
  }

  std::vector<c::ReadLease> execute(std::span<const c::ReadLease> inputs) {
    if (inputs.size()!=fragment.inputs.size()) throw std::runtime_error("proposal input count");
    for (std::size_t i=0;i<inputs.size();++i)
      check(invocation.bind(logical.inputs[i],fragment.description(fragment.inputs[i]),inputs[i]),"bind actual proposal input");
    for (std::size_t i=0;i<fragment.outputs.size();++i) {
      const auto &desc=fragment.description(fragment.outputs[i]);
      check(invocation.bind(logical.outputs[i],desc,take(outputs.allocate(
          take(desc.required_bytes(),"proposal output extent")),"new proposal result permission")),"bind proposal writer");
    }
    check(g::cuda::prepare_replay(invocation,*queue,
        {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare full proposal");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},queue}};
    {
      auto done=take(invocation.submit(queues),"strict prepared full proposal");
      check(done.wait(),"proposal actual terminal retirement");
    }
    std::vector<c::ReadLease> result;
    for (const auto id:logical.outputs) {
      auto output=take(invocation.result(id),"successful proposal output");
      result.push_back(take(output.storage_lease(),"independent proposal output"));
    }
    check(invocation.reset(),"withdraw completed proposal permissions");
    if (invocation.submit(queues)) throw std::runtime_error("reset proposal remained prepared");
    return result;
  }
};
} // namespace dspark_cuda_fixture
