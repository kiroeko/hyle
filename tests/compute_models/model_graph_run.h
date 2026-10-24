#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/graph/executable.h"
#include "../../engine/inference/prepared_output.h"

// Test-local ordinary Graph lifecycle shared by model consumers. Compilation,
// constants, model math, phase scheduling and accepted State stay with callers.
namespace model_graph_run {
namespace c=hyle::compute;
namespace g=hyle::graph;
struct Port final {
  g::ValueId id;
  c::TensorDesc description;
};
using OutputPort=hyle::inference::PreparedOutputFacts;
namespace detail {
template<class T> T take(hyle::Result<T> result,const char *where) {
  if(!result)throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status result,const char *where) {
  if(!result.ok())throw std::runtime_error(std::string(where)+": "+std::string(result.message()));
}
} // namespace detail
// These consumers deliberately keep their original independent caller pools.
// Capture facts at concrete cold compilation; do not infer ownership from a
// backend name or broaden this fixture's allocation policy during replay.
inline OutputPort prepare_output(const g::CompiledGraph &compiled,const Port &logical) {
  auto facts=detail::take(hyle::inference::prepare_output(compiled,logical.id),
      "prepare ordinary model Graph output");
  if(!facts.caller_owned())throw std::runtime_error("ordinary model Graph requires caller-owned result storage");
  if(facts.description!=logical.description)
    throw std::runtime_error("ordinary model Graph concrete output description changed");
  return facts;
}
inline std::vector<c::ReadLease> execute(g::Invocation &invocation,c::LeasePool &outputs,c::Queue &queue,
    std::span<const Port> input_ports,std::span<const c::ReadLease> inputs,
    std::span<const OutputPort> output_ports,std::size_t &completed_submissions,
    std::span<const std::size_t> externally_bound={},
    const std::function<void(g::Invocation&)> &bind_external={}) {
  if(inputs.size()!=input_ports.size())throw std::runtime_error("ordinary model Graph input count");
  std::vector<bool> external_ports(inputs.size(),false);
  for(const auto index:externally_bound) {
    if(index>=external_ports.size()||external_ports[index])throw std::runtime_error("ordinary model Graph resident input index");
    external_ports[index]=true;
  }
  if(!externally_bound.empty()&&!bind_external)throw std::runtime_error("ordinary model Graph resident binder missing");
  struct ResetOnFailure final {
    g::Invocation &active;bool done=false;
    ~ResetOnFailure(){if(!done)active.reset().IgnoreError();}
  } reset{invocation};
  for(std::size_t i=0;i<inputs.size();++i)if(!external_ports[i])
    detail::check(invocation.bind(input_ports[i].id,input_ports[i].description,inputs[i]),"bind ordinary model Graph input");
  if(bind_external)bind_external(invocation);
  for(const auto &port:output_ports) {
    const auto bytes=detail::take(port.description.required_bytes(),"ordinary model Graph output extent");
    detail::check(invocation.bind(port.value,port.description,
        detail::take(outputs.allocate(bytes),"ordinary model Graph output writer")),"bind ordinary model Graph output");
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  detail::check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),
      "prepare ordinary model Graph");
  {
    auto done=detail::take(invocation.submit(queues),"ordinary model Graph submit");
    detail::check(done.wait(),"ordinary model Graph retirement");
  }
  ++completed_submissions;
  std::vector<c::ReadLease> result;
  for(const auto &port:output_ports)
    result.push_back(detail::take(hyle::inference::retain_output(invocation,port),
        "retain ordinary model Graph result lease"));
  detail::check(invocation.reset(),"withdraw ordinary model Graph permissions");reset.done=true;
  if(invocation.submit(queues))throw std::runtime_error("reset ordinary model Graph retained stale replay");
  return result;
}
} // namespace model_graph_run
