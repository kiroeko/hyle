#pragma once

#include <string>
#include <string_view>
#include <stdexcept>

#include "dspark_complete_graph_fixture.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"

// One existing native cold-parameter preparation and bounded independent
// BF16-source check, shared by CPU and CUDA model consumers. No hot executor.
namespace dspark_prepare_constants {
namespace fixture=dspark_complete_graph;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Bytes=std::vector<std::byte>;
template<class T> T take(hyle::Result<T> value,std::string_view where) {
  if(!value) throw std::runtime_error(std::string(where)+": "+std::string(value.error().message()));
  return std::move(*value);
}
inline void check(hyle::Status status,std::string_view where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
inline m::Buffer buffer(const c::StorageOwner &owner,std::size_t bytes) {
  return take(m::Buffer::create(owner.ref(),bytes,256),"cold parameter backing");
}
template<class Compile>
std::vector<m::Buffer> prepare(fixture::CompleteProgram &program,const c::StorageOwner &owner,
    c::Queue &queue,const c::WorkloadContext &workload,Compile compile) {
  const auto &cold = *program.fragment.preparation;
  if (!cold.inputs.empty() || cold.preparation) throw std::runtime_error("unexpected nested cold program");
  const auto logical = sem::make_graph(cold);
  std::vector<m::Buffer> sources, outputs;
  std::vector<g::ConstantBinding> constants;
  for (std::size_t i=0;i<cold.constants.size();++i) {
    sources.push_back(buffer(owner,cold.constant_bytes[i].size()));
    check(sources.back().view().write(cold.constant_bytes[i]),"cold source upload");
    constants.push_back({logical.constants[i],take(c::ConstTensorView::create(
        cold.description(cold.constants[i]),sources.back().view()),"cold immutable binding")});
  }
  auto compiled = take(compile(logical.graph,g::CompileOptions{.constants=constants,.workload=workload}),
      "single cold parameter Graph");
  auto run = take(compiled.executable.create_invocation(),"cold Invocation");
  for (std::size_t i=0;i<cold.outputs.size();++i) {
    const auto &desc = cold.description(cold.outputs[i]);
    outputs.push_back(buffer(owner,take(desc.required_bytes(),"cold output extent")));
    check(run.bind(logical.outputs[i],take(c::TensorView::create(desc,outputs.back().view()),"cold output view")),"bind cold output");
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  check(run.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),"prepare cold Graph");
  {
    auto completion = take(run.submit(queues),"strict cold Graph submission");
    check(completion.wait(),"cold actual retirement");
  }
  constexpr std::size_t chunk_elements = 262144;
  for (std::size_t port=0;port<outputs.size();++port) {
    const auto type = cold.description(cold.outputs[port]).data_type();
    const std::size_t width = type==c::DataType::kFloat32 ? 4 : 2;
    if (type!=c::DataType::kFloat32 && type!=c::DataType::kBFloat16)
      throw std::runtime_error("unexpected DSpark cold output dtype");
    const auto widening = std::find_if(program.cold_widenings.begin(),program.cold_widenings.end(),
        [&](const auto &entry){return entry.output==port;});
    const Bytes *source = nullptr;
    if (widening!=program.cold_widenings.end()) {
      const auto it = std::find_if(cold.constants.begin(),cold.constants.end(),
          [&](auto value){return value.ordinal()==widening->source.ordinal();});
      if (it==cold.constants.end() || type!=c::DataType::kFloat32)
        throw std::runtime_error("cold BF16 source metadata mismatch");
      source=&cold.constant_bytes[std::size_t(it-cold.constants.begin())];
      if (source->size()*2!=outputs[port].size()) throw std::runtime_error("cold widening extent mismatch");
    } else if (cold.replays.at(0).outputs.at(port).size()*width!=outputs[port].size()) {
      throw std::runtime_error("stage cold oracle extent mismatch");
    }
    const auto elements=outputs[port].size()/width;
    for (std::size_t first=0;first<elements;first+=chunk_elements) {
      const auto count=std::min(chunk_elements,elements-first);
      Bytes chunk(count*width);
      check(take(outputs[port].view().subview(first*width,chunk.size()),"cold chunk").read(chunk),"read cold parameter chunk");
      for (std::size_t i=0;i<count;++i) {
        float actual;
        if (type==c::DataType::kFloat32) std::memcpy(&actual,chunk.data()+i*width,4);
        else {std::uint16_t bits;std::memcpy(&bits,chunk.data()+i*width,2);actual=fixture::widen(bits);}
        float expected;
        if (source) {std::uint16_t bits;std::memcpy(&bits,source->data()+(first+i)*2,2);expected=fixture::widen(bits);}
        else expected=cold.replays[0].outputs[port][first+i];
        if (actual!=expected) throw std::runtime_error("cold parameter mismatch at output "+std::to_string(port)+" element "+std::to_string(first+i));
      }
    }
  }
  return outputs;
}
} // namespace dspark_prepare_constants
