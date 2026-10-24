#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_qwen38_24t_semantic_fragments.h"

// Cold cuts of one actual model Fragment at its router frontiers.
// The caller runs Frontiers+1 ordinary Graphs on its existing Context/queues and
// forwards actual immutable leases by original ordinal. No executor or I/O
// callback is hidden in these pure model operations.
namespace model_resident_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using sem::ValueId;
struct Location final { std::size_t phase=0,port=0; };
struct Phase final {
  sem::Fragment fragment{.profile=nullptr,.region=0};
  // Parallel to fragment.inputs/outputs/constants. IDs address the original
  // complete Fragment, not another phase's local ValueId namespace.
  std::vector<ValueId> original_inputs,original_outputs,original_constants;
  std::optional<std::size_t> resident_stage{};
};
template<class Boundary,std::size_t Frontiers> struct ResidentDraft final {
  static_assert(Frontiers>0);
  std::array<Phase,Frontiers+1> phases;
  std::vector<Location> full_outputs;
  std::array<Boundary,Frontiers> routers;
};
template<class Boundary,std::size_t Frontiers>
inline ResidentDraft<Boundary,Frontiers> split(const sem::Fragment &full,
    const std::array<Boundary,Frontiers> &routers,std::uint64_t region_base) {
  static_assert(Frontiers>0);
  if(region_base>std::numeric_limits<std::uint64_t>::max()-Frontiers)
    throw std::runtime_error("resident phase region identities overflow");
  if(!full.state_ports.empty())throw std::runtime_error("resident cuts require the read-only encoded-history draft");
  constexpr auto missing=std::numeric_limits<std::size_t>::max();
  const auto count=full.descriptions.size();
  std::vector<std::size_t> producer(count,missing),uses(count,0),availability(count,0);
  std::vector<bool> constant(count,false),external(count,false),published(count,false);
  for(auto value:full.constants)constant.at(value.ordinal())=true;
  for(auto value:full.inputs)external.at(value.ordinal())=true;
  for(auto value:full.outputs)published.at(value.ordinal())=true;
  for(std::size_t index=0;index<full.nodes.size();++index) {
    for(auto value:full.nodes[index].outputs) {
      if(producer.at(value.ordinal())!=missing)throw std::runtime_error("duplicate resident value producer");
      producer[value.ordinal()]=index;
    }
    for(auto value:full.nodes[index].inputs)++uses.at(value.ordinal());
  }
  std::array<std::size_t,Frontiers> cuts;
  for(std::size_t stage=0;stage<routers.size();++stage) {
    const auto &router=routers[stage];
    cuts[stage]=0;
    for(auto value:{router.input,router.ids,router.weights}) {
      const auto source=producer.at(value.ordinal());
      if(source==missing)throw std::runtime_error("router frontier is not an actual Graph result");
      cuts[stage]=std::max(cuts[stage],source);
    }
    if(stage && cuts[stage]<=cuts[stage-1])throw std::runtime_error("router frontiers are not ordered");
    for(auto bank:router.banks) {
      if(!external.at(bank.ordinal()))throw std::runtime_error("resident bank is not an ordinary input");
      availability[bank.ordinal()]=stage+1;
    }
    if(!external.at(router.map.ordinal()))throw std::runtime_error("resident map is not an ordinary input");
    availability[router.map.ordinal()]=stage+1;
  }
  std::vector<std::size_t> node_phase(full.nodes.size());
  for(std::size_t index=0;index<full.nodes.size();++index) {
    auto phase=std::size_t(std::lower_bound(cuts.begin(),cuts.end(),index)-cuts.begin());
    const auto &node=full.nodes[index];
    // Terminal observation Copies were appended after cold assembly. Keep
    // them at their real predecessor's phase rather than retaining every
    // diagnostic through the final head just because of append order.
    if(node.operation.get_if<c::Copy>() && node.inputs.size()==1 && node.outputs.size()==1 &&
        published.at(node.outputs[0].ordinal()) && uses.at(node.outputs[0].ordinal())==0) {
      const auto source=producer.at(node.inputs[0].ordinal());
      phase=source==missing?availability[node.inputs[0].ordinal()]:node_phase.at(source);
    }
    node_phase[index]=phase;
    for(auto value:node.inputs) {
      const auto source=producer.at(value.ordinal());
      if(source!=missing && (source>=index || node_phase[source]>phase))
        throw std::runtime_error("resident cut would read a future producer");
      if(external[value.ordinal()] && availability[value.ordinal()]>phase)
        throw std::runtime_error("router prefix unexpectedly reads future resident weights");
    }
  }
  ResidentDraft<Boundary,Frontiers> result;result.routers=routers;
  std::array<std::vector<std::optional<ValueId>>,Frontiers+1> mapped;
  for(std::size_t phase=0;phase<Frontiers+1;++phase) {
    auto &part=result.phases[phase];
    part.fragment.profile=full.profile;part.fragment.region=region_base+phase;
    part.fragment.quantized=full.quantized;part.fragment.phase=full.phase;
    if(phase)part.resident_stage=phase-1;
    mapped[phase].resize(count);
  }
  const auto input=[&](std::size_t phase,ValueId original) {
    auto &entry=mapped[phase].at(original.ordinal());
    if(!entry) {
      auto &part=result.phases[phase];
      if(constant[original.ordinal()]) {
        // All full constants are cold prepared/uploaded once by the caller.
        // Each phase borrows that same ready immutable owner/lease.
        entry=part.fragment.constant(full.description(original),{});
        part.original_constants.push_back(original);
      } else {
        entry=part.fragment.input(full.description(original));
        part.original_inputs.push_back(original);
      }
    }
    return *entry;
  };
  for(std::size_t index=0;index<full.nodes.size();++index) {
    const auto phase=node_phase[index];auto &part=result.phases[phase];
    auto node=full.nodes[index];
    for(auto &value:node.inputs)value=input(phase,value);
    for(auto &value:node.outputs) {
      const auto original=value;
      if(mapped[phase][original.ordinal()])throw std::runtime_error("resident node result was bound as an input");
      value=part.fragment.value(full.description(original),sem::ValueKind::kNodeOutput);
      mapped[phase][original.ordinal()]=value;
    }
    part.fragment.nodes.push_back(std::move(node));
  }
  const auto expose=[&](std::size_t phase,ValueId original) {
    auto &part=result.phases[phase];
    const auto known=std::find_if(part.original_outputs.begin(),part.original_outputs.end(),
        [&](ValueId value){return value.ordinal()==original.ordinal();});
    if(known!=part.original_outputs.end())return Location{phase,std::size_t(known-part.original_outputs.begin())};
    const auto actual=input(phase,original);
    const auto source=producer[original.ordinal()];
    // Existing observer Copies already provide independent caller backing
    // and preserve the public Copy->actual-producer inspection seam.
    const bool copied=source!=missing && full.nodes[source].operation.get_if<c::Copy>();
    part.fragment.outputs.push_back(copied?actual:part.fragment.one(c::Operation(c::Copy{}),{actual}));
    part.original_outputs.push_back(original);
    return Location{phase,part.original_outputs.size()-1};
  };
  // Only real cross-phase dependencies are exposed. The caller retains each
  // origin lease until its last later reader; no host tensor round-trip or
  // pass-through arithmetic is inserted between phases.
  for(std::size_t phase=0;phase<Frontiers+1;++phase)
    for(auto original:result.phases[phase].original_inputs) {
      const auto source=producer[original.ordinal()];
      if(source!=missing) {
        const auto origin=node_phase[source];
        if(origin>=phase)throw std::runtime_error("resident cross-phase source is not earlier");
        (void)expose(origin,original);
      } else if(!external[original.ordinal()])throw std::runtime_error("resident input has no original binding or producer");
    }
  for(const auto &router:routers)
    for(auto value:{router.input,router.ids,router.weights})(void)expose(node_phase[producer[value.ordinal()]],value);
  for(auto original:full.outputs) {
    const auto source=producer[original.ordinal()];
    result.full_outputs.push_back(expose(source==missing?availability[original.ordinal()]:node_phase[source],original));
  }
  return result;
}
} // namespace model_resident_fragment
