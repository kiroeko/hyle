#pragma once

#include "hyle/graph/batch.h"
#include <algorithm>

namespace hyle::graph::batch::detail {
// Cold compiler qualification only. The original native compiler/runtime
// still qualifies the actual page math and owns every accepted submission.
inline Status audit_native_paged_nodes(const Graph& graph,
    std::span<const NativePagedNodeBinding> bindings,
    std::span<const std::uint8_t> live,
    std::span<const ExecutionTargetId> targets,ExecutionTargetId native) noexcept {
  for(std::size_t i=0;i<bindings.size();++i){
    const auto& binding=bindings[i];
    if(!binding.node.valid() || binding.node.ordinal()>=live.size())
      return Status::InvalidArgument("foreign native page node binding");
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(binding.node.ordinal()));
    if(node!=binding.node || !live[node.ordinal()] || targets[node.ordinal()]!=native ||
       !binding.program.valid() || !binding.program.schema().owns(binding.layout) ||
       !binding.options.max_metadata_backing_bytes || !binding.options.allocator.valid())
      return Status::InvalidArgument("native page node/recipe/layout/options/placement differs");
    for(std::size_t j=0;j<i;++j)if(bindings[j].node==node)
      return Status::InvalidArgument("duplicate native page node binding");
    HYLE_ASSIGN_OR_RETURN(const auto& operation,graph.operation(node));
    const auto descriptor=operation.custom_as<NativePagedAttention>();
    if(!descriptor || descriptor->capacity!=binding.program.schema().capacity() ||
       descriptor->active_query_rows!=binding.layout.facts().query_rows ||
       descriptor->scale!=binding.program.scale() || descriptor->causal!=binding.program.causal())
      return Status::InvalidArgument("native page node descriptor differs from its owned recipe/layout");
    for(std::size_t port=0;port<3;++port){
      HYLE_ASSIGN_OR_RETURN(auto input,graph.node_input(node,port));
      HYLE_ASSIGN_OR_RETURN(const auto& spec,graph.value_spec(input));
      HYLE_ASSIGN_OR_RETURN(auto description,spec.concrete_description());
      if(description!=binding.program.descriptions()[port])
        return Status::InvalidArgument("native page input is not its truthful recipe capacity");
      if(!port)continue;
      if(input.kind()!=ValueKind::kInput)
        return Status::InvalidArgument("native page K/V needs an external exact frame input");
      for(std::size_t output=0;output<graph.output_count();++output){
        HYLE_ASSIGN_OR_RETURN(auto published,graph.output_id(output));
        if(published==input)return Status::InvalidArgument("page capacity cannot be published");
      }
      for(std::size_t reader=0;reader<graph.node_count();++reader){
        if(!live[reader])continue;
        HYLE_ASSIGN_OR_RETURN(auto reader_node,graph.node_id(reader));
        HYLE_ASSIGN_OR_RETURN(auto count,graph.node_input_count(reader_node));
        for(std::size_t p=0;p<count;++p){
          HYLE_ASSIGN_OR_RETURN(auto value,graph.node_input(reader_node,p));
          if(value!=input)continue;
          const auto found=std::find_if(bindings.begin(),bindings.end(),
              [&](const auto& other){return other.node==reader_node;});
          if((p!=1 && p!=2) || found==bindings.end() || found->layout!=binding.layout)
            return Status::InvalidArgument("page capacity has an unqualified or different-layout reader");
        }
      }
    }
  }
  for(std::size_t i=0;i<graph.node_count();++i){
    if(!live[i])continue;
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(const auto& operation,graph.operation(node));
    if(operation.custom_as<NativePagedAttention>() &&
       std::none_of(bindings.begin(),bindings.end(),[&](const auto& b){return b.node==node;}))
      return Status::InvalidArgument("live native page node has no owned cold binding");
  }
  return Status::Ok();
}
} // namespace hyle::graph::batch::detail
