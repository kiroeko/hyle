#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include "qwen_full_attention_fragment.h"
#include "hyle/graph/executable.h"

// Only the common public-operation DAG and its closed-region report proof.
// Original-code centering, scale rounding and source metadata remain in the
// caller's typed codec, not a model-name branch or an execution abstraction.
namespace qwen_artifact_projection_fragment {
namespace attention=qwen_full_attention_fragment;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace sem=attention::sem;
using Value=attention::Value;
struct Arithmetic final {
  c::model_ops::WeightArtifactFormat format;
  std::uint32_t group_size;
  c::DataType decoded;
};
inline Value project(sem::Fragment &f,Value input,std::initializer_list<Value> original,
    const attention::Geometry &geometry,Arithmetic arithmetic) {
  const auto &activation=f.description(input);
  if(geometry.storage!=c::DataType::kBFloat16||activation.rank()!=2||!activation.is_plain()||
      !activation.is_dense()||activation.data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("original packed projection requires actual BF16 rows without activation quantization");
  const auto operand_type=activation.data_type(); // f.one may grow descriptions and invalidate references.
  const auto decoded=f.one(c::model_ops::artifact_weight_decode(
      arithmetic.format,arithmetic.group_size,arithmetic.decoded),original);
  // CT BF16->BF16 has no Cast. GPTQ F16->BF16 has one real Cast.
  const auto operand=arithmetic.decoded==operand_type?decoded:
      f.one(c::Operation(c::Cast{operand_type}),{decoded});
  constexpr std::array<std::int32_t,2> axes{1,0};
  const auto transpose=c::View::permute(axes);
  if(!transpose)throw std::runtime_error("original packed projection transpose description failed");
  const auto rhs=f.one(c::Operation(*transpose),{operand});
  return f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,rhs});
}
inline constexpr c::RoutePolicyId kPackedPolicy = g::kArtifactPackedProjectionPolicy;
// Call immediately after compile, before Invocation/workspace allocation.
// Node count is the actual public recipe: Decode/View/MatMul, with one Cast
// only when the original decoded dtype differs from the BF16 operand.
inline void require_packed(const g::CompileReport &report,std::span<const g::NodeId> matrices,
    std::size_t nodes) {
  if(matrices.empty()||(nodes!=3&&nodes!=4))
    throw std::runtime_error("original projection needs an explicit three/four-node packed recipe");
  for(const auto matrix:matrices) {
    std::size_t containing=0;
    for(std::size_t step_index=0;step_index<report.step_count();++step_index) {
      const auto step=report.step(step_index);
      if(!step)throw std::runtime_error("original packed compile step is unavailable");
      for(std::size_t index=0;index<step->node_count;++index) {
        const auto node=report.step_node(step_index,index);
        if(!node)throw std::runtime_error("original packed compile node is unavailable");
        if(*node!=matrix)continue;
        ++containing;
        if(step->policy!=kPackedPolicy||step->node_count!=nodes)
          throw std::runtime_error("original projection must retain its exact typed packed recipe");
      }
    }
    if(containing!=1)throw std::runtime_error("original matrix must occur in exactly one packed compile step");
  }
}
} // namespace qwen_artifact_projection_fragment
