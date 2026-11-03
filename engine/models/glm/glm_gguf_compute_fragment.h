#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>

#include "glm_gguf_model_types.h"
#include "glm_gguf_projection_fragment.h"
#include "glm_target_verifier_fragment.h"

// Explicit F32 COMPUTE leaves around original converted packed parameters.
// No implicit cache, hidden/journal store, index quantization, selection or
// state controller is chosen here. A caller must apply its declared store
// boundary and pass the resulting ACTUAL operand into the next leaf.
namespace glm_gguf_compute_fragment {
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
namespace model=glm_gguf_model_types;
namespace leaf=glm_gguf_projection_fragment;
using Value=sem::ValueId;
struct Norm final {
  Value input,output;
  template<class Visitor> void visit_values(Visitor &&v){v(input);v(output);}
};
Norm norm(sem::Fragment &f,Value actual,Value gamma,float epsilon);
Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> shape);
Value slice(sem::Fragment &f,Value value,std::int32_t axis,std::int64_t first,std::int64_t count);
Value copy(sem::Fragment &f,Value value);

struct Prefix final {
  Norm hidden_norm,embedding_norm,input_norm,query_norm;
  Value joined;
  leaf::Projection eh,query;
  template<class Visitor> void visit_values(Visitor &&v) {
    hidden_norm.visit_values(v);embedding_norm.visit_values(v);v(joined);
    eh.visit_values(v);input_norm.visit_values(v);query.visit_values(v);query_norm.visit_values(v);
  }
};
template<class Projections>
Prefix append_f32_prefix(sem::Fragment &f,Value actual_hidden,Value actual_embedding,
    const model::NextnParameters &p,const model::Geometry &g,const Projections &project) {
  model::require_geometry(g);
  const auto &h=f.description(actual_hidden),&e=f.description(actual_embedding);
  if(h!=e||h.rank()!=2||h.dimensions()[0]<=0||h.dimensions()[1]!=g.hidden||
      !h.is_plain()||!h.is_dense()||h.data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("converted GLM prefix requires actual F32 rows, not an implicit BF16 store");
  Prefix result;
  result.hidden_norm=norm(f,actual_hidden,p.hidden_norm,g.rms_epsilon);
  result.embedding_norm=norm(f,actual_embedding,p.embedding_norm,g.rms_epsilon);
  // The converted EH matrix retains original embedding-before-hidden order.
  result.joined=f.one(c::Operation(c::Concat{.axis=1}),
      {result.embedding_norm.output,result.hidden_norm.output});
  result.eh=project.linear(f,result.joined,p.eh);
  result.input_norm=norm(f,result.eh.output,p.block.input_norm,g.rms_epsilon);
  result.query=project.linear(f,result.input_norm.output,p.block.query_a);
  result.query_norm=norm(f,result.query.output,p.block.query_norm,g.rms_epsilon);
  return result;
}

struct Absorption final {
  leaf::Projection query,absorbed;
  Value nope;
  std::optional<Value> rope_suffix;
  template<class Visitor> void visit_values(Visitor &&v) {
    query.visit_values(v);v(nope);if(rope_suffix)v(*rope_suffix);absorbed.visit_values(v);
  }
};
template<class Projections>
Absorption append_f32_absorption(sem::Fragment &f,Value actual_query_norm,
    const model::Block &p,const model::Geometry &g,const Projections &project) {
  model::require_geometry(g);
  const auto &input=f.description(actual_query_norm);
  if(input.rank()!=2||input.dimensions()[0]<=0||input.dimensions()[1]!=g.query_rank)
    throw std::runtime_error("converted GLM absorbed query extent differs");
  const auto rows=input.dimensions()[0];
  Absorption result;result.query=project.linear(f,actual_query_norm,p.query_b);
  const auto heads=reshape(f,result.query.output,{rows,g.heads,g.nope+g.rope});
  result.nope=copy(f,slice(f,heads,2,0,g.nope));
  if(g.rope)result.rope_suffix=copy(f,slice(f,heads,2,g.nope,g.rope));
  result.absorbed=project.per_head(f,result.nope,p.key);
  // RoPE, any actual store, and concat with this suffix belong to the caller.
  // This is not the original unsplit E4 KV_b/BF16 BMM preparation contract.
  return result;
}
struct Output final {
  leaf::Projection value,output;
  template<class Visitor> void visit_values(Visitor &&v){value.visit_values(v);output.visit_values(v);}
};
template<class Projections>
Output append_f32_output(sem::Fragment &f,Value actual_attention,
    const model::Block &p,const model::Geometry &g,const Projections &project) {
  model::require_geometry(g);
  const auto &input=f.description(actual_attention);
  if(input.rank()!=3||input.dimensions()[0]<=0||input.dimensions()[1]!=g.heads||
      input.dimensions()[2]!=g.latent)
    throw std::runtime_error("converted GLM value projection needs actual [R,64,512] attention");
  const auto rows=input.dimensions()[0];Output result;
  result.value=project.per_head(f,actual_attention,p.value);
  result.output=project.linear(f,reshape(f,result.value.output,{rows,g.heads*g.value}),p.output);
  return result;
}

Value embedding_q8(sem::Fragment &f,Value actual_ids,Value original_rows,const model::Geometry &g);
glm_target_verifier_fragment::Ports append_q8_f32_verifier(sem::Fragment &f,
    Value actual_final_normalized_taps,Value original_head,const model::Geometry &g);
// Common accepted-prefix selection and the model-local accepted checkpoint/
// repair/index-catchup workflow consume these same direct leases.
} // namespace glm_gguf_compute_fragment
