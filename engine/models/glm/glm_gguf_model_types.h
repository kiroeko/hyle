#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "../detail/component_definition.h"

namespace glm_gguf_model_types {
namespace sem=hyle::inference::model_definition;
using Value=sem::ValueId;
struct Geometry final {
  std::int64_t hidden,query_rank,heads,nope,rope,latent,value,vocabulary;
  std::int64_t index_depth,index_heads,index_pool,index_topk;
  float rms_epsilon,index_epsilon,attention_scale;
};
// No family-name presets: e.g. source Flash index_topk may describe tokens or
// compressed rows depending on the actual artifact/runtime contract. Every
// caller supplies the authenticated geometry and score-domain TopK explicitly.
void require_geometry(const Geometry &g);
struct Projection final {
  Value input,weight,output;
  template<class Visitor> void visit_values(Visitor &&v){v(input);v(output);}
};
struct Block final {
  Value input_norm,query_a,query_norm,query_b,kv_a,kv_norm,key,value,output,post_norm;
  Value router,bias,shared_gate,shared_up,shared_down;
  Value index_key,index_query,index_norm,index_bias,index_weights;
};
struct NextnParameters final {
  Block block;
  Value eh,embedding_norm,hidden_norm,shared_norm,embedding,head;
};
} // namespace glm_gguf_model_types
