#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "d5_qwen38_24t_semantic_fragments.h"

namespace glm_gguf_model_types {
namespace sem=d5_qwen38_24t_semantic_fragments;
using Value=sem::ValueId;
struct Geometry final {
  std::int64_t hidden,query_rank,heads,nope,rope,latent,value,vocabulary;
  std::int64_t index_depth,index_heads,index_pool,index_topk;
  float rms_epsilon,index_epsilon,attention_scale;
};
// No family-name presets: e.g. source Flash index_topk may describe tokens or
// compressed rows depending on the actual artifact/runtime contract. Every
// caller supplies the authenticated geometry and score-domain TopK explicitly.
inline void require_geometry(const Geometry &g) {
  for(const auto n:{g.hidden,g.query_rank,g.heads,g.nope,g.latent,g.value,g.vocabulary,
      g.index_depth,g.index_heads,g.index_pool,g.index_topk})
    if(n<=0)throw std::runtime_error("converted GLM geometry must have positive extents");
  const auto limit=std::numeric_limits<std::int64_t>::max();
  if(g.rope<0||g.rope%2||g.rope>g.index_depth||g.nope>limit-g.rope||
      g.latent>limit-g.rope||g.heads>limit/(g.nope+g.rope)||g.heads>limit/g.value||
      g.index_heads>limit/g.index_depth||g.vocabulary>std::numeric_limits<std::int32_t>::max()||
      !(g.rms_epsilon>0)||!std::isfinite(g.rms_epsilon)||!(g.index_epsilon>0)||
      !std::isfinite(g.index_epsilon)||!(g.attention_scale>0)||!std::isfinite(g.attention_scale))
    throw std::runtime_error("converted GLM geometry/normalization/scale overflow or invalid value");
}
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
