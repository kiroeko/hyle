#include "glm_gguf_model_types.h"

namespace glm_gguf_model_types {
void require_geometry(const Geometry &g) {
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
}
