// Keep the locked runtime and its qualified carriers. Only the supported
// long-token CUDA mathematics use the newer upstream MMA implementation.
// Compiling both paths in this original unit also avoids another CUDA module.
#define ggml_cuda_op_gated_delta_net hyle_cuda_gdn_serial
#define ggml_cuda_op_gated_delta_net_indexed hyle_cuda_gdn_indexed_serial
#define ggml_cuda_op_gated_delta_net_fused_cache hyle_cuda_gdn_cache_serial
#include HYLE_GGML_GDN_SOURCE
#undef ggml_cuda_op_gated_delta_net
#undef ggml_cuda_op_gated_delta_net_indexed
#undef ggml_cuda_op_gated_delta_net_fused_cache

#include "cuda_gdn_mma.cuh"

namespace {
bool try_gdn_mma(ggml_backend_cuda_context &ctx, ggml_tensor *dst,
                 const ggml_cuda_gated_delta_net_fused_cache *cache,
                 bool indexed_entry) {
  if (!dst || dst->op != GGML_OP_GATED_DELTA_NET ||
      ggml_get_op_params_i32(dst, 0) != 1 || !dst->src[0] ||
      !dst->src[2] || !dst->src[3] || !dst->src[4] || !dst->src[5])
    return false;
  const auto *q = dst->src[0], *k = dst->src[1], *v = dst->src[2];
  const auto *g = dst->src[3], *beta = dst->src[4], *state = dst->src[5];
  if (v->ne[0] != 128 || v->ne[2] < 64 || g->ne[0] != 1 ||
      v->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32)
    return false; // Decode, KDA and typed widening keep the original path.
  const bool indexed = ggml_cuda_gdn_is_indexed(dst);
  const bool packed = ggml_cuda_gdn_is_r1_packed(dst);
  // Reuse original metadata qualifications and entry separation. Native index
  // carriers still come solely from the admitted Graph slot-write assembly.
  if (indexed != indexed_entry || (indexed && cache)) return false;
  if (indexed) {
    if (!ggml_cuda_supports_gated_delta_net_indexed(dst)) return false;
  } else if (packed) {
    if (!ggml_cuda_supports_gdn_r1_packed(dst)) return false;
  } else if (!ggml_cuda_supports_gated_delta_net_typed(dst)) {
    return false;
  }
  // The original MMA kernel shares scalar gate/beta strides. Other layouts
  // retain their existing addressing and numerical implementation.
  if (g->nb[1] != beta->nb[1] || g->nb[2] != beta->nb[2] ||
      g->nb[3] != beta->nb[3]) return false;
  const float scale = hyle::compute::ggml_runtime::gdn_output_scale(
      dst, 1.0f / sqrtf(128.0f));
  if (!(scale > 0) || !std::isfinite(scale)) return false;

  hyle_cuda_gdn_mma_args a{};
  a.q = static_cast<const float *>(q->data);
  a.k = static_cast<const float *>(k->data);
  a.v = static_cast<const float *>(v->data);
  a.g = static_cast<const float *>(g->data);
  a.beta = static_cast<const float *>(beta->data);
  a.state = static_cast<const float *>(state->data);
  a.dst = static_cast<float *>(dst->data);
  a.H = v->ne[1]; a.H_k = q->ne[1];
  a.n_tokens = v->ne[2]; a.n_seqs = v->ne[3];
  a.rq3 = v->ne[3] / q->ne[3];
  a.sq1 = q->nb[1] / sizeof(float); a.sq2 = q->nb[2] / sizeof(float);
  a.sq3 = q->nb[3] / sizeof(float);
  a.sv1 = v->nb[1] / sizeof(float); a.sv2 = v->nb[2] / sizeof(float);
  a.sv3 = v->nb[3] / sizeof(float);
  a.sb1 = beta->nb[1] / sizeof(float); a.sb2 = beta->nb[2] / sizeof(float);
  a.sb3 = beta->nb[3] / sizeof(float);
  a.scale = scale; a.eligible = true;
  if (indexed) {
    a.state_out = static_cast<float *>(state->data);
    a.sources = static_cast<const int32_t *>(dst->src[6]->data);
    a.destinations = static_cast<const int32_t *>(dst->src[7]->data);
    a.slot_stride = state->nb[3] / sizeof(float);
    a.slots = static_cast<int32_t>(state->ne[3]);
  } else {
    a.state_out = cache ? cache->data : a.dst + 128 * a.H * a.n_tokens * a.n_seqs;
  }
  if (packed) a.packed_offsets = static_cast<const int32_t *>(dst->src[8]->data);
  return hyle_cuda_gdn_mma_launch(ctx.device, a, ctx.stream());
}
} // namespace

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
  if (!try_gdn_mma(ctx, dst, nullptr, false)) hyle_cuda_gdn_serial(ctx, dst);
}
void ggml_cuda_op_gated_delta_net_indexed(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
  if (!try_gdn_mma(ctx, dst, nullptr, true)) hyle_cuda_gdn_indexed_serial(ctx, dst);
}
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context &ctx, ggml_tensor *dst,
                                            ggml_cuda_gated_delta_net_fused_cache cache) {
  if (!try_gdn_mma(ctx, dst, &cache, false)) hyle_cuda_gdn_cache_serial(ctx, dst, cache);
}
