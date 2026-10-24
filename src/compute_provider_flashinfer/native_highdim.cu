#include "native_batch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <stdexcept>

namespace hyle::compute::flashinfer::native {
namespace {
namespace fi = ::flashinfer;
constexpr unsigned kDepth = 512;

template <class Params>
void bind_plan(Params &params, const fi::PrefillPlanInfo &plan, void *integer) {
  const auto pointer = [&](std::int64_t offset) {
    return reinterpret_cast<std::int32_t *>(static_cast<char *>(integer) + offset);
  };
  params.request_indices = pointer(plan.request_indices_offset);
  params.qo_tile_indices = pointer(plan.qo_tile_indices_offset);
  params.kv_tile_indices = pointer(plan.kv_tile_indices_offset);
  params.o_indptr = pointer(plan.o_indptr_offset);
  params.kv_chunk_size_ptr = pointer(plan.kv_chunk_size_ptr_offset);
  params.padded_batch_size = std::uint32_t(plan.padded_batch_size);
  params.max_total_num_rows = std::uint32_t(plan.total_num_rows);
  if (plan.split_kv)
    params.merge_indptr = pointer(plan.merge_indptr_offset);
}

template <unsigned Tile, bool Paged, fi::MaskMode Mask, class Params>
cudaError_t launch_tile(Params params, const fi::PrefillPlanInfo &plan,
                        void *floating, cudaStream_t stream) {
  using Variant = fi::DefaultAttention<false, true, false, false>;
  auto *temporary = plan.split_kv ? reinterpret_cast<typename Params::DTypeO *>(
                                      static_cast<char *>(floating) + plan.v_offset)
                                : nullptr;
  auto *scores = plan.split_kv ? reinterpret_cast<float *>(
                                   static_cast<char *>(floating) + plan.s_offset)
                             : nullptr;
  if constexpr (Paged)
    return fi::BatchPrefillWithPagedKVCacheDispatched<
        Tile, kDepth, kDepth, fi::PosEncodingMode::kNone, false, Mask, Variant>(
        params, temporary, scores, false, stream);
  else
    return fi::BatchPrefillWithRaggedKVCacheDispatched<
        Tile, kDepth, kDepth, fi::PosEncodingMode::kNone, false, Mask, Variant>(
        params, temporary, scores, false, stream);
}

template <bool Paged, fi::MaskMode Mask, class Params>
cudaError_t launch(Params params, const fi::PrefillPlanInfo &plan,
                   void *floating, cudaStream_t stream) {
  // The locked planner selects exactly these tiles for value depth >= 512.
  // Its original VO-split kernels keep each per-warp output fragment bounded.
  switch (plan.cta_tile_q) {
  case 16:
    return launch_tile<16, Paged, Mask>(params, plan, floating, stream);
  case 32:
    return launch_tile<32, Paged, Mask>(params, plan, floating, stream);
  default:
    throw std::runtime_error("FlashInfer high-dimensional batch CTA differs from plan");
  }
}

template <bool Paged, class Params>
cudaError_t launch_mask(Params params, const fi::PrefillPlanInfo &plan,
                        BatchSpec spec, void *floating, cudaStream_t stream) {
  return spec.causal
             ? launch<Paged, fi::MaskMode::kCausal>(params, plan, floating, stream)
             : launch<Paged, fi::MaskMode::kNone>(params, plan, floating, stream);
}

template <class T>
cudaError_t run_typed(const fi::PrefillPlanInfo &plan, BatchSpec spec,
                      const void *q, const void *k, const void *v,
                      const void *metadata, void *output, void *floating,
                      void *integer, cudaStream_t stream) {
  const auto pointer = [&](std::size_t offset) {
    return reinterpret_cast<std::int32_t *>(
        const_cast<char *>(static_cast<const char *>(metadata)) + offset);
  };
  if (spec.paged) {
    fi::BatchPrefillPagedParams<T, T, T, std::int32_t> params;
    params.q = const_cast<T *>(static_cast<const T *>(q));
    params.o = static_cast<T *>(output);
    params.q_indptr = pointer(spec.query_indptr_offset);
    params.paged_kv = fi::paged_kv_t<T, std::int32_t>(
        spec.kv_heads, spec.page_size, kDepth, spec.requests, fi::QKVLayout::kNHD,
        const_cast<T *>(static_cast<const T *>(k)),
        const_cast<T *>(static_cast<const T *>(v)),
        pointer(spec.page_indices_offset), pointer(spec.kv_indptr_offset),
        pointer(spec.tail_offset));
    params.num_qo_heads = spec.query_heads;
    params.group_size = fi::uint_fastdiv(spec.query_heads / spec.kv_heads);
    params.q_stride_n = spec.query_heads * kDepth;
    params.q_stride_h = kDepth;
    params.window_left = spec.window_left;
    params.sm_scale = spec.scale;
    bind_plan(params, plan, integer);
    return launch_mask<true>(params, plan, spec, floating, stream);
  }
  fi::BatchPrefillRaggedParams<T, T, T, std::int32_t> params;
  params.q = const_cast<T *>(static_cast<const T *>(q));
  params.k = const_cast<T *>(static_cast<const T *>(k));
  params.v = const_cast<T *>(static_cast<const T *>(v));
  params.o = static_cast<T *>(output);
  params.q_indptr = pointer(spec.query_indptr_offset);
  params.kv_indptr = pointer(spec.kv_indptr_offset);
  params.num_qo_heads = spec.query_heads;
  params.num_kv_heads = spec.kv_heads;
  params.group_size = fi::uint_fastdiv(spec.query_heads / spec.kv_heads);
  params.q_stride_n = spec.query_heads * kDepth;
  params.q_stride_h = kDepth;
  params.k_stride_n = params.v_stride_n = spec.kv_heads * kDepth;
  params.k_stride_h = params.v_stride_h = kDepth;
  params.window_left = spec.window_left;
  params.sm_scale = spec.scale;
  bind_plan(params, plan, integer);
  return launch_mask<false>(params, plan, spec, floating, stream);
}
} // namespace

cudaError_t run_highdim_batch(const void *state, BatchSpec spec, const void *q,
                              const void *k, const void *v, const void *metadata,
                              void *output, void *floating, void *integer,
                              cudaStream_t stream) {
  if (!spec.query_rows)
    return cudaSuccess;
  if (spec.qk_head_dim != kDepth || spec.value_head_dim != kDepth || spec.fp8)
    return cudaErrorInvalidValue;
  const auto &plan = *static_cast<const fi::PrefillPlanInfo *>(state);
  return spec.bfloat16
             ? run_typed<__nv_bfloat16>(plan, spec, q, k, v, metadata, output,
                                       floating, integer, stream)
             : run_typed<half>(plan, spec, q, k, v, metadata, output,
                              floating, integer, stream);
}
} // namespace hyle::compute::flashinfer::native
