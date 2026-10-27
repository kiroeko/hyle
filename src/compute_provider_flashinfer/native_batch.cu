#include "native_batch.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <cstring>
#include <new>
#include <stdexcept>

namespace hyle::compute::flashinfer::native {
namespace fi=::flashinfer;
std::size_t batch_state_bytes() noexcept{return sizeof(fi::PrefillPlanInfo);}
void initialize_batch_state(void *p) noexcept{::new(p)fi::PrefillPlanInfo();}
void destroy_batch_state(void *p) noexcept{static_cast<fi::PrefillPlanInfo*>(p)->~PrefillPlanInfo();}
namespace {
// Fixed, small uniform query geometry is independent of the growing KV
// prefix. The scheduler still materializes actual chunk/merge/mask metadata.
// Other native dispatch domains keep their original scheduling semantics.
std::int64_t graph_query_length(BatchSpec s,const std::int32_t*q) noexcept {
  if(!s.paged || s.fp8 || !s.causal || s.window_left!=-1 || !s.requests || !q ||
      s.qk_head_dim!=s.value_head_dim ||
      (s.qk_head_dim!=64 && s.qk_head_dim!=128 && s.qk_head_dim!=256))return 0;
  const auto rows=q[1]-q[0];
  if(q[0]!=0 || (rows!=1 && rows!=2) ||
      std::uint64_t(rows)*s.requests!=s.query_rows)return 0;
  for(std::uint32_t r=0;r<s.requests;++r)
    if(q[r+1]-q[r]!=rows)return 0;
  return rows;
}
bool same_recorded_plan(const fi::PrefillPlanInfo&a,const fi::PrefillPlanInfo&b) noexcept {
  // Compare semantic fields, never struct padding or changing metadata bytes.
  return a.enable_cuda_graph && b.enable_cuda_graph &&
      a.padded_batch_size==b.padded_batch_size && a.total_num_rows==b.total_num_rows &&
      a.total_num_rows_offset==b.total_num_rows_offset && a.cta_tile_q==b.cta_tile_q &&
      a.request_indices_offset==b.request_indices_offset && a.qo_tile_indices_offset==b.qo_tile_indices_offset &&
      a.kv_tile_indices_offset==b.kv_tile_indices_offset && a.merge_indptr_offset==b.merge_indptr_offset &&
      a.o_indptr_offset==b.o_indptr_offset && a.kv_chunk_size_ptr_offset==b.kv_chunk_size_ptr_offset &&
      a.v_offset==b.v_offset && a.s_offset==b.s_offset &&
      a.block_valid_mask_offset==b.block_valid_mask_offset && a.split_kv==b.split_kv;
}
} // namespace
cudaError_t query_batch_workspace(BatchSpec s,const std::int32_t*q,const std::int32_t*k,cudaStream_t stream,BatchFacts&facts,
                                 const void*retained_plan,bool*preserves_commands,bool allow_graph_plan){
  facts={};
  if(preserves_commands)*preserves_commands=false;
  if(s.query_rows==0)return cudaSuccess;
  const auto uniform=allow_graph_plan?graph_query_length(s,q):0;
  fi::PrefillPlanInfo next;
  // The original scheduler's non-materializing implementation produces all
  // offsets and launch facts without changing the retained native plan.
  const auto status=fi::PrefillPlanImpl<false>(facts.float_bytes,facts.int_bytes,nullptr,0,nullptr,nullptr,0,next,
      const_cast<std::int32_t*>(q),const_cast<std::int32_t*>(k),s.query_rows,s.requests,s.query_heads,s.kv_heads,
      s.qk_head_dim,s.value_head_dim,s.paged?s.page_size:1,uniform>0,s.fp8?4:2,s.window_left,-1,false,0,uniform,
      stream,s.fp8?1:2,bf16_partial_domain(s));
  if(status!=cudaSuccess)return status;
  facts.cta_tile_q=std::uint32_t(next.cta_tile_q);facts.padded_batch_size=std::uint32_t(next.padded_batch_size);
  facts.split_kv=next.split_kv;facts.bf16_partials=bf16_partials_active(s,facts.cta_tile_q,facts.split_kv);
  if(preserves_commands && retained_plan)
    *preserves_commands=same_recorded_plan(*static_cast<const fi::PrefillPlanInfo*>(retained_plan),next);
  return status;
}
cudaError_t plan_batch(void *state,BatchSpec s,const std::int32_t*q,const std::int32_t*k,void*f,std::size_t fb,void*i,void*h,std::size_t ib,cudaStream_t stream,BatchFacts&facts,
                       bool preserve_commands,bool allow_graph_plan){
  auto&plan=*static_cast<fi::PrefillPlanInfo*>(state);
  if(s.query_rows==0){plan={};facts={};return cudaSuccess;}
  const auto uniform=allow_graph_plan?graph_query_length(s,q):0;
  fi::PrefillPlanInfo next;
  const auto result=fi::PrefillPlan(f,fb,i,h,ib,next,const_cast<std::int32_t*>(q),const_cast<std::int32_t*>(k),s.query_rows,s.requests,s.query_heads,s.kv_heads,s.qk_head_dim,s.value_head_dim,s.paged?s.page_size:1,uniform>0,s.fp8?4:2,s.window_left,-1,false,0,uniform,stream,s.fp8?1:2,bf16_partial_domain(s));
  if(result!=cudaSuccess)return result;
  // The original H2D may already be accepted. A violated preflight promise is
  // a terminal native error, never permission to replay changed arguments.
  if(preserve_commands && !same_recorded_plan(plan,next))return cudaErrorInvalidValue;
  plan=next;
  // The scheduler already materialized this scalar in the original host shadow.
  // Reuse must preserve that actual chunk size, rather than rerun its heuristic.
  if(plan.kv_chunk_size_ptr_offset<0)return cudaErrorInvalidValue;
  const auto offset=std::size_t(plan.kv_chunk_size_ptr_offset);
  if(offset>facts.int_bytes || facts.int_bytes-offset<sizeof(facts.kv_chunk_rows))return cudaErrorInvalidValue;
  std::memcpy(&facts.kv_chunk_rows,static_cast<const char*>(h)+offset,sizeof(facts.kv_chunk_rows));
  facts.cta_tile_q=std::uint32_t(plan.cta_tile_q);facts.padded_batch_size=std::uint32_t(plan.padded_batch_size);facts.split_kv=plan.split_kv;
  facts.bf16_partials=bf16_partials_active(s,facts.cta_tile_q,facts.split_kv);return result;
}
namespace {
template<unsigned D,bool Paged,class Params>cudaError_t launch(Params params,const fi::PrefillPlanInfo&plan,void*float_base,cudaStream_t stream){
  using Variant=fi::DefaultAttention<false,false,false,false>;
  auto*temporary=plan.split_kv?reinterpret_cast<typename Params::DTypeO*>(static_cast<char*>(float_base)+plan.v_offset):nullptr;
  auto*scores=plan.split_kv?reinterpret_cast<float*>(static_cast<char*>(float_base)+plan.s_offset):nullptr;
  switch(plan.cta_tile_q){
#define HYLE_BATCH_TILE(tile) case tile:if constexpr(Paged)return fi::BatchPrefillWithPagedKVCacheDispatched<tile,D,D,fi::PosEncodingMode::kNone,false,fi::MaskMode::kCausal,Variant>(params,temporary,scores,false,stream);else return fi::BatchPrefillWithRaggedKVCacheDispatched<tile,D,D,fi::PosEncodingMode::kNone,false,fi::MaskMode::kCausal,Variant>(params,temporary,scores,false,stream)
    HYLE_BATCH_TILE(16);HYLE_BATCH_TILE(64);HYLE_BATCH_TILE(128);
#undef HYLE_BATCH_TILE
  }
  throw std::runtime_error("FlashInfer batch CTA outside qualified dispatch");
}
template<class Params>void bind_plan(Params&p,const fi::PrefillPlanInfo&plan,void*integer){
  const auto ptr=[&](std::int64_t offset){return reinterpret_cast<std::int32_t*>(static_cast<char*>(integer)+offset);};
  p.request_indices=ptr(plan.request_indices_offset);p.qo_tile_indices=ptr(plan.qo_tile_indices_offset);p.kv_tile_indices=ptr(plan.kv_tile_indices_offset);
  p.o_indptr=ptr(plan.o_indptr_offset);p.kv_chunk_size_ptr=ptr(plan.kv_chunk_size_ptr_offset);p.padded_batch_size=std::uint32_t(plan.padded_batch_size);p.max_total_num_rows=std::uint32_t(plan.total_num_rows);
  if(plan.split_kv)p.merge_indptr=ptr(plan.merge_indptr_offset);
  if(plan.enable_cuda_graph){
    p.total_num_rows=reinterpret_cast<std::uint32_t*>(static_cast<char*>(integer)+plan.total_num_rows_offset);
    if(plan.split_kv)p.block_valid_mask=reinterpret_cast<bool*>(static_cast<char*>(integer)+plan.block_valid_mask_offset);
  }
}
template<class T,unsigned D>cudaError_t run_typed(const fi::PrefillPlanInfo&plan,BatchSpec s,const void*q,const void*k,const void*v,const void*metadata,void*o,void*f,void*i,cudaStream_t stream){
  const auto ptr=[&](std::size_t offset){return reinterpret_cast<std::int32_t*>(const_cast<char*>(static_cast<const char*>(metadata))+offset);};
  if(s.paged){fi::BatchPrefillPagedParams<T,T,T,std::int32_t>p;p.q=const_cast<T*>(static_cast<const T*>(q));p.o=static_cast<T*>(o);p.q_indptr=ptr(s.query_indptr_offset);
    p.paged_kv=fi::paged_kv_t<T,std::int32_t>(s.kv_heads,s.page_size,D,s.requests,fi::QKVLayout::kNHD,const_cast<T*>(static_cast<const T*>(k)),const_cast<T*>(static_cast<const T*>(v)),ptr(s.page_indices_offset),ptr(s.kv_indptr_offset),ptr(s.tail_offset));
    p.num_qo_heads=s.query_heads;p.group_size=fi::uint_fastdiv(s.query_heads/s.kv_heads);p.q_stride_n=s.query_heads*D;p.q_stride_h=D;p.window_left=-1;p.sm_scale=s.scale;bind_plan(p,plan,i);return launch<D,true>(p,plan,f,stream);
  }
  fi::BatchPrefillRaggedParams<T,T,T,std::int32_t>p;p.q=const_cast<T*>(static_cast<const T*>(q));p.k=const_cast<T*>(static_cast<const T*>(k));p.v=const_cast<T*>(static_cast<const T*>(v));p.o=static_cast<T*>(o);
  p.q_indptr=ptr(s.query_indptr_offset);p.kv_indptr=ptr(s.kv_indptr_offset);p.num_qo_heads=s.query_heads;p.num_kv_heads=s.kv_heads;p.group_size=fi::uint_fastdiv(s.query_heads/s.kv_heads);
  p.q_stride_n=s.query_heads*D;p.q_stride_h=D;p.k_stride_n=p.v_stride_n=s.kv_heads*D;p.k_stride_h=p.v_stride_h=D;p.window_left=-1;p.sm_scale=s.scale;bind_plan(p,plan,i);return launch<D,false>(p,plan,f,stream);
}
template<class T>cudaError_t dispatch(const fi::PrefillPlanInfo&plan,BatchSpec s,const void*q,const void*k,const void*v,const void*m,void*o,void*f,void*i,cudaStream_t stream){
  switch(s.qk_head_dim){case 64:return run_typed<T,64>(plan,s,q,k,v,m,o,f,i,stream);case 128:return run_typed<T,128>(plan,s,q,k,v,m,o,f,i,stream);case 256:return run_typed<T,256>(plan,s,q,k,v,m,o,f,i,stream);}
  throw std::runtime_error("FlashInfer batch head dimension outside admitted domain");
}
} // namespace
cudaError_t run_batch(const void*state,BatchSpec s,const void*q,const void*k,const void*v,const void*m,void*o,void*f,void*i,cudaStream_t stream){
  if(s.query_rows==0)return cudaSuccess;
  if(s.fp8 || !s.causal || s.window_left>=0 || s.qk_head_dim!=s.value_head_dim)
    return run_cache_batch(state,s,q,k,v,m,o,f,i,stream);
  const auto&plan=*static_cast<const fi::PrefillPlanInfo*>(state);
  return s.bfloat16?dispatch<__nv_bfloat16>(plan,s,q,k,v,m,o,f,i,stream):dispatch<half>(plan,s,q,k,v,m,o,f,i,stream);
}
} // namespace hyle::compute::flashinfer::native
