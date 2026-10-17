#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
namespace hyle::compute::flashinfer::native {
struct BatchSpec final {
  std::uint32_t query_heads=0,kv_heads=0,qk_head_dim=0,value_head_dim=0,page_size=0;
  std::uint32_t query_rows=0,requests=0;
  std::size_t query_indptr_offset=0,kv_indptr_offset=0,page_indices_offset=0,tail_offset=0;
  float scale=1.0F;
  float key_scale=1.0F,value_scale=1.0F;
  std::int32_t window_left=-1;
  bool paged=false,bfloat16=false,fp8=false,causal=true;
  bool allow_bf16_partials=false;
};
inline bool bf16_partial_domain(const BatchSpec &s) noexcept {
  return s.allow_bf16_partials && s.bfloat16 && s.fp8 && s.paged && s.page_size==16 &&
      s.causal && s.window_left==-1 && s.qk_head_dim==256 && s.value_head_dim==256;
}
inline bool bf16_partials_active(const BatchSpec &s,std::uint32_t tile,bool split) noexcept {
  return bf16_partial_domain(s) && tile==64 && split;
}
struct BatchFacts final {
  std::size_t float_bytes=0,int_bytes=0;
  std::uint32_t cta_tile_q=0,padded_batch_size=0,kv_chunk_rows=0;
  bool split_kv=false;
  bool bf16_partials=false;
};
std::size_t batch_state_bytes() noexcept;
void initialize_batch_state(void *) noexcept;
void destroy_batch_state(void *) noexcept;
// Only the native scheduler's H2D stage accepts GPU work. Query is host-only.
cudaError_t query_batch_workspace(BatchSpec,const std::int32_t *,const std::int32_t *,cudaStream_t,BatchFacts &);
cudaError_t plan_batch(void *,BatchSpec,const std::int32_t *,const std::int32_t *,void *,std::size_t,void *,void *,std::size_t,cudaStream_t,BatchFacts &);
cudaError_t run_batch(const void *,BatchSpec,const void *,const void *,const void *,const void *,void *,void *,void *,cudaStream_t);
cudaError_t run_cache_batch(const void *,BatchSpec,const void *,const void *,const void *,const void *,void *,void *,void *,cudaStream_t);
} // namespace hyle::compute::flashinfer::native
