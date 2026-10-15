#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::bench::resident::cuda_original {
// Benchmark-only bridge. No candidate FlashInfer header or patched arithmetic.
struct Spec {
  unsigned query_heads=8, kv_heads=2, key_depth=0, value_depth=0;
  unsigned page_size=16, requests=0, query_rows=0;
  std::size_t query_offset=0, kv_offset=0, indices_offset=0, tails_offset=0;
  float scale=1;
  int window=-1;
  bool causal=true;
};
struct Facts {
  std::size_t floating=0, integer=0;
  unsigned tile=0, padded=0;
  bool split=false;
};
void *create_plan();
void destroy_plan(void *) noexcept;
cudaError_t workspace(Spec,const std::int32_t *,const std::int32_t *,cudaStream_t,Facts &);
cudaError_t plan(void *,Spec,const std::int32_t *,const std::int32_t *,
                 void *,std::size_t,void *,void *,std::size_t,cudaStream_t,Facts &);
cudaError_t run(const void *,Spec,const void *,const void *,const void *,
                const void *,void *,void *,void *,cudaStream_t);
// Decode only authorized physical rows into an additional readonly BF16 cache.
cudaError_t decode(const void *,void *,const std::uint32_t *,unsigned,
                   unsigned,unsigned,unsigned,float,cudaStream_t);
}
