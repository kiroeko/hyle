#include "apple_ragged_reference.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <array>
#include <cstdio>
#include <cstdlib>

namespace hyle::bench::resident {
namespace c=compute;namespace a=c::apple;namespace mx=mlx::core;
a::Program ragged_reference_program(const a::Context&context,
    const c::BatchLayoutVersion&layout,const c::RaggedAttention&attention){
  const auto cap=layout.schema().capacity();const std::array inputs{
    a::InputSpec{{int(cap.max_query_rows),int(cap.query_heads),int(cap.head_dim)},mx::float16},
    a::InputSpec{{int(cap.max_kv_rows),int(cap.kv_heads),int(cap.head_dim)},mx::float16},
    a::InputSpec{{int(cap.max_kv_rows),int(cap.kv_heads),int(cap.head_dim)},mx::float16}};
  auto result=context.compile([layout,attention](const std::vector<mx::array>&inputs,mx::Stream stream)->Result<std::vector<mx::array>>{
    const auto cap=attention.capacity;const auto h=layout.host();std::vector<mx::array> outputs;
    for(unsigned r=0;r<layout.facts().requests;++r){const int nq=h.query_indptr[r+1]-h.query_indptr[r],nk=h.kv_indptr[r+1]-h.kv_indptr[r];if(!nq)continue;
      const auto view=[&](unsigned port,int start,int rows,int heads){auto slice=mx::slice(inputs[port],{start,0,0},{start+rows,heads,int(cap.head_dim)},stream);return mx::transpose(mx::reshape(slice,{1,rows,heads,int(cap.head_dim)},stream),{0,2,1,3},stream);};
      auto q=view(0,h.query_indptr[r],nq,int(cap.query_heads)),k=view(1,h.kv_indptr[r],nk,int(cap.kv_heads)),v=view(2,h.kv_indptr[r],nk,int(cap.kv_heads));
      // The original MLX fast implementation directly, never candidate lower.
      auto output=mx::fast::scaled_dot_product_attention(q,k,v,attention.scale,attention.causal?"causal":"",std::nullopt,std::nullopt,false,stream);
      outputs.push_back(mx::reshape(mx::transpose(output,{0,2,1,3},stream),{nq,int(cap.query_heads),int(cap.head_dim)},stream));
    }
    if(outputs.size()==1)return outputs;
    return std::vector{mx::concatenate(outputs,0,stream)};
  },inputs);
  if(!result){std::fprintf(stderr,"original Apple ragged: %.*s\n",int(result.error().message().size()),result.error().message().data());std::exit(2);}return std::move(*result);
}
}
