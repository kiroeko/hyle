#include "native_page_reference.h"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <array>

namespace hyle::bench::resident {
namespace c=compute;namespace n=c::apple;namespace mx=mlx::core;
Result<n::Program> page_reference_program(const n::Context& context,
    const c::NativePagedBatchProgram& recipe,c::BatchLayoutVersion layout,
    const c::WorkloadContext& workload){
  if(!recipe.valid() || !recipe.schema().owns(layout))return std::unexpected(Status::InvalidArgument("page peer schema"));
  std::array<n::InputSpec,4> specs;
  for(unsigned i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(specs[i],n::input_spec(recipe.descriptions()[i]));}
  return context.compile([recipe,layout](const std::vector<mx::array>& inputs,mx::Stream stream)->Result<std::vector<mx::array>>{
    const auto cap=recipe.schema().capacity();const auto host=layout.host();std::vector<mx::array> outputs;
    const mx::Shape physical{int(cap.max_pages*16),int(cap.kv_heads),int(cap.head_dim)};
    const auto kroot=mx::reshape(inputs[1],physical,stream),vroot=mx::reshape(inputs[2],physical,stream);
    for(unsigned r=0;r<layout.facts().requests;++r){
      const int nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
      const int begin=host.kv_indptr[r],end=host.kv_indptr[r+1];
      std::vector<mx::array> keys,values;
      for(int ref=begin;ref<end;++ref){
        const int rows=ref+1==end?host.last_page_lengths[r]:16,first=host.page_indices[ref]*16;
        keys.push_back(mx::slice(kroot,{first,0,0},{first+rows,int(cap.kv_heads),int(cap.head_dim)},stream));
        values.push_back(mx::slice(vroot,{first,0,0},{first+rows,int(cap.kv_heads),int(cap.head_dim)},stream));
      }
      const int nk=(end-begin-1)*16+host.last_page_lengths[r];
      auto k=mx::transpose(mx::reshape(mx::concatenate(keys,0,stream),{1,nk,int(cap.kv_heads),int(cap.head_dim)},stream),{0,2,1,3},stream);
      auto v=mx::transpose(mx::reshape(mx::concatenate(values,0,stream),{1,nk,int(cap.kv_heads),int(cap.head_dim)},stream),{0,2,1,3},stream);
      auto q=mx::slice(inputs[0],{host.query_indptr[r],0,0},{host.query_indptr[r+1],int(cap.query_heads),int(cap.head_dim)},stream);
      q=mx::transpose(mx::reshape(q,{1,nq,int(cap.query_heads),int(cap.head_dim)},stream),{0,2,1,3},stream);
      // Force the original fused entry, matching candidate's original vector/
      // NAX arithmetic. The upstream default may choose unfused D256 prefill.
      auto result=mx::fast::scaled_dot_product_attention(q,k,v,recipe.scale(),
          recipe.causal()?"causal":"",std::nullopt,std::nullopt,true,stream);
      outputs.push_back(mx::reshape(mx::transpose(result,{0,2,1,3},stream),{nq,int(cap.query_heads),int(cap.head_dim)},stream));
    }
    if(outputs.size()==1)return outputs;
    return std::vector{mx::concatenate(outputs,0,stream)};
  },specs,workload);
}
} // namespace hyle::bench::resident
