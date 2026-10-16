#include "hyle/compute/apple_batch.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

#include <array>
#include <algorithm>
#include <new>
#include <optional>
#include <cmath>

namespace hyle::compute::apple {
namespace {
bool cache_domain(const BatchCapacity& c) noexcept {
  const bool plain=c.key_storage==CacheStorage::plain(c.query_type).value_or(CacheStorage{}) &&
      c.value_storage==c.key_storage;
  const bool q8=c.key_storage==CacheStorage::q8_0().value() && c.value_storage==c.key_storage;
  return (plain || q8) && c.output_type==c.query_type &&
      (c.query_type==DataType::kFloat16 || c.query_type==DataType::kBFloat16) &&
      ((c.qk_head_dim==c.value_head_dim &&
        (c.qk_head_dim==64 || c.qk_head_dim==128 || c.qk_head_dim==256)) ||
       (c.qk_head_dim==64 && c.value_head_dim==128));
}
int cache_width(CacheStorage storage,std::uint32_t depth) noexcept {
  return int(storage==CacheStorage::q8_0().value()?depth/32*34:depth);
}
}
Result<Program> detail::PagedBatchAccess::prepare_program(const Context& context,
    const NativePagedBatchProgram& recipe, BatchLayoutVersion layout,
    const WorkloadContext& workload) noexcept {
  if(!context.valid() || !recipe.valid() || !recipe.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("Apple paged context/layout/schema differs"));
  const auto c=recipe.schema().capacity();
  if(!cache_domain(c))
    return std::unexpected(Status::Unimplemented("Apple paged cache dtype/encoding/depth is outside the finite domain"));
  if((c.query_type!=DataType::kFloat16 && c.query_type!=DataType::kBFloat16) ||
      c.page_size!=16 || c.max_pages>std::uint32_t(INT32_MAX/16) ||
      (c.qk_head_dim!=64 && c.qk_head_dim!=128 && c.qk_head_dim!=256))
    return std::unexpected(Status::Unimplemented("Apple paged qualified page16 flash dtype/depth"));
  const auto host=layout.host();
  for(std::size_t r=0;r<layout.facts().requests;++r){
    const auto nq=host.query_indptr[r+1]-host.query_indptr[r];
    const auto refs=host.kv_indptr[r+1]-host.kv_indptr[r];
    const auto nk=refs?std::uint64_t(refs-1)*16+std::uint32_t(host.last_page_lengths[r]):0;
    if(nq && (!nk || nk>INT32_MAX || (recipe.causal() && nk<std::uint64_t(nq))))
      return std::unexpected(Status::Unimplemented("Apple paged nonempty Q needs sufficient bounded logical KV"));
  }
  try {
    std::array<InputSpec,4> specs;
    for(std::size_t i=0;i<4;++i){HYLE_ASSIGN_OR_RETURN(specs[i],input_spec(recipe.descriptions()[i]));}
    if(layout.facts().query_rows==0 && layout.facts().page_refs==0){
      return prepare_runtime_program(context,layout,std::span(specs).first(1),
          [heads=int(c.query_heads),depth=int(c.value_head_dim)](const BatchLayoutVersion&,
          const std::vector<mlx::core::array>& inputs,
          mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
        if(inputs.size()!=1)return std::unexpected(Status::InvalidArgument("empty Apple page episode needs only Q"));
        HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));
        HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));
        return std::vector{mlx::core::reshape(sliced[0],{0,heads,depth},stream)};
      },workload);
    }
    return prepare_runtime_program(context,layout,specs,[recipe](const BatchLayoutVersion& layout,
        const std::vector<mlx::core::array>& inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=4)return std::unexpected(Status::InvalidArgument("Apple paged needs four native ports"));
      const auto c=recipe.schema().capacity();const auto host=layout.host();
      // Reshape and transpose are aliases of the truthful physical arena;
      // the private primitive owns the page table and logical row count.
      const mlx::core::Shape key_root{1,int(std::int64_t(c.max_pages)*16),int(c.kv_heads),cache_width(c.key_storage,c.qk_head_dim)};
      const mlx::core::Shape value_root{1,int(std::int64_t(c.max_pages)*16),int(c.kv_heads),cache_width(c.value_storage,c.value_head_dim)};
      const auto k=mlx::core::transpose(mlx::core::reshape(inputs[1],key_root,stream),{0,2,1,3},stream);
      const auto v=mlx::core::transpose(mlx::core::reshape(inputs[2],value_root,stream),{0,2,1,3},stream);
      std::vector<mlx::core::array> parts;
      for(std::size_t r=0;r<layout.facts().requests;++r){
        const auto nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
        const auto refs=host.kv_indptr[r+1]-host.kv_indptr[r];
        const auto nk=std::uint64_t(refs-1)*16+std::uint32_t(host.last_page_lengths[r]);
        HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,host.query_indptr[r],nq));
        HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(slice),std::span(&inputs[0],1),stream));
        auto q=mlx::core::transpose(mlx::core::reshape(sliced[0],{1,nq,int(c.query_heads),int(c.qk_head_dim)},stream),{0,2,1,3},stream);
        auto result=mlx::core::fast::hyle_paged_attention(q,k,v,inputs[3],int(nk),host.kv_indptr[r],recipe.scale(),recipe.causal(),stream,c.key_storage==CacheStorage::q8_0().value(),recipe.window_left());
        parts.push_back(mlx::core::reshape(mlx::core::transpose(result,{0,2,1,3},stream),{nq,int(c.query_heads),int(c.value_head_dim)},stream));
      }
      if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));return std::vector{mlx::core::reshape(sliced[0],{0,int(c.query_heads),int(c.value_head_dim)},stream)};}
      if(parts.size()==1)return parts;
      return lower(Operation(Concat{.axis=0}),parts,stream);
    },workload);
  } catch(const std::bad_alloc&) {
    return std::unexpected(Status::ResourceExhausted("Apple paged cold tracing metadata"));
  }
}
Result<Program> prepare_ragged_batch(const Context&context,BatchLayoutVersion layout,
    RaggedAttention attention,const WorkloadContext&workload)noexcept{
  if(!context.valid()||!layout.valid()||layout.schema().capacity()!=attention.capacity||
     attention.capacity.storage!=BatchStorage::kRagged||!std::isfinite(attention.scale))
    return std::unexpected(Status::InvalidArgument("Apple ragged context/layout/domain differs"));
  const auto c=attention.capacity;
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(attention.causal,attention.window_left));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,attention.causal,attention.window_left));
  if(!cache_domain(c))
    return std::unexpected(Status::Unimplemented("Apple ragged cache dtype/encoding/depth is outside the finite domain"));
  if((c.query_type!=DataType::kFloat16&&c.query_type!=DataType::kBFloat16)||
     (c.qk_head_dim!=64&&c.qk_head_dim!=128&&c.qk_head_dim!=256))
    return std::unexpected(Status::Unimplemented("Apple native ragged requires qualified flash dtype/depth"));
  try{
    HYLE_ASSIGN_OR_RETURN(auto q,TensorDesc::dense(c.query_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.qk_head_dim}));
    HYLE_ASSIGN_OR_RETURN(auto k,batch_cache_description(c,BatchCacheRole::kKey));
    HYLE_ASSIGN_OR_RETURN(auto v,batch_cache_description(c,BatchCacheRole::kValue));
    std::array<InputSpec,3> inputs;HYLE_ASSIGN_OR_RETURN(inputs[0],input_spec(q));HYLE_ASSIGN_OR_RETURN(inputs[1],input_spec(k));HYLE_ASSIGN_OR_RETURN(inputs[2],input_spec(v));
    const auto host=layout.host();
    for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];
      if(nq&&(!nk||(attention.causal&&nk<nq)))return std::unexpected(Status::Unimplemented("Apple ragged nonempty Q needs sufficient KV"));}
    // The original MLX compiler retains the callable. Retain the immutable
    // owned version, never spans from the caller's metadata arrays.
    return context.compile([layout=std::move(layout),attention](const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=3)return std::unexpected(Status::InvalidArgument("Apple ragged needs Q/K/V"));
      const auto c=attention.capacity;const auto host=layout.host();std::vector<mlx::core::array> parts;
      const auto view=[&](const mlx::core::array &input,std::int32_t start,std::int32_t rows,std::uint32_t heads,int width)->Result<mlx::core::array>{
        HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,start,rows));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(slice),std::span(&input,1),stream));
        HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,4>{1,rows,heads,width}));
        HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));HYLE_ASSIGN_OR_RETURN(auto reshaped,lower(Operation(reshape),sliced,stream));return reshaped[0];
      };
      for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];if(!nq)continue;
        HYLE_ASSIGN_OR_RETURN(auto q,view(inputs[0],host.query_indptr[r],nq,c.query_heads,int(c.qk_head_dim)));
        const bool q8=c.key_storage==CacheStorage::q8_0().value();
        // U8 carriers are flattened by input_spec; first restore exact rows
        // and bytes, then slice only this request without decoding history.
        auto key_rows=mlx::core::reshape(inputs[1],{int(c.max_kv_rows),int(c.kv_heads),cache_width(c.key_storage,c.qk_head_dim)},stream);
        auto value_rows=mlx::core::reshape(inputs[2],{int(c.max_kv_rows),int(c.kv_heads),cache_width(c.value_storage,c.value_head_dim)},stream);
        HYLE_ASSIGN_OR_RETURN(auto k,view(key_rows,host.kv_indptr[r],nk,c.kv_heads,cache_width(c.key_storage,c.qk_head_dim)));
        HYLE_ASSIGN_OR_RETURN(auto v,view(value_rows,host.kv_indptr[r],nk,c.kv_heads,cache_width(c.value_storage,c.value_head_dim)));
        auto output=mlx::core::fast::hyle_ragged_attention(
            mlx::core::transpose(q,{0,2,1,3},stream),mlx::core::transpose(k,{0,2,1,3},stream),
            mlx::core::transpose(v,{0,2,1,3},stream),attention.scale,attention.causal,stream,q8,attention.window_left);
        parts.push_back(mlx::core::reshape(mlx::core::transpose(output,{0,2,1,3},stream),{nq,int(c.query_heads),int(c.value_head_dim)},stream));
      }
      if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));return std::vector{mlx::core::reshape(sliced[0],{0,int(c.query_heads),int(c.value_head_dim)},stream)};}
      if(parts.size()==1)return parts;
      return lower(Operation(Concat{.axis=0}),parts,stream);
    },inputs,workload);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Apple ragged cold tracing metadata"));}
}
Result<Program> prepare_materialized_batch(const Context&context,
    const MaterializedBatchProgram&program,const WorkloadContext&workload)noexcept{
  if(!context.valid() || !program.valid())return std::unexpected(Status::InvalidArgument("Apple materialized batch context/program is empty"));
  try {
    std::array<InputSpec,5> inputs;const auto descriptions=program.descriptions();
    for(std::size_t i=0;i<inputs.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],input_spec(descriptions[i]));}
    // MLX retains the tracing callable. Copy its finite value records so it
    // does not borrow the common program's allocator/schema after prepare.
    // The retained native closure is separate, opaque native metadata.
    std::array<MaterializedBatchStep,19> recipe;const auto count=program.steps().size();
    std::copy(program.steps().begin(),program.steps().end(),recipe.begin());const auto output_value=program.output_value();
    return context.compile([recipe=std::move(recipe),count,output_value](const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=5)return std::unexpected(Status::InvalidArgument("Apple materialized batch needs five inputs"));
      std::array<std::optional<mlx::core::array>,24> values;
      for(std::size_t i=0;i<inputs.size();++i)values[i]=inputs[i];
      for(const auto&step:std::span(recipe).first(count)){
        std::vector<mlx::core::array> arguments;arguments.reserve(step.input_count);
        for(std::size_t i=0;i<step.input_count;++i)arguments.push_back(*values[step.inputs[i]]);
        HYLE_ASSIGN_OR_RETURN(auto result,lower(step.operation,arguments,stream));
        if(result.size()!=1)return std::unexpected(Status::Internal("Apple materialized recipe output arity"));
        values[step.output]=std::move(result[0]);
      }
      return std::vector<mlx::core::array>{*values[output_value]};
    },inputs,workload);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Apple materialized batch cold recipe metadata"));}
}
} // namespace hyle::compute::apple
