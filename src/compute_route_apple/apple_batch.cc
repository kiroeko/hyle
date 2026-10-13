#include "hyle/compute/apple_batch.h"

#include <array>
#include <algorithm>
#include <new>
#include <optional>
#include <cmath>

namespace hyle::compute::apple {
Result<Program> prepare_ragged_batch(const Context&context,BatchLayoutVersion layout,
    RaggedAttention attention,const WorkloadContext&workload)noexcept{
  if(!context.valid()||!layout.valid()||layout.schema().capacity()!=attention.capacity||
     attention.capacity.storage!=BatchStorage::kRagged||!std::isfinite(attention.scale))
    return std::unexpected(Status::InvalidArgument("Apple ragged context/layout/domain differs"));
  const auto c=attention.capacity;
  if((c.data_type!=DataType::kFloat16&&c.data_type!=DataType::kBFloat16)||
     (c.head_dim!=64&&c.head_dim!=128&&c.head_dim!=256))
    return std::unexpected(Status::Unimplemented("Apple native ragged requires qualified flash dtype/depth"));
  try{
    HYLE_ASSIGN_OR_RETURN(auto q,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.head_dim}));
    HYLE_ASSIGN_OR_RETURN(auto k,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,c.head_dim}));
    std::array<InputSpec,3> inputs;HYLE_ASSIGN_OR_RETURN(inputs[0],input_spec(q));HYLE_ASSIGN_OR_RETURN(inputs[1],input_spec(k));inputs[2]=inputs[1];
    const auto host=layout.host();
    for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];
      if(nq&&(!nk||(attention.causal&&nk<nq)))return std::unexpected(Status::Unimplemented("Apple ragged nonempty Q needs sufficient KV"));}
    // The original MLX compiler retains the callable. Retain the immutable
    // owned version, never spans from the caller's metadata arrays.
    return context.compile([layout=std::move(layout),attention](const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=3)return std::unexpected(Status::InvalidArgument("Apple ragged needs Q/K/V"));
      const auto c=attention.capacity;const auto host=layout.host();std::optional<mlx::core::array> combined;
      const auto view=[&](const mlx::core::array &input,std::int32_t start,std::int32_t rows,std::uint32_t heads)->Result<mlx::core::array>{
        HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,start,rows));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(slice),std::span(&input,1),stream));
        HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,4>{1,rows,heads,c.head_dim}));
        HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));HYLE_ASSIGN_OR_RETURN(auto reshaped,lower(Operation(reshape),sliced,stream));return reshaped[0];
      };
      for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];if(!nq)continue;
        HYLE_ASSIGN_OR_RETURN(auto q,view(inputs[0],host.query_indptr[r],nq,c.query_heads));
        HYLE_ASSIGN_OR_RETURN(auto k,view(inputs[1],host.kv_indptr[r],nk,c.kv_heads));
        HYLE_ASSIGN_OR_RETURN(auto v,view(inputs[2],host.kv_indptr[r],nk,c.kv_heads));const std::array ports{q,k,v};
        HYLE_ASSIGN_OR_RETURN(auto output,lower(Operation(ScaledDotProductAttention{.layout=AttentionLayout::kBatchSequenceHeadDepth,.scale=attention.scale,.causal=attention.causal}),ports,stream));
        HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,3>{nq,c.query_heads,c.head_dim}));
        HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));HYLE_ASSIGN_OR_RETURN(auto packed,lower(Operation(reshape),output,stream));
        if(combined){const std::array parts{*combined,packed[0]};HYLE_ASSIGN_OR_RETURN(auto joined,lower(Operation(Concat{.axis=0}),parts,stream));combined=joined[0];}else combined=packed[0];
      }
      if(!combined){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));return lower(Operation(empty),std::span(&inputs[0],1),stream);}
      return std::vector<mlx::core::array>{*combined};
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
