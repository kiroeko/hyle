#include "hyle/compute/vulkan_batch.h"
#include "hyle/compute/detail/vulkan_plan.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "../compute_provider_ggml_vulkan/region.h"

#include <new>
#include <cmath>
#include <limits>
#include <optional>

namespace hyle::compute::vulkan {
Result<Program> detail::PagedBatchAccess::prepare_program(const Context& context,
    const NativePagedBatchProgram& recipe, BatchLayoutVersion layout,
    const WorkloadContext& workload) noexcept {
  if(!context.valid() || !recipe.valid() || !recipe.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("Vulkan paged context/layout/schema differs"));
  const auto c=recipe.schema().capacity();
  if(c.data_type!=DataType::kFloat16 || c.page_size!=16 ||
      (c.head_dim!=64 && c.head_dim!=128 && c.head_dim!=256))
    return std::unexpected(Status::Unimplemented("Vulkan paged qualified page16/F16 flash depth"));
  try {
    detail::Plan plan;plan.workload=workload;plan.inputs={0,1,2,3};
    for(const auto& desc:recipe.descriptions())plan.values.push_back({desc,{}});
    const auto add=[&](Operation op,std::vector<std::uint32_t> inputs)->Result<std::uint32_t>{
      if(plan.values.size()>=UINT32_MAX)return std::unexpected(Status::ResourceExhausted("Vulkan paged recipe value count"));
      std::vector<TensorDesc> descriptions;
      for(auto id:inputs)descriptions.push_back(plan.values[id].description);
      TensorDesc output;HYLE_RETURN_IF_ERROR(infer_output_descriptions(op,descriptions,std::span(&output,1)));
      const auto id=static_cast<std::uint32_t>(plan.values.size());plan.values.push_back({output,{}});
      plan.steps.push_back({std::move(op),std::move(inputs),{id},{},{1,1,1},{}});return id;
    };
    HYLE_ASSIGN_OR_RETURN(auto root_shape,TensorShape::create(std::array<std::int64_t,4>{
        1,std::int64_t(c.max_pages)*16,c.kv_heads,c.head_dim}));
    HYLE_ASSIGN_OR_RETURN(auto root_view,View::reshape(root_shape));
    HYLE_ASSIGN_OR_RETURN(auto k,add(Operation(root_view),{1}));
    HYLE_ASSIGN_OR_RETURN(auto v,add(Operation(root_view),{2}));
    const auto host=layout.host();
    struct Request{std::uint32_t q,rows,kv,offset;};std::vector<Request> requests;
    for(std::size_t r=0;r<layout.facts().requests;++r){
      const auto nq=host.query_indptr[r+1]-host.query_indptr[r];
      const auto refs=host.kv_indptr[r+1]-host.kv_indptr[r];
      if(!nq)continue;
      const auto nk=refs?std::uint64_t(refs-1)*16+std::uint32_t(host.last_page_lengths[r]):0;
      if(!nk || nk>INT32_MAX || (recipe.causal() && nk<std::uint64_t(nq)))
        return std::unexpected(Status::Unimplemented("Vulkan paged nonempty Q needs sufficient bounded logical KV"));
      HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,host.query_indptr[r],nq));
      HYLE_ASSIGN_OR_RETURN(auto sliced,add(Operation(slice),{0}));
      HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,4>{1,nq,c.query_heads,c.head_dim}));
      HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));
      HYLE_ASSIGN_OR_RETURN(auto q,add(Operation(reshape),{sliced}));
      requests.push_back({q,std::uint32_t(nq),std::uint32_t(nk),std::uint32_t(host.kv_indptr[r])});
    }
    std::vector<std::uint32_t> parts;
    for(const auto& request:requests){
      const auto op=Operation::custom(ggml_vulkan::detail::PagedAttentionRead{
          .logical_rows=request.kv,.table_offset=request.offset,
          .scale=recipe.scale(),.causal=recipe.causal()});
      HYLE_ASSIGN_OR_RETURN(auto output,add(op,{request.q,k,v,3}));
      HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,3>{request.rows,c.query_heads,c.head_dim}));
      HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));
      HYLE_ASSIGN_OR_RETURN(auto packed,add(Operation(reshape),{output}));parts.push_back(packed);
    }
    std::uint32_t output;
    if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(output,add(Operation(empty),{0}));}
    else if(parts.size()==1)output=parts[0];
    else{HYLE_ASSIGN_OR_RETURN(output,add(Operation(Concat{.axis=0}),std::move(parts)));}
    plan.outputs={output};return detail::prepare_plan(context,plan);
  } catch(const std::bad_alloc&) {
    return std::unexpected(Status::ResourceExhausted("Vulkan paged cold recipe metadata"));
  }
}
Result<Program> prepare_ragged_batch(const Context&context,BatchLayoutVersion layout,
    RaggedAttention attention,const WorkloadContext&workload)noexcept{
  if(!context.valid()||!layout.valid()||layout.schema().capacity()!=attention.capacity||
     attention.capacity.storage!=BatchStorage::kRagged||!std::isfinite(attention.scale))
    return std::unexpected(Status::InvalidArgument("Vulkan ragged context/layout/domain differs"));
  const auto c=attention.capacity;
  if(c.data_type!=DataType::kFloat16||(c.head_dim!=64&&c.head_dim!=128&&c.head_dim!=256))
    return std::unexpected(Status::Unimplemented("Vulkan native ragged requires qualified F16 flash depth"));
  try{
    detail::Plan plan;plan.workload=workload;plan.inputs={0,1,2};
    HYLE_ASSIGN_OR_RETURN(auto q,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.head_dim}));
    HYLE_ASSIGN_OR_RETURN(auto k,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,c.head_dim}));
    plan.values={{q,{}},{k,{}},{k,{}}};
    const auto add=[&](Operation op,std::vector<std::uint32_t> inputs)->Result<std::uint32_t>{
      if(plan.values.size()>=UINT32_MAX)return std::unexpected(Status::ResourceExhausted("Vulkan ragged recipe value count"));
      std::vector<TensorDesc> descriptions;for(auto id:inputs)descriptions.push_back(plan.values[id].description);
      TensorDesc output;HYLE_RETURN_IF_ERROR(infer_output_descriptions(op,descriptions,std::span(&output,1)));
      const auto id=static_cast<std::uint32_t>(plan.values.size());plan.values.push_back({output,{}});
      plan.steps.push_back({std::move(op),std::move(inputs),{id},{},{1,1,1},{}});return id;
    };
    const auto view=[&](std::uint32_t input,std::int32_t start,std::int32_t rows,std::uint32_t heads)->Result<std::uint32_t>{
      HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,start,rows));HYLE_ASSIGN_OR_RETURN(auto sliced,add(Operation(slice),{input}));
      HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,4>{1,rows,heads,c.head_dim}));
      HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));return add(Operation(reshape),{sliced});
    };
    const auto host=layout.host();std::vector<std::uint32_t> parts;
    struct Request { std::uint32_t q,k,v;std::int32_t rows; };
    std::vector<Request> requests;
    for(std::size_t request=0;request<layout.facts().requests;++request){
      const auto nq=host.query_indptr[request+1]-host.query_indptr[request];
      const auto nk=host.kv_indptr[request+1]-host.kv_indptr[request];if(!nq)continue;
      if(!nk||(attention.causal&&nk<nq))return std::unexpected(Status::Unimplemented("Vulkan ragged nonempty Q needs sufficient KV"));
      HYLE_ASSIGN_OR_RETURN(auto qv,view(0,host.query_indptr[request],nq,c.query_heads));
      HYLE_ASSIGN_OR_RETURN(auto kv,view(1,host.kv_indptr[request],nk,c.kv_heads));
      HYLE_ASSIGN_OR_RETURN(auto vv,view(2,host.kv_indptr[request],nk,c.kv_heads));
      requests.push_back({qv,kv,vv,nq});
    }
    // External aliases precede the arithmetic region. Zero-cost output
    // reshapes can then remain inside one original managed submission.
    for(const auto &request:requests){
      HYLE_ASSIGN_OR_RETURN(auto output,add(Operation(ScaledDotProductAttention{.layout=AttentionLayout::kBatchSequenceHeadDepth,.scale=attention.scale,.causal=attention.causal}),{request.q,request.k,request.v}));
      HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::array<std::int64_t,3>{request.rows,c.query_heads,c.head_dim}));
      HYLE_ASSIGN_OR_RETURN(auto reshape,View::reshape(shape));HYLE_ASSIGN_OR_RETURN(auto packed,add(Operation(reshape),{output}));
      parts.push_back(packed);
    }
    std::uint32_t output;
    if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(output,add(Operation(empty),{0}));}
    else if(parts.size()==1)output=parts[0];
    else{HYLE_ASSIGN_OR_RETURN(output,add(Operation(Concat{.axis=0}),std::move(parts)));}
    plan.outputs={output};return detail::prepare_plan(context,plan);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan ragged cold recipe metadata"));}
}
Result<Program> prepare_materialized_batch(const Context&context,
    const MaterializedBatchProgram&program,const WorkloadContext&workload)noexcept{
  if(!context.valid() || !program.valid())return std::unexpected(Status::InvalidArgument("Vulkan materialized batch context/program is empty"));
  try {
    detail::Plan plan;plan.workload=workload;plan.inputs={0,1,2,3,4};plan.outputs={program.output_value()};
    plan.values.reserve(program.descriptions().size());plan.steps.reserve(program.steps().size());
    for(const auto&description:program.descriptions())plan.values.push_back({description,{}});
    for(const auto&step:program.steps()){
      detail::PlanStep native;native.operation=step.operation;native.outputs={step.output};
      native.inputs.assign(step.inputs.begin(),step.inputs.begin()+step.input_count);
      plan.steps.push_back(std::move(native));
    }
    return detail::prepare_plan(context,plan);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan materialized batch cold recipe metadata"));}
}
} // namespace hyle::compute::vulkan
