#include "vulkan_encoded_page_reference.h"
#include "reference.h"
#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <array>
#include <memory>
#include <vector>

namespace hyle::bench::resident {
namespace c=compute;namespace n=c::vulkan;namespace d=n::detail;
Result<n::Program> encoded_page_reference_program(const n::Context& context,
    const c::NativePagedBatchProgram& recipe,c::BatchLayoutVersion layout,
    const c::WorkloadContext& workload){
  if(!recipe.valid() || !recipe.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("encoded page peer schema"));
  const auto cap=recipe.schema().capacity();const auto host=layout.host();
  if(cap.key_storage!=c::CacheStorage::q8_0().value() ||
     cap.value_storage!=c::CacheStorage::q8_0().value() || cap.page_size!=16 ||
     cap.query_type!=c::DataType::kFloat16 || cap.output_type!=c::DataType::kFloat16)
    return std::unexpected(Status::InvalidArgument("encoded page peer finite domain"));
  auto state=std::make_shared<d::ProgramState>();
  state->runtime=d::Access::runtime(context);state->workload=workload;
  for(const auto& description:recipe.descriptions())state->values.push_back({description,{}});
  state->inputs={0,1,2,3};
  state->input_descriptions.assign(recipe.descriptions().begin(),recipe.descriptions().end());
  for(unsigned i=0;i<4;++i)state->aliases.push_back({i,0});
  const auto add=[&](const c::TensorDesc& description){
    const auto id=std::uint32_t(state->values.size());state->values.push_back({description,{}});
    state->aliases.push_back({id,0});return id;
  };
  std::vector<std::uint32_t> parts;
  for(unsigned r=0;r<layout.facts().requests;++r){
    const int nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
    const int begin=host.kv_indptr[r],end=host.kv_indptr[r+1];
    const int nk=(end-begin-1)*16+host.last_page_lengths[r];
    std::vector<std::int32_t> rows(std::size_t(nk),0);
    for(int row=0;row<nk;++row)rows[std::size_t(row)]=host.page_indices[begin+row/16]*16+row%16;
    HYLE_ASSIGN_OR_RETURN(auto index_desc,c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t,1>{nk}));
    const auto indices=add(index_desc);
    HYLE_ASSIGN_OR_RETURN(state->values[indices].constant,
        context.upload(index_desc,std::as_bytes(std::span(rows))));
    std::array<std::uint32_t,2> gathered;
    for(unsigned port=1;port<3;++port){
      const auto depth=port==1?cap.qk_head_dim:cap.value_head_dim;
      const auto encoded_depth=depth/32*34;
      HYLE_ASSIGN_OR_RETURN(auto root_desc,c::TensorDesc::dense(c::DataType::kUInt8,
          std::array<std::int64_t,3>{std::int64_t(cap.max_pages)*16,cap.kv_heads,encoded_depth}));
      const auto root=add(root_desc);state->aliases[root]={port,0};
      HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kUInt8,
          std::array<std::int64_t,3>{nk,cap.kv_heads,encoded_depth}));
      gathered[port-1]=add(desc);
      // The original GetRows I32 route copies opaque storage bits. Reuse that
      // bounded row adapter instead of O(page-count) three-input Concats.
      // Its index normalization and output copies remain charged as generated.
      HYLE_RETURN_IF_ERROR(d::lower_gather(*state,
          {c::Operation(c::Gather{0,c::IndexBoundsPolicy::kUnchecked}),
           {root,indices},{gathered[port-1]},{},{1,1,1},{}}));
    }
    HYLE_ASSIGN_OR_RETURN(auto qd,c::TensorDesc::dense(c::DataType::kFloat16,
        std::array<std::int64_t,4>{1,nq,cap.query_heads,cap.qk_head_dim}));
    HYLE_ASSIGN_OR_RETURN(auto kd,c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t,4>{1,nk,cap.kv_heads,cap.qk_head_dim/32*34}));
    HYLE_ASSIGN_OR_RETURN(auto vd,c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t,4>{1,nk,cap.kv_heads,cap.value_head_dim/32*34}));
    HYLE_ASSIGN_OR_RETURN(auto od,c::TensorDesc::dense(c::DataType::kFloat16,
        std::array<std::int64_t,4>{1,nq,cap.query_heads,cap.value_head_dim}));
    auto original=encoded_reference_program(context,qd,kd,vd,od,
        recipe.causal(),recipe.window_left(),recipe.scale());
    const auto& child=*d::Access::program(original);
    const auto base=std::uint32_t(state->values.size());
    state->values.insert(state->values.end(),child.values.begin(),child.values.end());
    for(const auto alias:child.aliases)state->aliases.push_back({base+alias.root,alias.byte_offset});
    state->aliases[base]={0,std::size_t(host.query_indptr[r])*cap.query_heads*cap.qk_head_dim*2};
    state->aliases[base+1]={gathered[0],0};state->aliases[base+2]={gathered[1],0};
    for(auto dispatch:child.dispatches){
      for(auto& id:dispatch.values)id+=base;
      state->dispatches.push_back(std::move(dispatch));
    }
    HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat16,
        std::array<std::int64_t,3>{nq,cap.query_heads,cap.value_head_dim}));
    const auto packed=add(desc);state->aliases[packed]={base+child.outputs[0],0};parts.push_back(packed);
  }
  HYLE_ASSIGN_OR_RETURN(auto description,c::TensorDesc::dense(c::DataType::kFloat16,
      std::array<std::int64_t,3>{layout.facts().query_rows,cap.query_heads,cap.value_head_dim}));
  const auto output=add(description);state->outputs={output};state->output_descriptions={description};
  HYLE_RETURN_IF_ERROR(d::lower_tensor(*state,{c::Operation(c::Concat{.axis=0}),
      std::move(parts),{output},{},{1,1,1},{}}));
  HYLE_RETURN_IF_ERROR(d::plan_storage(*state));return d::Access::program(std::move(state));
}
}
