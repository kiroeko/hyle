#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <limits>

#include "hyle/compute/model_ops.h"
#include "vulkan_shaders.h"
#include "../compute_batch/row_cache_map.h"

namespace hyle::compute::vulkan::detail {
Status lower_row_cache(ProgramState &program, const PlanStep &step,
                       RowCacheAccessFacts *access) noexcept {
  if (access) *access = {};
  const auto first_dispatch = program.dispatches.size();
  const auto finish = [&](RowCacheAccessFacts fact) {
    fact.dispatch_begin = first_dispatch;
    fact.dispatch_end = program.dispatches.size();
    if (access) *access = fact;
    return Status::Ok();
  };
  try {
    if(const auto map=step.operation.custom_as<compute::detail::RowCachePaddedMap>()){
      if(step.inputs.size()!=3 || step.outputs.size()!=2)return Status::InvalidArgument("Vulkan padded page map ports differ");
      const std::array inputs{program.values[step.inputs[0]].description,program.values[step.inputs[1]].description,program.values[step.inputs[2]].description};
      std::array<TensorDesc,2> inferred;HYLE_RETURN_IF_ERROR(map->infer_output_descriptions(inputs,inferred));
      for(std::size_t i=0;i<2;++i)if(inferred[i]!=program.values[step.outputs[i]].description)return Status::InvalidArgument("Vulkan padded map output differs");
      const auto &m=map->mapping;
      const RowCacheAccessFacts fact{.kind=RowCacheAccessKind::kPaddedMap,
        .query_rows=std::uint32_t(inputs[0].dimensions()[0]),
        .max_requests=m.max_requests(),.max_refs=m.max_refs(),.page_size=m.page_size};
      HYLE_ASSIGN_OR_RETURN(auto count,inferred[0].element_count());if(!count)return finish(fact);const auto&limits=program.runtime->device->properties.limits;
      if(limits.maxComputeWorkGroupInvocations<128 || limits.maxComputeWorkGroupSize[0]<128)return Status::Unimplemented("Vulkan padded map workgroup limit");
      for(const auto&desc:std::array{inputs[0],inputs[1],inputs[2],inferred[0],inferred[1]}){HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());
        if(bytes>limits.maxStorageBufferRange || bytes>UINT32_MAX-3u)return Status::Unimplemented("Vulkan padded map byte address limit");}
      Dispatch dispatch;dispatch.values={step.inputs[0],step.inputs[1],step.inputs[2],step.outputs[0],step.outputs[1]};
      const std::array<SpecializationConstant,11> constants{{{0,std::uint32_t(inputs[0].dimensions()[0])},{1,std::uint32_t(count)},{2,map->score_columns},
        {3,std::uint32_t(m.query_offset/4)},{4,std::uint32_t(m.kv_offset/4)},{5,std::uint32_t(m.page_offset/4)},{6,std::uint32_t(m.tail_offset/4)},
        {7,m.page_size},{8,m.max_pages},{9,m.max_requests()},{10,m.max_refs()}}};
      HYLE_ASSIGN_OR_RETURN(dispatch.pipeline,create_pipeline(program.runtime->device,{shaders::row_cache_padded_map(),5,20,constants}));
      const auto groups=(count+127)/128,x=std::min(groups,std::uint64_t(limits.maxComputeWorkGroupCount[0]));
      if(!x || (groups+x-1)/x>limits.maxComputeWorkGroupCount[1])return Status::Unimplemented("Vulkan padded map grid");
      dispatch.groups={std::uint32_t(x),std::uint32_t((groups+x-1)/x),1};dispatch.tensor_byte_origins=true;dispatch.push.resize(20);
      program.dispatches.push_back(std::move(dispatch));return finish(fact);
    }
    if(const auto map=step.operation.custom_as<compute::detail::RowCacheMap>()){
      if(step.inputs.size()!=2 || step.outputs.size()!=1)return Status::InvalidArgument("Vulkan row page map ports differ");
      const std::array inputs{program.values[step.inputs[0]].description,program.values[step.inputs[1]].description};
      const auto output=program.values[step.outputs[0]].description;std::array<TensorDesc,1> inferred;
      HYLE_RETURN_IF_ERROR(map->infer_output_descriptions(inputs,inferred));if(output!=inferred[0])return Status::InvalidArgument("Vulkan row page map output differs");
      const RowCacheAccessFacts fact{.kind=RowCacheAccessKind::kMap,
        .query_rows=std::uint32_t(inputs[0].dimensions()[0]),
        .max_requests=map->max_requests(),.max_refs=map->max_refs(),.page_size=map->page_size};
      HYLE_ASSIGN_OR_RETURN(auto count,output.element_count());if(!count)return finish(fact);
      const auto&limits=program.runtime->device->properties.limits;
      if(limits.maxComputeWorkGroupInvocations<128 || limits.maxComputeWorkGroupSize[0]<128 || limits.maxComputeSharedMemorySize<2080)
        return Status::Unimplemented("Vulkan row page map workgroup limits");
      for(const auto&desc:std::array{inputs[0],inputs[1],output}){HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());
        if(bytes>limits.maxStorageBufferRange || bytes>UINT32_MAX-3u)return Status::Unimplemented("Vulkan row page map byte address limit");}
      Dispatch dispatch;dispatch.values={step.inputs[0],step.inputs[1],step.inputs[0],step.outputs[0]};
      const std::array<SpecializationConstant,13> constants{{{0,4},{1,0},{2,std::uint32_t(inputs[0].dimensions()[0])},{3,std::uint32_t(count)},{4,0},
          {5,std::uint32_t(map->query_offset/4)},{6,std::uint32_t(map->kv_offset/4)},{7,std::uint32_t(map->page_offset/4)},{8,std::uint32_t(map->tail_offset/4)},
          {9,map->page_size},{10,map->max_pages},{11,map->max_requests()},{12,map->max_refs()}}};
      HYLE_ASSIGN_OR_RETURN(dispatch.pipeline,create_pipeline(program.runtime->device,{shaders::row_cache(),4,12,constants}));
      const auto groups=(count+127)/128,x=std::min(groups,std::uint64_t(limits.maxComputeWorkGroupCount[0]));
      if(!x || (groups+x-1)/x>limits.maxComputeWorkGroupCount[1])return Status::Unimplemented("Vulkan row page map dispatch grid");
      dispatch.groups={std::uint32_t(x),std::uint32_t((groups+x-1)/x),1};dispatch.tensor_byte_origins=true;dispatch.push.resize(12);
      program.dispatches.push_back(std::move(dispatch));return finish(fact);
    }
    const auto encode = step.operation.custom_as<model_ops::RowCacheEncode>();
    const auto gather = step.operation.custom_as<model_ops::RowCacheGatherDecode>();
    const auto padded = step.operation.custom_as<model_ops::RowCacheGatherDecodePadded>();
    if ((!encode && !gather && !padded) || step.outputs.size() != 1 ||
        step.inputs.size() != (encode ? 1u : 2u))
      return Status::InvalidArgument("Vulkan row cache ports differ");
    const auto encoding = encode ? encode->encoding : (padded?padded->encoding:gather->encoding);
    const std::array inputs{program.values[step.inputs[0]].description,
        encode ? TensorDesc{} : program.values[step.inputs[1]].description};
    const auto output = program.values[step.outputs[0]].description;
    std::array<TensorDesc, 1> inferred;
    HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation,
        std::span(inputs).first(step.inputs.size()), inferred));
    if (inferred[0] != output)
      return Status::InvalidArgument("Vulkan row cache output differs");
    const auto rows = static_cast<std::uint64_t>(inputs[0].dimensions()[0]);
    const auto selected = encode ? 0 : inputs[1].element_count().value();
    const bool checked = encode || padded || gather->bounds == IndexBoundsPolicy::kChecked;
    RowCacheAccessFacts fact;
    // The typed page-map chain uses rank-two selections. Other valid expert
    // decoder shapes retain their native implementation without this proof.
    if (!encode && checked && inputs[1].rank() == 2)
      fact = {.kind = padded ? RowCacheAccessKind::kPaddedCheckedDecode
                             : RowCacheAccessKind::kCheckedDecode,
              .bank_value = step.inputs[0],
              .query_rows = std::uint32_t(inputs[1].dimensions()[0])};
    if (encode ? rows == 0 : selected == 0) return finish(fact);
    const auto &limits = program.runtime->device->properties.limits;
    if (limits.maxComputeWorkGroupInvocations < 128 ||
        limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeSharedMemorySize < 2080)
      return Status::Unimplemented("Vulkan row cache workgroup limits");
    for (const auto value : step.inputs) {
      const auto bytes = program.values[value].description.required_bytes().value();
      if (bytes > limits.maxStorageBufferRange ||
          bytes > (std::numeric_limits<std::uint32_t>::max)() - 3u)
        return Status::Unimplemented("Vulkan row cache input exceeds byte address range");
    }
    const auto bytes = output.required_bytes().value();
    if (bytes > limits.maxStorageBufferRange ||
        bytes > (std::numeric_limits<std::uint32_t>::max)() - 3u)
      return Status::Unimplemented("Vulkan row cache output exceeds byte address range");
    while (program.aliases.size() < program.values.size()) {
      const auto id = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({id, 0});
    }
    if (checked && !program.status_value) {
      HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kInt32, {}));
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({description, {}});
      program.aliases.push_back({id, 0});
      program.status_value = id;
    }
    const auto append = [&](std::uint32_t phase, std::uint64_t groups) -> Status {
      Dispatch dispatch;
      dispatch.values = {padded && !rows ? step.inputs[1] : step.inputs[0], encode ? step.inputs[0] : step.inputs[1],
                         checked ? *program.status_value : step.inputs[0], step.outputs[0]};
      const std::array<SpecializationConstant, 6> constants{{
          {0, phase}, {1, encoding.format() == RowCacheFormat::kFloat4Block32 ? 1u : 0u},
          {2, static_cast<std::uint32_t>(rows)},
          {3, static_cast<std::uint32_t>(selected)}, {4, checked ? 1u : 0u},{13,padded?1u:0u}}};
      HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
          {shaders::row_cache(), 4, 12, constants}));
      const auto x = (std::min)(groups, std::uint64_t{limits.maxComputeWorkGroupCount[0]});
      if (x == 0 || (groups + x - 1) / x > limits.maxComputeWorkGroupCount[1])
        return Status::Unimplemented("Vulkan row cache dispatch exceeds device grid");
      dispatch.groups = {static_cast<std::uint32_t>(x),
                        static_cast<std::uint32_t>((groups + x - 1) / x), 1};
      dispatch.tensor_byte_origins = true;
      dispatch.checked_status = checked;
      dispatch.push.resize(12);
      program.dispatches.push_back(std::move(dispatch));
      return Status::Ok();
    };
    if (encode) {
      HYLE_RETURN_IF_ERROR(append(0, (rows * encoding.depth() + 127) / 128));
      return append(1, rows);
    }
    if (checked) HYLE_RETURN_IF_ERROR(append(2, (selected + 127) / 128));
    HYLE_RETURN_IF_ERROR(append(3, (selected * (encoding.depth() / 2) + 127) / 128));
    return finish(fact);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan row cache lowering metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
