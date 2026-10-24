#include "hyle/compute/vulkan_row_cache.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "hyle/compute/detail/vulkan_plan.h"
#include "hyle/compute/model_ops.h"
#include "../compute_batch/row_cache_native_controller.h"
#include "vulkan_internal.h"
namespace hyle::compute::vulkan {
namespace detail {
struct RowCacheTraits {
  using Context=vulkan::Context;using Memory=MemoryResource;using Invocation=vulkan::Invocation;using Program=vulkan::Program;
  using CopyInvocation=CopyBatchInvocation;using Submission=OwnedResult<Results>;
  static Result<Program> program(const Context&context,compute::detail::RowCacheMap map,RowCacheEncoding encoding,
      const std::array<TensorDesc,3>&inputs,TensorDesc output)noexcept{
    try{Plan plan;plan.inputs={0,1,2};plan.outputs={4};plan.values={{inputs[0],{}},{inputs[1],{}},{inputs[2],{}},{inputs[1],{}},{output,{}}};
      plan.steps.push_back({Operation::custom(map),{1,2},{3},{},{1,1,1},{}});
      plan.steps.push_back({model_ops::row_cache_gather_decode(encoding),{0,3},{4},{},{1,1,1},{}});
      return prepare_plan(context,plan);
    }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan row reader cold program"));}
  }
  static auto invocation(const Program&p,const Memory&d,const Memory&h,LeaseImportOptions o){return p.create_invocation(o.allocator,{.resource_budget=o.resource_budget,.device=d.ref(),.staging=h.ref()});}
  static auto copy_program(const Context&c,std::span<const CopyResourceSpec>s,CopyBatchOptions o){return CopyBatchProgram::prepare(c,s,o);}
  static auto copy_invocation(const CopyBatchProgram&p){return CopyBatchInvocation::create(p);}
  static Result<Tensor> binding(const Memory&m,ReadLease r,TensorDesc d,LeaseImportOptions o){
    HYLE_ASSIGN_OR_RETURN(auto bytes,d.required_bytes());if(bytes)return m.binding(std::move(r),d,o);
    HYLE_ASSIGN_OR_RETURN(auto one,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    HYLE_ASSIGN_OR_RETURN(auto base,m.binding(std::move(r),one,o));return Access::tensor(d,Access::buffer(base));
  }
  static auto paged_binding(const Memory&m,PagedRowCacheRead r,TensorDesc,LeaseImportOptions o){return PagedBatchAccess::binding(m,std::move(r),o);}
  static auto submit(Invocation&i,LeaseUseBatch u){return PagedBatchAccess::submit(i,std::move(u));}
  static auto reuse(Invocation&i){return PagedBatchAccess::reuse_gate(i);}
  static auto release_inputs(Invocation&i){return PagedBatchAccess::release_inputs(i);}
  static auto wait(Invocation&i){return PagedBatchAccess::wait_native(i);}
};
struct RowCachePaddedTraits final:RowCacheTraits{
  static Result<Program> padded_program(const Context&context,compute::detail::RowCachePaddedMap map,RowCacheEncoding encoding,
      const std::array<TensorDesc,4>&inputs,const std::array<TensorDesc,2>&outputs)noexcept{
    try{
      Plan plan;plan.inputs={0,1,2,3};plan.outputs={6,5};
      for(const auto&description:inputs)plan.values.push_back({description,{}});
      plan.values.push_back({inputs[1],{}});plan.values.push_back({outputs[1],{}});plan.values.push_back({outputs[0],{}});
      plan.steps.push_back({Operation::custom(map),{1,2,3},{4,5},{},{1,1,1},{}});
      plan.steps.push_back({model_ops::row_cache_gather_decode_padded(encoding),{0,4},{6},{},{1,1,1},{}});
      return prepare_plan(context,plan);
    }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan padded row cold program"));}
  }
};
struct PagedRowCacheGatherPaddedState final:compute::detail::RowCacheNativeState<RowCachePaddedTraits,true>{using RowCacheNativeState::RowCacheNativeState;};
static void destroy(PagedRowCacheGatherPaddedState*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
struct PagedRowCacheGatherState final:compute::detail::RowCacheNativeState<RowCacheTraits>{using RowCacheNativeState::RowCacheNativeState;};
static void destroy(PagedRowCacheGatherState*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
}
PagedRowCacheGatherInvocation::PagedRowCacheGatherInvocation(PagedRowCacheGatherInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
PagedRowCacheGatherInvocation&PagedRowCacheGatherInvocation::operator=(PagedRowCacheGatherInvocation&&v)noexcept{if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
PagedRowCacheGatherInvocation::~PagedRowCacheGatherInvocation(){detail::destroy(data_);}
Result<PagedRowCacheGatherInvocation> PagedRowCacheGatherInvocation::create(const Context&c,const MemoryResource&d,const MemoryResource&h,RowCacheBatchSchema s,TensorDesc ids,PagedRowCacheGatherOptions o)noexcept{
  auto a=o.allocator;if(!a.valid())return std::unexpected(Status::InvalidArgument("Vulkan row reader allocator"));if(o.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,o.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::PagedRowCacheGatherState>(a);if(!p)return std::unexpected(Status::ResourceExhausted("Vulkan row reader metadata"));PagedRowCacheGatherInvocation result(p);
  HYLE_RETURN_IF_ERROR(p->initialize(c,d,h,std::move(s),ids,o));return result;
}
TensorDesc PagedRowCacheGatherInvocation::output_description()const noexcept{return data_?data_->output:TensorDesc{};}
Status PagedRowCacheGatherInvocation::prepare_layout(RowCacheLayoutVersion v)noexcept{return data_?data_->prepare(std::move(v)):Status::FailedPrecondition("empty Vulkan row reader");}
Result<OwnedResult<Results>> PagedRowCacheGatherInvocation::submit(PagedRowCacheRead f,ReadLease ids)noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("empty Vulkan row reader"));return data_->submit(std::move(f),std::move(ids));}
Status PagedRowCacheGatherInvocation::close()noexcept{return data_?data_->close():Status::Ok();}
Status PagedRowCacheGatherInvocation::drain()noexcept{return data_?data_->drain():Status::Ok();}
PagedRowCacheGatherPaddedInvocation::PagedRowCacheGatherPaddedInvocation(PagedRowCacheGatherPaddedInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
PagedRowCacheGatherPaddedInvocation&PagedRowCacheGatherPaddedInvocation::operator=(PagedRowCacheGatherPaddedInvocation&&v)noexcept{if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
PagedRowCacheGatherPaddedInvocation::~PagedRowCacheGatherPaddedInvocation(){detail::destroy(data_);}
Result<PagedRowCacheGatherPaddedInvocation> PagedRowCacheGatherPaddedInvocation::create(const Context&context,const MemoryResource&device,const MemoryResource&host,
    RowCacheBatchSchema schema,TensorDesc ids,std::uint32_t columns,PagedRowCacheGatherOptions options)noexcept{
  auto allocator=options.allocator;if(!allocator.valid())return std::unexpected(Status::InvalidArgument("vulkan padded row allocator"));
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  auto*state=allocator.construct<detail::PagedRowCacheGatherPaddedState>(allocator);
  if(!state)return std::unexpected(Status::ResourceExhausted("vulkan padded row metadata"));
  PagedRowCacheGatherPaddedInvocation result(state);HYLE_RETURN_IF_ERROR(state->initialize(context,device,host,std::move(schema),ids,options,columns));return result;
}
TensorDesc PagedRowCacheGatherPaddedInvocation::output_description()const noexcept{return data_?data_->output:TensorDesc{};}
TensorDesc PagedRowCacheGatherPaddedInvocation::mask_description()const noexcept{return data_?data_->mask:TensorDesc{};}
Status PagedRowCacheGatherPaddedInvocation::prepare_layout(RowCacheLayoutVersion layout)noexcept{return data_?data_->prepare(std::move(layout)):Status::FailedPrecondition("empty vulkan padded row reader");}
Result<OwnedResult<Results>> PagedRowCacheGatherPaddedInvocation::submit(PagedRowCacheRead frame,ReadLease ids,ReadLease ranges)noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("empty vulkan padded row reader"));
  return data_->submit(std::move(frame),std::move(ids),std::move(ranges));
}
Status PagedRowCacheGatherPaddedInvocation::close()noexcept{return data_?data_->close():Status::Ok();}
Status PagedRowCacheGatherPaddedInvocation::drain()noexcept{return data_?data_->drain():Status::Ok();}

}
