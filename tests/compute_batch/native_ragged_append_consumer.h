#pragma once
namespace hyle_f5_cuda_ragged_fixture {
namespace c=hyle::compute;namespace cu=c::cuda;
using hyle::Status;
Status run(c::Context&context,c::Queue&queue,cu::MemoryResource&device,cu::MemoryResource&host){
  for(bool bf16:{false,true}){hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
    {
      HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),1u<<20));c::BatchProgram program;c::BatchInvocation invocation;
      const auto loader=[&](c::TensorDesc,std::span<const std::byte> bytes)->hyle::Result<c::ReadLease>{HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes.size()));HYLE_RETURN_IF_ERROR(writer.view()->write(bytes));return writer.publish();};
      const auto prepare=[&](std::span<const c::CopyResourceSpec> specs,c::CopyBatchOptions options){return c::CopyBatchProgram::prepare(context,specs,options);};
      const auto copier=[&](const c::CopyBatchProgram&copy){return c::CopyBatchInvocation::create(copy,queue);};
      const auto execute=[&](c::MaterializedBatchProgram recipe,c::BatchLayoutVersion layout,c::ReadLease q,c::ReadLease k,c::ReadLease v)->hyle::Result<c::ReadLease>{
        if(!program.valid()){HYLE_ASSIGN_OR_RETURN(program,c::BatchProgram::prepare(context,recipe.schema(),{.resource=device.ref(),.host_workspace=host.ref(),.max_float_workspace_bytes=1u<<20,.max_int_workspace_bytes=64u<<10,.scale=.125f,.allocator=metadata.get()}));HYLE_ASSIGN_OR_RETURN(invocation,c::BatchInvocation::create(program,queue));}
        HYLE_ASSIGN_OR_RETURN(auto prepared,invocation.update_batch(layout));HYLE_ASSIGN_OR_RETURN(auto done,prepared.run(std::array{q,k,v}));HYLE_RETURN_IF_ERROR(done.wait());return invocation.output();
      };
      const auto reader=[](const c::ReadLease&source,std::span<std::byte> bytes){return source.view()->read(bytes);};
      const auto status=hyle_f5_ragged_journey::exercise(bf16,device.ref(),metadata.get(),loader,prepare,copier,execute,reader,"CUDA Direct");if(!status.ok()){context.drain().IgnoreError();return status;}
    }
    HYLE_RETURN_IF_ERROR(context.drain());HYLE_RETURN_IF_ERROR(hyle_f5_ragged_journey::require(metadata.used()==0,"ragged CUDA Direct metadata leaked"));
  }return Status::Ok();
}
} // namespace hyle_f5_cuda_ragged_fixture
