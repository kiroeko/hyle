#pragma once
#include "hyle/compute/detail/paged_arena_access.h"
#include "row_cache_map.h"
#include <array>
namespace hyle::compute::detail {
// Only fixed typed-row metadata and original F1/F2 permission glue. Each
// backend supplies its original Program, Invocation and owning Results.
template<class T,bool Padded=false>struct RowCacheNativeState {
  explicit RowCacheNativeState(mem::AllocatorRef a):allocator(a),reads(a){}
  mem::AllocatorRef allocator;RowCacheBatchSchema schema;const typename T::Memory*memory=nullptr;
  std::array<TensorDesc,Padded?4:3> inputs;TensorDesc output,mask;
  std::uint32_t score_columns=0;RowCacheLayoutVersion layout;
  LeasePool staging_pool,mirror_pool;WriteLease staging_writer,mirror_writer;ReadLease staging_ready,mirror_ready;
  typename T::CopyInvocation upload;LeaseUsePool uses;BatchPageReads reads;
  LeaseImportOptions import;bool closed=false,failed=false,prepared=false;
  // Native invocation dies before external pool handles. Accepted original
  // native owners, not this controller, retain backing through late work.
  typename T::Invocation native;
  template<class Options>Status initialize(const typename T::Context&context,const typename T::Memory&device,
      const typename T::Memory&staging,RowCacheBatchSchema selected,TensorDesc ids,const Options&options,std::uint32_t columns=0)noexcept{
    if(!selected.valid() || !device.valid() || !staging.valid() || !ids.valid() || !ids.is_plain() || !ids.is_dense() ||
        ids.rank()!=2 || ids.data_type()!=DataType::kInt32 || std::uint64_t(ids.dimensions()[0])>selected.capacity().max_query_rows)
      return Status::InvalidArgument("native row reader schema/resources/IDs differ");
    score_columns=columns;schema=std::move(selected);memory=&device;import={.allocator=allocator,.resource_budget=options.resource_budget};
    const auto cap=schema.capacity();const auto offsets=schema.metadata_offsets();const auto physical=std::uint64_t(cap.max_pages)*cap.page_size;
    if(physical>std::uint64_t(INT32_MAX)+1)return Status::OutOfRange("native row reader physical I32 capacity");
    const auto bytes=offsets.bytes;if(bytes>SIZE_MAX/2 || 2*bytes>options.max_metadata_backing_bytes)return Status::ResourceExhausted("native row metadata byte cap");
    HYLE_ASSIGN_OR_RETURN(inputs[0],row_cache_carrier_description(cap.encoding,std::int64_t(physical)));inputs[1]=ids;
    HYLE_ASSIGN_OR_RETURN(inputs[Padded?3:2],TensorDesc::dense(DataType::kUInt8,std::array<std::int64_t,1>{std::int64_t(bytes)}));
    HYLE_ASSIGN_OR_RETURN(output,TensorDesc::dense(DataType::kBFloat16,std::array{ids.dimensions()[0],ids.dimensions()[1],std::int64_t(cap.encoding.depth())}));
    const RowCacheMap map{offsets.query_indptr,offsets.kv_indptr,offsets.page_indices,offsets.last_page_lengths,offsets.bytes,cap.page_size,cap.max_pages};
    typename T::Program program;
    if constexpr(Padded){
      HYLE_ASSIGN_OR_RETURN(inputs[2],TensorDesc::dense(DataType::kInt32,std::array{ids.dimensions()[0],std::int64_t{2}}));
      HYLE_ASSIGN_OR_RETURN(mask,TensorDesc::dense(DataType::kFloat32,ids.dimensions()));
      std::array<TensorDesc,2> inferred;const RowCachePaddedMap padded{map,columns};
      HYLE_RETURN_IF_ERROR(padded.infer_output_descriptions(std::array{ids,inputs[2],inputs[3]},inferred));
      HYLE_ASSIGN_OR_RETURN(program,T::padded_program(context,padded,cap.encoding,inputs,std::array{output,mask}));
    }else{HYLE_ASSIGN_OR_RETURN(program,T::program(context,map,cap.encoding,inputs,output));}
    HYLE_ASSIGN_OR_RETURN(native,T::invocation(program,device,staging,import));
    HYLE_ASSIGN_OR_RETURN(staging_pool,LeasePool::create(staging.ref(),bytes,{.max_blocks=1,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(mirror_pool,LeasePool::create(device.ref(),bytes,{.max_blocks=1,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(staging_writer,staging_pool.allocate(bytes,8));HYLE_ASSIGN_OR_RETURN(mirror_writer,mirror_pool.allocate(bytes,8));
    const std::array specs{CopyResourceSpec{staging.ref(),bytes},CopyResourceSpec{device.ref(),bytes}};
    HYLE_ASSIGN_OR_RETURN(auto copy,T::copy_program(context,specs,{.max_resources=2,.max_copies=1,.max_phases=1,.max_bytes=bytes,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(upload,T::copy_invocation(copy));
    if(cap.max_page_refs>UINT32_MAX-(Padded?3u:2u))return Status::OutOfRange("native row page permission capacity");
    HYLE_ASSIGN_OR_RETURN(uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=cap.max_page_refs+(Padded?3u:2u),.allocator=allocator,.resource_budget=options.resource_budget}));
    return reads.reserve(std::size_t(cap.max_page_refs)+(Padded?3u:2u));
  }
  Status reusable()noexcept{
    if(closed || failed)return Status::FailedPrecondition("native row reader closed/failed");
    if(uses.active_batches())return Status::Unavailable("native row reader actual page uses retained");
    return T::reuse(native);
  }
  Status prepare(RowCacheLayoutVersion next)noexcept{
    HYLE_RETURN_IF_ERROR(reusable());
    if(!schema.owns(next) || next.facts().query_rows!=std::uint64_t(inputs[1].dimensions()[0]))return Status::InvalidArgument("native row layout/selection rows differ");
    HYLE_ASSIGN_OR_RETURN(auto count,output.element_count());if(!Padded && count && !next.facts().page_refs)return Status::OutOfRange("native row nonempty IDs cannot read empty history");
    if(prepared && next==layout)return Status::Ok();
    HYLE_RETURN_IF_ERROR(T::release_inputs(native));prepared=false;layout={};
    if(staging_ready.valid()){HYLE_ASSIGN_OR_RETURN(staging_writer,staging_ready.try_write());}
    if(mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(mirror_writer,mirror_ready.try_write());}
    const auto transfer=[&]()noexcept->Status{
      HYLE_ASSIGN_OR_RETURN(auto metadata,next.metadata());HYLE_ASSIGN_OR_RETURN(auto source,metadata.view());HYLE_ASSIGN_OR_RETURN(auto mapped,source.map_host());
      HYLE_ASSIGN_OR_RETURN(auto target,staging_writer.view());auto status=target.write(mapped.bytes());status.Update(mapped.close());HYLE_RETURN_IF_ERROR(status);
      HYLE_ASSIGN_OR_RETURN(staging_ready,staging_writer.publish());HYLE_RETURN_IF_ERROR(upload.reset());
      HYLE_RETURN_IF_ERROR(upload.bind_source({0},staging_ready));HYLE_RETURN_IF_ERROR(upload.bind_mutable({1},std::move(mirror_writer)));
      const std::array<CopyPhase,1> phases{{{}}};const std::array<CopyRange,1> ranges{{{{0},{1},{0},0,0,schema.metadata_offsets().bytes}}};
      HYLE_RETURN_IF_ERROR(upload.prepare_replay(phases,ranges));{HYLE_ASSIGN_OR_RETURN(auto done,upload.submit());HYLE_RETURN_IF_ERROR(done.wait());}
      HYLE_ASSIGN_OR_RETURN(mirror_writer,upload.unbind_mutable({1}));HYLE_RETURN_IF_ERROR(upload.reset());
      HYLE_ASSIGN_OR_RETURN(mirror_ready,mirror_writer.publish());return Status::Ok();
    };
    const auto status=transfer();if(!status.ok()){failed=true;return status;}layout=std::move(next);prepared=true;return Status::Ok();
  }
  auto submit(PagedRowCacheRead frame,ReadLease ids,ReadLease ranges={})noexcept->Result<typename T::Submission>{
    HYLE_RETURN_IF_ERROR(reusable());if(!prepared || !frame.valid() || frame.layout()!=layout)return std::unexpected(Status::InvalidArgument("native row reader needs exact prepared frame"));
    HYLE_RETURN_IF_ERROR(PagedArenaAccess::validate_storage(frame,schema.capacity().encoding));
    HYLE_RETURN_IF_ERROR(T::release_inputs(native));
    HYLE_ASSIGN_OR_RETURN(auto bank,T::paged_binding(*memory,frame,inputs[0],import));
    HYLE_ASSIGN_OR_RETURN(auto index,T::binding(*memory,ids,inputs[1],import));
    HYLE_ASSIGN_OR_RETURN(auto metadata,T::binding(*memory,mirror_ready,inputs[Padded?3:2],import));
    if constexpr(Padded){HYLE_ASSIGN_OR_RETURN(auto range,T::binding(*memory,ranges,inputs[2],import));
      const std::array arguments{bank,index,range,metadata};HYLE_RETURN_IF_ERROR(native.bind(arguments));
    }else{const std::array arguments{bank,index,metadata};HYLE_RETURN_IF_ERROR(native.bind(arguments));}
    HYLE_ASSIGN_OR_RETURN(auto use,uses.begin());reads.reset();for(const auto&page:PagedArenaAccess::pages(frame))HYLE_RETURN_IF_ERROR(reads.add(page));
    HYLE_RETURN_IF_ERROR(reads.add(ids));if constexpr(Padded){HYLE_RETURN_IF_ERROR(reads.add(ranges));}HYLE_RETURN_IF_ERROR(reads.add(mirror_ready));HYLE_RETURN_IF_ERROR(reads.admit(use));
    return T::submit(native,std::move(use));
  }
  Status close()noexcept{closed=true;upload.close().IgnoreError();return uses.active_batches() || upload.in_flight()?Status::Unavailable("native row reader actual work retained"):Status::Ok();}
  Status drain()noexcept{close().IgnoreError();auto status=T::wait(native);status.Update(upload.drain());if(!status.ok() && uses.active_batches())return status;HYLE_RETURN_IF_ERROR(uses.drain());return status;}
};
}
