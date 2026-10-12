#pragma once
#include <array>
#include <optional>
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/paged_arena_access.h"

namespace hyle::compute::detail {
// Fixed F5 metadata/permission glue instantiated only by the two native owners.
// No math, executor, queue or fallback is defined here.
struct NativeMirrorOwner final {
  NativeMirrorOwner(MaterializedBatchProgram p,mem::AllocatorRef a,ResourceBudget b):program(std::move(p)),allocator(a),budget(std::move(b)){}
  MaterializedBatchProgram program;mem::AllocatorRef allocator;ResourceBudget budget;
};
template<class T>struct MaterializedNativeState {
  explicit MaterializedNativeState(mem::AllocatorRef a):allocator(a){}
  mem::AllocatorRef allocator;MaterializedBatchProgram recipe;
  const typename T::Memory*device=nullptr;const typename T::Memory*staging=nullptr;
  typename T::Invocation native;typename T::CopyInvocation upload;
  std::array<WriteLease,2> staging_writers;std::array<ReadLease,2> staging_ready;
  std::array<std::array<WriteLease,2>,2> mirror_writers;
  std::array<std::array<ReadLease,2>,2> mirror_ready;
  std::array<ReadLease,5> reads;PagedArenaRead keys,values;MaterializedBatchVersion version;
  LeaseUsePool uses;MaterializedNativeBatchReport report;
  int native_bank=-1,prepared_bank=0;
  bool prepared=false,bound=false,closed=false,failed=false;
  LeaseImportOptions import;

  Status initialize(const typename T::Context&context,typename T::Program program,
      MaterializedBatchProgram common,const typename T::Memory&d,const typename T::Memory&h,
      MaterializedNativeBatchOptions options)noexcept{
    if(!program.valid() || !common.valid() || !d.valid() || !h.valid())return Status::InvalidArgument("native batch program/resources invalid");
    recipe=std::move(common);device=&d;staging=&h;import={.allocator=allocator,.resource_budget=options.resource_budget};
    const auto costs=recipe.costs();if(costs.index_bytes>SIZE_MAX-costs.mask_bytes)return Status::OutOfRange("native metadata size overflow");
    const auto bytes=costs.index_bytes+costs.mask_bytes;
    if(bytes>(SIZE_MAX-378)/3 || 3*bytes+378>options.max_metadata_backing_bytes)return Status::ResourceExhausted("native metadata fixed backing cap");
    HYLE_ASSIGN_OR_RETURN(native,T::invocation(program,allocator));
    HYLE_ASSIGN_OR_RETURN(auto hp,LeasePool::create(h.ref(),bytes+126,{.max_blocks=2,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(auto dp,LeasePool::create(d.ref(),2*bytes+252,{.max_blocks=4,.allocator=allocator,.resource_budget=options.resource_budget}));
    const std::array lengths{costs.index_bytes,costs.mask_bytes};
    for(std::size_t i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(staging_writers[i],hp.allocate(lengths[i],64));for(auto&bank:mirror_writers){HYLE_ASSIGN_OR_RETURN(bank[i],dp.allocate(lengths[i],64));}}
    for(auto*pool:{&hp,&dp}){
      auto*owner=allocator.template construct<NativeMirrorOwner>(recipe,allocator,options.resource_budget);if(!owner)return Status::ResourceExhausted("native mirror owner metadata");
      LeaseAccess::retain_resource_owner(*pool,owner,[](void*opaque)noexcept{auto*p=static_cast<NativeMirrorOwner*>(opaque);const auto a=p->allocator;a.destroy(p);});LeaseAccess::detach_pool_root(*pool);
    }
    const std::array specs{CopyResourceSpec{h.ref(),lengths[0]},CopyResourceSpec{h.ref(),lengths[1]},CopyResourceSpec{d.ref(),lengths[0]},CopyResourceSpec{d.ref(),lengths[1]}};
    HYLE_ASSIGN_OR_RETURN(auto copy,T::copy_program(context,specs,{.max_resources=4,.max_copies=2,.max_phases=1,.max_bytes=bytes,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(upload,T::copy_invocation(copy));const auto refs=recipe.schema().capacity().max_page_refs;
    if(refs>(UINT32_MAX-5)/2)return Status::ResourceExhausted("native batch page use capacity");
    HYLE_ASSIGN_OR_RETURN(uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=5+2*refs,.allocator=allocator,.resource_budget=options.resource_budget}));
    report.staging_bytes=bytes;report.device_mirror_bytes=2*bytes;report.backing_bytes=3*bytes+378;return Status::Ok();
  }
  Status open()const noexcept{return closed || failed?Status::FailedPrecondition("native batch controller closed/failed"):Status::Ok();}
  Status prepare(BatchLayoutVersion layout)noexcept{
    HYLE_RETURN_IF_ERROR(open());HYLE_ASSIGN_OR_RETURN(auto next,recipe.version(std::move(layout)));
    HYLE_RETURN_IF_ERROR(T::reuse_gate(native));if(uses.active_batches())return Status::Unavailable("native batch actual permission use retained");
    HYLE_RETURN_IF_ERROR(upload.reset());prepared=false;bound=false;reads={};keys={};values={};version={};
    prepared_bank=native_bank==0?1:0;
    const auto transfer=[&]()noexcept->Status{
      const std::array lengths{recipe.costs().index_bytes,recipe.costs().mask_bytes};
      for(std::size_t i=0;i<2;++i){
        if(staging_ready[i].valid()){HYLE_ASSIGN_OR_RETURN(staging_writers[i],staging_ready[i].try_write());}
        if(mirror_ready[prepared_bank][i].valid()){HYLE_ASSIGN_OR_RETURN(mirror_writers[prepared_bank][i],mirror_ready[prepared_bank][i].try_write());}
        Result<ReadLease> source=i==0?next.indices():next.mask();if(!source)return source.error();HYLE_ASSIGN_OR_RETURN(auto mapping,source->view()->map_host());
        HYLE_RETURN_IF_ERROR(staging_writers[i].view()->write(mapping.bytes()));HYLE_ASSIGN_OR_RETURN(staging_ready[i],staging_writers[i].publish());
        HYLE_RETURN_IF_ERROR(upload.bind_source({std::uint32_t(i)},staging_ready[i]));HYLE_RETURN_IF_ERROR(upload.bind_mutable({std::uint32_t(i+2)},std::move(mirror_writers[prepared_bank][i])));
      }
      const std::array<CopyPhase,1> phases{{{}}};const std::array<CopyRange,2> ranges{{{{0},{2},{0},0,0,lengths[0]},{{1},{3},{0},0,0,lengths[1]}}};
      HYLE_RETURN_IF_ERROR(upload.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,upload.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
      for(std::size_t i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(mirror_ready[prepared_bank][i],submitted.output({0},{std::uint32_t(i+2)},0,lengths[i]));}
      const auto facts=submitted.report();report.layout_generation=next.layout().generation();report.uploaded_bytes=facts.logical_bytes;report.upload_host_waits=facts.host_waits;return Status::Ok();
    };
    const auto status=transfer();if(!status.ok()){failed=true;return status;}
    reads[3]=mirror_ready[prepared_bank][0];reads[4]=mirror_ready[prepared_bank][1];version=std::move(next);prepared=true;return Status::Ok();
  }
  Status bind(ReadLease query,PagedArenaRead k,PagedArenaRead v)noexcept{
    HYLE_RETURN_IF_ERROR(open());if(!prepared || bound)return Status::FailedPrecondition("native batch needs an unbound prepared episode");
    if(recipe.schema().capacity().storage!=BatchStorage::kPaged || k.layout()!=version.layout() || v.layout()!=version.layout())return Status::InvalidArgument("native page frames differ from prepared generation");
    const auto descriptions=recipe.descriptions();HYLE_ASSIGN_OR_RETURN(auto qt,query.tensor(descriptions[0]));if(qt.storage().resource()!=device->ref())return Status::InvalidArgument("native batch Q resource differs");
    HYLE_ASSIGN_OR_RETURN(auto q,T::binding(*device,query,descriptions[0],import));HYLE_ASSIGN_OR_RETURN(auto kn,T::paged_binding(*device,k,import));HYLE_ASSIGN_OR_RETURN(auto vn,T::paged_binding(*device,v,import));
    HYLE_ASSIGN_OR_RETURN(auto indices,T::binding(*device,reads[3],descriptions[3],import));HYLE_ASSIGN_OR_RETURN(auto mask,T::binding(*device,reads[4],descriptions[4],import));
    const std::array inputs{q,kn,vn,indices,mask};const auto status=native.bind(inputs);if(!status.ok()){failed=true;return status;}
    reads[0]=std::move(query);keys=std::move(k);values=std::move(v);native_bank=prepared_bank;bound=true;return Status::Ok();
  }
  Status bind(ReadLease query,ReadLease k,ReadLease v)noexcept{
    HYLE_RETURN_IF_ERROR(open());if(!prepared || bound)return Status::FailedPrecondition("native batch needs an unbound prepared episode");
    if(recipe.schema().capacity().storage!=BatchStorage::kRagged)return Status::InvalidArgument("native paged batch needs exact page frames");
    const std::array sources{query,k,v};const auto descriptions=recipe.descriptions();
    std::array<std::optional<typename T::View>,5> views;
    for(std::size_t i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(auto tensor,sources[i].tensor(descriptions[i]));if(tensor.storage().resource()!=device->ref())return Status::InvalidArgument("native batch source resource differs");HYLE_ASSIGN_OR_RETURN(views[i],T::binding(*device,sources[i],descriptions[i],import));}
    for(std::size_t i=3;i<5;++i){HYLE_ASSIGN_OR_RETURN(views[i],T::binding(*device,reads[i],descriptions[i],import));}
    const std::array inputs{*views[0],*views[1],*views[2],*views[3],*views[4]};const auto status=native.bind(inputs);if(!status.ok()){failed=true;return status;}
    for(std::size_t i=0;i<3;++i)reads[i]=sources[i];
    native_bank=prepared_bank;bound=true;return Status::Ok();
  }
  auto submit()noexcept->Result<typename T::Submission>{
    HYLE_RETURN_IF_ERROR(open());if(!prepared || !bound)return std::unexpected(Status::FailedPrecondition("native batch is not prepared/bound"));
    HYLE_ASSIGN_OR_RETURN(auto batch,uses.begin());for(const auto&lease:reads)if(lease.valid())HYLE_RETURN_IF_ERROR(batch.read(lease));
    for(const auto&page:PagedArenaAccess::pages(keys))HYLE_RETURN_IF_ERROR(batch.read(page));
    for(const auto&page:PagedArenaAccess::pages(values))HYLE_RETURN_IF_ERROR(batch.read(page));
    auto submitted=T::submit(native,std::move(batch));prepared=false;bound=false;if(!submitted)failed=true;return submitted;
  }
  Status close()noexcept{closed=true;upload.close().IgnoreError();return uses.close();}
  Status drain()noexcept{close().IgnoreError();auto status=T::wait_native(native);status.Update(upload.drain());status.Update(uses.drain());return status;}
};
} // namespace hyle::compute::detail
