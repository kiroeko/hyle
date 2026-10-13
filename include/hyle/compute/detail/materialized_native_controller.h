#pragma once
#include <array>
#include <optional>
#include <type_traits>
#include <utility>
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/paged_arena_access.h"

namespace hyle::compute::detail {
// Fixed F5 metadata/permission glue instantiated only by the two native owners.
// No math, executor, queue or fallback is defined here.
struct MaterializedMetadataPolicy final {
  using Recipe = MaterializedBatchProgram;
  using Version = MaterializedBatchVersion;
  using Options = MaterializedNativeBatchOptions;
  using Report = MaterializedNativeBatchReport;
  static constexpr std::size_t metadata_count = 2;
  static constexpr bool prepare_native = false;
  static auto lengths(const Recipe &p) noexcept { return std::array{p.costs().index_bytes, p.costs().mask_bytes}; }
  static Result<ReadLease> read(const Version &v, std::size_t i) noexcept { return i == 0 ? v.indices() : v.mask(); }
};
struct PagedMetadataPolicy final {
  using Recipe = NativePagedBatchProgram;
  using Version = NativePagedBatchVersion;
  using Options = NativePagedInvocationOptions;
  using Report = NativePagedBatchReport;
  static constexpr std::size_t metadata_count = 1;
  static constexpr bool prepare_native = true;
  static auto lengths(const Recipe &p) noexcept { return std::array{p.costs().index_bytes}; }
  static Result<ReadLease> read(const Version &v, std::size_t) noexcept { return v.indices(); }
};
template<class Recipe>struct NativeMirrorOwner final {
  NativeMirrorOwner(Recipe p,mem::AllocatorRef a,ResourceBudget b):program(std::move(p)),allocator(a),budget(std::move(b)){}
  Recipe program;mem::AllocatorRef allocator;ResourceBudget budget;
};
template<class View,std::size_t N,std::size_t...I>
auto native_input_array(const std::array<std::optional<View>,N>&views,std::index_sequence<I...>) {
  return std::array<View,N>{*views[I]...};
}
struct FixedNativePreparation final {};
template<class Context>struct LayoutNativePreparation final {
  const Context*context=nullptr;WorkloadContext workload;
};
template<class T,class Policy=MaterializedMetadataPolicy>struct MaterializedNativeState {
  static constexpr std::size_t metadata_count=Policy::metadata_count,input_count=3+metadata_count;
  explicit MaterializedNativeState(mem::AllocatorRef a):allocator(a){}
  mem::AllocatorRef allocator;typename Policy::Recipe recipe;
  [[no_unique_address]] std::conditional_t<Policy::prepare_native,
      LayoutNativePreparation<typename T::Context>,FixedNativePreparation> preparation;
  const typename T::Memory*device=nullptr;const typename T::Memory*staging=nullptr;
  typename T::Invocation native;typename T::CopyInvocation upload;
  std::array<WriteLease,metadata_count> staging_writers;std::array<ReadLease,metadata_count> staging_ready;
  std::array<std::array<WriteLease,metadata_count>,2> mirror_writers;
  std::array<std::array<ReadLease,metadata_count>,2> mirror_ready;
  std::array<ReadLease,input_count> reads;PagedArenaRead keys,values;typename Policy::Version version;
  LeaseUsePool uses;typename Policy::Report report;
  int native_bank=-1,prepared_bank=0;
  bool prepared=false,bound=false,closed=false,failed=false;
  LeaseImportOptions import;

  Status initialize(const typename T::Context&selected_context,typename T::Program program,
      typename Policy::Recipe common,const typename T::Memory&d,const typename T::Memory&h,
      typename Policy::Options options,const WorkloadContext&selected_workload={})noexcept{
    if((!Policy::prepare_native && !program.valid()) || !common.valid() || !d.valid() || !h.valid())return Status::InvalidArgument("native batch program/resources invalid");
    recipe=std::move(common);
    if constexpr(Policy::prepare_native){preparation.context=&selected_context;preparation.workload=selected_workload;}
    device=&d;staging=&h;import={.allocator=allocator,.resource_budget=options.resource_budget};
    const auto lengths=Policy::lengths(recipe);std::size_t bytes=0;
    for(const auto length:lengths){if(length>SIZE_MAX-bytes)return Status::OutOfRange("native metadata size overflow");bytes+=length;}
    constexpr std::size_t padding=3*metadata_count*63;
    if(bytes>(SIZE_MAX-padding)/3 || 3*bytes+padding>options.max_metadata_backing_bytes)return Status::ResourceExhausted("native metadata fixed backing cap");
    if(program.valid()){HYLE_ASSIGN_OR_RETURN(native,T::invocation(program,allocator));}
    HYLE_ASSIGN_OR_RETURN(auto hp,LeasePool::create(h.ref(),bytes+metadata_count*63,{.max_blocks=std::uint32_t(metadata_count),.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(auto dp,LeasePool::create(d.ref(),2*bytes+2*metadata_count*63,{.max_blocks=std::uint32_t(2*metadata_count),.allocator=allocator,.resource_budget=options.resource_budget}));
    for(std::size_t i=0;i<metadata_count;++i){HYLE_ASSIGN_OR_RETURN(staging_writers[i],hp.allocate(lengths[i],64));for(auto&bank:mirror_writers){HYLE_ASSIGN_OR_RETURN(bank[i],dp.allocate(lengths[i],64));}}
    for(auto*pool:{&hp,&dp}){
      auto*owner=allocator.template construct<NativeMirrorOwner<typename Policy::Recipe>>(recipe,allocator,options.resource_budget);if(!owner)return Status::ResourceExhausted("native mirror owner metadata");
      LeaseAccess::retain_resource_owner(*pool,owner,[](void*opaque)noexcept{auto*p=static_cast<NativeMirrorOwner<typename Policy::Recipe>*>(opaque);const auto a=p->allocator;a.destroy(p);});LeaseAccess::detach_pool_root(*pool);
    }
    std::array<CopyResourceSpec,2*metadata_count> specs;
    for(std::size_t i=0;i<metadata_count;++i){specs[i]={h.ref(),lengths[i]};specs[i+metadata_count]={d.ref(),lengths[i]};}
    HYLE_ASSIGN_OR_RETURN(auto copy,T::copy_program(selected_context,specs,{.max_resources=std::uint32_t(2*metadata_count),.max_copies=std::uint32_t(metadata_count),.max_phases=1,.max_bytes=bytes,.allocator=allocator,.resource_budget=options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(upload,T::copy_invocation(copy));const auto refs=recipe.schema().capacity().max_page_refs;
    if(refs>(UINT32_MAX-input_count)/2)return Status::ResourceExhausted("native batch page use capacity");
    HYLE_ASSIGN_OR_RETURN(uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=std::uint32_t(input_count)+2*refs,.allocator=allocator,.resource_budget=options.resource_budget}));
    report.staging_bytes=bytes;report.device_mirror_bytes=2*bytes;report.backing_bytes=3*bytes+padding;return Status::Ok();
  }
  Status open()const noexcept{return closed || failed?Status::FailedPrecondition("native batch controller closed/failed"):Status::Ok();}
  Status prepare(BatchLayoutVersion layout)noexcept{
    HYLE_RETURN_IF_ERROR(open());HYLE_ASSIGN_OR_RETURN(auto next,recipe.version(std::move(layout)));
    if(native.valid()){HYLE_RETURN_IF_ERROR(T::reuse_gate(native));}if(uses.active_batches())return Status::Unavailable("native batch actual permission use retained");
    if constexpr(Policy::prepare_native){
      HYLE_ASSIGN_OR_RETURN(auto program,T::prepare_program(*preparation.context,recipe,next.layout(),preparation.workload));
      HYLE_ASSIGN_OR_RETURN(auto prepared_native,T::invocation(program,allocator));
      // A rejected upload reset must leave the old prepared generation and
      // its native recipe together. Commit the cold replacement only after
      // this admission gate succeeds; later accepted failures poison the slot.
      HYLE_RETURN_IF_ERROR(upload.reset());
      native=std::move(prepared_native);
    }else{HYLE_RETURN_IF_ERROR(upload.reset());}
    prepared=false;bound=false;reads={};keys={};values={};version={};
    prepared_bank=native_bank==0?1:0;
    const auto transfer=[&]()noexcept->Status{
      const auto lengths=Policy::lengths(recipe);
      for(std::size_t i=0;i<metadata_count;++i){
        if(staging_ready[i].valid()){HYLE_ASSIGN_OR_RETURN(staging_writers[i],staging_ready[i].try_write());}
        if(mirror_ready[prepared_bank][i].valid()){HYLE_ASSIGN_OR_RETURN(mirror_writers[prepared_bank][i],mirror_ready[prepared_bank][i].try_write());}
        auto source=Policy::read(next,i);if(!source)return source.error();HYLE_ASSIGN_OR_RETURN(auto mapping,source->view()->map_host());
        HYLE_RETURN_IF_ERROR(staging_writers[i].view()->write(mapping.bytes()));HYLE_ASSIGN_OR_RETURN(staging_ready[i],staging_writers[i].publish());
        HYLE_RETURN_IF_ERROR(upload.bind_source({std::uint32_t(i)},staging_ready[i]));HYLE_RETURN_IF_ERROR(upload.bind_mutable({std::uint32_t(i+metadata_count)},std::move(mirror_writers[prepared_bank][i])));
      }
      const std::array<CopyPhase,1> phases{{{}}};std::array<CopyRange,metadata_count> ranges;
      for(std::size_t i=0;i<metadata_count;++i)ranges[i]={{std::uint32_t(i)},{std::uint32_t(i+metadata_count)},{0},0,0,lengths[i]};
      HYLE_RETURN_IF_ERROR(upload.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,upload.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
      for(std::size_t i=0;i<metadata_count;++i){HYLE_ASSIGN_OR_RETURN(mirror_ready[prepared_bank][i],submitted.output({0},{std::uint32_t(i+metadata_count)},0,lengths[i]));}
      const auto facts=submitted.report();report.layout_generation=next.layout().generation();report.uploaded_bytes=facts.logical_bytes;report.upload_host_waits=facts.host_waits;return Status::Ok();
    };
    const auto status=transfer();if(!status.ok()){failed=true;return status;}
    for(std::size_t i=0;i<metadata_count;++i)reads[3+i]=mirror_ready[prepared_bank][i];version=std::move(next);prepared=true;return Status::Ok();
  }
  Status bind(ReadLease query,PagedArenaRead k,PagedArenaRead v)noexcept{
    HYLE_RETURN_IF_ERROR(open());if(!prepared || bound)return Status::FailedPrecondition("native batch needs an unbound prepared episode");
    if constexpr(Policy::prepare_native){
      if(version.layout().facts().query_rows==0 && version.layout().facts().page_refs==0){
        if(k.valid() || v.valid())return Status::InvalidArgument("empty native page episode needs no KV frames");
        const auto description=recipe.descriptions()[0];HYLE_ASSIGN_OR_RETURN(auto qt,query.tensor(description));
        if(qt.storage().resource()!=device->ref())return Status::InvalidArgument("native batch Q resource differs");
        HYLE_ASSIGN_OR_RETURN(auto q,T::binding(*device,query,description,import));
        const std::array inputs{std::move(q)};const auto status=native.bind(inputs);
        if(!status.ok()){failed=true;return status;}
        reads[0]=std::move(query);native_bank=prepared_bank;bound=true;return Status::Ok();
      }
    }
    if(recipe.schema().capacity().storage!=BatchStorage::kPaged || k.layout()!=version.layout() || v.layout()!=version.layout())return Status::InvalidArgument("native page frames differ from prepared generation");
    const auto descriptions=recipe.descriptions();HYLE_ASSIGN_OR_RETURN(auto qt,query.tensor(descriptions[0]));if(qt.storage().resource()!=device->ref())return Status::InvalidArgument("native batch Q resource differs");
    HYLE_ASSIGN_OR_RETURN(auto q,T::binding(*device,query,descriptions[0],import));HYLE_ASSIGN_OR_RETURN(auto kn,T::paged_binding(*device,k,import));HYLE_ASSIGN_OR_RETURN(auto vn,T::paged_binding(*device,v,import));
    std::array<std::optional<typename T::View>,input_count> views;
    views[0]=std::move(q);views[1]=std::move(kn);views[2]=std::move(vn);
    for(std::size_t i=3;i<input_count;++i){HYLE_ASSIGN_OR_RETURN(views[i],T::binding(*device,reads[i],descriptions[i],import));}
    const auto inputs=native_input_array(views,std::make_index_sequence<input_count>{});
    const auto status=native.bind(inputs);if(!status.ok()){failed=true;return status;}
    reads[0]=std::move(query);keys=std::move(k);values=std::move(v);native_bank=prepared_bank;bound=true;return Status::Ok();
  }
  Status bind(ReadLease query,ReadLease k,ReadLease v)noexcept{
    HYLE_RETURN_IF_ERROR(open());if(!prepared || bound)return Status::FailedPrecondition("native batch needs an unbound prepared episode");
    if(recipe.schema().capacity().storage!=BatchStorage::kRagged)return Status::InvalidArgument("native paged batch needs exact page frames");
    const std::array sources{query,k,v};const auto descriptions=recipe.descriptions();
    std::array<std::optional<typename T::View>,input_count> views;
    for(std::size_t i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(auto tensor,sources[i].tensor(descriptions[i]));if(tensor.storage().resource()!=device->ref())return Status::InvalidArgument("native batch source resource differs");HYLE_ASSIGN_OR_RETURN(views[i],T::binding(*device,sources[i],descriptions[i],import));}
    for(std::size_t i=3;i<input_count;++i){HYLE_ASSIGN_OR_RETURN(views[i],T::binding(*device,reads[i],descriptions[i],import));}
    const auto inputs=native_input_array(views,std::make_index_sequence<input_count>{});const auto status=native.bind(inputs);if(!status.ok()){failed=true;return status;}
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
  Status drain()noexcept{close().IgnoreError();auto status=native.valid()?T::wait_native(native):Status::Ok();status.Update(upload.drain());status.Update(uses.drain());return status;}
};
} // namespace hyle::compute::detail
