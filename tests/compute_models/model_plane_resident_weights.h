#pragma once
#include <algorithm>
#include <memory>
#include <vector>
#include "glm_original_moe_roles.h"
#include "hyle/compute/resource_leases.h"
#if defined(HYLE_TEST_GLM_STATE_CUDA)
#include "hyle/compute/cuda_weights.h"
#endif

// Descriptor-driven ORIGINAL planes and one global->local map. This is only
// cold weight layout/copy supply over the existing LeasePool/WeightSlots API;
// no router, tensor math, provider selection, execution loop or lookup I/O.
namespace model_plane_resident_weights {
namespace c=hyle::compute;namespace m=hyle::mem;
using Selection=glm_original_moe_roles::Selection;
struct Layout final {
  std::int64_t domain=0,capacity=0;
  std::vector<c::TensorDesc> source,resident;
  std::vector<std::size_t> plane_bytes,bank_bytes;
  std::size_t map_bytes=0,logical_bytes=0,slot_bytes=0;
  static hyle::Result<Layout> create(std::span<const c::TensorDesc> actual_planes,std::int64_t domain,std::int64_t capacity) {
    if(actual_planes.empty()||actual_planes.size()>(std::numeric_limits<std::uint32_t>::max()/2)-1||
        domain<=0||domain>std::numeric_limits<std::int32_t>::max()||capacity<=0||capacity>domain)
      return std::unexpected(hyle::Status::InvalidArgument("original plane layout needs actual domain/capacity/planes"));
    Layout r;r.domain=domain;r.capacity=capacity;r.source.assign(actual_planes.begin(),actual_planes.end());
    const auto add=[&](std::size_t size)->bool {
      if(size>std::numeric_limits<std::size_t>::max()-255)return false;
      size=(size+255)&~std::size_t{255};
      if(size>std::numeric_limits<std::size_t>::max()-r.slot_bytes)return false;
      r.slot_bytes+=size;return true;
    };
    for(const auto &d:actual_planes) {
      if(!d.valid()||!d.is_plain()||!d.is_dense()||d.rank()!=2||d.dimensions()[0]<=0||d.dimensions()[1]<=0)
        return std::unexpected(hyle::Status::InvalidArgument("original plane must keep its actual dense rank2 encoding"));
      HYLE_ASSIGN_OR_RETURN(auto bytes,d.required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto bank,c::TensorDesc::dense(d.data_type(),std::array{capacity,d.dimensions()[0],d.dimensions()[1]}));
      HYLE_ASSIGN_OR_RETURN(auto total,bank.required_bytes());
      if(!add(total)||total>std::numeric_limits<std::size_t>::max()-r.logical_bytes)
        return std::unexpected(hyle::Status::InvalidArgument("original plane layout extent overflow"));
      r.resident.push_back(bank);r.plane_bytes.push_back(bytes);r.bank_bytes.push_back(total);r.logical_bytes+=total;
    }
    if(std::uint64_t(domain)>std::numeric_limits<std::size_t>::max()/sizeof(std::int32_t))
      return std::unexpected(hyle::Status::InvalidArgument("original plane map extent overflow"));
    r.map_bytes=std::size_t(domain)*sizeof(std::int32_t);
    if(!add(r.map_bytes)||r.map_bytes>std::numeric_limits<std::size_t>::max()-r.logical_bytes)
      return std::unexpected(hyle::Status::InvalidArgument("original plane map alignment overflow"));
    r.logical_bytes+=r.map_bytes;return r;
  }
  c::TensorDesc map_description()const{return c::TensorDesc::dense(c::DataType::kInt32,std::array{domain}).value();}
  hyle::Status validate(const Selection &s)const {
    if(s.globals.empty()||s.globals.size()>std::size_t(capacity))return hyle::Status::InvalidArgument("actual plane selection exceeds resident capacity");
    std::vector<std::int32_t> map(std::size_t(domain),-1);std::int32_t old=-1;
    for(std::size_t i=0;i<s.globals.size();++i) {
      const auto id=s.globals[i];if(id<0||id>=domain||id<=old)return hyle::Status::InvalidArgument("actual plane IDs must be sorted unique/in-domain");
      map[std::size_t(id)]=std::int32_t(i);old=id;
    }
    if(map!=s.map)return hyle::Status::InvalidArgument("original plane map differs from actual IDs");return hyle::Status::Ok();
  }
  std::int32_t global(const Selection &s,std::size_t slot)const{return slot<s.globals.size()?s.globals[slot]:s.globals.front();}
};
struct CpuBatch final {c::StorageOwner owner;std::vector<c::ReadLease> banks;};
class CpuSupply final {
  Layout layout_;c::StorageOwner owner_;c::LeasePool pool_;
 public:
  static hyle::Result<CpuSupply> create(const c::StorageOwner &owner,Layout layout) {
    if(!owner||layout.slot_bytes>std::numeric_limits<std::size_t>::max()/2)
      return std::unexpected(hyle::Status::InvalidArgument("original plane supply needs bounded actual owner"));
    CpuSupply r;r.layout_=std::move(layout);r.owner_=owner;
    HYLE_ASSIGN_OR_RETURN(r.pool_,c::LeasePool::create(owner.ref(),2*r.layout_.slot_bytes,
        {.max_blocks=static_cast<std::uint32_t>(2*(r.layout_.source.size()+1)),.backing_alignment=256}));return r;
  }
  template<class Source> hyle::Result<CpuBatch> upload(const Selection &selection,const Source &source) {
    HYLE_RETURN_IF_ERROR(layout_.validate(selection));CpuBatch r{owner_,{}};
    std::vector<c::WriteLease> writers(layout_.source.size()+1);r.banks.resize(writers.size());
    for(std::size_t p=0;p<layout_.source.size();++p) {
      HYLE_ASSIGN_OR_RETURN(writers[p],pool_.allocate(layout_.bank_bytes[p]));
      HYLE_ASSIGN_OR_RETURN(auto out,writers[p].view());
      for(std::size_t slot=0;slot<std::size_t(layout_.capacity);++slot) {
        const auto bytes=source.plane_bytes(p,layout_.global(selection,slot));
        if(bytes.size()!=layout_.plane_bytes[p])return std::unexpected(hyle::Status::InvalidArgument("actual original selected plane extent differs"));
        HYLE_ASSIGN_OR_RETURN(auto row,out.subview(slot*layout_.plane_bytes[p],layout_.plane_bytes[p]));HYLE_RETURN_IF_ERROR(row.write(bytes));
      }
    }
    HYLE_ASSIGN_OR_RETURN(writers.back(),pool_.allocate(layout_.map_bytes));
    HYLE_ASSIGN_OR_RETURN(auto map,writers.back().view());HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selection.map))));
    for(std::size_t i=0;i<writers.size();++i){HYLE_ASSIGN_OR_RETURN(r.banks[i],writers[i].publish());}return r;
  }
  const Layout &layout()const noexcept{return layout_;}
  c::LeasePoolStats stats()const noexcept{return pool_.stats();}
};
#if defined(HYLE_TEST_GLM_STATE_CUDA)
class CudaSupply final {
  struct State final {
    Layout layout;c::StorageOwner owner;c::cuda::MemoryResource host;c::cuda::WeightSlots slots;m::Buffer map;
    ~State(){slots.close().IgnoreError();}
  };
  std::unique_ptr<State> state_;
 public:
  static hyle::Result<CudaSupply> create(const c::Context &context,const c::StorageOwner &owner,
      const c::cuda::MemoryResource &device,Layout layout) {
    if(!owner||owner.ref()!=device.ref()||device.kind()!=c::cuda::MemoryKind::kDevice)
      return std::unexpected(hyle::Status::InvalidArgument("original planes need the compiled device and same actual owner"));
    CudaSupply r;r.state_=std::make_unique<State>();auto &s=*r.state_;s.owner=owner;s.layout=std::move(layout);
    HYLE_ASSIGN_OR_RETURN(s.host,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
    std::vector<c::cuda::WeightSpec> catalog;
    for(std::size_t p=0;p<s.layout.resident.size();++p)catalog.push_back({p+1,s.layout.resident[p]});
    catalog.push_back({catalog.size()+1,s.layout.map_description()});
    HYLE_ASSIGN_OR_RETURN(s.slots,c::cuda::WeightSlots::create(context,s.host,device,catalog,s.layout.slot_bytes,
        {.max_fragments=s.layout.source.size()*std::size_t(s.layout.capacity)+1}));
    HYLE_ASSIGN_OR_RETURN(s.map,m::Buffer::create(m::system_resource(),s.layout.map_bytes,256));return r;
  }
  template<class Source> hyle::Result<c::cuda::WeightBatch> upload(std::size_t slot,const Selection &selection,
      const Source &source,c::Queue &queue) {
    if(!state_)return std::unexpected(hyle::Status::FailedPrecondition("empty original plane supply"));
    auto &s=*state_;HYLE_RETURN_IF_ERROR(s.layout.validate(selection));
    HYLE_RETURN_IF_ERROR(s.map.view().write(std::as_bytes(std::span(selection.map))));std::vector<c::WeightSlice> slices;
    for(std::size_t p=0;p<s.layout.source.size();++p)for(std::size_t local=0;local<std::size_t(s.layout.capacity);++local) {
      HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(s.layout.source[p],source.plane_view(p,s.layout.global(selection,local))));
      slices.push_back({p+1,tensor,local*s.layout.plane_bytes[p]});
    }
    HYLE_ASSIGN_OR_RETURN(auto map,c::ConstTensorView::create(s.layout.map_description(),s.map.view()));
    slices.push_back({s.layout.source.size()+1,map,0});return s.slots.upload_slices(slot,slices,queue);
  }
  hyle::Result<c::WeightUploadReport> report(std::size_t slot)const {
    if(!state_)return std::unexpected(hyle::Status::FailedPrecondition("empty original plane supply"));return state_->slots.report(slot);
  }
  hyle::Status close(){return state_?state_->slots.close():hyle::Status::Ok();}
  const Layout &layout()const {if(!state_)throw std::runtime_error("empty original plane supply");return state_->layout;}
};
#endif
} // namespace model_plane_resident_weights
