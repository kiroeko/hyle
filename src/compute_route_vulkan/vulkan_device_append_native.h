// Included only by the original CopyRecording TU. This is a native encoding
// and its cold descriptors, never another Run, executor or retirement owner.
#pragma once

#include "vulkan_device_append_internal.h"
#include <algorithm>
#include <cstring>

namespace hyle::compute::vulkan::detail {
namespace shaders {
std::span<const std::uint32_t> device_lengths_append() noexcept;
std::span<const std::uint32_t> device_lengths_append_raw16() noexcept;
}
namespace {
namespace daw=compute::detail;
struct DeviceAppendState final {
  daw::DeviceLengthsPagedAppendSpec geometry;
  TensorDesc rows;
  InvocationAllocation allocation;
  mem::AllocatorRef allocator;
  std::array<std::array<std::shared_ptr<Pipeline>,3>,2> pipelines;
  std::shared_ptr<Buffer> metadata,checked,device_status,arena;
  mem::Buffer status_export;
  std::vector<daw::PhysicalCopyResource> targets,prefixes;
  std::vector<PlanValue> values;
  std::vector<Dispatch> dispatches;
  std::vector<std::shared_ptr<Buffer>> owners;
  std::uint32_t page_capacity=0,pages=0;
  std::size_t previous_checks=SIZE_MAX;
  std::uint32_t previous_unit=0,previous_pages=0;
  bool configured=false,bound=false;

  Status prepare(const std::shared_ptr<Runtime> &runtime, std::uint32_t capacity,
      daw::DeviceLengthsPagedAppendSpec spec,const TensorDesc &description,
      mem::AllocatorRef a,InvocationAllocation selected) noexcept {
    const auto s=spec.storage;
    if (!spec.requests || !s.page_rows || !s.physical_pages || !s.input_rows ||
        !s.row_bytes || s.row_bytes%2 || s.instruction_capacity<=spec.requests ||
        spec.requests>INT32_MAX || s.instruction_capacity>UINT32_MAX/32 ||
        std::uint64_t(s.physical_pages)*s.page_rows>UINT32_MAX ||
        std::uint64_t(s.physical_pages)*s.page_rows>UINT32_MAX/(s.row_bytes/2) ||
        std::uint64_t(s.input_rows)>UINT32_MAX/(s.row_bytes/2))
      return Status::InvalidArgument("Vulkan device append needs bounded whole-halfword physical rows");
    HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
    if (bytes!=std::size_t(s.input_rows)*s.row_bytes)
      return Status::InvalidArgument("Vulkan device append row carrier differs from its exact physical extent");
    const auto &limits=runtime->device->properties.limits;
    const auto bank_bytes=std::uint64_t(s.physical_pages)*s.page_rows*s.row_bytes;
    const auto metadata_bytes=sizeof(daw::PagedAppendMetadata)+std::size_t(s.instruction_capacity)*sizeof(daw::PagedAppendInstruction);
    page_capacity=std::min({capacity,s.physical_pages,s.instruction_capacity-spec.requests-1});
    if (!page_capacity || page_capacity>limits.maxComputeWorkGroupCount[0] ||
        bank_bytes>limits.maxStorageBufferRange || bytes>limits.maxStorageBufferRange ||
        metadata_bytes>limits.maxStorageBufferRange || limits.maxComputeWorkGroupInvocations<64 ||
        limits.maxComputeWorkGroupSize[0]<64 || limits.maxPushConstantsSize<32 ||
        limits.maxDescriptorSetStorageBuffers<8 || limits.maxPerStageDescriptorStorageBuffers<8)
      return Status::Unimplemented("Vulkan device append descriptor/dispatch/buffer limits");
    if (selected.resource_budget) {
      HYLE_RETURN_IF_ERROR(MemoryAccess::validate_resource(runtime,selected.device));
      HYLE_RETURN_IF_ERROR(MemoryAccess::validate_resource(runtime,selected.staging));
      if (selected.device.properties().has(mem::MemoryProperty::kHostAccessible) ||
          !selected.staging.properties().has(mem::MemoryProperty::kHostAccessible))
        return Status::InvalidArgument("Vulkan device append budget needs its device/staging resources");
    }
    geometry=spec;rows=description;allocation=std::move(selected);allocator=a;
    HYLE_ASSIGN_OR_RETURN(metadata,allocate_buffer(runtime->device,metadata_bytes,true,true,
        {.budget=allocation.resource_budget,.resource=allocation.staging,.allocator=a}));
    HYLE_ASSIGN_OR_RETURN(checked,allocate_buffer(runtime->device,4,true,true,
        {.budget=allocation.resource_budget,.resource=allocation.staging,.allocator=a}));
    HYLE_ASSIGN_OR_RETURN(device_status,allocate_buffer(runtime->device,4,false,true,
        {.budget=allocation.resource_budget,.resource=allocation.device,.allocator=a}));
    for (std::uint32_t kind=0;kind<2;++kind) {
      if ((!kind && s.row_bytes%4) || (kind && !runtime->device->provider.native().storage_buffer16)) continue;
      const auto unit=kind?2U:4U;
      for (std::uint32_t phase=0;phase<3;++phase) {
        const std::array<SpecializationConstant,8> constants{{{0,phase},{1,spec.requests},{2,s.physical_pages},
            {3,s.page_rows},{4,s.input_rows},{5,s.instruction_capacity},
            {6,static_cast<std::uint32_t>(s.row_bytes/unit)},{7,page_capacity}}};
        const SpirvDescription shader{kind?shaders::device_lengths_append_raw16():shaders::device_lengths_append(),8,32,constants};
        HYLE_ASSIGN_OR_RETURN(pipelines[kind][phase],kind?create_halfword_pipeline(runtime->device,shader):create_pipeline(runtime->device,shader));
      }
    }
    if (!pipelines[0][0] && !pipelines[1][0])
      return Status::Unimplemented("Vulkan exact device append needs enabled storage16 for its halfword rows");
    return Status::Ok();
  }
  static bool same_backing(const daw::PhysicalCopyResource &a,const daw::PhysicalCopyResource &b) noexcept {
    return a.address_space==b.address_space && a.backing==b.backing;
  }
  static bool overlap(const daw::PhysicalCopyResource &a,const daw::PhysicalCopyResource &b) noexcept {
    return same_backing(a,b) && (a.origin<=b.origin ? b.origin-a.origin<a.capacity : a.origin-b.origin<b.capacity);
  }
  Status configure(const std::shared_ptr<Runtime> &runtime,mem::ConstBufferView capacity,
      std::span<const daw::DeviceAppendRequest> requests,std::span<const DeviceAppendPageBinding> bindings) {
    configured=bound=false;
    const auto s=geometry.storage;
    if (requests.size()!=geometry.requests || bindings.size()>page_capacity ||
        capacity.size()!=std::size_t(s.physical_pages)*s.page_rows*s.row_bytes)
      return Status::InvalidArgument("Vulkan device append original arena/request/page capacity differs");
    HYLE_ASSIGN_OR_RETURN(auto bank,MemoryAccess::copy_span(runtime,capacity));
    if (bank.backing->mapped || bank.offset%2)
      return Status::InvalidArgument("Vulkan device append requires exact device halfword arena storage");
    std::uint64_t queries=0;
    for (const auto &r:requests) {
      for (auto reserved:r.reserved) if(reserved) return Status::InvalidArgument("Vulkan device append reserved request field");
      queries+=r.query_rows;
      if (queries>s.input_rows || r.minimum_previous_rows>r.maximum_previous_rows ||
          r.maximum_previous_rows>std::uint64_t(s.physical_pages)*s.page_rows ||
          r.minimum_previous_rows>r.maximum_kv_rows || r.maximum_kv_rows>std::uint64_t(s.physical_pages)*s.page_rows)
        return Status::InvalidArgument("Vulkan device append original request envelope differs");
    }
    std::vector<daw::PhysicalCopyResource> next_targets,next_prefixes;
    for (const auto &binding:bindings) {
      const auto p=binding.page;
      if (p.request>=requests.size() || p.destination_page>=s.physical_pages || !p.initialized_rows ||
          p.initialized_rows>s.page_rows || p.source_rows>s.page_rows || p.reserved[0] || p.reserved[1])
        return Status::InvalidArgument("Vulkan device append original page record differs");
      const auto begin=std::uint64_t(p.logical_page)*s.page_rows;
      const auto &r=requests[p.request];
      const auto inherited=begin<r.maximum_previous_rows ? std::min<std::uint64_t>(s.page_rows,r.maximum_previous_rows-begin):0;
      if (begin>=r.maximum_kv_rows || p.initialized_rows!=std::min<std::uint64_t>(s.page_rows,r.maximum_kv_rows-begin) ||
          p.source_rows!=inherited || binding.destination.size()!=std::size_t(p.initialized_rows)*s.row_bytes)
        return Status::InvalidArgument("Vulkan device append page differs from its original request envelope");
      HYLE_ASSIGN_OR_RETURN(auto target,MemoryAccess::copy_span(runtime,binding.destination));
      if (target.backing->mapped || !same_backing(target.physical,bank.physical) ||
          target.backing->handle!=bank.backing->handle ||
          target.physical.origin!=bank.physical.origin+std::size_t(p.destination_page)*s.page_rows*s.row_bytes)
        return Status::InvalidArgument("Vulkan device append destination lacks its exact original page authority");
      for (const auto &other:next_targets) if(overlap(target.physical,other))
        return Status::InvalidArgument("Vulkan device append target pages overlap");
      next_targets.push_back(target.physical);
      if (inherited) {
        if (p.source_page>=s.physical_pages || binding.prefix.size()!=inherited*s.row_bytes)
          return Status::InvalidArgument("Vulkan device append inherited prefix differs");
        HYLE_ASSIGN_OR_RETURN(auto prefix,MemoryAccess::copy_span(runtime,binding.prefix));
        if (!same_backing(prefix.physical,bank.physical) || prefix.backing->handle!=bank.backing->handle ||
            prefix.physical.origin!=bank.physical.origin+std::size_t(p.source_page)*s.page_rows*s.row_bytes)
          return Status::InvalidArgument("Vulkan device append prefix lacks its exact original authority");
        next_prefixes.push_back(prefix.physical);
      } else if (binding.prefix.valid() && binding.prefix.size())
        return Status::InvalidArgument("Vulkan device append unused inherited permission");
    }
    for (const auto &target:next_targets) for (const auto &prefix:next_prefixes) if(overlap(target,prefix))
      return Status::InvalidArgument("Vulkan device append requires fresh disjoint COW pages");
    auto native=std::allocate_shared<Buffer>(mem::StdAllocator<Buffer>(allocator));
    native->device=runtime->device;
    bind_view_buffer(*native,bank.backing,bank.offset-bank.backing->offset);
    if (!status_export.valid()) {
      HYLE_ASSIGN_OR_RETURN(auto status,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
      HYLE_ASSIGN_OR_RETURN(status_export,MemoryAccess::export_storage(runtime,capacity.resource(),Access::tensor(status,device_status),allocator));
    }
    std::memset(metadata->mapped,0,static_cast<std::size_t>(metadata->bytes));
    auto *bytes=static_cast<std::byte *>(metadata->mapped);
    const daw::PagedAppendMetadata header{static_cast<std::uint32_t>(1+requests.size()+bindings.size())};
    const daw::DeviceAppendDirectory directory{geometry.requests,static_cast<std::uint32_t>(bindings.size())};
    std::memcpy(bytes,&header,sizeof(header));bytes+=sizeof(header);
    std::memcpy(bytes,&directory,sizeof(directory));bytes+=sizeof(directory);
    std::memcpy(bytes,requests.data(),requests.size_bytes());bytes+=requests.size_bytes();
    for (const auto &binding:bindings) { std::memcpy(bytes,&binding.page,sizeof(binding.page));bytes+=sizeof(binding.page); }
    if (!(metadata->memory_flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,nullptr,metadata->memory,0,VK_WHOLE_SIZE};
      HYLE_RETURN_IF_ERROR(native_status(runtime->device->fn.FlushMappedMemoryRanges(runtime->device->native.device,1,&range),"vkFlushMappedMemoryRanges(device append metadata)"));
    }
    arena=std::move(native);targets=std::move(next_targets);prefixes=std::move(next_prefixes);
    pages=static_cast<std::uint32_t>(bindings.size());configured=true;
    return Status::Ok();
  }
  static void discard_descriptors(Run &run) noexcept {
    auto &device=*run.runtime->device;
    if (run.descriptor_pool) device.fn.DestroyDescriptorPool(device.native.device,run.descriptor_pool,nullptr);
    run.descriptor_pool=VK_NULL_HANDLE;
    if (run.descriptor_count) device.descriptors_in_pools.fetch_sub(run.descriptor_count,std::memory_order_relaxed);
    run.descriptor_count=0;run.descriptor_sets.clear();run.recorded_origins.clear();run.encoding_valid=false;
  }
  Status bind(const std::shared_ptr<Runtime> &runtime,const std::shared_ptr<Run> &original,
      std::span<const Tensor> inputs,std::span<const Tensor> checks) {
    auto &run=*original;
    bound=false;
    if (!configured || inputs.size()!=3) return Status::FailedPrecondition("Vulkan device append has no exact configured episode");
    if (checks.size()>UINT32_MAX-7) return Status::ResourceExhausted("Vulkan device append source check count");
    HYLE_ASSIGN_OR_RETURN(auto control,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{geometry.requests}));
    const std::array descriptions{rows,control,control};
    std::uint32_t unit=geometry.storage.row_bytes%4 || arena->offset%4 ? 2U:4U;
    for (std::size_t i=0;i<3+checks.size();++i) {
      const auto &input=i<3?inputs[i]:checks[i-3];
      if (!input.valid() || Access::buffer(input)->device!=runtime->device)
        return Status::InvalidArgument("Vulkan device append input belongs to a foreign device");
      if ((i<3 && input.description()!=descriptions[i]) ||
          (i>=3 && (input.description().data_type()!=DataType::kInt32 || input.description().element_count()!=1U)))
        return Status::InvalidArgument("Vulkan device append input description differs");
      const auto &buffer=Access::buffer(input);
      if (buffer->offset%2 || (i>0 && buffer->offset%4))
        return Status::Unimplemented("Vulkan device append input origin is outside its exact word domain");
      if (!i && buffer->offset%4) unit=2;
      // Native views retain original Buffer owners; this is only an alias
      // check. It never manufactures a ready lease or page permission.
      // Physical memory identity is checked below through the original arena
      // handle as well; independently allocated input buffers may share memory.
      for (const auto &target:targets) {
        const auto bytes=*input.description().required_bytes();
        if (buffer->handle==arena->handle &&
            (buffer->offset<=target.origin ? target.origin-buffer->offset<bytes : buffer->offset-target.origin<target.capacity))
          return Status::InvalidArgument("Vulkan device append input overlaps a dirty COW target");
      }
    }
    const auto kind=unit==2?1U:0U;
    if (!pipelines[kind][0]) return Status::Unimplemented("Vulkan exact append input requires enabled storage16");
    HYLE_ASSIGN_OR_RETURN(auto bank,TensorDesc::dense(DataType::kUInt8,std::array<std::int64_t,1>{
        static_cast<std::int64_t>(std::uint64_t(geometry.storage.physical_pages)*geometry.storage.page_rows*geometry.storage.row_bytes)}));
    HYLE_ASSIGN_OR_RETURN(auto metadata_description,TensorDesc::dense(DataType::kUInt32,
        std::array<std::int64_t,1>{static_cast<std::int64_t>(metadata->bytes/4)}));
    HYLE_ASSIGN_OR_RETURN(auto status,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    values={{rows,{}},{bank,{}},{status,{}},{metadata_description,{}},{control,{}},{control,{}},{status,{}}};
    owners={Access::buffer(inputs[0]),arena,checked,metadata,Access::buffer(inputs[1]),Access::buffer(inputs[2]),device_status};
    for (const auto &check:checks) { values.push_back({check.description(),{}});owners.push_back(Access::buffer(check)); }
    dispatches.clear();dispatches.reserve(checks.size()+2);
    const auto add=[&](std::uint32_t phase,std::uint32_t source) {
      Dispatch dispatch;dispatch.pipeline=pipelines[kind][phase];dispatch.values={0,1,2,3,4,5,6,source};
      dispatch.push.resize(32);dispatch.tensor_byte_origins=true;dispatch.checked_status=true;
      dispatch.groups={phase==2?std::max(1U,pages):1U,1,1};dispatches.push_back(std::move(dispatch));
    };
    add(0,2);for (std::uint32_t i=0;i<checks.size();++i) add(1,7+i);add(2,2);
    run.retained_buffers.assign(owners.begin(),owners.end());run.transfer_submission=false;
    if (previous_checks!=checks.size() || previous_unit!=unit || !run.descriptor_pool) {
      discard_descriptors(run);
      HYLE_RETURN_IF_ERROR(record(original,values,dispatches));
    } else {
      if (previous_pages!=pages) run.encoding_valid=false;
      HYLE_RETURN_IF_ERROR(rebind_descriptors(run,values,dispatches));
    }
    previous_checks=checks.size();previous_unit=unit;previous_pages=pages;bound=true;
    return Status::Ok();
  }
  void clear() noexcept { bound=false;owners.clear(); }
  Result<ConstTensorView> status_view() noexcept {
    if (!configured || !status_export.valid()) return std::unexpected(Status::FailedPrecondition("Vulkan device append has no original verdict"));
    HYLE_ASSIGN_OR_RETURN(auto description,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    return ConstTensorView::create(description,status_export.view());
  }
};
} // namespace
} // namespace hyle::compute::vulkan::detail
