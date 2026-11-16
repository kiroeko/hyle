// Private implementation included only by the original CopyRecording TU.
// It adds a native encoder mode, not another job, queue or retirement owner.
#pragma once

#include "apple_device_append_internal.h"
#include <algorithm>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace {
namespace daw = compute::detail;
constexpr auto device_append_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Geometry {
  uint page_rows,physical_pages,input_rows,instruction_capacity;
  ulong row_bytes;
  uint requests;
};
struct SourceCheck { const device int *word; };
kernel void hyle_device_lengths_append(device ushort *arena [[buffer(0)]],
    const device ushort *input [[buffer(1)]], const device uint *metadata [[buffer(2)]],
    const device int *query [[buffer(3)]], const device int *kv [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]], constant Geometry &s [[buffer(6)]],
    const device SourceCheck *checks [[buffer(7)]], constant uint &check_count [[buffer(8)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup int offsets[33],previous[32],last[32];
  threadgroup uint verdict;
  const device uint *words=metadata+4;
  if(tid==0) {
    bool valid=metadata[0]>=1+s.requests && metadata[0]<=s.instruction_capacity &&
        !metadata[1] && !metadata[2] && !metadata[3] && words[0]==s.requests &&
        words[1]<=metadata[0]-1-s.requests && words[1]<=s.physical_pages;
    for(uint i=2;i<8;++i)if(words[i])valid=false;
    for(uint i=0;i<check_count;++i)if(*checks[i].word)valid=false;
    offsets[0]=0;
    if(valid)for(uint r=0;r<s.requests;++r) {
      const device uint *b=words+8*(1+r);
      const int q=query[r],k=kv[r];const long old=long(k)-q,n=long(offsets[r])+q;
      for(uint i=4;i<8;++i)if(b[i])valid=false;
      if(q<0 || uint(q)>b[0] || k<0 || uint(k)>b[3] || old<long(b[1]) ||
          old>long(b[2]) || n<0 || n>long(s.input_rows)) {
        valid=false;offsets[r+1]=previous[r]=last[r]=0;
      } else {offsets[r+1]=int(n);previous[r]=int(old);last[r]=k;}
    }
    // Every group checks the same immutable episode and all controls before
    // any payload access. Verdict stores are identical, so a later valid
    // page cannot erase another group's invalid-control result.
    if(valid)for(uint i=0;i<words[1];++i) {
      const device uint *p=words+8*(1+s.requests+i);
      if(p[0]>=s.requests || p[2]>=s.physical_pages || !p[3] || p[3]>s.page_rows ||
          p[5]>s.page_rows || p[6] || p[7]) {valid=false;break;}
      const ulong begin=ulong(p[1])*s.page_rows,old=ulong(previous[p[0]]),end=ulong(last[p[0]]);
      const uint inherited=begin<old?uint(min(ulong(p[3]),old-begin)):0;
      const uint live=begin<end?uint(min(ulong(p[3]),end-begin)):0;
      const uint added=live>=inherited?live-inherited:0;
      const ulong source=added?ulong(offsets[p[0]])+max(begin,old)-old:0;
      if(inherited>live || (inherited && (p[4]>=s.physical_pages || inherited>p[5])) ||
          (added && (source>s.input_rows || added>s.input_rows-source))) {valid=false;break;}
    }
    verdict=valid?0:3;
    atomic_store_explicit(status,verdict,memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(verdict)return;
  const uint index=grid.x/3,phase=grid.x%3;
  if(index>=words[1])return;
  const device uint *p=words+8*(1+s.requests+index);
  const ulong begin=ulong(p[1])*s.page_rows,old=ulong(previous[p[0]]),end=ulong(last[p[0]]);
  const uint inherited=begin<old?uint(min(ulong(p[3]),old-begin)):0;
  const uint live=begin<end?uint(min(ulong(p[3]),end-begin)):0;
  const ulong source_row=live>inherited?ulong(offsets[p[0]])+max(begin,old)-old:0;
  const uint first=phase==0?0:phase==1?inherited:live;
  const uint stop=phase==0?inherited:phase==1?live:p[3];
  const ulong width=s.row_bytes/2,count=ulong(stop-first)*width;
  device ushort *destination=arena+(ulong(p[2])*s.page_rows+first)*width;
  for(ulong j=tid;j<count;j+=128) {
    if(phase==0)destination[j]=arena[ulong(p[4])*s.page_rows*width+j];
    else if(phase==1)destination[j]=input[source_row*width+j];
    else destination[j]=0;
  }
}
)METAL";
struct DeviceAppendState final {
  using Resources=std::vector<daw::PhysicalCopyResource,mem::StdAllocator<daw::PhysicalCopyResource>>;
  explicit DeviceAppendState(mem::AllocatorRef allocator):targets(mem::StdAllocator<daw::PhysicalCopyResource>(allocator)),prefixes(mem::StdAllocator<daw::PhysicalCopyResource>(allocator)){}
  daw::DeviceLengthsPagedAppendSpec geometry;
  TensorDesc rows;
  std::array<InputSpec,3> specs;
  std::array<InputBinding,3> inputs;
  std::array<std::uint64_t,3> consumed{},expected{};
  std::array<bool,3> pinned{};
  std::array<CopySpan,3> resolved;
  mem::Buffer status;
  id<MTLComputePipelineState> pipeline=nil;
  id<MTLBuffer> metadata=nil;
  CopySpan arena;
  // Bounded private error metadata, independent of model, K or row geometry.
  static constexpr std::size_t check_capacity=256;
  std::array<NativeStorage,check_capacity> check_words{};
  std::array<std::optional<mlx::core::array>,check_capacity> check_owners;
  std::uint32_t check_count=0;
  id<MTLBuffer> check_arguments=nil;
  Resources targets,prefixes;
  std::uint32_t pages=0;
  bool configured=false,bound=false;
  ~DeviceAppendState(){[check_arguments release];[metadata release];[pipeline release];}
  Status prepare(const std::shared_ptr<Runtime>&runtime,daw::DeviceLengthsPagedAppendSpec s,
      const TensorDesc &description) noexcept {
    const auto b=s.storage;
    if(!s.requests || s.requests>32)return Status::Unimplemented("Apple device append supports at most32 requests");
    if(!b.page_rows || !b.physical_pages || !b.input_rows || b.row_bytes%2 || !b.row_bytes ||
        b.instruction_capacity<=s.requests || b.instruction_capacity>UINT32_MAX/3 ||
        std::uint64_t(b.physical_pages)*b.page_rows>SIZE_MAX/b.row_bytes ||
        b.input_rows>SIZE_MAX/b.row_bytes || !description.is_dense() || !description.is_plain())
      return Status::InvalidArgument("Apple device append physical geometry differs");
    HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
    if(bytes!=std::size_t(b.input_rows)*b.row_bytes)
      return Status::InvalidArgument("Apple device append rows differ from original physical capacity");
    geometry=s;rows=description;
    HYLE_ASSIGN_OR_RETURN(specs[0],input_spec(rows));
    HYLE_ASSIGN_OR_RETURN(auto control,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{s.requests}));
    HYLE_ASSIGN_OR_RETURN(specs[1],input_spec(control));specs[2]=specs[1];
    @autoreleasepool {
      id<MTLDevice> device=(__bridge id<MTLDevice>)mlx::core::metal::device(runtime->stream->device).mtl_device();
      if(device.argumentBuffersSupport!=MTLArgumentBuffersTier2)
        return Status::Unimplemented("Apple device append requires native argument-buffer tier2 for original error words");
      NSError *error=nil;
      MTLCompileOptions *options=[[MTLCompileOptions alloc] init];options.languageVersion=MTLLanguageVersion3_0;
      id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:device_append_source] options:options error:&error];
      [options release];
      if(library==nil)return Status::Internal(error.localizedDescription.UTF8String?:"Apple device append library failed");
      id<MTLFunction> function=[library newFunctionWithName:@"hyle_device_lengths_append"];
      if(function!=nil)pipeline=[device newComputePipelineStateWithFunction:function error:&error];
      [function release];[library release];
      if(pipeline==nil)return Status::Internal(error.localizedDescription.UTF8String?:"Apple device append pipeline failed");
      const auto size=sizeof(daw::PagedAppendMetadata)+std::size_t(b.instruction_capacity)*sizeof(daw::PagedAppendInstruction);
      metadata=[device newBufferWithLength:size options:MTLResourceStorageModeShared];
      if(metadata==nil)return Status::ResourceExhausted("Apple device append capacity metadata");
      check_arguments=[device newBufferWithLength:check_capacity*sizeof(std::uint64_t) options:MTLResourceStorageModeShared];
      if(check_arguments==nil)return Status::ResourceExhausted("Apple device append source-check capacity");
    }
    return Status::Ok();
  }
  Status configure(std::uintptr_t device,mem::ConstBufferView capacity,
      std::span<const daw::DeviceAppendRequest> requests,std::span<const DeviceAppendPageBinding> bindings) noexcept {
    configured=bound=false;
    const auto s=geometry.storage;
    if(requests.size()!=geometry.requests || bindings.size()>s.physical_pages ||
        1+requests.size()+bindings.size()>s.instruction_capacity ||
        capacity.size()!=std::size_t(s.physical_pages)*s.page_rows*s.row_bytes)
      return Status::InvalidArgument("Apple device append original capacity/records differ");
    HYLE_ASSIGN_OR_RETURN(auto bank,copy_span(capacity,mem::HostAccess::kRead));
    if(bank.physical.address_space!=device)return Status::InvalidArgument("Apple device append arena Device differs");
    std::uint64_t queries=0;
    for(const auto &r:requests){
      for(auto reserved:r.reserved)if(reserved)return Status::InvalidArgument("Apple device append reserved request fields");
      queries+=r.query_rows;
      if(queries>s.input_rows || r.minimum_previous_rows>r.maximum_previous_rows ||
          r.maximum_previous_rows>std::uint64_t(s.physical_pages)*s.page_rows ||
          r.minimum_previous_rows>r.maximum_kv_rows || r.maximum_kv_rows>std::uint64_t(s.physical_pages)*s.page_rows)
        return Status::InvalidArgument("Apple device append original request bounds differ");
    }
    try {
      Resources new_targets(targets.get_allocator()),new_prefixes(prefixes.get_allocator());
      new_targets.reserve(bindings.size());new_prefixes.reserve(bindings.size());
      for(const auto &binding:bindings){
        const auto p=binding.page;
        if(p.request>=requests.size() || p.destination_page>=s.physical_pages || !p.initialized_rows ||
            p.initialized_rows>s.page_rows || p.source_rows>s.page_rows || p.reserved[0] || p.reserved[1])
          return Status::InvalidArgument("Apple device append original page record differs");
        const auto begin=std::uint64_t(p.logical_page)*s.page_rows;
        const auto &r=requests[p.request];
        const auto inherited=begin<r.maximum_previous_rows?std::min<std::uint64_t>(s.page_rows,r.maximum_previous_rows-begin):0;
        if(begin>=r.maximum_kv_rows || p.initialized_rows!=std::min<std::uint64_t>(s.page_rows,r.maximum_kv_rows-begin) ||
            p.source_rows!=inherited || binding.destination.size()!=std::size_t(p.initialized_rows)*s.row_bytes)
          return Status::InvalidArgument("Apple device append page does not match its original request envelope");
        HYLE_ASSIGN_OR_RETURN(auto target,copy_span(binding.destination,mem::HostAccess::kWrite));
        if(target.physical.address_space!=device || target.buffer!=bank.buffer ||
            target.physical.origin!=bank.physical.origin+std::size_t(p.destination_page)*s.page_rows*s.row_bytes ||
            target.physical.origin%2)
          return Status::InvalidArgument("Apple device append destination differs from exact original page authority");
        for(const auto &other:new_targets)if(other.origin==target.physical.origin)
          return Status::InvalidArgument("Apple device append destination page repeats");
        new_targets.push_back(target.physical);
        if(inherited){
          if(p.source_page>=s.physical_pages || binding.prefix.size()!=inherited*s.row_bytes)
            return Status::InvalidArgument("Apple device append inherited prefix differs");
          HYLE_ASSIGN_OR_RETURN(auto prefix,copy_span(binding.prefix,mem::HostAccess::kRead));
          if(prefix.physical.address_space!=device || prefix.buffer!=bank.buffer ||
              prefix.physical.origin!=bank.physical.origin+std::size_t(p.source_page)*s.page_rows*s.row_bytes || prefix.physical.origin%2)
            return Status::InvalidArgument("Apple device append source differs from exact original prefix authority");
          new_prefixes.push_back(prefix.physical);
        }else if(binding.prefix.valid() && binding.prefix.size())
          return Status::InvalidArgument("Apple device append unused inherited permission");
      }
      for(const auto &target:new_targets)for(const auto &prefix:new_prefixes)
        if(target.origin<=prefix.origin?prefix.origin-target.origin<target.capacity:target.origin-prefix.origin<prefix.capacity)
          return Status::InvalidArgument("Apple device append requires fresh disjoint COW pages");
      if(!status.valid()){
        HYLE_ASSIGN_OR_RETURN(status,mem::Buffer::create(capacity.resource(),sizeof(std::uint32_t),4));
        HYLE_RETURN_IF_ERROR(status.fill(0,sizeof(std::uint32_t),std::byte{0}));
      }
      auto *bytes=static_cast<std::byte *>(metadata.contents);
      const daw::PagedAppendMetadata header{static_cast<std::uint32_t>(1+requests.size()+bindings.size())};
      const daw::DeviceAppendDirectory directory{geometry.requests,static_cast<std::uint32_t>(bindings.size())};
      std::memcpy(bytes,&header,sizeof(header));bytes+=sizeof(header);
      std::memcpy(bytes,&directory,sizeof(directory));bytes+=sizeof(directory);
      std::memcpy(bytes,requests.data(),requests.size_bytes());bytes+=requests.size_bytes();
      for(const auto &binding:bindings){std::memcpy(bytes,&binding.page,sizeof(binding.page));bytes+=sizeof(binding.page);}
      targets=std::move(new_targets);prefixes=std::move(new_prefixes);arena=bank;
      pages=static_cast<std::uint32_t>(bindings.size());configured=true;
      return Status::Ok();
    }catch(...){return native_exception();}
  }
  Status bind(const std::shared_ptr<Runtime>&runtime,std::span<const InputBinding> values) noexcept {
    bound=false;
    if(!configured || values.size()!=3)return Status::InvalidArgument("Apple device append needs its three original inputs");
    try{
      std::array<std::uint64_t,3> epochs{};
      for(std::size_t i=0;i<3;++i){
        if(values[i].source.valid()==values[i].value.has_value())return Status::InvalidArgument("Apple device append needs exactly one actual carrier per input");
        if(values[i].source.valid()){
          HYLE_ASSIGN_OR_RETURN(epochs[i],InvocationAccess::source_epoch(values[i].source,runtime));
        }else{
          const auto &value=*values[i].value;
          if(value.status()!=mlx::core::array::available || value.shape()!=specs[i].shape || value.dtype()!=specs[i].dtype || !value.flags().row_contiguous)
            return Status::InvalidArgument("Apple device append fixed input is not its exact ready native value");
        }
      }
      for(std::size_t i=0;i<3;++i)inputs[i]=values[i];
      consumed=epochs;bound=true;return Status::Ok();
    }catch(...){return native_exception();}
  }
  void release() noexcept {
    for(std::size_t i=0;i<3;++i)if(pinned[i]){InvocationAccess::release_source(inputs[i].source);pinned[i]=false;}
    for(auto &owner:check_owners)owner.reset();check_count=0;
  }
  Status pin(const std::shared_ptr<Runtime>&runtime) noexcept {
    if(!bound)return Status::FailedPrecondition("Apple device append inputs are not bound");
    for(std::size_t i=0;i<3;++i)if(inputs[i].source.valid()){
      auto epoch=InvocationAccess::pin_source(inputs[i].source,runtime,consumed[i]);
      if(!epoch){release();return epoch.error();}expected[i]=*epoch;pinned[i]=true;
    }
    return Status::Ok();
  }
  void accepted() noexcept {for(std::size_t i=0;i<3;++i)if(pinned[i])consumed[i]=expected[i];}
  Status resolve(const std::shared_ptr<Runtime>&runtime,NativeDependencies &dependencies) noexcept {
    DeviceCheckInputs checks{check_words,check_owners};
    HYLE_RETURN_IF_ERROR(dependencies.device_checks(checks));
    for(std::size_t i=0;i<3;++i){
      std::optional<mlx::core::array> value;
      if(inputs[i].source.valid()){
        if(!pinned[i])return Status::FailedPrecondition("Apple device append source is not pinned");
        HYLE_ASSIGN_OR_RETURN(value,InvocationAccess::device_source_output(inputs[i].source,runtime,expected[i],checks));
      }else value=inputs[i].value;
      if(!value || value->shape()!=specs[i].shape || value->dtype()!=specs[i].dtype || !value->flags().row_contiguous)
        return Status::InvalidArgument("Apple device append actual native input geometry changed");
      HYLE_ASSIGN_OR_RETURN(auto storage,MemoryAccess::submitted_storage_info(*value));
      if(storage.view_bytes!=value->nbytes())return Status::InvalidArgument("Apple device append actual native span changed");
      id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)storage.buffer;
      if(buffer==nil || buffer.storageMode!=MTLStorageModeShared ||
          reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device)!=arena.physical.address_space ||
          storage.offset_bytes>buffer.length || storage.view_bytes>buffer.length-storage.offset_bytes ||
          storage.offset_bytes%(i?4:2))return Status::InvalidArgument("Apple device append input Device/alignment/span differs");
      const daw::PhysicalCopyResource physical{arena.physical.address_space,reinterpret_cast<std::uintptr_t>(storage.buffer),storage.offset_bytes,storage.view_bytes, i?4u:2u};
      if(physical.backing==arena.physical.backing)for(const auto &target:targets)
        if(physical.origin<=target.origin?target.origin-physical.origin<physical.capacity:physical.origin-target.origin<target.capacity)
          return Status::InvalidArgument("Apple device append input overlaps a dirty destination");
      resolved[i]={const_cast<void *>(storage.buffer),physical};
    }
    auto *addresses=static_cast<std::uint64_t *>(check_arguments.contents);
    for(std::size_t i=0;i<checks.count;++i){
      const auto &word=check_words[i];id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)word.buffer;
      if(buffer==nil || reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device)!=arena.physical.address_space ||
          buffer.storageMode!=MTLStorageModeShared || word.offset_bytes>buffer.length ||
          sizeof(std::int32_t)>buffer.length-word.offset_bytes || !buffer.gpuAddress)
        return Status::InvalidArgument("Apple source check Device/backing differs from its original native buffer");
      if(word.buffer==arena.buffer)for(const auto &target:targets)
        if(word.offset_bytes<=target.origin?target.origin-word.offset_bytes<sizeof(std::int32_t):
            word.offset_bytes-target.origin<target.capacity)
          return Status::InvalidArgument("Apple source check overlaps an original dirty COW destination");
      addresses[i]=buffer.gpuAddress+word.offset_bytes;
    }
    check_count=static_cast<std::uint32_t>(checks.count);
    return Status::Ok();
  }
  Status encode(id<MTLCommandBuffer> command,id<MTLFence> frontier,bool wait) noexcept {
    HYLE_ASSIGN_OR_RETURN(auto result,copy_span(status.view(),mem::HostAccess::kWrite));
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(encoder==nil)return Status::ResourceExhausted("Apple device append encoder");
    if(wait)[encoder waitForFence:frontier];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)arena.buffer offset:arena.physical.origin atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)resolved[0].buffer offset:resolved[0].physical.origin atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    for(std::size_t i=1;i<3;++i)[encoder setBuffer:(__bridge id<MTLBuffer>)resolved[i].buffer offset:resolved[i].physical.origin atIndex:i+2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)result.buffer offset:result.physical.origin atIndex:5];
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:6];
    [encoder setBuffer:check_arguments offset:0 atIndex:7];
    [encoder setBytes:&check_count length:sizeof(check_count) atIndex:8];
    for(std::size_t i=0;i<check_count;++i)
      [encoder useResource:(__bridge id<MTLBuffer>)check_words[i].buffer usage:MTLResourceUsageRead];
    [encoder dispatchThreadgroups:MTLSizeMake(std::max(1u,pages)*3u,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder updateFence:frontier];[encoder endEncoding];return Status::Ok();
  }
  Status observe() noexcept {
    std::uint32_t verdict=0;
    HYLE_RETURN_IF_ERROR(status.read(0,std::as_writable_bytes(std::span(&verdict,1))));
    return verdict?Status::InvalidArgument("Apple device append lengths exceed the original COW envelope"):Status::Ok();
  }
  Result<ConstTensorView> status_view() noexcept {
    if(!configured)return std::unexpected(Status::FailedPrecondition("Apple device append status has no original episode"));
    HYLE_ASSIGN_OR_RETURN(auto description,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    return ConstTensorView::create(description,status.view());
  }
};
} // namespace
} // namespace hyle::compute::apple::detail
