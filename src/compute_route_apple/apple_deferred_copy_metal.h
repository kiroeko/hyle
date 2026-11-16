// Private encoding metadata of the original CopyRecording. The same accepted
// copy Job, native buffer, Source pins and retirement remain authoritative.
#pragma once

#include "apple_copy_internal.h"

namespace hyle::compute::apple::detail {
namespace {
struct CopyCommand final {
  void *source = nullptr, *destination = nullptr;
  std::size_t source_offset = 0, destination_offset = 0, bytes = 0;
};

constexpr auto deferred_copy_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Copy { const device uint *source; device uint *destination; ulong words; };
struct Check { const device int *word; };
kernel void hyle_checked_native_copy(const device Copy *copies [[buffer(0)]],
    const device Check *checks [[buffer(1)]], constant uint &check_count [[buffer(2)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bool failed;
  if(tid==0) {
    failed=false;
    for(uint i=0;i<check_count;++i)if(*checks[i].word)failed=true;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // An invalid original producer must not even read its payload, nor change
  // any destination. Public failure still comes from that original episode.
  if(failed)return;
  const auto copy=copies[grid.x];
  for(ulong i=tid;i<copy.words;i+=128)copy.destination[i]=copy.source[i];
}
)METAL";

struct DeferredCopyState final {
  static constexpr std::size_t check_capacity=256;
  std::array<NativeStorage,check_capacity> words{};
  std::array<std::optional<mlx::core::array>,check_capacity> owners;
  std::array<DeviceCheckVisit,4096> visits{};
  std::size_t check_count=0;
  id<MTLBuffer> arguments=nil,checks=nil;
  ~DeferredCopyState(){[arguments release];[checks release];}

  // Shared pipeline belongs to the existing Runtime copy frontier. All
  // capacity and shader preparation occurs at binding, never submission.
  Result<bool> prepare(id<MTLDevice> device,std::size_t capacity,
      id<MTLComputePipelineState> &pipeline) noexcept {
    if(device.argumentBuffersSupport!=MTLArgumentBuffersTier2)return false;
    if(pipeline==nil){
      NSError *error=nil;
      MTLCompileOptions *options=[[MTLCompileOptions alloc] init];
      options.languageVersion=MTLLanguageVersion3_0;
      id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:deferred_copy_source]
          options:options error:&error];
      [options release];
      if(library==nil)return std::unexpected(Status::Internal(error.localizedDescription.UTF8String?:"Apple deferred copy library failed"));
      id<MTLFunction> function=[library newFunctionWithName:@"hyle_checked_native_copy"];
      if(function!=nil)pipeline=[device newComputePipelineStateWithFunction:function error:&error];
      [function release];[library release];
      if(pipeline==nil)return std::unexpected(Status::Internal(error.localizedDescription.UTF8String?:"Apple deferred copy pipeline failed"));
    }
    if(arguments==nil)arguments=[device newBufferWithLength:capacity*3*sizeof(std::uint64_t) options:MTLResourceStorageModeShared];
    if(checks==nil)checks=[device newBufferWithLength:check_capacity*sizeof(std::uint64_t) options:MTLResourceStorageModeShared];
    if(arguments==nil || checks==nil)return std::unexpected(Status::ResourceExhausted("Apple deferred copy fixed metadata"));
    return true;
  }
  void clear() noexcept {
    for(auto &owner:owners)owner.reset();
    visits.fill({});
    check_count=0;
  }
  DeviceCheckInputs inputs() noexcept {return {words,owners,check_count,visits};}
  Status encode(id<MTLCommandBuffer> command,id<MTLComputePipelineState> pipeline,
      id<MTLFence> fence,bool previous,std::span<const CopyCommand> copies) noexcept {
    auto *descriptions=static_cast<std::uint64_t *>(arguments.contents);
    auto *addresses=static_cast<std::uint64_t *>(checks.contents);
    for(std::size_t i=0;i<check_count;++i){
      const auto &word=words[i];id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)word.buffer;
      if(buffer==nil || buffer.device!=command.device || !buffer.gpuAddress ||
          word.offset_bytes>buffer.length || sizeof(std::int32_t)>buffer.length-word.offset_bytes || word.offset_bytes%4)
        return Status::InvalidArgument("Apple deferred copy check span/Device differs");
      addresses[i]=buffer.gpuAddress+word.offset_bytes;
    }
    for(std::size_t i=0;i<copies.size();++i){
      const auto &copy=copies[i];
      id<MTLBuffer> source=(__bridge id<MTLBuffer>)copy.source,destination=(__bridge id<MTLBuffer>)copy.destination;
      if(source==nil || destination==nil || source.device!=command.device || destination.device!=command.device ||
          !source.gpuAddress || !destination.gpuAddress || copy.source_offset%4 || copy.destination_offset%4 || copy.bytes%4 ||
          copy.source_offset>source.length || copy.bytes>source.length-copy.source_offset ||
          copy.destination_offset>destination.length || copy.bytes>destination.length-copy.destination_offset)
        return Status::InvalidArgument("Apple deferred copy original physical span differs");
      for(std::size_t j=0;j<check_count;++j){
        const auto &word=words[j];
        if(word.buffer==copy.destination && (word.offset_bytes<=copy.destination_offset
            ?copy.destination_offset-word.offset_bytes<sizeof(std::int32_t)
            :word.offset_bytes-copy.destination_offset<copy.bytes))
          return Status::InvalidArgument("Apple deferred copy destination aliases an original error word");
      }
      descriptions[3*i]=source.gpuAddress+copy.source_offset;
      descriptions[3*i+1]=destination.gpuAddress+copy.destination_offset;
      descriptions[3*i+2]=copy.bytes/4;
    }
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(encoder==nil)return Status::ResourceExhausted("Apple deferred copy encoder allocation failed");
    if(previous)[encoder waitForFence:fence];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:arguments offset:0 atIndex:0];[encoder setBuffer:checks offset:0 atIndex:1];
    const auto count=static_cast<std::uint32_t>(check_count);
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    for(std::size_t i=0;i<check_count;++i)[encoder useResource:(__bridge id<MTLBuffer>)words[i].buffer usage:MTLResourceUsageRead];
    for(const auto &copy:copies){
      [encoder useResource:(__bridge id<MTLBuffer>)copy.source usage:MTLResourceUsageRead];
      [encoder useResource:(__bridge id<MTLBuffer>)copy.destination usage:MTLResourceUsageWrite];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(copies.size(),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder updateFence:fence];[encoder endEncoding];
    return Status::Ok();
  }
};
} // namespace
} // namespace hyle::compute::apple::detail
