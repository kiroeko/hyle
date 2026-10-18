#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "ds4_roundtrip.h"

int main(int argc, char **argv) {
  @autoreleasepool {
    if(argc!=2) return 2;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
      encoding:NSUTF8StringEncoding error:&error];
    if(!source) {std::cerr<<[[error description] UTF8String]<<'\n';return 3;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) return 4;
    std::cout<<"device "<<[[device name] UTF8String]<<'\n';
    std::vector<float> input;
    for(std::uint32_t word=0;word<65536;++word)
      if((word&0x7f80)!=0x7f80) input.push_back(std::bit_cast<float>(word<<16));
    std::vector<float> expected=input;
    ds4_quantization::dsv4_fp4_act_quantize_row_inplace_cpu(expected.data(),expected.size());
    int result=0;
    for(int mode=0;mode<2;++mode) {
      MTLCompileOptions *options=[[MTLCompileOptions alloc] init];
      if(mode) options.mathMode=MTLMathModeSafe;
      id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
      if(!library) {std::cerr<<[[error description] UTF8String]<<'\n';return 5;}
      id<MTLFunction> fn=[library newFunctionWithName:@"original_ds4_fp4_pass"];
      id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
      if(!pipeline) {std::cerr<<[[error description] UTF8String]<<'\n';return 6;}
      id<MTLBuffer> in=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> out=[device newBufferWithLength:input.size()*4 options:MTLResourceStorageModeShared];
      id<MTLCommandQueue> queue=[device newCommandQueue];
      id<MTLCommandBuffer> command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:in offset:0 atIndex:0]; [encoder setBuffer:out offset:0 atIndex:1];
      [encoder setThreadgroupMemoryLength:32*4 atIndex:0];
      [encoder dispatchThreadgroups:MTLSizeMake(input.size()/32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      if(command.status!=MTLCommandBufferStatusCompleted) {std::cerr<<[[command.error description] UTF8String]<<'\n';return 7;}
      const auto *actual=static_cast<const float *>(out.contents);
      std::size_t mismatch=0, nonzero_lost=0;
      for(std::size_t i=0;i<input.size();++i) {
        if(std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(expected[i])) {
          ++mismatch; if(actual[i]==0 && expected[i]!=0) ++nonzero_lost;
          if(mismatch<=8) std::cout<<"index "<<i<<" in "<<std::hex<<std::bit_cast<std::uint32_t>(input[i])
           <<" expected "<<std::bit_cast<std::uint32_t>(expected[i])<<" actual "<<std::bit_cast<std::uint32_t>(actual[i])<<std::dec<<'\n';
        }
      }
      std::cout<<"mode "<<(mode?"safe":"default")<<" values "<<input.size()<<" mismatch "<<mismatch<<" nonzero_lost "<<nonzero_lost<<'\n';
      if(mode && mismatch) result=1;
      [queue release];[in release];[out release];[pipeline release];[fn release];[library release];[options release];
    }
    [device release]; return result;
  }
}
