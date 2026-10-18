#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mlx/allocator.h>
#include <mlx/array.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <optional>

int main() {
  @autoreleasepool {
    namespace mx = mlx::core;
    constexpr std::size_t bytes = 4096;
    auto allocation = mx::allocator::malloc(bytes);
    if (allocation.ptr() == nullptr) return 1;
    id<MTLBuffer> target = (__bridge id<MTLBuffer>)allocation.ptr();
    if (target.storageMode != MTLStorageModeShared || target.length < bytes) return 2;
    id<MTLDevice> device = target.device;
    id<MTLBuffer> source = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLSharedEvent> event = [device newSharedEvent];
    if (!source || !queue || !event) return 3;
    std::array<unsigned char, bytes> expected;
    for (std::size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>(i);
    std::memcpy(source.contents, expected.data(), bytes);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:target destinationOffset:0 size:bytes];
    [blit endEncoding];
    [command encodeSignalEvent:event value:1];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted || event.signaledValue != 1) return 4;
    int frees = 0;
    std::optional<mx::array> retained;
    {
      mx::array parent(mx::Shape{static_cast<int>(bytes)}, mx::uint8, nullptr, {});
      parent.set_data(allocation, [&frees](mx::allocator::Buffer buffer) {
        mx::allocator::free(buffer);
        ++frees;
      });
      allocation = mx::allocator::Buffer(nullptr);
      parent.set_status(mx::array::available);
      retained.emplace(parent);
    }
    if (frees != 0 || std::memcmp(retained->data<unsigned char>(), expected.data(), bytes) != 0) return 5;
    retained.reset();
    if (frees != 1) return 6;
    std::printf("{\"copy_bytes\":4096,\"shared_buffer\":true,\"metal_blit_completed\":true,"
                "\"shared_event_signaled\":true,\"native_owner_retained\":true,"
                "\"last_array_free_count\":1,\"gpu_start_seconds\":%.9f,\"gpu_end_seconds\":%.9f,"
                "\"scope\":\"buffer/copy/owner probe only; no mixed math or overlap claim\"}\n",
                command.GPUStartTime, command.GPUEndTime);
  }
}
