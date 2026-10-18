#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <hyle/compute/apple_weights.h>
#include <mlx/ops.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <span>
#include <vector>

namespace {
struct Record { unsigned trial, flags, dispatches; double start, end; int status; };
std::mutex mutex;
std::condition_variable committed;
std::vector<Record> records;
std::atomic<bool> enabled{false};
std::atomic<unsigned> trial{0}, compute_commits{0};
char flags_key, command_key, dispatches_key;
IMP original_commit, original_blit, original_compute, original_dispatch;
void hook_dispatch(id encoder, SEL selector, MTLSize groups, MTLSize threads) {
  id command = objc_getAssociatedObject(encoder, &command_key);
  const auto count = [objc_getAssociatedObject(command, &dispatches_key) unsignedIntValue];
  objc_setAssociatedObject(command, &dispatches_key, @(count + 1), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(original_dispatch)(encoder, selector, groups, threads);
}

unsigned flags(id command) {
  return [objc_getAssociatedObject(command, &flags_key) unsignedIntValue];
}
id hook_blit(id command, SEL selector) {
  objc_setAssociatedObject(command, &flags_key, @(flags(command) | 1u), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return reinterpret_cast<id (*)(id, SEL)>(original_blit)(command, selector);
}
id hook_compute(id command, SEL selector, MTLDispatchType type) {
  objc_setAssociatedObject(command, &flags_key, @(flags(command) | 2u), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  id encoder = reinterpret_cast<id (*)(id, SEL, MTLDispatchType)>(original_compute)(command, selector, type);
  objc_setAssociatedObject(encoder, &command_key, command, OBJC_ASSOCIATION_ASSIGN);
  return encoder;
}
void hook_commit(id command, SEL selector) {
  const auto kind = flags(command);
  const bool capture = enabled.load();
  const auto current = trial.load();
  const auto dispatches = [objc_getAssociatedObject(command, &dispatches_key) unsignedIntValue];
  if (capture) {
    [static_cast<id<MTLCommandBuffer>>(command) addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      std::lock_guard lock(mutex);
      records.push_back({current, kind, dispatches, buffer.GPUStartTime, buffer.GPUEndTime, int(buffer.status)});
    }];
  }
  reinterpret_cast<void (*)(id, SEL)>(original_commit)(command, selector);
  if (capture && (kind & 2u)) {
    { std::lock_guard lock(mutex); compute_commits.fetch_add(1); }
    committed.notify_all();
  }
}

hyle::Status install_hooks() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  Class type = object_getClass(command);
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  Class encoder_type = object_getClass(encoder);
  Method dispatch = class_getInstanceMethod(encoder_type, @selector(dispatchThreadgroups:threadsPerThreadgroup:));
  [encoder endEncoding];
  Method commit = class_getInstanceMethod(type, @selector(commit));
  Method blit = class_getInstanceMethod(type, @selector(blitCommandEncoder));
  Method compute = class_getInstanceMethod(type, @selector(computeCommandEncoderWithDispatchType:));
  if (commit == nullptr || blit == nullptr || compute == nullptr || dispatch == nullptr)
    return hyle::Status::Unimplemented("native command class lacks the declared capture selectors");
  original_commit = method_setImplementation(commit, reinterpret_cast<IMP>(&hook_commit));
  original_blit = method_setImplementation(blit, reinterpret_cast<IMP>(&hook_blit));
  original_compute = method_setImplementation(compute, reinterpret_cast<IMP>(&hook_compute));
  original_dispatch = method_setImplementation(dispatch, reinterpret_cast<IMP>(&hook_dispatch));
  [queue release];
  [device release];
  return hyle::Status::Ok();
}

namespace c = hyle::compute;
namespace a = c::apple;
namespace m = hyle::mem;
namespace mx = mlx::core;

hyle::Status run() {
  HYLE_RETURN_IF_ERROR(install_hooks());
  records.reserve(128);
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto square, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4096, 4096}));
  HYLE_ASSIGN_OR_RETURN(auto upload_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4096, 1024}));
  HYLE_ASSIGN_OR_RETURN(auto host, m::Buffer::create(m::system_resource(), 16 * 1024 * 1024));
  HYLE_RETURN_IF_ERROR(host.view().fill(std::byte{0x3c}));
  HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(upload_desc, host.view()));
  const std::array catalog{a::WeightSpec{7, upload_desc}};
  HYLE_ASSIGN_OR_RETURN(auto slots, a::WeightSlots::create(context, memory, catalog, 16 * 1024 * 1024));
  HYLE_ASSIGN_OR_RETURN(auto input, memory.allocate(square));
  HYLE_ASSIGN_OR_RETURN(auto weight, memory.allocate(square));
  HYLE_ASSIGN_OR_RETURN(auto initializer, m::Buffer::create(m::system_resource(), 64 * 1024 * 1024));
  std::vector<float> values(4096 * 4096, 1.0f / 4096);
  HYLE_RETURN_IF_ERROR(initializer.view().write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto tensor, c::TensorView::create(square, initializer.view()));
  HYLE_RETURN_IF_ERROR(memory.copy_from_host(input, tensor.as_const()));
  HYLE_RETURN_IF_ERROR(memory.copy_from_host(weight, tensor.as_const()));
  const std::array specs{*a::input_spec(square), *a::input_spec(square)};
  HYLE_ASSIGN_OR_RETURN(auto program, context.compile(
      [](const auto &inputs, mx::Stream stream) {
        auto output = inputs[0];
        output = mx::matmul(output, inputs[1], stream);
        return std::vector{output};
      }, specs));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
  const std::array inputs{input, weight};
  const std::array selection{a::WeightFragment{7, source}};
  for (unsigned index = 0; index < 2; ++index) {
    trial.store(index);
    HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
    const auto before = compute_commits.load();
    enabled.store(true);
    HYLE_ASSIGN_OR_RETURN(auto submitted, invocation.submit());
    {
      std::unique_lock lock(mutex);
      if (!committed.wait_for(lock, std::chrono::seconds(5), [&] { return compute_commits.load() > before; }))
        return hyle::Status::DeadlineExceeded("no original MLX compute commit observed");
    }
    HYLE_ASSIGN_OR_RETURN(auto batch, slots.upload(index % 2, selection));
    HYLE_RETURN_IF_ERROR(batch.wait());
    HYLE_ASSIGN_OR_RETURN(auto result, submitted.wait());
    enabled.store(false);
    HYLE_RETURN_IF_ERROR(memory.copy_to_host(result->arrays()[0], tensor));
    HYLE_RETURN_IF_ERROR(initializer.view().read(std::as_writable_bytes(std::span(values))));
    if (values.front() != 1.0f / 4096 || values.back() != 1.0f / 4096)
      return hyle::Status::Internal("original MLX repeated GEMM numeric result differs");
  }
  HYLE_RETURN_IF_ERROR(slots.close());
  std::printf("{\"trials\":2,\"copy_bytes_per_trial\":16777216,\"original_mlx_matmul_count_per_trial\":1,\"records\":[");
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto &r = records[i];
    std::printf("%s{\"trial\":%u,\"flags\":%u,\"dispatches\":%u,\"gpu_start_seconds\":%.9f,\"gpu_end_seconds\":%.9f,\"status\":%d}",
        i ? "," : "", r.trial, r.flags, r.dispatches, r.start, r.end, r.status);
  }
  std::printf("]}\n");
  return hyle::Status::Ok();
}
} // namespace

int main() {
  @autoreleasepool {
    const auto status = run();
    if (status.ok()) return 0;
    std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
    return 1;
  }
}
