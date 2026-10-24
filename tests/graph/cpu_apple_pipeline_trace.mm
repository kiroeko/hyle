// Validation only: decorate original Metal recording, preserve every original
// operation, and observe one warmed DAG at a time. This is not a P1 artifact.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <mach/mach_time.h>

#define HYLE_G09_APPLE 1
#define main hyle_g09_ordinary_apple_main
#include "native_joint_pipeline.cc"
#undef main

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>

namespace g09_metal {
struct CpuInterval { unsigned phase; std::uint64_t begin, end; };
struct GpuInterval {
  unsigned phase, flags, dispatches;
  double begin, end;
  int status;
  std::vector<std::string> kernels;
};
std::mutex monitor;
std::condition_variable settled;
std::vector<CpuInterval> cpu_intervals;
std::vector<GpuInterval> gpu_intervals;
std::atomic<bool> enabled{false};
std::atomic<unsigned> phase{0};
unsigned phase_base = 0, pending_handlers = 0;
mach_timebase_info_data_t timebase{};
char command_key, flags_key, count_key, names_key, pipeline_key, function_key;
IMP commit_impl, blit_impl, compute_impl, compute_type_impl;
IMP dispatch_groups_impl, dispatch_threads_impl, pipeline_impl;
IMP create_function_impl, create_descriptor_impl;
struct Replacement { Method method; IMP original; };
std::vector<Replacement> replacements;

std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(static_cast<long double>(mach_absolute_time()) *
      timebase.numer / timebase.denom);
}
void cpu_parallel(void *, std::size_t count, c::ParallelTaskFn task, void *context) noexcept {
  const auto observed = enabled.load(std::memory_order_acquire);
  const auto current = phase.load(std::memory_order_acquire);
  const auto begin = observed ? now_ns() : 0;
  for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  if (observed) {
    const auto end = now_ns();
    std::lock_guard lock(monitor);
    cpu_intervals.push_back({current, begin, end});
  }
}
unsigned flags(id command) { return [objc_getAssociatedObject(command, &flags_key) unsignedIntValue]; }
void attach_encoder(id encoder, id command) {
  if (encoder != nil) objc_setAssociatedObject(encoder, &command_key, command, OBJC_ASSOCIATION_ASSIGN);
  objc_setAssociatedObject(command, &flags_key, @(flags(command) | 2u), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}
id hook_compute(id command, SEL selector) {
  id encoder = reinterpret_cast<id (*)(id, SEL)>(compute_impl)(command, selector);
  attach_encoder(encoder, command); return encoder;
}
id hook_compute_type(id command, SEL selector, MTLDispatchType type) {
  id encoder = reinterpret_cast<id (*)(id, SEL, MTLDispatchType)>(compute_type_impl)(command, selector, type);
  attach_encoder(encoder, command); return encoder;
}
id hook_blit(id command, SEL selector) {
  objc_setAssociatedObject(command, &flags_key, @(flags(command) | 1u), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return reinterpret_cast<id (*)(id, SEL)>(blit_impl)(command, selector);
}
id hook_create_function(id device, SEL selector, id<MTLFunction> function, NSError **error) {
  id pipeline = reinterpret_cast<id (*)(id, SEL, id, NSError **)>(create_function_impl)(device, selector, function, error);
  if (pipeline != nil && function.name != nil)
    objc_setAssociatedObject(pipeline, &function_key, function.name, OBJC_ASSOCIATION_COPY_NONATOMIC);
  return pipeline;
}
id hook_create_descriptor(id device, SEL selector, MTLComputePipelineDescriptor *descriptor,
    MTLPipelineOption options, MTLComputePipelineReflection **reflection, NSError **error) {
  id pipeline = reinterpret_cast<id (*)(id, SEL, id, MTLPipelineOption, MTLComputePipelineReflection **, NSError **)>(
      create_descriptor_impl)(device, selector, descriptor, options, reflection, error);
  if (pipeline != nil && descriptor.computeFunction.name != nil)
    objc_setAssociatedObject(pipeline, &function_key, descriptor.computeFunction.name, OBJC_ASSOCIATION_COPY_NONATOMIC);
  return pipeline;
}
void hook_pipeline(id encoder, SEL selector, id<MTLComputePipelineState> pipeline) {
  NSString *name = objc_getAssociatedObject(pipeline, &function_key);
  // A missing original Function identity stays unknown; labels are not proof.
  objc_setAssociatedObject(encoder, &pipeline_key, name, OBJC_ASSOCIATION_COPY_NONATOMIC);
  reinterpret_cast<void (*)(id, SEL, id)>(pipeline_impl)(encoder, selector, pipeline);
}
void record_dispatch(id encoder) {
  id command = objc_getAssociatedObject(encoder, &command_key);
  const auto count = [objc_getAssociatedObject(command, &count_key) unsignedIntValue];
  objc_setAssociatedObject(command, &count_key, @(count + 1), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  NSMutableArray *names = objc_getAssociatedObject(command, &names_key);
  if (names == nil) {
    names = [NSMutableArray array];
    objc_setAssociatedObject(command, &names_key, names, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  NSString *name = objc_getAssociatedObject(encoder, &pipeline_key);
  [names addObject:name != nil ? name : @"UNKNOWN_FUNCTION"];
}
void hook_dispatch_groups(id encoder, SEL selector, MTLSize groups, MTLSize threads) {
  record_dispatch(encoder);
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(dispatch_groups_impl)(encoder, selector, groups, threads);
}
void hook_dispatch_threads(id encoder, SEL selector, MTLSize grid, MTLSize threads) {
  record_dispatch(encoder);
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(dispatch_threads_impl)(encoder, selector, grid, threads);
}
void hook_commit(id command, SEL selector) {
  if (enabled.load(std::memory_order_acquire)) {
    const auto current = phase.load(std::memory_order_acquire);
    const auto kind = flags(command);
    const auto dispatches = [objc_getAssociatedObject(command, &count_key) unsignedIntValue];
    NSArray *names = [objc_getAssociatedObject(command, &names_key) copy];
    { std::lock_guard lock(monitor); ++pending_handlers; }
    [static_cast<id<MTLCommandBuffer>>(command) addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      GpuInterval record{current, kind, dispatches, buffer.GPUStartTime, buffer.GPUEndTime, int(buffer.status), {}};
      for (NSString *name in names) record.kernels.emplace_back(name.UTF8String);
      {
        std::lock_guard lock(monitor);
        gpu_intervals.push_back(std::move(record));
        --pending_handlers;
      }
      settled.notify_all();
    }];
    [names release];
  }
  reinterpret_cast<void (*)(id, SEL)>(commit_impl)(command, selector);
}
bool replace(Class type, SEL selector, IMP hook, IMP &original) {
  const auto method = class_getInstanceMethod(type, selector);
  if (method == nullptr) return false;
  original = method_setImplementation(method, hook);
  replacements.push_back({method, original});
  return true;
}
struct Restore {
  ~Restore() {
    for (auto i = replacements.rbegin(); i != replacements.rend(); ++i)
      method_setImplementation(i->method, i->original);
  }
};
Status install() {
  if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0)
    return Status::Unavailable("mach clock timebase unavailable");
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  const auto command_class = object_getClass(command), encoder_class = object_getClass(encoder), device_class = object_getClass(device);
  [encoder endEncoding];
  const bool installed =
      replace(command_class, @selector(commit), reinterpret_cast<IMP>(&hook_commit), commit_impl) &&
      replace(command_class, @selector(blitCommandEncoder), reinterpret_cast<IMP>(&hook_blit), blit_impl) &&
      replace(command_class, @selector(computeCommandEncoder), reinterpret_cast<IMP>(&hook_compute), compute_impl) &&
      replace(command_class, @selector(computeCommandEncoderWithDispatchType:), reinterpret_cast<IMP>(&hook_compute_type), compute_type_impl) &&
      replace(encoder_class, @selector(setComputePipelineState:), reinterpret_cast<IMP>(&hook_pipeline), pipeline_impl) &&
      replace(encoder_class, @selector(dispatchThreadgroups:threadsPerThreadgroup:), reinterpret_cast<IMP>(&hook_dispatch_groups), dispatch_groups_impl) &&
      replace(encoder_class, @selector(dispatchThreads:threadsPerThreadgroup:), reinterpret_cast<IMP>(&hook_dispatch_threads), dispatch_threads_impl) &&
      replace(device_class, @selector(newComputePipelineStateWithFunction:error:), reinterpret_cast<IMP>(&hook_create_function), create_function_impl) &&
      replace(device_class, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), reinterpret_cast<IMP>(&hook_create_descriptor), create_descriptor_impl);
  [queue release]; [device release];
  return installed ? Status::Ok() : Status::Unavailable("original Metal selectors unavailable for transparent observation");
}
Status observe(unsigned step, bool before) {
  if (before) {
    phase.store(phase_base + step, std::memory_order_release);
    enabled.store(step != 0, std::memory_order_release); // Cold first replay only warms original math.
  } else {
    enabled.store(false, std::memory_order_release);
    std::unique_lock lock(monitor);
    if (!settled.wait_for(lock, std::chrono::seconds(5), [] { return pending_handlers == 0; }))
      return Status::DeadlineExceeded("original Metal completion observation did not settle");
  }
  return Status::Ok();
}
void print_string(const std::string &value) {
  std::putchar('"');
  for (unsigned char ch : value) {
    if (ch == '"' || ch == '\\') std::putchar('\\');
    if (ch < 0x20) std::printf("\\u%04x", unsigned(ch));
    else std::putchar(ch);
  }
  std::putchar('"');
}
Status verdict() {
  std::lock_guard lock(monitor);
  std::array<unsigned, 2> contained{}, pure{};
  constexpr std::uint64_t margin_ns = 1000; // Diagnostic guard, never a measured P1 confidence bound.
  for (const auto &cpu : cpu_intervals)
    std::printf("{\"cpu_task\":true,\"phase\":%u,\"begin_ns\":%llu,\"end_ns\":%llu}\n", cpu.phase,
        static_cast<unsigned long long>(cpu.begin), static_cast<unsigned long long>(cpu.end));
  for (const auto &gpu : gpu_intervals) {
    std::printf("{\"metal_buffer\":true,\"phase\":%u,\"flags\":%u,\"dispatches\":%u,\"status\":%d,\"gpu_start_seconds\":%.12f,\"gpu_end_seconds\":%.12f,\"functions\":[",
        gpu.phase, gpu.flags, gpu.dispatches, gpu.status, gpu.begin, gpu.end);
    for (std::size_t i = 0; i < gpu.kernels.size(); ++i) { if (i) std::putchar(','); print_string(gpu.kernels[i]); }
    std::puts("]}");
    const auto order = gpu.phase / 100;
    if (order >= 2 || gpu.dispatches != 1 || gpu.flags != 2 ||
        gpu.status != int(MTLCommandBufferStatusCompleted) || gpu.kernels.size() != 1 ||
        gpu.kernels[0] != "vn_Negativefloat32float32" ||
        !std::isfinite(gpu.begin) || !std::isfinite(gpu.end) || gpu.begin <= 0 || gpu.end <= gpu.begin) continue;
    ++pure[order];
    const auto begin = static_cast<std::uint64_t>(std::floor(gpu.begin * 1e9));
    const auto end = static_cast<std::uint64_t>(std::ceil(gpu.end * 1e9));
    for (const auto &cpu : cpu_intervals) {
      // The join is data-dependent on this GPU episode, so an executor task
      // containing its entire GPU interval belongs to the independent branch.
      if (cpu.phase == gpu.phase && cpu.begin + margin_ns < begin && end + margin_ns < cpu.end) {
        ++contained[order]; break;
      }
    }
  }
  std::printf("{\"single_dag\":true,\"warm_episodes_per_order\":8,\"clock\":\"system_mach_time\",\"timebase_numer\":%u,\"timebase_denom\":%u,\"guard_ns\":%llu,\"pure_negative_buffers\":[%u,%u],\"whole_gpu_inside_cpu_task\":[%u,%u]}\n",
      timebase.numer, timebase.denom, static_cast<unsigned long long>(margin_ns), pure[0], pure[1], contained[0], contained[1]);
  return require(contained[0] != 0 && contained[1] != 0,
      "both original insertion orders need an unmixed real Neg GPU interval inside the independent CPU task");
}
Status exercise() {
  Restore restore;
  cpu_intervals.reserve(256); gpu_intervals.reserve(64); replacements.reserve(16);
  HYLE_RETURN_IF_ERROR(install());
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context({.executor = c::ParallelExecutorRef{&phase, 1, &cpu_parallel}}));
  HYLE_ASSIGN_OR_RETURN(auto native, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, n::MemoryResource::create(native));
  const PipelineResources environment{cpu,native,memory};
  HYLE_RETURN_IF_ERROR(branches(environment, true, 2, true, 9, 1, &observe, true));
  phase_base = 100;
  HYLE_RETURN_IF_ERROR(branches(environment, false, 2, true, 9, 1, &observe, true));
  return verdict();
}
} // namespace g09_metal

int main(int argc, char **argv) {
  if (argc == 1) return hyle_g09_ordinary_apple_main();
  if (argc != 2 || std::string(argv[1]) != "--trace") return 2;
  @autoreleasepool {
    const auto status = g09_metal::exercise();
    if (status.ok()) return 0;
    std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
    return 1;
  }
}
