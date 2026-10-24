#pragma once

#include "hyle/graph/executable.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/result.h"
#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/graph/vulkan.h"
#elif defined(HYLE_COMPUTE_APPLE)
#include "hyle/graph/apple.h"
#endif
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"

#include <array>
#include <vector>

// Cold test assembly only: no invocation, submission or result wrapper.
namespace effect_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
#if defined(HYLE_COMPUTE_VULKAN)
namespace n = c::vulkan;
namespace ng = g::vulkan;
#elif defined(HYLE_COMPUTE_APPLE)
namespace n = c::apple;
namespace ng = g::apple;
#endif
inline constexpr g::ExecutionTargetId kNativeTarget{0}, kCpuTarget{1};
inline constexpr g::StorageDomainId kHostStorage{0}, kDeviceStorage{1};

inline hyle::Result<g::CompiledGraph> compile(const g::Graph &graph,
    const c::Context &cpu, const n::Context &native,
    hyle::mem::MemoryResourceRef device, hyle::mem::MemoryResourceRef staging,
    g::CompileOptions options = {}) {
  HYLE_ASSIGN_OR_RETURN(auto host_source, g::ExecutionSource::acquire(cpu));
  HYLE_ASSIGN_OR_RETURN(auto native_source, ng::execution_source(native));
  const std::array targets{g::ExecutionTarget{native_source, kDeviceStorage},
      g::ExecutionTarget{host_source, kHostStorage}};
  std::vector<g::StorageDomain> domains{{hyle::mem::system_resource()}, {device}};
  if (staging.valid() && staging != device) domains.push_back({staging});
  return g::compile(graph, {.storage_domains = domains, .targets = targets}, options);
}

inline hyle::Result<std::vector<g::QueueBinding>> queues(
    const g::CompileReport &report, c::Queue &cpu) {
  std::vector<g::QueueBinding> result;
  for (std::size_t i = 0; i < report.lane_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(i));
    if (!lane.queue_required) continue;
    if (lane.target != kCpuTarget || lane.runtime_mode != c::ProviderRuntimeMode::kSharedQueue)
      return std::unexpected(hyle::Status::Internal("native effect fixture requested a Compute Queue"));
    result.push_back({lane.lane, &cpu});
  }
  return result;
}

inline hyle::Result<float> read(const g::GraphOutput &output,
    const n::Context &native, const n::MemoryResource &memory,
    const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto owner, output.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto actual, memory.binding(std::move(owner), description));
#if defined(HYLE_COMPUTE_VULKAN)
  std::array<float, 1> value;
  HYLE_RETURN_IF_ERROR(native.download(actual, std::as_writable_bytes(std::span(value))));
  return value[0];
#else
  (void)native;
  return actual.template data<float>()[0];
#endif
}
} // namespace effect_test
