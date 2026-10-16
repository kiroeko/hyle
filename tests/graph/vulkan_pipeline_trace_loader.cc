// Validation only. Forward every original operation; decorate the selected
// original ggml Neg dispatch with queries. No replacement math or host waits.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include "vulkan_pipeline_trace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define G09_EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define G09_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
std::mutex monitor;
std::once_flag loader_once;
PFN_vkGetInstanceProcAddr real_get = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
VkInstance instance = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
VkQueryPool queries = VK_NULL_HANDLE;
constexpr std::uint32_t query_count = 512;
std::uint32_t next_query = 0;
std::atomic<std::uint32_t> observed_phase{0};
std::atomic<bool> failed{false};
g09_vk::DeviceInfo facts;
std::vector<unsigned char> reference;
std::string derived_reference_path;
std::unordered_set<VkShaderModule> neg_modules;
std::unordered_set<VkPipeline> neg_pipelines;
struct Command {
  bool negative = false;
  std::vector<std::uint32_t> query_pairs;
};
std::unordered_map<VkCommandBuffer, Command> commands;
struct Submitted { std::uint32_t phase, query; };
std::vector<Submitted> submitted;
PFN_vkCreateShaderModule create_shader;
PFN_vkDestroyShaderModule destroy_shader;
PFN_vkCreateComputePipelines create_pipelines;
PFN_vkDestroyPipeline destroy_pipeline;
PFN_vkBeginCommandBuffer begin_command;
PFN_vkFreeCommandBuffers free_commands;
PFN_vkCmdBindPipeline bind_pipeline;
PFN_vkCmdDispatch dispatch;
PFN_vkCmdResetQueryPool reset_queries;
PFN_vkCmdWriteTimestamp timestamp;
PFN_vkGetQueryPoolResults get_queries;
PFN_vkDestroyQueryPool destroy_queries;
PFN_vkDestroyDevice destroy_device;
PFN_vkQueueSubmit queue_submit;
PFN_vkQueueSubmit2 queue_submit2;
PFN_vkGetCalibratedTimestampsEXT calibrate;

void load() {
  std::call_once(loader_once, [] {
#if defined(_WIN32)
    const auto library = LoadLibraryW(L"vulkan-1.dll");
    real_get = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr")) : nullptr;
#elif defined(__APPLE__)
    auto *library = dlopen("/opt/homebrew/lib/libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    real_get = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr")) : nullptr;
#else
    auto *library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    real_get = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr")) : nullptr;
#endif
    // One test process pins this original loader through process exit. HYLE
    // still owns/destroys the original instance/device and all actual work.
    submitted.reserve(64); commands.reserve(128); reference.reserve(16384);
  });
}
template<class T> T from_instance(const char *name) { return reinterpret_cast<T>(real_get(instance, name)); }
template<class T> T from_device(const char *name) { return reinterpret_cast<T>(real_get_device(device, name)); }
bool prepare_reference(bool rte, bool denorm) {
  // Exact locked ggml-vulkan.cpp float-controls transformation, performed
  // only on the reference bytes. The original shader is always forwarded.
  // SPIR-V opcodes/enums are from the locked SDK unified1/spirv.hpp.
  if (reference.size() % 4 || reference.size() < 20) return false;
  if (rte || denorm) {
    std::vector<std::uint32_t> words(reference.size() / 4);
    std::memcpy(words.data(), reference.data(), reference.size());
    std::size_t pos = 5, cap = 5, ext = 5, exec = 5;
    std::uint32_t entry = 0;
    while (pos < words.size()) {
      const auto op = words[pos] & 0xffffu, count = words[pos] >> 16;
      if (!count || count > words.size() - pos) return false;
      if (op == 17) { cap = pos + count; ext = pos + count; }
      else if (op == 10) ext = pos + count;
      else if (op == 15) { if (count < 3) return false; entry = words[pos + 2]; exec = pos + count; }
      else if (op == 16 || op == 331) exec = pos + count;
      else if (entry) break;
      pos += count;
    }
    if (!entry) return false;
    const auto insert_mode = [&](std::uint32_t mode) {
      const std::uint32_t instruction[]{(4u << 16) | 16u, entry, mode, 16};
      words.insert(words.begin() + exec, std::begin(instruction), std::end(instruction));
    };
    if (rte) insert_mode(4462);
    if (denorm) insert_mode(4459);
    constexpr char text[] = "SPV_KHR_float_controls";
    std::vector<std::uint32_t> extension(1 + (sizeof(text) + 3) / 4, 0);
    extension[0] = (static_cast<std::uint32_t>(extension.size()) << 16) | 10u;
    std::memcpy(extension.data() + 1, text, sizeof(text));
    words.insert(words.begin() + ext, extension.begin(), extension.end());
    const auto insert_capability = [&](std::uint32_t capability) {
      const std::uint32_t instruction[]{(2u << 16) | 17u, capability};
      words.insert(words.begin() + cap, std::begin(instruction), std::end(instruction));
    };
    if (rte) insert_capability(4467);
    if (denorm) insert_capability(4464);
    reference.resize(words.size() * 4); std::memcpy(reference.data(), words.data(), reference.size());
  }
  std::ofstream output(derived_reference_path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(reference.data()), static_cast<std::streamsize>(reference.size()));
  return output.good();
}
bool sample(g09_vk::Calibration &result) {
  if (failed || !calibrate || !device) return false;
  VkCalibratedTimestampInfoEXT info[2]{};
  for (auto &i : info) i.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
  info[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
  info[1].timeDomain = static_cast<VkTimeDomainEXT>(facts.host_domain);
  std::uint64_t ticks[2]{};
  if (calibrate(device, 2, info, ticks, &result.max_deviation_ns) != VK_SUCCESS) return false;
  result.device_ticks = ticks[0]; result.host_ticks = ticks[1];
  return true;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo *original,
    const VkAllocationCallbacks *allocator, VkDevice *result) {
  const auto enumerate = from_instance<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
  std::uint32_t count = 0;
  enumerate(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  enumerate(physical, nullptr, &count, extensions.data());
  const char *extension = nullptr;
  for (const auto &e : extensions)
    if (std::strcmp(e.extensionName, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) == 0)
      extension = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  if (!extension) for (const auto &e : extensions)
    if (std::strcmp(e.extensionName, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) == 0)
      extension = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  auto info = *original;
  std::vector<const char *> enabled;
  for (std::uint32_t i = 0; i < original->enabledExtensionCount; ++i)
    enabled.push_back(original->ppEnabledExtensionNames[i]);
  if (extension && std::none_of(enabled.begin(), enabled.end(), [&](auto *e) { return std::strcmp(e, extension) == 0; }))
    enabled.push_back(extension);
  info.enabledExtensionCount = static_cast<std::uint32_t>(enabled.size());
  info.ppEnabledExtensionNames = enabled.data();
  const auto code = from_instance<PFN_vkCreateDevice>("vkCreateDevice")(physical, &info, allocator, result);
  if (code != VK_SUCCESS) return code;
  if (device != VK_NULL_HANDLE) { failed = true; return code; } // Exactly one original context per trace process.
  device = *result;
  real_get_device = from_instance<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
#define G09_LOAD(variable, type, name) variable = from_device<type>(name)
  G09_LOAD(create_shader, PFN_vkCreateShaderModule, "vkCreateShaderModule");
  G09_LOAD(destroy_shader, PFN_vkDestroyShaderModule, "vkDestroyShaderModule");
  G09_LOAD(create_pipelines, PFN_vkCreateComputePipelines, "vkCreateComputePipelines");
  G09_LOAD(destroy_pipeline, PFN_vkDestroyPipeline, "vkDestroyPipeline");
  G09_LOAD(begin_command, PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer");
  G09_LOAD(free_commands, PFN_vkFreeCommandBuffers, "vkFreeCommandBuffers");
  G09_LOAD(bind_pipeline, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
  G09_LOAD(dispatch, PFN_vkCmdDispatch, "vkCmdDispatch");
  G09_LOAD(reset_queries, PFN_vkCmdResetQueryPool, "vkCmdResetQueryPool");
  G09_LOAD(timestamp, PFN_vkCmdWriteTimestamp, "vkCmdWriteTimestamp");
  G09_LOAD(get_queries, PFN_vkGetQueryPoolResults, "vkGetQueryPoolResults");
  G09_LOAD(destroy_queries, PFN_vkDestroyQueryPool, "vkDestroyQueryPool");
  G09_LOAD(destroy_device, PFN_vkDestroyDevice, "vkDestroyDevice");
  G09_LOAD(queue_submit, PFN_vkQueueSubmit, "vkQueueSubmit");
  G09_LOAD(queue_submit2, PFN_vkQueueSubmit2, "vkQueueSubmit2");
  if (!queue_submit2) G09_LOAD(queue_submit2, PFN_vkQueueSubmit2, "vkQueueSubmit2KHR");
  G09_LOAD(calibrate, PFN_vkGetCalibratedTimestampsEXT, "vkGetCalibratedTimestampsEXT");
  if (!calibrate) G09_LOAD(calibrate, PFN_vkGetCalibratedTimestampsEXT, "vkGetCalibratedTimestampsKHR");
#undef G09_LOAD
  VkPhysicalDeviceProperties properties{};
  from_instance<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(physical, &properties);
  VkPhysicalDeviceVulkan12Properties v12{}; v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
  VkPhysicalDeviceProperties2 extended{}; extended.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2; extended.pNext = &v12;
  from_instance<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(physical, &extended);
  facts.fp16_controls = (v12.shaderRoundingModeRTEFloat16 ? 1u : 0u) | (v12.shaderDenormPreserveFloat16 ? 2u : 0u);
  const bool reference_ready = prepare_reference(v12.shaderRoundingModeRTEFloat16, v12.shaderDenormPreserveFloat16);
  facts.timestamp_period_ns = properties.limits.timestampPeriod;
  auto families = from_instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
  families(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> family(count); families(physical, &count, family.data());
  facts.timestamp_bits = 64;
  for (std::uint32_t i = 0; i < original->queueCreateInfoCount; ++i)
    facts.timestamp_bits = std::min(facts.timestamp_bits, family.at(original->pQueueCreateInfos[i].queueFamilyIndex).timestampValidBits);
#if defined(_WIN32)
  facts.host_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
  LARGE_INTEGER frequency{}; QueryPerformanceFrequency(&frequency); facts.host_frequency = frequency.QuadPart;
#else
  facts.host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT; facts.host_frequency = 1000000000;
#endif
  auto domains = from_instance<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>("vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
  if (!domains) domains = from_instance<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>("vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
  bool host = false, gpu = false;
  if (domains && domains(physical, &count, nullptr) == VK_SUCCESS) {
    std::vector<VkTimeDomainEXT> available(count);
    if (domains(physical, &count, available.data()) == VK_SUCCESS)
      for (const auto d : available) { host |= static_cast<unsigned>(d) == facts.host_domain; gpu |= d == VK_TIME_DOMAIN_DEVICE_EXT; }
  }
  failed = !reference_ready || !extension || !host || !gpu || !calibrate || !facts.host_frequency || facts.timestamp_bits < 32;
  VkQueryPoolCreateInfo pool{}; pool.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  pool.queryType = VK_QUERY_TYPE_TIMESTAMP; pool.queryCount = query_count;
  const auto create_queries = from_device<PFN_vkCreateQueryPool>("vkCreateQueryPool");
  if (create_queries(device, &pool, nullptr, &queries) != VK_SUCCESS) failed = true;
  std::printf("{\"vulkan_trace_device\":true,\"device\":\"%s\",\"calibrated_extension\":\"%s\",\"host_domain\":%u,\"host_frequency\":%llu,\"timestamp_bits\":%u,\"timestamp_period_ns\":%.9g,\"fp16_controls\":%u,\"derived_reference_bytes\":%zu,\"ready\":%s}\n",
      properties.deviceName, extension ? extension : "UNAVAILABLE", facts.host_domain,
      static_cast<unsigned long long>(facts.host_frequency), facts.timestamp_bits, facts.timestamp_period_ns,
      facts.fp16_controls, reference.size(), failed ? "false" : "true");
  return code;
}
VKAPI_ATTR void VKAPI_CALL hook_destroy_device(VkDevice target, const VkAllocationCallbacks *allocator) {
  if (target == device && queries) { destroy_queries(device, queries, nullptr); queries = VK_NULL_HANDLE; }
  destroy_device(target, allocator);
  if (target == device) device = VK_NULL_HANDLE;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_create_shader(VkDevice target, const VkShaderModuleCreateInfo *info,
    const VkAllocationCallbacks *allocator, VkShaderModule *result) {
  const auto code = create_shader(target, info, allocator, result);
  if (code == VK_SUCCESS && info->codeSize == reference.size() &&
      std::memcmp(info->pCode, reference.data(), reference.size()) == 0) {
    std::lock_guard lock(monitor); neg_modules.insert(*result); ++facts.matched_shader_modules;
  }
  return code;
}
VKAPI_ATTR void VKAPI_CALL hook_destroy_shader(VkDevice target, VkShaderModule shader, const VkAllocationCallbacks *allocator) {
  { std::lock_guard lock(monitor); neg_modules.erase(shader); }
  destroy_shader(target, shader, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_create_pipelines(VkDevice target, VkPipelineCache cache, std::uint32_t count,
    const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *allocator, VkPipeline *result) {
  const auto code = create_pipelines(target, cache, count, info, allocator, result);
  std::lock_guard lock(monitor);
  for (std::uint32_t i = 0; i < count; ++i) if (result[i] && neg_modules.contains(info[i].stage.module) &&
      info[i].stage.pName && std::strcmp(info[i].stage.pName, "main") == 0) {
    neg_pipelines.insert(result[i]); ++facts.matched_pipelines;
  }
  return code;
}
VKAPI_ATTR void VKAPI_CALL hook_destroy_pipeline(VkDevice target, VkPipeline pipeline, const VkAllocationCallbacks *allocator) {
  { std::lock_guard lock(monitor); neg_pipelines.erase(pipeline); }
  destroy_pipeline(target, pipeline, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_begin(VkCommandBuffer command, const VkCommandBufferBeginInfo *info) {
  const auto code = begin_command(command, info);
  if (code == VK_SUCCESS) {
    std::lock_guard lock(monitor); commands[command] = {};
    if (commands.size() > 256) failed = true;
  }
  return code;
}
VKAPI_ATTR void VKAPI_CALL hook_free(VkDevice target, VkCommandPool pool, std::uint32_t count, const VkCommandBuffer *list) {
  { std::lock_guard lock(monitor); for (std::uint32_t i = 0; i < count; ++i) commands.erase(list[i]); }
  free_commands(target, pool, count, list);
}
VKAPI_ATTR void VKAPI_CALL hook_bind(VkCommandBuffer command, VkPipelineBindPoint point, VkPipeline pipeline) {
  if (point == VK_PIPELINE_BIND_POINT_COMPUTE) {
    std::lock_guard lock(monitor); commands[command].negative = neg_pipelines.contains(pipeline);
  }
  bind_pipeline(command, point, pipeline);
}
VKAPI_ATTR void VKAPI_CALL hook_dispatch(VkCommandBuffer command, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  auto index = query_count;
  {
    std::lock_guard lock(monitor);
    auto &state = commands[command];
    if (state.negative && !failed) {
      if (next_query + 2 > query_count) failed = true;
      else { index = next_query; next_query += 2; state.query_pairs.push_back(index); }
    }
  }
  if (index < query_count) {
    // Reset belongs to this original CB and executes on every replay. Queries
    // are distinct across recordings; the fixture observes only one live slot.
    reset_queries(command, queries, index, 2);
    timestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, index);
  }
  dispatch(command, x, y, z);
  if (index < query_count) timestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, index + 1);
}
void attribute(VkCommandBuffer command) {
  const auto phase = observed_phase.load(std::memory_order_acquire);
  if (!phase) return;
  std::lock_guard lock(monitor);
  const auto found = commands.find(command);
  if (found == commands.end()) return;
  for (auto query : found->second.query_pairs) {
    if (submitted.size() == 64) { failed = true; break; }
    submitted.push_back({phase, query});
  }
}
VKAPI_ATTR VkResult VKAPI_CALL hook_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo *info, VkFence fence) {
  for (std::uint32_t i = 0; i < count; ++i)
    for (std::uint32_t j = 0; j < info[i].commandBufferCount; ++j) attribute(info[i].pCommandBuffers[j]);
  const auto code = queue_submit(queue, count, info, fence);
  if (code != VK_SUCCESS) failed = true;
  return code;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_submit2(VkQueue queue, std::uint32_t count, const VkSubmitInfo2 *info, VkFence fence) {
  for (std::uint32_t i = 0; i < count; ++i)
    for (std::uint32_t j = 0; j < info[i].commandBufferInfoCount; ++j) attribute(info[i].pCommandBufferInfos[j].commandBuffer);
  const auto code = queue_submit2(queue, count, info, fence);
  if (code != VK_SUCCESS) failed = true;
  return code;
}
PFN_vkVoidFunction replacement(const char *name) {
#define G09_REPLACE(n, f) if (std::strcmp(name, n) == 0) return reinterpret_cast<PFN_vkVoidFunction>(&f)
  G09_REPLACE("vkCreateDevice", hook_create_device); G09_REPLACE("vkDestroyDevice", hook_destroy_device);
  G09_REPLACE("vkCreateShaderModule", hook_create_shader); G09_REPLACE("vkDestroyShaderModule", hook_destroy_shader);
  G09_REPLACE("vkCreateComputePipelines", hook_create_pipelines); G09_REPLACE("vkDestroyPipeline", hook_destroy_pipeline);
  G09_REPLACE("vkBeginCommandBuffer", hook_begin); G09_REPLACE("vkFreeCommandBuffers", hook_free);
  G09_REPLACE("vkCmdBindPipeline", hook_bind); G09_REPLACE("vkCmdDispatch", hook_dispatch);
  G09_REPLACE("vkQueueSubmit", hook_submit); G09_REPLACE("vkQueueSubmit2", hook_submit2); G09_REPLACE("vkQueueSubmit2KHR", hook_submit2);
#undef G09_REPLACE
  return nullptr;
}
} // namespace

G09_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice target, const char *name) {
  const auto original = real_get_device ? real_get_device(target, name) : nullptr;
  if (!original) return nullptr;
  if (auto hook = replacement(name)) return hook;
  return original;
}
G09_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance target, const char *name) {
  load(); if (!real_get) return nullptr;
  const auto original = real_get(target, name); if (!original) return nullptr;
  if (target) instance = target;
  if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&vkGetInstanceProcAddr);
  if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&vkGetDeviceProcAddr);
  if (auto hook = replacement(name)) return hook;
  return original;
}
G09_EXPORT bool g09_trace_configure(const void *bytes, std::size_t size, const char *derived_path) {
  load(); if (device || !bytes || size < 20 || size > 65536) return false;
  const auto *first = static_cast<const unsigned char *>(bytes); reference.assign(first, first + size);
  if (!derived_path || !*derived_path) return false;
  derived_reference_path = derived_path;
  return real_get != nullptr;
}
G09_EXPORT bool g09_trace_ready(g09_vk::DeviceInfo *info) {
  std::lock_guard lock(monitor); *info = facts;
  return !failed && device && queries && !reference.empty();
}
G09_EXPORT bool g09_trace_begin(std::uint32_t phase, g09_vk::Calibration *before) {
  observed_phase.store(0, std::memory_order_release);
  { std::lock_guard lock(monitor); submitted.clear(); }
  if (!sample(*before)) return false;
  observed_phase.store(phase, std::memory_order_release); return true;
}
G09_EXPORT bool g09_trace_end(g09_vk::Observation *observation) {
  observed_phase.store(0, std::memory_order_release);
  if (!sample(observation->after)) return false;
  std::lock_guard lock(monitor); observation->count = 0;
  for (const auto &record : submitted) {
    std::uint64_t values[4]{};
    const auto code = get_queries(device, queries, record.query, 2, sizeof(values), values, 2 * sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (code != VK_SUCCESS || !values[1] || !values[3]) return false;
    observation->dispatches[observation->count++] = {values[0], values[2], record.phase, record.query};
  }
  return !failed;
}
