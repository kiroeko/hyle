// P0 mechanism probe only. Hooks decorate the original route's command
// recording with GPU timestamps; they never replace a copy or dispatch.
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include <hyle/compute/vulkan_weights.h>
#include "../../samples/package/compute/cpp/vulkan_ffn.h"

#include <algorithm>
#include <cstdio>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
constexpr std::uint32_t dispatches = 512;
constexpr std::uint32_t copy_query = dispatches * 2;
VkQueryPool queries = VK_NULL_HANDLE;
PFN_vkCmdDispatch original_dispatch = nullptr;
PFN_vkCmdCopyBuffer original_copy = nullptr;
PFN_vkCmdResetQueryPool reset_queries = nullptr;
PFN_vkCmdWriteTimestamp write_timestamp = nullptr;
std::uint32_t recorded = 0, copies = 0;
bool initializing_queries = false;
VKAPI_ATTR void VKAPI_CALL dispatch_hook(VkCommandBuffer command, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  if (initializing_queries) {
    reset_queries(command, queries, 0, copy_query + 2);
    original_dispatch(command, x, y, z);
    return;
  }
  const auto index = recorded++ * 2;
  write_timestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, index);
  original_dispatch(command, x, y, z);
  write_timestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, index + 1);
}
VKAPI_ATTR void VKAPI_CALL copy_hook(VkCommandBuffer command, VkBuffer source, VkBuffer destination,
                                    std::uint32_t count, const VkBufferCopy *regions) {
  ++copies;
  write_timestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, copy_query);
  original_copy(command, source, destination, count, regions);
  write_timestamp(command, VK_PIPELINE_STAGE_TRANSFER_BIT, queries, copy_query + 1);
}
struct Hooks final {
  d::Device &device;
  ~Hooks() { device.fn.CmdDispatch = original_dispatch; device.fn.CmdCopyBuffer = original_copy; }
};
struct QueryOwner final {
  VkDevice device;
  PFN_vkDestroyQueryPool destroy;
  ~QueryOwner() { if (queries != VK_NULL_HANDLE) destroy(device, queries, nullptr); queries = VK_NULL_HANDLE; }
};
hyle::Status vk_status(VkResult result, const char *operation) {
  return result == VK_SUCCESS ? hyle::Status::Ok() : hyle::Status::Internal(operation);
}
hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  auto &device = *d::Access::runtime(context)->device;
  const auto &native = device.provider.native();
  const auto families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      native.get_instance_proc_addr(native.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  std::uint32_t family_count = 0; families(native.physical, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> properties(family_count);
  families(native.physical, &family_count, properties.data());
  const auto bits = std::min(properties[native.queue_family].timestampValidBits,
      properties[native.transfer_queue_family].timestampValidBits);
  if (bits < 32)
    return hyle::Status::Unavailable("original queue lacks an applicable timestamp domain");
  const auto load = [&](const char *name) { return device.instance_fn.GetDeviceProcAddr(native.device, name); };
  const auto create_queries = reinterpret_cast<PFN_vkCreateQueryPool>(load("vkCreateQueryPool"));
  const auto get_queries = reinterpret_cast<PFN_vkGetQueryPoolResults>(load("vkGetQueryPoolResults"));
  const auto destroy_queries = reinterpret_cast<PFN_vkDestroyQueryPool>(load("vkDestroyQueryPool"));
  reset_queries = reinterpret_cast<PFN_vkCmdResetQueryPool>(load("vkCmdResetQueryPool"));
  write_timestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(load("vkCmdWriteTimestamp"));
  VkQueryPoolCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.queryType = VK_QUERY_TYPE_TIMESTAMP; info.queryCount = copy_query + 2;
  HYLE_RETURN_IF_ERROR(vk_status(create_queries(native.device, &info, nullptr, &queries), "vkCreateQueryPool"));
  QueryOwner owner{native.device, destroy_queries};
  // Existing public sample shader and exact fixed-point -1 oracle. This is an
  // application-owned native fragment, not a new product operator/provider.
  constexpr std::uint32_t elements = 65535 * 64;
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{elements}));
  const auto byte_count = std::size_t(elements) * sizeof(float);
  std::vector<float> host(elements, -1.0f);
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(description, std::as_bytes(std::span(host))));
  HYLE_ASSIGN_OR_RETURN(auto output, context.allocate(description));
  HYLE_ASSIGN_OR_RETURN(auto kernel, sample_vulkan_direct::affine(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto storage, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto payload, hyle::mem::Buffer::create(hyle::mem::system_resource(), byte_count));
  std::fill(host.begin(), host.end(), 0.5f);
  HYLE_RETURN_IF_ERROR(payload.view().write(std::as_bytes(std::span(host))));
  HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(description, payload.view()));
  const std::array catalog{v::WeightSpec{17, description}};
  HYLE_ASSIGN_OR_RETURN(auto weights, v::WeightSlots::create(context, staging, storage, catalog, byte_count));
  HYLE_ASSIGN_OR_RETURN(auto recorder, v::Recorder::create(context));
  original_dispatch = device.fn.CmdDispatch; original_copy = device.fn.CmdCopyBuffer;
  Hooks hooks{device}; device.fn.CmdDispatch = dispatch_hook; device.fn.CmdCopyBuffer = copy_hook;
  // Reset/warm once before either measured submission, not inside the measured
  // dispatch/copy sequence. Query-reset dependencies must not create overlap
  // or serialization that the original native execution does not require.
  initializing_queries = true;
  HYLE_ASSIGN_OR_RETURN(auto initializer, v::Recorder::create(context));
  HYLE_RETURN_IF_ERROR(initializer.dispatch(kernel, std::array{input, output},
      {65535, 1, 1}, sample_vulkan_direct::count_push(elements)));
  HYLE_ASSIGN_OR_RETURN(auto initialization, initializer.finish());
  HYLE_ASSIGN_OR_RETURN(auto initialized, initialization.submit());
  HYLE_RETURN_IF_ERROR(initialized.wait());
  initializing_queries = false;
  for (std::uint32_t index = 0; index < dispatches; ++index) {
    const std::array bindings = index % 2 == 0 ? std::array{input, output} : std::array{output, input};
    HYLE_RETURN_IF_ERROR(recorder.dispatch(kernel, bindings, {65535, 1, 1}, sample_vulkan_direct::count_push(elements)));
  }
  HYLE_ASSIGN_OR_RETURN(auto program, recorder.finish());
  if (recorded != dispatches) return hyle::Status::Internal("probe did not decorate the original dispatch chain");
  HYLE_ASSIGN_OR_RETURN(auto compute, program.submit());
  const std::array selection{v::WeightFragment{17, source}};
  HYLE_ASSIGN_OR_RETURN(auto copied, weights.upload(0, selection));
  HYLE_RETURN_IF_ERROR(copied.wait()); HYLE_RETURN_IF_ERROR(compute.wait());
  // Restore before oracle downloads; only the measured payload upload belongs
  // to the copy timestamp pair. All accepted work has actually retired here.
  device.fn.CmdDispatch = original_dispatch; device.fn.CmdCopyBuffer = original_copy;
  std::array<std::uint64_t, copy_query + 2> ticks{};
  HYLE_RETURN_IF_ERROR(vk_status(get_queries(native.device, queries, 0, copy_query + 2,
      sizeof(ticks), ticks.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults"));
  std::uint32_t intersections = 0; std::uint64_t overlap = 0;
  const auto mask = bits == 64 ? UINT64_MAX : (std::uint64_t{1} << bits) - 1;
  for (auto &tick : ticks) tick &= mask;
  for (std::uint32_t index = 0; index < dispatches; ++index) {
    const auto start = std::max(ticks[index * 2], ticks[copy_query]);
    const auto end = std::min(ticks[index * 2 + 1], ticks[copy_query + 1]);
    if (end > start) { ++intersections; overlap += end - start; }
  }
  HYLE_RETURN_IF_ERROR(context.download(input, std::as_writable_bytes(std::span(host))));
  if (!std::ranges::all_of(host, [](float value) { return value == -1.0f; }))
    return hyle::Status::Internal("original native compute oracle differs");
  HYLE_ASSIGN_OR_RETURN(auto weight, copied.binding(17));
  HYLE_RETURN_IF_ERROR(context.download(weight, std::as_writable_bytes(std::span(host))));
  if (!std::ranges::all_of(host, [](float value) { return value == 0.5f; }) || copies != 1)
    return hyle::Status::Internal("original weight transfer oracle or copy count differs");
  std::printf("{\"device\":\"%s\",\"queue_family\":%u,\"transfer_family\":%u,\"distinct_transfer_queue\":%s,\"timestamp_bits\":%u,\"period_ns\":%.9g,\"dispatch_count\":%u,\"copy_count\":%u,\"payload_bytes\":%zu,\"copy_ticks\":[%llu,%llu],\"intersected_dispatch_brackets\":%u,\"intersection_ns\":%.9g,\"math_and_payload\":\"PASS\"}\n",
      context.environment().device_name.c_str(), native.queue_family, native.transfer_queue_family,
      context.environment().distinct_transfer_queue ? "true" : "false", bits, double(device.properties.limits.timestampPeriod),
      recorded, copies, byte_count, static_cast<unsigned long long>(ticks[copy_query]), static_cast<unsigned long long>(ticks[copy_query + 1]),
      intersections, double(overlap) * device.properties.limits.timestampPeriod);
  for (std::uint32_t index = 0; index < dispatches; ++index)
    std::printf("dispatch,%u,%llu,%llu\n", index, static_cast<unsigned long long>(ticks[index * 2]), static_cast<unsigned long long>(ticks[index * 2 + 1]));
  return weights.close();
}
}
int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
