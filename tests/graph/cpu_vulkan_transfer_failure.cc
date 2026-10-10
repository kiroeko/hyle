// Real native copy prefix with a rejected handoff. Injection targets the
// status boundary only; it cannot qualify provider math or physical overlap.
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include <hyle/compute/vulkan_weights.h>

#include <array>
#include <algorithm>
#include <cstdio>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
PFN_vkQueueSubmit original_submit = nullptr;
VkQueue compute_queue = VK_NULL_HANDLE, transfer_queue = VK_NULL_HANDLE;
std::uint32_t copies = 0, rejected = 0;
bool reject_handoff = false;
VKAPI_ATTR VkResult VKAPI_CALL submit_hook(VkQueue queue, std::uint32_t count,
                                          const VkSubmitInfo *info, VkFence fence) {
  if (reject_handoff && queue == compute_queue && count == 1 && info[0].waitSemaphoreCount == 1) {
    ++rejected; return VK_ERROR_OUT_OF_HOST_MEMORY;
  }
  const auto result = original_submit(queue, count, info, fence);
  if (result == VK_SUCCESS && queue == transfer_queue && info[0].signalSemaphoreCount == 1) ++copies;
  return result;
}
struct Hook final {
  d::Device &device;
  ~Hook() { device.fn.QueueSubmit = original_submit; }
};
hyle::Status require(bool condition, const char *message) {
  return condition ? hyle::Status::Ok() : hyle::Status::Internal(message);
}
hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  HYLE_ASSIGN_OR_RETURN(auto row, c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}));
  HYLE_ASSIGN_OR_RETURN(auto source, hyle::mem::Buffer::create(hyle::mem::system_resource(), 256));
  std::array<float, 64> data; data.fill(3.0f);
  HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(data))));
  HYLE_ASSIGN_OR_RETURN(auto input, c::ConstTensorView::create(row, source.view()));
  HYLE_ASSIGN_OR_RETURN(auto legacy, context.allocate(row));
  HYLE_ASSIGN_OR_RETURN(auto legacy_info, device.storage_info(legacy));
  HYLE_RETURN_IF_ERROR(require(!legacy_info.transfer_eligible, "legacy storage changed family contract"));
  HYLE_ASSIGN_OR_RETURN(auto copier, v::TransferSlot::create(context, staging, 256));
  HYLE_ASSIGN_OR_RETURN(auto copied, copier.upload(legacy, input)); HYLE_RETURN_IF_ERROR(copied.wait());
  HYLE_RETURN_IF_ERROR(context.download(legacy, std::as_writable_bytes(std::span(data))));
  HYLE_RETURN_IF_ERROR(require(std::ranges::all_of(data, [](float x) { return x == 3.0f; }), "legacy same-queue upload differs"));
  if (!context.environment().distinct_transfer_queue) {
    std::printf("Legacy upload PASS; cross-queue rejection not applicable: native queues alias\n");
    return hyle::Status::Ok();
  }
  auto &native = *d::Access::runtime(context)->device;
  original_submit = native.fn.QueueSubmit; compute_queue = native.queue; transfer_queue = native.transfer_queue;
  const std::array limits{c::ResourceBudgetLimit{hyle::mem::system_resource(), 1024 * 1024},
      c::ResourceBudgetLimit{device.ref(), 4096}, c::ResourceBudgetLimit{staging.ref(), 4096}};
  HYLE_ASSIGN_OR_RETURN(auto budget, c::ResourceBudget::create({.limits = limits}));
  {
    const std::array catalog{v::WeightSpec{7, row}};
    HYLE_ASSIGN_OR_RETURN(auto pool, v::WeightSlots::create(context, staging, device, catalog, 256, {.resource_budget = budget}));
    const std::array selected{v::WeightFragment{7, input}};
    Hook hook{native}; native.fn.QueueSubmit = submit_hook; reject_handoff = true;
    HYLE_ASSIGN_OR_RETURN(auto batch, pool.upload(0, selected));
    reject_handoff = false; native.fn.QueueSubmit = original_submit;
    HYLE_RETURN_IF_ERROR(require(copies == 1 && rejected == 1, "copy was not accepted before handoff rejection"));
    const auto terminal = batch.wait();
    HYLE_RETURN_IF_ERROR(require(terminal.IsResourceExhausted(), "accepted copy lost the original handoff diagnosis"));
    HYLE_RETURN_IF_ERROR(require(!batch.binding(7), "failed handoff published ready weights"));
    batch = {};
    HYLE_RETURN_IF_ERROR(require(!pool.upload(0, selected), "failed native prefix became replayable"));
    const auto closed = pool.close();
    HYLE_RETURN_IF_ERROR(require(closed.IsResourceExhausted(), "pool close lost failed-prefix diagnosis"));
    HYLE_RETURN_IF_ERROR(context.collect());
  }
  HYLE_ASSIGN_OR_RETURN(auto usage, budget.usage(device.ref()));
  HYLE_RETURN_IF_ERROR(require(usage.live_bytes == 0, "retired failed prefix leaked device budget"));
  HYLE_ASSIGN_OR_RETURN(usage, budget.usage(staging.ref()));
  HYLE_RETURN_IF_ERROR(require(usage.live_bytes == 0, "retired failed prefix leaked staging budget"));
  HYLE_RETURN_IF_ERROR(context.close());
  std::printf("Native upload: legacy family, accepted copy/rejected handoff, diagnosis, no ready output and retirement PASS\n");
  return hyle::Status::Ok();
}
}
int main() {
  const auto status = run(); if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
