// compute-06: the VULKAN COMPUTE backend (coding-conventions §10: an independent
// composition unit -- one TU filling the manual vtable, NOT a switch in a god object).
// Compiled ONLY when HYLE_COMPUTE_VULKAN_ENABLED (volk + Vulkan-Headers + glslang fetched;
// see root CMakeLists). ALL Vulkan/volk types are confined to THIS .cc -- the public
// headers (backend.h is internal, kernel.h/stream.h public) never name a vk* type, same
// discipline as cuda_backend.cu and ggml_bridge.cc.
//
// This gives HYLE its OWN cross-vendor GPU compute path (the Vulkan twin of compute-04's
// CUDA NVRTC "any kernel"): SPIR-V module + descriptor set + push constants + dispatch.
// ggml is only an AI-operator supplier (compute-05); anything else -- plain vecadd/reduce
// or ② CAD/CAE geometry/SpMV kernels -- ggml can't run on GPU, so HYLE authors them here.
//
// MAPPING HYLE's CUDA-shaped vtable onto Vulkan (the real impedance, resolved by the
// compute-06 scouting -- see docs/milestones/compute-06-vulkan-spirv.md + the memory
// project_compute06_vulkan_scouting; each vtable concept -> its Vulkan analog):
//   Device   -> VkPhysicalDevice + a single VkDevice (created once, shared)
//   Stream   -> a VkQueue + its VkCommandPool (submission channel). ONE compute queue
//               (measured: multi-queue is 0.85x on same-shape compute -- batching wins).
//   Event    -> owned VkFence + record state. event_query = vkGetFenceStatus (measured
//               non-blocking, == cudaEventQuery semantics -> feeds the same GpuPoller).
//   Buffer   -> VkBuffer + VkDeviceMemory. memory-type picked per kind (ReBAR-aware).
//   Kernel   -> VkShaderModule + VkPipeline + VkPipelineLayout + VkDescriptorSetLayout.
//   launch   -> descriptor set (buffers) + push constants (scalar blob) + vkCmdDispatch.
//
// COMPLETION: a Stream owns a VkFence; launch records+submits with that fence signalled;
// event_query polls it (GpuPoller, compute-03, non-blocking). NO blocking waitForFences
// on the hot path (kompute's one blocking call, replaced -- the "second scheduler" rule).

#include "volk.h"  // must precede any vulkan header; VK_NO_PROTOTYPES set by target

#include <cstdint>
#include <cstdio>    // fopen/fread/fwrite -- pipeline cache 落盘 (checklist #1, env-gated)
#include <cstdlib>   // getenv -- pipeline-cache path (#1) + TEST-ONLY staging forcing (make_mem)
#include <cstring>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "backend.h"
#include "hyle/compute/types.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#if defined(HYLE_TESTING)
#  include "hyle/visibility.h"  // HYLE_EXPORT -- export the #1 cache-flush test hook across the DLL
#endif

namespace hyle::compute {
namespace {

// Translate a VkResult (external boundary) into a HYLE Status -- never leak VkResult
// across the public API. Canonical taxonomy mirrors from_cuda in cuda_backend.cu.
Status from_vk(VkResult r, const char* what) noexcept {
  if (r == VK_SUCCESS) return Status();
  switch (r) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return Status::ResourceExhausted(what);
    case VK_ERROR_INITIALIZATION_FAILED:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_DEVICE_LOST:
      return Status::Unavailable(what);
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return Status::Unimplemented(what);
    default:
      return Status::Internal(what);
  }
}

// ---- pipeline cache 落盘 (checklist #1) --------------------------------------------
// kompute builds an EMPTY per-algorithm cache = no reuse. HYLE keeps a device-shared cache
// (built once in init) AND optionally PERSISTS it across process runs: cold-start no longer
// re-compiles SPIR-V -> ISA (tens of ms/kernel). Persistence is OPT-IN via the env var
// HYLE_VK_PIPELINE_CACHE (a file path) -- a base library must NOT silently write to a user's
// disk, so unset = pure in-memory shared cache (the prior behavior). The file is our own
// framed format: a small header {magic, pipelineCacheUUID} + the raw vkGetPipelineCacheData
// blob. On load we reject a stale/foreign blob by UUID (换卡/换驱动 -> Vulkan changes the
// UUID) so the driver never ingests an incompatible cache (it would ignore it anyway, but we
// avoid even handing it over). Corruption/short reads fall back to an empty cache (never fatal
// -- the cache is a pure optimization).
struct PipeCacheHeader {
  char magic[8];                  // "HYLEPC01" -- format + version tag
  uint8_t uuid[VK_UUID_SIZE];     // pipelineCacheUUID this blob was produced under
};
constexpr char kPipeCacheMagic[8] = {'H', 'Y', 'L', 'E', 'P', 'C', '0', '1'};

const char* pipe_cache_path() noexcept {
  const char* p = std::getenv("HYLE_VK_PIPELINE_CACHE");
  return (p != nullptr && p[0] != '\0') ? p : nullptr;
}

// Read a persisted cache blob whose header UUID matches `uuid` into `out`. Returns true and
// fills `out` with the driver payload (header stripped) only on a full, matching read; any
// mismatch/short/absent file returns false (caller creates an empty cache).
bool load_pipe_cache(const char* path, const uint8_t uuid[VK_UUID_SIZE],
                     std::vector<uint8_t>* out) noexcept {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return false;  // first run / no file
  bool ok = false;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    long sz = std::ftell(f);
    if (sz > static_cast<long>(sizeof(PipeCacheHeader))) {
      std::rewind(f);
      PipeCacheHeader h{};
      if (std::fread(&h, sizeof(h), 1, f) == 1 &&
          std::memcmp(h.magic, kPipeCacheMagic, sizeof(kPipeCacheMagic)) == 0 &&
          std::memcmp(h.uuid, uuid, VK_UUID_SIZE) == 0) {
        std::size_t payload = static_cast<std::size_t>(sz) - sizeof(PipeCacheHeader);
        out->resize(payload);
        ok = std::fread(out->data(), 1, payload, f) == payload;
        if (!ok) out->clear();
      }
    }
  }
  std::fclose(f);
  return ok;
}

// Serialize `blob` (from vkGetPipelineCacheData) under a header stamped with `uuid`. Best-
// effort: a write failure is swallowed (the cache is an optimization, not correctness).
void save_pipe_cache(const char* path, const uint8_t uuid[VK_UUID_SIZE],
                     const std::vector<uint8_t>& blob) noexcept {
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return;
  PipeCacheHeader h{};
  std::memcpy(h.magic, kPipeCacheMagic, sizeof(kPipeCacheMagic));
  std::memcpy(h.uuid, uuid, VK_UUID_SIZE);
  std::fwrite(&h, sizeof(h), 1, f);
  if (!blob.empty()) std::fwrite(blob.data(), 1, blob.size(), f);
  std::fclose(f);
}

// ---- process-wide Vulkan context (created once, shared by all streams/buffers) -------
// Mirrors the CUDA backend's implicit process context. Built lazily on first vtable use
// via a Meyers singleton; if the loader/device is absent, `ok` stays false and
// vulkan_vtable() returns null so compute falls back to CUDA/host (no crash).
struct VkContext {
  bool ok = false;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t compute_qfi = 0;              // compute queue family index
  VkQueue queue = VK_NULL_HANDLE;        // the one compute queue (measured: 1 is enough)
  std::mutex submit_mtx;                 // serialize queue submits (a VkQueue is external-sync)
  std::mutex build_mtx;                  // serialize pipeline build: VkPipelineCache host access
                                         // is externally-synchronized (spec) + guards lazy
                                         // per-Kernel pipeline creation against concurrent first-launch.
  VkPipelineCache pipe_cache = VK_NULL_HANDLE;  // shared cache (checklist #1: not per-kernel)
  uint8_t cache_uuid[VK_UUID_SIZE] = {};  // pipelineCacheUUID: cache validity key (换卡/换驱动作废)
  uint32_t max_push = 0;                  // maxPushConstantsSize (checklist #4 device limit)
  uint32_t max_wg_count[3] = {0, 0, 0};   // maxComputeWorkGroupCount x/y/z (#20 clamp guard)
  std::size_t buf_alignment = 16;         // minStorageBufferOffsetAlignment (>=16); device const
  std::size_t buf_max_size = 0;           // maxStorageBufferRange; device const
  VkDeviceSize non_coherent_atom_size = 1;  // mapped flush/invalidate range granularity
  float timestamp_period = 0.0f;          // ns per timestamp tick (0 = timestamps unsupported)
  uint32_t timestamp_valid_bits = 0;       // queue-family timestamp counter width
  char dev_name[256] = "vulkan";

  VkContext() noexcept { init(); }

  void init() noexcept {
    if (volkInitialize() != VK_SUCCESS) return;  // no loader -> stay !ok
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    // MoltenVK (mac) is a PORTABILITY driver: vkCreateInstance fails with
    // VK_ERROR_INCOMPATIBLE_DRIVER (-9, measured on Apple M4) unless we opt in via
    // VK_KHR_portability_enumeration + the ENUMERATE_PORTABILITY flag. Probe the ext at
    // runtime and enable only when present, so ONE code path serves mac + the discrete-GPU
    // platforms (Win/WSL) where the ext is absent and must NOT be requested.
    const char* kPortEnum = "VK_KHR_portability_enumeration";
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
    std::vector<VkExtensionProperties> ie(nie);
    if (nie) vkEnumerateInstanceExtensionProperties(nullptr, &nie, ie.data());
    bool have_port = false;
    for (const auto& e : ie)
      if (std::strcmp(e.extensionName, kPortEnum) == 0) { have_port = true; break; }
    if (have_port) {
      ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
      ici.enabledExtensionCount = 1;
      ici.ppEnabledExtensionNames = &kPortEnum;
    }
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return;
    volkLoadInstance(instance);

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(instance, &n, nullptr) != VK_SUCCESS || n == 0) return;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());
    // Prefer a discrete GPU; else take the first with a compute queue.
    for (VkPhysicalDevice pd : devs) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(pd, &props);
      uint32_t nqf = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
      std::vector<VkQueueFamilyProperties> qfs(nqf);
      vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
      for (uint32_t i = 0; i < nqf; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          if (phys == VK_NULL_HANDLE ||
              props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            phys = pd;
            compute_qfi = i;
            std::memcpy(dev_name, props.deviceName,
                        sizeof(dev_name) < sizeof(props.deviceName)
                            ? sizeof(dev_name) : sizeof(props.deviceName));
            dev_name[sizeof(dev_name) - 1] = '\0';
          }
          break;
        }
      }
    }
    if (phys == VK_NULL_HANDLE) return;

    // A portability device (MoltenVK) advertises VK_KHR_portability_subset and the spec
    // REQUIRES enabling it at device creation if present. Probe + enable only then, so the
    // discrete-GPU platforms (which don't advertise it) request nothing extra.
    const char* kPortSubset = "VK_KHR_portability_subset";
    uint32_t nde = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, nullptr);
    std::vector<VkExtensionProperties> de(nde);
    if (nde) vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, de.data());
    bool need_subset = false;
    for (const auto& e : de)
      if (std::strcmp(e.extensionName, kPortSubset) == 0) { need_subset = true; break; }

    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = compute_qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &pri;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (need_subset) {
      dci.enabledExtensionCount = 1;
      dci.ppEnabledExtensionNames = &kPortSubset;
    }
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) return;
    volkLoadDevice(device);
    vkGetDeviceQueue(device, compute_qfi, 0, &queue);

    // Cache the device limits the launch path validates against (checklist #4 push size,
    // #20 workgroup count) -- read once here, not per-launch.
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    std::memcpy(cache_uuid, props.pipelineCacheUUID, VK_UUID_SIZE);  // #1 cache validity key

    // Shared pipeline cache (#1). If HYLE_VK_PIPELINE_CACHE names a file with a UUID-matching
    // blob, seed the cache from it (cold-start avoids re-compiling SPIR-V->ISA); else empty.
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    std::vector<uint8_t> seed;
    if (const char* cp = pipe_cache_path())
      if (load_pipe_cache(cp, cache_uuid, &seed) && !seed.empty()) {
        pcci.initialDataSize = seed.size();
        pcci.pInitialData = seed.data();
      }
    vkCreatePipelineCache(device, &pcci, nullptr, &pipe_cache);

    max_push = props.limits.maxPushConstantsSize;
    max_wg_count[0] = props.limits.maxComputeWorkGroupCount[0];
    max_wg_count[1] = props.limits.maxComputeWorkGroupCount[1];
    max_wg_count[2] = props.limits.maxComputeWorkGroupCount[2];
    // Buffer limits are device constants: cache here (like max_push above) so buffer_type
    // trait queries don't re-run vkGetPhysicalDeviceProperties on every Buffer::create.
    std::size_t a = props.limits.minStorageBufferOffsetAlignment;
    buf_alignment = a < 16 ? 16 : a;
    buf_max_size = props.limits.maxStorageBufferRange;
    non_coherent_atom_size = props.limits.nonCoherentAtomSize;
    // Timestamp support is a QUEUE-FAMILY property. timestampComputeAndGraphics only says
    // whether compute/graphics pipeline stages may be named; TOP/BOTTOM timestamps remain
    // usable when the selected queue family exposes timestampValidBits > 0. HYLE records at
    // BOTTOM_OF_PIPE, so key capability off the queue family rather than rejecting those
    // devices incorrectly.
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    if (nqf) vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs.data());
    if (compute_qfi < nqf) timestamp_valid_bits = qfs[compute_qfi].timestampValidBits;
    if (timestamp_valid_bits != 0)
      timestamp_period = props.limits.timestampPeriod;
    ok = true;
  }

  // Persist the pipeline cache on process exit (#1), if opt-in via HYLE_VK_PIPELINE_CACHE.
  // The Meyers-singleton dtor runs at exit while volk's device fns + our VkDevice are still
  // valid (same static lifetime). We only serialize the cache (not a full teardown of the
  // device/instance -- those are reclaimed by the OS at process exit); querying with a null
  // out-ptr gets the size, then we read the blob and frame it with the UUID header.
  ~VkContext() {
    if (!ok || pipe_cache == VK_NULL_HANDLE) return;
    const char* cp = pipe_cache_path();
    if (cp == nullptr) return;
    // No build_mtx here (unlike the flush hook): the Meyers-singleton dtor runs at process
    // exit AFTER worker threads/fibers have stopped, so no build races this read. Taking the
    // member mutex during its own object's destruction would be the riskier choice.
    std::size_t n = 0;
    if (vkGetPipelineCacheData(device, pipe_cache, &n, nullptr) != VK_SUCCESS || n == 0) return;
    std::vector<uint8_t> blob(n);
    if (vkGetPipelineCacheData(device, pipe_cache, &n, blob.data()) != VK_SUCCESS) return;
    blob.resize(n);  // n may shrink to the actual written size
    save_pipe_cache(cp, cache_uuid, blob);
  }
};

VkContext& ctx() noexcept {
  static VkContext c;
  return c;
}

// Pick a memory type index with ALL `want` bits, among `bits` allowed (checklist #6/#7).
// Returns UINT32_MAX if none (caller degrades). ReBAR (DEVICE_LOCAL|HOST_VISIBLE) is just
// another want-mask the caller can try first, then fall back.
uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags want) noexcept {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(ctx().phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  }
  return UINT32_MAX;
}

// ---- device enumeration ------------------------------------------------------
// One Vulkan device (the chosen compute GPU) when the context came up; else 0 (loader/
// device absent) -> vulkan_vtable() returns null, compute falls back.
std::size_t vk_device_count() noexcept { return ctx().ok ? 1 : 0; }
DeviceType vk_device_type(int) noexcept { return DeviceType::kGpu; }
const char* vk_device_name(int) noexcept { return ctx().dev_name; }
DeviceCaps vk_device_caps(int) noexcept {
  // Real async submit; host-visible staging available; fence-based events (query = poll).
  return DeviceCaps{/*async=*/true, /*host_buffer=*/true, /*events=*/true};
}
void vk_device_memory(int, std::uint64_t* free, std::uint64_t* total) noexcept {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(ctx().phys, &mp);
  std::uint64_t dl = 0;
  for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
    if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) dl += mp.memoryHeaps[i].size;
  if (total != nullptr) *total = dl;
  if (free != nullptr) *free = dl;  // Vulkan core has no live "free" query; report total
}

// ---- Stream = VkQueue + a per-stream VkCommandPool + in-flight tracking -------
// A HYLE Stream is a submission channel. Submits go through the shared compute VkQueue
// (VkQueue is externally synchronized -> ctx().submit_mtx guards it), but each Stream
// owns its OWN VkCommandPool (independent recording, checklist #14) AND tracks IN-FLIGHT
// submissions so launch() can return WITHOUT blocking (async, #21): each submit's
// command buffer + descriptor pool must outlive GPU execution, so we hold them keyed by
// the submit's fence and RETIRE (free) them once that fence signals. This is what makes
// `co_await stream` real -- launch submits + returns; the GpuPoller polls the fence; the
// resources are reclaimed lazily (next launch/sync) after completion. NO vkQueueWaitIdle
// on the launch hot path (that was the skeleton's blocking shortcut).
struct InFlight {
  VkFence fence = VK_NULL_HANDLE;   // the submit's completion fence (may be stream-owned or borrowed)
  VkCommandBuffer cb = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> dpools;  // descriptor pools to free on completion (0..N)
  bool owns_fence = true;           // false when `fence` is an Event's fence (borrowed; don't destroy)
};
struct VkStream {
  VkCommandPool pool = VK_NULL_HANDLE;
  std::vector<InFlight> inflight;   // submissions not yet known-complete
};

// Free every in-flight submission whose fence has signalled (GPU finished it). Called at
// launch entry + stream_sync; keeps the in-flight list from growing unbounded. `force`
// waits for all (stream_sync/destroy). Caller need not hold submit_mtx (per-stream state).
void retire(VkStream* s, bool force) noexcept {
  VkDevice dev = ctx().device;
  std::size_t keep = 0;
  for (auto& f : s->inflight) {
    bool done = vkGetFenceStatus(dev, f.fence) == VK_SUCCESS;
    if (!done && force) { vkWaitForFences(dev, 1, &f.fence, VK_TRUE, UINT64_MAX); done = true; }
    if (done) {
      if (f.cb != VK_NULL_HANDLE) vkFreeCommandBuffers(dev, s->pool, 1, &f.cb);
      for (VkDescriptorPool p : f.dpools) vkDestroyDescriptorPool(dev, p, nullptr);
      if (f.owns_fence) vkDestroyFence(dev, f.fence, nullptr);  // borrowed Event fences: skip
    } else {
      if (keep != &f - s->inflight.data()) s->inflight[keep] = std::move(f);
      ++keep;  // still running -> keep (move to compact, avoid copying the vector)
    }
  }
  s->inflight.resize(keep);
}

Status vk_stream_create(int, void** out_native) noexcept {
  auto* s = new (std::nothrow) VkStream{};
  if (s == nullptr) return Status::ResourceExhausted("vk stream alloc");
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.queueFamilyIndex = ctx().compute_qfi;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkResult r = vkCreateCommandPool(ctx().device, &ci, nullptr, &s->pool);
  if (r != VK_SUCCESS) { delete s; return from_vk(r, "vkCreateCommandPool"); }
  *out_native = s;
  return Status();
}

// Fence a precise queue position. A zero-command-buffer submit is sufficient: its fence is
// signalled only after all submissions queued before it complete. Unlike vkQueueWaitIdle this
// does not wait for work other threads enqueue after the marker while we are blocked.
Status wait_queue_position() noexcept {
  VkDevice dev = ctx().device;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  VkResult r = vkCreateFence(dev, &fci, nullptr, &fence);
  if (r != VK_SUCCESS) return from_vk(r, "vkCreateFence(stream sync)");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  {
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);
    r = vkQueueSubmit(ctx().queue, 1, &si, fence);
  }
  if (r == VK_SUCCESS)
    r = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  return from_vk(r, "vk stream sync marker");
}

void vk_stream_destroy(void* native) noexcept {
  auto* s = static_cast<VkStream*>(native);
  if (s == nullptr) return;
  retire(s, /*force=*/true);  // wait out + free all stream-owned in-flight submissions
  // Also drain untracked Event marker command buffers before destroying their pool.
  if (!wait_queue_position().ok()) {
    // Destructors cannot report allocation/device errors. Fall back to the broader wait so
    // the command pool is never destroyed while a marker command buffer may still execute.
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);
    (void)vkQueueWaitIdle(ctx().queue);
  }
  if (s->pool != VK_NULL_HANDLE) vkDestroyCommandPool(ctx().device, s->pool, nullptr);
  delete s;
}
Status vk_stream_sync(void* native) noexcept {
  // Retiring launch fences alone is insufficient: a timing/completion marker can be queued
  // after the last tracked launch. Fence the current queue position so sync really covers all
  // work submitted before the call, including those marker submits.
  auto* s = static_cast<VkStream*>(native);
  retire(s, /*force=*/true);
  return wait_queue_position();
}

// ---- Event = owned VkFence + record state ------------------------------------
// event_query = vkGetFenceStatus (measured non-blocking, == cudaEventQuery -> GpuPoller).
struct VkEvent {
  VkFence fence = VK_NULL_HANDLE;
  bool recorded = false;
};
Status vk_event_create(int, void** out_native) noexcept {
  auto* event = new (std::nothrow) VkEvent{};
  if (event == nullptr) return Status::ResourceExhausted("vk event alloc");
  VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkResult r = vkCreateFence(ctx().device, &ci, nullptr, &event->fence);
  if (r != VK_SUCCESS) {
    delete event;
    return from_vk(r, "vkCreateFence");
  }
  *out_native = event;
  return Status();
}
void vk_event_destroy(void* native) noexcept {
  auto* event = static_cast<VkEvent*>(native);
  if (event == nullptr) return;
  // RAII destruction must not destroy a fence still referenced by a queue submission.
  if (event->recorded)
    vkWaitForFences(ctx().device, 1, &event->fence, VK_TRUE, UINT64_MAX);
  if (event->fence != VK_NULL_HANDLE)
    vkDestroyFence(ctx().device, event->fence, nullptr);
  delete event;
}
// Record `event` on `stream`: the Vulkan analog of cudaEventRecord ("mark THIS point in
// the stream"). Submits an EMPTY command buffer signalling the event's fence. Because all
// submits on one VkQueue execute in submission order, this fence signals only after every
// prior launch on the stream has finished -- exactly cudaEventRecord's semantics, so
// `co_await StreamComplete` (create event -> record -> poll query) works unchanged.
//
// LIFETIME (why the marker cb is NOT parked in `inflight`): the event's fence is owned by
// the Event object, which the awaiting coroutine destroys right after the wait completes.
// Parking the marker cb in `inflight` keyed on that fence would make a later retire()/
// stream_destroy vkGetFenceStatus a DESTROYED fence = use-after-free (faults on Windows,
// silently "works" on Linux -- caught by testing BOTH). Instead the marker cb is left
// owned by the stream's COMMAND POOL, reclaimed wholesale at vkDestroyCommandPool
// (stream_destroy force-retires + waits stream-owned fences first). Markers accumulate over
// the stream's life (bounded by #co_awaits); per-marker reclaim is deferred, not needed
// for correctness.
void vk_event_record(void* stream_native, void* event_native) noexcept {
  auto* s = static_cast<VkStream*>(stream_native);
  auto* event = static_cast<VkEvent*>(event_native);
  VkDevice dev = ctx().device;
  // Re-recording a pending Vulkan fence is illegal. Match reusable Event semantics by
  // completing the previous record before resetting the native fence.
  if (event->recorded)
    vkWaitForFences(dev, 1, &event->fence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &event->fence);
  // Build the marker command buffer. If ANY step fails we must STILL signal the fence (we
  // already reset it above) -- otherwise the GpuPoller polls it forever and the awaiting
  // fiber hangs (gpu_await.h contract: a recorded event ALWAYS completes). So on failure
  // we fall through to a 0-command-buffer submit that just signals the fence: the fiber
  // resumes and observes completion via its own path, no silent hang.
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = s->pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  bool have_cb = vkAllocateCommandBuffers(dev, &cbai, &cb) == VK_SUCCESS;
  if (have_cb) {
    VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    have_cb = vkBeginCommandBuffer(cb, &bg) == VK_SUCCESS &&
              vkEndCommandBuffer(cb) == VK_SUCCESS;
    if (!have_cb) {
      vkFreeCommandBuffers(dev, s->pool, 1, &cb);
      cb = VK_NULL_HANDLE;
    }
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = have_cb ? 1u : 0u;
  si.pCommandBuffers = have_cb ? &cb : nullptr;
  std::lock_guard<std::mutex> lk(ctx().submit_mtx);
  // Even the empty (have_cb==false) submit correctly signals the fence AFTER prior stream
  // launches: those were separate submits already queued, and this marker executes after
  // them in VkQueue order -- so readback still sees completed data. A submit failure here
  // is device-lost class (fence can't signal); nothing short of that leaves ev unsignalled.
  event->recorded =
      vkQueueSubmit(ctx().queue, 1, &si, event->fence) == VK_SUCCESS;
  // cb NOT tracked in inflight (see LIFETIME note) -- the command pool reclaims it.
}
EventStatus vk_event_query(void* event_native) noexcept {
  auto* event = static_cast<VkEvent*>(event_native);
  HYLE_FATAL_ASSERT(event->recorded);
  VkResult r = vkGetFenceStatus(ctx().device, event->fence);
  return r == VK_SUCCESS ? EventStatus::kReady : EventStatus::kNotReady;
}
void vk_event_sync(void* event_native) noexcept {
  auto* event = static_cast<VkEvent*>(event_native);
  HYLE_FATAL_ASSERT(event->recorded);
  vkWaitForFences(ctx().device, 1, &event->fence, VK_TRUE, UINT64_MAX);
}

// ---- GPU timing (compute-06b #25): timestamp query + fence ---------------------
// A timing event = a VkFence (completion, like a plain event) + a 1-slot timestamp query
// pool. record_timing submits a marker cb that resets the pool + writes a timestamp at the
// bottom of pipe. BOTTOM is essential: a later submit's TOP stage may overlap earlier compute,
// so TOP does not represent "all work before this event completed". elapsed_ns reads two
// completed timing events' ticks and
// scales the difference by the device timestampPeriod. The native_ handle is a heap
// VkTimingEvent* (distinct from the plain VkEvent layout; the public layer routes them by
// Event::timing_).
struct VkTimingEvent {
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool pool = VK_NULL_HANDLE;
  bool recorded = false;
  bool timestamp_written = false;
};
Status vk_event_create_timing(int, void** out_native) noexcept {
  if (ctx().timestamp_period == 0.0f)
    return Status::Unimplemented("vk backend: device has no timestamp support");
  auto* t = new (std::nothrow) VkTimingEvent{};
  if (t == nullptr) return Status::ResourceExhausted("vk timing event alloc");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkResult r = vkCreateFence(ctx().device, &fci, nullptr, &t->fence);
  if (r != VK_SUCCESS) { delete t; return from_vk(r, "vkCreateFence(timing)"); }
  VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qci.queryCount = 1;
  r = vkCreateQueryPool(ctx().device, &qci, nullptr, &t->pool);
  if (r != VK_SUCCESS) { vkDestroyFence(ctx().device, t->fence, nullptr); delete t;
                         return from_vk(r, "vkCreateQueryPool"); }
  *out_native = t;
  return Status();
}
void vk_event_destroy_timing(void* native) noexcept {
  auto* t = static_cast<VkTimingEvent*>(native);
  if (t == nullptr) return;
  if (t->recorded)
    vkWaitForFences(ctx().device, 1, &t->fence, VK_TRUE, UINT64_MAX);
  if (t->pool != VK_NULL_HANDLE) vkDestroyQueryPool(ctx().device, t->pool, nullptr);
  if (t->fence != VK_NULL_HANDLE) vkDestroyFence(ctx().device, t->fence, nullptr);
  delete t;
}
void vk_event_record_timing(void* stream_native, void* event_native) noexcept {
  auto* s = static_cast<VkStream*>(stream_native);
  auto* t = static_cast<VkTimingEvent*>(event_native);
  VkDevice dev = ctx().device;
  if (t->recorded)
    vkWaitForFences(dev, 1, &t->fence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &t->fence);
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = s->pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  bool have_cb = vkAllocateCommandBuffers(dev, &cbai, &cb) == VK_SUCCESS;
  if (have_cb) {
    VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    have_cb = vkBeginCommandBuffer(cb, &bg) == VK_SUCCESS;
    if (have_cb) {
      vkCmdResetQueryPool(cb, t->pool, 0, 1);  // must reset before write
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, t->pool, 0);
      have_cb = vkEndCommandBuffer(cb) == VK_SUCCESS;
    }
    if (!have_cb) {
      vkFreeCommandBuffers(dev, s->pool, 1, &cb);
      cb = VK_NULL_HANDLE;
    }
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = have_cb ? 1u : 0u;
  si.pCommandBuffers = have_cb ? &cb : nullptr;
  std::lock_guard<std::mutex> lk(ctx().submit_mtx);
  t->timestamp_written = have_cb;
  t->recorded = vkQueueSubmit(ctx().queue, 1, &si, t->fence) == VK_SUCCESS;
  // marker cb reclaimed by command pool (see plain event LIFETIME)
}
EventStatus vk_event_query_timing(void* event_native) noexcept {
  auto* t = static_cast<VkTimingEvent*>(event_native);
  HYLE_FATAL_ASSERT(t->recorded);
  VkResult r = vkGetFenceStatus(ctx().device, t->fence);
  return r == VK_SUCCESS ? EventStatus::kReady : EventStatus::kNotReady;
}
void vk_event_sync_timing(void* event_native) noexcept {
  auto* t = static_cast<VkTimingEvent*>(event_native);
  HYLE_FATAL_ASSERT(t->recorded);
  vkWaitForFences(ctx().device, 1, &t->fence, VK_TRUE, UINT64_MAX);
}
Status vk_event_elapsed_ns(void* start_native, void* stop_native,
                           std::uint64_t* out_ns) noexcept {
  auto* a = static_cast<VkTimingEvent*>(start_native);
  auto* b = static_cast<VkTimingEvent*>(stop_native);
  if (!a->timestamp_written || !b->timestamp_written)
    return Status::ResourceExhausted("vk timing marker command buffer unavailable");
  VkDevice dev = ctx().device;
  std::uint64_t t0 = 0, t1 = 0;
  VkResult r = vkGetQueryPoolResults(dev, a->pool, 0, 1, sizeof(t0), &t0, sizeof(t0),
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (r != VK_SUCCESS) return from_vk(r, "vkGetQueryPoolResults(start)");
  r = vkGetQueryPoolResults(dev, b->pool, 0, 1, sizeof(t1), &t1, sizeof(t1),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (r != VK_SUCCESS) return from_vk(r, "vkGetQueryPoolResults(stop)");
  // Vulkan exposes only timestampValidBits low bits. Modular subtraction handles a single
  // counter wrap; the API contract requires ordered events on one stream, so a reversed pair
  // is caller misuse rather than something to silently clamp to zero.
  const uint32_t bits = ctx().timestamp_valid_bits;
  const std::uint64_t mask = bits >= 64 ? UINT64_MAX : ((std::uint64_t{1} << bits) - 1);
  const std::uint64_t ticks = (t1 - t0) & mask;
  *out_ns = static_cast<std::uint64_t>(static_cast<double>(ticks) * ctx().timestamp_period);
  return Status();
}

// ---- raw device alloc + async copy (compute-01 minimal ops) ------------------
// These predate the Buffer layer; for Vulkan they map onto a device-local VkBuffer.
// A raw "device pointer" in Vulkan isn't a real address, so we heap a VkBuffer+memory
// and hand back the pointer to a small record. Kept minimal -- the real path is Buffer.
struct VkMem {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* mapped = nullptr;      // persistent map if PRIMARY is host-visible (ReBAR, #9), else null
  std::size_t size = 0;         // logical Buffer size
  VkDeviceSize allocation_size = 0;  // VkDeviceMemory size (may exceed logical size)
  // Lazy synchronous-transfer staging for a device-local buffer that is NOT host-mappable
  // (no ReBAR). It is created on first host transfer and grows to the largest transfer, not
  // eagerly to the full device-buffer size. A future asynchronous upload path is a distinct
  // Event-owned StagingRing; this buffer never pretends to provide async overlap.
  VkBuffer staging_buf = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  void* staging_mapped = nullptr;  // persistent map of the staging buffer (host-visible+coherent)
  VkDeviceSize staging_capacity = 0;
  // #6 HOST_CACHED: the mapped PRIMARY (kHostPinned) may be HOST_CACHED for fast CPU readback.
  // HOST_CACHED memory is not guaranteed HOST_COHERENT, so a non-coherent map needs an explicit
  // flush after a host write (#10) and invalidate before a host read. false only for that case;
  // ReBAR device maps + staging are always coherent (true). The staging buffer is always chosen
  // COHERENT, so this flag governs only the `mapped` primary.
  bool coherent = true;
};

void free_mem(VkMem* m) noexcept;  // fwd: make_mem calls it to clean up partial allocations

// Create a VkBuffer + backing memory of `bytes`. `host_visible` picks the memory class:
// host-visible+coherent (staging / kHostPinned) vs device-local, trying ReBAR first for
// device buffers so host set/get can map directly (checklist #6 ReBAR + #9 persistent map).
Status make_mem(std::size_t bytes, bool host_visible, VkMem** out) noexcept {
  auto* m = new (std::nothrow) VkMem{};
  if (m == nullptr) return Status::ResourceExhausted("vk mem alloc");
  m->size = bytes;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = bytes ? bytes : 1;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult r = vkCreateBuffer(ctx().device, &bi, nullptr, &m->buf);
  if (r != VK_SUCCESS) { delete m; return from_vk(r, "vkCreateBuffer"); }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(ctx().device, m->buf, &mr);

  uint32_t mt = UINT32_MAX;
  bool will_map = host_visible;
  if (host_visible) {
    // #6 HOST_CACHED: kHostPinned is the readback/staging tier, so prefer HOST_CACHED (fast CPU
    // reads of D2H results) when present. Try CACHED|COHERENT (best: cached + no manual flush),
    // then CACHED alone (cached but non-coherent -> set/get flush/invalidate, #10), then plain
    // COHERENT (the original path). m->coherent tracks whether the chosen type is coherent.
    mt = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
#if defined(HYLE_TESTING)
    // TEST-ONLY: scan every compatible type instead of asking find_mem_type() for the first
    // match (which may be coherent while a later non-coherent type exists). Prefer cached,
    // then accept any non-coherent host-visible type so the visibility path is exercised
    // whenever the physical device can provide it. Absent from the shipped library.
    if (std::getenv("HYLE_VK_FORCE_NONCOHERENT")) {
      VkPhysicalDeviceMemoryProperties mp;
      vkGetPhysicalDeviceMemoryProperties(ctx().phys, &mp);
      auto find_noncoherent = [&](VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
          const VkMemoryPropertyFlags flags = mp.memoryTypes[i].propertyFlags;
          if ((mr.memoryTypeBits & (1u << i)) && (flags & want) == want &&
              !(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            return i;
        }
        return UINT32_MAX;
      };
      uint32_t nc = find_noncoherent(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      if (nc == UINT32_MAX)
        nc = find_noncoherent(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      if (nc != UINT32_MAX) mt = nc;
    }
#endif
    if (mt == UINT32_MAX)  // no cached type: try cached-non-coherent, then plain coherent
      mt = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (mt == UINT32_MAX)
      mt = find_mem_type(mr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt != UINT32_MAX) {  // record coherence of the chosen type (governs flush/invalidate)
      VkPhysicalDeviceMemoryProperties mp;
      vkGetPhysicalDeviceMemoryProperties(ctx().phys, &mp);
      m->coherent = (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
  } else {
    // Device buffer: try ReBAR (DEVICE_LOCAL|HOST_VISIBLE|COHERENT) first so set/get can
    // map directly with no staging copy; fall back to plain DEVICE_LOCAL (checklist #6/#7).
    mt = find_mem_type(mr.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt != UINT32_MAX) will_map = true;  // ReBAR -> host-mappable
    else { mt = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
           will_map = false; }
#if defined(HYLE_TESTING)
    // TEST-ONLY: on a ReBAR box (dev machines: RTX 4070) the device path always maps
    // directly, so the staging fallback (kompute #8/#13, for ordinary non-ReBAR GPUs)
    // is never exercised. HYLE_VK_FORCE_STAGING=1 forces the non-mapped path + plain
    // DEVICE_LOCAL so tests cover staging on hardware that wouldn't otherwise hit it.
    // Compiled ONLY under HYLE_TESTING (HYLE_BUILD_TESTS); absent from the shipped lib.
    if (will_map && std::getenv("HYLE_VK_FORCE_STAGING")) {
      uint32_t dl = find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (dl != UINT32_MAX) { mt = dl; will_map = false; }  // pretend "no ReBAR"
    }
#endif
  }
  if (mt == UINT32_MAX) {
    vkDestroyBuffer(ctx().device, m->buf, nullptr); delete m;
    return Status::ResourceExhausted("vk: no suitable memory type");
  }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = mt;
  m->allocation_size = ai.allocationSize;
  r = vkAllocateMemory(ctx().device, &ai, nullptr, &m->mem);
  if (r != VK_SUCCESS) { vkDestroyBuffer(ctx().device, m->buf, nullptr); delete m;
                         return from_vk(r, "vkAllocateMemory"); }
  r = vkBindBufferMemory(ctx().device, m->buf, m->mem, 0);
  if (r != VK_SUCCESS) { free_mem(m); return from_vk(r, "vkBindBufferMemory"); }
  if (will_map) {
    // Persistent map (#9). A map FAILURE (host address space / driver limits) must not be
    // ignored: leaving m->mapped null while will_map=true would skip the staging fallback,
    // then vk_buffer_set/get would deref a null staging_mapped -> crash. Fail at the boundary.
    r = vkMapMemory(ctx().device, m->mem, 0, VK_WHOLE_SIZE, 0, &m->mapped);
    if (r != VK_SUCCESS) { free_mem(m); return from_vk(r, "vkMapMemory"); }
  }

  *out = m;
  return Status();
}
void free_mem(VkMem* m) noexcept {
  if (m == nullptr) return;
  // Staging first (built last); each field guarded so a partial make_mem failure is safe.
  if (m->staging_mapped != nullptr) vkUnmapMemory(ctx().device, m->staging_mem);
  if (m->staging_buf != VK_NULL_HANDLE) vkDestroyBuffer(ctx().device, m->staging_buf, nullptr);
  if (m->staging_mem != VK_NULL_HANDLE) vkFreeMemory(ctx().device, m->staging_mem, nullptr);
  if (m->mapped != nullptr) vkUnmapMemory(ctx().device, m->mem);
  if (m->buf != VK_NULL_HANDLE) vkDestroyBuffer(ctx().device, m->buf, nullptr);
  if (m->mem != VK_NULL_HANDLE) vkFreeMemory(ctx().device, m->mem, nullptr);
  delete m;
}

// Ensure a coherent, persistently-mapped staging slice for one synchronous transfer.
// Build the replacement completely before dropping the old slice so OOM leaves the Buffer
// usable with its previous capacity. All callers wait for their copy before returning, hence
// replacement cannot race an in-flight transfer under Buffer's single-owner contract.
Status ensure_staging(VkMem* m, VkDeviceSize bytes) noexcept {
  if (bytes == 0 || m->staging_capacity >= bytes) return Status();
  VkDevice dev = ctx().device;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;

  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = bytes;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult r = vkCreateBuffer(dev, &bi, nullptr, &buffer);
  if (r != VK_SUCCESS) return from_vk(r, "vkCreateBuffer(staging)");

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buffer, &mr);
  const uint32_t mt = find_mem_type(
      mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mt == UINT32_MAX) {
    vkDestroyBuffer(dev, buffer, nullptr);
    return Status::ResourceExhausted("vk: no host-visible staging memory type");
  }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = mt;
  r = vkAllocateMemory(dev, &ai, nullptr, &memory);
  if (r == VK_SUCCESS) r = vkBindBufferMemory(dev, buffer, memory, 0);
  if (r == VK_SUCCESS) r = vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
  if (r != VK_SUCCESS) {
    if (mapped != nullptr) vkUnmapMemory(dev, memory);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(dev, memory, nullptr);
    vkDestroyBuffer(dev, buffer, nullptr);
    return from_vk(r, "vk staging allocation");
  }

  if (m->staging_mapped != nullptr) vkUnmapMemory(dev, m->staging_mem);
  if (m->staging_buf != VK_NULL_HANDLE) vkDestroyBuffer(dev, m->staging_buf, nullptr);
  if (m->staging_mem != VK_NULL_HANDLE) vkFreeMemory(dev, m->staging_mem, nullptr);
  m->staging_buf = buffer;
  m->staging_mem = memory;
  m->staging_mapped = mapped;
  m->staging_capacity = bytes;
  return Status();
}

// One-shot SYNCHRONOUS transfer: allocate a transient command buffer, let `rec` record
// transfer commands into it, submit on the compute queue, and WAIT for completion (own
// fence). This backs the SYNCHRONOUS buffer_set/get/copy on the staging path -- those
// have no Stream arg and are contractually synchronous (data in place on return, same as
// the CUDA backend's cudaMemcpyAsync+synchronize). NOT a hot path (the async kernel path
// is vk_kernel_launch); a transient pool per call keeps it self-contained. Returns a
// Status; frees everything before returning (success or failure).
template <class Rec>
Status staged_submit(Rec&& rec) noexcept {
  VkDevice dev = ctx().device;
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = ctx().compute_qfi;
  pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkResult r = vkCreateCommandPool(dev, &pci, nullptr, &pool);
  if (r != VK_SUCCESS) return from_vk(r, "vkCreateCommandPool(staged)");
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(dev, &cbai, &cb);
  if (r != VK_SUCCESS) { vkDestroyCommandPool(dev, pool, nullptr); return from_vk(r, "vkAllocateCommandBuffers(staged)"); }
  VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bg);
  rec(cb);
  vkEndCommandBuffer(cb);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  r = vkCreateFence(dev, &fci, nullptr, &fence);
  if (r != VK_SUCCESS) { vkDestroyCommandPool(dev, pool, nullptr); return from_vk(r, "vkCreateFence(staged)"); }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  {
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);  // VkQueue is externally synchronized
    r = vkQueueSubmit(ctx().queue, 1, &si, fence);
  }
  if (r == VK_SUCCESS) vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  vkDestroyCommandPool(dev, pool, nullptr);  // frees cb
  return from_vk(r, "vkQueueSubmit(staged)");
}

// Record a buffer copy with the barriers the staging D2H path needs (kompute OpSyncLocal):
// shaderWrite->transferRead before, transferWrite->hostRead after. For H2D the pre-barrier
// isn't needed (host write visibility comes from queue submit) but a transferWrite->
// shaderRead after makes the primary readable by a later kernel.
void record_copy(VkCommandBuffer cb, VkBuffer from, VkBuffer to,
                 std::size_t from_offset, std::size_t to_offset,
                 std::size_t bytes, bool d2h) noexcept {
  auto barrier = [&](VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags ss,
                     VkPipelineStageFlags ds) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = src; mb.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 1, &mb, 0, nullptr, 0, nullptr);
  };
  if (d2h)  // prior shader/transfer write -> make the primary transfer-readable
    barrier(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferCopy region{from_offset, to_offset, bytes};
  vkCmdCopyBuffer(cb, from, to, 1, &region);
  if (d2h)  // staging written by the copy -> make it host-readable
    barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
  else      // primary written by the copy -> make it shader-readable by a later kernel
    barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

// #6/#10 non-coherent HOST_CACHED: flush after a host WRITE (make it visible to the GPU) /
// invalidate before a host READ (pull the GPU's writes into the CPU cache). No-ops when the
// memory is coherent (the common case: ReBAR, staging, and CACHED|COHERENT types). Vulkan
// requires offset alignment to nonCoherentAtomSize and either an atom-multiple size or a range
// ending at the allocation boundary, so expand only the requested logical sub-range.
VkMappedMemoryRange mapped_range(const VkMem* m, std::size_t offset,
                                 std::size_t size) noexcept {
  const VkDeviceSize atom = ctx().non_coherent_atom_size;
  const VkDeviceSize begin = static_cast<VkDeviceSize>(offset) & ~(atom - 1);
  const VkDeviceSize end = static_cast<VkDeviceSize>(offset) + size;
  VkDeviceSize aligned_end = end;
  const VkDeviceSize remainder = end & (atom - 1);
  if (remainder != 0) {
    const VkDeviceSize padding = atom - remainder;
    aligned_end = padding > m->allocation_size - end
                      ? m->allocation_size
                      : end + padding;
  }
  if (aligned_end > m->allocation_size) aligned_end = m->allocation_size;
  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = m->mem;
  range.offset = begin;
  range.size = aligned_end - begin;
  return range;
}

Status flush_if_needed(const VkMem* m, std::size_t offset,
                       std::size_t size) noexcept {
  if (m->coherent || size == 0) return Status();
  VkMappedMemoryRange range = mapped_range(m, offset, size);
  return from_vk(vkFlushMappedMemoryRanges(ctx().device, 1, &range),
                 "vkFlushMappedMemoryRanges");
}
Status invalidate_if_needed(const VkMem* m, std::size_t offset,
                            std::size_t size) noexcept {
  if (m->coherent || size == 0) return Status();
  VkMappedMemoryRange range = mapped_range(m, offset, size);
  return from_vk(vkInvalidateMappedMemoryRanges(ctx().device, 1, &range),
                 "vkInvalidateMappedMemoryRanges");
}

// ---- buffer traits + alloc (compute-02) --------------------------------------
std::size_t vk_buffer_alignment(BufferKind) noexcept { return ctx().buf_alignment; }
std::size_t vk_buffer_max_size(int, BufferKind) noexcept { return ctx().buf_max_size; }
bool vk_buffer_is_host(BufferKind kind) noexcept { return kind == BufferKind::kHostPinned; }

Status vk_buffer_alloc(int, BufferKind kind, std::size_t bytes,
                       void** out_native, void** out_host) noexcept {
  VkMem* m = nullptr;
  Status s = make_mem(bytes, /*host_visible=*/kind == BufferKind::kHostPinned, &m);
  if (!s.ok()) return s;
  *out_native = m;
  *out_host = kind == BufferKind::kHostPinned ? m->mapped : nullptr;
  HYLE_FATAL_ASSERT(kind != BufferKind::kHostPinned || *out_host != nullptr);
  return Status();
}
void vk_buffer_free(int, BufferKind, void* ptr) noexcept { free_mem(static_cast<VkMem*>(ptr)); }
Status vk_buffer_flush_host(void* native, std::size_t offset,
                            std::size_t size) noexcept {
  auto* m = static_cast<VkMem*>(native);
  HYLE_FATAL_ASSERT(m != nullptr && m->mapped != nullptr);
  return flush_if_needed(m, offset, size);
}
Status vk_buffer_invalidate_host(void* native, std::size_t offset,
                                 std::size_t size) noexcept {
  auto* m = static_cast<VkMem*>(native);
  HYLE_FATAL_ASSERT(m != nullptr && m->mapped != nullptr);
  return invalidate_if_needed(m, offset, size);
}

// set/get/copy/clear: SYNCHRONOUS host<->device at a byte offset (compute-02 contract:
// data in place on return). TWO paths per the memory class:
//   - MAPPED primary (host-pinned, or ReBAR device): direct memcpy into the persistent
//     map (checklist #9). Zero GPU work -- the fast path on ReBAR hardware (RTX 4070).
//   - NON-MAPPED device (plain DEVICE_LOCAL, no ReBAR): lazily grow a transfer-sized staging
//     slice and reuse it; do not reserve one full host allocation per device buffer.
//     (kompute #8/#13): memcpy host<->staging map, then a GPU copyBuffer staging<->primary
//     via staged_submit (one-shot cb + wait). This is what makes the Vulkan backend work
//     on ordinary discrete GPUs / iGPUs, not just ReBAR cards.
Status vk_buffer_set(void* dst_base, std::size_t offset, const void* src,
                     std::size_t size) noexcept {
  auto* m = static_cast<VkMem*>(dst_base);
  if (m->mapped != nullptr) {  // ReBAR / host-visible: direct write
    std::memcpy(static_cast<std::byte*>(m->mapped) + offset, src, size);
    return flush_if_needed(m, offset, size);  // non-coherent HOST_CACHED -> publish the write
  }
  // Staging H2D: host -> staging map, then GPU copy staging -> primary (at `offset`).
  Status ready = ensure_staging(m, size);
  if (!ready.ok()) return ready;
  std::memcpy(m->staging_mapped, src, size);
  return staged_submit([&](VkCommandBuffer cb) {
    VkBufferCopy region{0, offset, size};  // staging[0] -> primary[offset]
    vkCmdCopyBuffer(cb, m->staging_buf, m->buf, 1, &region);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;   // primary now readable by a kernel
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
  });
}
Status vk_buffer_get(const void* src_base, std::size_t offset, void* dst,
                     std::size_t size) noexcept {
  auto* m = static_cast<VkMem*>(const_cast<void*>(src_base));
  if (m->mapped != nullptr) {  // ReBAR / host-visible: direct read
    Status visibility = invalidate_if_needed(m, offset, size);
    if (!visibility.ok()) return visibility;
    std::memcpy(dst, static_cast<std::byte*>(m->mapped) + offset, size);
    return Status();
  }
  // Staging D2H (kompute OpSyncLocal barrier chain): GPU copy primary -> staging with
  // shaderWrite->transferRead before + transferWrite->hostRead after, then host reads map.
  Status ready = ensure_staging(m, size);
  if (!ready.ok()) return ready;
  Status s = staged_submit([&](VkCommandBuffer cb) {
    record_copy(cb, m->buf, m->staging_buf, offset, 0, size,
                /*d2h=*/true);
  });
  if (!s.ok()) return s;
  std::memcpy(dst, m->staging_mapped, size);
  return Status();
}
Status vk_buffer_copy(void* dst_base, std::size_t dst_off, const void* src_base,
                      std::size_t src_off, std::size_t size) noexcept {
  auto* d = static_cast<VkMem*>(dst_base);
  auto* s = static_cast<VkMem*>(const_cast<void*>(src_base));
  if (d->mapped != nullptr && s->mapped != nullptr) {  // both host-mapped: CPU memcpy
    Status visibility = invalidate_if_needed(s, src_off, size);
    if (!visibility.ok()) return visibility;
    std::memcpy(static_cast<std::byte*>(d->mapped) + dst_off,
                static_cast<std::byte*>(s->mapped) + src_off, size);
    return flush_if_needed(d, dst_off, size);
  }
  // At least one is device-local: GPU copyBuffer src.primary -> dst.primary (D2D stays on
  // device -- no host round-trip). Order prior shader/host writes before the transfer, then
  // make the transfer write visible to a later kernel / readback.
  return staged_submit([&](VkCommandBuffer cb) {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before,
                         0, nullptr, 0, nullptr);
    VkBufferCopy region{src_off, dst_off, size};
    vkCmdCopyBuffer(cb, s->buf, d->buf, 1, &region);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
  });
}
Status vk_buffer_clear(void* base, std::size_t offset, std::uint8_t value,
                       std::size_t size) noexcept {
  auto* m = static_cast<VkMem*>(base);
  if (m->mapped != nullptr) {  // host-mapped: direct memset
    std::memset(static_cast<std::byte*>(m->mapped) + offset, value, size);
    return flush_if_needed(m, offset, size);
  }
  // Device-local: vkCmdFillBuffer writes a 32-bit pattern. buffer_clear fills whole bytes
  // with `value`, so replicate the byte across the 4-byte word. offset/size are byte-exact
  // here (the caller's contract keeps them 4-aligned for device buffers; VkFillBuffer
  // requires 4-aligned offset + size, so assert to fail-fast on a misaligned caller bug).
  HYLE_FATAL_ASSERT((offset % 4 == 0) && (size % 4 == 0));
  uint32_t word = value; word |= word << 8; word |= word << 16;
  return staged_submit([&](VkCommandBuffer cb) {
    vkCmdFillBuffer(cb, m->buf, offset, size, word);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
  });
}

// ---- device pool (compute-02c): cached VkBuffer blocks -----------------------
// Vulkan has no CUDA-VMM-equivalent path here, so use the honest analogue of the legacy
// pool: retain freed VkBuffer+VkDeviceMemory objects and best-fit reuse them. This preserves
// DevicePool's core promise (scratch churn avoids driver allocation) instead of presenting a
// pool-shaped API that called vkAllocateMemory/vkFreeMemory on every operation.
struct VkPool {
  static constexpr std::size_t kMaxBuffers = 256;
  struct Block {
    VkMem* mem = nullptr;
    std::size_t capacity = 0;
  };
  Block free_blocks[kMaxBuffers] = {};

  ~VkPool() {
    for (const Block& b : free_blocks) free_mem(b.mem);
  }
};

Status vk_pool_create(int, void** out_handle) noexcept {
  auto* pool = new (std::nothrow) VkPool{};
  if (pool == nullptr) return Status::ResourceExhausted("vk pool alloc");
  *out_handle = pool;
  return Status();
}
void vk_pool_destroy(void* handle) noexcept { delete static_cast<VkPool*>(handle); }
bool vk_pool_is_vmm(void*) noexcept { return false; }
Status vk_pool_alloc(void* handle, std::size_t size, void** out_ptr,
                     std::size_t* out_actual) noexcept {
  auto* pool = static_cast<VkPool*>(handle);
  const std::size_t needed = size == 0 ? 1 : size;
  std::size_t best = SIZE_MAX;
  std::size_t best_capacity = SIZE_MAX;
  for (std::size_t i = 0; i < VkPool::kMaxBuffers; ++i) {
    const std::size_t capacity = pool->free_blocks[i].capacity;
    if (capacity >= needed && capacity < best_capacity) {
      best = i;
      best_capacity = capacity;
      if (capacity == needed) break;
    }
  }
  if (best != SIZE_MAX) {
    VkPool::Block block = pool->free_blocks[best];
    pool->free_blocks[best] = {};
    *out_ptr = block.mem;
    if (out_actual != nullptr) *out_actual = block.capacity;
    return Status();
  }
  VkMem* m = nullptr;
  Status s = make_mem(size, /*host_visible=*/false, &m);
  if (!s.ok()) return s;
  *out_ptr = m;
  if (out_actual != nullptr) *out_actual = needed;
  return Status();
}
void vk_pool_free(void* handle, void* ptr, std::size_t) noexcept {
  if (ptr == nullptr) return;
  auto* pool = static_cast<VkPool*>(handle);
  auto* mem = static_cast<VkMem*>(ptr);
  for (VkPool::Block& block : pool->free_blocks) {
    if (block.mem == nullptr) {
      block = {mem, mem->size == 0 ? 1 : mem->size};
      return;
    }
  }
  free_mem(mem); // cache full: return the block to Vulkan
}

// ---- Kernel = SPIR-V module + pipeline + descriptor/pipeline layout ----------
// A HYLE Kernel's {module, func} handles hold: module = a heap VkKernel (all the pipeline
// objects); func = unused (nullptr). compile (GLSL source) is Unimplemented here -- Vulkan
// kernels are pre-compiled to SPIR-V (spec ④ pre-compile) and loaded via load_ptx, whose
// `ptx` arg we reinterpret as SPIR-V bytes. (A future runtime GLSL->SPIR-V via glslang's C
// interface is the compile() path; deferred.)
struct VkKernel {
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  uint32_t buffer_count = 0;   // descriptor bindings (set at load from SPIR-V reflection-lite)
  uint32_t push_size = 0;      // push-constant bytes
  // Specialization constants (#2/#3): copied from the load_ptx args (caller storage is
  // transient), consumed once in build_pipeline. entries[i] = {constant_id, offset, size}
  // into `spec_data`; empty (spec_map.empty()) = no spec constants.
  std::vector<VkSpecializationMapEntry> spec_map;
  std::vector<uint8_t> spec_data;
};

// Build a compute pipeline from SPIR-V. Because HYLE's Kernel API doesn't carry a
// separate binding count / push size, we derive them at LAUNCH from the KernelArgs
// (buffer_count + scalar_bytes) -- so load_ptx just makes the shader module and defers
// layout/pipeline creation to first launch (when we know the arg shape). We stash the
// SPIR-V module here; the pipeline is lazily built + cached on the VkKernel.
// `spirv`/`spirv_words`: SPIR-V passed through the `ptx` string arg (byte length via the
// entry-name convention is fragile, so we treat the C-string as raw bytes up to a NUL-safe
// length is WRONG for SPIR-V binary -- instead load_ptx receives the byte count via a
// length-prefixed contract documented in the header). For the first consumer the test
// passes SPIR-V through a dedicated path (see vulkan-specific load below).
Status build_pipeline(VkKernel* k, uint32_t buffer_count, uint32_t push_size) noexcept {
  // Serialize: VkPipelineCache host access is externally-synchronized (Vulkan spec), and
  // this guards the lazy per-Kernel build against concurrent first-launch from multiple
  // fibers. Double-check under the lock -- if another fiber already built this kernel while
  // we waited, use its result (no rebuild -> no leak / no torn k->pipeline write).
  std::lock_guard<std::mutex> lk(ctx().build_mtx);
  if (k->pipeline != VK_NULL_HANDLE) return Status();  // already built by a racing caller
  // Push-constant blob must fit the device limit (checklist #4). Exceeding it is a caller
  // contract violation (a kernel with too many scalars should use a uniform buffer) ->
  // fail-fast, not a silent vkCreatePipelineLayout failure.
  HYLE_FATAL_ASSERT(push_size <= ctx().max_push);
  // Descriptor set layout: `buffer_count` storage buffers at bindings 0..N-1 (checklist:
  // buffer args -> descriptor slots, positional).
  std::vector<VkDescriptorSetLayoutBinding> binds(buffer_count);
  for (uint32_t i = 0; i < buffer_count; ++i)
    binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = buffer_count;
  dlci.pBindings = buffer_count ? binds.data() : nullptr;
  VkResult r = vkCreateDescriptorSetLayout(ctx().device, &dlci, nullptr, &k->dsl);
  if (r != VK_SUCCESS) { k->dsl = VK_NULL_HANDLE; return from_vk(r, "vkCreateDescriptorSetLayout"); }

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &k->dsl;
  if (push_size) { plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr; }
  r = vkCreatePipelineLayout(ctx().device, &plci, nullptr, &k->layout);
  if (r != VK_SUCCESS) {
    // Roll back the dsl we just built: leaving it set means the NEXT launch re-enters
    // build_pipeline and OVERWRITES k->dsl (leaking this handle, repeatedly on retry).
    vkDestroyDescriptorSetLayout(ctx().device, k->dsl, nullptr);
    k->dsl = VK_NULL_HANDLE; k->layout = VK_NULL_HANDLE;
    return from_vk(r, "vkCreatePipelineLayout");
  }

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = k->module;
  cpci.stage.pName = "main";
  // Specialization constants (#2/#3): if the kernel carries any, bind them to the stage so the
  // driver folds them into the compiled ISA (dead-code elim / loop unroll / baked local_size).
  VkSpecializationInfo si{};
  if (!k->spec_map.empty()) {
    si.mapEntryCount = static_cast<uint32_t>(k->spec_map.size());
    si.pMapEntries = k->spec_map.data();
    si.dataSize = k->spec_data.size();
    si.pData = k->spec_data.data();
    cpci.stage.pSpecializationInfo = &si;
  }
  cpci.layout = k->layout;
  r = vkCreateComputePipelines(ctx().device, ctx().pipe_cache, 1, &cpci, nullptr, &k->pipeline);
  if (r != VK_SUCCESS) {
    // Roll back dsl + layout (same reason: a retry would overwrite + leak them).
    vkDestroyPipelineLayout(ctx().device, k->layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx().device, k->dsl, nullptr);
    k->dsl = VK_NULL_HANDLE; k->layout = VK_NULL_HANDLE; k->pipeline = VK_NULL_HANDLE;
    return from_vk(r, "vkCreateComputePipelines");
  }
  k->buffer_count = buffer_count;
  k->push_size = push_size;
  return Status();
}

Status vk_kernel_compile(int, const char*, const char*, void**, void**) noexcept {
  return Status::Unimplemented("vk backend: runtime GLSL compile deferred (pre-compile to SPIR-V, load_ptx)");
}

// load path: `code` is a SPIR-V byte stream of exactly `code_size` bytes (a SpirvBlob from the
// public load_spirv). SPIR-V is binary (has NULs) so the size is passed explicitly -- no
// length-prefix or strlen. Optional specialization constants (specs/spec_count/spec_data) are
// copied for consumption at pipeline build. This is the Vulkan side of Kernel::load_spirv.
Status vk_kernel_load_ptx(int, const char* code, unsigned code_size, const char* /*entry*/,
                          const SpecConstant* specs, unsigned spec_count,
                          const void* spec_data, unsigned spec_data_size,
                          void** out_module, void** out_func) noexcept {
  if (!ctx().ok) return Status::Unavailable("vk backend: no device");
  // `code` is the raw SPIR-V byte stream of exactly code_size bytes (SpirvBlob) -- no
  // length-prefix hack; the size is explicit. SPIR-V must be 4-byte aligned + a whole number
  // of words (a caller contract; a malformed module is a caller bug -> fail-fast).
  HYLE_FATAL_ASSERT(code_size >= 4 && code_size % 4 == 0);
  auto* k = new (std::nothrow) VkKernel{};
  if (k == nullptr) return Status::ResourceExhausted("vk kernel alloc");
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code_size;
  smci.pCode = reinterpret_cast<const uint32_t*>(code);
  VkResult r = vkCreateShaderModule(ctx().device, &smci, nullptr, &k->module);
  if (r != VK_SUCCESS) { delete k; return from_vk(r, "vkCreateShaderModule"); }
  // Copy spec constants (#2/#3): caller storage is transient, but build_pipeline runs at the
  // FIRST launch (later), so the VkKernel must own the map entries + value blob until then.
  if (spec_count > 0 && specs != nullptr && spec_data != nullptr) {
    k->spec_map.reserve(spec_count);
    for (unsigned i = 0; i < spec_count; ++i) {
      // Contract: each entry's [offset, offset+size) must lie within spec_data. A violation
      // would make VkSpecializationInfo read out of bounds at pipeline build -> fail-fast on
      // the caller bug (offset+size computed to avoid unsigned wrap), not a silent OOB read.
      HYLE_FATAL_ASSERT(specs[i].offset <= spec_data_size &&
                        specs[i].size <= spec_data_size - specs[i].offset);
      k->spec_map.push_back({specs[i].id, specs[i].offset, specs[i].size});
    }
    k->spec_data.assign(static_cast<const uint8_t*>(spec_data),
                        static_cast<const uint8_t*>(spec_data) + spec_data_size);
  }
  *out_module = k;
  *out_func = nullptr;
  return Status();
}
Status vk_kernel_load_host(int, void*, void**, void**) noexcept {
  return Status::Unimplemented("vk backend: no host kernels");
}
void vk_kernel_destroy(void* module, void*) noexcept {
  auto* k = static_cast<VkKernel*>(module);
  if (k == nullptr) return;
  if (k->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx().device, k->pipeline, nullptr);
  if (k->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx().device, k->layout, nullptr);
  if (k->dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx().device, k->dsl, nullptr);
  if (k->module != VK_NULL_HANDLE) vkDestroyShaderModule(ctx().device, k->module, nullptr);
  delete k;
}

// launch: build (lazily, once) the pipeline for this kernel's arg shape, allocate+update a
// descriptor set for the buffers, record bind+push+dispatch into a command buffer from the
// Stream's pool, and submit with the Stream's fence (the Event the caller queries). This is
// the KernelArgs -> Vulkan lowering (buffers -> descriptors, scalar blob -> push constants).
Status vk_kernel_launch(void* stream_native, void* module, void* /*func*/,
                        const void* config, const KernelArgs* args) noexcept {
  const auto* cfg = static_cast<const LaunchConfig*>(config);
  auto* k = static_cast<VkKernel*>(module);
  auto* stream = static_cast<VkStream*>(stream_native);
  VkDevice dev = ctx().device;

  // Lazily create the pipeline once we know the arg shape (buffer count + push size).
  if (k->pipeline == VK_NULL_HANDLE) {
    Status s = build_pipeline(k, args->buffer_count, args->scalar_bytes);
    if (!s.ok()) return s;
  }
  // Same kernel => same arg SHAPE. The pipeline layout was built (once, lazily) from the
  // FIRST launch's buffer_count + push_size and is frozen; relaunching with a different
  // shape would push past the layout's push-constant range (UB) or mis-bind descriptors.
  // Contract violation -> fail-fast (caller bug, not a recoverable external failure).
  HYLE_FATAL_ASSERT(k->buffer_count == args->buffer_count);
  HYLE_FATAL_ASSERT(k->push_size == args->scalar_bytes);

  // Descriptor pool + set for this launch's buffers (checklist #17/#18: pool sized to
  // buffers, ONE batched updateDescriptorSets). One-shot pool freed at end.
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  if (args->buffer_count) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, args->buffer_count};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkResult r = vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);
    if (r != VK_SUCCESS) return from_vk(r, "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &k->dsl;
    r = vkAllocateDescriptorSets(dev, &dsai, &dset);
    if (r != VK_SUCCESS) { vkDestroyDescriptorPool(dev, dpool, nullptr);
                           return from_vk(r, "vkAllocateDescriptorSets"); }
    // Batched update (checklist #18): one call for all bindings.
    std::vector<VkDescriptorBufferInfo> dbi(args->buffer_count);
    std::vector<VkWriteDescriptorSet> w(args->buffer_count);
    for (uint32_t i = 0; i < args->buffer_count; ++i) {
      auto* m = static_cast<VkMem*>(args->buffers[i]);
      dbi[i] = {m->buf, 0, VK_WHOLE_SIZE};
      w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w[i].dstSet = dset; w[i].dstBinding = i; w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, args->buffer_count, w.data(), 0, nullptr);
  }

  // Record: bind pipeline + descriptor set + push constants + dispatch (grid = groups).
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = stream->pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkResult r = vkAllocateCommandBuffers(dev, &cbai, &cb);
  if (r != VK_SUCCESS) { if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
                         return from_vk(r, "vkAllocateCommandBuffers"); }
  VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bg);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
  if (args->buffer_count)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &dset, 0, nullptr);
  if (args->scalar_bytes)
    vkCmdPushConstants(cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, args->scalar_bytes, args->scalars);
  // Grid = workgroup COUNT (checklist #20: local_size is in the shader, group=ceil(N/ls)).
  // Guard the device limit -- exceeding maxComputeWorkGroupCount is a caller bug (kompute's
  // no-clamp trap); fail-fast rather than a silent driver error.
  HYLE_FATAL_ASSERT(cfg->grid.x <= ctx().max_wg_count[0] && cfg->grid.y <= ctx().max_wg_count[1] &&
                    cfg->grid.z <= ctx().max_wg_count[2]);
  vkCmdDispatch(cb, cfg->grid.x, cfg->grid.y, cfg->grid.z);
  // Barrier (#11): make the compute writes to the storage buffers VISIBLE to a subsequent
  // host read (buffer_get maps the memory), a transfer (staging D2H), or a following
  // dispatch (read OR write -- SHADER_WRITE in dst orders a following submit's WAW on the
  // same buffer, 红队 D). ONE global memory barrier covers all buffers -- precise per-buffer
  // barriers matter for the multi-kernel batch path (above), not this single dispatch. This
  // replaces the skeleton's "rely on vkQueueWaitIdle + coherent memory" shortcut.
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                     VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
  vkEndCommandBuffer(cb);

  // Submit ASYNC (#21): own fence per submit, NO wait -- launch returns immediately; the
  // caller observes completion via `co_await StreamComplete` (create Event -> event_record
  // marker on this stream -> GpuPoller polls the fence) or stream_sync. The cb + descriptor
  // pool + fence are tracked in-flight and freed by retire() once the fence signals (they
  // MUST outlive GPU execution -- freeing them here would be use-after-free). retire() at
  // entry reclaims prior completed submissions so the list stays bounded.
  retire(stream, /*force=*/false);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  r = vkCreateFence(dev, &fci, nullptr, &fence);
  if (r != VK_SUCCESS) { vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
                         if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
                         return from_vk(r, "vkCreateFence"); }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  {
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);  // VkQueue is externally synchronized
    r = vkQueueSubmit(ctx().queue, 1, &si, fence);
  }
  if (r != VK_SUCCESS) { vkDestroyFence(dev, fence, nullptr);
                         vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
                         if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
                         return from_vk(r, "vkQueueSubmit"); }
  stream->inflight.push_back(InFlight{fence, cb, dpool ? std::vector<VkDescriptorPool>{dpool}
                                                       : std::vector<VkDescriptorPool>{},
                                      /*owns_fence=*/true});
  return Status();  // async: submitted, not waited
}

// BATCHED launch (#15, the measured perf lever): record ALL `count` dispatches into ONE
// command buffer + ONE submit (vs N launches = N submits+fences). Each item gets its own
// descriptor pool (freed together on retire); a barrier between dispatches makes each
// kernel's writes visible to the next (safe for dependent chains; independent kernels pay
// a cheap barrier). ONE fence for the whole batch owned by a SINGLE InFlight entry that
// also holds the cb + ALL the per-item descriptor pools (dpools vector); retire() frees
// them together when that one fence signals -- no shared-fence double-free.
Status vk_kernel_launch_batch(void* stream_native, const void* const* modules,
                              const void* const* funcs, const LaunchConfig* configs,
                              const KernelArgs* args_arr, unsigned count) noexcept {
  if (count == 0) return Status();
  auto* stream = static_cast<VkStream*>(stream_native);
  VkDevice dev = ctx().device;
  retire(stream, /*force=*/false);

  // Build pipelines (lazy) for every item up front. Same kernel => same arg shape: whether
  // freshly built or reused (pipeline already set), assert buffer_count + push_size match the
  // frozen layout -- a shape mismatch would mis-bind descriptors / overflow the push range
  // (caller bug -> fail-fast, same contract as the single-launch path).
  for (unsigned i = 0; i < count; ++i) {
    auto* k = static_cast<VkKernel*>(const_cast<void*>(modules[i]));
    if (k->pipeline == VK_NULL_HANDLE) {
      Status s = build_pipeline(k, args_arr[i].buffer_count, args_arr[i].scalar_bytes);
      if (!s.ok()) return s;
    }
    HYLE_FATAL_ASSERT(k->buffer_count == args_arr[i].buffer_count);
    HYLE_FATAL_ASSERT(k->push_size == args_arr[i].scalar_bytes);
  }
  // One command buffer for the whole batch.
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = stream->pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkResult r = vkAllocateCommandBuffers(dev, &cbai, &cb);
  if (r != VK_SUCCESS) return from_vk(r, "vkAllocateCommandBuffers(batch)");
  VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bg);

  std::vector<VkDescriptorPool> pools;  // one per item with buffers; freed on retire
  pools.reserve(count);
  // #11/#12 precise barriers: kompute inserts an unconditional barrier after EVERY dispatch
  // (over-serialization -- independent kernels can't overlap). HYLE inserts an inter-dispatch
  // barrier ONLY when an item touches a buffer a prior un-barriered item touched (a real
  // RAW/WAW/WAR hazard). Disjoint dispatches run with NO barrier so the GPU can overlap them.
  // `pending` = buffers touched since the last barrier (cleared when one is inserted, since a
  // barrier resolves all prior hazards). The ABI can't tell read from write, so any shared
  // buffer is treated as a hazard.
  std::vector<void*> pending;
  auto touches_pending = [&](const KernelArgs& a) noexcept {
    for (uint32_t b = 0; b < a.buffer_count; ++b)
      for (void* p : pending)
        if (a.buffers[b] == p) return true;
    return false;
  };
  for (unsigned i = 0; i < count; ++i) {
    auto* k = static_cast<VkKernel*>(const_cast<void*>(modules[i]));
    const KernelArgs& a = args_arr[i];
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (a.buffer_count) {
      VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, a.buffer_count};
      VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
      VkDescriptorPool dpool = VK_NULL_HANDLE;
      if (vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool) != VK_SUCCESS) {
        // Fail-fast, NOT silent-skip: a dropped dispatch would return OK while under-
        // computing (project rule: contract/resource failure surfaces, no silent兜底).
        vkEndCommandBuffer(cb);
        vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
        for (auto p : pools) vkDestroyDescriptorPool(dev, p, nullptr);
        return Status::ResourceExhausted("vk batch: descriptor pool alloc failed");
      }
      pools.push_back(dpool);
      VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &k->dsl;
      if (vkAllocateDescriptorSets(dev, &dsai, &dset) != VK_SUCCESS) {
        // Same fail-fast as the pool-alloc failure above (dpool is already in `pools`).
        vkEndCommandBuffer(cb);
        vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
        for (auto p : pools) vkDestroyDescriptorPool(dev, p, nullptr);
        return Status::ResourceExhausted("vk batch: descriptor set alloc failed");
      }
      std::vector<VkDescriptorBufferInfo> dbi(a.buffer_count);
      std::vector<VkWriteDescriptorSet> w(a.buffer_count);
      for (uint32_t b = 0; b < a.buffer_count; ++b) {
        auto* m = static_cast<VkMem*>(a.buffers[b]);
        dbi[b] = {m->buf, 0, VK_WHOLE_SIZE};
        w[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[b].dstSet = dset; w[b].dstBinding = b; w[b].descriptorCount = 1;
        w[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[b].pBufferInfo = &dbi[b];
      }
      vkUpdateDescriptorSets(dev, a.buffer_count, w.data(), 0, nullptr);
    }
    // #11/#12: barrier BEFORE this dispatch, ONLY if it shares a buffer with an un-barriered
    // prior dispatch (real hazard). Disjoint items skip it -> the GPU can overlap them. dst
    // includes SHADER_WRITE so a WAW/WAR hazard (this kernel writes a buffer a prior one
    // read/wrote) is ordered too (红队 D: the old barrier's dst omitted SHADER_WRITE).
    if (touches_pending(a)) {
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
      pending.clear();  // a barrier resolves all prior hazards
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    if (a.buffer_count)
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &dset, 0, nullptr);
    if (a.scalar_bytes)
      vkCmdPushConstants(cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, a.scalar_bytes, a.scalars);
    HYLE_FATAL_ASSERT(configs[i].grid.x <= ctx().max_wg_count[0] &&
                      configs[i].grid.y <= ctx().max_wg_count[1] &&
                      configs[i].grid.z <= ctx().max_wg_count[2]);  // #20 device limit
    vkCmdDispatch(cb, configs[i].grid.x, configs[i].grid.y, configs[i].grid.z);
    for (uint32_t b = 0; b < a.buffer_count; ++b) pending.push_back(a.buffers[b]);  // hazard set
  }
  // Terminal barrier (once): make ALL dispatch writes visible to a host readback (buffer_get
  // maps memory), a transfer (staging D2H), or a following submit's kernel. Only the FINAL
  // visibility needs host/transfer scope, so it's hoisted out of the per-dispatch loop.
  {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  }
  vkEndCommandBuffer(cb);

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  r = vkCreateFence(dev, &fci, nullptr, &fence);
  if (r != VK_SUCCESS) { vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
                         for (auto p : pools) vkDestroyDescriptorPool(dev, p, nullptr);
                         return from_vk(r, "vkCreateFence(batch)"); }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  {
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);
    r = vkQueueSubmit(ctx().queue, 1, &si, fence);
  }
  if (r != VK_SUCCESS) { vkDestroyFence(dev, fence, nullptr); vkFreeCommandBuffers(dev, stream->pool, 1, &cb);
                         for (auto p : pools) vkDestroyDescriptorPool(dev, p, nullptr);
                         return from_vk(r, "vkQueueSubmit(batch)"); }
  // ONE in-flight entry owns the batch's fence + cb + ALL its descriptor pools; retire()
  // frees them together when the single shared fence signals (no shared-fence double-free).
  InFlight bf; bf.fence = fence; bf.cb = cb; bf.dpools = std::move(pools); bf.owns_fence = true;
  stream->inflight.push_back(std::move(bf));
  return Status();
}

// ---- record / replay (checklist #16) -----------------------------------------
// A recorded launch owns a command buffer recorded ONCE (bind + push + dispatch + barrier),
// NOT one-time-submit, so it can be re-submitted many times with no re-recording -- the CPU
// win for iterative workloads. It also owns the descriptor pool/set the cb references (they
// must outlive every replay). replay() submits the cb with a fresh fence tracked in the
// stream's in-flight list, but with cb=NULL + no dpools in that entry so retire() reclaims
// only the fence (the cb + pool belong to VkRecorded, freed at destroy after a full drain).
struct VkRecorded {
  VkCommandBuffer cb = VK_NULL_HANDLE;   // recorded once, replayed many; owned here
  VkDescriptorPool dpool = VK_NULL_HANDLE;  // owns the descriptor set the cb binds (or NULL)
};

Status vk_recorded_create(void* stream_native, void* module, void* /*func*/,
                          const void* config, const KernelArgs* args,
                          void** out_recorded) noexcept {
  const auto* cfg = static_cast<const LaunchConfig*>(config);
  auto* k = static_cast<VkKernel*>(module);
  auto* stream = static_cast<VkStream*>(stream_native);
  VkDevice dev = ctx().device;
  if (k->pipeline == VK_NULL_HANDLE) {
    Status s = build_pipeline(k, args->buffer_count, args->scalar_bytes);
    if (!s.ok()) return s;
  }
  HYLE_FATAL_ASSERT(k->buffer_count == args->buffer_count);
  HYLE_FATAL_ASSERT(k->push_size == args->scalar_bytes);

  auto* rec = new (std::nothrow) VkRecorded{};
  if (rec == nullptr) return Status::ResourceExhausted("vk recorded alloc");
  VkDescriptorSet dset = VK_NULL_HANDLE;
  if (args->buffer_count) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, args->buffer_count};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkResult r = vkCreateDescriptorPool(dev, &dpci, nullptr, &rec->dpool);
    if (r != VK_SUCCESS) { delete rec; return from_vk(r, "vkCreateDescriptorPool(recorded)"); }
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = rec->dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &k->dsl;
    r = vkAllocateDescriptorSets(dev, &dsai, &dset);
    if (r != VK_SUCCESS) { vkDestroyDescriptorPool(dev, rec->dpool, nullptr); delete rec;
                           return from_vk(r, "vkAllocateDescriptorSets(recorded)"); }
    std::vector<VkDescriptorBufferInfo> dbi(args->buffer_count);
    std::vector<VkWriteDescriptorSet> w(args->buffer_count);
    for (uint32_t i = 0; i < args->buffer_count; ++i) {
      auto* m = static_cast<VkMem*>(args->buffers[i]);
      dbi[i] = {m->buf, 0, VK_WHOLE_SIZE};
      w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w[i].dstSet = dset; w[i].dstBinding = i; w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, args->buffer_count, w.data(), 0, nullptr);
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = stream->pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkResult r = vkAllocateCommandBuffers(dev, &cbai, &rec->cb);
  if (r != VK_SUCCESS) { if (rec->dpool) vkDestroyDescriptorPool(dev, rec->dpool, nullptr);
                         delete rec; return from_vk(r, "vkAllocateCommandBuffers(recorded)"); }
  // Replays are asynchronous, so callers may submit the same recording again before an
  // earlier replay completes. Vulkan requires SIMULTANEOUS_USE for that state transition;
  // merely omitting ONE_TIME_SUBMIT does not make a pending command buffer re-submittable.
  VkCommandBufferBeginInfo bg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bg.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(rec->cb, &bg);
  vkCmdBindPipeline(rec->cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
  if (args->buffer_count)
    vkCmdBindDescriptorSets(rec->cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &dset, 0, nullptr);
  if (args->scalar_bytes)
    vkCmdPushConstants(rec->cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, args->scalar_bytes, args->scalars);
  HYLE_FATAL_ASSERT(cfg->grid.x <= ctx().max_wg_count[0] && cfg->grid.y <= ctx().max_wg_count[1] &&
                    cfg->grid.z <= ctx().max_wg_count[2]);
  vkCmdDispatch(rec->cb, cfg->grid.x, cfg->grid.y, cfg->grid.z);
  // Same terminal barrier as the single launch: writes visible to host read / next dispatch.
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                     VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(rec->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  vkEndCommandBuffer(rec->cb);
  *out_recorded = rec;
  return Status();
}

Status vk_recorded_replay(void* stream_native, void* recorded) noexcept {
  auto* stream = static_cast<VkStream*>(stream_native);
  auto* rec = static_cast<VkRecorded*>(recorded);
  VkDevice dev = ctx().device;
  retire(stream, /*force=*/false);  // reclaim prior replays' fences
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  VkResult r = vkCreateFence(dev, &fci, nullptr, &fence);
  if (r != VK_SUCCESS) return from_vk(r, "vkCreateFence(replay)");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &rec->cb;
  {
    std::lock_guard<std::mutex> lk(ctx().submit_mtx);
    r = vkQueueSubmit(ctx().queue, 1, &si, fence);
  }
  if (r != VK_SUCCESS) { vkDestroyFence(dev, fence, nullptr); return from_vk(r, "vkQueueSubmit(replay)"); }
  // Track ONLY the fence: cb=NULL + no dpools so retire() frees the fence but NOT the cb /
  // pool (those belong to VkRecorded, reused across replays, freed at recorded_destroy).
  InFlight rf; rf.fence = fence; rf.cb = VK_NULL_HANDLE; rf.owns_fence = true;
  stream->inflight.push_back(std::move(rf));
  return Status();
}

void vk_recorded_destroy(void* stream_native, void* recorded) noexcept {
  auto* stream = static_cast<VkStream*>(stream_native);
  auto* rec = static_cast<VkRecorded*>(recorded);
  if (rec == nullptr) return;
  VkDevice dev = ctx().device;
  // Drain every outstanding replay (their fences are in the stream's in-flight list) so the GPU
  // is done with the cb before we free it -- else vkFreeCommandBuffers would race the GPU.
  retire(stream, /*force=*/true);
  if (rec->cb != VK_NULL_HANDLE) vkFreeCommandBuffers(dev, stream->pool, 1, &rec->cb);
  if (rec->dpool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, rec->dpool, nullptr);
  delete rec;
}

constexpr ComputeBackend kVulkanBackend = {
    /*name=*/"vulkan",
    /*device_count=*/vk_device_count,
    /*device_type=*/vk_device_type,
    /*device_name=*/vk_device_name,
    /*device_caps=*/vk_device_caps,
    /*device_memory=*/vk_device_memory,
    /*stream_create=*/vk_stream_create,
    /*stream_destroy=*/vk_stream_destroy,
    /*stream_sync=*/vk_stream_sync,
    /*event_create=*/vk_event_create,
    /*event_destroy=*/vk_event_destroy,
    /*event_record=*/vk_event_record,
    /*event_query=*/vk_event_query,
    /*event_sync=*/vk_event_sync,
    /*buffer_alignment=*/vk_buffer_alignment,
    /*buffer_max_size=*/vk_buffer_max_size,
    /*buffer_is_host=*/vk_buffer_is_host,
    /*buffer_alloc=*/vk_buffer_alloc,
    /*buffer_free=*/vk_buffer_free,
    /*buffer_flush_host=*/vk_buffer_flush_host,
    /*buffer_invalidate_host=*/vk_buffer_invalidate_host,
    /*buffer_set=*/vk_buffer_set,
    /*buffer_get=*/vk_buffer_get,
    /*buffer_copy=*/vk_buffer_copy,
    /*buffer_clear=*/vk_buffer_clear,
    /*pool_create=*/vk_pool_create,
    /*pool_destroy=*/vk_pool_destroy,
    /*pool_is_vmm=*/vk_pool_is_vmm,
    /*pool_alloc=*/vk_pool_alloc,
    /*pool_free=*/vk_pool_free,
    /*kernel_compile=*/vk_kernel_compile,
    /*kernel_load_ptx=*/vk_kernel_load_ptx,
    /*kernel_load_host=*/vk_kernel_load_host,
    /*kernel_destroy=*/vk_kernel_destroy,
    /*kernel_launch=*/vk_kernel_launch,
    /*kernel_launch_batch=*/vk_kernel_launch_batch,
    /*event_create_timing=*/vk_event_create_timing,
    /*event_destroy_timing=*/vk_event_destroy_timing,
    /*event_record_timing=*/vk_event_record_timing,
    /*event_query_timing=*/vk_event_query_timing,
    /*event_sync_timing=*/vk_event_sync_timing,
    /*event_elapsed_ns=*/vk_event_elapsed_ns,
    /*recorded_create=*/vk_recorded_create,
    /*recorded_replay=*/vk_recorded_replay,
    /*recorded_destroy=*/vk_recorded_destroy,
};

}  // namespace

const ComputeBackend* vulkan_vtable() noexcept {
  return ctx().ok ? &kVulkanBackend : nullptr;
}

#if defined(HYLE_TESTING)
// TEST-ONLY: expose the exact mapped-range normalization for synthetic boundary tests.
// This verifies nonCoherentAtomSize arithmetic even on devices that expose only coherent
// host memory (where the driver call itself must remain an honest hardware skip).
HYLE_EXPORT std::size_t vk_noncoherent_atom_size_for_test() noexcept {
  return static_cast<std::size_t>(ctx().non_coherent_atom_size);
}
HYLE_EXPORT void vk_mapped_range_for_test(void* native, std::size_t offset,
                                          std::size_t size,
                                          std::size_t* out_offset,
                                          std::size_t* out_size) noexcept {
  HYLE_FATAL_ASSERT(native != nullptr && out_offset != nullptr && out_size != nullptr);
  const VkMappedMemoryRange range =
      mapped_range(static_cast<const VkMem*>(native), offset, size);
  *out_offset = static_cast<std::size_t>(range.offset);
  *out_size = static_cast<std::size_t>(range.size);
}

// TEST-ONLY: report whether a host Buffer actually selected a non-coherent memory type.
// HYLE_VK_FORCE_NONCOHERENT scans every compatible memory type; false means the physical
// device truly exposes none and the driver-path test must skip honestly.
HYLE_EXPORT bool vk_buffer_is_noncoherent_for_test(void* native) noexcept {
  const auto* m = static_cast<const VkMem*>(native);
  return m != nullptr && m->mapped != nullptr && !m->coherent;
}

HYLE_EXPORT std::size_t vk_staging_capacity_for_test(void* native) noexcept {
  const auto* m = static_cast<const VkMem*>(native);
  return m == nullptr ? 0 : static_cast<std::size_t>(m->staging_capacity);
}

// TEST-ONLY (#1 pipeline cache 落盘): the persist path normally runs only in ~VkContext at
// process exit, which a single-process test can't observe. This hook flushes the current
// cache to HYLE_VK_PIPELINE_CACHE on demand and returns the payload byte count (0 if no env
// path, no device, or the cache is empty) so a test can prove a non-empty cache was written
// after building a pipeline. Compiled ONLY under HYLE_TESTING (absent from the shipped lib);
// HYLE_EXPORT'd because the test TU links it across the hyle_compute DLL boundary.
HYLE_EXPORT std::size_t vk_pipeline_cache_flush_for_test() noexcept {
  VkContext& c = ctx();
  if (!c.ok || c.pipe_cache == VK_NULL_HANDLE) return 0;
  const char* cp = pipe_cache_path();
  if (cp == nullptr) return 0;
  // vkGetPipelineCacheData is host access to the cache -> externally-synchronized (spec), and
  // build_pipeline writes it under build_mtx. Hold the SAME lock so a concurrent first-launch
  // build can't race this read (红队: same external-sync rule as the pipeline build).
  std::lock_guard<std::mutex> lk(c.build_mtx);
  std::size_t n = 0;
  if (vkGetPipelineCacheData(c.device, c.pipe_cache, &n, nullptr) != VK_SUCCESS || n == 0) return 0;
  std::vector<uint8_t> blob(n);
  if (vkGetPipelineCacheData(c.device, c.pipe_cache, &n, blob.data()) != VK_SUCCESS) return 0;
  blob.resize(n);
  save_pipe_cache(cp, c.cache_uuid, blob);
  return blob.size();
}
#endif

}  // namespace hyle::compute
