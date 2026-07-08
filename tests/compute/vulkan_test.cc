#include "hyle/compute/kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/stream.h"
#include "saxpy_spirv.h"  // kSaxpySpirv[] -- glslang-compiled saxpy.comp (checked in)

#if defined(HYLE_HAS_FCONTEXT)
#  include <atomic>
#  include <cmath>
#  include <memory>
#  include "hyle/compute/gpu_await.h"
#  include "hyle/mem/allocator.h"
#  include "hyle/sched/coro.h"
#  include "hyle/sched/fiber.h"
#  include "hyle/sched/scheduler.h"
#endif

// compute-06 test: run a self-authored SPIR-V compute kernel through HYLE's Vulkan
// backend end-to-end -- the Vulkan twin of kernel_test's CUDA path. saxpy:
// o[i] = alpha*x[i] + beta*y[i] + bias, with 3 storage buffers (bindings 0,1,2) and a
// push-constant block {uint n; float alpha,beta,bias}. Exercises: volk device, Buffer
// alloc (ReBAR host-mappable), the unified launch typed shell (buffers -> descriptors,
// scalars -> push constants), and readback. GPU-skips if no Vulkan device is present.

using namespace hyle;
using namespace hyle::compute;

// Threadsafe death style on Linux only (re-exec vs fork-after-driver-init). Windows death
// tests are always spawn-based (no fork) so the flag is unneeded -- and its symbol
// (testing::FLAGS_gtest_death_test_style) doesn't link against the gtest DLL. See
// contract_test.cc for the full rationale.
#if defined(_WIN32)
#  define HYLE_THREADSAFE_DEATH() ((void)0)
#else
#  define HYLE_THREADSAFE_DEATH() GTEST_FLAG_SET(death_test_style, "threadsafe")
#endif

namespace {

// Find the Vulkan-backed device among enumerated devices, or return false (no Vulkan
// loader / backend disabled). Device::backend() names the owning backend ("vulkan").
bool find_vulkan(Device& out) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::strcmp(devs[i].backend(), "vulkan") == 0) {
      out = devs[i];
      return true;
    }
  }
  return false;
}

// The push-constant block -- MUST match saxpy.comp's Params (std430: uint@0, 3 float).
struct SaxpyParams { std::uint32_t n; float alpha; float beta; float bias; };

}  // namespace

// End-to-end: SPIR-V kernel + buffers + scalars through the unified launch shell.
TEST(VulkanCompute, SaxpyEndToEnd) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device (backend disabled or no loader)";

  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());

  // Load the pre-compiled SPIR-V. The Vulkan backend's load_ptx expects a LENGTH-PREFIXED
  // blob: uint32 byte-length, then the SPIR-V words (kernel.h documents this for Vulkan).
  const std::uint32_t byte_len = sizeof(kSaxpySpirv);
  std::vector<std::byte> blob(sizeof(std::uint32_t) + byte_len);
  std::memcpy(blob.data(), &byte_len, sizeof(std::uint32_t));
  std::memcpy(blob.data() + sizeof(std::uint32_t), kSaxpySpirv, byte_len);
  auto k = Kernel::load_ptx(vk, reinterpret_cast<const char*>(blob.data()), "main");
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());

  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(100 + i); }
  ASSERT_TRUE(bx->set(0, hx.data(), bytes).ok());
  ASSERT_TRUE(by->set(0, hy.data(), bytes).ok());

  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};  // group count = ceil(N/local_size), local=256
  // Unified typed shell: buffers (x,y,o) -> descriptor bindings 0,1,2; scalar `p` -> push
  // constants. One call, backend-neutral (same source runs the CUDA path in kernel_test).
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  ASSERT_TRUE(s->sync().ok());

  std::vector<float> ho(kN, -1.0f);
  ASSERT_TRUE(bo->get(0, ho.data(), bytes).ok());
  for (int i = 0; i < kN; ++i) {
    float expect = 2.0f * hx[i] + 3.0f * hy[i] + 0.5f;
    ASSERT_NEAR(ho[i], expect, 1e-2f) << "at " << i;
  }
}

namespace {
// Load the saxpy SPIR-V kernel on `dev` (length-prefixed blob, as the Vulkan backend
// expects). Returned Kernel must outlive its launches.
Result<Kernel> load_saxpy(const Device& dev) {
  const std::uint32_t byte_len = sizeof(kSaxpySpirv);
  std::vector<std::byte> blob(sizeof(std::uint32_t) + byte_len);
  std::memcpy(blob.data(), &byte_len, sizeof(std::uint32_t));
  std::memcpy(blob.data() + sizeof(std::uint32_t), kSaxpySpirv, byte_len);
  return Kernel::load_ptx(dev, reinterpret_cast<const char*>(blob.data()), "main");
}
}  // namespace

// #15 batching: N saxpy dispatches (different buffer sets) in ONE launch_batch = ONE
// Vulkan command buffer + ONE submit. Each result golden-checked. Proves the batched path
// (the measured perf lever) is correct + the async in-flight tracking frees per-item pools.
TEST(VulkanCompute, BatchedSaxpy) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());
  auto k = load_saxpy(vk);
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kJobs = 3;
  constexpr int kN = 2048;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);

  // Keep buffers + params alive across the batch (their pointers go into KernelArgs).
  std::vector<Buffer> bx, by, bo;
  std::vector<std::vector<float>> hx(kJobs), hy(kJobs);
  SaxpyParams params[kJobs];
  void* bufptrs[kJobs][3];
  ScalarField fields[kJobs];
  BatchItem items[kJobs];
  bx.reserve(kJobs); by.reserve(kJobs); bo.reserve(kJobs);

  for (int j = 0; j < kJobs; ++j) {
    auto x = Buffer::create(bt, bytes); auto y = Buffer::create(bt, bytes);
    auto o = Buffer::create(bt, bytes);
    ASSERT_TRUE(x.has_value() && y.has_value() && o.has_value());
    bx.push_back(std::move(*x)); by.push_back(std::move(*y)); bo.push_back(std::move(*o));
    hx[j].resize(kN); hy[j].resize(kN);
    for (int i = 0; i < kN; ++i) { hx[j][i] = float(i + j); hy[j][i] = float(2 * i + j); }
    ASSERT_TRUE(bx[j].set(0, hx[j].data(), bytes).ok());
    ASSERT_TRUE(by[j].set(0, hy[j].data(), bytes).ok());
    params[j] = SaxpyParams{kN, 1.0f + j, 2.0f + j, float(j)};  // distinct scalars per job
    bufptrs[j][0] = bx[j].device_ptr(); bufptrs[j][1] = by[j].device_ptr(); bufptrs[j][2] = bo[j].device_ptr();
    fields[j] = ScalarField{0, sizeof(SaxpyParams)};
    items[j].kernel = &*k;
    items[j].config = LaunchConfig{};
    items[j].config.grid = {blocks_for(kN, 256), 1, 1};
    items[j].args = KernelArgs{bufptrs[j], 3, &params[j], sizeof(SaxpyParams), &fields[j], 1};
  }

  ASSERT_TRUE(launch_batch(*s, items, kJobs).ok());
  ASSERT_TRUE(s->sync().ok());

  for (int j = 0; j < kJobs; ++j) {
    std::vector<float> ho(kN, -1.0f);
    ASSERT_TRUE(bo[j].get(0, ho.data(), bytes).ok());
    for (int i = 0; i < kN; ++i) {
      float expect = params[j].alpha * hx[j][i] + params[j].beta * hy[j][i] + params[j].bias;
      ASSERT_NEAR(ho[i], expect, 1e-1f) << "job " << j << " at " << i;
    }
  }
}

// ---- Vulkan CONTRACT death tests (vulkan_backend.cc:561/665/666/714) ----------
// The Vulkan backend's launch contracts are HYLE_FATAL_ASSERTs (caller bugs, not
// recoverable Status). These prove they fire. All GPU-skip if no Vulkan device.
//
// Threadsafe death style: on this machine CUDA + Vulkan are both initialized by other
// tests; re-exec avoids fork-after-driver-init. Each scenario is a named function (the
// death statement's braces/commas would confuse EXPECT_DEATH). The function rebuilds all
// state (the threadsafe child re-execs from scratch) then triggers exactly one assert.
// Matcher "HYLE fatal" proves death came from the contract assert, not a driver crash.

namespace {

// Pipeline shape is FROZEN on first launch (buffer_count + push_size). A second launch of
// the SAME kernel with a different buffer_count must abort (vulkan_backend.cc:665). We
// launch saxpy correctly once (3 buffers, sizeof(SaxpyParams) push), then relaunch with a
// raw KernelArgs carrying only 2 buffers.
void vk_shape_freeze_buffer_count() {
  Device vk;
  if (!find_vulkan(vk)) std::abort();  // "skip" -> abort is fine; the test gates before calling
  auto s = Stream::create(vk);
  auto k = load_saxpy(vk);
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  SaxpyParams p{kN, 1.0f, 1.0f, 0.0f};
  LaunchConfig cfg; cfg.grid = {blocks_for(kN, 256), 1, 1};
  (void)launch(*s, *k, cfg, *bx, *by, *bo, p);  // first launch freezes {3 buffers, 16B push}
  (void)s->sync();
  // Relaunch with only 2 buffers (same push) -> buffer_count mismatch -> fatal (:665).
  void* two[2] = {bx->device_ptr(), by->device_ptr()};
  ScalarField f{0, sizeof(SaxpyParams)};
  KernelArgs bad{two, 2, &p, sizeof(SaxpyParams), &f, 1};
  (void)launch(*s, *k, cfg, bad);
}

// push_size > maxPushConstantsSize aborts in build_pipeline (vulkan_backend.cc:561) on the
// FIRST launch. The typed shell caps scalars at 256B, so we bypass it with a raw KernelArgs
// declaring scalar_bytes = 512 (> any real device's max_push, RTX 4070 = 256).
void vk_push_size_over_limit() {
  Device vk;
  if (!find_vulkan(vk)) std::abort();
  auto s = Stream::create(vk);
  auto k = load_saxpy(vk);
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  alignas(16) unsigned char big_scalars[512] = {};
  void* three[3] = {bx->device_ptr(), by->device_ptr(), bo->device_ptr()};
  ScalarField f{0, 512};
  KernelArgs bad{three, 3, big_scalars, 512, &f, 1};  // 512 > max_push -> fatal (:561)
  LaunchConfig cfg; cfg.grid = {blocks_for(kN, 256), 1, 1};
  (void)launch(*s, *k, cfg, bad);
}

// grid.x > maxComputeWorkGroupCount aborts at dispatch record (vulkan_backend.cc:714).
void vk_grid_over_limit() {
  Device vk;
  if (!find_vulkan(vk)) std::abort();
  auto s = Stream::create(vk);
  auto k = load_saxpy(vk);
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  SaxpyParams p{kN, 1.0f, 1.0f, 0.0f};
  LaunchConfig cfg;
  cfg.grid = {0xFFFFFFFFu, 1, 1};  // way past maxComputeWorkGroupCount[0] -> fatal (:714)
  (void)launch(*s, *k, cfg, *bx, *by, *bo, p);
}

}  // namespace

TEST(VulkanContractDeathTest, ShapeFreezeBufferCountMismatch) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(vk_shape_freeze_buffer_count(), "HYLE fatal");
}

TEST(VulkanContractDeathTest, PushConstantSizeOverLimit) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(vk_push_size_over_limit(), "HYLE fatal");
}

TEST(VulkanContractDeathTest, GridExceedsWorkgroupLimit) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(vk_grid_over_limit(), "HYLE fatal");
}

#if defined(HYLE_HAS_FCONTEXT)
namespace {
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;

// #21 async completion: a FIBER launches a Vulkan kernel (async submit, no wait) then
// `co_await StreamComplete` -- suspends until the GpuPoller sees the stream's fence (via
// event_record's marker submit + vkGetFenceStatus) and wakes it. Proves the Vulkan
// completion path is the same fiber-wake shape as CUDA (compute-03), not a blocking wait.
struct AwaitArg { int result = -100; std::atomic<bool> done{false}; };

sched::Task<int> vk_await_coro(mem::Allocator, int* out) {
  Device vk;
  if (!find_vulkan(vk)) { *out = 1; co_return 1; }  // skip-code (no device)
  auto s = Stream::create(vk);
  if (!s.has_value()) { *out = -1; co_return -1; }
  auto k = load_saxpy(vk);
  if (!k.has_value()) { *out = -2; co_return -2; }

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  if (!bx.has_value() || !by.has_value() || !bo.has_value()) { *out = -3; co_return -3; }
  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(i) + 1.0f; }
  if (!bx->set(0, hx.data(), bytes).ok() || !by->set(0, hy.data(), bytes).ok()) { *out = -4; co_return -4; }

  SaxpyParams p{kN, 2.0f, 3.0f, 1.0f};
  LaunchConfig cfg; cfg.grid = {blocks_for(kN, 256), 1, 1};
  if (!launch(*s, *k, cfg, *bx, *by, *bo, p).ok()) { *out = -5; co_return -5; }

  Status st = co_await StreamComplete{&*s};  // SUSPEND until GPU done (poller wakes us)
  if (!st.ok()) { *out = -6; co_return -6; }

  std::vector<float> ho(kN, -1.0f);
  if (!bo->get(0, ho.data(), bytes).ok()) { *out = -7; co_return -7; }
  for (int i = 0; i < kN; ++i)
    if (std::abs(ho[i] - (2.0f * hx[i] + 3.0f * hy[i] + 1.0f)) > 1e-2f) { *out = -8; co_return -8; }
  *out = 0;
  co_return 0;
}
struct Drive { sched::Task<int> task; std::atomic<bool> fin{false}; };
void drive(void* p) { auto* d = static_cast<Drive*>(p); d->task.resume(); d->fin.store(true, std::memory_order_release); }
}  // namespace

// Fiber co_await a Vulkan kernel's completion (async path via GpuPoller).
TEST(VulkanCompute, CoAwaitCompletion) {
  Scheduler sched(2, system_allocator(), 256);
  int result = -100;
  auto d = std::make_unique<Drive>();
  d->task = vk_await_coro(system_allocator(), &result);
  Fiber f(system_allocator(), kDefaultStackSize, &drive, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  compute::detail::gpu_poll_shutdown();
  EXPECT_TRUE(d->fin.load());
  if (result == 1) GTEST_SKIP() << "no Vulkan device";
  EXPECT_EQ(result, 0) << "vk co_await failed at step " << result;
}
#endif  // HYLE_HAS_FCONTEXT
