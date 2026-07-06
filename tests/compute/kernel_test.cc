#include "hyle/compute/kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/device_pool.h"
#include "hyle/compute/stream.h"

// compute-04 test: kernel launch. On a real GPU, NVRTC-compile a vecadd CUDA C source
// -> PTX -> load -> launch -> verify (device/buffer/pool/kernel + runtime compile all
// real). On the host fallback (no GPU), a registered host fn is the "kernel" and
// launch calls it inline -- exercising the abstraction with no device. The test TU is
// pure C++ (all CUDA/NVRTC inside hyle_compute), needs no nvcc; GPU vs host chosen at
// runtime.

using namespace hyle;
using namespace hyle::compute;

namespace {

bool has_gpu() {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i)
    if (devs[i].is_gpu()) return true;
  return false;
}

// The GPU kernel source (extern "C" so the entry name is verbatim).
constexpr const char* kVecAddSrc = R"(
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
)";

// The host-fallback "kernel": same vecadd semantics over host pointers. args mirrors
// the GPU layout: {&a, &b, &c, &n} where a/b/c are float* (host memory here) and n int.
void host_vecadd(const LaunchConfig&, void** args) noexcept {
  const float* a = *static_cast<const float* const*>(args[0]);
  const float* b = *static_cast<const float* const*>(args[1]);
  float* c = *static_cast<float* const*>(args[2]);
  int n = *static_cast<const int*>(args[3]);
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

}  // namespace

// ---- blocks_for helper ------------------------------------------------------

TEST(ComputeKernel, BlocksForHelper) {
  EXPECT_EQ(blocks_for(0, 256), 0u);
  EXPECT_EQ(blocks_for(1, 256), 1u);
  EXPECT_EQ(blocks_for(256, 256), 1u);
  EXPECT_EQ(blocks_for(257, 256), 2u);
  EXPECT_EQ(blocks_for(1000, 256), 4u);
}

// ---- NVRTC compile + launch end-to-end (GPU) --------------------------------

TEST(ComputeKernel, NvrtcVecAddEndToEnd) {
  if (!has_gpu()) GTEST_SKIP() << "no CUDA GPU -- NVRTC path is GPU-only";
  Device d = default_device();

  auto k = Kernel::compile(d, kVecAddSrc, "vecadd");
  ASSERT_TRUE(k.has_value()) << k.error().message();

  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  std::vector<float> ha(kN), hb(kN), hc(kN, -1.0f);
  for (int i = 0; i < kN; ++i) { ha[i] = float(i); hb[i] = float(2 * i); }

  auto da = Buffer::create(device_buffer_type(d), bytes);
  auto db = Buffer::create(device_buffer_type(d), bytes);
  auto dc = Buffer::create(device_buffer_type(d), bytes);
  ASSERT_TRUE(da.has_value() && db.has_value() && dc.has_value());
  ASSERT_TRUE(da->set(0, ha.data(), bytes).ok());
  ASSERT_TRUE(db->set(0, hb.data(), bytes).ok());

  // Launch args: pointers to each kernel argument (device ptrs + the scalar n).
  void* pa = da->device_ptr();
  void* pb = db->device_ptr();
  void* pc = dc->device_ptr();
  int n = kN;
  void* args[] = {&pa, &pb, &pc, &n};

  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*s, *k, cfg, args).ok());
  ASSERT_TRUE(s->sync().ok());  // wait for the kernel (co_await path tested in gpu_await)

  ASSERT_TRUE(dc->get(0, hc.data(), bytes).ok());
  for (int i = 0; i < kN; ++i)
    EXPECT_FLOAT_EQ(hc[i], float(i) + float(2 * i)) << "at " << i;
}

// ---- compile cache: second compile of same source hits cache ----------------

TEST(ComputeKernel, CompileCacheHit) {
  if (!has_gpu()) GTEST_SKIP() << "no CUDA GPU";
  Device d = default_device();
  // Two compiles of the same source; both must succeed and be launchable. (The cache
  // is internal; we assert correctness of the cached path, not timing.)
  auto k1 = Kernel::compile(d, kVecAddSrc, "vecadd");
  auto k2 = Kernel::compile(d, kVecAddSrc, "vecadd");
  ASSERT_TRUE(k1.has_value()) << k1.error().message();
  ASSERT_TRUE(k2.has_value()) << k2.error().message();
}

// ---- compile error surfaces NVRTC log ---------------------------------------

TEST(ComputeKernel, CompileErrorReturnsStatus) {
  if (!has_gpu()) GTEST_SKIP() << "no CUDA GPU";
  Device d = default_device();
  auto k = Kernel::compile(d, "this is not valid cuda", "nope");
  EXPECT_FALSE(k.has_value());  // returns a Status, not a crash
}

// ---- DevicePool as kernel scratch (compute-02c consumer) --------------------

TEST(ComputeKernel, DevicePoolScratch) {
  if (!has_gpu()) GTEST_SKIP() << "no CUDA GPU";
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());
  // A kernel's transient scratch buffer borrowed from the pool (LIFO via PoolAlloc).
  PoolAlloc<float> scratch(*pool, 1024);
  EXPECT_TRUE(scratch.valid());
  EXPECT_NE(scratch.get(), nullptr);
}

// ---- host fallback: load_host + launch calls the fn inline ------------------

TEST(ComputeKernel, HostFallbackLaunch) {
  if (has_gpu()) GTEST_SKIP() << "host-fallback path; a GPU present would use NVRTC";
  Device d = default_device();

  auto k = Kernel::load_host(d, &host_vecadd);
  ASSERT_TRUE(k.has_value()) << k.error().message();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 512;
  std::vector<float> a(kN), b(kN), c(kN, -1.0f);
  for (int i = 0; i < kN; ++i) { a[i] = float(i); b[i] = float(i) + 0.5f; }
  const float* pa = a.data();
  const float* pb = b.data();
  float* pc = c.data();
  int n = kN;
  void* args[] = {&pa, &pb, &pc, &n};

  LaunchConfig cfg;  // host ignores dims (fn loops n itself)
  ASSERT_TRUE(launch(*s, *k, cfg, args).ok());
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(c[i], float(i) + float(i) + 0.5f);
}

// load_host on a GPU device is Unimplemented (and compile on host is too).
TEST(ComputeKernel, BackendMismatchUnimplemented) {
  Device d = default_device();
  if (has_gpu()) {
    auto k = Kernel::load_host(d, &host_vecadd);
    EXPECT_FALSE(k.has_value());  // GPU can't run a host fn as a kernel
  } else {
    auto k = Kernel::compile(d, kVecAddSrc, "vecadd");
    EXPECT_FALSE(k.has_value());  // host has no NVRTC
  }
}
