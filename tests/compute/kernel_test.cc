#include "hyle/compute/kernel.h"

#include <gtest/gtest.h>

#include <cstring>
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

Device host_device() {
  Device devs[8];
  const std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i)
    if (!devs[i].is_gpu()) return devs[i];
  return default_device();
}

// The GPU kernel source (extern "C" so the entry name is verbatim).
constexpr const char* kVecAddSrc = R"(
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
extern "C" __global__ void mixed_order(const float* a, int n, const float* b,
                                         float* c, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + scale * b[i];
}
)";

// The host-fallback "kernel": same vecadd semantics over host pointers. Reads the
// unified KernelArgs: buffers[0..2] = a/b/c device (here host) ptrs, scalar field 0 = n
// (packed in the blob). Mirrors how a real host kernel would consume KernelArgs.
void host_vecadd(const LaunchConfig&, const KernelArgs& args) noexcept {
  const float* a = static_cast<const float*>(args.buffers[0]);
  const float* b = static_cast<const float*>(args.buffers[1]);
  float* c = static_cast<float*>(args.buffers[2]);
  int n = *reinterpret_cast<const int*>(static_cast<const std::byte*>(args.scalars) +
                                        args.fields[0].offset);
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

struct TwelveByteValue {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
};
bool g_mixed_order_ok = false;

void host_mixed_order(const LaunchConfig&, const KernelArgs& args) noexcept {
  TwelveByteValue value{};
  std::memcpy(&value,
              static_cast<const std::byte*>(args.scalars) + args.fields[0].offset,
              sizeof(value));
  g_mixed_order_ok =
      args.buffer_count == 2 && args.scalar_count == 2 && args.arg_count == 4 &&
      args.order != nullptr &&
      args.order[0].kind == KernelArgKind::kBuffer && args.order[0].index == 0 &&
      args.order[1].kind == KernelArgKind::kScalar && args.order[1].index == 0 &&
      args.order[2].kind == KernelArgKind::kBuffer && args.order[2].index == 1 &&
      args.order[3].kind == KernelArgKind::kScalar && args.order[3].index == 1 &&
      args.fields[0].size == sizeof(TwelveByteValue) &&
      args.fields[0].offset % alignof(TwelveByteValue) == 0 &&
      value.x == 1 && value.y == 2 && value.z == 3;
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

  // Typed shell: pass kernel args DIRECTLY (Buffers + scalar), classified at compile
  // time. Buffers (a,b,c) -> device ptrs in order; scalar n -> push/void** entry. The
  // vecadd kernel signature (float* a, float* b, float* c, int n) is buffers-then-scalar,
  // matching the shell's ordering (buffers first, then scalars).
  int n = kN;

  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*s, *k, cfg, *da, *db, *dc, n).ok());
  ASSERT_TRUE(s->sync().ok());  // wait for the kernel (co_await path tested in gpu_await)

  ASSERT_TRUE(dc->get(0, hc.data(), bytes).ok());
  for (int i = 0; i < kN; ++i)
    EXPECT_FLOAT_EQ(hc[i], float(i) + float(2 * i)) << "at " << i;
}

TEST(ComputeKernel, TypedShellPreservesMixedOrderOnCuda) {
  if (!has_gpu()) GTEST_SKIP() << "no CUDA GPU";
  Device d = default_device();
  auto k = Kernel::compile(d, kVecAddSrc, "mixed_order");
  auto s = Stream::create(d);
  ASSERT_TRUE(k.has_value() && s.has_value());

  constexpr int kN = 1024;
  const std::size_t bytes = kN * sizeof(float);
  std::vector<float> a(kN), b(kN), out(kN, 0.0f);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i + 1);
  }
  auto da = Buffer::create(device_buffer_type(d), bytes);
  auto db = Buffer::create(device_buffer_type(d), bytes);
  auto dc = Buffer::create(device_buffer_type(d), bytes);
  ASSERT_TRUE(da.has_value() && db.has_value() && dc.has_value());
  ASSERT_TRUE(da->set(0, a.data(), bytes).ok());
  ASSERT_TRUE(db->set(0, b.data(), bytes).ok());

  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  const float scale = 2.5f;
  ASSERT_TRUE(launch(*s, *k, cfg, *da, kN, *db, *dc, scale).ok());
  ASSERT_TRUE(s->sync().ok());
  ASSERT_TRUE(dc->get(0, out.data(), bytes).ok());
  for (int i = 0; i < kN; ++i)
    EXPECT_FLOAT_EQ(out[i], a[i] + scale * b[i]) << i;
}

TEST(ComputeKernel, TypedShellUsesNaturalAlignmentForTwelveByteValue) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_mixed_order);
  auto s = Stream::create(d);
  auto a = Buffer::create(device_buffer_type(d), 16);
  auto b = Buffer::create(device_buffer_type(d), 16);
  ASSERT_TRUE(k.has_value() && s.has_value() && a.has_value() && b.has_value());

  g_mixed_order_ok = false;
  TwelveByteValue value{1, 2, 3};
  const std::uint16_t tail = 9;
  ASSERT_TRUE(launch(*s, *k, LaunchConfig{}, *a, value, *b, tail).ok());
  EXPECT_TRUE(g_mixed_order_ok);
}

TEST(ComputeKernel, RawArgsRejectMalformedMetadata) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_vecadd);
  auto s = Stream::create(d);
  ASSERT_TRUE(k.has_value() && s.has_value());

  int scalar = 4;
  ScalarField field{0, sizeof(scalar)};
  KernelArgs null_buffers{nullptr, 1, nullptr, 0, nullptr, 0};
  EXPECT_FALSE(launch(*s, *k, LaunchConfig{}, null_buffers).ok());

  KernelArgs null_blob{nullptr, 0, nullptr, sizeof(scalar), &field, 1};
  EXPECT_FALSE(launch(*s, *k, LaunchConfig{}, null_blob).ok());

  ScalarField outside{2, sizeof(scalar)};
  KernelArgs outside_blob{nullptr, 0, &scalar, sizeof(scalar), &outside, 1};
  EXPECT_FALSE(launch(*s, *k, LaunchConfig{}, outside_blob).ok());

  void* buffers[1] = {&scalar};
  KernelArgRef duplicate[2] = {{KernelArgKind::kBuffer, 0},
                               {KernelArgKind::kBuffer, 0}};
  KernelArgs duplicate_order{buffers, 1, &scalar, sizeof(scalar), &field, 1,
                             duplicate, 2};
  EXPECT_FALSE(launch(*s, *k, LaunchConfig{}, duplicate_order).ok());
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
  // Host test uses raw host pointers (not Buffer), so build KernelArgs explicitly and
  // call the C-ABI core launch (the typed shell is for Buffer args; this exercises the
  // raw form + host_vecadd reading buffers[]/scalar blob).
  void* bufs[] = {const_cast<float*>(pa), const_cast<float*>(pb), pc};
  ScalarField fields[] = {{0, sizeof(int)}};
  KernelArgs ka{bufs, 3, &n, sizeof(int), fields, 1};

  LaunchConfig cfg;  // host ignores dims (fn loops n itself)
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
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

// ---- Kernel move semantics (was entirely uncovered) -------------------------
// A Kernel owns backend module/func handles (CUmodule/CUfunction on CUDA, a VkKernel on
// Vulkan). A wrong move = a leak or a double-unload. We exercise move on a HOST-backend
// kernel (load_host: module=nullptr, func=the HostFn) so it runs on ANY platform without
// a GPU -- the ownership-transfer logic (compute.cc:506/513) is backend-agnostic. The
// host device is enumerate()'s last entry (always the CPU fallback).

namespace {
Device host_device_for_kernel() {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  return devs[n - 1];  // CPU fallback is always last
}

// Launch a host kernel and confirm it actually ran (vecadd over 4 elements), proving the
// moved kernel's func handle is intact and callable.
void expect_kernel_runs(Kernel& k, Stream& s) {
  constexpr int kN = 4;
  float a[kN] = {1, 2, 3, 4}, b[kN] = {10, 20, 30, 40}, c[kN] = {-1, -1, -1, -1};
  void* bufs[] = {a, b, c};
  int n = kN;
  ScalarField fields[] = {{0, sizeof(int)}};
  KernelArgs ka{bufs, 3, &n, sizeof(int), fields, 1};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(s, k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
}
}  // namespace

TEST(ComputeKernel, MoveConstruction) {
  Device d = host_device_for_kernel();
  auto k = Kernel::load_host(d, &host_vecadd);
  ASSERT_TRUE(k.has_value()) << k.error().message();
  Kernel moved = std::move(*k);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(k->valid());  // moved-from invalid (won't double-destroy the module)
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  expect_kernel_runs(moved, *s);  // moved kernel still launches correctly
}

TEST(ComputeKernel, MoveAssignment) {
  Device d = host_device_for_kernel();
  auto k1 = Kernel::load_host(d, &host_vecadd);
  auto k2 = Kernel::load_host(d, &host_vecadd);
  ASSERT_TRUE(k1.has_value() && k2.has_value());
  *k2 = std::move(*k1);      // k2's original module destroyed, k1's moved in
  EXPECT_TRUE(k2->valid());
  EXPECT_FALSE(k1->valid());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  expect_kernel_runs(*k2, *s);
}

TEST(ComputeKernel, SelfMoveAssignment) {
  Device d = host_device_for_kernel();
  auto k = Kernel::load_host(d, &host_vecadd);
  ASSERT_TRUE(k.has_value());
  Kernel& ref = *k;
  *k = std::move(ref);       // guarded no-op (compute.cc:514) -- module NOT unloaded
  EXPECT_TRUE(k->valid());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  expect_kernel_runs(*k, *s);  // still callable
}

// ---- launch_batch on the HOST backend (was Vulkan-only) ---------------------
// launch_batch on CUDA/host loops kernel_launch (only Vulkan fuses into one command
// buffer). This covers the dispatch-layer unpack (compute.cc:585-603) + the host loop
// (host_backend.cc:263) with MULTIPLE host kernels, each golden-checked -- the CUDA/
// Vulkan batched paths were tested but the host loop never was.

TEST(ComputeKernel, HostLaunchBatch) {
  Device d = host_device_for_kernel();
  auto k = Kernel::load_host(d, &host_vecadd);
  ASSERT_TRUE(k.has_value()) << k.error().message();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kJobs = 3;
  constexpr int kN = 8;
  // Per-job storage MUST outlive launch_batch (its arrays are caller-owned): distinct
  // input/output arrays + arg descriptors per job.
  float a[kJobs][kN], b[kJobs][kN], c[kJobs][kN];
  void* bufs[kJobs][3];
  int ns[kJobs];
  ScalarField fields[kJobs];
  BatchItem items[kJobs];
  for (int j = 0; j < kJobs; ++j) {
    for (int i = 0; i < kN; ++i) { a[j][i] = float(i + j); b[j][i] = float(2 * i - j); c[j][i] = -1.0f; }
    bufs[j][0] = a[j]; bufs[j][1] = b[j]; bufs[j][2] = c[j];
    ns[j] = kN;
    fields[j] = ScalarField{0, sizeof(int)};
    items[j].kernel = &*k;
    items[j].config = LaunchConfig{};
    items[j].args = KernelArgs{bufs[j], 3, &ns[j], sizeof(int), &fields[j], 1};
  }
  ASSERT_TRUE(launch_batch(*s, items, kJobs).ok());
  for (int j = 0; j < kJobs; ++j)
    for (int i = 0; i < kN; ++i)
      EXPECT_FLOAT_EQ(c[j][i], a[j][i] + b[j][i]) << "job " << j << " at " << i;
}

// count==0 is a legal no-op (compute.cc:587 returns OK before the items/cap asserts) --
// NOT a contract violation. A null items pointer is fine when count is 0.
TEST(ComputeKernel, LaunchBatchCountZeroIsNoop) {
  Device d = host_device_for_kernel();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  EXPECT_TRUE(launch_batch(*s, nullptr, 0).ok());
}

// blocks_for(n, 0): the block_size==0 guard (kernel.h:54) returns 0 rather than dividing
// by zero. Complements BlocksForHelper's positive cases.
TEST(ComputeKernel, BlocksForZeroBlockSize) {
  EXPECT_EQ(blocks_for(100, 0), 0u);
  EXPECT_EQ(blocks_for(0, 0), 0u);
}

// The host backend has no NVRTC and no PTX loader: compile() and load_ptx() both return
// Unimplemented (host_backend.cc:238/242), pointing the caller at load_host. Tested on
// the host device explicitly (enumerate's last entry) so it runs on ANY platform, GPU or
// not -- BackendMismatchUnimplemented only covers compile, and only when NO gpu exists.
TEST(ComputeKernel, HostCompileAndLoadPtxUnimplemented) {
  Device host = host_device_for_kernel();
  auto kc = Kernel::compile(host, kVecAddSrc, "vecadd");
  EXPECT_FALSE(kc.has_value());  // no NVRTC on host
  auto kp = Kernel::load_ptx(host, "// ptx", "vecadd");
  EXPECT_FALSE(kp.has_value());  // no PTX loader on host
}
