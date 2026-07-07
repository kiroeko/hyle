#include "hyle/compute/kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/stream.h"
#include "saxpy_spirv.h"  // kSaxpySpirv[] -- glslang-compiled saxpy.comp (checked in)

// compute-06 test: run a self-authored SPIR-V compute kernel through HYLE's Vulkan
// backend end-to-end -- the Vulkan twin of kernel_test's CUDA path. saxpy:
// o[i] = alpha*x[i] + beta*y[i] + bias, with 3 storage buffers (bindings 0,1,2) and a
// push-constant block {uint n; float alpha,beta,bias}. Exercises: volk device, Buffer
// alloc (ReBAR host-mappable), the unified launch typed shell (buffers -> descriptors,
// scalars -> push constants), and readback. GPU-skips if no Vulkan device is present.

using namespace hyle;
using namespace hyle::compute;

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
