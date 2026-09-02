#include "hyle/compute/kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>   // getenv -- TEST-ONLY staging forcing
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/device.h"
#include "hyle/compute/device_pool.h"
#include "hyle/compute/memory.h"
#include "hyle/compute/stream.h"
#include "hyle/mem/buffer.h"
#include "hyle/platform/convenience.h"
#include "hyle/platform/fs.h"
// SPIR-V embedded as `inline constexpr SpirvBlob` by tools/hyle_shader.py (via
// hyle_embed_shader in tests/compute/CMakeLists.txt). Namespace hyle_test_shaders keeps the symbols from
// clashing -- the --namespace feature the tool provides. Pass the blob to Kernel::load_spirv.
#include "saxpy.spv.h"       // hyle_test_shaders::kSaxpySpirv (saxpy.comp)
#include "saxpy_spec.spv.h"  // hyle_test_shaders::kSaxpySpecSpirv (saxpy_spec.comp, #2/#3)

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

using hyle::mem::Buffer;

mem::MemoryResourceRef device_resource(const Device& device) {
  return device_memory_resource(device).value();
}

mem::MemoryResourceRef host_resource(const Device& device) {
  return host_memory_resource(device).value();
}

Status set_buffer(Buffer& buffer, std::size_t offset, const void* source,
                  std::size_t size) {
  return buffer.write(
      offset, {static_cast<const std::byte*>(source), size});
}

Status get_buffer(const Buffer& buffer, std::size_t offset, void* destination,
                  std::size_t size) {
  return buffer.read(offset, {static_cast<std::byte*>(destination), size});
}

Status clear_buffer(Buffer& buffer, std::size_t offset, std::uint8_t value,
                    std::size_t size) {
  return buffer.fill(offset, size, static_cast<std::byte>(value));
}

Status copy_buffer(Buffer& destination, const Buffer& source,
                   std::size_t destination_offset,
                   std::size_t source_offset, std::size_t size) {
  auto destination_view =
      destination.view().subview(destination_offset, size);
  if (!destination_view) return std::move(destination_view).error();
  auto source_view = source.view().subview(source_offset, size);
  if (!source_view) return std::move(source_view).error();
  return destination_view->copy_from(*source_view);
}

BufferBinding binding(const Buffer& buffer) noexcept {
  const mem::ConstBufferView view = buffer.view();
  return BufferBinding{view.native_handle(), view.offset(), view.size()};
}

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

#if defined(HYLE_TESTING)
bool environment_is_set(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const errno_t error = _dupenv_s(&value, &size, name);
  const bool present = error == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}
#endif

}  // namespace

TEST(VulkanCompute, HostBufferSeparatesNativeHandleFromMappedData) {
  Device d;
  if (!find_vulkan(d)) GTEST_SKIP() << "no Vulkan compute device";

  constexpr std::size_t kBytes = 4096;
  auto host = Buffer::create(host_resource(d), kBytes);
  ASSERT_TRUE(host.has_value()) << host.error().message();
  ASSERT_TRUE(
      host->properties().has(mem::MemoryProperty::kHostAccessible));
  ASSERT_NE(host->native_handle(), nullptr);
  auto mapping = host->view().map_host(mem::HostAccess::kReadWrite);
  ASSERT_TRUE(mapping.has_value()) << mapping.error().message();
  auto writable = mapping->writable_bytes();
  ASSERT_TRUE(writable.has_value());
  auto* mapped = reinterpret_cast<std::uint8_t*>(writable->data());
  EXPECT_NE(host->native_handle(), mapped);  // VkMem* versus VkMem::mapped
  for (std::size_t i = 0; i < kBytes; ++i)
    mapped[i] = static_cast<std::uint8_t>(i & 0xFF);
  ASSERT_TRUE(mapping->flush().ok());

  std::vector<std::uint8_t> out(kBytes, 0);
  ASSERT_TRUE(get_buffer(*host, 0, out.data(), kBytes).ok());
  for (std::size_t i = 0; i < kBytes; ++i) EXPECT_EQ(out[i], mapped[i]);
}

TEST(VulkanCompute, ZeroSizeBufferOperationsAreNoops) {
  Device d;
  if (!find_vulkan(d)) GTEST_SKIP() << "no Vulkan compute device";

  auto a = Buffer::create(device_resource(d), 0);
  auto b = Buffer::create(device_resource(d), 0);
  ASSERT_TRUE(a.has_value() && b.has_value());
  EXPECT_TRUE(set_buffer(*a, 0, nullptr, 0).ok());
  EXPECT_TRUE(get_buffer(*a, 0, nullptr, 0).ok());
  EXPECT_TRUE(copy_buffer(*b, *a, 0, 0, 0).ok());
  EXPECT_TRUE(clear_buffer(*a, 0, 0xA5, 0).ok());
}

TEST(VulkanCompute, EventRerecordAndPendingDestructionAreSafe) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";
  auto stream = Stream::create(vk);
  ASSERT_TRUE(stream.has_value()) << stream.error().message();
  {
    auto event = Event::create(vk);
    ASSERT_TRUE(event.has_value()) << event.error().message();
    stream->record(*event);
    // Vulkan fences cannot be reset while pending. The backend completes the prior record
    // before reusing the public Event, then makes destruction wait out the second record.
    stream->record(*event);
  }
  EXPECT_TRUE(stream->sync().ok());
}

#if defined(HYLE_TESTING)
// #1 pipeline cache 落盘: test-only flush hook (vulkan_backend.cc, HYLE_TESTING). Returns the
// bytes written to HYLE_VK_PIPELINE_CACHE (0 if unset/no device/empty). Declared here so the
// test can drive persistence on demand rather than waiting for process-exit (~VkContext).
// HYLE_IMPORT: the symbol lives in the hyle_compute DLL (exported under HYLE_TESTING).
namespace hyle::compute {
HYLE_IMPORT std::size_t vk_noncoherent_atom_size_for_test() noexcept;
HYLE_IMPORT void vk_mapped_range_for_test(void* native, std::size_t offset,
                                          std::size_t size,
                                          std::size_t* out_offset,
                                          std::size_t* out_size) noexcept;
HYLE_IMPORT bool vk_buffer_is_noncoherent_for_test(void* native) noexcept;
HYLE_IMPORT std::size_t vk_staging_capacity_for_test(void* native) noexcept;
HYLE_IMPORT std::size_t vk_pipeline_cache_flush_for_test() noexcept;
}

TEST(VulkanCompute, ForcedStagingIsLazyAndTransferSized) {
  if (!environment_is_set("HYLE_VK_FORCE_STAGING"))
    GTEST_SKIP() << "dedicated forced-staging CTest owns this contract";
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";

  constexpr std::size_t kBufferBytes = 1u << 20;
  constexpr std::size_t kTransferBytes = 64u << 10;
  constexpr std::size_t kOffset = 512u << 10;
  auto buffer = Buffer::create(device_resource(vk), kBufferBytes);
  ASSERT_TRUE(buffer.has_value()) << buffer.error().message();
  EXPECT_EQ(vk_staging_capacity_for_test(buffer->native_handle()), 0u);

  std::vector<std::uint8_t> input(kTransferBytes, 0xA5);
  ASSERT_TRUE(set_buffer(*buffer, kOffset, input.data(), input.size()).ok());
  const std::size_t capacity =
      vk_staging_capacity_for_test(buffer->native_handle());
  EXPECT_GE(capacity, kTransferBytes);
  EXPECT_LT(capacity, kBufferBytes);

  std::vector<std::uint8_t> output(kTransferBytes, 0);
  ASSERT_TRUE(get_buffer(*buffer, kOffset, output.data(), output.size()).ok());
  EXPECT_EQ(output, input);
  EXPECT_EQ(vk_staging_capacity_for_test(buffer->native_handle()), capacity);

  // Buffer::clear is byte-granular on every backend. vkCmdFillBuffer cannot encode
  // this unaligned range, so Vulkan must use its transfer-sized staging fallback.
  constexpr std::size_t kClearOffset = 3;
  constexpr std::size_t kClearBytes = 7;
  ASSERT_TRUE(clear_buffer(*buffer, kOffset + kClearOffset, 0x3C, kClearBytes).ok());
  ASSERT_TRUE(get_buffer(*buffer, kOffset, output.data(), output.size()).ok());
  for (std::size_t i = 0; i < output.size(); ++i) {
    const std::uint8_t expected =
        i >= kClearOffset && i < kClearOffset + kClearBytes ? 0x3C : 0xA5;
    EXPECT_EQ(output[i], expected) << i;
  }
}
#endif

TEST(VulkanCompute, DevicePoolReusesFreedVkBuffer) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";
  auto pool = DevicePool::create(vk);
  ASSERT_TRUE(pool.has_value()) << pool.error().message();
  auto first = pool->alloc(4096);
  ASSERT_TRUE(first.has_value()) << first.error().message();
  void* native = *first;
  pool->free(*first, 4096);
  auto reused = pool->alloc(4096);
  ASSERT_TRUE(reused.has_value()) << reused.error().message();
  EXPECT_EQ(*reused, native);
  pool->free(*reused, 4096);
}

TEST(VulkanCompute, TransferChainSetCopyGet) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";

  constexpr std::size_t kBytes = 4096;
  auto src = Buffer::create(device_resource(vk), kBytes);
  auto dst = Buffer::create(device_resource(vk), kBytes);
  ASSERT_TRUE(src.has_value() && dst.has_value());
  std::vector<std::uint8_t> input(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i)
    input[i] = static_cast<std::uint8_t>((i * 29) & 0xFF);
  ASSERT_TRUE(set_buffer(*src, 0, input.data(), kBytes).ok());
  ASSERT_TRUE(copy_buffer(*dst, *src, 0, 0, kBytes).ok());
  std::vector<std::uint8_t> output(kBytes, 0);
  ASSERT_TRUE(get_buffer(*dst, 0, output.data(), kBytes).ok());
  EXPECT_EQ(output, input);

  ASSERT_TRUE(clear_buffer(*src, 0, 0x5A, kBytes).ok());
  ASSERT_TRUE(copy_buffer(*dst, *src, 0, 0, kBytes).ok());
  ASSERT_TRUE(get_buffer(*dst, 0, output.data(), kBytes).ok());
  for (std::uint8_t byte : output) EXPECT_EQ(byte, 0x5A);
}

#if defined(HYLE_TESTING)
TEST(VulkanCompute, MappedRangeAlignsToNonCoherentAtom) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";

  const std::size_t atom = vk_noncoherent_atom_size_for_test();
  ASSERT_GT(atom, 0u);
  ASSERT_EQ(atom & (atom - 1), 0u);
  ASSERT_LE(atom, SIZE_MAX / 4);
  auto host = Buffer::create(host_resource(vk), atom * 4);
  ASSERT_TRUE(host.has_value()) << host.error().message();

  std::size_t offset = 0;
  std::size_t size = 0;
  vk_mapped_range_for_test(host->native_handle(), atom + 1, 1, &offset, &size);
  EXPECT_EQ(offset, atom);
  EXPECT_EQ(size, atom);

  vk_mapped_range_for_test(host->native_handle(), atom * 2 - 1, 2, &offset, &size);
  EXPECT_EQ(offset, atom);
  EXPECT_EQ(size, atom * 2);
}
#endif

// End-to-end: SPIR-V kernel + buffers + scalars through the unified launch shell.
TEST(VulkanCompute, SaxpyEndToEnd) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device (backend disabled or no loader)";

  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());

  // Load the pre-compiled SPIR-V: the embedded self-describing blob straight into load_spirv
  // (no length-prefix hand-packing -- the SpirvBlob carries its own byte size).
  auto k = Kernel::load_spirv(vk, hyle_test_shaders::kSaxpySpirv, "main");
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef resource = device_resource(vk);
  auto bx = Buffer::create(resource, bytes);
  auto by = Buffer::create(resource, bytes);
  auto bo = Buffer::create(resource, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());

  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(100 + i); }
  ASSERT_TRUE(set_buffer(*bx, 0, hx.data(), bytes).ok());
  ASSERT_TRUE(set_buffer(*by, 0, hy.data(), bytes).ok());

  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};  // group count = ceil(N/local_size), local=256
  // Unified typed shell: buffers (x,y,o) -> descriptor bindings 0,1,2; scalar `p` -> push
  // constants. One call, backend-neutral (same source runs the CUDA path in kernel_test).
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  ASSERT_TRUE(s->sync().ok());

  std::vector<float> ho(kN, -1.0f);
  ASSERT_TRUE(get_buffer(*bo, 0, ho.data(), bytes).ok());
  for (int i = 0; i < kN; ++i) {
    float expect = 2.0f * hx[i] + 3.0f * hy[i] + 0.5f;
    ASSERT_NEAR(ho[i], expect, 1e-2f) << "at " << i;
  }
}

TEST(VulkanCompute, SaxpySubviewBindingsPreserveDescriptorRanges) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";
  auto stream = Stream::create(vk);
  auto kernel =
      Kernel::load_spirv(vk, hyle_test_shaders::kSaxpySpirv, "main");
  ASSERT_TRUE(stream.has_value() && kernel.has_value());

  constexpr std::uint32_t kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  const mem::MemoryResourceRef resource = device_resource(vk);
  const std::size_t offset = resource.properties().minimum_alignment;
  auto x = Buffer::create(resource, offset + bytes);
  auto y = Buffer::create(resource, offset + bytes);
  auto output = Buffer::create(resource, offset + bytes);
  ASSERT_TRUE(x.has_value() && y.has_value() && output.has_value());
  auto xv = x->view().subview(offset, bytes);
  auto yv = y->view().subview(offset, bytes);
  auto ov = output->view().subview(offset, bytes);
  ASSERT_TRUE(xv.has_value() && yv.has_value() && ov.has_value());

  std::vector<float> host_x(kN), host_y(kN), observed(kN, 0.0f);
  for (std::uint32_t i = 0; i < kN; ++i) {
    host_x[i] = static_cast<float>(i);
    host_y[i] = static_cast<float>(1000 + i);
  }
  ASSERT_TRUE(xv->write(std::as_bytes(std::span(host_x))).ok());
  ASSERT_TRUE(yv->write(std::as_bytes(std::span(host_y))).ok());

  SaxpyParams params{kN, 1.5f, 2.0f, -3.0f};
  LaunchConfig config;
  config.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*stream, *kernel, config, *xv, *yv, *ov, params).ok());
  ASSERT_TRUE(stream->sync().ok());
  ASSERT_TRUE(ov->read(std::as_writable_bytes(std::span(observed))).ok());
  for (std::uint32_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(observed[i], 1.5f * host_x[i] + 2.0f * host_y[i] - 3.0f,
                1e-2f)
        << i;
  }
}

namespace {
// Load the saxpy SPIR-V kernel on `dev` from the embedded blob. Returned Kernel must outlive
// its launches.
Result<Kernel> load_saxpy(const Device& dev) {
  return Kernel::load_spirv(dev, hyle_test_shaders::kSaxpySpirv, "main");
}
}  // namespace

#if defined(HYLE_TESTING)
TEST(VulkanCompute, NonCoherentHostBufferRoundTrip) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan compute device";

  constexpr std::uint32_t kN = 256;
  constexpr std::size_t kBytes = kN * sizeof(float);
  mem::MemoryResourceRef host_type = host_resource(vk);
  auto bx = Buffer::create(host_type, kBytes);
  auto by = Buffer::create(host_type, kBytes);
  auto bo = Buffer::create(host_type, kBytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());
  if (!vk_buffer_is_noncoherent_for_test(bx->native_handle()))
    GTEST_SKIP() << "device exposes no non-coherent host-visible memory type";

  auto x_mapping = bx->view().map_host(mem::HostAccess::kReadWrite);
  auto y_mapping = by->view().map_host(mem::HostAccess::kReadWrite);
  auto out_mapping = bo->view().map_host(mem::HostAccess::kReadWrite);
  ASSERT_TRUE(x_mapping.has_value() && y_mapping.has_value() &&
              out_mapping.has_value());
  auto x_bytes = x_mapping->writable_bytes();
  auto y_bytes = y_mapping->writable_bytes();
  auto out_bytes = out_mapping->writable_bytes();
  ASSERT_TRUE(x_bytes.has_value() && y_bytes.has_value() &&
              out_bytes.has_value());
  auto* x = reinterpret_cast<float*>(x_bytes->data());
  auto* y = reinterpret_cast<float*>(y_bytes->data());
  auto* out = reinterpret_cast<float*>(out_bytes->data());
  for (std::uint32_t i = 0; i < kN; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = static_cast<float>(2 * i + 1);
    out[i] = -1.0f;
  }
  ASSERT_TRUE(x_mapping->flush().ok());
  ASSERT_TRUE(y_mapping->flush().ok());

  auto s = Stream::create(vk);
  auto k = load_saxpy(vk);
  ASSERT_TRUE(s.has_value() && k.has_value());
  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  ASSERT_TRUE(s->sync().ok());
  ASSERT_TRUE(out_mapping->invalidate().ok());

  for (std::uint32_t i = 0; i < kN; ++i)
    ASSERT_NEAR(out[i], 2.0f * x[i] + 3.0f * y[i] + 0.5f, 1e-2f) << "at " << i;
}
#endif

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
  mem::MemoryResourceRef bt = device_resource(vk);

  // Keep buffers + params alive across the batch (their pointers go into KernelArgs).
  std::vector<Buffer> bx, by, bo;
  std::vector<std::vector<float>> hx(kJobs), hy(kJobs);
  SaxpyParams params[kJobs];
  BufferBinding bufptrs[kJobs][3];
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
    ASSERT_TRUE(set_buffer(bx[j], 0, hx[j].data(), bytes).ok());
    ASSERT_TRUE(set_buffer(by[j], 0, hy[j].data(), bytes).ok());
    params[j] = SaxpyParams{kN, 1.0f + j, 2.0f + j, float(j)};  // distinct scalars per job
    bufptrs[j][0] = binding(bx[j]);
    bufptrs[j][1] = binding(by[j]);
    bufptrs[j][2] = binding(bo[j]);
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
    ASSERT_TRUE(get_buffer(bo[j], 0, ho.data(), bytes).ok());
    for (int i = 0; i < kN; ++i) {
      float expect = params[j].alpha * hx[j][i] + params[j].beta * hy[j][i] + params[j].bias;
      ASSERT_NEAR(ho[i], expect, 1e-1f) << "job " << j << " at " << i;
    }
  }
}

// #25 GPU timing (compute-06b): bracket a saxpy launch with two Vulkan timing events and
// read a NONZERO elapsed time. Exercises the Vulkan-specific timestamp path
// (VkQueryPool writeTimestamp + getQueryPoolResults * timestampPeriod) end-to-end -- the
// device_test GPU timing test picks the CUDA backend, so this is the only runtime check of
// the VkTimingEvent{VkFence,VkQueryPool} layout + elapsed_ns math. GPU-skips if no Vulkan.
TEST(VulkanCompute, TimingMeasuresElapsed) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());
  auto k = load_saxpy(vk);
  ASSERT_TRUE(k.has_value()) << k.error().message();

  auto start = Event::create_timing(vk);
  auto stop = Event::create_timing(vk);
  ASSERT_TRUE(start.has_value()) << start.error().message();
  ASSERT_TRUE(stop.has_value()) << stop.error().message();

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());
  std::vector<float> hx(kN, 1.0f), hy(kN, 2.0f);
  ASSERT_TRUE(set_buffer(*bx, 0, hx.data(), bytes).ok());
  ASSERT_TRUE(set_buffer(*by, 0, hy.data(), bytes).ok());

  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};

  s->record(*start);
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  s->record(*stop);
  ASSERT_TRUE(s->sync().ok());
  EXPECT_EQ(stop->query(), EventStatus::kReady)
      << "Stream::sync must include the timing marker queued after the launch";

  auto ns = elapsed_ns(*start, *stop);
  ASSERT_TRUE(ns.has_value()) << ns.error().message();
  EXPECT_GT(*ns, 0u);  // a real Vulkan dispatch took nonzero GPU time
}

// #1 pipeline cache 落盘: build a pipeline, flush the shared cache to HYLE_VK_PIPELINE_CACHE,
// and verify a NON-EMPTY blob was written. Persistence is opt-in via that env var, so this
// runs only in the dedicated CTest variant (hyle_compute_vulkan_cache_test) that sets it; the
// plain variant leaves it unset -> SKIP. Proves the save path (getPipelineCacheData -> framed
// file); the load+UUID path is exercised by the SAME variant's SECOND run seeding from this
// file (saxpy stays green => the seeded cache was accepted, not rejected).
#if defined(HYLE_TESTING)
TEST(VulkanCompute, PipelineCachePersist) {
  auto path = platform::get_env_var("HYLE_VK_PIPELINE_CACHE");
  if (!path || path->empty())
    GTEST_SKIP() << "HYLE_VK_PIPELINE_CACHE unset (persistence is opt-in; cache variant sets it)";
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";

  // Build at least one pipeline so the driver has something to cache: a full saxpy launch.
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());
  auto k = load_saxpy(vk);
  ASSERT_TRUE(k.has_value()) << k.error().message();
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());
  SaxpyParams p{kN, 1.0f, 1.0f, 0.0f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  ASSERT_TRUE(s->sync().ok());

  // Flush on demand (the process-exit dtor path can't be observed in-process).
  std::size_t written = hyle::compute::vk_pipeline_cache_flush_for_test();
  EXPECT_GT(written, 0u) << "cache flush wrote an empty payload";

  // The framed file must exist and exceed the header (magic + UUID = 24 bytes).
  auto size = platform::file_size(path->c_str());
  ASSERT_TRUE(size.has_value())
      << "cache file not created at " << *path << ": "
      << size.error().message();
  EXPECT_GT(*size, 24u) << "cache file has no payload beyond the header";
}
#endif

// #2/#3 specialization constants: load saxpy_spec.comp with local_size (id 0) and SCALE (id 1,
// default 7.0) as spec constants, overriding SCALE=3.0 at build. The result must be scaled by
// the SUPPLIED 3.0, not the shader default -- so a green check proves the spec values reached
// the driver's compile (a dropped spec would compute *7.0 and mismatch). local_size=64 also
// means the grid is ceil(N/64), exercising a spec-driven workgroup size.
TEST(VulkanCompute, SpecializationConstants) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());

  // Spec data blob: [uint32 local_size_x @0][float SCALE @4]; entries map GLSL constant_ids.
  constexpr std::uint32_t kLocalSize = 64;
  constexpr float kScale = 3.0f;  // != shader default 7.0 -> proves the override took
  alignas(4) unsigned char spec_data[8];
  std::memcpy(spec_data + 0, &kLocalSize, 4);
  std::memcpy(spec_data + 4, &kScale, 4);
  SpecConstant specs[2] = {{/*id=*/0, /*offset=*/0, /*size=*/4},
                           {/*id=*/1, /*offset=*/4, /*size=*/4}};
  auto k = Kernel::load_spirv(vk, hyle_test_shaders::kSaxpySpecSpirv, "main",
                              specs, 2, spec_data, sizeof(spec_data));
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());
  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(100 + i); }
  ASSERT_TRUE(set_buffer(*bx, 0, hx.data(), bytes).ok());
  ASSERT_TRUE(set_buffer(*by, 0, hy.data(), bytes).ok());

  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, kLocalSize), 1, 1};  // grid uses the spec-set local_size (64)
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, *bo, p).ok());
  ASSERT_TRUE(s->sync().ok());

  std::vector<float> ho(kN, -1.0f);
  ASSERT_TRUE(get_buffer(*bo, 0, ho.data(), bytes).ok());
  for (int i = 0; i < kN; ++i) {
    float expect = kScale * (2.0f * hx[i] + 3.0f * hy[i] + 0.5f);  // scaled by the SUPPLIED 3.0
    ASSERT_NEAR(ho[i], expect, 1e-1f) << "at " << i << " (spec SCALE not applied?)";
  }
}

// #16 record/replay: record saxpy ONCE, then replay it multiple times, changing the INPUT
// buffer contents between replays. The recorded command buffer is reused verbatim (same
// bindings/push/dispatch); each replay must reflect the CURRENT buffer data -- proving the
// record-once/replay-many semantics AND that data changes flow through without re-recording.
TEST(VulkanCompute, RecordReplayReflectsBufferChanges) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());
  auto k = load_saxpy(vk);
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 2048;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value() && bo.has_value());

  SaxpyParams p{kN, 2.0f, 3.0f, 1.0f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};

  // Record once (bindings to bx/by/bo + push p + dispatch frozen). Buffers must outlive `rl`.
  BufferBinding bufs[3] = {binding(*bx), binding(*by), binding(*bo)};
  ScalarField field{0, sizeof(p)};
  KernelArgs ka{bufs, 3, &p, sizeof(p), &field, 1};
  auto rl = RecordedLaunch::record(*s, *k, cfg, ka);
  ASSERT_TRUE(rl.has_value()) << rl.error().message();

  // Replay 3 times; each iteration writes NEW input data into the SAME buffers and expects the
  // output to track it (the recording didn't change, only the buffer contents did).
  for (int iter = 1; iter <= 3; ++iter) {
    std::vector<float> hx(kN), hy(kN);
    for (int i = 0; i < kN; ++i) { hx[i] = float(i * iter); hy[i] = float(i + iter); }
    ASSERT_TRUE(set_buffer(*bx, 0, hx.data(), bytes).ok());
    ASSERT_TRUE(set_buffer(*by, 0, hy.data(), bytes).ok());

    ASSERT_TRUE(rl->replay(*s).ok()) << "iter " << iter;
    // replay() is async. Submit the immutable command buffer again while the first replay
    // may still be pending; Vulkan requires SIMULTANEOUS_USE for this legal API shape.
    if (iter == 1) {
      ASSERT_TRUE(rl->replay(*s).ok());
    }
    ASSERT_TRUE(s->sync().ok());

    std::vector<float> ho(kN, -1.0f);
    ASSERT_TRUE(get_buffer(*bo, 0, ho.data(), bytes).ok());
    for (int i = 0; i < kN; ++i) {
      float expect = 2.0f * hx[i] + 3.0f * hy[i] + 1.0f;  // p.alpha/beta/bias from the recording
      ASSERT_NEAR(ho[i], expect, 1e-1f) << "iter " << iter << " at " << i;
    }
  }
}

// #11/#12 precise barriers: a DEPENDENT chain in one batch. Job j reads job (j-1)'s OUTPUT
// buffer (o[j-1] == x[j]), so the inter-dispatch hazard barrier MUST fire to order the chain;
// a wrong "no barrier" would race and read stale/partial data. Complements BatchedSaxpy
// (disjoint buffers -> the barrier is SKIPPED, letting the GPU overlap). Golden-checks the
// composed result so a missing barrier surfaces as a value mismatch, not just a hang.
TEST(VulkanCompute, BatchedDependentChain) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  ASSERT_TRUE(s.has_value());
  auto k = load_saxpy(vk);
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kJobs = 4;
  constexpr int kN = 2048;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);

  // kJobs+1 buffers b0..bkJobs chained: job j computes b[j+1] = 1*b[j] + 0*b[j] + bias_j
  // (alpha=1,beta=0 so it's b[j+1]=b[j]+bias_j -- a pure dependent accumulate). Each job's
  // x AND y bind the SAME input b[j] (2 reads of one buffer), output b[j+1].
  std::vector<Buffer> bufs;
  bufs.reserve(kJobs + 1);
  for (int j = 0; j <= kJobs; ++j) {
    auto b = Buffer::create(bt, bytes);
    ASSERT_TRUE(b.has_value());
    bufs.push_back(std::move(*b));
  }
  std::vector<float> seed(kN, 10.0f);
  ASSERT_TRUE(set_buffer(bufs[0], 0, seed.data(), bytes).ok());  // b0 = 10

  SaxpyParams params[kJobs];
  BufferBinding bufptrs[kJobs][3];
  ScalarField fields[kJobs];
  BatchItem items[kJobs];
  for (int j = 0; j < kJobs; ++j) {
    params[j] = SaxpyParams{kN, 1.0f, 0.0f, float(j + 1)};  // b[j+1] = b[j] + (j+1)
    bufptrs[j][0] = binding(bufs[j]);      // x = b[j]  (shared read)
    bufptrs[j][1] = binding(bufs[j]);      // y = b[j]  (beta=0)
    bufptrs[j][2] = binding(bufs[j + 1]);  // output becomes next input
    fields[j] = ScalarField{0, sizeof(SaxpyParams)};
    items[j].kernel = &*k;
    items[j].config = LaunchConfig{};
    items[j].config.grid = {blocks_for(kN, 256), 1, 1};
    items[j].args = KernelArgs{bufptrs[j], 3, &params[j], sizeof(SaxpyParams), &fields[j], 1};
  }

  ASSERT_TRUE(launch_batch(*s, items, kJobs).ok());
  ASSERT_TRUE(s->sync().ok());

  // b[kJobs] = 10 + sum(1..kJobs) = 10 + 10 = 20 for kJobs=4. Every stage must have been
  // ordered; a dropped barrier would leave a downstream buffer reading a pre-write value.
  float expect = 10.0f;
  for (int j = 0; j < kJobs; ++j) expect += float(j + 1);
  std::vector<float> ho(kN, -1.0f);
  ASSERT_TRUE(get_buffer(bufs[kJobs], 0, ho.data(), bytes).ok());
  for (int i = 0; i < kN; ++i) ASSERT_NEAR(ho[i], expect, 1e-2f) << "at " << i;
}

// ---- Vulkan CONTRACT death tests --------------------------------------------
// Frozen pipeline shape and grid-limit violations are caller bugs and remain fatal.
// Device capability differences such as push-constant capacity return Status instead.
// All GPU-skip if no Vulkan device.
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
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  SaxpyParams p{kN, 1.0f, 1.0f, 0.0f};
  LaunchConfig cfg; cfg.grid = {blocks_for(kN, 256), 1, 1};
  (void)launch(*s, *k, cfg, *bx, *by, *bo, p);  // first launch freezes {3 buffers, 16B push}
  (void)s->sync();
  // Relaunch with only 2 buffers (same push) -> buffer_count mismatch -> fatal (:665).
  BufferBinding two[2] = {binding(*bx), binding(*by)};
  ScalarField f{0, sizeof(SaxpyParams)};
  KernelArgs bad{two, 2, &p, sizeof(SaxpyParams), &f, 1};
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
  mem::MemoryResourceRef bt = device_resource(vk);
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

TEST(VulkanCompute, PushConstantSizeOverLimitReturnsStatus) {
  Device vk;
  if (!find_vulkan(vk)) GTEST_SKIP() << "no Vulkan device";
  auto s = Stream::create(vk);
  auto k = load_saxpy(vk);
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  ASSERT_TRUE(s.has_value() && k.has_value() && bx.has_value() &&
              by.has_value() && bo.has_value());
  alignas(16) unsigned char scalar = 0;
  BufferBinding three[3] = {binding(*bx), binding(*by), binding(*bo)};
  ScalarField f{0, UINT32_MAX};
  KernelArgs bad{three, 3, &scalar, UINT32_MAX, &f, 1};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  EXPECT_FALSE(launch(*s, *k, cfg, bad).ok());
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
  mem::MemoryResourceRef bt = device_resource(vk);
  auto bx = Buffer::create(bt, bytes); auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  if (!bx.has_value() || !by.has_value() || !bo.has_value()) { *out = -3; co_return -3; }
  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(i) + 1.0f; }
  if (!set_buffer(*bx, 0, hx.data(), bytes).ok() || !set_buffer(*by, 0, hy.data(), bytes).ok()) { *out = -4; co_return -4; }

  SaxpyParams p{kN, 2.0f, 3.0f, 1.0f};
  LaunchConfig cfg; cfg.grid = {blocks_for(kN, 256), 1, 1};
  if (!launch(*s, *k, cfg, *bx, *by, *bo, p).ok()) { *out = -5; co_return -5; }

  Status st = co_await StreamComplete{&*s};  // SUSPEND until GPU done (poller wakes us)
  if (!st.ok()) { *out = -6; co_return -6; }

  std::vector<float> ho(kN, -1.0f);
  if (!get_buffer(*bo, 0, ho.data(), bytes).ok()) { *out = -7; co_return -7; }
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
  Fiber f(kDefaultStackSize, &drive, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  compute::detail::gpu_poll_shutdown();
  EXPECT_TRUE(d->fin.load());
  if (result == 1) GTEST_SKIP() << "no Vulkan device";
  EXPECT_EQ(result, 0) << "vk co_await failed at step " << result;
}
#endif  // HYLE_HAS_FCONTEXT
