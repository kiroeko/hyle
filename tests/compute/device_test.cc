#include "hyle/compute/device.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "hyle/compute/device_pool.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/memory.h"
#include "hyle/compute/stream.h"
#include "hyle/mem/buffer.h"

// compute-01 tests. Two layers:
//   1. HOST fallback -- deterministic, runs on ANY platform (no GPU, CUDA not
//      built): enumeration, stream/event lifecycle, record->query==kReady, an
//      async copy that actually moves bytes. These assert the completion CONTRACT
//      the whole pillar rests on, without a device.
//   2. GPU (CUDA) -- SKIPPED when no GPU is present (enumerate finds none): the
//      real payload -- a device malloc + async H2D/D2H copy makes an Event
//      observably transition kNotReady -> kReady via cudaEventQuery polling. This
//      proves the reap model (compute-design §2) on real hardware, not a stub.
//
// The test TU is pure C++ (it only calls HYLE's Device/Stream/Event); all CUDA is
// inside hyle_compute. So it needs no nvcc and detects the GPU at RUNTIME.

using namespace hyle;
using namespace hyle::compute;

namespace {

// Find a GPU device if one exists (CUDA compiled in AND a card present), else return
// the host device. Lets each test target real hardware when available, host otherwise.
Device find_gpu_or_host(bool* is_gpu) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  EXPECT_GE(n, 1u);  // host is always present
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].is_gpu()) {
      *is_gpu = true;
      return devs[i];
    }
  }
  *is_gpu = false;
  return default_device();
}

bool find_cuda(Device* out) {
  Device devs[8];
  const std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (std::string_view(devs[i].backend()) == "cuda") {
      *out = devs[i];
      return true;
    }
  }
  return false;
}

void no_op_host_kernel(const LaunchConfig&, const KernelArgs&) noexcept {}

}  // namespace

// ---- device enumeration (host always present) -------------------------------

static_assert(std::is_trivially_copyable_v<Device>);

TEST(ComputeDevice, DefaultConstructionIsInvalidPlaceholder) {
  Device d;
  EXPECT_FALSE(d.valid());
  EXPECT_EQ(d.ordinal(), -1);
}

TEST(ComputeDevice, InvalidPlaceholderFactoriesReturnInvalidArgument) {
  const Device invalid;

  auto stream = Stream::create(invalid);
  ASSERT_FALSE(stream.has_value());
  EXPECT_TRUE(stream.error().IsInvalidArgument());
  auto event = Event::create(invalid);
  ASSERT_FALSE(event.has_value());
  EXPECT_TRUE(event.error().IsInvalidArgument());
  auto timing = Event::create_timing(invalid);
  ASSERT_FALSE(timing.has_value());
  EXPECT_TRUE(timing.error().IsInvalidArgument());
  auto pool = DevicePool::create(invalid);
  ASSERT_FALSE(pool.has_value());
  EXPECT_TRUE(pool.error().IsInvalidArgument());

  auto compiled = Kernel::compile(invalid, "", "entry");
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  auto ptx = Kernel::load_ptx(invalid, "", "entry");
  ASSERT_FALSE(ptx.has_value());
  EXPECT_TRUE(ptx.error().IsInvalidArgument());
  auto spirv = Kernel::load_spirv(invalid, SpirvBlob{}, "main");
  ASSERT_FALSE(spirv.has_value());
  EXPECT_TRUE(spirv.error().IsInvalidArgument());
  auto host = Kernel::load_host(invalid, &no_op_host_kernel);
  ASSERT_FALSE(host.has_value());
  EXPECT_TRUE(host.error().IsInvalidArgument());
}

TEST(ComputeDevice, EnumerateNeverEmpty) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  ASSERT_GE(n, 1u);  // the host CPU device is always last
  // The last device (or the only one) must be the host CPU fallback.
  bool has_cpu = false;
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].type() == DeviceType::kCpu) has_cpu = true;
  }
  EXPECT_TRUE(has_cpu);
}

TEST(ComputeDevice, EnumerateReportsCountBeyondBuffer) {
  // A null output is count-only even when a caller leaves a non-zero capacity.
  const std::size_t n = enumerate(nullptr, 0);
  EXPECT_GE(n, 1u);
  EXPECT_EQ(enumerate(nullptr, 8), n);
}

TEST(ComputeDevice, GlobalOrdinalsAndDefaultAgree) {
  Device devs[32];
  const std::size_t n = enumerate(devs, 32);
  ASSERT_GE(n, 1u);
  for (std::size_t i = 0; i < n && i < 32; ++i) {
    EXPECT_TRUE(devs[i].valid());
    EXPECT_EQ(devs[i].ordinal(), static_cast<int>(i));
  }

  const Device d = default_device();
  EXPECT_TRUE(d.valid());
  EXPECT_EQ(d.ordinal(), 0);
  EXPECT_EQ(d.type(), devs[0].type());
  EXPECT_STREQ(d.backend(), devs[0].backend());
}

TEST(ComputeDevice, DefaultDeviceValid) {
  Device d = default_device();
  const char* name = d.name();
  ASSERT_NE(name, nullptr);
  EXPECT_GT(std::string_view(name).size(), 0u);
  // caps: the completion contract (events) is honored by every backend, host too.
  EXPECT_TRUE(d.caps().events);
}

TEST(ComputeDevice, HostDeviceCaps) {
  // Reach the host device specifically (last in enumeration) and check its caps.
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  const Device* host = nullptr;
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].type() == DeviceType::kCpu) host = &devs[i];
  }
  ASSERT_NE(host, nullptr);
  EXPECT_STREQ(host->name(), "cpu");
  EXPECT_FALSE(host->caps().async);   // host has no real asynchrony
  EXPECT_TRUE(host->caps().events);   // but honors the event contract
  EXPECT_FALSE(host->is_gpu());
}

TEST(ComputeDevice, TypeNames) {
  EXPECT_STREQ(device_type_name(DeviceType::kCpu), "cpu");
  EXPECT_STREQ(device_type_name(DeviceType::kGpu), "gpu");
  EXPECT_STREQ(device_type_name(DeviceType::kIGpu), "igpu");
}

// Device::memory (compute.cc:121): free/total byte counts. The host backend reports
// (0,0) -- host RAM isn't meaningfully "device memory" (host_backend.cc:48). On a real
// GPU total>0 and free<=total. This just exercises the forwarding + host contract; a
// GPU asserts the sane inequality.
TEST(ComputeDevice, DeviceMemoryQuery) {
  bool is_gpu = false;
  Device d = find_gpu_or_host(&is_gpu);
  std::uint64_t free = 123, total = 456;
  d.memory(&free, &total);
  if (is_gpu) {
    EXPECT_GT(total, 0u);
    EXPECT_LE(free, total);
  } else {
    EXPECT_EQ(free, 0u);   // host: not meaningfully device memory
    EXPECT_EQ(total, 0u);
  }
}

// ---- stream / event lifecycle (works on host or GPU) ------------------------

TEST(ComputeStream, CreateAndSync) {
  Device d = default_device();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  EXPECT_TRUE(s->valid());
  EXPECT_EQ(s->device().type(), d.type());
  EXPECT_TRUE(s->sync().ok());
}

TEST(ComputeEvent, CreateRecordQuery) {
  Device d = default_device();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value()) << ev.error().message();

  // Record on an empty stream, then sync the stream: the event must be Ready.
  s->record(*ev);
  ASSERT_TRUE(s->sync().ok());
  EXPECT_EQ(ev->query(), EventStatus::kReady);
  ev->sync();  // no-op / returns immediately when already complete
}

TEST(ComputeStream, MoveSemantics) {
  Device d = default_device();
  auto s1 = Stream::create(d);
  ASSERT_TRUE(s1.has_value());
  Stream moved = std::move(*s1);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(s1->valid());  // moved-from is now invalid (null vt_)
  EXPECT_TRUE(moved.sync().ok());
}

// Move ASSIGNMENT (compute.cc:135): the destination already owns a stream, so operator=
// must destroy it before taking the source's -- no leak, no double-free. Only move
// CONSTRUCTION was covered above; assignment is a distinct code path.
TEST(ComputeStream, MoveAssignment) {
  Device d = default_device();
  auto s1 = Stream::create(d);
  auto s2 = Stream::create(d);
  ASSERT_TRUE(s1.has_value() && s2.has_value());
  *s2 = std::move(*s1);         // s2's original stream is destroyed, s1's moved in
  EXPECT_TRUE(s2->valid());
  EXPECT_FALSE(s1->valid());    // moved-from invalid
  EXPECT_TRUE(s2->sync().ok());
}

// Self-move-assignment must be a no-op (the `this != &o` guard, compute.cc:136): the
// stream stays valid and usable, its handle NOT freed. Done through a reference so the
// compiler can't see (and warn on) a literal `x = std::move(x)`.
TEST(ComputeStream, SelfMoveAssignment) {
  Device d = default_device();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  Stream& ref = *s;
  *s = std::move(ref);          // self-assign: guarded no-op
  EXPECT_TRUE(s->valid());      // still valid (handle not freed)
  EXPECT_TRUE(s->sync().ok());  // still usable
}

// ---- Event move semantics (was entirely uncovered) --------------------------
// Event owns a backend event handle (cudaEvent_t on GPU); move must transfer ownership
// so the moved-from won't double-destroy it. compute.cc:198 (ctor) / :204 (assign).

TEST(ComputeEvent, MoveConstruction) {
  Device d = default_device();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value()) << ev.error().message();
  Event moved = std::move(*ev);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(ev->valid());  // moved-from invalid (won't double-destroy)
}

TEST(ComputeEvent, MoveAssignment) {
  Device d = default_device();
  auto e1 = Event::create(d);
  auto e2 = Event::create(d);
  ASSERT_TRUE(e1.has_value() && e2.has_value());
  *e2 = std::move(*e1);       // e2's original event destroyed, e1's moved in
  EXPECT_TRUE(e2->valid());
  EXPECT_FALSE(e1->valid());
  // The moved-in event is usable: record it on a stream and observe completion.
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  s->record(*e2);
  ASSERT_TRUE(s->sync().ok());
  EXPECT_EQ(e2->query(), EventStatus::kReady);
}

TEST(ComputeEvent, SelfMoveAssignment) {
  Device d = default_device();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value());
  Event& ref = *ev;
  *ev = std::move(ref);       // guarded no-op (compute.cc:205)
  EXPECT_TRUE(ev->valid());   // handle not freed
}

// ---- GPU-only: event visibly polls NotReady -> Ready (the reap model) --------
// The async, stream-bound work that leaves an Event observably NotReady is a KERNEL
// LAUNCH (compute-04) -- compute-01's raw device_alloc/memcpy_async was removed (Buffer,
// compute-02, superseded it, but Buffer set/get is SYNCHRONOUS so it can't leave work in
// flight). We NVRTC-compile a deliberately-heavy spin kernel so it is still running when
// we first poll, exercising cudaEventQuery's NotReady->Ready transition. We do NOT assert
// we SEE NotReady (a fast GPU could finish first -- flaky); we assert the poll loop
// TERMINATES at kReady and sync() agrees.
TEST(ComputeEvent, GpuEventPollsToReady) {
  Device d;
  if (!find_cuda(&d)) {
    GTEST_SKIP() << "no CUDA GPU present -- host backend completes inline (always Ready)";
  }

  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value()) << ev.error().message();

  // A heavy spin kernel writing out[i]=i (the acc loop is masked to 0), still in flight
  // at the first poll below.
  constexpr const char* kSpin = R"(
extern "C" __global__ void spin(unsigned* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned acc = 0;
    for (int k = 0; k < iters; ++k) acc = acc * 1664525u + 1013904223u;
    out[i] = (unsigned)i + (acc & 0u);
  }
}
)";
  auto k = Kernel::compile(d, kSpin, "spin");
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 4096;
  auto out = mem::Buffer::create(device_memory_resource(d).value(),
                                 kN * sizeof(unsigned));
  ASSERT_TRUE(out.has_value()) << out.error().message();

  int n = kN, iters = 200000;
  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  ASSERT_TRUE(launch(*s, *k, cfg, *out, n, iters).ok());
  s->record(*ev);

  // Poll like a worker would: spin on query() until Ready. Bounded so a bug can't
  // hang the test forever.
  EventStatus st = EventStatus::kNotReady;
  for (int spin = 0; spin < 100'000'000 && st != EventStatus::kReady; ++spin) {
    st = ev->query();
  }
  EXPECT_EQ(st, EventStatus::kReady);
  EXPECT_TRUE(s->sync().ok());

  // The kernel actually ran: out[i] == i.
  std::vector<unsigned> host_out(kN, 0xFFFFFFFFu);
  ASSERT_TRUE(
      out->view().read(std::as_writable_bytes(std::span(host_out))).ok());
  for (int i = 0; i < kN; ++i) EXPECT_EQ(host_out[i], static_cast<unsigned>(i)) << "at " << i;
}

// ---- GPU timing (compute-06b #25): timestamp query --------------------------
// A timing event records a device timestamp on record(); elapsed_ns(start, stop) reports
// the GPU nanoseconds between two of them. On the host backend work runs inline so timing
// is a no-op that reads back 0 -- that path is exercised here on the host device so the
// contract (create_timing/record/sync/elapsed_ns all wired) is covered on ANY platform.

TEST(ComputeEvent, HostTimingInlineIsZero) {
  // Reach the host device specifically (it's always present, last in enumeration).
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  const Device* host = nullptr;
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].type() == DeviceType::kCpu) host = &devs[i];
  }
  ASSERT_NE(host, nullptr);

  auto s = Stream::create(*host);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto start = Event::create_timing(*host);
  auto stop = Event::create_timing(*host);
  ASSERT_TRUE(start.has_value()) << start.error().message();
  ASSERT_TRUE(stop.has_value()) << stop.error().message();

  s->record(*start);
  s->record(*stop);
  ASSERT_TRUE(s->sync().ok());
  // Timing events honor the same query/sync contract as plain events.
  EXPECT_EQ(start->query(), EventStatus::kReady);
  stop->sync();

  auto ns = elapsed_ns(*start, *stop);
  ASSERT_TRUE(ns.has_value()) << ns.error().message();
  EXPECT_EQ(*ns, 0u);  // host runs inline -> no device time
}

TEST(ComputeEvent, ElapsedRejectsPlainCompletionEvents) {
  Device devs[8];
  const std::size_t n = enumerate(devs, 8);
  const Device* host = nullptr;
  for (std::size_t i = 0; i < n && i < 8; ++i)
    if (devs[i].type() == DeviceType::kCpu) host = &devs[i];
  ASSERT_NE(host, nullptr);

  auto a = Event::create(*host);
  auto b = Event::create(*host);
  ASSERT_TRUE(a.has_value() && b.has_value());
  auto elapsed = elapsed_ns(*a, *b);
  EXPECT_FALSE(elapsed.has_value());
}

// GPU-only: bracket a heavy spin kernel with two timing events and read a NONZERO
// elapsed time. Proves the timestamp-query path (CUDA cudaEventElapsedTime / Vulkan
// VkQueryPool * timestampPeriod) actually measures device time. SKIPPED without a GPU.
TEST(ComputeEvent, GpuTimingMeasuresElapsed) {
  Device d;
  if (!find_cuda(&d)) {
    GTEST_SKIP() << "no CUDA GPU present -- Vulkan timing is covered by its SPIR-V test";
  }

  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto start = Event::create_timing(d);
  auto stop = Event::create_timing(d);
  ASSERT_TRUE(start.has_value()) << start.error().message();
  ASSERT_TRUE(stop.has_value()) << stop.error().message();

  constexpr const char* kSpin = R"(
extern "C" __global__ void spin(unsigned* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned acc = 0;
    for (int k = 0; k < iters; ++k) acc = acc * 1664525u + 1013904223u;
    out[i] = (unsigned)i + (acc & 0u);
  }
}
)";
  auto k = Kernel::compile(d, kSpin, "spin");
  ASSERT_TRUE(k.has_value()) << k.error().message();

  constexpr int kN = 4096;
  auto out = mem::Buffer::create(device_memory_resource(d).value(),
                                 kN * sizeof(unsigned));
  ASSERT_TRUE(out.has_value()) << out.error().message();

  int n = kN, iters = 200000;
  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};

  s->record(*start);
  ASSERT_TRUE(launch(*s, *k, cfg, *out, n, iters).ok());
  s->record(*stop);
  ASSERT_TRUE(s->sync().ok());

  auto ns = elapsed_ns(*start, *stop);
  ASSERT_TRUE(ns.has_value()) << ns.error().message();
  EXPECT_GT(*ns, 0u);  // a real kernel took nonzero GPU time
}
