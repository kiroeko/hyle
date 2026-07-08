#include "hyle/compute/device.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/compute/stream.h"

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

}  // namespace

// ---- device enumeration (host always present) -------------------------------

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
  // Passing max=0 still returns the true count (like vkEnumerate*).
  std::size_t n = enumerate(nullptr, 0);
  EXPECT_GE(n, 1u);
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

// ---- the completion proof: an async copy moves bytes, event observes done ----

TEST(ComputeStream, AsyncCopyRoundTrip) {
  bool is_gpu = false;
  Device d = find_gpu_or_host(&is_gpu);
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value()) << ev.error().message();

  constexpr std::size_t kN = 1024;
  std::vector<std::uint32_t> host_in(kN), host_out(kN, 0);
  for (std::size_t i = 0; i < kN; ++i) host_in[i] = static_cast<std::uint32_t>(i * 3 + 1);
  const std::size_t bytes = kN * sizeof(std::uint32_t);

  // Allocate device memory, copy H2D, then D2H back into host_out, record an event.
  auto dptr = s->device_alloc(bytes);
  ASSERT_TRUE(dptr.has_value()) << dptr.error().message();

  ASSERT_TRUE(s->memcpy_async(*dptr, host_in.data(), bytes).ok());
  ASSERT_TRUE(s->memcpy_async(host_out.data(), *dptr, bytes).ok());
  s->record(*ev);

  // Wait for completion (blocking here; compute-03 replaces this with co_await).
  ASSERT_TRUE(s->sync().ok());
  EXPECT_EQ(ev->query(), EventStatus::kReady);

  // The bytes must have round-tripped through device memory.
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(host_out[i], host_in[i]) << "mismatch at " << i;
  }
  s->device_free(*dptr);
}

// size=0 device_alloc: exercises the bytes==0?1:bytes guard (host_backend.cc:81). A
// zero-byte alloc still returns a valid, freeable pointer (a degenerate but legal
// request the completion-spine minimum must not choke on).
TEST(ComputeStream, ZeroSizeDeviceAlloc) {
  Device d = default_device();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  auto dptr = s->device_alloc(0);
  ASSERT_TRUE(dptr.has_value()) << dptr.error().message();
  EXPECT_NE(*dptr, nullptr);  // backend rounded 0 -> 1 byte
  s->device_free(*dptr);      // must be freeable without abort
}

// ---- GPU-only: event visibly polls NotReady -> Ready (the reap model) --------
TEST(ComputeEvent, GpuEventPollsToReady) {
  bool is_gpu = false;
  Device d = find_gpu_or_host(&is_gpu);
  if (!is_gpu) {
    GTEST_SKIP() << "no CUDA GPU present -- host backend completes inline (always Ready)";
  }

  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value()) << s.error().message();
  auto ev = Event::create(d);
  ASSERT_TRUE(ev.has_value()) << ev.error().message();

  // Enqueue enough async work that the event is plausibly NOT ready immediately,
  // then poll it to completion -- exercising cudaEventQuery's NotReady->Ready
  // transition (the compute-03 worker-reap primitive). We do NOT assert we SEE
  // NotReady (a fast GPU may finish before the first poll -- that would be flaky);
  // we assert the poll loop TERMINATES at kReady and sync() agrees.
  constexpr std::size_t kBig = 16 * 1024 * 1024;  // 16 MiB, a few copies
  auto dptr = s->device_alloc(kBig);
  ASSERT_TRUE(dptr.has_value()) << dptr.error().message();
  std::vector<std::uint8_t> host(kBig, 0xAB);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(s->memcpy_async(*dptr, host.data(), kBig).ok());
    ASSERT_TRUE(s->memcpy_async(host.data(), *dptr, kBig).ok());
  }
  s->record(*ev);

  // Poll like a worker would: spin on query() until Ready. Bounded so a bug can't
  // hang the test forever.
  EventStatus st = EventStatus::kNotReady;
  for (int spin = 0; spin < 100'000'000 && st != EventStatus::kReady; ++spin) {
    st = ev->query();
  }
  EXPECT_EQ(st, EventStatus::kReady);
  EXPECT_TRUE(s->sync().ok());
  s->device_free(*dptr);
}
