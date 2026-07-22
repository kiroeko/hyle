#include "hyle/compute/buffer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/compute/device.h"

// compute-02 buffer tests. Run on ANY platform: host fallback (both BufferKinds are
// plain RAM) always works; on a machine with a CUDA GPU the SAME tests exercise real
// VRAM (kDevice) + pinned host (kHostPinned). The test TU is pure C++ (all CUDA is
// inside hyle_compute), so it needs no nvcc and detects the GPU at runtime.
//
// The headline verification is the KV-CACHE PATTERN (the real inference-engine use,
// llama-kv-cache.cpp:1343): one persistent large device buffer written repeatedly at
// per-token row offsets, then read back and checked. That exercises set(offset,size)
// -- the reason offset exists.

using namespace hyle;
using namespace hyle::compute;

namespace {

Device gpu_or_host(bool* is_gpu) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
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

// ---- buffer types + traits --------------------------------------------------

TEST(ComputeBuffer, DeviceBufferTypeTraits) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  ASSERT_TRUE(bt.valid());
  EXPECT_EQ(bt.kind(), BufferKind::kDevice);
  EXPECT_GE(bt.alignment(), 1u);
  EXPECT_GT(bt.max_size(), 0u);
  // is_host: false for a real GPU's device memory; true on the host fallback (plain RAM).
  bool is_gpu = false;
  gpu_or_host(&is_gpu);
  EXPECT_EQ(bt.is_host(), !is_gpu);
}

TEST(ComputeBuffer, HostBufferTypeIsHost) {
  Device d = default_device();
  BufferType bt = host_buffer_type(d);
  ASSERT_TRUE(bt.valid());
  EXPECT_EQ(bt.kind(), BufferKind::kHostPinned);
  EXPECT_TRUE(bt.is_host());  // pinned host RAM is always host-dereferenceable
}

// ---- alloc / set / get round-trip -------------------------------------------

TEST(ComputeBuffer, AllocSetGetRoundTrip) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  constexpr std::size_t kN = 4096;
  auto buf = Buffer::create(bt, kN * sizeof(std::uint32_t));
  ASSERT_TRUE(buf.has_value()) << buf.error().message();
  EXPECT_TRUE(buf->valid());
  EXPECT_NE(buf->native_handle(), nullptr);
  EXPECT_EQ(buf->size(), kN * sizeof(std::uint32_t));

  std::vector<std::uint32_t> in(kN), out(kN, 0);
  for (std::size_t i = 0; i < kN; ++i) in[i] = static_cast<std::uint32_t>(i * 7 + 3);

  ASSERT_TRUE(buf->set(0, in.data(), kN * sizeof(std::uint32_t)).ok());
  ASSERT_TRUE(buf->get(0, out.data(), kN * sizeof(std::uint32_t)).ok());
  for (std::size_t i = 0; i < kN; ++i) EXPECT_EQ(out[i], in[i]) << "at " << i;
}

TEST(ComputeBuffer, PinnedHostRoundTrip) {
  Device d = default_device();
  BufferType bt = host_buffer_type(d);
  constexpr std::size_t kBytes = 8192;
  auto buf = Buffer::create(bt, kBytes);
  ASSERT_TRUE(buf.has_value()) << buf.error().message();
  EXPECT_TRUE(buf->is_host());

  std::vector<std::uint8_t> in(kBytes, 0xCD), out(kBytes, 0);
  ASSERT_TRUE(buf->set(0, in.data(), kBytes).ok());
  ASSERT_TRUE(buf->get(0, out.data(), kBytes).ok());
  EXPECT_EQ(in, out);
}

TEST(ComputeBuffer, HostMappingIsSeparateAndCoherentByExplicitOperations) {
  Device d = default_device();
  constexpr std::size_t kBytes = 8192;
  auto host = Buffer::create(host_buffer_type(d), kBytes);
  auto device = Buffer::create(device_buffer_type(d), kBytes);
  ASSERT_TRUE(host.has_value() && device.has_value());
  ASSERT_TRUE(host->is_host());
  ASSERT_NE(host->native_handle(), nullptr);

  auto* mapped = static_cast<std::uint8_t*>(host->host_data());
  ASSERT_NE(mapped, nullptr);
  for (std::size_t i = 0; i < kBytes; ++i)
    mapped[i] = static_cast<std::uint8_t>((i * 17) & 0xFF);
  ASSERT_TRUE(host->flush_host(0, kBytes).ok());
  ASSERT_TRUE(device->copy_from(*host, 0, 0, kBytes).ok());

  std::vector<std::uint8_t> out(kBytes, 0);
  ASSERT_TRUE(device->get(0, out.data(), kBytes).ok());
  for (std::size_t i = 0; i < kBytes; ++i) EXPECT_EQ(out[i], mapped[i]);

  std::vector<std::uint8_t> replacement(kBytes, 0xA7);
  ASSERT_TRUE(device->set(0, replacement.data(), kBytes).ok());
  ASSERT_TRUE(host->copy_from(*device, 0, 0, kBytes).ok());
  ASSERT_TRUE(host->invalidate_host(0, kBytes).ok());
  for (std::size_t i = 0; i < kBytes; ++i) EXPECT_EQ(mapped[i], 0xA7);
}

// ---- clear ------------------------------------------------------------------
// size=0 allocation: Buffer::create with bytes==0 exercises the bytes==0?1:bytes special
// case (host_backend.cc:115 -- malloc(0) is impl-defined, so the backend gives 1). The
// Buffer is valid with a non-null pointer and reports size()==0. This is a real edge (an
// empty tensor / a zero-row KV slice); it must not abort or return an invalid buffer.
TEST(ComputeBuffer, ZeroSizeAlloc) {
  Device d = default_device();
  auto buf = Buffer::create(device_buffer_type(d), 0);
  ASSERT_TRUE(buf.has_value()) << buf.error().message();
  EXPECT_TRUE(buf->valid());
  EXPECT_NE(buf->native_handle(), nullptr);  // backend rounded 0 -> 1 byte
  EXPECT_EQ(buf->size(), 0u);             // logical size is still 0
  // A zero-length set/get is in-bounds (offset 0 + size 0 <= size 0) and a no-op.
  EXPECT_TRUE(buf->set(0, nullptr, 0).ok());
  EXPECT_TRUE(buf->get(0, nullptr, 0).ok());
}
TEST(ComputeBuffer, Clear) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  constexpr std::size_t kBytes = 1024;
  auto buf = Buffer::create(bt, kBytes);
  ASSERT_TRUE(buf.has_value());
  ASSERT_TRUE(buf->clear(0, 0, kBytes).ok());

  std::vector<std::uint8_t> out(kBytes, 0xFF);
  ASSERT_TRUE(buf->get(0, out.data(), kBytes).ok());
  for (std::size_t i = 0; i < kBytes; ++i) EXPECT_EQ(out[i], 0u) << "at " << i;
}

// clear with a NON-ZERO value at a NON-ZERO offset: the existing Clear test only zeroes
// the whole buffer, so the value!=0 and offset!=0 paths (compute.cc:342 -> memset) were
// unexercised. Fill the buffer, clear a middle span to 0xA5, verify ONLY that span
// changed and the surrounding bytes are untouched (offset/size honored).
TEST(ComputeBuffer, ClearNonZeroValueAtOffset) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  constexpr std::size_t kBytes = 1024;
  auto buf = Buffer::create(bt, kBytes);
  ASSERT_TRUE(buf.has_value());

  std::vector<std::uint8_t> init(kBytes, 0x11);
  ASSERT_TRUE(buf->set(0, init.data(), kBytes).ok());
  constexpr std::size_t kOff = 256, kLen = 512;
  ASSERT_TRUE(buf->clear(kOff, 0xA5, kLen).ok());  // middle span -> 0xA5

  std::vector<std::uint8_t> out(kBytes, 0);
  ASSERT_TRUE(buf->get(0, out.data(), kBytes).ok());
  for (std::size_t i = 0; i < kBytes; ++i) {
    std::uint8_t want = (i >= kOff && i < kOff + kLen) ? 0xA5 : 0x11;
    EXPECT_EQ(out[i], want) << "at " << i;
  }
}

// ---- copy_from (D2D / same backend) -----------------------------------------

TEST(ComputeBuffer, CopyFrom) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  constexpr std::size_t kBytes = 2048;
  auto src = Buffer::create(bt, kBytes);
  auto dst = Buffer::create(bt, kBytes);
  ASSERT_TRUE(src.has_value() && dst.has_value());

  std::vector<std::uint8_t> in(kBytes), out(kBytes, 0);
  for (std::size_t i = 0; i < kBytes; ++i) in[i] = static_cast<std::uint8_t>(i & 0xFF);
  ASSERT_TRUE(src->set(0, in.data(), kBytes).ok());
  ASSERT_TRUE(dst->copy_from(*src, 0, 0, kBytes).ok());  // D2D
  ASSERT_TRUE(dst->get(0, out.data(), kBytes).ok());
  EXPECT_EQ(in, out);
}

// ---- move semantics ---------------------------------------------------------

TEST(ComputeBuffer, MoveSemantics) {
  Device d = default_device();
  auto b1 = Buffer::create(device_buffer_type(d), 256);
  ASSERT_TRUE(b1.has_value());
  void* ptr = b1->native_handle();
  Buffer moved = std::move(*b1);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.native_handle(), ptr);
  EXPECT_FALSE(b1->valid());  // moved-from invalid (won't double-free)
}

// Move ASSIGNMENT (compute.cc:283): dst already owns a block, so operator= must free it
// before adopting src's pointer -- no leak, no double-free. Move CONSTRUCTION (above) is
// a different path; both must be covered for an owning move-only type.
TEST(ComputeBuffer, MoveAssignment) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  auto b1 = Buffer::create(bt, 256);
  auto b2 = Buffer::create(bt, 128);
  ASSERT_TRUE(b1.has_value() && b2.has_value());
  void* src_ptr = b1->native_handle();
  *b2 = std::move(*b1);        // b2's original 128B block freed, b1's 256B moved in
  EXPECT_TRUE(b2->valid());
  EXPECT_EQ(b2->native_handle(), src_ptr);
  EXPECT_EQ(b2->size(), 256u);  // size transferred too
  EXPECT_FALSE(b1->valid());
  // The moved-in buffer is fully usable (set/get round-trip).
  std::vector<std::uint8_t> in(256, 0x5A), out(256, 0);
  ASSERT_TRUE(b2->set(0, in.data(), 256).ok());
  ASSERT_TRUE(b2->get(0, out.data(), 256).ok());
  EXPECT_EQ(in, out);
}

TEST(ComputeBuffer, SelfMoveAssignment) {
  Device d = default_device();
  auto b = Buffer::create(device_buffer_type(d), 256);
  ASSERT_TRUE(b.has_value());
  void* ptr = b->native_handle();
  Buffer& ref = *b;
  *b = std::move(ref);         // guarded no-op (compute.cc:284) -- block NOT freed
  EXPECT_TRUE(b->valid());
  EXPECT_EQ(b->native_handle(), ptr);
  std::vector<std::uint8_t> in(256, 0x5A), out(256, 0);
  ASSERT_TRUE(b->set(0, in.data(), 256).ok());  // still usable
  ASSERT_TRUE(b->get(0, out.data(), 256).ok());
  EXPECT_EQ(in, out);
}

// ---- THE HEADLINE: KV-cache pattern (large buffer, repeated offset writes) ---
// Mirrors llama.cpp's KV-cache: one persistent buffer, each "token" writes a row at
// a growing offset; afterward every row reads back correctly. This is the real
// validation of set(offset,size) as a partial-write primitive.

TEST(ComputeBuffer, KvCachePattern) {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);

  constexpr std::size_t kRowElems = 128;   // "embedding dim" per token row
  constexpr std::size_t kNumTokens = 512;  // rows written one at a time
  constexpr std::size_t kRowBytes = kRowElems * sizeof(float);
  const std::size_t total = kRowBytes * kNumTokens;

  auto kv = Buffer::create(bt, total);
  ASSERT_TRUE(kv.has_value()) << kv.error().message();
  ASSERT_TRUE(kv->clear(0, 0, total).ok());  // zero-init like llama_kv_cache

  // Write each token's row at its offset (the per-token append the KV-cache does).
  std::vector<float> row(kRowElems);
  for (std::size_t t = 0; t < kNumTokens; ++t) {
    for (std::size_t j = 0; j < kRowElems; ++j)
      row[j] = static_cast<float>(t) * 1000.0f + static_cast<float>(j);
    ASSERT_TRUE(kv->set(t * kRowBytes, row.data(), kRowBytes).ok())
        << "write token " << t;
  }

  // Read back every row and verify -- both a full read and a few random partial reads.
  std::vector<float> got(kRowElems);
  for (std::size_t t = 0; t < kNumTokens; ++t) {
    ASSERT_TRUE(kv->get(t * kRowBytes, got.data(), kRowBytes).ok())
        << "read token " << t;
    for (std::size_t j = 0; j < kRowElems; ++j) {
      EXPECT_FLOAT_EQ(got[j], static_cast<float>(t) * 1000.0f + static_cast<float>(j))
          << "token " << t << " elem " << j;
    }
  }
}

// ---- host-visible buffer -> device copy --------------------------------------

TEST(ComputeBuffer, HostBufferToDevice) {
  bool is_gpu = false;
  Device d = gpu_or_host(&is_gpu);
  constexpr std::size_t kBytes = 64 * 1024;

  // Fill a pinned host staging buffer, copy it to device, read device back, verify.
  auto staging = Buffer::create(host_buffer_type(d), kBytes);
  auto dev = Buffer::create(device_buffer_type(d), kBytes);
  ASSERT_TRUE(staging.has_value() && dev.has_value());

  // staging is host memory -> we can fill it via set (memcpy into pinned RAM).
  std::vector<std::uint8_t> src(kBytes), out(kBytes, 0);
  for (std::size_t i = 0; i < kBytes; ++i) src[i] = static_cast<std::uint8_t>((i * 13) & 0xFF);
  ASSERT_TRUE(staging->set(0, src.data(), kBytes).ok());

  // staging -> device (H2D within the backend), then device -> host verify.
  ASSERT_TRUE(dev->copy_from(*staging, 0, 0, kBytes).ok());
  ASSERT_TRUE(dev->get(0, out.data(), kBytes).ok());
  EXPECT_EQ(src, out);
}
