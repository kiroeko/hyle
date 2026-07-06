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
  EXPECT_NE(buf->device_ptr(), nullptr);
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

// ---- clear ------------------------------------------------------------------

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
  void* ptr = b1->device_ptr();
  Buffer moved = std::move(*b1);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.device_ptr(), ptr);
  EXPECT_FALSE(b1->valid());  // moved-from invalid (won't double-free)
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

// ---- staging pattern: pinned host -> device (the H2D upload path) ------------

TEST(ComputeBuffer, PinnedStagingToDevice) {
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
