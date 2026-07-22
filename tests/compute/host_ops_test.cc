#include "hyle/compute/host_ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/stream.h"

// compute-07 test: the CPU operator backend -- simd-backed host kernels driven through the
// SAME compute-04 Kernel/Stream/launch abstraction as the GPU backends. Proves (a) the
// abstraction runs on CPU with no GPU (functional baseline) and (b) each kernel's result
// matches a scalar reference (the GPU-parity oracle: host_vecadd here == the CUDA vecadd
// test kernel; host_saxpy == the Vulkan saxpy test kernel). Host completion is always-ready
// so no co_await is needed (launch is synchronous inline on the host backend).

using namespace hyle;
using namespace hyle::compute;

namespace {

// Pick a CPU/host device (the host backend is always present). enumerate() lists all;
// default_device() is the host fallback when no GPU. compute-07 targets that CPU backend.
Device host_device() {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i)
    if (!devs[i].is_gpu()) return devs[i];
  return default_device();  // default is the host fallback when no GPU enumerated
}

// Build a naturally-aligned scalar blob the way kernel.h's typed shell does; fields carry
// {offset,size}. Shader-facing callers remain responsible for matching their declared layout.
struct Blob {
  alignas(16) std::byte data[64];
  ScalarField fields[4];
  unsigned nfields = 0;
  unsigned off = 0;
  template <class T>
  void push(T v) {
    const unsigned sz = sizeof(T);
    const unsigned al = alignof(T);
    off = static_cast<unsigned>((off + al - 1) / al * al);
    std::memcpy(data + off, &v, sz);
    fields[nfields++] = ScalarField{off, sz};
    off += sz;
  }
};

}  // namespace

// ---- saxpy f32: y = a*x + y (== the Vulkan saxpy test kernel, parity oracle) ----
TEST(ComputeHostOps, SaxpyF32) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_saxpy_f32);
  ASSERT_TRUE(k.has_value()) << k.error().message();
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 1000;  // not a multiple of any SIMD width -> exercises simd tail
  std::vector<float> x(kN), y(kN), ref(kN);
  for (int i = 0; i < kN; ++i) { x[i] = float(i) * 0.5f; y[i] = float(i) + 1.0f; }
  const float a = 2.5f;
  for (int i = 0; i < kN; ++i) ref[i] = a * x[i] + y[i];

  void* bufs[] = {x.data(), y.data()};
  Blob blob; blob.push<int>(kN); blob.push<float>(a);  // scalars: n (int), a (float)
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(y[i], ref[i]) << "i=" << i;
}

// ---- vecadd f32: c = a + b (== the CUDA vecadd test kernel, parity oracle) ----
TEST(ComputeHostOps, VecAddF32) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_vecadd_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 4096;  // same size the CUDA NvrtcVecAddEndToEnd uses
  std::vector<float> a(kN), b(kN), c(kN, -1.0f);
  for (int i = 0; i < kN; ++i) { a[i] = float(i); b[i] = float(2 * i); }
  void* bufs[] = {a.data(), b.data(), c.data()};
  Blob blob; blob.push<int>(kN);
  KernelArgs ka{bufs, 3, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(c[i], float(i) + float(2 * i)) << "i=" << i;
}

// ---- dot f32: out[0] = sum(x*y) (reduction kernel) ----
TEST(ComputeHostOps, DotF32) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_dot_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 777;
  std::vector<float> x(kN), y(kN), out(1, -1.0f);
  double ref = 0.0;
  for (int i = 0; i < kN; ++i) { x[i] = float(i % 7) * 0.25f; y[i] = float(i % 5) - 1.0f; ref += double(x[i]) * y[i]; }
  void* bufs[] = {x.data(), y.data(), out.data()};
  Blob blob; blob.push<int>(kN);
  KernelArgs ka{bufs, 3, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  EXPECT_NEAR(out[0], float(ref), std::abs(float(ref)) * 1e-4f + 1e-4f);
}

// ---- exp f32: dst = exp(src) (elementwise transcendental via simd) ----
TEST(ComputeHostOps, ExpF32) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_exp_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 513;
  std::vector<float> src(kN), dst(kN, -1.0f);
  for (int i = 0; i < kN; ++i) src[i] = float(i - 256) * 0.05f;  // [-12.8, 12.8]
  void* bufs[] = {src.data(), dst.data()};
  Blob blob; blob.push<int>(kN);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i)
    EXPECT_NEAR(dst[i], std::exp(src[i]), std::abs(std::exp(src[i])) * 1e-5f + 1e-6f) << "i=" << i;
}

// ---- f64 path (CAE/scientific double; exercises simd f64 cores) ----
TEST(ComputeHostOps, SaxpyF64) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_saxpy_f64);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 1000;
  std::vector<double> x(kN), y(kN), ref(kN);
  for (int i = 0; i < kN; ++i) { x[i] = double(i) * 0.5; y[i] = double(i) + 1.0; }
  const double a = 2.5;
  for (int i = 0; i < kN; ++i) ref[i] = a * x[i] + y[i];
  void* bufs[] = {x.data(), y.data()};
  Blob blob; blob.push<int>(kN); blob.push<double>(a);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i) EXPECT_DOUBLE_EQ(y[i], ref[i]) << "i=" << i;
}

TEST(ComputeHostOps, DotF64) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_dot_f64);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 777;
  std::vector<double> x(kN), y(kN), out(1, -1.0);
  long double ref = 0.0L;
  for (int i = 0; i < kN; ++i) { x[i] = double(i % 7) * 0.25; y[i] = double(i % 5) - 1.0; ref += (long double)x[i] * y[i]; }
  void* bufs[] = {x.data(), y.data(), out.data()};
  Blob blob; blob.push<int>(kN);
  KernelArgs ka{bufs, 3, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  EXPECT_NEAR(out[0], double(ref), std::abs(double(ref)) * 1e-12 + 1e-12);
}

TEST(ComputeHostOps, ExpF64) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_exp_f64);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 513;
  std::vector<double> src(kN), dst(kN, -1.0);
  for (int i = 0; i < kN; ++i) src[i] = double(i - 256) * 0.05;
  void* bufs[] = {src.data(), dst.data()};
  Blob blob; blob.push<int>(kN);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  ASSERT_TRUE(launch(*s, *k, cfg, ka).ok());
  for (int i = 0; i < kN; ++i)
    EXPECT_NEAR(dst[i], std::exp(src[i]), std::abs(std::exp(src[i])) * 1e-13 + 1e-14) << "i=" << i;
}

// ---- typed-shell drive (the REAL launch ABI): pack via launch(s,k,cfg, Buffer..., scalar...) ----
// The other tests hand-build KernelArgs; this one goes through kernel.h's typed shell exactly
// like a GPU consumer, proving the shell's Buffer/scalar packing order matches host_saxpy_f32's
// unpacking. Host Buffers are RAM (Buffer::create on the host device = malloc), so the same call
// that targets a GPU works here.
TEST(ComputeHostOps, TypedShellSaxpyF32) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_saxpy_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());

  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  auto bx = Buffer::create(device_buffer_type(d), bytes);
  auto by = Buffer::create(device_buffer_type(d), bytes);
  ASSERT_TRUE(bx.has_value() && by.has_value());
  std::vector<float> hx(kN), hy(kN), ref(kN);
  const float a = 3.0f;
  for (int i = 0; i < kN; ++i) { hx[i] = float(i); hy[i] = float(i) * 0.5f; ref[i] = a * hx[i] + hy[i]; }
  ASSERT_TRUE(bx->set(0, hx.data(), bytes).ok());
  ASSERT_TRUE(by->set(0, hy.data(), bytes).ok());

  int n = kN;
  LaunchConfig cfg;
  // Typed shell: Buffers first (x,y), then scalars (n, a) -- the SAME order host_saxpy_f32 unpacks.
  ASSERT_TRUE(launch(*s, *k, cfg, *bx, *by, n, a).ok());
  ASSERT_TRUE(s->sync().ok());
  std::vector<float> got(kN);
  ASSERT_TRUE(by->get(0, got.data(), bytes).ok());
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(got[i], ref[i]) << "i=" << i;
}

// ---- contract death tests (caller-bug fail-fast, matching the GPU launch path) ----
TEST(ComputeHostOpsDeath, NullBufferAsserts) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_saxpy_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  void* bufs[] = {nullptr, nullptr};  // null buffers -> contract violation
  int n = 8; float a = 1.0f;
  Blob blob; blob.push<int>(n); blob.push<float>(a);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  EXPECT_DEATH(launch(*s, *k, cfg, ka), "");
}

TEST(ComputeHostOpsDeath, NegativeNAsserts) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_saxpy_f32);
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  std::vector<float> x(8), y(8);
  void* bufs[] = {x.data(), y.data()};
  int n = -1; float a = 1.0f;  // negative n -> would scan ~2^63 elements without the assert
  Blob blob; blob.push<int>(n); blob.push<float>(a);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  EXPECT_DEATH(launch(*s, *k, cfg, ka), "");
}

TEST(ComputeHostOpsDeath, InsufficientBufferCountAsserts) {
  Device d = host_device();
  auto k = Kernel::load_host(d, &host_dot_f32);  // needs 3 buffers
  ASSERT_TRUE(k.has_value());
  auto s = Stream::create(d);
  ASSERT_TRUE(s.has_value());
  std::vector<float> x(8), y(8);
  void* bufs[] = {x.data(), y.data()};  // only 2, dot needs 3 (x,y,out)
  int n = 8;
  Blob blob; blob.push<int>(n);
  KernelArgs ka{bufs, 2, blob.data, blob.off, blob.fields, blob.nfields};
  LaunchConfig cfg;
  EXPECT_DEATH(launch(*s, *k, cfg, ka), "");
}
