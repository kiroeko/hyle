#include "hyle/compute/host_ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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

// Build a std430-packed scalar blob the way kernel.h's typed shell does: each scalar aligned
// to its own size (<=16), fields carry {offset,size}. Small helper for the tests below.
struct Blob {
  alignas(16) std::byte data[64];
  ScalarField fields[4];
  unsigned nfields = 0;
  unsigned off = 0;
  template <class T>
  void push(T v) {
    const unsigned sz = sizeof(T);
    const unsigned al = sz < 16 ? sz : 16;
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
