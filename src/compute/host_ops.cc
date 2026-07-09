#include "hyle/compute/host_ops.h"

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (unpack scalar field from the packed blob)

#include "hyle/simd/math.h"  // exp_f32_batch
#include "hyle/simd/ops.h"   // dot_f32/f64, saxpy_f32/f64

// compute-07: CPU operator backend -- library host kernels wrapping hyle simd numeric cores.
// Each is a Kernel::HostFn: it unpacks the unified KernelArgs (buffers[] + a std430-packed
// scalar blob with per-field {offset,size} descriptors) exactly the way the typed-shell in
// kernel.h packed them, then calls the matching simd core. Synchronous, single-threaded,
// no allocation (scope: no second runtime). See host_ops.h for each kernel's arg contract.

namespace hyle::compute {
namespace {

// Read scalar field `i` (by the typed-shell's field order) out of the packed blob.
template <class T>
T scalar_at(const KernelArgs& args, unsigned i) noexcept {
  const auto* base = static_cast<const std::byte*>(args.scalars);
  T v{};
  const auto off = args.fields[i].offset;
  std::memcpy(&v, base + off, sizeof(T));  // fields[i].size == sizeof(T) by contract
  return v;
}

}  // namespace

void host_saxpy_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const float* x = static_cast<const float*>(args.buffers[0]);
  float* y = static_cast<float*>(args.buffers[1]);
  const int n = scalar_at<int>(args, 0);
  const float a = scalar_at<float>(args, 1);
  simd::saxpy_f32(a, x, y, static_cast<std::size_t>(n));  // y[i] = a*x[i] + y[i]
}

void host_vecadd_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  // c = a + b. Mirrors the CUDA vecadd test kernel exactly (parity oracle). vecadd is
  // saxpy with alpha=1 on a copy: seed c=b, then c += 1*a via saxpy (one simd pass, no loop).
  const float* a = static_cast<const float*>(args.buffers[0]);
  const float* b = static_cast<const float*>(args.buffers[1]);
  float* c = static_cast<float*>(args.buffers[2]);
  const int n = scalar_at<int>(args, 0);
  const std::size_t sz = static_cast<std::size_t>(n);
  for (std::size_t i = 0; i < sz; ++i) c[i] = b[i];   // c <- b
  simd::saxpy_f32(1.0f, a, c, sz);                    // c <- 1*a + c = a + b
}

void host_dot_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const float* x = static_cast<const float*>(args.buffers[0]);
  const float* y = static_cast<const float*>(args.buffers[1]);
  float* out = static_cast<float*>(args.buffers[2]);
  const int n = scalar_at<int>(args, 0);
  out[0] = simd::dot_f32(x, y, static_cast<std::size_t>(n));  // reduction -> out[0]
}

void host_exp_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const float* src = static_cast<const float*>(args.buffers[0]);
  float* dst = static_cast<float*>(args.buffers[1]);
  const int n = scalar_at<int>(args, 0);
  simd::exp_f32_batch(src, dst, static_cast<std::size_t>(n));
}

void host_saxpy_f64(const LaunchConfig&, const KernelArgs& args) noexcept {
  const double* x = static_cast<const double*>(args.buffers[0]);
  double* y = static_cast<double*>(args.buffers[1]);
  const int n = scalar_at<int>(args, 0);
  const double a = scalar_at<double>(args, 1);
  simd::saxpy_f64(a, x, y, static_cast<std::size_t>(n));
}

void host_dot_f64(const LaunchConfig&, const KernelArgs& args) noexcept {
  const double* x = static_cast<const double*>(args.buffers[0]);
  const double* y = static_cast<const double*>(args.buffers[1]);
  double* out = static_cast<double*>(args.buffers[2]);
  const int n = scalar_at<int>(args, 0);
  out[0] = simd::dot_f64(x, y, static_cast<std::size_t>(n));
}

}  // namespace hyle::compute
