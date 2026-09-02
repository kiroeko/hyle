#include "hyle/compute/host_ops.h"

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (unpack scalar field from the packed blob)

#include "hyle/core/compiler.h"     // HYLE_FATAL_ASSERT (contract fail-fast, like the GPU launch path)
#include "hyle/simd/math.h"  // exp_f32_batch / exp_f64_batch
#include "hyle/simd/ops.h"   // dot/saxpy/add f32/f64

// compute-07: CPU operator backend -- library host kernels wrapping hyle simd numeric cores.
// Each is a Kernel::HostFn: it unpacks the unified KernelArgs (buffers[] + a naturally-aligned
// scalar blob with per-field {offset,size} descriptors) exactly the way the typed-shell in
// kernel.h packed them, then calls the matching simd core. Synchronous, single-threaded,
// no allocation (scope: no second runtime). See host_ops.h for each kernel's arg contract.
//
// CONTRACT (caller bug -> fatal assert, matching the GPU launch path's HYLE_FATAL_ASSERT
// discipline, e.g. vulkan_backend buffer_count check): the KernelArgs shape is a CALLER
// contract (the typed shell guarantees it), NOT external input -- a violation is a bug,
// caught fail-fast, never silently UB. Each kernel asserts its required buffer/scalar count,
// non-null pointers, and n >= 0 (a negative n cast to size_t would scan ~2^63 elements =
// out-of-bounds memory walk in the simd core).

namespace hyle::compute {
namespace {

// Read scalar field `i` (by the typed-shell's field order) out of the packed blob. Asserts the
// field's packed size matches T (a type mismatch between pack and unpack is a caller bug).
template <class T>
T scalar_at(const KernelArgs& args, unsigned i) noexcept {
  HYLE_FATAL_ASSERT(args.scalar_count > i && args.fields != nullptr && args.scalars != nullptr);
  HYLE_FATAL_ASSERT(args.fields[i].size == sizeof(T));  // pack/unpack type must agree
  const auto* base = static_cast<const std::byte*>(args.scalars);
  T v{};
  std::memcpy(&v, base + args.fields[i].offset, sizeof(T));
  return v;
}

// Common buffer/count precondition for a kernel taking `nb` buffers (all non-null) + `n` >= 0.
void require(const KernelArgs& args, unsigned nb, int n) noexcept {
  HYLE_FATAL_ASSERT(args.buffer_count >= nb && args.buffers != nullptr);
  for (unsigned i = 0; i < nb; ++i) {
    HYLE_FATAL_ASSERT(args.buffers[i].native_handle != nullptr &&
                      args.buffers[i].size != 0);
  }
  HYLE_FATAL_ASSERT(n >= 0);
}

template <class T>
T* buffer_at(const KernelArgs& args, unsigned index,
             std::size_t elements) noexcept {
  const BufferBinding binding = args.buffers[index];
  HYLE_FATAL_ASSERT(elements <= binding.size / sizeof(T));
  auto* bytes = static_cast<std::byte*>(binding.native_handle) + binding.offset;
  HYLE_FATAL_ASSERT(reinterpret_cast<std::uintptr_t>(bytes) % alignof(T) == 0);
  return reinterpret_cast<T*>(bytes);
}

}  // namespace

void host_saxpy_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  const float a = scalar_at<float>(args, 1);
  require(args, 2, n);
  const float* x = buffer_at<float>(args, 0, static_cast<std::size_t>(n));
  float* y = buffer_at<float>(args, 1, static_cast<std::size_t>(n));
  simd::saxpy(a, x, y, static_cast<std::size_t>(n));  // y[i] = a*x[i] + y[i]
}

void host_vecadd_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  // c = a + b. Mirrors the CUDA vecadd test kernel (parity oracle). One simd pass via add_f32
  // (the elementwise-add primitive; no scalar copy + no double-write of the earlier saxpy hack).
  const int n = scalar_at<int>(args, 0);
  require(args, 3, n);
  const float* a = buffer_at<float>(args, 0, static_cast<std::size_t>(n));
  const float* b = buffer_at<float>(args, 1, static_cast<std::size_t>(n));
  float* c = buffer_at<float>(args, 2, static_cast<std::size_t>(n));
  simd::add(a, b, c, static_cast<std::size_t>(n));
}

void host_dot_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  require(args, 3, n);
  const float* x = buffer_at<float>(args, 0, static_cast<std::size_t>(n));
  const float* y = buffer_at<float>(args, 1, static_cast<std::size_t>(n));
  float* out = buffer_at<float>(args, 2, 1);
  out[0] = simd::dot(x, y, static_cast<std::size_t>(n));  // reduction -> out[0]
}

void host_exp_f32(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  require(args, 2, n);
  const float* src = buffer_at<float>(args, 0, static_cast<std::size_t>(n));
  float* dst = buffer_at<float>(args, 1, static_cast<std::size_t>(n));
  simd::exp_f32_batch(src, dst, static_cast<std::size_t>(n));
}

void host_saxpy_f64(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  const double a = scalar_at<double>(args, 1);
  require(args, 2, n);
  const double* x = buffer_at<double>(args, 0, static_cast<std::size_t>(n));
  double* y = buffer_at<double>(args, 1, static_cast<std::size_t>(n));
  simd::saxpy(a, x, y, static_cast<std::size_t>(n));
}

void host_dot_f64(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  require(args, 3, n);
  const double* x = buffer_at<double>(args, 0, static_cast<std::size_t>(n));
  const double* y = buffer_at<double>(args, 1, static_cast<std::size_t>(n));
  double* out = buffer_at<double>(args, 2, 1);
  out[0] = simd::dot(x, y, static_cast<std::size_t>(n));
}

void host_exp_f64(const LaunchConfig&, const KernelArgs& args) noexcept {
  const int n = scalar_at<int>(args, 0);
  require(args, 2, n);
  const double* src = buffer_at<double>(args, 0, static_cast<std::size_t>(n));
  double* dst = buffer_at<double>(args, 1, static_cast<std::size_t>(n));
  simd::exp_f64_batch(src, dst, static_cast<std::size_t>(n));
}

}  // namespace hyle::compute
