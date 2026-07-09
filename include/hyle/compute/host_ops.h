#pragma once

#include "hyle/compute/kernel.h"  // Kernel::HostFn, LaunchConfig, KernelArgs

// compute-07: CPU operator backend -- the real CPU implementation of compute-04's
// Kernel/Stream abstraction (the CPU twin of the CUDA-NVRTC and Vulkan-SPIR-V backends).
// These are library-provided `Kernel::HostFn`s that wrap hyle `simd/` numeric cores; a
// consumer registers one via `Kernel::load_host(cpu_device, host_saxpy_f32)` and launches it
// on a host Stream exactly like a GPU kernel (same LaunchConfig/KernelArgs ABI, same
// launch()/co_await path -- but completion is always-ready synchronous, host_event_query
// returns kReady). See docs/architecture/compute-design.md "compute-07 CPU 算子后端".
//
// TWO PURPOSES (README compute-07):
//   ① NO-GPU functional baseline: the compute-04 abstraction runs anywhere, no GPU needed.
//   ② GPU-vs-CPU parity ORACLE: feed the SAME inputs to a GPU kernel (CUDA vecadd / Vulkan
//      saxpy) and to the matching host kernel here, compare element-wise. These host kernels
//      deliberately mirror the GPU test kernels (vecadd, saxpy) so the oracle is exact.
//
// WHY simd, not ggml: this is NOT the AI-inference CPU path (that borrows ggml's CPU backend
// with its GEMM + quantization, compute-05). This is compute-04's "submit any kernel"
// abstraction given a CPU backend, whose kernels are hyle simd's numeric cores (dot/saxpy/
// elementwise). hyle simd has no matmul; GEMM stays with ggml. These two CPU paths are
// orthogonal (compute-design "compute-07 CPU 算子后端：接 hyle simd，不是借 ggml").
//
// NO SECOND RUNTIME: each host kernel runs single-threaded inline (scope 铁律). Data-parallel
// fan-out over many blocks is HYLE sched's fork-join (graph-02 parallel_for), never a thread
// pool spun inside the backend.
//
// KernelArgs UNPACKING CONTRACT (how each kernel reads its args -- mirrors the typed-shell
// packing order in kernel.h: Buffers first in order, then scalars packed into the blob):
//   host_saxpy_f32 : buffers[0]=x (in), buffers[1]=y (in/out); scalars: [0]=int n, [1]=float a.
//                    y[i] = a*x[i] + y[i].  (simd::saxpy_f32)
//   host_vecadd_f32: buffers[0]=a, buffers[1]=b, buffers[2]=c (out); scalars: [0]=int n.
//                    c[i] = a[i] + b[i].  (matches the CUDA vecadd test kernel for parity)
//   host_dot_f32   : buffers[0]=x, buffers[1]=y, buffers[2]=out (out[0] gets the scalar);
//                    scalars: [0]=int n.  out[0] = sum(x[i]*y[i]).  (simd::dot_f32 -- reduction)
//   host_exp_f32   : buffers[0]=src (in), buffers[1]=dst (out); scalars: [0]=int n.
//                    dst[i] = exp(src[i]).  (simd::exp_f32_batch -- elementwise transcendental)
//
// f64 siblings mirror the f32 ones (buffers are double*; scalar a is double for saxpy). They
// exercise the simd f64 path (compute-07's reason the f64 cores exist as a compute consumer).

namespace hyle::compute {

// f32 host kernels (register with Kernel::load_host).
HYLE_COMPUTE_API void host_saxpy_f32(const LaunchConfig&, const KernelArgs&) noexcept;
HYLE_COMPUTE_API void host_vecadd_f32(const LaunchConfig&, const KernelArgs&) noexcept;
HYLE_COMPUTE_API void host_dot_f32(const LaunchConfig&, const KernelArgs&) noexcept;
HYLE_COMPUTE_API void host_exp_f32(const LaunchConfig&, const KernelArgs&) noexcept;

// f64 host kernels (the CAE/scientific double path; exercises simd's f64 cores).
HYLE_COMPUTE_API void host_saxpy_f64(const LaunchConfig&, const KernelArgs&) noexcept;
HYLE_COMPUTE_API void host_dot_f64(const LaunchConfig&, const KernelArgs&) noexcept;
HYLE_COMPUTE_API void host_exp_f64(const LaunchConfig&, const KernelArgs&) noexcept;

}  // namespace hyle::compute
