#pragma once

#include <cstdint>

// compute-01 shared POD types (device kind + capability flags + event status).
// Split into their OWN header -- with NO dependency on error/result.h -- so the
// internal CUDA backend TU (cuda_backend.cu) can include backend.h without pulling
// std::expected in. nvcc 13.3 IGNORES -std=c++23 when forwarding to the host
// compiler (verified: "flag not supported with the configured host compiler"), so
// std::expected (a C++23 library type) is unavailable in a .cu TU. The CUDA
// backend's vtable slots only ever return Status (never Result), so keeping these
// enums free of result.h lets the .cu compile at CUDA's C++20 dialect while the
// host C++ TUs stay at C++23. device.h/stream.h include this; they add result.h
// themselves for their Result<> return types (host-only surface).

namespace hyle::compute {

// What kind of compute engine a Device is. Mirrors ggml_backend_dev_type: a GPU
// with dedicated VRAM vs an integrated GPU on host memory vs the CPU fallback --
// the distinction drives whether a buffer transfer is a real DMA copy (compute-02)
// or a no-op on coherent memory. kAccel (BLAS/AMX-style) and finer types are added
// when a backend needs them; compute-01 ships CPU + GPU.
enum class DeviceType : std::uint8_t {
  kCpu = 0,   // host CPU fallback (system memory; always available)
  kGpu = 1,   // discrete GPU with dedicated device memory (CUDA here)
  kIGpu = 2,  // integrated GPU on host memory (unified -- no host<->device copy)
};

// What a device supports -- mirrors ggml_backend_dev_caps. A consumer queries this
// to decide a code path (async submit vs sync fallback, pinned staging buffers,
// event-based completion) WITHOUT knowing the concrete backend. The host fallback
// reports {async=false, events=true} -- it has no real asynchrony but honors the
// event completion contract synchronously so the same code drives both.
struct DeviceCaps {
  bool async = false;        // stream ops run asynchronously (real GPU)
  bool host_buffer = false;  // can allocate pinned host staging memory (compute-02)
  bool events = false;       // event record/query/sync is supported
};

// The result of polling an Event: has the work it marks finished?
enum class EventStatus : std::uint8_t {
  kReady = 0,     // all stream work before the record() has completed
  kNotReady = 1,  // still running -- poll again later
};

// Which kind of memory a BufferType allocates (compute-02). Mirrors ggml's model
// where host/device/pinned are DISTINCT buffer_type instances (not a switch): one
// memory model = one BufferType. A Device exposes one of each it supports.
//   kDevice     : device-local memory (VRAM on a GPU; cudaMalloc). NOT host-
//                 dereferenceable. is_host()=false. On the host backend this is
//                 plain RAM (there is no separate "device").
//   kHostPinned : page-locked host RAM (cudaMallocHost) for fast DMA / async-copy
//                 overlap -- the staging tier for H2D/D2H. is_host()=true. On a
//                 GPU-less backend (or CUDA disabled) it falls back to plain host
//                 RAM (same as ggml's GGML_CUDA_NO_PINNED fallback).
enum class BufferKind : std::uint8_t {
  kDevice = 0,
  kHostPinned = 1,
};

// compute-04 kernel launch geometry (POD, kept here -- not in kernel.h -- so the CUDA
// backend TU can cast the vtable's config arg without including kernel.h, which pulls
// std::expected〔result.h〕that nvcc's C++ front-end can't compile, same reason as
// EventStatus). 3D grid/block extents + dynamic shared memory.
struct Dim3 {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;
};
struct LaunchConfig {
  Dim3 grid;
  Dim3 block;
  unsigned shared_mem = 0;  // dynamic shared memory bytes
};

// compute-04/06 UNIFIED kernel arguments (POD, here not kernel.h so the .cu TU can use
// it without result.h). This is the backend-neutral launch ABI: it lowers to BOTH CUDA
// cuLaunchKernel `void** args` AND Vulkan descriptor-set + push-constants. Its shape is
// established by REAL measurement (a saxpy kernel with 3 buffers + 4 mixed scalars run on
// both models -- see compute-06 spec "统一 launch ABI 真数据") + validated against a
// real renderer's Mesh(indices, ...vertexArrays) ctor (same variadic-fold + byte-pack +
// descriptor-array trick). A kernel's arguments are exactly two groups:
//   • BUFFERS  -- an ORDERED list of device pointers. Position i = CUDA arg-slot i =
//                 Vulkan descriptor binding i. Both models agree on "ordered buffer list".
//   • SCALARS  -- packed into ONE contiguous byte blob, PLUS a small descriptor array
//                 {offset,size} per scalar field. This resolves the CUDA-vs-Vulkan split
//                 the real data exposed: Vulkan pushes the WHOLE blob as push constants
//                 (ignores the descriptors); CUDA needs ONE void** entry PER scalar, which
//                 it rebuilds as (blob_base + descriptor[i].offset). The descriptors are
//                 filled at compile time by the variadic `launch` shell (it knows each
//                 scalar's offset/size), so neither backend needs a runtime type tag.
// Zero heap: buffers / scalar blob / descriptors all point at caller-owned STACK storage
// the typed `launch` shell builds -- same zero-alloc discipline as the old `void** args`.
// The backend reads them on the submit path (cold vs the kernel's GPU time), never copies.
//
// A buffers-only kernel: scalar_bytes=0, scalar_count=0. A scalars-only kernel:
// buffer_count=0. Grid/workgroup stays in LaunchConfig.grid (Vulkan group count == CUDA
// grid dim, same ceil(n/block) math).
struct ScalarField {
  unsigned offset;  // byte offset of this scalar within the blob
  unsigned size;    // byte size of this scalar
};
struct KernelArgs {
  void* const* buffers;         // ordered array of buffer device pointers (buffer_count)
  unsigned buffer_count;        // number of buffer arguments
  const void* scalars;          // packed scalar byte blob (push-constant data), or nullptr
  unsigned scalar_bytes;        // total size of the scalar blob, or 0
  const ScalarField* fields;    // per-scalar {offset,size} (scalar_count long), or nullptr
  unsigned scalar_count;        // number of scalar fields (for CUDA void** rebuild)
};

}  // namespace hyle::compute
