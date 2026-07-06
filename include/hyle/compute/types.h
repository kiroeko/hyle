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

}  // namespace hyle::compute
