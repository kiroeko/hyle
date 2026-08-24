#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/types.h"
#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-01: device enumeration + capability query -- the thinnest layer of the
// heterogeneous compute pillar (compute-design "模块形状" device.h). A Device is a
// handle to a compute engine (a CUDA GPU, or the host CPU fallback); it answers
// "what am I, how much memory, what can I do" and mints Streams (stream.h).
//
// BORROW-THE-SHAPE, NOT THE TYPE (compute-design §2 张力收口): the DeviceType /
// DeviceCaps mirror ggml-backend's dev_type + dev_caps -- the field set a real
// backend registry converged on -- but they are HYLE's OWN structs. We do NOT
// include any ggml header or expose cudaDevice_t; the backend lives behind the
// enumeration functions (device.cc, #if HYLE_COMPUTE_CUDA vs host fallback).
//
// SCOPE (compute-01, the io-01 analogue): device/stream/event abstraction + event
// POLLING primitive + two backends. It does NOT wire GPU completion into the sched
// worker loop (co_await stream) -- that is compute-03, an independent architectural
// problem (the set_io_poll seam is single-slot, held by io/). See the milestone doc.

namespace hyle::compute {

// DeviceType / DeviceCaps live in types.h (shared with the internal backend vtable
// without dragging in std::expected). Stable name of a DeviceType ("cpu"/"gpu"/
// "igpu"); never localized.
HYLE_COMPUTE_API const char* device_type_name(DeviceType t) noexcept;

// A handle to one compute engine. Non-owning value type (trivially copyable): the
// backend state (a CUDA device ordinal, or nothing for host) is small and lives
// behind `ordinal_`/`type_`; there is no per-Device resource to release (Streams
// and Events own theirs). A default-constructed Device is an invalid output
// placeholder; get a valid value from enumerate()/default_device() before using
// any observer other than valid()/ordinal() or creating resources from it.
class Device {
 public:
  Device() noexcept = default;

  bool valid() const noexcept { return ordinal_ >= 0; }
  // Requires valid().
  DeviceType type() const noexcept { return type_; }
  // Opaque device index minted by enumerate()/default_device(). Identifies the
  // physical engine; the dispatch layer maps it to a backend + backend-local
  // ordinal. Treat it as a handle, not an arithmetic value.
  int ordinal() const noexcept { return ordinal_; }
  // Requires valid().
  bool is_gpu() const noexcept { return type_ == DeviceType::kGpu || type_ == DeviceType::kIGpu; }

  // The remaining observers require valid(). Human-readable device name
  // ("NVIDIA GeForce RTX 4070" / "cpu"). Points at
  // backend-owned storage stable for the process; never null.
  HYLE_COMPUTE_API const char* name() const noexcept;
  // Backend that owns this device: "cuda" / "vulkan" / "host". Lets a consumer pick a
  // SPECIFIC backend when several expose the same physical GPU (e.g. CUDA and Vulkan both
  // enumerate an NVIDIA card). Process-stable pointer; never null.
  HYLE_COMPUTE_API const char* backend() const noexcept;
  // Capability flags for this device (queried from the backend).
  HYLE_COMPUTE_API DeviceCaps caps() const noexcept;
  // Free / total device memory in bytes (cudaMemGetInfo on GPU; host RAM on CPU).
  // Either pointer may be null to skip it. Best-effort: zeros if the backend can't
  // report (never fails -- this is an informational query, not an operation).
  HYLE_COMPUTE_API void memory(std::uint64_t* free, std::uint64_t* total) const noexcept;

 private:
  friend HYLE_COMPUTE_API std::size_t enumerate(Device* out, std::size_t max) noexcept;
  friend HYLE_COMPUTE_API Device default_device() noexcept;
  Device(DeviceType type, int ordinal) noexcept : type_(type), ordinal_(ordinal) {}

  DeviceType type_ = DeviceType::kCpu;
  int ordinal_ = -1;
};

// Fill `out` (a caller array of `max`) with the available devices, returning how
// many exist (which may exceed `max` -- call again with a bigger array, like
// vkEnumerate*). Passing null is a count-only query and ignores `max`. The host
// CPU device is ALWAYS present (index last), so the result
// is never zero; real GPUs (CUDA) come first when the backend is compiled in and a
// GPU is found at runtime. The allocation-free process table retains at most 32
// entries and reserves its final slot for that host fallback. Buffer-fill, no
// allocation (same discipline as platform fs iterate).
HYLE_COMPUTE_API std::size_t enumerate(Device* out, std::size_t max) noexcept;

// The best default device: the first GPU if one exists, else the host CPU. A
// caller that just wants "somewhere to run" uses this without enumerating.
HYLE_COMPUTE_API Device default_device() noexcept;

}  // namespace hyle::compute
