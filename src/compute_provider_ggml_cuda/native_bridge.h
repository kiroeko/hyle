#pragma once

// Private C++17 boundary compiled with the pinned ggml CUDA backend. The C++23
// adapter translates these native failures; no vendor types enter the SDK.
#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ggml.h>

struct ggml_backend;
struct ggml_backend_buffer;

namespace hyle::compute::ggml_cuda::bridge {

template <class Code> struct NativeFailure final {
  Code code;
  const char *operation;
};

template <class Code> [[noreturn]] void fail(Code code, const char *operation) {
  throw NativeFailure<Code>{code, operation};
}

// Maps the HYLE physical device to ggml's possibly virtualized device table.
// A negative result means the physical device is absent from that table.
int device_index(int physical_device);
// Cold geometry only: cached BF16/MMVF caps and the row-limited paired domain.
bool supports_bfloat_paired_projection(int device, std::uint64_t inner,
                                      std::uint64_t rows, std::uint64_t columns);
// Select the original caching pools. Native streams remain lazy: ordinary
// submission borrows the admitted stream; standalone packing creates its own.
void initialize_backend(ggml_backend *backend);
cudaError_t drain(ggml_backend *backend) noexcept;
cudaError_t query(ggml_backend *backend) noexcept;

// Cold-only dependency reporting, shared with the original native fusers.
// Reports actual storage roots and their last potential fused output's node
// ordinal in the original graph. No payload is read, graph reordered,
// stream created or optimizer enabled.
void collect_allocation_dependencies(
    ggml_backend *backend, ggml_cgraph *graph,
    void (*add)(void *user, ggml_tensor *root, int until_node), void *user);

// Exact original native graph qualification. No payload is read or submitted.
bool record_compatible(ggml_backend *backend, ggml_cgraph *graph);

// Protect the native cuBLAS destruction exclusion for the whole outer capture,
// not just the encoder call. The original preparation Completion releases it
// only after observing that caller capture ended or was aborted.
void retain_recording_guard();
void release_recording_guard() noexcept;

// Encode into the caller's already active capture, without native cache
// warmup, Begin/EndCapture, instantiation or GraphLaunch. Native resources stay
// with the original backend. A failed record cannot drain until the caller
// ends/aborts capture; retirement_error then reports cudaErrorNotReady so the
// original admitted Completion retains this backend and every binding.
ggml_status record_on_admitted_stream(ggml_backend *backend,
                                     ggml_cgraph *graph, cudaStream_t stream,
                                     cudaError_t *retirement_error);

// The caller owns admission and has rejected an already-captured stream.
// Borrow only for this original graph submission; every exit restores the
// backend's original (possibly null) stream table and current index. Restore
// errors propagate through the native failure boundary. retirement_error is
// required and initialized to success. A failed submission/restore reports the
// all-stream drain result there without replacing its original error/status.
ggml_status compute_on_admitted_stream(ggml_backend *backend,
                                      ggml_cgraph *graph, cudaStream_t stream,
                                      cudaError_t *retirement_error);

// Non-owning wrapper, created cold and rebound only while the invocation is
// idle. Destroying it never frees caller tensor payload. Capacity is fixed.
ggml_backend_buffer *binding_buffer(int device_index, std::size_t capacity);
void bind_buffer(ggml_backend_buffer *buffer, void *pointer) noexcept;

} // namespace hyle::compute::ggml_cuda::bridge
