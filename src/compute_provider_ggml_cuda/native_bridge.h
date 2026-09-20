#pragma once

// Private C++17 boundary compiled with the pinned ggml CUDA backend. The C++23
// adapter translates these native failures; no vendor types enter the SDK.
#include <cstddef>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

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
cudaStream_t root_stream(ggml_backend *backend);
cudaError_t drain(ggml_backend *backend) noexcept;

// Non-owning wrapper, created cold and rebound only while the invocation is
// idle. Destroying it never frees caller tensor payload. Capacity is fixed.
ggml_backend_buffer *binding_buffer(int device_index, std::size_t capacity);
void bind_buffer(ggml_backend_buffer *buffer, void *pointer) noexcept;

} // namespace hyle::compute::ggml_cuda::bridge
