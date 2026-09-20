#pragma once

// Included at the end of the locked ggml-cuda.cu, where its resource types are
// visible. This extends resource import only; all computation stays upstream.
#include "native_bridge.h"

namespace hyle::compute::ggml_cuda::bridge {

int device_index(int physical_device) {
  const auto &info = ggml_cuda_info();
  for (int index = 0; index < info.device_count; ++index)
    if (ggml_cuda_get_physical_device(index) == physical_device)
      return index;
  return -1;
}

cudaStream_t root_stream(ggml_backend *backend) {
  auto *context = static_cast<ggml_backend_cuda_context *>(backend->context);
  return context->stream(context->device, 0);
}

cudaError_t drain(ggml_backend *backend) noexcept {
  auto *context = static_cast<ggml_backend_cuda_context *>(backend->context);
  cudaError_t first = cudaSuccess;
  for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
    for (int index = 0; index < GGML_CUDA_MAX_STREAMS; ++index) {
      if (context->streams[device][index] == nullptr)
        continue;
      const auto status =
          cudaStreamSynchronize(context->streams[device][index]);
      if (first == cudaSuccess)
        first = status;
    }
  }
  return first;
}

ggml_backend_buffer *binding_buffer(int device, std::size_t capacity) {
  auto context =
      std::make_unique<ggml_backend_cuda_buffer_context>(device, nullptr);
  context->owns_payload = false;
  auto *buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(device),
                                          ggml_backend_cuda_buffer_interface,
                                          context.get(), capacity);
  context.release();
  return buffer;
}

void bind_buffer(ggml_backend_buffer *buffer, void *pointer) noexcept {
  auto *context =
      static_cast<ggml_backend_cuda_buffer_context *>(buffer->context);
  context->dev_ptr = pointer;
}

} // namespace hyle::compute::ggml_cuda::bridge
