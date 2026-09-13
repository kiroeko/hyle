#include <cuda_runtime_api.h>

namespace {

__global__ void illegal_address_kernel() {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *reinterpret_cast<volatile unsigned int *>(1) = 0x51a7b00fU;
  }
}

} // namespace

cudaError_t launch_illegal_address_kernel(cudaStream_t stream) noexcept {
  illegal_address_kernel<<<1, 1, 0, stream>>>();
  return cudaPeekAtLastError();
}
