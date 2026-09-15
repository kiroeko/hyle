#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#if defined(_WIN32)
#if defined(Q4_PREFLIGHT_KERNEL_BUILD)
#define Q4_PREFLIGHT_API __declspec(dllexport)
#else
#define Q4_PREFLIGHT_API __declspec(dllimport)
#endif
#else
#define Q4_PREFLIGHT_API __attribute__((visibility("default")))
#endif

namespace q4_preflight {

// ggml Q4_0 encoding, not NVIDIA FP4: 32 values, one IEEE FP16 scale,
// low nibbles for coordinates 0..15, high nibbles for coordinates 16..31.
struct Block final {
  std::uint16_t scale;
  std::uint8_t quants[16];
};
static_assert(sizeof(Block) == 18 && alignof(Block) == 2);

// A[M,K] * transpose(W[N,K]) -> Y[M,N]. All buffers are caller-owned.
// Research carrier only: no performance qualification or public HYLE format ID.
Q4_PREFLIGHT_API cudaError_t matmul(cudaStream_t stream, const float *input,
                                    const Block *weights, float *output,
                                    int rows, int columns, int inner) noexcept;

} // namespace q4_preflight
