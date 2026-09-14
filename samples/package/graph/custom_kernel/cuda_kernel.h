#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#if defined(_WIN32)
#if defined(HYLE_CUSTOM_CUDA_KERNEL_BUILD)
#define HYLE_CUSTOM_CUDA_KERNEL_API __declspec(dllexport)
#else
#define HYLE_CUSTOM_CUDA_KERNEL_API __declspec(dllimport)
#endif
#else
#define HYLE_CUSTOM_CUDA_KERNEL_API __attribute__((visibility("default")))
#endif

// Native kernel library: no HYLE runtime or provider SPI lives in this image.
HYLE_CUSTOM_CUDA_KERNEL_API cudaError_t launch_affine(
    cudaStream_t stream, const float *input, float *output, std::size_t count,
    float scale, float bias, unsigned block_size) noexcept;
