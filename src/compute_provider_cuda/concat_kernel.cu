#include "concat_kernel.h"

#include <cuda_runtime.h>

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kThreads = 256;
template <class T, unsigned Inputs, bool Dense>
__global__ void concat(ConcatPointers inputs, T *output, ConcatLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.elements)
    return;
  const unsigned group_chunk = layout.chunks[Inputs - 1];
  const unsigned row = index / group_chunk;
  const unsigned within = index % group_chunk;
  unsigned selected = 0;
#pragma unroll
  for (unsigned input = 0; input + 1 < Inputs; ++input)
    if (within >= layout.chunks[input])
      ++selected;
  const unsigned previous = selected == 0 ? 0 : layout.chunks[selected - 1];
  const unsigned source_chunk = layout.chunks[selected] - previous;
  const unsigned source_index = row * source_chunk + within - previous;
  T value;
  if constexpr (Dense)
    value = static_cast<const T *>(inputs.input[selected])[source_index];
  else {
    std::size_t offset = 0;
    unsigned remaining = source_index;
    for (unsigned axis = layout.rank; axis-- > 0;) {
      const unsigned extent = axis == layout.axis ? source_chunk / layout.inner
                                                  : layout.dimensions[axis];
      offset += (remaining % extent) * layout.strides[selected][axis];
      remaining /= extent;
    }
    const auto *source =
        static_cast<const unsigned char *>(inputs.input[selected]) + offset;
    auto *bytes = reinterpret_cast<unsigned char *>(&value);
#pragma unroll
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      bytes[byte] = source[byte];
  }
  output[static_cast<std::size_t>(row) * layout.output_row +
         layout.output_offset + within] = value;
}
template <class T, bool Dense> const void *bind(unsigned inputs) noexcept {
  switch (inputs) {
  case 1:
    return reinterpret_cast<const void *>(concat<T, 1, Dense>);
  case 2:
    return reinterpret_cast<const void *>(concat<T, 2, Dense>);
  case 3:
    return reinterpret_cast<const void *>(concat<T, 3, Dense>);
  case 4:
    return reinterpret_cast<const void *>(concat<T, 4, Dense>);
  }
  return nullptr;
}
template <class T> const void *bind(bool dense, unsigned inputs) noexcept {
  return dense ? bind<T, true>(inputs) : bind<T, false>(inputs);
}
} // namespace
cudaError_t concat_prepare(unsigned scalar_bytes, bool dense,
                           const ConcatLayout &layout,
                           ConcatProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  const void *kernel = nullptr;
  switch (scalar_bytes) {
  case 1:
    kernel = bind<unsigned char>(dense, layout.inputs);
    break;
  case 2:
    kernel = bind<unsigned short>(dense, layout.inputs);
    break;
  case 4:
    kernel = bind<unsigned>(dense, layout.inputs);
    break;
  case 8:
    kernel = bind<unsigned long long>(dense, layout.inputs);
    break;
  }
  if (kernel == nullptr)
    return cudaErrorInvalidValue;
  if (layout.elements != 0) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess)
      return status;
    if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
      return cudaErrorNotSupported;
  }
  *output = {kernel, layout};
  return cudaSuccess;
}
cudaError_t concat_launch(const ConcatProgram &program,
                          const ConcatPointers &inputs, void *output,
                          cudaStream_t stream) noexcept {
  if (program.layout.elements == 0)
    return cudaSuccess;
  auto layout = program.layout;
  auto pointers = inputs;
  void *arguments[]{&pointers, &output, &layout};
  return cudaLaunchKernel(program.kernel,
                          dim3((layout.elements + kThreads - 1) / kThreads),
                          dim3(kThreads), arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
