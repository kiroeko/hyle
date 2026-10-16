#include <hyle/compute/provider.h>
#if HYLE_G08_CUDA
#include <hyle/compute/providers/cuda.h>
#else
#include <hyle/compute/providers/ggml_cpu.h>
#endif

namespace g08 {
hyle::compute::Provider reference_provider() noexcept {
#if HYLE_G08_CUDA
  return hyle::compute::cuda_native::cccl_provider();
#else
  return hyle::compute::ggml_cpu::provider();
#endif
}
} // namespace g08
