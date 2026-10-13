#include "cpu_program_abi.h"
#include <cstddef>

namespace {
ggml_threadpool *create_pool(ggml_threadpool_params *parameters, int *error) {
  *error = 0;
  return ggml_threadpool_new(parameters);
}
const hyle::compute::ggml_runtime::CpuProgram original{
    hyle::compute::ggml_runtime::kCpuProgramAbi,
    sizeof(hyle::compute::ggml_runtime::CpuProgram), sizeof(ggml_tensor),
    sizeof(ggml_cplan), &ggml_cpu_init, &create_pool, &ggml_graph_plan,
    &ggml_graph_compute, &ggml_threadpool_free};
}

// Benchmark-only ABI handshake; all math, planning and pool retirement are
// compiled and linked from the pristine original source, never HYLE patches.
extern "C" __declspec(dllexport)
const hyle::compute::ggml_runtime::CpuProgram *
g01_cpu_paged_original_program() noexcept {
  return &original;
}
extern "C" __declspec(dllexport) std::size_t
g01_cpu_paged_original_tensor_src_offset() noexcept {
  return offsetof(ggml_tensor, src);
}
extern "C" __declspec(dllexport) int
g01_cpu_paged_original_bf16_type() noexcept {
  return GGML_TYPE_BF16;
}
