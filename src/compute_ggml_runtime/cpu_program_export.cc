#include "cpu_program_abi.h"
#include "hyle/visibility.h"

extern "C" ggml_threadpool *
ggml_hyle_threadpool_new(ggml_threadpool_params *, int *);

extern "C" HYLE_EXPORT const hyle::compute::ggml_runtime::CpuProgram *
hyle_ggml_cpu_program_v1() noexcept {
  using namespace hyle::compute::ggml_runtime;
  static const CpuProgram program{
      kCpuProgramAbi, sizeof(CpuProgram), sizeof(ggml_tensor), sizeof(ggml_cplan),
      &ggml_cpu_init, &ggml_hyle_threadpool_new, &ggml_graph_plan,
      &ggml_graph_compute, &ggml_threadpool_free};
  return &program;
}
