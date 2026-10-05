#include "cpu_program_abi.h"

extern "C" ggml_threadpool *
ggml_hyle_threadpool_new(ggml_threadpool_params *, int *);

#if defined(_WIN32)
#define HYLE_CPU_PROGRAM_EXPORT __declspec(dllexport)
#else
#define HYLE_CPU_PROGRAM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" HYLE_CPU_PROGRAM_EXPORT const hyle::compute::ggml_runtime::CpuProgram *
hyle_ggml_cpu_program_v1() noexcept {
  using namespace hyle::compute::ggml_runtime;
  static const CpuProgram program{
      kCpuProgramAbi, sizeof(CpuProgram), sizeof(ggml_tensor), sizeof(ggml_cplan),
      &ggml_cpu_init, &ggml_hyle_threadpool_new, &ggml_graph_plan,
      &ggml_graph_compute, &ggml_threadpool_free};
  return &program;
}
