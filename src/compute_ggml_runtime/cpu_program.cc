#include "cpu_program.h"
#include "hyle/core/environment.h"

#if defined(HYLE_GGML_CPU_AVX2_PROGRAM) && HYLE_OS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(HYLE_GGML_CPU_AVX2_PROGRAM)
#include <dlfcn.h>
#endif

extern "C" ggml_threadpool *
ggml_hyle_threadpool_new(ggml_threadpool_params *, int *);

namespace hyle::compute::ggml_runtime {
namespace {
const CpuProgram kBaseline{
    kCpuProgramAbi, sizeof(CpuProgram), sizeof(ggml_tensor), sizeof(ggml_cplan),
    &ggml_cpu_init, &ggml_hyle_threadpool_new, &ggml_graph_plan,
    &ggml_graph_compute, &ggml_threadpool_free};

#if defined(HYLE_GGML_CPU_AVX2_PROGRAM)
Result<const CpuProgram *> load_avx2_program() noexcept {
  using Entry = const CpuProgram *(*)() noexcept;
#if HYLE_OS_WINDOWS
  const auto library = LoadLibraryExA(HYLE_GGML_CPU_PROGRAM_LIBRARY, nullptr,
                                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (library == nullptr)
    return std::unexpected(status_from_win32(GetLastError(),
                                             "ggml CPU AVX2 program"));
  const auto entry = reinterpret_cast<Entry>(
      GetProcAddress(library, "hyle_ggml_cpu_program_v1"));
#else
  const auto library =
      dlopen(HYLE_GGML_CPU_PROGRAM_LIBRARY, RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    return std::unexpected(Status::NotFound("ggml CPU AVX2 program"));
  const auto entry =
      reinterpret_cast<Entry>(dlsym(library, "hyle_ggml_cpu_program_v1"));
#endif
  const auto *program = entry == nullptr ? nullptr : entry();
  if (program == nullptr || program->abi != kCpuProgramAbi ||
      program->descriptor_bytes != sizeof(CpuProgram) ||
      program->tensor_bytes != sizeof(ggml_tensor) ||
      program->plan_bytes != sizeof(ggml_cplan) || !program->initialize ||
      !program->create_pool || !program->plan || !program->compute ||
      !program->free_pool) {
#if HYLE_OS_WINDOWS
    FreeLibrary(library);
#else
    dlclose(library);
#endif
    return std::unexpected(Status::FailedPrecondition("ggml CPU program ABI"));
  }
  // One module per process, retained for the process lifetime like a linked
  // runtime. Execution slots retire their own pools/workspace independently.
  return program;
}
#endif
} // namespace

Result<const CpuProgram *> select_cpu_program() noexcept {
#if defined(HYLE_GGML_CPU_AVX2_PROGRAM)
  if (cpu_avx2_program_eligible(platform::cpu_features())) {
    static const auto optimized = load_avx2_program();
    return optimized;
  }
#endif
  return &kBaseline;
}

} // namespace hyle::compute::ggml_runtime
