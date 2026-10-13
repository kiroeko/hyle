#pragma once
#include "cpu_program_abi.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"
#include <cstddef>
#include <cstdio>
#include <filesystem>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace hyle::bench::resident {
inline Result<const compute::ggml_runtime::CpuProgram *> original_cpu_program() noexcept {
  using Program = compute::ggml_runtime::CpuProgram;
  using Entry = const Program *(*)() noexcept;
  static const auto loaded = []() -> Result<const Program *> {
    auto path=std::filesystem::path(HYLE_CPU_PAGED_ORIGINAL_DLL).make_preferred();
    const auto library = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library) {
      const auto error=GetLastError();
      std::fprintf(stderr,"# CPU_PAGED_ORIGINAL_LOAD path=%s error=%lu base_preloaded=%d cpu_preloaded=%d\n",HYLE_CPU_PAGED_ORIGINAL_DLL,error,GetModuleHandleA("ggml-base.dll")!=nullptr,GetModuleHandleA("ggml-cpu.dll")!=nullptr);
      return std::unexpected(status_from_win32(error, "original CPU program"));
    }
    const auto entry = reinterpret_cast<Entry>(GetProcAddress(library, "g01_cpu_paged_original_program"));
    const auto offset = reinterpret_cast<std::size_t(*)() noexcept>(GetProcAddress(library, "g01_cpu_paged_original_tensor_src_offset"));
    const auto type = reinterpret_cast<int(*)() noexcept>(GetProcAddress(library, "g01_cpu_paged_original_bf16_type"));
    const auto *program = entry ? entry() : nullptr;
    if (!program || program->abi != compute::ggml_runtime::kCpuProgramAbi ||
        program->descriptor_bytes != sizeof(Program) || program->tensor_bytes != sizeof(ggml_tensor) ||
        program->plan_bytes != sizeof(ggml_cplan) || !offset || offset() != offsetof(ggml_tensor, src) ||
        !type || type() != GGML_TYPE_BF16 || !program->initialize || !program->create_pool ||
        !program->plan || !program->compute || !program->free_pool) {
      FreeLibrary(library);
      return std::unexpected(Status::FailedPrecondition("original CPU ABI"));
    }
    HMODULE math_module=nullptr;
    wchar_t math_path[2048]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(program->compute),&math_module) ||
        !GetModuleFileNameW(math_module,math_path,2048))
      return std::unexpected(Status::FailedPrecondition("original CPU math image missing"));
    auto expected=path.parent_path()/"ggml-cpu.dll";
    if(_wcsicmp(math_path,expected.c_str())!=0)
      return std::unexpected(Status::FailedPrecondition("original CPU math image is foreign"));
    std::fprintf(stderr,"# CPU_PAGED_ORIGINAL_ABI tensor_bytes=%zu plan_bytes=%zu math_image=pristine-ggml-cpu.dll\n",program->tensor_bytes,program->plan_bytes);
    // Process-lifetime code image; each slot destroys its own original pool.
    return program;
  }();
  return loaded;
}

struct CpuPagedOriginalExecution final {
  const compute::ggml_runtime::CpuProgram *program = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_cplan plan{};
  mem::AllocatorRef allocator;
  CpuPagedOriginalExecution() = default;
  CpuPagedOriginalExecution(const CpuPagedOriginalExecution &) = delete;
  CpuPagedOriginalExecution &operator=(const CpuPagedOriginalExecution &) = delete;
  ~CpuPagedOriginalExecution() {
    if (plan.threadpool) program->free_pool(plan.threadpool);
    if (plan.work_data) allocator.deallocate(plan.work_data, plan.work_size, GGML_MEM_ALIGN);
  }
  Status prepare(ggml_cgraph *g, mem::AllocatorRef a) noexcept {
    HYLE_ASSIGN_OR_RETURN(program, original_cpu_program());
    graph = g; allocator = a; program->initialize();
    auto parameters = ggml_threadpool_params_default(1);
    parameters.poll = 0; parameters.prio = GGML_SCHED_PRIO_NORMAL;
    int error = 0;
    auto *pool = program->create_pool(&parameters, &error);
    if (!pool) return Status::ResourceExhausted("original CPU threadpool");
    plan = program->plan(graph, 1, pool);
    if (plan.work_size) {
      plan.work_data = static_cast<std::uint8_t *>(a.allocate(plan.work_size, GGML_MEM_ALIGN));
      if (!plan.work_data) return Status::ResourceExhausted("original CPU workspace");
    }
    return Status::Ok();
  }
  Status run() noexcept {
    return program->compute(graph, &plan) == GGML_STATUS_SUCCESS ? Status::Ok()
        : Status::Internal("original CPU math failed");
  }
};
}
