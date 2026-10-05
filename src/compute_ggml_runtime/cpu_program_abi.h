#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml-cpu.h"

namespace hyle::compute::ggml_runtime {

// Private, same-revision CPU program ABI. No tensor or pool crosses to a
// different ggml revision; allocation and retirement use the bound program.
struct CpuProgram {
  std::uint32_t abi;
  std::size_t descriptor_bytes;
  std::size_t tensor_bytes;
  std::size_t plan_bytes;
  void (*initialize)();
  ggml_threadpool *(*create_pool)(ggml_threadpool_params *, int *);
  ggml_cplan (*plan)(const ggml_cgraph *, int, ggml_threadpool *);
  ggml_status (*compute)(ggml_cgraph *, ggml_cplan *);
  void (*free_pool)(ggml_threadpool *);
};

inline constexpr std::uint32_t kCpuProgramAbi = 1;

} // namespace hyle::compute::ggml_runtime
