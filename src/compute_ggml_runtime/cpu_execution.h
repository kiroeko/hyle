#pragma once

#include "metadata_arena.h"

#include "ggml-cpu.h"

namespace hyle::compute::ggml_runtime {

struct CpuProgram;

// One synchronous invocation slot. Graph metadata, tensor payloads, callback
// data and allocator are borrowed and must outlive this object. Preparation
// validates a graph built by a trusted, capacity-checked HYLE lowering; this is
// not an API for arbitrary native graphs. No concurrent run/close on one slot.
// Distinct slots can run concurrently with shared immutable tensor payloads.
class CpuExecution final {
public:
  CpuExecution() noexcept = default;
  CpuExecution(const CpuExecution &) = delete;
  CpuExecution &operator=(const CpuExecution &) = delete;
  HYLE_GGML_RUNTIME_INTERNAL_API CpuExecution(CpuExecution &&other) noexcept;
  HYLE_GGML_RUNTIME_INTERNAL_API CpuExecution &
  operator=(CpuExecution &&other) noexcept;
  HYLE_GGML_RUNTIME_INTERNAL_API ~CpuExecution();

  HYLE_GGML_RUNTIME_INTERNAL_API static Result<CpuExecution>
  prepare(ggml_cgraph *graph, int threads, mem::AllocatorRef allocator,
          ggml_abort_callback abort = nullptr,
          void *abort_data = nullptr) noexcept;
  // Returns only after all workers have completed, including on cancellation.
  HYLE_GGML_RUNTIME_INTERNAL_API Status run() noexcept;
  std::size_t workspace_bytes() const noexcept { return plan_.work_size; }

private:
  void reset() noexcept;
  mem::AllocatorRef allocator_;
  const CpuProgram *program_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_cplan plan_{};
};

} // namespace hyle::compute::ggml_runtime
