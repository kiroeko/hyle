#include "cpu_execution.h"

#include <limits>
#include <utility>

// Private entry provided by the exact resource patch, never installed.
extern "C" ggml_threadpool *
ggml_hyle_threadpool_new(ggml_threadpool_params *params, int *native_error);

namespace hyle::compute::ggml_runtime {

Result<CpuExecution> CpuExecution::prepare(ggml_cgraph *graph, int threads,
                                           mem::AllocatorRef allocator,
                                           ggml_abort_callback abort,
                                           void *abort_data) noexcept {
  if (graph == nullptr || threads <= 0 || threads > GGML_MAX_N_THREADS ||
      !allocator.valid() || ggml_graph_n_nodes(graph) == 0)
    return std::unexpected(Status::InvalidArgument("ggml CPU execution"));

  ggml_cpu_init();
  auto params = ggml_threadpool_params_default(threads);
  params.poll = 0;
  params.prio = GGML_SCHED_PRIO_NORMAL;
  int native_error = 0;
  auto *pool = ggml_hyle_threadpool_new(&params, &native_error);
  if (pool == nullptr) {
    if (native_error == 0)
      return std::unexpected(Status::ResourceExhausted("ggml CPU pool"));
#if defined(_WIN32)
    return std::unexpected(status_from_win32(
        static_cast<std::uint32_t>(native_error), "ggml CPU pool"));
#else
    return std::unexpected(status_from_errno(native_error, "ggml CPU pool"));
#endif
  }
  CpuExecution execution;
  execution.allocator_ = allocator;
  execution.graph_ = graph;
  execution.plan_ = ggml_graph_plan(graph, threads, pool);
  execution.plan_.abort_callback = abort;
  execution.plan_.abort_callback_data = abort_data;
  if (execution.plan_.work_size >
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
    return std::unexpected(
        Status::ResourceExhausted("ggml CPU workspace size"));
  if (execution.plan_.work_size != 0) {
    execution.plan_.work_data = static_cast<std::uint8_t *>(
        allocator.allocate(execution.plan_.work_size, GGML_MEM_ALIGN));
    if (execution.plan_.work_data == nullptr)
      return std::unexpected(Status::ResourceExhausted("ggml CPU workspace"));
  }
  return execution;
}

Status CpuExecution::run() noexcept {
  if (graph_ == nullptr)
    return Status::FailedPrecondition("ggml CPU execution is empty");
  switch (ggml_graph_compute(graph_, &plan_)) {
  case GGML_STATUS_SUCCESS:
    return Status::Ok();
  case GGML_STATUS_ABORTED:
    return Status::Cancelled("ggml CPU execution");
  case GGML_STATUS_ALLOC_FAILED:
    return Status::ResourceExhausted("ggml CPU execution");
  case GGML_STATUS_FAILED:
    return Status::Internal("ggml CPU execution");
  }
  return Status::Internal("unknown ggml CPU status");
}

CpuExecution::CpuExecution(CpuExecution &&other) noexcept
    : allocator_(other.allocator_),
      graph_(std::exchange(other.graph_, nullptr)),
      plan_(std::exchange(other.plan_, ggml_cplan{})) {}

CpuExecution &CpuExecution::operator=(CpuExecution &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = other.allocator_;
    graph_ = std::exchange(other.graph_, nullptr);
    plan_ = std::exchange(other.plan_, ggml_cplan{});
  }
  return *this;
}

CpuExecution::~CpuExecution() { reset(); }

void CpuExecution::reset() noexcept {
  if (plan_.threadpool != nullptr)
    ggml_threadpool_free(plan_.threadpool);
  if (plan_.work_data != nullptr)
    allocator_.deallocate(plan_.work_data, plan_.work_size, GGML_MEM_ALIGN);
  plan_ = {};
  graph_ = nullptr;
}

} // namespace hyle::compute::ggml_runtime
