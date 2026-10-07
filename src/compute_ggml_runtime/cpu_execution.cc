#include "cpu_execution.h"
#include "hyle/core/environment.h"
#include "cpu_program.h"

#include <limits>
#include <utility>

namespace hyle::compute::ggml_runtime {

Result<CpuExecution> CpuExecution::prepare(ggml_cgraph *graph, int threads,
                                           mem::AllocatorRef allocator,
                                           ggml_abort_callback abort,
                                           void *abort_data) noexcept {
  if (graph == nullptr || threads <= 0 || threads > GGML_MAX_N_THREADS ||
      !allocator.valid() || ggml_graph_n_nodes(graph) == 0)
    return std::unexpected(Status::InvalidArgument("ggml CPU execution"));

  auto selected = select_cpu_program();
  if (!selected)
    return std::unexpected(selected.error());
  const auto *program = *selected;
  // Lowering and public ggml conversions can also use the linked baseline's
  // tables. Each isolated program initializes its own CPU tables cold.
  ggml_cpu_init();
  if (program->initialize != &ggml_cpu_init)
    program->initialize();
  auto params = ggml_threadpool_params_default(threads);
  params.poll = 0;
  params.prio = GGML_SCHED_PRIO_NORMAL;
  int native_error = 0;
  auto *pool = program->create_pool(&params, &native_error);
  if (pool == nullptr) {
    if (native_error == 0)
      return std::unexpected(Status::ResourceExhausted("ggml CPU pool"));
#if HYLE_OS_WINDOWS
    return std::unexpected(status_from_win32(
        static_cast<std::uint32_t>(native_error), "ggml CPU pool"));
#else
    return std::unexpected(status_from_errno(native_error, "ggml CPU pool"));
#endif
  }
  CpuExecution execution;
  execution.allocator_ = allocator;
  execution.program_ = program;
  execution.graph_ = graph;
  execution.plan_ = program->plan(graph, threads, pool);
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
  switch (program_->compute(graph_, &plan_)) {
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
      program_(std::exchange(other.program_, nullptr)),
      graph_(std::exchange(other.graph_, nullptr)),
      plan_(std::exchange(other.plan_, ggml_cplan{})) {}

CpuExecution &CpuExecution::operator=(CpuExecution &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = other.allocator_;
    program_ = std::exchange(other.program_, nullptr);
    graph_ = std::exchange(other.graph_, nullptr);
    plan_ = std::exchange(other.plan_, ggml_cplan{});
  }
  return *this;
}

CpuExecution::~CpuExecution() { reset(); }

void CpuExecution::reset() noexcept {
  if (plan_.threadpool != nullptr)
    program_->free_pool(plan_.threadpool);
  if (plan_.work_data != nullptr)
    allocator_.deallocate(plan_.work_data, plan_.work_size, GGML_MEM_ALIGN);
  plan_ = {};
  graph_ = nullptr;
  program_ = nullptr;
}

} // namespace hyle::compute::ggml_runtime
