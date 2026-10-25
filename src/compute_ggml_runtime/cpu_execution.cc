#include "cpu_execution.h"
#include "hyle/core/environment.h"
#include "cpu_program.h"
#include "hyle/util/ref_counted.h"

#include <limits>
#include <mutex>
#include <system_error>
#include <utility>

namespace hyle::compute::ggml_runtime {
namespace {

struct ActiveTeam final {
  const void *team;
  ActiveTeam *previous;
};
thread_local ActiveTeam *active_team = nullptr;

// The executor invokes the complete compute synchronously. Only that callback
// locks the native team; no lock is held while dispatching into the executor.
class TeamScope final {
public:
  TeamScope(std::mutex &gate, const void *team) noexcept
      : lock_(gate, std::defer_lock), frame_{team, nullptr} {}
  Status enter() noexcept {
    for (auto *current = active_team; current; current = current->previous)
      if (current->team == frame_.team)
        return Status::FailedPrecondition("ggml CPU team reentry");
    try {
      lock_.lock();
    } catch (const std::system_error &error) {
      return Status::Internal(error.what());
    }
    frame_.previous = active_team;
    active_team = &frame_;
    entered_ = true;
    return Status::Ok();
  }
  ~TeamScope() {
    if (entered_)
      active_team = frame_.previous;
  }

private:
  std::unique_lock<std::mutex> lock_;
  ActiveTeam frame_;
  bool entered_ = false;
};

} // namespace

struct CpuTeam::State final : util::RefCounted<State> {
  mem::AllocatorRef allocator;
  int threads;
  const CpuProgram *program = nullptr;
  ggml_threadpool *pool = nullptr;
  std::mutex gate;

  State(mem::AllocatorRef source, int count) noexcept
      : allocator(source), threads(count) {}
  void destroy() noexcept {
    const auto source = allocator;
    if (pool)
      program->free_pool(pool);
    this->~State();
    source.deallocate(this, sizeof(State), alignof(State));
  }
  Status initialize() noexcept {
    if (pool)
      return Status::Ok();
    auto selected = select_cpu_program();
    if (!selected)
      return selected.error();
    // Lowering and public ggml conversions can also use the linked baseline's
    // tables. Each isolated program initializes its own CPU tables cold.
    const auto *selected_program = *selected;
    ggml_cpu_init();
    if (selected_program->initialize != &ggml_cpu_init)
      selected_program->initialize();
    auto params = ggml_threadpool_params_default(threads);
    params.prio = GGML_SCHED_PRIO_NORMAL;
    int native_error = 0;
    auto *prepared = selected_program->create_pool(&params, &native_error);
    if (prepared == nullptr) {
      if (native_error == 0)
        return Status::ResourceExhausted("ggml CPU pool");
#if HYLE_OS_WINDOWS
      return status_from_win32(static_cast<std::uint32_t>(native_error),
                               "ggml CPU pool");
#else
      return status_from_errno(native_error, "ggml CPU pool");
#endif
    }
    program = selected_program;
    pool = prepared;
    return Status::Ok();
  }
};

Result<CpuTeam> CpuTeam::create(int threads,
                                mem::AllocatorRef allocator) noexcept {
  if (threads <= 0 || threads > GGML_MAX_N_THREADS || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("ggml CPU team"));
  auto *storage = allocator.allocate(sizeof(State), alignof(State));
  if (!storage)
    return std::unexpected(Status::ResourceExhausted("ggml CPU team"));
  return CpuTeam(::new (storage) State(allocator, threads));
}
CpuTeam::CpuTeam(const CpuTeam &other) noexcept : state_(other.state_) {
  if (state_)
    state_->incref();
}
CpuTeam &CpuTeam::operator=(const CpuTeam &other) noexcept {
  if (state_ != other.state_) {
    if (other.state_)
      other.state_->incref();
    if (state_)
      state_->decref();
    state_ = other.state_;
  }
  return *this;
}
CpuTeam::CpuTeam(CpuTeam &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
CpuTeam &CpuTeam::operator=(CpuTeam &&other) noexcept {
  if (this != &other) {
    if (state_)
      state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
CpuTeam::~CpuTeam() {
  if (state_)
    state_->decref();
}

Result<CpuExecution> CpuExecution::prepare(ggml_cgraph *graph,
                                           const CpuTeam &team,
                                           mem::AllocatorRef allocator,
                                           ggml_abort_callback abort,
                                           void *abort_data) noexcept {
  if (graph == nullptr || !team.valid() || !allocator.valid() ||
      ggml_graph_n_nodes(graph) == 0)
    return std::unexpected(Status::InvalidArgument("ggml CPU execution"));

  CpuExecution execution;
  execution.allocator_ = allocator;
  execution.team_ = team;
  execution.graph_ = graph;
  {
    auto &state = *execution.team_.state_;
    TeamScope scope(state.gate, &state);
    HYLE_RETURN_IF_ERROR(scope.enter());
    HYLE_RETURN_IF_ERROR(state.initialize());
    execution.program_ = state.program;
    // Planning reads the same native pool used by compute and is serialized
    // with it. Each slot still owns its independent plan and workspace.
    execution.plan_ = state.program->plan(graph, state.threads, state.pool);
  }
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
  if (graph_ == nullptr || !team_.valid())
    return Status::FailedPrecondition("ggml CPU execution is empty");
  auto &state = *team_.state_;
  TeamScope scope(state.gate, &state);
  HYLE_RETURN_IF_ERROR(scope.enter());
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
    : allocator_(other.allocator_), team_(std::move(other.team_)),
      program_(std::exchange(other.program_, nullptr)),
      graph_(std::exchange(other.graph_, nullptr)),
      plan_(std::exchange(other.plan_, ggml_cplan{})) {}

CpuExecution &CpuExecution::operator=(CpuExecution &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = other.allocator_;
    team_ = std::move(other.team_);
    program_ = std::exchange(other.program_, nullptr);
    graph_ = std::exchange(other.graph_, nullptr);
    plan_ = std::exchange(other.plan_, ggml_cplan{});
  }
  return *this;
}

CpuExecution::~CpuExecution() { reset(); }

void CpuExecution::reset() noexcept {
  const auto plan = std::exchange(plan_, ggml_cplan{});
  graph_ = nullptr;
  program_ = nullptr;
  // The last team drop joins/frees its native workers before releasing this
  // slot's workspace. Other active slots retain the team independently.
  team_ = {};
  if (plan.work_data != nullptr)
    allocator_.deallocate(plan.work_data, plan.work_size, GGML_MEM_ALIGN);
}

} // namespace hyle::compute::ggml_runtime
