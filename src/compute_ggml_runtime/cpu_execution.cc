#include "cpu_execution.h"
#include "hyle/core/environment.h"
#include "cpu_program.h"
#include "hyle/util/ref_counted.h"

#include <limits>
#include <mutex>
#include <system_error>
#include <utility>

#if defined(__APPLE__)
#include "cpu_blas_bridge.h"
#include "ggml-blas.h"
#include "ggml-impl.h"
#include "hyle/mem/std_allocator.h"
#include <algorithm>
#include <new>
#include <vector>
#endif
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
#include "hyle/platform/cpu_convenience.h"
#endif

namespace hyle::compute::ggml_runtime {
#if defined(__APPLE__)
struct CpuBlasSegments final {
  struct Segment final {
    ggml_cgraph graph{};
    ggml_cplan plan{};
    bool blas = false;
  };
  explicit CpuBlasSegments(mem::AllocatorRef allocator)
      : segments(mem::StdAllocator<Segment>(allocator)) {}
  std::vector<Segment, mem::StdAllocator<Segment>> segments;
  std::size_t cpu_bytes = 0, blas_bytes = 0;
};
#endif
namespace {
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
int performance_threads(int budget) noexcept {
  if (budget <= 1)
    return budget;
  const auto classes = platform::available_cpu_cache_classes();
  int count = 0;
  if (classes)
    for (const auto &level : *classes)
      if (level.kind == platform::CpuKind::kPerformance) {
        const auto cores = level.physical_cores ? level.physical_cores : level.logical_cpus;
        count += static_cast<int>((std::min)(cores, static_cast<unsigned>(budget - count)));
      }
  return count ? count : budget;
}

int plan_thread_count(const ggml_cgraph &graph, int performance, int budget) noexcept {
  if (performance == budget)
    return budget;
  // Keep the extra workers for single-row BF16 projections with enough
  // contiguous weight traffic per worker. Multirow matrices keep the
  // compact team so prefill does not pay mixed-core barrier costs.
  constexpr std::size_t bytes_per_worker = 128 * 1024;
  const auto minimum_bytes = static_cast<std::size_t>(budget) * bytes_per_worker;
  bool large_projection = false;
  for (int index = 0; index < graph.n_nodes; ++index) {
    const auto *node = graph.nodes[index];
    if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1])
      continue;
    if (ggml_nrows(node->src[1]) != 1)
      return performance;
    const auto *weights = node->src[0];
    large_projection |= weights->type == GGML_TYPE_BF16 && ggml_is_contiguous(weights) &&
                        ggml_nbytes(weights) >= minimum_bytes;
  }
  return large_projection ? budget : performance;
}
#endif


struct ActiveTeam final {
  const void *team;
  ActiveTeam *previous;
};
thread_local ActiveTeam *active_team = nullptr;

#if defined(__APPLE__)
Result<CpuBlasSegments *> prepare_blas_segments(ggml_cgraph *graph,
    const CpuProgram &program, int threads, ggml_threadpool *pool,
    mem::AllocatorRef allocator) noexcept {
  bool needed = false;
  for (int i = 0; i < graph->n_nodes; ++i)
    needed = needed || ggml_hyle_blas_supports(graph->nodes[i]);
  if (!needed)
    return nullptr; // Preserve the original single-program path exactly.
  auto *result = allocator.construct<CpuBlasSegments>(allocator);
  if (!result)
    return std::unexpected(Status::ResourceExhausted("CPU BLAS segment metadata"));
  try {
    for (int begin = 0; begin < graph->n_nodes;) {
      const bool blas = ggml_hyle_blas_supports(graph->nodes[begin]);
      int end = begin + 1;
      while (end < graph->n_nodes &&
             ggml_hyle_blas_supports(graph->nodes[end]) == blas)
        ++end;
      result->segments.push_back({.graph = ggml_graph_view(graph, begin, end),
                                  .blas = blas});
      begin = end;
    }
    // The vector is now fixed; plans see the final original graph-view addresses.
    for (auto &segment : result->segments) {
      if (!segment.blas) {
        segment.plan = program.plan(&segment.graph, threads, pool);
        result->cpu_bytes = (std::max)(result->cpu_bytes, segment.plan.work_size);
      } else for (int i = 0; i < segment.graph.n_nodes; ++i) {
        const auto *weights = segment.graph.nodes[i]->src[0];
        if (weights->type == GGML_TYPE_F32)
          continue;
        const auto count = ggml_nelements(weights);
        if (count <= 0 || static_cast<std::uint64_t>(count) >
                static_cast<std::uint64_t>(PTRDIFF_MAX) / sizeof(float)) {
          allocator.destroy(result);
          return std::unexpected(Status::ResourceExhausted("CPU BLAS weight workspace"));
        }
        result->blas_bytes = (std::max)(result->blas_bytes,
            static_cast<std::size_t>(count) * sizeof(float));
      }
    }
    return result;
  } catch (const std::bad_alloc &) {
    allocator.destroy(result);
    return std::unexpected(Status::ResourceExhausted("CPU BLAS segment metadata"));
  }
}

Status blas_status(ggml_status status) noexcept {
  switch (status) {
  case GGML_STATUS_SUCCESS: return Status::Ok();
  case GGML_STATUS_ABORTED: return Status::Cancelled("ggml CPU execution");
  case GGML_STATUS_ALLOC_FAILED: return Status::ResourceExhausted("ggml CPU execution");
  case GGML_STATUS_FAILED: return Status::Internal("ggml CPU execution");
  }
  return Status::Internal("unknown ggml CPU status");
}
#endif

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
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
  int preferred_threads;
#endif
  const CpuProgram *program = nullptr;
  ggml_threadpool *pool = nullptr;
  std::mutex gate;
#if defined(__APPLE__)
  ggml_backend_t blas = nullptr;
#endif

  State(mem::AllocatorRef source, int count) noexcept
      : allocator(source), threads(count)
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
      , preferred_threads(performance_threads(count))
#endif
  {}
  void destroy() noexcept {
    const auto source = allocator;
    if (pool)
      program->free_pool(pool);
#if defined(__APPLE__)
    if (blas)
      ggml_backend_free(blas);
#endif
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
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
    // Compact plans use only part of an expanded team. Park unused workers
    // between jobs so their polling does not compete with those plans.
    if (threads != preferred_threads)
      params.poll = 0;
#endif
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
#if defined(__APPLE__)
  Status prepare_blas(std::size_t bytes) noexcept try {
    // Called under the SAME gate as original native team planning/compute.
    // All slots share one bounded maximum scratch, not an F32 weight bank.
    if (!blas)
      blas = ggml_hyle_blas_init();
    if (!blas || !ggml_hyle_blas_reserve(blas, bytes))
      return Status::ResourceExhausted("native CPU BLAS workspace");
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("native CPU BLAS context");
  }
#endif
};

#if defined(__APPLE__)
bool blas_activation_widening_supported(const ggml_tensor &node) noexcept {
  if (node.op != GGML_OP_MUL_MAT || !node.src[1] ||
      (node.src[1]->type != GGML_TYPE_F16 && node.src[1]->type != GGML_TYPE_BF16) ||
      !ggml_is_contiguous(node.src[1]))
    return false;
  auto activation = *node.src[1];
  activation.type = GGML_TYPE_F32;
  activation.op = GGML_OP_CPY;
  activation.src[0] = node.src[1];
  activation.src[1] = nullptr;
  std::size_t stride = sizeof(float);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (activation.ne[i] < 0 || static_cast<std::uint64_t>(activation.ne[i]) >
            SIZE_MAX / stride)
      return false;
    activation.nb[i] = stride;
    stride *= static_cast<std::size_t>(activation.ne[i]);
    if (!stride)
      return false;
  }
  auto candidate = node;
  candidate.src[1] = &activation;
  return ggml_hyle_blas_supports(&candidate);
}
#endif

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
    int threads = state.threads;
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
    threads = plan_thread_count(*graph, state.preferred_threads, state.threads);
#endif
#if defined(__APPLE__)
    HYLE_ASSIGN_OR_RETURN(execution.blas_segments_, prepare_blas_segments(
        graph, *state.program, threads, state.pool, allocator));
    if (execution.blas_segments_) {
      HYLE_RETURN_IF_ERROR(state.prepare_blas(execution.blas_segments_->blas_bytes));
      execution.plan_.work_size = execution.blas_segments_->cpu_bytes;
    } else
#endif
    execution.plan_ = state.program->plan(graph, threads, state.pool);
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
#if defined(__APPLE__)
  if (execution.blas_segments_)
    for (auto &segment : execution.blas_segments_->segments) {
      segment.plan.work_data = execution.plan_.work_data;
      segment.plan.abort_callback = abort;
      segment.plan.abort_callback_data = abort_data;
    }
#endif
  return execution;
}

Status CpuExecution::run() noexcept {
  if (graph_ == nullptr || !team_.valid())
    return Status::FailedPrecondition("ggml CPU execution is empty");
  auto &state = *team_.state_;
  TeamScope scope(state.gate, &state);
  HYLE_RETURN_IF_ERROR(scope.enter());
#if defined(__APPLE__)
  if (blas_segments_) {
    for (auto &segment : blas_segments_->segments) {
      if (plan_.abort_callback && plan_.abort_callback(plan_.abort_callback_data))
        return Status::Cancelled("ggml CPU execution");
      const auto status = segment.blas
          ? ggml_hyle_blas_compute(state.blas, &segment.graph)
          : program_->compute(&segment.graph, &segment.plan);
      HYLE_RETURN_IF_ERROR(blas_status(status));
    }
    if (plan_.abort_callback && plan_.abort_callback(plan_.abort_callback_data))
      return Status::Cancelled("ggml CPU execution");
    return Status::Ok();
  }
#endif
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
      plan_(std::exchange(other.plan_, ggml_cplan{}))
#if defined(__APPLE__)
      , blas_segments_(std::exchange(other.blas_segments_, nullptr))
#endif
      {}

CpuExecution &CpuExecution::operator=(CpuExecution &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = other.allocator_;
    team_ = std::move(other.team_);
    program_ = std::exchange(other.program_, nullptr);
    graph_ = std::exchange(other.graph_, nullptr);
    plan_ = std::exchange(other.plan_, ggml_cplan{});
#if defined(__APPLE__)
    blas_segments_ = std::exchange(other.blas_segments_, nullptr);
#endif
  }
  return *this;
}

CpuExecution::~CpuExecution() { reset(); }

void CpuExecution::reset() noexcept {
  const auto plan = std::exchange(plan_, ggml_cplan{});
  graph_ = nullptr;
  program_ = nullptr;
#if defined(__APPLE__)
  if (auto *segments = std::exchange(blas_segments_, nullptr))
    allocator_.destroy(segments);
#endif
  // The last team drop joins/frees its native workers before releasing this
  // slot's workspace. Other active slots retain the team independently.
  team_ = {};
  if (plan.work_data != nullptr)
    allocator_.deallocate(plan.work_data, plan.work_size, GGML_MEM_ALIGN);
}

} // namespace hyle::compute::ggml_runtime
