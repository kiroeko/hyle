#include "task_scope_p1_common.h"

#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "hyle/mem/pmr.h"
#include "hyle/sched/join.h"
#include "hyle/sched/task_scope.h"
#include "hyle/util/spin_lock.h"

#if defined(HYLE_SCHED_V8_PLACEMENT_SHIFT)
extern "C" void hyle_sched_v8_placement_shift() noexcept;
#endif

namespace {

namespace nb = ankerl::nanobench;
namespace task_scope = hyle::bench::sched_task_scope_p1;
namespace common = hyle::bench::sched_p1;

constexpr const char *kTitle = "dynamic TaskScope owner cleanup aligned";
constexpr const char *kCandidateRow = "HYLE TaskScope";
constexpr const char *kPeerRow = "manual public task owner";
constexpr std::size_t kPairedWarmupOwners = 4'096;

bool peer_first_requested() noexcept {
  return std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
}

enum class AaRoute { kNone, kCandidate, kPeer };

AaRoute aa_route_requested() {
  const char *value = std::getenv("HYLE_BENCH_AA_ROUTE");
  if (value == nullptr)
    return AaRoute::kNone;
  if (std::strcmp(value, "candidate") == 0)
    return AaRoute::kCandidate;
  if (std::strcmp(value, "peer") == 0)
    return AaRoute::kPeer;
  common::fail("HYLE_BENCH_AA_ROUTE must be candidate or peer");
}

hyle::Status candidate_owner(hyle::sched::Scheduler &scheduler,
                             hyle::mem::AllocatorRef allocator,
                             std::span<task_scope::Child> children) {
  auto created = hyle::sched::TaskScope::create(scheduler, allocator);
  if (!created)
    return std::move(created).error();
  auto scope = *std::move(created);
  for (task_scope::Child &child : children) {
    const hyle::Status admitted =
        scope.add(task_scope::helper(allocator, &child));
    if (!admitted.ok())
      return admitted;
  }
  hyle::Status status = scope.join();
  scheduler.drain();
  return status;
}

hyle::Status manual_owner(hyle::sched::Scheduler &scheduler,
                          hyle::mem::AllocatorRef allocator,
                          std::span<task_scope::Child> children) {
  auto token_result = hyle::sched::CancelToken::try_create_root(allocator);
  if (!token_result)
    return std::move(token_result).error();
  auto group_token = *std::move(token_result);

  hyle::mem::PmrResource resource(allocator);
  std::pmr::vector<hyle::sched::JoinHandle<hyle::Status>> handles(&resource);
  handles.reserve(children.size());
  hyle::util::SpinLock admission_lock;
  bool closed = false;
  for (task_scope::Child &child : children) {
    std::lock_guard<hyle::util::SpinLock> lock(admission_lock);
    if (closed)
      return hyle::Status::FailedPrecondition("manual task owner is closed");
    handles.push_back(hyle::sched::spawn_task(
        scheduler, allocator,
        task_scope::helper(allocator, &child, &group_token)));
  }
  {
    std::lock_guard<hyle::util::SpinLock> lock(admission_lock);
    closed = true;
  }

  hyle::Status first_error;
  for (auto &handle : handles) {
    auto result = handle.join();
    if (!result)
      first_error.Update(result.error());
    else
      first_error.Update(*result);
  }
  scheduler.drain();
  return first_error;
}

struct RouteState {
  std::uint64_t next_owner = 0;
  std::uint64_t completed_owners = 0;
  std::uint64_t checksum = 0;
};

template <class Route>
void run_one_owner(Route &&route, RouteState &state,
                   hyle::sched::Scheduler &scheduler,
                   hyle::mem::AllocatorRef allocator,
                   common::WorkerHits &hits) {
  std::array<task_scope::Child, task_scope::kFanout> children{};
  const std::uint64_t owner = state.next_owner++;
  const bool expected_failure = owner % task_scope::kFailurePeriod == 7;
  for (std::size_t child = 0; child < children.size(); ++child) {
    children[child] = task_scope::Child{
        (owner << 32) | static_cast<std::uint64_t>(child + 1), 0,
        expected_failure && child == children.size() / 2, &hits};
  }

  const hyle::Status status = route(scheduler, allocator, children);
  if (expected_failure ? status.code() != hyle::StatusCode::Unavailable
                       : !status.ok()) {
    common::fail("dynamic helper owner returned the wrong aggregate status");
  }
  for (const task_scope::Child &child : children) {
    if (child.output != task_scope::child_value(child.seed))
      common::fail("dynamic helper output was not fully drained");
    state.checksum += child.output;
  }
  ++state.completed_owners;
  nb::doNotOptimizeAway(state.checksum);
}

struct RouteResult {
  std::uint64_t owners = 0;
  std::uint64_t callbacks = 0;
};

template <class Route>
RouteResult benchmark_route(
    const char *row_name, Route &&route,
    const std::array<unsigned, common::kWorkerCount> &cpus,
    hyle::mem::AllocatorRef allocator) {
  common::AffinityState affinity(cpus);
  hyle::sched::SchedulerOptions options;
  options.worker_cpu_set = std::span<const unsigned>(cpus);
  options.worker_config_policy = hyle::platform::ThreadConfigPolicy::kRequired;
  options.on_worker_start = +[](void *raw, unsigned worker) noexcept {
    static_cast<common::AffinityState *>(raw)->record(worker,
                                                      common::current_cpu());
  };
  options.worker_hook_context = &affinity;

  hyle::Status scheduler_status;
  hyle::sched::Scheduler scheduler(static_cast<unsigned>(common::kWorkerCount),
                                   allocator, 512, options, &scheduler_status);
  if (!scheduler_status.ok())
    common::fail("Scheduler construction failed");
  affinity.await_ready();
  if (!affinity.exact())
    common::fail("Scheduler worker placement is not exact");

  common::WorkerHits hits;
  RouteState state;
  for (std::size_t warmup = 0; warmup < kPairedWarmupOwners; ++warmup)
    run_one_owner(route, state, scheduler, allocator, hits);
  hits.reset();
  state.completed_owners = 0;

  nb::Bench bench;
  bench.title(kTitle)
      .batch(1)
      .epochs(9)
      .minEpochTime(std::chrono::milliseconds(100));
  bench.run(row_name, [&] {
    run_one_owner(route, state, scheduler, allocator, hits);
  });

  if (state.completed_owners == 0 ||
      hits.total() != state.completed_owners * task_scope::kFanout)
    common::fail("paired owner callback accounting is inconsistent");
  if (!affinity.exact() || hits.active() == 0)
    common::fail("paired owner worker placement is inconsistent");
  const RouteResult result{state.completed_owners, hits.total()};
  scheduler.shutdown();
  return result;
}

void print_telemetry(const std::array<unsigned, common::kWorkerCount> &cpus,
                     bool peer_first, AaRoute aa_route) {
  const char *topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
  const char *parent_mask = std::getenv("HYLE_PERF_PARENT_MASK");
  const char *priority = std::getenv("HYLE_PERF_PRIORITY");
  std::printf(
      "# HYLE_SCHED_TASK_SCOPE_V8_RUN execution_order=%s "
      "aa_route=%s affinity_topology=%s worker_count=%zu worker_cpus=",
      peer_first ? "peer-first" : "candidate-first",
      aa_route == AaRoute::kCandidate
          ? "candidate"
          : (aa_route == AaRoute::kPeer ? "peer" : "none"),
      topology != nullptr ? topology : "unmanaged", cpus.size());
  common::print_cpu_list(cpus);
  std::printf(" parent_mask=%s priority=%s affinity_ok=1 run_nonce=%s\n",
              parent_mask != nullptr ? parent_mask : "unmanaged",
              priority != nullptr ? priority : "unmanaged",
              hyle::bench::perf::run_nonce());
}

} // namespace

int main() {
#if defined(HYLE_SCHED_V8_PLACEMENT_SHIFT)
  // Keep the address perturbation outside every setup, warmup and timed row.
  hyle_sched_v8_placement_shift();
#endif
  const auto cpus = common::requested_cpus();
  const hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator();
  const bool peer_first = peer_first_requested();
  const AaRoute aa_route = aa_route_requested();
  using OwnerRoute = hyle::Status (*)(
      hyle::sched::Scheduler &, hyle::mem::AllocatorRef,
      std::span<task_scope::Child>);
  const OwnerRoute candidate_row_route =
      aa_route == AaRoute::kPeer ? &manual_owner : &candidate_owner;
  const OwnerRoute peer_row_route =
      aa_route == AaRoute::kCandidate ? &candidate_owner : &manual_owner;
  RouteResult candidate_result;
  RouteResult peer_result;
  if (peer_first) {
    peer_result = benchmark_route(kPeerRow, peer_row_route, cpus, allocator);
    candidate_result = benchmark_route(kCandidateRow, candidate_row_route,
                                       cpus, allocator);
  } else {
    candidate_result = benchmark_route(kCandidateRow, candidate_row_route,
                                       cpus, allocator);
    peer_result = benchmark_route(kPeerRow, peer_row_route, cpus, allocator);
  }

  std::printf("# HYLE_SCHED_TASK_SCOPE_V8_ACCOUNTING "
              "candidate_owners=%llu peer_owners=%llu "
              "candidate_callbacks=%llu peer_callbacks=%llu "
              "accounting_ok=1\n",
              static_cast<unsigned long long>(candidate_result.owners),
              static_cast<unsigned long long>(peer_result.owners),
              static_cast<unsigned long long>(candidate_result.callbacks),
              static_cast<unsigned long long>(peer_result.callbacks));
  print_telemetry(cpus, peer_first, aa_route);
  return 0;
}
