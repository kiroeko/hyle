// Product evidence for the error carrier as it is consumed from HYLE source.
//
// The algorithms below are deliberately shared between HYLE and Abseil.  They
// replay ownership/value-category traces that exist in current HYLE consumers;
// the only substituted type is the error carrier (Status/Result vs
// absl::Status/StatusOr).  This is not a primitive microbenchmark and it does
// not assign an invented frequency to a future inference engine.
#include <nanobench.h>

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "hyle/core/assert.h"
#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#include "../common/perf_process.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/status/statusor.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kGraphNodes = 64;
constexpr std::size_t kLastGraphNode = kGraphNodes - 1;
constexpr std::uint64_t kGraphChecksum =
    (kGraphNodes * (kGraphNodes + 1)) / 2;
constexpr std::string_view kFailureMessage =
    "resource not found in shard 7";

struct HyleCarrier {
  using Status = hyle::Status;
  template <class T>
  using Result = hyle::Result<T>;

  static Status ok_status() noexcept { return Status{}; }
  static Status failure() noexcept {
    return Status::NotFound(kFailureMessage);
  }
  static bool ok(const Status& status) noexcept { return status.ok(); }
  static bool is_expected_failure(const Status& status) noexcept {
    return status.IsNotFound() && status.message() == kFailureMessage;
  }
  template <class T>
  static Result<T> error_result(Status status) noexcept {
    return hyle::make_unexpected(std::move(status));
  }
  template <class T>
  static bool result_ok(const Result<T>& result) noexcept {
    return result.has_value();
  }
  template <class T>
  static bool result_is_expected_failure(const Result<T>& result) noexcept {
    return !result && is_expected_failure(result.error().status());
  }
};

#if defined(HYLE_BENCH_ABSL)
struct PeerCarrier {
  using Status = absl::Status;
  template <class T>
  using Result = absl::StatusOr<T>;

  static Status ok_status() noexcept { return absl::OkStatus(); }
  static Status failure() noexcept {
    return Status(absl::StatusCode::kNotFound, kFailureMessage,
                  absl::SourceLocation());
  }
  static bool ok(const Status& status) noexcept { return status.ok(); }
  static bool is_expected_failure(const Status& status) noexcept {
    const absl::string_view message = status.message();
    return status.code() == absl::StatusCode::kNotFound &&
           std::string_view(message.data(), message.size()) == kFailureMessage;
  }
  template <class T>
  static Result<T> error_result(Status status) noexcept {
    return Result<T>(std::move(status));
  }
  template <class T>
  static bool result_ok(const Result<T>& result) noexcept {
    return result.ok();
  }
  template <class T>
  static bool result_is_expected_failure(const Result<T>& result) noexcept {
    return !result.ok() && is_expected_failure(result.status());
  }
};
#endif

// TaskGraph::evaluate provenance: src/graph/taskgraph.cc.  The node stream,
// indirect calls, local carrier, status check, and dead-local move on failure
// are kept in one shared template.  Node/context storage lives outside the
// measured callback, just as a built graph does.
struct GraphTrace {
  std::uint64_t calls = 0;
  std::uint64_t checksum = 0;
  std::size_t failure_index = kGraphNodes;
};

struct GraphNodeContext {
  GraphTrace* trace = nullptr;
  std::size_t index = 0;
};

template <class Carrier>
struct GraphNode {
  using Function = typename Carrier::Status (*)(GraphNodeContext*) noexcept;
  Function function = nullptr;
  GraphNodeContext* context = nullptr;
};

template <class Carrier>
HYLE_NOINLINE typename Carrier::Status graph_success_node(
    GraphNodeContext* context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  return Carrier::ok_status();
}

template <class Carrier>
HYLE_NOINLINE typename Carrier::Status graph_error_node(
    GraphNodeContext* context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  context->trace->failure_index = context->index;
  return Carrier::failure();
}

template <class Carrier>
HYLE_NOINLINE typename Carrier::Status taskgraph_evaluate(
    const GraphNode<Carrier>* nodes, std::size_t size) noexcept {
  for (std::size_t index = 0; index < size; ++index) {
    typename Carrier::Status status =
        nodes[index].function(nodes[index].context);
    if (!Carrier::ok(status)) return std::move(status);
  }
  return Carrier::ok_status();
}

template <class Carrier>
void initialize_graph(
    std::array<GraphNode<Carrier>, kGraphNodes>& nodes,
    std::array<GraphNodeContext, kGraphNodes>& contexts, GraphTrace& trace,
    bool fail_last) noexcept {
  for (std::size_t index = 0; index < kGraphNodes; ++index) {
    contexts[index] = GraphNodeContext{&trace, index};
    nodes[index] = GraphNode<Carrier>{
        fail_last && index == kLastGraphNode ? graph_error_node<Carrier>
                                             : graph_success_node<Carrier>,
        &contexts[index]};
  }
}

// evaluate_parallel/evaluate_parallel_wavefront provenance:
// src/graph/parallel_eval.cc.  A worker creates an owning error, copies it to
// the per-node outcome, copies it through record(Status), moves it to the
// first-error slot, then the waiter copies the protected first error before the
// shared state dies.  Both carriers instantiate this exact ownership trace.
template <class Carrier>
HYLE_NOINLINE void record_first_error(typename Carrier::Status& first_error,
                                      typename Carrier::Status status) noexcept {
  first_error = std::move(status);
}

template <class Carrier>
HYLE_NOINLINE typename Carrier::Status parallel_first_error_handoff() noexcept {
  typename Carrier::Status outcome;
  typename Carrier::Status first_error;
  typename Carrier::Status status = Carrier::failure();
  outcome = status;
  if (!Carrier::ok(status)) record_first_error<Carrier>(first_error, status);
  typename Carrier::Status result;
  result = first_error;
  nb::doNotOptimizeAway(outcome);
  return result;
}

// JoinState provenance: include/hyle/sched/join.h.  The uncontended lock is
// common scaffolding, not a carrier substitute: producer and consumer are
// separated by noinline boundaries, the producer moves a fresh error into the
// shared slot, and take() copies under the lock before moving into Result.
class ReplaySpinLock {
 public:
  void lock() noexcept {
    while (flag_.test_and_set(std::memory_order_acquire)) {
    }
  }
  void unlock() noexcept { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

template <class Carrier>
struct JoinSlot {
  ReplaySpinLock lock;
  bool done = false;
  typename Carrier::Status status;

  void complete(typename Carrier::Status incoming) noexcept {
    lock.lock();
    status = std::move(incoming);
    done = true;
    lock.unlock();
  }

  void set_error(typename Carrier::Status incoming) noexcept {
    complete(std::move(incoming));
  }

  typename Carrier::template Result<int> take() noexcept {
    lock.lock();
    HYLE_FATAL_ASSERT(done);
    typename Carrier::Status local = status;
    if (!Carrier::ok(local)) {
      lock.unlock();
      return Carrier::template error_result<int>(std::move(local));
    }
    lock.unlock();
    return 7;
  }
};

template <class Carrier>
HYLE_NOINLINE void join_producer(JoinSlot<Carrier>* slot) noexcept {
  slot->set_error(Carrier::failure());
}

template <class Carrier>
HYLE_NOINLINE typename Carrier::template Result<int> join_consumer(
    JoinSlot<Carrier>* slot) noexcept {
  return slot->take();
}

template <class Carrier>
HYLE_NOINLINE typename Carrier::template Result<int>
failed_join_handoff() noexcept {
  JoinSlot<Carrier> slot;
  join_producer<Carrier>(&slot);
  return join_consumer<Carrier>(&slot);
}

bool rival_first_requested() {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, "HYLE_BENCH_RIVAL_FIRST") == 0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
#endif
}

template <class CandidateOp, class PeerOp>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, CandidateOp&& candidate,
              std::string_view peer_name, PeerOp&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<PeerOp>(peer));
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
  } else {
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
    bench.run(peer_name, std::forward<PeerOp>(peer));
  }
}

template <class Function>
std::uintptr_t function_address(Function function) noexcept {
  return reinterpret_cast<std::uintptr_t>(function);
}

int validate_contract() {
  GraphTrace hyle_success_trace;
  GraphTrace hyle_failure_trace;
  std::array<GraphNodeContext, kGraphNodes> hyle_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> hyle_failure_contexts{};
  std::array<GraphNode<HyleCarrier>, kGraphNodes> hyle_success_nodes{};
  std::array<GraphNode<HyleCarrier>, kGraphNodes> hyle_failure_nodes{};
  initialize_graph(hyle_success_nodes, hyle_success_contexts,
                   hyle_success_trace, false);
  initialize_graph(hyle_failure_nodes, hyle_failure_contexts,
                   hyle_failure_trace, true);
  const auto hyle_success = taskgraph_evaluate<HyleCarrier>(
      hyle_success_nodes.data(), hyle_success_nodes.size());
  const auto hyle_failure = taskgraph_evaluate<HyleCarrier>(
      hyle_failure_nodes.data(), hyle_failure_nodes.size());
  const auto hyle_parallel = parallel_first_error_handoff<HyleCarrier>();
  const auto hyle_join = failed_join_handoff<HyleCarrier>();

  bool graph_ok = HyleCarrier::ok(hyle_success) &&
                  HyleCarrier::is_expected_failure(hyle_failure) &&
                  hyle_success_trace.calls == kGraphNodes &&
                  hyle_failure_trace.calls == kGraphNodes &&
                  hyle_success_trace.checksum == kGraphChecksum &&
                  hyle_failure_trace.checksum == kGraphChecksum &&
                  hyle_failure_trace.failure_index == kLastGraphNode;
  bool parallel_ok = HyleCarrier::is_expected_failure(hyle_parallel);
  bool join_ok = HyleCarrier::result_is_expected_failure(hyle_join);
  std::uint64_t peer_success_calls = 0;
  std::uint64_t peer_failure_calls = 0;
  std::uint64_t peer_success_checksum = 0;
  std::uint64_t peer_failure_checksum = 0;
  std::size_t peer_failure_index = kGraphNodes;
  bool source_location_ok = false;

#if defined(HYLE_BENCH_ABSL)
  GraphTrace peer_success_trace;
  GraphTrace peer_failure_trace;
  std::array<GraphNodeContext, kGraphNodes> peer_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> peer_failure_contexts{};
  std::array<GraphNode<PeerCarrier>, kGraphNodes> peer_success_nodes{};
  std::array<GraphNode<PeerCarrier>, kGraphNodes> peer_failure_nodes{};
  initialize_graph(peer_success_nodes, peer_success_contexts,
                   peer_success_trace, false);
  initialize_graph(peer_failure_nodes, peer_failure_contexts,
                   peer_failure_trace, true);
  const auto peer_success = taskgraph_evaluate<PeerCarrier>(
      peer_success_nodes.data(), peer_success_nodes.size());
  const auto peer_failure = taskgraph_evaluate<PeerCarrier>(
      peer_failure_nodes.data(), peer_failure_nodes.size());
  const auto peer_parallel = parallel_first_error_handoff<PeerCarrier>();
  const auto peer_join = failed_join_handoff<PeerCarrier>();
  peer_success_calls = peer_success_trace.calls;
  peer_failure_calls = peer_failure_trace.calls;
  peer_success_checksum = peer_success_trace.checksum;
  peer_failure_checksum = peer_failure_trace.checksum;
  peer_failure_index = peer_failure_trace.failure_index;
  graph_ok = graph_ok && PeerCarrier::ok(peer_success) &&
             PeerCarrier::is_expected_failure(peer_failure) &&
             peer_success_calls == kGraphNodes &&
             peer_failure_calls == kGraphNodes &&
             peer_success_checksum == kGraphChecksum &&
             peer_failure_checksum == kGraphChecksum &&
             peer_failure_index == kLastGraphNode;
  parallel_ok = parallel_ok &&
                PeerCarrier::is_expected_failure(peer_parallel);
  join_ok = join_ok && PeerCarrier::result_is_expected_failure(peer_join);
  source_location_ok = peer_failure.GetSourceLocations().empty() &&
                       peer_parallel.GetSourceLocations().empty() &&
                       peer_join.status().GetSourceLocations().empty();
#endif

  const bool passed = graph_ok && parallel_ok && join_ok && source_location_ok;
  std::printf(
      "# HYLE_ERROR_SOURCE_CONSUMER_CONTRACT schema=1 nodes=%zu "
      "hyle_success_calls=%" PRIu64 " peer_success_calls=%" PRIu64 " "
      "hyle_failure_calls=%" PRIu64 " peer_failure_calls=%" PRIu64 " "
      "hyle_success_checksum=%" PRIu64 " peer_success_checksum=%" PRIu64
      " hyle_failure_checksum=%" PRIu64 " peer_failure_checksum=%" PRIu64
      " hyle_failure_index=%zu peer_failure_index=%zu graph=%s "
      "parallel_handoff=%s join_handoff=%s source_location=%s verdict=%s\n",
      kGraphNodes, hyle_success_trace.calls, peer_success_calls,
      hyle_failure_trace.calls, peer_failure_calls,
      hyle_success_trace.checksum, peer_success_checksum,
      hyle_failure_trace.checksum, peer_failure_checksum,
      hyle_failure_trace.failure_index, peer_failure_index,
      graph_ok ? "PASS" : "FAIL", parallel_ok ? "PASS" : "FAIL",
      join_ok ? "PASS" : "FAIL", source_location_ok ? "PASS" : "FAIL",
      passed ? "PASS" : "FAIL");

  std::printf(
      "# HYLE_ERROR_SOURCE_CONSUMER_LAYOUT hyle_eval=%" PRIxPTR
      " hyle_parallel=%" PRIxPTR " hyle_join_producer=%" PRIxPTR
      " hyle_join_consumer=%" PRIxPTR
#if defined(HYLE_BENCH_ABSL)
      " peer_eval=%" PRIxPTR " peer_parallel=%" PRIxPTR
      " peer_join_producer=%" PRIxPTR " peer_join_consumer=%" PRIxPTR
#endif
      "\n",
      function_address(taskgraph_evaluate<HyleCarrier>),
      function_address(parallel_first_error_handoff<HyleCarrier>),
      function_address(join_producer<HyleCarrier>),
      function_address(join_consumer<HyleCarrier>)
#if defined(HYLE_BENCH_ABSL)
          ,
      function_address(taskgraph_evaluate<PeerCarrier>),
      function_address(parallel_first_error_handoff<PeerCarrier>),
      function_address(join_producer<PeerCarrier>),
      function_address(join_consumer<PeerCarrier>)
#endif
  );
  return passed ? 0 : 2;
}

void run_taskgraph_benchmarks([[maybe_unused]] bool peer_first) {
  GraphTrace hyle_success_trace;
  GraphTrace hyle_failure_trace;
  std::array<GraphNodeContext, kGraphNodes> hyle_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> hyle_failure_contexts{};
  std::array<GraphNode<HyleCarrier>, kGraphNodes> hyle_success_nodes{};
  std::array<GraphNode<HyleCarrier>, kGraphNodes> hyle_failure_nodes{};
  initialize_graph(hyle_success_nodes, hyle_success_contexts,
                   hyle_success_trace, false);
  initialize_graph(hyle_failure_nodes, hyle_failure_contexts,
                   hyle_failure_trace, true);

#if defined(HYLE_BENCH_ABSL)
  GraphTrace peer_success_trace;
  GraphTrace peer_failure_trace;
  std::array<GraphNodeContext, kGraphNodes> peer_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> peer_failure_contexts{};
  std::array<GraphNode<PeerCarrier>, kGraphNodes> peer_success_nodes{};
  std::array<GraphNode<PeerCarrier>, kGraphNodes> peer_failure_nodes{};
  initialize_graph(peer_success_nodes, peer_success_contexts,
                   peer_success_trace, false);
  initialize_graph(peer_failure_nodes, peer_failure_contexts,
                   peer_failure_trace, true);
#endif

  {
    nb::Bench bench;
    bench.title("Source TaskGraph 64-node success propagation")
        .relative(true)
        .batch(1)
        .minEpochIterations(100000);
#if defined(HYLE_BENCH_ABSL)
    run_pair(
        bench, peer_first, "HYLE Status source TaskGraph success",
        [&] {
          auto status = taskgraph_evaluate<HyleCarrier>(
              hyle_success_nodes.data(), hyle_success_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(hyle_success_trace.checksum);
        },
        "absl::Status source TaskGraph success", [&] {
          auto status = taskgraph_evaluate<PeerCarrier>(
              peer_success_nodes.data(), peer_success_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(peer_success_trace.checksum);
        });
#else
    bench.run("HYLE Status source TaskGraph success", [&] {
      auto status = taskgraph_evaluate<HyleCarrier>(
          hyle_success_nodes.data(), hyle_success_nodes.size());
      nb::doNotOptimizeAway(status);
      nb::doNotOptimizeAway(hyle_success_trace.checksum);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Source TaskGraph 64-node owning last failure")
        .relative(true)
        .batch(1)
        .minEpochIterations(100000);
#if defined(HYLE_BENCH_ABSL)
    run_pair(
        bench, peer_first, "HYLE Status source TaskGraph owning failure",
        [&] {
          auto status = taskgraph_evaluate<HyleCarrier>(
              hyle_failure_nodes.data(), hyle_failure_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(hyle_failure_trace.failure_index);
        },
        "absl::Status source TaskGraph owning failure", [&] {
          auto status = taskgraph_evaluate<PeerCarrier>(
              peer_failure_nodes.data(), peer_failure_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(peer_failure_trace.failure_index);
        });
#else
    bench.run("HYLE Status source TaskGraph owning failure", [&] {
      auto status = taskgraph_evaluate<HyleCarrier>(
          hyle_failure_nodes.data(), hyle_failure_nodes.size());
      nb::doNotOptimizeAway(status);
      nb::doNotOptimizeAway(hyle_failure_trace.failure_index);
    });
#endif
  }
}

void run_handoff_benchmarks([[maybe_unused]] bool peer_first) {
  {
    nb::Bench bench;
    bench.title("Source parallel first-error owning handoff")
        .relative(true)
        .batch(1)
        .minEpochIterations(250000);
#if defined(HYLE_BENCH_ABSL)
    run_pair(
        bench, peer_first, "HYLE Status source parallel owning handoff",
        [&] {
          auto status = parallel_first_error_handoff<HyleCarrier>();
          nb::doNotOptimizeAway(status);
        },
        "absl::Status source parallel owning handoff", [&] {
          auto status = parallel_first_error_handoff<PeerCarrier>();
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status source parallel owning handoff", [&] {
      auto status = parallel_first_error_handoff<HyleCarrier>();
      nb::doNotOptimizeAway(status);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Source JoinState failed owning handoff")
        .relative(true)
        .batch(1)
        .minEpochIterations(250000);
#if defined(HYLE_BENCH_ABSL)
    run_pair(
        bench, peer_first, "HYLE Result source JoinState owning failure",
        [&] {
          auto result = failed_join_handoff<HyleCarrier>();
          nb::doNotOptimizeAway(HyleCarrier::result_ok(result));
        },
        "absl::StatusOr source JoinState owning failure", [&] {
          auto result = failed_join_handoff<PeerCarrier>();
          nb::doNotOptimizeAway(PeerCarrier::result_ok(result));
        });
#else
    bench.run("HYLE Result source JoinState owning failure", [&] {
      auto result = failed_join_handoff<HyleCarrier>();
      nb::doNotOptimizeAway(HyleCarrier::result_ok(result));
    });
#endif
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    return validate_contract();
  }

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = rival_first_requested();
  std::fprintf(
      stderr,
      "# HYLE_ERROR_SOURCE_CONSUMER_RUN benchmark=error-source-consumer "
      "execution_order=%s apple_qos=%s qos_ok=%s affinity=external "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      hyle::bench::perf::run_nonce());

  run_taskgraph_benchmarks(peer_first);
  run_handoff_benchmarks(peer_first);
  return 0;
}
