// Qualified error P1 workflows.  This executable intentionally contains only
// the frozen normal/replay rows from error-p1-v1.json.  Primitive and boundary
// diagnostics remain in status_bench.cc and normal_bench.cc, but are not paid
// for by every qualification child.
#include <nanobench.h>

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/format.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#include "../common/perf_process.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/status/status_builder.h"
#  include "absl/status/statusor.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;

constexpr std::size_t kGraphNodes = 64;
constexpr std::size_t kLastGraphNode = kGraphNodes - 1;
constexpr std::uint64_t kGraphChecksum =
    (kGraphNodes * (kGraphNodes + 1)) / 2;
constexpr std::string_view kMediumMessage =
    "resource not found in shard 7";
constexpr std::string_view kShortFormatMessage =
    "vulkan vkQueueSubmit failed: VK_ERROR_DEVICE_LOST (-4)";
static_assert(kMediumMessage.size() == 29);
static_assert(kShortFormatMessage.size() == 54);

struct GraphTrace {
  std::uint64_t calls = 0;
  std::uint64_t checksum = 0;
  std::size_t failure_index = kGraphNodes;
};

struct GraphNodeContext {
  GraphTrace *trace = nullptr;
  std::size_t index = 0;
};

using HyleGraphFn = Status (*)(GraphNodeContext *) noexcept;
struct HyleGraphNode {
  HyleGraphFn function = nullptr;
  GraphNodeContext *context = nullptr;
};

HYLE_NOINLINE Status hyle_graph_success_node(
    GraphNodeContext *context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  return Status{};
}

HYLE_NOINLINE Status hyle_graph_error_node(
    GraphNodeContext *context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  context->trace->failure_index = context->index;
  return Status(StatusCode::NotFound);
}

// This is the error-facing hot loop extracted from TaskGraph::evaluate: one
// active-node stream, indirect ComputeFn calls, and immediate value-less Status
// propagation.  It is noinline and accepts runtime node storage, so the
// compiler cannot specialize the 64 calls into benchmark-only direct calls.
HYLE_NOINLINE Status hyle_taskgraph_evaluate_replay(
    const HyleGraphNode *nodes, std::size_t size) noexcept {
  for (std::size_t index = 0; index < size; ++index) {
    Status status = nodes[index].function(nodes[index].context);
    HYLE_RETURN_IF_ERROR(status);
  }
  return Status{};
}

#if defined(HYLE_BENCH_ABSL)
using PeerGraphFn = absl::Status (*)(GraphNodeContext *) noexcept;
struct PeerGraphNode {
  PeerGraphFn function = nullptr;
  GraphNodeContext *context = nullptr;
};

HYLE_NOINLINE absl::Status peer_graph_success_node(
    GraphNodeContext *context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  return absl::OkStatus();
}

HYLE_NOINLINE absl::Status peer_graph_error_node(
    GraphNodeContext *context) noexcept {
  ++context->trace->calls;
  context->trace->checksum += context->index + 1;
  context->trace->failure_index = context->index;
  return absl::Status(absl::StatusCode::kNotFound, "",
                      absl::SourceLocation());
}

HYLE_NOINLINE absl::Status peer_taskgraph_evaluate_replay(
    const PeerGraphNode *nodes, std::size_t size) noexcept {
  for (std::size_t index = 0; index < size; ++index) {
    absl::Status status = nodes[index].function(nodes[index].context);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}
#endif

template <typename Node, typename SuccessFn, typename ErrorFn>
void initialize_graph_nodes(std::array<Node, kGraphNodes> &nodes,
                            std::array<GraphNodeContext, kGraphNodes> &contexts,
                            GraphTrace &trace, SuccessFn success,
                            ErrorFn error, bool fail_last) noexcept {
  for (std::size_t index = 0; index < kGraphNodes; ++index) {
    contexts[index] = GraphNodeContext{&trace, index};
    nodes[index] = Node{fail_last && index == kLastGraphNode ? error : success,
                        &contexts[index]};
  }
}

const Status &hyle_prebuilt_error() {
  static const Status status = Status::NotFound("resource not found");
  return status;
}

HYLE_NOINLINE Result<int> hyle_info_leaf(int value) {
  return value >= 0 ? Result<int>(hyle::make_unexpected(hyle_prebuilt_error()))
                    : Result<int>(value);
}

HYLE_NOINLINE Result<int> hyle_info_chain(int value) {
  HYLE_ASSIGN_OR_RETURN(int result, hyle_info_leaf(value));
  return result;
}

HYLE_NOINLINE Status hyle_update_first_prvalue() {
  return Status::NotFound("first");
}

HYLE_NOINLINE Status hyle_update_second_prvalue() {
  return Status::Internal("second");
}

#if defined(HYLE_BENCH_ABSL)
const absl::Status &peer_prebuilt_error() {
  static const absl::Status status(absl::StatusCode::kNotFound,
                                   "resource not found",
                                   absl::SourceLocation());
  return status;
}

HYLE_NOINLINE absl::StatusOr<int> peer_info_leaf(int value) {
  return value >= 0 ? absl::StatusOr<int>(peer_prebuilt_error())
                    : absl::StatusOr<int>(value);
}

HYLE_NOINLINE absl::StatusOr<int> peer_info_chain(int value) {
  auto result = peer_info_leaf(value);
  if (!result.ok()) return result.status();
  return *result;
}

HYLE_NOINLINE absl::Status peer_update_first_prvalue() {
  return absl::Status(absl::StatusCode::kNotFound, "first",
                      absl::SourceLocation());
}

HYLE_NOINLINE absl::Status peer_update_second_prvalue() {
  return absl::Status(absl::StatusCode::kInternal, "second",
                      absl::SourceLocation());
}
#endif

bool rival_first_requested() {
#if defined(_WIN32)
  char *value = nullptr;
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

template <typename CandidateOp, typename RivalOp>
void run_candidate_peer_pair(nb::Bench &bench, bool rival_first,
                             std::string_view candidate_name,
                             CandidateOp &&candidate,
                             std::string_view rival_name, RivalOp &&rival) {
  if (rival_first) {
    bench.run(rival_name, std::forward<RivalOp>(rival));
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
  } else {
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
    bench.run(rival_name, std::forward<RivalOp>(rival));
  }
}

template <typename Function>
std::uintptr_t function_address(Function function) noexcept {
  return reinterpret_cast<std::uintptr_t>(function);
}

#if defined(HYLE_BENCH_ABSL)
bool same_view(absl::string_view left, std::string_view right) {
  return left.size() == right.size() &&
         std::string_view(left.data(), left.size()) == right;
}

bool has_suffix(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}
#endif

int validate_contract() {
  GraphTrace hyle_success_trace;
  GraphTrace hyle_error_trace;
  std::array<GraphNodeContext, kGraphNodes> hyle_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> hyle_error_contexts{};
  std::array<HyleGraphNode, kGraphNodes> hyle_success_nodes{};
  std::array<HyleGraphNode, kGraphNodes> hyle_error_nodes{};
  initialize_graph_nodes(hyle_success_nodes, hyle_success_contexts,
                         hyle_success_trace, hyle_graph_success_node,
                         hyle_graph_error_node, false);
  initialize_graph_nodes(hyle_error_nodes, hyle_error_contexts,
                         hyle_error_trace, hyle_graph_success_node,
                         hyle_graph_error_node, true);

  const Status hyle_success = hyle_taskgraph_evaluate_replay(
      hyle_success_nodes.data(), hyle_success_nodes.size());
  const Status hyle_error = hyle_taskgraph_evaluate_replay(
      hyle_error_nodes.data(), hyle_error_nodes.size());
  bool graph_ok = hyle_success.ok() && hyle_error.IsNotFound() &&
                  hyle_success_trace.calls == kGraphNodes &&
                  hyle_error_trace.calls == kGraphNodes &&
                  hyle_success_trace.checksum == kGraphChecksum &&
                  hyle_error_trace.checksum == kGraphChecksum &&
                  hyle_error_trace.failure_index == kLastGraphNode;

  bool message_ok = false;
  bool context_ok = false;
  bool result_ok = false;
  bool format_ok = false;
  bool update_ok = false;
  bool source_location_ok = false;
  std::uint64_t peer_success_calls = 0;
  std::uint64_t peer_error_calls = 0;
  std::uint64_t peer_success_checksum = 0;
  std::uint64_t peer_error_checksum = 0;
  std::size_t peer_failure_index = kGraphNodes;

#if defined(HYLE_BENCH_ABSL)
  GraphTrace peer_success_trace;
  GraphTrace peer_error_trace;
  std::array<GraphNodeContext, kGraphNodes> peer_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> peer_error_contexts{};
  std::array<PeerGraphNode, kGraphNodes> peer_success_nodes{};
  std::array<PeerGraphNode, kGraphNodes> peer_error_nodes{};
  initialize_graph_nodes(peer_success_nodes, peer_success_contexts,
                         peer_success_trace, peer_graph_success_node,
                         peer_graph_error_node, false);
  initialize_graph_nodes(peer_error_nodes, peer_error_contexts,
                         peer_error_trace, peer_graph_success_node,
                         peer_graph_error_node, true);
  const absl::Status peer_success = peer_taskgraph_evaluate_replay(
      peer_success_nodes.data(), peer_success_nodes.size());
  const absl::Status peer_error = peer_taskgraph_evaluate_replay(
      peer_error_nodes.data(), peer_error_nodes.size());
  peer_success_calls = peer_success_trace.calls;
  peer_error_calls = peer_error_trace.calls;
  peer_success_checksum = peer_success_trace.checksum;
  peer_error_checksum = peer_error_trace.checksum;
  peer_failure_index = peer_error_trace.failure_index;
  graph_ok = graph_ok && peer_success.ok() &&
             peer_error.code() == absl::StatusCode::kNotFound &&
             peer_success_calls == kGraphNodes &&
             peer_error_calls == kGraphNodes &&
             peer_success_checksum == kGraphChecksum &&
             peer_error_checksum == kGraphChecksum &&
             peer_failure_index == kLastGraphNode;

  const Status hyle_message = Status::NotFound(kMediumMessage);
  const absl::Status peer_message(absl::StatusCode::kNotFound, kMediumMessage,
                                  absl::SourceLocation());
  const Status hyle_message_copy = hyle_message;
  const absl::Status peer_message_copy = peer_message;
  message_ok = hyle_message.IsNotFound() &&
               hyle_message.message() == kMediumMessage &&
               hyle_message_copy == hyle_message &&
               peer_message.code() == absl::StatusCode::kNotFound &&
               same_view(peer_message.message(), kMediumMessage) &&
               peer_message_copy == peer_message;

  constexpr std::string_view kExpectedContext =
      "serve request: execute graph: dispatch batch: device queue is "
      "unavailable";
  Status hyle_context = Status::Unavailable("device queue is unavailable")
                            .WithContext("dispatch batch")
                            .WithContext("execute graph")
                            .WithContext("serve request");
  absl::Status peer_context(absl::StatusCode::kUnavailable,
                            "device queue is unavailable",
                            absl::SourceLocation());
  peer_context =
      absl::StatusBuilder(std::move(peer_context), absl::SourceLocation())
          .SetPrepend()
      << "dispatch batch: ";
  peer_context =
      absl::StatusBuilder(std::move(peer_context), absl::SourceLocation())
          .SetPrepend()
      << "execute graph: ";
  peer_context =
      absl::StatusBuilder(std::move(peer_context), absl::SourceLocation())
          .SetPrepend()
      << "serve request: ";
  context_ok = hyle_context.message() == kExpectedContext &&
               same_view(peer_context.message(), kExpectedContext);

  const auto hyle_result = hyle_info_chain(1);
  const auto peer_result = peer_info_chain(1);
  result_ok = !hyle_result.has_value() && !peer_result.ok() &&
              hyle_result.error().status().IsNotFound() &&
              hyle_result.error().status().message() == "resource not found" &&
              peer_result.status().code() == absl::StatusCode::kNotFound &&
              same_view(peer_result.status().message(), "resource not found");

  const Status hyle_format_status =
      Status::Unavailable(kShortFormatMessage);
  const absl::Status peer_format_status(absl::StatusCode::kUnavailable,
                                        kShortFormatMessage,
                                        absl::SourceLocation());
  const std::string hyle_text = hyle::to_string(hyle_format_status);
  const std::string peer_text = peer_format_status.ToString();
  format_ok = hyle_text.size() == peer_text.size() &&
              has_suffix(hyle_text, kShortFormatMessage) &&
              has_suffix(peer_text, kShortFormatMessage);

  Status hyle_updated;
  hyle_updated.Update(hyle_update_first_prvalue());
  hyle_updated.Update(hyle_update_second_prvalue());
  absl::Status peer_updated;
  peer_updated.Update(peer_update_first_prvalue());
  peer_updated.Update(peer_update_second_prvalue());
  update_ok = hyle_updated.IsNotFound() && hyle_updated.message() == "first" &&
              peer_updated.code() == absl::StatusCode::kNotFound &&
              same_view(peer_updated.message(), "first");

  source_location_ok = peer_message.GetSourceLocations().empty() &&
                       peer_context.GetSourceLocations().empty() &&
                       peer_format_status.GetSourceLocations().empty() &&
                       peer_updated.GetSourceLocations().empty();
#endif

  const bool passed = graph_ok && message_ok && context_ok && result_ok &&
                      format_ok && update_ok && source_location_ok;
  std::printf(
      "# HYLE_ERROR_P1_CONTRACT schema=1 nodes=%zu "
      "hyle_success_calls=%" PRIu64 " peer_success_calls=%" PRIu64 " "
      "hyle_error_calls=%" PRIu64 " peer_error_calls=%" PRIu64 " "
      "hyle_success_checksum=%" PRIu64 " peer_success_checksum=%" PRIu64
      " hyle_error_checksum=%" PRIu64 " peer_error_checksum=%" PRIu64 " "
      "hyle_failure_index=%zu peer_failure_index=%zu graph=%s message=%s "
      "context=%s result=%s format=%s update=%s source_location=%s "
      "verdict=%s\n",
      kGraphNodes, hyle_success_trace.calls, peer_success_calls,
      hyle_error_trace.calls, peer_error_calls, hyle_success_trace.checksum,
      peer_success_checksum, hyle_error_trace.checksum, peer_error_checksum,
      hyle_error_trace.failure_index, peer_failure_index,
      graph_ok ? "PASS" : "FAIL", message_ok ? "PASS" : "FAIL",
      context_ok ? "PASS" : "FAIL", result_ok ? "PASS" : "FAIL",
      format_ok ? "PASS" : "FAIL", update_ok ? "PASS" : "FAIL",
      source_location_ok ? "PASS" : "FAIL", passed ? "PASS" : "FAIL");

  std::printf(
      "# HYLE_ERROR_P1_LAYOUT hyle_eval=%" PRIxPTR
      " hyle_success_node=%" PRIxPTR " hyle_error_node=%" PRIxPTR
      " hyle_info_leaf=%" PRIxPTR " hyle_info_chain=%" PRIxPTR
#if defined(HYLE_BENCH_ABSL)
      " peer_eval=%" PRIxPTR " peer_success_node=%" PRIxPTR
      " peer_error_node=%" PRIxPTR " peer_info_leaf=%" PRIxPTR
      " peer_info_chain=%" PRIxPTR
#endif
      "\n",
      function_address(hyle_taskgraph_evaluate_replay),
      function_address(hyle_graph_success_node),
      function_address(hyle_graph_error_node), function_address(hyle_info_leaf),
      function_address(hyle_info_chain)
#if defined(HYLE_BENCH_ABSL)
          ,
      function_address(peer_taskgraph_evaluate_replay),
      function_address(peer_graph_success_node),
      function_address(peer_graph_error_node), function_address(peer_info_leaf),
      function_address(peer_info_chain)
#endif
  );
  return passed ? 0 : 2;
}

void run_taskgraph_benchmarks([[maybe_unused]] bool rival_first) {
  GraphTrace hyle_success_trace;
  GraphTrace hyle_error_trace;
  std::array<GraphNodeContext, kGraphNodes> hyle_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> hyle_error_contexts{};
  std::array<HyleGraphNode, kGraphNodes> hyle_success_nodes{};
  std::array<HyleGraphNode, kGraphNodes> hyle_error_nodes{};
  initialize_graph_nodes(hyle_success_nodes, hyle_success_contexts,
                         hyle_success_trace, hyle_graph_success_node,
                         hyle_graph_error_node, false);
  initialize_graph_nodes(hyle_error_nodes, hyle_error_contexts,
                         hyle_error_trace, hyle_graph_success_node,
                         hyle_graph_error_node, true);

#if defined(HYLE_BENCH_ABSL)
  GraphTrace peer_success_trace;
  GraphTrace peer_error_trace;
  std::array<GraphNodeContext, kGraphNodes> peer_success_contexts{};
  std::array<GraphNodeContext, kGraphNodes> peer_error_contexts{};
  std::array<PeerGraphNode, kGraphNodes> peer_success_nodes{};
  std::array<PeerGraphNode, kGraphNodes> peer_error_nodes{};
  initialize_graph_nodes(peer_success_nodes, peer_success_contexts,
                         peer_success_trace, peer_graph_success_node,
                         peer_graph_error_node, false);
  initialize_graph_nodes(peer_error_nodes, peer_error_contexts,
                         peer_error_trace, peer_graph_success_node,
                         peer_graph_error_node, true);
#endif

  {
    nb::Bench bench;
    bench.title("TaskGraph evaluate 64 active function-pointer nodes success")
        .relative(true)
        .batch(1)
        .minEpochIterations(100000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first, "HYLE Status TaskGraph active stream success",
        [&] {
          Status status = hyle_taskgraph_evaluate_replay(
              hyle_success_nodes.data(), hyle_success_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(hyle_success_trace.calls);
          nb::doNotOptimizeAway(hyle_success_trace.checksum);
        },
        "absl::Status TaskGraph active stream success", [&] {
          absl::Status status = peer_taskgraph_evaluate_replay(
              peer_success_nodes.data(), peer_success_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(peer_success_trace.calls);
          nb::doNotOptimizeAway(peer_success_trace.checksum);
        });
#else
    bench.run("HYLE Status TaskGraph active stream success", [&] {
      Status status = hyle_taskgraph_evaluate_replay(
          hyle_success_nodes.data(), hyle_success_nodes.size());
      nb::doNotOptimizeAway(status);
      nb::doNotOptimizeAway(hyle_success_trace.calls);
      nb::doNotOptimizeAway(hyle_success_trace.checksum);
    });
#endif
  }

  {
    nb::Bench bench;
    bench
        .title(
            "TaskGraph evaluate 64 active function-pointer nodes last error")
        .relative(true)
        .batch(1)
        .minEpochIterations(100000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status TaskGraph active stream last code-only error",
        [&] {
          Status status = hyle_taskgraph_evaluate_replay(
              hyle_error_nodes.data(), hyle_error_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(hyle_error_trace.calls);
          nb::doNotOptimizeAway(hyle_error_trace.checksum);
          nb::doNotOptimizeAway(hyle_error_trace.failure_index);
        },
        "absl::Status TaskGraph active stream last code-only error", [&] {
          absl::Status status = peer_taskgraph_evaluate_replay(
              peer_error_nodes.data(), peer_error_nodes.size());
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(peer_error_trace.calls);
          nb::doNotOptimizeAway(peer_error_trace.checksum);
          nb::doNotOptimizeAway(peer_error_trace.failure_index);
        });
#else
    bench.run("HYLE Status TaskGraph active stream last code-only error", [&] {
      Status status = hyle_taskgraph_evaluate_replay(
          hyle_error_nodes.data(), hyle_error_nodes.size());
      nb::doNotOptimizeAway(status);
      nb::doNotOptimizeAway(hyle_error_trace.calls);
      nb::doNotOptimizeAway(hyle_error_trace.checksum);
      nb::doNotOptimizeAway(hyle_error_trace.failure_index);
    });
#endif
  }
}

void run_information_bearing_benchmarks([[maybe_unused]] bool rival_first) {
  {
    nb::Bench bench;
    bench.title("Status construct owning 29-byte message")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::NotFound owning 29-byte message (borrowed input)",
        [&] {
          Status status = Status::NotFound(kMediumMessage);
          nb::doNotOptimizeAway(status);
        },
        "absl::Status owning 29-byte message (borrowed input, no-loc)", [&] {
          absl::Status status(absl::StatusCode::kNotFound, kMediumMessage,
                              absl::SourceLocation());
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run(
        "HYLE Status::NotFound owning 29-byte message (borrowed input)", [&] {
          Status status = Status::NotFound(kMediumMessage);
          nb::doNotOptimizeAway(status);
        });
#endif
  }

  const Status hyle_status = Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
  const absl::Status peer_status(absl::StatusCode::kUnavailable,
                                 "device queue is unavailable",
                                 absl::SourceLocation());
#endif

  {
    const Status hyle_copy_source = Status::NotFound(kMediumMessage);
#if defined(HYLE_BENCH_ABSL)
    const absl::Status peer_copy_source(absl::StatusCode::kNotFound,
                                        kMediumMessage,
                                        absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Status copy sharing an owned 29-byte message")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status copy (shared owned 29-byte message)",
        [&] {
          Status copy = hyle_copy_source;
          nb::doNotOptimizeAway(copy);
        },
        "absl::Status copy (shared owned 29-byte message)", [&] {
          absl::Status copy = peer_copy_source;
          nb::doNotOptimizeAway(copy);
        });
#else
    bench.run("HYLE Status copy (shared owned 29-byte message)", [&] {
      Status copy = hyle_copy_source;
      nb::doNotOptimizeAway(copy);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Status prepend 1 context to a prebuilt message error")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::WithContext x1 (copied lvalue seed)",
        [&] {
          Status status = Status(hyle_status).WithContext("submit work");
          nb::doNotOptimizeAway(status);
        },
        "absl::StatusBuilder SetPrepend x1 (prebuilt lvalue seed)", [&] {
          absl::Status status =
              absl::StatusBuilder(peer_status, absl::SourceLocation())
                  .SetPrepend()
              << "submit work: ";
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status::WithContext x1 (copied lvalue seed)", [&] {
      Status status = Status(hyle_status).WithContext("submit work");
      nb::doNotOptimizeAway(status);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Status prepend 3 contexts to a prebuilt message error")
        .relative(true)
        .batch(1)
        .minEpochIterations(250000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::WithContext x3 (copied lvalue seed, rvalue chain)",
        [&] {
          Status status = Status(hyle_status)
                              .WithContext("dispatch batch")
                              .WithContext("execute graph")
                              .WithContext("serve request");
          nb::doNotOptimizeAway(status);
        },
        "absl::StatusBuilder SetPrepend x3 (prebuilt lvalue seed)", [&] {
          absl::Status status = peer_status;
          status =
              absl::StatusBuilder(std::move(status), absl::SourceLocation())
                  .SetPrepend()
              << "dispatch batch: ";
          status =
              absl::StatusBuilder(std::move(status), absl::SourceLocation())
                  .SetPrepend()
              << "execute graph: ";
          status =
              absl::StatusBuilder(std::move(status), absl::SourceLocation())
                  .SetPrepend()
              << "serve request: ";
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status::WithContext x3 (copied lvalue seed, rvalue chain)",
              [&] {
                Status status = Status(hyle_status)
                                    .WithContext("dispatch batch")
                                    .WithContext("execute graph")
                                    .WithContext("serve request");
                nb::doNotOptimizeAway(status);
              });
#endif
  }

  {
    int hyle_value = 1;
#if defined(HYLE_BENCH_ABSL)
    int peer_value = 1;
#endif
    nb::Bench bench;
    bench
        .title(
            "Result propagate prebuilt message error through 2 noinline frames")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first, "HYLE Result message error (2 noinline frames)",
        [&] {
          auto result = hyle_info_chain(hyle_value++);
          nb::doNotOptimizeAway(result.has_value());
        },
        "absl::StatusOr message error (2 noinline frames)", [&] {
          auto result = peer_info_chain(peer_value++);
          nb::doNotOptimizeAway(result.ok());
        });
#else
    bench.run("HYLE Result message error (2 noinline frames)", [&] {
      auto result = hyle_info_chain(hyle_value++);
      nb::doNotOptimizeAway(result.has_value());
    });
#endif
  }

  {
    const Status status = Status::Unavailable(kShortFormatMessage);
#if defined(HYLE_BENCH_ABSL)
    const absl::Status peer_format_status(absl::StatusCode::kUnavailable,
                                          kShortFormatMessage,
                                          absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Format 54-byte message to an owning string")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE hyle::to_string owning (54-byte message)",
        [&] {
          std::string text = hyle::to_string(status);
          nb::doNotOptimizeAway(text);
        },
        "absl::Status::ToString owning (54-byte message)", [&] {
          std::string text = peer_format_status.ToString();
          nb::doNotOptimizeAway(text);
        });
#else
    bench.run("HYLE hyle::to_string owning (54-byte message)", [&] {
      std::string text = hyle::to_string(status);
      nb::doNotOptimizeAway(text);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Status::Update keeps first of 2 fresh prvalue message errors")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::Update noinline factory prvalues x2",
        [&] {
          Status status;
          status.Update(hyle_update_first_prvalue());
          status.Update(hyle_update_second_prvalue());
          nb::doNotOptimizeAway(status);
        },
        "absl::Status::Update noinline factory prvalues x2", [&] {
          absl::Status status;
          status.Update(peer_update_first_prvalue());
          status.Update(peer_update_second_prvalue());
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status::Update noinline factory prvalues x2", [&] {
      Status status;
      status.Update(hyle_update_first_prvalue());
      status.Update(hyle_update_second_prvalue());
      nb::doNotOptimizeAway(status);
    });
#endif
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    return validate_contract();
  }

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool rival_first = rival_first_requested();
  std::fprintf(stderr,
               "# HYLE_ERROR_P1_RUN benchmark=error-p1 execution_order=%s "
               "apple_qos=%s qos_ok=%s affinity=external run_nonce=%s\n",
               rival_first ? "peer-first" : "candidate-first", qos.name,
               qos.ok, hyle::bench::perf::run_nonce());

  run_taskgraph_benchmarks(rival_first);
  run_information_bearing_benchmarks(rival_first);
  return 0;
}
