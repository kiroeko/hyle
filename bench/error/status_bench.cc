// Information-free error carrier paths: OK, code-only Status, and Result
// success. Every comparison keeps the same information contract.
// Message/context/native-detail paths live in normal_bench.cc and compare
// against information-bearing Abseil shapes.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <string_view>
#include <system_error>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/status/statusor.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;

// A std::expected baseline error type. std::error_code is the closest
// std-native "cheap error carrier" (an int enum + a category pointer = 2
// words), so expected<T,error_code> is the fair "what the standard gives you"
// rival to the expected-shaped Result<T> and its checked, same-size
// ResultError composition.
using StdResult = std::expected<int, std::error_code>;

// Make the selected canonical code unknown to the benchmark body.  Every
// carrier pays for the same opaque selector call, and the returned code changes
// at runtime so construction cannot collapse to a compile-time constant.
HYLE_NOINLINE bool opaque_code_choice(std::uint32_t value) noexcept {
  return (value & 1U) != 0U;
}

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

// Four consumer call frames exercise Result. Keep every layer opaque for every
// rival: otherwise a compiler can inline only one carrier's smaller wrapper
// and make the call topology unequal.
HYLE_NOINLINE Result<int> hyle_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE Result<int> hyle_leaf_error(int x) {
  return x >= 0
             ? Result<int>(hyle::make_unexpected(Status(StatusCode::NotFound)))
             : Result<int>(x);
}

HYLE_NOINLINE Result<int> hyle_ok_layer1(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_leaf_ok(x));
  return value + 1;
}
HYLE_NOINLINE Result<int> hyle_ok_layer2(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_ok_layer1(x));
  return value + 1;
}
HYLE_NOINLINE Result<int> hyle_chain_ok(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_ok_layer2(x));
  return value + 1;
}
HYLE_NOINLINE Result<int> hyle_error_layer1(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_leaf_error(x));
  return value + 1;
}
HYLE_NOINLINE Result<int> hyle_error_layer2(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_error_layer1(x));
  return value + 1;
}
HYLE_NOINLINE Result<int> hyle_chain_error(int x) {
  HYLE_ASSIGN_OR_RETURN(int value, hyle_error_layer2(x));
  return value + 1;
}

// std::expected equivalents (manual check-and-return, the idiom without a `?`
// operator).
HYLE_NOINLINE StdResult std_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE StdResult std_leaf_error(int x) {
  return x >= 0 ? StdResult(std::unexpected(std::make_error_code(
                      std::errc::no_such_file_or_directory)))
                : StdResult(x);
}
HYLE_NOINLINE StdResult std_ok_layer1(int x) {
  auto result = std_leaf_ok(x);
  if (!result) return result;
  return *result + 1;
}
HYLE_NOINLINE StdResult std_ok_layer2(int x) {
  auto result = std_ok_layer1(x);
  if (!result) return result;
  return *result + 1;
}
HYLE_NOINLINE StdResult std_chain_ok(int x) {
  auto result = std_ok_layer2(x);
  if (!result) return result;
  return *result + 1;
}
HYLE_NOINLINE StdResult std_error_layer1(int x) {
  auto result = std_leaf_error(x);
  if (!result) return result;
  return *result + 1;
}
HYLE_NOINLINE StdResult std_error_layer2(int x) {
  auto result = std_error_layer1(x);
  if (!result) return result;
  return *result + 1;
}
HYLE_NOINLINE StdResult std_chain_error(int x) {
  auto result = std_error_layer2(x);
  if (!result) return result;
  return *result + 1;
}

#if defined(HYLE_BENCH_ABSL)
HYLE_NOINLINE absl::StatusOr<int> absl_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE absl::StatusOr<int> absl_leaf_error(int x) {
  return x >= 0 ? absl::StatusOr<int>(absl::Status(absl::StatusCode::kNotFound,
                                                   "", absl::SourceLocation()))
                : absl::StatusOr<int>(x);
}
HYLE_NOINLINE absl::StatusOr<int> absl_ok_layer1(int x) {
  auto result = absl_leaf_ok(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
HYLE_NOINLINE absl::StatusOr<int> absl_ok_layer2(int x) {
  auto result = absl_ok_layer1(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
HYLE_NOINLINE absl::StatusOr<int> absl_chain_ok(int x) {
  auto result = absl_ok_layer2(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
HYLE_NOINLINE absl::StatusOr<int> absl_error_layer1(int x) {
  auto result = absl_leaf_error(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
HYLE_NOINLINE absl::StatusOr<int> absl_error_layer2(int x) {
  auto result = absl_error_layer1(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
HYLE_NOINLINE absl::StatusOr<int> absl_chain_error(int x) {
  auto result = absl_error_layer2(x);
  if (!result.ok()) return result.status();
  return *result + 1;
}
#endif

}  // namespace

int main() {
  int seed = 0;
  const bool rival_first = rival_first_requested();
  std::fprintf(stderr, "# HYLE_ERROR_BENCH order=%s\n",
               rival_first ? "peer-first" : "candidate-first");

  // ---- (1) OK construct + ok() check: THE hot path (every success returns
  // this) ----
  {
    nb::Bench b;
    b.title("OK status: construct + ok() check")
        .relative(true)
        .batch(1)
        .minEpochIterations(5000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first, "HYLE Status::Ok",
        [&] {
          Status s;  // rep_ = 0
          nb::doNotOptimizeAway(s.ok());
        },
        "absl::OkStatus", [&] {
          absl::Status s = absl::OkStatus();
          nb::doNotOptimizeAway(s.ok());
        });
#else
    b.run("HYLE Status::Ok", [&] {
      Status s;  // rep_ = 0
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("std::expected<int,ec> (success)", [&] {
      StdResult r = ++seed;
      nb::doNotOptimizeAway(r.has_value());
    });
  }

  // ---- (2) OK copy: propagating a success Status up the stack ----
  {
    Status ok_h;
#if defined(HYLE_BENCH_ABSL)
    absl::Status ok_a = absl::OkStatus();
#endif
    nb::Bench b;
    b.title("OK status: copy")
        .relative(true)
        .batch(1)
        .minEpochIterations(5000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first, "HYLE Status copy (OK)",
        [&] {
          Status c = ok_h;  // word copy, is_heap()==false -> no retain
          nb::doNotOptimizeAway(c.ok());
        },
        "absl::Status copy (OK)", [&] {
          absl::Status c = ok_a;
          nb::doNotOptimizeAway(c.ok());
        });
#else
    b.run("HYLE Status copy (OK)", [&] {
      Status c = ok_h;  // word copy, is_heap()==false -> no retain
      nb::doNotOptimizeAway(c.ok());
    });
#endif
  }

  // ---- (3) code-only construction from a runtime-selected canonical code.
  // The whole carrier escapes, rather than only querying a property that the
  // compiler can derive from a constant constructor argument.
  {
    std::uint32_t hyle_runtime_code_seed = 0;
    std::uint32_t std_runtime_code_seed = 0;
#if defined(HYLE_BENCH_ABSL)
    std::uint32_t absl_runtime_code_seed = 0;
#endif
    nb::Bench b;
    b.title("Code-only error construct from opaque runtime canonical code")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first,
        "HYLE Status runtime code, no message (carrier escapes)",
        [&] {
          const bool use_not_found =
              opaque_code_choice(++hyle_runtime_code_seed);
          Status status(use_not_found ? StatusCode::NotFound
                                      : StatusCode::InvalidArgument);
          nb::doNotOptimizeAway(status);
        },
        "absl::Status runtime code, empty, no-loc (carrier escapes)", [&] {
          const bool use_not_found =
              opaque_code_choice(++absl_runtime_code_seed);
          absl::Status status(
              use_not_found ? absl::StatusCode::kNotFound
                            : absl::StatusCode::kInvalidArgument,
              "", absl::SourceLocation());
          nb::doNotOptimizeAway(status);
        });
#else
    b.run("HYLE Status runtime code, no message (carrier escapes)", [&] {
      const bool use_not_found =
          opaque_code_choice(++hyle_runtime_code_seed);
      Status status(use_not_found ? StatusCode::NotFound
                                  : StatusCode::InvalidArgument);
      nb::doNotOptimizeAway(status);
    });
#endif
    b.run("std::error_code runtime errc (carrier escapes)", [&] {
      const bool use_not_found =
          opaque_code_choice(++std_runtime_code_seed);
      const std::error_code error = std::make_error_code(
          use_not_found ? std::errc::no_such_file_or_directory
                        : std::errc::invalid_argument);
      nb::doNotOptimizeAway(error);
    });
  }

  // Common public factory shape.  Abseil's no-argument convenience factory is
  // the appropriate code-only API peer; escaping the carrier prevents either
  // implementation from reducing the whole workload to a constant predicate.
  {
    nb::Bench b;
    b.title("Cancelled code-only factory + code query (carrier escapes)")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first,
        "HYLE Status::Cancelled() + code() (carrier escapes)",
        [&] {
          Status status = Status::Cancelled();
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(status.code());
        },
        "absl::CancelledError() + code() (carrier escapes)", [&] {
          absl::Status status = absl::CancelledError();
          nb::doNotOptimizeAway(status);
          nb::doNotOptimizeAway(status.code());
        });
#else
    b.run("HYLE Status::Cancelled() + code() (carrier escapes)", [&] {
      Status status = Status::Cancelled();
      nb::doNotOptimizeAway(status);
      nb::doNotOptimizeAway(status.code());
    });
#endif
    b.run("std::make_error_code(operation_canceled) (carrier escapes)", [&] {
      const std::error_code error =
          std::make_error_code(std::errc::operation_canceled);
      nb::doNotOptimizeAway(error);
      nb::doNotOptimizeAway(error.value());
    });
  }

  // Success propagation through four real call frames, followed by normal
  // caller consumption of the value.
  {
    std::uint64_t acc_hyle = 0, acc_std = 0;
#if defined(HYLE_BENCH_ABSL)
    std::uint64_t acc_absl = 0;
#endif
    nb::Bench b;
    b.title("Result<int> propagate OK through 4 frames + consume")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first, "HYLE Result OK (4 noinline frames + unwrap)",
        [&] {
          auto r = hyle_chain_ok(++seed);
          acc_hyle += r.value_or(0);
          nb::doNotOptimizeAway(acc_hyle);
        },
        "absl::StatusOr OK (4 noinline frames + unwrap)", [&] {
          auto r = absl_chain_ok(++seed);
          acc_absl += r.ok() ? static_cast<std::uint64_t>(*r) : 0;
          nb::doNotOptimizeAway(acc_absl);
        });
#else
    b.run("HYLE Result OK (4 noinline frames + unwrap)", [&] {
      auto r = hyle_chain_ok(++seed);
      acc_hyle += r.value_or(0);
      nb::doNotOptimizeAway(acc_hyle);
    });
#endif
    b.run("std::expected OK (4 noinline frames + unwrap)", [&] {
      auto r = std_chain_ok(++seed);
      acc_std += r.value_or(0);
      nb::doNotOptimizeAway(acc_std);
    });
  }

  // Failure propagation with code only: no message, context, payload, or
  // allocation.
  {
    int value = 1;
    nb::Bench b;
    b.title("Result<int> propagate code-only error through 4 noinline frames")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        b, rival_first, "HYLE Result code-only error (4 noinline frames)",
        [&] {
          auto result = hyle_chain_error(value++);
          nb::doNotOptimizeAway(result.has_value());
        },
        "absl::StatusOr code-only error (4 noinline frames)", [&] {
          auto result = absl_chain_error(value++);
          nb::doNotOptimizeAway(result.ok());
        });
#else
    b.run("HYLE Result code-only error (4 noinline frames)", [&] {
      auto result = hyle_chain_error(value++);
      nb::doNotOptimizeAway(result.has_value());
    });
#endif
    b.run("std::expected<error_code> error (4 noinline frames)", [&] {
      auto result = std_chain_error(value++);
      nb::doNotOptimizeAway(result.has_value());
    });
  }

  // ---- space dimension: sizes printed so the table's space column is real,
  // not claimed.
  std::printf("\n=== error carrier SIZES (space dimension) ===\n");
  std::printf(
      "  sizeof(hyle::Status)               = %zu bytes (one tagged word)\n",
      sizeof(Status));
  std::printf("  sizeof(hyle::Result<int>)          = %zu bytes\n",
              sizeof(Result<int>));
  std::printf("  sizeof(hyle::Result<void>)         = %zu bytes\n",
              sizeof(Result<void>));
  std::printf("  sizeof(std::expected<int,errc>)    = %zu bytes\n",
              sizeof(StdResult));
  std::printf("  sizeof(std::error_code)            = %zu bytes\n",
              sizeof(std::error_code));
#if defined(HYLE_BENCH_ABSL)
  std::printf("  sizeof(absl::Status)               = %zu bytes\n",
              sizeof(absl::Status));
  std::printf("  sizeof(absl::StatusOr<int>)        = %zu bytes\n",
              sizeof(absl::StatusOr<int>));
#endif
  return 0;
}
