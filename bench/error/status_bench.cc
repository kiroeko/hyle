// Information-free error carrier paths: OK, code-only Status, and Result
// success. Every comparison keeps the same information contract.
// Message/context/native-detail paths live in normal_bench.cc and compare
// against information-bearing Abseil shapes.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <system_error>

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
// rival to Result<T> = expected<T, Status>.
using StdResult = std::expected<int, std::error_code>;

// Four consumer call frames exercise Result. Keep both the leaf and outer chain
// opaque for every rival: otherwise GCC inlines only the smaller std/Abseil
// wrapper and charges HYLE an extra call boundary.
HYLE_NOINLINE Result<int> hyle_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE Result<int> hyle_leaf_error(int x) {
  return x >= 0
             ? Result<int>(hyle::make_unexpected(Status(StatusCode::NotFound)))
             : Result<int>(x);
}

HYLE_NOINLINE Result<int> hyle_chain_ok(int x) {
  HYLE_ASSIGN_OR_RETURN(int a, hyle_leaf_ok(x));
  HYLE_ASSIGN_OR_RETURN(int b, hyle_leaf_ok(a));
  HYLE_ASSIGN_OR_RETURN(int c, hyle_leaf_ok(b));
  return c;
}
HYLE_NOINLINE Result<int> hyle_chain_error(int x) {
  HYLE_ASSIGN_OR_RETURN(int result, hyle_leaf_error(x));
  return result;
}

// std::expected equivalents (manual check-and-return, the idiom without a `?`
// operator).
HYLE_NOINLINE StdResult std_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE StdResult std_leaf_error(int x) {
  return x >= 0 ? StdResult(std::unexpected(std::make_error_code(
                      std::errc::no_such_file_or_directory)))
                : StdResult(x);
}
HYLE_NOINLINE StdResult std_chain_ok(int x) {
  auto a = std_leaf_ok(x);
  if (!a) return a;
  auto b = std_leaf_ok(*a);
  if (!b) return b;
  auto c = std_leaf_ok(*b);
  if (!c) return c;
  return *c;
}
HYLE_NOINLINE StdResult std_chain_error(int x) {
  auto result = std_leaf_error(x);
  if (!result) return result;
  return *result;
}

#if defined(HYLE_BENCH_ABSL)
HYLE_NOINLINE absl::StatusOr<int> absl_leaf_ok(int x) { return x + 1; }
HYLE_NOINLINE absl::StatusOr<int> absl_leaf_error(int x) {
  return x >= 0 ? absl::StatusOr<int>(absl::Status(absl::StatusCode::kNotFound,
                                                   "", absl::SourceLocation()))
                : absl::StatusOr<int>(x);
}
HYLE_NOINLINE absl::StatusOr<int> absl_chain_ok(int x) {
  auto a = absl_leaf_ok(x);
  if (!a.ok()) return a.status();
  auto b = absl_leaf_ok(*a);
  if (!b.ok()) return b.status();
  auto c = absl_leaf_ok(*b);
  if (!c.ok()) return c.status();
  return *c;
}
HYLE_NOINLINE absl::StatusOr<int> absl_chain_error(int x) {
  auto result = absl_leaf_error(x);
  if (!result.ok()) return result.status();
  return *result;
}
#endif

}  // namespace

int main() {
  int seed = 0;

  // ---- (1) OK construct + ok() check: THE hot path (every success returns
  // this) ----
  {
    nb::Bench b;
    b.title("OK status: construct + ok() check")
        .relative(true)
        .batch(1)
        .minEpochIterations(5000000);
    b.run("std::expected<int,ec> (success)", [&] {
      StdResult r = ++seed;
      nb::doNotOptimizeAway(r.has_value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::OkStatus", [&] {
      absl::Status s = absl::OkStatus();
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("HYLE Status::Ok", [&] {
      Status s;  // rep_ = 0
      nb::doNotOptimizeAway(s.ok());
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
    b.run("absl::Status copy (OK)", [&] {
      absl::Status c = ok_a;
      nb::doNotOptimizeAway(c.ok());
    });
#endif
    b.run("HYLE Status copy (OK)", [&] {
      Status c = ok_h;  // word copy, is_heap()==false -> no retain
      nb::doNotOptimizeAway(c.ok());
    });
  }

  // ---- (3) code-only error construct: both mature carriers inline this shape
  // ----
  {
    nb::Bench b;
    b.title("error construct (code only, no message)")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    b.run("std::error_code(errc)", [&] {
      const std::error_code error =
          std::make_error_code(std::errc::invalid_argument);
      nb::doNotOptimizeAway(error.value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::Status(code,empty,no-loc)", [&] {
      absl::Status s(absl::StatusCode::kInvalidArgument, "",
                     absl::SourceLocation());
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("HYLE Status(code) inline", [&] {
      Status s(StatusCode::InvalidArgument);  // (code<<2)|1, zero heap
      nb::doNotOptimizeAway(s.ok());
    });
  }

  // Success propagation through a 4-deep consumer call chain.
  {
    nb::Bench b;
    b.title("Result<int> propagate OK through 4 frames")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    b.run("std::expected (manual check)", [&] {
      auto r = std_chain_ok(++seed);
      nb::doNotOptimizeAway(r.has_value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::StatusOr (manual check)", [&] {
      auto r = absl_chain_ok(++seed);
      nb::doNotOptimizeAway(r.ok());
    });
#endif
    b.run("HYLE Result + ASSIGN_OR_RETURN", [&] {
      auto r = hyle_chain_ok(++seed);
      nb::doNotOptimizeAway(r.has_value());
    });
  }

  // Success propagation with the value consumed, as it is in normal caller
  // code.
  {
    std::uint64_t acc_hyle = 0, acc_std = 0;
    nb::Bench b;
    b.title("Result<int> propagate OK + consume value")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    b.run("std::expected (unwrap + use)", [&] {
      auto r = std_chain_ok(++seed);
      acc_std += r.value_or(0);
      nb::doNotOptimizeAway(acc_std);
    });
    b.run("HYLE Result (unwrap + use)", [&] {
      auto r = hyle_chain_ok(++seed);
      acc_hyle += r.value_or(0);
      nb::doNotOptimizeAway(acc_hyle);
    });
  }

  // Failure propagation with code only: no message, context, payload, or
  // allocation.
  {
    int value = 1;
    nb::Bench b;
    b.title("Result<int> propagate code-only error")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("std::expected<error_code> error", [&] {
      auto result = std_chain_error(value++);
      nb::doNotOptimizeAway(result.has_value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::StatusOr code-only error", [&] {
      auto result = absl_chain_error(value++);
      nb::doNotOptimizeAway(result.ok());
    });
#endif
    b.run("HYLE Result code-only error", [&] {
      auto result = hyle_chain_error(value++);
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
