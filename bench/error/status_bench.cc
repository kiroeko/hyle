// error-module bench (zero-dependency root, previously UN-benchmarked despite being
// on every hot-path return). Status is an 8-byte tagged-pointer error carrier; Result<T>
// = std::expected<T, Status>. Both race the SOTA + std equivalents (win/lose printed):
//
//   * Status vs absl::Status (the SAME design: tagged pointer, canonical gRPC code set)
//     and std::expected<T,std::error_code>. abseil is gated (-DHYLE_BENCH_ABSL); without
//     it the std rivals still anchor the numbers.
//
// The hot path is the OK path: every successful call returns Status::Ok() / a value, so
// OK-construct + ok()-check + OK-copy must be ~free (HYLE: rep_==0, a single word). The
// error path splits two ways HYLE optimizes that absl does NOT: a message-LESS error is a
// pure inline code (NO heap), only a WITH-message error touches the heap (refcounted rep).
// absl::Status ALWAYS heap-allocates a rep (even code-only) -- so HYLE's inline-code path
// should win construct/copy on code-only errors. We measure all of it, honestly.
//
// Time/space both reported: the ns rows below + a sizeof block at the end (Status must stay
// one word; Result<T> = max(sizeof T, 8)+align). On a time/space conflict HYLE favors TIME.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "hyle/error/format.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

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

// Isolation probe: a TRIVIALLY-destructible error type (one int, no heap, no refcount).
// expected<int, TrivialErr> has a trivial dtor/move, exactly like expected<int,error_code>.
// If a chain over THIS matches std but the Status chain is slower, the gap is proven to be
// Status's NON-trivial dtor (the is_heap() branch on every scope exit / value plumbing),
// NOT the propagation idiom. That isolates "by-design refcount-carrier tax" from "macro tax".
struct TrivialErr {
  int code = 0;
};
using TrivResult = std::expected<int, TrivialErr>;
TrivResult triv_leaf_ok(int x) { return x + 1; }
TrivResult triv_chain_ok(int x) {
  auto a = triv_leaf_ok(x);
  if (!a) return a;
  auto b = triv_leaf_ok(*a);
  if (!b) return b;
  auto c = triv_leaf_ok(*b);
  if (!c) return c;
  return *c;
}

// A std::expected baseline error type. std::error_code is the closest std-native "cheap
// error carrier" (an int enum + a category pointer = 2 words), so expected<T,error_code>
// is the fair "what the standard gives you" rival to Result<T> = expected<T, Status>.
using StdResult = std::expected<int, std::error_code>;

// --- propagation depth: a 4-deep call chain, the realistic "error bubbles up N frames"
// pattern. Each layer forwards via ASSIGN_OR_RETURN (HYLE) so the macro + FailReturn
// conversion cost is in the measurement, not just a single construct. The ERROR-path leaf
// returns a COPY of a pre-built error (no per-call malloc) so the chain measures the
// PROPAGATION/unwind mechanism, not error construction (that's rows 3-4). Otherwise HYLE's
// per-call malloc(msg) vs std::errc's zero-alloc static code would be apples-to-oranges.
Result<int> hyle_leaf_ok(int x) { return x + 1; }
const Status& hyle_prebuilt_err() {
  static const Status e = Status::NotFound("missing");
  return e;
}
Result<int> hyle_leaf_err(int x) {
  // x is folded in so the preceding ok-chain isn't dead-code-eliminated (the error
  // still propagates regardless of x; this just keeps the prefix work real).
  return x >= 0 ? Result<int>(std::unexpected(hyle_prebuilt_err())) : Result<int>(x);
}

Result<int> hyle_chain_ok(int x) {
  HYLE_ASSIGN_OR_RETURN(int a, hyle_leaf_ok(x));
  HYLE_ASSIGN_OR_RETURN(int b, hyle_leaf_ok(a));
  HYLE_ASSIGN_OR_RETURN(int c, hyle_leaf_ok(b));
  return c;
}
Result<int> hyle_chain_err(int x) {
  HYLE_ASSIGN_OR_RETURN(int a, hyle_leaf_ok(x));
  HYLE_ASSIGN_OR_RETURN(int b, hyle_leaf_ok(a));
  HYLE_ASSIGN_OR_RETURN(int c, hyle_leaf_err(b));  // fails here, unwinds the error up
  return c;
}

Result<void> hyle_void_leaf(bool fail) {
  return fail ? Result<void>(hyle::make_unexpected(hyle_prebuilt_err()))
              : Result<void>();
}
Status hyle_void_chain(bool fail_first, bool fail_second) {
  HYLE_RETURN_IF_ERROR(hyle_void_leaf(fail_first));
  HYLE_RETURN_IF_ERROR(hyle_void_leaf(fail_second));
  return Status{};
}
Status hyle_void_chain_manual(bool fail_first, bool fail_second) {
  Result<void> first = hyle_void_leaf(fail_first);
  if (!first) return first.error();
  Result<void> second = hyle_void_leaf(fail_second);
  if (!second) return second.error();
  return Status{};
}

// std::expected equivalents (manual check-and-return, the idiom without a `?` operator).
StdResult std_leaf_ok(int x) { return x + 1; }
StdResult std_leaf_err(int x) {
  return x >= 0
             ? StdResult(std::unexpected(std::make_error_code(std::errc::no_such_file_or_directory)))
             : StdResult(x);
}
StdResult std_chain_ok(int x) {
  auto a = std_leaf_ok(x);
  if (!a) return a;
  auto b = std_leaf_ok(*a);
  if (!b) return b;
  auto c = std_leaf_ok(*b);
  if (!c) return c;
  return *c;
}
StdResult std_chain_err(int x) {
  auto a = std_leaf_ok(x);
  if (!a) return a;
  auto b = std_leaf_ok(*a);
  if (!b) return b;
  auto c = std_leaf_err(*b);
  if (!c) return c;
  return *c;
}

#if defined(HYLE_BENCH_ABSL)
absl::StatusOr<int> absl_leaf_ok(int x) { return x + 1; }
const absl::Status& absl_prebuilt_err() {
  static const absl::Status e = absl::NotFoundError("missing");
  return e;
}
absl::StatusOr<int> absl_leaf_err(int x) {
  return x >= 0 ? absl::StatusOr<int>(absl_prebuilt_err()) : absl::StatusOr<int>(x);
}
absl::StatusOr<int> absl_chain_ok(int x) {
  auto a = absl_leaf_ok(x);
  if (!a.ok()) return a.status();
  auto b = absl_leaf_ok(*a);
  if (!b.ok()) return b.status();
  auto c = absl_leaf_ok(*b);
  if (!c.ok()) return c.status();
  return *c;
}
absl::StatusOr<int> absl_chain_err(int x) {
  auto a = absl_leaf_ok(x);
  if (!a.ok()) return a.status();
  auto b = absl_leaf_ok(*a);
  if (!b.ok()) return b.status();
  auto c = absl_leaf_err(*b);
  if (!c.ok()) return c.status();
  return *c;
}
#endif

}  // namespace

int main() {
  int seed = 0;

  // ---- (1) OK construct + ok() check: THE hot path (every success returns this) ----
  {
    nb::Bench b;
    b.title("OK status: construct + ok() check").relative(true).batch(1).minEpochIterations(5000000);
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
    b.title("OK status: copy").relative(true).batch(1).minEpochIterations(5000000);
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

  // ---- (3) code-only error construct: HYLE inline (NO heap) vs absl (ALWAYS heap) ----
  {
    nb::Bench b;
    b.title("error construct (code only, no message)").relative(true).batch(1).minEpochIterations(2000000);
    b.run("std::unexpected<error_code>", [&] {
      StdResult r = std::unexpected(std::make_error_code(std::errc::invalid_argument));
      nb::doNotOptimizeAway(r.has_value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::InvalidArgumentError(\"\")", [&] {
      absl::Status s(absl::StatusCode::kInvalidArgument, "");  // absl still heap-allocs a rep
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("HYLE Status(code) inline", [&] {
      Status s(StatusCode::InvalidArgument);  // (code<<2)|1, zero heap
      nb::doNotOptimizeAway(s.ok());
    });
  }

  // ---- (4) with-message error construct: both heap-allocate a rep here ----
  {
    std::string_view msg = "resource not found in shard 7";
    nb::Bench b;
    b.title("error construct (with message -> heap)").relative(true).batch(1).minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::NotFoundError(msg)", [&] {
      absl::Status s = absl::NotFoundError(msg);
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("HYLE Status::NotFound(msg)", [&] {
      Status s = Status::NotFound(msg);  // single malloc(rep+len+1)
      nb::doNotOptimizeAway(s.ok());
    });
  }

  // ---- (5) heap-Status copy: refcount retain/release (both bump an atomic) ----
  {
    Status err_h = Status::NotFound("resource not found in shard 7");
#if defined(HYLE_BENCH_ABSL)
    absl::Status err_a = absl::NotFoundError("resource not found in shard 7");
#endif
    nb::Bench b;
    b.title("heap status: copy (retain+release)").relative(true).batch(1).minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::Status copy (error)", [&] {
      absl::Status c = err_a;  // refcount bump
      nb::doNotOptimizeAway(c.ok());
    });
#endif
    b.run("HYLE Status copy (error)", [&] {
      Status c = err_h;  // detail::status_retain (relaxed) + release on scope exit
      nb::doNotOptimizeAway(c.ok());
    });
  }

  // ---- (5b) add one semantic context layer to an existing error ----
  {
    Status err_h = Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
    absl::Status err_a = absl::UnavailableError("device queue is unavailable");
#endif
    nb::Bench b;
    b.title("error context: prefix one semantic layer")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::StatusBuilder SetPrepend", [&] {
      absl::Status s = absl::StatusBuilder(err_a).SetPrepend() << "submit work: ";
      nb::doNotOptimizeAway(s.ok());
    });
#endif
    b.run("HYLE Status::WithContext", [&] {
      Status s = Status(err_h).WithContext("submit work");
      nb::doNotOptimizeAway(s.ok());
    });
  }

  // The no-op path stays inline in consumer code; it must remain suitable for
  // unconditional contextualization at a boundary.
  {
    Status ok;
    nb::Bench b;
    b.title("WithContext on OK (no-op)")
        .relative(true)
        .batch(1)
        .minEpochIterations(5000000);
    b.run("HYLE Status::WithContext(OK)", [&] {
      Status s = ok.WithContext("submit work");
      nb::doNotOptimizeAway(s.ok());
    });
  }

  // ---- (6) success propagation through a 4-deep chain (macro + value plumbing) ----
  {
    nb::Bench b;
    b.title("Result<int> propagate OK through 4 frames").relative(true).batch(1).minEpochIterations(2000000);
    b.run("std::expected (manual check)", [&] {
      auto r = std_chain_ok(++seed);
      nb::doNotOptimizeAway(r.has_value());
    });
    b.run("expected<int,TrivialErr> (isolation: trivial dtor)", [&] {
      auto r = triv_chain_ok(++seed);
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

  // ---- (6b) success propagation, REALISTIC: the value is UNWRAPPED and consumed (the way
  // real code uses it: ASSIGN_OR_RETURN the result, then use the int). The (6) micro-pattern
  // builds a Result and immediately destroys it in the loop -- which over-weights the one
  // extra dtor branch that expected<int,Status> carries (Status is non-trivially destructible
  // so it can OWN a heap message; std::error_code is trivial so it can't). This variant folds
  // the unwrapped value into an accumulator, so the dtor interleaves with real work like it
  // would in production -- the honest "does the carrier cost show when you actually use it".
  {
    std::uint64_t acc_hyle = 0, acc_std = 0;
    nb::Bench b;
    b.title("Result<int> propagate OK + consume value").relative(true).batch(1).minEpochIterations(2000000);
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

  // ---- (7) error propagation: unwinding the error up the chain ----
  {
    nb::Bench b;
    b.title("Result<int> propagate ERROR up 3 frames").relative(true).batch(1).minEpochIterations(1000000);
    b.run("std::expected (manual check)", [&] {
      auto r = std_chain_err(++seed);
      nb::doNotOptimizeAway(r.has_value());
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::StatusOr (manual check)", [&] {
      auto r = absl_chain_err(++seed);
      nb::doNotOptimizeAway(r.ok());
    });
#endif
    b.run("HYLE Result + ASSIGN_OR_RETURN", [&] {
      auto r = hyle_chain_err(++seed);
      nb::doNotOptimizeAway(r.has_value());
    });
  }

  // ---- (8) public Result<void> path: sched join/send/change operations use
  // this shape, so RETURN_IF_ERROR must stay as cheap as the manual check.
  {
    nb::Bench b;
    b.title("Result<void> propagate OK through 2 calls")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    b.run("manual Result<void> checks", [&] {
      bool fail_first = false;
      bool fail_second = false;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = hyle_void_chain_manual(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
    b.run("HYLE RETURN_IF_ERROR(Result<void>)", [&] {
      bool fail_first = false;
      bool fail_second = false;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = hyle_void_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    nb::Bench b;
    b.title("Result<void> propagate ERROR from second call")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    b.run("manual Result<void> checks", [&] {
      bool fail_first = false;
      bool fail_second = true;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = hyle_void_chain_manual(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
    b.run("HYLE RETURN_IF_ERROR(Result<void>)", [&] {
      bool fail_first = false;
      bool fail_second = true;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = hyle_void_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  // ---- (9) normal diagnostics: fixed-buffer formatting is the C/FFI/logging
  // primitive; allocating C++ formatting is an opt-in convenience layer.
  {
    Status status = Status::Unavailable(
        "vulkan vkQueueSubmit failed: VK_ERROR_DEVICE_LOST (-4)");
    char buffer[128];
    nb::Bench b;
    b.title("Status human-readable formatting")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
    b.run("HYLE ToString caller buffer", [&] {
      const std::size_t size = status.ToString(buffer, sizeof(buffer));
      nb::doNotOptimizeAway(size);
      nb::doNotOptimizeAway(buffer[0]);
    });
    b.run("HYLE to_string owning", [&] {
      std::string text = hyle::to_string(status);
      nb::doNotOptimizeAway(text);
    });
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status =
        absl::UnavailableError(status.message());
    b.run("absl::Status::ToString owning", [&] {
      std::string text = absl_status.ToString();
      nb::doNotOptimizeAway(text);
    });
#endif
  }

  // ---- (10) deterministic aggregation: keep the first non-OK status.
  {
    const Status hyle_first = Status::NotFound("first");
    const Status hyle_second = Status::Internal("second");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_first = absl::NotFoundError("first");
    const absl::Status absl_second = absl::InternalError("second");
#endif
    nb::Bench b;
    b.title("Status aggregate first error")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::Status::Update", [&] {
      absl::Status status;
      status.Update(absl_first);
      status.Update(absl_second);
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    b.run("HYLE Status::Update", [&] {
      Status status;
      status.Update(hyle_first);
      status.Update(hyle_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  // ---- space dimension: sizes printed so the table's space column is real, not claimed.
  std::printf("\n=== error carrier SIZES (space dimension) ===\n");
  std::printf("  sizeof(hyle::Status)               = %zu bytes (one tagged word)\n", sizeof(Status));
  std::printf("  sizeof(hyle::Result<int>)          = %zu bytes\n", sizeof(Result<int>));
  std::printf("  sizeof(hyle::Result<void>)         = %zu bytes\n", sizeof(Result<void>));
  std::printf("  sizeof(std::expected<int,errc>)    = %zu bytes\n", sizeof(StdResult));
  std::printf("  sizeof(std::error_code)            = %zu bytes\n", sizeof(std::error_code));
#if defined(HYLE_BENCH_ABSL)
  std::printf("  sizeof(absl::Status)               = %zu bytes\n", sizeof(absl::Status));
  std::printf("  sizeof(absl::StatusOr<int>)        = %zu bytes\n", sizeof(absl::StatusOr<int>));
#endif
  return 0;
}
