// Normal user paths for error utilities. Kept separate from status_bench.cc so
// adding diagnostics/aggregation workloads cannot change the compiler's inline
// budget and therefore the codegen of the core carrier micro-invariants.
#include <nanobench.h>

#include <cerrno>
#include <cstddef>
#include <string>
#include <utility>

#include "hyle/error/format.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

using hyle::Result;
using hyle::Status;

const Status &prebuilt_error() {
  static const Status status = Status::NotFound("missing");
  return status;
}

Result<void> void_leaf(bool fail) {
  return fail ? Result<void>(hyle::make_unexpected(prebuilt_error()))
              : Result<void>();
}

Status macro_chain(bool fail_first, bool fail_second) {
  HYLE_RETURN_IF_ERROR(void_leaf(fail_first));
  HYLE_RETURN_IF_ERROR(void_leaf(fail_second));
  return {};
}

Status manual_chain(bool fail_first, bool fail_second) {
  Result<void> first = void_leaf(fail_first);
  if (!first) return std::move(first).error();
  Result<void> second = void_leaf(fail_second);
  if (!second) return std::move(second).error();
  return {};
}

}  // namespace

int main() {
  // sched JoinHandle/channel operations publicly return Result<void>. Keep both
  // booleans opaque to the optimizer so neither side can delete a whole call.
  {
    nb::Bench bench;
    bench.title("Result<void> propagate OK through 2 calls")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    bench.run("manual Result<void> checks (OK)", [&] {
      bool fail_first = false;
      bool fail_second = false;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = manual_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
    bench.run("HYLE RETURN_IF_ERROR(Result<void>) (OK)", [&] {
      bool fail_first = false;
      bool fail_second = false;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = macro_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    nb::Bench bench;
    bench.title("Result<void> propagate ERROR from second call")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
    bench.run("manual Result<void> checks (ERROR)", [&] {
      bool fail_first = false;
      bool fail_second = true;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = manual_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
    bench.run("HYLE RETURN_IF_ERROR(Result<void>) (ERROR)", [&] {
      bool fail_first = false;
      bool fail_second = true;
      nb::doNotOptimizeAway(fail_first);
      nb::doNotOptimizeAway(fail_second);
      Status status = macro_chain(fail_first, fail_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  // Fixed-buffer formatting is the C/FFI/logging primitive. Owning formatting
  // is an opt-in C++ convenience and races Abseil's same-shape operation.
  {
    const Status status = Status::Unavailable(
        "vulkan vkQueueSubmit failed: VK_ERROR_DEVICE_LOST (-4)");
    char buffer[128];
    nb::Bench bench;
    bench.title("Status human-readable formatting")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
    bench.run("HYLE ToString caller buffer", [&] {
      const std::size_t size = status.ToString(buffer, sizeof(buffer));
      nb::doNotOptimizeAway(size);
      nb::doNotOptimizeAway(buffer[0]);
    });
    bench.run("HYLE to_string owning", [&] {
      std::string text = hyle::to_string(status);
      nb::doNotOptimizeAway(text);
    });
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status =
        absl::UnavailableError(status.message());
    bench.run("absl::Status::ToString owning", [&] {
      std::string text = absl_status.ToString();
      nb::doNotOptimizeAway(text);
    });
#endif
  }

  // Deterministic fan-out/fan-in aggregation keeps the first non-OK Status.
  {
    const Status hyle_first = Status::NotFound("first");
    const Status hyle_second = Status::Internal("second");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_first = absl::NotFoundError("first");
    const absl::Status absl_second = absl::InternalError("second");
#endif
    nb::Bench bench;
    bench.title("Status aggregate first error")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::Status::Update", [&] {
      absl::Status status;
      status.Update(absl_first);
      status.Update(absl_second);
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE Status::Update", [&] {
      Status status;
      status.Update(hyle_first);
      status.Update(hyle_second);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  // Real native-leaf construction: classify errno, obtain the CRT diagnostic,
  // and prepend the operation in one Status allocation. Abseil exposes the
  // same canonical adapter and is the direct mature-library comparison.
  {
    nb::Bench bench;
    bench.title("errno to canonical Status with operation context")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::ErrnoToStatus", [&] {
      absl::Status status = absl::ErrnoToStatus(ENOENT, "opening model");
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE status_from_errno", [&] {
      Status status = hyle::status_from_errno(ENOENT, "opening model");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  return 0;
}
