// Normal user paths for error utilities. Kept separate from status_bench.cc so
// adding diagnostics/aggregation workloads cannot change the compiler's inline
// budget and therefore the codegen of the core carrier micro-invariants.
#include <nanobench.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/c/error.h"
#include "hyle/error/format.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/strings/cord.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

using hyle::Result;
using hyle::Status;

// Keep the shared error behind a call boundary. Otherwise GCC can prove that
// the local retain/release pair is unobservable in the hand-written variant
// but not through the generic return-type adaptor, which benchmarks escape
// analysis rather than propagation semantics.
HYLE_NOINLINE const Status &prebuilt_error() {
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

hyle_status c_macro_ok_chain() {
  HYLE_C_RETURN_IF_ERROR(hyle_status_ok());
  HYLE_C_RETURN_IF_ERROR(hyle_status_ok());
  return hyle_status_ok();
}

hyle_status c_manual_ok_chain() {
  hyle_status first = hyle_status_ok();
  if (first.rep != 0) return first;
  hyle_status second = hyle_status_ok();
  if (second.rep != 0) return second;
  return hyle_status_ok();
}

#if defined(HYLE_BENCH_ABSL)
constexpr std::string_view kVulkanPayloadUrl =
    "type.hyle.dev/native-error/vulkan";
absl::Status absl_native_error() {
  absl::Status status(absl::StatusCode::kUnavailable, "device lost",
                      absl::SourceLocation());
  const std::int64_t code = -4;
  status.SetPayload(
      kVulkanPayloadUrl,
      absl::Cord(absl::string_view(reinterpret_cast<const char *>(&code),
                                   sizeof(code))));
  return status;
}
#endif

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
    bench.title("C ABI Status propagate OK through 2 calls")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
    bench.run("C manual rep checks (OK)", [&] {
      hyle_status status = c_manual_ok_chain();
      nb::doNotOptimizeAway(status.rep);
    });
    bench.run("C HYLE_C_RETURN_IF_ERROR (OK)", [&] {
      hyle_status status = c_macro_ok_chain();
      nb::doNotOptimizeAway(status.rep);
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
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   status.message(),
                                   absl::SourceLocation());
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
    const absl::Status absl_first(absl::StatusCode::kNotFound, "first",
                                  absl::SourceLocation());
    const absl::Status absl_second(absl::StatusCode::kInternal, "second",
                                   absl::SourceLocation());
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
      absl::Status status = absl::ErrnoToStatus(
          ENOENT, "opening model", absl::SourceLocation());
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE status_from_errno", [&] {
      Status status = hyle::status_from_errno(ENOENT, "opening model");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    nb::Bench bench;
    bench.title("Status construct with machine-readable native detail")
        .relative(true)
        .batch(1)
        .minEpochIterations(300000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::Status + SetPayload", [&] {
      absl::Status status = absl_native_error();
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE Status::FromNative", [&] {
      Status status = Status::FromNative(
          hyle::StatusCode::Unavailable,
          {hyle::NativeErrorDomain::Vulkan, -4}, "device lost");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    const Status hyle_status = Status::FromNative(
        hyle::StatusCode::Unavailable,
        {hyle::NativeErrorDomain::Vulkan, -4}, "device lost");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status = absl_native_error();
#endif
    nb::Bench bench;
    bench.title("Status query machine-readable native detail")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::Status GetPayload + decode", [&] {
      std::int64_t code = 0;
      const auto payload = absl_status.GetPayload(kVulkanPayloadUrl);
      if (payload.has_value()) {
        const auto bytes = payload->TryFlat();
        if (bytes.has_value() && bytes->size() == sizeof(code)) {
          std::memcpy(&code, bytes->data(), sizeof(code));
        }
      }
      nb::doNotOptimizeAway(code);
    });
#endif
    bench.run("HYLE Status::native_error", [&] {
      const hyle::NativeError native = hyle_status.native_error();
      nb::doNotOptimizeAway(native.code);
    });
  }

  return 0;
}
