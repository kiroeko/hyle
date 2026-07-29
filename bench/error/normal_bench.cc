// Information-bearing error paths. Every SOTA comparison carries the same
// message/context/native detail; information-free carrier paths are isolated in
// status_bench.cc.
#include <nanobench.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/error/format.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/status/status_builder.h"
#  include "absl/status/statusor.h"
#  include "absl/strings/cord.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;

const Status &hyle_prebuilt_error() {
  static const Status status = Status::NotFound("resource not found");
  return status;
}

// Use the same two real call boundaries for HYLE and Abseil. The benchmark
// measures carrier propagation, not a compiler's asymmetric inlining choice.
HYLE_NOINLINE Result<int> hyle_info_leaf(int value) {
  return value >= 0 ? Result<int>(hyle::make_unexpected(hyle_prebuilt_error()))
                    : Result<int>(value);
}
HYLE_NOINLINE Result<int> hyle_info_chain(int value) {
  HYLE_ASSIGN_OR_RETURN(int result, hyle_info_leaf(value));
  return result;
}

#if defined(HYLE_BENCH_ABSL)
constexpr std::string_view kVulkanPayloadUrl =
    "type.hyle.dev/native-error/vulkan";
constexpr std::string_view kPosixPayloadUrl =
    "type.hyle.dev/native-error/posix-errno";

void set_absl_native_payload(absl::Status &status, std::string_view type_url,
                             std::int64_t code) {
  status.SetPayload(type_url,
                    absl::Cord(absl::string_view(
                        reinterpret_cast<const char *>(&code), sizeof(code))));
}

absl::Status absl_native_error() {
  absl::Status status(absl::StatusCode::kUnavailable, "device lost",
                      absl::SourceLocation());
  set_absl_native_payload(status, kVulkanPayloadUrl, -4);
  return status;
}

absl::Status absl_errno_with_native() {
  absl::Status status =
      absl::ErrnoToStatus(ENOENT, "opening model", absl::SourceLocation());
  set_absl_native_payload(status, kPosixPayloadUrl, ENOENT);
  return status;
}

const absl::Status &absl_prebuilt_error() {
  static const absl::Status status(absl::StatusCode::kNotFound,
                                   "resource not found",
                                   absl::SourceLocation());
  return status;
}

HYLE_NOINLINE absl::StatusOr<int> absl_info_leaf(int value) {
  return value >= 0 ? absl::StatusOr<int>(absl_prebuilt_error())
                    : absl::StatusOr<int>(value);
}

HYLE_NOINLINE absl::StatusOr<int> absl_info_chain(int value) {
  auto result = absl_info_leaf(value);
  if (!result.ok()) return result.status();
  return *result;
}
#endif

}  // namespace

int main() {
  // Owning human-readable error construction: same canonical code and message,
  // with Abseil source-location capture explicitly disabled on the rival.
  {
    constexpr std::string_view message = "resource not found in shard 7";
    nb::Bench bench;
    bench.title("Status construct with message")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::Status(code,msg,no-loc)", [&] {
      absl::Status status(absl::StatusCode::kNotFound, message,
                          absl::SourceLocation());
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE Status::NotFound(msg)", [&] {
      Status status = Status::NotFound(message);
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    const Status hyle_status =
        Status::NotFound("resource not found in shard 7");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kNotFound,
                                   "resource not found in shard 7",
                                   absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Status copy with message")
        .relative(true)
        .batch(1)
        .minEpochIterations(2000000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::Status copy (message)", [&] {
      absl::Status copy = absl_status;
      nb::doNotOptimizeAway(copy.ok());
    });
#endif
    bench.run("HYLE Status copy (message)", [&] {
      Status copy = hyle_status;
      nb::doNotOptimizeAway(copy.ok());
    });
  }

  {
    const Status hyle_status =
        Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   "device queue is unavailable",
                                   absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Status prepend one semantic context")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::StatusBuilder SetPrepend", [&] {
      absl::Status status =
          absl::StatusBuilder(absl_status, absl::SourceLocation()).SetPrepend()
          << "submit work: ";
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE Status::WithContext", [&] {
      Status status = Status(hyle_status).WithContext("submit work");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    const Status hyle_status =
        Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   "device queue is unavailable",
                                   absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Status prepend three semantic contexts")
        .relative(true)
        .batch(1)
        .minEpochIterations(250000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::StatusBuilder SetPrepend x3", [&] {
      absl::Status status = absl_status;
      status = absl::StatusBuilder(std::move(status), absl::SourceLocation())
                   .SetPrepend()
               << "dispatch batch: ";
      status = absl::StatusBuilder(std::move(status), absl::SourceLocation())
                   .SetPrepend()
               << "execute graph: ";
      status = absl::StatusBuilder(std::move(status), absl::SourceLocation())
                   .SetPrepend()
               << "serve request: ";
      nb::doNotOptimizeAway(status.ok());
    });
#endif
    bench.run("HYLE Status::WithContext x3", [&] {
      Status status = Status(hyle_status)
                          .WithContext("dispatch batch")
                          .WithContext("execute graph")
                          .WithContext("serve request");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    int value = 1;
    nb::Bench bench;
    bench.title("Result propagate message error")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::StatusOr message error", [&] {
      auto result = absl_info_chain(value++);
      nb::doNotOptimizeAway(result.ok());
    });
#endif
    bench.run("HYLE Result message error", [&] {
      auto result = hyle_info_chain(value++);
      nb::doNotOptimizeAway(result.has_value());
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
                                   status.message(), absl::SourceLocation());
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

  // Real native-leaf construction: both sides carry canonical code, message,
  // operation context, and a machine-readable errno value.
  {
    nb::Bench bench;
    bench.title("errno to canonical Status with operation context")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::ErrnoToStatus + native payload", [&] {
      absl::Status status = absl_errno_with_native();
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
      Status status = Status::FromNative(hyle::StatusCode::Unavailable,
                                         {hyle::NativeErrorDomain::Vulkan, -4},
                                         "device lost");
      nb::doNotOptimizeAway(status.ok());
    });
  }

  {
    const Status hyle_status = Status::FromNative(
        hyle::StatusCode::Unavailable, {hyle::NativeErrorDomain::Vulkan, -4},
        "device lost");
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
