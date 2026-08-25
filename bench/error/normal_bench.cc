// Information-bearing error paths. Every SOTA comparison carries the same
// message/context/native detail; information-free carrier paths are isolated in
// status_bench.cc.
#include <nanobench.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr std::string_view kSsoSizedMessage = "missing";
constexpr std::string_view kMediumMessage = "resource not found in shard 7";
constexpr std::string_view kShortFormatMessage =
    "vulkan vkQueueSubmit failed: VK_ERROR_DEVICE_LOST (-4)";
static_assert(kSsoSizedMessage.size() == 7);
static_assert(kMediumMessage.size() == 29);
static_assert(kShortFormatMessage.size() == 54);

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

const Status &hyle_prebuilt_error() {
  static const Status status = Status::NotFound("resource not found");
  return status;
}

HYLE_NOINLINE Status hyle_update_first_prvalue() {
  return Status::NotFound("first");
}

HYLE_NOINLINE Status hyle_update_second_prvalue() {
  return Status::Internal("second");
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

HYLE_NOINLINE absl::Status absl_update_first_prvalue() {
  return absl::Status(absl::StatusCode::kNotFound, "first",
                      absl::SourceLocation());
}

HYLE_NOINLINE absl::Status absl_update_second_prvalue() {
  return absl::Status(absl::StatusCode::kInternal, "second",
                      absl::SourceLocation());
}
#endif

void run_message_construct_benchmark(bool rival_first,
                                     const char *message_shape,
                                     std::string_view message,
                                     std::uint64_t min_iterations) {
  const std::string title =
      std::string("Status construct owning ") + message_shape;
  nb::Bench bench;
  bench.title(title).relative(true).batch(1).minEpochIterations(min_iterations);
#if defined(HYLE_BENCH_ABSL)
  run_candidate_peer_pair(
      bench, rival_first,
      std::string("HYLE Status::NotFound owning ") + message_shape +
          " (borrowed input)",
      [&] {
        Status status = Status::NotFound(message);
        nb::doNotOptimizeAway(status);
      },
      std::string("absl::Status owning ") + message_shape +
          " (borrowed input, no-loc)",
      [&] {
        absl::Status status(absl::StatusCode::kNotFound, message,
                            absl::SourceLocation());
        nb::doNotOptimizeAway(status);
      });
#else
  bench.run(std::string("HYLE Status::NotFound owning ") + message_shape +
                " (borrowed input)",
            [&] {
              Status status = Status::NotFound(message);
              nb::doNotOptimizeAway(status);
            });
#endif
}

}  // namespace

int main() {
  const bool rival_first = rival_first_requested();
  std::fprintf(stderr, "# HYLE_ERROR_NORMAL_BENCH order=%s\n",
               rival_first ? "peer-first" : "candidate-first");

  // Owning human-readable error construction: same canonical code and message,
  // with Abseil source-location capture explicitly disabled on the rival.  A
  // short, a current medium, and one long input cover the important size
  // regimes without creating a cross-product of message lengths.
  run_message_construct_benchmark(rival_first, "7-byte SSO-sized message",
                                  kSsoSizedMessage, 1000000);
  run_message_construct_benchmark(rival_first, "29-byte message",
                                  kMediumMessage, 1000000);
  const std::string long_construct_message(256, 'x');
  run_message_construct_benchmark(rival_first, "256-byte long message",
                                  long_construct_message, 250000);

  {
    const Status hyle_status =
        Status::NotFound("resource not found in shard 7");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kNotFound,
                                   "resource not found in shard 7",
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
          Status copy = hyle_status;
          nb::doNotOptimizeAway(copy);
        },
        "absl::Status copy (shared owned 29-byte message)", [&] {
          absl::Status copy = absl_status;
          nb::doNotOptimizeAway(copy);
        });
#else
    bench.run("HYLE Status copy (shared owned 29-byte message)", [&] {
      Status copy = hyle_status;
      nb::doNotOptimizeAway(copy);
    });
#endif
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
              absl::StatusBuilder(absl_status, absl::SourceLocation())
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
    const Status hyle_status =
        Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   "device queue is unavailable",
                                   absl::SourceLocation());
#endif
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
          absl::Status status = absl_status;
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

  // One deeper point exposes growth with context depth while keeping the
  // matrix bounded.  Both peers create a fresh seed and then update an rvalue
  // through the same number of context layers.
  {
    constexpr int kContextDepth = 16;
    nb::Bench bench;
    bench.title("Status prepend 16 contexts to a fresh message error")
        .relative(true)
        .batch(1)
        .minEpochIterations(50000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::WithContext x16 (fresh seed, rvalue chain)",
        [&] {
          Status status = Status::Unavailable("device queue is unavailable");
          for (int depth = 0; depth < kContextDepth; ++depth) {
            status = std::move(status).WithContext("process dependency");
          }
          nb::doNotOptimizeAway(status);
        },
        "absl::StatusBuilder SetPrepend x16 (fresh seed, rvalue chain)", [&] {
          absl::Status status(absl::StatusCode::kUnavailable,
                              "device queue is unavailable",
                              absl::SourceLocation());
          for (int depth = 0; depth < kContextDepth; ++depth) {
            status =
                absl::StatusBuilder(std::move(status), absl::SourceLocation())
                    .SetPrepend()
                << "process dependency: ";
          }
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status::WithContext x16 (fresh seed, rvalue chain)", [&] {
      Status status = Status::Unavailable("device queue is unavailable");
      for (int depth = 0; depth < kContextDepth; ++depth) {
        status = std::move(status).WithContext("process dependency");
      }
      nb::doNotOptimizeAway(status);
    });
#endif
  }

  // Shared/lvalue callers cannot donate storage.  Keep this contract separate
  // from the unique-rvalue chain above so a cheap move path cannot mask the
  // cost of preserving the source representation at every layer.
  {
    constexpr int kContextDepth = 16;
    const Status hyle_seed =
        Status::Unavailable("device queue is unavailable");
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_seed(absl::StatusCode::kUnavailable,
                                 "device queue is unavailable",
                                 absl::SourceLocation());
#endif
    nb::Bench bench;
    bench.title("Status prepend 16 contexts through an lvalue-derived chain")
        .relative(true)
        .batch(1)
        .minEpochIterations(25000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE Status::WithContext x16 (prebuilt seed, lvalue each layer)",
        [&] {
          Status status = hyle_seed;
          for (int depth = 0; depth < kContextDepth; ++depth) {
            status = status.WithContext("process dependency");
          }
          nb::doNotOptimizeAway(status);
        },
        "absl::StatusBuilder SetPrepend x16 (prebuilt seed, lvalue each layer)",
        [&] {
          absl::Status status = absl_seed;
          for (int depth = 0; depth < kContextDepth; ++depth) {
            status = absl::StatusBuilder(status, absl::SourceLocation())
                         .SetPrepend()
                     << "process dependency: ";
          }
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run(
        "HYLE Status::WithContext x16 (prebuilt seed, lvalue each layer)", [&] {
          Status status = hyle_seed;
          for (int depth = 0; depth < kContextDepth; ++depth) {
            status = status.WithContext("process dependency");
          }
          nb::doNotOptimizeAway(status);
        });
#endif
  }

  {
    int value = 1;
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
          auto result = hyle_info_chain(value++);
          nb::doNotOptimizeAway(result.has_value());
        },
        "absl::StatusOr message error (2 noinline frames)", [&] {
          auto result = absl_info_chain(value++);
          nb::doNotOptimizeAway(result.ok());
        });
#else
    bench.run("HYLE Result message error (2 noinline frames)", [&] {
      auto result = hyle_info_chain(value++);
      nb::doNotOptimizeAway(result.has_value());
    });
#endif
  }

  // Fixed-buffer formatting is the C/FFI/logging primitive. Owning formatting
  // is an opt-in C++ convenience and races Abseil's same-shape operation.
  {
    const Status status = Status::Unavailable(kShortFormatMessage);
    char buffer[128];
    nb::Bench bench;
    bench
        .title("Format 54-byte message: HYLE cap=128 buffer and owning strings")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   status.message(), absl::SourceLocation());
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE hyle::to_string owning (54-byte message)",
        [&] {
          std::string text = hyle::to_string(status);
          nb::doNotOptimizeAway(text);
        },
        "absl::Status::ToString owning (54-byte message)", [&] {
          std::string text = absl_status.ToString();
          nb::doNotOptimizeAway(text);
        });
#else
    bench.run("HYLE hyle::to_string owning (54-byte message)", [&] {
      std::string text = hyle::to_string(status);
      nb::doNotOptimizeAway(text);
    });
#endif
    bench.run(
        "HYLE Status::ToString caller buffer (54-byte msg, cap=128)", [&] {
          const std::size_t size = status.ToString(buffer, sizeof(buffer));
          nb::doNotOptimizeAway(size);
          nb::doNotOptimizeAway(buffer[0]);
        });
  }

  {
    const std::string long_format_message(512, 'x');
    const Status status = Status::Unavailable(long_format_message);
    nb::Bench bench;
    bench.title("Format 512-byte message to a full owning string")
        .relative(true)
        .batch(1)
        .minEpochIterations(100000);
#if defined(HYLE_BENCH_ABSL)
    const absl::Status absl_status(absl::StatusCode::kUnavailable,
                                   long_format_message,
                                   absl::SourceLocation());
    run_candidate_peer_pair(
        bench, rival_first,
        "HYLE hyle::to_string owning (512-byte message)",
        [&] {
          std::string text = hyle::to_string(status);
          nb::doNotOptimizeAway(text);
        },
        "absl::Status::ToString owning (512-byte message)", [&] {
          std::string text = absl_status.ToString();
          nb::doNotOptimizeAway(text);
        });
#else
    bench.run("HYLE hyle::to_string owning (512-byte message)", [&] {
      std::string text = hyle::to_string(status);
      nb::doNotOptimizeAway(text);
    });
#endif
  }

  // Deterministic fan-out/fan-in aggregation keeps the first non-OK Status.
  // Keep the historical stored-lvalue shape, but name its ownership contract
  // explicitly rather than presenting it as the only Update workload.
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
    bench.title("Status::Update keeps first of 2 stored lvalue message errors")
        .relative(true)
        .batch(1)
        .minEpochIterations(1000000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first, "HYLE Status::Update lvalue message errors x2",
        [&] {
          Status status;
          status.Update(hyle_first);
          status.Update(hyle_second);
          nb::doNotOptimizeAway(status);
        },
        "absl::Status::Update lvalue message errors x2", [&] {
          absl::Status status;
          status.Update(absl_first);
          status.Update(absl_second);
          nb::doNotOptimizeAway(status);
        });
#else
    bench.run("HYLE Status::Update lvalue message errors x2", [&] {
      Status status;
      status.Update(hyle_first);
      status.Update(hyle_second);
      nb::doNotOptimizeAway(status);
    });
#endif
  }

  // Normal fan-in call sites commonly pass a freshly returned Status.  The
  // noinline factories ensure this is a true prvalue producer/consumer path on
  // every iteration instead of a compiler-spelled copy of a stored object.
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
          status.Update(absl_update_first_prvalue());
          status.Update(absl_update_second_prvalue());
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

  // Real native-leaf construction: both sides carry canonical code, message,
  // operation context, and a machine-readable errno value.
  {
    nb::Bench bench;
    bench.title("errno to canonical Status with operation context")
        .relative(true)
        .batch(1)
        .minEpochIterations(500000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first, "HYLE status_from_errno",
        [&] {
          Status status = hyle::status_from_errno(ENOENT, "opening model");
          nb::doNotOptimizeAway(status.ok());
        },
        "absl::ErrnoToStatus + native payload", [&] {
          absl::Status status = absl_errno_with_native();
          nb::doNotOptimizeAway(status.ok());
        });
#else
    bench.run("HYLE status_from_errno", [&] {
      Status status = hyle::status_from_errno(ENOENT, "opening model");
      nb::doNotOptimizeAway(status.ok());
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("Status construct with machine-readable native detail")
        .relative(true)
        .batch(1)
        .minEpochIterations(300000);
#if defined(HYLE_BENCH_ABSL)
    run_candidate_peer_pair(
        bench, rival_first, "HYLE Status::FromNative",
        [&] {
          Status status = Status::FromNative(
              hyle::StatusCode::Unavailable,
              {hyle::NativeErrorDomain::Vulkan, -4}, "device lost");
          nb::doNotOptimizeAway(status.ok());
        },
        "absl::Status + SetPayload", [&] {
          absl::Status status = absl_native_error();
          nb::doNotOptimizeAway(status.ok());
        });
#else
    bench.run("HYLE Status::FromNative", [&] {
      Status status = Status::FromNative(hyle::StatusCode::Unavailable,
                                         {hyle::NativeErrorDomain::Vulkan, -4},
                                         "device lost");
      nb::doNotOptimizeAway(status.ok());
    });
#endif
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
    run_candidate_peer_pair(
        bench, rival_first, "HYLE Status::native_error",
        [&] {
          const hyle::NativeError native = hyle_status.native_error();
          nb::doNotOptimizeAway(native.code);
        },
        "absl::Status GetPayload + decode", [&] {
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
#else
    bench.run("HYLE Status::native_error", [&] {
      const hyle::NativeError native = hyle_status.native_error();
      nb::doNotOptimizeAway(native.code);
    });
#endif
  }

  return 0;
}
