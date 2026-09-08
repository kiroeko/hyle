#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/io/io_context.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_TRANSFER_API
#elif defined(HYLE_IO_TRANSFER_BUILD)
#define HYLE_IO_TRANSFER_API HYLE_EXPORT
#else
#define HYLE_IO_TRANSFER_API HYLE_IMPORT
#endif

namespace hyle::io {

class TlsContext;

// Reliable bounded transfer is an IO mechanism, not a repository client. It
// copies one local file or HTTP(S) representation into a sibling temporary
// file and atomically replaces the destination only after validation + sync.
// Repository schemas, credentials, archives and model parsing stay above IO.

enum class TransferSourceKind : std::uint8_t {
  kLocalFile,
  kHttp,
};

enum class TransferPartialPolicy : std::uint8_t {
  // Best-effort removal on failure. A cleanup failure is returned as an owning
  // checkpoint so the artifact never becomes an unreported orphan.
  kDiscard,
  // Return an owning checkpoint. Dropping it removes the partial unless the
  // caller explicitly keeps it or moves it into TransferPlan::resume().
  kRetain,
};

enum class TransferResumeMismatchPolicy : std::uint8_t {
  // Safely truncate and restart when the source identity/validator changed or
  // an HTTP server ignored a Range request.
  kRestart,
  // Preserve the partial and report FailedPrecondition instead.
  kFail,
};

enum class TransferPhase : std::uint8_t {
  kPreparing,
  kTransferring,
  kVerifying,
  kPublishing,
};

enum class TransferDisposition : std::uint8_t {
  kNoArtifact,
  kPartialDiscarded,
  kPartialRetained,
  kPublished,
};

enum class TransferDurability : std::uint8_t {
  kNone,
  // File data and metadata were synced before atomic replacement. Directory
  // entry durability is not claimed because Platform currently has no portable
  // directory-sync contract.
  kFileDataAndMetadata,
};

struct TransferLimits {
  // Maximum size of the final artifact, including bytes already present in a
  // resumed partial. UINT64_MAX means no library-imposed byte limit.
  std::uint64_t max_bytes = (std::numeric_limits<std::uint64_t>::max)();
  // One local-copy buffer and the maximum HTTP body chunk. A resumed local
  // copy temporarily uses a second buffer of this size to compare the source
  // prefix byte-for-byte with the retained partial. Memory stays O(1) in
  // artifact size. Must be nonzero.
  // 1 MiB amortizes one completion per block on local model-file paths while
  // keeping the per-transfer working set bounded and modest. Callers with a
  // tighter memory budget may select a smaller nonzero block explicitly.
  std::size_t buffer_bytes = 1024 * 1024;
  std::size_t max_http_head_bytes = 64 * 1024;
  std::size_t max_http_header_count = 128;
  unsigned max_redirects = 5;
};

struct TransferOptions {
  TransferLimits limits{};
  TransferPartialPolicy partial_on_failure = TransferPartialPolicy::kRetain;
  TransferResumeMismatchPolicy resume_mismatch =
      TransferResumeMismatchPolicy::kRestart;
  // Secure default: an HTTPS source may not redirect to cleartext HTTP.
  bool allow_https_to_http_redirect = false;
};

struct LocalTransferSource {
  container::StringView path{};
};

// Accepts an absolute http:// or https:// URL. Plans copy it immediately.
// Credentials/userinfo and URL fragments are rejected; authentication policy
// is intentionally not part of this foundation mechanism.
struct HttpTransferSource {
  container::StringView url{};
};

// Publication always uses a sibling temporary file and atomic replacement.
// The old destination remains intact unless and until the commit succeeds.
struct AtomicFileDestination {
  container::StringView path{};
};

struct TransferProgress {
  TransferPhase phase = TransferPhase::kPreparing;
  std::uint64_t artifact_bytes = 0;
  std::uint64_t bytes_this_attempt = 0;
  std::optional<std::uint64_t> total_bytes{};
  bool resumed = false;
  bool restarted = false;
};

// Borrowed, allocation-free callback. It runs synchronously on the transfer's
// owner fiber and must not re-enter the same transfer. A non-OK return stops
// the pipeline through its ordinary partial-retention policy.
class TransferProgressObserver {
public:
  using NotifyFn = Status (*)(void *, const TransferProgress &) noexcept;

  constexpr TransferProgressObserver() noexcept = default;

  template <class Observer>
    requires requires(Observer &observer, const TransferProgress &progress) {
      { observer(progress) } noexcept -> std::same_as<Status>;
    }
  static TransferProgressObserver from(Observer &observer) noexcept {
    return from_callback(
        std::addressof(observer),
        [](void *context, const TransferProgress &progress) noexcept {
          return (*static_cast<Observer *>(context))(progress);
        });
  }

  static constexpr TransferProgressObserver
  from_callback(void *context, NotifyFn notify) noexcept {
    return TransferProgressObserver(context, notify);
  }

  constexpr bool valid() const noexcept { return notify_ != nullptr; }
  Status notify(const TransferProgress &progress) const noexcept {
    return notify_ == nullptr ? Status{} : notify_(context_, progress);
  }

private:
  constexpr TransferProgressObserver(void *context, NotifyFn notify) noexcept
      : context_(context), notify_(notify) {}

  void *context_ = nullptr;
  NotifyFn notify_ = nullptr;
};

// Borrowed incremental whole-artifact verifier. reset() starts a new digest,
// update() consumes bytes in artifact order, and finish(total_bytes) compares
// against caller-owned expectations. Resume replays the retained prefix through
// update(), so the hook always observes the complete artifact exactly once per
// attempt. IO deliberately mandates no checksum algorithm.
class TransferVerifier {
public:
  using ResetFn = Status (*)(void *) noexcept;
  using UpdateFn = Status (*)(void *, std::span<const std::byte>) noexcept;
  using FinishFn = Status (*)(void *, std::uint64_t) noexcept;

  constexpr TransferVerifier() noexcept = default;

  template <class Verifier>
    requires requires(Verifier &verifier, std::span<const std::byte> bytes,
                      std::uint64_t size) {
      { verifier.reset() } noexcept -> std::same_as<Status>;
      { verifier.update(bytes) } noexcept -> std::same_as<Status>;
      { verifier.finish(size) } noexcept -> std::same_as<Status>;
    }
  static TransferVerifier from(Verifier &verifier) noexcept {
    return from_callbacks(
        std::addressof(verifier),
        [](void *context) noexcept {
          return static_cast<Verifier *>(context)->reset();
        },
        [](void *context, std::span<const std::byte> bytes) noexcept {
          return static_cast<Verifier *>(context)->update(bytes);
        },
        [](void *context, std::uint64_t size) noexcept {
          return static_cast<Verifier *>(context)->finish(size);
        });
  }

  static constexpr TransferVerifier from_callbacks(void *context, ResetFn reset,
                                                   UpdateFn update,
                                                   FinishFn finish) noexcept {
    return TransferVerifier(context, reset, update, finish);
  }

  constexpr bool valid() const noexcept {
    return reset_ != nullptr && update_ != nullptr && finish_ != nullptr;
  }
  Status reset() const noexcept {
    return valid() ? reset_(context_)
                   : Status::FailedPrecondition("transfer verifier is invalid");
  }
  Status update(std::span<const std::byte> bytes) const noexcept {
    return valid() ? update_(context_, bytes)
                   : Status::FailedPrecondition("transfer verifier is invalid");
  }
  Status finish(std::uint64_t size) const noexcept {
    return valid() ? finish_(context_, size)
                   : Status::FailedPrecondition("transfer verifier is invalid");
  }

private:
  constexpr TransferVerifier(void *context, ResetFn reset, UpdateFn update,
                             FinishFn finish) noexcept
      : context_(context), reset_(reset), update_(update), finish_(finish) {}

  void *context_ = nullptr;
  ResetFn reset_ = nullptr;
  UpdateFn update_ = nullptr;
  FinishFn finish_ = nullptr;
};

namespace detail {
struct TransferState;
class TransferAccess;
} // namespace detail

// Stable path value returned when the caller takes cleanup ownership of a
// partial file. The file remains on disk after this object is destroyed; the
// object only owns the path storage needed to find it. Move-only ownership
// makes the lifetime explicit without a second allocation or path copy.
class KeptTransferPartial {
public:
  KeptTransferPartial() noexcept = default;
  HYLE_IO_TRANSFER_API ~KeptTransferPartial();
  HYLE_IO_TRANSFER_API KeptTransferPartial(KeptTransferPartial &&) noexcept;
  HYLE_IO_TRANSFER_API KeptTransferPartial &
  operator=(KeptTransferPartial &&) noexcept;
  KeptTransferPartial(const KeptTransferPartial &) = delete;
  KeptTransferPartial &operator=(const KeptTransferPartial &) = delete;

  HYLE_IO_TRANSFER_API bool valid() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView path() const noexcept;
  HYLE_IO_TRANSFER_API const char *c_str() const noexcept;
  explicit operator bool() const noexcept { return valid(); }

private:
  friend class TransferCheckpoint;
  explicit KeptTransferPartial(detail::TransferState *state) noexcept
      : state_(state) {}
  detail::TransferState *state_ = nullptr;
};

// Owns the exact open temporary file and all source/validator identity needed
// for an in-process retry. It is move-only. Destruction performs best-effort
// close + removal; keep_partial() is the explicit escape hatch for handing the
// artifact to another process or diagnostic tool.
class TransferCheckpoint {
public:
  TransferCheckpoint() noexcept = default;
  HYLE_IO_TRANSFER_API ~TransferCheckpoint();
  HYLE_IO_TRANSFER_API TransferCheckpoint(TransferCheckpoint &&) noexcept;
  HYLE_IO_TRANSFER_API TransferCheckpoint &
  operator=(TransferCheckpoint &&) noexcept;
  TransferCheckpoint(const TransferCheckpoint &) = delete;
  TransferCheckpoint &operator=(const TransferCheckpoint &) = delete;

  HYLE_IO_TRANSFER_API bool valid() const noexcept;
  HYLE_IO_TRANSFER_API bool resumable() const noexcept;
  HYLE_IO_TRANSFER_API TransferSourceKind source_kind() const noexcept;
  HYLE_IO_TRANSFER_API std::uint64_t partial_bytes() const noexcept;
  HYLE_IO_TRANSFER_API std::optional<std::uint64_t>
  expected_bytes() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView source() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView destination() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView partial_path() const noexcept;

  // Closes the open partial, disarms automatic removal and transfers its stable
  // path storage to an owning C++ value. The kept file remains the caller's
  // responsibility even after the returned value is destroyed.
  HYLE_IO_TRANSFER_API Result<KeptTransferPartial> keep_partial() noexcept;
  HYLE_IO_TRANSFER_API Status discard() noexcept;

private:
  friend class detail::TransferAccess;
  explicit TransferCheckpoint(detail::TransferState *state) noexcept
      : state_(state) {}
  detail::TransferState *state_ = nullptr;
};

// Move-only owning operation description. Factory arguments are borrowed only
// for the call; the returned plan owns every path/URL and remains valid while a
// coroutine is suspended. Dropping a resumed plan cleans its partial. The
// allocator passed to the factory backs plans, checkpoints and kept-path values
// and must outlive every value produced from that plan.
class TransferPlan {
public:
  TransferPlan() noexcept = default;
  HYLE_IO_TRANSFER_API ~TransferPlan();
  HYLE_IO_TRANSFER_API TransferPlan(TransferPlan &&) noexcept;
  HYLE_IO_TRANSFER_API TransferPlan &operator=(TransferPlan &&) noexcept;
  TransferPlan(const TransferPlan &) = delete;
  TransferPlan &operator=(const TransferPlan &) = delete;

  HYLE_IO_TRANSFER_API static Result<TransferPlan>
  local(mem::AllocatorRef allocator, LocalTransferSource source,
        AtomicFileDestination destination,
        TransferOptions options = {}) noexcept;
  HYLE_IO_TRANSFER_API static Result<TransferPlan>
  http(mem::AllocatorRef allocator, HttpTransferSource source,
       AtomicFileDestination destination,
       TransferOptions options = {}) noexcept;
  // Reuses the exact open partial and original configuration. Deadline/cancel
  // and callbacks belong to each execution and are supplied to transfer().
  HYLE_IO_TRANSFER_API static Result<TransferPlan>
  resume(TransferCheckpoint &&checkpoint) noexcept;
  // Keeps source/destination/validator identity while allowing a retry to
  // change byte/redirect limits and failure policy explicitly.
  HYLE_IO_TRANSFER_API static Result<TransferPlan>
  resume(TransferCheckpoint &&checkpoint, TransferOptions options) noexcept;

  HYLE_IO_TRANSFER_API bool valid() const noexcept;
  HYLE_IO_TRANSFER_API TransferSourceKind source_kind() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView source() const noexcept;
  HYLE_IO_TRANSFER_API container::StringView destination() const noexcept;
  HYLE_IO_TRANSFER_API const TransferOptions &options() const noexcept;

private:
  friend class detail::TransferAccess;
  explicit TransferPlan(detail::TransferState *state) noexcept
      : state_(state) {}
  detail::TransferState *state_ = nullptr;
};

struct TransferExecution {
  unsigned owner_worker = 0;
  OperationOptions operation{};
  TransferVerifier verifier{};
  TransferProgressObserver progress{};
  // Optional borrowed client context for custom trust roots, mTLS or other
  // deployment TLS policy. nullptr uses a secure system-trust context limited
  // to HTTP/1.1. The context must outlive the returned Task.
  const TlsContext *tls_context = nullptr;
};

// Always returns a report so failure can preserve progress and a checkpoint.
// Callbacks and their state, plus coroutine_allocator, must outlive the
// returned Task. The task must run on owner_worker; commit becomes
// non-cancellable once atomic replacement begins.
struct [[nodiscard]] TransferReport {
  Status status{};
  TransferDisposition disposition = TransferDisposition::kNoArtifact;
  TransferDurability durability = TransferDurability::kNone;
  std::uint64_t bytes_this_attempt = 0;
  std::uint64_t artifact_bytes = 0;
  std::optional<std::uint64_t> total_bytes{};
  bool resumed = false;
  bool restarted = false;
  TransferCheckpoint checkpoint{};

  bool ok() const noexcept { return status.ok(); }
  bool published() const noexcept {
    return ok() && disposition == TransferDisposition::kPublished;
  }
  explicit operator bool() const noexcept { return published(); }
};

HYLE_IO_TRANSFER_API sched::Task<TransferReport>
transfer(mem::AllocatorRef coroutine_allocator, IoContext &io,
         TransferPlan plan, TransferExecution execution = {}) noexcept;

static_assert(std::is_nothrow_move_constructible_v<TransferCheckpoint>);
static_assert(std::is_nothrow_move_constructible_v<KeptTransferPartial>);
static_assert(std::is_nothrow_move_constructible_v<TransferPlan>);

} // namespace hyle::io
