#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/compute/detail/dependent_completion.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/core/assert.h"

namespace hyle::compute::detail {
struct CopyOutcomeSourceState;
struct CopyOutcomeSourceOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget;
};

// Same-build callbacks operate on the one original Update payload. They do not
// execute work or implement native completion/queue progress.
struct CopyOutcomeSourceControl final {
  void (*construct)(void *) noexcept = nullptr;
  void (*destroy)(void *) noexcept = nullptr;
  Status (*observe)(void *) noexcept = nullptr;
  Status (*wait)(void *) noexcept = nullptr;
  Status (*request_cancel)(void *) noexcept = nullptr;
  void (*abandon)(void *) noexcept = nullptr;
};
struct CopyOutcomeSourceAccess final {
  static HYLE_COMPUTE_API Result<CopyOutcomeSourceState *> create(
      std::size_t capacity, std::size_t payload_bytes, std::size_t payload_alignment,
      CopyOutcomeSourceControl, CopyOutcomeSourceOptions) noexcept;
  static HYLE_COMPUTE_API void *payload(CopyOutcomeSourceState *) noexcept;
  static HYLE_COMPUTE_API bool available(const CopyOutcomeSourceState *) noexcept;
  static HYLE_COMPUTE_API Result<std::span<Completion *>> prepare(
      CopyOutcomeSourceState *, std::size_t count) noexcept;
  static HYLE_COMPUTE_API OutcomeSources commit(
      CopyOutcomeSourceState *, std::size_t count) noexcept;
  static HYLE_COMPUTE_API void release(CopyOutcomeSourceState *) noexcept;
};

// Cold, move-only owner for an original CopyBatchUpdate and bounded retirement
// nodes. All observations and handle mutations are externally serialized;
// native callbacks may retire independently. Allocate before source acceptance.
// Access borrows the original native controls: those completion addresses must
// survive moving Update's handles, as with the four original F2 source families.
template<class Update, class Access> class CopyOutcomeSource final {
  static_assert(std::is_nothrow_default_constructible_v<Update>);
  static_assert(std::is_nothrow_move_assignable_v<Update>);
  static_assert(std::is_nothrow_destructible_v<Update>);
public:
  CopyOutcomeSource() noexcept = default;
  CopyOutcomeSource(const CopyOutcomeSource &) = delete;
  CopyOutcomeSource &operator=(const CopyOutcomeSource &) = delete;
  CopyOutcomeSource(CopyOutcomeSource &&other) noexcept
      : state_(std::exchange(other.state_, nullptr)) {}
  CopyOutcomeSource &operator=(CopyOutcomeSource &&other) noexcept {
    if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
    return *this;
  }
  ~CopyOutcomeSource() { reset(); }
  static Result<CopyOutcomeSource> create(std::size_t capacity,
      CopyOutcomeSourceOptions options = {}) noexcept {
    constexpr CopyOutcomeSourceControl control{
        [](void *p) noexcept { std::construct_at(static_cast<Update *>(p)); },
        [](void *p) noexcept { std::destroy_at(static_cast<Update *>(p)); },
        [](void *p) noexcept { return static_cast<Update *>(p)->observe(); },
        [](void *p) noexcept { return static_cast<Update *>(p)->wait(); },
        [](void *p) noexcept { return static_cast<Update *>(p)->request_cancel(); },
        [](void *p) noexcept { *static_cast<Update *>(p) = {}; }};
    HYLE_ASSIGN_OR_RETURN(auto state, CopyOutcomeSourceAccess::create(
        capacity, sizeof(Update), alignof(Update), control, std::move(options)));
    return CopyOutcomeSource(state);
  }
  bool available() const noexcept { return CopyOutcomeSourceAccess::available(state_); }
  // After completed observation, update retains its original retired controls.
  // Recover the original invocation/writers here to reuse their cold metadata.
  // Before suffix acceptance. Rejection leaves both owner and Update intact.
  // Success consumes the Update and returns one reference for publish; on
  // rejected suffix admission, release that reference without subscribing.
  Result<OutcomeSources> take(Update &&source) noexcept {
    if (!available() || !source.valid())
      return std::unexpected(Status::FailedPrecondition("copy outcome source slot is unavailable"));
    HYLE_ASSIGN_OR_RETURN(auto dependency, Access::dependency(source));
    (void)dependency;
    HYLE_ASSIGN_OR_RETURN(auto count, Access::retirement_sources(source, {}));
    HYLE_ASSIGN_OR_RETURN(auto destinations, CopyOutcomeSourceAccess::prepare(state_, count));
    HYLE_ASSIGN_OR_RETURN(auto filled, Access::retirement_sources(source, destinations));
    HYLE_FATAL_ASSERT(filled == count);
    for (auto *completion : destinations) {
      if (!completion)
        return std::unexpected(Status::InvalidArgument("copy outcome source completion"));
      HYLE_ASSIGN_OR_RETURN(auto original, RetainedCompletion::native_dependency(*completion));
      if (original.terminal_hint && !original.terminal_hint->ok())
        return std::unexpected(*original.terminal_hint);
    }
    update() = std::move(source);
    return CopyOutcomeSourceAccess::commit(state_, count);
  }
  // Trusted input/prefix/guard access only. Do not observe/reset while a suffix
  // borrows this original Update; its protocol remains the sole observer.
  Update &update() noexcept {
    HYLE_FATAL_ASSERT(state_);
    return *static_cast<Update *>(CopyOutcomeSourceAccess::payload(state_));
  }
  void reset() noexcept {
    if (auto *state = std::exchange(state_, nullptr)) CopyOutcomeSourceAccess::release(state);
  }
private:
  explicit CopyOutcomeSource(CopyOutcomeSourceState *state) noexcept : state_(state) {}
  CopyOutcomeSourceState *state_ = nullptr;
};
} // namespace hyle::compute::detail
