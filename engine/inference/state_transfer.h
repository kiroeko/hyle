#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "backend.h"
#include "execution_resources.h"
#include "retirement_observation.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::inference {

// One exact initialized byte prefix. The original exclusive destination may
// cover a larger page: take_writer() returns that whole writer, never a claim
// that its untouched suffix was initialized. No tensor/model meaning lives here.
struct StateTransferCopy final {
  compute::ReadLease source;
  compute::WriteLease destination;
  std::size_t bytes = 0;
};

// Cold, single-use ownership of one original CPU/CUDA CopyBatch episode.
// prepare consumes destinations only on success; sources remain unchanged.
// submit/wait return the original status without discarding this owner. A
// successful wait permits take_writer; no bytes become ready here implicitly.
// lifetime owns ALL borrowed resource/allocator backings, not the application
// Model. It is held only by this temporary transfer, never copied into an image.
class StateTransfer final {
  struct Scope final {
    explicit Scope(std::shared_ptr<void> retained) noexcept : lifetime(std::move(retained)) {}
    // Reverse destruction keeps original resources/Context after every control.
    std::shared_ptr<void> lifetime;
    compute::detail::ContextRef context;
    compute::Queue queue;
    compute::CopyBatchProgram program;
    compute::CopyBatchInvocation invocation;
    compute::CopyBatchUpdate update;
    RetainedResources<Scope> retirement;
    hyle::Status terminal;
    std::size_t copies = 0;
    bool attempted = false, successful_wait = false;
  };
public:
  StateTransfer() noexcept = default;
  StateTransfer(const StateTransfer &) = delete;
  StateTransfer &operator=(const StateTransfer &) = delete;
  StateTransfer(StateTransfer &&other) noexcept : scope_(std::move(other.scope_)) {}
  StateTransfer &operator=(StateTransfer &&other) noexcept {
    if (this != &other) { abandon(); scope_ = std::move(other.scope_); }
    return *this;
  }
  ~StateTransfer() { abandon(); }

  static Result<StateTransfer> prepare(Backend &backend, std::span<StateTransferCopy> copies,
      std::shared_ptr<void> lifetime, compute::ResourceBudget budget = {}) noexcept {
    if (!lifetime)
      return std::unexpected(Status::InvalidArgument("state transfer requires its original resource lifetime"));
    if (copies.size() > (std::numeric_limits<std::uint32_t>::max)() / 2)
      return std::unexpected(Status::OutOfRange("state transfer resource count"));
    HYLE_ASSIGN_OR_RETURN(auto *context, backend.common_context());
    auto allocator = context->metadata_allocator();
    if (budget) { HYLE_ASSIGN_OR_RETURN(allocator, budget.metadata_allocator()); }
    try {
      auto scope = std::allocate_shared<Scope>(mem::StdAllocator<Scope>(allocator), std::move(lifetime));
      HYLE_ASSIGN_OR_RETURN(scope->context, compute::detail::ContextRef::acquire(*context));
      scope->copies = copies.size();
      StateTransfer result(std::move(scope));
      if (copies.empty()) return result;
      auto &owner = *result.scope_;
      std::vector<compute::CopyResourceSpec, mem::StdAllocator<compute::CopyResourceSpec>> resources{
          mem::StdAllocator<compute::CopyResourceSpec>(allocator)};
      std::vector<compute::CopyRange, mem::StdAllocator<compute::CopyRange>> ranges{
          mem::StdAllocator<compute::CopyRange>(allocator)};
      if (copies.size() * 2 > resources.max_size() || copies.size() > ranges.max_size())
        return std::unexpected(Status::OutOfRange("state transfer cold metadata count"));
      resources.reserve(copies.size() * 2); ranges.reserve(copies.size());
      std::size_t total = 0;
      for (std::size_t i = 0; i < copies.size(); ++i) {
        const auto &copy = copies[i];
        HYLE_ASSIGN_OR_RETURN(auto source, copy.source.view());
        HYLE_ASSIGN_OR_RETURN(auto destination, copy.destination.view());
        if (!copy.bytes || copy.bytes > source.size() || copy.bytes > destination.size())
          return std::unexpected(Status::InvalidArgument("state transfer initialized prefix exceeds its original leases"));
        if (copy.bytes > SIZE_MAX - total)
          return std::unexpected(Status::OutOfRange("state transfer byte count"));
        total += copy.bytes;
        resources.push_back({source.resource(), source.size()});
        resources.push_back({destination.resource(), destination.size()});
        ranges.push_back({{static_cast<std::uint32_t>(2 * i)}, {static_cast<std::uint32_t>(2 * i + 1)},
            {0}, 0, 0, copy.bytes});
      }
      HYLE_ASSIGN_OR_RETURN(owner.queue, owner.context.create_queue());
      const compute::CopyBatchOptions options{
          .max_resources = static_cast<std::uint32_t>(resources.size()),
          .max_copies = static_cast<std::uint32_t>(copies.size()), .max_phases = 1,
          .max_bytes = total, .allocator = allocator, .resource_budget = std::move(budget)};
      HYLE_ASSIGN_OR_RETURN(owner.program, compute::detail::CopyStreamAccess::prepare(owner.context, resources, options));
      HYLE_ASSIGN_OR_RETURN(owner.invocation, compute::CopyBatchInvocation::create(owner.program, owner.queue));
      std::size_t bound = 0;
      hyle::Status status;
      for (std::size_t i = 0; i < copies.size(); ++i) {
        status = owner.invocation.bind_source({static_cast<std::uint32_t>(2 * i)}, copies[i].source);
        if (!status.ok()) break;
        status = owner.invocation.bind_mutable({static_cast<std::uint32_t>(2 * i + 1)},
            std::move(copies[i].destination));
        if (!status.ok()) break;
        ++bound;
      }
      constexpr std::array<compute::CopyPhase, 1> phases{};
      if (status.ok()) status = owner.invocation.prepare_replay(phases, ranges);
      if (!status.ok()) {
        // No submit/producer/native work exists. Original withdrawal is
        // nonallocating and proves each successful binding can be returned.
        for (std::size_t i = 0; i < bound; ++i) {
          auto writer = owner.invocation.unbind_mutable({static_cast<std::uint32_t>(2 * i + 1)});
          HYLE_FATAL_ASSERT(writer.has_value()); copies[i].destination = std::move(*writer);
        }
        return std::unexpected(status);
      }
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("state transfer cold metadata"));
    }
  }

  bool valid() const noexcept { return bool(scope_); }
  bool in_flight() const noexcept {
    return scope_ && (scope_->update.in_flight() || scope_->invocation.in_flight() || scope_->retirement.pending());
  }
  compute::CopyBatchReport report() const noexcept {
    if (!scope_) return {};
    return scope_->update.valid() ? scope_->update.report() : scope_->invocation.report();
  }
  Status submit() noexcept {
    if (!scope_ || scope_->attempted)
      return Status::FailedPrecondition("state transfer is empty or was already submitted");
    auto &owner = *scope_; owner.attempted = true;
    if (!owner.copies) return Status::Ok();
    const auto reserved = owner.retirement.reserve(scope_);
    if (!reserved.ok()) { owner.terminal = reserved; return owner.terminal; }
    auto submitted = owner.invocation.submit();
    if (!submitted) {
      owner.retirement.reject(); owner.terminal = submitted.error(); return owner.terminal;
    }
    // Exactly one phase, no producer bindings and a sole move-only observer.
    auto update = compute::CopyBatchUpdate::take(std::move(owner.invocation), std::move(*submitted));
    HYLE_FATAL_ASSERT(update.has_value()); owner.update = std::move(*update);
    std::array<compute::Completion *, 1> original{};
    auto count = compute::detail::CopyStreamAccess::retirement_sources(owner.update, original);
    HYLE_FATAL_ASSERT(count && *count == 1);
    HYLE_FATAL_ASSERT(owner.retirement.attach(*original[0]).ok());
    return Status::Ok();
  }
  Status wait() noexcept {
    if (!scope_ || !scope_->attempted)
      return Status::FailedPrecondition("state transfer has not been submitted");
    auto &owner = *scope_;
    if (owner.update.valid()) record_retirement_status(owner.terminal, owner.update.wait());
    owner.successful_wait = owner.terminal.ok() && !in_flight();
    return owner.terminal;
  }
  Result<compute::WriteLease> take_writer(std::size_t index) noexcept {
    if (!scope_ || index >= scope_->copies)
      return std::unexpected(Status::OutOfRange("state transfer destination index"));
    if (!scope_->terminal.ok()) return std::unexpected(scope_->terminal);
    if (!scope_->successful_wait || in_flight())
      return std::unexpected(Status::FailedPrecondition("state transfer destination requires actual successful retirement"));
    return scope_->update.take_mutable({static_cast<std::uint32_t>(2 * index + 1)});
  }

private:
  explicit StateTransfer(std::shared_ptr<Scope> scope) noexcept : scope_(std::move(scope)) {}
  void abandon() noexcept {
    auto held = std::move(scope_);
    if (!held) return;
    // Break the user-observation cycle explicitly. Dropping both original F2
    // controls hands late completions to their native runtime; its attached
    // RetainedResources keeps this entire stable Scope/Queue alive until real
    // retirement. No wait, speculative release or permanent self-cycle.
    held->update = {}; held->invocation = {};
  }
  std::shared_ptr<Scope> scope_;
};

} // namespace hyle::inference
