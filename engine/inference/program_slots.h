#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>

#include "hyle/core/assert.h"
#include "program_carrier.h"
#include "slot_transactions.h"

namespace hyle::inference {

// Stable physical carriers for one already selected concrete program. The
// borrowed CompiledGraph, resource Graph and fixed-view owners must outlive
// their original carriers. Keep the complete group in its existing cold owner;
// neither the group nor its entries can move after IO handles are registered.
// Model-specific tensor descriptions and state meanings remain in Payload.
template<class Scope, class Payload, std::size_t Capacity>
class ProgramSlots final {
  static_assert(Capacity != 0);

 public:
  struct Slot final {
    Payload data;
    std::size_t planned_uses = 0;
    std::size_t preparation_uses = 0;
    graph::ReplayMode replay_mode = graph::ReplayMode::kCommands;
    ProgramIo io;
    SlotTransactions transactions;
    // Destroy the original Invocation before its transactions, IO inventories
    // and borrowed fixed Tensor owners. Accepted native work still keeps the
    // original Scope through ProgramCarrier's one retirement attachment.
    ProgramCarrier<Scope> carrier;
  };

  // Synchronous installation facts only. The selected entry supplies its own
  // original IO; callers cannot accidentally install a peer's fixed bindings.
  // No callback, span or borrowed recipe is retained after initialize().
  struct Initialization final {
    graph::InvocationOptions invocation{};
    ProgramInitialization::Progress progress{};
    std::optional<ProgramInitialization::Slots> slots{};
    std::optional<ProgramInitialization::Exact> exact{};
  };

  ProgramSlots() = default;
  ProgramSlots(const ProgramSlots&) = delete;
  ProgramSlots& operator=(const ProgramSlots&) = delete;
  ProgramSlots(ProgramSlots&&) = delete;
  ProgramSlots& operator=(ProgramSlots&&) = delete;

  // One unpublished cold construction. PrepareIo is a direct, statically
  // dispatched recipe which registers and seals each entry's original IO;
  // it may also fill Payload and the entry's cold replay facts. It must not
  // initialize or submit an Invocation. An error or the recipe's original
  // exception leaves size()==0 and forbids retry of this partial group: its
  // unpublished cold owner must destroy it. No native rollback is invented.
  template<class PrepareIo>
  Status prepare(const graph::CompiledGraph& compiled,
      std::span<const graph::QueueBinding> queues, std::size_t count,
      std::span<const SlotTransactionSpec> transactions, PrepareIo&& prepare_io) {
    if (!count || count > Capacity)
      return Status::InvalidArgument("program slot count is outside its cold capacity");
    if (preparation_attempted_)
      return Status::FailedPrecondition("program slots already attempted cold preparation");
    preparation_attempted_ = true;
    for (std::size_t index = 0; index < count; ++index) {
      auto& slot = entries_[index];
      HYLE_RETURN_IF_ERROR(slot.carrier.prepare(compiled, queues));
      HYLE_RETURN_IF_ERROR(slot.transactions.prepare(transactions));
      HYLE_RETURN_IF_ERROR(prepare_io(index, slot));
      HYLE_RETURN_IF_ERROR(slot.io.check_program(compiled));
    }
    size_ = count;
    return Status::Ok();
  }

  std::size_t size() const noexcept { return size_; }
  std::span<Slot> slots() noexcept { return {entries_.data(), size_}; }
  std::span<const Slot> slots() const noexcept { return {entries_.data(), size_}; }

  Slot& operator[](std::size_t index) noexcept {
    HYLE_DEBUG_ASSERT(index < size_);
    return entries_[index];
  }
  const Slot& operator[](std::size_t index) const noexcept {
    HYLE_DEBUG_ASSERT(index < size_);
    return entries_[index];
  }

  // Idempotence keeps an already installed original run; it grants no new
  // borrow or submit permission. A closed truthy run still needs the original
  // retired reset. Failed candidate installation never replaces its peer.
  Status initialize(std::size_t index, const Initialization& options) {
    if (index >= size_)
      return Status::OutOfRange("program slot is outside its cold carrier plan");
    auto& slot = entries_[index];
    if (slot.carrier.closing())
      return Status::FailedPrecondition("program slot needs its original retired reset before initialization");
    if (slot.carrier.initialized()) return Status::Ok();
    HYLE_RETURN_IF_ERROR(slot.carrier.initialize({
        .io = slot.io,
        .invocation = options.invocation,
        .progress = options.progress,
        .slots = options.slots,
        .exact = options.exact}));
    slot.preparation_uses = 0;
    return Status::Ok();
  }

  // A later failure preserves earlier successful original installations.
  // The caller can retry the failed entry without recreating its peers.
  Status initialize_all(const Initialization& options) {
    if (!size_)
      return Status::FailedPrecondition("program slots are not cold prepared");
    for (std::size_t index = 0; index < size_; ++index)
      HYLE_RETURN_IF_ERROR(initialize(index, options));
    return Status::Ok();
  }

  bool pending() const noexcept {
    for (const auto& slot : slots())
      if (slot.carrier.pending()) return true;
    return false;
  }

  // Attempt every original carrier even when a peer reports an error. Actual
  // retirement and terminal status remain independent; no reset, transaction
  // take, state publication or new Completion is performed by this scan.
  template<class Environment>
  RetirementObservation retire_all(Environment& environment) noexcept {
    RetirementObservation result{true, {}};
    for (auto& slot : slots()) {
      if (!slot.carrier.initialized() && !slot.carrier.pending()) continue;
      const auto observed = slot.carrier.retire(environment);
      result.retired = result.retired && observed.retired;
      record_retirement_status(result.status, observed.status);
    }
    return result;
  }

  // Teardown only, after the existing owner hands off its original Backend.
  // Each carrier's irreversible handoff breaks its application Graph root;
  // accepted native work retains the same real retirement attachment.
  void release_invocation_roots() noexcept {
    for (auto& slot : slots()) slot.carrier.release_invocation_root();
  }

 private:
  std::array<Slot, Capacity> entries_;
  std::size_t size_ = 0;
  bool preparation_attempted_ = false;
};

} // namespace hyle::inference
