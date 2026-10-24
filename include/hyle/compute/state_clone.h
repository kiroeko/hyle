#pragma once
#include "hyle/compute/state_versions.h"
#include "hyle/compute/detail/state_clone_access.h"
#include "hyle/mem/std_allocator.h"
#include <array>
#include <functional>
#include <new>
#include <mutex>
#include <type_traits>
#include <vector>

namespace hyle::compute {
struct StateCloneOptions final {
  mem::MemoryResourceRef resource;
  std::size_t capacity_bytes = 0;
  std::uint32_t slots = 1;
  std::size_t alignment = 256;
};

class StateCloneRoute final {
public:
  StateCloneRoute() noexcept = default;
  template<class Program, class Factory>
  static Result<StateCloneRoute> prepare(const StateStore &, StateCloneOptions,
                                        const Program &, Factory) noexcept;
  bool valid() const noexcept { return store_.valid(); }
  // Explicit cold recovery after actual retirement. Original cancelled F2
  // invocations cannot be reopened by reset; this constructs original slots
  // through the retained typed cold factory. No submit-time recovery.
  HYLE_COMPUTE_STATE_API Status rearm() noexcept;
  // Last episode of each original slot, not lifetime totals. Shared program
  // metadata is counted once; rearm starts a new cold slot/report identity.
  // Unavailable while native work or an explicit host observer is active.
  HYLE_COMPUTE_STATE_API Result<CopyBatchReport> report() const noexcept;
private:
  friend struct detail::StateCloneAccess;
  friend struct detail::StateAccess;
  StateCloneRoute(StateStore store, std::uint32_t route) noexcept
      : store_(std::move(store)), route_(route) {}
  StateStore store_;
  std::uint32_t route_ = 0;
};

// Owns draft/store and the actual F1 mutable permission. lease() is the
// original writer, suitable for the original typed native consumers. Complete
// only with a successfully ready reader of that same block/generation.
class StateWriter final {
public:
  StateWriter() noexcept = default;
  StateWriter(const StateWriter &) = delete;
  StateWriter &operator=(const StateWriter &) = delete;
  HYLE_COMPUTE_STATE_API StateWriter(StateWriter &&) noexcept;
  HYLE_COMPUTE_STATE_API StateWriter &operator=(StateWriter &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateWriter();
  bool valid() const noexcept { return store_.valid(); }
  WriteLease &lease() noexcept { return writer_; }
  HYLE_COMPUTE_STATE_API Status publish() noexcept;
  HYLE_COMPUTE_STATE_API Status complete(const ReadLease &) noexcept;
private:
  friend class CloneTicket;
  StateWriter(StateStore, std::uint32_t, std::uint32_t, WriteLease) noexcept;
  StateStore store_;
  std::uint32_t draft_ = 0, entry_ = 0;
  WriteLease writer_;
  const void *block_ = nullptr;
  std::uint64_t generation_ = 0;
  bool complete_ = false;
};
class CloneTicket final {
public:
  CloneTicket() noexcept = default;
  CloneTicket(const CloneTicket &) = delete;
  CloneTicket &operator=(const CloneTicket &) = delete;
  HYLE_COMPUTE_STATE_API CloneTicket(CloneTicket &&) noexcept;
  HYLE_COMPUTE_STATE_API CloneTicket &operator=(CloneTicket &&) noexcept;
  HYLE_COMPUTE_STATE_API ~CloneTicket();
  bool valid() const noexcept { return store_.valid(); }
  HYLE_COMPUTE_STATE_API Result<StateWriter> poll() noexcept;
  HYLE_COMPUTE_STATE_API Result<StateWriter> wait() noexcept;
  HYLE_COMPUTE_STATE_API Status request_cancel() noexcept;
private:
  friend class StateDraft;
  CloneTicket(StateStore store, std::uint32_t draft, std::uint32_t entry,
              std::uint32_t route, std::uint32_t slot) noexcept
      : store_(std::move(store)), draft_(draft), entry_(entry), route_(route), slot_(slot) {}
  Result<StateWriter> take(bool wait) noexcept;
  StateStore store_;
  std::uint32_t draft_ = 0, entry_ = 0, route_ = 0, slot_ = 0;
};

namespace detail {
template<class Program, class Factory> struct TypedStateCopy final {
  using Invocation = typename std::invoke_result_t<Factory &, const Program &>::value_type;
  using Submission = typename decltype(std::declval<Invocation &>().submit())::value_type;
  struct Slot final {
    explicit Slot(Invocation i) : invocation(std::move(i)) {}
    Slot(Slot &&s) noexcept : invocation(std::move(s.invocation)), submission(std::move(s.submission)),
        output(std::move(s.output)), bytes(s.bytes), live(s.live), abandoned(s.abandoned), disabled(s.disabled) {}
    Invocation invocation;
    Submission submission;
    ReadLease output;
    std::size_t bytes = 0;
    bool live = false, abandoned = false, disabled = false;
    std::uint32_t observers = 0;
    std::mutex wait_lock; // only explicit host observers of original F2 waits
  };
  TypedStateCopy(Program p, Factory f, mem::AllocatorRef a)
      : program(std::move(p)), factory(std::move(f)), allocator(a), slots(mem::StdAllocator<Slot>(a)) {}
  Program program;
  Factory factory;
  mem::AllocatorRef allocator;
  std::vector<Slot, mem::StdAllocator<Slot>> slots;
  bool closed = false;
  Status cleanup(Slot &s) noexcept {
    if (s.observers) return Status::Unavailable("state copy has an explicit host observer");
    if (s.invocation.in_flight()) return Status::Unavailable("state copy retirement is unproven");
    s.submission = {}; s.output.reset();
    const auto reset = s.invocation.reset();
    if (reset.ok()) s.live = false;
    return reset;
  }
  Result<std::uint32_t> start(const ReadLease &source, WriteLease &&destination, std::size_t bytes) noexcept {
    if (closed) return std::unexpected(Status::FailedPrecondition("state copy route is closed"));
    for (std::uint32_t i = 0; i < slots.size(); ++i) {
      auto &s = slots[i]; if (s.observers || s.live || s.disabled) continue;
      s.bytes = bytes; s.abandoned = false;
      auto status = s.invocation.bind_source({0}, source);
      if (status.ok()) status = s.invocation.bind_mutable({1}, std::move(destination));
      const std::array<CopyPhase,1> phases{{{}}};
      const std::array<CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes}}};
      if (status.ok()) status = s.invocation.prepare_replay(phases, ranges);
      if (!status.ok()) { s.invocation.reset().IgnoreError(); return std::unexpected(status); }
      auto submitted = s.invocation.submit();
      if (!submitted) {
        s.disabled = true; s.abandoned = true; s.live = true;
        cleanup(s).IgnoreError(); return std::unexpected(submitted.error());
      }
      s.submission = std::move(*submitted); s.live = true; return i;
    }
    return std::unexpected(Status::Unavailable("state clone slots retained or require cold rearm"));
  }
  Result<WriteLease> take(std::uint32_t index, bool wait) noexcept {
    if (index >= slots.size() || !slots[index].live)
      return std::unexpected(Status::FailedPrecondition("state clone ticket is retired"));
    auto &s = slots[index];
    if (s.observers) return std::unexpected(Status::Unavailable("state clone wait observation still active"));
    if (!wait && s.invocation.in_flight()) return std::unexpected(Status::Unavailable("state clone pending"));
    if (s.submission.valid()) {
      const auto terminal = s.submission.wait();
      if (!terminal.ok()) {
        if (!terminal.IsUnavailable()) { s.disabled = true; s.abandoned = true; cleanup(s).IgnoreError(); }
        return std::unexpected(terminal);
      }
      auto output = s.submission.output({0},{1},0,s.bytes);
      if (!output) return std::unexpected(output.error());
      s.output = std::move(*output); s.submission = {};
    }
    HYLE_RETURN_IF_ERROR(s.invocation.reset());
    HYLE_ASSIGN_OR_RETURN(auto writer, s.output.try_write());
    s.live = false; return writer;
  }
  Status wait(std::uint32_t index) noexcept {
    if (index >= slots.size()) return Status::InvalidArgument("state clone slot index");
    auto &s = slots[index]; std::lock_guard guard(s.wait_lock);
    return s.submission.valid() ? s.submission.wait() : s.invocation.drain();
  }
  void observe(std::uint32_t index, bool add) noexcept {
    for (std::uint32_t i=0;i<slots.size();++i) if (index==UINT32_MAX || index==i) {
      if(add) ++slots[i].observers; else --slots[i].observers;
    }
  }
  Status cancel(std::uint32_t i) noexcept {
    if (i >= slots.size()) return Status::InvalidArgument("state clone slot index");
    auto &s = slots[i]; s.abandoned = true; s.disabled = true;
    // The store monitor protects observer admission, not the original F2
    // handle while its host wait runs outside that monitor. Defer native
    // cancellation/cleanup while that observer owns it; close never waits.
    if (s.observers) return Status::Unavailable("state copy has an explicit host observer");
    return s.submission.valid() ? s.submission.request_cancel() : Status::Ok();
  }
  Status collect() noexcept {
    for (auto &s : slots) if (s.live && s.abandoned && !s.observers && !s.invocation.in_flight()) cleanup(s).IgnoreError();
    for (const auto &s : slots) if (s.live && s.abandoned) return Status::Unavailable("state abandoned clone still retires");
    return Status::Ok();
  }
  Status close() noexcept {
    closed = true;
    for (std::uint32_t i = 0; i < slots.size(); ++i) {
      if (slots[i].live) cancel(i).IgnoreError();
      if (!slots[i].observers) slots[i].invocation.close().IgnoreError();
    }
    return active() ? Status::Unavailable("state clone route has native work") : Status::Ok();
  }
  Status drain() noexcept {
    Status first;
    // begin observation pins all slot handles before leaving the store lock.
    for (auto &s : slots) {
      std::lock_guard guard(s.wait_lock);
      const auto status = s.submission.valid() ? s.submission.wait() : s.invocation.drain();
      if (first.ok()) first = status;
    }
    return first;
  }
  Status rearm() noexcept {
    if (closed) return Status::FailedPrecondition("state copy route is closed");
    collect().IgnoreError();
    try {
      for (auto &s : slots) if (s.disabled) {
        if (s.observers || s.live) return Status::Unavailable("state copy actual retirement is unproven");
        HYLE_ASSIGN_OR_RETURN(auto native, std::invoke(factory, std::as_const(program)));
        s.invocation = std::move(native); s.disabled = false; s.abandoned = false;
      }
      return Status::Ok();
    } catch (const std::bad_alloc &) { return Status::ResourceExhausted("state copy cold rearm metadata"); }
    catch (...) { return Status::Internal("state copy cold factory failed"); }
  }
  std::uint32_t active() noexcept { std::uint32_t count = 0; for (const auto &s : slots) count += s.live; return count; }
  Result<CopyBatchReport> report() noexcept {
    for(const auto &s:slots) if(s.observers || s.invocation.in_flight())
      return std::unexpected(Status::Unavailable("state copy report requires retired native work and observers"));
    CopyBatchReport total;
    for(const auto &s:slots) {
      const auto r=s.invocation.report();
      total.resources+=r.resources;total.copies+=r.copies;total.active_copies+=r.active_copies;
      total.phases+=r.phases;total.no_op_copies+=r.no_op_copies;
      total.logical_bytes+=r.logical_bytes;total.physical_bytes+=r.physical_bytes;total.staging_bytes+=r.staging_bytes;
      if(r.metadata_bytes>total.metadata_bytes) total.metadata_bytes=r.metadata_bytes;
      total.native_calls+=r.native_calls;total.native_waits+=r.native_waits;
      total.phase_completions+=r.phase_completions;total.host_waits+=r.host_waits;
      total.native_metadata_unknown|=r.native_metadata_unknown;
    }
    return total;
  }
  StateCopyDriver bridge() noexcept {
    return {this, std::uint32_t(slots.size()),
      [](void *p,const ReadLease &r,WriteLease &&w,std::size_t n) noexcept { return static_cast<TypedStateCopy *>(p)->start(r,std::move(w),n); },
      [](void *p,std::uint32_t i,bool w) noexcept { return static_cast<TypedStateCopy *>(p)->take(i,w); },
      [](void *p,std::uint32_t i) noexcept { return static_cast<TypedStateCopy *>(p)->wait(i); },
      [](void *p,std::uint32_t i,bool a) noexcept { static_cast<TypedStateCopy *>(p)->observe(i,a); },
      [](void *p,std::uint32_t i) noexcept { return static_cast<TypedStateCopy *>(p)->cancel(i); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->collect(); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->close(); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->drain(); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->rearm(); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->active(); },
      [](void *p) noexcept { return static_cast<TypedStateCopy *>(p)->report(); },
      [](void *p) noexcept { auto *self=static_cast<TypedStateCopy *>(p);const auto a=self->allocator;a.destroy(self); }};
  }
};
} // namespace detail

// Program/Invocation/Submission are the original CPU/CUDA, Apple or Vulkan
// F2 types. The cold factory's captures, resource and original Queue/Context
// stay alive at stable addresses through the route and actual native uses.
template<class Program, class Factory>
Result<StateCloneRoute> StateCloneRoute::prepare(const StateStore &store, StateCloneOptions o,
                                                const Program &program, Factory factory) noexcept {
  static_assert(std::is_trivially_destructible_v<Factory>,
      "state copy cold factories borrow native endpoints; do not capture owning state handles");
  if (!o.resource.valid() || !o.capacity_bytes || !o.slots || o.slots == UINT32_MAX ||
      !o.alignment || (o.alignment & (o.alignment-1)))
    return std::unexpected(Status::InvalidArgument("invalid state clone capacity/resource/alignment"));
  HYLE_ASSIGN_OR_RETURN(auto allocator, detail::StateCloneAccess::allocator(store));
  HYLE_RETURN_IF_ERROR(detail::StateCloneAccess::preflight(store,o.slots,o.capacity_bytes));
  using Native = detail::TypedStateCopy<Program,Factory>;
  try {
    auto owner = allocator.make_unique<Native>(program,std::move(factory),allocator);
    if (!owner) return std::unexpected(Status::ResourceExhausted("state clone route metadata"));
    owner->slots.reserve(o.slots);
    for (std::uint32_t i=0;i<o.slots;++i) {
      HYLE_ASSIGN_OR_RETURN(auto slot,std::invoke(owner->factory,std::as_const(owner->program)));
      owner->slots.emplace_back(std::move(slot));
    }
    auto driver=owner->bridge();
    HYLE_ASSIGN_OR_RETURN(auto route,detail::StateCloneAccess::register_driver(store,driver,o.resource,o.capacity_bytes,o.alignment));
    owner.release();return route;
  } catch (const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("state clone cold metadata")); }
  catch (...) { return std::unexpected(Status::Internal("state clone cold factory failed")); }
}
} // namespace hyle::compute
