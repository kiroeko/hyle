#include "../../engine/inference/host_state_storage.h"
#include "../../engine/inference/state_transfer.h"
#include "../../src/compute/context_state.h"
#include "../../src/compute/completion_internal.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
namespace i = hyle::inference;
namespace c = hyle::compute;
using hyle::Status;

void check(Status status) {
  if (!status.ok()) {
    char text[1024];
    status.ToString(text, sizeof(text));
    throw std::runtime_error(text);
  }
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) check(result.error());
  return std::move(*result);
}
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

// All borrowed resources and pools outlive the original native CopyBatch.
// The escaped host lease deliberately retains no Backend or transfer scope.
struct TransferResources final {
  std::shared_ptr<i::Backend> backend;
  std::shared_ptr<i::HostStateStorage> storage;
  c::LeasePool source, host, restored;
};

#if defined(HYLE_COMPUTE_CPU)
namespace cd = c::detail;

// The original CPU copies really execute. Only their Completion observation
// and retirement callback are controlled, as in Compute's AsyncProbe tests.
// This proves the adapter protocol, not pending GPU work or a driver failure.
struct ManualRetirement final {
  bool released = false, unknown = false, cancel_supported = false, cancelled = false;
  unsigned waits = 0, polls = 0, destroys = 0, callbacks = 0, cancels = 0;
  cd::CompletionLifetime deferred{};
  Status hint;
  void finish(cd::CompletionLifetime lifetime, Status terminal) noexcept {
    if (!lifetime.valid()) return;
    if (cancelled) terminal.Update(Status::Cancelled("controlled CPU cancellation"));
    ++callbacks; lifetime.terminal(lifetime.state, terminal); lifetime.release(lifetime.state);
  }
  void release() noexcept {
    released = true;
    const auto lifetime = std::exchange(deferred, {});
    finish(lifetime, hint);
  }
  ~ManualRetirement() { release(); } // Exception cleanup, without waiting or executing payload.
  static const cd::CompletionVTable table;
};

inline const cd::CompletionVTable ManualRetirement::table{
      .destroy = +[](void *opaque, cd::CompletionLifetime lifetime, Status terminal) noexcept {
        auto &self = *static_cast<ManualRetirement *>(opaque); ++self.destroys;
        if (self.released) self.finish(lifetime, std::move(terminal));
        else { self.deferred = lifetime; self.hint = std::move(terminal); }
      },
      .poll = +[](void *opaque) noexcept -> hyle::Result<c::CompletionPoll> {
        auto &self = *static_cast<ManualRetirement *>(opaque); ++self.polls;
        if (!self.released && self.unknown)
          return std::unexpected(Status::Unavailable("controlled CPU observation is unknown"));
        if (!self.released) return c::CompletionPoll::kPending;
        if (self.cancelled) return std::unexpected(Status::Cancelled("controlled CPU cancellation"));
        return c::CompletionPoll::kComplete;
      },
      .wait = +[](void *opaque) noexcept {
        auto &self = *static_cast<ManualRetirement *>(opaque); ++self.waits;
        if (!self.released) return Status::Unavailable("controlled CPU retirement is still unknown");
        return self.cancelled ? Status::Cancelled("controlled CPU cancellation") : Status::Ok();
      },
      .request_cancel = +[](void *opaque) noexcept {
        auto &self = *static_cast<ManualRetirement *>(opaque); ++self.cancels;
        if (!self.cancel_supported) return Status::Unimplemented("controlled CPU cancellation is unsupported");
        self.cancelled = true; return Status::Ok();
      }};

struct CpuFaultBoundary final {
  cd::ContextRef context_owner;
  cd::ContextState *context;
  const cd::BackendVTable *original;
  cd::BackendVTable table;
  ManualRetirement retirement;
  unsigned calls = 0, accepted = 0, reject_at = 0, sequences = 0;
  bool delayed = false;
  static inline thread_local CpuFaultBoundary *active = nullptr;
  explicit CpuFaultBoundary(c::Context &selected)
      : context_owner(take(cd::ContextRef::acquire(selected))), context(cd::ContextAccess::state(selected)),
        original(context->backend_vtable), table(*original) {
    require(!active, "CPU transfer fault boundaries must not overlap");
    table.copy_async = +[](void *queue, hyle::mem::BufferView destination,
        hyle::mem::ConstBufferView source) noexcept {
      auto &self = *active; ++self.calls;
      if (self.reject_at == self.calls)
        return cd::EnqueueResult::rejected(Status::ResourceExhausted("controlled CPU copy rejection"));
      auto result = self.original->copy_async(queue, destination, source);
      self.accepted += result.accepted; return result;
    };
    table.finish_sequence = +[](void *queue, void *sequence, bool accepted, Status terminal) noexcept {
      auto &self = *active; ++self.sequences;
      auto completion = self.original->finish_sequence(queue, sequence, accepted, std::move(terminal));
      if (!self.delayed || !accepted) return completion;
      // CPU finish is synchronous. Preserve any real error before replacing
      // only its successful observation with the controlled original callback.
      const auto status = completion.wait();
      if (!status.ok()) return c::Completion::completed(status);
      return cd::CompletionAccess::make(&ManualRetirement::table, &self.retirement);
    };
    active = this; context->backend_vtable = &table;
  }
  ~CpuFaultBoundary() {
    // Failure cleanup also releases every saved callback before restoring the
    // original endpoint. context_owner keeps this exact ContextState alive.
    retirement.release(); context->backend_vtable = original; active = nullptr;
  }
};

struct FaultResources final {
  std::shared_ptr<i::Backend> backend;
  std::shared_ptr<i::HostStateStorage> storage;
  std::shared_ptr<const void> image;
  c::LeasePool sources, destinations;
};

enum class FaultMode { kPending, kAcceptedPrefix, kUnknown, kAbandon, kRejectedFirst,
                       kCancelSupported, kCancelUnsupported };

void fault_case(FaultMode mode) {
  constexpr std::size_t prefix = 64, extent = 128;
  auto resources = std::make_shared<FaultResources>();
  resources->backend = std::make_shared<i::Backend>(take(i::Backend::create({.primary = i::BackendKind::kCpu})));
  resources->storage = take(i::HostStateStorage::create(*resources->backend, 1, 1u << 20));
  resources->image = take(resources->storage->reserve_image());
  const auto budget = resources->storage->budget();
  resources->sources = take(c::LeasePool::create(resources->backend->device(), 256, {.max_blocks = 1}));
  resources->destinations = take(c::LeasePool::create(resources->storage->resource(), 1024,
      {.max_blocks = 3, .resource_budget = budget}));
  std::array<std::byte, prefix> expected{};
  for (std::size_t n = 0; n < prefix; ++n) expected[n] = std::byte((n * 37 + 11) & 255);
  auto writer = take(resources->sources.allocate(prefix));
  check(take(writer.view()).write(expected)); auto source = take(writer.publish());
  const auto count = mode == FaultMode::kAcceptedPrefix ? std::size_t{3} : std::size_t{1};
  std::array<i::StateTransferCopy, 3> copies;
  for (std::size_t n = 0; n < count; ++n) {
    auto destination = take(resources->destinations.allocate(extent));
    check(take(destination.view()).fill(std::byte{0xcd}));
    copies[n] = {source, std::move(destination), prefix};
  }
  const auto generation = copies[0].destination.generation();
  const auto baseline_bytes = take(resources->storage->usage()).live_bytes;
  auto *context = take(resources->backend->common_context());
  CpuFaultBoundary boundary(*context);
  boundary.delayed = mode != FaultMode::kAcceptedPrefix && mode != FaultMode::kRejectedFirst;
  boundary.retirement.unknown = mode == FaultMode::kUnknown;
  boundary.retirement.cancel_supported = mode == FaultMode::kCancelSupported;
  boundary.reject_at = mode == FaultMode::kAcceptedPrefix ? 2u : mode == FaultMode::kRejectedFirst ? 1u : 0u;
  auto transfer = take(i::StateTransfer::prepare(*resources->backend, std::span(copies).first(count), resources, budget));
  require(!transfer.take_writer(0), "prepared transfer returned an unpublished writer");
  const auto submitted = transfer.submit();
  if (mode == FaultMode::kRejectedFirst) {
    require(submitted.IsResourceExhausted() && boundary.calls == 1 && !boundary.accepted && !transfer.in_flight(),
        "first rejected CPU copy fabricated accepted work or lost its original error");
    require(transfer.wait().IsResourceExhausted() && !transfer.take_writer(0),
        "rejected CPU transfer published a writer or lost its error");
  } else {
    check(submitted);
    require(boundary.accepted == 1, "CPU transfer did not execute its original accepted copy");
    if (mode == FaultMode::kAcceptedPrefix) {
      require(boundary.calls == 2 && transfer.wait().IsResourceExhausted() && !transfer.in_flight(),
          "rejected suffix executed a third copy or lost accepted-prefix retirement/error");
      for (std::size_t n = 0; n < count; ++n)
        require(!transfer.take_writer(n), "failed accepted CPU prefix published a destination");
    } else {
      require(transfer.in_flight() && resources->sources.stats().active_uses &&
          resources->destinations.stats().active_uses && !transfer.take_writer(0) && !boundary.retirement.callbacks,
          "controlled pending transfer released permissions or returned an early writer");
      const auto pending = transfer.poll();
      require(!pending.retired && pending.status.IsUnavailable() && transfer.in_flight() && !transfer.take_writer(0),
          "nonblocking observation turned pending/unknown work into retirement or a ready writer");
      if (mode == FaultMode::kUnknown) {
        const auto again = transfer.poll();
        require(!again.retired && again.status.IsUnavailable() && transfer.in_flight() && !transfer.take_writer(0) &&
            boundary.retirement.deferred.valid() && !boundary.retirement.callbacks,
            "unknown observation manufactured retirement or a ready writer");
        boundary.retirement.release();
        const auto failed = transfer.poll();
        require(failed.retired && !transfer.in_flight() && failed.status.IsUnavailable() && !transfer.take_writer(0),
            "later actual callback erased the original unknown error or published failed bytes");
      } else if (mode == FaultMode::kPending || mode == FaultMode::kCancelUnsupported) {
        if (mode == FaultMode::kCancelUnsupported) {
          require(transfer.request_cancel().IsUnimplemented() && boundary.retirement.cancels == 1 &&
              !boundary.retirement.cancelled && transfer.in_flight() && !transfer.take_writer(0),
              "unsupported cancellation changed the original transfer or returned its writer");
        }
        boundary.retirement.release();
        const auto completed = transfer.poll(); check(completed.status);
        require(completed.retired && !boundary.retirement.waits,
            "successful nonblocking retirement required a wait or retained native work");
        auto restored = take(transfer.take_writer(0));
        require(restored.size() == extent && restored.generation() == generation && !transfer.take_writer(0),
            "successful callback lost the original complete writer or returned it twice");
        std::array<std::byte, extent> actual{}; check(take(restored.view()).read(actual));
        require(std::equal(expected.begin(), expected.end(), actual.begin()) &&
            std::all_of(actual.begin() + prefix, actual.end(), [](auto b) { return b == std::byte{0xcd}; }),
            "controlled success changed copied bytes or the untouched suffix");
      } else if (mode == FaultMode::kCancelSupported) {
        check(transfer.request_cancel());
        const auto cancelling = transfer.poll();
        require(boundary.retirement.cancels == 1 && !cancelling.retired && transfer.in_flight() &&
            !transfer.take_writer(0) && !boundary.retirement.callbacks,
            "accepted cancellation claimed retirement before its original callback");
        boundary.retirement.release();
        const auto cancelled = transfer.poll();
        require(cancelled.retired && cancelled.status.IsCancelled() && !transfer.in_flight() && !transfer.take_writer(0),
            "retired cancellation lost its status or published a destination");
      } else {
        auto storage = resources->storage;
        std::weak_ptr<FaultResources> alive = resources;
        for (auto &copy : copies) copy.source.reset();
        source.reset(); resources.reset();
        transfer = {};
        {
          const auto retained = alive.lock();
          require(retained && storage->live_images() == 1 && boundary.retirement.deferred.valid() &&
              !boundary.retirement.waits && !boundary.retirement.callbacks && retained->sources.stats().active_uses &&
              retained->destinations.stats().active_uses && take(storage->usage()).live_bytes > baseline_bytes,
              "abandon waited or released the original resource/allocator/image before its callback");
        }
        boundary.retirement.release();
        require(alive.expired() && !storage->live_images() && take(storage->usage()).live_bytes <= baseline_bytes,
            "actual callback did not release the abandoned scope and its original budget/image uses");
      }
    }
  }
  if (resources) {
    std::array<std::byte, prefix> unchanged{}; check(take(source.view()).read(unchanged));
    require(unchanged == expected, "failed or successful transfer changed its readonly source");
    require(!resources->sources.stats().active_uses && !resources->destinations.stats().active_uses,
        "actually retired CPU transfer retained native use permissions");
  }
}

void duplicate_attachment() {
  auto owner = std::make_shared<unsigned>(7);
  std::weak_ptr<unsigned> alive = owner;
  i::RetainedResources<unsigned> retained;
  ManualRetirement retirement;
  auto completion = cd::CompletionAccess::make(&ManualRetirement::table, &retirement);
  check(retained.reserve(owner)); check(retained.attach(completion));
  owner.reset();
  require(retained.attach(completion).IsFailedPrecondition() && retained.pending() && !alive.expired() &&
      !retirement.callbacks, "duplicate carrier attachment rejected its original retained owner");
  retirement.release();
  require(take(completion.poll()) == c::CompletionPoll::kComplete && !retained.pending() && alive.expired() &&
      retirement.callbacks == 1, "original carrier attachment did not release once at its actual callback");
}

void closed_control_admission() {
  auto context = take(c::Context::create(c::cpu::device()));
  auto queue = take(context.create_queue());
  auto pool = take(c::LeasePool::create(hyle::mem::system_resource(), 512, {.max_blocks = 2}));
  auto source_writer = take(pool.allocate(64));
  check(take(source_writer.view()).fill(std::byte{0x3a}));
  auto source = take(source_writer.publish());
  auto destination = take(pool.allocate(128));
  const auto generation = destination.generation();
  const std::array specs{c::CopyResourceSpec{hyle::mem::system_resource(), 64},
                         c::CopyResourceSpec{hyle::mem::system_resource(), 128}};
  auto program = take(c::CopyBatchProgram::prepare(context, specs));
  auto invocation = take(c::CopyBatchInvocation::create(program, queue));
  check(invocation.bind_source({0}, source));
  check(invocation.bind_mutable({1}, std::move(destination)));
  constexpr std::array<c::CopyPhase, 1> phases{};
  constexpr std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 64}}};
  check(invocation.prepare_replay(phases, ranges));
  auto submitted = take(invocation.submit()); check(submitted.wait());
  check(submitted.request_cancel()); // Close only after known successful retirement.
  submitted = {};
  const auto replay = invocation.prepare_replay(phases, ranges);
  const auto resubmitted = invocation.submit();
  require(replay.IsFailedPrecondition() && replay.message().find("closed") != std::string_view::npos &&
      !resubmitted && resubmitted.error().IsFailedPrecondition() &&
      resubmitted.error().message().find("closed") != std::string_view::npos,
      "closed CopyBatch prepare/submit did not reject at the original closed admission gate");
  auto recovered = take(invocation.unbind_mutable({1}));
  require(recovered.size() == 128 && recovered.generation() == generation,
      "closed successful control did not recover its original complete writer");
  require(!invocation.bind_source({0}, source).ok() &&
      !invocation.bind_mutable({1}, std::move(recovered)).ok() && recovered.valid() &&
      recovered.generation() == generation && !invocation.prepare_replay(phases, ranges).ok() &&
      !invocation.submit(), "writer recovery reopened closed CopyBatch admission or consumed a rejected writer");
}

void public_retirement_sources() {
  auto context = take(c::Context::create(c::cpu::device()));
  auto queue = take(context.create_queue());
  auto pool = take(c::LeasePool::create(hyle::mem::system_resource(), 1024, {.max_blocks = 4}));
  auto source_writer = take(pool.allocate(64));
  check(take(source_writer.view()).fill(std::byte{0x3a}));
  auto source = take(source_writer.publish());
  auto unused_writer = take(pool.allocate(64));
  check(take(unused_writer.view()).fill(std::byte{0x5b}));
  auto unused = take(unused_writer.publish());
  const std::array specs{c::CopyResourceSpec{hyle::mem::system_resource(), 64},
      c::CopyResourceSpec{hyle::mem::system_resource(), 64},
      c::CopyResourceSpec{hyle::mem::system_resource(), 64}};
  auto program = take(c::CopyBatchProgram::prepare(context, specs));
  const auto make_update = [&](bool producers, bool two_phases) {
    auto invocation = take(c::CopyBatchInvocation::create(program, queue));
    auto destination = take(pool.allocate(64));
    check(invocation.bind_source({0}, source));
    check(invocation.bind_mutable({1}, std::move(destination)));
    if (producers) {
      check(invocation.bind_source({2}, unused));
      check(invocation.bind_producer({0}, c::Completion::completed()));
      check(invocation.bind_producer({2}, c::Completion::completed()));
    }
    constexpr std::array<c::CopyPhase, 2> phases{{{}, {.dependency_mask = 1}}};
    const std::array<c::CopyRange, 2> ranges{{
        {{0}, {1}, {0}, 0, 0, two_phases ? 32u : 64u},
        {{0}, {1}, {1}, 32, 32, 32}}};
    const auto count = two_phases ? std::size_t{2} : std::size_t{1};
    check(invocation.prepare_replay(std::span(phases).first(count), std::span(ranges).first(count)));
    auto submitted = take(invocation.submit());
    return take(c::CopyBatchUpdate::take(std::move(invocation), std::move(submitted)));
  };
  const auto release = +[](void *opaque) noexcept { ++*static_cast<unsigned *>(opaque); };
  {
    auto update = make_update(true, true);
    // One producer used by both accepted phases occurs once; the other bound
    // producer is unused and must not enter the episode's source inventory.
    require(take(update.retirement_count()) == 3, "public copy retirement source inventory is not exact");
    std::array<c::CompletionRetention, 3> nodes;
    std::array<unsigned, 3> released{};
    unsigned rejected = 0;
    c::CompletionRetention invalid;
    require(update.attach_retention(3, invalid, &rejected, release).IsOutOfRange() && !rejected &&
        update.attach_retention(0, invalid, nullptr, release).IsInvalidArgument() &&
        update.attach_retention(0, invalid, &rejected, nullptr).IsInvalidArgument() && !rejected,
        "rejected public attachment consumed a reference or accepted an invalid source");
    for (std::size_t at = 0; at < nodes.size(); ++at)
      check(update.attach_retention(at, nodes[at], &released[at], release));
    require(released == std::array<unsigned, 3>{1, 1, 1}, "inline copy sources did not release distinct nodes once");
    check(update.wait());
  }
  {
    struct SelfOwned final {
      c::CopyBatchUpdate update;
      c::CompletionRetention node;
      unsigned *released;
    };
    unsigned released = 0;
    auto *owner = new SelfOwned{make_update(true, true), {}, &released};
    check(owner->update.attach_retention(0, owner->node, owner, +[](void *opaque) noexcept {
      auto *retained = static_cast<SelfOwned *>(opaque);
      ++*retained->released;
      delete retained;
    }));
    require(released == 1, "inline copy attachment did not safely release its entire Update owner");
  }
  {
    CpuFaultBoundary boundary(context); boundary.delayed = true;
    auto update = make_update(false, false);
    require(take(update.retirement_count()) == 1, "single pending copy gained an unrelated source");
    c::CompletionRetention node;
    unsigned released = 0, rejected = 0;
    check(update.attach_retention(0, node, &released, release));
    require(update.attach_retention(0, node, &rejected, release).IsFailedPrecondition() &&
        !released && !rejected && update.in_flight(),
        "duplicate public attachment consumed a reference or released the first pending owner");
    require(update.observe().IsUnavailable() && !released && !boundary.retirement.callbacks,
        "pending public attachment retired on observation alone");
    boundary.retirement.release();
    check(update.observe());
    require(released == 1 && !rejected && !update.in_flight(),
        "original retirement did not release the public attachment exactly once");
  }
  {
    CpuFaultBoundary boundary(context); boundary.delayed = true; boundary.retirement.unknown = true;
    auto update = make_update(false, false);
    c::CompletionRetention original, late;
    unsigned released = 0, rejected = 0;
    check(update.attach_retention(0, original, &released, release));
    require(update.observe().IsUnavailable() && update.in_flight() &&
        boundary.retirement.deferred.valid() && !released,
        "unknown observation did not retain the already attached original owner");
    require(update.attach_retention(0, late, &rejected, release).IsUnavailable() &&
        !released && !rejected && update.in_flight(),
        "late unknown attachment consumed a reference or fabricated inline retirement");
    boundary.retirement.release();
    require(update.observe().IsUnavailable() && !update.in_flight() && released == 1 && !rejected,
        "actual callback lost the original unknown outcome or released a rejected reference");
  }
  {
    CpuFaultBoundary boundary(context); boundary.reject_at = 2;
    auto update = make_update(true, true);
    require(take(update.retirement_count()) == 2,
        "rejected suffix entered the retirement inventory or erased its accepted prefix");
    std::array<c::CompletionRetention, 2> nodes;
    std::array<unsigned, 2> released{};
    for (std::size_t at = 0; at < nodes.size(); ++at)
      check(update.attach_retention(at, nodes[at], &released[at], release));
    require(released == std::array<unsigned, 2>{1, 1} && update.wait().IsResourceExhausted(),
        "known retired failure refused retention or lost its original accepted-prefix error");
  }
}

void run_faults() {
  for (const auto mode : {FaultMode::kPending, FaultMode::kAcceptedPrefix, FaultMode::kUnknown,
          FaultMode::kAbandon, FaultMode::kRejectedFirst, FaultMode::kCancelSupported,
          FaultMode::kCancelUnsupported}) fault_case(mode);
  duplicate_attachment();
  closed_control_admission();
  public_retirement_sources();
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"real_cpu_copy\":true,"
      "\"controlled_completion_protocol\":true,\"pending_success\":true,\"accepted_prefix_failure\":true,"
      "\"unknown_retirement\":true,\"abandon_keeps_scope\":true,\"first_copy_rejection\":true,"
      "\"nonblocking_success\":true,\"supported_cancel_retirement\":true,\"unsupported_cancel_success\":true,"
      "\"duplicate_attachment_keeps_owner\":true,"
      "\"closed_control_admission_rejected\":true,"
      "\"public_copy_retention\":true,"
      "\"early_or_failed_writer_publication\":false,\"GPU_driver_fault\":false}\n";
}
#endif

void run(i::BackendKind kind, bool closed_recovery = false) {
  constexpr std::size_t prefix = 64, extent = 128, limit = 1u << 20;
  std::array<std::byte, extent> expected{};
  expected.fill(std::byte{0xcd});
  for (std::size_t n = 0; n < prefix; ++n) expected[n] = std::byte((n * 37 + 11) & 255);
  std::shared_ptr<i::HostStateStorage> escaped_storage;
  c::ReadLease escaped;
  std::weak_ptr<i::Backend> old_backend;
  bool export_pending = false, import_pending = false;
  {
    auto resources = std::make_shared<TransferResources>();
    resources->backend = std::make_shared<i::Backend>(take(i::Backend::create({.primary = kind})));
    auto &backend = *resources->backend;
    old_backend = resources->backend;
    resources->storage = take(i::HostStateStorage::create(backend, 2, limit));
    check(resources->storage->validate_backend(backend));
    {
      auto foreign = take(i::Backend::create({.primary = kind}));
      require(resources->storage->validate_backend(foreign).IsInvalidArgument(),
          "independent Backends shared a host-storage identity");
    }
    const auto budget = resources->storage->budget();
    const auto allocator = take(budget.metadata_allocator());
    resources->source = take(c::LeasePool::create(backend.device(), 256, {.max_blocks = 1}));
    resources->restored = take(c::LeasePool::create(backend.device(), 256, {.max_blocks = 1}));
    resources->host = take(c::LeasePool::create(resources->storage->resource(), 256,
        {.max_blocks = 1, .allocator = allocator, .resource_budget = budget}));
    auto source_writer = take(resources->source.allocate(prefix));
    check(take(source_writer.view()).write(std::span(expected).first(prefix)));
    auto source = take(source_writer.publish());
    auto destination = take(resources->host.allocate(extent));
    check(take(destination.view()).fill(std::byte{0xcd}));
    const auto generation = destination.generation();
    std::array<i::StateTransferCopy, 1> copies{{{source, std::move(destination), prefix + 1}}};
    auto invalid = i::StateTransfer::prepare(backend, copies, resources, budget);
    require(!invalid && invalid.error().IsInvalidArgument() && copies[0].destination.valid() &&
        copies[0].destination.size() == extent && copies[0].destination.generation() == generation,
        "invalid initialized prefix consumed the original destination writer");
    copies[0].bytes = prefix;
    auto transfer = take(i::StateTransfer::prepare(backend, copies, resources, budget));
    require(!copies[0].destination.valid(), "successful preparation did not consume its destination");
    require(!transfer.take_writer(0), "destination escaped before successful native retirement");
    check(transfer.submit());
    export_pending = transfer.in_flight();
    require(transfer.submit().IsFailedPrecondition(), "single-use transfer accepted a second submit");
    check(transfer.wait());
    require(!transfer.in_flight(), "successful wait left original export work pending");
    if (closed_recovery) check(transfer.request_cancel());
    auto host_writer = take(transfer.take_writer(0));
    require(host_writer.size() == extent && host_writer.generation() == generation,
        "export did not return its original complete writer");
    require(!transfer.take_writer(0), "export returned the same writer twice");
    auto host = take(host_writer.publish());
    std::array<std::byte, extent> actual{};
    check(take(host.view()).read(actual));
    require(actual == expected, "export changed its initialized suffix or copied incorrect bytes");

    auto restored_writer = take(resources->restored.allocate(extent));
    check(take(restored_writer.view()).fill(std::byte{0xef}));
    std::array<i::StateTransferCopy, 1> restore_copies{{{
        take(host.subview(0, prefix)), std::move(restored_writer), prefix}}};
    auto restore = take(i::StateTransfer::prepare(backend, restore_copies, resources, budget));
    check(restore.submit());
    import_pending = restore.in_flight();
    check(restore.wait());
    require(!restore.in_flight(), "successful wait left original import work pending");
    if (closed_recovery) check(restore.request_cancel());
    auto restored = take(restore.take_writer(0));
    require(restored.size() == extent, "import lost its original untouched writer suffix");
    check(take(restored.view()).read(actual));
    require(std::equal(actual.begin(), actual.begin() + prefix, expected.begin()) &&
        std::all_of(actual.begin() + prefix, actual.end(), [](auto byte) { return byte == std::byte{0xef}; }),
        "import copied incorrect bytes or changed its initialized suffix");
    check(take(restored.view()).fill(std::byte{0x5a}));
    check(take(host.view()).read(actual));
    require(actual == expected, "restored device mutation changed its retained host image");
    escaped_storage = resources->storage;
    escaped = std::move(host);
  }
  require(old_backend.expired(), "escaped host storage retained its execution Backend");
  std::array<std::byte, extent> after{};
  check(take(escaped.view()).read(after));
  require(after == expected, "host image failed after original Backend and pools were destroyed");
  escaped = {};
  require(take(escaped_storage->usage()).live_bytes <= limit, "host budget exceeded its requested-byte bound");
  std::cout << "state_transfer_pass prefix_bytes=" << prefix << " writer_bytes=" << extent
            << " export_pending_observed=" << export_pending << " import_pending_observed=" << import_pending
            << " backend_expired=1";
  if (closed_recovery) std::cout << " closed_recovery=1";
  std::cout << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    const bool closed_recovery = argc == 3 && std::string_view(argv[2]) == "--closed-recovery";
    if (argc != 2 && !closed_recovery)
      throw std::runtime_error("usage: inference_state_transfer_test --cpu|--cuda|--apple|--vulkan [--closed-recovery], or --faults");
    const std::string_view selected(argv[1]);
    if (selected == "--faults") {
      if (closed_recovery) throw std::runtime_error("--faults does not take a backend or recovery option");
#if defined(HYLE_COMPUTE_CPU)
      run_faults(); return 0;
#else
      std::cerr << "SKIP: CPU transfer fault fixture is not built\n"; return 77;
#endif
    }
    auto kind = i::BackendKind::kDefault;
    if (selected == "--cpu") kind = i::BackendKind::kCpu;
    else if (selected == "--cuda") kind = i::BackendKind::kCuda;
    else if (selected == "--apple") kind = i::BackendKind::kApple;
    else if (selected == "--vulkan") kind = i::BackendKind::kVulkan;
    else throw std::runtime_error("unknown transfer-test backend");
    const auto compiled = i::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), kind) == compiled.end()) {
      std::cerr << "SKIP: selected route is not built\n"; return 77;
    }
    run(kind, closed_recovery);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
