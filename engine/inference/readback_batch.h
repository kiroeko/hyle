#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

#include "backend.h"
#include "execution_resources.h"
#include "readback.h"
#include "retirement_observation.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"

#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
#include "../../tests/tools/inference_readback_batch_probe.h"
#endif

namespace hyle::inference {

// One cold serial slot for at most two independent completed Graph outputs.
// The original CopyBatch owns all transfer commands, permissions and native
// retirement. Source helpers must stay at their cold addresses through read;
// their pointers are never used by late native callbacks or destruction.
class PreparedReadbackBatch final {
  static constexpr std::size_t kMaxPorts = 2;
  template<class T> struct ScopeAllocator final {
    using value_type = T;
    using is_always_equal = std::false_type;
    template<class U> struct rebind { using other = ScopeAllocator<U>; };
    ScopeAllocator(mem::AllocatorRef allocator, compute::StorageOwner retained,
        HostStorage storage, compute::ResourceBudget charged) noexcept
        : backing(allocator), primary(std::move(retained)), host(std::move(storage)),
          budget(std::move(charged)) {}
    template<class U> ScopeAllocator(const ScopeAllocator<U> &other) noexcept
        : backing(other.backing), primary(other.primary), host(other.host), budget(other.budget) {}
    T *allocate(std::size_t count) { return backing.allocate(count); }
    void deallocate(T *pointer, std::size_t count) noexcept { backing.deallocate(pointer, count); }
    template<class U> bool operator==(const ScopeAllocator<U> &other) const noexcept {
      return backing == other.backing;
    }
    mem::StdAllocator<T> backing;
    // allocate_shared destroys Scope before deallocating its control block.
    compute::StorageOwner primary;
    HostStorage host;
    compute::ResourceBudget budget;
  };
  struct Scope final {
    // Reverse destruction preserves every borrowed resource and Queue address
    // until the original copy controls and their descriptors have gone away.
    compute::StorageOwner primary;
    std::array<compute::StorageOwner, kMaxPorts> source_owners;
    HostStorage host;
    compute::ResourceBudget budget;
    compute::LeasePool host_pool;
    compute::WriteLease host_writer;
    compute::Queue queue;
    compute::CopyBatchProgram program;
    compute::CopyBatchInvocation invocation;
    compute::CopyBatchUpdate update;
    RetainedResources<void> source_retirement;
    RetainedResources<Scope> retirement;
    std::array<const PreparedReadback *, kMaxPorts> sources{};
    std::array<PreparedOutputFacts, kMaxPorts> facts{};
    std::array<compute::CopyResourceSpec, kMaxPorts + 1> resources{};
    std::array<compute::CopyRange, kMaxPorts> ranges{};
    std::size_t ports = 0, bytes = 0;
    compute::CopyBatchReport abandoned_report{};
    Status terminal;
    bool dirty = false, closed = false;
  };

public:
  PreparedReadbackBatch() noexcept = default;
  PreparedReadbackBatch(const PreparedReadbackBatch &) = delete;
  PreparedReadbackBatch &operator=(const PreparedReadbackBatch &) = delete;
  PreparedReadbackBatch(PreparedReadbackBatch &&other) noexcept : scope_(std::move(other.scope_)) {}
  PreparedReadbackBatch &operator=(PreparedReadbackBatch &&other) noexcept {
    if (this != &other) { abandon(); scope_ = std::move(other.scope_); }
    return *this;
  }
  ~PreparedReadbackBatch() { abandon(); }

  // This qualification is cold. Unsupported ownership/layout/backend returns
  // Unimplemented before any transfer; callers may select ordinary readback.
  // Budget rejection is likewise a cold admission outcome, never a hot retry.
  static Result<PreparedReadbackBatch> prepare(Backend &backend,
      std::span<const PreparedReadback *const> sources,
      compute::ResourceBudget budget = {}) noexcept {
    if (backend.kind() != BackendKind::kCpu && backend.kind() != BackendKind::kCuda)
      return std::unexpected(Status::Unimplemented("batched readback requires a CPU or CUDA endpoint"));
    if (sources.empty())
      return std::unexpected(Status::InvalidArgument("batched readback has no output ports"));
    if (sources.size() > kMaxPorts)
      return std::unexpected(Status::Unimplemented("batched readback supports at most two output ports"));
    std::size_t total = 0;
    std::array<compute::CopyResourceSpec, kMaxPorts + 1> resources{};
    std::array<compute::StorageOwner, kMaxPorts> source_owners;
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto *source = sources[i];
      if (!source) return std::unexpected(Status::InvalidArgument("batched readback output is null"));
      const auto &facts = source->facts;
      if (!facts.caller_owned() || !facts.description.is_plain() || !facts.description.is_dense())
        return std::unexpected(Status::Unimplemented("batched readback requires plain dense caller outputs"));
      HYLE_ASSIGN_OR_RETURN(auto bytes, facts.description.required_bytes());
      if (!bytes || !source->pool().valid() || !source->resource().valid())
        return std::unexpected(Status::Unimplemented("batched readback requires nonempty prepared caller storage"));
      if (bytes > (std::numeric_limits<std::size_t>::max)() - total)
        return std::unexpected(Status::OutOfRange("batched readback packet byte count"));
      HYLE_ASSIGN_OR_RETURN(source_owners[i], backend.storage_owner_for(source->resource()));
      resources[i] = {source->resource(), bytes};
      total += bytes;
    }
    HYLE_ASSIGN_OR_RETURN(auto *context, backend.common_context());
    HYLE_ASSIGN_OR_RETURN(auto host, backend.retain_staging_storage());
    resources[sources.size()] = {host.ref(), total};
    auto allocator = mem::system_allocator();
    if (budget) { HYLE_ASSIGN_OR_RETURN(allocator, budget.metadata_allocator()); }
    try {
      const ScopeAllocator<Scope> scope_allocator(allocator, backend.storage_owner(), host, budget);
      auto scope = std::allocate_shared<Scope>(scope_allocator);
      scope->primary = backend.storage_owner();
      scope->source_owners = std::move(source_owners);
      scope->host = std::move(host);
      scope->budget = std::move(budget);
      scope->ports = sources.size(); scope->bytes = total; scope->resources = resources;
      std::size_t offset = 0;
      for (std::size_t i = 0; i < sources.size(); ++i) {
        scope->sources[i] = sources[i]; scope->facts[i] = sources[i]->facts;
        scope->ranges[i] = {{static_cast<std::uint32_t>(i)},
            {static_cast<std::uint32_t>(sources.size())}, {0}, 0, offset, resources[i].capacity_bytes};
        offset += resources[i].capacity_bytes;
      }
      HYLE_ASSIGN_OR_RETURN(scope->host_pool, compute::LeasePool::create(scope->host.ref(), total,
          {.max_blocks = 1, .allocator = allocator, .resource_budget = scope->budget}));
      HYLE_ASSIGN_OR_RETURN(scope->host_writer, scope->host_pool.allocate(total));
      HYLE_ASSIGN_OR_RETURN(scope->queue, context->create_queue());
      HYLE_ASSIGN_OR_RETURN(scope->program, compute::CopyBatchProgram::prepare(*context,
          std::span(resources).first(sources.size() + 1),
          {.max_resources = static_cast<std::uint32_t>(sources.size() + 1),
           .max_copies = static_cast<std::uint32_t>(sources.size()), .max_phases = 1,
           .max_bytes = total, .allocator = allocator, .resource_budget = scope->budget}));
      HYLE_ASSIGN_OR_RETURN(scope->invocation,
          compute::CopyBatchInvocation::create(scope->program, scope->queue));
      return PreparedReadbackBatch(std::move(scope));
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("batched readback cold metadata"));
    }
  }

  bool valid() const noexcept { return bool(scope_); }
  bool in_flight() const noexcept {
    return scope_ && (scope_->retirement.pending() || scope_->source_retirement.pending() ||
        scope_->update.in_flight() ||
        scope_->invocation.in_flight());
  }
  compute::CopyBatchReport report() const noexcept {
    if (!scope_) return {};
    if (scope_->update.valid()) return scope_->update.report();
    return scope_->invocation.valid() ? scope_->invocation.report() : scope_->abandoned_report;
  }

  // Irreversible observation handoff, without waiting or closing the actual
  // Queue. The original runtime/reaper owns all late native retirement; both
  // attached guards remain intact. Keeping this cold scope permits later
  // in_flight/recover queries even after its sole original controls are gone.
  void abandon_observation() noexcept {
    auto held = scope_;
    if (!held || held->closed) return;
    held->abandoned_report = held->update.valid() ? held->update.report() : held->invocation.report();
    held->closed = true;
    // Source release may destroy the Phase containing this helper. Use only
    // the retained Scope after dropping either original observer below.
    held->update = {};
    held->invocation = {};
  }

  // The caller observes the original Graph completion first. Every source is
  // retained from that exact run and revalidated; no source reader survives a
  // successful call. The one copy phase has its own original completion.
  // An accepted failure stays here for recover(), with no synchronous fallback.
  // An optional original source scope is retained only through that same
  // native retirement, including abandonment without an escaped Session.
  Status read(graph::Invocation &run, std::span<const std::span<std::byte>> outputs,
      std::shared_ptr<void> source_lifetime = {}) noexcept {
    if (!scope_) return Status::FailedPrecondition("batched readback is empty");
    auto &owner = *scope_;
    if (owner.closed) return Status::FailedPrecondition("batched readback observation is closed");
    if (outputs.size() != owner.ports)
      return Status::InvalidArgument("batched readback output count differs");
    for (std::size_t i = 0; i < owner.ports; ++i) {
      if (outputs[i].size() != owner.resources[i].capacity_bytes)
        return Status::InvalidArgument("batched readback output byte extent differs");
      const auto &facts = owner.sources[i]->facts;
      const auto &prepared = owner.facts[i];
      if (facts.value != prepared.value || facts.description != prepared.description ||
          facts.storage != prepared.storage || facts.ownership != prepared.ownership ||
          owner.sources[i]->resource() != owner.resources[i].resource)
        return Status::InvalidArgument("batched readback output changed after cold preparation");
    }
    if (owner.dirty || owner.update.valid() || in_flight())
      return Status::Unavailable("batched readback requires original copy retirement and recovery");
    owner.terminal = {};
    if (!owner.host_writer) {
      auto writer = owner.host_pool.allocate(owner.bytes);
      if (!writer) return fail(owner, writer.error());
      owner.host_writer = std::move(*writer);
    }
    owner.dirty = true;
    for (std::size_t i = 0; i < owner.ports; ++i) {
      auto source = owner.sources[i]->retain_storage(run);
      if (!source) return fail(owner, source.error());
      const auto bound = owner.invocation.bind_source({static_cast<std::uint32_t>(i)}, *source);
      if (!bound.ok()) return fail(owner, bound);
    }
    const compute::CopyResourceId destination{static_cast<std::uint32_t>(owner.ports)};
    auto status = owner.invocation.bind_mutable(destination, std::move(owner.host_writer));
    if (!status.ok()) return fail(owner, status);
    constexpr std::array<compute::CopyPhase, 1> phases{};
    status = owner.invocation.prepare_replay(phases, std::span(owner.ranges).first(owner.ports));
    if (!status.ok()) return fail(owner, status);
    status = owner.retirement.reserve(scope_);
    if (!status.ok()) return fail(owner, status);
    if (source_lifetime) {
      status = owner.source_retirement.reserve(source_lifetime);
      if (!status.ok()) { owner.retirement.reject(); return fail(owner, status); }
    }
#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
    status = testing::readback_batch_before_submit(owner.queue);
    if (!status.ok()) {
      if (source_lifetime) owner.source_retirement.reject();
      owner.retirement.reject(); return fail(owner, status);
    }
#endif
    auto submitted = owner.invocation.submit();
    if (!submitted) {
      if (source_lifetime) owner.source_retirement.reject();
      owner.retirement.reject();
      return fail(owner, submitted.error());
    }
    auto update = compute::CopyBatchUpdate::take(std::move(owner.invocation), std::move(*submitted));
    HYLE_FATAL_ASSERT(update.has_value());
    owner.update = std::move(*update);
    const auto count = owner.update.retirement_count();
    HYLE_FATAL_ASSERT(count && *count == 1);
    // Attach the source first, then the self-scope outside it. Source teardown
    // may destroy the Phase owning this batch; the outer attachment must keep
    // this stable Scope and both original nodes alive until that release ends.
    if (source_lifetime)
      HYLE_FATAL_ASSERT(owner.source_retirement.attach(owner.update, 0).ok());
    HYLE_FATAL_ASSERT(owner.retirement.attach(owner.update, 0).ok());
#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
    status = testing::readback_batch_wait(owner.update);
#else
    status = owner.update.wait();
#endif
    if (!status.ok()) return fail(owner, status);
    auto writer = owner.update.take_mutable(destination);
    if (!writer) return fail(owner, writer.error());
    owner.host_writer = std::move(*writer);
    status = release_controls(owner);
    if (!status.ok()) return fail(owner, status);
    auto storage = owner.host_writer.view();
    if (!storage) return fail(owner, storage.error());
    for (std::size_t i = 0; i < owner.ports; ++i) {
      auto part = storage->subview(owner.ranges[i].destination_offset, outputs[i].size());
      if (!part) return fail(owner, part.error());
      status = part->read(outputs[i]);
      if (!status.ok()) return fail(owner, status);
    }
    return {};
  }

  // Nonblocking cleanup through the original CopyBatch observer. The first
  // error is preserved, but an observation error alone never proves native
  // retirement. On real retirement, release every source before Graph cleanup.
  // A failed copy may discard its host writer; the cold pool can supply it on
  // the next read. No result is interpreted or published by this operation.
  RetirementObservation recover(bool wait = false) noexcept {
    if (!scope_) return {true, {}};
    auto &owner = *scope_;
    if (owner.update.valid()) {
      if (wait) record_retirement_status(owner.terminal, owner.update.wait());
      const auto observed = owner.update.observe();
      if (in_flight()) {
        auto status = owner.terminal;
        record_retirement_status(status, observed);
        return {false, status};
      }
      record_retirement_status(owner.terminal, owner.update.observe());
    } else if (in_flight()) {
      return {false, owner.terminal};
    }
    const auto released = release_controls(owner);
    record_retirement_status(owner.terminal, released);
    return {!owner.dirty && !in_flight(), owner.terminal};
  }

private:
  static Status release_controls(Scope &owner) noexcept {
    if (owner.update.valid()) {
      auto invocation = owner.update.release_invocation();
      if (!invocation) return invocation.error();
      owner.invocation = std::move(*invocation);
      owner.update = {};
    }
    if (owner.invocation.valid()) {
      HYLE_RETURN_IF_ERROR(owner.invocation.reset());
    }
    owner.dirty = false;
    return {};
  }
  static Status fail(Scope &owner, Status error) noexcept {
    record_retirement_status(owner.terminal, error);
    // No accepted update means no native prefix escaped submit. Reset only
    // that unsubmitted slot; accepted failures keep their actual controls.
    if (!owner.update.valid() && !owner.retirement.pending() &&
        !owner.source_retirement.pending() && !owner.invocation.in_flight())
      record_retirement_status(owner.terminal, release_controls(owner));
    return owner.terminal;
  }
  explicit PreparedReadbackBatch(std::shared_ptr<Scope> scope) noexcept : scope_(std::move(scope)) {}
  void abandon() noexcept {
    abandon_observation();
    scope_.reset();
  }
  std::shared_ptr<Scope> scope_;
};

} // namespace hyle::inference
