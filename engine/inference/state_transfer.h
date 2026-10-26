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
#include <variant>
#include <vector>

#include "backend.h"
#include "execution_resources.h"
#include "retirement_observation.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/copy_stream_access.h"
#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/detail/apple_copy_stream_access.h"
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#endif
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

// Cold, single-use ownership of one original endpoint's CopyBatch episode.
// prepare consumes destinations only on success; sources remain unchanged.
// submit/wait return the original status without discarding this owner. A
// successful wait permits take_writer; no bytes become ready here implicitly.
// lifetime owns ALL borrowed resource/allocator backings, not the application
// Model. It is held only by this temporary transfer, never copied into an image.
class StateTransfer final {
  // allocate_shared frees its control block after destroying Scope. Keep the
  // caller's real allocator owner through that final original allocation too.
  template<class T> struct ScopeAllocator final {
    using value_type = T;
    using is_always_equal = std::false_type;
    template<class U> struct rebind { using other = ScopeAllocator<U>; };
    ScopeAllocator(mem::AllocatorRef allocator, std::shared_ptr<void> retained) noexcept
        : backing(allocator), lifetime(std::move(retained)) {}
    template<class U> ScopeAllocator(const ScopeAllocator<U> &other) noexcept
        : backing(other.backing), lifetime(other.lifetime) {}
    T *allocate(std::size_t count) { return backing.allocate(count); }
    void deallocate(T *pointer, std::size_t count) noexcept { backing.deallocate(pointer, count); }
    template<class U> bool operator==(const ScopeAllocator<U> &other) const noexcept {
      return backing == other.backing;
    }
    mem::StdAllocator<T> backing;
    std::shared_ptr<void> lifetime;
  };
  struct CommonControl final {
    compute::detail::ContextRef context;
    compute::Queue queue;
    compute::CopyBatchProgram program;
    compute::CopyBatchInvocation invocation;
    compute::CopyBatchUpdate update;
  };
  template<class Program, class Invocation, class Update>
  struct NativeControl final {
    Program program;
    Invocation invocation;
    Update update;
  };
#if defined(HYLE_COMPUTE_APPLE)
  using AppleControl = NativeControl<compute::apple::CopyBatchProgram,
      compute::apple::CopyBatchInvocation, compute::apple::CopyBatchUpdate>;
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  using VulkanControl = NativeControl<compute::vulkan::CopyBatchProgram,
      compute::vulkan::CopyBatchInvocation, compute::vulkan::CopyBatchUpdate>;
#endif
  using Control = std::variant<CommonControl
#if defined(HYLE_COMPUTE_APPLE)
      , AppleControl
#endif
#if defined(HYLE_COMPUTE_VULKAN)
      , VulkanControl
#endif
      >;
  struct Scope final {
    explicit Scope(std::shared_ptr<void> retained) noexcept : lifetime(std::move(retained)) {}
    // Reverse destruction keeps borrowed resources after original controls.
    // Native Program already retains its exact Runtime; no Queue shell exists.
    std::shared_ptr<void> lifetime;
    Control control;
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
    auto allocator = mem::system_allocator();
    if (backend.kind() == BackendKind::kCpu || backend.kind() == BackendKind::kCuda) {
      HYLE_ASSIGN_OR_RETURN(auto *context, backend.common_context());
      allocator = context->metadata_allocator();
    }
    // The original Apple/Vulkan copy APIs default to system metadata. A real
    // budget replaces that allocator for every transfer-owned metadata object.
    if (budget) { HYLE_ASSIGN_OR_RETURN(allocator, budget.metadata_allocator()); }
    try {
      const ScopeAllocator<Scope> scope_allocator(allocator, lifetime);
      auto scope = std::allocate_shared<Scope>(scope_allocator, std::move(lifetime));
      scope->copies = copies.size();
      StateTransfer result(std::move(scope));
      auto &owner = *result.scope_;
      HYLE_RETURN_IF_ERROR(select_control(backend, owner));
      if (copies.empty()) return result;
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
      const compute::CopyBatchOptions options{
          .max_resources = static_cast<std::uint32_t>(resources.size()),
          .max_copies = static_cast<std::uint32_t>(copies.size()), .max_phases = 1,
          .max_bytes = total, .allocator = allocator, .resource_budget = std::move(budget)};
      HYLE_RETURN_IF_ERROR(prepare_control(backend, owner, resources, options));
      const auto status = std::visit([&](auto &control) noexcept {
        return bind_copies(control.invocation, copies, ranges);
      }, owner.control);
      if (!status.ok()) return std::unexpected(status);
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("state transfer cold metadata"));
    }
  }

  bool valid() const noexcept { return bool(scope_); }
  bool in_flight() const noexcept {
    return scope_ && (scope_->retirement.pending() || std::visit([](const auto &control) noexcept {
      return control.update.in_flight() || control.invocation.in_flight();
    }, scope_->control));
  }
  compute::CopyBatchReport report() const noexcept {
    if (!scope_) return {};
    return std::visit([](const auto &control) noexcept {
      return control.update.valid() ? control.update.report() : control.invocation.report();
    }, scope_->control);
  }
  Status submit() noexcept {
    if (!scope_ || scope_->attempted)
      return Status::FailedPrecondition("state transfer is empty or was already submitted");
    auto &owner = *scope_; owner.attempted = true;
    if (!owner.copies) return Status::Ok();
    const auto reserved = owner.retirement.reserve(scope_);
    if (!reserved.ok()) { owner.terminal = reserved; return owner.terminal; }
    return std::visit([&](auto &control) noexcept -> Status {
      auto submitted = control.invocation.submit();
      if (!submitted) {
        owner.retirement.reject(); owner.terminal = submitted.error(); return owner.terminal;
      }
      using Update = std::remove_cvref_t<decltype(control.update)>;
      auto update = Update::take(std::move(control.invocation), std::move(*submitted));
      HYLE_FATAL_ASSERT(update.has_value()); control.update = std::move(*update);
      attach_retirement(owner, control);
      return Status::Ok();
    }, owner.control);
  }
  Status wait() noexcept {
    if (!scope_ || !scope_->attempted)
      return Status::FailedPrecondition("state transfer has not been submitted");
    auto &owner = *scope_;
    std::visit([&](auto &control) noexcept {
      if (control.update.valid()) record_retirement_status(owner.terminal, control.update.wait());
    }, owner.control);
    owner.successful_wait = owner.terminal.ok() && !in_flight();
    return owner.terminal;
  }
  Result<compute::WriteLease> take_writer(std::size_t index) noexcept {
    if (!scope_ || index >= scope_->copies)
      return std::unexpected(Status::OutOfRange("state transfer destination index"));
    if (!scope_->terminal.ok()) return std::unexpected(scope_->terminal);
    if (!scope_->successful_wait || in_flight())
      return std::unexpected(Status::FailedPrecondition("state transfer destination requires actual successful retirement"));
    return std::visit([&](auto &control) noexcept {
      return control.update.take_mutable({static_cast<std::uint32_t>(2 * index + 1)});
    }, scope_->control);
  }

private:
  static Status select_control(Backend &backend, Scope &owner) noexcept {
    switch (backend.kind()) {
    case BackendKind::kCpu:
    case BackendKind::kCuda: {
      HYLE_ASSIGN_OR_RETURN(auto *context, backend.common_context());
      HYLE_ASSIGN_OR_RETURN(std::get<CommonControl>(owner.control).context,
          compute::detail::ContextRef::acquire(*context));
      return {};
    }
#if defined(HYLE_COMPUTE_APPLE)
    case BackendKind::kApple: {
      HYLE_ASSIGN_OR_RETURN(auto *context, backend.apple_context());
      if (!context->valid()) return Status::FailedPrecondition("state transfer Apple Context is empty");
      owner.control.emplace<AppleControl>(); return {};
    }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
    case BackendKind::kVulkan: {
      HYLE_ASSIGN_OR_RETURN(auto *context, backend.vulkan_context());
      if (!context->valid()) return Status::FailedPrecondition("state transfer Vulkan Context is empty");
      owner.control.emplace<VulkanControl>(); return {};
    }
#endif
    default: return Status::Unimplemented("state transfer endpoint is unavailable");
    }
  }
  static Status prepare_control(Backend &backend, Scope &owner,
      std::span<const compute::CopyResourceSpec> resources,
      const compute::CopyBatchOptions &options) noexcept {
    (void)backend; // Common-only builds use the already acquired ContextRef.
    return std::visit([&](auto &control) noexcept -> Status {
      using Selected = std::remove_cvref_t<decltype(control)>;
      if constexpr (std::is_same_v<Selected, CommonControl>) {
        HYLE_ASSIGN_OR_RETURN(control.queue, control.context.create_queue());
        HYLE_ASSIGN_OR_RETURN(control.program,
            compute::detail::CopyStreamAccess::prepare(control.context, resources, options));
        HYLE_ASSIGN_OR_RETURN(control.invocation,
            compute::CopyBatchInvocation::create(control.program, control.queue));
      }
#if defined(HYLE_COMPUTE_APPLE)
      else if constexpr (std::is_same_v<Selected, AppleControl>) {
        HYLE_ASSIGN_OR_RETURN(auto *context, backend.apple_context());
        HYLE_ASSIGN_OR_RETURN(control.program, compute::apple::CopyBatchProgram::prepare(*context, resources, options));
        HYLE_ASSIGN_OR_RETURN(control.invocation, compute::apple::CopyBatchInvocation::create(control.program));
      }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
      else if constexpr (std::is_same_v<Selected, VulkanControl>) {
        HYLE_ASSIGN_OR_RETURN(auto *context, backend.vulkan_context());
        HYLE_ASSIGN_OR_RETURN(control.program, compute::vulkan::CopyBatchProgram::prepare(*context, resources, options));
        HYLE_ASSIGN_OR_RETURN(control.invocation, compute::vulkan::CopyBatchInvocation::create(control.program));
      }
#endif
      return {};
    }, owner.control);
  }
  template<class Invocation>
  static Status bind_copies(Invocation &invocation, std::span<StateTransferCopy> copies,
      std::span<const compute::CopyRange> ranges) noexcept {
    std::size_t bound = 0;
    Status status;
    for (std::size_t i = 0; i < copies.size(); ++i) {
      status = invocation.bind_source({static_cast<std::uint32_t>(2 * i)}, copies[i].source);
      if (!status.ok()) break;
      status = invocation.bind_mutable({static_cast<std::uint32_t>(2 * i + 1)}, std::move(copies[i].destination));
      if (!status.ok()) break;
      ++bound;
    }
    constexpr std::array<compute::CopyPhase, 1> phases{};
    if (status.ok()) status = invocation.prepare_replay(phases, ranges);
    if (!status.ok()) for (std::size_t i = 0; i < bound; ++i) {
      // No producer, submit or native work exists. Restore every successfully
      // bound original writer before a preparation failure releases controls.
      auto writer = invocation.unbind_mutable({static_cast<std::uint32_t>(2 * i + 1)});
      HYLE_FATAL_ASSERT(writer.has_value()); copies[i].destination = std::move(*writer);
    }
    return status;
  }
  template<class Access, class Update>
  static void attach_original(Scope &owner, Update &update) noexcept {
    // Exactly one phase, no producer bindings and a sole move-only observer.
    std::array<compute::Completion *, 1> original{};
    const auto count = Access::retirement_sources(update, original);
    HYLE_FATAL_ASSERT(count && *count == 1);
    HYLE_FATAL_ASSERT(owner.retirement.attach(*original[0]).ok());
  }
  static void attach_retirement(Scope &owner, CommonControl &control) noexcept {
    attach_original<compute::detail::CopyStreamAccess>(owner, control.update);
  }
#if defined(HYLE_COMPUTE_APPLE)
  static void attach_retirement(Scope &owner, AppleControl &control) noexcept {
    struct Attachment { Scope &owner; compute::apple::CopyBatchUpdate &update; } attachment{owner, control.update};
    // The original FIFO preflight can observe the same Completion. Both the
    // source lookup and attachment must share its recursive observation lock.
    compute::apple::detail::CopyStreamAccess::serialize(control.update, +[](void *opaque) noexcept {
      auto &call = *static_cast<Attachment *>(opaque);
      attach_original<compute::apple::detail::CopyStreamAccess>(call.owner, call.update);
    }, &attachment);
  }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  static void attach_retirement(Scope &owner, VulkanControl &control) noexcept {
    attach_original<compute::vulkan::detail::CopyStreamAccess>(owner, control.update);
  }
#endif
  explicit StateTransfer(std::shared_ptr<Scope> scope) noexcept : scope_(std::move(scope)) {}
  void abandon() noexcept {
    auto held = std::move(scope_);
    if (!held) return;
    // Break the user-observation cycle explicitly. Dropping both original F2
    // controls hands late completions to their native runtime; its attached
    // RetainedResources keeps this entire stable Scope/Queue alive until real
    // retirement. No wait, speculative release or permanent self-cycle.
    std::visit([](auto &control) noexcept { control.update = {}; control.invocation = {}; }, held->control);
  }
  std::shared_ptr<Scope> scope_;
};

} // namespace hyle::inference
