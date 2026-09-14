#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/compute/visibility.h"

namespace hyle::compute {

namespace detail {
struct ProviderInstanceAccess;
} // namespace detail

struct ProviderId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t provider_id = 0;

  constexpr bool valid() const noexcept {
    return namespace_id != 0 || provider_id != 0;
  }
  friend constexpr bool operator==(ProviderId, ProviderId) noexcept = default;
};

struct ProviderIdentity final {
  ProviderId id{};
  std::string_view name{};
  std::string_view version{};

  constexpr bool valid() const noexcept { return id.valid(); }
  friend constexpr bool operator==(const ProviderIdentity &left,
                                   const ProviderIdentity &right) noexcept {
    return left.id == right.id;
  }
};

// One configured provider slot in one Context, not a provider family or a
// portable build/device fingerprint. Values are unique only within one loaded
// Compute runtime lifetime; never persist them or compare across processes.
// This non-owning diagnostic value remains comparable after owner destruction
// but does not prove liveness, authorize work or qualify a timing measurement.
class ProviderInstanceId final {
public:
  constexpr ProviderInstanceId() noexcept = default;
  constexpr bool valid() const noexcept { return value_ != 0; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint64_t value() const noexcept { return value_; }
  friend constexpr bool operator==(ProviderInstanceId,
                                   ProviderInstanceId) noexcept = default;

private:
  friend struct detail::ProviderInstanceAccess;
  constexpr explicit ProviderInstanceId(std::uint64_t value) noexcept
      : value_(value) {}
  std::uint64_t value_ = 0;
};

// A prepared provider either records commands into a caller-owned HYLE Queue
// or owns an independent runtime island. The two modes have disjoint submit
// callbacks; callers never infer ownership from a provider name.
enum class ProviderRuntimeMode : std::uint8_t {
  kSharedQueue = 0,
  kDelegated,
};

enum class ProviderWorkspaceOwner : std::uint8_t {
  kCaller = 0,
  kProvider,
};

// The provider's successful replay cost, independent of runtime ownership.
// Caller-supplied executors/callbacks and failure diagnostics retain their own
// cost contracts. Neither a provider name nor capture compatibility proves this.
enum class ProviderReplayClass : std::uint8_t {
  kUnknown = 0,
  // Already-prepared commands: no graph/shape inference, heap allocation, JIT,
  // tactic search or constant packing in the provider's successful replay.
  kPrepared,
  // A precompiled program may instantiate execution metadata/arrays and do
  // declared internal cache/shape work. It still must not JIT/search/repack or
  // hide copy/sync; all internal costs belong in the full replay boundary.
  kManaged,
};

// Cold-path facts that remain true for the complete lifetime of one prepared
// artifact. They are diagnostic/planning data, not a runtime policy oracle.
struct ProviderRuntimeProperties final {
  ProviderRuntimeMode mode = ProviderRuntimeMode::kSharedQueue;
  ProviderWorkspaceOwner workspace_owner = ProviderWorkspaceOwner::kCaller;
  ProviderReplayClass replay_class = ProviderReplayClass::kUnknown;
  bool blocks_host = false;
  bool uses_private_threads = false;
  // Per prepared artifact. False requires direct callers to serialize all
  // its Invocations through terminal Completion, not just through enqueue.
  // Graph enforces the corresponding exclusion within one Executable.
  bool supports_concurrent_invocations = true;
  bool capture_compatible = false;

  constexpr bool valid() const noexcept {
    return (mode == ProviderRuntimeMode::kSharedQueue ||
            mode == ProviderRuntimeMode::kDelegated) &&
           (workspace_owner == ProviderWorkspaceOwner::kCaller ||
            workspace_owner == ProviderWorkspaceOwner::kProvider) &&
           (replay_class == ProviderReplayClass::kUnknown ||
            replay_class == ProviderReplayClass::kPrepared ||
            replay_class == ProviderReplayClass::kManaged) &&
           (!capture_compatible || mode == ProviderRuntimeMode::kSharedQueue);
  }
  friend constexpr bool operator==(const ProviderRuntimeProperties &,
                                   const ProviderRuntimeProperties &) noexcept =
      default;
};

// Stable identity for an optional provider-owned capability. The version is
// part of the identity: consumers never reinterpret an interface table from a
// different contract version. Extension lookup is explicit and cold-path;
// normal operation preparation does not search this namespace.
struct ProviderExtensionId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t extension_id = 0;
  std::uint32_t version = 0;

  constexpr bool valid() const noexcept {
    return (namespace_id != 0 || extension_id != 0) && version != 0;
  }
  friend constexpr bool operator==(ProviderExtensionId,
                                   ProviderExtensionId) noexcept = default;
};

using ParallelTaskFn = void (*)(void *task_context, std::size_t index,
                                std::size_t count) noexcept;
using ParallelForFn = void (*)(void *executor_context, std::size_t count,
                               ParallelTaskFn task,
                               void *task_context) noexcept;
using InParallelFn = bool (*)(void *executor_context) noexcept;

// Borrowed executor seam for CPU providers. A custom backing must outlive each
// Context/artifact/invocation created from it; parallel_for is synchronous,
// infallible, non-throwing, and must make progress when nested.
class ParallelExecutorRef final {
public:
  constexpr ParallelExecutorRef() noexcept = default;
  constexpr ParallelExecutorRef(void *context, std::size_t concurrency,
                                ParallelForFn parallel_for,
                                InParallelFn in_parallel = nullptr) noexcept
      : context_(context),
        concurrency_(
            parallel_for == nullptr ? 1 : (concurrency == 0 ? 1 : concurrency)),
        parallel_for_(parallel_for), in_parallel_(in_parallel) {}

  constexpr std::size_t concurrency() const noexcept { return concurrency_; }
  bool in_parallel() const noexcept {
    return in_parallel_ != nullptr && in_parallel_(context_);
  }
  void parallel_for(std::size_t count, ParallelTaskFn task,
                    void *task_context) const noexcept {
    if (count == 0 || task == nullptr)
      return;
    if (parallel_for_ != nullptr) {
      parallel_for_(context_, count, task, task_context);
      return;
    }
    for (std::size_t index = 0; index < count; ++index)
      task(task_context, index, count);
  }

private:
  void *context_ = nullptr;
  std::size_t concurrency_ = 1;
  ParallelForFn parallel_for_ = nullptr;
  InParallelFn in_parallel_ = nullptr;
};

namespace detail {
struct ProviderAccess;
struct ProviderVTable;
} // namespace detail

// Small, copyable configuration, not a backend runtime. Builtin configurations
// are inline; make_custom_provider owns its captured factory. Context copies
// retain that configuration until all artifacts/in-flight work release it.
// Copying separate handles is thread-safe; mutating one handle is exclusive.
// The implementation's code image and any explicitly borrowed resources must
// outlive every retained handle. Identity views are borrowed from this owner.
class Provider final {
public:
  constexpr Provider() noexcept = default;
  HYLE_COMPUTE_API Provider(const Provider &other) noexcept;
  HYLE_COMPUTE_API Provider &operator=(const Provider &other) noexcept;
  HYLE_COMPUTE_API Provider(Provider &&other) noexcept;
  HYLE_COMPUTE_API Provider &operator=(Provider &&other) noexcept;
  HYLE_COMPUTE_API ~Provider();

  constexpr bool valid() const noexcept { return vtable_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API ProviderIdentity identity() const noexcept;
  HYLE_COMPUTE_API ProviderId id() const noexcept;
  HYLE_COMPUTE_API std::string_view name() const noexcept;
  HYLE_COMPUTE_API std::string_view version() const noexcept;

private:
  friend struct detail::ProviderAccess;
  static constexpr std::size_t kInlineBytes = sizeof(void *) * 6;

  const detail::ProviderVTable *vtable_ = nullptr;
  alignas(void *) std::array<std::byte, kInlineBytes> payload_{};
};

} // namespace hyle::compute
