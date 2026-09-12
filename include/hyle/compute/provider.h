#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/compute/visibility.h"

namespace hyle::compute {

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

// Small, copyable provider configuration value. It owns no backend runtime;
// Context creation copies this value and creates provider state once. Provider
// identity is open and versioned, so independently shipped implementations do
// not require a central enum update.
class Provider final {
public:
  constexpr Provider() noexcept = default;

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
