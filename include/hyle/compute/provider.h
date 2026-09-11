#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/visibility.h"

namespace hyle::compute {

enum class ProviderKind : std::uint8_t {
  kUnknown = 0,
  kHyleNative,
  kOneDnn,
  kMixed,
};

HYLE_COMPUTE_API std::string_view provider_name(ProviderKind provider) noexcept;

// Infallible unit of a synchronous parallel episode. Every index in [0,count)
// is invoked exactly once before ParallelForFn returns.
using ParallelTaskFn = void (*)(void *task_context, std::size_t index,
                                std::size_t count) noexcept;
using ParallelForFn = void (*)(void *executor_context, std::size_t count,
                               ParallelTaskFn task,
                               void *task_context) noexcept;
using InParallelFn = bool (*)(void *executor_context) noexcept;

// Borrowed executor seam for CPU providers. The default value is a valid
// caller-thread serial executor. A custom backing and its callbacks must
// outlive every Context created from this value; parallel_for must be
// synchronous, infallible, non-throwing, and make progress when nested.
class ParallelExecutorRef final {
public:
  constexpr ParallelExecutorRef() noexcept = default;
  constexpr ParallelExecutorRef(void *context, std::size_t concurrency,
                                ParallelForFn parallel_for,
                                InParallelFn in_parallel = nullptr) noexcept
      : context_(context),
        concurrency_(parallel_for == nullptr ? 1
                                             : (concurrency == 0 ? 1 : concurrency)),
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
} // namespace detail

// Small, copyable provider configuration value. It owns no backend runtime;
// Context creation copies this value and creates the corresponding provider
// state. Factories such as onednn::provider() are the only construction path.
class Provider final {
public:
  constexpr Provider() noexcept = default;

  constexpr bool valid() const noexcept { return vtable_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API ProviderKind kind() const noexcept;
  HYLE_COMPUTE_API std::string_view name() const noexcept;

private:
  friend struct detail::ProviderAccess;
  static constexpr std::size_t kInlineBytes = sizeof(void *) * 6;

  const void *vtable_ = nullptr;
  alignas(void *) std::array<std::byte, kInlineBytes> payload_{};
};

inline constexpr std::size_t kMaxContextProviders = 4;

// Fixed-capacity cold-path list: enough for native-library choices without a
// hidden heap or a dangling initializer_list. HYLE-native is always the final
// fallback and is therefore not stored here.
class ProviderList final {
public:
  constexpr ProviderList() noexcept = default;

  constexpr bool add(Provider provider) noexcept {
    if (!provider.valid() || size_ == providers_.size())
      return false;
    providers_[size_++] = provider;
    return true;
  }

  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr std::span<const Provider> providers() const noexcept {
    return std::span<const Provider>(providers_.data(), size_);
  }

private:
  std::array<Provider, kMaxContextProviders> providers_{};
  std::size_t size_ = 0;
};

} // namespace hyle::compute
