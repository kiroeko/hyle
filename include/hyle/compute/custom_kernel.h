#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/compute/detail/prepared_callbacks.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/resource_requirement.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {

// A custom kernel records into an already-admitted HYLE Queue. An accepted
// prefix, including a partial failure, must be represented by Completion;
// rejected means no work or external effect was accepted. Do not call a
// second submit()/submit_host() inside enqueue. No-op records no work.
struct KernelCommandResult final {
  bool accepted = false;
  Status status{};

  static KernelCommandResult
  accepted_command(Status terminal = Status::Ok()) noexcept {
    return {true, std::move(terminal)};
  }
  static KernelCommandResult rejected(Status error) noexcept {
    return {false, std::move(error)};
  }
  static KernelCommandResult no_op() noexcept { return {}; }
};

struct KernelProperties final {
  WorkspaceRequirement workspace{};
  ProviderRuntimeProperties runtime{};
  std::optional<std::size_t> persistent_bytes{};
  // Upper bound outside the create callback's allocator and caller workspace,
  // including creation, replay and destruction. Unknown is not zero.
  std::optional<std::size_t> invocation_private_bytes{};

  constexpr bool valid() const noexcept {
    return std::has_single_bit(workspace.alignment) && runtime.valid() &&
           runtime.mode == ProviderRuntimeMode::kSharedQueue &&
           runtime.workspace_owner == ProviderWorkspaceOwner::kCaller;
  }
};

// Borrowed for the duration of propose/prepare only; never retain these spans.
// Descriptions are exact points. Unsupported is a cold capability decline;
// propose must not compile, read tensor payload, or accept external work.
struct KernelRequest final {
  Device device{};
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
};

struct KernelProposal final {
  std::uint64_t route = 0;
  ProviderRuntimeProperties runtime{};
  ResourceRequirement workspace_resource{};
  ProviderRouteCostFacts cost{};
  // Optional stable implementation key; independent of the prepare token.
  // Leaving this unknown never disables an otherwise valid execution route.
  ProviderRouteKey profile_key{};
};

struct KernelPrepareRequest final {
  Device device{};
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
  OperationResourceBindings resources{};
  std::uint64_t route = 0;
};

namespace detail {
struct CustomKernelAccess;
template <class Kernel> struct CustomKernelStorage;
} // namespace detail

// Owning, move-only result of cold kernel preparation. The same typed kernel
// is immutable and shared by its Invocations; each Invocation owns its mutable
// run state. The factory's configuration remains alive through the Context.
class CustomKernel final {
public:
  CustomKernel() noexcept = default;
  CustomKernel(const CustomKernel &) = delete;
  CustomKernel &operator=(const CustomKernel &) = delete;
  CustomKernel(CustomKernel &&other) noexcept
      : table_(std::exchange(other.table_, nullptr)),
        state_(std::exchange(other.state_, nullptr)),
        properties_(other.properties_) {}
  CustomKernel &operator=(CustomKernel &&other) noexcept {
    if (this != &other) {
      reset();
      table_ = std::exchange(other.table_, nullptr);
      state_ = std::exchange(other.state_, nullptr);
      properties_ = other.properties_;
    }
    return *this;
  }
  ~CustomKernel() { reset(); }

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  KernelProperties properties() const noexcept { return properties_; }
  void reset() noexcept {
    if (state_ != nullptr)
      table_->destroy_prepared(state_, {});
    table_ = nullptr;
    state_ = nullptr;
  }

private:
  friend struct detail::CustomKernelAccess;
  template <class Kernel> friend struct detail::CustomKernelStorage;
  CustomKernel(const detail::ProviderPreparedVTable *table, void *state,
               KernelProperties properties) noexcept
      : table_(table), state_(state), properties_(properties) {}

  const detail::ProviderPreparedVTable *table_ = nullptr;
  void *state_ = nullptr;
  KernelProperties properties_{};
};

// Distinguishes a safe, rolled-back decline from a real preparation error.
// All partial CustomKernel values are RAII-owned, including error results.
struct KernelPreparation final {
  ProviderPrepareOutcome outcome{};
  CustomKernel kernel{};

  static KernelPreparation prepared(CustomKernel kernel) noexcept {
    return {ProviderPrepareOutcome::prepared(), std::move(kernel)};
  }
  static KernelPreparation unsupported(Status reason) noexcept {
    return {ProviderPrepareOutcome::unsupported(std::move(reason)), {}};
  }
  static KernelPreparation resource_declined(Status reason) noexcept {
    return {ProviderPrepareOutcome::resource_declined(std::move(reason)), {}};
  }
  static KernelPreparation error(Status reason) noexcept {
    return {ProviderPrepareOutcome::error(std::move(reason)), {}};
  }
};

template <class Kernel>
concept CustomKernelType =
    std::is_nothrow_destructible_v<Kernel> &&
    requires(const Kernel &kernel, typename Kernel::Invocation &invocation,
             Queue &queue, mem::BufferView workspace,
             mem::AllocatorRef allocator,
             std::span<const ConstTensorView> inputs,
             std::span<const TensorView> outputs) {
      requires std::is_nothrow_move_constructible_v<
          typename Kernel::Invocation>;
      requires std::is_nothrow_destructible_v<typename Kernel::Invocation>;
      {
        kernel.create_invocation(workspace, allocator)
      } noexcept -> std::same_as<Result<typename Kernel::Invocation>>;
      {
        kernel.enqueue(invocation, queue, inputs, outputs)
      } noexcept -> std::same_as<KernelCommandResult>;
    };

namespace detail {

template <class Kernel> struct CustomKernelStorage final {
  mem::AllocatorRef owner_allocator;
  Kernel kernel_value;

  using Invocation = typename Kernel::Invocation;
  static const ProviderPreparedVTable *callbacks() noexcept {
    static constexpr ProviderPreparedVTable table{
        .destroy_prepared =
            [](void *state, mem::AllocatorRef) noexcept {
              auto *owned = static_cast<CustomKernelStorage *>(state);
              const auto allocator = owned->owner_allocator;
              std::destroy_at(owned);
              allocator.deallocate(owned, sizeof(CustomKernelStorage),
                                   alignof(CustomKernelStorage));
            },
        .alias_support =
            [](void *state, std::size_t output, std::size_t input) noexcept {
              const auto &value =
                  static_cast<CustomKernelStorage *>(state)->kernel_value;
              if constexpr (requires { value.alias_support(output, input); }) {
                static_assert(noexcept(value.alias_support(output, input)));
                static_assert(
                    std::same_as<decltype(value.alias_support(output, input)),
                                 AliasSupport>);
                return value.alias_support(output, input);
              } else {
                return AliasSupport{};
              }
            },
        .create_invocation = [](void *prepared, mem::AllocatorRef allocator,
                                mem::BufferView workspace,
                                void **output) noexcept -> Status {
          *output = nullptr;
          void *storage =
              allocator.allocate(sizeof(Invocation), alignof(Invocation));
          if (storage == nullptr)
            return Status::ResourceExhausted(
                "custom kernel invocation allocation failed");
          const auto &kernel =
              static_cast<CustomKernelStorage *>(prepared)->kernel_value;
          auto value = kernel.create_invocation(workspace, allocator);
          if (!value) {
            allocator.deallocate(storage, sizeof(Invocation),
                                 alignof(Invocation));
            return std::move(value).error();
          }
          *output = std::construct_at(static_cast<Invocation *>(storage),
                                      std::move(*value));
          return Status::Ok();
        },
        .destroy_invocation =
            [](void *state, mem::AllocatorRef allocator) noexcept {
              std::destroy_at(static_cast<Invocation *>(state));
              allocator.deallocate(state, sizeof(Invocation),
                                   alignof(Invocation));
            },
        .shared_submit =
            [](void *prepared, void *state, Queue &queue,
               std::span<const ConstTensorView> inputs,
               std::span<const TensorView> outputs) noexcept -> EnqueueResult {
          const auto &kernel =
              static_cast<CustomKernelStorage *>(prepared)->kernel_value;
          auto result = kernel.enqueue(*static_cast<Invocation *>(state), queue,
                                       inputs, outputs);
          return {result.accepted, std::move(result.status)};
        },
    };
    return &table;
  }

  template <class Value>
  static Result<CustomKernel> make(Value &&value, KernelProperties properties,
                                   mem::AllocatorRef allocator) noexcept {
    if (!properties.valid())
      return std::unexpected(
          Status::InvalidArgument("invalid shared custom kernel properties"));
    if (!allocator.valid())
      allocator = mem::system_allocator();
    void *storage = allocator.allocate(sizeof(CustomKernelStorage),
                                       alignof(CustomKernelStorage));
    if (storage == nullptr)
      return std::unexpected(
          Status::ResourceExhausted("custom kernel allocation failed"));
    auto *owned = ::new (storage)
        CustomKernelStorage{allocator, std::forward<Value>(value)};
    return CustomKernel(callbacks(), owned, properties);
  }
};

struct CustomFactoryCallbacks final {
  void (*destroy)(void *) noexcept = nullptr;
  Result<KernelProposal> (*propose)(
      const void *, const KernelRequest &, std::span<ResourceRequirement>,
      std::span<ResourceRequirement>) noexcept = nullptr;
  KernelPreparation (*prepare)(const void *, const KernelPrepareRequest &,
                               mem::AllocatorRef) noexcept = nullptr;
};

// Internal typed-factory bridge. Construction runs only after validation and
// successful allocation; no user state is consumed on failure.
HYLE_COMPUTE_API Result<Provider>
make_custom_provider(ProviderIdentity identity, BackendId backend,
                     const CustomFactoryCallbacks *callbacks, std::size_t size,
                     std::size_t alignment, void *source,
                     void (*construct)(void *, void *) noexcept,
                     mem::AllocatorRef allocator) noexcept;

} // namespace detail

// One cold allocation for the typed kernel, followed by one allocation for
// each typed Invocation. The allocator owner must outlive them. Failed
// validation/allocation does not move from the supplied value. Successful
// replay adds no allocation or dispatch beyond the existing provider callback.
template <class Kernel>
  requires CustomKernelType<std::remove_cvref_t<Kernel>> &&
           std::is_nothrow_constructible_v<std::remove_cvref_t<Kernel>,
                                           Kernel &&>
Result<CustomKernel>
make_custom_kernel(Kernel &&kernel, KernelProperties properties,
                   mem::AllocatorRef allocator = {}) noexcept {
  return detail::CustomKernelStorage<std::remove_cvref_t<Kernel>>::make(
      std::forward<Kernel>(kernel), properties, allocator);
}

template <class Factory>
concept CustomKernelFactory =
    std::is_nothrow_destructible_v<Factory> &&
    requires(const Factory &factory, const KernelRequest &request,
             const KernelPrepareRequest &prepare_request,
             mem::AllocatorRef allocator, std::span<ResourceRequirement> inputs,
             std::span<ResourceRequirement> outputs) {
      {
        factory.propose(request, inputs, outputs)
      } noexcept -> std::same_as<Result<KernelProposal>>;
      {
        factory.prepare(prepare_request, allocator)
      } noexcept -> std::same_as<KernelPreparation>;
    };

// Owns factory (including move-only captures) and copies identity strings in
// one cold allocation. A Context retains it; no global registration is used.
// const factory methods must support concurrent Context/prepare calls. They
// never submit user work. Borrowed resources and the code image remain the
// consumer's lifetime responsibility. Only shared-Queue kernels are accepted;
// delegated runtimes use their dedicated HYLE provider adapters.
template <class Factory>
  requires CustomKernelFactory<std::remove_cvref_t<Factory>> &&
           std::is_nothrow_constructible_v<std::remove_cvref_t<Factory>,
                                           Factory &&>
Result<Provider>
make_custom_provider(ProviderIdentity identity, BackendId backend,
                     Factory &&factory,
                     mem::AllocatorRef allocator = {}) noexcept {
  using Value = std::remove_cvref_t<Factory>;
  static const detail::CustomFactoryCallbacks callbacks{
      .destroy =
          [](void *state) noexcept {
            std::destroy_at(static_cast<Value *>(state));
          },
      .propose =
          [](const void *state, const KernelRequest &request,
             std::span<ResourceRequirement> inputs,
             std::span<ResourceRequirement> outputs) noexcept {
            return static_cast<const Value *>(state)->propose(request, inputs,
                                                              outputs);
          },
      .prepare =
          [](const void *state, const KernelPrepareRequest &request,
             mem::AllocatorRef allocator) noexcept {
            return static_cast<const Value *>(state)->prepare(request,
                                                              allocator);
          },
  };
  auto construct = [](void *destination, void *source) noexcept {
    auto &value = *static_cast<std::remove_reference_t<Factory> *>(source);
    std::construct_at(static_cast<Value *>(destination),
                      std::forward<Factory>(value));
  };
  return detail::make_custom_provider(
      identity, backend, &callbacks, sizeof(Value), alignof(Value),
      const_cast<void *>(static_cast<const void *>(std::addressof(factory))),
      construct, allocator);
}

} // namespace hyle::compute
