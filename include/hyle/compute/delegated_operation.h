#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>

#include "hyle/compute/prepared_operation.h"

namespace hyle::compute {

namespace detail {
struct DirectDelegatedState;
}

struct DelegatedInvocationOptions final {
  mem::BufferView workspace{};
  // Zero reserves input_count + 1 entries during cold creation.
  std::size_t dependency_capacity = 0;
};

// A distinct type for an independent runtime. It has no Queue submit entry.
// Create once per concurrent run. Bindings and source Completions are borrowed
// through dependent completion; do not move, destroy or settle a dependency
// while that runtime may still use its native state. A Provider must reject
// unsupported dependencies before acceptance, without an implicit host wait.
class DelegatedInvocation final {
public:
  DelegatedInvocation() noexcept = default;
  DelegatedInvocation(const DelegatedInvocation &) = delete;
  DelegatedInvocation &operator=(const DelegatedInvocation &) = delete;
  HYLE_COMPUTE_API DelegatedInvocation(DelegatedInvocation &&other) noexcept;
  HYLE_COMPUTE_API DelegatedInvocation &
  operator=(DelegatedInvocation &&other) noexcept;
  HYLE_COMPUTE_API ~DelegatedInvocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit_after(std::span<const Completion *const> dependencies,
               std::span<const ConstTensorView> inputs,
               std::span<const TensorView> outputs) noexcept;
  Result<Completion> submit(std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
    return submit_after({}, inputs, outputs);
  }
  Result<Completion> submit(std::span<const ConstTensorView> inputs,
                            const TensorView &output) noexcept {
    return submit(inputs, std::span<const TensorView>(&output, 1));
  }
  Result<Completion> submit(const ConstTensorView &input,
                            const TensorView &output) noexcept {
    return submit(std::span<const ConstTensorView>(&input, 1), output);
  }
  Result<Completion> submit(const ConstTensorView &first,
                            const ConstTensorView &second,
                            const TensorView &output) noexcept {
    const std::array inputs{first, second};
    return submit(inputs, output);
  }

private:
  friend class PreparedDelegatedOperation;
  explicit DelegatedInvocation(detail::DirectDelegatedState *state) noexcept
      : state_(state) {}
  detail::DirectDelegatedState *state_ = nullptr;
};

class PreparedDelegatedOperation final {
public:
  PreparedDelegatedOperation() noexcept = default;
  bool valid() const noexcept { return prepared_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  OperationKind kind() const noexcept { return prepared_.kind(); }
  ProviderIdentity provider() const noexcept { return prepared_.provider(); }
  ProviderInstanceId provider_instance_id() const noexcept {
    return prepared_.provider_instance_id();
  }
  ProviderRuntimeProperties runtime_properties() const noexcept {
    return prepared_.runtime_properties();
  }
  WorkspaceRequirement workspace_requirement() const noexcept {
    return prepared_.workspace_requirement();
  }
  std::optional<std::size_t> persistent_bytes() const noexcept {
    return prepared_.persistent_bytes();
  }
  std::optional<std::size_t> invocation_private_bytes() const noexcept {
    return prepared_.invocation_private_bytes();
  }
  Result<AliasSupport> alias_support(std::size_t output,
                                     std::size_t input) const noexcept {
    return prepared_.alias_support(output, input);
  }
  std::size_t input_count() const noexcept { return prepared_.input_count(); }
  std::size_t output_count() const noexcept { return prepared_.output_count(); }
  Result<const TensorDesc &>
  input_description(std::size_t index = 0) const noexcept {
    return prepared_.input_description(index);
  }
  Result<const TensorDesc &>
  output_description(std::size_t index = 0) const noexcept {
    return prepared_.output_description(index);
  }
  HYLE_COMPUTE_API Result<DelegatedInvocation>
  create_invocation(DelegatedInvocationOptions options = {}) const noexcept;

private:
  friend HYLE_COMPUTE_API Result<PreparedDelegatedOperation>
  prepare_delegated(const Context &, Operation, std::span<const TensorDesc>,
                    std::span<const TensorDesc>,
                    OperationResourceBindings) noexcept;
  friend HYLE_COMPUTE_API Result<PreparedDelegatedOperation> prepare_delegated(
      const Context &, Operation, std::span<const ConstTensorView>,
      std::span<const TensorView>, mem::MemoryResourceRef) noexcept;
  explicit PreparedDelegatedOperation(PreparedOperation prepared) noexcept
      : prepared_(std::move(prepared)) {}
  PreparedOperation prepared_{};
};

HYLE_COMPUTE_API Result<PreparedDelegatedOperation>
prepare_delegated(const Context &context, Operation operation,
                  std::span<const TensorDesc> inputs,
                  std::span<const TensorDesc> outputs,
                  OperationResourceBindings resources) noexcept;
HYLE_COMPUTE_API Result<PreparedDelegatedOperation>
prepare_delegated(const Context &context, Operation operation,
                  std::span<const ConstTensorView> inputs,
                  std::span<const TensorView> outputs,
                  mem::MemoryResourceRef workspace_resource = {}) noexcept;

} // namespace hyle::compute
