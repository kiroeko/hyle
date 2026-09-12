#pragma once

#include <cstddef>
#include <span>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/tensor.h"
#include "hyle/compute/visibility.h"

namespace hyle::compute {

namespace detail {
struct InvocationState;
struct PreparedState;
} // namespace detail

struct WorkspaceRequirement final {
  std::size_t size = 0;
  std::size_t alignment = 1;
};

class PreparedOperation;
class OperationInvocation;

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs) noexcept;

class PreparedOperation final {
public:
  PreparedOperation() noexcept = default;
  HYLE_COMPUTE_API PreparedOperation(const PreparedOperation &other) noexcept;
  HYLE_COMPUTE_API PreparedOperation &
  operator=(const PreparedOperation &other) noexcept;
  HYLE_COMPUTE_API PreparedOperation(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API PreparedOperation &
  operator=(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API ~PreparedOperation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API OperationKind kind() const noexcept;
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;
  HYLE_COMPUTE_API WorkspaceRequirement workspace_requirement() const noexcept;
  HYLE_COMPUTE_API std::size_t input_count() const noexcept;
  HYLE_COMPUTE_API std::size_t output_count() const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc &>
  input_description(std::size_t index = 0) const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc &>
  output_description(std::size_t index = 0) const noexcept;

  HYLE_COMPUTE_API Result<OperationInvocation>
  create_invocation(mem::BufferView workspace = {}) const noexcept;

private:
  friend HYLE_COMPUTE_API Result<PreparedOperation>
  prepare(const Context &, Operation, std::span<const TensorDesc>,
          std::span<const TensorDesc>) noexcept;

  explicit PreparedOperation(detail::PreparedState *state) noexcept
      : state_(state) {}

  detail::PreparedState *state_ = nullptr;
};

// Exclusive mutable state for one execution lane. Create it once per
// concurrent run, then rebind and submit repeatedly without provider search,
// descriptor construction or workspace allocation.
class OperationInvocation final {
public:
  OperationInvocation() noexcept = default;
  OperationInvocation(const OperationInvocation &) = delete;
  OperationInvocation &operator=(const OperationInvocation &) = delete;
  HYLE_COMPUTE_API OperationInvocation(OperationInvocation &&other) noexcept;
  HYLE_COMPUTE_API OperationInvocation &
  operator=(OperationInvocation &&other) noexcept;
  HYLE_COMPUTE_API ~OperationInvocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;

  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const ConstTensorView &input,
                                             const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(Queue &queue,
                                             const ConstTensorView &input,
                                             const ConstTensorView &weight,
                                             const TensorView &output) noexcept;

private:
  friend class PreparedOperation;
  explicit OperationInvocation(detail::InvocationState *state) noexcept
      : state_(state) {}

  detail::InvocationState *state_ = nullptr;
};

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, View operation, TensorDesc input) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Copy operation, TensorDesc input) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation> prepare(const Context &context,
                                                   Fill operation) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation> prepare(const Context &context,
                                                   RmsNorm operation,
                                                   TensorDesc input,
                                                   TensorDesc output) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, RmsNorm operation, TensorDesc input,
        TensorDesc weight, TensorDesc output) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, MatMul operation, TensorDesc left,
        TensorDesc right, TensorDesc output) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation> prepare(const Context &context,
                                                   GatedActivation operation,
                                                   TensorDesc input,
                                                   TensorDesc output) noexcept;

} // namespace hyle::compute
