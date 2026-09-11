#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {

namespace detail {
struct PreparedState;
}

enum class OperationKind : std::uint8_t {
  kUnknown = 0,
  kRmsNorm,
  kMatMul,
  kSiluAndMul,
};

struct RmsNorm final {
  float epsilon = 1.0e-5F;
};

// Logical rank-2 matrix multiplication: A[M,K] * B[K,N] -> C[M,N].
// Physical orientation is described by TensorDesc strides, not flags here.
struct MatMul final {};

// Packed SwiGLU used by current inference engines:
// input[..., 2*d] -> silu(input[..., :d]) * input[..., d:].
struct SiluAndMul final {};

// Typed operation value. The public shape grows by adding typed descriptors,
// never a string property bag or a provider-specific node pointer.
class Operation final {
public:
  constexpr explicit Operation(RmsNorm rms_norm) noexcept
      : kind_(OperationKind::kRmsNorm), rms_norm_(rms_norm) {}
  constexpr explicit Operation(MatMul) noexcept
      : kind_(OperationKind::kMatMul) {}
  constexpr explicit Operation(SiluAndMul) noexcept
      : kind_(OperationKind::kSiluAndMul) {}

  constexpr OperationKind kind() const noexcept { return kind_; }
  constexpr const RmsNorm *rms_norm() const noexcept {
    return kind_ == OperationKind::kRmsNorm ? &rms_norm_ : nullptr;
  }

private:
  OperationKind kind_ = OperationKind::kRmsNorm;
  RmsNorm rms_norm_{};
};

HYLE_COMPUTE_API std::string_view
operation_name(OperationKind operation) noexcept;

struct WorkspaceRequirement final {
  std::size_t size = 0;
  std::size_t alignment = 1;
};

// The single source of shape/layout rules shared by direct Compute and Graph.
// Every v2 operation has exactly one output.
HYLE_COMPUTE_API Result<TensorDesc>
infer_output_description(Operation operation,
                         std::span<const TensorDesc> inputs) noexcept;

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
  HYLE_COMPUTE_API ProviderKind provider() const noexcept;
  HYLE_COMPUTE_API bool used_fallback() const noexcept;
  HYLE_COMPUTE_API WorkspaceRequirement
  workspace_requirement() const noexcept;
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
  HYLE_COMPUTE_API ProviderKind provider() const noexcept;

  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs) noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &input,
         const TensorView &output) noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &input,
         const ConstTensorView &weight, const TensorView &output) noexcept;

private:
  friend class PreparedOperation;
  OperationInvocation(detail::PreparedState *state,
                      void *provider_state) noexcept;

  Result<Completion>
  submit_impl(Queue &queue, const ConstTensorView *first,
              const ConstTensorView *second, std::size_t provided_input_count,
              const TensorView *output,
              std::size_t provided_output_count) noexcept;

  detail::PreparedState *state_ = nullptr;
  void *provider_state_ = nullptr;
};

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
                                                   SiluAndMul operation,
                                                   TensorDesc input,
                                                   TensorDesc output) noexcept;

} // namespace hyle::compute
