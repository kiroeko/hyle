#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {

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

enum class ProviderKind : std::uint8_t {
  kUnknown = 0,
  kHyleNative,
};

HYLE_COMPUTE_API std::string_view provider_name(ProviderKind provider) noexcept;
HYLE_COMPUTE_API std::string_view
operation_name(OperationKind operation) noexcept;

// The single source of shape/layout rules shared by direct Compute and Graph.
// Every v2 operation has exactly one output.
HYLE_COMPUTE_API Result<TensorDesc>
infer_output_description(Operation operation,
                         std::span<const TensorDesc> inputs) noexcept;

class PreparedOperation;

HYLE_COMPUTE_API Result<PreparedOperation>
prepare(const Context &context, Operation operation,
        std::span<const TensorDesc> inputs,
        std::span<const TensorDesc> outputs) noexcept;

class PreparedOperation final {
public:
  PreparedOperation() noexcept = default;
  PreparedOperation(const PreparedOperation &) = delete;
  PreparedOperation &operator=(const PreparedOperation &) = delete;
  HYLE_COMPUTE_API PreparedOperation(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API PreparedOperation &
  operator=(PreparedOperation &&other) noexcept;
  HYLE_COMPUTE_API ~PreparedOperation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  OperationKind kind() const noexcept {
    return valid() ? operation_.kind() : OperationKind::kUnknown;
  }
  ProviderKind provider() const noexcept {
    return valid() ? ProviderKind::kHyleNative : ProviderKind::kUnknown;
  }
  std::size_t input_count() const noexcept { return input_count_; }
  std::size_t output_count() const noexcept { return valid() ? 1 : 0; }
  HYLE_COMPUTE_API Result<const TensorDesc &>
  input_description(std::size_t index = 0) const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc &>
  output_description(std::size_t index = 0) const noexcept;

  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs) const noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &input,
         const TensorView &output) const noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, const ConstTensorView &input,
         const ConstTensorView &weight,
         const TensorView &output) const noexcept;

private:
  friend HYLE_COMPUTE_API Result<PreparedOperation>
  prepare(const Context &, Operation, std::span<const TensorDesc>,
          std::span<const TensorDesc>) noexcept;

  PreparedOperation(detail::ContextState *state, Operation operation,
                    std::span<const TensorDesc> inputs, TensorDesc output,
                    std::array<std::size_t, 3> parameters, float scalar,
                    void (*route)(const float *, const float *, float *,
                                  const std::array<std::size_t, 3> &,
                                  float) noexcept) noexcept;

  Result<Completion>
  submit_impl(Queue &queue, const ConstTensorView *first,
              const ConstTensorView *second, std::size_t provided_input_count,
              const TensorView *output,
              std::size_t provided_output_count) const noexcept;

  using Route = void (*)(const float *, const float *, float *,
                         const std::array<std::size_t, 3> &, float) noexcept;

  detail::ContextState *state_ = nullptr;
  Operation operation_{RmsNorm{}};
  std::array<TensorDesc, 2> inputs_{};
  TensorDesc output_{};
  std::array<std::size_t, 3> parameters_{};
  float scalar_ = 0.0F;
  std::uint8_t input_count_ = 0;
  Route route_ = nullptr;
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
