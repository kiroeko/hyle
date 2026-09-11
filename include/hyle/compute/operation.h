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
  kRmsNorm = 0,
};

struct RmsNorm final {
  float epsilon = 1.0e-5F;
};

// Typed operation value. The public shape grows by adding typed descriptors,
// never a string property bag or a provider-specific node pointer.
class Operation final {
 public:
  constexpr explicit Operation(RmsNorm rms_norm) noexcept
      : rms_norm_(rms_norm) {}

  constexpr OperationKind kind() const noexcept {
    return OperationKind::kRmsNorm;
  }
  constexpr const RmsNorm& rms_norm() const noexcept { return rms_norm_; }

 private:
  RmsNorm rms_norm_{};
};

enum class ProviderKind : std::uint8_t {
  kHyleNative = 0,
};

HYLE_COMPUTE_API std::string_view provider_name(
    ProviderKind provider) noexcept;
HYLE_COMPUTE_API std::string_view operation_name(
    OperationKind operation) noexcept;

class PreparedOperation;

HYLE_COMPUTE_API Result<PreparedOperation> prepare(
    const Context& context, Operation operation,
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs) noexcept;

class PreparedOperation final {
 public:
  PreparedOperation() noexcept = default;
  PreparedOperation(const PreparedOperation&) = delete;
  PreparedOperation& operator=(const PreparedOperation&) = delete;
  HYLE_COMPUTE_API PreparedOperation(PreparedOperation&& other) noexcept;
  HYLE_COMPUTE_API PreparedOperation& operator=(
      PreparedOperation&& other) noexcept;
  HYLE_COMPUTE_API ~PreparedOperation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  OperationKind kind() const noexcept { return operation_.kind(); }
  ProviderKind provider() const noexcept { return ProviderKind::kHyleNative; }
  std::size_t input_count() const noexcept { return input_count_; }
  std::size_t output_count() const noexcept { return valid() ? 1 : 0; }
  HYLE_COMPUTE_API Result<const TensorDesc&> input_description(
      std::size_t index = 0) const noexcept;
  HYLE_COMPUTE_API Result<const TensorDesc&> output_description(
      std::size_t index = 0) const noexcept;

  HYLE_COMPUTE_API Result<Completion> submit(
      Queue& queue, std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) const noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(
      Queue& queue, ConstTensorView input, TensorView output) const noexcept;
  HYLE_COMPUTE_API Result<Completion> submit(
      Queue& queue, ConstTensorView input, ConstTensorView weight,
      TensorView output) const noexcept;

 private:
  friend HYLE_COMPUTE_API Result<PreparedOperation> prepare(
      const Context&, Operation, std::span<const TensorDesc>,
      std::span<const TensorDesc>) noexcept;

  PreparedOperation(detail::ContextState* state, Operation operation,
                    std::span<const TensorDesc> inputs,
                    TensorDesc output) noexcept;

  using Route = void (*)(const float*, const float*, float*, std::size_t,
                         float) noexcept;

  detail::ContextState* state_ = nullptr;
  Operation operation_{RmsNorm{}};
  std::array<TensorDesc, 2> inputs_{};
  TensorDesc output_{};
  std::size_t element_count_ = 0;
  std::uint8_t input_count_ = 0;
  Route route_ = nullptr;
};

HYLE_COMPUTE_API Result<PreparedOperation> prepare(
    const Context& context, RmsNorm operation, TensorDesc input,
    TensorDesc output) noexcept;
HYLE_COMPUTE_API Result<PreparedOperation> prepare(
    const Context& context, RmsNorm operation, TensorDesc input,
    TensorDesc weight, TensorDesc output) noexcept;

}  // namespace hyle::compute
