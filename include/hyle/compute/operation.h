#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/compute/ops_visibility.h"
#include "hyle/compute/tensor_desc.h"

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

// Provider-neutral typed schema. Runtime preparation, bindings and submission
// deliberately live in prepared_operation.h so Graph can consume this value
// without acquiring Context, Queue, Mem or provider runtime dependencies.
class Operation final {
 public:
  constexpr Operation() noexcept = default;
  constexpr explicit Operation(RmsNorm rms_norm) noexcept
      : kind_(OperationKind::kRmsNorm), rms_norm_(rms_norm) {}
  constexpr explicit Operation(MatMul) noexcept
      : kind_(OperationKind::kMatMul) {}
  constexpr explicit Operation(SiluAndMul) noexcept
      : kind_(OperationKind::kSiluAndMul) {}

  constexpr bool valid() const noexcept {
    return kind_ != OperationKind::kUnknown;
  }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr OperationKind kind() const noexcept { return kind_; }
  constexpr const RmsNorm* rms_norm() const noexcept {
    return kind_ == OperationKind::kRmsNorm ? &rms_norm_ : nullptr;
  }

 private:
  OperationKind kind_ = OperationKind::kUnknown;
  RmsNorm rms_norm_{};
};

HYLE_COMPUTE_OPS_API std::string_view
operation_name(OperationKind operation) noexcept;

// The single source of shape/layout rules shared by direct Compute and Graph.
// This compatibility-shaped entry is replaced by the exact-span multi-output
// contract as the v1 catalogue lands.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
infer_output_description(Operation operation,
                         std::span<const TensorDesc> inputs) noexcept;

static_assert(std::is_trivially_copyable_v<Operation>);

}  // namespace hyle::compute
