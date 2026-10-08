#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include <cstddef>

namespace hyle::compute::apple::detail {

// Compute-owned, cold region boundary used by the native Graph compiler.
// It deliberately carries neither Graph types nor a per-kernel executor.
enum class RegionKind { kMlx, kGgml };

struct Region final {
  RegionKind kind = RegionKind::kMlx;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  RegionFunction mlx;
  Operation operation;
  std::vector<TensorDesc> native_inputs;
};

struct RegionPlan final {
  WorkloadContext workload;
  std::vector<InputSpec> values;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> constants;
  std::vector<mlx::core::array> constant_values;
  std::vector<std::size_t> outputs;
  std::vector<Region> regions;
};

struct ContextAccess final {
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program>
  compile(const Context &context, RegionPlan plan) noexcept;
};

// One cold admission predicate for typed model-native MLX custom lowering.
// Direct lowering and both Graph validation/tracing use the same domain.
HYLE_COMPUTE_ROUTE_APPLE_API bool
is_mlx_tensor_custom(const Operation &operation) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API bool
is_mlx_exact_gguf_gather(const Operation &operation,
                        std::span<const TensorDesc> inputs) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::vector<mlx::core::array>>
lower_mlx_exact_gguf_gather(const Operation &operation,
    std::span<const TensorDesc> descriptions, std::span<const mlx::core::array> inputs,
    mlx::core::Stream stream) noexcept;

// Cold lowering for model-native FP8 storage. Logical descriptions remain
// distinct from the uint8 MLX carrier used at the native boundary.
HYLE_COMPUTE_ROUTE_APPLE_API bool
is_mlx_float_operation(const Operation &operation,
                       std::span<const TensorDesc> inputs) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<InputSpec>
mlx_float_input_spec(const TensorDesc &description) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::vector<mlx::core::array>>
lower_mlx_float(const Operation &operation,
                std::span<const TensorDesc> descriptions,
                std::span<const mlx::core::array> inputs,
                mlx::core::Stream stream) noexcept;

} // namespace hyle::compute::apple::detail
#endif
