#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include <cstddef>
#include <cstdint>
#include <optional>

namespace hyle::compute::apple::detail {

// Compute-owned, cold region boundary used by the native Graph compiler.
// It deliberately carries neither Graph types nor a per-kernel executor.
enum class RegionKind { kMlx, kGgml, kAttentionBlocks };

HYLE_COMPUTE_ROUTE_APPLE_API bool
is_mlx_attention_blocks(const Operation &, std::span<const TensorDesc>) noexcept;

struct Region final {
  RegionKind kind = RegionKind::kMlx;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  RegionFunction mlx;
  Operation operation;
  std::vector<TensorDesc> native_inputs;
  // Graph-only SSA proof: TopK indices (optionally reshaped), selected axis
  // exactly equal to this GroupedMatMul's expert extent. Never a caller flag.
  std::size_t proven_index_port = static_cast<std::size_t>(-1);
  std::int64_t proven_index_extent = 0;
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
  static const std::shared_ptr<Runtime> &runtime(const Context &context) noexcept { return context.state_; }
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program>
  compile(const Context &context, RegionPlan plan) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program>
  compile(std::shared_ptr<Runtime> runtime, RegionPlan plan,
          std::span<const std::size_t> descriptor_values = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program>
  compile(std::shared_ptr<Runtime> runtime, RegionFunction region,
          std::span<const InputSpec> inputs,
          const WorkloadContext &workload = {},
          std::span<const std::size_t> descriptor_inputs = {}) noexcept;
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
// Original row math's bank port for shape/dtype-only cold warmup. This is
// neither mathematical admission nor permission to read any bank payload.
HYLE_COMPUTE_ROUTE_APPLE_API std::optional<std::size_t>
mlx_row_bank_input(const Operation &) noexcept;
// Cold facts of the original MLX row implementation. Descriptor-only warmup
// inputs are not read permissions. Actual lowering must return these same
// facts before a selected Graph artifact may authorize a projected bank.
enum class RowCacheAccessKind : std::uint8_t {
  kNone, kMap, kPaddedMap, kCheckedDecode, kPaddedCheckedDecode, kPagedIndexer
};
struct RowCacheAccessFacts final {
  RowCacheAccessKind kind = RowCacheAccessKind::kNone;
  std::uint32_t projected_input = UINT32_MAX;
  std::uint32_t query_rows = 0, max_requests = 0, max_refs = 0, page_size = 0;
  friend bool operator==(RowCacheAccessFacts, RowCacheAccessFacts) noexcept = default;
};
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::optional<RowCacheAccessFacts>>
analyze_mlx_row_access(const Operation &, std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::vector<mlx::core::array>>
lower_mlx_float(const Operation &operation,
                std::span<const TensorDesc> descriptions,
                std::span<const mlx::core::array> inputs,
                mlx::core::Stream stream, RowCacheAccessFacts * = nullptr) noexcept;

} // namespace hyle::compute::apple::detail
#endif
