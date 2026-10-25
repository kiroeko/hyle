#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include "hyle/graph/graph.h"

namespace hyle::inference::ct_block128 {

// Cold ordinary Graph mathematics for the explicit CT BF16 block128 reader.
// These are borrowed Value/description facts, not storage or a second IR.
// The caller supplies their actual Value descriptions, not requested coercions,
// and retains/binds their real leases. Graph keeps its normal Value identity,
// operation inference and constant/storage validation.
struct Operand final {
  graph::ValueId value{};
  compute::TensorDesc description{};
};
struct Constants final {
  Operand limit_448, epsilon_1_over_128, positive_zero;
};

namespace detail {
using Type = compute::DataType;
inline Status plain(const Operand &v, Type type, std::size_t rank) noexcept {
  if (!v.value.valid() || !v.description.valid() || !v.description.is_plain() ||
      !v.description.is_dense() || v.description.data_type() != type || v.description.rank() != rank)
    return Status::InvalidArgument("CT block128 requires its declared plain dense operand type/rank");
  for (auto n : v.description.dimensions())
    if (n <= 0) return Status::InvalidArgument("CT block128 requires positive dimensions");
  return {};
}
inline Status constants(const Constants &v) noexcept {
  HYLE_RETURN_IF_ERROR(plain(v.limit_448, Type::kBFloat16, 0));
  HYLE_RETURN_IF_ERROR(plain(v.epsilon_1_over_128, Type::kBFloat16, 0));
  return plain(v.positive_zero, Type::kBFloat16, 0);
}
inline Status activation(const Operand &a) noexcept {
  HYLE_RETURN_IF_ERROR(plain(a, Type::kBFloat16, 2));
  if (a.description.dimensions()[1] % 128)
    return Status::InvalidArgument("CT activation requires complete contiguous group128 rows");
  return {};
}
struct WeightShape final { std::int64_t columns, depth, rows, slots; };
inline Result<WeightShape> weight(const Operand &w, const Operand &sf,
    const Operand &row_blocks, const std::optional<Operand> &ids) noexcept {
  const auto rank = ids ? std::size_t{3} : std::size_t{2};
  HYLE_RETURN_IF_ERROR(plain(w, Type::kFloat8E4M3FN, rank));
  HYLE_RETURN_IF_ERROR(plain(sf, Type::kBFloat16, rank));
  HYLE_RETURN_IF_ERROR(plain(row_blocks, Type::kInt32, 1));
  const auto wd = w.description.dimensions(), sd = sf.description.dimensions();
  const auto n = wd[rank - 2], k = wd[rank - 1];
  const auto blocks = (n - 1) / 128 + 1;
  if (k % 128 || row_blocks.description.dimensions()[0] != n ||
      sd[rank - 2] != blocks || sd[rank - 1] != k / 128 ||
      blocks > (std::numeric_limits<std::int32_t>::max)())
    return std::unexpected(Status::InvalidArgument("CT weight requires original E4/BF16 block128 axes and one I32 block row per output row"));
  std::int64_t rows = 1, slots = 1;
  if (ids) {
    HYLE_RETURN_IF_ERROR(plain(*ids, Type::kInt32, 2));
    if (wd[0] != sd[0] || wd[0] > (std::numeric_limits<std::int32_t>::max)())
      return std::unexpected(Status::InvalidArgument("CT expert W/SF domains differ"));
    rows = ids->description.dimensions()[0]; slots = ids->description.dimensions()[1];
    if (rows > (std::numeric_limits<std::int64_t>::max)() / slots)
      return std::unexpected(Status::InvalidArgument("CT selected expert extent overflows"));
  }
  return WeightShape{n, k, rows, slots};
}
template<std::size_t N>
inline Result<graph::ValueId> reshape(graph::GraphBuilder &b, graph::ValueId v,
    const std::array<std::int64_t, N> &shape) noexcept {
  std::array<graph::Dimension, N> dimensions;
  for (std::size_t i = 0; i < N; ++i) dimensions[i] = graph::Dimension::fixed(shape[i]);
  HYLE_ASSIGN_OR_RETURN(auto view, graph::ViewSpec::reshape(dimensions));
  return b.add(view, v);
}
template<std::size_t N>
inline Result<Operand> described(graph::ValueId v, Type type,
    const std::array<std::int64_t, N> &shape) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto desc, compute::TensorDesc::dense(type, shape));
  return Operand{v, desc};
}
} // namespace detail

// Constants are the original BF16 literals 448, 1/128 and +0. Only an exactly
// zero group scale uses epsilon. No floor, finite guard or scale conversion is
// inserted. Keep +0 and -0: they are separate producer rounding/store points.
inline Result<Operand> append_activation(graph::GraphBuilder &b, Operand a,
    const Constants &literals) noexcept {
  using Type = compute::DataType;
  HYLE_RETURN_IF_ERROR(detail::activation(a));
  HYLE_RETURN_IF_ERROR(detail::constants(literals));
  const auto m = a.description.dimensions()[0], k = a.description.dimensions()[1];
  HYLE_ASSIGN_OR_RETURN(auto grouped, detail::reshape(b, a.value, std::array{m, k / 128, std::int64_t{128}}));
  HYLE_ASSIGN_OR_RETURN(auto absolute, b.add(compute::Unary{compute::UnaryKind::kAbs}, grouped));
  HYLE_ASSIGN_OR_RETURN(auto axis, compute::Axes::from(std::array<std::int32_t, 1>{2}));
  HYLE_ASSIGN_OR_RETURN(auto maximum, b.add(compute::Reduce{.kind = compute::ReductionKind::kMaximum,
      .axes = axis, .keep_dimensions = true, .accumulation_type = Type::kFloat32}, absolute));
  HYLE_ASSIGN_OR_RETURN(auto raw_scale, b.add(compute::Binary{compute::BinaryKind::kDivide}, maximum, literals.limit_448.value));
  HYLE_ASSIGN_OR_RETURN(auto nonzero, b.add(compute::Cast{Type::kBool}, raw_scale));
  HYLE_ASSIGN_OR_RETURN(auto scale, b.add(compute::Select{}, nonzero, raw_scale, literals.epsilon_1_over_128.value));
  HYLE_ASSIGN_OR_RETURN(auto normalized, b.add(compute::Binary{compute::BinaryKind::kDivide}, grouped, scale));
  HYLE_ASSIGN_OR_RETURN(auto shifted, b.add(compute::Binary{compute::BinaryKind::kAdd}, normalized, literals.positive_zero.value));
  HYLE_ASSIGN_OR_RETURN(auto encoded, b.add(compute::Cast{Type::kFloat8E4M3FN}, shifted));
  HYLE_ASSIGN_OR_RETURN(auto widened, b.add(compute::Cast{Type::kBFloat16}, encoded));
  HYLE_ASSIGN_OR_RETURN(auto centered, b.add(compute::Binary{compute::BinaryKind::kSubtract}, widened, literals.positive_zero.value));
  HYLE_ASSIGN_OR_RETURN(auto restored, b.add(compute::Binary{compute::BinaryKind::kMultiply}, centered, scale));
  HYLE_ASSIGN_OR_RETURN(auto output, detail::reshape(b, restored, std::array{m, k}));
  return detail::described(output, Type::kBFloat16, std::array{m, k});
}

// row_blocks is the caller's cold I32[N] literal row/128, never a hot policy.
// Original SF remains BF16 [ceil(N/128),K/128], or [E,ceil(N/128),K/128]. Select checked
// expert IDs before decode; no full E*N*K floating bank is materialized.
inline Result<Operand> append_weight(graph::GraphBuilder &b, Operand w, Operand sf,
    Operand row_blocks, std::optional<Operand> ids = {}) noexcept {
  using Type = compute::DataType;
  HYLE_ASSIGN_OR_RETURN(auto shape, detail::weight(w, sf, row_blocks, ids));
  const auto n = shape.columns, k = shape.depth, selected = shape.rows * shape.slots;
  auto raw = w.value, scales = sf.value;
  if (ids) {
    HYLE_ASSIGN_OR_RETURN(auto flat, detail::reshape(b, ids->value, std::array{selected}));
    HYLE_ASSIGN_OR_RETURN(raw, b.add(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked}, raw, flat));
    HYLE_ASSIGN_OR_RETURN(scales, b.add(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked}, scales, flat));
  }
  HYLE_ASSIGN_OR_RETURN(auto selected_sf, b.add(compute::Gather{.axis = ids ? 1 : 0,
      .bounds = compute::IndexBoundsPolicy::kChecked}, scales, row_blocks.value));
  graph::ValueId scales_view;
  if (ids) { HYLE_ASSIGN_OR_RETURN(scales_view, detail::reshape(b, selected_sf, std::array{selected, n, k / 128, std::int64_t{1}})); }
  else { HYLE_ASSIGN_OR_RETURN(scales_view, detail::reshape(b, selected_sf, std::array{n, k / 128, std::int64_t{1}})); }
  HYLE_ASSIGN_OR_RETURN(auto codes, b.add(compute::Cast{Type::kBFloat16}, raw));
  graph::ValueId codes_view;
  if (ids) { HYLE_ASSIGN_OR_RETURN(codes_view, detail::reshape(b, codes, std::array{selected, n, k / 128, std::int64_t{128}})); }
  else { HYLE_ASSIGN_OR_RETURN(codes_view, detail::reshape(b, codes, std::array{n, k / 128, std::int64_t{128}})); }
  HYLE_ASSIGN_OR_RETURN(auto restored, b.add(compute::Binary{compute::BinaryKind::kMultiply}, codes_view, scales_view));
  if (ids) {
    HYLE_ASSIGN_OR_RETURN(auto output, detail::reshape(b, restored, std::array{selected, n, k}));
    return detail::described(output, Type::kBFloat16, std::array{selected, n, k});
  }
  HYLE_ASSIGN_OR_RETURN(auto output, detail::reshape(b, restored, std::array{n, k}));
  return detail::described(output, Type::kBFloat16, std::array{n, k});
}

inline Result<Operand> append_matmul(graph::GraphBuilder &b, Operand aq,
    Operand decoded, std::optional<Operand> ids = {}) noexcept {
  using Type = compute::DataType;
  HYLE_RETURN_IF_ERROR(detail::activation(aq));
  HYLE_RETURN_IF_ERROR(detail::plain(decoded, Type::kBFloat16, ids ? 3 : 2));
  const auto m = aq.description.dimensions()[0], k = aq.description.dimensions()[1];
  const auto wd = decoded.description.dimensions();
  const auto n = wd[ids ? 1 : 0];
  if (wd[ids ? 2 : 1] != k)
    return std::unexpected(Status::InvalidArgument("CT projection activation/weight depth differs"));
  if (ids) {
    HYLE_RETURN_IF_ERROR(detail::plain(*ids, Type::kInt32, 2));
    const auto slots = ids->description.dimensions()[1];
    if (ids->description.dimensions()[0] != m ||
        m > (std::numeric_limits<std::int64_t>::max)() / slots || wd[0] != m * slots)
      return std::unexpected(Status::InvalidArgument("CT projection selected expert rows differ"));
    HYLE_ASSIGN_OR_RETURN(auto rhs, detail::reshape(b, decoded.value, std::array{m, slots, n, k}));
    HYLE_ASSIGN_OR_RETURN(auto view, graph::ViewSpec::permute(std::array<std::int32_t, 4>{0, 1, 3, 2}));
    HYLE_ASSIGN_OR_RETURN(auto transposed, b.add(view, rhs));
    HYLE_ASSIGN_OR_RETURN(auto lhs, detail::reshape(b, aq.value, std::array{m, std::int64_t{1}, std::int64_t{1}, k}));
    HYLE_ASSIGN_OR_RETURN(auto product, b.add(compute::MatMul{.accumulation_type = Type::kFloat32}, lhs, transposed));
    HYLE_ASSIGN_OR_RETURN(auto output, detail::reshape(b, product, std::array{m, slots, n}));
    return detail::described(output, Type::kBFloat16, std::array{m, slots, n});
  }
  HYLE_ASSIGN_OR_RETURN(auto view, graph::ViewSpec::permute(std::array<std::int32_t, 2>{1, 0}));
  HYLE_ASSIGN_OR_RETURN(auto rhs, b.add(view, decoded.value));
  HYLE_ASSIGN_OR_RETURN(auto output, b.add(compute::MatMul{.accumulation_type = Type::kFloat32}, aq.value, rhs));
  return detail::described(output, Type::kBFloat16, std::array{m, n});
}

inline Result<Operand> append_projection(graph::GraphBuilder &b, Operand a,
    Operand w, Operand sf, Operand row_blocks, const Constants &literals,
    std::optional<Operand> ids = {}) noexcept {
  HYLE_RETURN_IF_ERROR(detail::activation(a));
  HYLE_RETURN_IF_ERROR(detail::constants(literals));
  HYLE_ASSIGN_OR_RETURN(auto shape, detail::weight(w, sf, row_blocks, ids));
  if (shape.depth != a.description.dimensions()[1] ||
      (ids && shape.rows != a.description.dimensions()[0]))
    return std::unexpected(Status::InvalidArgument("CT projection activation/expert rows or depth differ"));
  HYLE_ASSIGN_OR_RETURN(auto aq, append_activation(b, a, literals));
  HYLE_ASSIGN_OR_RETURN(auto decoded, append_weight(b, w, sf, row_blocks, ids));
  return append_matmul(b, aq, decoded, ids);
}

} // namespace hyle::inference::ct_block128
