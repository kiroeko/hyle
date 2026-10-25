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
// One borrowed Graph spec carries both fixed component and symbolic family
// shapes. Callers retain/bind their real leases; Graph still validates actual
// values, operation inference and constant/storage authority.
struct Operand final {
  graph::ValueId value{};
  graph::TensorSpec description{};
  Operand() noexcept = default;
  Operand(graph::ValueId v, graph::TensorSpec spec) noexcept : value(v), description(spec) {}
  Operand(graph::ValueId v, compute::TensorDesc desc) noexcept
      : value(v), description(graph::TensorSpec::concrete(desc).value_or(graph::TensorSpec{})) {}
};
struct Constants final { Operand limit_448, epsilon_1_over_128, positive_zero; };
// Optional cold facts for the caller's original node-placement contract. This
// does not own an executor, storage, or a second graph; the default path records
// nothing. A complete selected-expert projection fits in this fixed capacity.
struct Nodes final {
  std::array<graph::NodeId, 32> values{};
  std::size_t count = 0;
};

namespace detail {
using Type = compute::DataType;
using D = graph::Dimension;
inline Status plain(const Operand &v, Type type, std::size_t rank) noexcept {
  const auto &s = v.description;
  if (!v.value.valid() || !s.valid() || !s.storage_format().is_plain() ||
      s.data_type() != type || s.rank() != rank)
    return Status::InvalidArgument("CT block128 requires its declared plain dense operand type/rank");
  if (s.kind() != graph::TensorSpecKind::kDense) {
    HYLE_ASSIGN_OR_RETURN(auto concrete, s.concrete_description());
    if (!concrete.is_dense()) return Status::InvalidArgument("CT block128 requires dense operands");
  }
  for (auto n : s.dimensions())
    if (!n.valid() || (n.is_fixed() && n.fixed_value() <= 0))
      return Status::InvalidArgument("CT block128 requires positive dimensions");
  return {};
}
inline Status constants(const Constants &v) noexcept {
  HYLE_RETURN_IF_ERROR(plain(v.limit_448, Type::kBFloat16, 0));
  HYLE_RETURN_IF_ERROR(plain(v.epsilon_1_over_128, Type::kBFloat16, 0));
  return plain(v.positive_zero, Type::kBFloat16, 0);
}
inline Result<std::int64_t> fixed(D d) noexcept {
  if (!d.is_fixed() || d.fixed_value() <= 0)
    return std::unexpected(Status::InvalidArgument("CT weight and K axes must be fixed positive dimensions"));
  return d.fixed_value();
}
inline Result<std::int64_t> activation(const Operand &a) noexcept {
  HYLE_RETURN_IF_ERROR(plain(a, Type::kBFloat16, 2));
  HYLE_ASSIGN_OR_RETURN(auto k, fixed(a.description.dimensions()[1]));
  if (k % 128) return std::unexpected(Status::InvalidArgument("CT activation requires complete contiguous group128 rows"));
  return k;
}
struct WeightShape final { std::int64_t columns, depth; D rows; std::int64_t slots; };
inline Result<WeightShape> weight(const Operand &w, const Operand &sf,
    const Operand &row_blocks, const std::optional<Operand> &ids) noexcept {
  const auto rank = ids ? std::size_t{3} : std::size_t{2};
  HYLE_RETURN_IF_ERROR(plain(w, Type::kFloat8E4M3FN, rank));
  HYLE_RETURN_IF_ERROR(plain(sf, Type::kBFloat16, rank));
  HYLE_RETURN_IF_ERROR(plain(row_blocks, Type::kInt32, 1));
  const auto wd = w.description.dimensions(), sd = sf.description.dimensions();
  HYLE_ASSIGN_OR_RETURN(auto n, fixed(wd[rank - 2]));
  HYLE_ASSIGN_OR_RETURN(auto k, fixed(wd[rank - 1]));
  const auto blocks = (n - 1) / 128 + 1;
  if (k % 128 || row_blocks.description.dimensions()[0] != D::fixed(n) ||
      sd[rank - 2] != D::fixed(blocks) || sd[rank - 1] != D::fixed(k / 128) ||
      blocks > (std::numeric_limits<std::int32_t>::max)())
    return std::unexpected(Status::InvalidArgument("CT weight requires original E4/BF16 block128 axes and one I32 block row per output row"));
  auto rows = D::fixed(1); std::int64_t slots = 1;
  if (ids) {
    HYLE_RETURN_IF_ERROR(plain(*ids, Type::kInt32, 2));
    HYLE_ASSIGN_OR_RETURN(auto experts, fixed(wd[0]));
    if (wd[0] != sd[0] || experts > (std::numeric_limits<std::int32_t>::max)())
      return std::unexpected(Status::InvalidArgument("CT expert W/SF domains differ"));
    rows = ids->description.dimensions()[0];
    HYLE_ASSIGN_OR_RETURN(slots, fixed(ids->description.dimensions()[1]));
    if (rows.is_fixed() && rows.fixed_value() > (std::numeric_limits<std::int64_t>::max)() / slots)
      return std::unexpected(Status::InvalidArgument("CT selected expert extent overflows"));
  }
  return WeightShape{n, k, rows, slots};
}
inline Result<D> selected(graph::GraphBuilder &b, D rows, std::int64_t slots) noexcept {
  if (rows.is_fixed()) return D::fixed(rows.fixed_value() * slots);
  HYLE_ASSIGN_OR_RETURN(auto product, b.dimension_product(rows, D::fixed(slots)));
  return D(product);
}
template<std::size_t N>
inline Result<graph::TensorSpec> spec(Type type, const std::array<D, N> &shape) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto s, graph::TensorSpec::dense(type, shape));
  if (s.is_symbolic()) return s;
  HYLE_ASSIGN_OR_RETURN(auto concrete, s.concrete_description());
  return graph::TensorSpec::concrete(concrete);
}
struct Append final {
  graph::GraphBuilder &builder;
  Nodes *nodes;
  Status record(graph::NodeId node) const noexcept {
    if (nodes) {
      if (nodes->count == nodes->values.size()) return Status::ResourceExhausted("CT cold node list is full");
      nodes->values[nodes->count++] = node;
    }
    return {};
  }
  template<class Op, class... Inputs>
  Result<graph::ValueId> add(Op operation, const graph::TensorSpec &output, Inputs... inputs) const noexcept {
    const std::array<graph::ValueId, sizeof...(Inputs)> in{inputs...};
    const std::array out{output};
    HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(compute::Operation(operation), in, out));
    HYLE_RETURN_IF_ERROR(record(node));
    return builder.result(node);
  }
  template<std::size_t N>
  Result<graph::ValueId> reshape(graph::ValueId v, const std::array<D, N> &shape) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto view, graph::ViewSpec::reshape(shape));
    HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(view, v));
    HYLE_RETURN_IF_ERROR(record(node));
    return builder.result(node);
  }
  template<std::size_t N>
  Result<graph::ValueId> permute(graph::ValueId v, const std::array<std::int32_t, N> &axes) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto view, graph::ViewSpec::permute(axes));
    HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(view, v));
    HYLE_RETURN_IF_ERROR(record(node));
    return builder.result(node);
  }
};
} // namespace detail

// Constants are the original BF16 literals 448, 1/128 and +0. Only an exactly
// zero group scale uses epsilon. Keep the separate +0 and -0 store points.
inline Result<Operand> append_activation(graph::GraphBuilder &b, Operand a,
    const Constants &literals, Nodes *nodes = nullptr) noexcept {
  using Type = compute::DataType; using D = graph::Dimension;
  HYLE_ASSIGN_OR_RETURN(auto k, detail::activation(a));
  HYLE_RETURN_IF_ERROR(detail::constants(literals));
  const auto m = a.description.dimensions()[0]; const detail::Append out{b, nodes};
  const std::array grouped_shape{m, D::fixed(k / 128), D::fixed(128)};
  HYLE_ASSIGN_OR_RETURN(auto grouped_spec, detail::spec(Type::kBFloat16, grouped_shape));
  HYLE_ASSIGN_OR_RETURN(auto scale_spec, detail::spec(Type::kBFloat16, std::array{m, D::fixed(k / 128), D::fixed(1)}));
  HYLE_ASSIGN_OR_RETURN(auto bool_spec, detail::spec(Type::kBool, std::array{m, D::fixed(k / 128), D::fixed(1)}));
  HYLE_ASSIGN_OR_RETURN(auto e4_spec, detail::spec(Type::kFloat8E4M3FN, grouped_shape));
  HYLE_ASSIGN_OR_RETURN(auto grouped, out.reshape(a.value, grouped_shape));
  HYLE_ASSIGN_OR_RETURN(auto absolute, out.add(compute::Unary{compute::UnaryKind::kAbs}, grouped_spec, grouped));
  HYLE_ASSIGN_OR_RETURN(auto axis, compute::Axes::from(std::array<std::int32_t, 1>{2}));
  HYLE_ASSIGN_OR_RETURN(auto maximum, out.add(compute::Reduce{.kind = compute::ReductionKind::kMaximum,
      .axes = axis, .keep_dimensions = true, .accumulation_type = Type::kFloat32}, scale_spec, absolute));
  HYLE_ASSIGN_OR_RETURN(auto raw_scale, out.add(compute::Binary{compute::BinaryKind::kDivide}, scale_spec, maximum, literals.limit_448.value));
  HYLE_ASSIGN_OR_RETURN(auto nonzero, out.add(compute::Cast{Type::kBool}, bool_spec, raw_scale));
  HYLE_ASSIGN_OR_RETURN(auto scale, out.add(compute::Select{}, scale_spec, nonzero, raw_scale, literals.epsilon_1_over_128.value));
  HYLE_ASSIGN_OR_RETURN(auto normalized, out.add(compute::Binary{compute::BinaryKind::kDivide}, grouped_spec, grouped, scale));
  HYLE_ASSIGN_OR_RETURN(auto shifted, out.add(compute::Binary{compute::BinaryKind::kAdd}, grouped_spec, normalized, literals.positive_zero.value));
  HYLE_ASSIGN_OR_RETURN(auto encoded, out.add(compute::Cast{Type::kFloat8E4M3FN}, e4_spec, shifted));
  HYLE_ASSIGN_OR_RETURN(auto widened, out.add(compute::Cast{Type::kBFloat16}, grouped_spec, encoded));
  HYLE_ASSIGN_OR_RETURN(auto centered, out.add(compute::Binary{compute::BinaryKind::kSubtract}, grouped_spec, widened, literals.positive_zero.value));
  HYLE_ASSIGN_OR_RETURN(auto restored, out.add(compute::Binary{compute::BinaryKind::kMultiply}, grouped_spec, centered, scale));
  HYLE_ASSIGN_OR_RETURN(auto output, out.reshape(restored, std::array{m, D::fixed(k)}));
  HYLE_ASSIGN_OR_RETURN(auto result, detail::spec(Type::kBFloat16, std::array{m, D::fixed(k)}));
  return Operand{output, result};
}

// Original SF remains BF16 [ceil(N/128),K/128], or [E,ceil(N/128),K/128].
// row_blocks is the caller's cold I32[N] row/128. IDs are checked before decode.
inline Result<Operand> append_weight(graph::GraphBuilder &b, Operand w, Operand sf,
    Operand row_blocks, std::optional<Operand> ids = {}, Nodes *nodes = nullptr) noexcept {
  using Type = compute::DataType; using D = graph::Dimension;
  HYLE_ASSIGN_OR_RETURN(auto shape, detail::weight(w, sf, row_blocks, ids));
  const auto n = shape.columns, k = shape.depth; const detail::Append out{b, nodes};
  HYLE_ASSIGN_OR_RETURN(auto selected, detail::selected(b, shape.rows, shape.slots));
  auto raw = w.value, scales = sf.value;
  if (ids) {
    HYLE_ASSIGN_OR_RETURN(auto flat, out.reshape(ids->value, std::array{selected}));
    HYLE_ASSIGN_OR_RETURN(auto raw_spec, detail::spec(Type::kFloat8E4M3FN, std::array{selected, D::fixed(n), D::fixed(k)}));
    HYLE_ASSIGN_OR_RETURN(auto scales_spec, detail::spec(Type::kBFloat16, std::array{selected, D::fixed((n-1)/128+1), D::fixed(k/128)}));
    HYLE_ASSIGN_OR_RETURN(raw, out.add(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked}, raw_spec, raw, flat));
    HYLE_ASSIGN_OR_RETURN(scales, out.add(compute::Gather{.axis = 0, .bounds = compute::IndexBoundsPolicy::kChecked}, scales_spec, scales, flat));
  }
  const auto emit = [&]<std::size_t N>(const std::array<D,N> &expanded, const std::array<D,N> &grouped,
      const auto &flat_shape, const auto &selected_sf_shape) -> Result<Operand> {
    HYLE_ASSIGN_OR_RETURN(auto selected_sf_spec, detail::spec(Type::kBFloat16, selected_sf_shape));
    HYLE_ASSIGN_OR_RETURN(auto selected_sf, out.add(compute::Gather{.axis = ids ? 1 : 0,
        .bounds = compute::IndexBoundsPolicy::kChecked}, selected_sf_spec, scales, row_blocks.value));
    HYLE_ASSIGN_OR_RETURN(auto scales_view, out.reshape(selected_sf, expanded));
    HYLE_ASSIGN_OR_RETURN(auto codes_spec, detail::spec(Type::kBFloat16, flat_shape));
    HYLE_ASSIGN_OR_RETURN(auto codes, out.add(compute::Cast{Type::kBFloat16}, codes_spec, raw));
    HYLE_ASSIGN_OR_RETURN(auto codes_view, out.reshape(codes, grouped));
    HYLE_ASSIGN_OR_RETURN(auto restored_spec, detail::spec(Type::kBFloat16, grouped));
    HYLE_ASSIGN_OR_RETURN(auto restored, out.add(compute::Binary{compute::BinaryKind::kMultiply}, restored_spec, codes_view, scales_view));
    HYLE_ASSIGN_OR_RETURN(auto output, out.reshape(restored, flat_shape));
    return Operand{output, codes_spec};
  };
  if (ids) return emit(std::array{selected,D::fixed(n),D::fixed(k/128),D::fixed(1)},
      std::array{selected,D::fixed(n),D::fixed(k/128),D::fixed(128)},
      std::array{selected,D::fixed(n),D::fixed(k)}, std::array{selected,D::fixed(n),D::fixed(k/128)});
  return emit(std::array{D::fixed(n),D::fixed(k/128),D::fixed(1)},
      std::array{D::fixed(n),D::fixed(k/128),D::fixed(128)},
      std::array{D::fixed(n),D::fixed(k)}, std::array{D::fixed(n),D::fixed(k/128)});
}

inline Result<Operand> append_matmul(graph::GraphBuilder &b, Operand aq,
    Operand decoded, std::optional<Operand> ids = {}, Nodes *nodes = nullptr) noexcept {
  using Type = compute::DataType; using D = graph::Dimension;
  HYLE_ASSIGN_OR_RETURN(auto k, detail::activation(aq));
  HYLE_RETURN_IF_ERROR(detail::plain(decoded, Type::kBFloat16, ids ? 3 : 2));
  const auto m = aq.description.dimensions()[0]; const auto wd = decoded.description.dimensions();
  HYLE_ASSIGN_OR_RETURN(auto n, detail::fixed(wd[ids ? 1 : 0]));
  if (wd[ids ? 2 : 1] != D::fixed(k))
    return std::unexpected(Status::InvalidArgument("CT projection activation/weight depth differs"));
  const detail::Append out{b,nodes};
  if (ids) {
    HYLE_RETURN_IF_ERROR(detail::plain(*ids, Type::kInt32, 2));
    HYLE_ASSIGN_OR_RETURN(auto slots, detail::fixed(ids->description.dimensions()[1]));
    if (ids->description.dimensions()[0] != m ||
        (m.is_fixed() && m.fixed_value() > (std::numeric_limits<std::int64_t>::max)()/slots))
      return std::unexpected(Status::InvalidArgument("CT projection selected expert rows differ"));
    HYLE_ASSIGN_OR_RETURN(auto selected, detail::selected(b,m,slots));
    HYLE_RETURN_IF_ERROR(b.require_dimension_equal(wd[0], selected));
    HYLE_ASSIGN_OR_RETURN(auto rhs, out.reshape(decoded.value,std::array{m,D::fixed(slots),D::fixed(n),D::fixed(k)}));
    HYLE_ASSIGN_OR_RETURN(auto transposed, out.permute(rhs,std::array<std::int32_t,4>{0,1,3,2}));
    HYLE_ASSIGN_OR_RETURN(auto lhs, out.reshape(aq.value,std::array{m,D::fixed(1),D::fixed(1),D::fixed(k)}));
    HYLE_ASSIGN_OR_RETURN(auto product_spec, detail::spec(Type::kBFloat16,std::array{m,D::fixed(slots),D::fixed(1),D::fixed(n)}));
    HYLE_ASSIGN_OR_RETURN(auto product, out.add(compute::MatMul{.accumulation_type=Type::kFloat32},product_spec,lhs,transposed));
    HYLE_ASSIGN_OR_RETURN(auto output, out.reshape(product,std::array{m,D::fixed(slots),D::fixed(n)}));
    HYLE_ASSIGN_OR_RETURN(auto result, detail::spec(Type::kBFloat16,std::array{m,D::fixed(slots),D::fixed(n)}));
    return Operand{output,result};
  }
  HYLE_ASSIGN_OR_RETURN(auto rhs, out.permute(decoded.value,std::array<std::int32_t,2>{1,0}));
  HYLE_ASSIGN_OR_RETURN(auto result, detail::spec(Type::kBFloat16,std::array{m,D::fixed(n)}));
  HYLE_ASSIGN_OR_RETURN(auto output, out.add(compute::MatMul{.accumulation_type=Type::kFloat32},result,aq.value,rhs));
  return Operand{output,result};
}

inline Result<Operand> append_projection(graph::GraphBuilder &b, Operand a,
    Operand w, Operand sf, Operand row_blocks, const Constants &literals,
    std::optional<Operand> ids = {}, Nodes *nodes = nullptr) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto k, detail::activation(a));
  HYLE_RETURN_IF_ERROR(detail::constants(literals));
  HYLE_ASSIGN_OR_RETURN(auto shape, detail::weight(w,sf,row_blocks,ids));
  if (shape.depth != k || (ids && shape.rows != a.description.dimensions()[0]))
    return std::unexpected(Status::InvalidArgument("CT projection activation/expert rows or depth differ"));
  HYLE_ASSIGN_OR_RETURN(auto aq, append_activation(b,a,literals,nodes));
  HYLE_ASSIGN_OR_RETURN(auto decoded, append_weight(b,w,sf,row_blocks,ids,nodes));
  return append_matmul(b,aq,decoded,ids,nodes);
}
} // namespace hyle::inference::ct_block128
