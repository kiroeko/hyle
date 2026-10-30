// Cold graph construction for one Qwen model package. Compile under each
// consumer's existing Backend capability contract, independently of main.cc.
#include "graph_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../../inference/backend.h"
#include "../../inference/ct_block128_graph.h"
#include "hyle/simd/dtype.h"

namespace qwen35 {
namespace inference = hyle::inference;
using inference::Backend;

GraphAssembly::Scope::Scope(GraphAssembly &selected, Location location)
    : assembly(selected), previous(selected.location_) { assembly.location_ = location; }
GraphAssembly::Scope::~Scope() { assembly.location_ = previous; }

GraphAssembly::GraphAssembly(const Backend &environment, PrecisionProfile precision,
                         const c::ResourceBudget &budget)
      : bindings_(environment.storage_domains()), backend_(environment), budget_(budget), precision_(precision), mixed_(environment.has_cpu_assist()),
        predicate_type_(environment.native_generated_storage() ? Type::kFloat32 : Type::kBool) {}

GraphAssembly::Location GraphAssembly::primary() { return {g::ExecutionTargetId{0}, g::StorageDomainId{0}}; }

GraphAssembly::Location GraphAssembly::cpu() const {
  if (!mixed_) throw std::runtime_error("CPU FFN requires the configured CPU endpoint");
  return {g::ExecutionTargetId{1}, g::StorageDomainId{1}};
}

Tensor &GraphAssembly::literal_at(Location location, c::TensorDesc description) {
  literals.emplace_back(take(backend_.resource_for(location.storage)), description, true, budget_);
  return literals.back();
}

Tensor &GraphAssembly::literal(c::TensorDesc description) { return literal_at(location_, description); }

Type GraphAssembly::predicate_type() const { return predicate_type_; }

Type GraphAssembly::activation_type() const noexcept { return precision_.activation_type(); }

void GraphAssembly::node(g::NodeId node) {
  if (mixed_) check(bindings_.node(node, location_));
}

void GraphAssembly::external(V value, g::StorageDomainId storage) {
  if (!mixed_) return;
  check(bindings_.external(value, storage));
}

std::span<const g::ConstantBinding> GraphAssembly::constants() const noexcept { return bindings_.constants(); }

g::CompileOptions GraphAssembly::options(g::CompileOptions options) const {
  return bindings_.options(options);
}

void GraphAssembly::bind_constant(V value, Tensor &tensor) {
  tensor.freeze(backend_);
  const auto read = tensor.read();
  const auto owner = take(backend_.storage_owner_for(read.storage().resource()));
  // Native resources are explicitly borrowed from Model's Backend. The
  // bank/literal owns its lease through every compiled child. A native lease
  // requires its actual StorageOwner; only process-system storage can be
  // retained without one. Graph validates that allocator identity separately.
  const auto retained = owner || read.storage().resource() == mem::system_resource()
      ? tensor.ready : c::ReadLease{};
  check(bindings_.bind_constant({value, read, g::ConstantPolicy::kBorrow, owner, retained}));
}

Recipe::Recipe(std::map<std::string, LoadedWeight> &selected, const ProjectionBindings &selected_projections,
         GraphAssembly &graph_assembly, const ModelGeometry &model_geometry,
         std::size_t max_query_rows, std::size_t requests,
         const WeightMeanings *parameter_meanings)
      : builder(take(g::GraphBuilder::create())), bank(selected), projections(selected_projections), meanings(parameter_meanings), assembly(graph_assembly), geometry(model_geometry), hidden(model_geometry.hidden) {
  // L is a zero-based last-token position, independently bound at startup.
  // T=L+1 avoids subtraction or a second independent (possibly inconsistent)
  // shape binding. All other numeric axes remain fixed model dimensions.
  last_row = take(builder.dimension({0, static_cast<std::int64_t>(max_query_rows - 1), 1}));
  token_rows = D(take(builder.dimension_sum(D(last_row), f(1))));
  packed_rows = requests == 1 ? token_rows :
      D(take(builder.dimension_product(f(static_cast<std::int64_t>(requests)), token_rows)));
  sample_rows = D(take(builder.dimension_sum(token_rows, f(geometry.conv_kernel - 1))));
}

g::TensorSpec Recipe::dense_spec(Type type, std::span<const D> dimensions) {
  return take(g::TensorSpec::dense(type, dimensions));
}

g::TensorSpec Recipe::dense_spec(Type type, std::initializer_list<D> dimensions) {
  return dense_spec(type, std::span(dimensions.begin(), dimensions.size()));
}

Type Recipe::activation_type() const noexcept { return assembly.activation_type(); }

const g::TensorSpec &Recipe::spec(V value) const { return specs.at(value.ordinal()); }

V Recipe::remember(V value, g::TensorSpec description) {
  if (specs.size() <= value.ordinal()) specs.resize(value.ordinal() + 1);
  specs[value.ordinal()] = description;
  return value;
}

V Recipe::input(g::TensorSpec description) {
  const auto value = remember(take(builder.input(description)), description); assembly.external(value); return value;
}

V Recipe::input(c::TensorDesc description) { return input(take(g::TensorSpec::concrete(description))); }

V Recipe::constant(Tensor &tensor) {
  const auto value = remember(take(builder.constant(tensor.description)),
                              take(g::TensorSpec::concrete(tensor.description)));
  assembly.bind_constant(value, tensor);
  return value;
}

V Recipe::operand(Tensor &tensor) {
  const auto found = operands.find(&tensor);
  if (found != operands.end()) return found->second;
  const auto value = constant(tensor);
  operands.emplace(&tensor, value);
  return value;
}

V Recipe::weight_absolute(const std::string &name) {
  const auto original = bank.find(name);
  if (original == bank.end()) throw std::runtime_error("missing original model role: " + name);
  return operand(original->second.tensor);
}

V Recipe::weight(const std::string &name) {
  return weight_absolute("model.language_model." + name);
}

prepared::ParameterMeaning Recipe::meaning(const std::string &name) const {
  if (meanings) if (const auto found = meanings->find(name); found != meanings->end()) return found->second;
  return prepared::ParameterMeaning::kHfRaw;
}

V Recipe::embedding(V tokens, const std::string &name) {
  const auto ids = spec(tokens);
  if (const auto found = projections.find(name); found != projections.end())
    if (const auto *mlx = std::get_if<MlxAffineProjection>(&found->second)) {
      if (activation_type() != Type::kBFloat16 || ids.rank() != 1 || ids.data_type() != Type::kInt32)
        throw std::runtime_error("MLX source embedding requires BF16 and checked I32 rows");
      const auto shape = projection_detail::logical_shape(found->second);
      if (shape[1] != hidden) throw std::runtime_error("MLX embedding logical width differs");
      // The embedding has typed BF16 product/add boundaries. Its tied head
      // reuses these original operands as a linear operation, not this table.
      return one(c::model_ops::mlx_quantized_embedding(
          mlx->equation.bits, mlx->equation.group_size),
          {operand(*mlx->codes), operand(*mlx->scales), operand(*mlx->biases), tokens},
          dense_spec(Type::kBFloat16, {ids.dimensions()[0], f(hidden)}));
    }
  const auto original = weight_absolute(name);
  const auto stored = spec(original);
  if (ids.rank() != 1 || ids.data_type() != Type::kInt32 || stored.rank() != 2 ||
      stored.dimensions()[1] != f(hidden))
    throw std::runtime_error("embedding requires I32 rows and original [vocabulary,hidden] storage: " + name);
  // Gather decodes selected original rows only, then applies the model's
  // activation boundary. The shared vocabulary bank is never expanded.
  // Encoded row decoders carry checked IDs as part of their storage contract.
  const auto bounds = c::IndexBoundsPolicy::kChecked;
  return cast(one(c::Gather{.bounds = bounds}, {original, tokens},
      dense_spec(stored.data_type(), {ids.dimensions()[0], stored.dimensions()[1]})), activation_type());
}

V Recipe::scalar(float value, Type type) {
  auto &literal = assembly.literal(dense(type, {1}));
  if (type == Type::kFloat32) literal.write<float>(std::span(&value, 1));
  else if (type == Type::kFloat16) {
    const auto f16 = hyle::simd::f32_to_f16(value);
    literal.write<hyle::simd::float16_t>(std::span(&f16, 1));
  } else if (type == Type::kBFloat16) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    const auto bf16 = static_cast<std::uint16_t>(bits >> 16);
    literal.write<std::uint16_t>(std::span(&bf16, 1));
  } else throw std::runtime_error("floating scalar requires F32/F16/BF16");
  return constant(literal);
}

std::vector<V> Recipe::op(c::Operation operation, std::initializer_list<V> arguments,
                    std::initializer_list<g::TensorSpec> outputs,
                    g::NodeId *node_id) {
  const auto node = take(builder.add_node(operation,
      std::span(arguments.begin(), arguments.size()),
      std::span(outputs.begin(), outputs.size())));
  assembly.node(node);
  if (node_id) *node_id = node;
  std::vector<V> result;
  std::size_t index = 0;
  for (const auto &output : outputs)
    result.push_back(remember(take(builder.result(node, index++)), output));
  return result;
}

std::vector<V> Recipe::op(c::Operation operation, std::initializer_list<V> arguments,
                    g::NodeId *node_id) {
  std::vector<c::TensorDesc> in;
  for (auto value : arguments) in.push_back(take(spec(value).concrete_description()));
  std::vector<c::TensorDesc> out(c::operation_traits(operation).outputs);
  check(c::infer_output_descriptions(operation, in, out));
  const auto node = take(builder.add_node(operation, std::span(arguments.begin(), arguments.size())));
  assembly.node(node);
  if (node_id) *node_id = node;
  std::vector<V> result;
  for (std::size_t i = 0; i < out.size(); ++i)
    result.push_back(remember(take(builder.result(node, i)), take(g::TensorSpec::concrete(out[i]))));
  return result;
}

V Recipe::cast(V x, Type type) {
  return spec(x).data_type() == type ? x :
      one(c::Cast{type}, {x}, dense_spec(type, spec(x).dimensions()));
}

V Recipe::host_predicate(V value) {
  return cast(value, assembly.predicate_type());
}

V Recipe::unary(c::UnaryKind kind, V x) { return same(c::Unary{kind}, {x}, x); }

V Recipe::binary(c::BinaryKind kind, V x, V y) {
  const auto left = spec(x), right = spec(y);
  if (left.data_type() != right.data_type()) throw std::runtime_error("binary dtype mismatch");
  const auto rank = std::max(left.rank(), right.rank());
  std::vector<D> dimensions(rank, f(1));
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const auto a = axis + left.rank() < rank ? f(1) : left.dimensions()[axis + left.rank() - rank];
    const auto b = axis + right.rank() < rank ? f(1) : right.dimensions()[axis + right.rank() - rank];
    if (a == b || b == f(1)) dimensions[axis] = a;
    else if (a == f(1)) dimensions[axis] = b;
    else dimensions[axis] = D(take(builder.dimension_broadcast(a, b)));
  }
  return one(c::Binary{kind}, {x, y}, dense_spec(left.data_type(), dimensions));
}

V Recipe::reshape(V x, std::initializer_list<D> dimensions) {
  // A fixed constant may still need the original stride-chunk Copy. Dynamic
  // views preserve inference and receive explicit Copy only where the actual
  // model layout requires it (the self-attention interleaved gate below).
  bool fixed = true;
  std::vector<std::int64_t> concrete;
  for (auto dimension : dimensions) {
    fixed = fixed && dimension.is_fixed();
    concrete.push_back(dimension.fixed_value());
  }
  if (auto input = spec(x).concrete_description(); input && fixed) {
    auto view = take(c::View::reshape(take(c::TensorShape::create(concrete))));
    auto inferred = c::infer_single_output(c::Operation(view), std::array{*input});
    if (!inferred && inferred.error().IsUnimplemented()) x = copy(x);
  }
  const auto view = take(g::ViewSpec::reshape(std::span(dimensions.begin(), dimensions.size())));
  const auto node = take(builder.add_node(view, x)); assembly.node(node);
  const auto output = take(builder.result(node));
  return remember(output, take(g::TensorSpec::inferred(spec(x).data_type(),
      std::span(dimensions.begin(), dimensions.size()), spec(x).storage_format())));
}

V Recipe::copy(V x) { return same(c::Copy{}, {x}, x); }

V Recipe::concat(std::span<const V> inputs, int axis, g::TensorSpec output) {
  if (inputs.size() == 1) return inputs.front();
  const std::array<g::TensorSpec, 1> outputs{output};
  const auto node = take(builder.add_node(c::Operation(c::Concat{.axis = axis}), inputs, outputs));
  assembly.node(node);
  return remember(take(builder.result(node)), std::move(output));
}

V Recipe::slice(V x, int axis, D first, D count) {
  const auto source = spec(x);
  std::vector<D> dimensions(source.dimensions().begin(), source.dimensions().end());
  dimensions.at(static_cast<std::size_t>(axis)) = count;
  const auto node = take(builder.add_node(take(g::ViewSpec::slice(axis, first, count)), x)); assembly.node(node);
  const auto output = take(builder.result(node));
  return remember(output, take(g::TensorSpec::inferred(source.data_type(), dimensions,
                                                     source.storage_format())));
}

V Recipe::linear(V x, const std::string &name) {
  return linear_absolute(x, "model.language_model." + name);
}

const ProjectionBinding &Recipe::projection(const std::string &name) const {
  const auto found = projections.find(name);
  if (found == projections.end()) throw std::runtime_error("missing cold model projection binding: " + name);
  return found->second;
}

V Recipe::emit_projection(V x, const ProjectionBinding &binding, bool scores) {
  if (spec(x).rank() != 2 || spec(x).data_type() != activation_type())
    throw std::runtime_error("model projection differs from source activation [rows,K] precision");
  auto dimensions = std::vector<D>(spec(x).dimensions().begin(), spec(x).dimensions().end());
  // Cold bindings already validate the source logical [N,K] layout.
  dimensions.back() = f(projection_detail::logical_shape(binding)[0]);
  const auto result = std::visit([&](const auto &projection) -> V {
    using P = std::decay_t<decltype(projection)>;
    if constexpr (std::is_same_v<P, StoredProjection>) {
      const auto original = operand(*projection.weight);
      const auto stored = spec(original);
      if (stored.rank() != 2 || spec(x).dimensions()[1] != stored.dimensions()[1])
        throw std::runtime_error("original stored projection type/shape differs");
      if (activation_type() == Type::kFloat16) {
        if (stored.data_type() != Type::kFloat16 || !projection.weight->description.is_plain())
          throw std::runtime_error("F16 source projection requires a dense F16 operand");
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        return one(c::MatMul{.accumulation_type = Type::kFloat32},
            {x, one(take(c::View::permute(axes)), {original})}, dense_spec(Type::kFloat16, dimensions));
      }
      if (c::gguf_quantized_type(stored.storage_format())) {
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {original});
        const auto output = one(c::MatMul{}, {cast(x, Type::kFloat32), rhs}, dense_spec(Type::kFloat32, dimensions));
        return scores ? output : cast(output, Type::kBFloat16);
      }
      if (scores || stored.data_type() == Type::kFloat16 || stored.data_type() == Type::kFloat32) {
        const auto output = one(c::model_ops::dense_linear_f32(),
            {stored.data_type() == Type::kBFloat16 ? x : cast(x, Type::kFloat32), original},
            dense_spec(Type::kFloat32, dimensions));
        return scores ? output : cast(output, Type::kBFloat16);
      }
      if (stored.data_type() != Type::kBFloat16)
        throw std::runtime_error("original dense projection requires a supported floating operand");
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      const auto rhs = one(take(c::View::permute(axes)), {original});
      return one(c::MatMul{}, {x, rhs}, dense_spec(Type::kBFloat16, dimensions));
    } else if constexpr (std::is_same_v<P, MlxAffineProjection>) {
      if (activation_type() != Type::kBFloat16) throw std::runtime_error("MLX source projection requires BF16 activation");
      return one(c::Operation::custom(projection.equation),
          {x, operand(*projection.codes), operand(*projection.scales), operand(*projection.biases)},
          dense_spec(Type::kBFloat16, dimensions));
    } else if constexpr (std::is_same_v<P, Nf4Projection>) {
      if (activation_type() != Type::kFloat16) throw std::runtime_error("NF4 source equation requires F16 activation");
      PackedProjectionNodes nodes; nodes.count = 3;
      const auto decoded = one(c::Operation::custom(projection.decode),
          {operand(*projection.original[0]), operand(*projection.original[1]),
           operand(*projection.original[2]), operand(*projection.original[3]),
           operand(*projection.original[4]), operand(*projection.original[5])}, &nodes.chain[0]);
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      const auto rhs = one(take(c::View::permute(axes)), {decoded}, &nodes.chain[1]);
      const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
          dense_spec(Type::kFloat16, dimensions), &nodes.chain[2]);
      packed_projection_nodes.push_back(nodes);
      return output;
    } else if constexpr (std::is_same_v<P, GptqProjection>) {
      if (activation_type() != Type::kBFloat16) throw std::runtime_error("GPTQ source equation requires BF16 activation");
      const auto codes = operand(*projection.codes), zeros = operand(*projection.zeros);
      const auto scales = operand(*projection.scales), groups = operand(*projection.groups);
      // Preserve F16 decode, BF16 operand and the same four-node fusion region.
      PackedProjectionNodes nodes;
      const auto decoded = one(c::Operation::custom(projection.decode), {codes, zeros, scales, groups}, &nodes.chain[0]);
      const auto converted = one(c::Cast{Type::kBFloat16}, {decoded}, &nodes.chain[1]);
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      const auto rhs = one(take(c::View::permute(axes)), {converted}, &nodes.chain[2]);
      const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
          dense_spec(Type::kBFloat16, dimensions), &nodes.chain[3]);
      packed_projection_nodes.push_back(nodes);
      return output;
    } else if constexpr (std::is_same_v<P, CtPackedProjection>) {
      if (activation_type() != Type::kBFloat16) throw std::runtime_error("CT packed source equation requires BF16 activation");
      PackedProjectionNodes nodes; nodes.count = 3;
      const auto decoded = projection.zeros ? one(c::Operation::custom(projection.decode),
          {operand(*projection.codes), operand(*projection.zeros), operand(*projection.scales)}, &nodes.chain[0]) :
          one(c::Operation::custom(projection.decode), {operand(*projection.codes), operand(*projection.scales)}, &nodes.chain[0]);
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      const auto rhs = one(take(c::View::permute(axes)), {decoded}, &nodes.chain[1]);
      const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
          dense_spec(Type::kBFloat16, dimensions), &nodes.chain[2]);
      packed_projection_nodes.push_back(nodes);
      return output;
    } else if constexpr (std::is_same_v<P, LocalNvfp4Projection>) {
      const auto codes = operand(*projection.codes);
      return one(c::Operation::custom(projection.equation),
          {x, codes, operand(*projection.scales), operand(*projection.weight_global), operand(*projection.input_global)},
          dense_spec(Type::kBFloat16, dimensions));
    } else if constexpr (std::is_same_v<P, ModelOptNvfp4Projection>) {
      const auto codes = operand(*projection.codes);
      return one(c::model_ops::nvfp4_linear(),
          {x, codes, operand(*projection.scales), operand(*projection.weight_global), operand(*projection.input_global)},
          dense_spec(Type::kBFloat16, dimensions));
    } else if constexpr (std::is_same_v<P, BlockFp8Projection>) {
      const auto carrier = operand(*projection.carrier);
      return one(c::Operation::custom(projection.equation), {x, carrier}, dense_spec(Type::kBFloat16, dimensions));
    } else if constexpr (std::is_same_v<P, ModelOptStaticFp8Projection>) {
      if (activation_type() != Type::kBFloat16)
        throw std::runtime_error("selected static FP8 source reader requires BF16 activation");
      const auto original = operand(*projection.codes);
      const auto sa = operand(*projection.activation_scale), sw = operand(*projection.weight_scale);
      // This exact 12-node public graph is the existing StaticFloat8Pattern.
      // Cold selection owns contraction or decomposition; no provider switch,
      // decoded bank, second executor or private Compute operation is added.
      const auto a = cast(x, Type::kFloat32);
      const auto one_value = one(c::Fill{
          .output_shape = take(c::TensorShape::create(projection.activation_scale->description.dimensions())),
          .output_type = Type::kFloat32, .value = c::ScalarValue(1.F)}, {}, spec(sa));
      const auto inverse = binary(c::BinaryKind::kDivide, one_value, sa);
      const auto normalized = binary(c::BinaryKind::kMultiply, a, inverse);
      const auto encoded = cast(normalized, Type::kFloat8E4M3FN);
      const auto q = cast(encoded, Type::kFloat32);
      const auto w = cast(original, Type::kFloat32);
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      const auto rhs = one(take(c::View::permute(axes)), {w});
      const auto dot = one(c::MatMul{.accumulation_type = Type::kFloat32},
          {q, rhs}, dense_spec(Type::kFloat32, dimensions));
      const auto scaled = binary(c::BinaryKind::kMultiply,
          binary(c::BinaryKind::kMultiply, dot, sa), sw);
      return cast(scaled, Type::kBFloat16);
    } else if constexpr (std::is_same_v<P, CtBlock128Projection>) {
      namespace math = inference::ct_block128;
      if (activation_type() != Type::kBFloat16)
        throw std::runtime_error("CT block128 source reader requires BF16 activation");
      const auto raw = operand(*projection.codes), sf = operand(*projection.block_scales);
      const auto n = projection.codes->description.dimensions()[0];
      auto &row_map = assembly.literal(dense(Type::kInt32, {n}));
      std::vector<std::int32_t> rows(static_cast<std::size_t>(n));
      for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<std::int32_t>(i / 128);
      row_map.template write<std::int32_t>(rows);
      const auto row_value = constant(row_map);
      const auto literal = [&](std::uint16_t bits) -> math::Operand {
        auto &value = assembly.literal(dense(Type::kBFloat16, {}));
        value.write<std::uint16_t>(std::span(&bits, 1));
        const auto id = constant(value); return {id, spec(id)};
      };
      const math::Constants literals{literal(0x43e0), literal(0x3c00), literal(0)};
      math::Nodes nodes;
      const auto projected = take(math::append_projection(builder, {x, spec(x)},
          {raw, spec(raw)}, {sf, spec(sf)}, {row_value, spec(row_value)}, literals, {}, &nodes));
      for (std::size_t i = 0; i < nodes.count; ++i) assembly.node(nodes.values[i]);
      return remember(projected.value, projected.description);
    } else if constexpr (std::is_same_v<P, ChannelTokenFp8Projection>) {
      const auto original = operand(*projection.codes);
      // The selected channel/token reader uses typed BF16 QDQ. Only an
      // exactly-zero scale gets epsilon; all original rounding stays here.
      const auto maximum = scalar(448.F, Type::kBFloat16);
      const auto raw_scale = binary(c::BinaryKind::kDivide,
          one(c::Reduce{.kind = c::ReductionKind::kMaximum,
              .axes = take(c::Axes::from(std::array<std::int32_t, 1>{1})),
              .keep_dimensions = true, .accumulation_type = Type::kFloat32},
              {unary(c::UnaryKind::kAbs, x)},
              dense_spec(Type::kBFloat16, {spec(x).dimensions()[0], f(1)})), maximum);
      const auto scale = one(c::Select{}, {cast(raw_scale, Type::kBool), raw_scale,
          scalar(1.F / 128.F, Type::kBFloat16)}, spec(raw_scale));
      const auto normalized = binary(c::BinaryKind::kDivide, x, scale);
      const auto clipped = binary(c::BinaryKind::kMinimum,
          binary(c::BinaryKind::kMaximum, normalized, scalar(-448.F, Type::kBFloat16)), maximum);
      const auto activation = binary(c::BinaryKind::kMultiply,
          cast(cast(clipped, Type::kFloat8E4M3FN), Type::kBFloat16), scale);
      const auto decoded = binary(c::BinaryKind::kMultiply, cast(original, Type::kBFloat16), operand(*projection.channel_scales));
      constexpr std::array<std::int32_t, 2> axes{1, 0};
      return one(c::MatMul{.accumulation_type = Type::kFloat32},
          {activation, one(take(c::View::permute(axes)), {decoded})}, dense_spec(Type::kBFloat16, dimensions));
    }
  }, binding);
  return scores ? cast(result, Type::kFloat32) : result;
}

V Recipe::linear_absolute(V x, const std::string &name) {
  return emit_projection(x, projection(name), false);
}

V Recipe::scores(V x, const std::string &name) {
  return emit_projection(x, projection(name), true);
}

V Recipe::norm_absolute(V x, const std::string &name) {
  const auto type = spec(x).data_type();
  auto scale = cast(weight_absolute(name), Type::kFloat32);
  if (meaning(name) != prepared::ParameterMeaning::kRmsEffectiveGamma)
    scale = binary(c::BinaryKind::kAdd, scale, scalar(1));
  const auto converted = cast(x, Type::kFloat32);
  return cast(same(c::RmsNorm{.epsilon = geometry.epsilon}, {converted, scale}, converted), type);
}

V Recipe::norm(V x, const std::string &name) { return norm_absolute(x, "model.language_model." + name); }

V Recipe::decay(const std::string &name) {
  auto value = cast(weight_absolute(name), Type::kFloat32);
  return meaning(name) == prepared::ParameterMeaning::kGdnNegativeExpDecay ? value :
      unary(c::UnaryKind::kNeg, unary(c::UnaryKind::kExp, value));
}

// Literal BF16 boundaries from the independent local reference, including
// BF16 square/sum/epsilon/rsqrt. GDN applies its own 1/sqrt(128) exactly once.
V Recipe::l2(V x) {
  const auto square = binary(c::BinaryKind::kMultiply, x, x);
  const std::array<std::int32_t, 1> axes{-1};
  auto dimensions = std::vector<D>(spec(square).dimensions().begin(), spec(square).dimensions().end());
  dimensions.back() = f(1);
  auto sum = one(c::Reduce{.kind = c::ReductionKind::kSum,
      .axes = take(c::Axes::from(axes)), .keep_dimensions = true,
      .accumulation_type = Type::kFloat32}, {square}, dense_spec(spec(square).data_type(), dimensions));
  sum = binary(c::BinaryKind::kAdd, sum, scalar(1.e-6F, spec(x).data_type()));
  return binary(c::BinaryKind::kMultiply, x, unary(c::UnaryKind::kRsqrt, sum));
}

V Recipe::gdn_key_rows(V rows) {
  if (geometry.gdn_layout == GdnHeadLayout::kCyclic || geometry.gdn_key_heads == geometry.gdn_value_heads)
    return rows;
  // HF value heads are grouped by key head. Converted GGUF has already
  // reordered its V/conv/gates/output weights into the native cyclic order.
  // Preserve either source; only grouped HF needs this explicit boundary.
  std::vector<std::int32_t> expansion(static_cast<std::size_t>(geometry.gdn_value_heads));
  for (std::size_t head = 0; head < expansion.size(); ++head)
    expansion[head] = static_cast<std::int32_t>(head / static_cast<std::size_t>(geometry.gdn_value_heads / geometry.gdn_key_heads));
  auto &literal = assembly.literal(dense(Type::kInt32, {geometry.gdn_value_heads}));
  literal.write<std::int32_t>(expansion);
  const auto indices = constant(literal);
  auto dimensions = std::vector<D>(spec(rows).dimensions().begin(), spec(rows).dimensions().end());
  dimensions.at(2) = f(geometry.gdn_value_heads);
  return one(c::Gather{.axis = 2, .bounds = c::IndexBoundsPolicy::kUnchecked}, {rows, indices},
      dense_spec(spec(rows).data_type(), dimensions));
}

void Recipe::publish(V value) { check(builder.publish(value)); assembly.external(value); }

} // namespace qwen35
