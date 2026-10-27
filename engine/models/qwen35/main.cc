// Qwen3.5 model implementation for the Hyle inference executable. All model
// mathematics uses Compute/Graph; native execution remains backend-owned.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "schedule.h"
#include "session_schedule.h"
#include "../../inference/session_cache.h"
#include "session_state.h"
#include "host_image.h"
#include "../../inference/host_state_storage.h"
#include "../../inference/state_transfer.h"
#include "prefix_cache.h"
#include "batch_schedule.h"
#include "mtp_graph.h"
#include "../../inference/mtp_acceptance.h"
#include "../../inference/sampling_graph.h"
#include "../../inference/ct_block128_graph.h"
#include "../../inference/execution_resources.h"
#include "../../inference/retirement_observation.h"
#include "../../inference/evaluation_failure.h"
#include "../../inference/request_execution_failure.h"
#include "../../inference/prepared_output.h"
#include "../../inference/request_progress.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/request_window.h"
#include "../../inference/request_loop.h"
#include "../../inference/prefill_plan.h"
#include "../../inference/version_admission.h"
#include "../../inference/batch_dispatch.h"
#include "../../inference/execution_observation.h"
#include "../../reporting/execution_report.h"
#include "generation_schedule.h"
#include "generation_tokens.h"
#include "generation_ledger.h"
#include "prepared_roles.h"
#include "ffn_placement.h"
#include "projection_binding.h"
#include "weights.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/core/assert.h"
#include "precision_profile.h"
#include "cache_precision.h"
#include "hyle/simd/dtype.h"
#include "hyle/container/small_vector.h"

#include "../../inference/backend.h"
#include "../../inference/readback.h"
#include "driver.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/slots.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/store_source.h"
#include "hyle/graph/detail/paged_append_source.h"
#include "hyle/mem/buffer.h"
#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/sched/join.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"
#include "hyle/util/scope_guard.h"

namespace qwen35 {
namespace sampling = hyle::inference::sampling;
namespace inference = hyle::inference;
using inference::Backend;
using inference::BackendKind;
using inference::BackendSelection;
using inference::PreparedReadback;
using inference::PreparedOutputFacts;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using V = g::ValueId;
using D = g::Dimension;
constexpr D f(std::int64_t value) noexcept { return D::fixed(value); }
using Type = c::DataType;
using Clock = std::chrono::steady_clock;
enum class ReplayPolicy { kAuto, kCommands, kPreferCapture };
constexpr g::ReplayMode select_replay_mode(ReplayPolicy policy,
    std::size_t known_uses, bool sampling_enabled) noexcept {
  return sampling_enabled || policy == ReplayPolicy::kCommands
      || (policy == ReplayPolicy::kAuto && known_uses <= 1)
      ? g::ReplayMode::kCommands : g::ReplayMode::kPreferCapture;
}
constexpr std::array<const char *, 9> layer0_probe_names{
    "01-gdn-output", "02-out-proj-input", "03-out-proj-output",
    "04-down-proj-input", "05-layer-output", "06-post-attn-residual",
    "07-post-attn-norm", "08-gate-proj-output", "09-up-proj-output"};

struct StateSchema final {
  c::TensorDesc history, recurrent;
};
std::vector<StateSchema> state_schema_for(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision = {}) {
  geometry.validate();
  if (capacity == 0 || capacity > static_cast<std::size_t>(geometry.context_limit)) throw std::runtime_error("capacity outside model context limit");
  std::vector<StateSchema> result(geometry.layers.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto &layer = result[i];
    if (geometry.is_gdn(i)) {
      layer.history = dense(precision.activation_type(), {1, geometry.conv_kernel - 1, geometry.gdn_qkv_width()});
      layer.recurrent = dense(Type::kFloat32, {1, geometry.gdn_value_heads, geometry.gdn_key_depth, geometry.gdn_value_depth});
    }
  }
  return result;
}
std::size_t recurrent_bytes(std::span<const StateSchema> schema, const ModelGeometry &geometry) {
  std::size_t bytes = 0;
  const auto add = [&](const c::TensorDesc &description, std::size_t copies) {
    const auto size = take(description.required_bytes());
    if (size > ((std::numeric_limits<std::size_t>::max)() - bytes) / copies)
      throw std::runtime_error("session cache byte count overflows");
    bytes += size * copies;
  };
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (geometry.is_gdn(i)) { add(schema[i].history, 1); add(schema[i].recurrent, 1); }
  }
  return bytes;
}
struct PackedProjectionNodes final {
  std::array<g::NodeId, 4> chain;
  std::size_t count = 4;
};
void validate_mtp_source(const ModelGeometry &geometry, bool enabled) {
  if (enabled && geometry.mtp_layers != 1)
    throw std::runtime_error("MTP requires one NextN layer in the prepared source geometry");
}
BackendSelection select_execution(BackendSelection selection, std::size_t cpu_layers) {
  if (selection.primary == BackendKind::kDefault)
    selection.primary = take(Backend::default_selection()).primary;
  if (std::ranges::find(Backend::compiled_backends(), selection.primary) == Backend::compiled_backends().end())
    throw std::runtime_error("selected backend is not compiled into this image");
  selection.cpu_assist = cpu_layers != 0;
  if (selection.cpu_assist && (selection.primary == BackendKind::kCpu ||
      std::ranges::find(Backend::compiled_backends(), BackendKind::kCpu) == Backend::compiled_backends().end()))
    throw std::runtime_error("--cpu-ffn-layers requires a GPU primary and the compiled CPU route");
  return selection;
}
Backend create_execution(const ModelGeometry &geometry, const cache::Plan &plan,
    std::size_t cpu_layers, BackendSelection selection,
    const c::cpu::DefaultContextOptions *cpu_options = nullptr) {
  geometry.validate();
  validate_mtp_source(geometry, plan.draft_cache);
  FfnPlacement::validate(cpu_layers, geometry.layers.size());
  selection = select_execution(selection, cpu_layers);
  if (selection.primary != BackendKind::kCpu && !cpu_layers) {
    if (cpu_options) return take(Backend::create(selection, {}, *cpu_options));
    return take(Backend::create(selection));
  }
  // Conservative finite admission for this graph's known workspace only.
  // Sum model-sized activation envelopes instead of assuming workspace reuse;
  // unknown native-private storage remains explicitly outside this bound.
  const auto rows = static_cast<std::size_t>(plan.max_query_rows);
  const auto activation_elements = cache::add(cache::multiply(geometry.layers.size(),
      cache::add(cache::multiply(16, geometry.intermediate), cache::multiply(32, geometry.hidden))), cache::multiply(2, geometry.vocabulary));
  const auto compute = cache::add(64u << 20, cache::multiply(cache::multiply(rows, activation_elements), sizeof(float)));
  if (selection.primary == BackendKind::kCpu) {
    std::cerr << "cpu_workspace_admission host=" << compute
        << " scope=known_per_invocation_workspace native_private_unknown=1 physical_memory_cap=0\n";
    // CPU staging aliases its host allocation domain. Give that domain one
    // finite limit; do not count the same memory as a second staging budget.
    if (cpu_options) return take(Backend::create(selection, {.host = compute}, *cpu_options));
    return take(Backend::create(selection, {.host = compute}));
  }
  const auto staging = cache::add(64u << 20,
      cache::multiply(cache::multiply(rows, cache::multiply(static_cast<std::size_t>(geometry.hidden), 8)), cpu_layers));
  std::cerr << "hybrid_workspace_admission device=" << compute << " host=" << compute
      << " staging=" << staging << " scope=known_per_invocation_workspace native_private_unknown=1 physical_vram_cap=0\n";
  if (cpu_options) return take(Backend::create(selection, {compute, compute, staging}, *cpu_options));
  return take(Backend::create(selection, {compute, compute, staging}));
}
// Cold graph resources and placement, independent of the model equations.
// The original prepared graph owns the moved literals; Backend outlives both.
struct GraphAssembly final {
  struct Location final {
    g::ExecutionTargetId target;
    g::StorageDomainId storage;
  };
  struct Scope final {
    GraphAssembly &assembly;
    Location previous;
    Scope(GraphAssembly &selected, Location location)
        : assembly(selected), previous(selected.location_) { assembly.location_ = location; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { assembly.location_ = previous; }
  };

  std::vector<g::ConstantBinding> constants;
  std::vector<Tensor> literals;
  std::vector<g::NodePlacement> node_placements;
  std::vector<g::ExternalResourceContract> external_resources;

  explicit GraphAssembly(const Backend &environment, PrecisionProfile precision = {},
                         const c::ResourceBudget &budget = {})
      : backend_(environment), budget_(budget), precision_(precision), mixed_(environment.has_cpu_assist()),
        predicate_type_(environment.native_generated_storage() ? Type::kFloat32 : Type::kBool) {}
  GraphAssembly(const GraphAssembly &) = delete;
  GraphAssembly &operator=(const GraphAssembly &) = delete;
  static Location primary() { return {g::ExecutionTargetId{0}, g::StorageDomainId{0}}; }
  Location cpu() const {
    if (!mixed_) throw std::runtime_error("CPU FFN requires the configured CPU endpoint");
    return {g::ExecutionTargetId{1}, g::StorageDomainId{1}};
  }
  Tensor &literal_at(Location location, c::TensorDesc description) {
    literals.emplace_back(take(backend_.resource_for(location.storage)), description, true, budget_);
    return literals.back();
  }
  Tensor &literal(c::TensorDesc description) { return literal_at(location_, description); }
  Type predicate_type() const { return predicate_type_; }
  Type activation_type() const noexcept { return precision_.activation_type(); }
  void node(g::NodeId node) {
    if (mixed_) node_placements.push_back({node, location_.target, g::PlacementStrength::kRequired});
  }
  void external(V value, g::StorageDomainId storage = g::StorageDomainId{0}) {
    if (!mixed_) return;
    const auto found = std::ranges::find(external_resources, value, &g::ExternalResourceContract::value);
    if (found == external_resources.end()) external_resources.push_back({value, storage});
    else if (found->storage != storage) throw std::runtime_error("conflicting external tensor resource contracts");
  }
  g::CompileOptions options(g::CompileOptions options) const {
    options.node_placements = node_placements; options.external_resources = external_resources; return options;
  }
  void bind_constant(V value, Tensor &tensor) {
    tensor.freeze(backend_);
    const auto read = tensor.read();
    const auto owner = take(backend_.storage_owner_for(read.storage().resource()));
    if (mixed_) external(value, g::StorageDomainId{read.storage().resource() == backend_.device() ? 0u : 1u});
    // Native resources are explicitly borrowed from Model's Backend. The
    // bank/literal owns its lease through every compiled child. A native lease
    // requires its actual StorageOwner; only process-system storage can be
    // retained without one. Graph validates that allocator identity separately.
    const auto retained = owner || read.storage().resource() == mem::system_resource()
        ? tensor.ready : c::ReadLease{};
    constants.push_back({value, read, g::ConstantPolicy::kBorrow, owner, retained});
  }

private:
  const Backend &backend_;
  const c::ResourceBudget budget_;
  const PrecisionProfile precision_;
  const bool mixed_;
  const Type predicate_type_;
  Location location_ = primary();
};

struct Recipe final {
  g::GraphBuilder builder = take(g::GraphBuilder::create());
  std::vector<g::TensorSpec> specs;
  std::map<Tensor *, V> operands;
  std::vector<PackedProjectionNodes> packed_projection_nodes;
  std::map<std::string, LoadedWeight> &bank;
  const ProjectionBindings &projections;
  const WeightMeanings *meanings = nullptr;
  GraphAssembly &assembly;
  const ModelGeometry &geometry;
  const std::int64_t hidden;
  g::DimensionId last_row;
  D token_rows, packed_rows, sample_rows;
  Recipe(std::map<std::string, LoadedWeight> &selected, const ProjectionBindings &selected_projections,
         GraphAssembly &graph_assembly, const ModelGeometry &model_geometry,
         std::size_t max_query_rows, std::size_t requests = 1,
         const WeightMeanings *parameter_meanings = nullptr)
      : bank(selected), projections(selected_projections), meanings(parameter_meanings), assembly(graph_assembly), geometry(model_geometry), hidden(model_geometry.hidden) {
    // L is a zero-based last-token position, independently bound at startup.
    // T=L+1 avoids subtraction or a second independent (possibly inconsistent)
    // shape binding. All other numeric axes remain fixed model dimensions.
    last_row = take(builder.dimension({0, static_cast<std::int64_t>(max_query_rows - 1), 1}));
    token_rows = D(take(builder.dimension_sum(D(last_row), f(1))));
    packed_rows = requests == 1 ? token_rows :
        D(take(builder.dimension_product(f(static_cast<std::int64_t>(requests)), token_rows)));
    sample_rows = D(take(builder.dimension_sum(token_rows, f(geometry.conv_kernel - 1))));
  }
  static g::TensorSpec dense_spec(Type type, std::span<const D> dimensions) {
    return take(g::TensorSpec::dense(type, dimensions));
  }
  static g::TensorSpec dense_spec(Type type, std::initializer_list<D> dimensions) {
    return dense_spec(type, std::span(dimensions.begin(), dimensions.size()));
  }
  Type activation_type() const noexcept { return assembly.activation_type(); }
  const g::TensorSpec &spec(V value) const { return specs.at(value.ordinal()); }
  V remember(V value, g::TensorSpec description) {
    if (specs.size() <= value.ordinal()) specs.resize(value.ordinal() + 1);
    specs[value.ordinal()] = description;
    return value;
  }
  V input(g::TensorSpec description) {
    const auto value = remember(take(builder.input(description)), description); assembly.external(value); return value;
  }
  V input(c::TensorDesc description) { return input(take(g::TensorSpec::concrete(description))); }
  V constant(Tensor &tensor) {
    const auto value = remember(take(builder.constant(tensor.description)),
                                take(g::TensorSpec::concrete(tensor.description)));
    assembly.bind_constant(value, tensor);
    return value;
  }
  V operand(Tensor &tensor) {
    const auto found = operands.find(&tensor);
    if (found != operands.end()) return found->second;
    const auto value = constant(tensor);
    operands.emplace(&tensor, value);
    return value;
  }
  V weight_absolute(const std::string &name) {
    const auto original = bank.find(name);
    if (original == bank.end()) throw std::runtime_error("missing original model role: " + name);
    return operand(original->second.tensor);
  }
  V weight(const std::string &name) {
    return weight_absolute("model.language_model." + name);
  }
  prepared::ParameterMeaning meaning(const std::string &name) const {
    if (meanings) if (const auto found = meanings->find(name); found != meanings->end()) return found->second;
    return prepared::ParameterMeaning::kHfRaw;
  }
  V embedding(V tokens, const std::string &name) {
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
  V scalar(float value, Type type = Type::kFloat32) {
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
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    std::initializer_list<g::TensorSpec> outputs,
                    g::NodeId *node_id = nullptr) {
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
  // Only concrete constant transforms use the Compute-only inference shortcut.
  // Symbolic operations declare their real logical outputs; each exact child
  // still runs the authoritative Compute inference during Graph compilation.
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    g::NodeId *node_id = nullptr) {
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
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, node_id).at(0);
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::TensorSpec output, g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, {output}, node_id).at(0);
  }
  template<class O> V same(O operation, std::initializer_list<V> arguments, V source) {
    return one(operation, arguments, dense_spec(spec(source).data_type(), spec(source).dimensions()));
  }
  V cast(V x, Type type) {
    return spec(x).data_type() == type ? x :
        one(c::Cast{type}, {x}, dense_spec(type, spec(x).dimensions()));
  }
  V host_predicate(V value) {
    return cast(value, assembly.predicate_type());
  }
  V unary(c::UnaryKind kind, V x) { return same(c::Unary{kind}, {x}, x); }
  V binary(c::BinaryKind kind, V x, V y) {
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
  V reshape(V x, std::initializer_list<D> dimensions) {
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
  V copy(V x) { return same(c::Copy{}, {x}, x); }
  V concat(std::span<const V> inputs, int axis, g::TensorSpec output) {
    if (inputs.size() == 1) return inputs.front();
    const std::array<g::TensorSpec, 1> outputs{output};
    const auto node = take(builder.add_node(c::Operation(c::Concat{.axis = axis}), inputs, outputs));
    assembly.node(node);
    return remember(take(builder.result(node)), std::move(output));
  }
  V slice(V x, int axis, D first, D count) {
    const auto source = spec(x);
    std::vector<D> dimensions(source.dimensions().begin(), source.dimensions().end());
    dimensions.at(static_cast<std::size_t>(axis)) = count;
    const auto node = take(builder.add_node(take(g::ViewSpec::slice(axis, first, count)), x)); assembly.node(node);
    const auto output = take(builder.result(node));
    return remember(output, take(g::TensorSpec::inferred(source.data_type(), dimensions,
                                                       source.storage_format())));
  }
  V linear(V x, const std::string &name) {
    return linear_absolute(x, "model.language_model." + name);
  }
  const ProjectionBinding &projection(const std::string &name) const {
    const auto found = projections.find(name);
    if (found == projections.end()) throw std::runtime_error("missing cold model projection binding: " + name);
    return found->second;
  }
  V emit_projection(V x, const ProjectionBinding &binding, bool scores) {
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
  V linear_absolute(V x, const std::string &name) {
    return emit_projection(x, projection(name), false);
  }
  V scores(V x, const std::string &name) {
    return emit_projection(x, projection(name), true);
  }
  V norm_absolute(V x, const std::string &name) {
    const auto type = spec(x).data_type();
    auto scale = cast(weight_absolute(name), Type::kFloat32);
    if (meaning(name) != prepared::ParameterMeaning::kRmsEffectiveGamma)
      scale = binary(c::BinaryKind::kAdd, scale, scalar(1));
    const auto converted = cast(x, Type::kFloat32);
    return cast(same(c::RmsNorm{.epsilon = geometry.epsilon}, {converted, scale}, converted), type);
  }
  V norm(V x, const std::string &name) { return norm_absolute(x, "model.language_model." + name); }
  V decay(const std::string &name) {
    auto value = cast(weight_absolute(name), Type::kFloat32);
    return meaning(name) == prepared::ParameterMeaning::kGdnNegativeExpDecay ? value :
        unary(c::UnaryKind::kNeg, unary(c::UnaryKind::kExp, value));
  }
  // Literal BF16 boundaries from the independent local reference, including
  // BF16 square/sum/epsilon/rsqrt. GDN applies its own 1/sqrt(128) exactly once.
  V l2(V x) {
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
  V ffn(V x, const std::string &prefix, std::span<V> probes = {}) {
    auto normalized = norm(x, prefix + "post_attention_layernorm.weight");
    auto projected_gate = linear(normalized, prefix + "mlp.gate_proj.weight");
    auto gate = unary(c::UnaryKind::kSilu, projected_gate);
    auto up = linear(normalized, prefix + "mlp.up_proj.weight");
    auto product = binary(c::BinaryKind::kMultiply, gate, up);
    if (!probes.empty()) {
      probes[3] = product; probes[5] = x; probes[6] = normalized;
      probes[7] = projected_gate; probes[8] = up;
    }
    auto result = linear(product, prefix + "mlp.down_proj.weight");
    return binary(c::BinaryKind::kAdd, x, result);
  }
  V gdn_key_rows(V rows) {
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
  void publish(V value) { check(builder.publish(value)); assembly.external(value); }
};

// Distinguish direct packed contractions from the ordinary Decode/Cast route.
// The latter has per-projection Invocation workspace and may use managed
// provider temporaries; it does not replace the original constant weight bank.
void report_packed_projections(const g::Graph &graph, const g::CompileReport &report,
                               std::span<const PackedProjectionNodes> projections, std::size_t graph_workspace_bytes) {
  std::size_t packed = 0;
  for (const auto &projection : projections) {
    if (projection.count != 3 && projection.count != 4)
      throw std::runtime_error("packed projection requires its actual three/four-node recipe");
    std::size_t containing = 0;
    for (std::size_t step_index = 0; step_index < report.step_count(); ++step_index) {
      const auto step = take(report.step(step_index));
      std::array<bool, 4> included{};
      for (std::size_t i = 0; i < step.node_count; ++i) {
        const auto node = take(report.step_node(step_index, i));
        for (std::size_t j = 0; j < projection.count; ++j) included[j] = included[j] || node == projection.chain[j];
      }
      if (!included[projection.count - 1]) continue;
      ++containing;
      const bool contracted = step.policy == g::kArtifactPackedProjectionPolicy && step.node_count == projection.count &&
          std::ranges::all_of(std::span(included).first(projection.count), [](bool value) { return value; });
      if (contracted) {
        for (std::size_t j = 0; j + 1 < projection.count; ++j) {
          const auto intermediate = take(graph.result(projection.chain[j]));
          for (std::size_t i = 0; i < report.physical_value_count(); ++i)
            if (take(report.physical_value(i)).value == intermediate)
              throw std::runtime_error("packed projection materialized a floating weight intermediate");
        }
        ++packed;
      }
    }
    if (containing != 1) throw std::runtime_error("original packed projection is missing from its selected plan");
  }
  if (!projections.empty())
    std::cerr << "original_quantized_projections=" << projections.size()
              << " packed_contractions=" << packed
              << " decomposed_projections=" << projections.size() - packed
              << " graph_workspace_bytes=" << graph_workspace_bytes << '\n';
}

struct SamplingConfiguration final {
  bool enabled = false;
  sampling::Policy policy;
  sampling::LogitsPolicy modifiers;
};
// Typed ready inputs for one ordinary output episode. Copies only retain the
// original read permission. Use a lease carrying its existing StorageOwner,
// or keep its borrowed allocator/resource/Context alive through every retained
// copy and native use. Owners may supply different leases next episode; no
// modifier payload enters a Snapshot, commit ledger or KV cache.
struct LogitsInput final {
  c::TensorDesc description;
  c::ReadLease lease;
};
struct LogitsInputs final { LogitsInput bias, allowed; };
struct PreparedSampling;
struct ColdResources;
void validate_sampling_configuration(const SamplingConfiguration &value, std::int64_t vocabulary) {
  if (vocabulary <= 0 || vocabulary > INT32_MAX || !std::isfinite(value.policy.temperature) || value.policy.temperature <= 0 ||
      !std::isfinite(value.policy.top_p) || value.policy.top_p <= 0 || value.policy.top_p > 1 ||
      value.policy.top_k > static_cast<std::size_t>(vocabulary) ||
      (!value.enabled && (value.policy.temperature != 1 || value.policy.top_k || value.policy.top_p != 1)))
    throw std::runtime_error("invalid finite sampling configuration for vocabulary");
}
bool same_sampling_configuration(const SamplingConfiguration &a, const SamplingConfiguration &b) noexcept {
  return a.enabled == b.enabled && a.policy.temperature == b.policy.temperature &&
      a.policy.top_k == b.policy.top_k && a.policy.top_p == b.policy.top_p && a.modifiers == b.modifiers;
}

void validate_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
                            std::int64_t vocabulary) {
  const auto require = [&](const LogitsInput &input, bool enabled, Type type) {
    if (input.lease.valid() != enabled || input.description.valid() != enabled)
      throw std::runtime_error("logits input does not match its cold port configuration");
    if (enabled) {
      if (input.description != dense(type, {1, vocabulary}))
        throw std::runtime_error("logits input requires the exact cold dense [1,V] descriptor");
      (void)take(input.lease.tensor(input.description));
    }
  };
  require(inputs.bias, configuration.modifiers.bias, Type::kFloat32);
  require(inputs.allowed, configuration.modifiers.allowed, Type::kBool);
}
sampling::LogitsInputsFor<V> prepare_logits_inputs(Recipe &r, sampling::LogitsPolicy policy) {
  sampling::LogitsInputsFor<V> result;
  if (policy.bias) result.bias = r.input(dense(Type::kFloat32, {1, r.geometry.vocabulary}));
  if (policy.allowed) result.allowed = r.input(dense(Type::kBool, {1, r.geometry.vocabulary}));
  return result;
}

GenerationBoundary generation_boundary(const cache::State &head) noexcept {
  return {head.length, head.draft ? head.draft->frontier : 0,
      head.draft ? head.draft->lag : 0, head.next_id,
      head.draft ? head.draft->catchup_token : -1, head.sampling};
}
struct FamilyKey final {
  std::size_t requests, verification_rows;
  GraphRole role;
  TextCompletion completion;
  std::size_t sampling_index;
  FamilyKey(std::size_t count, GraphRole selected = GraphRole::kText,
            std::size_t rows = 0, TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0) noexcept
      : requests(count), verification_rows(selected == GraphRole::kVerify ? rows : 0), role(selected), completion(mode), sampling_index(sampling) {}
  static FamilyKey exact(std::size_t count, GraphRole selected, std::size_t rows,
                         TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0) noexcept {
    FamilyKey key(count, selected, 0, mode, sampling); key.verification_rows = rows; return key;
  }
  auto operator<=>(const FamilyKey &) const = default;
};
struct ExecutionShape final {
  std::size_t requests, tokens;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t sampling_index = 0;
  bool imported_sources = false;
  auto operator<=>(const ExecutionShape &) const = default;
};
struct StatePorts final {
  V pool, source_ids, destination_ids;
  g::NodeId gather, store;
};
struct Ports final {
  V key, value;
  StatePorts history, recurrent;
};
struct FamilyPorts {
  std::size_t prefix_rows = 1;
  V token_port, position_port, last_ids_port, metadata_port, logits_port, id_port, hidden_port;
  V sampling_identity_port, sampling_valid_port, sampling_accept_port;
  bool batched_text_sampling = false; // Cold [R,4] identity-port contract.
  std::vector<V> logits_bias_ports, logits_allowed_ports; // Cold per-member inputs only.
  std::vector<g::NodeId> attention_nodes, append_nodes;
  std::vector<g::slots::SourceBinding> state_sources;
  V draft_key_port, draft_value_port, draft_metadata_port, draft_position_port;
  V carry_input_port, prior_input_port, catchup_token_port, draft_id_port;
  std::vector<std::array<V, 2>> draft_hidden_inputs;
  g::NodeId draft_attention_node;
  std::vector<V> state_resource_ports;
  std::vector<std::vector<V>> carry_ports;
  std::array<V, 9> probe_ports;
  std::vector<Ports> layers;
};
struct FamilyChild final {
  g::CompiledGraph compiled; // One immutable selected child shared by every session.
  // Only needed when the original symbolic graph cannot describe a state row.
  // This is cold metadata with the same IDs, never another executable.
  g::Graph state_graph;
  std::vector<PreparedOutputFacts> outputs;
  std::vector<std::uint32_t> output_indices;
  const PreparedOutputFacts &output(V value) const {
    if (!value || value.ordinal() >= output_indices.size())
      throw std::runtime_error("value is outside the prepared child output index");
    const auto index = output_indices[value.ordinal()];
    if (index >= outputs.size() || outputs[index].value != value)
      throw std::runtime_error("value is not a prepared output of this exact child");
    return outputs[index];
  }
};
struct Family final : FamilyPorts {
  const PreparedSampling *sampling = nullptr;
  std::size_t requests = 1;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  bool admitted_exact = false;
  std::vector<Tensor> literals; // Backing survives every child and its slots.
  std::vector<g::TensorSpec> specs;
  c::BatchSchema attention_schema;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::DimensionId last_row;
  std::vector<std::size_t> chunks;
  std::vector<std::unique_ptr<FamilyChild>> children;
};
// Physical IO/control carriers are independent of candidate depth. A reader
// may still retain the preceding carrier after its producer publishes.
constexpr std::size_t execution_slot_count = 3;
using ResourceRetention = inference::RetainedResources<ColdResources>;
struct HiddenOutputCopy final {
  std::vector<Tensor> scratch;
  std::vector<c::ReadLease> readers;
  std::vector<c::CopyRange> ranges;
  c::CopyBatchProgram program;
  c::CopyBatchInvocation invocation;
  c::CopyBatchUpdate update;
  ResourceRetention retirement;

  // Copy controls borrow the original Queue. Their accepted Completion, not
  // a lease or observed error, retains ColdResources until actual retirement.
  hyle::Result<bool> recover(hyle::Status &terminal) noexcept {
    bool failed = false;
    if (update.valid()) {
      const auto observed = update.observe();
      inference::record_retirement_status(terminal, observed);
      if (update.in_flight()) return false;
      failed = !observed.ok();
      HYLE_ASSIGN_OR_RETURN(invocation, update.release_invocation());
    }
    if (invocation.valid()) {
      if (invocation.in_flight()) return false;
      HYLE_RETURN_IF_ERROR(invocation.reset());
      // reset cannot reopen a closed failed slot. Recreate only after the
      // original controls and every accepted native use actually retire.
      if (failed) invocation = {};
    }
    return !retirement.pending();
  }
  // Call only after original Graph unbind/reset AND copy binding withdrawal.
  // A native/escaped alias rejects promotion instead of exposing a writer.
  hyle::Status restore_writers() noexcept {
    for (std::size_t i = 0; i < scratch.size(); ++i) if (!scratch[i].writer) {
      if (readers[i]) {
        HYLE_ASSIGN_OR_RETURN(scratch[i].writer, readers[i].try_write());
      } else {
        // Abort before result export: Graph reset returned the original range.
        HYLE_ASSIGN_OR_RETURN(const auto bytes, scratch[i].description.required_bytes());
        HYLE_ASSIGN_OR_RETURN(scratch[i].writer, scratch[i].pool.allocate(bytes, cache::state_pool_alignment));
      }
    }
    return hyle::Status::Ok();
  }
};
struct StateBinding final {
  std::size_t layer = 0;
  bool recurrent = false;
  g::slots::SlotBindings transaction;
  std::vector<c::ReadLease> sources;
  std::vector<c::WriteLease> destinations;
};
// Mutable input, result and layout carriers belong to one actual Invocation.
// Sharing a finite child/Queue never grants another slot access to these bytes.
struct PhaseIo final {
  c::BatchLayoutVersion attention_layout;
  Tensor tokens, positions, last_ids, logits, final_hidden;
  Tensor sampling_identities, draft_positions, catchup_token;
  PreparedReadback next_id, draft_id, sampling_valid, sampling_accept;
  std::vector<std::uint8_t> sampling_valid_values, sampling_accept_values;
  std::vector<float> sampling_scratch;
  std::vector<std::int32_t> draft_ids, prefix_candidates;
  std::array<Tensor, 9> probe_outputs;
};
struct Phase final : FamilyPorts {
  inference::QueueBindings queues;
  std::size_t query_rows = 0;
  c::BatchSchema attention_schema;
  // The selected child is shared immutable model data, not copied per session.
  const FamilyChild *child = nullptr;
  const Family *family = nullptr;
  std::size_t variant_index = 0;
  std::size_t slot_count = 1;
  bool imported_sources = false;
  std::array<g::Invocation, execution_slot_count> slots;
  std::array<std::vector<StateBinding>, execution_slot_count> state_bindings;
  std::array<std::vector<g::slots::BindRequest>, execution_slot_count> state_requests;
  const g::Graph &state_graph() const noexcept {
    return child->state_graph.valid() ? child->state_graph : family->graph;
  }
  std::array<ResourceRetention, execution_slot_count> retirement;
  std::array<HiddenOutputCopy, execution_slot_count> hidden_output;
  std::array<std::size_t, execution_slot_count> planned_slot_uses{};
  std::array<std::size_t, execution_slot_count> preparation_uses{};
  std::array<g::ReplayMode, execution_slot_count> replay_modes{};
  std::array<PhaseIo, execution_slot_count> io;
  const PreparedOutputFacts *logits_facts = nullptr, *hidden_facts = nullptr;
  std::array<const PreparedOutputFacts *, 9> probe_facts{};
};

// One original Graph episode owns all dirty pages and candidate state. Until
// finish replaces the Session head, neither a ready layer nor an error changes
// the committed snapshot. Failure recovery remains attached to that Session.
// One host-only upload workspace for the original serial Model carrier.
// Its integer ranges never escape synchronous Tensor::write; retained model
// state, page permissions and submitted resources keep their existing owners.
struct InputPreparationScratch final {
  std::vector<std::int32_t> packed_ids, positions, draft_positions, catchups;
  std::vector<std::uint32_t> sampling_identities;
  // BatchSchema::version synchronously copies this integer source range.
  // Keep page readers, immutable layout versions and NodeLayouts outside scratch.
  std::vector<std::int32_t> target_page_ids;
  void prepare(const Phase &phase) {
    const auto &io = phase.io.front(); // All slots share the same cold descriptions.
    const auto reserve = [](auto &values, const Tensor &tensor) {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      values.reserve(take(tensor.description.required_bytes()) / sizeof(Element));
    };
    target_page_ids.reserve(phase.attention_schema.capacity().max_page_refs);
    if (phase.family->requests > 1) reserve(packed_ids, io.tokens);
    reserve(positions, io.positions);
    if (io.sampling_identities.caller_owned()) reserve(sampling_identities, io.sampling_identities);
    if (io.draft_positions.caller_owned()) reserve(draft_positions, io.draft_positions);
    if (io.catchup_token.caller_owned()) reserve(catchups, io.catchup_token);
  }
};
class Session;
struct StepMember final {
  std::shared_ptr<Session> session;
  cache::WorkingState *owner_state = nullptr;
  const cache::State *base = nullptr;
  std::uint64_t base_epoch = 0;
  // Synchronous episode borrow, like its original request input. No native
  // callback reads this policy; the original owner keeps it through finish.
  const inference::RequestStopPolicy *stop_policy = nullptr;
  std::unique_ptr<cache::State> next{};
  hyle::container::SmallVector<c::PagedArenaCommit, 16> key_updates{}, value_updates{};
  std::array<c::PagedArenaCommit, 2> draft_updates{};
  hyle::container::SmallVector<c::PagedPageAppender, 32> next_tails{};
  std::array<c::PagedPageAppender, 2> next_draft_tails{};
  c::BatchLayoutVersion layout{};
  c::BatchLayoutVersion draft_layout{};
  std::size_t accepted_tokens = 0;
  std::uint32_t verified_last_page = 0;
  std::int32_t first_input = -1, draft_token = -1;
  std::vector<std::int32_t> verification_inputs{};
  std::vector<c::ReadLease> carry_prefix{};
  std::size_t discarded_draft_rows = 0;
  bool sampled_accept = false;
  GenerationCommitRecord *commit_record = nullptr;
  GenerationLedger *commit_ledger = nullptr;

  const c::PagedArenaFrame &frame(std::size_t layer, bool key) const noexcept {
    const auto &candidate = key ? next->keys[layer] : next->values[layer];
    return candidate.valid() ? candidate : key ? base->keys[layer] : base->values[layer];
  }
  c::PagedArenaFootprint source(std::size_t layer, bool key) {
    auto &candidate = key ? next->keys[layer] : next->values[layer];
    const auto &original = key ? base->keys[layer] : base->values[layer];
    if (!candidate.valid() && original.sealed())
      candidate = take(c::PagedArenaFrame::fork(take(original.view())));
    return take(frame(layer, key).view());
  }
  const c::PagedArenaFrame &draft_frame(bool key) const noexcept {
    const auto &candidate = key ? next->draft->key : next->draft->value;
    return candidate.valid() ? candidate : key ? base->draft->key : base->draft->value;
  }
  c::PagedArenaFootprint draft_source(bool key) {
    auto &candidate = key ? next->draft->key : next->draft->value;
    const auto &original = key ? base->draft->key : base->draft->value;
    if (!candidate.valid() && original.sealed())
      candidate = take(c::PagedArenaFrame::fork(take(original.view())));
    return take(draft_frame(key).view());
  }
  void preflight_frames() const {
    for (std::size_t i = 0; i < key_updates.size(); ++i) {
      check(frame(i, true).validate_commit(key_updates[i]));
      check(frame(i, false).validate_commit(value_updates[i]));
    }
    for (std::size_t i = 0; i < draft_updates.size(); ++i) if (draft_updates[i].valid())
      check(draft_frame(i == 0).validate_commit(draft_updates[i]));
  }
  void publish_frames(cache::State *previous, cache::State &candidate) noexcept {
    const auto publish = [](c::PagedArenaFrame &destination, c::PagedArenaFrame *old,
        c::PagedArenaCommit &ticket) noexcept {
      if (!destination.valid()) {
        HYLE_FATAL_ASSERT(old && old->valid() && !old->sealed());
        destination = std::move(*old);
      }
      const auto status = destination.commit(std::move(ticket));
      HYLE_FATAL_ASSERT(status.ok());
    };
    for (std::size_t i = 0; i < key_updates.size(); ++i) {
      publish(candidate.keys[i], previous ? &previous->keys[i] : nullptr, key_updates[i]);
      publish(candidate.values[i], previous ? &previous->values[i] : nullptr, value_updates[i]);
    }
    for (std::size_t i = 0; i < draft_updates.size(); ++i) if (draft_updates[i].valid()) {
      auto &destination = i == 0 ? candidate.draft->key : candidate.draft->value;
      auto *old = previous ? (i == 0 ? &previous->draft->key : &previous->draft->value) : nullptr;
      publish(destination, old, draft_updates[i]);
    }
  }
};
// Typed views of one original Graph submission. They contain no executor,
// model-state snapshot or observer independent of the original proxy. A
// successor may keep these views after its producer's host episode retires.
struct StepSources final {
  struct Boundary final {
    std::weak_ptr<Session> session;
    const cache::State *candidate = nullptr;
    std::uint64_t epoch = 0;
  };
  g::detail::SubmissionSource submission;
  g::detail::OutputSource token;
  std::vector<g::detail::StoreSource> states;
  std::vector<g::batch::detail::PagedAppendSource> pages;
  std::vector<Boundary> boundaries;
  std::vector<c::ReadLease> published_states;
};
struct IncomingStepSources final {
  std::shared_ptr<StepSources> owner;
  std::vector<g::detail::OutputBinding> bindings;
  std::vector<V> outside_inputs;
  bool bound = false;
};
struct StepState final {
  // Model owns the episode; these strong pins outlive all accepted native work.
  // Session's pending link borrows this owner and never creates a cycle.
  hyle::container::SmallVector<StepMember, 1> members;
  hyle::container::SmallVector<std::vector<c::PagedPageWriter>, 34> dirty;
  // One inherited partial page per request/bank. The common R1 case stays inline.
  hyle::container::SmallVector<hyle::container::SmallVector<c::PagedPageAppender, 1>, 34> tails;
  hyle::container::SmallVector<g::batch::PagedAppendPages, 34> appends;
  hyle::container::SmallVector<g::NodeId, 34> nodes;
  hyle::container::SmallVector<std::uint8_t, 34> bound;
  void configure(std::size_t banks) {
    if (!dirty.try_resize(banks) || !tails.try_resize(banks) || !appends.try_resize(banks) ||
        !nodes.try_resize(banks) || !bound.try_resize(banks)) throw std::bad_alloc{};
  }
  c::BatchLayoutVersion draft_layout;
  g::Invocation *run = nullptr;
  Phase *phase = nullptr;
  c::Completion completion;
  std::shared_ptr<StepSources> sources;
  std::unique_ptr<IncomingStepSources> incoming;
  std::shared_ptr<StepState> predecessor_episode;
  bool has_completion() const noexcept {
    return completion.valid() || (sources && sources->submission.valid());
  }
  hyle::Result<c::CompletionPoll> poll() noexcept {
    return sources && sources->submission.valid() ? sources->submission.poll() : completion.poll();
  }
  hyle::Status wait() noexcept {
    return sources && sources->submission.valid() ? sources->submission.wait() : completion.wait();
  }
  hyle::Status request_cancel() noexcept {
    return sources && sources->submission.valid() ? sources->submission.request_cancel() : completion.request_cancel();
  }
  hyle::Status terminal;
  Clock::time_point step_start{}, start{};
  double submit_seconds = 0, cold_batch_seconds = 0, shape_admission_seconds = 0;
  std::size_t input_tokens = 0, capture_recorded = 0, capture_reused = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool prepared = false, submitted = false, cancelled = false;
  std::size_t state_slot = 0;
  bool states_bound = false;
  bool cleanup_pending = false;
  bool discard_stopped_members = false;
  // Model admission is serial even when a cancelled native episode has not
  // retired. Return it only after publication and final result reads, or after
  // complete abort cleanup.
  std::size_t *candidate_admission = nullptr;
  HiddenOutputCopy *hidden_output = nullptr;
  void release_admission() noexcept {
    if (candidate_admission) {
      auto *count = std::exchange(candidate_admission, nullptr);
      HYLE_FATAL_ASSERT(*count); --*count;
    }
  }

  hyle::Result<bool> try_abort() noexcept {
    // Even an unsubmitted carrier may own a capture/workspace whose native
    // destruction synchronizes its context. Keep the rejected candidate
    // intact until its actual predecessor retires; let the owner observe the
    // original preparation error without waiting behind that predecessor.
    if (cleanup_pending && predecessor_episode && predecessor_episode->run &&
        predecessor_episode->run->in_flight()) return false;
    if (has_completion()) {
      auto observed = poll();
      if (observed && *observed == c::CompletionPoll::kPending) return false;
      if (!observed) inference::record_retirement_status(terminal, observed.error());
      completion = {}; // Error/abandonment keeps actual native guards retained.
    }
    if (sources) {
      if (sources->submission.valid() && sources.use_count() != 1) return false;
      // Failed token views cannot publish/detach. Drop this host reference;
      // an actual successor still owns its own view and prevents premature
      // producer withdrawal until the successor's original reads retire.
      sources->token = {};
      if (sources->submission.valid()) {
        const auto detached = sources->submission.detach();
        if (detached.IsUnavailable() || detached.IsFailedPrecondition()) return false;
        HYLE_RETURN_IF_ERROR(detached);
      }
      sources.reset(); // Unescaped tickets must end before writer discard.
    }
    if (hidden_output) {
      HYLE_ASSIGN_OR_RETURN(auto retired, hidden_output->recover(terminal));
      if (!retired) return false;
    }
    if (run) {
      if (run->in_flight()) return false;
      for (std::size_t i = 0; i < appends.size(); ++i) if (bound[i]) {
        auto recovered = g::batch::take_append(*run, nodes[i]);
        if (!recovered) {
          if (recovered.error().IsUnavailable()) return false;
          return std::unexpected(recovered.error());
        }
        appends[i] = std::move(*recovered); bound[i] = false;
      }
      // One batch bind transfers every state transaction together. A partial
      // take on recovery is remembered by the valid caller-owned metadata.
      if (phase) for (std::size_t i = 0; i < phase->state_bindings[state_slot].size(); ++i) {
        auto &bank = phase->state_bindings[state_slot][i];
        if (states_bound && !bank.transaction.valid()) {
          auto recovered = g::slots::take(*run, phase->state_sources[i].store);
          if (!recovered) {
            if (recovered.error().IsUnavailable()) return false;
            return std::unexpected(recovered.error());
          }
          bank.transaction = std::move(*recovered);
        }
        bank.transaction.clear();
        for (auto &source : bank.sources) source = {};
        for (auto &writer : bank.destinations) writer = {};
      }
      states_bound = false;
      if (incoming && incoming->bound) {
        HYLE_RETURN_IF_ERROR(g::detail::release_outputs(*run, incoming->bindings));
      }
      incoming.reset(); predecessor_episode.reset();
      *run = {}; // Release failed ordinary outputs and their original bindings.
    }
    if (hidden_output) {
      const auto restored = hidden_output->restore_writers();
      if (restored.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(restored);
      hidden_output = nullptr;
    }
    const auto recover = [](c::PagedPageWriter &page) noexcept -> hyle::Status {
      if (!page.valid()) return hyle::Status::Ok(); // Already successfully published.
      page.lease().reset();
      HYLE_RETURN_IF_ERROR(page.discard()); // Unavailable retains reservation.
      page = {}; // Return the recovered pristine writer; retries skip this page.
      return hyle::Status::Ok();
    };
    for (auto &append : appends) if (append.valid()) {
      for (std::size_t i = 0; i < append.appender_count(); ++i) {
        HYLE_ASSIGN_OR_RETURN(auto &tail, append.appender(i));
        if (!tail.valid() || !tail.reserved_rows()) continue;
        auto rolled_back = tail.rollback();
        if (rolled_back.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(rolled_back);
      }
      for (std::size_t page = 0; page < append.page_count(); ++page) {
        HYLE_ASSIGN_OR_RETURN(auto &writer, append.page(page));
        auto recovered = recover(writer);
        if (recovered.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(recovered);
      }
    }
    for (auto &bank : tails) for (auto &tail : bank) {
      if (!tail.valid() || !tail.reserved_rows()) continue;
      auto rolled_back = tail.rollback();
      if (rolled_back.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(rolled_back);
    }
    for (auto &bank : dirty) for (auto &writer : bank) {
      auto recovered = recover(writer);
      if (recovered.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(recovered);
    }
    appends = {};
    for (auto &bank : dirty) bank.clear();
    tails = {};
    for (auto &member : members) {
      member.owner_state->recycle_candidate(std::move(member.next)); member.base = nullptr;
    }
    run = nullptr; phase = nullptr;
    release_admission();
    return true;
  }
};

class Model;
// Admission borrows the live Model. The cold resource owner and an explicit
// original retirement continuation survive host Model teardown when needed.
class Session final : public std::enable_shared_from_this<Session> {
  friend class Model;
  Model *model_;
  std::shared_ptr<ColdResources> resources_;
  explicit Session(Model &model, std::shared_ptr<ColdResources> resources, const PreparedSampling &sampling) noexcept
      : model_(&model), resources_(std::move(resources)), sampling_(sampling) {}
  const PreparedSampling &sampling_;
public:
  cache::WorkingState head;
  hyle::container::SmallVector<c::PagedPageAppender, 32> tails;
  std::array<c::PagedPageAppender, 2> draft_tails;
  std::uint64_t tail_epoch = 0, draft_tail_epoch = 0;
  StepState *pending = nullptr; // Borrowed from Model; episode pins this Session.
  bool failed = false, stopping = false;
  hyle::Status retirement_status;
  std::size_t length() const noexcept { return head ? head->length : 0; }
  cache::SnapshotRef checkpoint() {
    if (pending) throw std::runtime_error("checkpoint requires an idle committed model boundary");
    return head.checkpoint();
  }
  const PreparedSampling &sampling_configuration() const noexcept { return sampling_; }
  std::weak_ptr<ColdResources> sampling_owner() const noexcept { return resources_; }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  // After ordinary multi-request submission, stop withdraws only this member's
  // publication while a survivor remains. Other roles and pre-submit stop abort
  // the whole episode. Cancellation never proves native retirement.
  hyle::Status stop() noexcept;
  inference::RetirementObservation try_retire() noexcept;
  inference::RetirementObservation drain() noexcept;
  ~Session();
};
struct EvaluationRequest final {
  std::shared_ptr<Session> session;
  std::span<const std::int32_t> ids;
  const inference::RequestStopPolicy *stop_policy = nullptr;
  LogitsInputs logits;
};

// CPU topology only, never a candidate/head or permission. A weak identity and
// the next committed epoch qualify consumption after the original predecessor.
struct DecodeAheadMetadata final {
  template<class T, std::size_t N>
  using Values = hyle::container::SmallVector<T, N, mem::AllocatorRef>;
  struct Member final {
    std::weak_ptr<Session> session;
    std::uint64_t epoch = 0;
    std::size_t base_length = 0, maximum_length = 0, sampling_index = 0;
    bool ready = false, selected = false;
  };
  c::ResourceBudget budget; // Outlives its borrowed metadata allocators.
  Values<Member, 2> members;
  // One fixed stripe of original target-page IDs per member. Selecting a
  // prefix must not overwrite a still-unconsumed sibling's integers.
  Values<std::int32_t, 2> candidate_ids;
  Values<std::size_t, 2> selected;
  Values<std::int32_t, 2> positions, old_ids, old_indptr, old_last, old_query;
  std::size_t page_stride = 0, maximum_rows = 1;
  bool ready = false;
  DecodeAheadMetadata(c::ResourceBudget owner, mem::AllocatorRef allocator)
      : budget(std::move(owner)), members(allocator), candidate_ids(allocator), selected(allocator),
        positions(allocator), old_ids(allocator), old_indptr(allocator), old_last(allocator), old_query(allocator) {}
  void clear_output() noexcept {
    for (const auto index : selected) members[index].selected = false;
    selected.clear(); positions.clear(); old_ids.clear(); old_indptr.clear(); old_last.clear(); old_query.clear();
  }
  void refresh() noexcept {
    ready = std::ranges::any_of(members, [](const auto &member) { return member.ready; });
  }
  void clear() noexcept {
    ready = false;
    for (auto &member : members) member = {};
    clear_output();
  }
};
enum class EvaluationDisposition : std::uint8_t { kNotPublished, kPublished, kCancelled };
struct EvaluationOutput final {
  std::shared_ptr<Session> session;
  bool capture_tensors = false, capture_probes = false;
  double *complete_seconds = nullptr;
  Clock::time_point *host_endpoint = nullptr;
  inference::StepObservation *observation = nullptr;
  inference::PreparationObservation *preparation = nullptr;
  GenerationLedger *generation_ledger = nullptr;
  const GenerationCommitRecord *expected_episode = nullptr;
  // Optional actual publication fact. A stopped member's raw next_ids element
  // is not a delivered result; kCancelled has no observer/head/ledger commit.
  EvaluationDisposition *disposition = nullptr;
};
// Borrowed only during the synchronous post-commit observer. Original output
// leases and the carrier admission guard remain in finish's stack scope.
struct CompletedEvaluationView final {
  inference::StepObservation &observation;
  const cache::State &head;
  std::size_t member_index = 0, requests = 1;
  c::ConstTensorView logits{}, hidden{};
  std::span<const c::ConstTensorView> probes{};
};
// Entry-owned naming and IO policy. The model receives only capture demands.
struct EvaluationIO final {
  std::string_view name;
  const std::filesystem::path *trace = nullptr;
  bool trace_states = true;
  std::string_view probe_subdirectory{};
};
struct MtpRoundResult final {
  std::int32_t pending = -1, draft = -1, target_first = -1, target_second = -1, next_id = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  std::size_t processed_target = 0, processed_draft = 0;
  std::size_t proposal_count = 0, consumed_drafts = 0, verified_rows = 0;
  std::size_t auxiliary_graph_calls = 0;
  std::vector<std::int32_t> emitted;
};
struct MtpGenerationSelection final {
  std::shared_ptr<Session> session;
  std::int32_t expected_pending = -1;
  std::size_t output_quota = 0;
  std::span<const c::ReadLease> draft_log_probabilities{};
  std::size_t proposal_limit = 0;
};

// Finite NextN-only programs share the original model weights and execution
// endpoint. Their speculative pages never become a Session head. Target
// verification repairs the retained draft prefix with actual target hidden
// rows before the existing Snapshot transaction can publish it.
struct DraftProgram final {
  inference::QueueBindings queues;
  std::size_t rows = 0;
  bool head = false;
  g::ReplayMode replay_mode = g::ReplayMode::kCommands;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  ResourceRetention retirement;
  V tokens, positions, key, value, metadata, hidden, id, identity, log_probabilities, valid, probability;
  std::vector<V> hidden_inputs, resource_ports;
  mtp::AppendNodes append;
  g::NodeId attention;
  Tensor token_values, position_values, identity_values;
  PreparedReadback id_value, valid_value, probability_value;
  c::LeasePool probability_pool;
  PreparedOutputFacts hidden_output, probabilities_output;
};
struct DraftWork final {
  cache::DraftState state;
  std::array<c::PagedPageAppender, 2> tails;
};
struct DraftProposal final {
  std::int32_t id = -1;
  float probability = 0;
  c::ReadLease hidden, log_probabilities;
};
struct PrefixSamplingProgram final {
  inference::QueueBindings queues;
  std::size_t proposals = 0;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  ResourceRetention retirement;
  V logits, identities, ids;
  std::vector<V> q, accepted, final_ids, valid;
  std::vector<V> resource_ports;
  Tensor identity_values, id_values;
  std::vector<PreparedReadback> accepted_values, final_values, valid_values;
};
struct PrefixSamplingResult final {
  std::vector<std::uint8_t> accepted;
  std::vector<std::int32_t> final_ids;
};

struct SamplingFrontier final {
  inference::QueueBindings queues;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::DimensionId last_row;
  V hidden_port, identity_port, id_port, valid_port, bias_port, allowed_port;
  Tensor identity;
  PreparedReadback id, valid;
  ResourceRetention retirement;
};

struct PreparedSampling final {
  const std::size_t index;
  const SamplingConfiguration configuration;
  std::unique_ptr<SamplingFrontier> frontier;
  PreparedSampling(std::size_t ordinal, SamplingConfiguration value) : index(ordinal), configuration(value) {}
};

// Actual cold storage and policy addresses borrowed by submitted Graphs. The
// application Model and Session links are absent: a failed native prefix must
// never retain or later dereference a destroyed host application object.
struct ColdResources final {
  // The Model hands off its original endpoint on teardown, including normal
  // retirement. Destroy payloads before this resource controller's final owner.
  std::optional<Backend> retirement_backend;
  std::vector<c::LeasePool> weight_pools;
  std::map<std::string, LoadedWeight> weights;
  ProjectionBindings projections;
  WeightMeanings weight_meanings;
  std::string source_identity;
  std::vector<StateSchema> state_schema;
  const prepared::CachePolicy cache_policy;
  const std::vector<CacheStoragePair> attention_storage;
  c::ResourceBudget cache_budget, execution_budget, allocation_budget;
  cache::VersionCount snapshot_versions;
  std::size_t carrier_borrowed = 0;
  cache::PageInventory page_inventory, draft_inventory;
  std::vector<cache::StatePools> state_pools;
  c::LeasePool hidden_pool;
  std::vector<std::unique_ptr<PreparedSampling>> sampling_configurations;
  const bool retain_sampling_frontier;
  std::map<FamilyKey, std::unique_ptr<Family>> families, probe_families;
  std::map<ExecutionShape, std::unique_ptr<Phase>> phases, probe_phases;
  std::map<std::pair<std::size_t, bool>, std::unique_ptr<DraftProgram>> draft_programs;
  std::map<std::size_t, std::unique_ptr<PrefixSamplingProgram>> prefix_sampling_programs;

  explicit ColdResources(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision,
      SamplingConfiguration first, std::span<const SamplingConfiguration> additional,
      prepared::CachePolicy policy, std::vector<CacheStoragePair> storage)
      : state_schema(state_schema_for(geometry, capacity, precision)),
        cache_policy(policy), attention_storage(std::move(storage)), state_pools(geometry.layers.size()),
        retain_sampling_frontier(first.enabled || !first.modifiers.empty() || std::ranges::any_of(additional,
            [](const auto &value) { return value.enabled || !value.modifiers.empty(); })) {
    sampling_configurations.reserve(cache::add(additional.size(), 1));
    const auto add = [&](SamplingConfiguration value) {
      validate_sampling_configuration(value, geometry.vocabulary);
      for (const auto &old : sampling_configurations)
        if (same_sampling_configuration(old->configuration, value))
          throw std::runtime_error("duplicate cold sampling configuration");
      sampling_configurations.push_back(std::make_unique<PreparedSampling>(sampling_configurations.size(), value));
    };
    add(first); for (const auto &value : additional) add(value);
  }

  const PreparedSampling &sampling_configuration(std::size_t index) const {
    if (index >= sampling_configurations.size()) throw std::runtime_error("sampling configuration was not cold prepared");
    return *sampling_configurations[index];
  }
  std::size_t sampling_index(const PreparedSampling *value) const {
    if (!value) return 0;
    for (const auto &entry : sampling_configurations) if (entry.get() == value) return entry->index;
    throw std::runtime_error("sampling configuration belongs to another model");
  }
  bool pending() const noexcept {
    for (const auto *cache : {&phases, &probe_phases})
      for (const auto &[shape, phase] : *cache) {
        (void)shape;
        for (const auto &pin : phase->retirement) if (pin.pending()) return true;
        for (const auto &copy : phase->hidden_output) if (copy.retirement.pending()) return true;
      }
    for (const auto &[shape, program] : draft_programs) {
      (void)shape; if (program->retirement.pending()) return true;
    }
    for (const auto &[width, program] : prefix_sampling_programs) {
      (void)width; if (program->retirement.pending()) return true;
    }
    return std::ranges::any_of(sampling_configurations, [](const auto &entry) {
      return entry->frontier && entry->frontier->retirement.pending();
    });
  }
  void release_invocation_roots() noexcept {
    // The accepted attachment is the only temporary scope->Graph->scope
    // cycle. Drop these application roots at handoff; the original native
    // Completion still retains its Graph until actual retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape; for (auto &run : phase->slots) run = {};
        for (auto &copy : phase->hidden_output) { copy.update = {}; copy.invocation = {}; }
      }
    for (auto &[shape, program] : draft_programs) { (void)shape; program->run = {}; }
    for (auto &[width, program] : prefix_sampling_programs) { (void)width; program->run = {}; }
  }
  inference::RetirementObservation retire(bool wait) noexcept {
    hyle::Status status;
    if (retirement_backend) {
      g::Invocation empty;
      status = wait ? retirement_backend->drain_retired(empty) : retirement_backend->collect_retired();
    }
    // A pending original guard is not an observation failure. Preserve actual
    // backend errors independently; neither fact substitutes for the other.
    return {!pending(), status};
  }
};

class Model final {
public:
  struct Sampling final {
    bool enabled = false;
    sampling::Policy policy;
    std::uint64_t seed = 0;
    sampling::LogitsPolicy modifiers;
  };
  struct DraftPolicy final {
    std::size_t max_drafts = 1;
    float minimum_probability = 0;
  };
private:
  DraftPolicy draft_policy_;
  bool preparation_started_ = false;
public:
  const std::uint64_t sampling_seed;
  std::size_t max_drafts() const noexcept { return draft_policy_.max_drafts; }
  float min_draft_probability() const noexcept { return draft_policy_.minimum_probability; }
  void configure_draft_policy(DraftPolicy policy) {
    if (preparation_started_)
      throw std::runtime_error("configure draft policy before the first model preparation");
    if (!std::isfinite(policy.minimum_probability) || policy.minimum_probability < 0 || policy.minimum_probability > 1)
      throw std::runtime_error("MTP minimum draft probability must be finite and within [0,1]");
    if (policy.max_drafts >= capacity ||
        (cache_plan.draft_cache && policy.max_drafts >= cache_plan.candidate_boundaries))
      throw std::runtime_error("MTP maximum drafts exceed the original context or candidate capacity");
    draft_policy_ = policy;
  }
  bool recursive_mtp() const noexcept { return max_drafts() && (max_drafts() > 1 || min_draft_probability() > 0); }
  const FfnPlacement ffn_placement;
  std::size_t host_weight_bytes = 0, device_weight_bytes = 0;
  std::size_t physical_text_roles = 0, physical_mtp_roles = 0, canonical_weight_bytes = 0;
  std::size_t ct_channel_groups = 0, ct_local_groups = 0, ct_block_groups = 0, omitted_cache_roles = 0;
  std::size_t fp8_weight_groups = 0, fp8_import_peak_bytes = 0;
  bool float8_weight_policy = false;
  std::array<std::size_t, 3> known_workspace_peak{}, known_workspace_limits{};
  const PrecisionProfile precision;
  const ModelGeometry geometry;
  const std::int64_t hidden, intermediate, vocabulary;
  const std::size_t target_kv_layers, draft_bank;
  const std::vector<std::size_t> attention_layer_ids;
  Backend execution;
  std::shared_ptr<ColdResources> resources_;
  const SamplingConfiguration &default_sampler; // The immutable default, shared by original MTP paths.
  // These original runtimes outlive every phase/slot and its native retirement.
  std::unique_ptr<hyle::sched::Scheduler> scheduler;
  std::unique_ptr<hyle::sched::BlockingPool> workers;
  std::vector<c::LeasePool> &weight_pools;
  std::map<std::string, LoadedWeight> &weights;
  ProjectionBindings &projections;
  WeightMeanings &weight_meanings;
  std::string &source_identity;
  std::vector<StateSchema> &state_schema;
  c::ResourceBudget &cache_budget, &execution_budget, &allocation_budget;
  std::size_t known_cache_backing_bytes = 0;
  bool state_storage_prepared = false;
  bool hidden_writers = false;
  bool generated_state_outputs = false;
  const cache::Plan cache_plan;
  c::BatchSchema attention_schema;
  std::vector<c::BatchSchema> attention_schemas;
  const c::BatchSchema &target_attention_schema(std::size_t layer) const {
    if (layer >= target_kv_layers) throw std::runtime_error("invalid target attention cache slot");
    return attention_schemas.empty() ? attention_schema : attention_schemas.at(layer);
  }
  cache::VersionCount &snapshot_versions;
  std::size_t &carrier_borrowed;
  static constexpr std::size_t slot_index = 0;
  cache::PageInventory &page_inventory, &draft_inventory;
  std::vector<cache::StatePools> &state_pools;
  c::LeasePool &hidden_pool;
  // Family/child owners are shared by sessions; no session addresses enter compile.
  std::map<FamilyKey, std::unique_ptr<Family>> &families, &probe_families;
  std::vector<FamilyKey> admitted_exact_keys;
  std::size_t exact_admissions = 0, exact_evictions = 0;
  std::size_t exact_shape_capacity = 0;
  // One carrier per actually prepared finite shape. Serial admission borrows
  // its one slot and shared IO through native retirement and all result reads.
  std::map<ExecutionShape, std::unique_ptr<Phase>> &phases, &probe_phases;
  std::map<ExecutionShape, std::array<std::size_t, execution_slot_count>> planned_slot_uses;
  std::map<std::pair<std::size_t, bool>, std::unique_ptr<DraftProgram>> &draft_programs;
  std::map<std::size_t, std::unique_ptr<PrefixSamplingProgram>> &prefix_sampling_programs;
  std::size_t draft_program_graph_calls = 0, prefix_decision_graph_calls = 0;
  const PreparedSampling &sampling_configuration(std::size_t index = 0) const { return resources_->sampling_configuration(index); }
  std::size_t sampling_configuration_count() const noexcept { return resources_->sampling_configurations.size(); }
  std::size_t sampling_index(const PreparedSampling *value) const { return resources_->sampling_index(value); }
  std::size_t sampling_index(const GenerationLedger &ledger) const {
    if (ledger.sampling && (ledger.sampling_owner.expired() || ledger.sampling_owner.owner_before(resources_) || resources_.owner_before(ledger.sampling_owner)))
      throw std::runtime_error("generation ledger belongs to another or destroyed model");
    return sampling_index(ledger.sampling);
  }
  bool retains_sampling_frontier() const noexcept {
    return resources_->retain_sampling_frontier;
  }
  std::vector<std::weak_ptr<Session>> session_links;
  std::size_t frontier_sampling_graph_calls = 0;
  double frontier_sampling_seconds = 0;
  // Destroy episode pins/borrowed links before carriers and the raw policy
  // owners referenced by Snapshot/PageRef, including exceptional shutdown.
  InputPreparationScratch input_scratch;
  std::shared_ptr<StepState> active_step;
  std::shared_ptr<StepState> successor_step;
  std::optional<DecodeAheadMetadata> decode_ahead;
  std::size_t cpu_ahead_prepared = 0, cpu_ahead_hits = 0, cpu_ahead_discarded = 0;
  double cpu_ahead_prepare_seconds = 0;
  const std::size_t capacity;
  const ReplayPolicy replay_policy;
  const std::filesystem::path layer0_probe;
  std::filesystem::path trace_routes;
  explicit Model(const std::filesystem::path &manifest, const ModelGeometry &model_geometry, std::size_t selected_capacity,
                 cache::Plan selected_cache,
                 std::filesystem::path probe = {}, bool progress = false,
                  ReplayPolicy replay = ReplayPolicy::kAuto,
                  std::size_t max_cache_bytes = (std::numeric_limits<std::size_t>::max)(),
                  Sampling sampling_configuration = {false, {}, 0, {}},
                  std::size_t cpu_ffn_layer_count = 0,
                  BackendSelection backend_selection = {}, std::span<const SamplingConfiguration> additional_sampling = {})
       : Model(prepare_weight_source(manifest, selected_cache.draft_cache), model_geometry, selected_capacity,
           selected_cache, std::move(probe), progress, replay, max_cache_bytes,
           sampling_configuration, cpu_ffn_layer_count, backend_selection, additional_sampling) {}

  Model(PreparedWeightSource source, const ModelGeometry &model_geometry, std::size_t selected_capacity,
      cache::Plan selected_cache, std::filesystem::path probe, bool progress, ReplayPolicy replay,
      std::size_t max_cache_bytes, Sampling sampling_configuration, std::size_t cpu_ffn_layer_count,
      BackendSelection backend_selection, std::span<const SamplingConfiguration> additional_sampling = {})
       : Model(create_execution(model_geometry, selected_cache, cpu_ffn_layer_count, backend_selection), {},
           std::move(source), model_geometry, selected_capacity, selected_cache, std::move(probe), progress,
           replay, max_cache_bytes, sampling_configuration, cpu_ffn_layer_count, additional_sampling) {}

  // Explicit cold seam: endpoints exist before the caller constructs one
  // requested-byte scope from their actual resource identities. No new owner
  // or endpoint is created here. The default constructor delegates with no
  // allocation budget, retaining its original allocation path.
  Model(Backend &&prepared_execution, c::ResourceBudget shared_budget,
      PreparedWeightSource source, const ModelGeometry &model_geometry, std::size_t selected_capacity,
      cache::Plan selected_cache, std::filesystem::path probe, bool progress, ReplayPolicy replay,
      std::size_t max_cache_bytes, Sampling sampling_configuration, std::size_t cpu_ffn_layer_count,
      std::span<const SamplingConfiguration> additional_sampling = {})
       : sampling_seed(sampling_configuration.seed), ffn_placement(source.roles, cpu_ffn_layer_count),
         precision(PrecisionProfile::from_roles(source.roles)),
         geometry(model_geometry), hidden(geometry.hidden), intermediate(geometry.intermediate), vocabulary(geometry.vocabulary),
         target_kv_layers(geometry.attention_layers()), draft_bank(target_kv_layers * 2),
         attention_layer_ids([&] {
           std::vector<std::size_t> indices; indices.reserve(target_kv_layers);
           for (std::size_t i = 0; i < geometry.layers.size(); ++i) if (!geometry.is_gdn(i)) indices.push_back(i);
           return indices;
         }()),
         execution(std::move(prepared_execution)),
         resources_(std::make_shared<ColdResources>(geometry, selected_capacity, precision,
              SamplingConfiguration{sampling_configuration.enabled, sampling_configuration.policy, sampling_configuration.modifiers}, additional_sampling,
              source.roles.cache_policy, std::move(source.cache_storage))),
          default_sampler(resources_->sampling_configurations[0]->configuration),
         weight_pools(resources_->weight_pools), weights(resources_->weights), projections(resources_->projections),
         weight_meanings(resources_->weight_meanings), source_identity(resources_->source_identity),
         state_schema(resources_->state_schema), cache_budget(resources_->cache_budget), execution_budget(resources_->execution_budget),
         allocation_budget(resources_->allocation_budget),
         cache_plan(selected_cache), snapshot_versions(resources_->snapshot_versions), carrier_borrowed(resources_->carrier_borrowed),
         page_inventory(resources_->page_inventory), draft_inventory(resources_->draft_inventory),
         state_pools(resources_->state_pools), hidden_pool(resources_->hidden_pool),
         families(resources_->families), probe_families(resources_->probe_families),
         phases(resources_->phases), probe_phases(resources_->probe_phases),
         draft_programs(resources_->draft_programs), prefix_sampling_programs(resources_->prefix_sampling_programs),
         capacity(selected_capacity),
         replay_policy(replay), layer0_probe(std::move(probe)) {
    geometry.validate();
    validate_mtp_source(geometry, cache_plan.draft_cache);
    FfnPlacement::validate(ffn_placement.cpu_layers(), geometry.layers.size());
    if (execution.kind() == BackendKind::kDefault || execution.has_cpu_assist() != (ffn_placement.cpu_layers() != 0))
      throw std::runtime_error("prepared Backend differs from the model CPU placement");
    allocation_budget = std::move(shared_budget);
    if (allocation_budget) {
      // Validate actual rows before any managed source allocation. Resources
      // are explicit; absent rows never become inferred/unattributed charges.
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        (void)take(allocation_budget.usage(resource));
      if (execution.has_cpu_assist()) (void)take(allocation_budget.usage(execution.host_device()));
    }
    const auto &source_roles = source.roles;
    if (source_roles.geometry != geometry) throw std::runtime_error("source geometry changed before weight loading");
    const bool calibrated = resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration;
    if ((calibrated && resources_->attention_storage.size() != target_kv_layers) ||
        (!calibrated && !resources_->attention_storage.empty()) || (calibrated && cache_plan.draft_cache))
      throw std::runtime_error("prepared E4 cache storage requires its complete ordinary target pairs");
    if (cache_plan.row_bytes != source.cache_row_bytes ||
        cache_plan.key_bank_bytes != cache::add(cache::multiply(cache::multiply(cache_plan.max_pages,
            cache::Plan::page_rows), source.cache_row_bytes.key), 1) ||
        cache_plan.value_bank_bytes != cache::add(cache::multiply(cache::multiply(cache_plan.max_pages,
            cache::Plan::page_rows), source.cache_row_bytes.value), 1))
      throw std::runtime_error("cache backing plan differs from prepared physical storage");
    if (cache_plan.hidden_row_bytes != geometry.hidden_bytes() || cache_plan.attention_layers != target_kv_layers ||
        cache_plan.recurrent_layers != geometry.recurrent_layers())
      throw std::runtime_error("cache layout differs from source model geometry");
    if (cache_plan.draft_cache && (!additional_sampling.empty() || !default_sampler.modifiers.empty()))
      throw std::runtime_error("nondefault sampling configurations are currently ordinary Text only");
    if (retains_sampling_frontier() && (!cache_plan.hidden_bytes || cache_plan.hidden_slots < cache_plan.snapshot_slots))
      throw std::runtime_error("sampling requires a cache plan with retained frontier storage");
    const std::array limits{c::ResourceBudgetLimit{execution.device(), max_cache_bytes},
        c::ResourceBudgetLimit{mem::system_resource(), (std::numeric_limits<std::size_t>::max)()}};
    // CPU payload and host metadata can use the same real resource. Keep the
    // caller's finite cache limit when they coincide; never replace it with
    // the otherwise independent unlimited metadata-domain entry.
    const auto limit_count = execution.device() == mem::system_resource() ? 1u : 2u;
    // This allocator couples only actual CPU metadata. The explicit second
    // source-factory constraint below separately gates original cache payloads.
    const auto cache_allocator = allocation_budget ? take(allocation_budget.metadata_allocator()) : mem::system_allocator();
    cache_budget = take(c::ResourceBudget::create({.limits = std::span(limits).first(limit_count),
        .allocator = cache_allocator}));
    if (allocation_budget) execution_budget = allocation_budget;
    else {
      // Execution controls and native outputs have their own actual resource
      // scope on every backend. State/KV/hidden pools remain cache storage;
      // private slot indices and their pinned staging do not consume its plan.
      std::vector<c::ResourceBudgetLimit> execution_limits;
      execution_limits.reserve(4);
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        if (std::ranges::none_of(execution_limits, [resource](const auto &limit) {
              return limit.resource == resource;
            }))
          execution_limits.push_back({resource});
      if (ffn_placement.cpu_layers()) {
        const auto host = weight_resource(execution, true);
        if (std::ranges::none_of(execution_limits, [host](const auto &limit) { return limit.resource == host; }))
          execution_limits.push_back({host});
      }
      execution_budget = take(c::ResourceBudget::create({.limits = execution_limits, .allow_unattributed = true}));
    }
    if (progress) {
      auto status = hyle::Status::Ok();
      scheduler = std::make_unique<hyle::sched::Scheduler>(1, mem::system_allocator(), 64, &status);
      check(status);
      workers = std::make_unique<hyle::sched::BlockingPool>(mem::system_allocator(), 1,
                                                          std::chrono::seconds(10), &status);
      check(status);
    }
    const auto weight_load_start = Clock::now();
    auto loading = load_selected_weight_roles(source_roles, execution, weights, weight_pools, ffn_placement, allocation_budget);
    source_identity = loading.identity;
    weight_meanings = std::move(loading.meanings);
    projections = prepare_projection_bindings(weights, cache_plan.draft_cache, geometry);
    host_weight_bytes = loading.host_bytes; device_weight_bytes = loading.device_bytes;
    physical_text_roles = loading.physical_text_roles; physical_mtp_roles = loading.physical_mtp_roles;
    canonical_weight_bytes = loading.canonical_bytes; fp8_weight_groups = loading.fp8_groups;
    fp8_import_peak_bytes = loading.import_peak_bytes; float8_weight_policy = loading.float8_policy;
    ct_channel_groups = loading.ct_channel_groups; ct_local_groups = loading.ct_local_groups;
    ct_block_groups = loading.ct_block_groups;
    omitted_cache_roles = loading.omitted_cache_roles;
    const auto weight_load_seconds = source.metadata_prepare_seconds +
        std::chrono::duration<double>(Clock::now() - weight_load_start).count();
    std::cerr << "resident text weights: " << weights.size() << " tensors, " << loading.bytes << " bytes\n";
    if (float8_weight_policy) std::cerr << "fp8_weight_import original_physical_roles="
        << physical_text_roles + physical_mtp_roles << " original_bytes=" << loading.bytes
        << " canonical_tensors=" << weights.size() << " canonical_bytes=" << canonical_weight_bytes
        << " groups=" << fp8_weight_groups << " temporary_group_peak_bytes=" << fp8_import_peak_bytes << '\n';
    if (loading.nf4_groups) std::cerr << "nf4_source_operands groups=" << loading.nf4_groups
        << " original_group_bytes=" << loading.nf4_original_bytes << " canonical_bank_bytes=" << canonical_weight_bytes
        << " input_copy_bytes=" << loading.nf4_input_copy_bytes
        << " known_group_peak_bytes=" << loading.nf4_known_group_peak_bytes
        << " source_operands_resident=true decode_precision=F16 selected_plan_reports_packed_or_decomposed=true\n";
    std::cerr << "weights_load_seconds=" << weight_load_seconds << '\n';
    if (ffn_placement.cpu_layers()) std::cerr << "weight_placement cpu_ffn_layers=" << ffn_placement.cpu_layers()
        << " original_host_weight_bytes=" << host_weight_bytes
        << " original_device_weight_bytes=" << device_weight_bytes
        << " duplicated_original_weight_bytes=0\n";
    std::cerr << "weight_loader backend=" << execution.name() << " staging_bytes=" << loading.stream.staging_bytes
              << " io_calls=" << loading.stream.io_calls
              << " native_calls=" << loading.stream.native_calls
              << " native_waits=" << loading.stream.native_waits
              << " host_waits=" << loading.stream.host_waits
              << " phase_completions=" << loading.stream.phase_completions
              << " copied_physical_bytes=" << loading.stream.copied_physical_bytes
              << " peak_backing_bytes=" << loading.stream.peak_backing_bytes << '\n';
  }

  ~Model() {
    // Stop the host driver before joining its original scheduler/pool. Native
    // cancellation remains advisory and never serves as retirement proof.
    for (auto &weak : session_links) if (auto session = weak.lock()) session->stop().IgnoreError();
    (void)abort_episode();
    // Attempt actual native retirement before detaching application links.
    // Scheduler idleness alone does not prove native/completion retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[tokens, phase] : *cache) {
        (void)tokens;
        for (auto &run : phase->slots) execution.drain_retired(run).IgnoreError();
      }
    if (workers) workers->shutdown();
    if (scheduler) scheduler->drain();
    (void)abort_episode();
    // A persistent unknown observation may outlive this application Model.
    // Preserve the original native endpoint as an explicit Session retirement
    // continuation, while the accepted guard pins the complete cold scope.
    resources_->retirement_backend.emplace(std::move(execution));
    for (auto &weak : session_links) if (auto session = weak.lock()) {
      session->model_ = nullptr;
      session->stopping = true;
      session->pending = nullptr;
      if (active_step) inference::record_retirement_status(session->retirement_status, active_step->terminal);
    }
    resources_->release_invocation_roots();
    // Teardown never promotes an unpublished successor back into escaped
    // Sessions whose Model link has already been withdrawn above.
    if (successor_step) successor_step->candidate_admission = nullptr;
    if (active_step) active_step->candidate_admission = nullptr;
    successor_step.reset(); active_step.reset();
  }

  c::Completion submit_retained(g::Invocation &run, std::span<const g::QueueBinding> queues,
                                ResourceRetention &retention, IncomingStepSources *incoming = nullptr) {
    check(retention.reserve(resources_));
    auto accepted = incoming ? g::detail::submit_outputs(run, incoming->bindings, queues) : run.submit(queues);
    if (!accepted) { retention.reject(); return take(std::move(accepted)); }
    check(retention.attach(*accepted));
    return std::move(*accepted);
  }

  std::string_view score_precision_policy() const {
    return score_policy(projections.at(std::string(geometry.head_role())), precision);
  }
  std::string weight_inventory_report() const {
    auto result = ",\"weight_count_scope\":\"selected_physical_roles\",\"text_tensors\":" +
        std::to_string(physical_text_roles) + ",\"mtp_tensors\":" + std::to_string(physical_mtp_roles);
    if (float8_weight_policy) result += ",\"canonical_weight_tensors\":" + std::to_string(weights.size()) +
        ",\"canonical_weight_bytes\":" + std::to_string(canonical_weight_bytes) +
        ",\"fp8_block128_groups\":" + std::to_string(fp8_weight_groups) +
        ",\"fp8_import_temporary_group_peak_bytes\":" + std::to_string(fp8_import_peak_bytes) +
        ",\"fp8_block128_policy\":{\"activation_scale\":\"divide\",\"activation_floor_f32_bits\":\"2edbe6ff\","
        "\"input_dtype\":\"BF16\",\"accumulation_dtype\":\"F32\",\"output_dtype\":\"BF16\","
        "\"weight_scale_conversion\":\"bf16_to_f32_exact\",\"reader\":\"" + std::string(prepared::fp8_reader) + "\"}";
    if (ct_channel_groups || ct_local_groups) result += ",\"ct_channel_token_groups\":" + std::to_string(ct_channel_groups) +
        ",\"ct_local_nvfp4_groups\":" + std::to_string(ct_local_groups);
    if (ct_block_groups) result += ",\"ct_block128_bf16_groups\":" + std::to_string(ct_block_groups);
    if (omitted_cache_roles) result += ",\"kv_cache_dtype\":\"BF16\",\"omitted_original_cache_calibration_roles\":" +
        std::to_string(omitted_cache_roles);
    if (resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration)
      result += ",\"kv_cache_dtype\":\"E4M3FN\",\"original_cache_calibration_roles\":" +
          std::to_string(resources_->attention_storage.size() * 2);
    return result;
  }
  Tensor caller_tensor(mem::MemoryResourceRef resource, c::TensorDesc description, bool caller = true) const {
    return Tensor(resource, description, caller, allocation_budget);
  }

  c::ResourceBudgetUsage cache_usage() const {
    const auto usage = take(cache_budget.usage(execution.device()));
    const auto metadata = usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)];
    const auto payload = usage.live_bytes - metadata;
    if (payload < known_cache_backing_bytes ||
        (!generated_state_outputs && payload != known_cache_backing_bytes))
      throw std::runtime_error("actual cache backing differs: payload=" + std::to_string(payload) +
          " known=" + std::to_string(known_cache_backing_bytes) + " metadata=" + std::to_string(metadata));
    return usage;
  }

  // Report the same accounting scope for ordinary, MTP and scheduled runs.
  // Native execution allocates before a completed result is retained; its
  // temporaries/private allocations do not become a physical-memory cap here.
  std::string cache_report() const {
    const auto usage = cache_usage();
    std::ostringstream out;
    if (sampling_configuration_count() > 1) out << ",\"sampling_scope\":\"default_configuration\"";
    out << ",\"sampling\":\"" << (default_sampler.enabled ? "random" : "greedy") << '"'
        << ",\"sampling_policy_order\":\"temperature_top_k_top_p\""
        << ",\"sampling_temperature\":" << default_sampler.policy.temperature
        << ",\"sampling_top_k\":" << default_sampler.policy.top_k
        << ",\"sampling_top_p\":" << default_sampler.policy.top_p
        << ",\"sampling_seed\":" << sampling_seed
        << ",\"mtp_acceptance_policy\":\"" << (!cache_plan.draft_cache ? "disabled" : default_sampler.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '"'
        << ",\"frontier_sampling_graph_calls\":" << frontier_sampling_graph_calls
        << ",\"mtp_max_drafts\":" << (cache_plan.draft_cache ? max_drafts() : 0)
        << ",\"mtp_min_draft_probability\":" << (cache_plan.draft_cache ? min_draft_probability() : 0)
        << ",\"draft_program_graph_calls\":" << draft_program_graph_calls
        << ",\"prefix_decision_graph_calls\":" << prefix_decision_graph_calls
        << ",\"cold_exact_admissions\":" << exact_admissions
        << ",\"cold_exact_evictions\":" << exact_evictions
        << ",\"cpu_ahead_prepared\":" << cpu_ahead_prepared
        << ",\"cpu_ahead_hits\":" << cpu_ahead_hits
        << ",\"cpu_ahead_discarded\":" << cpu_ahead_discarded
        << ",\"cpu_ahead_prepare_seconds\":" << cpu_ahead_prepare_seconds
        << ",\"backend\":\"" << execution.name()
        << "\",\"cache_budget_scope\":\"shared_pools_retained_outputs_and_resource_metadata\""
        << ",\"cpu_ffn_layers\":" << ffn_placement.cpu_layers()
        << ",\"original_host_weight_bytes\":" << host_weight_bytes
        << ",\"original_device_weight_bytes\":" << device_weight_bytes
        << ",\"cache_shared_backing_bytes\":" << known_cache_backing_bytes
        << ",\"cache_retained_output_backing_bytes\":" << usage.live_bytes -
            usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] - known_cache_backing_bytes
        << ",\"cache_metadata_requested_bytes\":" <<
            usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)]
        << ",\"cache_live_requested_bytes\":" << usage.live_bytes
        << ",\"cache_peak_requested_bytes\":" << usage.peak_bytes
        << ",\"cache_requested_byte_limit\":";
    if (usage.limit_bytes == SIZE_MAX) out << "null";
    else out << usage.limit_bytes;
    out << ",\"native_execution_private_bytes_unknown\":" <<
        ((execution.native_generated_storage() || execution.host_only() || ffn_placement.cpu_layers()) ? "true" : "false")
        << ",\"physical_vram_limit_known\":false";
    if (execution.host_only() || ffn_placement.cpu_layers()) {
      out << ",\"execution_shared_requested_byte_limit\":null"
          << ",\"workspace_limit_scope\":\"known_per_invocation_graph_workspace\""
          << ",\"workspace_domains\":[";
      const auto domain_count = execution.host_only() ? 1u : known_workspace_peak.size();
      for (std::size_t i = 0; i < domain_count; ++i) {
        if (i) out << ',';
        out << "{\"domain\":" << i << ",\"known_peak_bytes\":" << known_workspace_peak[i]
            << ",\"limit_bytes\":" << known_workspace_limits[i] << '}';
      }
      out << ']';
    }
    return out.str();
  }

  void record_execution_plan(const g::CompileReport &report, std::span<const g::ConstantBinding> constants) {
    if (execution.host_only()) {
      if (report.storage_domain_count() != 1 || report.target_count() != 1)
        throw std::runtime_error("CPU graph lost its single host allocation domain");
      const auto domain = take(report.storage_domain(0));
      if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
        throw std::runtime_error("CPU known workspace lacks its finite domain admission limit");
      known_workspace_peak[0] = (std::max)(known_workspace_peak[0], domain.workspace_bytes);
      known_workspace_limits[0] = domain.workspace_budget_bytes;
      std::cerr << "cpu_plan known_workspace_host=" << domain.workspace_bytes
          << " workspace_host_limit=" << domain.workspace_budget_bytes
          << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_memory_cap=0\n";
      return;
    }
    if (!ffn_placement.cpu_layers()) return;
    if (report.storage_domain_count() != 3 || report.target_count() != 2)
      throw std::runtime_error("CPU FFN graph lost its configured CPU/GPU domains");
    for (std::size_t i = 0; i < known_workspace_peak.size(); ++i) {
      const auto domain = take(report.storage_domain(i));
      if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
        throw std::runtime_error("hybrid known workspace lacks its finite domain admission limit");
      known_workspace_peak[i] = (std::max)(known_workspace_peak[i], domain.workspace_bytes);
      known_workspace_limits[i] = domain.workspace_budget_bytes;
    }
    std::size_t host_constants = 0;
    for (std::size_t i = 0; i < report.constant_count(); ++i) {
      const auto constant = take(report.constant(i));
      const auto source = std::ranges::find(constants, constant.value, &g::ConstantBinding::value);
      if (source == constants.end()) throw std::runtime_error("hybrid constant lost its original binding");
      if (source->tensor.storage().resource() != weight_resource(execution, true)) continue;
      if (constant.storage != g::StorageDomainId{1} || constant.materialized_bytes)
        throw std::runtime_error("CPU FFN constant acquired an unexpected canonical replica");
      for (std::size_t value = 0; value < report.physical_value_count(); ++value) {
        const auto physical = take(report.physical_value(value));
        if (physical.value == constant.value && physical.constant && physical.storage != g::StorageDomainId{1})
          throw std::runtime_error("CPU FFN constant escaped its host allocation domain");
      }
      ++host_constants;
    }
    std::cerr << "hybrid_plan host_constants=" << host_constants << " materialized_constant_bytes="
        << report.materialized_constant_bytes() << " activation_copy_bytes=" << report.copy_bytes()
        << " known_workspace_device=" << take(report.storage_domain(0)).workspace_bytes
        << " known_workspace_host=" << take(report.storage_domain(1)).workspace_bytes
        << " known_workspace_staging=" << take(report.storage_domain(2)).workspace_bytes
        << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_vram_cap=0\n";
  }

  void validate_state_output(const PreparedOutputFacts &output, const c::TensorDesc &expected) const {
    if (output.description != expected)
      throw std::runtime_error("prepared state output differs from its model state description");
    if (output.caller_owned() && take(execution.resource_for(output.storage)) != execution.device())
      throw std::runtime_error("prepared state output is outside its original state pool resource");
  }

  c::ReadLease retain_state_storage(g::Invocation &run, const PreparedOutputFacts &output) const {
    auto retained = inference::retain_output(run, output, cache_budget, allocation_budget);
    if (retained) return std::move(*retained);
    const auto usage = cache_usage();
    throw std::runtime_error("cache retained-state admission failed before snapshot publication: " +
        std::string(retained.error().message()) + "; live=" + std::to_string(usage.live_bytes) +
        ", limit=" + std::to_string(usage.limit_bytes) +
        "; adjust the exhausted cache/shared byte limit or reduce retained sessions/context; old snapshots remain valid");
  }

  void prepare_cache() {
    preparation_started_ = true;
    if (attention_schema.valid()) {
      return;
    }
    const auto storage = take(c::CacheStorage::plain(precision.activation_type()));
    auto prepared_schema = take(c::BatchSchema::create({
        .storage = c::BatchStorage::kPaged, .query_type = precision.activation_type(),
        .output_type = precision.activation_type(), .key_storage = storage, .value_storage = storage,
        .max_requests = cache_plan.max_requests, .max_query_rows = cache_plan.max_query_rows,
        .max_kv_rows = cache_plan.max_kv_rows, .max_pages = cache_plan.max_pages,
        .max_page_refs = cache_plan.max_page_refs, .query_heads = static_cast<std::uint32_t>(geometry.query_heads), .kv_heads = static_cast<std::uint32_t>(geometry.kv_heads),
        .qk_head_dim = static_cast<std::uint32_t>(geometry.key_depth), .value_head_dim = static_cast<std::uint32_t>(geometry.value_depth), .page_size = cache::Plan::page_rows},
        {.resource_budget = cache_budget}));
    std::vector<c::BatchSchema> prepared_schemas;
    prepared_schemas.reserve(resources_->attention_storage.size());
    for (const auto &pair : resources_->attention_storage)
      prepared_schemas.push_back(take(prepared_schema.with_cache_storage(pair[0], pair[1])));
    cache::PageInventory prepared_inventory, prepared_draft;
    prepared_inventory.prepare(cache_plan.max_pages, target_kv_layers);
    for (std::size_t i = 0; i < prepared_inventory.keys.size(); ++i) {
      const auto &schema = prepared_schemas.empty() ? prepared_schema : prepared_schemas[i];
      prepared_inventory.keys[i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_inventory.values[i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    if (cache_plan.draft_cache) {
      prepared_draft.prepare(cache_plan.max_pages, 1);
      prepared_draft.keys[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_draft.values[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    const auto prepared_bytes = cache::multiply(cache::add(cache_plan.key_bank_bytes, cache_plan.value_bank_bytes),
        target_kv_layers + (cache_plan.draft_cache ? 1u : 0u));
    // No PageRef exists until this complete cold source tuple is published.
    // Leave the schema sentinel invalid on failure so released quota can retry.
    static_assert(std::is_nothrow_move_assignable_v<cache::PageInventory>);
    static_assert(std::is_nothrow_move_assignable_v<c::BatchSchema>);
    page_inventory = std::move(prepared_inventory);
    draft_inventory = std::move(prepared_draft);
    known_cache_backing_bytes = prepared_bytes;
    snapshot_versions.capacity = cache_plan.snapshot_slots;
    attention_schemas = std::move(prepared_schemas);
    attention_schema = std::move(prepared_schema);
    prepare_decode_ahead_storage();
  }

  void prepare_state_storage() {
    preparation_started_ = true;
    if (state_storage_prepared) return;
    prepare_sampling_frontier();
    // History and recurrence always use the original versioned state pools.
    // Hidden results keep their actual caller/generated output contract.
    // Publish together so a failed allocation releases its temporary backing
    // and leaves this Model ready for a later admission attempt.
    std::vector<cache::StatePools> prepared_pools(geometry.layers.size());
    c::LeasePool prepared_hidden;
    auto prepared_bytes = known_cache_backing_bytes;
    const auto initial_slots = static_cast<std::uint32_t>(cache_plan.resident_sessions);
    for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
      const auto &description = state_schema[i]; auto &pools = prepared_pools[i];
      const auto history_slots = cache_plan.state_slots;
      const auto recurrent_slots = cache_plan.state_slots;
      const auto history_bytes = cache::state_pool_backing_bytes(take(description.history.required_bytes()), history_slots);
      const auto recurrent_bytes = cache::state_pool_backing_bytes(take(description.recurrent.required_bytes()), recurrent_slots);
      pools.history = take(c::LeasePool::create(execution.device(), history_bytes,
          {.max_blocks = history_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      pools.recurrent = take(c::LeasePool::create(execution.device(), recurrent_bytes,
          {.max_blocks = recurrent_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      const auto history_payload = take(description.history.required_bytes());
      const auto recurrent_payload = take(description.recurrent.required_bytes());
      pools.history_layout = take(c::LeaseSlotLayout::create(pools.history, description.history,
          history_slots, cache::add(history_payload, cache::state_pool_padding(history_payload))));
      pools.recurrent_layout = take(c::LeaseSlotLayout::create(pools.recurrent, description.recurrent,
          recurrent_slots, cache::add(recurrent_payload, cache::state_pool_padding(recurrent_payload))));
      check(pools.history_reuse.prepare(cache_plan.active_sequences, history_payload));
      check(pools.recurrent_reuse.prepare(cache_plan.active_sequences, recurrent_payload));
      prepared_bytes = cache::add(prepared_bytes, cache::add(history_bytes, recurrent_bytes));
    }
    if (cache_plan.hidden_bytes) {
      const auto slots = hidden_writers ? cache_plan.hidden_slots : initial_slots;
      const auto bytes = cache::state_pool_backing_bytes(geometry.hidden_bytes(), slots);
      prepared_hidden = take(c::LeasePool::create(execution.device(), bytes,
          {.max_blocks = slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      prepared_bytes = cache::add(prepared_bytes, bytes);
    }
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes - usage.live_by_category[
            static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] != prepared_bytes)
      throw std::runtime_error("shared cache backing inventory differs from the actual reserved bytes");
    state_pools = std::move(prepared_pools); hidden_pool = std::move(prepared_hidden);
    known_cache_backing_bytes = prepared_bytes;
    state_storage_prepared = true;
  }

  void prepare_family(std::span<const std::size_t> planned_chunks, bool diagnostic = false,
                      std::size_t requests = 1, GraphRole role = GraphRole::kText,
                      bool admitted_exact = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && (sampling_index || !selected_sampler.modifiers.empty()))
      throw std::runtime_error("nondefault sampling is ordinary Text only");
    if ((completion != TextCompletion::kEmitNext && completion != TextCompletion::kStateOnly) ||
        (completion == TextCompletion::kStateOnly && role != GraphRole::kText))
      throw std::runtime_error("state-only completion requires an ordinary Text family");
    preparation_started_ = true;
    if (state_storage_prepared && !admitted_exact)
      throw std::runtime_error("compile finite families before preparing their selected state storage");
    if (admitted_exact && (active_step || carrier_borrowed || diagnostic || planned_chunks.size() != 1 || verifies_prefix(role)))
      throw std::runtime_error("exact suffix admission requires an idle single cold shape within the original state plan");
    if (planned_chunks.empty()) throw std::runtime_error("text finite family requires known nonempty chunks");
    // Snapshot count is a fixed operation descriptor. Each verifier width
    // therefore owns its exact cold family, rather than changing it at submit.
    if (role == GraphRole::kVerify && planned_chunks.size() != 1) {
      std::set<std::size_t> widths(planned_chunks.begin(), planned_chunks.end());
      for (const auto width : widths) {
        const std::array chunk{width}; prepare_family(chunk, diagnostic, requests, role);
      }
      return;
    }
    const std::size_t prefix_rows = role == GraphRole::kVerify ? planned_chunks.front()
        : drafts_token(role) ? 2 : 1;
    if (!prefix_rows || prefix_rows > cache_plan.candidate_boundaries)
      throw std::runtime_error("verification width exceeds the positive prepared prefix capacity");
    const auto ports_per_request = cache::add(76, cache::multiply(38, prefix_rows - 1));
    if (!requests || requests > cache_plan.active_sequences ||
        requests > (std::numeric_limits<std::size_t>::max)() / ports_per_request)
      throw std::runtime_error("text finite family request count exceeds its configured capacity");
    if (verifies_prefix(role) && (diagnostic || !layer0_probe.empty() ||
        std::ranges::any_of(planned_chunks, [&](auto t) { return t != prefix_rows; })))
      throw std::runtime_error("verification requires one exact width and independently budgeted prefix boundaries");
    if (has_mtp(role) && (!cache_plan.draft_cache || diagnostic || !layer0_probe.empty()))
      throw std::runtime_error("MTP requires its complete cache plan and a non-diagnostic family");
    if (has_mtp(role) && !drafts_token(role) && !is_mtp_prefill(role) &&
        std::ranges::any_of(planned_chunks, [](auto t) { return t != 1; }))
      throw std::runtime_error("MTP single-token families require T1");
    if (is_mtp_prefill(role) && std::ranges::any_of(planned_chunks, [](auto t) { return t < 2; }))
      throw std::runtime_error("MTP multi-token prefill requires at least two rows; use its single-token family for T1");
    const auto family_key = admitted_exact ? FamilyKey::exact(requests, role, planned_chunks.front(), completion, sampling_index) : FamilyKey(requests, role, prefix_rows, completion, sampling_index);
    auto &selected = diagnostic ? probe_families[family_key] : families[family_key];
    if (selected) throw std::runtime_error("text finite family was already prepared");
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.layers.resize(geometry.layers.size()); definition.attention_nodes.resize(target_kv_layers);
    definition.append_nodes.resize(draft_bank + 2);
    definition.sampling = &prepared_sampling;
    definition.requests = requests; definition.role = role; definition.prefix_rows = prefix_rows; definition.completion = completion;
    definition.admitted_exact = admitted_exact;
    definition.state_resource_ports.reserve(ports_per_request * requests);
    if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) {
      definition.carry_ports.resize(requests);
      for (auto &ports : definition.carry_ports) ports.resize(prefix_rows);
    }
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity)
        throw std::runtime_error("finite text chunk outside nonempty context capacity");
      if (std::find(definition.chunks.begin(), definition.chunks.end(), tokens) == definition.chunks.end())
        definition.chunks.push_back(tokens);
    }
    const auto start = Clock::now();
    const auto max_query_rows = *std::max_element(definition.chunks.begin(), definition.chunks.end());
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, max_query_rows, requests, &weight_meanings);
    const auto t = r.token_rows, packed = r.packed_rows;
    const auto batch = f(static_cast<std::int64_t>(requests));
    definition.last_row = r.last_row;
    if (requests > cache_plan.active_sequences || cache::multiply(requests, max_query_rows) > cache_plan.max_query_rows)
      throw std::runtime_error("finite family exceeds the configured batch capacity");
    prepare_cache();
    definition.attention_schema = attention_schema;
    // A sampler produces [B,1]. Preserve that exact external token carrier
    // for ordinary T1 and flatten only the mathematical embedding indices.
    definition.token_port = role == GraphRole::kText
        ? r.input(r.dense_spec(Type::kInt32, {packed, f(1)}))
        : r.input(r.dense_spec(Type::kInt32,
            {has_mtp(role) && !is_mtp_prefill(role) ? batch : packed}));
    definition.position_port = requests == 1 ? r.input(r.dense_spec(Type::kInt32, {t})) :
        r.input(r.dense_spec(Type::kInt32, {batch, t}));
    if (requests != 1 && !verifies_prefix(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    definition.metadata_port = r.input(take(definition.attention_schema.metadata_description()));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    V target_tokens = role == GraphRole::kText
        ? r.reshape(definition.token_port, {packed}) : definition.token_port;
    std::vector<sampling::Distribution> draft_distributions;
    std::vector<V> proposal_ids;
    definition.batched_text_sampling = requests > 1 && role == GraphRole::kText &&
        completion == TextCompletion::kEmitNext && selected_sampler.enabled &&
        selected_sampler.modifiers.empty() && selected_sampler.policy.top_p == 1.F &&
        (!selected_sampler.policy.top_k || selected_sampler.policy.top_k == static_cast<std::size_t>(vocabulary));
    if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify)
      definition.sampling_identity_port = r.input(definition.batched_text_sampling
          ? dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 4})
          : dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4}));
    const auto sampling_identity = [&](std::size_t request, std::size_t draw) {
      return r.copy(r.reshape(r.slice(r.slice(definition.sampling_identity_port, 0,
          f(static_cast<std::int64_t>(request)), f(1)), 1,
          f(static_cast<std::int64_t>(draw)), f(1)), {f(1), f(4)}));
    };
    const auto sampling_row = [&](V values, std::size_t row) {
      return r.copy(r.slice(values, 0, f(static_cast<std::int64_t>(row)), f(1)));
    };
    D draft_sequence_rows = f(0), draft_packed_rows = f(0);
    if (has_mtp(role)) {
      std::vector<V> carries, priors;
      definition.draft_hidden_inputs.resize(requests);
      for (auto &inputs : definition.draft_hidden_inputs) {
        inputs[0] = r.input(dense(precision.activation_type(), {1, hidden}));
        definition.state_resource_ports.push_back(inputs[0]); carries.push_back(inputs[0]);
        if (draft_lag(role)) {
          inputs[1] = r.input(dense(precision.activation_type(), {1, hidden}));
          definition.state_resource_ports.push_back(inputs[1]); priors.push_back(inputs[1]);
        }
      }
      definition.carry_input_port = r.concat(carries, 0, r.dense_spec(precision.activation_type(), {batch, f(hidden)}));
      const auto draft_rows = is_mtp_prefill(role)
          ? (draft_lag(role) ? D(take(r.builder.dimension_sum(t, f(1)))) : t)
          : f(static_cast<std::int64_t>(draft_lag(role) + 1));
      const auto packed_draft_rows = !draft_lag(role) && is_mtp_prefill(role) ? packed : requests == 1 ? draft_rows :
          draft_rows.is_fixed() ? f(static_cast<std::int64_t>(cache::multiply(requests, static_cast<std::size_t>(draft_rows.fixed_value())))) :
          D(take(r.builder.dimension_product(draft_rows, batch)));
      draft_sequence_rows = draft_rows; draft_packed_rows = packed_draft_rows;
      definition.draft_position_port = r.input(requests == 1 ? r.dense_spec(Type::kInt32, {draft_rows})
          : r.dense_spec(Type::kInt32, {batch, draft_rows}));
      if (draft_lag(role)) {
        definition.prior_input_port = r.concat(priors, 0, r.dense_spec(precision.activation_type(), {batch, f(hidden)}));
        definition.catchup_token_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      }
      if (drafts_token(role) || !is_mtp_prefill(role)) {
        V ids = definition.token_port, hidden_rows = definition.carry_input_port;
        if (draft_lag(role)) {
          if (requests == 1) {
            ids = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids}, r.dense_spec(Type::kInt32, {f(2)}));
            hidden_rows = r.one(c::Concat{.axis = 0}, {definition.prior_input_port, hidden_rows}, r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
          } else {
            ids = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.catchup_token_port, {batch, f(1)}), r.reshape(ids, {batch, f(1)})},
                r.dense_spec(Type::kInt32, {batch, f(2)})), {packed_draft_rows});
            hidden_rows = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.prior_input_port, {batch, f(1), f(hidden)}), r.reshape(hidden_rows, {batch, f(1), f(hidden)})},
                r.dense_spec(precision.activation_type(), {batch, f(2), f(hidden)})), {packed_draft_rows, f(hidden)});
          }
        }
        const auto projection = mtp::input_projection(r, ids, hidden_rows);
        const mtp::Geometry row_geometry{requests, draft_rows, packed_draft_rows};
        const auto kv = mtp::project_kv(r, projection, definition.draft_position_port, frequencies, row_geometry);
        mtp::AppendNodes append;
        if (drafts_token(role)) {
          definition.draft_key_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
          definition.draft_value_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
          definition.draft_metadata_port = r.input(take(attention_schema.metadata_description()));
          const auto head = mtp::full_head(r, projection, kv, definition.draft_position_port, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port, definition.draft_metadata_port},
              f(static_cast<std::int64_t>(draft_lag(role))), row_geometry);
          append = head.append_nodes; definition.draft_attention_node = head.attention_node;
          if (!selected_sampler.enabled) definition.draft_id_port = r.op(c::Operation(c::TopK{.count = 1}), {head.logits},
              {r.dense_spec(Type::kFloat32, {batch, f(1)}), r.dense_spec(Type::kInt32, {batch, f(1)})})[1];
          if (selected_sampler.enabled) {
            for (std::size_t request = 0; request < requests; ++request) {
              draft_distributions.push_back(sampling::distribution(r, sampling_row(head.logits, request), selected_sampler.policy));
              proposal_ids.push_back(r.one(c::sampling_ops::categorical(),
                  {draft_distributions.back().log_probabilities, sampling_identity(request, 0)}));
            }
            definition.draft_id_port = r.reshape(r.concat(proposal_ids, 0,
                r.dense_spec(Type::kInt32, {batch})), {batch, f(1)});
          }
          r.publish(definition.draft_id_port);
          target_tokens = requests == 1 ? r.one(c::Concat{.axis = 0},
              {definition.token_port, r.reshape(definition.draft_id_port, {f(1)})}, r.dense_spec(Type::kInt32, {packed})) :
              r.reshape(r.one(c::Concat{.axis = 1},
                  {r.reshape(definition.token_port, {batch, f(1)}), r.reshape(definition.draft_id_port, {batch, f(1)})},
                  r.dense_spec(Type::kInt32, {batch, f(2)})), {packed});
        } else append = mtp::append_kv(r, attention_schema, kv);
        definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      }
    }
    auto x = r.embedding(target_tokens, "model.language_model.embed_tokens.weight");
    // Writer-only draft maintenance has no attention data edge into target.
    // Order its real cache effects before target's first append as well.
    g::NodeId prior_append = has_mtp(role) && !is_mtp_prefill(role)
        ? definition.append_nodes[draft_bank + 1] : g::NodeId{};
    bool has_prior_append = has_mtp(role) && !is_mtp_prefill(role);
    for (std::size_t i = 0; i < state_schema.size(); ++i) {
      const auto prefix = "layers." + std::to_string(i) + ".";
      auto normalized = r.norm(x, prefix + "input_layernorm.weight");
      auto &ports = definition.layers[i];
      V mixed;
      if (geometry.is_gdn(i)) {
        const auto state_port = [&](StatePorts &port, const c::TensorDesc &item) {
          const auto bytes = take(item.required_bytes());
          const auto schema = take(g::slots::Schema::create(item, cache_plan.state_slots,
              cache::add(bytes, cache::state_pool_padding(bytes))));
          port.pool = r.input(take(schema.capacity_description()));
          port.source_ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
          port.destination_ids = r.input(dense(Type::kInt32,
              {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
          auto rows = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {port.pool, port.source_ids}, r.dense_spec(item.data_type(),
                  {batch, f(schema.item_elements())}), &port.gather);
          return std::pair{schema, rows};
        };
        const auto [history_schema, history_rows] = state_port(ports.history, state_schema[i].history);
        const auto [recurrent_schema, recurrent_rows] = state_port(ports.recurrent, state_schema[i].recurrent);
        const auto history = r.reshape(history_rows,
            {batch, f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())});
        const auto state = r.reshape(recurrent_rows,
            {batch, f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)});
        const auto store_state = [&](StatePorts &port, const g::slots::Schema &schema, V rows) {
          r.op(c::Operation::custom(g::slots::Store{schema}), {rows, port.destination_ids}, {}, &port.store);
          if (has_prior_append) check(r.builder.depends_on(port.store, prior_append));
          prior_append = port.store; has_prior_append = true;
          definition.state_sources.push_back({port.store, port.gather});
        };
        auto qkv = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight"), {batch, t, f(geometry.gdn_qkv_width())});
        auto samples = r.one(c::Concat{.axis = 1}, {history, qkv},
            r.dense_spec(precision.activation_type(), {batch, r.sample_rows, f(geometry.gdn_qkv_width())}));
        // K1 records the final boundary; current K>1 families have T==K.
        // The ordinary graph stores raw history before Cast F32. A single
        // window may have a batch gap and therefore needs real materialization.
        std::vector<V> windows;
        windows.reserve(prefix_rows);
        for (std::size_t row = 0; row < prefix_rows; ++row) {
          const auto boundary = prefix_rows == 1 ? t : f(static_cast<std::int64_t>(row + 1));
          auto window = r.slice(samples, 1, boundary, f(geometry.conv_kernel - 1));
          if (prefix_rows == 1) window = r.one(c::Copy{}, {window},
              r.dense_spec(precision.activation_type(),
                  {batch, f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())}));
          else window = r.reshape(window,
              {batch, f(1), f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())});
          windows.push_back(window);
        }
        const auto histories = prefix_rows == 1 ? windows[0] : r.concat(windows, 1,
            r.dense_spec(precision.activation_type(), {batch, f(static_cast<std::int64_t>(prefix_rows)),
                f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())}));
        store_state(ports.history, history_schema, r.reshape(histories,
            {f(static_cast<std::int64_t>(requests * prefix_rows)), f(history_schema.item_elements())}));
        auto convolution = r.one(c::model_ops::causal_conv1d(), {r.cast(samples, Type::kFloat32),
            r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"), {f(geometry.gdn_qkv_width()), f(geometry.conv_kernel)}), Type::kFloat32)},
            r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_qkv_width())}));
        auto activated = r.unary(c::UnaryKind::kSilu, r.cast(convolution, precision.activation_type()));
        auto q = r.reshape(r.slice(activated, 2, f(0), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto k = r.reshape(r.slice(activated, 2, f(geometry.gdn_key_width()), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto v = r.reshape(r.slice(activated, 2, f(2 * geometry.gdn_key_width()), f(geometry.gdn_value_width())), {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto decay = r.reshape(r.decay("model.language_model." + prefix + "linear_attn.A_log"), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto g = r.binary(c::BinaryKind::kMultiply, decay,
            r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
        auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
        auto recurrent = r.op(c::Operation(c::GatedDeltaNet{
            .snapshot_count = static_cast<std::uint32_t>(prefix_rows)}),
            {r.cast(r.gdn_key_rows(r.l2(q)), Type::kFloat32), r.cast(r.gdn_key_rows(r.l2(k)), Type::kFloat32),
             r.cast(v, Type::kFloat32), g, beta, state},
            {r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}),
             prefix_rows > 1
                 ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)})
                 : r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)})});
        store_state(ports.recurrent, recurrent_schema, r.reshape(recurrent[1],
            {f(static_cast<std::int64_t>(requests * prefix_rows)), f(recurrent_schema.item_elements())}));
        auto y = r.cast(recurrent[0], precision.activation_type());
        if (i == 0) definition.probe_ports[0] = y;
        const auto y_f32 = r.cast(y, Type::kFloat32);
        y = r.cast(r.same(c::RmsNorm{.epsilon = geometry.epsilon}, {y_f32}, y_f32), precision.activation_type());
        // Original storage is F32 in the official checkpoint and BF16 in
        // these quantized artifacts. Preserve its actual product rounding.
        const auto norm_scale = r.weight(prefix + "linear_attn.norm.weight");
        y = r.binary(c::BinaryKind::kMultiply, r.cast(y, r.spec(norm_scale).data_type()), norm_scale);
        auto z = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_z.weight"), {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        y = r.cast(r.binary(c::BinaryKind::kMultiply, r.cast(y, Type::kFloat32),
            r.unary(c::UnaryKind::kSilu, r.cast(z, Type::kFloat32))), precision.activation_type());
        auto projection_input = r.reshape(y, {packed, f(geometry.gdn_value_width())});
        if (i == 0) definition.probe_ports[1] = y; // Same BF16 payload before its metadata-only out_proj reshape.
        mixed = r.linear(projection_input, prefix + "linear_attn.out_proj.weight");
        if (i == 0) definition.probe_ports[2] = mixed;
      } else {
        const auto layer = geometry.attention_slot(i);
        const auto &schema = target_attention_schema(layer);
        ports.key = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kKey)));
        ports.value = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kValue)));
        auto projected = r.reshape(r.linear(normalized, prefix + "self_attn.q_proj.weight"), {batch, t, f(geometry.query_heads), f(geometry.key_depth + geometry.value_depth)});
        auto q = r.norm(r.slice(projected, 3, f(0), f(geometry.key_depth)), prefix + "self_attn.q_norm.weight");
        auto gate = r.reshape(r.copy(r.slice(projected, 3, f(geometry.key_depth), f(geometry.value_depth))), {packed, f(geometry.attention_width())});
        auto k = r.norm(r.reshape(r.linear(normalized, prefix + "self_attn.k_proj.weight"), {batch, t, f(geometry.kv_heads), f(geometry.key_depth)}), prefix + "self_attn.k_norm.weight");
        auto v = r.linear(normalized, prefix + "self_attn.v_proj.weight");
        const c::RoPE rope{.rotary_dimensions = static_cast<std::size_t>(geometry.rotary), .sequence_axis = 1, .layout = c::RopeLayout::kHalfSplit};
        q = r.same(rope, {q, definition.position_port, frequencies}, q);
        k = r.same(rope, {k, definition.position_port, frequencies}, k);
        auto &key_append = definition.append_nodes[layer * 2];
        auto &value_append = definition.append_nodes[layer * 2 + 1];
        auto key_rows = r.reshape(k, {packed, f(geometry.kv_heads), f(geometry.key_depth)});
        auto value_rows = r.reshape(v, {packed, f(geometry.kv_heads), f(geometry.value_depth)});
        const auto cache_capacity = schema.capacity();
        if (cache_capacity.key_storage.data_type() == Type::kFloat8E4M3FN)
          key_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.key_storage}),
              {key_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.key_depth)}));
        if (cache_capacity.value_storage.data_type() == Type::kFloat8E4M3FN)
          value_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.value_storage}),
              {value_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.value_depth)}));
        r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kKey))),
            {key_rows}, {}, &key_append);
        r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kValue))),
            {value_rows}, {}, &value_append);
        if (has_prior_append) check(r.builder.depends_on(key_append, prior_append));
        check(r.builder.depends_on(value_append, key_append));
        prior_append = value_append; has_prior_append = true;
        auto attention = r.one(c::Operation::custom(c::PagedAttention{
            .capacity = cache_capacity, .scale = 1.F / std::sqrt(static_cast<float>(geometry.key_depth)), .causal = true}),
            {r.reshape(q, {packed, f(geometry.query_heads), f(geometry.key_depth)}), ports.key, ports.value, definition.metadata_port},
            r.dense_spec(precision.activation_type(), {packed, f(geometry.query_heads), f(geometry.value_depth)}), &definition.attention_nodes[layer]);
        check(r.builder.depends_on(definition.attention_nodes[layer], key_append));
        check(r.builder.depends_on(definition.attention_nodes[layer], value_append));
        auto gated = r.binary(c::BinaryKind::kMultiply, r.reshape(attention, {packed, f(geometry.attention_width())}), r.unary(c::UnaryKind::kSigmoid, gate));
        mixed = r.linear(gated, prefix + "self_attn.o_proj.weight");
      }
      const auto ffn_input = r.binary(c::BinaryKind::kAdd, x, mixed);
      {
        GraphAssembly::Scope scope(assembly, ffn_placement.ffn_on_cpu(i) ? assembly.cpu() : assembly.primary());
        x = r.ffn(ffn_input, prefix, i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
      }
      if (i == 0) definition.probe_ports[4] = x;
    }
    if (diagnostic) for (std::size_t i = 0; i < definition.probe_ports.size(); ++i) {
      r.publish(definition.probe_ports[i]);
      auto description = r.spec(definition.probe_ports[i]);
      if (description.data_type() != precision.activation_type()) throw std::runtime_error("layer0 diagnostic differs from source activation precision");
    }
    // The selected child decides caller-written versus generated result storage.
    const auto normalized_last = r.norm(x, "norm.weight");
    if (is_mtp_prefill(role)) {
      const auto kv = mtp::prefill_kv(r, definition.token_port, normalized_last,
          definition.carry_input_port, definition.prior_input_port, definition.catchup_token_port,
          mtp::Geometry{requests, t, packed}, mtp::Geometry{requests, draft_sequence_rows, draft_packed_rows}, definition.draft_position_port,
          frequencies, draft_lag(role) != 0);
      const auto append = mtp::append_kv(r, attention_schema, kv);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
    }
    auto last = verifies_prefix(role) ? r.copy(normalized_last) :
        (requests == 1 ? r.copy(r.slice(normalized_last, 0, D(r.last_row), f(1))) :
        r.one(c::Gather{}, {normalized_last, definition.last_ids_port},
            r.dense_spec(precision.activation_type(), {batch, f(hidden)})));
    if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) for (std::size_t request = 0; request < requests; ++request)
      for (std::size_t row = 0; row < prefix_rows; ++row) {
        const auto carry = r.copy(r.slice(normalized_last, 0,
            verifies_prefix(role) ? f(static_cast<std::int64_t>(request * prefix_rows + row)) :
                (requests == 1 ? D(r.last_row) : D(take(r.builder.dimension_sum(
                    D(take(r.builder.dimension_product(t, f(static_cast<std::int64_t>(request))))), D(r.last_row))))), f(1)));
        definition.carry_ports[request][row] = carry;
        r.publish(carry); definition.state_resource_ports.push_back(carry);
      }
    definition.hidden_port = last;
    if (completion == TextCompletion::kEmitNext) {
      definition.logits_port = r.scores(last, std::string(geometry.head_role()));
      if ((!selected_sampler.enabled && selected_sampler.modifiers.empty()) || role == GraphRole::kVerify) definition.id_port = r.op(c::Operation(c::TopK{.count = 1}), {definition.logits_port},
          {r.dense_spec(Type::kFloat32, {verifies_prefix(role) ? packed : batch, f(1)}),
           r.dense_spec(Type::kInt32, {verifies_prefix(role) ? packed : batch, f(1)})})[1];
      else if (definition.batched_text_sampling) {
        const auto p = sampling::distribution(r, definition.logits_port, selected_sampler.policy);
        definition.id_port = r.reshape(r.one(c::sampling_ops::categorical(),
            {p.log_probabilities, definition.sampling_identity_port}), {batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.reshape(p.valid, {batch}));
        r.publish(definition.sampling_valid_port);
      } else if (!selected_sampler.modifiers.empty()) {
        std::vector<V> ids, valid;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
          if (inputs.bias) { definition.logits_bias_ports.push_back(*inputs.bias); definition.state_resource_ports.push_back(*inputs.bias); }
          if (inputs.allowed) { definition.logits_allowed_ports.push_back(*inputs.allowed); definition.state_resource_ports.push_back(*inputs.allowed); }
          const auto selection = sampling::ordinary_selection(r, sampling_row(definition.logits_port, request),
              selected_sampler.enabled ? std::optional<V>(sampling_identity(request, 0)) : std::nullopt,
              selected_sampler.policy, inputs);
          ids.push_back(r.reshape(selection.id, {f(1)})); valid.push_back(*selection.valid);
        }
        definition.id_port = r.reshape(r.concat(ids, 0, r.dense_spec(Type::kInt32, {batch})), {batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
      } else {
        std::vector<V> ids, valid, accepted;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto p = sampling::distribution(r, sampling_row(definition.logits_port,
              verifies_prefix(role) ? request * 2 : request), selected_sampler.policy);
          if (drafts_token(role)) {
            const auto bonus = sampling::distribution(r, sampling_row(definition.logits_port, request * 2 + 1), selected_sampler.policy);
            const auto decision = sampling::decision(r, p.log_probabilities,
                draft_distributions[request].log_probabilities, bonus.log_probabilities,
                proposal_ids[request], sampling_identity(request, 1), sampling_identity(request, 2));
            ids.push_back(r.one(c::Select{}, {decision.accepted, proposal_ids[request], decision.final_id}));
            ids.push_back(decision.final_id); accepted.push_back(decision.accepted);
            auto ok = r.one(c::Select{}, {p.valid, draft_distributions[request].valid, r.cast(r.scalar(0), Type::kBool)});
            ok = r.one(c::Select{}, {ok, bonus.valid, r.cast(r.scalar(0), Type::kBool)});
            valid.push_back(r.one(c::Select{}, {ok, decision.valid, r.cast(r.scalar(0), Type::kBool)}));
          } else {
            ids.push_back(r.one(c::sampling_ops::categorical(), {p.log_probabilities, sampling_identity(request, 0)}));
            valid.push_back(p.valid);
          }
        }
        definition.id_port = r.reshape(r.concat(ids, 0, r.dense_spec(Type::kInt32,
            {f(static_cast<std::int64_t>(ids.size()))})), {verifies_prefix(role) ? packed : batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
        if (drafts_token(role)) {
          definition.sampling_accept_port = r.host_predicate(r.concat(accepted, 0, r.dense_spec(Type::kBool, {batch})));
          r.publish(definition.sampling_accept_port);
        }
      }
    }
    r.publish(definition.hidden_port);
    if (completion == TextCompletion::kEmitNext) { r.publish(definition.logits_port); r.publish(definition.id_port); }
    definition.graph = take(std::move(r.builder).finish());
    definition.literals = std::move(assembly.literals);
    definition.specs = std::move(r.specs);
    std::vector<std::array<g::ShapeBinding, 1>> bindings;
    bindings.reserve(definition.chunks.size());
    for (const auto tokens : definition.chunks)
      bindings.push_back({g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(tokens - 1)}});
    std::vector<g::ExactShapeVariant> variants;
    variants.reserve(bindings.size());
    for (const auto &point : bindings) variants.push_back({point});
    g::CompileOptions compile_options{.shape_policy = g::CompileShapePolicy::kFiniteExact,
        .exact_variants = variants, .constants = assembly.constants};
    // The library's cold operation admission cap covers the whole family.
    // Budget the requested finite children explicitly at the per-graph default;
    // this is host search/storage policy, not device or execution capacity.
    const auto per_graph_operations = g::CompileOptions{}.max_operation_proposals;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_operations)
      throw std::runtime_error("finite text operation proposal budget overflows");
    compile_options.max_operation_proposals = per_graph_operations * variants.size();
    const auto per_graph_preparations = g::CompileOptions{}.max_placement_attempts;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_preparations)
      throw std::runtime_error("finite text artifact preparation budget overflows");
    compile_options.max_placement_attempts = per_graph_preparations * variants.size();
    std::cerr << "compile full text finite family requests=" << requests << " variants=" << variants.size()
              << " max_query_rows=" << max_query_rows << " nodes=" << definition.graph.node_count()
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts << '\n';
    compile_options.resource_budget = execution_budget;
    definition.compiled = take(execution.compile(definition.graph, assembly.constants, assembly.options(compile_options)));
    const auto storage = definition.compiled.report.total_compiled_storage();
    std::cerr << "compiled text finite family variants=" << definition.compiled.report.exact_variant_count()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " operation_proposals=" << definition.compiled.report.total_search().operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts
              << " placement_attempts=" << definition.compiled.report.total_search().placement_attempts
              << " cold_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    definition.children.reserve(definition.chunks.size());
    for (std::size_t index = 0; index < definition.chunks.size(); ++index) {
      auto child = std::make_unique<FamilyChild>();
      child->compiled.executable = take(definition.compiled.executable.exact_variant(index));
      child->compiled.report = take(definition.compiled.report.exact_variant(index));
      record_execution_plan(child->compiled.report, assembly.constants);
      report_packed_projections(definition.graph, child->compiled.report, r.packed_projection_nodes,
          child->compiled.executable.workspace_bytes());
      child->outputs.resize(child->compiled.executable.output_count());
      child->output_indices.resize(definition.specs.size());
      check(inference::prepare_outputs(child->compiled, child->outputs, child->output_indices));
      bool concrete_states = true;
      for (const auto &association : definition.state_sources) {
        for (const auto value : {take(definition.graph.node_input(association.gather, 0)),
                 take(definition.graph.node_input(association.gather, 1)),
                 take(definition.graph.result(association.gather, 0)),
                 take(definition.graph.node_input(association.store, 0)),
                 take(definition.graph.node_input(association.store, 1))})
          if (auto description = definition.graph.value_description(value); !description) {
            if (!description.error().IsFailedPrecondition()) check(description.error());
            concrete_states = false;
          }
      }
      if (!concrete_states) {
        auto specialized = take(g::specialize_exact(definition.graph, bindings[index],
            take(execution_budget.metadata_allocator())));
        if (specialized.signature != child->compiled.executable.specialization())
          throw std::runtime_error("state graph specialization differs from its selected child");
        child->state_graph = std::move(specialized.graph);
      }
      for (const auto &ports : definition.carry_ports) for (const auto value : ports) {
        if (!value.valid()) continue;
        const auto &output = child->output(value);
        validate_state_output(output, dense(precision.activation_type(), {1, hidden}));
        hidden_writers = hidden_writers || output.caller_owned();
        generated_state_outputs = generated_state_outputs || !output.caller_owned();
      }
      definition.children.push_back(std::move(child));
    }
    selected = std::move(result);
  }

  Family *find_family(std::size_t requests, std::size_t tokens, GraphRole role,
                      bool diagnostic = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    auto &table = diagnostic ? probe_families : families;
    const auto primary = table.find({requests, role, tokens, completion, sampling_index});
    if (primary != table.end() && primary->second &&
        std::ranges::find(primary->second->chunks, tokens) != primary->second->chunks.end()) return primary->second.get();
    const auto exact = table.find(FamilyKey::exact(requests, role, tokens, completion, sampling_index));
    return exact == table.end() ? nullptr : exact->second.get();
  }

  // A generated-token checkpoint can expose a suffix whose length was unknown
  // at model setup. Specialize only that actual cold request. The same original
  // Family/Phase owners remain bounded by the frozen cold shape plan; decoded
  // hot-loop shapes stay in their original immutable finite families.
  void freeze_shape_plan() {
    if (exact_shape_capacity) return;
    if (!admitted_exact_keys.empty()) throw std::logic_error("freeze the original shape plan before dynamic admission");
    exact_shape_capacity = (std::max)(std::size_t{1}, planned_slot_uses.size());
    std::cerr << "cold_shape_plan planned_shapes=" << planned_slot_uses.size()
              << " extra_capacity=" << exact_shape_capacity << '\n';
  }

  void admit_exact_shape(std::size_t tokens, std::size_t requests, GraphRole role,
                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (find_family(requests, tokens, role, false, completion, sampling_index)) return;
    const auto admission_start = Clock::now();
    if (!state_storage_prepared || active_step || carrier_borrowed || !tokens || tokens > capacity ||
        !requests || requests > cache_plan.active_sequences ||
        cache::multiply(tokens, requests) > cache_plan.max_query_rows || verifies_prefix(role))
      throw std::runtime_error("actual suffix admission exceeds the idle model's original query/state bounds");
    // Direct component consumers may not call the explicit startup boundary.
    // Freeze before the first extra shape can mutate the original inventory.
    freeze_shape_plan();
    const auto limit = exact_shape_capacity;
    if (admitted_exact_keys.capacity() < limit) admitted_exact_keys.reserve(limit);
    if (admitted_exact_keys.size() == limit) {
      const auto victim = admitted_exact_keys.front();
      auto old = families.find(victim);
      if (old == families.end() || !old->second->admitted_exact || old->second->chunks.size() != 1)
        throw std::runtime_error("exact family inventory is inconsistent");
      const ExecutionShape shape{victim.requests, old->second->chunks[0], victim.role, victim.completion, victim.sampling_index};
      for (const bool imported : {false, true}) {
        auto physical = shape; physical.imported_sources = imported;
        if (auto phase = phases.find(physical); phase != phases.end()) {
          for (auto &run : std::span(phase->second->slots).first(phase->second->slot_count)) check(run.drain());
          phases.erase(phase);
        }
      }
      planned_slot_uses.erase(shape); families.erase(old);
      admitted_exact_keys.erase(admitted_exact_keys.begin()); ++exact_evictions;
    }
    const auto before_hidden = hidden_writers, before_generated = generated_state_outputs;
    const auto key = FamilyKey::exact(requests, role, tokens, completion, sampling_index);
    try {
      const std::array chunk{tokens}; prepare_family(chunk, false, requests, role, true, completion, sampling_index);
      if (!before_hidden && hidden_writers)
        throw std::runtime_error("new suffix requires caller-written hidden storage absent from the original cache plan; reconfigure the idle model");
      planned_slot_uses[{requests, tokens, role, completion, sampling_index}].fill(1);
      admitted_exact_keys.push_back(key); ++exact_admissions;
      std::cerr << "cold_exact_admission requests=" << requests << " T=" << tokens
                << " role=" << static_cast<int>(role) << " retained=" << admitted_exact_keys.size()
                << " limit=" << limit
                << " seconds=" << std::chrono::duration<double>(Clock::now() - admission_start).count() << '\n';
    } catch (...) {
      families.erase(key);
      hidden_writers = before_hidden; generated_state_outputs = before_generated;
      throw;
    }
  }

  void prepare_draft_program(std::size_t rows, bool head) {
    preparation_started_ = true;
    if (state_storage_prepared || !cache_plan.draft_cache || !rows || rows > capacity)
      throw std::runtime_error("prepare a positive NextN width before selected cache storage");
    const auto key = std::pair{rows, head};
    if (draft_programs.contains(key)) return;
    prepare_cache();
    auto prepared = std::make_unique<DraftProgram>(); auto &program = *prepared;
    program.rows = rows; program.head = head;
    // No reliable auxiliary-use hint: Auto retains the original Commands path.
    program.replay_mode = select_replay_mode(replay_policy, 1, default_sampler.enabled);
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, rows, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(rows));
    program.tokens = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.positions = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    for (std::size_t row = 0; row < rows; ++row) {
      program.hidden_inputs.push_back(r.input(dense(precision.activation_type(), {1, hidden})));
      program.resource_ports.push_back(program.hidden_inputs.back());
    }
    const auto shifted = r.concat(program.hidden_inputs, 0, r.dense_spec(precision.activation_type(), {count, f(hidden)}));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    const mtp::Geometry row_geometry{1, count, count};
    const auto projection = mtp::input_projection(r, program.tokens, shifted);
    const auto kv = mtp::project_kv(r, projection, program.positions, frequencies, row_geometry);
    if (head) {
      program.key = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
      program.value = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
      program.metadata = r.input(take(attention_schema.metadata_description()));
      const auto result = mtp::full_head(r, projection, kv, program.positions, frequencies,
          {attention_schema, program.key, program.value, program.metadata}, f(static_cast<std::int64_t>(rows - 1)), row_geometry);
      program.append = result.append_nodes; program.attention = result.attention_node;
      program.hidden = result.selected_hidden;
      if (!default_sampler.enabled) program.id = r.op(c::Operation(c::TopK{.count = 1}), {result.logits},
          {r.dense_spec(Type::kFloat32, {f(1), f(1)}), r.dense_spec(Type::kInt32, {f(1), f(1)})})[1];
      if (default_sampler.enabled || min_draft_probability() > 0) {
        const auto distribution = sampling::distribution(r, result.logits,
            default_sampler.enabled ? default_sampler.policy : sampling::Policy{});
        program.log_probabilities = distribution.log_probabilities; program.valid = r.host_predicate(distribution.valid);
        if (default_sampler.enabled) {
          program.identity = r.input(dense(Type::kUInt32, {1, 4}));
          program.id = r.one(c::sampling_ops::categorical(), {program.log_probabilities, program.identity});
          r.publish(program.log_probabilities);
        }
        if (min_draft_probability() > 0) {
          const auto maximum = r.copy(r.reshape(r.one(c::Reduce{.kind = c::ReductionKind::kMaximum,
              .accumulation_type = Type::kFloat32}, {program.log_probabilities}), {f(1)}));
          program.probability = r.unary(c::UnaryKind::kExp, maximum); r.publish(program.probability);
        }
        r.publish(program.valid);
      }
      r.publish(program.hidden); r.publish(program.id);
    } else program.append = mtp::append_kv(r, attention_schema, kv);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, static_cast<std::int64_t>(rows - 1)}};
    const std::array variants{g::ExactShapeVariant{binding}};
    program.compiled = take(execution.compile(program.graph, assembly.constants,
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants, .resource_budget = execution_budget})));
    program.compiled.executable = take(program.compiled.executable.exact_variant(0));
    program.compiled.report = take(program.compiled.report.exact_variant(0));
    program.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), program.compiled.report));
    record_execution_plan(program.compiled.report, assembly.constants);
    program.literals = std::move(assembly.literals);
    program.token_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.position_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    if (head) {
      program.hidden_output = take(inference::prepare_output(program.compiled, program.hidden));
      validate_state_output(program.hidden_output, dense(precision.activation_type(), {1, hidden}));
      if (default_sampler.enabled) program.probabilities_output =
          take(inference::prepare_output(program.compiled, program.log_probabilities));
    }
    for (std::size_t output = 0; output < program.compiled.executable.output_count(); ++output) {
      if (take(program.compiled.executable.output_ownership(output)) != g::OutputOwnership::kCaller) continue;
      const auto port = take(program.compiled.executable.output_id(output));
      if (port == program.hidden) {
        hidden_writers = true; program.resource_ports.push_back(port);
      } else if (port == program.log_probabilities) {
        program.resource_ports.push_back(port);
        const auto bytes = cache::multiply(vocabulary * sizeof(float), cache_plan.candidate_boundaries);
        program.probability_pool = take(c::LeasePool::create(
            take(execution.resource_for(program.probabilities_output.storage)), cache::add(bytes, 1),
            {.max_blocks = cache_plan.candidate_boundaries, .resource_budget = execution_budget}));
      }
    }
    if (head) {
      program.id_value = take(PreparedReadback::prepare(program.compiled, program.id, execution, allocation_budget));
      if (default_sampler.enabled || min_draft_probability() > 0)
        program.valid_value = take(PreparedReadback::prepare(program.compiled, program.valid, execution, allocation_budget));
      if (min_draft_probability() > 0)
        program.probability_value = take(PreparedReadback::prepare(program.compiled, program.probability, execution, allocation_budget));
    }
    initialize_draft_program(program);
    draft_programs.emplace(key, std::move(prepared));
  }

  void initialize_draft_program(DraftProgram &program) {
    program.run = take(program.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(program.run, *scheduler, *workers));
    check(program.run.bind(program.tokens, program.token_values.read()));
    check(program.run.bind(program.positions, program.position_values.read()));
    if (program.head) check(program.id_value.bind(program.run));
    if (program.head && default_sampler.enabled) {
      if (!program.identity_values.caller_owned()) program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
      check(program.run.bind(program.identity, program.identity_values.read()));
    }
    if (program.head && (default_sampler.enabled || min_draft_probability() > 0)) check(program.valid_value.bind(program.run));
    if (program.head && min_draft_probability() > 0) check(program.probability_value.bind(program.run));
  }

  // Append to an unpublished draft only. The first append uses COW; later
  // recursive rows may extend that private tail. Published Session tails are
  // never borrowed, so aborting proposal/verification leaves the old head valid.
  DraftProposal execute_draft(DraftWork &work, std::span<const std::int32_t> ids,
      std::span<const c::ReadLease> shifted, bool head, const std::array<std::uint32_t, 4> &identity = {}) {
    if (ids.empty() || ids.size() != shifted.size() || !state_storage_prepared ||
        work.state.frontier > capacity || ids.size() > capacity - work.state.frontier)
      throw std::runtime_error("NextN input rows exceed the prepared context or hidden geometry");
    auto found = draft_programs.find({ids.size(), head});
    if (found == draft_programs.end()) throw std::runtime_error("NextN finite program was not cold prepared");
    auto &program = *found->second;
    if (!program.run) initialize_draft_program(program);
    auto &run = program.run;
    StepState guard; guard.configure(draft_bank + 2); guard.run = &run;
    guard.nodes[draft_bank] = program.append.key; guard.nodes[draft_bank + 1] = program.append.value;
    cache::DraftState next;
    DraftProposal output;
    try {
      const auto length = work.state.frontier;
      next.frontier = cache::add(length, ids.size());
      const auto prefix_pages = length / cache::Plan::page_rows;
      const auto page_count = (next.frontier + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      const auto dirty_count = page_count - prefix_pages;
      const auto tail_rows = length % cache::Plan::page_rows;
      bool reuse = tail_rows && !work.state.pages.empty();
      if (reuse) for (const auto &tail : work.tails)
        reuse = reuse && tail.valid() && !tail.reserved_rows() && tail.initialized_rows() == tail_rows &&
            tail.page() == work.state.pages.back().page();
      if (reuse) {
        const auto rows = static_cast<std::uint32_t>((std::min)(ids.size(), cache::Plan::page_rows - tail_rows));
        std::size_t reserved = 0; hyle::Status status;
        for (; reserved < work.tails.size(); ++reserved) {
          status = work.tails[reserved].reserve_rows(rows); if (!status.ok()) break;
        }
        if (!status.ok()) {
          for (std::size_t i = 0; i < reserved; ++i) check(work.tails[i].rollback());
          if (!status.IsUnavailable()) check(status);
          reuse = false;
        }
      }
      next.pages.reserve(page_count);
      for (std::size_t i = 0; i < prefix_pages; ++i) next.pages.push_back(work.state.pages.at(i));
      if (reuse) {
        next.pages.push_back(work.state.pages.back());
        for (std::size_t bank = 0; bank < 2; ++bank) guard.tails[draft_bank + bank].push_back(std::move(work.tails[bank]));
      } else work.tails = {};
      for (std::size_t i = reuse ? 1 : 0; i < dirty_count; ++i) {
        auto page = take(draft_inventory.acquire()); next.pages.push_back(std::move(page.reference));
        guard.dirty[draft_bank].push_back(std::move(page.keys[0])); guard.dirty[draft_bank + 1].push_back(std::move(page.values[0]));
      }
      std::vector<std::int32_t> page_ids, positions;
      for (const auto &page : next.pages) page_ids.push_back(static_cast<std::int32_t>(page.page()));
      for (std::size_t i = 0; i < ids.size(); ++i) positions.push_back(static_cast<std::int32_t>(length + i));
      const std::array<std::int32_t, 2> query{0, static_cast<std::int32_t>(ids.size())},
          kv{0, static_cast<std::int32_t>(page_ids.size())};
      const std::array last{static_cast<std::int32_t>((next.frontier - 1) % cache::Plan::page_rows + 1)};
      const auto layout = take(attention_schema.version({query, kv, page_ids, last}));
      const std::array sources{take(work.state.key.view()), take(work.state.value.view())};
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        auto &arena = bank == draft_bank ? draft_inventory.keys[0] : draft_inventory.values[0];
        const auto &source = sources[bank - draft_bank];
        guard.appends[bank] = take(g::batch::PagedAppendPages::create(arena, layout,
            source.layout(), std::span(&source, 1), guard.dirty[bank], guard.tails[bank],
            {.resource_budget = cache_budget}));
      }
      program.token_values.write<std::int32_t>(ids); program.position_values.write<std::int32_t>(positions);
      for (std::size_t row = 0; row < ids.size(); ++row)
        check(run.bind(program.hidden_inputs[row], dense(precision.activation_type(), {1, hidden}), shifted[row]));
      if (program.hidden_output.value && program.hidden_output.caller_owned())
        check(run.bind(program.hidden_output.value, program.hidden_output.description,
            take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment))));
      if (head && default_sampler.enabled) {
        program.identity_values.write<std::uint32_t>(identity);
        if (program.probabilities_output.caller_owned()) check(run.bind(program.probabilities_output.value,
            program.probabilities_output.description, take(program.probability_pool.allocate(vocabulary * sizeof(float)))));
      }
      const std::array layouts{g::batch::NodeLayout{program.attention, layout}};
      const auto queues = inference::queue_span(program.queues);
      check(g::batch::prepare(run, queues, head ? std::span<const g::batch::NodeLayout>(layouts) : std::span<const g::batch::NodeLayout>{},
          {.host_workspace = execution.staging()}));
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        if (head) check(g::batch::bind_append(run, program.graph, guard.nodes[bank],
            bank == draft_bank ? program.key : program.value, std::move(guard.appends[bank])));
        else check(g::batch::bind_append(run, program.graph, guard.nodes[bank], std::move(guard.appends[bank])));
        guard.bound[bank] = true;
      }
      check(run.prepare_replay(queues, {.mode = program.replay_mode, .host_wait = g::HostWaitPolicy::kAllow}));
      guard.completion = submit_retained(run, queues, program.retirement); check(guard.completion.wait()); guard.completion = {};
      ++draft_program_graph_calls;
      if (head) {
        if (default_sampler.enabled || min_draft_probability() > 0) {
          const auto flag = take(program.valid_value.predicate(execution, run));
          if (flag != 1) throw std::runtime_error("recursive draft distribution is invalid; no head or RNG was published");
          if (default_sampler.enabled) output.log_probabilities =
              take(inference::retain_output(run, program.probabilities_output, execution_budget));
          if (min_draft_probability() > 0) {
            check(program.probability_value.read(execution, run, std::as_writable_bytes(std::span(&output.probability, 1))));
            if (!std::isfinite(output.probability) || output.probability < 0 || output.probability > 1)
              throw std::runtime_error("draft maximum probability is outside its declared probability domain");
          }
        }
        check(program.id_value.read(execution, run, std::as_writable_bytes(std::span(&output.id, 1))));
        if (output.id < 0 || output.id >= vocabulary) throw std::runtime_error("NextN proposal is outside vocabulary");
        output.hidden = retain_state_storage(run, program.hidden_output);
      }
      check(g::batch::take_appends(run, std::span(guard.nodes.data() + draft_bank, 2),
          std::span(guard.appends.data() + draft_bank, 2)));
      std::fill_n(guard.bound.begin() + draft_bank, 2, std::uint8_t{});
      auto key_delta = take(guard.appends[draft_bank].ready_delta(sources[0], layout));
      auto value_delta = take(guard.appends[draft_bank + 1].ready_delta(sources[1], layout));
      auto key_commit = take(work.state.key.prepare_commit(std::move(key_delta)));
      auto value_commit = take(work.state.value.prepare_commit(std::move(value_delta)));
      std::array<c::PagedPageAppender, 2> tails;
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) for (std::size_t i = 0; i < guard.appends[bank].appender_count(); ++i) {
        auto &tail = take(guard.appends[bank].appender(i)); if (!tail.valid()) continue;
        if (tail.page() != next.pages.back().page() || tail.reserved_rows() || tails[bank - draft_bank].valid() ||
            tail.initialized_rows() != next.frontier % cache::Plan::page_rows)
          throw std::runtime_error("NextN append returned an invalid private tail");
        tails[bank - draft_bank] = std::move(tail);
      }
      check(run.unbind(program.resource_ports));
      check(work.state.key.validate_commit(key_commit));
      check(work.state.value.validate_commit(value_commit));
      next.key = std::move(work.state.key); next.value = std::move(work.state.value);
      const auto key_status = next.key.commit(std::move(key_commit));
      const auto value_status = next.value.commit(std::move(value_commit));
      HYLE_FATAL_ASSERT(key_status.ok() && value_status.ok());
      work.tails = std::move(tails); work.state = std::move(next);
      return output;
    } catch (...) {
      // An unsuccessful private append is never a partially published head.
      run.close().IgnoreError(); execution.drain_retired(run).IgnoreError();
      auto recovered = guard.try_abort();
      if (!recovered || !*recovered) run = {};
      throw;
    }
  }

  void prepare_prefix_sampling(std::size_t proposals) {
    preparation_started_ = true;
    if (!default_sampler.enabled || state_storage_prepared || proposals >= cache_plan.candidate_boundaries)
      throw std::runtime_error("prepare bounded speculative-sampling widths before cache storage");
    if (prefix_sampling_programs.contains(proposals)) return;
    auto prepared = std::make_unique<PrefixSamplingProgram>(); auto &program = *prepared;
    program.proposals = proposals;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(proposals + 1));
    program.logits = r.input(r.dense_spec(Type::kFloat32, {count, f(vocabulary)}));
    program.identities = r.input(r.dense_spec(Type::kUInt32, {count, f(4)}));
    if (proposals) program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    auto assembled = sampling::speculative_prefix(r, program.logits, program.identities,
        program.ids, default_sampler.policy);
    program.q = std::move(assembled.q); program.accepted = std::move(assembled.accepted);
    program.final_ids = std::move(assembled.final_ids); program.valid = std::move(assembled.valid);
    for (std::size_t row = 0; row < program.final_ids.size(); ++row) {
      if (proposals) {
        program.accepted[row] = r.host_predicate(program.accepted[row]);
        r.publish(program.accepted[row]);
      }
      program.valid[row] = r.host_predicate(program.valid[row]);
      r.publish(program.final_ids[row]); r.publish(program.valid[row]);
    }
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, 0}};
    const std::array variants{g::ExactShapeVariant{binding}};
    program.compiled = take(execution.compile(program.graph, assembly.constants,
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants, .resource_budget = execution_budget})));
    program.compiled.executable = take(program.compiled.executable.exact_variant(0));
    program.compiled.report = take(program.compiled.report.exact_variant(0));
    program.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), program.compiled.report));
    record_execution_plan(program.compiled.report, assembly.constants);
    program.literals = std::move(assembly.literals);
    program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {static_cast<std::int64_t>(proposals + 1), 4}));
    input_scratch.sampling_identities.reserve(take(program.identity_values.description.required_bytes()) / sizeof(std::uint32_t));
    if (proposals) program.id_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    const auto prepare = [&](const std::vector<V> &ports, std::vector<PreparedReadback> &values) {
      values.reserve(ports.size());
      for (const auto port : ports) values.push_back(take(PreparedReadback::prepare(program.compiled, port, execution, allocation_budget)));
    };
    if (proposals) prepare(program.accepted, program.accepted_values);
    prepare(program.final_ids, program.final_values); prepare(program.valid, program.valid_values);
    program.resource_ports = program.q; program.resource_ports.push_back(program.logits);
    initialize_prefix_sampling(program);
    prefix_sampling_programs.emplace(proposals, std::move(prepared));
  }

  void initialize_prefix_sampling(PrefixSamplingProgram &program) {
    program.run = take(program.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(program.run, *scheduler, *workers));
    check(program.run.bind(program.identities, program.identity_values.read()));
    if (program.proposals) check(program.run.bind(program.ids, program.id_values.read()));
    for (auto &output : program.accepted_values) check(output.bind(program.run));
    for (auto &output : program.final_values) check(output.bind(program.run));
    for (auto &output : program.valid_values) check(output.bind(program.run));
  }

  // Target family widths and replay counts stay with the finite request plan;
  // every recursive consumer shares this model's actual NextN/sampler programs.
  void prepare_recursive_round_programs() {
    for (std::size_t rows = 1; rows <= cache_plan.candidate_boundaries; ++rows) {
      prepare_draft_program(rows, false);
      if (default_sampler.enabled && (rows > 1 || min_draft_probability() > 0)) prepare_prefix_sampling(rows - 1);
    }
    prepare_draft_program(1, true);
    if (capacity > 1) prepare_draft_program(2, true);
  }

  PrefixSamplingResult sample_recursive_prefix(c::ConstTensorView logits,
      std::span<const std::int32_t> proposals, std::span<const c::ReadLease> q,
      cache::SamplingCursor cursor) {
    if (!default_sampler.enabled || proposals.size() != q.size())
      throw std::runtime_error("speculative sampling requires every actual proposal distribution");
    auto found = prefix_sampling_programs.find(proposals.size());
    if (found == prefix_sampling_programs.end()) throw std::runtime_error("speculative-sampling width was not cold prepared");
    auto &program = *found->second;
    if (!program.run) initialize_prefix_sampling(program);
    auto &run = program.run;
    const auto count = proposals.size();
    (void)cursor.advanced(cache::add(cache::multiply(2, count), 1));
    auto &identities = input_scratch.sampling_identities;
    identities.clear();
    for (std::size_t row = 0; row <= count; ++row) {
      const auto identity = cursor.identity(row == count ? 2 * count : 2 * row + 1);
      identities.insert(identities.end(), identity.begin(), identity.end());
    }
    program.identity_values.write<std::uint32_t>(identities);
    if (count) program.id_values.write<std::int32_t>(proposals);
    try {
    check(run.bind(program.logits, logits));
    for (std::size_t row = 0; row < count; ++row)
      check(run.bind(program.q[row], dense(Type::kFloat32, {1, vocabulary}), q[row]));
    const auto queues = inference::queue_span(program.queues);
    check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
    auto done = submit_retained(run, queues, program.retirement); check(done.wait());
    ++prefix_decision_graph_calls;
    PrefixSamplingResult result; result.accepted.resize(count); result.final_ids.resize(program.final_ids.size());
    for (std::size_t row = 0; row < program.final_ids.size(); ++row) {
      const auto valid = take(program.valid_values[row].predicate(execution, run));
      if (valid != 1) throw std::runtime_error("recursive target/q/residual distribution is invalid; no head or RNG was published");
      if (count) result.accepted[row] = take(program.accepted_values[row].predicate(execution, run));
      check(program.final_values[row].read(execution, run, std::as_writable_bytes(std::span(&result.final_ids[row], 1))));
    }
    check(run.unbind(program.resource_ports));
    return result;
    } catch (...) {
      // Failed p/q validation or native work must release the same transient
      // input pins as success. A later explicit retry recreates only this
      // original finite invocation, never a Session or an RNG cursor.
      run.close().IgnoreError(); execution.drain_retired(run).IgnoreError(); run = {};
      throw;
    }
  }

  static c::PagedArenaFrame working_frame(c::PagedArenaRead read) {
    return take(c::PagedArenaFrame::fork(take(c::PagedArenaFootprint::from_read(read))));
  }
  static c::PagedArenaFrame snapshot_alias(const c::PagedArenaFrame &frame) {
    return take(c::PagedArenaFrame::from_frozen(take(frame.snapshot())));
  }
  static cache::DraftState fork_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.pages = source.pages; next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    next.key = take(c::PagedArenaFrame::fork(take(source.key.view())));
    next.value = take(c::PagedArenaFrame::fork(take(source.value.view())));
    return next;
  }
  static cache::DraftState alias_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.pages = source.pages; next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    next.key = snapshot_alias(source.key); next.value = snapshot_alias(source.value);
    return next;
  }

  std::unique_ptr<cache::State> reserve_state(cache::WorkingState &owner) {
    auto version = take(cache::VersionRef::acquire(snapshot_versions));
    auto result = owner.acquire_candidate();
    result->resource_owner = resources_;
    if (!result->keys.try_resize(target_kv_layers) || !result->values.try_resize(target_kv_layers) ||
        !result->state.try_resize(geometry.layers.size())) throw std::bad_alloc{};
    result->version = std::move(version); result->model = this; result->math = math_identity();
    return result;
  }

  // A request's pending token/cursor is not part of the reusable model state.
  // Wrapper origins are canonical so aliases share exactly one finite version;
  // mutable page-tail capabilities stay with the original Session only.
  std::shared_ptr<cache::State> sampling_fork(cache::SnapshotRef base,
      cache::SamplingCursor cursor, std::int32_t pending) const {
    if (!base || base->model != this || base->math != math_identity())
      throw std::runtime_error("sampling fork belongs to another model or compiled policy");
    auto next = std::make_shared<cache::State>();
    next->state_origin = base->state_origin ? base->state_origin : base;
    next->pages = base->pages;
    if (!next->keys.try_resize(base->keys.size()) || !next->values.try_resize(base->values.size()))
      throw std::bad_alloc{};
    for (std::size_t i = 0; i < base->keys.size(); ++i) {
      next->keys[i] = snapshot_alias(base->keys[i]); next->values[i] = snapshot_alias(base->values[i]);
    }
    next->state = base->state; next->carry_hidden = base->carry_hidden;
    next->prior_hidden = base->prior_hidden;
    if (base->draft) next->draft.emplace(alias_draft(*base->draft));
    next->model = base->model; next->math = base->math; next->length = base->length;
    next->next_id = pending; next->sampling = cursor; return next;
  }

  void prepare_sampling_frontier() {
    preparation_started_ = true;
    for (auto &entry : resources_->sampling_configurations) {
      const auto &selected_sampler = entry->configuration;
      if (!selected_sampler.enabled && selected_sampler.modifiers.empty() && sampling_configuration_count() == 1) continue;
      if (!entry->frontier) {
        auto prepared = std::make_unique<SamplingFrontier>();
        GraphAssembly assembly(execution, precision, allocation_budget);
        Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
        prepared->last_row = r.last_row;
        prepared->hidden_port = r.input(dense(precision.activation_type(), {1, hidden}));
        if (selected_sampler.enabled) prepared->identity_port = r.input(dense(Type::kUInt32, {1, 4}));
        const auto inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
        if (inputs.bias) prepared->bias_port = *inputs.bias;
        if (inputs.allowed) prepared->allowed_port = *inputs.allowed;
        const auto selection = sampling::ordinary_selection(r,
            r.scores(prepared->hidden_port, std::string(geometry.head_role())),
            selected_sampler.enabled ? std::optional<V>(prepared->identity_port) : std::nullopt,
            selected_sampler.policy, inputs);
        prepared->id_port = selection.id;
        if (selection.valid) prepared->valid_port = r.host_predicate(*selection.valid);
        r.publish(prepared->id_port); if (prepared->valid_port) r.publish(prepared->valid_port);
        prepared->graph = take(std::move(r.builder).finish());
        // The frontier is a fixed [1,H] projection with no symbolic model rows.
        const std::array shape{g::ShapeBinding{r.last_row, 0}};
        const std::array variants{g::ExactShapeVariant{shape}};
        prepared->compiled = take(execution.compile(prepared->graph, assembly.constants,
            assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
             .constants = assembly.constants, .resource_budget = execution_budget})));
        prepared->compiled.executable = take(prepared->compiled.executable.exact_variant(0));
        prepared->compiled.report = take(prepared->compiled.report.exact_variant(0));
        prepared->queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), prepared->compiled.report));
        record_execution_plan(prepared->compiled.report, assembly.constants);
        prepared->literals = std::move(assembly.literals);
        if (selected_sampler.enabled) prepared->identity = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
        prepared->id = take(PreparedReadback::prepare(prepared->compiled, prepared->id_port, execution, allocation_budget));
        if (prepared->valid_port) prepared->valid = take(PreparedReadback::prepare(prepared->compiled, prepared->valid_port, execution, allocation_budget));
        entry->frontier = std::move(prepared);
      }
    }
  }

  cache::SnapshotRef sample_prefix(cache::SnapshotRef base, cache::SamplingCursor cursor, std::size_t sampling_index = 0,
                                   const LogitsInputs &inputs = {}) {
    const auto &selected = sampling_configuration(sampling_index);
    const auto &selected_sampler = selected.configuration;
    validate_logits_inputs(selected_sampler, inputs, vocabulary);
    if (!base || !base->length || !base->carry_hidden.valid() || active_step || carrier_borrowed || !selected.frontier)
      throw std::runtime_error("prefix sampling requires its cold-prepared graph and an idle complete model-state hidden frontier");
    const auto next_cursor = selected_sampler.enabled ? cursor.advanced(1) : cache::SamplingCursor{};
    const auto started = Clock::now();
    auto &frontier = *selected.frontier;
    if (frontier.retirement.pending())
      throw std::runtime_error("original sampling frontier resource scope is still retained");
    // This is the same Graph execution API/runtime as target/draft. It reads
    // the original immutable hidden lease and returns only two scalar results.
    auto run = take(frontier.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(run, *scheduler, *workers));
    if (selected_sampler.enabled) { const auto identity = cursor.identity(); frontier.identity.write<std::uint32_t>(identity); }
    check(run.bind(frontier.hidden_port, dense(precision.activation_type(), {1, hidden}), base->carry_hidden));
    if (selected_sampler.enabled) check(run.bind(frontier.identity_port, frontier.identity.read()));
    if (frontier.bias_port) check(run.bind(frontier.bias_port, inputs.bias.description, inputs.bias.lease));
    if (frontier.allowed_port) check(run.bind(frontier.allowed_port, inputs.allowed.description, inputs.allowed.lease));
    check(frontier.id.bind(run)); if (frontier.valid_port) check(frontier.valid.bind(run));
    const auto queues = inference::queue_span(frontier.queues);
    check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
    auto done = submit_retained(run, queues, frontier.retirement); check(done.wait());
    ++frontier_sampling_graph_calls;
    const auto valid = frontier.valid_port ? take(frontier.valid.predicate(execution, run)) : std::uint8_t{1};
    std::int32_t pending = -1;
    check(frontier.id.read(execution, run, std::as_writable_bytes(std::span(&pending, 1))));
    if (valid != 1 || pending < 0 || pending >= vocabulary)
      throw std::runtime_error("cached frontier sampling is invalid; no request RNG cursor was published");
    auto next = sampling_fork(std::move(base), next_cursor, pending);
    frontier_sampling_seconds += std::chrono::duration<double>(Clock::now() - started).count();
    return next;
  }

  const void *math_identity() const noexcept {
    // Every family uses this Model's source-derived score boundary. These identities also
    // separate probe math; no prefix can cross a Model or its compiled policy.
    return layer0_probe.empty() ? static_cast<const void *>(&families) : static_cast<const void *>(&probe_families);
  }

  // Aggregate ordered call plans before any execution carriers are created.
  // Prefix adoption can skip full-hit calls. Counts describe actual possible
  // uses of the serial slot, not additional execution carriers.
  void plan_execution(std::span<const std::size_t> calls, std::size_t requests = 1, GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    for (const auto tokens : calls) {
      if (!families.contains({requests, role, tokens, completion, sampling_index}) || std::find(families.at({requests, role, tokens, completion, sampling_index})->chunks.begin(), families.at({requests, role, tokens, completion, sampling_index})->chunks.end(), tokens) == families.at({requests, role, tokens, completion, sampling_index})->chunks.end())
        throw std::runtime_error("execution plan chunk is outside the shared finite family");
      auto &uses = planned_slot_uses[{requests, tokens, role, completion, sampling_index}];
      uses[slot_index] = cache::add(uses[slot_index], 1);
    }
  }

  // Finite wave/suffix plans can count many possible restores without
  // materializing repeated host call tapes or execution carriers.
  void plan_replay_uses(std::size_t tokens, std::size_t requests, std::size_t count,
                        GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    if (!families.contains({requests, role, tokens, completion, sampling_index}) ||
        std::find(families.at({requests, role, tokens, completion, sampling_index})->chunks.begin(), families.at({requests, role, tokens, completion, sampling_index})->chunks.end(), tokens) == families.at({requests, role, tokens, completion, sampling_index})->chunks.end())
      throw std::runtime_error("execution use shape is outside the shared finite family");
    for (auto &uses : planned_slot_uses[{requests, tokens, role, completion, sampling_index}]) uses = cache::add(uses, count);
  }

  std::shared_ptr<Session> create_session(std::span<const std::size_t> planned_chunks,
                                        cache::SnapshotRef prefix = {},
                                        GraphRole role = GraphRole::kText,
                                        std::optional<cache::SamplingCursor> cursor = {},
                                        bool resample_frontier = false,
                                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                                         const LogitsInputs &logits = {}) {
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && (sampling_index || !selected_sampler.modifiers.empty()))
      throw std::runtime_error("nondefault sampling is ordinary Text only");
    if (resample_frontier) validate_logits_inputs(selected_sampler, logits, vocabulary);
    prepare_state_storage();
    const auto initial_rows = planned_chunks.empty() ? 0 : planned_chunks.front();
    for (const auto rows : planned_chunks)
      if (!find_family(1, rows, role, false, completion, sampling_index)) admit_exact_shape(rows, 1, role, completion, sampling_index);
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("session must use its complete target/draft cache family");
    if ((planned_chunks.empty() ? !families.contains({1, role, 0, completion, sampling_index}) : !find_family(1, initial_rows, role, false, completion, sampling_index)) ||
        (!layer0_probe.empty() && !probe_families.contains({1, role, initial_rows, completion, sampling_index})))
      throw std::runtime_error("session requires the shared text families first");
    if (planned_chunks.empty() && !prefix) throw std::runtime_error("session requires an ordered call plan or a complete prefix");
    if (prefix && (prefix->model != this || prefix->math != math_identity() || prefix->length > capacity ||
        (planned_chunks.empty() && (!prefix->length || prefix->next_id < 0))))
      throw std::runtime_error("prefix belongs to another model or context");
    if (prefix) validate_mtp_snapshot(*prefix);
    auto session = std::shared_ptr<Session>(new Session(*this, resources_, prepared_sampling));
    std::erase_if(session_links, [](const auto &weak) { return weak.expired(); });
    session_links.push_back(session);
    if (!session->tails.try_resize(draft_bank)) throw std::bad_alloc{};
    std::size_t total = prefix ? prefix->length : 0;
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity - total)
        throw std::runtime_error("session plan exceeds nonempty context capacity");
      for (const auto *selected : {find_family(1, tokens, role, false, completion, sampling_index), layer0_probe.empty() ? nullptr : probe_families.at({1, role, tokens, completion, sampling_index}).get()}) {
        if (selected && std::find(selected->chunks.begin(), selected->chunks.end(), tokens) == selected->chunks.end())
          throw std::runtime_error("session chunk is outside the shared finite family");
      }
      total += tokens;
      if (!planned_slot_uses.contains({1, tokens, role, completion, sampling_index}))
        throw std::runtime_error("session chunk has no model execution use plan");
    }
    if (prefix) {
      if (resample_frontier && (selected_sampler.enabled || !selected_sampler.modifiers.empty() || sampling_configuration_count() > 1))
        prefix = sample_prefix(std::move(prefix), cursor.value_or(cache::SamplingCursor{}), sampling_index, logits);
      else if ((selected_sampler.enabled || !selected_sampler.modifiers.empty()) && cursor)
        prefix = sampling_fork(std::move(prefix), selected_sampler.enabled ? *cursor : cache::SamplingCursor{}, -1);
      else if (resample_frontier) {
        // Recovery wrappers can preserve a historical pending token. A new
        // greedy request starts from the immutable evaluated state's result.
        const auto fresh = prefix->state_identity()->next_id;
        if (fresh < 0 || fresh >= vocabulary)
          throw std::runtime_error("cached evaluated frontier has no valid greedy result");
        prefix = sampling_fork(std::move(prefix), {}, fresh);
      } else if (cursor && sampling_configuration_count() > 1) {
        // A new greedy request can adopt consumed state from a random request.
        // The old sampled pending/cursor is not this request's boundary.
        prefix = sampling_fork(std::move(prefix), {}, -1);
      }
      session->head.adopt(std::move(prefix)); // Exact evaluated-state fork; no KV/state payload copy.
    }
    else {
      auto initial = reserve_state(session->head);
      initial->sampling = selected_sampler.enabled ? cursor.value_or(cache::SamplingCursor{sampling_seed, 0, 0})
          : cache::SamplingCursor{};
      const std::array<std::int32_t, 2> empty{0, 0};
      const std::array<std::int32_t, 1> tail{0};
      const auto layout = take(attention_schema.version({empty, empty, {}, tail}));
      for (std::size_t i = 0; i < initial->keys.size(); ++i) {
        initial->keys[i] = working_frame(take(page_inventory.keys[i].read(layout, {})));
        initial->values[i] = working_frame(take(page_inventory.values[i].read(layout, {})));
      }
      if (cache_plan.draft_cache) {
        initial->draft.emplace();
        initial->draft->key = working_frame(take(draft_inventory.keys[0].read(layout, {})));
        initial->draft->value = working_frame(take(draft_inventory.values[0].read(layout, {})));
        auto zero = take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment));
        check(zero.view()->fill(std::byte{})); initial->carry_hidden = take(zero.publish());
      }
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
        auto history = take(state_pools[i].acquire_history(take(state_schema[i].history.required_bytes())));
        auto recurrent = take(state_pools[i].acquire_recurrent(take(state_schema[i].recurrent.required_bytes())));
        check(history.view()->fill(std::byte{})); check(recurrent.view()->fill(std::byte{}));
        initial->state[i].history = take(history.publish());
        initial->state[i].recurrent = take(recurrent.publish());
      }
      session->head.commit(std::move(initial));
    }
    // State admission does not allocate execution carriers. The first actual
    // explicit prepare creates its needed Phase and slot; idle/full-hit cached
    // Sessions never duplicate IO, workspace or native execution preparation.
    return session;
  }

  void restore_pending_boundary(Session &session, cache::SamplingCursor cursor, std::int32_t pending) const {
    if (session.pending || active_step || carrier_borrowed || pending < -1 || pending >= vocabulary)
      throw std::runtime_error("sampling recovery requires an idle committed boundary and historical pending token");
    session.head.preflight_commit();
    auto replacement = sampling_fork(session.checkpoint(), cursor, pending);
    for (auto &tail : session.tails) tail = {};
    session.draft_tails = {};
    session.tail_epoch = 0; session.draft_tail_epoch = 0;
    session.head.adopt(std::move(replacement));
  }

  // Complete same-model cold parking. No token replay, sampling or native
  // state publication occurs until the original byte transfer succeeds.
  cache::HostImageSchema host_image_schema() const;
  cache::HostImageIdentity host_image_identity(std::size_t sampling_index) const;
  std::shared_ptr<const cache::HostImage> export_session(Session &session,
      const std::shared_ptr<inference::HostStateStorage> &storage);
  std::shared_ptr<Session> restore_session(std::shared_ptr<const cache::HostImage> image);

  void copy_selected_hidden_outputs(StepState &step) {
    if (!step.hidden_output) return;
    auto &copy = *step.hidden_output; auto &phase = *step.phase; auto &run = *step.run;
    const auto count = copy.scratch.size();
    copy.ranges.clear();
    // Export ALL fixed outputs before take_append revokes results. The unused
    // rows are retained only until Graph withdrawal to recover their writer.
    for (std::size_t j = 0; j < step.members.size(); ++j)
      for (std::size_t row = 0; row < phase.prefix_rows; ++row) {
        const auto index = j * phase.prefix_rows + row;
        copy.readers[index] = take(inference::retain_output(run, phase.child->output(phase.carry_ports[j][row])));
      }
    for (std::size_t j = 0; j < step.members.size(); ++j) {
      const auto last = verifies_prefix(phase.family->role) ? step.members[j].accepted_tokens - 1 : 0;
      const auto bind = [&](std::size_t row) {
        const auto index = j * phase.prefix_rows + row;
        const auto bytes = take(copy.scratch[index].description.required_bytes());
        check(copy.invocation.bind_source({static_cast<std::uint32_t>(index)}, copy.readers[index]));
        check(copy.invocation.bind_mutable({static_cast<std::uint32_t>(count + index)},
            take(hidden_pool.allocate(bytes, cache::state_pool_alignment))));
        copy.ranges.push_back({{static_cast<std::uint32_t>(index)}, {static_cast<std::uint32_t>(count + index)}, {0}, 0, 0, bytes});
      };
      bind(last);
      if (verifies_prefix(phase.family->role) && step.members[j].accepted_tokens > 1) bind(last - 1);
    }
    constexpr std::array<c::CopyPhase, 1> copy_phases{};
    check(copy.invocation.prepare_replay(copy_phases, copy.ranges));
    check(copy.retirement.reserve(resources_));
    auto submitted = copy.invocation.submit();
    if (!submitted) { copy.retirement.reject(); (void)take(std::move(submitted)); }
    // One original phase, no producer bindings, sole move-only observer:
    // these accepted-control attachments are capacity-proven and nonallocating.
    auto update = c::CopyBatchUpdate::take(std::move(copy.invocation), std::move(*submitted));
    HYLE_FATAL_ASSERT(update.has_value()); copy.update = std::move(*update);
    const auto sources = copy.update.retirement_count();
    HYLE_FATAL_ASSERT(sources && *sources == 1);
    HYLE_FATAL_ASSERT(copy.retirement.attach(copy.update, 0).ok());
    check(copy.update.wait());
    for (std::size_t j = 0; j < step.members.size(); ++j) {
      auto &member = step.members[j];
      const auto last = verifies_prefix(phase.family->role) ? member.accepted_tokens - 1 : 0;
      const auto ready = [&](std::size_t row) {
        auto writer = take(copy.update.take_mutable({static_cast<std::uint32_t>(count + j * phase.prefix_rows + row)}));
        // The exact full row was copied and actually retired; no padding or
        // untouched half is published as initialized canonical state.
        return take(writer.publish());
      };
      member.next->carry_hidden = ready(last);
      if (verifies_prefix(phase.family->role) && member.accepted_tokens > 1) member.next->prior_hidden = ready(last - 1);
    }
    hyle::Status terminal;
    if (!take(copy.recover(terminal))) throw std::runtime_error("fixed hidden output copy retirement is pending");
    check(terminal);
    // readers stay readonly until original Graph unbind below. In particular,
    // copy completion alone does not permit their promotion/reuse.
  }

  void validate_mtp_snapshot(const cache::State &head, const StepMember *candidate = nullptr) const {
    if (cache_plan.draft_cache != head.draft.has_value())
      throw std::runtime_error("snapshot does not contain the configured complete MTP state");
    if (!head.draft) return;
    const auto &draft = *head.draft;
    const auto key = candidate && candidate->draft_updates[0].valid()
        ? take(candidate->draft_updates[0].view()) : take(draft.key.view());
    const auto value = candidate && candidate->draft_updates[1].valid()
        ? take(candidate->draft_updates[1].view()) : take(draft.value.view());
    if (draft.lag > 1 || head.length < draft.lag || draft.frontier != head.length - draft.lag ||
        !key.valid() || !value.valid() || key.layout().facts().kv_rows != draft.frontier ||
        value.layout().facts().kv_rows != draft.frontier || !head.carry_hidden.valid() ||
        (draft.lag && (!head.prior_hidden.valid() || draft.catchup_token < 0 || draft.catchup_token >= vocabulary)) ||
        (!draft.lag && draft.catchup_token != -1))
      throw std::runtime_error("inconsistent MTP frontier, catch-up token or owned target hidden state");
  }

  void initialize_hidden_output(Phase &phase, std::size_t slot) {
    if (!drafts_token(phase.family->role) ||
        recursive_mtp() || max_drafts() != 1 ||
        phase.replay_modes[slot] != g::ReplayMode::kPreferCapture) return;
    auto &copy = phase.hidden_output[slot];
    // Gate before touching a writer or a Graph binding, even on recovery.
    if (copy.retirement.pending() || copy.update.in_flight() || copy.invocation.in_flight())
      throw std::runtime_error("fixed hidden output still retains original copy work");
    if (!copy.program.valid()) {
      const auto description = dense(precision.activation_type(), {1, hidden});
      const auto bytes = take(description.required_bytes());
      const auto count = cache::multiply(phase.family->requests, phase.prefix_rows);
      if (!count || count > UINT32_MAX / 2) throw std::runtime_error("fixed hidden output capacity overflow");
      // Ownership and actual resource are compiled facts, not backend names.
      for (const auto &ports : phase.carry_ports) for (auto port : ports) {
        const auto &output = phase.child->output(port);
        if (!output.caller_owned()) return;
        validate_state_output(output, description);
      }
      auto context = execution.common_context(); auto queue = execution.common_queue();
      if (!context || !queue) return; // No selected common CopyBatch endpoint.
      std::vector<Tensor> scratch; std::vector<c::CopyResourceSpec> specs;
      scratch.reserve(count); specs.reserve(count * 2);
      for (std::size_t i = 0; i < count; ++i) {
        Tensor value; value.description = description;
        value.pool = take(c::LeasePool::create(execution.device(), bytes,
            {.max_blocks = 1, .backing_alignment = cache::state_pool_alignment,
             .resource_budget = execution_budget}));
        value.writer = take(value.pool.allocate(bytes, cache::state_pool_alignment));
        scratch.push_back(std::move(value)); specs.push_back({execution.device(), bytes});
      }
      for (std::size_t i = 0; i < count; ++i) specs.push_back({execution.device(), bytes});
      auto program = take(c::CopyBatchProgram::prepare(**context, specs,
          {.max_resources = static_cast<std::uint32_t>(count * 2),
           .max_copies = static_cast<std::uint32_t>(count), .max_phases = 1,
           .max_bytes = cache::multiply(count, bytes), .resource_budget = execution_budget}));
      auto invocation = take(c::CopyBatchInvocation::create(program, **queue));
      copy.readers.resize(count); copy.ranges.reserve(count);
      copy.scratch = std::move(scratch); copy.program = std::move(program); copy.invocation = std::move(invocation);
    } else if (!copy.invocation.valid())
      copy.invocation = take(c::CopyBatchInvocation::create(copy.program, *take(execution.common_queue())));
    check(copy.restore_writers());
  }

  void initialize_slot(Phase &phase, std::size_t slot) {
    if (slot >= phase.slot_count) throw std::runtime_error("execution slot is outside its cold carrier plan");
    auto &io = phase.io[slot];
    initialize_hidden_output(phase, slot);
    if (phase.slots[slot].valid()) return;
    const auto &definition = *phase.family;
    const auto &selected_sampler = definition.sampling->configuration;
    const std::array shape{g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(phase.query_rows - 1)}};
    auto run = take(definition.compiled.executable.create_invocation(
        {.shape = shape, .resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (run.exact_variant_index() != phase.variant_index)
      throw std::runtime_error("finite family selected an unexpected exact child");
    if (scheduler) check(g::prepare_progress(run, *scheduler, *workers));
    check(run.bind(phase.token_port, io.tokens.read()));
    check(run.bind(phase.position_port, io.positions.read()));
    if (selected_sampler.enabled && definition.completion == TextCompletion::kEmitNext && definition.role != GraphRole::kVerify) {
      check(run.bind(phase.sampling_identity_port, io.sampling_identities.read()));
    }
    if (phase.sampling_valid_port) {
      check(io.sampling_valid.bind(run));
      if (drafts_token(definition.role)) check(io.sampling_accept.bind(run));
    }
    if (definition.requests != 1 && !verifies_prefix(definition.role)) check(run.bind(phase.last_ids_port, io.last_ids.read()));
    if (io.logits.caller_owned()) check(run.bind(phase.logits_port, io.logits.view()));
    if (definition.completion == TextCompletion::kEmitNext) check(io.next_id.bind(run));
    if (io.final_hidden.caller_owned()) check(run.bind(phase.hidden_port, io.final_hidden.view()));
    if (has_mtp(definition.role)) {
      check(run.bind(phase.draft_position_port, io.draft_positions.read()));
      if (draft_lag(definition.role)) check(run.bind(phase.catchup_token_port, io.catchup_token.read()));
      if (drafts_token(definition.role)) check(io.draft_id.bind(run));
    }
    if (!layer0_probe.empty()) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      if (const auto *output = phase.probe_facts[i]; output && output->caller_owned())
        check(run.bind(output->value, io.probe_outputs[i].view()));
    if (!phase.state_sources.empty())
      check(g::slots::prepare(run, phase.state_graph(), inference::queue_span(phase.queues), phase.state_sources,
          {.host_workspace = execution.staging(), .resource_budget = execution_budget ? execution_budget : cache_budget}));
    // Cold preparation can also run without a pending StepState. Publish only
    // the fully bound slot, so an allocation/binding failure cannot leave an
    // apparently valid half-prepared Invocation in the reusable Model pool.
    phase.slots[slot] = std::move(run);
    phase.preparation_uses[slot] = 0;
  }

  Phase &prepare_child(std::size_t tokens, bool diagnostic, std::size_t requests = 1, GraphRole role = GraphRole::kText,
                       TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                       bool imported_sources = false) {
    auto &cache = diagnostic ? probe_phases : phases;
    const auto found = cache.find({requests, tokens, role, completion, sampling_index, imported_sources});
    if (found != cache.end()) return *found->second;
    auto *selected = find_family(requests, tokens, role, diagnostic, completion, sampling_index);
    if (!selected) throw std::runtime_error("text finite family has not been prepared");
    auto &definition = *selected;
    const auto &selected_sampler = definition.sampling->configuration;
    const auto point = std::find(definition.chunks.begin(), definition.chunks.end(), tokens);
    if (point == definition.chunks.end()) throw std::runtime_error("unlisted text chunk has no prepared exact child");
    const auto start = Clock::now();
    const auto index = static_cast<std::size_t>(point - definition.chunks.begin());
    auto result = std::make_unique<Phase>();
    auto &phase = *result;
    phase.imported_sources = imported_sources;
    if (imported_sources) {
      if (tokens != 1 || diagnostic || role != GraphRole::kText || completion != TextCompletion::kEmitNext ||
          cache_plan.draft_cache || cache_plan.in_flight_batches < 2 || scheduler)
        throw std::runtime_error("imported decode carriers require ordinary T1 and two candidate depths");
      phase.slot_count = execution_slot_count;
    }
    phase.query_rows = tokens;
    static_cast<FamilyPorts &>(phase) = static_cast<const FamilyPorts &>(definition);
    if (imported_sources) for (auto &source : phase.state_sources) source.mode = g::slots::SourceMode::kImported;
    phase.planned_slot_uses = planned_slot_uses.at({requests, tokens, role, completion, sampling_index});
    for (std::size_t slot = 0; slot < phase.slot_count; ++slot) {
      phase.replay_modes[slot] = imported_sources ? g::ReplayMode::kPreferCapture : select_replay_mode(
          replay_policy, phase.planned_slot_uses[slot], selected_sampler.enabled);
    }
    phase.attention_schema = definition.attention_schema;
    phase.child = definition.children.at(index).get();
    phase.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), phase.child->compiled.report));
    phase.family = &definition; phase.variant_index = index;
    for (std::size_t slot = 0; slot < phase.slot_count; ++slot) {
      auto &banks = phase.state_bindings[slot];
      banks.reserve(phase.state_sources.size());
      for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (geometry.is_gdn(layer))
        for (const bool recurrent : {false, true}) {
          StateBinding bank{.layer = layer, .recurrent = recurrent};
          if (!imported_sources) bank.sources.resize(requests);
          bank.destinations.resize(cache::multiply(requests, phase.prefix_rows));
          banks.push_back(std::move(bank));
        }
      phase.state_requests[slot].resize(banks.size());
      for (std::size_t bank = 0; bank < banks.size(); ++bank)
        phase.state_requests[slot][bank] = {phase.state_sources[bank].store, &banks[bank].transaction};
    }
    std::erase_if(phase.state_resource_ports, [&](V value) {
      return value.kind() != g::ValueKind::kInput && !phase.child->output(value).caller_owned();
    });
    const auto t = static_cast<std::int64_t>(tokens);
    if (completion == TextCompletion::kEmitNext)
      phase.logits_facts = &phase.child->output(phase.logits_port);
    phase.hidden_facts = &phase.child->output(phase.hidden_port);
    if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      phase.probe_facts[i] = &phase.child->output(phase.probe_ports[i]);
    for (auto &io : std::span(phase.io).first(phase.slot_count)) {
      const auto token_count = has_mtp(role) && !is_mtp_prefill(role)
          ? static_cast<std::int64_t>(requests) : t * static_cast<std::int64_t>(requests);
      io.tokens = caller_tensor(execution.device(), role == GraphRole::kText
          ? dense(Type::kInt32, {token_count, 1}) : dense(Type::kInt32, {token_count}));
      io.positions = caller_tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {t})
          : dense(Type::kInt32, {static_cast<std::int64_t>(requests), t}));
      if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) {
        io.sampling_identities = caller_tensor(execution.device(), phase.batched_text_sampling
            ? dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 4})
            : dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4}));
      }
      if (phase.sampling_valid_port) {
        io.sampling_valid = take(PreparedReadback::prepare(phase.child->compiled, phase.sampling_valid_port, execution, allocation_budget));
        io.sampling_valid_values.resize(requests);
        bool float_predicates = io.sampling_valid.facts.description.data_type() == Type::kFloat32;
        if (drafts_token(role)) {
          io.sampling_accept = take(PreparedReadback::prepare(phase.child->compiled, phase.sampling_accept_port, execution, allocation_budget));
          io.sampling_accept_values.resize(requests);
          float_predicates = float_predicates || io.sampling_accept.facts.description.data_type() == Type::kFloat32;
        }
        if (float_predicates) io.sampling_scratch.resize(requests);
      }
      if (has_mtp(role)) {
        const auto draft_rows = is_mtp_prefill(role) ? t + static_cast<std::int64_t>(draft_lag(role)) : static_cast<std::int64_t>(draft_lag(role) + 1);
        io.draft_positions = caller_tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {draft_rows})
            : dense(Type::kInt32, {static_cast<std::int64_t>(requests), draft_rows}));
        if (draft_lag(role)) io.catchup_token = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        if (drafts_token(role)) {
          io.draft_id = take(PreparedReadback::prepare(phase.child->compiled, phase.draft_id_port, execution, allocation_budget));
          io.draft_ids.resize(requests);
        }
      }
      if (requests != 1 && !verifies_prefix(role)) {
        io.last_ids = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        std::vector<std::int32_t> last_ids(requests);
        for (std::size_t j = 0; j < requests; ++j) last_ids[j] = static_cast<std::int32_t>((j + 1) * tokens - 1);
        io.last_ids.write<std::int32_t>(last_ids);
      }
      if (completion == TextCompletion::kEmitNext) {
        io.logits = caller_tensor(take(execution.resource_for(phase.logits_facts->storage)),
            phase.logits_facts->description, phase.logits_facts->caller_owned());
        io.next_id = take(PreparedReadback::prepare(phase.child->compiled, phase.id_port, execution, allocation_budget));
      }
      if (verifies_prefix(role)) io.prefix_candidates.resize(cache::multiply(requests, phase.prefix_rows));
      io.final_hidden = caller_tensor(take(execution.resource_for(phase.hidden_facts->storage)),
          phase.hidden_facts->description, phase.hidden_facts->caller_owned());
      if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
        const auto &output = *phase.probe_facts[i];
        io.probe_outputs[i] = caller_tensor(take(execution.resource_for(output.storage)), output.description, output.caller_owned());
      }
    }
    const auto &route_directory = layer0_probe.empty() ? trace_routes : layer0_probe;
    if (!route_directory.empty()) {
      std::filesystem::create_directories(route_directory);
      const auto label = std::string(diagnostic ? "published" : "default") + (has_mtp(role) ? "-role" + std::to_string(static_cast<int>(role)) : "")
          + (sampling_configuration_count() > 1 ? "-sampling" + std::to_string(sampling_index) : "");
      std::ofstream route(route_directory / (std::string(label) + "-R" + std::to_string(requests) + "-T" + std::to_string(tokens) + ".route.tsv"));
      route << "step\tkind\tprovider\timplementation\tnodes\n";
      for (std::size_t step = 0; step < phase.child->compiled.report.step_count(); ++step) {
        const auto detail = take(phase.child->compiled.report.step(step));
        route << step << '\t' << static_cast<unsigned>(detail.kind) << '\t' << detail.provider.name
              << '\t' << detail.implementation.value() << '\t';
        for (std::size_t n = 0; n < detail.node_count; ++n) {
          const auto node = take(phase.child->compiled.report.step_node(step, n));
          const auto kind = take(definition.graph.operation_kind(node));
          route << (n ? "," : "") << node.ordinal() << ':'
                << (kind == c::OperationKind::kView ? c::operation_name(kind)
                    : c::operation_name(take(definition.graph.operation(node))));
        }
        route << '\n';
      }
      std::ofstream boundaries(route_directory / (std::string(label) + "-R" + std::to_string(requests) + "-T" + std::to_string(tokens) + ".boundaries.tsv"));
      boundaries << "point\tvalue\tproducer\tshape\n";
      const auto signature = phase.child->compiled.executable.specialization();
      for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
        const auto value = phase.probe_ports[i];
        boundaries << layer0_probe_names[i] << '\t' << value.ordinal() << '\t'
                   << take(definition.graph.producer(value)).ordinal() << '\t';
        for (const auto dimension : definition.specs.at(value.ordinal()).dimensions()) {
          if (dimension.is_fixed()) { boundaries << dimension.fixed_value() << ','; continue; }
          bool resolved = false;
          for (std::size_t binding = 0; binding < signature.binding_count(); ++binding) {
            const auto entry = take(signature.binding(binding));
            if (entry.dimension != dimension.symbol()) continue;
            boundaries << entry.value << ',';
            resolved = true;
            break;
          }
          if (!resolved) throw std::runtime_error("exact child diagnostic dimension is unresolved");
        }
        boundaries << '\n';
      }
      if (!route || !boundaries) throw std::runtime_error("cannot write layer0 route diagnostics");
    }
    const auto storage = phase.child->compiled.report.compiled_storage();
    std::cerr << "planned_replay T=" << tokens
              << " slot_count=" << phase.slot_count
              << " slot0_uses=" << phase.planned_slot_uses[0]
              << " slot0=" << (phase.replay_modes[0] == g::ReplayMode::kCommands ? "commands" : "prefer-capture") << '\n';
    std::cerr << "compiled T=" << tokens << " steps=" << phase.child->compiled.report.step_count()
              << " workspace/slot=" << phase.child->compiled.executable.workspace_bytes()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " child_prepare_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    input_scratch.prepare(phase);
    auto *address = result.get(); cache.emplace(ExecutionShape{requests, tokens, role, completion, sampling_index, imported_sources}, std::move(result)); return *address;
  }

  // An engine with a known ordered request can explicitly prepare its actual
  // carriers before admission/ready, preserving the original cold boundary.
  // Candidate prefix suffixes remain unmaterialized until a hit selects one.
  void prepare_execution(std::span<const std::size_t> calls,
                         std::size_t requests = 1, GraphRole role = GraphRole::kText,
                          TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (carrier_borrowed || active_step)
      throw std::runtime_error("cannot cold-prepare carriers during an active execution loan");
    for (const auto tokens : calls) {
      // Explicit idle warmup can cover a suffix not present in the initial
      // schedule. Use the same bounded admission and original family owners
      // as execution, before borrowing a Session or accepting native work.
      if (state_storage_prepared && (role == GraphRole::kText || is_mtp_prefill(role)) &&
          !find_family(requests, tokens, role, false, completion, sampling_index))
        admit_exact_shape(tokens, requests, role, completion, sampling_index);
      if (!layer0_probe.empty()) prepare_child(tokens, false, requests, role, completion, sampling_index);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), requests, role, completion, sampling_index);
      initialize_slot(phase, slot_index);
    }
  }

  void prepare_relay_execution(std::size_t requests = 1, std::size_t sampling_index = 0) {
    if (active_step || carrier_borrowed) throw std::runtime_error("relay carrier preparation requires an idle model");
    auto &phase = prepare_child(1, false, requests, GraphRole::kText, TextCompletion::kEmitNext, sampling_index, true);
    for (std::size_t slot = 0; slot < phase.slot_count; ++slot) initialize_slot(phase, slot);
  }

  // Prepare the existing cold plan, not every possible context length. This
  // creates carriers only: no Session, state transition, sampling or submit.
  // Native capture still needs actual bindings and remains a measured cost.
  std::size_t prepare_planned_execution() {
    if (active_step || carrier_borrowed)
      throw std::runtime_error("cannot prepare planned shapes during an active episode");
    std::size_t count = 0;
    for (const auto &[shape, uses] : planned_slot_uses) {
      if (std::ranges::none_of(uses, [](auto planned_uses) { return planned_uses != 0; })) continue;
      const std::array rows{shape.tokens};
      prepare_execution(rows, shape.requests, shape.role, shape.completion, shape.sampling_index);
      ++count;
    }
    return count;
  }

  void detach_episode() noexcept {
    if (!active_step) return;
    if (successor_step && successor_step->cleanup_pending) {
      const auto cleanup = abort_successor_episode();
      inference::record_retirement_status(active_step->terminal, cleanup.status);
      if (!cleanup.retired) {
        return; // Retain both owners for the original retirement retry.
      }
    }
    // Only a caller-requested, actually retired shared cancellation is local
    // to this episode. An unchanged surviving Session remains runnable; do not
    // poison its later retirement with a peer's normal Graph cancellation.
    // Preserve native and all other failures, and the cancelled owner's fact.
    const bool cancelled_episode = active_step->cancelled &&
        active_step->terminal.IsCancelled() && !active_step->terminal.has_native_error();
    for (auto &member : active_step->members) {
      if (!cancelled_episode || member.session->stopping)
        inference::record_retirement_status(member.session->retirement_status, active_step->terminal);
      member.session->pending = successor_step && std::ranges::any_of(successor_step->members,
          [&](const auto &next) { return next.session == member.session; }) ? successor_step.get() : nullptr;
    }
    active_step = std::move(successor_step);
    if (active_step) for (auto &member : active_step->members) {
      if (member.session->head.get() == member.base && member.session->head.epoch() == member.base_epoch)
        member.first_input = member.session->head->next_id;
      else active_step->cancelled = true;
    }
  }

  inference::RetirementObservation try_retire_episode() noexcept {
    if (!active_step) return {true, {}};
    if (successor_step) {
      inference::record_retirement_status(successor_step->terminal, execution.collect_retired());
      const auto retired = successor_step->try_abort();
      if (!retired) return {false, retired.error()};
      if (!*retired) return {false, successor_step->terminal};
      successor_step.reset();
    }
    // Dropping the public Completion does not remove original native work.
    // Drive its owning environment before checking the unchanged Graph/page
    // retirement gates; an observation error alone never proves retirement.
    inference::record_retirement_status(active_step->terminal, execution.collect_retired());
    const auto retired = active_step->try_abort();
    if (!retired) {
      inference::record_retirement_status(active_step->terminal, retired.error());
      return {false, active_step->terminal};
    }
    const auto terminal = active_step->terminal;
    if (*retired) detach_episode();
    return {*retired, terminal};
  }

  inference::RetirementObservation abort_successor_episode() noexcept {
    if (!successor_step) return {true, {}};
    if (!successor_step->submitted && successor_step->run && active_step &&
        active_step->run && active_step->run->in_flight()) {
      successor_step->cancelled = true;
      successor_step->cleanup_pending = true;
      return {false, {}};
    }
    if (successor_step->run)
      inference::record_retirement_status(successor_step->terminal, execution.drain_retired(*successor_step->run));
    const auto retired = successor_step->try_abort();
    if (!retired) return {false, retired.error()};
    const auto status = successor_step->terminal;
    if (*retired) successor_step.reset();
    return {*retired, status};
  }

  inference::RetirementObservation abort_episode() noexcept {
    discard_decode_ahead();
    if (!active_step) return {true, {}};
    if (successor_step && successor_step->run)
      inference::record_retirement_status(successor_step->terminal, execution.drain_retired(*successor_step->run));
    // Nonblocking collection cannot drain a partially accepted Vulkan prefix.
    // Hand off once to the original blocking owner, then check the unchanged
    // Graph/page gates. Unknown native failure keeps this candidate, Session
    // pins and admission intact for an explicit retry; it must not spin.
    if (active_step->run)
      inference::record_retirement_status(active_step->terminal, execution.drain_retired(*active_step->run));
    // Episode retirement alone does not release a Session's published head.
    return try_retire_episode();
  }

  // Selection and equal-T bucketing belong to the caller. A batch owns one
  // original Invocation, IO loan, Completion and publication transaction.
  void prepare_draft_append(StepState &step, GraphRole role) {
    auto &phase = *step.phase; auto &run = *step.run;
    auto &io = phase.io[step.state_slot];
    const auto width = step.members.size();
    const auto rows = is_mtp_prefill(role) ? cache::add(step.input_tokens, draft_lag(role)) : draft_lag(role) + 1;
    hyle::container::SmallVector<std::int32_t, 2> query{0}, old_ptr{0}, next_ptr{0}, old_last, next_last;
    std::vector<std::int32_t> old_ids, next_ids;
    auto &positions = input_scratch.draft_positions, &catchups = input_scratch.catchups;
    positions.clear(); catchups.clear();
    for (auto *table : {&query, &old_ptr, &next_ptr})
      if (!table->try_reserve(width + 1)) throw std::bad_alloc{};
    for (auto *table : {&old_last, &next_last})
      if (!table->try_reserve(width)) throw std::bad_alloc{};
    std::size_t old_count = 0, next_count = 0;
    for (const auto &member : step.members) {
      old_count = cache::add(old_count, member.base->draft->pages.size());
      next_count = cache::add(next_count, (cache::add(member.base->draft->frontier, rows) + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
    }
    old_ids.reserve(old_count); next_ids.reserve(next_count);
    for (std::size_t j = 0; j < width; ++j) {
      auto &member = step.members[j]; auto &session = *member.session;
      const auto &old = *member.base->draft;
      member.next->draft.emplace(); auto &next = *member.next->draft;
      next.frontier = cache::add(old.frontier, rows);
      if (next.frontier > capacity) throw std::runtime_error("draft cache exceeds context capacity");
      const auto prefix_pages = old.frontier / cache::Plan::page_rows;
      const auto count = (next.frontier + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      const auto dirty_count = count - prefix_pages, old_tail = old.frontier % cache::Plan::page_rows;
      bool reuse = old_tail && session.draft_tail_epoch == member.base_epoch && !old.pages.empty();
      if (reuse) for (const auto &tail : session.draft_tails)
        reuse = reuse && tail.valid() && !tail.reserved_rows() &&
            tail.page() == old.pages.back().page() && tail.initialized_rows() == old_tail;
      next.pages.reserve(count);
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) step.dirty[bank].reserve(cache::add(step.dirty[bank].size(), dirty_count));
      if (reuse) {
        const auto added = static_cast<std::uint32_t>((std::min)(rows, cache::Plan::page_rows - old_tail));
        std::size_t reserved = 0; hyle::Status status;
        for (; reserved < session.draft_tails.size(); ++reserved) {
          status = session.draft_tails[reserved].reserve_rows(added); if (!status.ok()) break;
        }
        if (!status.ok()) {
          for (std::size_t i = 0; i < reserved; ++i) check(session.draft_tails[i].rollback());
          if (!status.IsUnavailable()) check(status);
          reuse = false;
        }
      }
      for (std::size_t page = 0; page < prefix_pages; ++page) next.pages.push_back(old.pages.at(page));
      if (reuse) {
        next.pages.push_back(old.pages.back());
        for (std::size_t bank = 0; bank < 2; ++bank) step.tails[draft_bank + bank].push_back(std::move(session.draft_tails[bank]));
      } else session.draft_tails = {};
      session.draft_tail_epoch = 0;
      for (std::size_t page = reuse ? 1 : 0; page < dirty_count; ++page) {
        auto reservation = take(draft_inventory.acquire()); next.pages.push_back(std::move(reservation.reference));
        step.dirty[draft_bank].push_back(std::move(reservation.keys[0])); step.dirty[draft_bank + 1].push_back(std::move(reservation.values[0]));
      }
      for (const auto &page : old.pages) old_ids.push_back(static_cast<std::int32_t>(page.page()));
      const auto begin = next_ids.size();
      for (const auto &page : next.pages) next_ids.push_back(static_cast<std::int32_t>(page.page()));
      query.push_back(static_cast<std::int32_t>((j + 1) * rows));
      old_ptr.push_back(static_cast<std::int32_t>(old_ids.size())); next_ptr.push_back(static_cast<std::int32_t>(next_ids.size()));
      old_last.push_back(old.frontier ? static_cast<std::int32_t>((old.frontier - 1) % cache::Plan::page_rows + 1) : 0);
      next_last.push_back(static_cast<std::int32_t>((next.frontier - 1) % cache::Plan::page_rows + 1));
      const std::array<std::int32_t, 2> one_query{0, static_cast<std::int32_t>(rows)}, one_kv{0, static_cast<std::int32_t>(count)};
      member.draft_layout = take(attention_schema.version({one_query, one_kv, std::span(next_ids).subspan(begin, count), std::span(&next_last.back(), 1)}));
      for (std::size_t i = 0; i < rows; ++i) positions.push_back(static_cast<std::int32_t>(old.frontier + i));
      const auto &ports = phase.draft_hidden_inputs[j];
      check(run.bind(ports[0], dense(precision.activation_type(), {1, hidden}), member.base->carry_hidden));
      if (draft_lag(role)) {
        catchups.push_back(old.catchup_token);
        check(run.bind(ports[1], dense(precision.activation_type(), {1, hidden}), member.base->prior_hidden));
      }
    }
    const auto layout = width == 1 ? step.members[0].draft_layout : take(attention_schema.version({query, next_ptr, next_ids, next_last}));
    step.draft_layout = layout;
    std::vector<std::int32_t> old_query(width == 1 ? 0 : width + 1, 0);
    const auto old_layout = width == 1 ? c::BatchLayoutVersion{} : take(attention_schema.version({old_query, old_ptr, old_ids, old_last}));
    std::vector<c::PagedArenaFootprint> sources(width);
    for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
      auto &arena = bank == draft_bank ? draft_inventory.keys[0] : draft_inventory.values[0];
      for (std::size_t j = 0; j < width; ++j) sources[j] = step.members[j].draft_source(bank == draft_bank);
      const auto previous = width == 1 ? sources[0].layout() : old_layout;
      step.appends[bank] = take(g::batch::PagedAppendPages::create(arena, layout, previous, sources,
          step.dirty[bank], step.tails[bank], {.resource_budget = cache_budget}));
    }
    io.draft_positions.write<std::int32_t>(positions);
    if (draft_lag(role)) io.catchup_token.write<std::int32_t>(catchups);
  }

  void discard_decode_ahead(const Session *session = nullptr) noexcept {
    if (!decode_ahead) return;
    auto &prepared = *decode_ahead;
    if (!session) {
      if (prepared.ready) ++cpu_ahead_discarded;
      prepared.clear(); return;
    }
    bool discarded = false;
    for (auto &member : prepared.members)
      if (member.ready && member.session.lock().get() == session) {
        member = {}; discarded = true;
      }
    if (discarded) ++cpu_ahead_discarded;
    prepared.refresh();
  }

  void prune_decode_ahead() noexcept {
    if (!decode_ahead || !decode_ahead->ready) return;
    bool discarded = false;
    for (auto &member : decode_ahead->members) if (member.ready) {
      const auto owner = member.session.lock();
      if (!owner || owner->failed || owner->stopping || !owner->head ||
          owner->head.epoch() != member.epoch || owner->length() <= member.base_length ||
          owner->length() > member.maximum_length) {
        member = {}; discarded = true;
      }
    }
    if (discarded) ++cpu_ahead_discarded;
    decode_ahead->refresh();
  }

  // Only integer metadata, cold-bounded O(R*pages + R*K). No head, state slot,
  // draft frontier or native permission is retained by this optional cache.
  void prepare_decode_ahead_storage() noexcept {
    if ((!cache_plan.draft_cache && cache_plan.active_sequences < 2) || decode_ahead) return;
    try {
      decode_ahead.emplace(execution_budget, take(execution_budget.metadata_allocator()));
      auto &prepared = *decode_ahead;
      const auto width = cache_plan.active_sequences;
      prepared.maximum_rows = cache_plan.draft_cache ? cache_plan.candidate_boundaries : 1;
      prepared.page_stride = cache::add(capacity, cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      if (!prepared.members.try_resize(width) || !prepared.selected.try_reserve(width) ||
          !prepared.candidate_ids.try_resize(cache::multiply(width, prepared.page_stride)) ||
          !prepared.positions.try_reserve(cache::multiply(width, prepared.maximum_rows)) ||
          !prepared.old_ids.try_reserve(attention_schema.capacity().max_page_refs) ||
          !prepared.old_indptr.try_reserve(width + 1) || !prepared.old_last.try_reserve(width) ||
          !prepared.old_query.try_reserve(width + 1)) decode_ahead.reset();
    } catch (...) { decode_ahead.reset(); } // Optional, before any accepted work.
  }

  // Save each candidate's page stripe and finite accepted-length range, not
  // K^R outcomes. Actual next role/rows, draft repair and permissions remain
  // in the original prepare/finish path.
  hyle::Status prepare_decode_ahead() noexcept {
    if (!decode_ahead) return hyle::Status::Ok();
    if (!active_step || !active_step->submitted || active_step->cancelled || !active_step->phase ||
        active_step->phase->family->completion != TextCompletion::kEmitNext ||
        is_mtp_prefill(active_step->phase->family->role) ||
        (active_step->phase->family->role != GraphRole::kText && !verifies_prefix(active_step->phase->family->role) && !has_mtp(active_step->phase->family->role)))
      return hyle::Status::FailedPrecondition("decode metadata requires an original submitted generation episode");
    const auto begin = Clock::now();
    const auto fail = [&](hyle::Status status) noexcept {
      discard_decode_ahead();
      cpu_ahead_prepare_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
      return status;
    };
    try {
      auto &prepared = *decode_ahead;
      const auto width = active_step->members.size();
      if (!width || width > prepared.members.size() || !active_step->input_tokens ||
          active_step->input_tokens > prepared.maximum_rows)
        return fail(hyle::Status::Unavailable("decode topology exceeds its cold workspace"));
      for (const auto &member : active_step->members)
        if (!member.next || member.session->failed || member.session->stopping ||
            member.base->length >= capacity - 1 || member.base_epoch == UINT64_MAX ||
            member.next->pages.size() > prepared.page_stride)
          return fail(hyle::Status::Unavailable("decode metadata member cannot continue"));
      prune_decode_ahead(); prepared.clear_output();
      bool replaced = false;
      for (const auto &member : active_step->members) {
        auto index = prepared.members.size();
        // Preserve sibling subgroups: prefer this Session, then a vacant slot,
        // otherwise a bounded unused entry. Never replace a slot selected by
        // an earlier member of this same submitted episode.
        for (std::size_t i = 0; i < prepared.members.size(); ++i)
          if (!prepared.members[i].selected && prepared.members[i].ready && prepared.members[i].session.lock() == member.session) {
            index = i; break;
          }
        if (index == prepared.members.size())
          for (std::size_t i = 0; i < prepared.members.size(); ++i)
            if (!prepared.members[i].selected && !prepared.members[i].ready) { index = i; break; }
        if (index == prepared.members.size())
          for (std::size_t i = 0; i < prepared.members.size(); ++i)
            if (!prepared.members[i].selected) { index = i; break; }
        if (index == prepared.members.size())
          return fail(hyle::Status::Unavailable("decode metadata has no cold member slot"));
        auto &saved = prepared.members[index]; replaced = replaced || saved.ready;
        saved = {member.session, member.base_epoch + 1, member.base->length,
            member.next->length, active_step->phase->family->sampling->index, true, true};
        for (std::size_t page = 0; page < member.next->pages.size(); ++page)
          prepared.candidate_ids[index * prepared.page_stride + page] =
              static_cast<std::int32_t>(member.next->pages[page].page());
        prepared.selected.push_back(index);
      }
      if (replaced) ++cpu_ahead_discarded;
      for (const auto index : prepared.selected) prepared.members[index].selected = false;
      prepared.selected.clear(); prepared.refresh(); ++cpu_ahead_prepared;
      cpu_ahead_prepare_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
      return hyle::Status::Ok();
    } catch (const std::bad_alloc &) {
      return fail(hyle::Status::ResourceExhausted("optional decode metadata allocation"));
    } catch (...) {
      return fail(hyle::Status::Unavailable("optional decode metadata preparation"));
    }
  }

  bool take_decode_ahead(std::span<const EvaluationRequest> requests, GraphRole role,
      TextCompletion completion, std::size_t sampling_index, std::size_t tokens) noexcept {
    if (!decode_ahead || !decode_ahead->ready) return false;
    auto &prepared = *decode_ahead;
    prune_decode_ahead(); prepared.clear_output();
    if (requests.empty() || requests.size() > prepared.members.size() || !tokens ||
        tokens > prepared.maximum_rows || completion != TextCompletion::kEmitNext || is_mtp_prefill(role) ||
        (role != GraphRole::kText && !verifies_prefix(role) && !has_mtp(role))) return false;
    const auto miss = [&]() noexcept { prepared.clear_output(); return false; };
    // Standard prepare has already validated actual MTP repair/lag, inputs,
    // role and capacity. Only this exact published target-page prefix may
    // reuse the integers; no predicted draft state or permission is trusted.
    for (const auto &request : requests) {
      auto index = prepared.members.size();
      for (std::size_t i = 0; i < prepared.members.size(); ++i) {
        const auto &saved = prepared.members[i];
        if (!saved.ready || saved.sampling_index != sampling_index) continue;
        auto owner = saved.session.lock();
        if (owner != request.session || owner.owner_before(request.session) || request.session.owner_before(owner)) continue;
        if (owner->head.epoch() != saved.epoch || owner->length() <= saved.base_length ||
            owner->length() > saved.maximum_length || owner->head->pages.size() > prepared.page_stride) break;
        bool same_pages = true;
        for (std::size_t page = 0; page < owner->head->pages.size(); ++page)
          if (owner->head->pages[page].page() != static_cast<std::uint32_t>(
              prepared.candidate_ids[i * prepared.page_stride + page])) { same_pages = false; break; }
        if (same_pages) index = i;
        break;
      }
      if (index == prepared.members.size() || prepared.members[index].selected) return miss();
      prepared.members[index].selected = true; prepared.selected.push_back(index);
    }
    prepared.old_indptr.push_back(0); prepared.old_query.resize(requests.size() + 1, 0);
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto index = prepared.selected[j]; const auto &owner = *requests[j].session;
      for (std::size_t row = 0; row < tokens; ++row)
        prepared.positions.push_back(static_cast<std::int32_t>(owner.length() + row));
      for (std::size_t page = 0; page < owner.head->pages.size(); ++page)
        prepared.old_ids.push_back(prepared.candidate_ids[index * prepared.page_stride + page]);
      prepared.old_indptr.push_back(static_cast<std::int32_t>(prepared.old_ids.size()));
      prepared.old_last.push_back(static_cast<std::int32_t>((owner.length() - 1) % cache::Plan::page_rows + 1));
    }
    for (const auto index : prepared.selected) prepared.members[index] = {};
    prepared.selected.clear(); prepared.refresh(); ++cpu_ahead_hits;
    return true;
  }

  void prepare_evaluations(std::span<const EvaluationRequest> requests,
                           inference::PreparationObservation *preparation = nullptr,
                           GraphRole role = GraphRole::kText,
                           TextCompletion completion = TextCompletion::kEmitNext,
                           std::shared_ptr<StepSources> incoming = {}, std::size_t physical_slot = 0) {
    const auto predecessor = incoming && active_step ? active_step : std::shared_ptr<StepState>{};
    const bool relay = bool(incoming);
    if ((completion != TextCompletion::kEmitNext && completion != TextCompletion::kStateOnly) ||
        (completion == TextCompletion::kStateOnly && role != GraphRole::kText))
      throw std::runtime_error("state-only evaluation requires ordinary Text");
    if (requests.empty() || requests.size() > cache_plan.active_sequences)
      throw std::runtime_error("batch width exceeds the configured active sequence capacity");
    if (!relay && (carrier_borrowed || active_step))
      throw std::runtime_error("the original execution carrier has not retired");
    if (relay && (successor_step || scheduler || cache_plan.draft_cache || cache_plan.in_flight_batches < 2 ||
        role != GraphRole::kText || completion != TextCompletion::kEmitNext || physical_slot >= execution_slot_count ||
        incoming->boundaries.size() != requests.size() || !incoming->submission.valid() || !incoming->token.valid() ||
        incoming->states.size() != geometry.recurrent_layers() * 2 || incoming->pages.size() != draft_bank ||
        (predecessor && (predecessor->sources != incoming || predecessor->cancelled || !predecessor->submitted ||
            predecessor->members.size() != requests.size()))))
      throw std::runtime_error("decode successor requires one exact complete ordinary predecessor and independent carriers");
    if (!requests.front().session || requests.front().session->model_ != this)
      throw std::runtime_error("batch session belongs to another model");
    const auto &prepared_sampling = requests.front().session->sampling_configuration();
    const auto sampling_index = prepared_sampling.index;
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && (sampling_index || !selected_sampler.modifiers.empty()))
      throw std::runtime_error("nondefault sampling is ordinary Text only");
    const auto supplied_tokens = relay ? std::size_t{1} : requests.front().ids.size();
    const auto tokens = drafts_token(role) ? 2u : supplied_tokens;
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("evaluation must use the configured complete target/draft state");
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      if (!request.session || request.session->model_ != this)
        throw std::runtime_error("batch session belongs to another model");
      const auto &session = *request.session;
      const auto &stop_policy = inference::request_stop_policy(request.stop_policy);
      stop_policy.validate(vocabulary);
      if (role != GraphRole::kText && !stop_policy.is_default())
        throw std::runtime_error("nondefault request stop policy is ordinary Text only");
      if (&session.sampling_configuration() != &prepared_sampling) throw std::runtime_error("batch requires one cold sampling configuration");
      if (completion == TextCompletion::kEmitNext) validate_logits_inputs(selected_sampler, request.logits, vocabulary);
      if (session.stopping || session.failed || (session.pending && (!predecessor || session.pending != predecessor.get())))
        throw std::runtime_error("batch session is stopped, failed or already pending");
      session.head.preflight_commit();
      const auto *base = predecessor ? predecessor->members[j].next.get() : session.head.get();
      if (relay) {
        auto owner = incoming->boundaries[j].session.lock();
        const auto epoch = session.head.epoch();
        if (!owner || owner != request.session || !base || incoming->boundaries[j].candidate != base ||
            epoch > UINT64_MAX - (predecessor ? 2u : 1u) ||
            incoming->boundaries[j].epoch != epoch + (predecessor ? 1u : 0u) ||
            (predecessor && predecessor->members[j].session != request.session) ||
            (!request.ids.empty() && (predecessor || request.ids.size() != 1 || request.ids[0] != base->next_id)))
          throw std::runtime_error("successor Source differs from its exact Session candidate and epoch");
      }
      if (!supplied_tokens || (!relay && request.ids.size() != supplied_tokens) || tokens > capacity - base->length)
        throw std::runtime_error("batch requires equal nonempty chunks within each remaining context");
      validate_mtp_snapshot(*session.head);
      if (has_mtp(role)) {
        if (session.head->draft->lag != draft_lag(role) ||
            (!is_mtp_prefill(role) && supplied_tokens != 1))
          throw std::runtime_error("MTP role, pending token or catch-up lag does not match its snapshot");
      }
      for (std::size_t other = 0; other < j; ++other)
        if (requests[other].session == request.session)
          throw std::runtime_error("batch repeats a mutable session");
      for (auto id : request.ids) if (id < 0 || id >= vocabulary)
        throw std::runtime_error("token ID outside vocabulary");
      // Raw Text callers have no turn quota. Still reject their actual draw
      // before creating an episode, borrowing the carrier or writing inputs.
      if (role == GraphRole::kText && completion == TextCompletion::kEmitNext && selected_sampler.enabled)
        (void)base->sampling.advanced(1);
    }
    // A suffix first seen at a request boundary may compile a new exact family.
    // Charge that real synchronous work to this step, before the carrier loan.
    const auto step_start = Clock::now();
    double shape_admission_seconds = 0;
    if ((role == GraphRole::kText || is_mtp_prefill(role)) &&
        !find_family(requests.size(), tokens, role, false, completion, sampling_index)) {
      admit_exact_shape(tokens, requests.size(), role, completion, sampling_index);
      shape_admission_seconds = std::chrono::duration<double>(Clock::now() - step_start).count();
    }
    const bool ahead = !relay && take_decode_ahead(requests, role, completion, sampling_index, tokens);
    auto candidate = std::make_shared<StepState>(); candidate->configure(draft_bank + 2);
    if (!candidate->members.try_reserve(requests.size())) throw std::bad_alloc{};
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      candidate->members.push_back({.session = request.session, .owner_state = &request.session->head,
          .base = predecessor ? predecessor->members[j].next.get() : request.session->head.get(),
          .base_epoch = request.session->head.epoch() + (predecessor ? 1u : 0u),
          .stop_policy = request.stop_policy});
      candidate->members.back().first_input = relay ? (predecessor ? -1 : request.session->head->next_id) : request.ids.front();
      if (role == GraphRole::kVerify)
        candidate->members.back().verification_inputs.assign(request.ids.begin(), request.ids.end());
    }
    if (relay) {
      candidate->incoming = std::make_unique<IncomingStepSources>();
      candidate->incoming->owner = std::move(incoming);
    }
    auto &step = *candidate;
    if (predecessor) successor_step = candidate; else active_step = candidate;
    ++carrier_borrowed; step.candidate_admission = &carrier_borrowed;
    step.predecessor_episode = predecessor;
    step.step_start = step_start; step.input_tokens = tokens;
    step.shape_admission_seconds = shape_admission_seconds;
    if (!predecessor) for (auto &member : step.members) member.session->pending = &step;
    try {
      const auto width = requests.size(), slot = relay ? physical_slot : slot_index;
      if (!layer0_probe.empty()) prepare_child(tokens, false, width, role, completion, sampling_index);
      if (predecessor) {
        const auto found = phases.find({width, tokens, role, completion, sampling_index, true});
        if (found == phases.end() || !found->second->slots[slot].valid())
          throw std::runtime_error("successor carrier must be cold-prepared before its predecessor submission");
      }
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), width, role, completion, sampling_index, relay);
      auto &run = phase.slots[slot];
      auto &io = phase.io[slot];
      // A rejected successor never owns or clears its predecessor's run.
      if (predecessor && predecessor->run == &run)
        throw std::runtime_error("successor cannot reuse its live predecessor carrier");
      step.phase = &phase; step.run = &run; step.state_slot = slot; step.nodes.assign(phase.append_nodes.begin(), phase.append_nodes.end());
      initialize_slot(phase, slot);
      auto &packed_ids = input_scratch.packed_ids, &positions = input_scratch.positions;
      packed_ids.clear(); positions.clear();
      for (std::size_t j = 0; j < width; ++j) {
        const auto &request = requests[j];
        if (!relay && width != 1) packed_ids.insert(packed_ids.end(), request.ids.begin(), request.ids.end());
        if (!ahead) for (std::size_t row = 0; row < tokens; ++row)
          positions.push_back(static_cast<std::int32_t>(step.members[j].base->length + row));
      }
      for (std::size_t j = 0; j < width; ++j) {
        if (!phase.logits_bias_ports.empty()) check(run.bind(phase.logits_bias_ports[j], requests[j].logits.bias.description, requests[j].logits.bias.lease));
        if (!phase.logits_allowed_ports.empty()) check(run.bind(phase.logits_allowed_ports[j], requests[j].logits.allowed.description, requests[j].logits.allowed.lease));
      }
      if (!relay) io.tokens.write<std::int32_t>(width == 1 ? requests[0].ids : std::span<const std::int32_t>(packed_ids));
      if (ahead) {
        io.positions.write<std::int32_t>({decode_ahead->positions.data(), decode_ahead->positions.size()});
        decode_ahead->positions.clear();
      } else io.positions.write<std::int32_t>(positions);
      if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) {
        auto &identities = input_scratch.sampling_identities;
        identities.clear();
        for (const auto &member : step.members) {
          if (role != GraphRole::kText) (void)member.base->sampling.advanced(drafts_token(role) ? 3 : 1);
          for (std::size_t draw = 0; draw < (phase.batched_text_sampling ? 1u : 3u); ++draw) {
            // Batched ordinary Text binds one original slot0 key per row.
            // Other non-round roles retain three slots and consume slot0.
            const auto identity = member.base->sampling.identity(drafts_token(role) ? draw : 0);
            identities.insert(identities.end(), identity.begin(), identity.end());
          }
        }
        io.sampling_identities.write<std::uint32_t>(identities);
      }
      const auto cold_start = Clock::now();
      hyle::container::SmallVector<std::int32_t, 2> query_indptr{0}, old_indptr{0}, new_indptr{0}, old_last, new_last;
      for (auto *table : {&query_indptr, &old_indptr, &new_indptr, &old_last, &new_last})
        if (!table->try_reserve(width + 1)) throw std::bad_alloc{};
      std::vector<std::int32_t> old_ids;
      auto &new_ids = input_scratch.target_page_ids;
      new_ids.clear();
      std::size_t old_refs = 0, new_refs = 0;
      for (const auto &member : step.members) {
        old_refs = cache::add(old_refs, member.base->pages.size());
        new_refs = cache::add(new_refs, (member.base->length + tokens + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
      }
      if (width != 1 && !ahead) old_ids.reserve(old_refs);
      if (new_refs > new_ids.capacity())
        throw std::runtime_error("target page-index scratch exceeds prepared schema capacity");
      for (auto &bank : step.tails) if (!bank.try_reserve(width)) throw std::bad_alloc{};
      for (std::size_t j = 0; j < width; ++j) {
        auto &member = step.members[j]; auto &session = *member.session;
        const auto length = member.base->length;
        member.next = reserve_state(session.head);
        if (!member.next_tails.try_resize(draft_bank)) throw std::bad_alloc{};
        member.next->length = length + tokens;
        member.next->sampling = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify ? member.base->sampling.advanced(drafts_token(role) ? 3 : 1)
            : member.base->sampling;
        const auto full_prefix_pages = length / cache::Plan::page_rows;
        const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
        const auto dirty_count = page_count - full_prefix_pages;
        const auto tail_rows = length % cache::Plan::page_rows;
        bool reuse_tail = !relay && tail_rows && session.tail_epoch == member.base_epoch && !member.base->pages.empty();
        if (reuse_tail) for (const auto &tail : session.tails)
          reuse_tail = reuse_tail && tail.valid() && !tail.reserved_rows() &&
              tail.page() == member.base->pages.back().page() && tail.initialized_rows() == tail_rows;
        member.next->pages.reserve(page_count);
        for (auto &bank : step.dirty) bank.reserve(cache::add(bank.size(), dirty_count - (reuse_tail ? 1 : 0)));
        if (reuse_tail) {
          const auto rows = static_cast<std::uint32_t>((std::min)(tokens, cache::Plan::page_rows - tail_rows));
          std::size_t reserved = 0; hyle::Status status;
          for (; reserved < session.tails.size(); ++reserved) {
            status = session.tails[reserved].reserve_rows(rows); if (!status.ok()) break;
          }
          if (!status.ok()) {
            for (std::size_t i = 0; i < reserved; ++i) check(session.tails[i].rollback());
            if (!status.IsUnavailable()) check(status);
            reuse_tail = false;
            for (auto &bank : step.dirty) bank.reserve(cache::add(bank.size(), dirty_count));
          }
        }
        for (std::size_t i = 0; i < full_prefix_pages; ++i) member.next->pages.push_back(member.base->pages.at(i));
        if (reuse_tail) {
          member.next->pages.push_back(member.base->pages.back());
          for (std::size_t bank = 0; bank < session.tails.size(); ++bank)
            step.tails[bank].push_back(std::move(session.tails[bank]));
        } else if (!relay) for (auto &tail : session.tails) tail = {};
        if (!relay) session.tail_epoch = 0;
        for (std::size_t page = reuse_tail ? 1 : 0; page < dirty_count; ++page) {
          auto reservation = take(page_inventory.acquire());
          member.next->pages.push_back(std::move(reservation.reference));
          for (std::size_t layer = 0; layer < reservation.keys.size(); ++layer) {
            step.dirty[layer * 2].push_back(std::move(reservation.keys[layer]));
            step.dirty[layer * 2 + 1].push_back(std::move(reservation.values[layer]));
          }
        }
        if (width != 1 && !ahead) for (const auto &page : member.base->pages) old_ids.push_back(static_cast<std::int32_t>(page.page()));
        const auto begin = new_ids.size();
        for (const auto &page : member.next->pages) new_ids.push_back(static_cast<std::int32_t>(page.page()));
        query_indptr.push_back(static_cast<std::int32_t>((j + 1) * tokens));
        old_indptr.push_back(static_cast<std::int32_t>(old_ids.size()));
        new_indptr.push_back(static_cast<std::int32_t>(new_ids.size()));
        old_last.push_back(length ? static_cast<std::int32_t>((length - 1) % cache::Plan::page_rows + 1) : 0);
        new_last.push_back(static_cast<std::int32_t>((member.next->length - 1) % cache::Plan::page_rows + 1));
        const std::array<std::int32_t, 2> one_query{0, static_cast<std::int32_t>(tokens)}, one_kv{0, static_cast<std::int32_t>(page_count)};
        member.layout = take(attention_schema.version({one_query, one_kv,
            std::span(new_ids).subspan(begin, page_count), std::span(&new_last.back(), 1)}));
      }
      io.attention_layout = width == 1 ? step.members[0].layout
          : take(attention_schema.version({query_indptr, new_indptr, new_ids, new_last}));
      // Old layouts describe no new queries, but retain each independent history.
      const std::vector<std::int32_t> old_query(width == 1 || ahead ? 0 : width + 1, 0);
      const auto old_layout = width == 1 ? c::BatchLayoutVersion{}
          : ahead ? take(attention_schema.version({
              {decode_ahead->old_query.data(), decode_ahead->old_query.size()},
              {decode_ahead->old_indptr.data(), decode_ahead->old_indptr.size()},
              {decode_ahead->old_ids.data(), decode_ahead->old_ids.size()},
              {decode_ahead->old_last.data(), decode_ahead->old_last.size()}}))
          : take(attention_schema.version({old_query, old_indptr, old_ids, old_last}));
      if (ahead) decode_ahead->clear_output(); // version() copied this group's borrowed integers.
      std::vector<c::PagedArenaFootprint> sources(width);
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        // A ready immutable head needs its own mutable frame metadata, just as
        // on the direct path. A pending predecessor has no ready frame yet;
        // that original frame is transferred only after its ordered commit.
        if (relay && !predecessor) for (auto &member : step.members) {
          (void)member.source(layer, true); (void)member.source(layer, false);
        }
        const auto compose = [&](bool key) {
          for (std::size_t j = 0; j < width; ++j)
            sources[j] = step.members[j].source(layer, key);
          return width == 1 ? sources[0].layout() : old_layout;
        };
        auto &key_arena = page_inventory.keys[layer]; auto &value_arena = page_inventory.values[layer];
        step.appends[layer * 2] = relay ? take(g::batch::detail::make_append_from_source(key_arena, io.attention_layout,
            step.incoming->owner->pages[layer * 2], step.dirty[layer * 2], {.resource_budget = cache_budget}))
          : take(g::batch::PagedAppendPages::create(key_arena, io.attention_layout,
            compose(true), sources, step.dirty[layer * 2], step.tails[layer * 2], {.resource_budget = cache_budget}));
        step.appends[layer * 2 + 1] = relay ? take(g::batch::detail::make_append_from_source(value_arena, io.attention_layout,
            step.incoming->owner->pages[layer * 2 + 1], step.dirty[layer * 2 + 1], {.resource_budget = cache_budget}))
          : take(g::batch::PagedAppendPages::create(value_arena, io.attention_layout,
            compose(false), sources, step.dirty[layer * 2 + 1], step.tails[layer * 2 + 1], {.resource_budget = cache_budget}));
      }
      if (has_mtp(role)) prepare_draft_append(step, role);
      for (auto &bank : phase.state_bindings[slot]) {
        auto &pools = state_pools[bank.layer];
        const auto &description = bank.recurrent ? state_schema[bank.layer].recurrent : state_schema[bank.layer].history;
        const auto bytes = take(description.required_bytes());
        for (std::size_t j = 0; !relay && j < width; ++j) {
          const auto &state = step.members[j].base->state[bank.layer];
          bank.sources[j] = bank.recurrent ? state.recurrent : state.history;
        }
        for (auto &writer : bank.destinations)
          writer = take(bank.recurrent ? pools.acquire_recurrent(bytes) : pools.acquire_history(bytes));
        if (bank.transaction.valid()) check(bank.transaction.refill(bank.sources, bank.destinations));
        else bank.transaction = take(g::slots::SlotBindings::create(
            bank.recurrent ? pools.recurrent_layout : pools.history_layout,
            bank.sources, bank.destinations, {.resource_budget = cache_budget}));
        // The transaction now owns these source guards. Do not keep a second
        // Phase reader that would block old-head recovery after take/clear.
        for (auto &source : bank.sources) source = {};
      }
      auto &fixed_output = phase.hidden_output[slot];
      if (fixed_output.program.valid()) step.hidden_output = &fixed_output;
      std::size_t carry_index = 0;
      if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) for (const auto &ports : phase.carry_ports)
        for (std::size_t i = 0; i < phase.prefix_rows; ++i, ++carry_index)
          if (const auto &output = phase.child->output(ports[i]); output.caller_owned()) {
            // Still a leased Graph OUTPUT: original write/native-use admission
            // owns it until retirement/publication. Never bind writer.view().
            auto writer = fixed_output.program.valid() ? std::move(fixed_output.scratch[carry_index].writer)
                : take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment));
            check(run.bind(output.value, output.description, std::move(writer)));
          }
      std::vector<g::batch::NodeLayout> layouts;
      layouts.reserve(target_kv_layers + (drafts_token(role) ? 1 : 0));
      for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i) layouts.push_back({phase.attention_nodes[i], io.attention_layout});
      if (drafts_token(role)) layouts.push_back({phase.draft_attention_node, step.draft_layout});
      const auto queues = inference::queue_span(phase.queues);
      if (preparation) { preparation->slot = slot; preparation->phase_slot_prior_uses = phase.preparation_uses[slot]; }
      const auto batch_start = Clock::now();
      check(g::batch::prepare(run, queues, layouts, {.host_workspace = execution.staging()}));
      for (std::size_t layer = 0; layer < phase.attention_nodes.size(); ++layer) {
        const auto &ports = phase.layers[attention_layer_ids[layer]];
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2], ports.key,
            std::move(step.appends[layer * 2]))); step.bound[layer * 2] = true;
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2 + 1], ports.value,
            std::move(step.appends[layer * 2 + 1]))); step.bound[layer * 2 + 1] = true;
      }
      if (has_mtp(role)) for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        if (drafts_token(role)) check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank],
            bank == draft_bank ? phase.draft_key_port : phase.draft_value_port, std::move(step.appends[bank])));
        else check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank], std::move(step.appends[bank])));
        step.bound[bank] = true;
      }
      if (relay) {
        auto &input = *step.incoming; auto &previous = *input.owner;
        input.bindings.reserve(1 + previous.states.size() + previous.pages.size());
        input.bindings.push_back({phase.token_port, &previous.token});
        const bool ready_states = previous.published_states.size() == previous.states.size() * width &&
            std::ranges::all_of(previous.published_states, [](const auto &lease) { return lease.valid(); });
        for (std::size_t bank = 0; bank < previous.states.size(); ++bank)
          input.bindings.push_back(take(previous.states[bank].reader_binding(run, phase.state_graph(),
              phase.state_sources[bank].store, ready_states ? std::span<const c::ReadLease>(previous.published_states).subspan(bank * width, width)
                  : std::span<const c::ReadLease>{})));
        for (std::size_t bank = 0; bank < previous.pages.size(); ++bank)
          input.bindings.push_back(take(g::batch::detail::append_binding(run, phase.append_nodes[bank], previous.pages[bank])));
        for (auto value : phase.state_resource_ports) {
          const bool attached = std::ranges::any_of(input.bindings, [&](const auto &binding) {
            return binding.input == value || (binding.store && (binding.projection.capacity == value || binding.projection.metadata == value));
          });
          if (!attached) input.outside_inputs.push_back(value);
        }
        check(g::detail::bind_outputs(run, input.bindings)); input.bound = true;
      }
      check(g::slots::bind(run, phase.state_graph(), phase.state_requests[slot]));
      step.states_bound = !phase.state_sources.empty();
      const auto batch_end = Clock::now();
      if (relay) check(g::detail::prepare_outputs(run, step.incoming->bindings, queues));
      else check(run.prepare_replay(queues, {.mode = phase.replay_modes[slot], .host_wait = g::HostWaitPolicy::kAllow}));
      const auto replay_end = Clock::now();
      if (preparation) {
        preparation->layout_seconds = std::chrono::duration<double>(batch_start - cold_start).count();
        preparation->batch_prepare_seconds = std::chrono::duration<double>(batch_end - batch_start).count();
        preparation->replay_prepare_seconds = std::chrono::duration<double>(replay_end - batch_end).count();
      }
      step.replay_route = run.replay_route();
      const auto replay_report = take(run.replay_report());
      step.capture_recorded = replay_report.recorded_capture_range_count;
      step.capture_reused = replay_report.reused_capture_range_count;
      step.cold_batch_seconds = std::chrono::duration<double>(Clock::now() - cold_start).count();
      ++phase.preparation_uses[slot]; step.prepared = true;
    } catch (inference::EvaluationFailure &failure) {
      failure.observe_retirement(predecessor ? abort_successor_episode() : abort_episode());
      throw;
    } catch (...) { if (predecessor) (void)abort_successor_episode(); else (void)abort_episode(); throw; }
  }

  void prepare_evaluation(Session &session, std::span<const std::int32_t> ids,
                          inference::PreparationObservation *preparation = nullptr,
                          GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext,
                          const inference::RequestStopPolicy *stop_policy = nullptr, const LogitsInputs &logits = {}) {
    const std::array requests{EvaluationRequest{session.shared_from_this(), ids, stop_policy, logits}};
    prepare_evaluations(requests, preparation, role, completion);
  }

  StepState &prepare_relay_evaluations(std::span<const EvaluationRequest> requests,
      std::shared_ptr<StepSources> previous, std::size_t physical_slot) {
    prepare_evaluations(requests, nullptr, GraphRole::kText, TextCompletion::kEmitNext, std::move(previous), physical_slot);
    return successor_step ? *successor_step : *active_step;
  }

  // Reserve typed metadata before accepting native work. The caller can retain
  // the same original views for a successor; no candidate head is exported and
  // the ordinary API remains on its existing direct Completion path.
  std::shared_ptr<StepSources> prepare_submission_sources(StepState &step) {
    if (!step.prepared || step.submitted || step.cancelled || step.sources || !step.phase ||
        step.phase->family->role != GraphRole::kText || cache_plan.draft_cache ||
        step.phase->family->completion != TextCompletion::kEmitNext || scheduler)
      throw std::runtime_error("submission views require one prepared direct ordinary episode");
    auto &phase = *step.phase;
    const auto storage = execution.storage_owner();
    if (!storage || inference::queue_span(phase.queues).size() != 1 ||
        !phase.io[step.state_slot].next_id.pool().valid())
      throw std::runtime_error("submission views require the original owned shared Queue and full token writer");
    auto sources = std::make_shared<StepSources>();
    sources->states.reserve(phase.state_sources.size());
    sources->pages.reserve(draft_bank);
    sources->boundaries.reserve(step.members.size());
    sources->published_states.resize(phase.state_sources.size() * step.members.size());
    for (const auto &member : step.members) {
      if (member.base_epoch == UINT64_MAX) throw std::overflow_error("source candidate epoch exhausted");
      sources->boundaries.push_back({member.session, member.next.get(), member.base_epoch + 1});
    }
    for (const auto &state : phase.state_sources)
      sources->states.push_back(take(g::detail::StoreSource::prepare(*step.run,
          state.store, storage, {.resource_budget = execution_budget})));
    for (std::size_t bank = 0; bank < draft_bank; ++bank)
      sources->pages.push_back(take(g::batch::detail::PagedAppendSource::prepare(
          *step.run, step.nodes[bank], storage, {.resource_budget = execution_budget})));
    step.sources = sources;
    return sources;
  }

  std::shared_ptr<StepSources> prepare_submission_sources() {
    if (!active_step) throw std::runtime_error("submission views require a prepared episode");
    return prepare_submission_sources(*active_step);
  }

  void submit_evaluations(StepState &step, bool optional_sources = false) {
    if ((active_step.get() != &step && successor_step.get() != &step) || !step.prepared ||
        step.submitted || step.cancelled || !step.run)
      throw std::runtime_error("batch is not prepared or was already submitted/cancelled");
    for (const auto &member : step.members)
      if (member.session->stopping || member.session->failed)
        throw std::runtime_error("batch participant stopped before submission");
    const auto queues = inference::queue_span(step.phase->queues);
    step.start = Clock::now(); step.submitted = true;
    step.completion = submit_retained(*step.run, queues, step.phase->retirement[step.state_slot], step.incoming.get());
    if (step.sources) {
      auto &sources = *step.sources;
      auto exported = g::detail::SubmissionSource::create(
          *step.run, std::move(step.completion), execution.storage_owner());
      if (!exported && optional_sources && !exported.error().has_native_error() &&
          (exported.error().IsUnimplemented() || exported.error().IsUnavailable())) {
        // create transfers the original Completion only on success. A legal
        // command route can finish normally without opting into device relay.
        step.sources.reset();
      } else {
        sources.submission = take(std::move(exported));
        sources.token = take(g::detail::OutputSource::create(sources.submission,
            step.phase->id_port, execution.storage_owner(), step.phase->io[step.state_slot].next_id.pool()));
        for (auto &state : sources.states) check(state.capture(sources.submission));
        for (auto &pages : sources.pages) check(pages.capture(sources.submission));
      }
    }
    step.submit_seconds = std::chrono::duration<double>(Clock::now() - step.start).count();
  }

  void submit_evaluations() {
    if (!active_step) throw std::runtime_error("submission requires a prepared episode");
    submit_evaluations(*active_step);
  }

  void submit_evaluation(Session &session) {
    if (session.model_ != this || !active_step || session.pending != active_step.get() || active_step->members.size() != 1)
      throw std::runtime_error("single submission requires its original one-request episode");
    submit_evaluations();
  }

  template<class Select, class Complete, class Observe>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Complete complete, Observe observe) {
    if (!active_step || outputs.size() != active_step->members.size() || next_ids.size() != outputs.size() ||
        active_step->cancelled || !active_step->submitted || !active_step->has_completion())
      throw std::runtime_error("finish requires its original complete noncancelled batch");
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      if (outputs[j].session != active_step->members[j].session || outputs[j].session->failed ||
          (outputs[j].session->stopping && !active_step->discard_stopped_members))
        throw std::runtime_error("finish participant does not match the original active batch");
      if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kNotPublished;
    }
    auto &step = *active_step; auto &phase = *step.phase; auto &run = *step.run;
    auto &io = phase.io[step.state_slot];
    const auto step_start = step.step_start, start = step.start;
    const auto tokens = step.input_tokens;
    const bool state_only = phase.family->completion == TextCompletion::kStateOnly;
    const auto submit_seconds = step.submit_seconds, cold_batch_seconds = step.cold_batch_seconds;
    const auto shape_admission_seconds = step.shape_admission_seconds;
    const auto replay_route = step.replay_route;
    const auto capture_recorded = step.capture_recorded, capture_reused = step.capture_reused;
    std::array<c::ConstTensorView, 9> probe_views;
    std::array<c::ReadLease, 9> probe_owners;
    c::ReadLease generated_logits, generated_hidden;
    // An empty local mask means all members publish. Allocate only for an
    // actual withdrawal, before any publication; ordinary R>2 adds no buffer.
    // It stays owned here through detach and all output observers.
    hyle::container::SmallVector<std::uint8_t, 2> published_members;
    struct ReturnReadback final {
      std::size_t *admission = nullptr;
      ~ReturnReadback() { if (admission) { HYLE_FATAL_ASSERT(*admission); --*admission; } }
    } return_readback;
    try {
      if (step.discard_stopped_members &&
          !published_members.try_resize(outputs.size(), std::uint8_t{1})) throw std::bad_alloc{};
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        if (outputs[j].generation_ledger) {
          if (cache_plan.draft_cache != (phase.family->role != GraphRole::kText) ||
              step.members[j].base->draft.has_value() != cache_plan.draft_cache)
            throw std::runtime_error("generation role differs from its complete target/draft participant");
          for (std::size_t k = 0; k < j; ++k) if (outputs[k].generation_ledger == outputs[j].generation_ledger)
            throw std::runtime_error("batch participants cannot share a mutable generation ledger");
          if (outputs[j].generation_ledger->stop_policy() !=
              inference::request_stop_policy(step.members[j].stop_policy))
            throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
          auto &record = outputs[j].generation_ledger->next();
          step.members[j].commit_ledger = outputs[j].generation_ledger; step.members[j].commit_record = &record;
          if (record.role != phase.family->role || record.completion != phase.family->completion || record.before != generation_boundary(*step.members[j].base) ||
              record.ids().size() != (record.generation ? 1u : tokens) ||
              record.ids().empty() || record.ids().front() != step.members[j].first_input)
            throw std::runtime_error("generation ledger entry differs from its actual episode");
          if (outputs[j].expected_episode && !outputs[j].expected_episode->committed)
            throw std::runtime_error("expected episode record is not committed");
          if (step.discard_stopped_members && outputs[j].session->stopping) {
            step.members[j].commit_ledger = nullptr; step.members[j].commit_record = nullptr;
            outputs[j].generation_ledger->discard_uncommitted();
          }
        } else if (outputs[j].expected_episode)
          throw std::runtime_error("expected episode record requires a fresh commit witness");
      }
      check(step.wait()); step.completion = {};
      if (step.sources) check(step.sources->token.detach());
      // Trace storage must be pinned before take_append/unbind revoke Graph
      // output observation. Generated arrays keep their actual native owner.
      if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_tensors; })) {
        if (!state_only && !io.logits.caller_owned()) generated_logits = take(inference::retain_output(run, *phase.logits_facts, allocation_budget));
        if (!io.final_hidden.caller_owned()) generated_hidden = take(inference::retain_output(run, *phase.hidden_facts, allocation_budget));
      }
      if (drafts_token(phase.family->role)) {
        check(io.draft_id.read(execution, run, std::as_writable_bytes(std::span(io.draft_ids))));
        for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].draft_token = io.draft_ids[j];
      }
      if (phase.sampling_valid_port && !state_only && phase.family->role != GraphRole::kVerify) {
        check(io.sampling_valid.predicates(execution, run, io.sampling_valid_values, io.sampling_scratch));
        auto bad = std::ranges::find_if(io.sampling_valid_values, [](auto value) { return value != 1; });
        if (step.discard_stopped_members)
          while (bad != io.sampling_valid_values.end() &&
              step.members[static_cast<std::size_t>(bad - io.sampling_valid_values.begin())].session->stopping)
            bad = std::find_if(bad + 1, io.sampling_valid_values.end(), [](auto value) { return value != 1; });
        if (bad != io.sampling_valid_values.end()) {
          if (phase.family->role == GraphRole::kText) {
            std::optional<std::size_t> participant;
            // Everything before the first bad value is already proven 1.
            // Only the failure path inspects the remainder for a unique 0.
            bool unique = true;
            for (auto it = bad + 1; it != io.sampling_valid_values.end(); ++it)
              if (*it != 1 && !(step.discard_stopped_members &&
                  step.members[static_cast<std::size_t>(it - io.sampling_valid_values.begin())].session->stopping)) unique = false;
            if (*bad == 0 && unique)
              participant = static_cast<std::size_t>(bad - io.sampling_valid_values.begin());
            throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
                "sampling distribution or proposal is invalid; no session or RNG cursor was published"), participant);
          }
          throw std::runtime_error("sampling distribution or proposal is invalid; no session or RNG cursor was published");
        }
        if (drafts_token(phase.family->role)) {
          check(io.sampling_accept.predicates(execution, run, io.sampling_accept_values, io.sampling_scratch));
          for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].sampled_accept = io.sampling_accept_values[j] != 0;
        }
      }
      // Selection observes only retired original Graph results. No head has
      // been published and all state/append guards still belong to this episode.
      if (verifies_prefix(phase.family->role)) {
        check(io.next_id.read(execution, run, std::as_writable_bytes(std::span(io.prefix_candidates))));
        select(std::span<const std::int32_t>(io.prefix_candidates), step, next_ids);
      } else if (state_only) {
        std::ranges::fill(next_ids, -1);
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      } else {
        check(io.next_id.read(execution, run, std::as_writable_bytes(next_ids)));
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      }
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        auto &member = step.members[j];
        if (!member.accepted_tokens || member.accepted_tokens > tokens ||
            (member.accepted_tokens != tokens && !verifies_prefix(phase.family->role)) ||
            (!state_only && !(step.discard_stopped_members && member.session->stopping) &&
                (next_ids[j] < 0 || next_ids[j] >= vocabulary)))
          throw std::runtime_error("invalid accepted target boundary");
        member.verified_last_page = member.next->pages.back().page();
        if (member.accepted_tokens != tokens) {
          member.next->length = member.base->length + member.accepted_tokens;
          const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
          member.next->pages.resize(page_count);
          std::vector<std::int32_t> ids;
          ids.reserve(page_count);
          for (const auto &page : member.next->pages) ids.push_back(static_cast<std::int32_t>(page.page()));
          const std::array<std::int32_t, 2> query{0, static_cast<std::int32_t>(member.accepted_tokens)},
              kv{0, static_cast<std::int32_t>(page_count)};
          const std::array last{static_cast<std::int32_t>((member.next->length - 1) % cache::Plan::page_rows + 1)};
          member.layout = take(attention_schema.version({query, kv, ids, last}));
        }
      }
      // Export every participant's ordinary outputs before taking ANY append:
      // the first take_append revokes Graph output-ready for the entire batch.
      copy_selected_hidden_outputs(step);
      if (!step.hidden_output && (verifies_prefix(phase.family->role) || has_mtp(phase.family->role) || retains_sampling_frontier()))
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          const auto &ports = phase.carry_ports[j];
          member.next->carry_hidden = retain_state_storage(run,
              phase.child->output(ports[verifies_prefix(phase.family->role) ? member.accepted_tokens - 1 : 0]));
          if (verifies_prefix(phase.family->role) && member.accepted_tokens > 1) {
            member.next->prior_hidden = retain_state_storage(run, phase.child->output(ports[member.accepted_tokens - 2]));
          }
          if (phase.family->role == GraphRole::kVerify && cache_plan.draft_cache) {
            member.carry_prefix.reserve(member.accepted_tokens);
            for (std::size_t row = 0; row < member.accepted_tokens; ++row)
              member.carry_prefix.push_back(retain_state_storage(run, phase.child->output(ports[row])));
          }
        }
      if (has_mtp(phase.family->role)) for (auto &member : step.members) {
        auto &draft = *member.next->draft;
        draft.lag = static_cast<std::uint32_t>(member.next->length - draft.frontier);
        draft.catchup_token = draft.lag ? member.draft_token : -1;
      }
      if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_probes; }))
        for (std::size_t i = 0; i < probe_views.size(); ++i) {
          const auto &output = *phase.probe_facts[i];
          if (output.caller_owned()) probe_views[i] = take(run.output(output.value));
          else {
            probe_owners[i] = take(inference::retain_output(run, output, allocation_budget));
            probe_views[i] = take(probe_owners[i].tensor(output.description));
          }
        }
      // Attest the original append success before any slot withdrawal revokes
      // Graph output-ready. Every ordinary output has already been retained.
      if (step.sources) check(step.sources->submission.detach());
      const auto append_banks = draft_bank + (has_mtp(phase.family->role) ? 2u : 0u);
      check(g::batch::take_appends(run, std::span(step.nodes.data(), append_banks),
          std::span(step.appends.data(), append_banks)));
      std::fill_n(step.bound.begin(), append_banks, std::uint8_t{});
      // Slot take proves actual retirement, not success; the original whole
      // Graph completion above remains the success witness for every bank.
      for (std::size_t bank_index = 0; bank_index < phase.state_bindings[step.state_slot].size(); ++bank_index) {
        auto &bank = phase.state_bindings[step.state_slot][bank_index];
        bank.transaction = take(g::slots::take(run, phase.state_sources[bank_index].store));
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          const auto prefix = verifies_prefix(phase.family->role) ? step.members[j].accepted_tokens - 1 : 0;
          const auto selected = j * phase.prefix_rows + prefix;
          auto ready = step.sources ? take(step.sources->states[bank_index].publish(
              selected, bank.transaction.destinations()[selected]))
              : take(bank.transaction.destinations()[selected].publish());
          if (step.sources) step.sources->published_states[bank_index * outputs.size() + j] = ready;
          auto &state = step.members[j].next->state[bank.layer];
          (bank.recurrent ? state.recurrent : state.history) = std::move(ready);
        }
        // Release old source guards before the joint head-commit recycle hook.
        // Unaccepted MTP candidates are discarded, never made visible.
        bank.transaction.clear();
      }
      step.states_bound = false;
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          if (!member.key_updates.try_resize(page_inventory.keys.size()) ||
              !member.value_updates.try_resize(page_inventory.keys.size())) throw std::bad_alloc{};
          auto key = step.sources ? take(step.sources->pages[layer * 2].ready_delta(
              step.appends[layer * 2], take(member.frame(layer, true).view()), member.layout))
              : take(step.appends[layer * 2].ready_delta(take(member.frame(layer, true).view()), member.layout));
          auto value = step.sources ? take(step.sources->pages[layer * 2 + 1].ready_delta(
              step.appends[layer * 2 + 1], take(member.frame(layer, false).view()), member.layout))
              : take(step.appends[layer * 2 + 1].ready_delta(take(member.frame(layer, false).view()), member.layout));
          member.key_updates[layer] = take(member.frame(layer, true).prepare_commit(std::move(key)));
          member.value_updates[layer] = take(member.frame(layer, false).prepare_commit(std::move(value)));
        }
      }
      if (has_mtp(phase.family->role)) {
        std::vector<std::pair<std::uint32_t, std::size_t>> owners; owners.reserve(outputs.size());
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j]; auto &draft = *member.next->draft;
          auto key = take(step.appends[draft_bank].ready_delta(take(member.draft_frame(true).view()), member.draft_layout));
          auto value = take(step.appends[draft_bank + 1].ready_delta(take(member.draft_frame(false).view()), member.draft_layout));
          member.draft_updates[0] = take(member.draft_frame(true).prepare_commit(std::move(key)));
          member.draft_updates[1] = take(member.draft_frame(false).prepare_commit(std::move(value)));
          owners.emplace_back(draft.pages.back().page(), j);
        }
        std::sort(owners.begin(), owners.end());
        for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank)
          for (std::size_t i = 0; i < step.appends[bank].appender_count(); ++i) {
            auto &tail = take(step.appends[bank].appender(i)); if (!tail.valid()) continue;
            const auto found = std::lower_bound(owners.begin(), owners.end(), std::pair{tail.page(), std::size_t{0}});
            if (found == owners.end() || found->first != tail.page())
              throw std::runtime_error("draft append returned a tail outside all final pages");
            auto &member = step.members[found->second]; auto &draft = *member.next->draft;
            if (tail.reserved_rows() || member.next_draft_tails[bank - draft_bank].valid() ||
                tail.initialized_rows() != draft.frontier % cache::Plan::page_rows)
              throw std::runtime_error("draft append returned an invalid committed frontier");
            member.next_draft_tails[bank - draft_bank] = std::move(tail);
          }
      }
      // Caller-supplied completion is confined to this unpublished episode.
      // In particular, recursive proposals must repair their retained KV from
      // actual target hidden rows here, before ledger checks or any head swap.
      complete(step);
      if (cache_plan.draft_cache)
        for (const auto &member : step.members) validate_mtp_snapshot(*member.next, &member);
      for (std::size_t j = 0; j < outputs.size(); ++j) if (auto *record = step.members[j].commit_record) {
        const auto &member = step.members[j];
        record->requests = outputs.size(); record->member_index = j;
        record->accepted = member.accepted_tokens;
        record->draft_rows = member.next->draft
            ? member.next->draft->frontier - member.base->draft->frontier : 0;
        if (phase.family->role == GraphRole::kVerify)
          record->draft_rows = cache::add(record->draft_rows,
              tokens - 1 + member.base->draft->lag + member.discarded_draft_rows);
        record->after = generation_boundary(*member.next); record->after.next_id = next_ids[j];
        if (phase.family->role == GraphRole::kVerify)
          record->accepted_inputs.assign(member.verification_inputs.begin(),
              member.verification_inputs.begin() + member.accepted_tokens);
        else record->accepted_inputs = {member.first_input, member.accepted_tokens == 2 ? member.draft_token : -1};
        if (record->generation) {
          if (phase.family->role == GraphRole::kVerify) record->verification_inputs.assign(
              member.verification_inputs.begin(), member.verification_inputs.end());
          else if (drafts_token(phase.family->role)) record->verification_inputs = {member.first_input, member.draft_token};
          else record->verification_inputs = {member.first_input};
        }
        record->verified_rows = tokens;
        if (!record->generation || phase.family->role == GraphRole::kText) {
          record->draft = -1; record->target_first = next_ids[j]; record->target_second = -1;
          record->agreed = false; record->accepted_draft = false;
          record->stopped = !state_only && (phase.family->role == GraphRole::kText
              ? inference::request_stop_policy(member.stop_policy).stopped(next_ids[j], is_end_of_generation(next_ids[j]))
              : is_end_of_generation(next_ids[j]));
          if (state_only) { record->emitted.clear(); record->emitted_count = 0; }
          else { record->emitted = {next_ids[j], -1}; record->emitted_count = 1; }
        }
        if (outputs[j].expected_episode && !record->same_result(*outputs[j].expected_episode))
          throw std::runtime_error("MTP replay natural decision or state boundary diverged from its recorded episode");
      }
      hyle::container::SmallVector<std::pair<std::uint32_t, std::size_t>, 1> tail_owners;
      if (!tail_owners.try_reserve(step.members.size())) throw std::bad_alloc{};
      for (std::size_t j = 0; j < step.members.size(); ++j)
        tail_owners.emplace_back(step.members[j].verified_last_page, j);
      std::sort(tail_owners.begin(), tail_owners.end());
      for (std::size_t bank = 0; bank < draft_bank; ++bank)
        for (std::size_t i = 0; i < step.appends[bank].appender_count(); ++i) {
          auto &tail = take(step.appends[bank].appender(i)); if (!tail.valid()) continue;
          const auto found = std::lower_bound(tail_owners.begin(), tail_owners.end(), std::pair{tail.page(), std::size_t{0}});
          if (found == tail_owners.end() || found->first != tail.page())
            throw std::runtime_error("batch append returned a tail outside all final pages");
          auto &member = step.members[found->second];
          // A ready T2 tail already owns the verified physical frontier.
          // Partial acceptance only narrows read authority; discard its mutable
          // capability and use the original COW path next, never rewind it.
          if (member.accepted_tokens != tokens) { tail = {}; continue; }
          if (tail.reserved_rows() || member.next_tails[bank].valid() || tail.initialized_rows() != member.next->length % cache::Plan::page_rows)
            throw std::runtime_error("batch append returned an invalid committed tail");
          member.next_tails[bank] = std::move(tail);
        }
      if (step.incoming) {
        check(g::detail::release_outputs(run, step.incoming->bindings));
        check(run.unbind(step.incoming->outside_inputs));
        step.incoming.reset(); step.predecessor_episode.reset();
      } else check(run.unbind(phase.state_resource_ports));
      if (step.hidden_output) {
        check(step.hidden_output->restore_writers());
        step.hidden_output = nullptr;
      }
      for (const auto &member : step.members)
        if (member.session->head.get() != member.base || member.session->head.epoch() != member.base_epoch ||
            member.session->pending != &step ||
            (member.session->stopping && !step.discard_stopped_members))
          throw std::runtime_error("batch snapshot expected-base conflict");
      for (const auto &member : step.members) member.preflight_frames();
      // All fallible native/export/metadata work is complete. The head swaps
      // below cannot throw. Stopped ordinary members discard their unpublished
      // candidate; every surviving member keeps this original successful work.
      for (std::size_t j = 0; j < step.members.size(); ++j) {
        auto &member = step.members[j];
        if (step.discard_stopped_members && member.session->stopping) {
          published_members[j] = 0;
          if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kCancelled;
          continue;
        }
        if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kPublished;
        member.next->next_id = next_ids[j];
        member.session->tails = std::move(member.next_tails);
        member.session->draft_tails = std::move(member.next_draft_tails);
        member.session->head.commit(std::move(member.next),
            [&](cache::State *previous, cache::State &candidate) noexcept { member.publish_frames(previous, candidate); },
            [&](cache::State &previous) noexcept {
          // Only an exclusively owned successful old working head reaches
          // here, after Graph withdrawal. Frozen checkpoints stay untouched;
          // try_write also rejects escaped readers and original native uses.
          const auto count = (std::min)(previous.state.size(), state_pools.size());
          for (std::size_t i = 0; i < count; ++i) {
            state_pools[i].history_reuse.recycle(previous.state[i].history);
            state_pools[i].recurrent_reuse.recycle(previous.state[i].recurrent);
          }
        });
        member.session->tail_epoch = member.session->head.epoch();
        member.session->draft_tail_epoch = member.session->head.epoch();
      }
      // Every bound and expected decision was checked before publication.
      // The reserved host slot is now authoritative even if trace/report IO
      // below throws. No callback, allocation or permission is acquired here.
      for (auto &member : step.members) if (member.commit_record) {
        member.commit_record->committed = true;
        ++member.commit_ledger->committed_count;
      }
      return_readback.admission = std::exchange(step.candidate_admission, nullptr);
      detach_episode(); // Old states/transactions retire inside the host endpoint.
    } catch (inference::EvaluationFailure &failure) {
      failure.observe_retirement(abort_episode());
      throw;
    } catch (...) { (void)abort_episode(); throw; }
    const auto finished = Clock::now();
    for (std::size_t j = 0; j < outputs.size(); ++j) if ((published_members.empty() || published_members[j]) && outputs[j].generation_ledger) {
      const auto &output = outputs[j];
      auto &record = output.generation_ledger->records[output.generation_ledger->committed_count - 1];
      record.endpoint = finished;
      record.step_seconds = std::chrono::duration<double>(finished - step_start).count();
    }
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      if (!published_members.empty() && !published_members[j]) continue;
      const auto &output = outputs[j]; const auto &session = *output.session; const auto next = next_ids[j];
      if (output.complete_seconds) *output.complete_seconds = std::chrono::duration<double>(finished - step_start).count();
      if (output.host_endpoint) *output.host_endpoint = finished;
      if (output.preparation) {
        const auto report_start = Clock::now();
        output.preparation->nodes.resize(phase.attention_nodes.size());
        output.preparation->reports.resize(phase.attention_nodes.size());
        for (std::size_t i = 0; i < output.preparation->reports.size(); ++i) {
          output.preparation->nodes[i] = phase.attention_nodes[i];
          output.preparation->reports[i] = take(g::batch::report(run, phase.attention_nodes[i]));
        }
        output.preparation->report_snapshot_seconds = std::chrono::duration<double>(Clock::now() - report_start).count();
      }
      inference::StepObservation observed;
      auto &observation = output.observation ? *output.observation : observed;
      observation.input_tokens = tokens; observation.visible = session.length(); observation.next = next;
      observation.graph_executed = true; observation.graph_input_tokens = tokens;
      observation.seconds = std::chrono::duration<double>(finished - start).count();
      observation.submit_seconds = submit_seconds; observation.step_seconds = std::chrono::duration<double>(finished - step_start).count();
      observation.cold_batch_seconds = cold_batch_seconds; observation.replay_route = replay_route;
      observation.shape_admission_seconds = shape_admission_seconds;
      observation.capture_recorded = capture_recorded; observation.capture_reused = capture_reused;
      observation.scheduled_progress = static_cast<bool>(scheduler);
      CompletedEvaluationView completed{.observation = observation, .head = *session.head,
          .member_index = j, .requests = outputs.size()};
      if (output.capture_tensors) {
        if (!state_only) completed.logits = io.logits.caller_owned() ? io.logits.view().as_const()
            : take(generated_logits.tensor(io.logits.description));
        completed.hidden = io.final_hidden.caller_owned() ? io.final_hidden.view().as_const()
            : take(generated_hidden.tensor(io.final_hidden.description));
      }
      if (output.capture_probes) completed.probes = probe_views;
      observe(j, completed);
    }
  }

  template<class Select, class Observe>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Observe observe) {
    finish_selected_evaluations(outputs, next_ids, std::move(select), [](StepState &) {}, std::move(observe));
  }

  // Returns the actual verifier input [pending, proposal0, ...]. Approximate
  // hidden is local to this recursion and is deliberately discarded with its
  // private pages; it is never installed in a committed draft cache.
  std::vector<std::int32_t> propose_recursive(Session &session, std::size_t maximum,
      std::span<const std::int32_t> eos = {}, std::vector<c::ReadLease> *q = nullptr) {
    if (active_step || carrier_borrowed || session.pending || session.stopping || session.failed ||
        !session.head || !session.head->draft || (default_sampler.enabled && !q) ||
        maximum >= capacity - session.length() || maximum >= cache_plan.candidate_boundaries)
      throw std::runtime_error("recursive proposals require an idle prepared head, bounded width and every stochastic q row");
    validate_mtp_snapshot(*session.head);
    const auto ended = [&](auto token) { return std::find(eos.begin(), eos.end(), token) != eos.end(); };
    std::vector<std::int32_t> inputs; inputs.reserve(maximum + 1); inputs.push_back(session.head->next_id);
    if (q) { q->clear(); q->reserve(maximum); }
    if (!maximum) return inputs;
    if (inputs.front() < 0 || inputs.front() >= vocabulary || ended(inputs.front()))
      throw std::runtime_error("recursive proposals require a valid nonterminal emitted pending token");
    DraftWork work; work.state = fork_draft(*session.head->draft);
    c::ReadLease approximate;
    for (std::size_t i = 0; i < maximum; ++i) {
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (!i && work.state.lag) { ids.push_back(work.state.catchup_token); shifted.push_back(session.head->prior_hidden); }
      ids.push_back(inputs.back()); shifted.push_back(i ? approximate : session.head->carry_hidden);
      auto proposal = execute_draft(work, ids, shifted, true,
          default_sampler.enabled ? session.head->sampling.identity(2 * i) : std::array<std::uint32_t, 4>{});
      // Stop on a distribution property, not the sampled proposal's own
      // probability. The stopping condition is independent of this draw; its
      // unpublished address may then be the target's final draw at 2*m.
      if (min_draft_probability() > 0 && proposal.probability < min_draft_probability()) break;
      inputs.push_back(proposal.id); approximate = std::move(proposal.hidden);
      if (q && default_sampler.enabled) q->push_back(std::move(proposal.log_probabilities));
      if (ended(proposal.id)) break;
    }
    return inputs;
  }

  void repair_recursive_draft(StepState &step) {
    if (step.phase->family->role != GraphRole::kVerify || !cache_plan.draft_cache)
      throw std::runtime_error("draft repair requires its unpublished target verifier");
    for (auto &member : step.members) {
      const auto n = member.accepted_tokens;
      if (!n || member.verification_inputs.size() < n || member.carry_prefix.size() != n)
        throw std::runtime_error("draft repair requires every actually consumed target prefix row");
      DraftWork repaired; repaired.state = fork_draft(*member.base->draft);
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (repaired.state.lag) { ids.push_back(repaired.state.catchup_token); shifted.push_back(member.base->prior_hidden); }
      // A one-row target ends at lag0. Longer accepted prefixes retain the
      // canonical lag1 frontier, with the final consumed token left to catch up.
      const auto retained = n == 1 ? 1 : n - 1;
      for (std::size_t i = 0; i < retained; ++i) {
        ids.push_back(member.verification_inputs[i]);
        shifted.push_back(i ? member.carry_prefix[i - 1] : member.base->carry_hidden);
      }
      (void)execute_draft(repaired, ids, shifted, false);
      repaired.state.lag = n == 1 ? 0u : 1u;
      repaired.state.catchup_token = n == 1 ? -1 : member.verification_inputs[n - 1];
      member.next->draft = std::move(repaired.state);
      member.next_draft_tails = std::move(repaired.tails);
      if (member.next->draft->frontier + member.next->draft->lag != member.next->length)
        throw std::runtime_error("draft repair did not reach the selected target prefix");
    }
  }

  template<class Observe>
  void finish_recursive_mtp(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe) {
    if (!active_step || selections.empty() || active_step->phase->family->role != GraphRole::kVerify ||
        !cache_plan.draft_cache || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size())
      throw std::runtime_error("recursive finish requires the complete original verification batch");
    const auto ended = [&](auto token) { return std::find(eos.begin(), eos.end(), token) != eos.end(); };
    std::vector<std::int32_t> next(selections.size(), -1);
    for (std::size_t j = 0; j < selections.size(); ++j) {
      const auto &selection = selections[j];
      if (!selection.output_quota || selection.session != active_step->members[j].session ||
          selection.session != outputs[j].session || ended(selection.expected_pending))
        throw std::runtime_error("recursive generation requires its nonterminal pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(active_step->input_tokens);
    }
    finish_selected_evaluations(outputs, next,
        [&](std::span<const std::int32_t> candidates, StepState &step, std::span<std::int32_t> selected) {
      c::ReadLease logits_owner;
      mem::ConstBufferView logits_storage;
      if (default_sampler.enabled) {
        auto &logits = step.phase->io[step.state_slot].logits;
        if (logits.caller_owned()) logits_storage = logits.storage();
        else {
          logits_owner = take(inference::retain_output(*step.run, *step.phase->logits_facts, allocation_budget));
          logits_storage = take(logits_owner.view());
        }
      }
      for (std::size_t j = 0; j < selections.size(); ++j) {
        auto &member = step.members[j]; auto &result = results[j]; const auto &selection = selections[j];
        if (member.first_input != selection.expected_pending || selection.expected_pending != member.base->next_id)
          throw std::runtime_error("recursive verifier input differs from the emitted pending token");
        const auto width = step.input_tokens;
        const auto rows = candidates.subspan(j * width, width);
        result.pending = selection.expected_pending; result.proposal_count = width - 1; result.verified_rows = width;
        result.auxiliary_graph_calls = width + (default_sampler.enabled ? 1u : 0u);
        result.draft = width > 1 ? member.verification_inputs[1] : -1;
        result.target_first = rows[0]; result.target_second = width > 1 ? rows[1] : -1;
        hyle::inference::mtp::AcceptedPrefix accepted;
        if (default_sampler.enabled) {
          if (selection.draft_log_probabilities.size() != width - 1)
            throw std::runtime_error("stochastic verification requires every actual q row");
          const auto bytes = cache::multiply(cache::multiply(width, vocabulary), sizeof(float));
          const auto logits = take(c::ConstTensorView::create(dense(Type::kFloat32,
              {static_cast<std::int64_t>(width), vocabulary}), take(logits_storage.subview(j * bytes, bytes))));
          const auto decision = sample_recursive_prefix(logits,
              std::span<const std::int32_t>(member.verification_inputs).subspan(1),
              selection.draft_log_probabilities, member.base->sampling);
          accepted = hyle::inference::mtp::select_sampled_prefix(member.verification_inputs, decision.accepted, decision.final_ids,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
          member.next->sampling = member.base->sampling.advanced(cache::add(cache::multiply(2, width - 1), 1));
          result.target_first = result.emitted.front();
          result.target_second = result.emitted.size() > 1 ? result.emitted[1] : -1;
        } else {
          accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, rows,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
        }
        member.accepted_tokens = accepted.consumed; selected[j] = result.next_id = accepted.next_id;
        result.agreed = accepted.first_agrees; result.consumed_drafts = accepted.consumed_drafts;
        const bool clipped = min_draft_probability() > 0 && selection.proposal_limit > width - 1 &&
            !ended(member.verification_inputs.back());
        member.discarded_draft_rows = clipped ? (width == 1 ? member.base->draft->lag + 1 : 1) : 0;
        result.auxiliary_graph_calls += clipped ? 1u : 0u;
        result.accepted_draft = result.consumed_drafts != 0;
        result.processed_target = member.accepted_tokens;
        result.processed_draft = width - 1 + 2 * member.base->draft->lag +
            (member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1) + member.discarded_draft_rows;
        result.stopped = accepted.stopped;
        if (outputs[j].generation_ledger) {
          auto &record = outputs[j].generation_ledger->next();
          if (!record.generation || record.quota != selection.output_quota)
            throw std::runtime_error("recursive MTP ledger differs from its original output quota");
          record.draft = result.draft; record.target_first = result.target_first;
          record.target_second = result.target_second; record.agreed = result.agreed;
          record.accepted_draft = result.accepted_draft; record.consumed_drafts = result.consumed_drafts;
          record.auxiliary_graph_calls = result.auxiliary_graph_calls;
          record.stopped = result.stopped; record.emitted.assign(result.emitted.begin(), result.emitted.end());
          record.emitted_count = result.emitted.size();
        }
      }
    }, [&](StepState &step) { repair_recursive_draft(step); }, std::move(observe));
  }

  template<class Observe>
  void finish_evaluations(std::span<const EvaluationOutput> outputs,
                         std::span<std::int32_t> next_ids, Observe observe) {
    if (active_step && active_step->phase && verifies_prefix(active_step->phase->family->role))
      throw std::runtime_error("verification requires explicit accepted-boundary selection");
    finish_selected_evaluations(outputs, next_ids,
        [](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> next) {
          for (std::size_t j = 0; j < step.members.size(); ++j) {
            step.members[j].accepted_tokens = step.input_tokens; next[j] = ids[j];
          }
        }, std::move(observe));
  }

  // Generation policy observes only the original terminal graph. Pending has
  // already been emitted; an accepted draft and its bonus are new outputs.
  template<class Observe>
  void finish_mtp_rounds(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe) {
    if (!active_step || selections.empty() || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size() || !active_step->phase ||
        !has_mtp(active_step->phase->family->role) || is_mtp_prefill(active_step->phase->family->role))
      throw std::runtime_error("MTP finish requires its original complete generation batch");
    const auto role = active_step->phase->family->role;
    const auto ended = [&](std::int32_t id) { return std::find(eos.begin(), eos.end(), id) != eos.end(); };
    for (std::size_t j = 0; j < selections.size(); ++j) {
      if (!selections[j].output_quota || selections[j].session != active_step->members[j].session ||
          selections[j].session != outputs[j].session || selections[j].expected_pending < 0 ||
          selections[j].expected_pending >= vocabulary || ended(selections[j].expected_pending))
        throw std::runtime_error("MTP generation requires its emitted pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(2);
      results[j].pending = selections[j].expected_pending;
    }
    hyle::container::SmallVector<std::int32_t, 2> next;
    if (!next.try_resize(selections.size(), -1)) throw std::bad_alloc{};
    finish_selected_evaluations(outputs, std::span(next.data(), next.size()),
        [&](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> selected) {
          for (std::size_t j = 0; j < selections.size(); ++j) {
            const auto &selection = selections[j]; auto &member = step.members[j]; auto &result = results[j];
            if (member.first_input != selection.expected_pending)
              throw std::runtime_error("MTP generation input differs from its emitted pending token");
            result.target_first = ids[drafts_token(role) ? 2 * j : j]; result.draft = member.draft_token;
            result.target_second = -1;
            if (drafts_token(role)) {
              if (result.draft < 0 || result.draft >= vocabulary) throw std::runtime_error("MTP draft ID outside vocabulary");
              result.target_second = ids[2 * j + 1];
            }
            const auto rows = drafts_token(role) ? std::size_t{2} : std::size_t{1};
            const std::array inputs{selection.expected_pending, result.draft};
            const auto verified_inputs = std::span<const std::int32_t>(inputs).first(rows);
            hyle::inference::mtp::AcceptedPrefix accepted;
            if (default_sampler.enabled) {
              const std::array flags{std::uint8_t(member.sampled_accept)};
              // The original K1 graph's second ID is its residual/bonus final;
              // its first ID is already Select(accept, proposal, final).
              const std::array finals{drafts_token(role) ? result.target_second : result.target_first};
              accepted = hyle::inference::mtp::select_sampled_prefix(verified_inputs,
                  std::span<const std::uint8_t>(flags).first(rows - 1), finals,
                  selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
            } else {
              const std::array targets{result.target_first, result.target_second};
              accepted = hyle::inference::mtp::select_greedy_prefix(verified_inputs,
                  std::span<const std::int32_t>(targets).first(rows), selection.output_quota, eos,
                  static_cast<std::int32_t>(vocabulary), result.emitted);
            }
            member.accepted_tokens = accepted.consumed;
            selected[j] = result.next_id = accepted.next_id;
            result.agreed = accepted.first_agrees; result.accepted_draft = accepted.consumed_drafts != 0;
            result.processed_target = accepted.consumed; result.processed_draft = draft_lag(role) + 1;
            result.proposal_count = drafts_token(role) ? 1 : 0;
            result.consumed_drafts = accepted.consumed_drafts; result.verified_rows = step.input_tokens;
            result.stopped = accepted.stopped;
            if (outputs[j].generation_ledger) {
              auto &record = outputs[j].generation_ledger->next();
              if (!record.generation || record.quota != selection.output_quota)
                throw std::runtime_error("MTP ledger quota differs from its original selection");
              record.draft = result.draft; record.target_first = result.target_first;
              record.target_second = result.target_second; record.agreed = result.agreed;
              record.accepted_draft = result.accepted_draft; record.stopped = result.stopped;
              record.consumed_drafts = result.consumed_drafts;
              record.emitted = {result.emitted[0], result.emitted.size() == 2 ? result.emitted[1] : -1};
              record.emitted_count = result.emitted.size();
            }
          }
        }, std::move(observe));
  }

};

} // namespace qwen35
#include "session_transfer.h"
namespace qwen35 {

// Entry IO runs synchronously inside Model's original completed-output scope.
// Neither this view nor its borrowed storage may escape or reenter preparation.
void observe_completed_evaluation(Model &model, const EvaluationOutput &output,
    const EvaluationIO &io, const CompletedEvaluationView &completed) {
  auto &observation = completed.observation;
  if (!output.observation) { observation.name = io.name; hyle::engine::reporting::write_step_observation(std::cerr, observation); }
  const auto dump_bytes = [&](mem::ConstBufferView view, const std::filesystem::path &path) {
    std::vector<std::byte> bytes(view.size()); check(model.execution.read(view, bytes));
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot write Qwen trace: " + path.string());
  };
  if (io.trace && !io.trace->empty()) {
    const auto directory = *io.trace / io.name; std::filesystem::create_directories(directory);
    const auto dump_row = [&](c::ConstTensorView tensor, const char *file) {
      const auto bytes = take(tensor.description().required_bytes()) / completed.requests;
      dump_bytes(take(tensor.storage().subview(completed.member_index * bytes, bytes)), directory / file);
    };
    if (completed.logits.valid()) dump_row(completed.logits, "logits.f32");
    dump_row(completed.hidden, model.precision.is_f16() ? "hidden.f16" : "hidden.bf16");
    if (io.trace_states) {
      for (std::size_t i = 0; i < model.state_schema.size(); ++i) if (model.geometry.is_gdn(i)) {
        const auto prefix = "layer" + std::to_string(i);
        dump_bytes(take(completed.head.state[i].history.view()), directory / (prefix + (model.precision.is_f16() ? ".conv.f16" : ".conv.bf16")));
        dump_bytes(take(completed.head.state[i].recurrent.view()), directory / (prefix + ".gdn.f32"));
      }
      std::ofstream layout_trace(directory / "kv-layout.json");
      layout_trace << "{\"visible_tokens\":" << completed.head.length << ",\"page_rows\":" << cache::Plan::page_rows << ",\"pages\":[";
      for (std::size_t i = 0; i < completed.head.pages.size(); ++i) layout_trace << (i ? "," : "") << completed.head.pages[i].page();
      layout_trace << "]}\n"; if (!layout_trace) throw std::runtime_error("cannot write KV layout trace");
    }
  }
  if (output.capture_probes) {
    const auto directory = io.probe_subdirectory.empty() ? model.layer0_probe / io.name
        : model.layer0_probe / std::filesystem::path(io.probe_subdirectory) / io.name;
    std::filesystem::create_directories(directory);
    for (std::size_t i = 0; i < completed.probes.size(); ++i) {
      const auto view = completed.probes[i]; const auto bytes = take(view.description().required_bytes());
      if ((completed.requests != 1 && !view.description().is_dense()) || bytes % completed.requests)
        throw std::runtime_error("published batch diagnostic is not request-major dense");
      dump_bytes(take(view.storage().subview(completed.member_index * (bytes / completed.requests), bytes / completed.requests)),
          directory / (std::string(layer0_probe_names[i]) + (model.precision.is_f16() ? ".f16" : ".bf16")));
    }
  }
}

EvaluationOutput evaluation_output(Model &model, const std::shared_ptr<Session> &session,
    const EvaluationIO &io, double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    GenerationLedger *ledger = nullptr, const GenerationCommitRecord *expected = nullptr) {
  return {.session = session, .capture_tensors = io.trace && !io.trace->empty(),
      .capture_probes = !model.layer0_probe.empty(), .complete_seconds = complete_seconds,
      .host_endpoint = host_endpoint, .observation = observation, .preparation = preparation,
      .generation_ledger = ledger, .expected_episode = expected};
}

MtpRoundResult finish_mtp_round(Model &model, Session &session, std::int32_t expected_pending,
    std::size_t output_quota, std::span<const std::int32_t> eos = {},
    const EvaluationOutput *output = nullptr, const EvaluationIO &io = {.name = "mtp"}) {
  const std::array selections{MtpGenerationSelection{session.shared_from_this(), expected_pending, output_quota}};
  const std::array outputs{output ? *output : evaluation_output(model, session.shared_from_this(), io)};
  std::array<MtpRoundResult, 1> results;
  model.finish_mtp_rounds(selections, eos, outputs, results, [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  });
  return std::move(results[0]);
}

std::int32_t finish_evaluation(Model &model, Session &session, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {},
    GenerationLedger *generation_ledger = nullptr, const GenerationCommitRecord *expected_episode = nullptr) {
  const EvaluationIO io{name, &trace, trace_states, probe_subdirectory};
  const std::array outputs{evaluation_output(model, session.shared_from_this(), io,
      complete_seconds, host_endpoint, observation, preparation, generation_ledger, expected_episode)};
  std::int32_t next = -1;
  model.finish_evaluations(outputs, std::span(&next, 1), [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  });
  return next;
}

std::int32_t evaluate(Model &model, Session &session, std::span<const std::int32_t> ids, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {}, const inference::RequestStopPolicy *stop_policy = nullptr,
    const LogitsInputs &logits = {}) {
  model.prepare_evaluation(session, ids, preparation, GraphRole::kText, TextCompletion::kEmitNext, stop_policy, logits);
  try { model.submit_evaluation(session); return finish_evaluation(model, session, name, trace, trace_states,
      complete_seconds, host_endpoint, observation, preparation, probe_subdirectory); }
  catch (inference::EvaluationFailure &failure) {
    failure.observe_retirement(model.abort_episode());
    throw;
  } catch (...) { (void)model.abort_episode(); throw; }
}

hyle::Status Session::stop() noexcept {
  if (stopping) return retirement_status;
  stopping = true;
  if (pending) {
    if (!pending->cancelled && pending->submitted && pending->has_completion() &&
        pending->phase && pending->phase->family->role == GraphRole::kText &&
        model_ && !model_->cache_plan.draft_cache && pending->members.size() > 1 &&
        std::ranges::any_of(pending->members, [](const auto &member) { return !member.session->stopping; })) {
      pending->discard_stopped_members = true;
      // This acknowledges the requested withdrawal, rather than reporting a
      // device/retirement error. The completed member reports kCancelled.
      return retirement_status;
    }
    pending->cancelled = true;
    if (pending->has_completion()) {
      const auto cancelled = pending->request_cancel();
      if (!cancelled.IsUnimplemented()) inference::record_retirement_status(retirement_status, cancelled);
    }
    if (pending->run) pending->run->close().IgnoreError();
  }
  return retirement_status;
}
inference::RetirementObservation Session::try_retire() noexcept {
  if (!stopping) return {false, hyle::Status::FailedPrecondition("stop the Qwen session before retirement")};
  // The original caller must finish successful work for surviving members.
  // Neither a stopped owner nor its drain may turn this into a peer abort.
  if (pending && pending->discard_stopped_members && !pending->cancelled)
    return {false, retirement_status};
  inference::RetirementObservation observed{true, {}};
  if (!model_) observed = resources_->retire(false);
  else if (pending) observed = model_->try_retire_episode();
  inference::record_retirement_status(retirement_status, observed.status);
  if (observed.retired) {
    tails = {}; draft_tails = {}; tail_epoch = 0; draft_tail_epoch = 0; head.reset();
  }
  return {observed.retired, retirement_status};
}
inference::RetirementObservation Session::drain() noexcept {
  if (pending && pending->discard_stopped_members && !pending->cancelled)
    return {false, retirement_status};
  inference::RetirementObservation observed{true, {}};
  if (!model_) observed = resources_->retire(true);
  else if (pending) observed = model_->abort_episode();
  // Driving an unstopped Session preserves its head and continued-use contract.
  // In particular, a peer's ordinary episode cancellation is not made sticky.
  if (!stopping) {
    auto status = retirement_status;
    inference::record_retirement_status(status, observed.status);
    return {false, status};
  }
  // A stopped Session still proves/releases through the original retirement gate.
  inference::record_retirement_status(retirement_status, observed.status);
  return try_retire();
}
Session::~Session() { (void)drain(); }

struct GenerationRestoreStats final {
  std::size_t graph_calls = 0, target_rows = 0, draft_rows = 0;
  std::size_t normalized_lag_rows = 0;
  struct DecisionChange { std::size_t episode; std::int32_t historical, fresh; };
  std::vector<DecisionChange> decision_changes;
  double step_seconds = 0, seconds = 0;
};

enum class GenerationMode { kText, kMtp };

struct GenerationRequest {
  std::string id, memory_name;
  std::span<const std::int32_t> prompt;
  std::filesystem::path trace;
  inference::RequestProgress progress;
  inference::RequestKey request_key{};
  std::shared_ptr<Session> session;
  cache::TokenPlanRef input;
  std::shared_ptr<const cache::HostImage> host_image;
  cache::Checkpoint completed_checkpoint;
  std::shared_ptr<cache::ReplayHistory> history;
  std::vector<MtpRoundResult> rounds;
  std::vector<std::vector<std::int32_t>> round_emitted;
  Clock::time_point prefill_start{}, prefill_endpoint{}, endpoint{};
  double prefill_step_seconds = 0, decode_step_sum_seconds = 0;
  std::size_t verified_rows = 0, draft_rows = 0, draft_rounds = 0, agreements = 0, accepts = 0;
  std::size_t visible_tokens = 0, draft_frontier = 0, draft_lag = 0;
  std::size_t reused_tokens = 0, prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0;
  std::size_t prefill_graph_calls = 0, prefill_graph_tokens = 0, prefill_draft_rows = 0;
  std::size_t prefix_sampling_calls = 0;
  std::size_t auxiliary_graph_calls = 0;
  double prefix_sampling_seconds = 0;
  const inference::RequestStopPolicy stop_policy;
  std::unique_ptr<GenerationLedger> ledger;
  struct TurnObservation {
    Clock::time_point start{}, prefill_endpoint{}, endpoint{};
    std::size_t output_begin = 0, output_end = 0;
    std::size_t context_before = 0, context_after = 0;
  };
  std::vector<TurnObservation> turn_observations;
  std::vector<std::shared_ptr<std::vector<std::int32_t>>> warm_inputs;
  std::size_t last_idle = 0;
  const std::size_t sampling_index;
  LogitsInputs logits; // Owner may replace ready inputs between output episodes.
  std::size_t input_turn = SIZE_MAX, turn_input_consumed = 0;
  Clock::time_point admission_start{};
  GenerationRestoreStats restored;
  std::size_t restore_count = 0, idle_evictions = 0;
  std::size_t checkpoint_resume_hits = 0, checkpoint_resume_misses = 0;
  std::int32_t fresh_next_id = -1;

  GenerationRequest(std::string name, std::span<const std::int32_t> ids,
                       std::filesystem::path output, std::size_t steps, std::size_t sampling = 0,
                       inference::RequestStopPolicy stop = {})
      : id(std::move(name)), memory_name(id.empty() ? "generation" : "generation_" + id),
        prompt(ids), trace(std::move(output)), progress(steps), stop_policy(std::move(stop)), sampling_index(sampling) {
    rounds.reserve(steps - 1);
  }
};

// Cold episode count; imported source chunks have their own explicit bound.
std::size_t generation_record_capacity(std::span<const inference::GenerationTurn> turns,
    std::size_t max_prefill_rows = SIZE_MAX, std::size_t imported_records = 0) {
  const inference::PrefillPolicy prefill(max_prefill_rows);
  auto count = imported_records;
  for (std::size_t turn = 0; turn < turns.size(); ++turn) {
    const auto rows = cache::add(turns[turn].prompt.size(), turn ? 1u : 0u);
    if (!rows || !turns[turn].max_outputs) throw std::runtime_error("generation requires nonempty turn input and positive quota");
    const auto chunks = turn == 0 && imported_records ? prefill.parts_bound(rows) : prefill.parts(rows);
    count = cache::add(count, cache::add(chunks, turns[turn].max_outputs - 1));
  }
  return count;
}

GenerationCommitRecord &prepare_generation_record(GenerationLedger &ledger, const Session &session,
    GraphRole role, std::span<const std::int32_t> ids, bool generation,
    std::size_t quota, std::size_t turn, std::shared_ptr<const void> owner = {}, std::size_t maximum = 0,
    TextCompletion completion = TextCompletion::kEmitNext, std::size_t input_begin = 0, bool prefill_complete = true) {
  if (completion == TextCompletion::kStateOnly && (role != GraphRole::kText || generation || prefill_complete))
    throw std::runtime_error("state-only ledger entry requires an incomplete ordinary prefill");
  if (ids.empty() || !session.head || (generation && ids.size() != 1))
    throw std::runtime_error("MTP replay record requires its actual input and complete base");
  auto &record = ledger.next();
  ledger.bind_sampling(session.sampling_configuration(), session.sampling_owner());
  auto emitted = std::move(record.emitted), consumed = std::move(record.accepted_inputs);
  auto verified = std::move(record.verification_inputs);
  record = {}; record.emitted = std::move(emitted); record.accepted_inputs = std::move(consumed);
  record.verification_inputs = std::move(verified); record.verification_inputs.clear();
  record.emitted.assign({-1, -1}); record.accepted_inputs.assign({-1, -1});
  record.role = role; record.generation = generation; record.quota = quota; record.turn = turn;
  record.completion = completion; record.input_begin = input_begin; record.prefill_complete = prefill_complete;
  record.user_output = completion == TextCompletion::kEmitNext;
  record.draft_limit = role == GraphRole::kVerify ? maximum : drafts_token(role) ? 1 : 0;
  record.before = generation_boundary(*session.head);
  if (generation) record.pending_input = ids.front();
  else { record.input_owner = std::move(owner); record.prompt_ids = ids; }
  return record;
}

void preflight_generation_prompt_ledger(const GenerationLedger &ledger, const cache::ReplayPrefix &prefix) {
  if (ledger.committed_count || !prefix.history || !prefix.chunks ||
      prefix.chunks > prefix.history->chunks.size() || prefix.chunks > ledger.records.size())
    throw std::runtime_error("MTP initial prompt import requires an empty bounded ledger");
  std::size_t covered = 0;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    if (ledger.records[i].committed || ledger.records[i].emitted.capacity() < 2 ||
        !part.input || !part.count || part.first != covered || part.first > part.input->ids.size() ||
        part.count > part.input->ids.size() - part.first || covered > prefix.length ||
        part.count > prefix.length - covered || (part.state_only ? part.next_id != -1 : part.next_id < 0))
      throw std::runtime_error("prompt import requires complete original ranges and uncommitted reserved records");
    covered += part.count;
  }
  if (covered != prefix.length)
    throw std::runtime_error("prompt import history does not cover its complete boundary");
}

void import_generation_prompt_ledger(GenerationLedger &ledger, const cache::ReplayPrefix &prefix,
                              bool full_hit, bool draft_cache) {
  preflight_generation_prompt_ledger(ledger, prefix);
  std::int32_t previous = -1;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    auto &record = ledger.records[i];
    record.role = !draft_cache ? GraphRole::kText
        : part.count == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
    record.input_owner = part.input;
    record.prompt_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
    record.before = {part.first, draft_cache ? part.first : 0, 0, previous, -1};
    record.after = {part.first + part.count, draft_cache ? part.first + part.count : 0, 0, part.next_id, -1};
    record.accepted = part.count; record.draft_rows = draft_cache ? part.count : 0;
    record.completion = part.state_only ? TextCompletion::kStateOnly : TextCompletion::kEmitNext;
    record.prefill_complete = !part.state_only;
    record.input_begin = part.first;
    record.target_first = part.next_id; record.emitted = {part.next_id, -1}; record.emitted_count = part.state_only ? 0 : 1;
    record.stopped = !draft_cache
        ? ledger.stop_policy().stopped(part.next_id, is_end_of_generation(part.next_id))
        : is_end_of_generation(part.next_id);
    record.user_output = full_hit && i + 1 == prefix.chunks;
    record.imported = true;
    record.requests = 0; // Role-only source history does not claim original batch geometry.
    previous = part.next_id;
  }
  // All ranges/capacities were checked before population. shared_ptr/span and
  // scalar writes, including the two inline emitted IDs, allocate nothing.
  for (std::size_t i = 0; i < prefix.chunks; ++i) ledger.records[i].committed = true;
  ledger.committed_count = prefix.chunks;
}

// Consume committed host facts even when a subsequent diagnostic write threw.
// The authoritative ledger was installed by the original head-swap transaction.
void observe_generation_commits(GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  while (request.progress.observed_commits() < request.ledger->committed_count) {
    const auto ordinal = request.progress.observed_commits();
    const auto &record = request.ledger->records[ordinal];
    if (record.emitted_count > record.emitted.size())
      throw std::runtime_error("committed output count exceeds its original record storage");
    const auto emitted = std::span(record.emitted).first(record.emitted_count);
    const bool prefill_complete = !record.generation && !record.imported && record.prefill_complete;
    if (!record.generation && !record.imported &&
        (record.turn != request.progress.turn_index || request.turn_input_consumed > request.prompt.size() || record.input_begin != request.turn_input_consumed ||
         record.ids().size() > request.prompt.size() - request.turn_input_consumed ||
         record.ids().data() != request.prompt.data() + request.turn_input_consumed ||
         record.prefill_complete != (record.ids().size() == request.prompt.size() - request.turn_input_consumed)))
      throw std::runtime_error("committed prefill input is not the next actual turn slice");
    const bool stopped = record.role == GraphRole::kText ? record.stopped : is_end_of_generation(record.after.next_id);
    if (!request.progress.preflight_commit(ordinal, emitted, record.user_output, prefill_complete, stopped))
      continue;
    if (record.user_output && record.generation &&
        (request.rounds.size() == request.rounds.capacity() || request.rounds.size() >= request.round_emitted.size() ||
         record.emitted_count > request.round_emitted[request.rounds.size()].capacity()))
      throw std::runtime_error("generation observer requires its original cold-reserved round storage");
    const auto output_end = request.progress.generated.size() + (record.user_output ? emitted.size() : 0);
    if (record.user_output) {
      if (record.generation) {
        MtpRoundResult result;
        result.pending = record.pending_input; result.draft = record.draft;
        result.target_first = record.target_first; result.target_second = record.target_second;
        result.next_id = record.after.next_id; result.agreed = record.agreed;
        result.accepted_draft = record.accepted_draft; result.stopped = record.stopped;
        result.processed_target = record.accepted; result.processed_draft = record.draft_rows;
        result.proposal_count = record.verified_rows ? record.verified_rows - 1 : 0;
        result.verified_rows = record.verified_rows; result.consumed_drafts = record.consumed_drafts;
        result.auxiliary_graph_calls = record.auxiliary_graph_calls;
        result.emitted = std::move(request.round_emitted[request.rounds.size()]);
        result.emitted.assign(record.emitted.begin(), record.emitted.begin() + record.emitted_count);
        request.rounds.push_back(std::move(result));
        request.verified_rows += record.verified_rows;
        request.draft_rows += record.draft_rows; request.draft_rounds += record.draft_limit ? 1 : 0;
        request.agreements += record.agreed ? 1 : 0; request.accepts += record.consumed_drafts;
        request.auxiliary_graph_calls += record.auxiliary_graph_calls;
        request.decode_step_sum_seconds += record.step_seconds;
      }
    }
    if (!record.generation && !record.imported) {
      ++request.prefill_graph_calls; request.prefill_graph_tokens += record.ids().size();
      request.prefill_draft_rows += record.draft_rows;
      request.prefill_step_seconds += record.step_seconds;
    }
    request.visible_tokens = record.after.length; request.draft_frontier = record.after.frontier;
    request.draft_lag = record.after.lag; request.fresh_next_id = record.after.next_id;
    if (!record.imported) request.endpoint = record.endpoint;
    if (prefill_complete) {
      request.prefill_endpoint = record.endpoint;
      if (!request.turn_observations.empty()) {
        auto &turn = request.turn_observations[request.progress.turn_index];
        turn.start = request.prefill_start; turn.prefill_endpoint = record.endpoint;
      }
    }
    if (!request.turn_observations.empty()) {
      auto &turn = request.turn_observations[request.progress.turn_index];
      turn.endpoint = request.endpoint; turn.output_end = output_end;
      turn.context_after = request.visible_tokens;
    }
    // No fallible family bookkeeping follows the one host observation mark.
    (void)request.progress.observe_commit(ordinal, emitted, record.user_output, prefill_complete, stopped);
    if (!record.generation && !record.imported) request.turn_input_consumed += record.ids().size();
  }
}

// Only the serialized request owners and this directory retain committed
// Snapshots here. Hit/local Snapshot references are dropped before this gate.
inference::VersionHolder<Session, cache::State> version_holder(const Session *session) noexcept {
  return {session, session && session->head ? session->head->state_identity() : nullptr};
}

void make_generation_room(Model &model, cache::PrefixCache *prefix, Session *owner,
                    std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const std::array selected{version_holder(owner)};
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
  });
  const auto observe = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation admission requires retired committed heads within N");
    return inference::VersionRoomObservation{{live, limit}, prefix ? prefix->size() : 0,
        inference::version_admission({live, limit}, selected, residents,
            [&](const auto *head) { return prefix && prefix->retains(head); }).fits};
  };
  if (inference::make_version_room(observe, [&] { return prefix && prefix->evict_one(); }) !=
      inference::VersionRoomResult::kReady)
    throw std::runtime_error("generation checkpoints retain all committed versions; no unfinished session may be evicted");
}

std::shared_ptr<Session> restore_mtp_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = model.sampling_index(ledger);
  const auto started = Clock::now();
  if (sampling_index || !model.cache_plan.draft_cache || !ledger.stop_policy().is_default())
    throw std::runtime_error("strict MTP replay requires the original complete target/draft model and default stop policy");
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP restore requires a complete zero-origin ledger and an idle carrier");
  for (std::size_t i = 0; i < ledger.committed_count; ++i)
    if (ledger.records[i].requests != 1 || ledger.records[i].member_index != 0)
      throw std::runtime_error("strict MTP replay requires recorded original R1 geometry; use committed-token rebuild for cache recovery");
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before restoring within N committed versions");
  const auto &first = ledger.records[0];
  const std::array calls{first.ids().size()};
  auto session = model.create_session(calls, {}, first.role, first.before.sampling);
  GenerationLedger actual(ledger.committed_count, model.cache_plan.candidate_boundaries);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  const auto &eos = model_eos_tokens;
  try {
    for (std::size_t index = 0; index < ledger.committed_count; ++index) {
      const auto &expected = ledger.records[index];
      if (!expected.committed) throw std::runtime_error("MTP restore ledger contains an uncommitted gap");
      const auto pending = expected.pending_input;
      const auto ids = expected.generation ? std::span<const std::int32_t>(&pending, 1) : expected.ids();
      prepare_generation_record(actual, *session, expected.role, ids, expected.generation,
          expected.quota, expected.turn, expected.input_owner, expected.draft_limit);
      make_generation_room(model, prefix, session.get(), requests);
      std::vector<c::ReadLease> q;
      const auto verification = expected.role == GraphRole::kVerify
          ? model.propose_recursive(*session, expected.draft_limit, eos, &q) : std::vector<std::int32_t>{};
      model.prepare_evaluation(*session, expected.role == GraphRole::kVerify ? std::span<const std::int32_t>(verification) : ids,
          nullptr, expected.role); model.submit_evaluation(*session);
      if (stats) {
        ++stats->graph_calls;
        stats->target_rows = cache::add(stats->target_rows, expected.verified_rows);
        stats->draft_rows = cache::add(stats->draft_rows, expected.draft_rows);
      }
      const auto name = "episode" + std::to_string(index);
      double step_seconds = 0;
      if (expected.generation) {
        const EvaluationIO io{name, &trace, trace_states};
        const auto output = evaluation_output(model, session, io, &step_seconds,
            nullptr, nullptr, nullptr, &actual, &expected);
        if (expected.role == GraphRole::kVerify) {
          const std::array selections{MtpGenerationSelection{session, pending, expected.quota, q, expected.draft_limit}};
          const std::array outputs{output}; std::array<MtpRoundResult, 1> results;
          model.finish_recursive_mtp(selections, eos, outputs, results,
              [&](std::size_t j, const CompletedEvaluationView &view) { observe_completed_evaluation(model, outputs[j], io, view); });
        } else (void)qwen35::finish_mtp_round(model, *session, pending, expected.quota, eos, &output, io);
      } else {
        (void)qwen35::finish_evaluation(model, *session, name, trace, trace_states, &step_seconds,
            nullptr, nullptr, nullptr, {}, &actual, &expected);
      }
      if (stats) stats->step_seconds += step_seconds;
    }
  } catch (...) {
    session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// Cache reconstruction consumes immutable committed token chunks. It does not
// resample past outputs or claim to replay an old batch's natural decisions.
std::shared_ptr<Session> rebuild_generation_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = model.sampling_index(ledger);
  ledger.stop_policy().validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !ledger.stop_policy().is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  const auto started = Clock::now();
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP rebuild requires a complete zero-origin committed history and idle carrier");
  std::size_t committed_length = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i];
    if (!record.committed || !record.accepted || (record.generation &&
        (record.accepted > record.accepted_inputs.size() || record.accepted_inputs[0] != record.pending_input ||
          record.consumed_drafts + 1 != record.accepted ||
          (record.accepted > 1 && (!record.accepted_draft || record.accepted_inputs[1] != record.draft)))))
      throw std::runtime_error("MTP rebuild has an invalid committed input boundary");
    const auto ids = record.committed_ids();
    if (record.before.length != committed_length || ids.empty() || ids.size() != record.accepted ||
        record.accepted > model.capacity - committed_length)
      throw std::runtime_error("MTP rebuild committed input chunks are not a continuous bounded prefix");
    committed_length = cache::add(committed_length, record.accepted);
    if (record.after.length != committed_length) throw std::runtime_error("MTP rebuild committed length is inconsistent");
    for (auto id : ids) if (id < 0 || id >= model.vocabulary) throw std::runtime_error("MTP rebuild token is outside vocabulary");
  }
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before rebuilding within N committed versions");
  if (stats) stats->decision_changes.reserve(cache::add(stats->decision_changes.size(), ledger.committed_count));
  // Modifier values are per-output inputs, not replay history. Rebuild only
  // consumed state, then restore each recorded pending/cursor below; no old
  // mask/bias retention, historical resampling or extra RNG draw is required.
  const bool state_rebuild = !model.sampling_configuration(sampling_index).configuration.modifiers.empty();
  const auto first_ids = ledger.records[0].committed_ids();
  const std::array calls{first_ids.size()};
  const auto role_for = [&](std::size_t rows) {
    return !model.cache_plan.draft_cache ? GraphRole::kText
        : rows == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
  };
  auto session = model.create_session(calls, {}, role_for(first_ids.size()),
      ledger.records[0].before.sampling, false, state_rebuild ? TextCompletion::kStateOnly : ledger.records[0].completion, sampling_index);
  GenerationLedger actual(ledger.committed_count);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  std::size_t observed = 0;
  const auto observe = [&] {
    while (observed < actual.committed_count) {
      const auto &fresh = actual.records[observed]; const auto &old = ledger.records[observed];
      if (stats) {
        ++stats->graph_calls; stats->target_rows = cache::add(stats->target_rows, fresh.accepted);
        stats->draft_rows = cache::add(stats->draft_rows, fresh.draft_rows);
        stats->step_seconds += fresh.step_seconds;
        if (!state_rebuild && fresh.after.next_id != old.after.next_id)
          stats->decision_changes.push_back({observed, old.after.next_id, fresh.after.next_id});
      }
      ++observed;
    }
  };
  try {
    for (std::size_t i = 0; i < ledger.committed_count; ++i) {
      const auto &source = ledger.records[i]; const auto ids = source.committed_ids();
      const auto role = role_for(ids.size());
      const auto completion = state_rebuild ? TextCompletion::kStateOnly : source.completion;
      prepare_generation_record(actual, *session, role, ids, false, 0, source.turn, source.input_owner, 0,
          completion, source.input_begin, state_rebuild ? false : source.prefill_complete);
      make_generation_room(model, prefix, session.get(), requests);
      model.prepare_evaluation(*session, ids, nullptr, role, completion, &actual.stop_policy()); model.submit_evaluation(*session);
      (void)qwen35::finish_evaluation(model, *session, "episode" + std::to_string(i), trace, trace_states,
          nullptr, nullptr, nullptr, nullptr, {}, &actual);
      observe();
      // Rebuilding committed tokens is not a fresh user sampling episode.
      // The historical pending token and cursor remain authoritative even if
      // a differently partitioned recomputation produces different logits.
      model.restore_pending_boundary(*session, source.after.sampling, source.after.next_id);
    }
    if (session->length() != ledger.records[ledger.committed_count - 1].after.length ||
        (session->head->draft && session->head->draft->lag))
      throw std::runtime_error("MTP committed-token rebuild has an inconsistent complete boundary");
    if (stats) stats->normalized_lag_rows = cache::add(stats->normalized_lag_rows, ledger.records[ledger.committed_count - 1].after.lag);
  } catch (...) {
    observe(); session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// A completed checkpoint is only a possible recovery shortcut. A newer
// mid-turn commit leaves that old complete key behind; compare it with the
// actual immutable ledger at this cold recovery boundary, without a new key.
bool checkpoint_covers_generation_ledger(const cache::Checkpoint &checkpoint,
                                         const GenerationLedger &ledger) noexcept {
  if (!checkpoint.input || !checkpoint.replay.history || !ledger.committed_count ||
      ledger.committed_count > ledger.records.size()) return false;
  const auto &last = ledger.records[ledger.committed_count - 1];
  if (last.after.next_id < 0 || checkpoint.input->ids.size() != last.after.length ||
      checkpoint.replay.length != last.after.length || checkpoint.replay.chunks != ledger.committed_count ||
      checkpoint.replay.chunks > checkpoint.replay.history->chunks.size() ||
      checkpoint.replay.history->chunks[checkpoint.replay.chunks - 1].next_id != last.after.next_id)
    return false;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i]; const auto ids = record.committed_ids();
    if (!record.committed || ids.empty() || ids.size() != record.accepted || record.before.length != offset ||
        offset > checkpoint.input->ids.size() || ids.size() > checkpoint.input->ids.size() - offset ||
        !std::equal(ids.begin(), ids.end(), checkpoint.input->ids.begin() + static_cast<std::ptrdiff_t>(offset)))
      return false;
    offset += ids.size();
    if (record.after.length != offset) return false;
  }
  return offset == last.after.length;
}

// Prove the whole remaining ordinary turn against its authoritative cursor.
// This is admission arithmetic only: no RNG reservation or mutable copy. A new
// prefix hit must start from this request, never from its cached publisher.
void preflight_generation_sampling(const Model &model, const GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (model.cache_plan.draft_cache ||
      !model.sampling_configuration(request.sampling_index).configuration.enabled) return;
  const auto &progress = request.progress;
  std::size_t outputs = 0;
  if (progress.turn_started() && progress.turn_done && progress.more_turns()) {
    // Open input may append its successor while the finished turn is paused.
    // Prove that actual next quota before the shared owner evicts any peer.
    outputs = progress.turns()[progress.turn_index + 1].max_outputs;
    if (progress.generated.size() > progress.maximum_outputs() ||
        outputs > progress.maximum_outputs() - progress.generated.size())
      throw std::runtime_error("request turn exceeds its prepared output limit");
  }
  else if (progress.turn_started()) outputs = progress.remaining_outputs();
  else {
    if (progress.generated.size() > progress.maximum_outputs())
      throw std::runtime_error("request outputs exceed its prepared output limit");
    const auto turns = progress.turns();
    if (turns.empty()) outputs = progress.maximum_outputs() - progress.generated.size();
    else {
      if (progress.turn_index >= turns.size())
        throw std::runtime_error("request turn is outside its prepared schedule");
      outputs = turns[progress.turn_index].max_outputs;
      if (outputs > progress.maximum_outputs() - progress.generated.size())
        throw std::runtime_error("request turn exceeds its prepared output limit");
    }
  }
  if (!outputs) return;
  auto cursor = progress.initial_sampling;
  if (request.session) {
    if (!request.session->head || request.session->stopping || request.session->failed || request.session->pending)
      throw std::runtime_error("ordinary sampling admission requires a live idle Session head");
    cursor = request.session->head->sampling;
  }
  else if (request.ledger->committed_count) {
    if (request.ledger->committed_count > request.ledger->records.size())
      throw std::runtime_error("request recovery exceeds its original committed ledger");
    cursor = request.ledger->records[request.ledger->committed_count - 1].after.sampling;
  }
  // Includes the prefill/full-hit output; the last pending token needs no
  // additional draw. State-only chunks do not consume this checked budget.
  (void)cursor.advanced(outputs);
}

// Only restore the original Session. Request progress, turn input/offset,
// output delivery and its key remain with the serialized owner. In particular,
// this function never begins a turn or observes/emits historical output again.
void restore_generation_session(Model &model, GenerationRequest &request,
    cache::PrefixCache *prefix = nullptr, bool trace_states = true,
    std::span<const GenerationRequest> requests = {}) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !request.stop_policy.is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (request.session) {
    if (&request.session->sampling_configuration() != &prepared_sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    return;
  }
  if (!request.ledger->committed_count) return;
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  if (model.sampling_index(*request.ledger) != request.sampling_index)
    throw std::runtime_error("request recovery sampling differs from its committed ledger");
  preflight_generation_sampling(model, request);
  if (model.active_step || model.carrier_borrowed)
    throw std::runtime_error("request recovery requires the original idle execution carrier");
  const auto &ledger = *request.ledger;
  if (ledger.committed_count > ledger.records.size())
    throw std::runtime_error("request recovery exceeds its original committed ledger");
  const auto &last = ledger.records[ledger.committed_count - 1];
  std::shared_ptr<Session> restored;
  if (prefix && request.completed_checkpoint.input) {
    if (ledger.sampling == &prepared_sampling &&
        checkpoint_covers_generation_ledger(request.completed_checkpoint, ledger)) {
      const auto length = request.completed_checkpoint.input->ids.size();
      auto hit = prefix->lookup(*request.completed_checkpoint.input, length, length, &model, model.math_identity());
      if (hit) {
        // A peer may have published the same consumed key with another pending
        // or RNG cursor. Restore only those historical scalar facts; the fork
        // cannot repair an old physical length/frontier/lag/catchup boundary.
        auto head = model.sampling_fork(std::move(hit.payload.snapshot), last.after.sampling, last.after.next_id);
        hit.reset();
        if (generation_boundary(*head) == last.after &&
            bool(head->draft) == model.cache_plan.draft_cache) {
          restored = model.create_session({}, std::move(head),
              model.cache_plan.draft_cache ? GraphRole::kMtpSingle : GraphRole::kText,
              {}, false, TextCompletion::kEmitNext, request.sampling_index);
          ++request.checkpoint_resume_hits;
        }
      }
    }
    if (!restored) ++request.checkpoint_resume_misses;
  }
  if (!restored) {
    const auto trace = request.trace.empty() ? request.trace
        : request.trace / "restores" / ("restore" + std::to_string(request.restore_count));
    restored = rebuild_generation_session(model, ledger, trace, trace_states, &request.restored, prefix, requests);
    ++request.restore_count;
  }
  // All fallible recovery precedes installing the new original Session owner.
  request.session = std::move(restored);
}

struct PreparedGenerationCall final {
  GenerationRequest *request = nullptr;
  GraphRole role = GraphRole::kText;
  std::span<const std::int32_t> prompt;
  std::int32_t pending = -1;
  std::size_t quota = 0, steps = 0;
  std::size_t maximum = 0;
  std::vector<std::int32_t> verification_ids;
  std::vector<c::ReadLease> draft_log_probabilities;
  bool generation = false, full_hit = false, initial_prefix = false, prefill_complete = true;
  EvaluationDisposition disposition = EvaluationDisposition::kNotPublished;
  LogitsInputs logits; // Snapshot the caller's input leases for this prepared episode.
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::string name;
  double step_seconds = 0, proposal_seconds = 0;
  std::span<const std::int32_t> ids() const noexcept {
    if (generation && role == GraphRole::kVerify && !verification_ids.empty()) return verification_ids;
    return generation ? std::span(&pending, 1) : prompt;
  }
  std::size_t planned_rows() const noexcept { return role == GraphRole::kVerify ? maximum + 1 : drafts_token(role) ? 2 : ids().size(); }
};

// One actually retired target batch, never a planned request bucket. Members
// are borrowed synchronously; a report owner copies identities before return.
// Episode durations can overlap and must not be added as a wall-time union.
struct GenerationBatchFacts final {
  GraphRole role;
  TextCompletion completion;
  bool generation;
  std::size_t input_rows;
  std::span<const PreparedGenerationCall> members;
  double step_seconds;
};

template<class Observer>
void observe_generation_batch(Observer &observer, std::span<const PreparedGenerationCall> members,
    std::size_t rows, double seconds) {
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Observer>>)
    observer(GenerationBatchFacts{members[0].role, members[0].completion, members[0].generation,
        rows, members, seconds});
}

// Serialized owner policy only. A wave still retires every original episode;
// only its latest successful ready Source crosses waves. The exact Model and
// Graph checks, not these warm-direction hints, authorize pending execution.
struct GenerationContinuation final {
  struct Member final {
    inference::RequestKey key;
    std::size_t turn;
    friend bool operator==(const Member &, const Member &) = default;
  };
  bool enabled = true, unavailable = false;
  hyle::container::SmallVector<Member, 2> members;
  std::size_t next_slot = 0;
  std::optional<std::size_t> ready_slot;
  std::array<bool, execution_slot_count> direction_warm{};
  std::size_t ready_steps = 0;
  std::shared_ptr<StepSources> ready;
  std::size_t completed_steps = 0, successor_submissions = 0, successor_publications = 0,
      successor_discards = 0, successor_refusals = 0;

  void prepare(std::size_t maximum_members) {
    if (!members.try_reserve(maximum_members)) throw std::bad_alloc{};
  }
  void release_ready() noexcept {
    ready.reset(); ready_slot.reset(); next_slot = 0;
  }
  void reset() noexcept {
    release_ready(); members.clear(); direction_warm.fill(false); ready_steps = 0;
  }
  bool select(std::span<const PreparedGenerationCall> calls) noexcept {
    if (calls.empty() || calls.size() > members.capacity()) { reset(); return false; }
    bool same = members.size() == calls.size();
    for (std::size_t j = 0; same && j < calls.size(); ++j)
      same = members[j] == Member{calls[j].request->request_key, calls[j].request->progress.turn_index};
    if (!same) {
      reset();
      for (const auto &call : calls)
        members.push_back({call.request->request_key, call.request->progress.turn_index});
    }
    return true;
  }
  // Admission may run for one member before the next compatible batch exists.
  // Keep a matching cohort, but release it before another owner needs slots.
  void admission(inference::RequestKey selected, std::size_t selected_turn) noexcept {
    if (std::ranges::none_of(members, [&](const auto &member) {
          return member.key == selected && member.turn == selected_turn;
        })) reset();
  }
  void release(inference::RequestKey selected) noexcept {
    if (std::ranges::any_of(members, [&](const auto &member) { return member.key == selected; })) reset();
  }
  bool matches(std::span<const PreparedGenerationCall> calls) const noexcept {
    if (!ready || calls.size() != members.size() || ready->boundaries.size() != calls.size()) return false;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      const auto &request = *calls[j].request;
      if (members[j] != Member{request.request_key, request.progress.turn_index} || !request.session || !request.session->head)
        return false;
      const auto &boundary = ready->boundaries[j]; const auto owner = boundary.session.lock();
      if (owner != request.session || owner.owner_before(request.session) || request.session.owner_before(owner) ||
          boundary.candidate != request.session->head.get() || boundary.epoch != request.session->head.epoch()) return false;
    }
    return true;
  }
};

PreparedGenerationCall prepare_generation_call(Model &model, GenerationRequest &request, std::size_t steps,
    cache::PrefixCache *prefix, std::span<const GenerationRequest> requests,
    std::size_t output_credit = SIZE_MAX, std::size_t max_prefill_rows = SIZE_MAX) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !request.stop_policy.is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (model.cache_plan.draft_cache && (request.sampling_index || !prepared_sampling.configuration.modifiers.empty()))
    throw std::runtime_error("nondefault request sampling is ordinary Text only");
  if (request.session && &request.session->sampling_configuration() != &prepared_sampling)
    throw std::runtime_error("request sampling cannot change after Session admission");
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  const auto &sampler = prepared_sampling.configuration;
  if (!max_prefill_rows || (model.cache_plan.draft_cache && max_prefill_rows != SIZE_MAX))
    throw std::runtime_error("bounded prefill rows require a positive ordinary Text quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  // Credit belongs to this episode, not the turn's immutable output quota.
  // Reject before prefix lookup, session creation or any other admission work.
  if (!output_credit) throw std::runtime_error("generation must wait for output credit before preparation");
  if (steps != request.progress.output_limit())
    throw std::runtime_error("generation call quota differs from its prepared request turn");
  preflight_generation_sampling(model, request);
  if (!sampler.modifiers.empty()) {
    bool emits = request.progress.prefilled;
    if (!emits) {
      auto consumed = request.input_turn == request.progress.turn_index ? request.turn_input_consumed : 0;
      if (consumed > request.prompt.size()) throw std::runtime_error("prefill offset exceeds its turn input");
      if (prefix && request.progress.turn_index == 0 && !request.session) {
        if (!request.input) throw std::runtime_error("prefix admission requires its complete token plan");
        // No eviction/publication or directory mutation occurs between this
        // pure length preview and the original pinned lookup below.
        consumed = prefix->match_length(*request.input, request.prompt.size(), 1, &model, model.math_identity());
      }
      emits = prefill.next(request.prompt.size(), consumed).complete;
    }
    if (emits) validate_logits_inputs(sampler, request.logits, model.vocabulary);
  }
  if (!request.progress.turn_started()) (void)request.progress.begin_turn();
  if (request.progress.turn_done) throw std::runtime_error("MTP request has already completed");
  if (request.round_emitted.size() < steps - 1)
    throw std::runtime_error("MTP observer slots must be reserved for every turn before execution");
  request.ledger->bind_stop_policy(request.stop_policy, model.vocabulary);
  PreparedGenerationCall call; call.request = &request; call.steps = steps; call.logits = request.logits;
  call.generation = request.progress.prefilled;
  if (!call.generation) {
    if (request.input_turn != request.progress.turn_index) {
      const auto before = request.session ? request.session->length() : 0;
      if (before > model.capacity || request.prompt.empty() || request.prompt.size() > model.capacity - before)
        throw std::runtime_error("complete turn input exceeds the remaining context");
      // The turn's final output remains pending. Admit its full maximum journey
      // before any input chunk, prefix import or Session is published.
      const auto outputs = request.progress.remaining_outputs();
      if (!outputs || outputs - 1 > model.capacity - before - request.prompt.size())
        throw std::runtime_error("complete turn input and output quota exceed the remaining context");
      for (auto id : request.prompt) if (id < 0 || id >= model.vocabulary)
        throw std::runtime_error("token ID outside vocabulary");
      request.input_turn = request.progress.turn_index; request.turn_input_consumed = 0;
    }
    request.ledger->validate_sampling(prepared_sampling, model.resources_);
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_begin\n";
    if (!request.turn_input_consumed) request.prefill_start = request.admission_start == Clock::time_point{} ? Clock::now()
        : std::exchange(request.admission_start, Clock::time_point{});
    call.initial_prefix = prefix && request.progress.turn_index == 0;
    if (call.initial_prefix && !request.session) {
      auto hit = prefix->lookup(*request.input, request.prompt.size(), 1, &model, model.math_identity());
      if (hit) {
        if (hit.payload.snapshot->draft.has_value() != model.cache_plan.draft_cache)
          throw std::runtime_error("prefix directory contains a different target/draft state contract");
        const auto cached_lag = hit.payload.snapshot->draft ? hit.payload.snapshot->draft->lag : 0;
        const auto reused_tokens = hit.payload.snapshot->length;
        const bool full_hit = reused_tokens == request.prompt.size();
        auto history = cache::copy_history(hit.payload.replay, model.capacity);
        preflight_generation_prompt_ledger(*request.ledger, hit.payload.replay);
        request.progress.preflight_adopt_committed_history();
        const auto suffix_part = prefill.next(request.prompt.size(), reused_tokens);
        const auto suffix = request.prompt.size() - suffix_part.begin;
        const auto suffix_rows = suffix_part.rows;
        const auto suffix_completion = suffix_part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
        std::string hit_name = "prefill";
        if (!suffix_part.complete) hit_name += "_chunk" + std::to_string(reused_tokens);
        const std::array chunks{suffix_rows};
        const auto prior_sampling_seconds = model.frontier_sampling_seconds;
        const auto suffix_role = !model.cache_plan.draft_cache ? GraphRole::kText : (suffix == 1 || request.prompt.size() == 1)
            ? (cached_lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
            : (cached_lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
        // Keep every new request fact local until the actual Session/frontier
        // succeeds. Failed native/resource admission must permit the same retry.
        const auto admission_role = !model.cache_plan.draft_cache ? GraphRole::kText
            : suffix ? suffix_role : request.prompt.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
        auto adopted = model.create_session(suffix ? std::span<const std::size_t>(chunks) : std::span<const std::size_t>{},
            hit.payload.snapshot, admission_role,
            request.progress.initial_sampling, suffix == 0, suffix_completion, request.sampling_index, call.logits);
        if (!suffix) history->chunks.back().next_id = adopted->head->next_id;
        // Only original reserved metadata remains. Preconditions are unchanged
        // since preflight; no allocation or native work follows this commit.
        import_generation_prompt_ledger(*request.ledger, hit.payload.replay, full_hit, model.cache_plan.draft_cache);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
            auto &record = request.ledger->records[i];
            record.before.sampling = sampler.enabled ? request.progress.initial_sampling : cache::SamplingCursor{};
            record.after.sampling = record.before.sampling;
            record.emitted_count = 0; record.emitted = {-1, -1};
          }
          request.ledger->records[request.ledger->committed_count - 1].after.sampling = adopted->head->sampling;
        }
        if (!suffix) {
          auto &last = request.ledger->records[request.ledger->committed_count - 1];
          last.after.sampling = adopted->head->sampling;
          last.after.next_id = adopted->head->next_id;
          last.target_first = last.after.next_id; last.emitted = {last.after.next_id, -1};
          last.emitted_count = 1; last.stopped = !model.cache_plan.draft_cache
              ? request.stop_policy.stopped(last.after.next_id, is_end_of_generation(last.after.next_id))
              : is_end_of_generation(last.after.next_id);
        }
        request.ledger->bind_sampling(prepared_sampling, model.resources_);
        request.progress.adopt_committed_history(request.ledger->committed_count);
        request.history = std::move(history);
        request.session = std::move(adopted);
        request.reused_tokens = reused_tokens; request.prefix_hits = 1;
        request.prefix_full_hits = full_hit ? 1 : 0;
        request.turn_input_consumed = reused_tokens;
        call.name = std::move(hit_name);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          request.prefix_sampling_calls = suffix == 0 ? 1 : 0;
          request.prefix_sampling_seconds = model.frontier_sampling_seconds - prior_sampling_seconds;
        }
        hit.reset();
      } else { request.prefix_misses = 1; request.history = std::make_shared<cache::ReplayHistory>(model.capacity); }
    }
    call.input_begin = request.turn_input_consumed;
    const auto part = prefill.next(request.prompt.size(), call.input_begin);
    call.prefill_complete = part.complete;
    call.completion = call.prefill_complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
    call.prompt = request.prompt.subspan(part.begin, part.rows);
    const auto lag = request.session && request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : call.prompt.size() == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
    if (!request.session) {
      make_generation_room(model, prefix, nullptr, requests);
      const std::array chunks{call.prompt.size()}; request.session = model.create_session(chunks, {}, call.role, request.progress.initial_sampling, false, call.completion, request.sampling_index);
    }
    if (!request.ledger->sampling) request.ledger->bind_sampling(prepared_sampling, model.resources_);
    if (!request.turn_observations.empty() &&
        call.input_begin == (request.progress.turn_index ? 0 : request.reused_tokens))
      request.turn_observations[request.progress.turn_index].context_before = request.session->length();
    call.full_hit = call.prompt.empty();
    if (call.name.empty()) {
      call.name = request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_prefill" : "prefill";
      if (!call.prefill_complete || call.input_begin > request.reused_tokens)
        call.name += "_chunk" + std::to_string(call.input_begin);
    }
  } else {
    call.quota = (std::min)(request.progress.remaining_outputs(), output_credit);
    call.pending = request.session->head->next_id;
    if (request.progress.generated.empty() || call.pending != request.progress.generated.back())
      throw std::runtime_error("generation pending differs from its last observed output");
    const auto remaining = model.capacity - request.session->length();
    if (!remaining) throw std::runtime_error("MTP has exhausted the configured context capacity");
    request.ledger->bind_sampling(prepared_sampling, model.resources_);
    call.maximum = model.cache_plan.draft_cache
        ? (std::min)({model.max_drafts(), call.quota - 1, remaining - 1}) : 0;
    const bool one = call.maximum == 0;
    const auto lag = request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : one ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : !model.recursive_mtp() ? (lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : GraphRole::kVerify;
    call.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
        (model.cache_plan.draft_cache ? "mtp" : "decode") + std::to_string(request.rounds.size() + 1);
  }
  if (call.completion == TextCompletion::kStateOnly) call.logits = {};
  return call;
}

bool make_generation_batch_room(Model &model, cache::PrefixCache *prefix,
    std::span<const PreparedGenerationCall> calls, std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const auto selected = calls | std::views::transform([](const auto &call) {
    return version_holder(call.request->session.get());
  });
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
  });
  const auto observe = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation batch admission requires retired heads within N");
    return inference::VersionRoomObservation{{live, limit}, prefix ? prefix->size() : 0,
        inference::version_admission({live, limit}, selected, residents,
            [&](const auto *head) { return prefix && prefix->retains(head); }).fits};
  };
  return inference::make_version_room(observe, [&] { return prefix && prefix->evict_one(); }) ==
      inference::VersionRoomResult::kReady;
}

void finish_generation_call(Model &model, PreparedGenerationCall &call) {
  if (call.disposition == EvaluationDisposition::kCancelled) return;
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (call.full_hit && request.progress.prefilled) return; // A committed cache hit is observed once.
  if (call.full_hit) {
    const auto next_id = request.session->head->next_id;
    const auto emitted = std::span<const std::int32_t>(&next_id, 1);
    const bool stopped = call.role == GraphRole::kText
        ? request.stop_policy.stopped(next_id, is_end_of_generation(next_id)) : is_end_of_generation(next_id);
    request.progress.preflight_outputs(emitted, true, true, stopped);
    // The imported request ledger already adopted source history. The new
    // frontier output is observed directly, without inventing a model ordinal.
    request.progress.apply_outputs(emitted, true, true, stopped);
    request.prefill_endpoint = Clock::now(); request.endpoint = request.prefill_endpoint;
  } else observe_generation_commits(request);
  if (!call.generation && call.prefill_complete) {
    request.endpoint = request.prefill_endpoint;
  }
  request.visible_tokens = request.session->length();
  request.draft_frontier = request.session->head->draft ? request.session->head->draft->frontier : 0;
  request.draft_lag = request.session->head->draft ? request.session->head->draft->lag : 0;
  request.fresh_next_id = request.session->head->next_id;
  if (!request.turn_observations.empty()) {
    auto &turn = request.turn_observations[request.progress.turn_index];
    if (!call.generation && call.prefill_complete) { turn.start = request.prefill_start; turn.prefill_endpoint = request.prefill_endpoint; }
    turn.endpoint = request.endpoint;
    turn.output_end = request.progress.generated.size();
    turn.context_after = request.visible_tokens;
  }
  if (call.full_hit)
    std::cerr << "prefix_hit name=prefill reused_tokens=" << request.reused_tokens
              << " prompt_graph_calls=0 frontier_sampling_graph_calls=" << request.prefix_sampling_calls
              << " visible=" << request.session->length() << '\n';
  if (!call.generation && call.prefill_complete)
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_end\n"
              << "memory_phase=" << request.memory_name << "_decode_begin\n";
  if (request.progress.turn_done) std::cerr << "memory_phase=" << request.memory_name << "_decode_end\n";
  if (call.full_hit) {
    if (!request.trace.empty()) {
      const auto path = request.trace / "prefill"; std::filesystem::create_directories(path);
      std::ofstream metadata(path / "cache-hit.json");
      if (!metadata) throw std::runtime_error("cannot write MTP prefix-hit metadata");
      metadata << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << request.prefix_sampling_calls
               << ",\"graph_calls\":" << request.prefix_sampling_calls << ",\"tensor_payload_available\":false,\"reused_tokens\":"
               << request.reused_tokens << ",\"next_id\":" << request.progress.generated.back() << "}\n";
    }
  }
  (void)model;
}

template<class AfterSubmit, class BatchObserver>
bool execute_ordinary_continuation(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    GenerationContinuation &continuation, AfterSubmit &after_submit, BatchObserver &batch_observer) {
  if (!continuation.enabled || continuation.unavailable || model.cache_plan.in_flight_batches < 2 ||
      model.cache_plan.draft_cache || model.scheduler || model.execution.host_only() ||
      model.execution.native_generated_storage() || model.execution.has_cpu_assist() ||
      std::ranges::any_of(calls, [&](const auto &call) {
        const auto &request = *call.request;
        const auto &sampling = request.session->sampling_configuration().configuration;
        return sampling.enabled || !sampling.modifiers.empty() || call.role != GraphRole::kText ||
            call.completion != TextCompletion::kEmitNext || call.full_hit || !call.generation ||
            (trace_states && !request.trace.empty());
      })) {
    continuation.reset(); return false;
  }
  if (!continuation.select(calls)) return false;
  if (continuation.ready && !continuation.matches(calls)) {
    continuation.reset(); if (!continuation.select(calls)) return false;
  }
  const auto slot = continuation.next_slot;
  const auto prior_slot = continuation.ready_slot;
  const bool imported = continuation.matches(calls);
  hyle::container::SmallVector<PreparedGenerationCall, 2> second;
  hyle::container::SmallVector<EvaluationRequest, 2> direct, relay;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<std::int32_t, 2> next;
  if (!second.try_reserve(calls.size()) || !direct.try_reserve(calls.size()) ||
      !relay.try_reserve(calls.size()) || !outputs.try_reserve(calls.size()) ||
      !next.try_resize(calls.size(), -1)) throw std::bad_alloc{};
  std::shared_ptr<StepSources> a_source, b_source;
  bool submitted_second = false, refused_second = false;
  Clock::time_point first_start{}, second_start{};
  std::optional<inference::EvaluationFailure> refused_failure;
  const auto notify = [&] {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) after_submit();
  };
  const auto release_and_abort = [&]() noexcept {
    b_source.reset(); a_source.reset(); continuation.reset();
    return model.abort_episode();
  };
  const auto require_retired = [](inference::RetirementObservation observation) {
    if (!observation.retired || (!observation.status.ok() &&
        (!observation.status.IsCancelled() || observation.status.has_native_error()))) {
      inference::EvaluationFailure failure(observation.status.ok()
          ? hyle::Status::Unavailable("ordinary continuation still retains its original episode")
          : observation.status);
      failure.observe_retirement(observation); throw failure;
    }
  };
  const auto discard_records = [&] {
    for (const auto &call : calls) {
      auto &ledger = *call.request->ledger;
      if (ledger.committed_count < ledger.records.size()) ledger.discard_uncommitted();
    }
  };
  const auto observe_commits = [&] {
    for (const auto &call : calls) observe_generation_commits(*call.request);
  };
  const auto finish = [&](std::span<PreparedGenerationCall> current) {
    const auto rows = model.active_step->input_tokens;
    const auto started = model.active_step->step_start;
    outputs.clear();
    for (auto &call : current) {
      auto &request = *call.request;
      outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
          &call.step_seconds, &request.endpoint, nullptr, nullptr, request.ledger.get()));
      outputs.back().disposition = &call.disposition;
    }
    model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()),
        [&](std::size_t j, const CompletedEvaluationView &view) {
          auto &call = current[j]; auto &request = *call.request;
          observe_completed_evaluation(model, outputs[j], {call.name, &request.trace, trace_states}, view);
        });
    double seconds = 0;
    for (const auto &call : current) seconds = (std::max)(seconds, call.step_seconds);
    if (!seconds) seconds = std::chrono::duration<double>(Clock::now() - started).count();
    observe_generation_batch(batch_observer, current, rows, seconds);
    for (auto &call : current) finish_generation_call(model, call);
  };
  const auto observe_cancelled = [&](std::span<PreparedGenerationCall> current,
      Clock::time_point started, Clock::time_point retired_at) {
    const auto seconds = std::chrono::duration<double>(retired_at - started).count();
    for (auto &call : current) {
      call.disposition = EvaluationDisposition::kCancelled; call.step_seconds = seconds;
    }
    observe_generation_batch(batch_observer, current, 1, seconds);
  };
  try {
    for (auto &call : calls) {
      auto &request = *call.request;
      prepare_generation_record(*request.ledger, *request.session, call.role, call.ids(), call.generation,
          call.quota, request.progress.turn_index, {}, call.maximum, call.completion,
          call.input_begin, call.prefill_complete);
      direct.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits});
      relay.push_back({request.session, {}, &request.ledger->stop_policy(), call.logits});
    }
    auto &a = imported ? model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), continuation.ready, slot)
        : (model.prepare_evaluations(std::span(direct.data(), direct.size()), nullptr, GraphRole::kText), *model.active_step);
    first_start = a.step_start; a_source = model.prepare_submission_sources(a);
    model.submit_evaluations(a, true);
    if (!a.sources) a_source.reset();
    notify();
    const auto abort_a = [&] {
      // Ordinary withdrawal can leave A's other members publishable. Only a
      // real whole-episode cancellation/failure takes this abort branch.
      return a.cancelled || std::ranges::any_of(calls, [](const auto &call) { return call.request->session->failed; });
    };
    if (abort_a()) {
      require_retired(release_and_abort()); discard_records();
      observe_cancelled(calls, first_start, Clock::now()); return true;
    }
    if (!a_source) {
      finish(calls); ++continuation.completed_steps; discard_records();
      continuation.reset(); continuation.unavailable = true; return true;
    }
    const auto next_slot = (slot + 1) % execution_slot_count;
    // Original B prepare/submit checks its complete tuple and Source. A's
    // possible future replay is not authority for B; warm flags are hints.
    if (imported && continuation.ready_steps >= 6 && continuation.direction_warm[next_slot] &&
        std::ranges::all_of(calls, [&](const auto &call) {
          const auto &request = *call.request; const auto &ledger = *request.ledger;
          return !request.session->stopping && !request.session->failed && call.quota >= 2 &&
              request.progress.remaining_outputs() >= 2 && model.capacity - request.session->length() >= 2 &&
              request.round_emitted.size() - request.rounds.size() >= 2 &&
              request.rounds.capacity() - request.rounds.size() >= 2 &&
              ledger.records.size() - ledger.committed_count >= 2;
        })) {
      for (std::size_t j = 0; j < calls.size(); ++j) {
        const auto &call = calls[j]; auto &request = *call.request;
        second.push_back(call); auto &following = second.back();
        following.pending = -1; following.quota = call.quota - 1;
        following.step_seconds = following.proposal_seconds = 0;
        following.disposition = EvaluationDisposition::kNotPublished;
        following.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
            "decode" + std::to_string(request.rounds.size() + 2);
        request.ledger->reserve_successor(request.progress.observed_commits(), following.quota,
            request.progress.turn_index, calls.size(), j);
      }
      StepState *b = nullptr;
      try { b = &model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), a_source, next_slot); }
      catch (const inference::EvaluationFailure &failure) {
        if (!failure.status().IsUnavailable() || failure.status().has_native_error() ||
            (failure.retirement_observed() && !failure.retirement().status.ok()) ||
            (model.successor_step && model.successor_step->submitted)) throw;
        refused_failure = failure; refused_second = true; ++continuation.successor_refusals;
      }
      if (b) {
        second_start = b->step_start; b_source = model.prepare_submission_sources(*b);
        model.submit_evaluations(*b); submitted_second = true; ++continuation.successor_submissions;
        notify();
      }
    }
    if (abort_a()) {
      if (submitted_second) ++continuation.successor_discards;
      require_retired(release_and_abort()); discard_records();
      const auto retired_at = Clock::now(); observe_cancelled(calls, first_start, retired_at);
      if (submitted_second) observe_cancelled(std::span(second.data(), second.size()), second_start, retired_at);
      return true;
    }
    finish(calls); ++continuation.completed_steps;
    continuation.ready.reset();
    if (refused_second) {
      inference::RetirementObservation retired{!model.active_step && !model.successor_step, {}};
      for (const auto &call : calls) {
        retired.retired = retired.retired && !call.request->session->pending;
        inference::record_retirement_status(retired.status, call.request->session->retirement_status);
      }
      refused_failure->observe_retirement(retired);
      if (!retired.retired || !refused_failure->retirement().status.ok()) throw *refused_failure;
      discard_records(); a_source.reset(); continuation.reset(); continuation.unavailable = true;
      return true;
    }
    if (submitted_second) {
      // Pending sources are a complete same-order cohort. A partial A result
      // cannot be projected into B. Retire all B, preserve every successful A,
      // and let the original owner choose the next runnable subset.
      if ((model.active_step && model.active_step->cancelled) ||
          std::ranges::any_of(calls, [](const auto &call) {
            const auto &request = *call.request;
            return call.disposition != EvaluationDisposition::kPublished || request.progress.turn_done ||
                request.session->stopping || request.session->failed || !request.progress.remaining_outputs();
          })) {
        ++continuation.successor_discards;
        require_retired(release_and_abort()); discard_records();
        const auto retired_at = Clock::now();
        observe_cancelled(std::span(second.data(), second.size()), second_start, retired_at);
        for (const auto &call : calls) {
          auto &request = *call.request; request.endpoint = retired_at;
          if (!request.turn_observations.empty()) request.turn_observations[request.progress.turn_index].endpoint = retired_at;
        }
        return true;
      }
      for (std::size_t j = 0; j < calls.size(); ++j) {
        auto &request = *calls[j].request;
        auto &record = request.ledger->bind_successor(generation_boundary(*request.session->head),
            request.progress.observed_commits(), request.progress.turn_index, request.progress.generated.back());
        second[j].pending = record.pending_input;
      }
      finish(std::span(second.data(), second.size())); ++continuation.completed_steps; ++continuation.successor_publications;
      a_source.reset(); continuation.ready = std::move(b_source); continuation.ready_slot = next_slot;
      continuation.next_slot = (next_slot + 1) % execution_slot_count;
    } else {
      continuation.ready = std::move(a_source);
      if (imported) {
        ++continuation.ready_steps;
        if (prior_slot && (*prior_slot + 1) % execution_slot_count == slot) continuation.direction_warm[slot] = true;
        continuation.ready_slot = slot; continuation.next_slot = next_slot;
      } else { continuation.ready_slot.reset(); continuation.next_slot = 0; }
      // Withdrawn A records have no commit; clear only their unpublished work.
      discard_records();
    }
    if (std::ranges::any_of(calls, [](const auto &call) {
          const auto &request = *call.request;
          return call.disposition != EvaluationDisposition::kPublished || request.progress.turn_done ||
              request.session->stopping || request.session->failed;
        }) || continuation.unavailable) continuation.reset();
    return true;
  } catch (inference::EvaluationFailure &failure) {
    failure.observe_retirement(release_and_abort()); observe_commits();
    if (!model.active_step && !model.successor_step) discard_records();
    throw;
  } catch (...) {
    (void)release_and_abort(); observe_commits();
    if (!model.active_step && !model.successor_step) discard_records();
    throw;
  }
}

// The serialized owner may process controls after the original native submit
// and before completion. Static optional dispatch adds no callback table, native
// executor or cost for the ordinary synchronous caller. A throwing control
// observer aborts through the original episode owner, retaining unretired work.
template<class AfterSubmit = std::nullptr_t, class BatchObserver = std::nullptr_t>
void execute_generation_calls(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    cache::PrefixCache *prefix = nullptr, std::span<const GenerationRequest> requests = {},
    AfterSubmit &&after_submit = nullptr, GenerationContinuation *continuation = nullptr,
    BatchObserver &&batch_observer = nullptr) {
  if (calls.empty()) return;
  // Validate every request's cold metadata before prefix eviction, proposal
  // work or preparing even the first uncommitted ledger slot.
  for (std::size_t member = 0; member < calls.size(); ++member) {
    const auto &call = calls[member];
    if (!call.request || !call.request->ledger)
      throw std::runtime_error("generation request requires its cold-prepared commit ledger");
    const auto &request = *call.request;
    if (!request.session || !request.session->head || request.session->stopping ||
        request.session->failed || request.session->pending)
      throw std::runtime_error("generation batch requires live idle Session heads");
    for (const auto &prior : calls.first(member))
      if (prior.request->session.get() == request.session.get())
        throw std::runtime_error("generation batch requires distinct Session owners");
    const auto &sampling = model.sampling_configuration(request.sampling_index);
    if (&request.session->sampling_configuration() != &sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    request.ledger->validate_sampling(sampling, model.resources_);
    if (call.completion == TextCompletion::kEmitNext) validate_logits_inputs(sampling.configuration, call.logits, model.vocabulary);
    if (request.ledger->stop_policy() != request.stop_policy)
      throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
    if ((call.full_hit && calls.size() != 1) ||
        call.request->sampling_index != calls[0].request->sampling_index || call.role != calls[0].role ||
        call.completion != calls[0].completion || call.generation != calls[0].generation ||
        (call.role != GraphRole::kVerify && call.ids().size() != calls[0].ids().size()))
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    if (!call.full_hit) {
      if (call.completion == TextCompletion::kStateOnly &&
          (call.role != GraphRole::kText || call.generation || call.prefill_complete))
        throw std::runtime_error("state-only ledger entry requires an incomplete ordinary prefill");
      if (call.ids().empty() || (call.role == GraphRole::kVerify && !call.generation))
        throw std::runtime_error("generation call requires its actual prepared input");
      (void)request.ledger->next();
      if (!call.generation) (void)request.warm_inputs.at(request.progress.turn_index);
    }
  }
  if (calls.size() == 1 && calls[0].full_hit) {
    if (continuation) continuation->reset();
    finish_generation_call(model, calls[0]); calls[0].disposition = EvaluationDisposition::kPublished; return;
  }
  if (!make_generation_batch_room(model, prefix, calls, requests)) throw std::runtime_error("MTP batch exceeds committed head budget");
  if (continuation) {
    if (execute_ordinary_continuation(model, calls, trace_states, *continuation,
        after_submit, batch_observer)) return;
    continuation->reset();
  }
  const auto &eos = model_eos_tokens;
  if (calls[0].role == GraphRole::kVerify) {
    for (auto &call : calls) {
      if (!call.generation || call.role != GraphRole::kVerify)
        throw std::runtime_error("recursive MTP batch requires generation verifiers only");
      if (call.verification_ids.empty()) {
        const auto begin = Clock::now();
        call.verification_ids = model.propose_recursive(*call.request->session, call.maximum, eos, &call.draft_log_probabilities);
        call.proposal_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      }
    }
    // EOS can shorten different requests to different exact widths. Preserve
    // real query geometry rather than padding or consuming a terminal token.
    std::map<std::size_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < calls.size(); ++i) groups[calls[i].ids().size()].push_back(i);
    if (groups.size() > 1) {
      for (const auto &[rows, members] : groups) {
        (void)rows;
        std::vector<PreparedGenerationCall> selected;
        for (const auto index : members) selected.push_back(std::move(calls[index]));
        try {
          execute_generation_calls(model, selected, trace_states, prefix, requests, after_submit, nullptr, batch_observer);
        } catch (const inference::EvaluationFailure &failure) {
          // A child index is not an index into the original request bucket.
          // Preserve the original error and actual retirement independently.
          const auto child = failure.participant();
          inference::EvaluationFailure mapped(failure.status(), child && *child < members.size()
              ? std::optional<std::size_t>(members[*child]) : std::nullopt);
          if (failure.retirement_observed()) mapped.observe_retirement(failure.retirement());
          throw mapped;
        }
        for (std::size_t j = 0; j < members.size(); ++j) calls[members[j]] = std::move(selected[j]);
      }
      return;
    }
  }
  const auto role = calls[0].role; const auto generation = calls[0].generation;
  hyle::container::SmallVector<EvaluationRequest, 2> inputs;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  if (!inputs.try_reserve(calls.size()) || !outputs.try_reserve(calls.size()) ||
      !selections.try_reserve(calls.size()) || !results.try_resize(generation && role != GraphRole::kText ? calls.size() : 0) || !next.try_resize(calls.size()))
    throw std::bad_alloc{};
  for (auto &call : calls) {
    auto &request = *call.request;
    // Recursive proposals may have shortened since the initial metadata pass.
    if (role == GraphRole::kVerify && call.ids().size() != calls[0].ids().size())
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    prepare_generation_record(*request.ledger, *request.session, role,
        generation ? std::span<const std::int32_t>(&call.pending, 1) : call.ids(), generation,
        call.quota, request.progress.turn_index, generation ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(request.warm_inputs.at(request.progress.turn_index)), call.maximum, call.completion, call.input_begin, call.prefill_complete);
    inputs.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits});
    outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
        &call.step_seconds, generation ? &request.endpoint : &request.prefill_endpoint,
        nullptr, nullptr, request.ledger.get()));
    outputs.back().disposition = &call.disposition;
    if (generation && role != GraphRole::kText) selections.push_back({request.session, call.pending, call.quota, call.draft_log_probabilities, call.maximum});
  }
  const auto observe = [&](std::size_t j, const CompletedEvaluationView &view) {
    const auto &call = calls[j];
    observe_completed_evaluation(model, outputs[j], {call.name, &call.request->trace, trace_states}, view);
  };
  model.prepare_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion); model.submit_evaluations();
  const auto input_rows = model.active_step->input_tokens;
  const auto episode_start = model.active_step->step_start;
  bool prepared_ahead = false;
  try {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) {
      try { after_submit(); }
      catch (inference::EvaluationFailure &failure) {
        failure.observe_retirement(model.abort_episode()); throw;
      }
      catch (...) { (void)model.abort_episode(); throw; }
    }
    // Controls retain their original post-submit ordering. Optional CPU-only
    // work follows them, and never turns its own allocation failure into a
    // failure or publication of this already submitted episode.
    if (generation && ((calls.size() > 1 && role == GraphRole::kText) || verifies_prefix(role) || has_mtp(role)) &&
        calls[0].completion == TextCompletion::kEmitNext) {
      const bool continues = std::ranges::all_of(calls, [](const auto &call) {
        const auto &request = *call.request;
        return call.quota > 1 && request.progress.remaining_outputs() > 1 && request.session &&
            !request.session->stopping && !request.session->failed;
      });
      if (continues) {
        (void)model.prepare_decode_ahead();
        prepared_ahead = model.decode_ahead && model.decode_ahead->ready;
      }
    }
    if (generation && role == GraphRole::kVerify) model.finish_recursive_mtp(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else if (generation && role != GraphRole::kText) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe);
    double target_seconds = 0, proposal_seconds = 0;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      target_seconds = (std::max)(target_seconds, calls[j].step_seconds);
      proposal_seconds += calls[j].proposal_seconds;
      calls[j].step_seconds += calls[j].proposal_seconds;
      if (calls[j].proposal_seconds) {
        auto &ledger = *calls[j].request->ledger;
        ledger.records[ledger.committed_count - 1].step_seconds += calls[j].proposal_seconds;
      }
    }
    // Shared target work is counted once; recursive proposals were serial
    // per member and are all retained. Cancelled members may have no output
    // endpoint, so a fully withdrawn batch uses its actual retired call end.
    if (target_seconds == 0) target_seconds = std::chrono::duration<double>(Clock::now() - episode_start).count();
    observe_generation_batch(batch_observer, calls, input_rows, target_seconds + proposal_seconds);
    for (auto &call : calls) finish_generation_call(model, call);
    if (prepared_ahead)
      for (const auto &call : calls)
        if (call.disposition != EvaluationDisposition::kPublished || call.request->progress.turn_done ||
            !call.request->session || call.request->session->stopping) {
          model.discard_decode_ahead(call.request->session.get());
        }
  } catch (...) {
    for (auto &call : calls) observe_generation_commits(*call.request);
    throw;
  }
}

void publish_generation_prompt(Model &model, PreparedGenerationCall &call, cache::PrefixCache *prefix) {
  if (!call.initial_prefix || call.full_hit || !call.prefill_complete ||
      call.disposition == EvaluationDisposition::kCancelled) return;
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  for (const auto &record : std::span(request.ledger->records).first(request.ledger->committed_count)) {
    if (record.imported || record.generation || record.turn != 0) continue;
    if (record.input_begin != request.history->length)
      throw std::runtime_error("complete prompt is missing its actual committed prefill chunk");
    request.history->append(request.input, record.input_begin, record.accepted, record.after.next_id,
        record.completion == TextCompletion::kStateOnly);
  }
  if (request.history->length != request.prompt.size())
    throw std::runtime_error("committed prefill chunks do not cover the complete prompt");
  if (request.session->head->draft && request.session->head->draft->lag)
    throw std::runtime_error("only a complete lag0 MTP prompt may enter the directory");
  prefix->publish(request.input, request.session->checkpoint(), {request.history, request.history->chunks.size(), request.prompt.size()});
  (void)model;
}

void publish_generation_completed(Model &model, GenerationRequest &request, cache::PrefixCache *prefix) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (!prefix || !request.progress.turn_done || !request.session || !request.session->head)
    return;
  std::vector<cache::ConsumedChunk> chunks;
  chunks.reserve(request.ledger->committed_count);
  for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
    const auto &record = request.ledger->records[i];
    if (!record.committed) throw std::runtime_error("completed checkpoint has a gap in its original committed ledger");
    chunks.push_back({record.committed_ids(), record.after.next_id, record.completion == TextCompletion::kStateOnly});
  }
  request.completed_checkpoint = cache::consumed_checkpoint(chunks, *request.session->head, model.capacity);
  prefix->publish(request.completed_checkpoint.input, request.session->checkpoint(), request.completed_checkpoint.replay);
}

void advance_generation_request(Model &model, GenerationRequest &request,
    std::size_t steps, bool trace_states, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  std::array calls{prepare_generation_call(model, request, steps, prefix, requests)};
  execute_generation_calls(model, calls, trace_states, prefix, requests);
  publish_generation_prompt(model, calls[0], prefix);
}

inference::RetirementObservation retire_generation_request(GenerationRequest &request) {
  // Stop/observe even on error; the runtime owner returns payload/key capacity
  // only after proof, before reporting the original observation status.
  (void)request.session->stop();
  return request.session->try_retire();
}

// Model-specific state and execution only. RequestLoop owns FIFO, idle
// selection and turn/key transitions; no native work is owned by this adapter.
template<class ExecuteWave>
struct GenerationRequestAdapter final {
  Model &model;
  cache::PrefixCache *prefix;
  std::span<GenerationRequest> requests;
  bool trace_states;
  std::vector<std::size_t> *evictions;
  ExecuteWave &execute;
  // Optional explicit policy; the default remains committed-token recovery.
  // Host storage has independent finite image-count and actual-byte limits.
  std::shared_ptr<inference::HostStateStorage> host_storage{};
  GenerationContinuation *continuation = nullptr;

  inference::RequestProgress &progress(inference::RequestKey key) const noexcept {
    return requests[key.slot].progress;
  }
  bool resident(inference::RequestKey key) const noexcept {
    return bool(requests[key.slot].session);
  }
  void admission_preflight(inference::RequestKey key) const {
    auto &request = requests[key.slot];
    // Release the previous request before any restoration/eviction can need
    // its state slots; execute_wave would be too late for this boundary.
    if (continuation) continuation->admission(key, request.progress.turn_index);
    preflight_generation_sampling(model, request);
    request.admission_start = Clock::now();
  }
  void admit(inference::RequestKey key, std::span<const inference::RequestKey>) const {
    auto &request = requests[key.slot];
    if (!request.session && request.host_image) {
      if (request.host_image->identity.sampling_index != request.sampling_index)
        throw std::runtime_error("parked request sampling configuration changed");
      make_generation_room(model, prefix, nullptr, requests);
      request.session = model.restore_session(request.host_image);
      request.host_image.reset(); // Only successful complete import releases it.
    }
    if (!request.session)
      restore_generation_session(model, request, prefix, trace_states, requests);
    if (request.progress.turn_index && !request.session)
      throw std::runtime_error("later generation turn requires its rebuilt complete head");
    if (!request.progress.turn_started()) {
      request.prompt = request.progress.begin_turn(*request.warm_inputs[request.progress.turn_index],
          request.progress.turn_index ? std::optional<std::int32_t>(request.session->head->next_id) : std::nullopt);
      request.turn_observations[request.progress.turn_index].output_begin = request.progress.turn_output_begin;
    }
  }
  void execute_wave(std::span<const inference::RequestKey> active) const { execute(active); }
  hyle::Status stop(inference::RequestKey key) const noexcept {
    if (continuation) continuation->release(key);
    const auto &request = requests[key.slot];
    return request.session ? request.session->stop() : hyle::Status::Ok();
  }
  void publish_turn(inference::RequestKey key) const {
    if (continuation) continuation->release(key);
    publish_generation_completed(model, requests[key.slot], prefix);
  }
  inference::RetirementObservation retire(inference::RequestKey key, inference::RequestRetirement reason) const {
    if (continuation) continuation->release(key);
    auto &request = requests[key.slot];
    if (!request.session) {
      if (reason != inference::RequestRetirement::kEvicted) request.host_image.reset();
      return {true, {}};
    }
    if (reason == inference::RequestRetirement::kEvicted && host_storage && !request.host_image)
      request.host_image = model.export_session(*request.session, host_storage);
    const auto retired = retire_generation_request(request);
    if (retired.retired) {
      request.session.reset();
      if (reason == inference::RequestRetirement::kEvicted) {
        ++request.idle_evictions;
        if (evictions) evictions->push_back(key.slot);
      }
    }
    return retired;
  }
};

void print_generation_rebuild(const GenerationRestoreStats &stats, bool draft_cache) {
  std::cout << ",\"restore_contract\":\"recompute_committed_inputs\",\"restored_graph_calls\":" << stats.graph_calls
      << ",\"restored_target_rows\":" << stats.target_rows;
  if (draft_cache) std::cout << ",\"restored_draft_rows\":" << stats.draft_rows
      << ",\"normalized_lag_rows\":" << stats.normalized_lag_rows;
  std::cout << ",\"decision_changes\":[";
  for (std::size_t i = 0; i < stats.decision_changes.size(); ++i) {
    const auto &change = stats.decision_changes[i];
    std::cout << (i ? "," : "") << "{\"episode\":" << change.episode << ",\"historical_next\":" << change.historical
        << ",\"fresh_next\":" << change.fresh << '}';
  }
  std::cout << ']';
}

void write_request_sampling(std::ostream &out, const Model &model, const GenerationRequest &request) {
  if (model.sampling_configuration_count() == 1) return;
  const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
  out << ",\"sampling_configuration_index\":" << request.sampling_index
      << ",\"request_sampling\":\"" << (configuration.enabled ? "random" : "greedy") << '\"'
      << ",\"request_temperature\":" << configuration.policy.temperature
      << ",\"request_top_k\":" << configuration.policy.top_k << ",\"request_top_p\":" << configuration.policy.top_p;
}

void print_generation_request(const Model &model, const GenerationRequest &request, std::size_t steps,
                       Clock::time_point startup, Clock::time_point ready) {
  const auto &prompt = request.prompt;
  const auto &generated = request.progress.generated;
  const auto &rounds = request.rounds;
  const auto prefill_start = request.turn_observations.empty() ? request.prefill_start : request.turn_observations.front().start;
  const auto prefill_endpoint = request.turn_observations.empty() ? request.prefill_endpoint : request.turn_observations.front().prefill_endpoint;
  const auto endpoint = request.endpoint;
  const auto prefill_step_seconds = request.prefill_step_seconds,
      decode_step_sum_seconds = request.decode_step_sum_seconds;
  const auto verified_rows = request.verified_rows, draft_rows = request.draft_rows,
      draft_rounds = request.draft_rounds, agreements = request.agreements, accepts = request.accepts;
  const auto &trace = request.trace;
  // The original finish endpoint follows publication and host token visibility,
  // before its trailing trace/log output. Inter-call work remains in elapsed.
  auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
  auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
  std::size_t logical_prompt = prompt.size(), turn_count = 1;
  if (!request.turn_observations.empty()) {
    prefill_seconds = 0; decode_seconds = 0; logical_prompt = request.progress.turns().size() - 1; turn_count = request.progress.turns().size();
    for (std::size_t i = 0; i < request.progress.turns().size(); ++i) {
      logical_prompt = cache::add(logical_prompt, request.progress.turns()[i].prompt.size());
      const auto &turn = request.turn_observations[i];
      prefill_seconds += std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count();
      decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count();
    }
  }
  const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
  std::cout << std::setprecision(17)
            << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
            << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << model.weight_inventory_report()
            << model.cache_report() << ",\"visible_tokens\":" << request.visible_tokens;
  write_request_sampling(std::cout, model, request);
  if (model.cache_plan.draft_cache) std::cout << ",\"draft_frontier\":" << request.draft_frontier
      << ",\"draft_lag\":" << request.draft_lag;
  std::cout << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
  std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
            << ",\"session_seconds\":" << session_seconds
            << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
            << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
            << ",\"prefill_step_seconds\":" << prefill_step_seconds + request.prefix_sampling_seconds
            << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
            << ",\"step_sum_seconds\":" << prefill_step_seconds + decode_step_sum_seconds + request.restored.step_seconds + request.prefix_sampling_seconds
            << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
            << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated.size() - turn_count) / decode_seconds : 0)
            << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
            << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
            << ",\"generated_decode_tokens\":" << generated.size() - turn_count << ",\"verified_target_rows\":" << verified_rows
            << (request.turn_observations.empty() ? ",\"actual_graph_calls\":" : ",\"graph_participations\":")
            << rounds.size() + request.prefill_graph_calls + request.restored.graph_calls + request.prefix_sampling_calls + request.auxiliary_graph_calls
            << ",\"prefix_sampling_graph_calls\":" << request.prefix_sampling_calls
            << ",\"sampling_stream\":" << request.progress.initial_sampling.stream
            << ",\"sampling_draw_cursor\":" << (request.ledger && request.ledger->committed_count
                ? request.ledger->records[request.ledger->committed_count - 1].after.sampling.draw
                : request.session ? request.session->head->sampling.draw : 0)
            << ",\"prefill_graph_calls\":" << request.prefill_graph_calls << ",\"decode_graph_calls\":" << rounds.size()
            << ",\"prefill_target_rows\":" << request.prefill_graph_tokens
            << ",\"actual_target_rows\":" << request.prefill_graph_tokens + verified_rows + request.restored.target_rows
            << ",\"committed_target_rows\":" << request.visible_tokens
            << ",\"committed_target_decode_rows\":" << request.visible_tokens - logical_prompt;
  if (model.cache_plan.draft_cache) std::cout << ",\"prefill_draft_rows\":" << request.prefill_draft_rows
      << ",\"actual_draft_rows\":" << request.prefill_draft_rows + draft_rows + request.restored.draft_rows
      << ",\"draft_rows\":" << draft_rows << ",\"draft_rounds\":" << draft_rounds
      << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts;
  std::cout << ",\"diagnostic\":" << (trace.empty() ? "false" : "true");
  if (!request.turn_observations.empty()) {
    std::cout << ",\"emitted_pending\":" << generated.back() << ",\"fresh_next_id\":" << request.fresh_next_id;
    print_generation_rebuild(request.restored, model.cache_plan.draft_cache);
  }
  if (!request.id.empty())
    std::cout << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"reused_tokens\":" << request.reused_tokens
              << ",\"prefix_hits\":" << request.prefix_hits << ",\"prefix_misses\":" << request.prefix_misses
              << ",\"prefix_full_hits\":" << request.prefix_full_hits;
  if (!request.turn_observations.empty()) {
    std::cout << ",\"between_turn_seconds\":" << (std::max)(0.0, session_seconds - prefill_seconds - decode_seconds)
              << ",\"phase_timing\":\"sum_per_turn_latency_including_interleaving\",\"turns\":[";
    for (std::size_t i = 0; i < request.turn_observations.size(); ++i) {
      const auto &turn = request.turn_observations[i];
      std::cout << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << request.progress.turns()[i].max_outputs
          << ",\"new_user_tokens\":" << request.progress.turns()[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
          << ",\"context_before\":" << turn.context_before << ",\"context_after\":" << turn.context_after
          << ",\"output_begin\":" << turn.output_begin << ",\"output_end\":" << turn.output_end
          << ",\"prefill_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count()
          << ",\"decode_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count()
          << ",\"input_ids\":[";
      for (std::size_t k = 0; k < request.warm_inputs[i]->size(); ++k)
        std::cout << (k ? "," : "") << (*request.warm_inputs[i])[k];
      std::cout << "]}";
    }
    std::cout << ']';
  }
  std::cout << ",\"rounds\":[";
  for (std::size_t i = 0; i < rounds.size(); ++i) {
    const auto &row = rounds[i];
    std::cout << (i ? "," : "") << "{\"pending\":" << row.pending;
    if (!model.cache_plan.draft_cache) {
      std::cout << ",\"next_id\":" << row.next_id << ",\"committed_target_rows\":" << row.processed_target << '}';
      continue;
    }
    std::cout << ",\"draft\":" << row.draft
              << ",\"target_first\":" << row.target_first << ",\"target_second\":" << row.target_second
              << ",\"accepted_draft\":" << (row.accepted_draft ? "true" : "false")
              << ",\"proposals\":" << row.proposal_count << ",\"verified_rows\":" << row.verified_rows
              << ",\"consumed_drafts\":" << row.consumed_drafts << ",\"auxiliary_graph_calls\":" << row.auxiliary_graph_calls
              << ",\"committed_target_rows\":" << row.processed_target << ",\"draft_rows\":" << row.processed_draft << '}';
  }
  std::cout << "]}";
}

template<GenerationMode Mode>
int run_generation_sessions(Model &model, std::span<const inference::GenerationSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans,
    std::span<const std::size_t> sampling_indices = {}, std::span<const inference::RequestStopPolicy> stop_policies = {},
    std::size_t max_prefill_rows = SIZE_MAX, bool warmup_planned_shapes = false,
    bool decode_overlap = false) {
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  if (!max_prefill_rows || (mtp && max_prefill_rows != SIZE_MAX))
    throw std::runtime_error("bounded prefill rows require a positive ordinary Text quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  if (!stop_policies.empty() && stop_policies.size() != schedules.size())
    throw std::runtime_error("one immutable stop policy is required per request");
  for (const auto &policy : stop_policies) {
    policy.validate(model.vocabulary);
    if (mtp && !policy.is_default()) throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  }
  if (!sampling_indices.empty() && sampling_indices.size() != schedules.size())
    throw std::runtime_error("one cold sampling choice is required per request");
  for (const auto index : sampling_indices) {
    (void)model.sampling_configuration(index);
    if (mtp && index) throw std::runtime_error("nondefault request sampling is ordinary Text only");
  }
  if (schedules.empty() || model.cache_plan.draft_cache != mtp)
    throw std::runtime_error("natural generation mode differs from its complete model/cache plan");
  if (prefix_enabled && token_plans.size() != schedules.size())
    throw std::runtime_error("natural generation requires one validated prefix input per request");
  const auto prompt_role = [](std::size_t rows, bool lag = false) {
    if constexpr (!mtp) { (void)rows; (void)lag; return GraphRole::kText; }
    else return rows == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
  };
  auto prefix = prefix_enabled ? std::make_unique<cache::PrefixCache>(model.cache_plan.resident_sessions) : nullptr;
  std::vector<GenerationRequest> requests;
  requests.reserve(schedules.size());
  std::size_t total_turns = 0, requested_outputs = 0, decode_bound = 0, restore_bound = 0, dispatch_bound = 0;
  for (const auto &schedule : schedules) {
    if (schedule.turns.empty()) throw std::runtime_error("natural request requires a nonempty turn list");
    if (schedule.turns.front().prompt.empty())
      throw std::runtime_error("natural request requires a nonempty initial prompt");
    std::size_t outputs = 0;
    for (const auto &turn : schedule.turns) {
      if (!turn.max_outputs)
        throw std::runtime_error("natural turns require positive output quota");
      outputs = cache::add(outputs, turn.max_outputs);
    }
    const auto &first = schedule.turns.front().prompt;
    requests.emplace_back(schedule.id, first, trace.empty() ? trace : trace /
        ("session" + std::to_string(requests.size()) + "_" + schedule.id), outputs, sampling_indices.empty() ? 0 : sampling_indices[requests.size()],
        stop_policies.empty() ? inference::RequestStopPolicy{} : stop_policies[requests.size()]);
    auto &request = requests.back(); request.progress.bind_turns(schedule.turns);
    request.progress.initial_sampling = {model.sampling_seed, static_cast<std::uint64_t>(requests.size() - 1), 0};
    request.turn_observations.resize(schedule.turns.size());
    request.warm_inputs.reserve(schedule.turns.size());
    for (std::size_t i = 0; i < schedule.turns.size(); ++i) {
      const auto &ids = schedule.turns[i].prompt;
      auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(ids.size(), i ? 1 : 0));
      std::copy(ids.begin(), ids.end(), owner->begin() + (i ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
    request.prompt = *request.warm_inputs[0];
    const auto ledger_capacity = generation_record_capacity(schedule.turns, max_prefill_rows, first.size());
    request.ledger = std::make_unique<GenerationLedger>(ledger_capacity, model.cache_plan.candidate_boundaries);
    dispatch_bound = cache::add(dispatch_bound, ledger_capacity);
    request.round_emitted.resize(outputs - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(model.cache_plan.candidate_boundaries);
    if (prefix) request.input = token_plans[requests.size() - 1];
    total_turns = cache::add(total_turns, schedule.turns.size());
    requested_outputs = cache::add(requested_outputs, outputs);
    decode_bound = cache::add(decode_bound, outputs - schedule.turns.size());
    if (model.cache_plan.resident_sessions < schedules.size())
      restore_bound = cache::add(restore_bound, cache::multiply(schedule.turns.size() - 1, ledger_capacity));
  }
  const std::array pair{std::size_t{2}};
  using PromptKey = std::pair<GraphRole, TextCompletion>;
  std::map<PromptKey, std::map<std::size_t, std::size_t>> prompt_uses;
  std::map<PromptKey, std::map<std::size_t, std::vector<std::size_t>>> support;
  const auto note_prompt = [&](GraphRole role, std::size_t tokens, std::size_t member,
                               TextCompletion completion = TextCompletion::kEmitNext, bool possible = false) {
    const PromptKey key{role, completion};
    auto &count = prompt_uses[key][tokens];
    count = possible ? (std::max)(count, std::size_t{1}) : cache::add(count, 1);
    auto &members = support[key][tokens];
    if (std::find(members.begin(), members.end(), member) == members.end()) members.push_back(member);
  };
  const auto note_prefill = [&](std::size_t tokens, std::size_t member, bool lag = false) {
    for (std::size_t consumed = 0; consumed < tokens;) {
      const auto part = prefill.next(tokens, consumed);
      const auto completion = part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
      note_prompt(prompt_role(part.rows, lag), part.rows, member, completion);
      consumed = part.begin + part.rows;
    }
  };
  for (std::size_t i = 0; i < requests.size(); ++i) {
    std::vector<std::size_t> suffixes{requests[i].prompt.size()};
    if (prefix) for (std::size_t j = 0; j < i; ++j) {
      const auto length = requests[j].prompt.size();
      if (length < requests[i].prompt.size() && cache::same_prefix(*requests[i].input, *requests[j].input, length))
        suffixes.push_back(requests[i].prompt.size() - length);
    }
    std::sort(suffixes.begin(), suffixes.end()); suffixes.erase(std::unique(suffixes.begin(), suffixes.end()), suffixes.end());
    for (auto tokens : suffixes) note_prefill(tokens, i);
    if constexpr (!mtp) if (prefix && prefill.bounded()) {
      // A generated checkpoint may match inside this prompt at any token, not
      // only at another request's original prompt boundary. Register its real
      // finite tail shapes and potential batch members before any carrier exists.
      prefill.for_each_shape(requests[i].prompt.size(), [&](std::size_t rows, bool complete) {
        note_prompt(GraphRole::kText, rows, i,
            complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly, true);
      });
    }
    for (std::size_t turn = 1; turn < requests[i].progress.turns().size(); ++turn) {
      const auto tokens = cache::add(requests[i].progress.turns()[turn].prompt.size(), 1);
      note_prefill(tokens, i);
      if constexpr (mtp) note_prefill(tokens, i, true);
    }
  }
  const std::array roles{GraphRole::kMtpRound, GraphRole::kMtpCatchupRound,
      GraphRole::kMtpSingle, GraphRole::kMtpCatchupSingle};
  // Normal rebuild preserves original committed chunks, not original proposals.
  // Every consumed verifier prefix can become a teacher-forced prefill chunk.
  prompt_uses[{prompt_role(1), TextCompletion::kEmitNext}].try_emplace(1, 0);
  if constexpr (mtp) for (std::size_t rows = 2; rows <= model.cache_plan.candidate_boundaries; ++rows)
    prompt_uses[{GraphRole::kMtpPrefill, TextCompletion::kEmitNext}].try_emplace(rows, 0);
  for (std::size_t sampling_index = 0; sampling_index < model.sampling_configuration_count(); ++sampling_index) {
    if (std::ranges::none_of(requests, [&](const auto &request) { return request.sampling_index == sampling_index; })) continue;
    const auto policy_decoding_requests = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
    }));
  for (std::size_t width = 1; width <= model.cache_plan.active_sequences; ++width) {
    std::map<PromptKey, std::map<std::size_t, std::size_t>> shapes;
    for (const auto &[key, tokens] : prompt_uses) for (const auto &[count, uses] : tokens) {
      const auto found_role = support.find(key);
      const auto eligible = found_role == support.end() || !found_role->second.contains(count) ? 0 : static_cast<std::size_t>(std::count_if(found_role->second.at(count).begin(), found_role->second.at(count).end(), [&](auto member) { return requests[member].sampling_index == sampling_index; }));
      if (width == 1 || width <= eligible) shapes[key][count] = cache::add(uses, width == 1 ? restore_bound : 0);
    }
    if (decode_bound && width <= policy_decoding_requests) {
      if constexpr (mtp) for (auto role : roles) {
        if (drafts_token(role) && (model.max_drafts() != 1 || model.recursive_mtp())) continue;
        auto &uses = shapes[{role, TextCompletion::kEmitNext}][drafts_token(role) ? 2 : 1]; uses = cache::add(uses, decode_bound);
      }
      else {
        auto &uses = shapes[{GraphRole::kText, TextCompletion::kEmitNext}][1];
        uses = cache::add(uses, decode_bound);
      }
    }
    if (mtp && decode_bound && width <= policy_decoding_requests && model.recursive_mtp())
      for (std::size_t rows = 1; rows <= model.cache_plan.candidate_boundaries; ++rows)
        shapes[{GraphRole::kVerify, TextCompletion::kEmitNext}][rows] = decode_bound;
    // R1 restoration also needs lag0 versions of each original warm chunk.
    if (mtp && width == 1) for (const auto &[tokens, uses] : prompt_uses[{GraphRole::kMtpCatchupPrefill, TextCompletion::kEmitNext}]) {
      (void)uses; shapes[{GraphRole::kMtpPrefill, TextCompletion::kEmitNext}].try_emplace(tokens, restore_bound);
    }
    for (const auto &[key, entries] : shapes) {
      const auto [role, completion] = key;
      std::vector<std::size_t> chunks; chunks.reserve(entries.size());
      for (const auto &[tokens, uses] : entries) { (void)uses; chunks.push_back(tokens); }
      model.prepare_family(chunks, false, width, role, false, completion, sampling_index);
      for (const auto &[tokens, uses] : entries) model.plan_replay_uses(tokens, width,
          cache::add(uses, width == 1 ? restore_bound : 0), role, completion, sampling_index);
    }
  }
  }
  if (mtp && decode_bound && model.recursive_mtp()) model.prepare_recursive_round_programs();
  model.freeze_shape_plan();
  if (warmup_planned_shapes) {
    const auto begin = Clock::now();
    const auto shapes = model.prepare_planned_execution();
    std::cerr << "shape_warmup policy=planned shapes=" << shapes
              << " seconds=" << std::chrono::duration<double>(Clock::now() - begin).count() << '\n';
  }
  const auto first_part = prefill.next(requests.front().prompt.size());
  const std::array first_prefill{first_part.rows};
  const auto first_completion = first_part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
  model.prepare_execution(first_prefill, 1, prompt_role(first_prefill[0]), first_completion, requests.front().sampling_index);
  if constexpr (!mtp) if (decode_bound) {
    const std::array decode{std::size_t{1}};
    for (std::size_t i = 0; i < model.sampling_configuration_count(); ++i)
      if (std::ranges::any_of(requests, [i](const auto &request) { return request.sampling_index == i &&
          std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }); }))
      {
        model.prepare_execution(decode, 1, GraphRole::kText, TextCompletion::kEmitNext, i);
        if (decode_overlap && model.cache_plan.in_flight_batches >= 2 &&
            !model.sampling_configuration(i).configuration.enabled &&
            model.sampling_configuration(i).configuration.modifiers.empty()) {
          const auto eligible = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [i](const auto &request) {
            return request.sampling_index == i && std::ranges::any_of(request.progress.turns(),
                [](const auto &turn) { return turn.max_outputs > 1; });
          }));
          for (std::size_t width = 1; width <= (std::min)(eligible, model.cache_plan.active_sequences); ++width) {
            if (width != 1) model.prepare_execution(decode, width, GraphRole::kText, TextCompletion::kEmitNext, i);
            model.prepare_relay_execution(width, i);
          }
        }
      }
  }
  if (mtp && decode_bound && model.max_drafts() == 1 && !model.recursive_mtp()) for (auto role : roles) if (drafts_token(role)) model.prepare_execution(pair, 1, role);
  inference::RequestLoop request_loop;
  request_loop.prepare(requests.size(), model.cache_plan.resident_sessions);
  for (std::size_t index = 0; index < requests.size(); ++index)
    requests[index].request_key = request_loop.enqueue(index);
  std::vector<std::size_t> dispatch, dispatch_turns;
  dispatch.reserve(dispatch_bound); dispatch_turns.reserve(dispatch_bound);
  std::vector<std::size_t> evictions; evictions.reserve(total_turns);
  std::size_t completed = 0, admitted = 0;
  struct BatchObservation { std::size_t tokens; GraphRole role; bool generation; hyle::container::SmallVector<std::size_t, 2> members; };
  std::vector<BatchObservation> batches; batches.reserve(dispatch_bound);
  std::size_t actual_prefill_calls = 0, actual_decode_calls = 0;
  std::size_t actual_fresh_participations = 0, actual_fresh_target_rows = 0;
  double actual_step_sum = 0;
  std::size_t peak_resident = 0, peak_committed = 0;
  const auto resident_count = [&] {
    return static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [](const auto &request) { return bool(request.session); }));
  };
  inference::BatchIndices batch_indices;
  batch_indices.prepare(model.cache_plan.resident_sessions, model.cache_plan.active_sequences);
  GenerationContinuation continuation; continuation.enabled = decode_overlap;
  if (decode_overlap) continuation.prepare(model.cache_plan.active_sequences);
  const auto ready = Clock::now();
  std::cerr << "generation_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n'
            << "measure_memory name=ready_before_first\n";
  const auto start = Clock::now(); auto endpoint = start;
  double prefill_seconds = 0, decode_seconds = 0;
  const auto observe_batch = [&](const GenerationBatchFacts &facts) {
    const auto participations = cache::add(actual_fresh_participations, facts.members.size());
    const auto rows = cache::add(actual_fresh_target_rows, cache::multiply(facts.input_rows, facts.members.size()));
    BatchObservation batch{facts.input_rows, facts.role, facts.generation, {}};
    if (!batch.members.try_reserve(facts.members.size())) throw std::bad_alloc{};
    for (const auto &member : facts.members) {
      const auto key = member.request->request_key;
      if (!request_loop.matches(key) || key.slot >= requests.size() || &requests[key.slot] != member.request)
        throw std::runtime_error("actual batch fact lost its original request identity");
      batch.members.push_back(key.slot);
    }
    // All vectors have a cold bound from the maximum actual episode count.
    // Prepare the variable-width member storage before committing any report.
    if (batches.size() == batches.capacity() || facts.members.size() > dispatch.capacity() - dispatch.size() ||
        facts.members.size() > dispatch_turns.capacity() - dispatch_turns.size())
      throw std::runtime_error("actual batch facts exceed their cold report capacity");
    batches.push_back(std::move(batch));
    for (const auto &member : facts.members) {
      dispatch.push_back(member.request->request_key.slot);
      dispatch_turns.push_back(member.request->progress.turn_index);
    }
    ++(facts.generation ? actual_decode_calls : actual_prefill_calls);
    actual_fresh_participations = participations; actual_fresh_target_rows = rows;
    actual_step_sum += facts.step_seconds;
  };
  const auto execute_wave = [&](std::span<const inference::RequestKey> active) {
    // Resolve every lookup/admission in this wave before any checkpoint is
    // published. A same-wave prompt cannot observe another member's new head.
    hyle::container::SmallVector<PreparedGenerationCall, 2> calls;
    if (!calls.try_reserve(active.size())) throw std::bad_alloc{};
    for (auto key : active) {
      auto &request = requests[key.slot];
      std::cerr << "generation_session id=" << request.id << " turn=" << request.progress.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      calls.push_back(prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests,
          request_loop.credit(key, request.progress.generated.size()), max_prefill_rows));
    }
    inference::dispatch_batches(batch_indices, calls.size(),
        [](std::size_t) { return true; },
        [&](std::size_t first) { return calls[first].full_hit ? std::size_t{1} : model.cache_plan.active_sequences; },
        [&](std::size_t first, std::size_t j) {
        return calls[j].full_hit == calls[first].full_hit &&
            calls[j].request->sampling_index == calls[first].request->sampling_index &&
            calls[j].generation == calls[first].generation && calls[j].role == calls[first].role &&
            calls[j].completion == calls[first].completion &&
            calls[j].planned_rows() == calls[first].planned_rows();
        }, [&](std::span<const std::size_t> selected) -> std::size_t {
          hyle::container::SmallVector<PreparedGenerationCall, 2> bucket;
          if (!bucket.try_reserve(selected.size())) throw std::bad_alloc{};
          for (auto j : selected) bucket.push_back(calls[j]);
          while (bucket.size() > 1 && !bucket[0].full_hit && !make_generation_batch_room(model, prefix.get(), std::span(bucket.data(), bucket.size()), requests)) {
            bucket.pop_back();
          }
          std::cerr << "generation_batch requests=" << bucket.size() << " T=" << bucket[0].planned_rows()
                    << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit << '\n';
          try {
            execute_generation_calls(model, std::span(bucket.data(), bucket.size()), trace_states, prefix.get(), requests,
                nullptr, decode_overlap ? &continuation : nullptr, observe_batch);
          } catch (const inference::EvaluationFailure &failure) {
            // The final submitted bucket may be smaller or differently ordered
            // than the active request wave. Attribute only its original member.
            throw inference::RequestExecutionFailure::from_evaluation(failure, bucket.size(),
                [&](std::size_t member) { return bucket[member].request->request_key; });
          }
          auto finished = endpoint;
          for (std::size_t j = 0; j < bucket.size(); ++j) {
            const auto index = active[selected[j]].slot;
            if (bucket[j].full_hit) {
              // A cache hit is a logical dispatch, never an invented Graph.
              dispatch.push_back(index); dispatch_turns.push_back(requests[index].progress.turn_index);
            }
            finished = (std::max)(finished, requests[index].endpoint);
          }
          const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
          (bucket[0].generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
          peak_resident = (std::max)(peak_resident, resident_count());
          peak_committed = (std::max)(peak_committed, std::size_t(model.snapshot_versions.live));
          if (model.snapshot_versions.live > model.cache_plan.resident_sessions) throw std::runtime_error("MTP committed version count exceeds N after publication");
          return bucket.size();
        });
    for (auto &call : calls) publish_generation_prompt(model, call, prefix.get());
    if (decode_overlap && continuation.ready)
      for (const auto &member : continuation.members)
        if (!request_loop.credit(member.key, requests[member.key.slot].progress.generated.size())) {
          continuation.release_ready(); break;
        }
  };
  GenerationRequestAdapter adapter{model, prefix.get(), std::span(requests), trace_states, &evictions, execute_wave,
      {}, decode_overlap ? &continuation : nullptr};
  while (completed != requests.size()) {
    const auto pumped = request_loop.pump_once(adapter);
    completed = cache::add(completed, pumped.completed);
    admitted = cache::add(admitted, pumped.admitted);
    if (pumped.waiting_retirement)
      throw std::runtime_error("generation request still retains an in-flight episode");
    if (!pumped.executed && completed != requests.size())
      throw std::runtime_error("generation admission has no runnable resident request");
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  std::size_t logical_prompt = 0, user_prompt = 0, prefill_tokens = 0, prefill_draft = 0;
  std::size_t generated = 0, verified = 0, drafted = 0, committed = 0;
  std::size_t draft_rounds = 0, agreements = 0, accepts = 0, hits = 0, misses = 0, full_hits = 0, reused = 0;
  std::size_t restores = 0, recomputed_changes = 0; GenerationRestoreStats restored; double step_sum_seconds = 0;
  for (const auto &request : requests) {
    for (const auto &turn : request.progress.turns()) user_prompt = cache::add(user_prompt, turn.prompt.size());
    prefill_tokens = cache::add(prefill_tokens, request.prefill_graph_tokens);
    prefill_draft = cache::add(prefill_draft, request.prefill_draft_rows);
    generated = cache::add(generated, request.progress.generated.size());
    verified = cache::add(verified, request.verified_rows); drafted = cache::add(drafted, request.draft_rows);
    committed = cache::add(committed, request.visible_tokens); draft_rounds = cache::add(draft_rounds, request.draft_rounds);
    agreements = cache::add(agreements, request.agreements); accepts = cache::add(accepts, request.accepts);
    hits = cache::add(hits, request.prefix_hits); misses = cache::add(misses, request.prefix_misses);
    full_hits = cache::add(full_hits, request.prefix_full_hits); reused = cache::add(reused, request.reused_tokens);
    step_sum_seconds += request.prefill_step_seconds + request.decode_step_sum_seconds;
    restores = cache::add(restores, request.restore_count);
    restored.graph_calls = cache::add(restored.graph_calls, request.restored.graph_calls);
    restored.target_rows = cache::add(restored.target_rows, request.restored.target_rows);
    restored.draft_rows = cache::add(restored.draft_rows, request.restored.draft_rows);
    restored.normalized_lag_rows = cache::add(restored.normalized_lag_rows, request.restored.normalized_lag_rows);
    recomputed_changes = cache::add(recomputed_changes, request.restored.decision_changes.size());
    restored.step_seconds += request.restored.step_seconds; restored.seconds += request.restored.seconds;
  }
  logical_prompt = cache::add(user_prompt, total_turns - requests.size());
  const auto graph_calls = cache::add(cache::add(cache::add(actual_prefill_calls, actual_decode_calls), restored.graph_calls),
      cache::add(model.frontier_sampling_graph_calls, cache::add(model.draft_program_graph_calls, model.prefix_decision_graph_calls)));
  std::cout << std::setprecision(17) << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
      << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << model.weight_inventory_report()
      << ",\"session_count\":" << requests.size() << ",\"turn_count\":" << total_turns << ",\"active_sequences\":" << model.cache_plan.active_sequences << ",\"max_sessions\":" << model.cache_plan.resident_sessions
      << ",\"decode_overlap\":" << (decode_overlap ? "true" : "false")
      << ",\"decode_overlap_available\":" << (decode_overlap && !continuation.unavailable ? "true" : "false")
      << ",\"overlap_successor_submissions\":" << continuation.successor_submissions
      << ",\"overlap_successor_publications\":" << continuation.successor_publications
      << ",\"overlap_successor_discards\":" << continuation.successor_discards
      << ",\"overlap_successor_refusals\":" << continuation.successor_refusals
      << model.cache_report() << ",\"prefix_cache\":" << (prefix_enabled ? "true" : "false")
      << ",\"admitted_requests\":" << requests.size() << ",\"admitted_turns\":" << admitted << ",\"retired_requests\":" << completed
      << ",\"peak_resident_requests\":" << peak_resident << ",\"peak_committed_versions\":" << peak_committed
      << ",\"live_committed_versions\":" << model.snapshot_versions.live << ",\"idle_evictions\":" << evictions.size()
      << ",\"prefix_hits\":" << hits << ",\"prefix_misses\":" << misses << ",\"prefix_full_hits\":" << full_hits << ",\"reused_tokens\":" << reused
      << ",\"prefix_entries\":" << (prefix ? prefix->size() : 0) << ",\"prefix_evictions\":" << (prefix ? prefix->evictions() : 0)
      << ",\"requested_output_tokens\":" << requested_outputs << ",\"generated_tokens\":" << generated << ",\"generated_decode_tokens\":" << generated - total_turns
      << ",\"actual_graph_calls\":" << graph_calls << ",\"prefill_graph_calls\":" << actual_prefill_calls << ",\"decode_graph_calls\":" << actual_decode_calls
      << ",\"graph_participations\":" << actual_fresh_participations + restored.graph_calls + model.frontier_sampling_graph_calls + model.draft_program_graph_calls + model.prefix_decision_graph_calls
      << ",\"logical_dispatches\":" << dispatch.size() << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"logical_user_prompt_tokens\":" << user_prompt
      << ",\"pending_flushed_tokens\":" << total_turns - requests.size() << ",\"prefill_target_rows\":" << prefill_tokens
      << ",\"verified_target_rows\":" << verified
      << ",\"actual_target_rows\":" << cache::add(actual_fresh_target_rows, restored.target_rows)
      << ",\"committed_target_rows\":" << committed << ",\"committed_target_decode_rows\":" << committed - logical_prompt
      << ",\"restore_count\":" << restores << ",\"restored_graph_calls\":" << restored.graph_calls << ",\"restored_target_rows\":" << restored.target_rows
      << ",\"restore_step_seconds\":" << restored.step_seconds << ",\"restore_seconds\":" << restored.seconds
      << ",\"restore_contract\":\"recompute_committed_inputs\"";
  if constexpr (mtp) std::cout << ",\"prefill_draft_rows\":" << prefill_draft << ",\"draft_rows\":" << drafted
      << ",\"actual_draft_rows\":" << cache::add(cache::add(prefill_draft, drafted), restored.draft_rows)
      << ",\"draft_rounds\":" << draft_rounds << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts
      << ",\"restored_draft_rows\":" << restored.draft_rows << ",\"normalized_lag_rows\":" << restored.normalized_lag_rows;
  std::cout << ",\"recomputed_decision_changes\":" << recomputed_changes
      << ",\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << session_seconds
      << ",\"step_sum_seconds\":" << actual_step_sum + restored.step_seconds + model.frontier_sampling_seconds
      << ",\"participation_step_sum_seconds\":" << step_sum_seconds << ",\"post_endpoint_seconds\":" << post_endpoint_seconds
      << ",\"phase_timing\":\"disjoint_continuous_episode_intervals\",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count() << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
      << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(requests.front().turn_observations.front().prefill_endpoint - startup).count()
      << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
      << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated - total_turns) / decode_seconds : 0)
      << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated / session_seconds : 0)
      << ",\"diagnostic\":" << (trace.empty() ? "false" : "true") << ",\"dispatch\":[";
  for (std::size_t i = 0; i < dispatch.size(); ++i)
    std::cout << (i ? "," : "") << "{\"session_id\":\"" << requests[dispatch[i]].id << "\",\"turn\":" << dispatch_turns[i] + 1 << '}';
  std::cout << "],\"batches\":[";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto &batch = batches[i];
    std::cout << (i ? "," : "") << "{\"index\":" << i << ",\"requests\":" << batch.members.size()
        << ",\"tokens\":" << batch.tokens << ",\"role\":" << static_cast<int>(batch.role)
        << ",\"generation\":" << (batch.generation ? "true" : "false") << ",\"session_ids\":[";
    for (std::size_t j = 0; j < batch.members.size(); ++j)
      std::cout << (j ? "," : "") << '"' << requests[batch.members[j]].id << '"';
    std::cout << "]}";
  }
  std::cout << "],\"sessions\":[";
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto &request = requests[i]; std::size_t quota = 0;
    for (const auto &turn : request.progress.turns()) quota = cache::add(quota, turn.max_outputs);
    std::cout << (i ? "," : "") << "{\"session_id\":\"" << request.id << "\",\"result\":";
    print_generation_request(model, request, quota, startup, ready);
    std::cout << ",\"restore_count\":" << request.restore_count << ",\"idle_evictions\":" << request.idle_evictions
        << ",\"checkpoint_resume_hits\":" << request.checkpoint_resume_hits
        << ",\"checkpoint_resume_misses\":" << request.checkpoint_resume_misses << ",\"ledger\":[";
    std::size_t round = 0;
    for (std::size_t j = 0; j < request.ledger->committed_count; ++j) {
      const auto &record = request.ledger->records[j]; if (record.generation) ++round;
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto token : record.ids()) hash = (hash ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
      const auto prefix_name = record.turn ? "round" + std::to_string(record.turn + 1) + "_" : "";
      const auto name = prefix_name + (record.generation ? (mtp ? "mtp" : "decode") + std::to_string(round) : "prefill");
      std::cout << (j ? "," : "") << "{\"index\":" << j << ",\"turn\":" << record.turn + 1 << ",\"role\":" << static_cast<int>(record.role)
          << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
          << ",\"generation\":" << (record.generation ? "true" : "false") << ",\"input_rows\":" << record.ids().size()
          << ",\"verified_rows\":" << record.verified_rows << ",\"draft_limit\":" << record.draft_limit
          << ",\"consumed_drafts\":" << record.consumed_drafts << ",\"auxiliary_graph_calls\":" << record.auxiliary_graph_calls
          << ",\"input_hash\":" << hash << ",\"first_input\":" << record.ids().front() << ",\"quota\":" << record.quota
          << ",\"before_length\":" << record.before.length << ",\"before_frontier\":" << record.before.frontier << ",\"before_lag\":" << record.before.lag
          << ",\"before_next_id\":" << record.before.next_id << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first << ",\"target_second\":" << record.target_second
          << ",\"sampling_seed\":" << record.before.sampling.seed << ",\"sampling_stream\":" << record.before.sampling.stream
          << ",\"sampling_draw_before\":" << record.before.sampling.draw << ",\"sampling_draw_after\":" << record.after.sampling.draw
          << ",\"accepted\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows << ",\"after_length\":" << record.after.length
          << ",\"after_frontier\":" << record.after.frontier << ",\"after_lag\":" << record.after.lag << ",\"next_id\":" << record.after.next_id
          << ",\"committed_input_ids\":[";
      if (record.generation) for (std::size_t k = 0; k < record.accepted; ++k)
        std::cout << (k ? "," : "") << record.accepted_inputs[k];
      std::cout << "],\"verification_input_ids\":[";
      for (std::size_t k = 0; k < record.verification_inputs.size(); ++k)
        std::cout << (k ? "," : "") << record.verification_inputs[k];
      std::cout << "],\"imported\":" << (record.imported ? "true" : "false")
          << ",\"source_geometry\":\"" << (record.imported ? "unknown" : "recorded")
          << "\",\"trace_name\":\"" << (record.imported ? "" : name) << "\"}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen35

int qwen35::run(int argc, char **argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Hyle inference engine (Qwen text)\n"
        "Usage: hyle --inputs DIR [options]\n"
        "  --inputs DIR                Locally prepared checkpoint and token IDs\n"
        "  --backend cpu|cuda|apple|vulkan  Primary backend; required when multiple GPU routes are compiled\n"
        "  --mtp off|on                Multi-token prediction (default: off)\n"
        "  --mtp-max-drafts N          Maximum proposals; 0 disables drafting (default: 1)\n"
        "  --mtp-min-probability FLOAT  Stop below max draft probability, [0,1] (default: 0)\n"
        "  --sampling greedy|random    Token selection (default: greedy)\n"
        "  --temperature FLOAT         Random policy temperature > 0 (default: 1)\n"
        "  --top-k N                    Random policy top-k; 0 disables (default: 0)\n"
        "  --top-p FLOAT                Random policy nucleus in (0,1] (default: 1)\n"
        "  --seed N                     Unsigned 64-bit random seed (default: 0)\n"
        "  --steps N                   Maximum output tokens (default: 4)\n"
        "  --capacity N                Total context capacity (default: 128)\n"
        "  --prefill-chunk-tokens N    Maximum real rows per ordinary prefill episode\n"
        "  --warmup-shapes lazy|planned  Prepare known carriers before ready (default: lazy)\n"
        "  --decode-overlap off|auto   Bounded CUDA greedy batch continuation (default: off)\n"
        "  --cpu-ffn-layers N          First N target FFNs on CPU; needs CPU+GPU build (default: 0)\n"
        "  --schedule FILE             One session's prompt/generate turns\n"
        "  --session-schedules FILE    Multiple named session schedules\n"
        "  --max-sessions auto|N       Bounded retained session states\n"
        "  --active-sequences auto|N   Maximum actual execution batch width\n"
        "  --cache-bytes auto|N        Session cache backing budget\n"
        "  --prefix-cache off|auto     Reuse completed prompt/turn checkpoints\n"
        "  --replay auto|commands|prefer-capture\n"
        "Prepare inputs with engine/models/qwen35/prepare.py; no model is downloaded at runtime.\n";
    return 0;
  }
  const auto startup = qwen35::Clock::now();
  try {
    std::filesystem::path input, continuation, trace, layer0_probe, decode_ids, schedule_file, session_file;
    std::size_t steps = 4, capacity = 128, max_sessions = 0, active_sequences = 0, cache_bytes = 0, max_drafts = 1, cpu_ffn_layers = 0;
    std::size_t max_prefill_rows = SIZE_MAX;
    bool warmup_planned_shapes = false, decode_overlap = false;
    bool progress = false, trace_states = true, explicit_steps = false;
    bool mtp_enabled = false;
    bool explicit_mtp_drafts = false;
    float min_draft_probability = 0;
    bool explicit_mtp_probability = false;
    qwen35::Model::Sampling sampler;
    bool explicit_sampling_parameters = false;
    bool single_schedule = false, multiple_schedules = false;
    std::string prefix_policy;
    bool explicit_continuation = false, explicit_decode_ids = false, preparation_diagnostics = false;
    auto replay = qwen35::ReplayPolicy::kAuto;
    qwen35::BackendSelection backend_selection;
    const auto positive_size = [](const std::string &option, const std::string &value) {
      std::size_t result = 0;
      const auto converted = std::from_chars(value.data(), value.data() + value.size(), result);
      if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(option + " requires a positive integer");
      return result;
    };
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("option needs a value");
      const std::string value = argv[++i];
      if (option == "--inputs") input = value;
      else if (option == "--prefill-chunk-tokens") max_prefill_rows = positive_size(option, value);
      else if (option == "--decode-overlap") {
        if (value != "off" && value != "auto") throw std::runtime_error("--decode-overlap requires off or auto");
        decode_overlap = value == "auto";
      }
      else if (option == "--warmup-shapes") {
        if (value != "lazy" && value != "planned")
          throw std::runtime_error("--warmup-shapes requires lazy or planned");
        warmup_planned_shapes = value == "planned";
      }
      else if (option == "--backend") {
        if (value == "cpu") backend_selection.primary = qwen35::BackendKind::kCpu;
        else if (value == "cuda") backend_selection.primary = qwen35::BackendKind::kCuda;
        else if (value == "apple") backend_selection.primary = qwen35::BackendKind::kApple;
        else if (value == "vulkan") backend_selection.primary = qwen35::BackendKind::kVulkan;
        else throw std::runtime_error("--backend requires cpu, cuda, apple or vulkan");
      }
      else if (option == "--sampling") {
        if (value != "greedy" && value != "random") throw std::runtime_error("--sampling requires greedy or random");
        sampler.enabled = value == "random";
      }
      else if (option == "--temperature" || option == "--top-p" || option == "--mtp-min-probability") {
        float number = 0;
        std::istringstream parsed(value); parsed.imbue(std::locale::classic());
        parsed >> std::noskipws >> number;
        const bool probability = option == "--mtp-min-probability";
        if (probability && (!parsed || parsed.peek() != std::char_traits<char>::eof() ||
            !std::isfinite(number) || number < 0 || number > 1))
          throw std::runtime_error("--mtp-min-probability requires a finite probability within [0,1]");
        if (!probability && (!parsed || parsed.peek() != std::char_traits<char>::eof() || !std::isfinite(number) ||
            number <= 0 || (option == "--top-p" && number > 1)))
          throw std::runtime_error(option + " requires a finite positive number" + (option == "--top-p" ? " at most one" : ""));
        if (probability) { min_draft_probability = number; explicit_mtp_probability = true; }
        else {
          if (option == "--temperature") sampler.policy.temperature = number; else sampler.policy.top_p = number;
          explicit_sampling_parameters = true;
        }
      }
      else if (option == "--top-k" || option == "--seed") {
        std::uint64_t number = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            (option == "--top-k" && number > INT32_MAX))
          throw std::runtime_error(option + " requires an unsigned integer" + (option == "--top-k" ? " no greater than the vocabulary" : ""));
        if (option == "--seed") sampler.seed = number; else sampler.policy.top_k = static_cast<std::size_t>(number);
        explicit_sampling_parameters = true;
      }
      else if (option == "--mtp") {
        if (value != "on" && value != "off") throw std::runtime_error("--mtp requires on or off");
        mtp_enabled = value == "on";
      }
      else if (option == "--mtp-max-drafts") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), max_drafts);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--mtp-max-drafts requires a nonnegative integer within the supported size range");
        explicit_mtp_drafts = true;
      }
      else if (option == "--cpu-ffn-layers") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), cpu_ffn_layers);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--cpu-ffn-layers requires a nonnegative integer within the supported size range");
      }
      else if (option == "--continuation") { continuation = value; explicit_continuation = true; }
      else if (option == "--schedule") { schedule_file = value; single_schedule = true; }
      else if (option == "--session-schedules") { session_file = value; multiple_schedules = true; }
      else if (option == "--max-sessions") max_sessions = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--active-sequences") active_sequences = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--cache-bytes") cache_bytes = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--prefix-cache") {
        if (value != "off" && value != "auto") throw std::runtime_error("--prefix-cache requires off or auto");
        prefix_policy = value;
      }
      else if (option == "--trace") trace = value;
      else if (option == "--preparation-diagnostics") {
        if (value != "0" && value != "1") throw std::runtime_error("--preparation-diagnostics requires 0 or 1");
        preparation_diagnostics = value == "1";
      }
      else if (option == "--trace-level") {
        if (value != "full" && value != "decisions")
          throw std::runtime_error("--trace-level requires full or decisions");
        trace_states = value == "full";
      }
      else if (option == "--decode-ids") { decode_ids = value; explicit_decode_ids = true; }
      else if (option == "--layer0-probe") layer0_probe = value;
      else if (option == "--steps") { steps = positive_size(option, value); explicit_steps = true; }
      else if (option == "--capacity") capacity = positive_size(option, value);
      else if (option == "--progress") {
        if (value != "direct" && value != "scheduler")
          throw std::runtime_error("--progress requires direct or scheduler");
        progress = value == "scheduler";
      }
      else if (option == "--replay") {
        if (value != "auto" && value != "commands" && value != "prefer-capture")
          throw std::runtime_error("--replay requires auto, commands or prefer-capture");
        replay = value == "auto" ? qwen35::ReplayPolicy::kAuto
            : value == "prefer-capture" ? qwen35::ReplayPolicy::kPreferCapture
                                         : qwen35::ReplayPolicy::kCommands;
      }
      else throw std::runtime_error("unknown option: " + option);
    }
    const bool scheduled = single_schedule || multiple_schedules;
    if (input.empty()) throw std::runtime_error("--inputs DIR is required");
    backend_selection = qwen35::select_execution(backend_selection, cpu_ffn_layers);
    auto weight_source = qwen35::prepare_weight_source(input / "weights.tsv", mtp_enabled);
    const auto geometry = weight_source.roles.geometry;
    qwen35::validate_mtp_source(geometry, mtp_enabled);
    qwen35::FfnPlacement::validate(cpu_ffn_layers, geometry.layers.size());
    if (sampler.policy.top_k > static_cast<std::size_t>(geometry.vocabulary))
      throw std::runtime_error("--top-k exceeds the source vocabulary");
    if (explicit_mtp_drafts && !mtp_enabled) throw std::runtime_error("--mtp-max-drafts requires --mtp on");
    if (explicit_mtp_probability && !mtp_enabled) throw std::runtime_error("--mtp-min-probability requires --mtp on");
    if (explicit_sampling_parameters && !sampler.enabled)
      throw std::runtime_error("--temperature, --top-k, --top-p and --seed require --sampling random");
    if (sampler.enabled && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("random sampling does not support layer or preparation diagnostics");
    const bool prefix_enabled = prefix_policy == "auto" || (prefix_policy.empty() && multiple_schedules);
    if (mtp_enabled) {
      if (explicit_continuation || explicit_decode_ids || !layer0_probe.empty())
        throw std::runtime_error("--mtp on uses natural generation; --continuation, --decode-ids and --layer0-probe are not supported");
      if (preparation_diagnostics)
        throw std::runtime_error("--preparation-diagnostics is not supported with --mtp on");

    }
    if (prefix_enabled && !scheduled) throw std::runtime_error("--prefix-cache auto requires a schedule");
    if (single_schedule && multiple_schedules)
      throw std::runtime_error("--schedule and --session-schedules are exclusive");
    using Grammar = qwen35::benchmark::ScheduleGrammar;
    const auto grammar = single_schedule ? qwen35::benchmark::schedule_grammar(schedule_file)
        : multiple_schedules ? qwen35::benchmark::session_schedule_grammar(session_file,
            std::vector<qwen35::benchmark::SessionManifestEntry>{}.max_size()) : Grammar::kPromptOnly;
    const bool natural_schedule = scheduled && grammar != Grammar::kFixedInput;
    const bool natural_single = !scheduled && !explicit_continuation && !explicit_decode_ids &&
        layer0_probe.empty() && !preparation_diagnostics;
    const bool natural_generation = natural_schedule || natural_single;
    if (max_prefill_rows != SIZE_MAX && (!natural_generation || mtp_enabled))
      throw std::runtime_error("--prefill-chunk-tokens requires ordinary natural generation; MTP and fixed replay are not supported");
    if (mtp_enabled && grammar == Grammar::kFixedInput)
      throw std::runtime_error("--mtp on requires natural prompt/generate requests; decode IDSFILE is fixed replay");
    if (natural_schedule && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("natural requests do not support layer or preparation diagnostics");
    if (scheduled && (explicit_continuation || explicit_decode_ids || (explicit_steps && !natural_schedule)))
      throw std::runtime_error("schedule inputs are exclusive with continuation, decode IDs and fixed-replay --steps");
    if ((single_schedule && schedule_file.empty()) || (multiple_schedules && session_file.empty()))
      throw std::runtime_error("schedule requires a nonempty manifest path");
    if (preparation_diagnostics && !scheduled)
      throw std::runtime_error("--preparation-diagnostics requires a fixed replay schedule");
    std::vector<qwen35::benchmark::SessionSchedule> schedules;
    std::vector<hyle::inference::GenerationSessionSchedule> generation_schedules;
    if (natural_schedule && single_schedule)
      generation_schedules.push_back({"default", qwen35::benchmark::read_generation_schedule(schedule_file,
          geometry.vocabulary, capacity, steps, explicit_steps)});
    else if (natural_schedule && multiple_schedules)
      generation_schedules = qwen35::benchmark::read_generation_session_schedules(session_file, geometry.vocabulary,
          capacity, std::vector<hyle::inference::GenerationSessionSchedule>{}.max_size(), steps, explicit_steps);
    else if (single_schedule)
      schedules.push_back({"default", qwen35::benchmark::read_schedule(schedule_file, geometry.vocabulary, capacity, true)});
    else if (multiple_schedules)
      schedules = qwen35::benchmark::read_session_schedules(session_file, geometry.vocabulary, capacity,
          std::vector<qwen35::benchmark::SessionSchedule>{}.max_size(), true);
    const auto input_sessions = natural_schedule ? generation_schedules.size() : scheduled ? schedules.size() : std::size_t{1};
    const auto recurrent_payload = qwen35::recurrent_bytes(qwen35::state_schema_for(geometry, capacity), geometry);
    // Configuration describes a maximum, never permission to overrun context.
    // Actual calls additionally clip against output quota and remaining rows.
    max_drafts = (std::min)(max_drafts, capacity - 1);
    if (max_sessions == 0) max_sessions = input_sessions;
    const auto cached_slots = (std::min)(max_sessions, input_sessions);
    if (active_sequences > max_sessions)
      throw std::runtime_error("--active-sequences cannot exceed --max-sessions");
    const auto forced = decode_ids.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(decode_ids, geometry.vocabulary, capacity);
    if (!decode_ids.empty()) {
      if (forced.empty()) throw std::runtime_error("--decode-ids requires a nonempty token tape");
      const auto forced_steps = qwen35::cache::add(forced.size(), 1);
      if (explicit_steps && steps != forced_steps)
        throw std::runtime_error("--steps must equal decode input count plus one");
      steps = forced_steps;
    }
    if (input.empty())
      throw std::runtime_error("usage: hyle --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--decode-overlap auto|off] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
    const auto prompt = scheduled ? std::vector<std::int32_t>{} : qwen35::read_ids(input / "prompt.ids", geometry.vocabulary, capacity);
    const auto extra = continuation.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(continuation, geometry.vocabulary, capacity);
    if (!scheduled) {
      if (prompt.empty()) throw std::runtime_error("generation requires a nonempty prompt before loading weights");
      if (prompt.size() > capacity || extra.size() > capacity - prompt.size() ||
          steps - 1 > capacity - prompt.size() - extra.size())
        throw std::runtime_error("prompt, continuation and requested generation exceed --capacity");
    }
    if (natural_single) generation_schedules.push_back({"default", {{prompt, steps}}});
    const inference::PrefillPolicy prefill(max_prefill_rows);
    std::vector<std::size_t> planned_chunks;
    std::size_t mtp_schema_rows = 0;
    bool mtp_has_decode = !scheduled && steps > 1;
    if (natural_generation) {
      for (const auto &request : generation_schedules) for (std::size_t i = 0; i < request.turns.size(); ++i) {
        const auto tokens = qwen35::cache::add(request.turns[i].prompt.size(), i ? 1 : 0);
        planned_chunks.push_back(prefill.next(tokens).rows);
        if (mtp_enabled) mtp_schema_rows = (std::max)(mtp_schema_rows, qwen35::cache::add(tokens, i ? 1 : 0));
        mtp_has_decode = mtp_has_decode || request.turns[i].max_outputs > 1;
      }
    } else if (scheduled) {
      for (const auto &request : schedules) for (const auto &phase : request.phases) {
        const auto tokens = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? phase.ids.size() : std::size_t{1};
        if (std::find(planned_chunks.begin(), planned_chunks.end(), tokens) == planned_chunks.end())
          planned_chunks.push_back(tokens);
      }
    } else {
      planned_chunks.push_back(prompt.size());
      if (!extra.empty()) planned_chunks.push_back(extra.size());
      planned_chunks.insert(planned_chunks.end(), steps - 1, 1);
    }
    // Immutable host token owners are bounded by the validated input batch.
    // Only suffixes at real prompt/turn boundaries enter the cold shape union.
    std::vector<qwen35::cache::TokenPlanRef> token_plans;
    if (prefix_enabled) {
      token_plans.reserve(input_sessions);
      if (natural_generation) for (const auto &request : generation_schedules) {
        token_plans.push_back(qwen35::cache::token_plan(request.turns.front().prompt, capacity));
      } else for (const auto &request : schedules) {
        std::vector<std::span<const std::int32_t>> chunks;
        chunks.reserve(request.phases.size());
        for (const auto &phase : request.phases) chunks.push_back(phase.ids);
        token_plans.push_back(qwen35::cache::token_plan(chunks, capacity));
      }
    }
    const auto max_query_rows = (std::max)({*std::max_element(planned_chunks.begin(), planned_chunks.end()), mtp_schema_rows,
        mtp_enabled && mtp_has_decode ? max_drafts + 1 : std::size_t{1}});
    const auto requested_active = (std::min)(active_sequences, input_sessions);
    // Reserve D2 only for the original device-ordered caller-owned route.
    // The original capacity formula already accounts for the actual batch R.
    const bool overlap_selected = decode_overlap && natural_generation && !mtp_enabled && !progress &&
        !sampler.enabled && backend_selection.primary == qwen35::BackendKind::kCuda && !cpu_ffn_layers &&
        replay != qwen35::ReplayPolicy::kCommands && (trace.empty() || !trace_states) &&
        std::ranges::any_of(generation_schedules, [](const auto &request) {
          return std::ranges::any_of(request.turns, [](const auto &turn) { return turn.max_outputs >= 10; });
        });
    const auto plan_for = [&](std::size_t residents) {
      return qwen35::cache::plan(geometry, weight_source.cache_row_bytes, capacity, residents,
          mtp_enabled && mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? residents : requested_active,
          mtp_enabled ? max_drafts + 1 : 1, mtp_enabled, sampler.enabled, overlap_selected ? 2 : 1);
    };
    const auto minimum_residents = active_sequences == 0 ? std::size_t{1} : requested_active;
    auto selected_cache = plan_for(cached_slots);
    // Caller-written pools have an exact cold formula. Native generated
    // results instead enter the same byte scope after real completion, before
    // any append is taken or head published. Do not estimate native N from the
    // CUDA pool formula or pretend auto bounds opaque execution allocations.
    // CPU cache metadata shares the payload's actual host resource. The GPU
    // payload formula is not an upper bound for that combined byte scope.
    // Auto keeps the configured version/batch bounds; an explicit byte cap
    // still accounts for and limits every cache allocation, including metadata.
    const bool native_generated_storage = backend_selection.primary == qwen35::BackendKind::kApple ||
        backend_selection.primary == qwen35::BackendKind::kVulkan;
    if (cache_bytes == 0) cache_bytes = (native_generated_storage || backend_selection.primary == qwen35::BackendKind::kCpu)
        ? SIZE_MAX : selected_cache.backing_bytes;
    else if (!native_generated_storage && selected_cache.backing_bytes > cache_bytes) {
      const auto minimum = plan_for(minimum_residents);
      if (minimum.backing_bytes > cache_bytes)
        throw std::runtime_error("--cache-bytes cannot hold the configured active sequences: requires " +
            std::to_string(minimum.backing_bytes) + " shared KV/RNN backing bytes");
      std::size_t low = minimum_residents, high = cached_slots;
      while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (plan_for(middle).backing_bytes <= cache_bytes) low = middle;
        else high = middle - 1;
      }
      selected_cache = plan_for(low);
    }
    const auto cache_slots = selected_cache.resident_sessions;
    const auto active_slots = selected_cache.active_sequences;
    const auto metadata_prepare_seconds = weight_source.metadata_prepare_seconds;
    const auto model_start = qwen35::Clock::now();
    qwen35::Model model(std::move(weight_source), geometry, capacity, selected_cache, layer0_probe, progress, replay, cache_bytes, sampler, cpu_ffn_layers, backend_selection);
    model.configure_draft_policy({max_drafts, min_draft_probability});
    const auto constructed = qwen35::Clock::now();
    if (!trace.empty()) model.trace_routes = trace / "routes";
    if (natural_generation) {
      if (mtp_enabled) return qwen35::run_generation_sessions<qwen35::GenerationMode::kMtp>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, SIZE_MAX, warmup_planned_shapes);
      return qwen35::run_generation_sessions<qwen35::GenerationMode::kText>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected);
    }

    // These owners are declared after Model, including on exception paths.
    // The schedule is consumer policy; every call uses the same Model/Graph chain.
    struct PhaseSummary {
      std::size_t context_before = 0, context_after = 0;
      double seconds = 0, step_sum_seconds = 0;
      std::string name, memory_name;
      qwen35::Clock::time_point start{};
    };
    struct Call { std::size_t phase, local; };
    // Count/idle policy only. Actual shared arena and state-pool backings own
    // the byte reservations; charging this directory per Session would double
    // count shared pages and immutable snapshots.
    hyle::inference::SessionCache resident_cache({cache_slots, (std::numeric_limits<std::size_t>::max)()});
    // Destroy runs (and their real slots) before this directory and Model.
    auto prefix_cache = prefix_enabled ? std::make_unique<qwen35::cache::PrefixCache>(cache_slots) : nullptr;
    struct SessionRun {
      struct RecomputedDecision { std::size_t chunk; std::int32_t previous, actual; };
      std::shared_ptr<qwen35::Session> session;
      hyle::inference::SessionCache::Pin pin;
      std::vector<std::size_t> chunks;
      std::shared_ptr<qwen35::cache::ReplayHistory> history;
      std::vector<PhaseSummary> summaries;
      std::vector<Call> calls;
      std::vector<std::int32_t> generated, consumed;
      std::vector<inference::StepObservation> observations;
      std::vector<inference::PreparationObservation> preparation;
      qwen35::cache::SamplingCursor saved_sampling;
      std::int32_t saved_pending = -1;
      std::filesystem::path trace;
      std::string trace_subdirectory;
      qwen35::Clock::time_point start{}, endpoint{};
      std::size_t visible_tokens = 0, cache_hits = 0, cache_misses = 0;
      std::size_t restored_calls = 0, restored_tokens = 0;
      std::vector<RecomputedDecision> recomputed_decisions;
      std::size_t graph_participations = 0, actual_graph_tokens = 0;
      std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
      std::size_t prefix_sampling_calls = 0;
      double step_sum_seconds = 0, first_token_elapsed_seconds = 0, cache_prepare_seconds = 0;
    };
    std::vector<SessionRun> runs(schedules.size());
    std::size_t peak_committed_versions = 0;
    const auto note_committed_versions = [&] {
      peak_committed_versions = (std::max)(peak_committed_versions, std::size_t(model.snapshot_versions.live));
    };
    std::size_t all_calls = 0;
    for (std::size_t s = 0; s < schedules.size(); ++s) {
      const auto &request = schedules[s];
      auto &run = runs[s];
      run.saved_sampling = {model.sampling_seed, static_cast<std::uint64_t>(s), 0};
      run.summaries.resize(request.phases.size());
      if (prefix_enabled) run.history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
      run.trace_subdirectory = "session" + std::to_string(s) + "_" + request.id;
      run.trace = trace.empty() || !multiple_schedules ? trace : trace / run.trace_subdirectory;
      for (std::size_t i = 0; i < request.phases.size(); ++i) {
        const auto &phase = request.phases[i];
        auto &summary = run.summaries[i];
        summary.name = "round" + std::to_string(phase.turn) +
            (phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? "_prefill" : "_decode");
        summary.memory_name = multiple_schedules ? request.id + "_" + summary.name : summary.name;
        const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
        const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
        for (std::size_t local = 0; local < count; ++local) {
          run.calls.push_back({i, local});
          run.chunks.push_back(is_prompt ? phase.ids.size() : std::size_t{1});
        }
      }
      if (run.calls.size() > (std::numeric_limits<std::size_t>::max)() - all_calls)
        throw std::runtime_error("total schedule call count overflows");
      all_calls += run.calls.size();
      run.generated.reserve(run.calls.size()); run.consumed.reserve(run.calls.size());
      run.observations.resize(run.calls.size());
      if (preparation_diagnostics) run.preparation.resize(run.calls.size());
      for (std::size_t i = 0; i < run.calls.size(); ++i) {
        const auto call = run.calls[i];
        run.observations[i].name = qwen35::benchmark::phase_name(request.phases[call.phase], call.local);
      }
    }
    const auto waves = scheduled ? qwen35::benchmark::make_schedule_waves(schedules, active_slots)
        : std::vector<qwen35::benchmark::ScheduleWave>{};
    std::vector<std::pair<std::size_t, std::size_t>> dispatch;
    dispatch.reserve(all_calls);
    std::vector<std::vector<std::size_t>> query_options(all_calls);
    std::vector<std::map<std::size_t, std::size_t>> published_at(schedules.size());
    std::vector<std::size_t> prior_tokens(schedules.size());
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      dispatch.emplace_back(call.session, call.call);
      prior_tokens[call.session] = qwen35::cache::add(prior_tokens[call.session], call.query_rows);
      const auto &phase = schedules[call.session].phases[call.phase];
      if (phase.kind == qwen35::benchmark::PhaseKind::kPrompt || call.local + 1 == phase.ids.size())
        published_at[call.session].emplace(prior_tokens[call.session], w);
    }
    std::fill(prior_tokens.begin(), prior_tokens.end(), 0);
    std::size_t restore_call_bound = 0;
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      auto &before = prior_tokens[call.session];
      const auto target = qwen35::cache::add(before, call.query_rows);
      if (call.kind == qwen35::benchmark::PhaseKind::kPrompt) {
        if (schedules.size() > cache_slots) restore_call_bound = qwen35::cache::add(restore_call_bound, before);
        if (prefix_enabled) for (std::size_t candidate = 0; candidate < token_plans.size(); ++candidate)
          for (const auto boundary : token_plans[candidate]->checkpoints) {
            const auto producer = published_at[candidate].find(boundary);
            if (boundary < before || boundary > target || producer == published_at[candidate].end() || producer->second >= w ||
                !qwen35::cache::same_prefix(*token_plans[candidate], *token_plans[call.session], boundary)) continue;
            query_options[call.ordinal].push_back(target - boundary);
          }
      }
      before = target;
    }
    const auto shape_uses = qwen35::benchmark::plan_batch_shape_uses(waves, query_options);
    if (!scheduled) {
      model.prepare_family(planned_chunks);
      if (!layer0_probe.empty()) model.prepare_family(planned_chunks, true);
      model.plan_execution(planned_chunks);
    } else {
      std::map<std::size_t, std::vector<std::size_t>> chunks_by_width;
      for (const auto &shape : shape_uses) chunks_by_width[shape.requests].push_back(shape.query_rows);
      for (const auto &[width, chunks] : chunks_by_width) {
        model.prepare_family(chunks, false, width);
        if (!layer0_probe.empty()) model.prepare_family(chunks, true, width);
      }
      if (!multiple_schedules && !prefix_enabled) model.plan_execution(runs.front().chunks);
      else for (const auto &shape : shape_uses)
        model.plan_replay_uses(shape.query_rows, shape.requests,
            qwen35::cache::add(shape.max_wave_calls, shape.requests == 1 ? restore_call_bound : 0));
    }
    struct BatchObservation {
      qwen35::benchmark::PhaseKind kind;
      std::size_t tokens;
      bool restore;
      double seconds;
      hyle::container::SmallVector<std::size_t, 4> sessions;
    };
    struct WaveObservation {
      int kind;
      std::size_t outputs = 0, logical_tokens = 0, actual_tokens = 0, graph_calls = 0;
      double seconds = 0;
    };
    std::vector<BatchObservation> batches;
    batches.reserve(all_calls);
    std::vector<WaveObservation> wave_observations;
    wave_observations.reserve(waves.size());
    std::size_t executed_graph_calls = 0, executed_graph_tokens = 0, max_batch_requests = 0;
    const auto record_batch = [&](qwen35::benchmark::PhaseKind kind, std::size_t tokens,
                                   std::span<const std::size_t> members, bool restore, double seconds) {
      BatchObservation value{kind, tokens, restore, seconds, {}};
      if (!value.sessions.try_reserve(members.size())) throw std::bad_alloc{};
      for (auto s : members) value.sessions.push_back(s);
      batches.push_back(std::move(value));
      executed_graph_calls = qwen35::cache::add(executed_graph_calls, 1);
      executed_graph_tokens = qwen35::cache::add(executed_graph_tokens, qwen35::cache::multiply(tokens, members.size()));
      max_batch_requests = (std::max)(max_batch_requests, members.size());
    };
    std::vector<std::size_t> evicted_sessions;
    evicted_sessions.reserve(all_calls);
    const auto make_initial_room = [&] {
      if (model.active_step) throw std::runtime_error("initial admission requires retired sessions");
      const std::array selected{qwen35::version_holder(nullptr)};
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache ? prefix_cache->size() : 0,
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache && prefix_cache->retains(head); }).fits};
      };
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache && prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("prefix/session heads retain the committed version budget before initial admission");
    };
    const auto make_batch_room = [&](std::span<qwen35::Session *const> owners) {
      if (owners.empty() || model.active_step || model.snapshot_versions.live > cache_slots)
        throw std::runtime_error("batch advance requires retired sessions within the committed version budget");
      for (auto *owner : owners) if (!owner || !owner->head || owner->pending || owner->failed || owner->stopping)
        throw std::runtime_error("batch advance contains a stopped, failed or pending session");
      // With prefix reuse disabled, each resident Session has its own sole
      // committed head. Advancing it replaces one version; no directory scan.
      if (!prefix_cache) return;
      const auto selected = owners | std::views::transform([](const auto *owner) {
        return qwen35::version_holder(owner);
      });
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache->size(),
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache->retains(head); }).fits};
      };
      // Dropping a directory alias need not free a version. Re-read both live
      // inventory and external retainers after EVERY eviction, including full hits.
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("batch commit exceeds N retained snapshots; release a checkpoint/session or increase the cache");
    };
    const auto make_advance_room = [&](qwen35::Session &owner) {
      const std::array owners{&owner}; make_batch_room(owners);
    };
    const auto ensure_resident = [&](std::size_t s, bool defer_restore) {
      auto &run = runs[s];
      if (run.pin) return;
      const auto begin = qwen35::Clock::now();
      const auto admission = resident_cache.admit(s, 0, [&](std::size_t victim) {
        auto &old = runs[victim];
        if (old.pin) return hyle::inference::RetirementObservation{};
        hyle::inference::RetirementObservation retired{true, {}};
        if (old.session) {
          (void)old.session->stop();
          retired = old.session->try_retire();
          if (!retired.retired) return retired;
        }
        old.session.reset(); // Actual state leases retire before quota returns.
        evicted_sessions.push_back(victim);
        return retired;
      });
      check(admission.retirement_status);
      if (!admission.accepted())
        throw std::runtime_error(admission.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven
            ? "session cache cannot prove retirement; the original owner remains charged"
            : "session cache admission exhausted; reduce --active-sequences or increase --max-sessions/--cache-bytes");
      run.pin = resident_cache.pin(s);
      if (!run.pin) throw std::runtime_error("session cache could not pin the admitted owner");
      if (run.session) ++run.cache_hits;
      else {
        ++run.cache_misses;
        if (!defer_restore) {
          make_initial_room();
          run.session = model.create_session(run.chunks, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute the original token/chunk tape; never resample or replace
          // already returned tokens. A different legal batch reduction may
          // change logits/greedy decisions. The rebuilt real head is authority
          // for new work; old decisions remain diagnostics, not state payload.
          for (std::size_t i = 0; i < run.generated.size(); ++i) {
            const auto call = run.calls[i];
            const auto &phase = schedules[s].phases[call.phase];
            const auto ids = phase.kind == qwen35::benchmark::PhaseKind::kPrompt
                ? std::span<const std::int32_t>(phase.ids)
                : std::span<const std::int32_t>(&phase.ids[call.local], 1);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_call" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != run.generated[i]) run.recomputed_decisions.push_back({i, run.generated[i], next});
            ++run.restored_calls; run.restored_tokens += ids.size();
            ++run.graph_participations; run.actual_graph_tokens += ids.size();
            const std::array members{s};
            record_batch(phase.kind, ids.size(), members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
            note_committed_versions();
          }
          if (run.session->length() != run.visible_tokens)
            throw std::runtime_error("restored cache context differs from retained consumer history");
          if (run.saved_pending >= 0) model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
        }
      }
      (void)model.cache_usage();
      run.cache_prepare_seconds += std::chrono::duration<double>(qwen35::Clock::now() - begin).count();
    };
    model.freeze_shape_plan();
    if (warmup_planned_shapes) {
      const auto begin = qwen35::Clock::now();
      const auto shapes = model.prepare_planned_execution();
      std::cerr << "shape_warmup policy=planned shapes=" << shapes
                << " seconds=" << std::chrono::duration<double>(qwen35::Clock::now() - begin).count() << '\n';
    }
    // Prefix mode admits at the timed request boundary, before any initial
    // state allocation. A full hit must not create or evaluate an empty state.
    if (scheduled && !prefix_enabled) {
      // Exact nominal buckets are known before admission. Suffix alternatives
      // stay cold metadata until an actual prefix hit selects one.
      if (!multiple_schedules) model.prepare_execution(runs.front().chunks);
      else for (const auto &wave : waves) {
        std::vector<std::size_t> rows;
        rows.reserve(wave.calls.size());
        for (const auto &call : wave.calls) rows.push_back(call.query_rows);
        for (const auto &bucket : qwen35::benchmark::bucket_schedule_wave(wave, rows)) {
          const std::array chunks{bucket.query_rows};
          model.prepare_execution(chunks, bucket.calls.size());
        }
      }
      if (schedules.size() > cache_slots) for (const auto &run : runs)
        model.prepare_execution(run.chunks);
    }
    if (!prefix_enabled) for (std::size_t s = 0; s < (std::min)(active_slots, runs.size()); ++s) ensure_resident(s, false);
    auto session = scheduled ? std::shared_ptr<qwen35::Session>{} : model.create_session(planned_chunks);
    if (session) model.prepare_execution(planned_chunks);
    inference::BatchIndices batch_indices;
    if (scheduled) batch_indices.prepare(active_slots, active_slots);
    std::vector<std::int32_t> generated, consumed;
    if (!scheduled) { generated.reserve(steps); consumed.reserve(steps - 1); }
    const auto ready = qwen35::Clock::now();
    std::cerr << "initialized model_seconds=" << metadata_prepare_seconds +
                  std::chrono::duration<double>(constructed - model_start).count()
              << " shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
              << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
    std::cerr << "cache_limits context=" << capacity << " max_sessions=" << max_sessions
              << " active_sequences=" << active_slots << " effective_cache_sessions=" << cache_slots
              << " recurrent_payload_bytes_per_snapshot=" << recurrent_payload
              << " backend=" << model.execution.name()
              << " shared_cache_backing_bytes=" << model.known_cache_backing_bytes
              << " native_execution_private_bytes_unknown=" << model.execution.native_generated_storage()
              << " cache_budget_scope=shared_pools_and_retained_output_backing";
    if (cache_bytes == SIZE_MAX) std::cerr << " cache_requested_byte_limit=none";
    else std::cerr << " cache_requested_byte_limit=" << cache_bytes;
    std::cerr << " physical_vram_limit_known=false\n";
    if (scheduled) {
      qwen35::Clock::time_point journey_start{}, endpoint{};
      struct PreparedCall {
        std::span<const std::int32_t> ids, logical_ids;
        std::size_t before = 0, reused = 0;
        bool used_prefix = false;
        qwen35::Clock::time_point call_start{};
        double complete_seconds = 0;
        std::int32_t next = -1;
      };
      std::vector<PreparedCall> prepared;
      std::vector<std::size_t> actual_rows, members;
      std::vector<qwen35::Session *> owners;
      std::vector<qwen35::EvaluationRequest> requests;
      std::vector<qwen35::EvaluationOutput> outputs;
      std::vector<std::int32_t> next_ids;
      prepared.reserve(active_slots); actual_rows.reserve(active_slots); members.reserve(active_slots);
      owners.reserve(active_slots); requests.reserve(active_slots); outputs.reserve(active_slots); next_ids.reserve(active_slots);
      for (const auto &wave : waves) {
        const auto wave_start = qwen35::Clock::now();
        if (journey_start == qwen35::Clock::time_point{}) journey_start = wave_start;
        const auto graphs_before = executed_graph_calls, tokens_before = executed_graph_tokens;
        WaveObservation wave_observation{wave.calls.front().kind == qwen35::benchmark::PhaseKind::kPrompt ? 0 : 1};
        prepared.clear(); prepared.resize(wave.calls.size()); actual_rows.clear();
        // Resolve the whole wave against PREVIOUS publications. A full hit
        // stays pinned and charged while the other members are evaluated.
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s];
          const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase];
          auto &summary = run.summaries[call.phase];
          if (call.local == 0) {
            summary.context_before = run.visible_tokens;
            std::cerr << "memory_phase=" << summary.memory_name << "_begin\n";
            summary.start = qwen35::Clock::now();
            if (index == 0) run.start = summary.start;
          }
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j];
          work.call_start = qwen35::Clock::now();
          if (is_prompt) ensure_resident(s, prefix_enabled);
          const auto logical_ids = is_prompt ? std::span<const std::int32_t>(phase.ids)
              : std::span<const std::int32_t>(&phase.ids[call.local], 1);
          auto ids = logical_ids;
          const auto before = run.visible_tokens;
          const auto target = before + logical_ids.size();
          std::size_t reused = 0;
          bool used_prefix = false;
        if (prefix_enabled && is_prompt) {
          // An equal-length checkpoint is useful after eviction, but must not
          // replace a resident same-boundary Session and lose its recordings.
          auto hit = prefix_cache->lookup(*token_plans[s], target,
              before + (run.session ? std::size_t{1} : std::size_t{0}), &model, model.math_identity());
          if (hit) {
            if (hit.payload.snapshot->next_id >= geometry.vocabulary)
              throw std::runtime_error("prefix next_id is outside the model vocabulary");
            reused = hit.payload.snapshot->length - before;
            const auto suffix = target - hit.payload.snapshot->length;
            std::vector<std::size_t> remaining;
            remaining.reserve(run.chunks.size() - index);
            if (suffix) remaining.push_back(suffix);
            remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index + 1), run.chunks.end());
            // All fallible host/IO preparation precedes withdrawal of the old
            // Session. These are real original Model slots, not shadow state.
            auto history = qwen35::cache::copy_history(hit.payload.replay, capacity);
            const auto cursor = run.session ? run.session->head->sampling : run.saved_sampling;
            auto adopted = model.create_session(remaining, hit.payload.snapshot, qwen35::GraphRole::kText, cursor, suffix == 0);
            if (model.default_sampler.enabled && !suffix) {
              history->chunks.back().next_id = adopted->head->next_id; ++run.prefix_sampling_calls;
            }
            if (run.session) {
              (void)run.session->stop();
              const auto retired = run.session->try_retire();
              if (retired.retired) run.session.reset();
              check(retired.status);
              if (!retired.retired)
                throw std::runtime_error("prefix adoption cannot retire the original session");
            }
            run.session = std::move(adopted);
            run.history = std::move(history);
            ids = logical_ids.subspan(reused);
            hit.reset(); // No temporary Snapshot owner may hide from advance admission.
            used_prefix = true; ++run.prefix_hits; run.reused_tokens += reused;
            if (ids.empty()) ++run.prefix_full_hits;
          } else ++run.prefix_misses;
        }
        if (prefix_enabled && !run.session) {
          make_initial_room();
          std::vector<std::size_t> remaining;
          remaining.reserve(run.history->chunks.size() + run.chunks.size() - index);
          for (const auto &part : run.history->chunks) remaining.push_back(part.count);
          remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index), run.chunks.end());
          run.session = model.create_session(remaining, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute actual evaluated chunks, including imported prefixes.
          // Keep any old shared history immutable: new checkpoint metadata
          // must describe the REAL rebuilt head and its recomputed decisions.
          auto rebuilt_history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
          for (std::size_t i = 0; i < run.history->chunks.size(); ++i) {
            const auto &part = run.history->chunks[i];
            const auto restored_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_chunk" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, restored_ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != part.next_id) run.recomputed_decisions.push_back({i, part.next_id, next});
            rebuilt_history->append(part.input, part.first, part.count, next);
            ++run.restored_calls; run.restored_tokens += restored_ids.size();
            ++run.graph_participations; run.actual_graph_tokens += restored_ids.size();
            const std::array restored_members{s};
            record_batch(phase.kind, restored_ids.size(), restored_members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
          }
          if (run.session->length() != before)
            throw std::runtime_error("actual-chunk recompute differs from the retained logical boundary");
          if (run.saved_pending >= 0) {
            model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
            if (model.default_sampler.enabled && !rebuilt_history->chunks.empty())
              rebuilt_history->chunks.back().next_id = run.saved_pending;
          }
          run.history = std::move(rebuilt_history);
        }
          work.ids = ids; work.logical_ids = logical_ids;
          work.before = before; work.reused = reused; work.used_prefix = used_prefix;
          actual_rows.push_back(ids.size());
          wave_observation.logical_tokens += logical_ids.size();
          if (wave_observation.kind != (is_prompt ? 0 : 1)) wave_observation.kind = -1;
          if (ids.empty()) {
            work.next = run.session->head->next_id;
            run.endpoint = qwen35::Clock::now();
            auto &observation = run.observations[index];
            observation.graph_executed = false; observation.graph_input_tokens = 0;
            observation.visible = run.session->length(); observation.next = work.next;
            observation.scheduled_progress = progress;
          }
        }
        // Validate the entire resolved wave before any bucket can submit.
        // A later incompatible member must not turn an old preflight error
        // into a partially committed wave.
        if (actual_rows.size() != wave.calls.size())
          throw std::runtime_error("batch wave query lengths do not match its calls");
        for (std::size_t j = 0; j < wave.calls.size(); ++j)
          if (actual_rows[j] > wave.calls[j].query_rows)
            throw std::runtime_error("batch wave actual query exceeds its logical input");
        inference::dispatch_batches(batch_indices, wave.calls.size(),
            [&](std::size_t index) { return actual_rows[index] != 0; },
            [&](std::size_t) -> std::size_t { return active_slots; },
            [&](std::size_t first, std::size_t j) {
              return wave.calls[j].kind == wave.calls[first].kind && actual_rows[j] == actual_rows[first];
            }, [&](std::span<const std::size_t> selected) -> std::size_t {
              const auto kind = wave.calls[selected.front()].kind;
              const auto query_rows = actual_rows[selected.front()];
              requests.clear(); outputs.clear(); owners.clear(); members.clear(); next_ids.resize(selected.size());
              for (const auto j : selected) {
                const auto &call = wave.calls[j]; auto &run = runs[call.session]; auto &work = prepared[j];
                requests.push_back({run.session, work.ids, nullptr, {}});
                owners.push_back(run.session.get()); members.push_back(call.session);
                outputs.push_back(qwen35::evaluation_output(model, run.session,
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}},
                    &work.complete_seconds, &run.endpoint, &run.observations[call.call],
                    preparation_diagnostics ? &run.preparation[call.call] : nullptr));
              }
              const auto batch_start = qwen35::Clock::now();
              make_batch_room(owners);
              inference::PreparationObservation preparation;
              model.prepare_evaluations(requests, preparation_diagnostics ? &preparation : nullptr);
              if (preparation_diagnostics) for (const auto j : selected)
                runs[wave.calls[j].session].preparation[wave.calls[j].call] = preparation;
              model.submit_evaluations();
              model.finish_evaluations(outputs, next_ids, [&](std::size_t at, const qwen35::CompletedEvaluationView &view) {
                const auto &call = wave.calls[selected[at]]; const auto &run = runs[call.session];
                qwen35::observe_completed_evaluation(model, outputs[at],
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}}, view);
              });
              const auto batch_endpoint = runs[members.front()].endpoint;
              record_batch(kind, query_rows, members, false,
                  std::chrono::duration<double>(batch_endpoint - batch_start).count());
              for (std::size_t at = 0; at < selected.size(); ++at) {
                const auto j = selected[at]; const auto &call = wave.calls[j];
                auto &run = runs[call.session]; auto &work = prepared[j]; work.next = next_ids[at];
                ++run.graph_participations; run.actual_graph_tokens += work.ids.size();
                if (prefix_enabled) run.history->append(token_plans[call.session], work.before + work.reused, work.ids.size(), work.next);
              }
              // No temporary strong Session owner may retain an old head across
              // next-wave adoption or eviction outside the charged directory.
              requests.clear(); outputs.clear();
              if (model.snapshot_versions.live > cache_slots)
                throw std::runtime_error("committed batch versions exceed the finite cache budget");
              note_committed_versions();
              return selected.size();
            });
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s]; const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase]; auto &summary = run.summaries[call.phase];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j]; auto &observation = run.observations[index];
          const auto logical_ids = work.logical_ids;
          const auto call_start = work.call_start;
          const auto next = work.next; const auto reused = work.reused; const auto used_prefix = work.used_prefix;
          // Per-session latency includes time waiting for its batch. Aggregate
          // throughput uses the wave wall once, never the sum of these spans.
          auto complete_seconds = std::chrono::duration<double>(run.endpoint - call_start).count();
          observation.step_seconds = complete_seconds;
          if (prefix_enabled) {
            observation.input_tokens = logical_ids.size();
            observation.prefix_hit = used_prefix; observation.reused_tokens = reused;
            if (model.snapshot_versions.live > cache_slots)
              throw std::runtime_error("committed prefix/session versions exceed the finite cache budget");
            if (!observation.graph_executed) {
              observation.seconds = complete_seconds;
              const auto write_hit_trace = [&](const std::filesystem::path &base) {
                if (base.empty()) return;
                const auto directory = base / observation.name;
                std::filesystem::create_directories(directory);
                std::ofstream output(directory / "cache-hit.json");
                output << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << (sampler.enabled ? 1 : 0)
                       << ",\"graph_calls\":" << (sampler.enabled ? 1 : 0) << ",\"tensor_payload_available\":false,\"visible_tokens\":"
                       << run.session->length() << ",\"next_id\":" << next << "}\n";
                if (!output) throw std::runtime_error("cannot write prefix-hit metadata trace");
              };
              write_hit_trace(run.trace);
              if (!layer0_probe.empty()) write_hit_trace(multiple_schedules ? layer0_probe / run.trace_subdirectory : layer0_probe);
            }
          }
          note_committed_versions();
          endpoint = (std::max)(endpoint, run.endpoint);
          run.visible_tokens = run.session->length();
          summary.step_sum_seconds += complete_seconds; run.step_sum_seconds += complete_seconds;
          run.generated.push_back(next);
          run.saved_sampling = run.session->head->sampling; run.saved_pending = next;
          if (!is_prompt) run.consumed.push_back(phase.ids[call.local]);
          if (index == 0) run.first_token_elapsed_seconds = std::chrono::duration<double>(run.endpoint - startup).count();
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 == count) {
            summary.context_after = run.visible_tokens;
            summary.seconds = std::chrono::duration<double>(run.endpoint - summary.start).count();
            std::cerr << "memory_phase=" << summary.memory_name << "_end\n";
            // Terminal prompts are real completed requests too.
            if (!is_prompt || call.phase + 1 == schedules[s].phases.size()) run.pin.reset();
          }
        }
        // Publication is a wave boundary. Same-wave calls never consume each
        // other's new checkpoint, matching the dependency and cold shape plan.
        if (prefix_enabled) for (const auto &call : wave.calls) {
          const auto &phase = schedules[call.session].phases[call.phase];
          const auto count = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 != count) continue;
          auto &run = runs[call.session];
          const qwen35::cache::ReplayPrefix history{run.history, run.history->chunks.size(), run.history->length};
          (void)prefix_cache->publish(token_plans[call.session], run.session->checkpoint(), history);
        }
        wave_observation.outputs = wave.calls.size();
        wave_observation.actual_tokens = executed_graph_tokens - tokens_before;
        wave_observation.graph_calls = executed_graph_calls - graphs_before;
        endpoint = qwen35::Clock::now(); // Include all result/checkpoint publication once.
        wave_observation.seconds = std::chrono::duration<double>(endpoint - wave_start).count();
        wave_observations.push_back(wave_observation);
      }
      for (std::size_t s = 0; s < runs.size(); ++s) {
        const auto &run = runs[s];
        if (multiple_schedules) std::cerr << "session_steps id=" << schedules[s].id << '\n';
        for (const auto &observation : run.observations) hyle::engine::reporting::write_step_observation(std::cerr, observation);
        if (preparation_diagnostics) for (std::size_t i = 0; i < run.observations.size(); ++i)
          if (run.observations[i].graph_executed)
            hyle::engine::reporting::write_preparation_observation(std::cerr, run.observations[i], run.preparation[i]);
      }
      // Define after the execution observations. Call once in the single
      // schedule's top-level object, or once in the multi-schedule top-level
      // object; do not call for each nested Session. Existing output sets
      // setprecision(17). Session IDs were validated as ASCII [A-Za-z0-9_-].
      const auto print_execution = [&] {
        std::cout << ",\"actual_graph_calls\":" << executed_graph_calls + model.frontier_sampling_graph_calls
                  << ",\"actual_graph_tokens\":" << executed_graph_tokens
                  << ",\"max_batch_requests\":" << max_batch_requests
                  << ",\"batches\":[";
        for (std::size_t i = 0; i < batches.size(); ++i) {
          const auto &batch = batches[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (batch.kind == qwen35::benchmark::PhaseKind::kPrompt ? "prefill" : "decode")
                    << "\",\"restore\":" << (batch.restore ? "true" : "false")
                    << ",\"query_rows\":" << batch.tokens
                    << ",\"requests\":" << batch.sessions.size()
                    << ",\"seconds\":" << batch.seconds << ",\"session_ids\":[";
          for (std::size_t j = 0; j < batch.sessions.size(); ++j)
            std::cout << (j ? "," : "") << '"' << schedules[batch.sessions[j]].id << '"';
          std::cout << "]}";
        }
        std::cout << "],\"waves\":[";
        for (std::size_t i = 0; i < wave_observations.size(); ++i) {
          const auto &wave = wave_observations[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (wave.kind == 0 ? "prefill" : wave.kind == 1 ? "decode" : "mixed")
                    << "\",\"outputs\":" << wave.outputs
                    << ",\"logical_tokens\":" << wave.logical_tokens
                    << ",\"actual_tokens\":" << wave.actual_tokens
                    << ",\"graph_calls\":" << wave.graph_calls
                    << ",\"seconds\":" << wave.seconds << '}';
        }
        std::cout << ']';
      };
      const auto print_result = [&](std::size_t s) {
        const auto &run = runs[s];
        const auto &phases = schedules[s].phases;
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << model.weight_inventory_report() << ",\"visible_tokens\":"
                  << run.visible_tokens << ",\"generated_ids\":[";
        for (std::size_t i = 0; i < run.generated.size(); ++i) std::cout << (i ? "," : "") << run.generated[i];
        std::cout << "],\"fixed_decode\":true,\"decode_input_ids\":[";
        for (std::size_t i = 0; i < run.consumed.size(); ++i) std::cout << (i ? "," : "") << run.consumed[i];
        std::cout << "],\"schedule\":true,\"phase_count\":" << phases.size()
                  << ",\"turn_count\":" << phases.back().turn
                  << ",\"processed_tokens\":" << run.visible_tokens
                  << ",\"generated_tokens\":" << run.generated.size()
                  << ",\"session_seconds\":" << std::chrono::duration<double>(run.endpoint - run.start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(run.endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << run.step_sum_seconds
                  << ",\"cache_hits\":" << run.cache_hits << ",\"cache_misses\":" << run.cache_misses
                  << ",\"restored_calls\":" << run.restored_calls << ",\"restored_tokens\":" << run.restored_tokens
                  << ",\"restore_contract\":\"recompute_original_inputs\""
                  << ",\"cache_prepare_seconds\":" << run.cache_prepare_seconds
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_hits\":" << run.prefix_hits << ",\"prefix_misses\":" << run.prefix_misses
                  << ",\"prefix_full_hits\":" << run.prefix_full_hits << ",\"reused_tokens\":" << run.reused_tokens
                  << ",\"graph_participations\":" << run.graph_participations + run.prefix_sampling_calls
                  << ",\"prefix_sampling_graph_calls\":" << run.prefix_sampling_calls
                  << ",\"sampling_stream\":" << run.saved_sampling.stream
                  << ",\"sampling_draw_cursor\":" << run.saved_sampling.draw
                  << ",\"graph_input_tokens\":" << run.actual_graph_tokens
                  << model.cache_report()
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false");
        std::cout << ",\"recomputed_decision_changes\":[";
        for (std::size_t i = 0; i < run.recomputed_decisions.size(); ++i) {
          const auto &change = run.recomputed_decisions[i];
          std::cout << (i ? "," : "") << "{\"chunk\":" << change.chunk
                    << ",\"previous\":" << change.previous << ",\"actual\":" << change.actual << '}';
        }
        std::cout << ']';
        if (multiple_schedules) std::cout << ",\"session_id\":\"" << schedules[s].id
            << "\",\"trace_subdirectory\":\"" << run.trace_subdirectory << '"';
        if (!multiple_schedules) print_execution();
        std::cout << ",\"phases\":[";
        for (std::size_t i = 0; i < phases.size(); ++i) {
          const auto &phase = phases[i]; const auto &summary = run.summaries[i];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          std::cout << (i ? "," : "") << "{\"turn\":" << phase.turn
                    << ",\"kind\":\"" << (is_prompt ? "prefill" : "decode") << "\",\"name\":\"" << summary.name
                    << "\",\"input_tokens\":" << phase.ids.size() << ",\"calls\":" << count << ",\"generated_tokens\":" << count
                    << ",\"context_before\":" << summary.context_before << ",\"context_after\":" << summary.context_after
                    << ",\"seconds\":" << summary.seconds << ",\"step_sum_seconds\":" << summary.step_sum_seconds << '}';
        }
        std::cout << "]}";
      };
      std::cerr << "first_token_elapsed_seconds=" << runs.front().first_token_elapsed_seconds << '\n';
      std::cout << std::setprecision(17);
      if (!multiple_schedules) print_result(0);
      else {
        std::size_t processed = 0, graph_participations = 0;
        std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
        double step_sum = 0;
        for (const auto &run : runs) {
          processed += run.visible_tokens; step_sum += run.step_sum_seconds;
          graph_participations += run.graph_participations + run.prefix_sampling_calls;
          prefix_hits += run.prefix_hits; prefix_misses += run.prefix_misses;
          prefix_full_hits += run.prefix_full_hits; reused_tokens += run.reused_tokens;
        }
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << model.weight_inventory_report() << ",\"session_count\":" << runs.size()
                  << ",\"execution\":\"equal_T_batched\",\"processed_tokens\":" << processed
                  << ",\"generated_tokens\":" << all_calls
                  << ",\"graph_participations\":" << graph_participations
                  << ",\"prefix_hits\":" << prefix_hits << ",\"prefix_misses\":" << prefix_misses
                  << ",\"prefix_full_hits\":" << prefix_full_hits << ",\"reused_tokens\":" << reused_tokens
                  << ",\"interleaved_seconds\":" << std::chrono::duration<double>(endpoint - journey_start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << step_sum
                  << ",\"max_cached_sessions\":" << max_sessions << ",\"active_sequences\":" << active_slots
                  << ",\"effective_cache_sessions\":" << cache_slots
                  << ",\"cache_recurrent_payload_bytes_per_snapshot\":" << recurrent_payload
                  << model.cache_report()
                  << ",\"cache_page_rows\":" << qwen35::cache::Plan::page_rows
                  << ",\"cache_max_physical_pages\":" << selected_cache.max_pages
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_retained_sessions\":" << resident_cache.retained_sessions()
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_entry_capacity\":" << (prefix_enabled ? cache_slots : 0)
                  << ",\"prefix_entries\":" << (prefix_cache ? prefix_cache->size() : 0)
                  << ",\"prefix_evictions\":" << (prefix_cache ? prefix_cache->evictions() : 0)
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false")
                  << ",\"logical_dispatch_session_ids\":[";
        for (std::size_t i = 0; i < dispatch.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[dispatch[i].first].id << '"';
        std::cout << "],\"evicted_session_ids\":[";
        for (std::size_t i = 0; i < evicted_sessions.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[evicted_sessions[i]].id << '"';
        std::cout << ']';
        print_execution();
        std::cout << ",\"sessions\":[";
        for (std::size_t s = 0; s < runs.size(); ++s) { if (s) std::cout << ','; print_result(s); }
        std::cout << "]}";
      }
      std::cout << '\n';
      return 0;
    }
    std::cerr << "measure_memory name=ready_before_first\nmemory_phase=generation_prefill_begin\n";
    double prefill_step_seconds = 0, continuation_step_seconds = 0, continuation_seconds = 0, decode_step_sum_seconds = 0;
    qwen35::Clock::time_point prefill_endpoint{};
    const auto prefill_start = qwen35::Clock::now();
    auto next = qwen35::evaluate(model, *session, prompt, "prefill", trace, trace_states,
        &prefill_step_seconds, &prefill_endpoint);
    auto endpoint = prefill_endpoint;
    std::cerr << "first_token_elapsed_seconds=" << std::chrono::duration<double>(prefill_endpoint - startup).count() << '\n';
    std::cerr << "memory_phase=generation_prefill_end\nmemory_phase=generation_decode_begin\n";
    if (!extra.empty()) {
      const auto continuation_start = qwen35::Clock::now();
      next = qwen35::evaluate(model, *session, extra, "continuation", trace, trace_states,
          &continuation_step_seconds, &endpoint);
      continuation_seconds = std::chrono::duration<double>(endpoint - continuation_start).count();
    }
    generated.push_back(next);
    for (std::size_t step = 1; step < steps && (!forced.empty() || !qwen35::is_end_of_generation(next)); ++step) {
      const auto previous = forced.empty() ? next : forced[step - 1];
      double step_seconds = 0;
      next = qwen35::evaluate(model, *session, std::span(&previous, 1), "decode" + std::to_string(step), trace, trace_states,
          &step_seconds, &endpoint);
      decode_step_sum_seconds += step_seconds;
      consumed.push_back(previous); generated.push_back(next);
    }
    std::cerr << "memory_phase=generation_decode_end\n";
    const auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
    const auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
    const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
    const auto graph_calls = 1 + std::size_t(!extra.empty()) + consumed.size();
    std::cout << std::setprecision(17)
              << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
              << "\",\"mtp\":false" << model.weight_inventory_report() << model.cache_report()
              << ",\"sampling_stream\":" << session->head->sampling.stream
              << ",\"sampling_draw_cursor\":" << session->head->sampling.draw
              << ",\"visible_tokens\":" << session->length() << ",\"generated_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
    std::cout << "],\"fixed_decode\":" << (forced.empty() ? "false" : "true") << ",\"decode_input_ids\":[";
    for (std::size_t i = 0; i < consumed.size(); ++i) std::cout << (i ? "," : "") << consumed[i];
    std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
              << ",\"session_seconds\":" << session_seconds
              << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
              << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
              << ",\"prefill_step_seconds\":" << prefill_step_seconds
              << ",\"continuation_seconds\":" << continuation_seconds << ",\"continuation_step_seconds\":" << continuation_step_seconds
              << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
              << ",\"step_sum_seconds\":" << prefill_step_seconds + continuation_step_seconds + decode_step_sum_seconds
              << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? prompt.size() / prefill_seconds : 0)
              << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? consumed.size() / decode_seconds : 0)
              << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
              << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
              << ",\"generated_decode_tokens\":" << consumed.size() << ",\"actual_graph_calls\":" << graph_calls
              << ",\"prefill_graph_calls\":1,\"decode_graph_calls\":" << consumed.size()
              << ",\"continuation_graph_calls\":" << std::size_t(!extra.empty())
              << ",\"prefill_target_rows\":" << prompt.size() << ",\"prefill_draft_rows\":0"
              << ",\"continuation_target_rows\":" << extra.size() << ",\"verified_target_rows\":" << consumed.size()
              << ",\"actual_target_rows\":" << prompt.size() + extra.size() + consumed.size() << ",\"actual_draft_rows\":0"
              << ",\"committed_target_rows\":" << session->length()
              << ",\"committed_target_decode_rows\":" << consumed.size()
              << ",\"draft_rows\":0,\"draft_rounds\":0,\"agreements\":0,\"consumed_drafts\":0"
              << ",\"decode_seconds_includes_continuation\":" << (extra.empty() ? "false" : "true")
              << ",\"diagnostic\":" << (!trace.empty() || !layer0_probe.empty() ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen3.5 text consumer: " << error.what() << '\n'; return 1;
  }
}
