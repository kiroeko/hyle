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
#include <exception>
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
#include <tuple>
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
#include "block_graph.h"
#include "../../inference/mtp_acceptance.h"
#include "../../inference/mtp_acceptance_graph.h"
#include "../../inference/sampling_graph.h"
#include "../../inference/logits_inputs.h"
#include "../../inference/ct_block128_graph.h"
#include "../../inference/execution_resources.h"
#include "../../inference/program_carrier.h"
#include "../../inference/program_slots.h"
#include "../../inference/execution_episode.h"
#include "../../inference/program_io.h"
#include "../../inference/graph_bindings.h"
#include "../../inference/prepared_program_family.h"
#include "../../inference/cold_shape_plan.h"
#include "../../inference/cold_demand.h"
#include "../../inference/slot_transactions.h"
#include "../../inference/append_transactions.h"
#include "../../inference/retirement_observation.h"
#include "../../inference/evaluation_failure.h"
#include "../../inference/request_execution_failure.h"
#include "../../inference/prepared_output.h"
#include "../../inference/request_progress.h"
#include "../../inference/commit_observation.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/request_window.h"
#include "../../inference/request_loop.h"
#include "../../inference/request_wave_execution.h"
#include "../../inference/continuation.h"
#include "../../inference/prefill_plan.h"
#include "../../inference/packed_rows.h"
#include "../../inference/paged_append.h"
#include "../../inference/version_admission.h"
#include "../../inference/batch_dispatch.h"
#include "../../inference/request_batch_execution.h"
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
#include "hyle/compute/sequence_ops.h"
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
      layer.recurrent = dense(Type::kFloat32, {1, geometry.gdn_value_heads, geometry.gdn_value_depth, geometry.gdn_key_depth});
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
  using Location = inference::GraphBindings::Location;
  struct Scope final {
    GraphAssembly &assembly;
    Location previous;
    Scope(GraphAssembly &selected, Location location)
        : assembly(selected), previous(selected.location_) { assembly.location_ = location; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { assembly.location_ = previous; }
  };

  std::vector<Tensor> literals;

  explicit GraphAssembly(const Backend &environment, PrecisionProfile precision = {},
                         const c::ResourceBudget &budget = {})
      : bindings_(environment.storage_domains()), backend_(environment), budget_(budget), precision_(precision), mixed_(environment.has_cpu_assist()),
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
    if (mixed_) check(bindings_.node(node, location_));
  }
  void external(V value, g::StorageDomainId storage = g::StorageDomainId{0}) {
    if (!mixed_) return;
    check(bindings_.external(value, storage));
  }
  std::span<const g::ConstantBinding> constants() const noexcept { return bindings_.constants(); }
  g::CompileOptions options(g::CompileOptions options) const {
    return bindings_.options(options);
  }
  void bind_constant(V value, Tensor &tensor) {
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

private:
  inference::GraphBindings bindings_;
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
// Typed ready inputs shared by an episode or prepared for each conditional.
// Bias/mask are immutable [1,V] snapshots; counts is C0 and mathematical
// prefix updates create later exact I32 counts. Copies retain
// the original read permission. Use a lease carrying its existing StorageOwner,
// or keep its borrowed allocator/resource/Context alive through every retained
// copy and native use. Owners may supply different leases next episode; no
// modifier payload enters a Snapshot or KV cache; the ledger holds only its
// own budgeted derived host counts, never external bias/mask payloads.
using inference::LogitsInput;
using inference::LogitsInputs;
struct PreparedSampling;
struct ColdResources;
void validate_sampling_configuration(const SamplingConfiguration &value, std::int64_t vocabulary) {
  if (vocabulary <= 0 || vocabulary > INT32_MAX || !std::isfinite(value.policy.temperature) || value.policy.temperature <= 0 ||
      !std::isfinite(value.policy.top_p) || value.policy.top_p <= 0 || value.policy.top_p > 1 ||
      value.policy.top_k > static_cast<std::size_t>(vocabulary) ||
      (!value.enabled && (value.policy.temperature != 1 || value.policy.top_k || value.policy.top_p != 1)) ||
      !sampling::valid_history_policy(value.modifiers.history))
    throw std::runtime_error("invalid finite sampling configuration for vocabulary");
}
bool same_sampling_configuration(const SamplingConfiguration &a, const SamplingConfiguration &b) noexcept {
  return a.enabled == b.enabled && a.policy.temperature == b.policy.temperature &&
      a.policy.top_k == b.policy.top_k && a.policy.top_p == b.policy.top_p && a.modifiers == b.modifiers;
}

// Only deterministic port/descriptor facts: no payload reads, lease admission
// or numeric distribution checks. Those remain with their original owners.
hyle::Status logits_inputs_metadata_status(const SamplingConfiguration &configuration,
    const LogitsInputs &inputs, std::int64_t vocabulary) noexcept {
  const auto require = [&](const LogitsInput &input, bool enabled, Type type) -> hyle::Status {
    if (input.lease.valid() != enabled || input.description.valid() != enabled)
      return hyle::Status::InvalidArgument("logits input does not match its cold port configuration");
    if (enabled) {
      const std::array<std::int64_t, 2> shape{1, vocabulary};
      HYLE_ASSIGN_OR_RETURN(const auto expected, c::TensorDesc::dense(type, shape));
      if (input.description != expected)
        return hyle::Status::InvalidArgument("logits input requires the exact cold dense [1,V] descriptor");
    }
    return hyle::Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(require(inputs.bias, configuration.modifiers.bias, Type::kFloat32));
  HYLE_RETURN_IF_ERROR(require(inputs.allowed, configuration.modifiers.allowed, Type::kBool));
  return require(inputs.counts, !configuration.modifiers.history.empty(), Type::kInt32);
}
void validate_logits_input_leases(const SamplingConfiguration &configuration, const LogitsInputs &inputs) {
  if (configuration.modifiers.bias) (void)take(inputs.bias.lease.tensor(inputs.bias.description));
  if (configuration.modifiers.allowed) (void)take(inputs.allowed.lease.tensor(inputs.allowed.description));
  if (!configuration.modifiers.history.empty()) (void)take(inputs.counts.lease.tensor(inputs.counts.description));
}
void validate_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
                            std::int64_t vocabulary) {
  check(logits_inputs_metadata_status(configuration, inputs, vocabulary));
  validate_logits_input_leases(configuration, inputs);
}
// The base retains the one C0 history. Conditional rows supply only immutable
// bias/mask snapshots. Descriptor errors can retain the original request key;
// lease permission and native checks remain with their original mechanisms.
hyle::Status prefix_logits_inputs_metadata_status(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional, std::size_t rows, std::int64_t vocabulary) noexcept {
  if (conditional.empty()) return logits_inputs_metadata_status(configuration, inputs, vocabulary);
  if (conditional.size() != rows || (!configuration.modifiers.bias && !configuration.modifiers.allowed))
    return hyle::Status::InvalidArgument("conditional logits require one configured bias/mask row per actual target including bonus");
  auto base = configuration, per_row = configuration;
  base.modifiers.bias = false; base.modifiers.allowed = false;
  per_row.modifiers.history = {};
  HYLE_RETURN_IF_ERROR(logits_inputs_metadata_status(base, inputs, vocabulary));
  for (const auto &row : conditional) HYLE_RETURN_IF_ERROR(logits_inputs_metadata_status(per_row, row, vocabulary));
  return hyle::Status::Ok();
}
void validate_prefix_logits_input_leases(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional) {
  if (conditional.empty()) { validate_logits_input_leases(configuration, inputs); return; }
  auto base = configuration, per_row = configuration;
  base.modifiers.bias = false; base.modifiers.allowed = false; per_row.modifiers.history = {};
  validate_logits_input_leases(base, inputs);
  for (const auto &row : conditional) validate_logits_input_leases(per_row, row);
}
void validate_prefix_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional, std::size_t rows, std::int64_t vocabulary) {
  check(prefix_logits_inputs_metadata_status(configuration, inputs, conditional, rows, vocabulary));
  validate_prefix_logits_input_leases(configuration, inputs, conditional);
}
// A registered request has its own complete key before any bucket or native
// episode exists. Do not manufacture EvaluationFailure.participant from it.
void check_request_logits_metadata(hyle::Status status, inference::RequestKey key) {
  if (status.IsInvalidArgument() && !status.has_native_error() && key.generation &&
      key.slot != (std::numeric_limits<std::size_t>::max)())
    throw inference::RequestExecutionFailure::from_preflight(std::move(status), key);
  check(std::move(status)); // Unregistered/raw consumers retain their original error path.
}
void validate_request_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::int64_t vocabulary, inference::RequestKey key) {
  check_request_logits_metadata(logits_inputs_metadata_status(configuration, inputs, vocabulary), key);
  validate_logits_input_leases(configuration, inputs);
}
sampling::LogitsInputsFor<V> prepare_logits_inputs(Recipe &r, sampling::LogitsPolicy policy) {
  sampling::LogitsInputsFor<V> result;
  if (policy.bias) result.bias = r.input(dense(Type::kFloat32, {1, r.geometry.vocabulary}));
  if (policy.allowed) result.allowed = r.input(dense(Type::kBool, {1, r.geometry.vocabulary}));
  if (!policy.history.empty()) result.counts = r.input(dense(Type::kInt32, {1, r.geometry.vocabulary}));
  return result;
}

// Ledger token storage is reserved with the request. Its sources are separate
// episode/result values; replacement assignment would discard that capacity.
void write_generation_tokens(hyle::container::SmallVector<std::int32_t, 2> &tokens,
    std::span<const std::int32_t> values) {
  if (!tokens.try_resize(values.size())) throw std::bad_alloc{};
  std::copy(values.begin(), values.end(), tokens.begin());
}
void write_generation_tokens(hyle::container::SmallVector<std::int32_t, 2> &tokens,
    std::initializer_list<std::int32_t> values) {
  write_generation_tokens(tokens, std::span<const std::int32_t>(values.begin(), values.size()));
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
  RowPacking packing = RowPacking::kRectangular;
  FamilyKey(std::size_t count, GraphRole selected = GraphRole::kText,
            std::size_t rows = 0, TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0, RowPacking packing = RowPacking::kRectangular) noexcept
      : requests(count), verification_rows(selected == GraphRole::kVerify ? rows : 0), role(selected), completion(mode), sampling_index(sampling), packing(packing) {}
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
  RowPacking packing = RowPacking::kRectangular;
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
  V sequence_offsets_port, history_select_port, prefix_ids_port;
  bool batched_text_sampling = false; // Cold [R,4] identity-port contract.
  std::vector<V> logits_bias_ports, logits_allowed_ports, logits_counts_ports; // Cold per-member inputs only.
  std::vector<g::NodeId> attention_nodes, append_nodes;
  std::vector<V> append_cache_inputs; // Cold same-cohort reader ports; invalid means write-only.
  std::vector<g::slots::SourceBinding> state_sources;
  V draft_key_port, draft_value_port, draft_metadata_port, draft_position_port;
  V carry_input_port, prior_input_port, catchup_token_port, draft_id_port;
  V successor_token_port, successor_catchup_port;
  V draft_hidden_select_port, draft_token_select_port, draft_last_ids_port;
  StatePorts hidden_state;
  g::NodeId draft_attention_node;
  std::vector<V> state_resource_ports;
  std::array<V, 9> probe_ports;
  std::vector<Ports> layers;
};
struct Family final : FamilyPorts {
  const PreparedSampling *sampling = nullptr;
  std::size_t requests = 1;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  bool admitted_exact = false;
  RowPacking packing = RowPacking::kRectangular;
  std::vector<Tensor> literals; // Backing survives every child and its slots.
  std::vector<g::TensorSpec> specs;
  c::BatchSchema attention_schema;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  g::DimensionId last_row;
  std::vector<std::size_t> chunks;
};
struct FamilyChild final {
  Family &family;
  const inference::PreparedProgram &program;
};
// Physical IO/control carriers are independent of candidate depth. A reader
// may still retain the preceding carrier after its producer publishes.
constexpr std::size_t execution_slot_count = 3;
enum class StateBankKind { kHistory, kRecurrent, kHidden };
struct StateBank final {
  std::size_t layer = 0;
  StateBankKind kind = StateBankKind::kHistory;
  std::size_t source_index = SIZE_MAX; // Only a real paired Gather has a Source.
};
// Mutable input, result and layout carriers belong to one actual Invocation.
// Sharing a finite child/Queue never grants another slot access to these bytes.
struct PhaseIo final {
  c::BatchLayoutVersion attention_layout;
  // Original per-Invocation planning storage. Nodes are fixed at cold
  // preparation; layout aliases exist only during batch::prepare below.
  std::vector<g::batch::NodeLayout> batch_layouts;
  Tensor tokens, positions, last_ids, logits, final_hidden, history_select, prefix_ids;
  std::vector<std::int32_t> history_select_values, last_id_values, prefix_id_values;
  Tensor sampling_identities, draft_positions, catchup_token;
  Tensor draft_hidden_select, draft_token_select, draft_last_ids;
  std::vector<std::int32_t> draft_hidden_indices, draft_token_indices;
  inference::ProgramIo::Readback next_id, draft_id, sampling_valid, sampling_accept;
  inference::ProgramIo::Readback successor_token, successor_catchup;
  std::vector<std::uint8_t> sampling_valid_values, sampling_accept_values;
  std::vector<float> sampling_scratch;
  std::vector<std::int32_t> draft_ids, prefix_candidates;
  std::array<Tensor, 9> probe_outputs;
};
struct Phase final : FamilyPorts {
  std::size_t query_rows = 0;
  c::BatchSchema attention_schema;
  // The selected child is shared immutable model data, not copied per session.
  const inference::PreparedProgram *child = nullptr;
  const Family *family = nullptr;
  bool imported_sources = false;
  std::vector<StateBank> state_banks;
  using ExecutionSlots = inference::ProgramSlots<ColdResources, PhaseIo, execution_slot_count>;
  ExecutionSlots slots;
  const g::Graph &state_graph() const noexcept {
    return child->resource_graph();
  }
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
  std::vector<inference::PackedRowInput> row_inputs;
  std::vector<std::int32_t> query_indptr, last_rows;
  std::vector<std::uint32_t> sampling_identities;
  // Shared preparation reuses cold-bounded integers and temporary footprints.
  // Its source pins never survive a call; Graph retains its original leases.
  inference::PagedAppendScratch paged_appends;
  void prepare(const Phase &phase) {
    const auto &io = phase.slots[0].data; // All slots share the same cold descriptions.
    const auto reserve = [](auto &values, const Tensor &tensor) {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      values.reserve(take(tensor.description.required_bytes()) / sizeof(Element));
    };
    row_inputs.reserve(phase.family->requests);
    query_indptr.reserve(phase.family->requests + 1);
    last_rows.reserve(phase.family->requests);
    if (phase.family->requests > 1 || packs_target(phase.family->packing)) reserve(packed_ids, io.tokens);
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
  LogitsInputs logits{}; // Original admitted episode snapshot, never history.
  inference::ConditionalLogitsInputs conditional_logits{};
  std::unique_ptr<cache::State> next{};
  hyle::container::SmallVector<c::PagedArenaCommit, 32> target_updates{};
  std::array<c::PagedArenaCommit, 2> draft_updates{};
  // Armed only after members and ticket storage stop moving, then consumed
  // before publication recycles their original candidate or previous state.
  inference::PagedSequencePublication target_publication, draft_publication;
  hyle::container::SmallVector<c::PagedPageAppender, 32> next_tails{};
  std::array<c::PagedPageAppender, 2> next_draft_tails{};
  c::BatchLayoutVersion layout{};
  c::BatchLayoutVersion draft_layout{};
  std::size_t input_tokens = 0, input_row_begin = 0, accepted_tokens = 0;
  std::uint32_t verified_last_page = 0;
  std::int32_t first_input = -1, draft_token = -1;
  std::vector<std::int32_t> verification_inputs{};
  std::vector<c::ReadLease> carry_prefix{};
  std::size_t discarded_draft_rows = 0;
  bool sampled_accept = false;
  GenerationCommitRecord *commit_record = nullptr;
  GenerationLedger *commit_ledger = nullptr;
  // Borrowed only by finish, after the original ledger slot is checked. No
  // native work or successor reads this host publication ticket.
  std::optional<inference::PreparedRecordCommit<GenerationCommitRecord>> commit_publication;

  inference::PagedSequenceEdit target_edit() noexcept {
    return {base->paged, next->paged, {target_updates.data(), target_updates.size()}};
  }
  inference::PagedSequenceEdit draft_edit() noexcept {
    return {base->draft->paged, next->draft->paged, draft_updates};
  }
  void preflight_frames() {
    target_publication = {}; draft_publication = {};
    target_publication = take(target_edit().prepare_publication());
    if (draft_updates[0].valid() || draft_updates[1].valid())
      draft_publication = take(draft_edit().prepare_publication());
  }
  void publish_frames(cache::State *previous, cache::State &candidate) noexcept {
    HYLE_FATAL_ASSERT(target_publication.candidate() == &candidate.paged);
    target_publication.publish(previous ? &previous->paged : nullptr);
    if (draft_publication.valid()) {
      HYLE_FATAL_ASSERT(candidate.draft && draft_publication.candidate() == &candidate.draft->paged &&
          (!previous || previous->draft));
      draft_publication.publish(previous ? &previous->draft->paged : nullptr);
    }
  }
};
// Typed views of one original Graph submission. They contain no executor,
// model-state snapshot or observer independent of the original proxy. A
// successor may keep these views after its producer's host episode retires.
struct StepSources final : inference::SubmissionViews {
  struct Boundary final {
    std::weak_ptr<Session> session;
    const cache::State *candidate = nullptr;
    std::uint64_t epoch = 0;
    std::size_t length = 0, draft_frontier = 0;
  };
  std::vector<Boundary> boundaries;
  std::vector<c::ReadLease> published_states;
  // K1 keeps the producer's full two-prefix publication rights in states.
  // These independent original final-prefix views serve the next Gather.
  std::vector<g::detail::StoreSource> successor_states;
  bool mtp_full_candidate = false;
  void release_publication_views() noexcept {
    if (!mtp_full_candidate) return;
    // Model-owned full-prefix views have no consumer binding. Once A has
    // published, B needs only its independent last-prefix views. Keeping the
    // obsolete prefix tickets would exceed the original N + K*D*R pool bound.
    // Preserve hidden's two rows and the identities of every successor view.
    for (std::size_t bank = 0; bank < successor_states.size(); ++bank) states[bank] = {};
  }
};
using IncomingStepSources = inference::IncomingExecutionSources<StepSources>;
struct StepState final {
  // Keep the model's original participant pins alive until after the common
  // resource episode is destroyed. Session pending still borrows this Step.
  hyle::container::SmallVector<StepMember, 1> members;
  inference::ExecutionEpisode<ColdResources, StepSources, 34> execution;
  c::BatchLayoutVersion draft_layout;
  Phase *phase = nullptr;
  Clock::time_point step_start{}, start{};
  double submit_seconds = 0, cold_batch_seconds = 0, shape_admission_seconds = 0;
  std::size_t input_tokens = 0, active_input_rows = 0, capture_recorded = 0, capture_reused = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool prepared = false, submitted = false, cancelled = false;
  std::size_t state_slot = 0;
  bool discard_stopped_members = false;
  bool withdrawn(const StepMember &member) const noexcept;

  hyle::Result<bool> try_abort() noexcept {
    return execution.recover([&]() noexcept {
      for (auto &member : members) {
        member.target_publication = {}; member.draft_publication = {};
        member.owner_state->recycle_candidate(std::move(member.next)); member.base = nullptr;
      }
      phase = nullptr;
    });
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
  // After multi-request submission, stop withdraws only this member's
  // publication while a survivor remains, including MTP. Single/all-member and
  // pre-submit stops abort the whole episode. Cancellation is not retirement.
  hyle::Status stop() noexcept;
  inference::RetirementObservation try_retire() noexcept;
  inference::RetirementObservation drain() noexcept;
  ~Session();
};
bool StepState::withdrawn(const StepMember &member) const noexcept {
  return discard_stopped_members && member.session->stopping;
}
struct EvaluationRequest final {
  std::shared_ptr<Session> session;
  std::span<const std::int32_t> ids;
  const inference::RequestStopPolicy *stop_policy = nullptr;
  LogitsInputs logits{};
  inference::ConditionalLogitsInputs conditional_logits{};
  inference::RequestKey request_key{}; // Diagnostic identity only; not admission authority.
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
  // Packed Verify exposes compact input rows; other outputs stay request-major.
  std::size_t packed_input_begin = 0, packed_input_rows = 0;
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
  void reset() noexcept {
    auto storage = std::move(emitted);
    *this = MtpRoundResult{};
    emitted = std::move(storage); emitted.clear();
  }
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
  std::size_t rows = 0, sampling_index = 0;
  bool head = false;
  g::ReplayMode replay_mode = g::ReplayMode::kCommands;
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  V tokens, positions, key, value, metadata, hidden, id, identity, log_probabilities, valid, probability;
  V bias, allowed, counts, next_counts;
  std::vector<V> hidden_inputs;
  std::vector<inference::ProgramIo::Input> hidden_bindings;
  inference::ProgramIo::Input bias_input, allowed_input, counts_input;
  StatePorts hidden_state;
  inference::SlotTransactions hidden_transactions;
  StepState guard; // The same original episode, cold-prepared for recursive reuse.
  mtp::AppendNodes append;
  g::NodeId attention;
  Tensor token_values, position_values, identity_values;
  inference::ProgramIo io;
  inference::ProgramIo::Readback id_value, valid_value, probability_value;
  c::LeasePool probability_pool, next_counts_pool;
  inference::ProgramIo::Output probabilities_output, next_counts_output;
};
struct DraftWork final {
  cache::DraftState state;
  std::array<c::PagedPageAppender, 2> tails;
};
struct DraftProposal final {
  std::int32_t id = -1;
  float probability = 0;
  c::ReadLease hidden, log_probabilities, next_counts;
};
// Call-local linear proposal resolution. Row zero was resolved before request
// admission; later rows are retained once and frozen for the original verifier.
struct ConditionalLogitsProposal final {
  const inference::LogitsInputProvider &provider;
  std::span<const std::int32_t> committed_outputs;
  std::vector<LogitsInputs> rows;
};
struct PrefixSamplingKey final {
  std::size_t proposals, sampling_index;
  bool conditional = false;
  friend auto operator<=>(const PrefixSamplingKey &, const PrefixSamplingKey &) = default;
};
struct PrefixSamplingProgram final {
  std::size_t proposals = 0, sampling_index = 0;
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  V logits, identities, ids, bias, allowed, counts, continuation, packet;
  struct ConditionalInput final {
    V bias, allowed;
    inference::ProgramIo::Input bias_input, allowed_input;
  };
  std::vector<ConditionalInput> conditional_inputs;
  std::vector<V> q, accepted, final_ids, valid;
  Tensor identity_values, id_values, continuation_values;
  std::vector<float> continuation_host;
  std::vector<std::int32_t> packet_host;
  inference::ProgramIo io;
  inference::ProgramIo::Input logits_input, bias_input, allowed_input, counts_input;
  std::vector<inference::ProgramIo::Input> q_inputs;
  inference::ProgramIo::Readback packet_value;
};
struct PrefixSamplingResult final {
  std::vector<std::int32_t> final_ids;
  inference::mtp::AcceptedPrefix prefix;
};

struct SamplingFrontier final {
  std::vector<Tensor> literals;
  g::Graph graph;
  inference::PreparedProgramFamily programs;
  inference::ProgramCarrier<ColdResources> carrier;
  g::DimensionId last_row;
  V hidden_port, identity_port, id_port, valid_port, bias_port, allowed_port, counts_port;
  Tensor identity;
  inference::ProgramIo io;
  inference::ProgramIo::Input hidden_input, bias_input, allowed_input, counts_input;
  inference::ProgramIo::Readback id, valid;
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
  hyle::inference::PagedPageInventory page_inventory, draft_inventory;
  std::vector<cache::StatePools> state_pools;
  c::LeasePool hidden_pool, history_counts_pool;
  c::LeaseSlotLayout hidden_layout;
  std::vector<std::unique_ptr<PreparedSampling>> sampling_configurations;
  const bool retain_sampling_frontier;
  std::map<FamilyKey, std::unique_ptr<Family>> families, probe_families;
  std::map<ExecutionShape, std::unique_ptr<Phase>> phases, probe_phases;
  std::map<std::tuple<std::size_t, bool, std::size_t>, std::unique_ptr<DraftProgram>> draft_programs;
  std::map<PrefixSamplingKey, std::unique_ptr<PrefixSamplingProgram>> prefix_sampling_programs;

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
        if (phase->slots.pending()) return true;
      }
    for (const auto &[shape, program] : draft_programs) {
      (void)shape; if (program->carrier.pending()) return true;
    }
    for (const auto &[width, program] : prefix_sampling_programs) {
      (void)width; if (program->carrier.pending()) return true;
    }
    return std::ranges::any_of(sampling_configurations, [](const auto &entry) {
      return entry->frontier && entry->frontier->carrier.pending();
    });
  }
  void release_invocation_roots() noexcept {
    // The accepted attachment is the only temporary scope->Graph->scope
    // cycle. Drop these application roots at handoff; the original native
    // Completion still retains its Graph until actual retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape; phase->slots.release_invocation_roots();
      }
    for (auto &[shape, program] : draft_programs) {
      (void)shape;
      program->guard.execution.completion = {}; program->guard.execution.run = nullptr; program->guard.execution.carrier = nullptr;
      program->carrier.release_invocation_root();
    }
    for (auto &[width, program] : prefix_sampling_programs) { (void)width; program->carrier.release_invocation_root(); }
    for (auto &entry : sampling_configurations) if (entry->frontier)
      entry->frontier->carrier.release_invocation_root();
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
  const SamplingConfiguration &default_sampler; // Aggregate diagnostics describe this immutable default.
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
  const cache::Plan cache_plan;
  // The original plain storage/layout root also serves draft attention.
  // Calibrated target storage views share this metadata domain, not its dtype.
  c::BatchSchema attention_schema;
  std::vector<c::BatchSchema> attention_schemas;
  const c::BatchSchema &target_attention_schema(std::size_t layer) const {
    if (layer >= target_kv_layers) throw std::runtime_error("invalid target attention cache slot");
    return attention_schemas.empty() ? attention_schema : attention_schemas.at(layer);
  }
  cache::VersionCount &snapshot_versions;
  std::size_t &carrier_borrowed;
  static constexpr std::size_t slot_index = 0;
  hyle::inference::PagedPageInventory &page_inventory, &draft_inventory;
  std::vector<cache::StatePools> &state_pools;
  c::LeasePool &hidden_pool;
  c::LeaseSlotLayout &hidden_layout;
  // Family/child owners are shared by sessions; no session addresses enter compile.
  std::map<FamilyKey, std::unique_ptr<Family>> &families, &probe_families;
  inference::ColdShapePlan<ExecutionShape, FamilyKey, execution_slot_count> shape_plan;
  // One carrier per actually prepared finite shape. Serial admission borrows
  // its one slot and shared IO through native retirement and all result reads.
  std::map<ExecutionShape, std::unique_ptr<Phase>> &phases, &probe_phases;
  std::map<std::tuple<std::size_t, bool, std::size_t>, std::unique_ptr<DraftProgram>> &draft_programs;
  std::map<PrefixSamplingKey, std::unique_ptr<PrefixSamplingProgram>> &prefix_sampling_programs;
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
  std::size_t packed_prefill_calls = 0, packed_prefill_active_rows = 0, packed_prefill_physical_rows = 0;
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
         state_pools(resources_->state_pools), hidden_pool(resources_->hidden_pool), hidden_layout(resources_->hidden_layout),
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
    if ((calibrated && (!target_kv_layers || resources_->attention_storage.size() != target_kv_layers)) ||
        (!calibrated && !resources_->attention_storage.empty()))
      throw std::runtime_error("prepared E4 cache storage requires its complete original target calibration pairs");
    const auto plain = take(c::CacheStorage::plain(precision.activation_type()));
    const auto plain_rows = cache_storage_row_bytes(geometry, {plain, plain});
    const auto target_rows = calibrated ? cache_storage_row_bytes(geometry, resources_->attention_storage.front()) : plain_rows;
    const cache::KvRowBytes actual_target{target_rows[0], target_rows[1]}, actual_draft{plain_rows[0], plain_rows[1]};
    if (source.cache_row_bytes.target != actual_target ||
        ((source.cache_row_bytes.draft.key || source.cache_row_bytes.draft.value || cache_plan.draft_cache) &&
            source.cache_row_bytes.draft != actual_draft) ||
        std::ranges::any_of(resources_->attention_storage, [&](const auto &pair) {
          return cache_storage_row_bytes(geometry, pair) != target_rows;
        }))
      throw std::runtime_error("prepared target/draft row extents differ from their actual typed source carriers");
    const auto bank_rows = cache::multiply(cache_plan.max_pages, cache::Plan::page_rows);
    if (cache_plan.row_bytes != source.cache_row_bytes ||
        cache_plan.key_bank_bytes != cache::add(cache::multiply(bank_rows, source.cache_row_bytes.target.key), 1) ||
        cache_plan.value_bank_bytes != cache::add(cache::multiply(bank_rows, source.cache_row_bytes.target.value), 1) ||
        cache_plan.draft_key_bank_bytes != (cache_plan.draft_cache
            ? cache::add(cache::multiply(bank_rows, source.cache_row_bytes.draft.key), 1) : 0) ||
        cache_plan.draft_value_bank_bytes != (cache_plan.draft_cache
            ? cache::add(cache::multiply(bank_rows, source.cache_row_bytes.draft.value), 1) : 0))
      throw std::runtime_error("cache backing plan differs from independent target/draft physical storage");
    if (cache_plan.hidden_row_bytes != geometry.hidden_bytes() || cache_plan.attention_layers != target_kv_layers ||
        cache_plan.recurrent_layers != geometry.recurrent_layers())
      throw std::runtime_error("cache layout differs from source model geometry");
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
    if (std::ranges::any_of(resources_->sampling_configurations,
        [](const auto &configuration) { return !configuration->configuration.modifiers.history.empty(); })) {
      const auto blocks = cache_plan.active_sequences;
      if (!blocks || blocks >= UINT32_MAX) throw std::overflow_error("history snapshot block capacity exceeds lease indexing");
      const auto bytes = cache::multiply(static_cast<std::size_t>(vocabulary), sizeof(std::int32_t));
      resources_->history_counts_pool = take(c::LeasePool::create(execution.device(),
          cache::add(cache::multiply(cache::add(bytes, cache::state_pool_padding(bytes)), blocks), 1),
          {.max_blocks = static_cast<std::uint32_t>(blocks), .resource_budget = execution_budget}));
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
        (void)phase->slots.retire_all(execution);
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
      if (active_step) inference::record_retirement_status(session->retirement_status, active_step->execution.terminal);
    }
    resources_->release_invocation_roots();
    // Teardown never promotes an unpublished successor back into escaped
    // Sessions whose Model link has already been withdrawn above.
    if (successor_step) successor_step->execution.candidate_admission = nullptr;
    if (active_step) active_step->execution.candidate_admission = nullptr;
    successor_step.reset(); active_step.reset();
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
    if (cache_plan.draft_cache) {
      const auto plain_dtype = precision.is_f16() ? "F16" : "BF16";
      result += ",\"target_kv_cache_dtype\":\"" + std::string(resources_->cache_policy ==
          prepared::CachePolicy::kFloat8OriginalCalibration ? "E4M3FN" : plain_dtype) +
          "\",\"draft_kv_cache_dtype\":\"" + plain_dtype + "\"";
    }
    return result;
  }
  Tensor caller_tensor(mem::MemoryResourceRef resource, c::TensorDesc description, bool caller = true) const {
    return Tensor(resource, description, caller, allocation_budget);
  }

  c::ResourceBudgetUsage cache_usage() const {
    const auto usage = take(cache_budget.usage(execution.device()));
    const auto metadata = usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)];
    const auto payload = usage.live_bytes - metadata;
    if (payload != known_cache_backing_bytes)
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
    if (sampling_configuration_count() > 1 || !default_sampler.modifiers.history.empty()) out << ",\"sampling_scope\":\"default_configuration\"";
    out << ",\"sampling\":\"" << (default_sampler.enabled ? "random" : "greedy") << '"'
        << ",\"sampling_policy_order\":\"" << (default_sampler.modifiers.history.empty()
            ? "temperature_top_k_top_p" : "bias_repetition_frequency_presence_mask_temperature_top_k_top_p") << '\"'
        << ",\"history_repetition\":" << default_sampler.modifiers.history.repetition
        << ",\"history_presence\":" << default_sampler.modifiers.history.presence
        << ",\"history_frequency\":" << default_sampler.modifiers.history.frequency
        << ",\"history_counts_snapshot_blocks\":" << (resources_->history_counts_pool.valid() ? cache_plan.active_sequences : 0)
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
        << ",\"packed_prefill_calls\":" << packed_prefill_calls
        << ",\"packed_prefill_active_rows\":" << packed_prefill_active_rows
        << ",\"packed_prefill_physical_rows\":" << packed_prefill_physical_rows
        << ",\"cold_exact_admissions\":" << shape_plan.admissions()
        << ",\"cold_exact_evictions\":" << shape_plan.evictions()
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

  // C0 includes only authoritative committed IDs and this first conditional's
  // actual input. It is never an alternative history, Session or RNG owner.
  // Raw EvaluationRequest callers can instead supply their own exact C0 lease.
  LogitsInputs history_logits_inputs(GenerationLedger &ledger, std::span<const std::int32_t> first_condition_ids,
      LogitsInputs inputs = {}, std::size_t sampling_index = 0) {
    const auto &prepared = sampling_configuration(sampling_index);
    ledger.validate_sampling(prepared, resources_);
    if (prepared.configuration.modifiers.history.empty()) return inputs;
    if (inputs.counts.lease.valid() || inputs.counts.description.valid())
      throw std::invalid_argument("derived history C0 cannot replace an externally supplied counts input");
    auto &cache = ledger.history_counts(static_cast<std::size_t>(vocabulary), execution_budget,
        take(execution_budget.metadata_allocator()));
    const auto conditional = cache.condition(first_condition_ids);
    const auto bytes = std::as_bytes(cache.values());
    auto writer = take(resources_->history_counts_pool.allocate(bytes.size()));
    check(execution.write(take(writer.view()), bytes));
    inputs.counts = {dense(Type::kInt32, {1, vocabulary}), take(writer.publish())};
    return inputs;
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
    hyle::inference::PagedPageInventory prepared_inventory, prepared_draft;
    check(prepared_inventory.prepare(cache_plan.max_pages, target_kv_layers * 2));
    for (std::size_t i = 0; i < prepared_inventory.banks.size() / 2; ++i) {
      const auto &schema = prepared_schemas.empty() ? prepared_schema : prepared_schemas[i];
      prepared_inventory.banks[2 * i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_inventory.banks[2 * i + 1] = take(c::PagedArena::create(schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    if (cache_plan.draft_cache) {
      check(prepared_draft.prepare(cache_plan.max_pages, 2));
      prepared_draft.banks[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.draft_key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_draft.banks[1] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.draft_value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    const auto prepared_bytes = cache::add(cache::multiply(
        cache::add(cache_plan.key_bank_bytes, cache_plan.value_bank_bytes), target_kv_layers),
        cache::add(cache_plan.draft_key_bank_bytes, cache_plan.draft_value_bank_bytes));
    const auto layout_capacity = prepared_schema.capacity();
    input_scratch.paged_appends.prepare(layout_capacity.max_requests, layout_capacity.max_page_refs);
    // No PageRef exists until this complete cold source tuple is published.
    // Leave the schema sentinel invalid on failure so released quota can retry.
    static_assert(std::is_nothrow_move_assignable_v<hyle::inference::PagedPageInventory>);
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
    // Every canonical state row uses its original versioned pool. Logical
    // Store outputs do not depend on a provider's ordinary output ownership.
    // Publish together so a failed allocation releases its temporary backing
    // and leaves this Model ready for a later admission attempt.
    std::vector<cache::StatePools> prepared_pools(geometry.layers.size());
    c::LeasePool prepared_hidden;
    c::LeaseSlotLayout prepared_hidden_layout;
    auto prepared_bytes = known_cache_backing_bytes;
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
      const auto slots = cache_plan.hidden_slots;
      const auto bytes = cache::state_pool_backing_bytes(geometry.hidden_bytes(), slots);
      prepared_hidden = take(c::LeasePool::create(execution.device(), bytes,
          {.max_blocks = slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      prepared_hidden_layout = take(c::LeaseSlotLayout::create(prepared_hidden,
          dense(precision.activation_type(), {1, hidden}), slots,
          cache::add(geometry.hidden_bytes(), cache::state_pool_padding(geometry.hidden_bytes()))));
      prepared_bytes = cache::add(prepared_bytes, bytes);
    }
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes - usage.live_by_category[
            static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] != prepared_bytes)
      throw std::runtime_error("shared cache backing inventory differs from the actual reserved bytes");
    state_pools = std::move(prepared_pools); hidden_pool = std::move(prepared_hidden);
    hidden_layout = std::move(prepared_hidden_layout);
    known_cache_backing_bytes = prepared_bytes;
    state_storage_prepared = true;
    for (auto *children : {&phases, &probe_phases})
      for (auto &[key, phase] : *children) prepare_transaction_layouts(*phase);
    for (auto &[key, program] : draft_programs) if (program->head)
      check(program->hidden_transactions.set_layout(0, hidden_layout));
  }

  // Graph/counts may be prepared before admission allocates canonical storage.
  // The pool's immutable layout is attached only at that original cold boundary.
  void prepare_transaction_layouts(Phase &phase) {
    for (std::size_t bank = 0; bank < phase.state_banks.size(); ++bank) {
      const auto &facts = phase.state_banks[bank];
      const auto &layout = facts.kind == StateBankKind::kHidden ? hidden_layout
          : facts.kind == StateBankKind::kRecurrent ? state_pools[facts.layer].recurrent_layout
              : state_pools[facts.layer].history_layout;
      for (std::size_t slot = 0; slot < phase.slots.size(); ++slot)
        check(phase.slots[slot].transactions.set_layout(bank, layout));
    }
  }

  bool prepare_family(std::span<const std::size_t> planned_chunks, bool diagnostic = false,
                      std::size_t requests = 1, GraphRole role = GraphRole::kText,
                      bool admitted_exact = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                      RowPacking packing = RowPacking::kRectangular) {
    const bool packed_prefill = packs_prefill(packing);
    const bool compact_draft = packs_draft(packing);
    const bool packed_verify = packs_verify(packing), packed_target = packs_target(packing);
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (!supports_completion(role, completion))
      throw std::runtime_error("state-only completion requires a Text or teacher-forced MTP input family");
    if (packed_prefill && (diagnostic || admitted_exact ||
        (role != GraphRole::kText && !is_mtp_prefill(role))))
      throw std::runtime_error("packed prefill requires an original cold Text or MTP prefill family");
    if (compact_draft && (diagnostic || admitted_exact || requests < 2 ||
        role != GraphRole::kMtpCatchupRound || completion != TextCompletion::kEmitNext))
      throw std::runtime_error("compact draft requires a multi-member K1 round capacity");
    if (packed_verify && (diagnostic || admitted_exact || role != GraphRole::kVerify ||
        completion != TextCompletion::kEmitNext || planned_chunks.size() != 1 || !target_kv_layers))
      throw std::runtime_error("packed verification requires one original cold prefix capacity and attention layout");
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
        const std::array chunk{width};
        prepare_family(chunk, diagnostic, requests, role, admitted_exact, completion, sampling_index, packing);
      }
      return true;
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
    const auto family_key = admitted_exact ? FamilyKey::exact(requests, role, planned_chunks.front(), completion, sampling_index) : FamilyKey(requests, role, prefix_rows, completion, sampling_index, packing);
    auto &table = diagnostic ? probe_families : families;
    if (table.contains(family_key)) throw std::runtime_error("text finite family was already prepared");
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.layers.resize(geometry.layers.size()); definition.attention_nodes.resize(target_kv_layers);
    definition.append_nodes.resize(draft_bank + 2);
    definition.append_cache_inputs.resize(draft_bank + 2);
    definition.sampling = &prepared_sampling;
    definition.requests = requests; definition.role = role; definition.prefix_rows = prefix_rows; definition.completion = completion;
    definition.admitted_exact = admitted_exact; definition.packing = packing;
    definition.state_resource_ports.reserve(ports_per_request * requests);
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
    // In a packed family [batch,t] partitions the physical carrier only.
    // Original cu_seqlens and explicit positions describe logical members.
    const auto batch = f(static_cast<std::int64_t>(requests));
    g::slots::Schema hidden_schema;
    if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) {
      hidden_schema = take(g::slots::Schema::create(dense(precision.activation_type(), {1, hidden}),
          cache_plan.hidden_slots,
          cache::add(geometry.hidden_bytes(), cache::state_pool_padding(geometry.hidden_bytes()))));
      definition.hidden_state.destination_ids = r.input(dense(Type::kInt32,
          {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
    }
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
    if ((requests != 1 || packed_prefill) && !verifies_prefix(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    if (packed_target) {
      definition.sequence_offsets_port = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(requests + 1)}));
      definition.history_select_port = r.input(dense(Type::kInt32,
          {static_cast<std::int64_t>(cache::multiply(cache::multiply(requests, prefix_rows), geometry.conv_kernel - 1))}));
    }
    if (packed_verify) definition.prefix_ids_port = r.input(dense(Type::kInt32,
        {static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))}));
    definition.metadata_port = r.input(take(definition.attention_schema.metadata_description()));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    V target_tokens = role == GraphRole::kText
        ? r.reshape(definition.token_port, {packed}) : definition.token_port;
    std::vector<sampling::Distribution> draft_distributions;
    std::vector<V> proposal_ids, draft_valid;
    std::vector<sampling::LogitsInputsFor<V>> logits_inputs(requests);
    if (completion == TextCompletion::kEmitNext && role != GraphRole::kVerify && !selected_sampler.modifiers.empty())
      for (auto &inputs : logits_inputs) {
        inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
        if (inputs.bias) { definition.logits_bias_ports.push_back(*inputs.bias); definition.state_resource_ports.push_back(*inputs.bias); }
        if (inputs.allowed) { definition.logits_allowed_ports.push_back(*inputs.allowed); definition.state_resource_ports.push_back(*inputs.allowed); }
        if (inputs.counts) { definition.logits_counts_ports.push_back(*inputs.counts); definition.state_resource_ports.push_back(*inputs.counts); }
      }
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
      auto &port = definition.hidden_state;
      const auto source_rows = f(static_cast<std::int64_t>(cache::multiply(requests, draft_lag(role) + 1)));
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(r.dense_spec(Type::kInt32, {source_rows}));
      const auto gathered = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {source_rows, f(hidden)}), &port.gather);
      // One fixed pool and private IDs; the original ready leases still prove
      // exactly these rows. Batched/packed order is [all carries, all priors].
      const bool chronological = requests == 1 && role == GraphRole::kMtpCatchupRound &&
          packing == RowPacking::kRectangular;
      // A K1 producer stores [prior, carry]. The lag1 R1 reader uses that same
      // real ID order, in ready and imported modes; only the graph interprets
      // the two chronological rows. No source ticket/ID is manufactured.
      definition.carry_input_port = r.slice(gathered, 0, chronological ? batch : f(0), batch);
      const auto draft_rows = is_mtp_prefill(role)
          ? (draft_lag(role) ? D(take(r.builder.dimension_sum(t, f(1)))) : t)
          : f(static_cast<std::int64_t>(draft_lag(role) + 1));
      const auto packed_draft_rows = !draft_lag(role) && is_mtp_prefill(role) ? packed : requests == 1 ? draft_rows :
          draft_rows.is_fixed() ? f(static_cast<std::int64_t>(cache::multiply(requests, static_cast<std::size_t>(draft_rows.fixed_value())))) :
          D(take(r.builder.dimension_product(draft_rows, batch)));
      draft_sequence_rows = draft_rows; draft_packed_rows = packed_draft_rows;
      if (packed_prefill || compact_draft) {
        definition.draft_hidden_select_port = r.input(r.dense_spec(Type::kInt32, {packed_draft_rows}));
        if (draft_lag(role)) definition.draft_token_select_port = r.input(r.dense_spec(Type::kInt32, {packed_draft_rows}));
      }
      if (compact_draft) definition.draft_last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      definition.draft_position_port = r.input(requests == 1 ? r.dense_spec(Type::kInt32, {draft_rows})
          : r.dense_spec(Type::kInt32, {batch, draft_rows}));
      if (draft_lag(role)) {
        definition.prior_input_port = r.slice(gathered, 0, chronological ? f(0) : batch, batch);
        definition.catchup_token_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      }
      if (drafts_token(role) || !is_mtp_prefill(role)) {
        V ids = definition.token_port, hidden_rows = definition.carry_input_port;
        if (compact_draft) {
          const auto token_source = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids},
              r.dense_spec(Type::kInt32, {packed_draft_rows}));
          const auto hidden_source = r.one(c::Concat{.axis = 0}, {hidden_rows, definition.prior_input_port},
              r.dense_spec(precision.activation_type(), {packed_draft_rows, f(hidden)}));
          ids = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {token_source, definition.draft_token_select_port}, r.dense_spec(Type::kInt32, {packed_draft_rows}));
          hidden_rows = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {hidden_source, definition.draft_hidden_select_port}, r.dense_spec(precision.activation_type(), {packed_draft_rows, f(hidden)}));
        } else if (draft_lag(role)) {
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
          definition.append_cache_inputs[draft_bank] = definition.draft_key_port;
          definition.append_cache_inputs[draft_bank + 1] = definition.draft_value_port;
          const auto head = mtp::full_head(r, projection, kv, definition.draft_position_port, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port, definition.draft_metadata_port},
              f(static_cast<std::int64_t>(draft_lag(role))), row_geometry, definition.draft_last_ids_port);
          append = head.append_nodes; definition.draft_attention_node = head.attention_node;
          if (!selected_sampler.enabled && selected_sampler.modifiers.empty())
            definition.draft_id_port = r.op(c::Operation(c::TopK{.count = 1}), {head.logits},
                {r.dense_spec(Type::kFloat32, {batch, f(1)}), r.dense_spec(Type::kInt32, {batch, f(1)})})[1];
          else {
            for (std::size_t request = 0; request < requests; ++request) {
              const auto processed = sampling::process_logits(r, sampling_row(head.logits, request), logits_inputs[request], selected_sampler.modifiers.history);
              if (selected_sampler.enabled) {
                draft_distributions.push_back(sampling::processed_distribution(r, processed, selected_sampler.policy));
                proposal_ids.push_back(r.one(c::sampling_ops::categorical(),
                    {draft_distributions.back().log_probabilities, sampling_identity(request, 0)}));
              } else {
                const auto selection = sampling::processed_selection(r, processed, std::optional<V>{}, selected_sampler.policy);
                proposal_ids.push_back(r.reshape(selection.id, {f(1)})); draft_valid.push_back(*selection.valid);
              }
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
            {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)});
        const auto store_state = [&](StatePorts &port, const g::slots::Schema &schema, V rows) {
          r.op(c::Operation::custom(g::slots::Store{schema}), {rows, port.destination_ids}, {}, &port.store);
          if (has_prior_append) check(r.builder.depends_on(port.store, prior_append));
          prior_append = port.store; has_prior_append = true;
          definition.state_sources.push_back({port.store, port.gather});
        };
        if (packed_target) {
          const auto channels = f(geometry.gdn_qkv_width());
          const auto qkv = r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight");
          const auto history_count = f(static_cast<std::int64_t>(cache::multiply(requests, geometry.conv_kernel - 1)));
          const auto candidate_count = f(static_cast<std::int64_t>(cache::multiply(requests, prefix_rows)));
          const auto history_outputs = f(static_cast<std::int64_t>(cache::multiply(
              cache::multiply(requests, prefix_rows), geometry.conv_kernel - 1)));
          const auto sample_count = D(take(r.builder.dimension_sum(history_count, packed)));
          const auto raw_samples = r.one(c::Concat{.axis = 0},
              {r.reshape(history, {history_count, channels}), qkv},
              r.dense_spec(precision.activation_type(), {sample_count, channels}));
          // Preserve the original activation bytes in persistent history. In
          // particular, F32 convolution must not round-trip NaN payloads through
          // a different storage type just to recover the true history tail.
          const auto next_history = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {raw_samples, definition.history_select_port},
              r.dense_spec(precision.activation_type(), {history_outputs, channels}));
          store_state(ports.history, history_schema,
              r.reshape(next_history, {candidate_count, f(history_schema.item_elements())}));
          const auto convolution = r.op(c::sequence_ops::packed_causal_conv1d(),
              {r.cast(qkv, Type::kFloat32),
               r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"),
                   {channels, f(geometry.conv_kernel)}), Type::kFloat32),
               r.cast(history, Type::kFloat32), definition.sequence_offsets_port},
              {r.dense_spec(Type::kFloat32, {packed, channels}),
               r.dense_spec(Type::kFloat32, {batch, f(geometry.conv_kernel - 1), channels})});
          const auto activated = r.reshape(r.unary(c::UnaryKind::kSilu,
              r.cast(convolution[0], precision.activation_type())), {batch, t, channels});
          auto q = r.reshape(r.slice(activated, 2, f(0), f(geometry.gdn_key_width())),
              {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
          auto k = r.reshape(r.slice(activated, 2, f(geometry.gdn_key_width()), f(geometry.gdn_key_width())),
              {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
          auto v = r.reshape(r.slice(activated, 2, f(2 * geometry.gdn_key_width()), f(geometry.gdn_value_width())),
              {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
          v = precision.activation_type() == Type::kFloat32 ? r.copy(v) : r.cast(v, Type::kFloat32);
          q = r.gdn_key_rows(r.l2(q)); k = r.gdn_key_rows(r.l2(k));
          const auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32),
              {packed, f(geometry.gdn_value_heads), f(1)});
          const auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"),
              {packed, f(geometry.gdn_value_heads), f(1)});
          const auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32),
              {f(1), f(geometry.gdn_value_heads), f(1)});
          const auto decay = r.reshape(r.decay("model.language_model." + prefix + "linear_attn.A_log"),
              {f(1), f(geometry.gdn_value_heads), f(1)});
          const auto g = r.binary(c::BinaryKind::kMultiply, decay,
              r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
          const auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
          const auto recurrent = r.op(packed_verify ? c::sequence_ops::packed_gated_delta_net_prefixes(
              static_cast<std::uint32_t>(prefix_rows)) : c::sequence_ops::packed_gated_delta_net(),
              {r.cast(r.reshape(q, {packed, r.spec(q).dimensions()[2], f(geometry.gdn_key_depth)}), Type::kFloat32),
               r.cast(r.reshape(k, {packed, r.spec(k).dimensions()[2], f(geometry.gdn_key_depth)}), Type::kFloat32),
               r.cast(v, Type::kFloat32), g, beta,
               r.reshape(recurrent_rows, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)}),
               definition.sequence_offsets_port},
              {r.dense_spec(Type::kFloat32, {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}),
                packed_verify
                    ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})
                    : r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})});
          store_state(ports.recurrent, recurrent_schema,
              r.reshape(recurrent[1], {candidate_count, f(recurrent_schema.item_elements())}));
          mixed = block_graph::emit_gdn_epilogue(r, recurrent[0], normalized,
              "model.language_model." + prefix,
              {packed, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}, packed,
              i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
        } else {
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
                   ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})
                   : r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth), f(geometry.gdn_key_depth)})});
          store_state(ports.recurrent, recurrent_schema, r.reshape(recurrent[1],
              {f(static_cast<std::int64_t>(requests * prefix_rows)), f(recurrent_schema.item_elements())}));
          mixed = block_graph::emit_gdn_epilogue(r, recurrent[0], normalized,
              "model.language_model." + prefix,
              {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}, packed,
              i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
        }
      } else {
        const auto layer = geometry.attention_slot(i);
        const auto &schema = target_attention_schema(layer);
        ports.key = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kKey)));
        ports.value = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kValue)));
        definition.append_cache_inputs[layer * 2] = ports.key;
        definition.append_cache_inputs[layer * 2 + 1] = ports.value;
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
        x = block_graph::emit_ffn(r, ffn_input, "model.language_model." + prefix,
            i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
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
          frequencies, draft_lag(role) != 0,
          {definition.draft_hidden_select_port, definition.draft_token_select_port});
      const auto append = mtp::append_kv(r, attention_schema, kv);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      prior_append = append.value; has_prior_append = true;
    }
    auto last = verifies_prefix(role) ? r.copy(normalized_last) : packed_prefill
        ? r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
            {normalized_last, definition.last_ids_port}, r.dense_spec(precision.activation_type(), {batch, f(hidden)})) :
        (requests == 1 ? r.copy(r.slice(normalized_last, 0, D(r.last_row), f(1))) :
        r.one(c::Gather{}, {normalized_last, definition.last_ids_port},
            r.dense_spec(precision.activation_type(), {batch, f(hidden)})));
    if (hidden_schema.valid()) {
      auto &port = definition.hidden_state;
      // Verifiers retain every target prefix; prefill/single retain only the
      // same final row per member as their existing complete boundary.
      const auto retained = packed_verify
          ? r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
              {normalized_last, definition.prefix_ids_port}, r.dense_spec(precision.activation_type(),
                  {f(static_cast<std::int64_t>(cache::multiply(requests, prefix_rows))), f(hidden)}))
          : verifies_prefix(role) ? normalized_last : last;
      r.op(c::Operation::custom(g::slots::Store{hidden_schema}),
          {retained, port.destination_ids}, {}, &port.store);
      if (has_prior_append) check(r.builder.depends_on(port.store, prior_append));
      if (has_mtp(role)) definition.state_sources.push_back({port.store, port.gather});
    }
    definition.hidden_port = last;
    const auto bonus_logits_inputs = [&](std::size_t request) {
      auto inputs = logits_inputs[request];
      if (!selected_sampler.modifiers.history.empty())
        inputs.counts = sampling::advance_history_counts(r, *inputs.counts, r.reshape(proposal_ids[request], {f(1)}));
      return inputs;
    };
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
      } else if (!selected_sampler.enabled) {
        std::vector<V> ids, valid;
        V false_value;
        if (drafts_token(role)) false_value = r.cast(r.scalar(0), Type::kBool);
        for (std::size_t request = 0; request < requests; ++request) {
          const auto first_row = drafts_token(role) ? request * 2 : request;
          const auto selection = sampling::processed_selection(r,
              sampling::process_logits(r, sampling_row(definition.logits_port, first_row), logits_inputs[request], selected_sampler.modifiers.history),
              std::optional<V>{}, selected_sampler.policy);
          ids.push_back(r.reshape(selection.id, {f(1)}));
          auto ok = *selection.valid;
          if (drafts_token(role)) {
            const auto bonus = sampling::processed_selection(r,
                sampling::process_logits(r, sampling_row(definition.logits_port, first_row + 1), bonus_logits_inputs(request), selected_sampler.modifiers.history),
                std::optional<V>{}, selected_sampler.policy);
            ids.push_back(r.reshape(bonus.id, {f(1)}));
            ok = r.one(c::Select{}, {ok, draft_valid[request], false_value});
            ok = r.one(c::Select{}, {ok, *bonus.valid, false_value});
          }
          valid.push_back(ok);
        }
        definition.id_port = r.reshape(r.concat(ids, 0,
            r.dense_spec(Type::kInt32, {drafts_token(role) ? packed : batch})),
            {drafts_token(role) ? packed : batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
      } else {
        std::vector<V> ids, valid, accepted;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto p = sampling::processed_distribution(r, sampling::process_logits(r,
              sampling_row(definition.logits_port, drafts_token(role) ? request * 2 : request), logits_inputs[request], selected_sampler.modifiers.history), selected_sampler.policy);
          if (drafts_token(role)) {
            const auto bonus = sampling::processed_distribution(r, sampling::process_logits(r,
                sampling_row(definition.logits_port, request * 2 + 1), bonus_logits_inputs(request), selected_sampler.modifiers.history), selected_sampler.policy);
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
    if (drafts_token(role) && requests == 1 && packing == RowPacking::kRectangular &&
        !selected_sampler.enabled && selected_sampler.modifiers.empty() &&
        cache_plan.in_flight_batches >= 2 && !scheduler) {
      // Two complete, independently owned I32[1] outputs. OutputSource cannot
      // grant a fabricated subview of the two-row target-ID writer to B.
      definition.successor_token_port = r.copy(r.reshape(r.slice(definition.id_port, 0, f(1), f(1)), {f(1)}));
      definition.successor_catchup_port = r.copy(r.reshape(definition.draft_id_port, {f(1)}));
      r.publish(definition.successor_token_port); r.publish(definition.successor_catchup_port);
    }
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
        .exact_variants = variants, .constants = assembly.constants()};
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
    auto compiled = execution.compile(definition.graph, assembly.constants(), assembly.options(compile_options));
    if (!compiled && (packed_target || compact_draft) && compiled.error().IsUnimplemented() && !compiled.error().has_native_error())
      return false; // Cold capability only; no execution/device error is a fallback.
    auto compiled_family = take(std::move(compiled));
    const auto &family_report = compiled_family.report;
    const auto storage = family_report.total_compiled_storage();
    std::cerr << "compiled text finite family variants=" << family_report.exact_variant_count()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " operation_proposals=" << family_report.total_search().operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts
              << " placement_attempts=" << family_report.total_search().placement_attempts
              << " cold_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    const auto hidden_store = definition.hidden_state.store;
    const auto extra_stores = hidden_store.valid() ? std::span(&hidden_store, 1) : std::span<const g::NodeId>{};
    check(definition.programs.prepare(std::move(compiled_family), definition.graph, variants,
        {definition.state_sources, extra_stores}, take(execution_budget.metadata_allocator())));
    for (std::size_t index = 0; index < definition.chunks.size(); ++index) {
      const auto &child = definition.programs.child(index);
      record_execution_plan(child.compiled.report, assembly.constants());
      report_packed_projections(definition.graph, child.compiled.report, r.packed_projection_nodes,
          child.compiled.executable.workspace_bytes());
    }
    // Publish only a complete cold candidate. Unsupported routes and failures
    // leave no slot for callers to repair; the original Family address stays
    // stable for every concrete program and its later execution carriers.
    table.emplace(family_key, std::move(result));
    return true;
  }

  // These capacities are physical carriers. Actual prefill consumption remains
  // the caller's true input slice and never becomes a padded Session boundary.
  bool prepare_prefill_family(std::span<const std::size_t> capacities,
      TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    // This adapter reuses the original attention mirror's true query control.
    // A graph without that carrier, or a diagnostic graph with different
    // published boundaries, has no implementation of this packed signature.
    if (!target_kv_layers || !layer0_probe.empty()) return false;
    return prepare_family(capacities, false, requests, role, false, completion, sampling_index, RowPacking::kPrefill);
  }

  std::optional<FamilyChild> find_child(std::size_t requests, std::size_t tokens, GraphRole role,
                      bool diagnostic = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                      RowPacking packing = RowPacking::kRectangular) {
    auto &table = diagnostic ? probe_families : families;
    const auto select = [&](const FamilyKey &key) -> std::optional<FamilyChild> {
      const auto found = table.find(key);
      if (found == table.end()) return {};
      auto &family = *found->second;
      const auto point = std::ranges::find(family.chunks, tokens);
      if (point == family.chunks.end()) return {};
      return FamilyChild{family, family.programs.child(static_cast<std::size_t>(point - family.chunks.begin()))};
    };
    if (auto primary = select({requests, role, tokens, completion, sampling_index, packing})) return primary;
    if (packing != RowPacking::kRectangular) return {};
    return select(FamilyKey::exact(requests, role, tokens, completion, sampling_index));
  }

  std::size_t prefill_capacity(std::size_t rows, TextCompletion completion, std::size_t sampling_index,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) const {
    if (!rows || !layer0_probe.empty()) return 0;
    const auto found = families.find({requests, role, 0, completion, sampling_index, RowPacking::kPrefill});
    if (found == families.end() || !found->second) return 0;
    std::size_t selected = 0;
    for (const auto physical : found->second->chunks)
      if (physical >= rows && (!selected || physical < selected)) selected = physical;
    return selected;
  }

  std::size_t verification_capacity(std::size_t rows, std::size_t requests = 1,
                                    std::size_t sampling_index = 0) const {
    std::size_t selected = 0;
    for (const auto &[key, family] : families)
      if (family && key.role == GraphRole::kVerify && key.requests == requests &&
          key.completion == TextCompletion::kEmitNext && key.sampling_index == sampling_index && packs_verify(key.packing))
        for (const auto extent : family->chunks)
          if (rows <= extent && (!selected || extent < selected)) selected = extent;
    return selected;
  }

  // Session admission proves a prepared cohort can execute this member. The
  // actual batch still selects its exact width/capacity before taking a loan.
  bool planned_prefill_member(std::size_t rows, TextCompletion completion, std::size_t sampling_index,
      GraphRole role = GraphRole::kText) const {
    for (const auto &[shape, uses] : shape_plan.entries()) {
      if (!packs_prefill(shape.packing) || shape.role != role || shape.completion != completion ||
          shape.sampling_index != sampling_index || shape.tokens < rows ||
          std::ranges::none_of(uses, [](auto count) { return count != 0; })) continue;
      if (prefill_capacity(rows, completion, sampling_index, shape.requests, role)) return true;
    }
    return false;
  }

  // A generated-token checkpoint can expose a suffix whose length was unknown
  // at model setup. Specialize only that actual cold request. The same original
  // Family/Phase owners remain bounded by the frozen cold shape plan; decoded
  // hot-loop shapes stay in their original immutable finite families.
  void freeze_shape_plan() {
    if (shape_plan.freeze())
      std::cerr << "cold_shape_plan planned_shapes=" << shape_plan.size()
                << " extra_capacity=" << shape_plan.capacity() << '\n';
  }

  void admit_exact_shape(std::size_t tokens, std::size_t requests, GraphRole role,
                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (find_child(requests, tokens, role, false, completion, sampling_index)) return;
    const auto admission_start = Clock::now();
    if (!state_storage_prepared || active_step || carrier_borrowed || !tokens || tokens > capacity ||
        !requests || requests > cache_plan.active_sequences ||
        cache::multiply(tokens, requests) > cache_plan.max_query_rows || verifies_prefix(role))
      throw std::runtime_error("actual suffix admission exceeds the idle model's original query/state bounds");
    // Direct component consumers may not call the explicit startup boundary.
    // Freeze before the first extra shape can mutate the original inventory.
    freeze_shape_plan();
    const auto key = FamilyKey::exact(requests, role, tokens, completion, sampling_index);
    const ExecutionShape shape{requests, tokens, role, completion, sampling_index};
    shape_plan.admit(shape, key,
        [&](const FamilyKey &victim, const ExecutionShape &old_shape) {
          auto old = families.find(victim);
          if (old == families.end() || !old->second->admitted_exact || old->second->chunks.size() != 1 ||
              old_shape != ExecutionShape{victim.requests, old->second->chunks[0], victim.role,
                  victim.completion, victim.sampling_index})
            throw std::runtime_error("exact family inventory is inconsistent");
          for (const bool imported : {false, true}) {
            auto physical = old_shape; physical.imported_sources = imported;
            if (auto phase = phases.find(physical); phase != phases.end()) {
              const auto retired = phase->second->slots.retire_all(execution);
              check(retired.status);
              if (!retired.retired) throw std::runtime_error("exact family carrier has not actually retired");
              phases.erase(phase);
            }
          }
          families.erase(old);
        },
        [&] {
          const std::array chunk{tokens};
          prepare_family(chunk, false, requests, role, true, completion, sampling_index);
        },
        [&]() noexcept { families.erase(key); });
    std::cerr << "cold_exact_admission requests=" << requests << " T=" << tokens
              << " role=" << static_cast<int>(role) << " retained=" << shape_plan.retained()
              << " limit=" << shape_plan.capacity()
              << " seconds=" << std::chrono::duration<double>(Clock::now() - admission_start).count() << '\n';
  }

  void prepare_draft_program(std::size_t rows, bool head, std::size_t sampling_index = 0) {
    preparation_started_ = true;
    if (state_storage_prepared || !cache_plan.draft_cache || !rows || rows > capacity)
      throw std::runtime_error("prepare a positive NextN width before selected cache storage");
    const auto selected_index = head ? sampling_index : 0;
    const auto &selected_sampler = sampling_configuration(selected_index).configuration;
    const auto key = std::tuple{rows, head, selected_index};
    if (draft_programs.contains(key)) return;
    prepare_cache();
    auto prepared = std::make_unique<DraftProgram>(); auto &program = *prepared;
    program.rows = rows; program.head = head; program.sampling_index = selected_index;
    // No reliable auxiliary-use hint: Auto retains the original Commands path.
    program.replay_mode = !selected_sampler.modifiers.history.empty() && head ? g::ReplayMode::kCommands
        : select_replay_mode(replay_policy, 1, selected_sampler.enabled);
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, rows, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(rows));
    program.tokens = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.positions = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    V shifted;
    g::slots::Schema hidden_schema;
    if (head) {
      hidden_schema = take(g::slots::Schema::create(dense(precision.activation_type(), {1, hidden}),
          cache_plan.hidden_slots,
          cache::add(geometry.hidden_bytes(), cache::state_pool_padding(geometry.hidden_bytes()))));
      auto &port = program.hidden_state;
      port.pool = r.input(take(hidden_schema.capacity_description()));
      port.source_ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
      port.destination_ids = r.input(dense(Type::kInt32, {1}));
      shifted = r.one(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          {port.pool, port.source_ids}, r.dense_spec(precision.activation_type(), {count, f(hidden)}), &port.gather);
    } else {
      // This maintenance graph only reads hidden and writes draft KV. Keep
      // its original ready-leaf Commands path; no dummy Store or pool reader.
      for (std::size_t row = 0; row < rows; ++row) {
        program.hidden_inputs.push_back(r.input(dense(precision.activation_type(), {1, hidden})));
      }
      shifted = r.concat(program.hidden_inputs, 0, r.dense_spec(precision.activation_type(), {count, f(hidden)}));
    }
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
      const auto inputs = prepare_logits_inputs(r, selected_sampler.modifiers);
      if (inputs.bias) program.bias = *inputs.bias;
      if (inputs.allowed) program.allowed = *inputs.allowed;
      if (inputs.counts) program.counts = *inputs.counts;
      const auto processed = sampling::process_logits(r, result.logits, inputs, selected_sampler.modifiers.history);
      if (!selected_sampler.enabled) {
        if (selected_sampler.modifiers.empty()) program.id = r.op(c::Operation(c::TopK{.count = 1}), {result.logits},
            {r.dense_spec(Type::kFloat32, {f(1), f(1)}), r.dense_spec(Type::kInt32, {f(1), f(1)})})[1];
        else if (min_draft_probability() == 0) {
          const auto selection = sampling::processed_selection(r, processed, std::optional<V>{}, selected_sampler.policy);
          program.id = selection.id; program.valid = r.host_predicate(*selection.valid);
        }
      }
      if (selected_sampler.enabled || min_draft_probability() > 0) {
        const auto distribution = sampling::processed_distribution(r, processed,
            selected_sampler.enabled ? selected_sampler.policy : sampling::Policy{});
        program.log_probabilities = distribution.log_probabilities; program.valid = r.host_predicate(distribution.valid);
        if (selected_sampler.enabled) {
          program.identity = r.input(dense(Type::kUInt32, {1, 4}));
          program.id = r.one(c::sampling_ops::categorical(), {program.log_probabilities, program.identity});
          r.publish(program.log_probabilities);
        } else if (!selected_sampler.modifiers.empty()) {
          // Threshold already requires this one distribution's complete
          // validity. Reuse it for a safe greedy ID instead of validating the
          // same processed row a second time.
          const auto safe = r.one(c::Select{}, {distribution.valid, processed.logits, r.scalar(0)});
          program.id = r.op(c::Operation(c::TopK{.count = 1}), {safe})[1];
        }
        if (min_draft_probability() > 0) {
          const auto maximum = r.copy(r.reshape(r.one(c::Reduce{.kind = c::ReductionKind::kMaximum,
              .accumulation_type = Type::kFloat32}, {program.log_probabilities}), {f(1)}));
          program.probability = r.unary(c::UnaryKind::kExp, maximum); r.publish(program.probability);
        }
      }
      if (inputs.counts) {
        program.next_counts = sampling::advance_history_counts(r, *inputs.counts, r.reshape(program.id, {f(1)}));
        r.publish(program.next_counts);
      }
      if (program.valid) r.publish(program.valid);
      r.op(c::Operation::custom(g::slots::Store{hidden_schema}),
          {program.hidden, program.hidden_state.destination_ids}, {}, &program.hidden_state.store);
      check(r.builder.depends_on(program.hidden_state.store, program.append.value));
      r.publish(program.id);
    } else program.append = mtp::append_kv(r, attention_schema, kv);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, static_cast<std::int64_t>(rows - 1)}};
    const std::array variants{g::ExactShapeVariant{binding}};
    const auto hidden_store = program.hidden_state.store;
    const auto extra_stores = hidden_store.valid() ? std::span(&hidden_store, 1) : std::span<const g::NodeId>{};
    check(program.programs.prepare(take(execution.compile(program.graph, assembly.constants(),
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants(), .resource_budget = execution_budget}))), program.graph, variants,
        {{}, extra_stores}, take(execution_budget.metadata_allocator())));
    const auto &compiled_program = program.programs.child(0).compiled;
    check(program.carrier.prepare(compiled_program, execution.queue_bindings()));
    record_execution_plan(compiled_program.report, assembly.constants());
    program.literals = std::move(assembly.literals);
    program.token_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.position_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    input_scratch.draft_positions.reserve(rows);
    if (head && selected_sampler.enabled)
      program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
    check(program.io.prepare(compiled_program, head && selected_sampler.enabled ? 3 : 2,
        program.hidden_inputs.size() + bool(program.bias) + bool(program.allowed) + bool(program.counts),
        head ? 1 + bool(program.valid) + (min_draft_probability() > 0) : 0,
        head ? bool(selected_sampler.enabled) + bool(program.next_counts) : 0));
    program.hidden_bindings.reserve(program.hidden_inputs.size());
    for (const auto port : program.hidden_inputs) program.hidden_bindings.push_back(take(program.io.add_input(port)));
    if (program.bias) program.bias_input = take(program.io.add_input(program.bias));
    if (program.allowed) program.allowed_input = take(program.io.add_input(program.allowed));
    if (program.counts) program.counts_input = take(program.io.add_input(program.counts));
    if (head) {
      if (selected_sampler.enabled) program.probabilities_output = take(program.io.add_output(program.log_probabilities));
      if (program.next_counts) program.next_counts_output = take(program.io.add_output(program.next_counts));
    }
    for (const auto &output : program.programs.child(0).outputs()) {
      if (!output.caller_owned()) continue;
      const auto port = output.value;
      if (port == program.next_counts) {
        const auto &facts = program.io.output_facts(program.next_counts_output);
        const auto bytes = take(facts.description.required_bytes());
        program.next_counts_pool = take(c::LeasePool::create(
            take(execution.resource_for(facts.storage)),
            cache::add(cache::multiply(cache::add(bytes, cache::state_pool_padding(bytes)), 2), 1),
            {.max_blocks = 2, .resource_budget = execution_budget}));
      } else if (port == program.log_probabilities) {
        // One verifier cohort retains every member's q until target selection.
        // Keep the original per-member allowance within the existing budget.
        const auto blocks = cache::multiply(cache_plan.active_sequences, cache_plan.candidate_boundaries);
        if (blocks >= UINT32_MAX) throw std::overflow_error("NextN probability block capacity exceeds lease indexing");
        const auto bytes = cache::multiply(cache::multiply(static_cast<std::size_t>(vocabulary), sizeof(float)), blocks);
        program.probability_pool = take(c::LeasePool::create(
            take(execution.resource_for(program.io.output_facts(program.probabilities_output).storage)), cache::add(bytes, 1),
            {.max_blocks = static_cast<std::uint32_t>(blocks), .resource_budget = execution_budget}));
      }
    }
    check(program.io.add_fixed(program.tokens, program.token_values.read()));
    check(program.io.add_fixed(program.positions, program.position_values.read()));
    if (head) {
      program.id_value = take(program.io.add_readback(program.id, execution, allocation_budget));
      if (selected_sampler.enabled) check(program.io.add_fixed(program.identity, program.identity_values.read()));
      if (program.valid) program.valid_value = take(program.io.add_readback(program.valid, execution, allocation_budget));
      if (min_draft_probability() > 0)
        program.probability_value = take(program.io.add_readback(program.probability, execution, allocation_budget));
    }
    check(program.io.seal());
    if (head) {
      const std::array specs{inference::SlotTransactionSpec{program.hidden_state.store,
          rows, 1, {.resource_budget = cache_budget}}};
      check(program.hidden_transactions.prepare(specs));
    }
    program.guard.execution.configure(draft_bank + 2);
    const auto dirty_bound = cache::add(rows, cache::Plan::page_rows - 2) / cache::Plan::page_rows + 1;
    for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank)
      program.guard.execution.dirty[bank].reserve(dirty_bound);
    program.guard.execution.nodes[draft_bank] = program.append.key;
    program.guard.execution.nodes[draft_bank + 1] = program.append.value;
    initialize_draft_program(program);
    draft_programs.emplace(key, std::move(prepared));
  }

  void initialize_draft_program(DraftProgram &program) {
    const std::array sources{g::slots::SourceBinding{program.hidden_state.store, program.hidden_state.gather}};
    inference::ProgramInitialization initialization{
        .io = program.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}};
    if (program.head) initialization.slots.emplace(inference::ProgramInitialization::Slots{
        program.programs.child(0).resource_graph(), sources,
        {.host_workspace = execution.staging(), .resource_budget = execution_budget ? execution_budget : cache_budget}});
    check(program.carrier.initialize(initialization));
  }

  // The shared carrier observes its original native owner. The original Step
  // alone can take back this NextN append/Store cohort before carrier reset.
  hyle::Result<bool> recover_draft_program(DraftProgram &program) noexcept {
    auto &guard = program.guard;
    const auto observed = guard.execution.retire(execution);
    if (!observed.retired) return false;
    return guard.try_abort();
  }

  // Append to an unpublished draft only. The first append uses COW; later
  // recursive rows may extend that private tail. Published Session tails are
  // never borrowed, so aborting proposal/verification leaves the old head valid.
  DraftProposal execute_draft(DraftWork &work, std::span<const std::int32_t> ids,
      std::span<const c::ReadLease> shifted, bool head, const std::array<std::uint32_t, 4> &identity = {},
      std::size_t sampling_index = 0, const LogitsInputs &logits = {}) {
    if (ids.empty() || ids.size() != shifted.size() || !state_storage_prepared ||
        work.state.frontier > capacity || ids.size() > capacity - work.state.frontier)
      throw std::runtime_error("NextN input rows exceed the prepared context or hidden geometry");
    auto found = draft_programs.find({ids.size(), head, head ? sampling_index : 0});
    if (found == draft_programs.end()) throw std::runtime_error("NextN finite program was not cold prepared");
    auto &program = *found->second;
    const auto &selected_sampler = sampling_configuration(program.sampling_index).configuration;
    if (head) validate_logits_inputs(selected_sampler, logits, vocabulary);
    auto &guard = program.guard;
    if (guard.execution.cleanup_pending) {
      // Keep the same original failed episode and all its append/state facts
      // until real retirement; no new carrier can reuse these permissions.
      if (!take(recover_draft_program(program)))
        throw std::runtime_error("NextN original episode is still retiring");
      guard.execution.cleanup_pending = false; guard.execution.terminal = {};
    }
    if (!program.carrier.initialized()) initialize_draft_program(program);
    auto &run = *take(program.carrier.borrow());
    guard.execution.run = &run; guard.execution.carrier = &program.carrier;
    if (head) guard.execution.state_transactions = &program.hidden_transactions;
    cache::DraftState next;
    DraftProposal output;
    try {
      const auto length = work.state.frontier;
      next.frontier = length + ids.size();
      if (!next.paged.frames.try_resize(work.state.paged.frames.size())) throw std::bad_alloc{};
      std::array<c::PagedArenaCommit, 2> commits;
      inference::PagedSequenceEdit edit(work.state.paged, next.paged, commits);
      c::BatchLayoutVersion layout;
      const auto member_at = [&](std::size_t) noexcept -> inference::PagedAppendMember {
        return {work.state.paged, &next.paged, layout, length, ids.size(), work.tails, true};
      };
      layout = take(inference::prepare_paged_append(attention_schema, 1,
          inference::PagedAppendBanks{draft_inventory,
              std::span(guard.execution.dirty.data() + draft_bank, 2),
              std::span(guard.execution.tails.data() + draft_bank, 2),
              std::span(guard.execution.appends.data() + draft_bank, 2)},
          input_scratch.paged_appends, member_at,
          {.capacity_rows = capacity, .append_options = {.resource_budget = cache_budget}}));
      auto &positions = input_scratch.draft_positions;
      positions.clear();
      for (std::size_t i = 0; i < ids.size(); ++i) positions.push_back(static_cast<std::int32_t>(length + i));
      program.token_values.write<std::int32_t>(ids); program.position_values.write<std::int32_t>(positions);
      if (head) {
        auto &transactions = program.hidden_transactions;
        std::copy(shifted.begin(), shifted.end(), transactions.sources(0).begin());
        transactions.writers(0)[0] = take(hidden_pool.allocate(transactions.writer_bytes(0), cache::state_pool_alignment));
        check(transactions.stage(0));
      } else for (std::size_t row = 0; row < ids.size(); ++row)
        check(program.io.bind(run, program.hidden_bindings[row], dense(precision.activation_type(), {1, hidden}), shifted[row]));
      if (program.bias_input) check(program.io.bind(run, program.bias_input, logits.bias.description, logits.bias.lease));
      if (program.allowed_input) check(program.io.bind(run, program.allowed_input, logits.allowed.description, logits.allowed.lease));
      if (program.counts_input) check(program.io.bind(run, program.counts_input, logits.counts.description, logits.counts.lease));
      if (head && selected_sampler.enabled) {
        program.identity_values.write<std::uint32_t>(identity);
        check(program.io.bind_output(run, program.probabilities_output, program.probability_pool));
      }
      if (program.next_counts) check(program.io.bind_output(run, program.next_counts_output, program.next_counts_pool));
      const std::array layouts{g::batch::NodeLayout{program.attention, layout}};
      const std::array cache_inputs{head ? program.key : V{}, head ? program.value : V{}};
      check(guard.execution.prepare_appends(program.graph,
          head ? std::span<const g::batch::NodeLayout>(layouts) : std::span<const g::batch::NodeLayout>{},
          draft_bank, 2, cache_inputs, {.host_workspace = execution.staging()}));
      check(guard.execution.bind_consumers(program.programs.child(0).resource_graph()));
      check(guard.execution.prepare_replay({.mode = program.replay_mode, .host_wait = g::HostWaitPolicy::kAllow}));
      check(guard.execution.submit(resources_)); check(guard.execution.wait_success());
      ++draft_program_graph_calls;
      if (head) {
        if (program.valid) {
          const auto flag = take(program.io.predicate(execution, run, program.valid_value));
          if (flag == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
              "recursive draft distribution is invalid; no head or RNG was published"));
          if (flag != 1) throw std::runtime_error("recursive draft validity is outside its predicate encoding");
          if (selected_sampler.enabled) output.log_probabilities =
              take(program.io.retain(run, program.probabilities_output, execution_budget));
          if (min_draft_probability() > 0) {
            check(program.io.read(execution, run, program.probability_value, std::as_writable_bytes(std::span(&output.probability, 1))));
            if (!std::isfinite(output.probability) || output.probability < 0 || output.probability > 1)
              throw std::runtime_error("draft maximum probability is outside its declared probability domain");
          }
        }
        check(program.io.read(execution, run, program.id_value, std::as_writable_bytes(std::span(&output.id, 1))));
        if (output.id < 0 || output.id >= vocabulary) throw std::runtime_error("NextN proposal is outside vocabulary");
        if (program.next_counts) output.next_counts =
            take(program.io.retain(run, program.next_counts_output, execution_budget));
      }
      check(guard.execution.append_transactions().subspan(draft_bank, 2).take(run));
      if (head) {
        auto writers = take(program.hidden_transactions.take(run, 0));
        output.hidden = take(writers[0].publish());
        check(program.hidden_transactions.clear(0));
      }
      std::array<c::PagedPageAppender, 2> tails;
      const auto finish_at = [&](std::size_t) noexcept -> inference::PagedAppendFinishMember {
        return {work.state.paged, next.paged, commits, layout, next.paged.pages.back().page(),
            static_cast<std::uint32_t>(next.frontier % cache::Plan::page_rows), tails};
      };
      check(inference::prepare_paged_append_commits(
          std::span(guard.execution.appends.data() + draft_bank, 2), 1, finish_at));
      check(inference::return_paged_append_tails(
          guard.execution.append_transactions().subspan(draft_bank, 2), 2, 1,
          input_scratch.paged_appends, finish_at));
      check(program.io.release_inputs(program.carrier));
      auto publication = take(edit.prepare_publication());
      publication.publish(&work.state.paged);
      work.tails = std::move(tails); work.state = std::move(next);
      for (auto &append : guard.execution.appends) append = {};
      for (auto &bank : guard.execution.dirty) bank.clear();
      for (auto &bank : guard.execution.tails) bank.clear();
      guard.execution.run = nullptr; guard.execution.carrier = nullptr; guard.execution.state_transactions = nullptr;
      return output;
    } catch (inference::EvaluationFailure &failure) {
      // This validator has no target Step participant. Keep its original
      // private guard's actual retirement observation for a known caller key.
      guard.execution.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (!recovered) inference::record_retirement_status(guard.execution.terminal, recovered.error());
      failure.observe_retirement({recovered && *recovered, guard.execution.terminal});
      if (recovered && *recovered) { guard.execution.cleanup_pending = false; guard.execution.terminal = {}; }
      throw;
    } catch (...) {
      // An unsuccessful private append is never a partially published head.
      guard.execution.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (recovered && *recovered) { guard.execution.cleanup_pending = false; guard.execution.terminal = {}; }
      // Otherwise the original cold guard/run retain the partial cleanup facts;
      // the next use must finish this same retirement before it can initialize.
      throw;
    }
  }

  void prepare_prefix_sampling(std::size_t proposals, std::size_t sampling_index = 0, bool conditional = false) {
    preparation_started_ = true;
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    if ((!selected_sampler.enabled && selected_sampler.modifiers.empty()) || state_storage_prepared || proposals >= cache_plan.candidate_boundaries)
      throw std::runtime_error("prepare bounded speculative-sampling widths before cache storage");
    if (conditional && !selected_sampler.modifiers.bias && !selected_sampler.modifiers.allowed)
      throw std::runtime_error("conditional prefix inputs require configured bias or mask ports");
    const PrefixSamplingKey key{proposals, sampling_index, conditional};
    if (prefix_sampling_programs.contains(key)) return;
    auto prepared = std::make_unique<PrefixSamplingProgram>(); auto &program = *prepared;
    program.proposals = proposals; program.sampling_index = sampling_index;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(proposals + 1));
    program.logits = r.input(r.dense_spec(Type::kFloat32, {count, f(vocabulary)}));
    auto shared_policy = selected_sampler.modifiers;
    if (conditional) { shared_policy.bias = false; shared_policy.allowed = false; }
    const auto inputs = prepare_logits_inputs(r, shared_policy);
    std::vector<sampling::LogitsInputsFor<V>> conditional_ports;
    if (conditional) {
      conditional_ports.reserve(proposals + 1); program.conditional_inputs.resize(proposals + 1);
      const sampling::LogitsPolicy policy{.bias = selected_sampler.modifiers.bias, .allowed = selected_sampler.modifiers.allowed};
      for (auto &row : program.conditional_inputs) {
        auto ports = prepare_logits_inputs(r, policy);
        if (ports.bias) row.bias = *ports.bias;
        if (ports.allowed) row.allowed = *ports.allowed;
        conditional_ports.push_back(std::move(ports));
      }
    }
    if (inputs.bias) program.bias = *inputs.bias;
    if (inputs.allowed) program.allowed = *inputs.allowed;
    if (inputs.counts) program.counts = *inputs.counts;
    if (selected_sampler.enabled) {
      program.identities = r.input(r.dense_spec(Type::kUInt32, {count, f(4)}));
      if (proposals) program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
      auto assembled = sampling::speculative_prefix(r, program.logits, program.identities,
          program.ids, selected_sampler.policy, inputs, selected_sampler.modifiers.history,
          std::span<const sampling::LogitsInputsFor<V>>(conditional_ports));
      program.q = std::move(assembled.q); program.accepted = std::move(assembled.accepted);
      program.final_ids = std::move(assembled.final_ids); program.valid = std::move(assembled.valid);
    } else {
      if (!selected_sampler.modifiers.history.empty() && proposals)
        program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
      auto row_inputs = inputs;
      // A greedy verifier still needs the exact processed target of every row;
      // no proposal-q or RNG ports belong to this program.
      for (std::size_t row = 0; row <= proposals; ++row) {
        if (conditional) {
          row_inputs.bias = conditional_ports[row].bias; row_inputs.allowed = conditional_ports[row].allowed;
        }
        const auto raw = r.copy(r.slice(program.logits, 0, f(static_cast<std::int64_t>(row)), f(1)));
        const auto selection = sampling::processed_selection(r,
            sampling::process_logits(r, raw, row_inputs, selected_sampler.modifiers.history), std::optional<V>{}, selected_sampler.policy);
        program.final_ids.push_back(r.reshape(selection.id, {f(1)})); program.valid.push_back(*selection.valid);
        if (!selected_sampler.modifiers.history.empty() && row < proposals)
          row_inputs.counts = sampling::advance_history_counts(r, *row_inputs.counts,
              r.copy(r.slice(program.ids, 0, f(static_cast<std::int64_t>(row)), f(1))));
      }
    }
    if (selected_sampler.enabled) {
      if (proposals) program.continuation = r.input(dense(Type::kFloat32,
          {static_cast<std::int64_t>(proposals)}));
      program.packet = inference::mtp::sampled_prefix_packet(r, program.ids, program.continuation,
          std::span<const V>(program.accepted), std::span<const V>(program.final_ids), std::span<const V>(program.valid));
    } else {
      program.packet = inference::mtp::validated_ids_packet(r,
          std::span<const V>(program.final_ids), std::span<const V>(program.valid));
    }
    r.publish(program.packet);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, 0}};
    const std::array variants{g::ExactShapeVariant{binding}};
    check(program.programs.prepare(take(execution.compile(program.graph, assembly.constants(),
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants(), .resource_budget = execution_budget}))), program.graph, variants));
    const auto &compiled_program = program.programs.child(0).compiled;
    check(program.carrier.prepare(compiled_program, execution.queue_bindings()));
    record_execution_plan(compiled_program.report, assembly.constants());
    program.literals = std::move(assembly.literals);
    if (program.identities) {
      program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {static_cast<std::int64_t>(proposals + 1), 4}));
      input_scratch.sampling_identities.reserve(take(program.identity_values.description.required_bytes()) / sizeof(std::uint32_t));
    }
    if (program.ids) program.id_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    if (program.continuation) {
      program.continuation_values = caller_tensor(execution.device(), dense(Type::kFloat32,
          {static_cast<std::int64_t>(proposals)}));
      program.continuation_host.resize(proposals);
    }
    program.packet_host.resize(selected_sampler.enabled ? proposals + 5 : proposals + 2);
    check(program.io.prepare(compiled_program, bool(program.identities) + bool(program.ids) + bool(program.continuation),
        cache::add(1 + program.q.size() + bool(program.bias) + bool(program.allowed) + bool(program.counts),
            cache::multiply(program.conditional_inputs.size(), std::size_t(selected_sampler.modifiers.bias) + std::size_t(selected_sampler.modifiers.allowed))),
        1, 0));
    if (program.identities) check(program.io.add_fixed(program.identities, program.identity_values.read()));
    if (program.ids) check(program.io.add_fixed(program.ids, program.id_values.read()));
    if (program.continuation) check(program.io.add_fixed(program.continuation, program.continuation_values.read()));
    program.packet_value = take(program.io.add_readback(program.packet, execution, allocation_budget));
    if (program.bias) program.bias_input = take(program.io.add_input(program.bias));
    if (program.allowed) program.allowed_input = take(program.io.add_input(program.allowed));
    if (program.counts) program.counts_input = take(program.io.add_input(program.counts));
    for (auto &row : program.conditional_inputs) {
      if (row.bias) row.bias_input = take(program.io.add_input(row.bias));
      if (row.allowed) row.allowed_input = take(program.io.add_input(row.allowed));
    }
    program.q_inputs.reserve(program.q.size());
    for (const auto port : program.q) program.q_inputs.push_back(take(program.io.add_input(port)));
    program.logits_input = take(program.io.add_input(program.logits));
    check(program.io.seal());
    initialize_prefix_sampling(program);
    prefix_sampling_programs.emplace(key, std::move(prepared));
  }

  void initialize_prefix_sampling(PrefixSamplingProgram &program) {
    check(program.carrier.initialize({
        .io = program.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}}));
  }

  // Pure private consumers have no append/Store transaction to recover. A
  // previously closed run can be replaced only after its real owner retires.
  void recover_private_carrier(inference::ProgramCarrier<ColdResources> &carrier) {
    if (!carrier.closing()) return;
    const auto observed = carrier.retire(execution);
    if (!observed.retired) {
      check(observed.status);
      throw std::runtime_error("original private program carrier is still retiring");
    }
    check(carrier.reset_retired());
    check(observed.status);
  }

  // Target family widths and replay counts stay with the finite request plan;
  // every recursive consumer shares this model's actual NextN/sampler programs.
  void prepare_recursive_round_programs(std::size_t sampling_index = 0, bool conditional = false) {
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    for (std::size_t rows = 1; rows <= cache_plan.candidate_boundaries; ++rows) {
      prepare_draft_program(rows, false);
      if (selected_sampler.enabled || !selected_sampler.modifiers.empty())
        prepare_prefix_sampling(rows - 1, sampling_index, conditional);
    }
    prepare_draft_program(1, true, sampling_index);
    if (capacity > 1) prepare_draft_program(2, true, sampling_index);
  }

  PrefixSamplingResult sample_recursive_prefix(c::ConstTensorView logits,
      std::span<const std::int32_t> proposals, std::span<const c::ReadLease> q,
      cache::SamplingCursor cursor, std::size_t quota, std::span<const std::int32_t> eos,
      const inference::RequestStopPolicy *stop_policy, std::size_t sampling_index = 0, const LogitsInputs &inputs = {},
      std::span<const LogitsInputs> conditional_inputs = {}) {
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    if ((selected_sampler.enabled ? proposals.size() != q.size() : !q.empty() || selected_sampler.modifiers.empty()))
      throw std::runtime_error("speculative sampling requires its actual proposal distributions or greedy modifiers");
    if (!quota) throw std::runtime_error("speculative prefix requires a positive output quota");
    inference::mtp::validate_token_ids(proposals, static_cast<std::int32_t>(vocabulary));
    inference::request_stop_policy(stop_policy).validate(vocabulary);
    validate_prefix_logits_inputs(selected_sampler, inputs, conditional_inputs, cache::add(proposals.size(), 1), vocabulary);
    auto found = prefix_sampling_programs.find({proposals.size(), sampling_index, !conditional_inputs.empty()});
    if (found == prefix_sampling_programs.end()) throw std::runtime_error("speculative-sampling width was not cold prepared");
    auto &program = *found->second;
    recover_private_carrier(program.carrier);
    if (!program.carrier.initialized()) initialize_prefix_sampling(program);
    auto &run = *take(program.carrier.borrow());
    const auto count = proposals.size();
    if (selected_sampler.enabled) {
      (void)cursor.advanced(cache::add(cache::multiply(2, count), 1));
      auto &identities = input_scratch.sampling_identities;
      identities.clear();
      for (std::size_t row = 0; row <= count; ++row) {
        const auto identity = cursor.identity(row == count ? 2 * count : 2 * row + 1);
        identities.insert(identities.end(), identity.begin(), identity.end());
      }
      program.identity_values.write<std::uint32_t>(identities);
    }
    if (program.ids) program.id_values.write<std::int32_t>(proposals);
    if (program.continuation) {
      for (std::size_t row = 0; row < count; ++row)
        program.continuation_host[row] = row < quota - 1 &&
            !inference::request_token_stopped(proposals[row], eos, stop_policy) ? 1.F : 0.F;
      program.continuation_values.write<float>(program.continuation_host);
    }
    try {
      check(program.io.bind(run, program.logits_input, logits));
      if (program.bias_input) check(program.io.bind(run, program.bias_input, inputs.bias.description, inputs.bias.lease));
      if (program.allowed_input) check(program.io.bind(run, program.allowed_input, inputs.allowed.description, inputs.allowed.lease));
      if (program.counts_input) check(program.io.bind(run, program.counts_input, inputs.counts.description, inputs.counts.lease));
      for (std::size_t row = 0; row < program.conditional_inputs.size(); ++row) {
        const auto &ports = program.conditional_inputs[row]; const auto &values = conditional_inputs[row];
        if (ports.bias_input) check(program.io.bind(run, ports.bias_input, values.bias.description, values.bias.lease));
        if (ports.allowed_input) check(program.io.bind(run, ports.allowed_input, values.allowed.description, values.allowed.lease));
      }
      for (std::size_t row = 0; row < q.size(); ++row)
        check(program.io.bind(run, program.q_inputs[row], dense(Type::kFloat32, {1, vocabulary}), q[row]));
      const auto queues = program.carrier.queues();
      check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      auto done = take(program.carrier.submit(resources_)); check(done.wait());
      ++prefix_decision_graph_calls;
      check(program.io.read(execution, run, program.packet_value,
          std::as_writable_bytes(std::span(program.packet_host))));
      const auto words = std::span<const std::int32_t>(program.packet_host);
      if (words[0] == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
          "recursive target/q/residual distribution is invalid; no head or RNG was published"));
      if (words[0] != 1) throw std::runtime_error("recursive target validity is outside its predicate encoding");
      PrefixSamplingResult result;
      if (selected_sampler.enabled) {
        const auto consumed = words[1];
        const auto emitted = words.subspan(4);
        inference::mtp::validate_token_ids(emitted, static_cast<std::int32_t>(vocabulary));
        if (consumed <= 0 || static_cast<std::size_t>(consumed) > (std::min)(quota, count + 1) ||
            (words[3] != 0 && words[3] != 1) || words[2] != emitted[static_cast<std::size_t>(consumed) - 1])
          throw std::runtime_error("device prefix result differs from its bounded emitted rows");
        result.prefix = {.consumed = static_cast<std::size_t>(consumed),
            .consumed_drafts = static_cast<std::size_t>(consumed) - 1, .next_id = words[2],
            .first_agrees = words[3] != 0,
            .stopped = inference::request_token_stopped(words[2], eos, stop_policy)};
        result.final_ids.assign(emitted.begin(), emitted.begin() + consumed);
      } else {
        const auto ids = words.subspan(1);
        inference::mtp::validate_token_ids(ids, static_cast<std::int32_t>(vocabulary));
        result.final_ids.assign(ids.begin(), ids.end());
      }
      check(program.io.release_inputs(program.carrier));
      return result;
    } catch (inference::EvaluationFailure &failure) {
      const auto observed = program.carrier.retire(execution);
      failure.observe_retirement(observed);
      if (observed.retired) program.carrier.reset_retired().IgnoreError();
      throw;
    } catch (...) {
      // The outer target episode retains its original member proof and drives
      // actual retirement; this finite private consumer releases only its pins.
      const auto observed = program.carrier.retire(execution);
      if (observed.retired) program.carrier.reset_retired().IgnoreError();
      throw;
    }
  }

  static c::PagedArenaFrame working_frame(c::PagedArenaRead read) {
    return take(c::PagedArenaFrame::fork(take(c::PagedArenaFootprint::from_read(read))));
  }
  static cache::DraftState fork_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.paged = take(source.paged.fork()); next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    return next;
  }
  static cache::DraftState alias_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.paged = take(source.paged.frozen_alias()); next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    return next;
  }

  std::unique_ptr<cache::State> reserve_state(cache::WorkingState &owner) {
    auto version = take(cache::VersionRef::acquire(snapshot_versions));
    auto result = owner.acquire_candidate();
    result->resource_owner = resources_;
    if (!result->paged.frames.try_resize(target_kv_layers * 2) ||
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
    next->paged = take(base->paged.frozen_alias());
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
        if (inputs.counts) prepared->counts_port = *inputs.counts;
        const auto selection = sampling::ordinary_selection(r,
            r.scores(prepared->hidden_port, std::string(geometry.head_role())),
            selected_sampler.enabled ? std::optional<V>(prepared->identity_port) : std::nullopt,
            selected_sampler.policy, inputs, selected_sampler.modifiers.history);
        prepared->id_port = selection.id;
        if (selection.valid) prepared->valid_port = r.host_predicate(*selection.valid);
        r.publish(prepared->id_port); if (prepared->valid_port) r.publish(prepared->valid_port);
        prepared->graph = take(std::move(r.builder).finish());
        // The frontier is a fixed [1,H] projection with no symbolic model rows.
        const std::array shape{g::ShapeBinding{r.last_row, 0}};
        const std::array variants{g::ExactShapeVariant{shape}};
        check(prepared->programs.prepare(take(execution.compile(prepared->graph, assembly.constants(),
            assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
             .constants = assembly.constants(), .resource_budget = execution_budget}))), prepared->graph, variants));
        const auto &compiled_program = prepared->programs.child(0).compiled;
        check(prepared->carrier.prepare(compiled_program, execution.queue_bindings()));
        record_execution_plan(compiled_program.report, assembly.constants());
        prepared->literals = std::move(assembly.literals);
        if (selected_sampler.enabled) prepared->identity = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
        check(prepared->io.prepare(compiled_program, bool(prepared->identity_port),
            1 + bool(prepared->bias_port) + bool(prepared->allowed_port) + bool(prepared->counts_port),
            1 + bool(prepared->valid_port), 0));
        if (prepared->identity_port) check(prepared->io.add_fixed(prepared->identity_port, prepared->identity.read()));
        prepared->id = take(prepared->io.add_readback(prepared->id_port, execution, allocation_budget));
        if (prepared->valid_port) prepared->valid = take(prepared->io.add_readback(prepared->valid_port, execution, allocation_budget));
        prepared->hidden_input = take(prepared->io.add_input(prepared->hidden_port));
        if (prepared->bias_port) prepared->bias_input = take(prepared->io.add_input(prepared->bias_port));
        if (prepared->allowed_port) prepared->allowed_input = take(prepared->io.add_input(prepared->allowed_port));
        if (prepared->counts_port) prepared->counts_input = take(prepared->io.add_input(prepared->counts_port));
        check(prepared->io.seal());
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
    recover_private_carrier(frontier.carrier);
    // This is the same Graph execution API/runtime as target/draft. Keep the
    // original fresh-run frontier policy, with atomic fixed-IO installation.
    check(frontier.carrier.initialize({
        .io = frontier.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}}));
    auto &run = *take(frontier.carrier.borrow());
    try {
      if (selected_sampler.enabled) { const auto identity = cursor.identity(); frontier.identity.write<std::uint32_t>(identity); }
      check(frontier.io.bind(run, frontier.hidden_input, dense(precision.activation_type(), {1, hidden}), base->carry_hidden));
      if (frontier.bias_input) check(frontier.io.bind(run, frontier.bias_input, inputs.bias.description, inputs.bias.lease));
      if (frontier.allowed_input) check(frontier.io.bind(run, frontier.allowed_input, inputs.allowed.description, inputs.allowed.lease));
      if (frontier.counts_input) check(frontier.io.bind(run, frontier.counts_input, inputs.counts.description, inputs.counts.lease));
      const auto queues = frontier.carrier.queues();
      check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      auto done = take(frontier.carrier.submit(resources_)); check(done.wait());
      ++frontier_sampling_graph_calls;
      const auto valid = frontier.valid_port ? take(frontier.io.predicate(execution, run, frontier.valid)) : std::uint8_t{1};
      std::int32_t pending = -1;
      check(frontier.io.read(execution, run, frontier.id, std::as_writable_bytes(std::span(&pending, 1))));
      if (valid == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
          "cached frontier sampling is invalid; no request RNG cursor was published"));
      if (valid != 1 || pending < 0 || pending >= vocabulary)
        throw std::runtime_error("cached frontier sampling is outside its declared result domain");
      auto next = sampling_fork(std::move(base), next_cursor, pending);
      frontier_sampling_seconds += std::chrono::duration<double>(Clock::now() - started).count();
      check(frontier.carrier.reset_retired());
      return next;
    } catch (inference::EvaluationFailure &failure) {
      const auto observed = frontier.carrier.retire(execution);
      failure.observe_retirement(observed);
      if (observed.retired) frontier.carrier.reset_retired().IgnoreError();
      throw;
    } catch (...) {
      const auto observed = frontier.carrier.retire(execution);
      if (observed.retired) frontier.carrier.reset_retired().IgnoreError();
      throw;
    }
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
      if (!find_child(requests, tokens, role, false, completion, sampling_index))
        throw std::runtime_error("execution plan chunk is outside the shared finite family");
      shape_plan.add_use({requests, tokens, role, completion, sampling_index}, slot_index);
    }
  }

  // Finite wave/suffix plans can count many possible restores without
  // materializing repeated host call tapes or execution carriers.
  void plan_replay_uses(std::size_t tokens, std::size_t requests, std::size_t count,
                        GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                        RowPacking packing = RowPacking::kRectangular) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    if (!find_child(requests, tokens, role, false, completion, sampling_index, packing))
      throw std::runtime_error("execution use shape is outside the shared finite family");
    shape_plan.add_uses({requests, tokens, role, completion, sampling_index, false, packing}, count);
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
    if (resample_frontier) validate_logits_inputs(selected_sampler, logits, vocabulary);
    prepare_state_storage();
    const auto initial_rows = planned_chunks.empty() ? 0 : planned_chunks.front();
    const auto prepared_rows = [&](std::size_t rows) {
      return find_child(1, rows, role, false, completion, sampling_index) ||
          ((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(rows, completion, sampling_index, role));
    };
    for (const auto rows : planned_chunks)
      if (!prepared_rows(rows)) admit_exact_shape(rows, 1, role, completion, sampling_index);
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("session must use its complete target/draft cache family");
    const bool prepared_adoption = families.contains({1, role, 0, completion, sampling_index}) ||
        ((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(1, completion, sampling_index, role));
    if ((planned_chunks.empty() ? !prepared_adoption : !prepared_rows(initial_rows)) ||
        (!layer0_probe.empty() && (planned_chunks.empty()
            ? !probe_families.contains({1, role, 0, completion, sampling_index})
            : !find_child(1, initial_rows, role, true, completion, sampling_index))))
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
      if (!layer0_probe.empty() && !find_child(1, tokens, role, true, completion, sampling_index))
        throw std::runtime_error("session chunk is outside the shared diagnostic family");
      total += tokens;
      if (!shape_plan.contains({1, tokens, role, completion, sampling_index}) &&
          !((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(tokens, completion, sampling_index, role)))
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
      for (std::size_t i = 0; i < initial->paged.frames.size() / 2; ++i) {
        initial->paged.frames[2 * i] = working_frame(take(page_inventory.banks[2 * i].read(layout, {})));
        initial->paged.frames[2 * i + 1] = working_frame(take(page_inventory.banks[2 * i + 1].read(layout, {})));
      }
      if (cache_plan.draft_cache) {
        initial->draft.emplace();
        if (!initial->draft->paged.frames.try_resize(2)) throw std::bad_alloc{};
        initial->draft->paged.frames[0] = working_frame(take(draft_inventory.banks[0].read(layout, {})));
        initial->draft->paged.frames[1] = working_frame(take(draft_inventory.banks[1].read(layout, {})));
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

  void validate_mtp_snapshot(const cache::State &head, const StepMember *candidate = nullptr) const {
    if (cache_plan.draft_cache != head.draft.has_value())
      throw std::runtime_error("snapshot does not contain the configured complete MTP state");
    if (!head.draft) return;
    const auto &draft = *head.draft;
    if (draft.paged.frames.size() != 2)
      throw std::runtime_error("snapshot does not contain its complete draft bank cohort");
    const auto key = candidate && candidate->draft_updates[0].valid()
        ? take(candidate->draft_updates[0].view()) : take(draft.paged.frames[0].view());
    const auto value = candidate && candidate->draft_updates[1].valid()
        ? take(candidate->draft_updates[1].view()) : take(draft.paged.frames[1].view());
    if (draft.lag > 1 || head.length < draft.lag || draft.frontier != head.length - draft.lag ||
        !key.valid() || !value.valid() || key.layout().facts().kv_rows != draft.frontier ||
        value.layout().facts().kv_rows != draft.frontier || !head.carry_hidden.valid() ||
        (draft.lag && (!head.prior_hidden.valid() || draft.catchup_token < 0 || draft.catchup_token >= vocabulary)) ||
        (!draft.lag && draft.catchup_token != -1))
      throw std::runtime_error("inconsistent MTP frontier, catch-up token or owned target hidden state");
  }

  Phase::ExecutionSlots::Initialization slot_initialization(const Phase &phase) const {
    Phase::ExecutionSlots::Initialization initialization{
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}};
    if (!phase.state_banks.empty()) initialization.slots.emplace(inference::ProgramInitialization::Slots{
        phase.state_graph(), phase.state_sources,
        {.host_workspace = execution.staging(), .resource_budget = execution_budget ? execution_budget : cache_budget}});
    return initialization;
  }

  void initialize_slot(Phase &phase, std::size_t slot) {
    check(phase.slots.initialize_with(slot, [&] { return slot_initialization(phase); }));
  }

  Phase &prepare_child(std::size_t tokens, bool diagnostic, std::size_t requests = 1, GraphRole role = GraphRole::kText,
                       TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                       bool imported_sources = false, RowPacking packing = RowPacking::kRectangular) {
    const bool compact_draft = packs_draft(packing);
    const bool packed_verify = packs_verify(packing), packed_target = packs_target(packing);
    auto &cache = diagnostic ? probe_phases : phases;
    const auto found = cache.find({requests, tokens, role, completion, sampling_index, imported_sources, packing});
    if (found != cache.end()) return *found->second;
    const auto selected = find_child(requests, tokens, role, diagnostic, completion, sampling_index, packing);
    if (!selected) throw std::runtime_error("text finite family has not been prepared");
    auto &definition = selected->family;
    const auto &selected_sampler = definition.sampling->configuration;
    const auto start = Clock::now();
    auto result = std::make_unique<Phase>();
    auto &phase = *result;
    std::size_t physical_slots = 1;
    phase.imported_sources = imported_sources;
    if (imported_sources) {
      const bool ordinary = tokens == 1 && role == GraphRole::kText && !cache_plan.draft_cache;
      const bool mtp = tokens == 2 && requests == 1 && role == GraphRole::kMtpCatchupRound &&
          cache_plan.draft_cache && definition.successor_token_port &&
          !selected_sampler.enabled && selected_sampler.modifiers.empty() && packing == RowPacking::kRectangular;
      if ((!ordinary && !mtp) || diagnostic || completion != TextCompletion::kEmitNext ||
          cache_plan.in_flight_batches < 2 || scheduler)
        throw std::runtime_error("imported decode carriers require a complete supported predecessor and two candidate depths");
      physical_slots = execution_slot_count;
    }
    phase.query_rows = tokens;
    static_cast<FamilyPorts &>(phase) = static_cast<const FamilyPorts &>(definition);
    if (imported_sources) for (auto &source : phase.state_sources) source.mode = g::slots::SourceMode::kImported;
    const auto planned_slot_uses = shape_plan.uses({requests, tokens, role, completion, sampling_index, false, packing});
    phase.attention_schema = definition.attention_schema;
    phase.child = &selected->program;
    phase.family = &definition;
    std::vector<inference::SlotTransactionSpec> transactions;
    phase.state_banks.reserve(phase.state_sources.size() + (phase.hidden_state.store.valid() ? 1u : 0u));
    transactions.reserve(phase.state_banks.capacity());
    const auto add_bank = [&](StateBank bank, const StatePorts &port, std::size_t sources) {
      const auto paired = std::ranges::find(phase.state_sources, port.store, &g::slots::SourceBinding::store);
      if (paired != phase.state_sources.end())
        bank.source_index = static_cast<std::size_t>(paired - phase.state_sources.begin());
      phase.state_banks.push_back(bank);
      transactions.push_back({port.store, imported_sources ? 0 : sources,
          cache::multiply(requests, phase.prefix_rows), {.resource_budget = cache_budget}});
    };
    for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (geometry.is_gdn(layer))
      for (const bool recurrent : {false, true}) {
        const auto &port = recurrent ? phase.layers[layer].recurrent : phase.layers[layer].history;
        add_bank({.layer = layer, .kind = recurrent ? StateBankKind::kRecurrent : StateBankKind::kHistory},
            port, requests);
      }
    if (phase.hidden_state.store.valid())
      add_bank({.kind = StateBankKind::kHidden}, phase.hidden_state,
          has_mtp(role) ? cache::multiply(requests, draft_lag(role) + 1) : 0);
    std::erase_if(phase.state_resource_ports, [&](V value) {
      return value.kind() != g::ValueKind::kInput && !phase.child->output(value).caller_owned();
    });
    const auto t = static_cast<std::int64_t>(tokens);
    if (completion == TextCompletion::kEmitNext)
      phase.logits_facts = &phase.child->output(phase.logits_port);
    phase.hidden_facts = &phase.child->output(phase.hidden_port);
    if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      phase.probe_facts[i] = &phase.child->output(phase.probe_ports[i]);
    check(phase.slots.prepare(phase.child->compiled, execution.queue_bindings(), physical_slots, transactions,
        [&](std::size_t slot, auto &program_slot) {
      auto &io = program_slot.data;
      io.batch_layouts.reserve(phase.attention_nodes.size() + drafts_token(role));
      for (auto node : phase.attention_nodes) io.batch_layouts.push_back({node, {}});
      if (drafts_token(role)) io.batch_layouts.push_back({phase.draft_attention_node, {}});
      program_slot.planned_uses = planned_slot_uses[slot];
      program_slot.replay_mode = !selected_sampler.modifiers.history.empty() ? g::ReplayMode::kCommands
          : imported_sources ? g::ReplayMode::kPreferCapture : select_replay_mode(
          replay_policy, program_slot.planned_uses, selected_sampler.enabled);
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
        io.sampling_valid_values.resize(requests);
        bool float_predicates = phase.child->output(phase.sampling_valid_port).description.data_type() == Type::kFloat32;
        if (phase.sampling_accept_port) {
          io.sampling_accept_values.resize(requests);
          float_predicates = float_predicates || phase.child->output(phase.sampling_accept_port).description.data_type() == Type::kFloat32;
        }
        if (float_predicates) io.sampling_scratch.resize(requests);
      }
      if (has_mtp(role)) {
        const auto draft_rows = is_mtp_prefill(role) ? t + static_cast<std::int64_t>(draft_lag(role)) : static_cast<std::int64_t>(draft_lag(role) + 1);
        io.draft_positions = caller_tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {draft_rows})
            : dense(Type::kInt32, {static_cast<std::int64_t>(requests), draft_rows}));
        if (draft_lag(role)) io.catchup_token = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        if (packs_prefill(definition.packing) || compact_draft) {
          const auto count = cache::multiply(requests, static_cast<std::size_t>(draft_rows));
          const auto description = dense(Type::kInt32, {static_cast<std::int64_t>(count)});
          io.draft_hidden_select = caller_tensor(execution.device(), description);
          io.draft_hidden_indices.resize(count);
          io.draft_hidden_select.write<std::int32_t>(io.draft_hidden_indices);
          if (draft_lag(role)) {
            io.draft_token_select = caller_tensor(execution.device(), description);
            io.draft_token_indices.resize(count);
            io.draft_token_select.write<std::int32_t>(io.draft_token_indices);
          }
        }
        if (compact_draft) io.draft_last_ids = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        if (drafts_token(role)) {
          io.draft_ids.resize(requests);
        }
      }
      if ((requests != 1 || packs_prefill(definition.packing)) && !verifies_prefix(role)) {
        io.last_ids = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
        std::vector<std::int32_t> last_ids(requests);
        for (std::size_t j = 0; j < requests; ++j) last_ids[j] = static_cast<std::int32_t>((j + 1) * tokens - 1);
        io.last_ids.write<std::int32_t>(last_ids);
        if (packs_prefill(definition.packing)) io.last_id_values.resize(requests);
      }
      if (packed_target) {
        const auto count = cache::multiply(cache::multiply(requests, phase.prefix_rows), geometry.conv_kernel - 1);
        io.history_select = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(count)}));
        io.history_select_values.resize(count);
        if (packed_verify) {
          io.last_id_values.resize(requests);
          const auto prefixes = cache::multiply(requests, phase.prefix_rows);
          io.prefix_ids = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(prefixes)}));
          io.prefix_id_values.resize(prefixes);
        }
      }
      if (completion == TextCompletion::kEmitNext) {
        io.logits = caller_tensor(take(execution.resource_for(phase.logits_facts->storage)),
            phase.logits_facts->description, phase.logits_facts->caller_owned());
      }
      if (verifies_prefix(role)) io.prefix_candidates.resize(cache::multiply(requests, phase.prefix_rows));
      io.final_hidden = caller_tensor(take(execution.resource_for(phase.hidden_facts->storage)),
          phase.hidden_facts->description, phase.hidden_facts->caller_owned());
      if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
        const auto &output = *phase.probe_facts[i];
        io.probe_outputs[i] = caller_tensor(take(execution.resource_for(output.storage)), output.description, output.caller_owned());
      }
      const bool sampling_identity = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify;
      const bool last_ids = (requests != 1 || packs_prefill(definition.packing)) && !verifies_prefix(role);
      std::size_t fixed_count = 2 + sampling_identity + last_ids + packed_target + packed_verify +
          io.logits.caller_owned() + io.final_hidden.caller_owned();
      if (has_mtp(role)) fixed_count += 1 + bool(draft_lag(role)) +
          ((packs_prefill(definition.packing) || compact_draft) ? 1 + bool(draft_lag(role)) : 0) + compact_draft;
      if (diagnostic) for (const auto *output : phase.probe_facts)
        fixed_count += output && output->caller_owned();
      const auto readback_count = bool(phase.sampling_valid_port) + bool(phase.sampling_accept_port) +
          (completion == TextCompletion::kEmitNext) + drafts_token(role) +
          bool(phase.successor_token_port) + bool(phase.successor_catchup_port);
      check(program_slot.io.prepare(phase.child->compiled, fixed_count, 0, readback_count, 0));
      // Register once in the original cold installation order. Tensor owners
      // stay here; each readback's only owner moves into the same shared IO.
      check(program_slot.io.add_fixed(phase.token_port, io.tokens.read()));
      check(program_slot.io.add_fixed(phase.position_port, io.positions.read()));
      if (sampling_identity) check(program_slot.io.add_fixed(phase.sampling_identity_port, io.sampling_identities.read()));
      if (phase.sampling_valid_port) {
        io.sampling_valid = take(program_slot.io.add_readback(phase.sampling_valid_port, execution, allocation_budget));
        if (phase.sampling_accept_port)
          io.sampling_accept = take(program_slot.io.add_readback(phase.sampling_accept_port, execution, allocation_budget));
      }
      if (last_ids) check(program_slot.io.add_fixed(phase.last_ids_port, io.last_ids.read()));
      if (packed_target) check(program_slot.io.add_fixed(phase.history_select_port, io.history_select.read()));
      if (packed_verify) check(program_slot.io.add_fixed(phase.prefix_ids_port, io.prefix_ids.read()));
      if (io.logits.caller_owned()) check(program_slot.io.add_fixed_output(phase.logits_port, io.logits.view()));
      if (completion == TextCompletion::kEmitNext)
        io.next_id = take(program_slot.io.add_readback(phase.id_port, execution, allocation_budget));
      if (phase.successor_token_port)
        io.successor_token = take(program_slot.io.add_readback(phase.successor_token_port, execution, allocation_budget));
      if (phase.successor_catchup_port)
        io.successor_catchup = take(program_slot.io.add_readback(phase.successor_catchup_port, execution, allocation_budget));
      if (io.final_hidden.caller_owned()) check(program_slot.io.add_fixed_output(phase.hidden_port, io.final_hidden.view()));
      if (has_mtp(role)) {
        check(program_slot.io.add_fixed(phase.draft_position_port, io.draft_positions.read()));
        if (draft_lag(role)) check(program_slot.io.add_fixed(phase.catchup_token_port, io.catchup_token.read()));
        if (packs_prefill(definition.packing) || compact_draft) {
          check(program_slot.io.add_fixed(phase.draft_hidden_select_port, io.draft_hidden_select.read()));
          if (draft_lag(role)) check(program_slot.io.add_fixed(phase.draft_token_select_port, io.draft_token_select.read()));
        }
        if (compact_draft) check(program_slot.io.add_fixed(phase.draft_last_ids_port, io.draft_last_ids.read()));
        if (drafts_token(role)) io.draft_id = take(program_slot.io.add_readback(phase.draft_id_port, execution, allocation_budget));
      }
      if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
        if (const auto *output = phase.probe_facts[i]; output && output->caller_owned())
          check(program_slot.io.add_fixed_output(output->value, io.probe_outputs[i].view()));
      check(program_slot.io.seal());
      return hyle::Status::Ok();
    }));
    if (state_storage_prepared) prepare_transaction_layouts(phase);
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
              << " slot_count=" << phase.slots.size()
              << " slot0_uses=" << phase.slots[0].planned_uses
              << " slot0=" << (phase.slots[0].replay_mode == g::ReplayMode::kCommands ? "commands" : "prefer-capture") << '\n';
    std::cerr << "compiled T=" << tokens << " steps=" << phase.child->compiled.report.step_count()
              << " workspace/slot=" << phase.child->compiled.executable.workspace_bytes()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " child_prepare_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    input_scratch.prepare(phase);
    auto *address = result.get(); cache.emplace(ExecutionShape{requests, tokens, role, completion, sampling_index, imported_sources, packing}, std::move(result)); return *address;
  }

  // An engine with a known ordered request can explicitly prepare its actual
  // carriers before admission/ready, preserving the original cold boundary.
  // Candidate prefix suffixes remain unmaterialized until a hit selects one.
  void prepare_execution(std::span<const std::size_t> calls,
                         std::size_t requests = 1, GraphRole role = GraphRole::kText,
                          TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0, RowPacking packing = RowPacking::kRectangular) {
    const bool packed_prefill = packs_prefill(packing);
    if (carrier_borrowed || active_step)
      throw std::runtime_error("cannot cold-prepare carriers during an active execution loan");
    for (const auto tokens : calls) {
      // Explicit idle warmup can cover a suffix not present in the initial
      // schedule. Use the same bounded admission and original family owners
      // as execution, before borrowing a Session or accepting native work.
      if (!packed_prefill && state_storage_prepared && (role == GraphRole::kText || is_mtp_prefill(role)) &&
          !find_child(requests, tokens, role, false, completion, sampling_index))
        admit_exact_shape(tokens, requests, role, completion, sampling_index);
      if (!layer0_probe.empty()) prepare_child(tokens, false, requests, role, completion, sampling_index);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), requests, role, completion, sampling_index, false, packing);
      initialize_slot(phase, slot_index);
    }
  }

  void prepare_prefill_execution(std::span<const std::size_t> capacities,
      TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    prepare_execution(capacities, requests, role, completion, sampling_index, RowPacking::kPrefill);
  }

  void prepare_relay_execution(std::size_t requests = 1, std::size_t sampling_index = 0,
      GraphRole role = GraphRole::kText) {
    if (active_step || carrier_borrowed) throw std::runtime_error("relay carrier preparation requires an idle model");
    auto &phase = prepare_child(drafts_token(role) ? 2 : 1, false, requests, role,
        TextCompletion::kEmitNext, sampling_index, true);
    check(phase.slots.initialize_all_with([&] { return slot_initialization(phase); }));
  }

  // Prepare the existing cold plan, not every possible context length. This
  // creates carriers only: no Session, state transition, sampling or submit.
  // Native capture still needs actual bindings and remains a measured cost.
  std::size_t prepare_planned_execution() {
    if (active_step || carrier_borrowed)
      throw std::runtime_error("cannot prepare planned shapes during an active episode");
    std::size_t count = 0;
    for (const auto &[shape, uses] : shape_plan.entries()) {
      if (std::ranges::none_of(uses, [](auto planned_uses) { return planned_uses != 0; })) continue;
      const std::array rows{shape.tokens};
      prepare_execution(rows, shape.requests, shape.role, shape.completion, shape.sampling_index, shape.packing);
      ++count;
    }
    return count;
  }

  void detach_episode() noexcept {
    if (!active_step) return;
    if (successor_step && successor_step->execution.cleanup_pending) {
      const auto cleanup = abort_successor_episode();
      inference::record_retirement_status(active_step->execution.terminal, cleanup.status);
      if (!cleanup.retired) {
        return; // Retain both owners for the original retirement retry.
      }
    }
    // Only a caller-requested, actually retired shared cancellation is local
    // to this episode. An unchanged surviving Session remains runnable; do not
    // poison its later retirement with a peer's normal Graph cancellation.
    // Preserve native and all other failures, and the cancelled owner's fact.
    const bool cancelled_episode = active_step->cancelled &&
        active_step->execution.terminal.IsCancelled() && !active_step->execution.terminal.has_native_error();
    for (auto &member : active_step->members) {
      if (!cancelled_episode || member.session->stopping)
        inference::record_retirement_status(member.session->retirement_status, active_step->execution.terminal);
      member.session->pending = successor_step && std::ranges::any_of(successor_step->members,
          [&](const auto &next) { return next.session == member.session; }) ? successor_step.get() : nullptr;
    }
    active_step = std::move(successor_step);
    if (active_step) for (std::size_t j = 0; j < active_step->members.size(); ++j) {
      auto &member = active_step->members[j];
      const auto *source = active_step->execution.incoming ? active_step->execution.incoming->owner.get() : nullptr;
      const auto *boundary = source ? &source->boundaries[j] : nullptr;
      const bool full_boundary = !boundary || (member.session->head && member.session->head->length == boundary->length &&
          (!source->mtp_full_candidate || (member.session->head->draft &&
              member.session->head->draft->frontier == boundary->draft_frontier &&
              member.session->head->draft->lag == 1 && !inference::request_token_stopped(
                  member.session->head->next_id, model_eos_tokens, member.stop_policy))));
      if (member.session->head.get() == member.base && member.session->head.epoch() == member.base_epoch && full_boundary)
        member.first_input = member.session->head->next_id;
      else active_step->cancelled = true;
    }
  }

  inference::RetirementObservation try_retire_episode() noexcept {
    if (!active_step) return {true, {}};
    if (successor_step) {
      inference::record_retirement_status(successor_step->execution.terminal, execution.collect_retired());
      const auto retired = successor_step->try_abort();
      if (!retired) return {false, retired.error()};
      if (!*retired) return {false, successor_step->execution.terminal};
      successor_step.reset();
    }
    // Dropping the public Completion does not remove original native work.
    // Drive its owning environment before checking the unchanged Graph/page
    // retirement gates; an observation error alone never proves retirement.
    inference::record_retirement_status(active_step->execution.terminal, execution.collect_retired());
    const auto retired = active_step->try_abort();
    if (!retired) {
      inference::record_retirement_status(active_step->execution.terminal, retired.error());
      return {false, active_step->execution.terminal};
    }
    const auto terminal = active_step->execution.terminal;
    if (*retired) detach_episode();
    return {*retired, terminal};
  }

  inference::RetirementObservation abort_successor_episode() noexcept {
    if (!successor_step) return {true, {}};
    if (!successor_step->submitted && successor_step->execution.run && active_step &&
        active_step->execution.run && active_step->execution.run->in_flight()) {
      successor_step->cancelled = true;
      successor_step->execution.cleanup_pending = true;
      return {false, {}};
    }
    if (successor_step->execution.run && successor_step->phase) {
      (void)successor_step->execution.retire(execution);
    }
    const auto retired = successor_step->try_abort();
    if (!retired) return {false, retired.error()};
    const auto status = successor_step->execution.terminal;
    if (*retired) successor_step.reset();
    return {*retired, status};
  }

  inference::RetirementObservation abort_episode() noexcept {
    discard_decode_ahead();
    if (!active_step) return {true, {}};
    if (successor_step && successor_step->execution.run && successor_step->phase) {
      // Capture destruction for an unsubmitted B can synchronize its native
      // context. Keep the original predecessor gate before even draining B.
      if (!successor_step->submitted && active_step->execution.run && active_step->execution.run->in_flight()) {
        successor_step->cancelled = true;
        successor_step->execution.cleanup_pending = true;
      } else {
        (void)successor_step->execution.retire(execution);
      }
    }
    // Nonblocking collection cannot drain a partially accepted Vulkan prefix.
    // Hand off once to the original blocking owner, then check the unchanged
    // Graph/page gates. Unknown native failure keeps this candidate, Session
    // pins and admission intact for an explicit retry; it must not spin.
    if (active_step->execution.run && active_step->phase) {
      (void)active_step->execution.retire(execution);
    }
    // Episode retirement alone does not release a Session's published head.
    return try_retire_episode();
  }

  // Cohort selection and physical capacity belong to the caller. A batch
  // owns one original Invocation, IO loan, Completion and publication transaction.
  void prepare_draft_append(StepState &step, GraphRole role) {
    auto &phase = *step.phase;
    auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
    const auto width = step.members.size();
    const bool relay = bool(step.execution.incoming);
    const bool compact_draft = packs_draft(phase.family->packing);
    const auto physical_rows = cache::multiply(width, is_mtp_prefill(role)
        ? cache::add(phase.query_rows, draft_lag(role)) : draft_lag(role) + 1);
    auto &positions = input_scratch.draft_positions, &catchups = input_scratch.catchups;
    positions.resize(physical_rows); catchups.clear();
    auto &row_inputs = input_scratch.row_inputs;
    auto &offsets = input_scratch.query_indptr;
    auto &last_rows = input_scratch.last_rows;
    row_inputs.clear(); offsets.resize(width + 1); last_rows.resize(width);
    for (const auto &member : step.members) row_inputs.push_back({member.base->draft->frontier,
        compact_draft ? member.base->draft->lag + 1 : is_mtp_prefill(role)
            ? cache::add(member.input_tokens, draft_lag(role)) : draft_lag(role) + 1});
    (void)inference::pack_sequence_rows(row_inputs, physical_rows, offsets, last_rows, positions);
    if (compact_draft) {
      if (io.draft_hidden_indices.size() != physical_rows || io.draft_token_indices.size() != physical_rows)
        throw std::logic_error("MTP draft selectors differ from their cold physical carrier");
      std::ranges::fill(io.draft_hidden_indices, 0); std::ranges::fill(io.draft_token_indices, 0);
      for (std::size_t j = 0; j < width; ++j) {
        const auto lag = step.members[j].base->draft->lag;
        const auto begin = static_cast<std::size_t>(offsets[j]);
        if (lag > 1 || offsets[j + 1] <= offsets[j] ||
            last_rows[j] != offsets[j + 1] - 1 || row_inputs[j].rows != lag + 1)
          throw std::logic_error("MTP draft selection differs from its actual member rows");
        if (lag) {
          io.draft_hidden_indices[begin] = static_cast<std::int32_t>(width + j);
          io.draft_token_indices[begin] = static_cast<std::int32_t>(j);
        }
        io.draft_hidden_indices[begin + lag] = static_cast<std::int32_t>(j);
        io.draft_token_indices[begin + lag] = static_cast<std::int32_t>(width + j);
      }
    } else if (packs_prefill(phase.family->packing)) {
      const auto target_rows = cache::multiply(width, phase.query_rows);
      const auto prefix_rows = cache::multiply(width, draft_lag(role) + 1);
      const auto hidden_rows = cache::add(prefix_rows, target_rows);
      const auto token_rows = cache::add(width, target_rows);
      if (hidden_rows > static_cast<std::size_t>(INT32_MAX) || token_rows > static_cast<std::size_t>(INT32_MAX) ||
          io.draft_hidden_indices.size() != physical_rows ||
          (draft_lag(role) && io.draft_token_indices.size() != physical_rows))
        throw std::out_of_range("MTP compact selection exceeds its original I32 carrier");
      for (std::size_t j = 0; j < width; ++j) {
        const auto &member = step.members[j];
        if (!member.input_tokens || member.input_row_begin > target_rows ||
            member.input_tokens > target_rows - member.input_row_begin ||
            row_inputs[j].rows != cache::add(member.input_tokens, draft_lag(role)))
          throw std::logic_error("MTP compact selection differs from the target and draft row facts");
      }
      std::ranges::fill(io.draft_hidden_indices, 0);
      std::ranges::fill(io.draft_token_indices, 0);
      for (std::size_t j = 0; j < width; ++j) {
        const auto &member = step.members[j];
        const auto target_begin = member.input_row_begin;
        const auto draft_begin = static_cast<std::size_t>(offsets[j]);
        auto destination = draft_begin;
        if (draft_lag(role)) {
          io.draft_hidden_indices[destination++] = static_cast<std::int32_t>(width + j);
          io.draft_token_indices[draft_begin] = static_cast<std::int32_t>(j);
          for (std::size_t row = 0; row < member.input_tokens; ++row)
            io.draft_token_indices[draft_begin + 1 + row] = static_cast<std::int32_t>(width + target_begin + row);
        }
        io.draft_hidden_indices[destination] = static_cast<std::int32_t>(j);
        for (std::size_t row = 1; row < member.input_tokens; ++row)
          io.draft_hidden_indices[destination + row] = static_cast<std::int32_t>(prefix_rows + target_begin + row - 1);
      }
    }
    const auto begin_member = [&](std::size_t j) {
      auto &member = step.members[j];
      member.next->draft.emplace(); auto &next = *member.next->draft;
      if (!next.paged.frames.try_resize(2)) throw std::bad_alloc{};
      next.frontier = member.base->draft->frontier + row_inputs[j].rows;
    };
    const auto member_at = [&](std::size_t j) noexcept -> inference::PagedAppendMember {
      auto &member = step.members[j]; auto &session = *member.session;
      return {member.base->draft->paged, member.next->draft ? &member.next->draft->paged : nullptr,
          member.draft_layout, member.base->draft->frontier, row_inputs[j].rows,
          relay ? std::span<c::PagedPageAppender>{} : std::span(session.draft_tails),
          !relay && session.draft_tail_epoch == member.base_epoch};
    };
    const auto invalidate_tail = [&](std::size_t j) noexcept {
      if (!relay) step.members[j].session->draft_tail_epoch = 0;
    };
    step.draft_layout = take(inference::prepare_paged_append(attention_schema, width,
        inference::PagedAppendBanks{draft_inventory,
            std::span(step.execution.dirty.data() + draft_bank, 2),
            std::span(step.execution.tails.data() + draft_bank, 2),
            std::span(step.execution.appends.data() + draft_bank, 2)},
        input_scratch.paged_appends, member_at,
        {.capacity_rows = capacity,
         .incoming = relay ? std::span(step.execution.incoming->owner->pages).subspan(draft_bank, 2)
             : std::span<g::batch::detail::PagedAppendSource>{},
         .pending_predecessor = bool(step.execution.predecessor_episode),
         .append_options = {.resource_budget = cache_budget}},
        invalidate_tail, begin_member));
    if (draft_lag(role) && !relay) for (const auto &member : step.members) {
      const auto &old = *member.base->draft;
      catchups.push_back(compact_draft && !old.lag ? member.first_input : old.catchup_token);
    }
    io.draft_positions.write<std::int32_t>(positions);
    if (draft_lag(role) && !relay) io.catchup_token.write<std::int32_t>(catchups);
    if (packs_prefill(phase.family->packing) || compact_draft) {
      io.draft_hidden_select.write<std::int32_t>(io.draft_hidden_indices);
      if (draft_lag(role)) io.draft_token_select.write<std::int32_t>(io.draft_token_indices);
    }
    if (compact_draft) io.draft_last_ids.write<std::int32_t>(last_rows);
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
            member.next->paged.pages.size() > prepared.page_stride)
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
        for (std::size_t page = 0; page < member.next->paged.pages.size(); ++page)
          prepared.candidate_ids[index * prepared.page_stride + page] =
              static_cast<std::int32_t>(member.next->paged.pages[page].page());
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
            owner->length() > saved.maximum_length || owner->head->paged.pages.size() > prepared.page_stride) break;
        bool same_pages = true;
        for (std::size_t page = 0; page < owner->head->paged.pages.size(); ++page)
          if (owner->head->paged.pages[page].page() != static_cast<std::uint32_t>(
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
      for (std::size_t page = 0; page < owner.head->paged.pages.size(); ++page)
        prepared.old_ids.push_back(prepared.candidate_ids[index * prepared.page_stride + page]);
      prepared.old_indptr.push_back(static_cast<std::int32_t>(prepared.old_ids.size()));
      prepared.old_last.push_back(static_cast<std::int32_t>((owner.length() - 1) % cache::Plan::page_rows + 1));
    }
    for (const auto index : prepared.selected) prepared.members[index] = {};
    prepared.selected.clear(); prepared.refresh(); ++cpu_ahead_hits;
    return true;
  }

  void prepare_prefill_evaluations(std::span<const EvaluationRequest> requests,
      inference::PreparationObservation *preparation = nullptr, GraphRole role = GraphRole::kText,
      TextCompletion completion = TextCompletion::kEmitNext) {
    std::size_t maximum = 0;
    for (const auto &request : requests) maximum = (std::max)(maximum, request.ids.size());
    const auto physical = !requests.empty() && requests[0].session &&
        (role == GraphRole::kText || is_mtp_prefill(role))
        ? prefill_capacity(maximum, completion, requests[0].session->sampling_configuration().index, requests.size(), role) : 0;
    prepare_evaluations(requests, preparation, role, completion, {}, 0, physical,
        physical ? RowPacking::kPrefill : RowPacking::kRectangular);
  }

  void prepare_verification_evaluations(std::span<const EvaluationRequest> requests,
      std::size_t physical_rows, inference::PreparationObservation *preparation = nullptr) {
    prepare_evaluations(requests, preparation, GraphRole::kVerify, TextCompletion::kEmitNext,
        {}, 0, physical_rows, RowPacking::kVerify);
  }

  void prepare_evaluations(std::span<const EvaluationRequest> requests,
                           inference::PreparationObservation *preparation = nullptr,
                           GraphRole role = GraphRole::kText,
                           TextCompletion completion = TextCompletion::kEmitNext,
                           std::shared_ptr<StepSources> incoming = {}, std::size_t physical_slot = 0,
                            std::size_t physical_rows = 0, RowPacking packing = RowPacking::kRectangular) {
    const auto predecessor = incoming && active_step ? active_step : std::shared_ptr<StepState>{};
    const bool relay = bool(incoming);
    const bool mtp_relay = relay && incoming->mtp_full_candidate;
    if (!supports_completion(role, completion))
      throw std::runtime_error("state-only evaluation requires Text or teacher-forced MTP input");
    if (requests.empty() || requests.size() > cache_plan.active_sequences)
      throw std::runtime_error("batch width exceeds the configured active sequence capacity");
    if (!relay && (carrier_borrowed || active_step))
      throw std::runtime_error("the original execution carrier has not retired");
    const bool ordinary_relay = relay && !mtp_relay && !cache_plan.draft_cache && role == GraphRole::kText;
    const bool complete_mtp_relay = mtp_relay && cache_plan.draft_cache && requests.size() == 1 &&
        role == GraphRole::kMtpCatchupRound && (!predecessor || (predecessor->phase &&
            drafts_token(predecessor->phase->family->role) && predecessor->input_tokens == 2)) &&
        incoming->extra_outputs.size() == 1 && incoming->extra_outputs[0].valid() &&
        incoming->successor_states.size() == geometry.recurrent_layers() * 2;
    if (relay && (successor_step || scheduler || cache_plan.in_flight_batches < 2 ||
        (!ordinary_relay && !complete_mtp_relay) || completion != TextCompletion::kEmitNext || physical_slot >= execution_slot_count ||
        incoming->boundaries.size() != requests.size() || !incoming->submission.valid() || !incoming->token.valid() ||
        incoming->states.size() != geometry.recurrent_layers() * 2 + (mtp_relay ? 1u : 0u) ||
        incoming->pages.size() != draft_bank + (mtp_relay ? 2u : 0u) ||
        (predecessor && (predecessor->execution.sources != incoming || predecessor->cancelled || !predecessor->submitted ||
            predecessor->members.size() != requests.size()))))
      throw std::runtime_error("decode successor requires one exact complete predecessor and independent carriers");
    if (!requests.front().session || requests.front().session->model_ != this)
      throw std::runtime_error("batch session belongs to another model");
    if (!requests.front().session->head)
      throw std::runtime_error("batch session has no live committed head");
    const auto &prepared_sampling = requests.front().session->sampling_configuration();
    const auto sampling_index = prepared_sampling.index;
    const auto &selected_sampler = prepared_sampling.configuration;
    if (mtp_relay && (selected_sampler.enabled || !selected_sampler.modifiers.empty()))
      throw std::runtime_error("MTP full-candidate successor requires its original greedy sampling configuration");
    const auto supplied_tokens = relay ? std::size_t{1} : requests.front().ids.size();
    const bool packed_prefill = packs_prefill(packing), packed_verify = packs_verify(packing);
    const bool packed_target = packs_target(packing);
    if (packed_target != (physical_rows != 0) || (packed_verify && role != GraphRole::kVerify))
      throw std::runtime_error("target row capacity differs from its declared packing");
    // Mixed K1 members share a fixed two-row draft carrier, but keep their
    // actual one/two-row append and original mathematical role in the ledger.
    const auto first_lag = drafts_token(role) && requests.size() > 1 && requests.front().session->head->draft
        ? requests.front().session->head->draft->lag : 0;
    const bool compact_draft = !relay && !packed_prefill && requests.size() > 1 && drafts_token(role) &&
        std::ranges::any_of(requests, [&](const auto &request) {
          return request.session && request.session->head && request.session->head->draft &&
              request.session->head->draft->lag != first_lag;
        });
    if (compact_draft && first_lag != draft_lag(role))
      throw std::runtime_error("MTP role differs from its first member snapshot");
    if (compact_draft) role = GraphRole::kMtpCatchupRound;
    if (compact_draft) packing = RowPacking::kDraft;
    auto tokens = drafts_token(role) ? std::size_t{2} : supplied_tokens;
    if (packed_target) for (const auto &request : requests) tokens = (std::max)(tokens, request.ids.size());
    const auto physical_tokens = packed_target ? physical_rows : tokens;
    if (compact_draft && (completion != TextCompletion::kEmitNext ||
        !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, packing)))
      throw std::runtime_error("mixed MTP draft rows require their original cold compact family");
    if (packed_prefill && (relay || (role != GraphRole::kText && !is_mtp_prefill(role)) ||
        physical_tokens < tokens || !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, RowPacking::kPrefill)))
      throw std::runtime_error("prefill extent is outside its original cold packed family");
    if (packed_verify && (relay || physical_tokens < tokens || completion != TextCompletion::kEmitNext ||
        !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, packing)))
      throw std::runtime_error("verification extent is outside its original cold prefix capacity");
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
      if (&session.sampling_configuration() != &prepared_sampling) throw std::runtime_error("batch requires one cold sampling configuration");
      if (!request.conditional_logits.empty()) {
        const auto rows = request.conditional_logits.rows();
        auto status = role == GraphRole::kVerify && completion == TextCompletion::kEmitNext
            ? prefix_logits_inputs_metadata_status(selected_sampler, request.logits, rows, request.ids.size(), vocabulary)
            : hyle::Status::InvalidArgument("conditional evaluation logits require an emitting Verify episode");
        if (status.ok() && (request.ids.empty() || !prefix_sampling_programs.contains(
            {request.ids.size() - 1, sampling_index, true})))
          status = hyle::Status::InvalidArgument("conditional verification width was not cold prepared");
        check_request_logits_metadata(std::move(status), request.request_key);
        validate_prefix_logits_input_leases(selected_sampler, request.logits, rows);
      } else if (completion == TextCompletion::kEmitNext)
        validate_request_logits_inputs(selected_sampler, request.logits, vocabulary, request.request_key);
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
      const auto member_tokens = packed_target ? request.ids.size() : tokens;
      if (!member_tokens || (!packed_target && !relay && request.ids.size() != supplied_tokens) ||
          member_tokens > capacity - base->length)
        throw std::runtime_error("batch requires valid nonempty chunks within each remaining context");
      validate_mtp_snapshot(*session.head);
      if (has_mtp(role)) {
        const bool lag_matches = mtp_relay ? base->draft && base->draft->frontier == incoming->boundaries[j].draft_frontier &&
            base->length == incoming->boundaries[j].length && base->length - base->draft->frontier == 1 :
            compact_draft ? session.head->draft->lag <= 1 : session.head->draft->lag == draft_lag(role);
        if (!lag_matches ||
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
    if (!packed_prefill && (role == GraphRole::kText || is_mtp_prefill(role)) &&
        !find_child(requests.size(), tokens, role, false, completion, sampling_index)) {
      admit_exact_shape(tokens, requests.size(), role, completion, sampling_index);
      shape_admission_seconds = std::chrono::duration<double>(Clock::now() - step_start).count();
    }
    const bool ahead = !relay && !packed_target && take_decode_ahead(requests, role, completion, sampling_index, tokens);
    auto candidate = std::make_shared<StepState>(); candidate->execution.configure(draft_bank + 2);
    if (!candidate->members.try_reserve(requests.size())) throw std::bad_alloc{};
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      candidate->members.push_back({.session = request.session, .owner_state = &request.session->head,
          .base = predecessor ? predecessor->members[j].next.get() : request.session->head.get(),
          .base_epoch = request.session->head.epoch() + (predecessor ? 1u : 0u),
          .stop_policy = request.stop_policy,
          .logits = completion == TextCompletion::kEmitNext ? request.logits : LogitsInputs{},
          .conditional_logits = request.conditional_logits,
          .input_tokens = packed_target ? request.ids.size() : tokens});
      candidate->members.back().first_input = relay ? (predecessor ? -1 : request.session->head->next_id) : request.ids.front();
      if (role == GraphRole::kVerify)
        candidate->members.back().verification_inputs.assign(request.ids.begin(), request.ids.end());
    }
    if (relay) {
      candidate->execution.incoming = std::make_unique<IncomingStepSources>();
      candidate->execution.incoming->owner = std::move(incoming);
    }
    auto &step = *candidate;
    if (predecessor) successor_step = candidate; else active_step = candidate;
    ++carrier_borrowed; step.execution.candidate_admission = &carrier_borrowed;
    if (predecessor) step.execution.predecessor_episode = {predecessor, &predecessor->execution};
    step.step_start = step_start; step.input_tokens = tokens;
    for (auto &member : step.members) {
      member.input_row_begin = step.active_input_rows;
      step.active_input_rows = cache::add(step.active_input_rows, member.input_tokens);
    }
    step.shape_admission_seconds = shape_admission_seconds;
    if (!predecessor) for (auto &member : step.members) member.session->pending = &step;
    try {
      const auto width = requests.size(), slot = relay ? physical_slot : slot_index;
      if (!layer0_probe.empty()) prepare_child(tokens, false, width, role, completion, sampling_index);
      if (predecessor) {
        const auto found = phases.find({width, tokens, role, completion, sampling_index, true});
        if (found == phases.end() || !found->second->slots[slot].carrier.initialized())
          throw std::runtime_error("successor carrier must be cold-prepared before its predecessor submission");
      }
      auto &phase = prepare_child(physical_tokens, !layer0_probe.empty(), width, role, completion, sampling_index, relay, packing);
      auto &carrier = phase.slots[slot].carrier;
      auto &program_slot = phase.slots[slot]; auto &io = program_slot.data;
      // A rejected successor never claims or clears its predecessor's run.
      if (predecessor && predecessor->phase == &phase && predecessor->state_slot == slot)
        throw std::runtime_error("successor cannot reuse its live predecessor carrier");
      initialize_slot(phase, slot);
      auto &run = *take(carrier.borrow());
      step.phase = &phase; step.execution.run = &run; step.execution.carrier = &carrier; step.state_slot = slot;
      step.execution.state_transactions = &phase.slots[slot].transactions;
      step.execution.nodes.assign(phase.append_nodes.begin(), phase.append_nodes.end());
      if (packed_target) {
        // The prior step has really retired. Release this exact mirror alias
        // before batch::prepare tries to refresh its original metadata writer.
        const std::array offset_input{phase.sequence_offsets_port};
        check(run.unbind(offset_input));
      }
      auto &packed_ids = input_scratch.packed_ids, &positions = input_scratch.positions;
      packed_ids.clear(); positions.clear();
      for (std::size_t j = 0; j < width; ++j) {
        const auto &request = requests[j];
        if (!relay && width != 1) packed_ids.insert(packed_ids.end(), request.ids.begin(), request.ids.end());
        if (!ahead && !packed_target) for (std::size_t row = 0; row < step.members[j].input_tokens; ++row)
          positions.push_back(static_cast<std::int32_t>(step.members[j].base->length + row));
      }
      for (std::size_t j = 0; j < width; ++j) {
        if (!phase.logits_bias_ports.empty()) check(run.bind(phase.logits_bias_ports[j], requests[j].logits.bias.description, requests[j].logits.bias.lease));
        if (!phase.logits_allowed_ports.empty()) check(run.bind(phase.logits_allowed_ports[j], requests[j].logits.allowed.description, requests[j].logits.allowed.lease));
        if (!phase.logits_counts_ports.empty()) check(run.bind(phase.logits_counts_ports[j], requests[j].logits.counts.description, requests[j].logits.counts.lease));
      }
      if (packed_target) {
        if (width == 1) packed_ids.assign(requests[0].ids.begin(), requests[0].ids.end());
        const auto carrier_rows = cache::multiply(width, physical_tokens);
        packed_ids.resize(carrier_rows, 0); // Only the inactive carrier suffix is scratch.
        positions.resize(carrier_rows);
        auto &row_inputs = input_scratch.row_inputs;
        auto &offsets = input_scratch.query_indptr;
        row_inputs.clear(); offsets.resize(width + 1);
        for (const auto &member : step.members) row_inputs.push_back({member.base->length, member.input_tokens});
        const auto row_facts = inference::pack_sequence_rows(row_inputs, carrier_rows, offsets, io.last_id_values, positions);
        if (row_facts.active_rows != step.active_input_rows || row_facts.max_member_rows != step.input_tokens)
          throw std::logic_error("packed target rows differ from the admitted episode");
        const auto history_rows = static_cast<std::size_t>(geometry.conv_kernel - 1);
        const auto history_count = cache::multiply(width, history_rows);
        if (packed_verify) {
          (void)inference::pack_sequence_prefixes(offsets, phase.prefix_rows, history_rows,
              io.prefix_id_values, io.history_select_values);
          io.prefix_ids.write<std::int32_t>(io.prefix_id_values);
        } else for (std::size_t j = 0; j < width; ++j) {
          const auto begin = step.members[j].input_row_begin;
          const auto length = step.members[j].input_tokens;
          for (std::size_t row = 0; row < history_rows; ++row) {
            const auto tail = length + row;
            const auto source = tail < history_rows ? j * history_rows + tail
                : history_count + begin + tail - history_rows;
            io.history_select_values[j * history_rows + row] = static_cast<std::int32_t>(source);
          }
        }
        if (packed_prefill) io.last_ids.write<std::int32_t>(io.last_id_values);
        io.history_select.write<std::int32_t>(io.history_select_values);
      }
      if (!relay) io.tokens.write<std::int32_t>(packed_target || width != 1
          ? std::span<const std::int32_t>(packed_ids) : requests[0].ids);
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
      const auto begin_member = [&](std::size_t j) {
        // The common target cohort already reserved its own tail metadata.
        // Keep the remaining original banks before the first model candidate.
        if (!j) for (auto &bank : std::span(step.execution.tails.data(), step.execution.tails.size()).subspan(draft_bank))
          if (!bank.try_reserve(width)) throw std::bad_alloc{};
        auto &member = step.members[j]; auto &session = *member.session;
        member.next = reserve_state(session.head);
        if (!member.next_tails.try_resize(draft_bank)) throw std::bad_alloc{};
        member.next->length = member.base->length + member.input_tokens;
        member.next->sampling = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify
            ? member.base->sampling.advanced(drafts_token(role) ? 3 : 1) : member.base->sampling;
      };
      const auto member_at = [&](std::size_t j) noexcept -> inference::PagedAppendMember {
        auto &member = step.members[j]; auto &session = *member.session;
        return {member.base->paged, member.next ? &member.next->paged : nullptr, member.layout,
            member.base->length, member.input_tokens,
            relay ? std::span<c::PagedPageAppender>{} : std::span(session.tails.data(), session.tails.size()),
            !relay && session.tail_epoch == member.base_epoch};
      };
      const auto invalidate_tail = [&](std::size_t j) noexcept {
        if (!relay) step.members[j].session->tail_epoch = 0;
      };
      const c::HostBatchLayout previous_layout = ahead ? c::HostBatchLayout{
          {decode_ahead->old_query.data(), decode_ahead->old_query.size()},
          {decode_ahead->old_indptr.data(), decode_ahead->old_indptr.size()},
          {decode_ahead->old_ids.data(), decode_ahead->old_ids.size()},
          {decode_ahead->old_last.data(), decode_ahead->old_last.size()}} : c::HostBatchLayout{};
      io.attention_layout = take(inference::prepare_paged_append(attention_schema, width,
          inference::PagedAppendBanks{page_inventory,
              std::span(step.execution.dirty.data(), draft_bank),
              std::span(step.execution.tails.data(), draft_bank),
              std::span(step.execution.appends.data(), draft_bank)},
          input_scratch.paged_appends, member_at,
          {.capacity_rows = capacity,
           .incoming = relay ? std::span(step.execution.incoming->owner->pages).first(draft_bank)
               : std::span<g::batch::detail::PagedAppendSource>{},
           .pending_predecessor = bool(predecessor),
           .previous_layout = ahead && width != 1 ? &previous_layout : nullptr,
           .append_options = {.resource_budget = cache_budget}},
          invalidate_tail, begin_member));
      if (ahead) decode_ahead->clear_output(); // Schema copied the original borrowed integers.
      if (has_mtp(role)) prepare_draft_append(step, role);
      auto &transactions = phase.slots[slot].transactions;
      for (std::size_t index = 0; index < phase.state_banks.size(); ++index) {
        const auto &bank = phase.state_banks[index];
        auto state_sources = transactions.sources(index);
        const bool is_hidden = bank.kind == StateBankKind::kHidden;
        const bool recurrent = bank.kind == StateBankKind::kRecurrent;
        const auto bytes = transactions.writer_bytes(index);
        if (!relay) for (std::size_t j = 0; j < width; ++j) {
          if (is_hidden) {
            if (has_mtp(role)) {
              const bool chronological = width == 1 && role == GraphRole::kMtpCatchupRound && packing == RowPacking::kRectangular;
              state_sources[chronological ? width + j : j] = step.members[j].base->carry_hidden;
              if (draft_lag(role)) state_sources[chronological ? j : width + j] = compact_draft && !step.members[j].base->draft->lag
                  ? step.members[j].base->carry_hidden : step.members[j].base->prior_hidden;
            }
          } else {
            const auto &state = step.members[j].base->state[bank.layer];
            state_sources[j] = recurrent ? state.recurrent : state.history;
          }
        }
        for (auto &writer : transactions.writers(index))
          writer = is_hidden ? take(hidden_pool.allocate(bytes, cache::state_pool_alignment))
              : take(recurrent ? state_pools[bank.layer].acquire_recurrent(bytes)
                  : state_pools[bank.layer].acquire_history(bytes));
        check(transactions.stage(index));
      }
      if (preparation) { preparation->slot = slot; preparation->phase_slot_prior_uses = phase.slots[slot].preparation_uses; }
      const auto batch_start = Clock::now();
      const auto append_banks = draft_bank + (has_mtp(role) ? 2u : 0u);
      {
        hyle::util::ScopeGuard release_layouts{[&]() noexcept {
          for (auto &entry : io.batch_layouts) entry.layout = {};
        }};
        for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i) io.batch_layouts[i].layout = io.attention_layout;
        if (drafts_token(role)) io.batch_layouts.back().layout = step.draft_layout;
        check(step.execution.prepare_appends(phase.family->graph, io.batch_layouts, 0, append_banks,
            std::span<const V>(phase.append_cache_inputs).first(append_banks), {.host_workspace = execution.staging()}));
      }
      if (relay) {
        auto &input = *step.execution.incoming; auto &previous = *input.owner;
        input.bindings.reserve(1 + previous.states.size() + previous.pages.size() + (mtp_relay ? 1u : 0u));
        input.bindings.push_back({phase.token_port, &previous.token});
        if (mtp_relay) input.bindings.push_back({phase.catchup_token_port, &previous.extra_outputs[0]});
        const bool ready_states = previous.published_states.size() == previous.states.size() * width + (mtp_relay ? 1u : 0u) &&
            std::ranges::all_of(previous.published_states, [](const auto &lease) { return lease.valid(); });
        for (std::size_t bank = 0; bank < previous.states.size(); ++bank) {
          const auto &source = mtp_relay && bank < previous.successor_states.size()
              ? previous.successor_states[bank] : previous.states[bank];
          const auto count = mtp_relay && bank == previous.successor_states.size() ? 2u : width;
          input.bindings.push_back(take(source.reader_binding(run, phase.state_graph(),
              phase.state_sources[bank].store, ready_states ? std::span<const c::ReadLease>(previous.published_states).subspan(bank * width, count)
                  : std::span<const c::ReadLease>{})));
        }
        for (std::size_t bank = 0; bank < previous.pages.size(); ++bank)
          input.bindings.push_back(take(g::batch::detail::append_binding(run, phase.append_nodes[bank], previous.pages[bank])));
        for (auto value : phase.state_resource_ports) {
          const bool attached = std::ranges::any_of(input.bindings, [&](const auto &binding) {
            return binding.input == value || (binding.store && (binding.projection.capacity == value || binding.projection.metadata == value));
          });
          if (!attached) input.outside_inputs.push_back(value);
        }
      }
      check(step.execution.bind_consumers(phase.state_graph()));
      if (packed_target) {
        auto offsets = take(g::batch::query_offsets(run, phase.attention_nodes.front()));
        check(run.bind(phase.sequence_offsets_port, dense(Type::kInt32, {static_cast<std::int64_t>(width + 1)}), offsets));
      }
      const auto batch_end = Clock::now();
      check(step.execution.prepare_replay({.mode = phase.slots[slot].replay_mode, .host_wait = g::HostWaitPolicy::kAllow}));
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
      ++phase.slots[slot].preparation_uses; step.prepared = true;
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
      std::shared_ptr<StepSources> previous, std::size_t physical_slot, GraphRole role = GraphRole::kText) {
    prepare_evaluations(requests, nullptr, role, TextCompletion::kEmitNext, std::move(previous), physical_slot);
    return successor_step ? *successor_step : *active_step;
  }

  // Reserve typed metadata before accepting native work. The caller can retain
  // the same original views for a successor; no candidate head is exported and
  // the ordinary API remains on its existing direct Completion path.
  std::shared_ptr<StepSources> prepare_submission_sources(StepState &step) {
    const bool mtp = step.phase && drafts_token(step.phase->family->role) && step.members.size() == 1 &&
        step.phase->successor_token_port && step.phase->successor_catchup_port;
    if (!step.prepared || step.submitted || step.cancelled || step.execution.sources || !step.phase ||
        (!mtp && (step.phase->family->role != GraphRole::kText || cache_plan.draft_cache)) ||
        step.phase->family->completion != TextCompletion::kEmitNext || scheduler)
      throw std::runtime_error("submission views require one complete supported prepared episode");
    auto &phase = *step.phase;
    const auto storage = execution.storage_owner();
    if (!storage || phase.slots[step.state_slot].carrier.queues().size() != 1 ||
        !take(phase.slots[step.state_slot].io.readback(phase.slots[step.state_slot].data.next_id))->pool().valid())
      throw inference::EvaluationFailure(hyle::Status::Unimplemented(
          "submission views require the original owned shared Queue and full token writer"));
    auto sources = std::make_shared<StepSources>();
    sources->mtp_full_candidate = mtp;
    sources->boundaries.reserve(step.members.size());
    sources->published_states.resize(phase.state_sources.size() * step.members.size() + (mtp ? 1u : 0u));
    for (const auto &member : step.members) {
      if (member.base_epoch == UINT64_MAX) throw std::overflow_error("source candidate epoch exhausted");
      sources->boundaries.push_back({member.session, member.next.get(), member.base_epoch + 1,
          member.next->length, member.next->draft ? member.next->draft->frontier : 0});
    }
    check(sources->prepare(*step.execution.run, phase.state_sources,
        std::span<const g::NodeId>(step.execution.nodes.data(), draft_bank + (mtp ? 2u : 0u)), storage,
        execution_budget, mtp ? 1u : 0u));
    if (mtp) {
      sources->successor_states.reserve(geometry.recurrent_layers() * 2);
      for (std::size_t bank = 0; bank < geometry.recurrent_layers() * 2; ++bank)
        sources->successor_states.push_back(take(g::detail::StoreSource::prepare(*step.execution.run,
            phase.state_sources[bank].store, storage, g::detail::StoreDestinationRange{1, 1},
            {.resource_budget = execution_budget})));
    }
    step.execution.sources = sources;
    return sources;
  }

  std::shared_ptr<StepSources> prepare_submission_sources() {
    if (!active_step) throw std::runtime_error("submission views require a prepared episode");
    return prepare_submission_sources(*active_step);
  }

  void submit_evaluations(StepState &step, bool optional_sources = false) {
    if ((active_step.get() != &step && successor_step.get() != &step) || !step.prepared ||
        step.submitted || step.cancelled || !step.execution.run)
      throw std::runtime_error("batch is not prepared or was already submitted/cancelled");
    for (const auto &member : step.members)
      if (member.session->stopping || member.session->failed)
        throw std::runtime_error("batch participant stopped before submission");
    step.start = Clock::now(); step.submitted = true;
    check(step.execution.submit(resources_));
    if (step.execution.sources) {
      auto &phase = *step.phase; auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
      auto &sources = *step.execution.sources;
      bool captured = false;
      if (sources.mtp_full_candidate) {
        const std::array<inference::SubmissionViews::OutputCapture, 2> outputs{{
            {phase.successor_token_port, take(program_slot.io.readback(io.successor_token))->pool()},
            {phase.successor_catchup_port, take(program_slot.io.readback(io.successor_catchup))->pool()}}};
        captured = take(sources.capture_outputs(*step.execution.run, step.execution.completion, outputs,
            execution.storage_owner(), optional_sources));
        if (captured) for (auto &state : sources.successor_states) check(state.capture(sources.submission));
      } else captured = take(sources.capture(*step.execution.run, step.execution.completion,
          phase.id_port, execution.storage_owner(), take(program_slot.io.readback(io.next_id))->pool(), optional_sources));
      if (!captured) step.execution.sources.reset();
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

  // Private validators can outlive the target's failure. Read only their
  // original Invocation/retention facts; this failure-only check neither
  // starts another recovery nor treats a cleanup hint as proof of retirement.
  bool private_publications_retired() const noexcept {
    for (const auto &[key, program] : draft_programs) {
      (void)key;
      if (program->guard.execution.run || program->guard.execution.cleanup_pending ||
          !program->carrier.retired()) return false;
    }
    for (const auto &[key, program] : prefix_sampling_programs) {
      (void)key;
      if (!program->carrier.retired()) return false;
    }
    for (const auto &entry : resources_->sampling_configurations)
      if (entry->frontier && !entry->frontier->carrier.retired()) return false;
    return true;
  }

  inference::RetirementObservation retire_failed_publication(bool private_pending = false) noexcept {
    // The ledger is a call-scoped borrow, not owned by the Step's native pins.
    // Never retain a cleanup callback for a later Session/Model retirement.
    const auto failed_step = active_step;
    if (failed_step) for (auto &member : failed_step->members) member.commit_publication.reset();
    auto retired = abort_episode();
    retired.retired = retired.retired && !private_pending && private_publications_retired();
    if (failed_step) for (auto &member : failed_step->members) {
      if (retired.retired && member.commit_ledger)
        (void)member.commit_ledger->discard_attempt(member.commit_record);
      // An unretired Step may outlive this caller and its ledger. Later native
      // retirement must never revisit these synchronous publication borrows.
      member.commit_record = nullptr; member.commit_ledger = nullptr;
    }
    return retired;
  }

  template<class Select, class Complete, class Observe>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Complete complete, Observe observe) {
    if (!active_step || outputs.size() != active_step->members.size() || next_ids.size() != outputs.size() ||
        active_step->cancelled || !active_step->submitted || !active_step->execution.has_completion())
      throw std::runtime_error("finish requires its original complete noncancelled batch");
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      if (outputs[j].session != active_step->members[j].session || outputs[j].session->failed ||
          (outputs[j].session->stopping && !active_step->discard_stopped_members))
        throw std::runtime_error("finish participant does not match the original active batch");
      if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kNotPublished;
    }
    auto &step = *active_step; auto &phase = *step.phase; auto &run = *step.execution.run;
    auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
    const auto step_start = step.step_start, start = step.start;
    hyle::container::SmallVector<std::size_t, 2> member_input_rows;
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
      if (!member_input_rows.try_reserve(outputs.size())) throw std::bad_alloc{};
      for (const auto &member : step.members) member_input_rows.push_back(member.input_tokens);
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
          const auto member_role = packs_draft(phase.family->packing)
              ? (step.members[j].base->draft->lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : phase.family->role;
          if (record.role != member_role || record.completion != phase.family->completion || record.before != generation_boundary(*step.members[j].base) ||
              record.ids().size() != (record.generation ? 1u : step.members[j].input_tokens) ||
              record.ids().empty() || record.ids().front() != step.members[j].first_input)
            throw std::runtime_error("generation ledger entry differs from its actual episode");
          if (outputs[j].expected_episode && !outputs[j].expected_episode->committed)
            throw std::runtime_error("expected episode record is not committed");
          if (step.withdrawn(step.members[j])) {
            step.members[j].commit_ledger = nullptr; step.members[j].commit_record = nullptr;
            outputs[j].generation_ledger->discard_uncommitted();
          } else {
            auto &ledger = *outputs[j].generation_ledger;
            step.members[j].commit_publication.emplace(take(inference::prepare_record_commit(
                std::span(ledger.records), ledger.committed_count, record)));
          }
        } else if (outputs[j].expected_episode)
          throw std::runtime_error("expected episode record requires a fresh commit witness");
      }
      check(step.execution.wait_success());
      if (step.execution.sources) check(step.execution.sources->detach_outputs());
      // Trace storage must be pinned before take_append/unbind revoke Graph
      // output observation. Generated arrays keep their actual native owner.
      if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_tensors; })) {
        if (!state_only && !io.logits.caller_owned()) generated_logits = take(inference::retain_output(run, *phase.logits_facts, allocation_budget));
        if (!io.final_hidden.caller_owned()) generated_hidden = take(inference::retain_output(run, *phase.hidden_facts, allocation_budget));
      }
      if (drafts_token(phase.family->role)) {
        check(program_slot.io.read(execution, run, io.draft_id, std::as_writable_bytes(std::span(io.draft_ids))));
        for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].draft_token = io.draft_ids[j];
      }
      if (phase.sampling_valid_port && !state_only && phase.family->role != GraphRole::kVerify) {
        check(take(program_slot.io.readback(io.sampling_valid))->predicates(execution, run, io.sampling_valid_values, io.sampling_scratch));
        auto bad = std::ranges::find_if(io.sampling_valid_values, [](auto value) { return value != 1; });
        if (step.discard_stopped_members)
          while (bad != io.sampling_valid_values.end() &&
              step.withdrawn(step.members[static_cast<std::size_t>(bad - io.sampling_valid_values.begin())]))
            bad = std::find_if(bad + 1, io.sampling_valid_values.end(), [](auto value) { return value != 1; });
        if (bad != io.sampling_valid_values.end()) {
          std::optional<std::size_t> participant;
          // Everything before the first bad value is already proven 1.
          // Only the failure path inspects the remainder for a unique 0.
          bool unique = true;
          for (auto it = bad + 1; it != io.sampling_valid_values.end(); ++it)
            if (*it != 1 && !step.withdrawn(
                step.members[static_cast<std::size_t>(it - io.sampling_valid_values.begin())])) unique = false;
          if (*bad == 0 && unique)
            participant = static_cast<std::size_t>(bad - io.sampling_valid_values.begin());
          throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
              "sampling distribution or proposal is invalid; no session or RNG cursor was published"), participant);
        }
        if (phase.sampling_accept_port) {
          check(take(program_slot.io.readback(io.sampling_accept))->predicates(execution, run, io.sampling_accept_values, io.sampling_scratch));
          for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].sampled_accept = io.sampling_accept_values[j] != 0;
        }
      }
      // Selection observes only retired original Graph results. No head has
      // been published and all state/append guards still belong to this episode.
      if (verifies_prefix(phase.family->role)) {
        check(program_slot.io.read(execution, run, io.next_id, std::as_writable_bytes(std::span(io.prefix_candidates))));
        select(std::span<const std::int32_t>(io.prefix_candidates), step, next_ids);
      } else if (state_only) {
        std::ranges::fill(next_ids, -1);
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      } else {
        check(program_slot.io.read(execution, run, io.next_id, std::as_writable_bytes(next_ids)));
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      }
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        auto &member = step.members[j];
        // Retain every original physical owner for whole-cohort tail return,
        // even when no accepted prefix will be published for this member.
        member.verified_last_page = member.next->paged.pages.back().page();
        if (step.withdrawn(member)) { next_ids[j] = -1; continue; }
        if (!member.accepted_tokens || member.accepted_tokens > member.input_tokens ||
            (member.accepted_tokens != member.input_tokens && !verifies_prefix(phase.family->role)) ||
            (!state_only && (next_ids[j] < 0 || next_ids[j] >= vocabulary)))
          throw std::runtime_error("invalid accepted target boundary");
        if (member.accepted_tokens != member.input_tokens) {
          member.next->length = member.base->length + member.accepted_tokens;
          const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
          member.next->paged.pages.resize(page_count);
          auto &layout_rows = input_scratch.paged_appends.next_rows;
          layout_rows.reset();
          check(layout_rows.append(member.accepted_tokens, member.next->length, cache::Plan::page_rows,
              std::span<const hyle::inference::PagedPageRef>(member.next->paged.pages)));
          member.layout = take(layout_rows.member_version(attention_schema, 0));
        }
      }
      if (has_mtp(phase.family->role)) for (auto &member : step.members) {
        if (step.withdrawn(member)) continue;
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
      if (step.execution.sources) check(step.execution.sources->submission.detach());
      const auto append_banks = draft_bank + (has_mtp(phase.family->role) ? 2u : 0u);
      check(step.execution.append_transactions().subspan(0, append_banks).take(run));
      // Slot take proves actual retirement, not success; the original whole
      // Graph completion above remains the success witness for every bank.
      auto &transactions = phase.slots[step.state_slot].transactions;
      check(transactions.take_all(run));
      for (std::size_t bank_index = 0; bank_index < phase.state_banks.size(); ++bank_index) {
        const auto &bank = phase.state_banks[bank_index];
        auto writers = take(transactions.returned(bank_index));
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          if (step.withdrawn(member)) continue;
          if (bank.kind == StateBankKind::kHidden) {
            const auto count = verifies_prefix(phase.family->role) ? member.accepted_tokens : 1;
            const bool repair = phase.family->role == GraphRole::kVerify && cache_plan.draft_cache;
            if (repair) member.carry_prefix.reserve(count);
            for (std::size_t row = 0; row < count; ++row) {
              // Exactly one publication of each accepted canonical row. The
              // final head and draft repair share aliases of that same lease.
              const auto destination = j * phase.prefix_rows + row;
              auto ready = step.execution.sources && bank.source_index != SIZE_MAX
                  ? take(step.execution.sources->states[bank.source_index].publish(destination, writers[destination]))
                  : take(writers[destination].publish());
              if (step.execution.sources && step.execution.sources->mtp_full_candidate)
                step.execution.sources->published_states[bank.source_index + row] = ready;
              if (row + 1 == count) member.next->carry_hidden = ready;
              if (row + 2 == count) member.next->prior_hidden = ready;
              if (repair) member.carry_prefix.push_back(std::move(ready));
            }
            continue;
          }
          const auto prefix = verifies_prefix(phase.family->role) ? step.members[j].accepted_tokens - 1 : 0;
          const auto selected = j * phase.prefix_rows + prefix;
          auto ready = step.execution.sources ? take(step.execution.sources->states[bank.source_index].publish(
              selected, writers[selected]))
              : take(writers[selected].publish());
          if (step.execution.sources) step.execution.sources->published_states[bank.source_index * outputs.size() + j] = ready;
          auto &state = step.members[j].next->state[bank.layer];
          (bank.kind == StateBankKind::kRecurrent ? state.recurrent : state.history) = std::move(ready);
        }
        // Release old source guards before the joint head-commit recycle hook.
        // Unaccepted MTP candidates are discarded, never made visible.
        check(transactions.clear(bank_index));
      }
      for (auto &member : step.members)
        if (!step.withdrawn(member) && !member.target_updates.try_resize(page_inventory.banks.size()))
          throw std::bad_alloc{};
      const auto target_finish_at = [&](std::size_t j) noexcept -> inference::PagedAppendFinishMember {
        auto &member = step.members[j];
        return {member.base->paged, member.next->paged,
            {member.target_updates.data(), member.target_updates.size()}, member.layout,
            member.verified_last_page, static_cast<std::uint32_t>(member.next->length % cache::Plan::page_rows),
            {member.next_tails.data(), member.next_tails.size()}, !step.withdrawn(member),
            member.accepted_tokens == member.input_tokens};
      };
      check(inference::prepare_paged_append_commits(
          std::span(step.execution.appends.data(), draft_bank), outputs.size(), target_finish_at,
          step.execution.sources ? std::span<const g::batch::detail::PagedAppendSource>(step.execution.sources->pages).first(draft_bank)
              : std::span<const g::batch::detail::PagedAppendSource>{}));
      if (has_mtp(phase.family->role)) {
        const auto draft_finish_at = [&](std::size_t j) noexcept -> inference::PagedAppendFinishMember {
          auto &member = step.members[j]; auto &draft = *member.next->draft;
          return {member.base->draft->paged, draft.paged, member.draft_updates, member.draft_layout,
              draft.paged.pages.back().page(), static_cast<std::uint32_t>(draft.frontier % cache::Plan::page_rows),
              member.next_draft_tails, !step.withdrawn(member)};
        };
        check(inference::prepare_paged_append_commits(
            std::span(step.execution.appends.data() + draft_bank, 2), outputs.size(), draft_finish_at,
            step.execution.sources ? std::span<const g::batch::detail::PagedAppendSource>(step.execution.sources->pages).subspan(draft_bank, 2)
                : std::span<const g::batch::detail::PagedAppendSource>{}));
        check(inference::return_paged_append_tails(
            step.execution.append_transactions().subspan(draft_bank, 2), 2, outputs.size(),
            input_scratch.paged_appends, draft_finish_at));
      }
      // Caller-supplied completion is confined to this unpublished episode.
      // In particular, recursive proposals must repair their retained KV from
      // actual target hidden rows here, before ledger checks or any head swap.
      complete(step);
      if (cache_plan.draft_cache)
        for (const auto &member : step.members)
          if (!step.withdrawn(member)) validate_mtp_snapshot(*member.next, &member);
      for (std::size_t j = 0; j < outputs.size(); ++j) if (auto *record = step.members[j].commit_record) {
        const auto &member = step.members[j];
        record->requests = outputs.size(); record->member_index = j;
        record->accepted = member.accepted_tokens;
        record->draft_rows = member.next->draft
            ? member.next->draft->frontier - member.base->draft->frontier : 0;
        if (phase.family->role == GraphRole::kVerify)
          record->draft_rows = cache::add(record->draft_rows,
              member.input_tokens - 1 + (member.input_tokens > 1 ? member.base->draft->lag : 0) +
              member.discarded_draft_rows);
        record->after = generation_boundary(*member.next); record->after.next_id = next_ids[j];
        if (phase.family->role == GraphRole::kVerify)
          write_generation_tokens(record->accepted_inputs,
              std::span<const std::int32_t>(member.verification_inputs).first(member.accepted_tokens));
        else write_generation_tokens(record->accepted_inputs, {member.first_input, member.accepted_tokens == 2 ? member.draft_token : -1});
        if (record->generation) {
          if (phase.family->role == GraphRole::kVerify) write_generation_tokens(record->verification_inputs, member.verification_inputs);
          else if (drafts_token(phase.family->role)) write_generation_tokens(record->verification_inputs, {member.first_input, member.draft_token});
          else write_generation_tokens(record->verification_inputs, {member.first_input});
        }
        record->verified_rows = member.input_tokens;
        if (!record->generation || phase.family->role == GraphRole::kText) {
          record->draft = -1; record->target_first = next_ids[j]; record->target_second = -1;
          record->agreed = false; record->accepted_draft = false;
          record->stopped = !state_only && inference::request_token_stopped(
              next_ids[j], model_eos_tokens, member.stop_policy);
          if (state_only) { record->emitted.clear(); record->emitted_count = 0; }
          else { write_generation_tokens(record->emitted, {next_ids[j], -1}); record->emitted_count = 1; }
        }
        if (outputs[j].expected_episode && !record->same_result(*outputs[j].expected_episode))
          throw std::runtime_error("MTP replay natural decision or state boundary diverged from its recorded episode");
      }
      // Partial acceptance narrows Frame authority without rewinding the
      // verified physical tail; the next append retains its original COW path.
      check(inference::return_paged_append_tails(
          step.execution.append_transactions().subspan(0, draft_bank), draft_bank, step.members.size(),
          input_scratch.paged_appends, target_finish_at));
      check(step.execution.release_consumers(phase.state_resource_ports));
      for (const auto &member : step.members)
        if (member.session->head.get() != member.base || member.session->head.epoch() != member.base_epoch ||
            member.session->pending != &step ||
            (member.session->stopping && !step.discard_stopped_members))
          throw std::runtime_error("batch snapshot expected-base conflict");
      for (auto &member : step.members)
        if (!step.withdrawn(member)) member.preflight_frames();
      // Every delta owns its changed guards and every tail has transferred.
      // A successor still retains this original Step, but must not thereby
      // pin replaced input tails from its consumed returned append wrappers.
      for (auto &append : step.execution.appends) append = {};
      // All fallible native/export/metadata work is complete. The head swaps
      // below cannot throw. Stopped members discard their unpublished
      // candidate; every surviving member keeps this original successful work.
      for (std::size_t j = 0; j < step.members.size(); ++j) {
        auto &member = step.members[j];
        if (step.withdrawn(member)) {
          published_members[j] = 0;
          if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kCancelled;
          continue;
        }
        if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kPublished;
        member.next->next_id = next_ids[j];
        member.session->tails = std::move(member.next_tails);
        member.session->draft_tails = std::move(member.next_draft_tails);
        const auto publish = [&]() noexcept -> hyle::Status {
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
          return {};
        };
        // Mark each original record immediately after its own no-fail head
        // swap, before any subsequent reporting or observation can fail.
        if (member.commit_publication) {
          const auto status = member.commit_publication->publish(publish);
          HYLE_FATAL_ASSERT(status.ok());
          member.commit_publication.reset();
        } else {
          (void)publish();
        }
        member.session->tail_epoch = member.session->head.epoch();
        member.session->draft_tail_epoch = member.session->head.epoch();
      }
      if (step.execution.sources) step.execution.sources->release_publication_views();
      if (packs_prefill(phase.family->packing)) {
        ++packed_prefill_calls;
        packed_prefill_active_rows = cache::add(packed_prefill_active_rows, step.active_input_rows);
        packed_prefill_physical_rows = cache::add(packed_prefill_physical_rows,
            cache::multiply(phase.query_rows, phase.family->requests));
      }
      return_readback.admission = std::exchange(step.execution.candidate_admission, nullptr);
      detach_episode(); // Old states/transactions retire inside the host endpoint.
    } catch (inference::EvaluationFailure &failure) {
      // A nested private validator can still own native work independently of
      // this target episode. Its negative observation cannot be overwritten
      // by retiring only the target; retain it until its original owner drains.
      const bool private_pending = failure.retirement_observed() && !failure.retirement().retired;
      failure.observe_retirement(retire_failed_publication(private_pending));
      throw;
    } catch (...) {
      (void)retire_failed_publication(); throw;
    }
    const auto finished = Clock::now();
    for (std::size_t j = 0; j < outputs.size(); ++j) if ((published_members.empty() || published_members[j]) && outputs[j].generation_ledger) {
      const auto &output = outputs[j];
      auto &record = output.generation_ledger->records[output.generation_ledger->committed_count - 1];
      record.endpoint = finished;
      record.step_seconds = std::chrono::duration<double>(finished - step_start).count();
    }
    std::size_t compact_input_begin = 0;
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      const auto member_input_begin = compact_input_begin;
      compact_input_begin = cache::add(compact_input_begin, member_input_rows[j]);
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
      observation.input_tokens = member_input_rows[j]; observation.visible = session.length(); observation.next = next;
      observation.graph_executed = true; observation.graph_input_tokens = member_input_rows[j];
      observation.seconds = std::chrono::duration<double>(finished - start).count();
      observation.submit_seconds = submit_seconds; observation.step_seconds = std::chrono::duration<double>(finished - step_start).count();
      observation.cold_batch_seconds = cold_batch_seconds; observation.replay_route = replay_route;
      observation.shape_admission_seconds = shape_admission_seconds;
      observation.capture_recorded = capture_recorded; observation.capture_reused = capture_reused;
      observation.scheduled_progress = static_cast<bool>(scheduler);
      CompletedEvaluationView completed{.observation = observation, .head = *session.head,
          .member_index = j, .requests = outputs.size(),
          .packed_input_begin = packs_verify(phase.family->packing) ? member_input_begin : 0,
          .packed_input_rows = packs_verify(phase.family->packing) ? member_input_rows[j] : 0};
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
      std::span<const std::int32_t> eos = {}, std::vector<c::ReadLease> *q = nullptr,
      const inference::RequestStopPolicy *stop_policy = nullptr, const LogitsInputs &logits = {},
      std::span<const LogitsInputs> conditional_inputs = {}, ConditionalLogitsProposal *dynamic = nullptr) {
    if (session.model_ != this)
      throw std::runtime_error("recursive proposal Session belongs to another model");
    const auto &selected_sampling = session.sampling_configuration();
    const auto &selected_sampler = selected_sampling.configuration;
    validate_prefix_logits_inputs(selected_sampler, logits, conditional_inputs, cache::add(maximum, 1), vocabulary);
    if (dynamic && (!conditional_inputs.empty() || !dynamic->provider || dynamic->rows.size() != 1))
      throw std::runtime_error("dynamic proposal inputs require one previously resolved first row");
    if (dynamic) {
      auto row_policy = selected_sampler; row_policy.modifiers.history = {};
      validate_logits_inputs(row_policy, dynamic->rows.front(), vocabulary);
    }
    if (active_step || carrier_borrowed || session.pending || session.stopping || session.failed ||
        !session.head || !session.head->draft || (selected_sampler.enabled && !q) ||
        maximum >= capacity - session.length() || maximum >= cache_plan.candidate_boundaries)
      throw std::runtime_error("recursive proposals require an idle prepared head, bounded width and every stochastic q row");
    validate_mtp_snapshot(*session.head);
    inference::request_stop_policy(stop_policy).validate(vocabulary);
    const auto ended = [&](auto token) { return inference::request_token_stopped(token, eos, stop_policy); };
    std::vector<std::int32_t> inputs; inputs.reserve(maximum + 1); inputs.push_back(session.head->next_id);
    if (q) { q->clear(); q->reserve(maximum); }
    if (!maximum) return inputs;
    if (inputs.front() < 0 || inputs.front() >= vocabulary || ended(inputs.front()))
      throw std::runtime_error("recursive proposals require a valid nonterminal emitted pending token");
    DraftWork work; work.state = fork_draft(*session.head->draft);
    c::ReadLease approximate;
    auto current_logits = logits;
    const auto resolve = [&]() -> const LogitsInputs & {
      const auto row = inputs.size() - 1;
      if (row < dynamic->rows.size()) return dynamic->rows[row];
      if (row != dynamic->rows.size()) throw std::logic_error("conditional proposal skipped its linear prefix");
      auto supplied = dynamic->provider(dynamic->committed_outputs, std::span<const std::int32_t>(inputs).subspan(1));
      if (!supplied) throw inference::EvaluationFailure(hyle::Status(supplied.error()));
      auto policy = selected_sampler; policy.modifiers.history = {};
      auto status = logits_inputs_metadata_status(policy, *supplied, vocabulary);
      if (!status.ok()) throw inference::EvaluationFailure(std::move(status));
      validate_logits_input_leases(policy, *supplied);
      dynamic->rows.push_back(std::move(*supplied)); return dynamic->rows.back();
    };
    try {
    for (std::size_t i = 0; i < maximum; ++i) {
      if (dynamic) {
        const auto &row = resolve(); current_logits.bias = row.bias; current_logits.allowed = row.allowed;
      } else if (!conditional_inputs.empty()) {
        current_logits.bias = conditional_inputs[i].bias; current_logits.allowed = conditional_inputs[i].allowed;
      }
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (!i && work.state.lag) { ids.push_back(work.state.catchup_token); shifted.push_back(session.head->prior_hidden); }
      ids.push_back(inputs.back()); shifted.push_back(i ? approximate : session.head->carry_hidden);
      auto proposal = execute_draft(work, ids, shifted, true,
          selected_sampler.enabled ? session.head->sampling.identity(2 * i) : std::array<std::uint32_t, 4>{}, selected_sampling.index, current_logits);
      // Stop on a distribution property, not the sampled proposal's own
      // probability. The stopping condition is independent of this draw; its
      // unpublished address may then be the target's final draw at 2*m.
      if (min_draft_probability() > 0 && proposal.probability < min_draft_probability()) break;
      inputs.push_back(proposal.id); approximate = std::move(proposal.hidden);
      if (proposal.next_counts.valid()) current_logits.counts.lease = std::move(proposal.next_counts);
      if (q && selected_sampler.enabled) q->push_back(std::move(proposal.log_probabilities));
      if (ended(proposal.id)) break;
    }
    if (dynamic) {
      // An accepted stop proposal ends selection before its bonus; rejection
      // also ends at that proposal. Preserve q and extend the last live policy
      // for this unreachable row, without querying a terminated grammar.
      if (ended(inputs.back()) && dynamic->rows.size() < inputs.size())
        dynamic->rows.push_back(dynamic->rows.back());
      else (void)resolve(); // A threshold-rejected draft already resolved this bonus.
    }
    return inputs;
    } catch (inference::EvaluationFailure &failure) {
      if (dynamic && !failure.retirement_observed()) failure.observe_retirement(resources_->retire(false));
      throw;
    } catch (...) {
      if (dynamic) (void)resources_->retire(false);
      throw;
    }
  }

  void repair_recursive_draft(StepState &step) {
    if (step.phase->family->role != GraphRole::kVerify || !cache_plan.draft_cache)
      throw std::runtime_error("draft repair requires its unpublished target verifier");
    for (auto &member : step.members) {
      if (step.withdrawn(member)) continue;
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
    if (!active_step || selections.empty() || !active_step->phase || active_step->phase->family->role != GraphRole::kVerify ||
        !cache_plan.draft_cache || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size())
      throw std::runtime_error("recursive finish requires the complete original verification batch");
    const auto &selected_sampling = *active_step->phase->family->sampling;
    const auto &selected_sampler = selected_sampling.configuration;
    hyle::container::SmallVector<std::int32_t, 2> next;
    if (!next.try_resize(selections.size(), -1)) throw std::bad_alloc{};
    for (std::size_t j = 0; j < selections.size(); ++j) {
      const auto &selection = selections[j];
      if (selection.session != active_step->members[j].session || selection.session != outputs[j].session)
        throw std::runtime_error("recursive selection differs from its original batch member");
      results[j].reset();
      if (active_step->withdrawn(active_step->members[j])) continue;
      if (!selection.output_quota || inference::request_token_stopped(
              selection.expected_pending, eos, active_step->members[j].stop_policy))
        throw std::runtime_error("recursive generation requires its nonterminal pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(active_step->input_tokens);
    }
    finish_selected_evaluations(outputs, std::span(next.data(), next.size()),
        [&](std::span<const std::int32_t> candidates, StepState &step, std::span<std::int32_t> selected) {
      c::ReadLease logits_owner;
      mem::ConstBufferView logits_storage;
      if (selected_sampler.enabled || !selected_sampler.modifiers.empty()) {
        auto &logits = step.phase->slots[step.state_slot].data.logits;
        if (logits.caller_owned()) logits_storage = logits.storage();
        else {
          logits_owner = take(inference::retain_output(*step.execution.run, *step.phase->logits_facts, allocation_budget));
          logits_storage = take(logits_owner.view());
        }
      }
      for (std::size_t j = 0; j < selections.size(); ++j) {
        auto &member = step.members[j]; auto &result = results[j]; const auto &selection = selections[j];
        if (step.withdrawn(member)) { selected[j] = -1; continue; }
        if (member.first_input != selection.expected_pending || selection.expected_pending != member.base->next_id)
          throw std::runtime_error("recursive verifier input differs from the emitted pending token");
         const auto width = member.input_tokens;
         const auto rows = candidates.subspan(member.input_row_begin, width);
        result.pending = selection.expected_pending; result.proposal_count = width - 1; result.verified_rows = width;
        result.auxiliary_graph_calls = width + (selected_sampler.enabled || !selected_sampler.modifiers.empty() ? 1u : 0u);
        result.draft = width > 1 ? member.verification_inputs[1] : -1;
        result.target_first = rows[0]; result.target_second = width > 1 ? rows[1] : -1;
        hyle::inference::mtp::AcceptedPrefix accepted;
        if (selected_sampler.enabled || !selected_sampler.modifiers.empty()) {
          if (selected_sampler.enabled && selection.draft_log_probabilities.size() != width - 1)
            throw std::runtime_error("stochastic verification requires every actual q row");
          const auto bytes = cache::multiply(cache::multiply(width, vocabulary), sizeof(float));
          const auto logits = take(c::ConstTensorView::create(dense(Type::kFloat32,
               {static_cast<std::int64_t>(width), vocabulary}), take(logits_storage.subview(
                   cache::multiply(cache::multiply(member.input_row_begin, vocabulary), sizeof(float)), bytes))));
          PrefixSamplingResult decision;
          try {
            decision = sample_recursive_prefix(logits,
                std::span<const std::int32_t>(member.verification_inputs).subspan(1),
                selection.draft_log_probabilities, member.base->sampling, selection.output_quota,
                eos, member.stop_policy, selected_sampling.index, member.logits, member.conditional_logits.rows());
          } catch (const inference::EvaluationFailure &failure) {
            // Exact private validation of this original submitted member's
            // target rows; neither a request bucket nor a q-pool index.
            if (!failure.status().IsInvalidArgument() || failure.status().has_native_error() || failure.participant() ||
                !failure.retirement_observed() || !failure.retirement().retired || failure.retirement().status.has_native_error()) throw;
            inference::EvaluationFailure mapped(failure.status(), j);
            if (failure.retirement_observed()) mapped.observe_retirement(failure.retirement());
            throw mapped;
          }
          if (selected_sampler.enabled) {
            accepted = decision.prefix;
            result.emitted.assign(decision.final_ids.begin(), decision.final_ids.end());
            member.next->sampling = member.base->sampling.advanced(cache::add(cache::multiply(2, width - 1), 1));
            result.target_first = result.emitted.front();
            result.target_second = result.emitted.size() > 1 ? result.emitted[1] : -1;
          } else {
            result.target_first = decision.final_ids[0]; result.target_second = width > 1 ? decision.final_ids[1] : -1;
            accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, decision.final_ids,
                selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
          }
        } else {
          accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, rows,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
        }
        member.accepted_tokens = accepted.consumed; selected[j] = result.next_id = accepted.next_id;
        result.agreed = accepted.first_agrees; result.consumed_drafts = accepted.consumed_drafts;
        const bool clipped = min_draft_probability() > 0 && selection.proposal_limit > width - 1 &&
            !inference::request_token_stopped(member.verification_inputs.back(), eos, member.stop_policy);
        member.discarded_draft_rows = clipped ? (width == 1 ? member.base->draft->lag + 1 : 1) : 0;
        result.auxiliary_graph_calls += clipped ? 1u : 0u;
        result.accepted_draft = result.consumed_drafts != 0;
        result.processed_target = member.accepted_tokens;
        // Zero proposals never ran the draft head or its catch-up. A threshold-
        // rejected first head is counted completely in discarded_draft_rows.
        result.processed_draft = width - 1 + (width > 1 ? member.base->draft->lag : 0) +
            member.base->draft->lag + (member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1) +
            member.discarded_draft_rows;
        result.stopped = accepted.stopped;
        if (outputs[j].generation_ledger) {
          auto &record = outputs[j].generation_ledger->next();
          if (!record.generation || record.quota != selection.output_quota)
            throw std::runtime_error("recursive MTP ledger differs from its original output quota");
          record.draft = result.draft; record.target_first = result.target_first;
          record.target_second = result.target_second; record.agreed = result.agreed;
          record.accepted_draft = result.accepted_draft; record.consumed_drafts = result.consumed_drafts;
          record.auxiliary_graph_calls = result.auxiliary_graph_calls;
          record.stopped = result.stopped; write_generation_tokens(record.emitted, result.emitted);
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
            if (step.withdrawn(step.members[j])) { next[j] = -1; continue; }
            step.members[j].accepted_tokens = step.members[j].input_tokens; next[j] = ids[j];
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
    const auto &selected_sampler = active_step->phase->family->sampling->configuration;
    for (std::size_t j = 0; j < selections.size(); ++j) {
      if (selections[j].session != active_step->members[j].session || selections[j].session != outputs[j].session)
        throw std::runtime_error("MTP selection differs from its original batch member");
      results[j].reset();
      if (active_step->withdrawn(active_step->members[j])) continue;
      if (!selections[j].output_quota || selections[j].expected_pending < 0 ||
          selections[j].expected_pending >= vocabulary || inference::request_token_stopped(
              selections[j].expected_pending, eos, active_step->members[j].stop_policy))
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
            if (step.withdrawn(member)) { selected[j] = -1; continue; }
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
            if (selected_sampler.enabled) {
              const std::array flags{std::uint8_t(member.sampled_accept)};
              // The original K1 graph's second ID is its residual/bonus final;
              // its first ID is already Select(accept, proposal, final).
              const std::array finals{drafts_token(role) ? result.target_second : result.target_first};
              accepted = hyle::inference::mtp::select_sampled_prefix(verified_inputs,
                  std::span<const std::uint8_t>(flags).first(rows - 1), finals,
                  selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
            } else {
              const std::array targets{result.target_first, result.target_second};
              accepted = hyle::inference::mtp::select_greedy_prefix(verified_inputs,
                  std::span<const std::int32_t>(targets).first(rows), selection.output_quota, eos,
                  static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
            }
            member.accepted_tokens = accepted.consumed;
            selected[j] = result.next_id = accepted.next_id;
            result.agreed = accepted.first_agrees; result.accepted_draft = accepted.consumed_drafts != 0;
            result.processed_target = accepted.consumed; result.processed_draft = member.base->draft->lag + 1;
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
              write_generation_tokens(record.emitted, {result.emitted[0], result.emitted.size() == 2 ? result.emitted[1] : -1});
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
      if (completed.packed_input_rows) {
        const auto &description = tensor.description();
        if (!description.is_dense() || description.rank() != 2)
          throw std::runtime_error("packed verification trace requires dense row outputs");
        const auto row_bytes = take(description.required_bytes()) / static_cast<std::size_t>(description.dimensions()[0]);
        dump_bytes(take(tensor.storage().subview(cache::multiply(completed.packed_input_begin, row_bytes),
            cache::multiply(completed.packed_input_rows, row_bytes))), directory / file);
      } else {
        const auto bytes = take(tensor.description().required_bytes()) / completed.requests;
        dump_bytes(take(tensor.storage().subview(completed.member_index * bytes, bytes)), directory / file);
      }
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
      for (std::size_t i = 0; i < completed.head.paged.pages.size(); ++i) layout_trace << (i ? "," : "") << completed.head.paged.pages[i].page();
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
    failure.observe_retirement(model.retire_failed_publication(
        failure.retirement_observed() && !failure.retirement().retired));
    throw;
  } catch (...) { (void)model.retire_failed_publication(); throw; }
}

hyle::Status Session::stop() noexcept {
  if (stopping) return retirement_status;
  stopping = true;
  if (pending) {
    if (!pending->cancelled && pending->submitted && pending->execution.has_completion() &&
        pending->phase && model_ && pending->members.size() > 1 &&
        std::ranges::any_of(pending->members, [](const auto &member) { return !member.session->stopping; })) {
      pending->discard_stopped_members = true;
      // This acknowledges the requested withdrawal, rather than reporting a
      // device/retirement error. The completed member reports kCancelled.
      return retirement_status;
    }
    pending->cancelled = true;
    if (pending->execution.has_completion()) {
      const auto cancelled = pending->execution.request_cancel();
      if (!cancelled.IsUnimplemented()) inference::record_retirement_status(retirement_status, cancelled);
    }
    if (pending->execution.run) pending->execution.run->close().IgnoreError();
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
  LogitsInputs logits; // One [1,V] snapshot for every conditional row in an output episode.
  inference::LogitsInputProvider logits_provider;
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
    const auto chunks = prefill.parts(rows);
    count = cache::add(count, cache::add(chunks, turns[turn].max_outputs - 1));
  }
  return count;
}

GenerationCommitRecord &prepare_generation_record(GenerationLedger &ledger, const Session &session,
    GraphRole role, std::span<const std::int32_t> ids, bool generation,
    std::size_t quota, std::size_t turn, std::shared_ptr<const void> owner = {}, std::size_t maximum = 0,
    TextCompletion completion = TextCompletion::kEmitNext, std::size_t input_begin = 0, bool prefill_complete = true) {
  if (!supports_commit_completion(role, completion, generation, prefill_complete))
    throw std::runtime_error("state-only ledger entry requires an incomplete teacher-forced prefill");
  if (ids.empty() || !session.head || (generation && ids.size() != 1))
    throw std::runtime_error("MTP replay record requires its actual input and complete base");
  auto &record = ledger.next();
  ledger.bind_sampling(session.sampling_configuration(), session.sampling_owner());
  auto emitted = std::move(record.emitted), consumed = std::move(record.accepted_inputs);
  auto verified = std::move(record.verification_inputs);
  record = {}; record.emitted = std::move(emitted); record.accepted_inputs = std::move(consumed);
  record.verification_inputs = std::move(verified); record.verification_inputs.clear();
  write_generation_tokens(record.emitted, {-1, -1}); write_generation_tokens(record.accepted_inputs, {-1, -1});
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
    record.target_first = part.next_id; write_generation_tokens(record.emitted, {part.next_id, -1}); record.emitted_count = part.state_only ? 0 : 1;
    record.stopped = !part.state_only && inference::request_token_stopped(
        part.next_id, model_eos_tokens, &ledger.stop_policy());
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

void apply_generation_observation(GenerationRequest &request, const GenerationCommitRecord &record,
    const inference::CommitObservation &observation) noexcept {
  const auto output_end = request.progress.generated.size();
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
  if (observation.prefill_complete) {
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
}

// Consume committed host facts even when a subsequent diagnostic write threw.
// The authoritative ledger was installed by the original head-swap transaction.
void observe_generation_commits(GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  while (request.progress.observed_commits() < request.ledger->committed_count) {
    const auto ordinal = request.progress.observed_commits();
    (void)inference::observe_record_commit(std::span<const GenerationCommitRecord>(request.ledger->records),
        request.ledger->committed_count, ordinal, request.progress,
        [](const GenerationCommitRecord &record) {
          if (record.emitted_count > record.emitted.size())
            throw std::runtime_error("committed output count exceeds its original record storage");
          return inference::CommitObservation{
              std::span(record.emitted).first(record.emitted_count), record.user_output,
              !record.generation && !record.imported && record.prefill_complete,
              record.completion == TextCompletion::kEmitNext && record.stopped,
              !record.generation && !record.imported
                  ? std::optional<inference::RequestInputCommit>{{record.turn, record.input_begin, record.ids()}}
                  : std::nullopt};
        },
        [&](const GenerationCommitRecord &record, const inference::CommitObservation &) {
          if (record.user_output && record.generation &&
        (request.rounds.size() == request.rounds.capacity() || request.rounds.size() >= request.round_emitted.size() ||
         record.emitted_count > request.round_emitted[request.rounds.size()].capacity()))
            throw std::runtime_error("generation observer requires its original cold-reserved round storage");
        },
        [&](const GenerationCommitRecord &record, const inference::CommitObservation &observation) noexcept {
          apply_generation_observation(request, record, observation);
        });
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
  ledger.stop_policy().validate(model.vocabulary);
  const auto &replay_modifiers = model.sampling_configuration(sampling_index).configuration.modifiers;
  if (!model.cache_plan.draft_cache || replay_modifiers.bias || replay_modifiers.allowed)
    throw std::runtime_error("strict MTP replay requires the original complete target/draft model without external logits modifiers");
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
  auto session = model.create_session(calls, {}, first.role, first.before.sampling, false, first.completion, sampling_index);
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
          expected.quota, expected.turn, expected.input_owner, expected.draft_limit,
          expected.completion, expected.input_begin, expected.prefill_complete);
      make_generation_room(model, prefix, session.get(), requests);
      LogitsInputs replay_logits;
      if (expected.completion == TextCompletion::kEmitNext)
        replay_logits = model.history_logits_inputs(actual, ids, {}, sampling_index);
      std::vector<c::ReadLease> q;
      const auto verification = expected.role == GraphRole::kVerify
          ? model.propose_recursive(*session, expected.draft_limit, eos, &q, &actual.stop_policy(), replay_logits) : std::vector<std::int32_t>{};
      if (!expected.generation && is_mtp_prefill(expected.role)) {
        const std::array input{EvaluationRequest{session, ids, &actual.stop_policy(), replay_logits}};
        model.prepare_prefill_evaluations(input, nullptr, expected.role, expected.completion);
      } else model.prepare_evaluation(*session, expected.role == GraphRole::kVerify ? std::span<const std::int32_t>(verification) : ids,
          nullptr, expected.role, expected.completion, &actual.stop_policy(), replay_logits);
      model.submit_evaluation(*session);
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
      if ((!source.generation && role == GraphRole::kText) || is_mtp_prefill(role)) {
        const std::array input{EvaluationRequest{session, ids, &actual.stop_policy()}};
        model.prepare_prefill_evaluations(input, nullptr, role, completion);
      } else model.prepare_evaluation(*session, ids, nullptr, role, completion, &actual.stop_policy());
      model.submit_evaluation(*session);
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
  // A full hit may have sampled its new frontier before a later request's
  // preparation failed. Import adopted the source ordinals, but its first
  // output is still unobserved. Budget only future draws from that exact
  // published frontier, including when restoring its original ledger.
  const auto &ledger = *request.ledger;
  if (progress.turn_started() && !progress.turn_index && !progress.prefilled &&
      progress.generated.empty() && progress.input_admitted() &&
      progress.input_consumed() == progress.input().size() && ledger.committed_count &&
      ledger.committed_count <= ledger.records.size() && progress.observed_commits() == ledger.committed_count) {
    const auto &last = ledger.records[ledger.committed_count - 1];
    if (last.committed && last.imported && last.user_output && last.turn == progress.turn_index &&
        last.emitted_count == 1 && !last.emitted.empty() && last.emitted[0] == last.after.next_id &&
        last.after.next_id >= 0 && last.after.length == progress.input().size() && last.after.sampling == cursor &&
        (!request.session || generation_boundary(*request.session->head) == last.after))
      --outputs;
  }
  // Includes each not-yet-executed prefill/full-hit output. The last pending
  // token and StateOnly chunks require no additional draw.
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
  inference::LogitsInputProvider logits_provider;
  inference::ConditionalLogitsInputs conditional_logits;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::string name;
  double step_seconds = 0, proposal_seconds = 0;
  std::span<const std::int32_t> ids() const noexcept {
    if (generation && role == GraphRole::kVerify && !verification_ids.empty()) return verification_ids;
    return generation ? std::span(&pending, 1) : prompt;
  }
  // Actual submitted target rows, independent of the cold proposal bound or
  // accepted prefix. Verify has no actual rows until its proposals are ready.
  std::size_t actual_target_rows() const noexcept {
    if (generation && role == GraphRole::kVerify) return verification_ids.size();
    return generation && drafts_token(role) ? 2 : ids().size();
  }
  std::size_t planned_rows() const noexcept { return role == GraphRole::kVerify ? maximum + 1 : drafts_token(role) ? 2 : ids().size(); }
};

// Failure recovery may observe already authoritative records, but a second
// host-observation error must not replace the original Status/participant or
// retirement facts. Try every original peer independently. A rejected observer
// leaves its ordinal in the existing ledger for a later observation-only retry.
void observe_generation_commits_after_failure(std::span<const PreparedGenerationCall> calls) noexcept {
  for (const auto &call : calls) {
    try { observe_generation_commits(*call.request); }
    catch (...) {}
  }
}

void validate_generation_logits_inputs(const SamplingConfiguration &configuration,
    const PreparedGenerationCall &call, std::int64_t vocabulary) {
  if (call.conditional_logits.empty()) {
    validate_request_logits_inputs(configuration, call.logits, vocabulary, call.request->request_key); return;
  }
  const auto rows = call.conditional_logits.rows();
  check_request_logits_metadata(prefix_logits_inputs_metadata_status(configuration, call.logits, rows,
      call.ids().size(), vocabulary), call.request->request_key);
  validate_prefix_logits_input_leases(configuration, call.logits, rows);
}

std::size_t generation_prefill_capacity(const Model &model, std::span<const PreparedGenerationCall> calls) {
  if (calls.empty() || !calls[0].request || calls[0].generation || calls[0].full_hit ||
      (calls[0].role != GraphRole::kText && !is_mtp_prefill(calls[0].role))) return 0;
  std::size_t rows = 0;
  for (const auto &call : calls) {
    if (!call.request || call.generation || call.full_hit || call.role != calls[0].role ||
        call.completion != calls[0].completion || call.request->sampling_index != calls[0].request->sampling_index)
      return 0;
    rows = (std::max)(rows, call.ids().size());
  }
  return model.prefill_capacity(rows, calls[0].completion, calls[0].request->sampling_index, calls.size(), calls[0].role);
}

std::size_t generation_verification_capacity(const Model &model, std::span<const PreparedGenerationCall> calls) {
  if (calls.empty() || !calls[0].request || !calls[0].generation || calls[0].full_hit ||
      calls[0].role != GraphRole::kVerify || calls[0].completion != TextCompletion::kEmitNext) return 0;
  std::size_t rows = 0;
  for (const auto &call : calls) {
    if (!call.request || !call.generation || call.full_hit || call.role != GraphRole::kVerify ||
        call.completion != calls[0].completion || call.request->sampling_index != calls[0].request->sampling_index) return 0;
    rows = (std::max)(rows, call.verification_ids.empty() ? call.planned_rows() : call.ids().size());
  }
  return model.verification_capacity(rows, calls.size(), calls[0].request->sampling_index);
}

// One actually retired target batch, never a planned request bucket. Members
// are borrowed synchronously; a report owner copies identities before return.
// Episode durations can overlap and must not be added as a wall-time union.
struct GenerationBatchFacts final {
  GraphRole role; // First member's role; mixed K1 members retain their own roles below.
  TextCompletion completion;
  bool generation;
  std::size_t input_rows; // Maximum actual member width, never total work.
  std::size_t active_rows, physical_rows;
  std::span<const PreparedGenerationCall> members;
  double step_seconds;
};

template<class Observer>
void observe_generation_batch(Observer &observer, std::span<const PreparedGenerationCall> members,
    std::size_t rows, double seconds, std::size_t active_rows = 0, std::size_t physical_rows = 0) {
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Observer>>) {
    if (!active_rows)
      for (const auto &member : members) active_rows = cache::add(active_rows, member.actual_target_rows());
    if (!physical_rows) physical_rows = cache::multiply(rows, members.size());
    observer(GenerationBatchFacts{members[0].role, members[0].completion, members[0].generation,
        rows, active_rows, physical_rows, members, seconds});
  }
}

// Serialized host metadata only. Original Step/Source owns all native work;
// a deferred generation call may cross pumps without exporting another head.
// Exact Model/Graph checks, not warm-direction hints, authorize execution.
struct GenerationContinuation final {
  bool enabled = true, unavailable = false;
  inference::ContinuationCohort cohort;
  std::size_t next_slot = 0;
  std::optional<std::size_t> ready_slot;
  std::array<bool, execution_slot_count> direction_warm{};
  std::size_t ready_steps = 0;
  std::shared_ptr<StepSources> ready;
  hyle::container::SmallVector<PreparedGenerationCall, 2> pending_calls;
  std::size_t completed_steps = 0, successor_submissions = 0, successor_publications = 0,
      successor_discards = 0, successor_refusals = 0;

  void prepare(std::size_t maximum_members) {
    cohort.prepare(maximum_members);
    if (!pending_calls.try_reserve(maximum_members)) throw std::bad_alloc{};
  }
  void release_ready() noexcept {
    ready.reset(); ready_slot.reset(); next_slot = 0;
  }
  void reset_reuse() noexcept {
    release_ready(); direction_warm.fill(false); ready_steps = 0;
  }
  void reset() noexcept { reset_reuse(); cohort.reset_ready(); }
  static inference::ContinuationMember identity(const PreparedGenerationCall &call) noexcept {
    return {call.request->request_key, call.request->progress.turn_index};
  }
  bool select(std::span<const PreparedGenerationCall> calls) {
    const auto selected = cohort.select(calls, identity, !pending_calls.empty());
    if (selected == inference::ContinuationCohort::Selection::kPendingConflict)
      throw std::logic_error("pending continuation cannot change its original cohort");
    if (selected == inference::ContinuationCohort::Selection::kRejected) { reset(); return false; }
    if (selected == inference::ContinuationCohort::Selection::kChanged) reset_reuse();
    return true;
  }
  // Admission may run for one member before the next compatible batch exists.
  // Keep a matching cohort, but release it before another owner needs slots.
  void admission(inference::RequestKey selected, std::size_t selected_turn) noexcept {
    if (cohort.admission({selected, selected_turn}, !pending_calls.empty()) == inference::ContinuationCohort::Release::kReady)
      reset();
  }
  void release(inference::RequestKey selected) noexcept {
    if (cohort.release(selected, !pending_calls.empty()) == inference::ContinuationCohort::Release::kReady) reset();
  }
  bool matches(std::span<const PreparedGenerationCall> calls) const noexcept {
    const auto members = cohort.members();
    if (!ready || calls.size() != members.size() || ready->boundaries.size() != calls.size()) return false;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      const auto &request = *calls[j].request;
      if (members[j] != identity(calls[j]) || !request.session || !request.session->head)
        return false;
      const auto &boundary = ready->boundaries[j]; const auto owner = boundary.session.lock();
      if (owner != request.session || owner.owner_before(request.session) || request.session.owner_before(owner) ||
          boundary.candidate != request.session->head.get() || boundary.epoch != request.session->head.epoch()) return false;
      if (ready->mtp_full_candidate && (calls.size() != 1 || !drafts_token(calls[j].role) ||
          request.session->head->length != boundary.length || !request.session->head->draft ||
          request.session->head->draft->frontier != boundary.draft_frontier || request.session->head->draft->lag != 1))
        return false;
    }
    return true;
  }
};

std::span<const std::int32_t> provider_committed_outputs(const GenerationRequest &request) {
  if (!request.progress.turn_started()) return {};
  return std::span<const std::int32_t>(request.progress.generated).subspan(request.progress.turn_output_begin);
}

PreparedGenerationCall prepare_generation_call(Model &model, GenerationRequest &request, std::size_t steps,
    cache::PrefixCache *prefix, std::span<const GenerationRequest> requests,
    std::size_t output_credit = SIZE_MAX, std::size_t max_prefill_rows = SIZE_MAX) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (request.session && &request.session->sampling_configuration() != &prepared_sampling)
    throw std::runtime_error("request sampling cannot change after Session admission");
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  const auto &sampler = prepared_sampling.configuration;
  if (!max_prefill_rows) throw std::runtime_error("prefill rows require a positive quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  // Credit belongs to this episode, not the turn's immutable output quota.
  // Reject before prefix lookup, session creation or any other admission work.
  if (!output_credit) throw std::runtime_error("generation must wait for output credit before preparation");
  if (steps != request.progress.output_limit())
    throw std::runtime_error("generation call quota differs from its prepared request turn");
  if (request.progress.turn_started() && request.progress.turn_done)
    throw std::runtime_error("generation request has already completed its current turn");
  if (request.logits_provider && request.session && (!request.session->head || request.session->stopping ||
      request.session->failed || request.session->pending))
    throw std::runtime_error("dynamic logits preparation requires a live idle Session");
  preflight_generation_sampling(model, request);
  LogitsInputs episode_logits = request.logits;
  inference::LogitsInputProvider episode_provider;
  if (request.logits_provider && !sampler.modifiers.bias && !sampler.modifiers.allowed)
    check_request_logits_metadata(hyle::Status::InvalidArgument("dynamic logits require configured bias or mask ports"), request.request_key);
  if (!sampler.modifiers.empty()) {
    bool emits = request.progress.prefilled;
    if (!emits) {
      auto consumed = request.progress.input_consumed();
      if (consumed > request.prompt.size()) throw std::runtime_error("prefill offset exceeds its turn input");
      if (prefix && request.progress.turn_index == 0 && !request.session) {
        if (!request.input) throw std::runtime_error("prefix admission requires its complete token plan");
        // No eviction/publication or directory mutation occurs between this
        // pure length preview and the original pinned lookup below.
        consumed = prefix->match_length(*request.input, request.prompt.size(), 1, &model, model.math_identity());
      }
      emits = prefill.next(request.prompt.size(), consumed).complete;
    }
    if (emits) {
      auto external = sampler;
      external.modifiers.history = {}; // The ledger supplies natural C0 below.
      if (request.logits_provider) {
        auto base = external; base.modifiers.bias = false; base.modifiers.allowed = false;
        validate_request_logits_inputs(base, request.logits, model.vocabulary, request.request_key);
        episode_provider = request.logits_provider;
        auto supplied = episode_provider(provider_committed_outputs(request), {});
        if (!supplied) {
          auto status = hyle::Status(supplied.error());
          if (status.IsInvalidArgument()) check_request_logits_metadata(std::move(status), request.request_key);
          throw inference::EvaluationFailure(std::move(status));
        }
        episode_logits = std::move(*supplied);
      }
      validate_request_logits_inputs(external, episode_logits, model.vocabulary, request.request_key);
    }
  }
  if (!request.progress.turn_started()) (void)request.progress.begin_turn();
  if (request.progress.turn_done) throw std::runtime_error("MTP request has already completed");
  if (request.round_emitted.size() < steps - 1)
    throw std::runtime_error("MTP observer slots must be reserved for every turn before execution");
  request.ledger->bind_stop_policy(request.stop_policy, model.vocabulary);
  PreparedGenerationCall call; call.request = &request; call.steps = steps; call.logits = std::move(episode_logits);
  call.logits_provider = std::move(episode_provider);
  call.generation = request.progress.prefilled;
  if (!call.generation) {
    if (!request.progress.input_admitted()) {
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
      request.progress.bind_input(request.prompt);
    }
    request.ledger->validate_sampling(prepared_sampling, model.resources_);
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_begin\n";
    if (!request.progress.input_consumed()) request.prefill_start = request.admission_start == Clock::time_point{} ? Clock::now()
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
        request.progress.preflight_adopt_committed_history(reused_tokens);
        const auto suffix_part = prefill.next(request.prompt.size(), reused_tokens);
        const auto suffix = request.prompt.size() - suffix_part.begin;
        const auto suffix_rows = suffix_part.rows;
        const auto suffix_completion = suffix_part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
        std::string hit_name = "prefill";
        if (!suffix_part.complete) hit_name += "_chunk" + std::to_string(reused_tokens);
        const std::array chunks{suffix_rows};
        const auto prior_sampling_seconds = model.frontier_sampling_seconds;
        const auto suffix_role = !model.cache_plan.draft_cache ? GraphRole::kText : suffix_rows == 1
            ? (cached_lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
            : (cached_lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
        // Keep every new request fact local until the actual Session/frontier
        // succeeds. Failed native/resource admission must permit the same retry.
        const auto admission_role = !model.cache_plan.draft_cache ? GraphRole::kText
            : suffix ? suffix_role : request.prompt.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
        if (full_hit && !sampler.modifiers.history.empty())
          call.logits = model.history_logits_inputs(*request.ledger,
              std::span<const std::int32_t>(request.input->ids).first(reused_tokens), call.logits, request.sampling_index);
        std::shared_ptr<Session> adopted;
        try {
          adopted = model.create_session(suffix ? std::span<const std::size_t>(chunks) : std::span<const std::size_t>{},
              hit.payload.snapshot, admission_role,
              request.progress.initial_sampling, suffix == 0, suffix_completion, request.sampling_index, call.logits);
        } catch (const inference::EvaluationFailure &failure) {
          const auto key = request.request_key;
          if (failure.status().IsInvalidArgument() && !failure.status().has_native_error() && !failure.participant() &&
              failure.retirement_observed() && failure.retirement().retired && !failure.retirement().status.has_native_error() &&
              key.generation && key.slot != (std::numeric_limits<std::size_t>::max)())
            throw inference::RequestExecutionFailure::from_request_evaluation(failure, key);
          throw;
        }
        if (!suffix) history->chunks.back().next_id = adopted->head->next_id;
        // Only original reserved metadata remains. Preconditions are unchanged
        // since preflight; no allocation or native work follows this commit.
        import_generation_prompt_ledger(*request.ledger, hit.payload.replay, full_hit, model.cache_plan.draft_cache);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
            auto &record = request.ledger->records[i];
            record.before.sampling = sampler.enabled ? request.progress.initial_sampling : cache::SamplingCursor{};
            record.after.sampling = record.before.sampling;
            record.emitted_count = 0; write_generation_tokens(record.emitted, {-1, -1});
          }
          request.ledger->records[request.ledger->committed_count - 1].after.sampling = adopted->head->sampling;
        }
        if (!suffix) {
          auto &last = request.ledger->records[request.ledger->committed_count - 1];
          last.after.sampling = adopted->head->sampling;
          last.after.next_id = adopted->head->next_id;
          last.target_first = last.after.next_id; write_generation_tokens(last.emitted, {last.after.next_id, -1});
          last.emitted_count = 1; last.stopped = inference::request_token_stopped(
              last.after.next_id, model_eos_tokens, &request.stop_policy);
        }
        request.ledger->bind_sampling(prepared_sampling, model.resources_);
        request.progress.adopt_committed_history(request.ledger->committed_count, reused_tokens);
        request.history = std::move(history);
        request.session = std::move(adopted);
        request.reused_tokens = reused_tokens; request.prefix_hits = 1;
        request.prefix_full_hits = full_hit ? 1 : 0;
        call.name = std::move(hit_name);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          request.prefix_sampling_calls = suffix == 0 ? 1 : 0;
          request.prefix_sampling_seconds = model.frontier_sampling_seconds - prior_sampling_seconds;
        }
        hit.reset();
      } else { request.prefix_misses = 1; request.history = std::make_shared<cache::ReplayHistory>(model.capacity); }
    }
    const auto part = request.progress.next_input(prefill);
    call.input_begin = part.begin;
    call.prefill_complete = part.complete;
    call.completion = call.prefill_complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
    call.prompt = part.ids;
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
        : !model.recursive_mtp() && !call.logits_provider
            ? (lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : GraphRole::kVerify;
    call.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
        (model.cache_plan.draft_cache ? "mtp" : "decode") + std::to_string(request.rounds.size() + 1);
  }
  if (call.completion == TextCompletion::kStateOnly) call.logits = {};
  else {
    // A wave can exceed the actual physical cohort. Allocate its bounded C0
    // only when this call enters an executing bucket below. A full-hit's
    // frontier has already retired here, so it needs no later counts pin.
    if (call.full_hit) call.logits.counts = {};
    auto external = sampler; external.modifiers.history = {};
    validate_request_logits_inputs(external, call.logits, model.vocabulary, request.request_key);
  }
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
  if (call.disposition == EvaluationDisposition::kCancelled) { call.logits.counts = {}; return; }
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (call.full_hit && request.progress.prefilled) { call.logits.counts = {}; return; } // A committed cache hit is observed once.
  if (call.full_hit) {
    const auto next_id = request.session->head->next_id;
    const auto emitted = std::span<const std::int32_t>(&next_id, 1);
    const bool stopped = inference::request_token_stopped(next_id, model_eos_tokens, &request.stop_policy);
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
  call.logits.counts = {}; // Native consumers retired; the completed call no longer needs C0.
  (void)model;
}

template<class AfterSubmit, class BatchObserver>
bool execute_generation_continuation(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    GenerationContinuation &continuation, AfterSubmit &after_submit, BatchObserver &batch_observer,
    bool defer_successor = false, std::span<const inference::RequestCreditGrant> credits = {},
    bool drain_requested = false) {
  const bool resumed = !continuation.pending_calls.empty();
  const bool mtp = model.cache_plan.draft_cache;
  const auto maximum_outputs = mtp ? std::size_t{2} : std::size_t{1};
  const auto successor_role = mtp ? GraphRole::kMtpCatchupRound : GraphRole::kText;
  struct RetiredCalls final {
    Model &model; GenerationContinuation &continuation;
    int exceptions = std::uncaught_exceptions();
    ~RetiredCalls() {
      if (!model.active_step && !model.successor_step &&
          (std::uncaught_exceptions() == exceptions || model.private_publications_retired())) {
        continuation.pending_calls.clear(); continuation.cohort.clear_drain();
      }
    }
  } retired_calls{model, continuation};
  if (!resumed && (!continuation.enabled || continuation.unavailable || model.cache_plan.in_flight_batches < 2 ||
      (mtp && (calls.size() != 1 || model.max_drafts() != 1 || model.recursive_mtp())) ||
      model.scheduler || model.execution.host_only() ||
      model.execution.native_generated_storage() || model.execution.has_cpu_assist() ||
      std::ranges::any_of(calls, [&](const auto &call) {
        const auto &request = *call.request;
        const auto &sampling = request.session->sampling_configuration().configuration;
        return sampling.enabled || !sampling.modifiers.empty() ||
            (mtp ? !drafts_token(call.role) || call.maximum != 1 : call.role != GraphRole::kText) ||
            call.completion != TextCompletion::kEmitNext || call.full_hit || !call.generation ||
            (trace_states && !request.trace.empty());
      }))) {
    continuation.reset(); return false;
  }
  if (resumed && (calls.data() != continuation.pending_calls.data() ||
      calls.size() != continuation.pending_calls.size() || !model.active_step || model.successor_step ||
      !model.active_step->submitted || model.active_step->members.size() != calls.size()))
    throw std::logic_error("generation wave lost its original pending batch");
  if (!resumed && !continuation.select(calls)) return false;
  if (!resumed && continuation.ready && !continuation.matches(calls)) {
    continuation.reset(); if (!continuation.select(calls)) return false;
  }
  // A discarded successor really closed its original Invocation. Reinstall
  // missing carriers only while idle, before admitting this new predecessor.
  if (!resumed) model.prepare_relay_execution(calls.size(), calls[0].request->sampling_index, successor_role);
  const auto slot = resumed ? model.active_step->state_slot : continuation.next_slot;
  const auto prior_slot = continuation.ready_slot;
  const bool imported = resumed || continuation.matches(calls);
  hyle::container::SmallVector<PreparedGenerationCall, 2> second;
  hyle::container::SmallVector<EvaluationRequest, 2> direct, relay;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  if (!second.try_reserve(calls.size()) || !direct.try_reserve(calls.size()) ||
      !relay.try_reserve(calls.size()) || !outputs.try_reserve(calls.size()) ||
      !next.try_resize(calls.size(), -1) || (mtp &&
          (!selections.try_reserve(calls.size()) || !results.try_resize(calls.size())))) throw std::bad_alloc{};
  std::shared_ptr<StepSources> a_source, b_source;
  bool submitted_second = false, refused_second = false;
  Clock::time_point first_start{}, second_start{};
  std::optional<inference::EvaluationFailure> refused_failure;
  const auto notify = [&] {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) after_submit();
  };
  const auto release_and_abort = [&](const inference::EvaluationFailure *failure = nullptr) noexcept {
    b_source.reset(); a_source.reset(); continuation.reset();
    return model.retire_failed_publication(
        failure && failure->retirement_observed() && !failure->retirement().retired);
  };
  const auto require_retired = [](inference::RetirementObservation observation) {
    if (!observation.retired || (!observation.status.ok() &&
        (!observation.status.IsCancelled() || observation.status.has_native_error()))) {
      inference::EvaluationFailure failure(observation.status.ok()
          ? hyle::Status::Unavailable("generation continuation still retains its original episode")
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
  const auto recover_records = [&](bool retired) noexcept {
    observe_generation_commits_after_failure(calls);
    if (retired && !model.active_step && !model.successor_step) {
      // Host record recovery is secondary to the original error too. Native
      // retirement still comes only from release_and_abort above this call.
      try { discard_records(); }
      catch (...) {}
    }
  };
  const auto finish = [&](std::span<PreparedGenerationCall> current) {
    const auto rows = model.active_step->input_tokens;
    const auto started = model.active_step->step_start;
    outputs.clear(); selections.clear();
    for (auto &call : current) {
      auto &request = *call.request;
      outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
          &call.step_seconds, &request.endpoint, nullptr, nullptr, request.ledger.get()));
      outputs.back().disposition = &call.disposition;
      if (mtp) selections.push_back({request.session, call.pending, call.quota, {}, call.maximum});
    }
    const auto observe = [&](std::size_t j, const CompletedEvaluationView &view) {
          auto &call = current[j]; auto &request = *call.request;
          observe_completed_evaluation(model, outputs[j], {call.name, &request.trace, trace_states}, view);
        };
    if (mtp) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), model_eos_tokens,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe);
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
    observe_generation_batch(batch_observer, current, maximum_outputs, seconds);
  };
  try {
    for (auto &call : calls) {
      auto &request = *call.request;
      if (!resumed)
        prepare_generation_record(*request.ledger, *request.session, call.role, call.ids(), call.generation,
            call.quota, request.progress.turn_index, {}, call.maximum, call.completion,
            call.input_begin, call.prefill_complete);
      direct.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits});
      relay.push_back({request.session, {}, &request.ledger->stop_policy(), call.logits});
    }
    auto &a = resumed ? *model.active_step
        : imported ? model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), continuation.ready, slot, successor_role)
        : (model.prepare_evaluations(std::span(direct.data(), direct.size()), nullptr, calls[0].role), *model.active_step);
    first_start = a.step_start;
    if (resumed) a_source = a.execution.sources;
    else {
      a_source = model.prepare_submission_sources(a);
      model.submit_evaluations(a, true);
      if (!a.execution.sources) a_source.reset();
      notify();
    }
    const auto abort_a = [&] {
      // Ordinary withdrawal can leave A's other members publishable. Only a
      // real whole-episode cancellation/failure takes this abort branch.
      return a.cancelled || std::ranges::any_of(calls, [](const auto &call) { return call.request->session->failed; });
    };
    if (abort_a()) {
      if (resumed) ++continuation.successor_discards;
      require_retired(release_and_abort()); discard_records();
      observe_cancelled(calls, first_start, Clock::now()); return true;
    }
    if (!a_source) {
      finish(calls); ++continuation.completed_steps; discard_records();
      continuation.reset(); continuation.unavailable = true; return true;
    }
    const bool draining = continuation.cohort.draining(drain_requested, credits);
    if (resumed && !draining) {
      if (credits.size() != calls.size()) throw std::logic_error("pending batch lost its original request grants");
      for (std::size_t j = 0; j < calls.size(); ++j) {
        auto &call = calls[j]; auto &request = *call.request;
        if (credits[j].cancelled || request.session->stopping || request.progress.turn_done) continue;
        call.quota = request.ledger->renew_pending_quota(generation_boundary(*request.session->head),
            request.progress.observed_commits(), request.progress.turn_index, request.progress.remaining_outputs(),
            credits[j].reserved, credits[j].additional).quota;
      }
    }
    const auto next_slot = (slot + 1) % execution_slot_count;
    // Original B prepare/submit checks its complete tuple and Source. A's
    // possible future replay is not authority for B; warm flags are hints.
    if (!draining && imported && continuation.ready_steps >= 6 && continuation.direction_warm[next_slot] &&
        std::ranges::all_of(calls, [&](const auto &call) {
          const auto &request = *call.request; const auto &ledger = *request.ledger;
          return !request.session->stopping && !request.session->failed && call.quota >= 2 * maximum_outputs &&
              request.progress.remaining_outputs() >= 2 * maximum_outputs &&
              model.capacity - request.session->length() >= 2 * maximum_outputs &&
              request.round_emitted.size() - request.rounds.size() >= 2 &&
              request.rounds.capacity() - request.rounds.size() >= 2 &&
              ledger.records.size() - ledger.committed_count >= 2;
        })) {
      for (std::size_t j = 0; j < calls.size(); ++j) {
        const auto &call = calls[j]; auto &request = *call.request;
        second.push_back(call); auto &following = second.back();
        following.pending = -1; following.quota = call.quota - maximum_outputs;
        following.role = successor_role;
        following.step_seconds = following.proposal_seconds = 0;
        following.disposition = EvaluationDisposition::kNotPublished;
        following.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
            (mtp ? "mtp" : "decode") + std::to_string(request.rounds.size() + 2);
        request.ledger->reserve_successor(request.progress.observed_commits(), following.quota,
            request.progress.turn_index, calls.size(), j);
      }
      StepState *b = nullptr;
      try { b = &model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), a_source, next_slot, successor_role); }
      catch (const inference::EvaluationFailure &failure) {
        if (!failure.status().IsUnavailable() || failure.status().has_native_error() ||
            (failure.retirement_observed() && !failure.retirement().status.ok()) ||
            (model.successor_step && model.successor_step->submitted)) throw;
        refused_failure = failure; refused_second = true; ++continuation.successor_refusals;
        std::cerr << "decode_overlap_refused code=" << hyle::status_code_name(failure.status().code())
            << " native=" << failure.status().has_native_error() << " predecessor_slot=" << slot
            << " successor_slot=" << next_slot << " members=" << calls.size()
            << " committed_length=" << calls.front().request->session->length()
            << " candidate_length=" << a.members.front().next->length
            << " reason=" << failure.status().message() << '\n';
      }
      if (b) {
        second_start = b->step_start; b_source = model.prepare_submission_sources(*b);
        model.submit_evaluations(*b); submitted_second = true; ++continuation.successor_submissions;
        notify();
      }
    }
    if (abort_a()) {
      if (submitted_second) ++continuation.successor_discards;
      if (resumed) ++continuation.successor_discards;
      require_retired(release_and_abort()); discard_records();
      const auto retired_at = Clock::now(); observe_cancelled(calls, first_start, retired_at);
      if (submitted_second) observe_cancelled(std::span(second.data(), second.size()), second_start, retired_at);
      return true;
    }
    finish(calls); ++continuation.completed_steps;
    if (resumed) ++continuation.successor_publications;
    continuation.ready.reset();
    if (refused_second) {
      inference::RetirementObservation retired{!model.active_step && !model.successor_step, {}};
      for (const auto &call : calls) {
        retired.retired = retired.retired && !call.request->session->pending;
        inference::record_retirement_status(retired.status, call.request->session->retirement_status);
      }
      refused_failure->observe_retirement(retired);
      if (!retired.retired || !refused_failure->retirement().status.ok()) throw *refused_failure;
      // Optional device-ordered reuse was unavailable for this physical tuple.
      // A is now actually retired and B was never submitted. Re-prime through
      // the original ready-input route so it can capture the new layout; a
      // temporary coverage miss does not disable every later continuation.
      discard_records(); a_source.reset(); continuation.reset();
      return true;
    }
    if (submitted_second) {
      // Pending sources are a complete same-order cohort. A partial A result
      // cannot be projected into B. Retire all B, preserve every successful A,
      // and let the original owner choose the next runnable subset.
      if (inference::continuation_retention(calls, [](const auto &call) {
            const auto &request = *call.request;
            return inference::ContinuationProgress{call.disposition == EvaluationDisposition::kPublished,
                request.progress.turn_done, request.session->stopping, request.session->failed,
                request.progress.remaining_outputs()};
          }, true, model.active_step && model.active_step->cancelled, continuation.unavailable, draining) ==
          inference::ContinuationRetention::kRetirePending) {
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
      if (defer_successor) {
        // All borrowed A metadata has been observed. B's native owner remains
        // the original promoted active_step; only its host call crosses pumps.
        continuation.pending_calls.clear();
        for (auto &call : second) continuation.pending_calls.push_back(std::move(call));
        continuation.cohort.clear_drain();
        continuation.next_slot = (next_slot + 1) % execution_slot_count;
        return true;
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
      // A K1 full Source is no longer an input fact after partial acceptance.
      // It cannot seed B, even when the head address and epoch still match.
      if (mtp && !continuation.matches(calls)) continuation.release_ready();
      // Withdrawn A records have no commit; clear only their unpublished work.
      discard_records();
    }
    const auto retention = inference::continuation_retention(calls, [](const auto &call) {
          const auto &request = *call.request;
          return inference::ContinuationProgress{call.disposition == EvaluationDisposition::kPublished,
              request.progress.turn_done, request.session->stopping, request.session->failed,
              request.progress.remaining_outputs()};
        }, false, false, continuation.unavailable, draining);
    if (retention == inference::ContinuationRetention::kResetReady) continuation.reset();
    else if (retention == inference::ContinuationRetention::kReleaseReady ||
        std::ranges::any_of(calls, [](const auto &call) { return call.quota <= 1; })) continuation.release_ready();
    return true;
  } catch (inference::EvaluationFailure &failure) {
    const auto retired = release_and_abort(&failure);
    failure.observe_retirement(retired); recover_records(retired.retired);
    throw;
  } catch (...) {
    const auto retired = release_and_abort(); recover_records(retired.retired);
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
    BatchObserver &&batch_observer = nullptr, bool defer_successor = false,
    std::span<const inference::RequestCreditGrant> credits = {}) {
  if (calls.empty()) return;
  const bool packed_prefill = generation_prefill_capacity(model, calls) != 0;
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
    if (drafts_token(call.role) && (!request.session->head->draft ||
        request.session->head->draft->lag != draft_lag(call.role)))
      throw std::runtime_error("generation K1 member role differs from its actual draft lag");
    for (const auto &prior : calls.first(member))
      if (prior.request->session.get() == request.session.get())
        throw std::runtime_error("generation batch requires distinct Session owners");
    const auto &sampling = model.sampling_configuration(request.sampling_index);
    if (&request.session->sampling_configuration() != &sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    request.ledger->validate_sampling(sampling, model.resources_);
    if (call.completion == TextCompletion::kEmitNext) {
      auto required = sampling.configuration;
      if (call.full_hit || !call.logits.counts.lease.valid()) required.modifiers.history = {};
      validate_generation_logits_inputs(required, call, model.vocabulary);
    }
    if (request.ledger->stop_policy() != request.stop_policy)
      throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
    if ((call.full_hit && calls.size() != 1) ||
        call.request->sampling_index != calls[0].request->sampling_index ||
        (call.role != calls[0].role && !(call.generation && calls[0].generation &&
            drafts_token(call.role) && drafts_token(calls[0].role))) ||
        call.completion != calls[0].completion || call.generation != calls[0].generation ||
        (!packed_prefill && call.role != GraphRole::kVerify && call.ids().size() != calls[0].ids().size()))
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    if (!call.full_hit) {
      if (!supports_commit_completion(call.role, call.completion, call.generation, call.prefill_complete))
        throw std::runtime_error("state-only ledger entry requires an incomplete teacher-forced prefill");
      if (call.ids().empty() || (call.role == GraphRole::kVerify && !call.generation))
        throw std::runtime_error("generation call requires its actual prepared input");
      (void)request.ledger->next();
      if (!call.generation) (void)request.warm_inputs.at(request.progress.turn_index);
    }
  }
  // C0 storage is bounded by the actual executing cohort, never the number
  // of queued requests. All original request metadata above is checked first.
  for (auto &call : calls) if (!call.full_hit && call.completion == TextCompletion::kEmitNext) {
    auto &request = *call.request;
    const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
    if (!configuration.modifiers.history.empty() && !call.logits.counts.lease.valid())
      call.logits = model.history_logits_inputs(*request.ledger,
          call.generation ? std::span<const std::int32_t>(&call.pending, 1) : call.prompt,
          call.logits, request.sampling_index);
    validate_generation_logits_inputs(configuration, call, model.vocabulary);
  }
  if (calls.size() == 1 && calls[0].full_hit) {
    if (continuation) continuation->reset();
    finish_generation_call(model, calls[0]); calls[0].disposition = EvaluationDisposition::kPublished; return;
  }
  if (!make_generation_batch_room(model, prefix, calls, requests)) throw std::runtime_error("MTP batch exceeds committed head budget");
  if (continuation) {
    if (execute_generation_continuation(model, calls, trace_states, *continuation,
        after_submit, batch_observer, defer_successor, credits)) return;
    continuation->reset();
  }
  const auto &eos = model_eos_tokens;
  std::size_t packed_verification_capacity = 0;
  if (calls[0].role == GraphRole::kVerify) {
    const auto discard_proposals = [&] {
      for (auto &peer : calls) {
        if (!peer.conditional_logits.empty()) {
          const auto &first = peer.conditional_logits.rows().front();
          peer.logits.bias = first.bias; peer.logits.allowed = first.allowed;
        }
        peer.draft_log_probabilities.clear(); peer.conditional_logits = {}; peer.verification_ids.clear();
      }
    };
    for (auto &call : calls) {
      if (!call.generation || call.role != GraphRole::kVerify)
        throw std::runtime_error("recursive MTP batch requires generation verifiers only");
      if (call.verification_ids.empty()) {
        const auto begin = Clock::now();
        try {
          std::optional<ConditionalLogitsProposal> dynamic;
          if (call.logits_provider) {
            dynamic.emplace(ConditionalLogitsProposal{call.logits_provider, provider_committed_outputs(*call.request), {}});
            dynamic->rows.reserve(call.maximum + 1);
            dynamic->rows.push_back({call.logits.bias, call.logits.allowed});
          }
          call.verification_ids = model.propose_recursive(*call.request->session, call.maximum, eos,
              &call.draft_log_probabilities, &call.request->ledger->stop_policy(), call.logits, {}, dynamic ? &*dynamic : nullptr);
          if (dynamic) {
            call.conditional_logits = inference::ConditionalLogitsInputs(dynamic->rows);
            call.logits.bias = {}; call.logits.allowed = {};
          }
        } catch (const inference::EvaluationFailure &failure) {
          const auto key = call.request->request_key;
          if (failure.retirement_observed() && failure.retirement().retired) discard_proposals();
          // The original one-request q validator has no target participant.
          // Preserve unknown/native/unfinished failures without reclassifying.
          if (failure.status().IsInvalidArgument() && !failure.status().has_native_error() && !failure.participant() &&
              failure.retirement_observed() && failure.retirement().retired && !failure.retirement().status.has_native_error() &&
              key.generation && key.slot != (std::numeric_limits<std::size_t>::max)())
            throw inference::RequestExecutionFailure::from_request_evaluation(failure, key);
          throw;
        } catch (...) {
          const auto observed = model.resources_->retire(false);
          if (observed.retired) discard_proposals();
          throw;
        }
        call.proposal_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      }
    }
    // Keep exact R1/uniform graphs. Only an unequal cohort selects the cold
    // capacity whose offsets express every member's real rows.
    const auto uniform_rows = calls[0].ids().size();
    const bool unequal = std::ranges::any_of(calls, [&](const auto &call) { return call.ids().size() != uniform_rows; });
    if (unequal) packed_verification_capacity = generation_verification_capacity(model, calls);
    // Missing provider capability keeps exact grouping, never retries native failure.
    if (unequal && !packed_verification_capacity) {
      std::map<std::size_t, std::vector<std::size_t>> groups;
      for (std::size_t i = 0; i < calls.size(); ++i) groups[calls[i].ids().size()].push_back(i);
      for (const auto &[rows, members] : groups) {
        (void)rows;
        hyle::container::SmallVector<PreparedGenerationCall, 2> selected;
        if (!selected.try_reserve(members.size())) throw std::bad_alloc{};
        inference::SelectedRequestCalls<PreparedGenerationCall, 2> borrowed(calls, members, selected);
        try {
          execute_generation_calls(model, borrowed.members(), trace_states, prefix, requests, after_submit, nullptr, batch_observer);
        } catch (const inference::EvaluationFailure &failure) {
          // A child index is not an index into the original request bucket.
          // Preserve the original error and actual retirement independently.
          const auto child = failure.participant();
          inference::EvaluationFailure mapped(failure.status(), child && *child < members.size()
              ? std::optional<std::size_t>(members[*child]) : std::nullopt);
          if (failure.retirement_observed()) mapped.observe_retirement(failure.retirement());
          throw mapped;
        }
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
    if (role == GraphRole::kVerify && !packed_verification_capacity && call.ids().size() != calls[0].ids().size())
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    prepare_generation_record(*request.ledger, *request.session, call.role,
        generation ? std::span<const std::int32_t>(&call.pending, 1) : call.ids(), generation,
        call.quota, request.progress.turn_index, generation ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(request.warm_inputs.at(request.progress.turn_index)), call.maximum, call.completion, call.input_begin, call.prefill_complete);
    inputs.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits, call.conditional_logits, request.request_key});
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
  if (!generation && packed_prefill) model.prepare_prefill_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion);
  else if (packed_verification_capacity)
    model.prepare_verification_evaluations(std::span(inputs.data(), inputs.size()), packed_verification_capacity);
  else model.prepare_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion);
  model.submit_evaluations();
  const auto input_rows = model.active_step->input_tokens;
  const auto active_rows = model.active_step->active_input_rows;
  const auto physical_rows = cache::multiply(model.active_step->phase->query_rows, calls.size());
  const auto episode_start = model.active_step->step_start;
  bool prepared_ahead = false;
  try {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) {
      try { after_submit(); }
      catch (inference::EvaluationFailure &failure) {
        failure.observe_retirement(model.retire_failed_publication(
            failure.retirement_observed() && !failure.retirement().retired)); throw;
      }
      catch (...) { (void)model.retire_failed_publication(); throw; }
    }
    if (model.active_step && model.active_step->cancelled) {
      // A single/all-member or pre-submit stop cancels the whole episode.
      // Its original transactions still require actual native retirement.
      const bool requested = std::ranges::any_of(model.active_step->members,
          [](const auto &member) { return member.session->stopping; });
      auto retired = model.abort_episode();
      for (const auto &call : calls) if (call.request->session->stopping)
        inference::record_retirement_status(retired.status, call.request->session->retirement_status);
      if (!requested || !retired.retired || (!retired.status.ok() &&
          (!retired.status.IsCancelled() || retired.status.has_native_error()))) {
        auto status = retired.status;
        if (status.ok()) status = retired.retired
            ? hyle::Status::FailedPrecondition("cancelled generation episode has no stopped member")
            : hyle::Status::Unavailable("cancelled generation episode is still retiring");
        inference::EvaluationFailure failure(std::move(status));
        failure.observe_retirement(retired); throw failure;
      }
      const auto seconds = std::chrono::duration<double>(Clock::now() - episode_start).count();
      double proposal_seconds = 0;
      for (auto &call : calls) {
        call.request->ledger->discard_uncommitted();
        call.draft_log_probabilities.clear(); call.logits.counts = {}; call.conditional_logits = {};
        call.disposition = EvaluationDisposition::kCancelled;
        call.step_seconds = seconds + call.proposal_seconds;
        proposal_seconds += call.proposal_seconds;
      }
      observe_generation_batch(batch_observer, calls, input_rows, seconds + proposal_seconds, active_rows, physical_rows);
      return;
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
    if (generation && role == GraphRole::kVerify) {
      model.finish_recursive_mtp(std::span(selections.data(), selections.size()), eos,
          std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
      // The original verifier and prefix decision have retired and released
      // their reads. Completed call records no longer need these private q
      // leases; retaining them would charge the next episode's bounded pool.
      for (auto &call : calls) call.draft_log_probabilities.clear();
    }
    else if (generation && role != GraphRole::kText) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe);
    double target_seconds = 0, proposal_seconds = 0;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      target_seconds = (std::max)(target_seconds, calls[j].step_seconds);
      proposal_seconds += calls[j].proposal_seconds;
      calls[j].step_seconds += calls[j].proposal_seconds;
      if (calls[j].proposal_seconds && calls[j].disposition == EvaluationDisposition::kPublished) {
        auto &ledger = *calls[j].request->ledger;
        ledger.records[ledger.committed_count - 1].step_seconds += calls[j].proposal_seconds;
      }
    }
    // Shared target work is counted once; recursive proposals were serial
    // per member and are all retained. Cancelled members may have no output
    // endpoint, so a fully withdrawn batch uses its actual retired call end.
    if (target_seconds == 0) target_seconds = std::chrono::duration<double>(Clock::now() - episode_start).count();
    observe_generation_batch(batch_observer, calls, input_rows, target_seconds + proposal_seconds, active_rows, physical_rows);
    for (auto &call : calls) { finish_generation_call(model, call); call.conditional_logits = {}; }
    if (prepared_ahead)
      for (const auto &call : calls)
        if (call.disposition != EvaluationDisposition::kPublished || call.request->progress.turn_done ||
            !call.request->session || call.request->session->stopping) {
          model.discard_decode_ahead(call.request->session.get());
        }
  } catch (...) {
    observe_generation_commits_after_failure(calls);
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

// Explicit synchronous consumer, including component drivers. The request
// adapter below has one wave contract for both synchronous and pipelined use.
template<class Execute>
struct SynchronousGenerationWave final {
  Model &model; Execute &execute;
  std::size_t pending_count() const noexcept { return 0; }
  inference::RequestWavePendingMember pending_member(std::size_t) const noexcept { return {}; }
  inference::RequestWaveAdvance advance(inference::RequestWaveAction action, inference::RequestWave wave) const {
    if (action == inference::RequestWaveAction::kDrain)
      return {.remaining = retire(inference::RequestWaveRetirement::kObserve)};
    execute(wave.keys);
    if (model.active_step || model.successor_step)
      throw std::logic_error("synchronous request wave retained an unfinished episode");
    return {.advanced = true, .remaining = {true, {}}};
  }
  inference::RetirementObservation retire(inference::RequestWaveRetirement action) const noexcept {
    if (action == inference::RequestWaveRetirement::kAbort) return model.retire_failed_publication();
    auto retired = model.try_retire_episode();
    retired.retired = retired.retired && model.private_publications_retired();
    return retired;
  }
  inference::RequestWaveResult operator()(inference::RequestWave wave,
      std::span<std::size_t> remaining_reserved) const {
    return inference::advance_request_wave(wave, remaining_reserved, *this);
  }
};

// Native work and deferred calls stay in the original Model/continuation.
// Common wave control owns only the cohort/credit and retirement transitions.
template<class Ready, class AfterSubmit = std::nullptr_t, class BatchObserver = std::nullptr_t>
inference::RequestWaveResult advance_generation_wave(Model &model, inference::RequestWave wave,
    std::span<std::size_t> remaining_reserved, bool trace_states, GenerationContinuation &continuation,
    Ready &&execute_ready, AfterSubmit &&after_submit = nullptr, BatchObserver &&batch_observer = nullptr) {
  struct Stage final {
    Model &model;
    GenerationContinuation &continuation;
    bool trace_states;
    Ready &execute_ready;
    AfterSubmit &after_submit;
    BatchObserver &batch_observer;

    std::size_t pending_count() const noexcept { return continuation.pending_calls.size(); }
    inference::RequestWavePendingMember pending_member(std::size_t j) const noexcept {
      const auto &call = continuation.pending_calls[j];
      return {call.request->request_key, (std::min)(call.quota, drafts_token(call.role) ? std::size_t{2} : std::size_t{1})};
    }
    inference::RequestWaveAdvance advance(inference::RequestWaveAction action, inference::RequestWave current) {
      if (action == inference::RequestWaveAction::kResume) {
        auto calls = std::span(continuation.pending_calls.data(), continuation.pending_calls.size());
        try {
          if (!execute_generation_continuation(model, calls, trace_states, continuation,
              after_submit, batch_observer, true, current.credits, current.drain_requested))
            throw std::logic_error("pending generation wave lost its original continuation");
        } catch (const inference::EvaluationFailure &failure) {
          throw inference::RequestExecutionFailure::from_evaluation(failure, current.keys.size(),
              [&](std::size_t j) { return current.keys[j]; });
        }
      } else if (action == inference::RequestWaveAction::kStart) execute_ready(current);
      else continuation.reset();
      const bool pending = !continuation.pending_calls.empty();
      if (pending) {
        if (!model.active_step || model.successor_step || !model.active_step->submitted ||
            continuation.pending_calls.size() != current.keys.size())
          throw std::logic_error("generation wave retained metadata without its original submission");
      } else if (model.active_step || model.successor_step)
        throw std::logic_error("generation wave lost the host metadata for its original submission");
      return {.advanced = true, .remaining = {!pending, {}}};
    }
    inference::RetirementObservation retire(inference::RequestWaveRetirement action) noexcept {
      auto retired = action == inference::RequestWaveRetirement::kObserve
          ? model.try_retire_episode() : model.retire_failed_publication();
      retired.retired = retired.retired && model.private_publications_retired();
      if (retired.retired) {
        continuation.pending_calls.clear(); continuation.reset(); continuation.cohort.clear_drain();
      }
      return retired;
    }
  } stage{model, continuation, trace_states, execute_ready, after_submit, batch_observer};
  return inference::advance_request_wave(wave, remaining_reserved, stage);
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
  inference::RequestWaveResult execute_wave(inference::RequestWave wave,
      std::span<std::size_t> remaining_reserved) const {
    if (remaining_reserved.size() != wave.keys.size())
      throw std::logic_error("request wave reservation extent differs from its original cohort");
    return execute(wave, remaining_reserved);
  }
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
  const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
  if (model.sampling_configuration_count() == 1 && configuration.modifiers.history.empty()) return;
  out << ",\"sampling_configuration_index\":" << request.sampling_index
      << ",\"request_sampling\":\"" << (configuration.enabled ? "random" : "greedy") << '\"'
      << ",\"request_temperature\":" << configuration.policy.temperature
      << ",\"request_top_k\":" << configuration.policy.top_k << ",\"request_top_p\":" << configuration.policy.top_p
      << ",\"request_history_repetition\":" << configuration.modifiers.history.repetition
      << ",\"request_history_presence\":" << configuration.modifiers.history.presence
      << ",\"request_history_frequency\":" << configuration.modifiers.history.frequency
      << ",\"request_mtp_acceptance_policy\":\"" << (!model.cache_plan.draft_cache ? "disabled" : configuration.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '\"';
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

template<GenerationMode Mode, class BatchObserver = std::nullptr_t>
int run_generation_sessions(Model &model, std::span<const inference::GenerationSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans,
    std::span<const std::size_t> sampling_indices = {}, std::span<const inference::RequestStopPolicy> stop_policies = {},
    std::size_t max_prefill_rows = SIZE_MAX, bool warmup_planned_shapes = false,
    bool decode_overlap = false, std::span<const inference::LogitsInputProvider> logits_providers = {},
    BatchObserver &&batch_observer = nullptr) {
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  if (!max_prefill_rows) throw std::runtime_error("prefill rows require a positive quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  if (!stop_policies.empty() && stop_policies.size() != schedules.size())
    throw std::runtime_error("one immutable stop policy is required per request");
  for (const auto &policy : stop_policies) {
    policy.validate(model.vocabulary);
  }
  if (!sampling_indices.empty() && sampling_indices.size() != schedules.size())
    throw std::runtime_error("one cold sampling choice is required per request");
  for (const auto index : sampling_indices) {
    (void)model.sampling_configuration(index);
  }
  if (!logits_providers.empty() && logits_providers.size() != schedules.size())
    throw std::runtime_error("one optional logits provider is required per request");
  for (std::size_t i = 0; i < logits_providers.size(); ++i) if (logits_providers[i]) {
    const auto &policy = model.sampling_configuration(sampling_indices.empty() ? 0 : sampling_indices[i]).configuration.modifiers;
    if (!policy.bias && !policy.allowed)
      throw std::runtime_error("dynamic logits require configured bias or mask ports");
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
    if (!logits_providers.empty()) request.logits_provider = logits_providers[requests.size() - 1];
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
  using Demands = inference::ColdDemandInventory<PromptKey>;
  Demands prompt_demand, carrier_demand;
  const auto note_prompt = [&](GraphRole role, std::size_t tokens, std::size_t member,
                               TextCompletion completion = TextCompletion::kEmitNext, bool possible = false) {
    prompt_demand.note({role, completion}, tokens, member,
        possible ? inference::ColdDemandKind::kPossible : inference::ColdDemandKind::kCounted);
  };
  const auto note_carrier = [&](GraphRole role, std::size_t rows, std::size_t member, TextCompletion completion, bool possible = false) {
    carrier_demand.note({role, completion}, rows, member,
        possible ? inference::ColdDemandKind::kPossible : inference::ColdDemandKind::kCounted);
  };
  const auto note_prefill = [&](std::size_t tokens, std::size_t member, bool lag = false) {
    for (std::size_t consumed = 0; consumed < tokens;) {
      const auto part = prefill.next(tokens, consumed);
      const auto completion = part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
      // A committed prefill consumes an old catch-up row exactly once and
      // leaves a complete draft frontier for every later input chunk.
      const auto role = prompt_role(part.rows, lag && consumed == 0);
      note_prompt(role, part.rows, member, completion);
      if (role == GraphRole::kText || is_mtp_prefill(role))
        note_carrier(role, (std::min)(tokens, max_prefill_rows), member, completion);
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
    if (prefix && prefill.bounded()) {
      // A generated checkpoint can leave any real suffix. Keep one capacity
      // per completion/role, including the original T1 path, rather than an
      // exact graph for every possible suffix length.
      prefill.for_each_capacity(requests[i].prompt.size(), [&](std::size_t rows, bool complete) {
        const auto completion = complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
        const auto note_suffix = [&](bool lag) {
          // A nonempty prefix already consumed at least one target row. Its
          // catch-up carrier need not exceed the complete prompt capacity.
          const auto suffix_rows = lag ? (std::min)(rows, requests[i].prompt.size() - 1) : rows;
          if (!suffix_rows) return;
          const auto role = prompt_role(suffix_rows, lag);
          if (role == GraphRole::kText || is_mtp_prefill(role)) note_carrier(role, suffix_rows, i, completion, true);
          else note_prompt(role, 1, i, completion, true);
        };
        note_suffix(false);
        if constexpr (mtp) note_suffix(true);
      });
      if constexpr (mtp) {
        note_prompt(GraphRole::kMtpSingle, 1, i, TextCompletion::kEmitNext, true);
        if (requests[i].prompt.size() > 1)
          note_prompt(GraphRole::kMtpCatchupSingle, 1, i, TextCompletion::kEmitNext, true);
      }
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
  prompt_demand.inventory({prompt_role(1), TextCompletion::kEmitNext}, 1);
  if constexpr (mtp) for (std::size_t rows = 2;
      rows <= (std::min)(model.cache_plan.candidate_boundaries, model.cache_plan.max_query_rows); ++rows)
    prompt_demand.inventory({GraphRole::kMtpPrefill, TextCompletion::kEmitNext}, rows);
  for (std::size_t sampling_index = 0; sampling_index < model.sampling_configuration_count(); ++sampling_index) {
    if (std::ranges::none_of(requests, [&](const auto &request) { return request.sampling_index == sampling_index; })) continue;
    const auto matches_member = [&](std::size_t member) { return requests[member].sampling_index == sampling_index; };
    const auto policy_decoding_requests = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
    }));
    const auto dynamic_decoding_requests = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && request.logits_provider &&
          std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
    }));
    const auto recursive_requests = model.recursive_mtp() ? policy_decoding_requests : dynamic_decoding_requests;
  for (std::size_t width = 1; width <= model.cache_plan.active_sequences; ++width) {
    std::map<PromptKey, std::map<std::size_t, std::size_t>> shapes;
    for (const auto &[key, tokens] : prompt_demand.entries()) for (const auto &[count, demand] : tokens) {
      const auto eligible = demand.eligible_count(matches_member);
      if (width == 1 || width <= eligible) shapes[key][count] = cache::add(demand.uses, width == 1 ? restore_bound : 0);
    }
    if (decode_bound && width <= policy_decoding_requests) {
      if constexpr (mtp) for (auto role : roles) {
        if (drafts_token(role) && (model.max_drafts() != 1 || model.recursive_mtp() ||
            width > policy_decoding_requests - dynamic_decoding_requests)) continue;
        auto &uses = shapes[{role, TextCompletion::kEmitNext}][drafts_token(role) ? 2 : 1]; uses = cache::add(uses, decode_bound);
      }
      else {
        auto &uses = shapes[{GraphRole::kText, TextCompletion::kEmitNext}][1];
        uses = cache::add(uses, decode_bound);
      }
    }
    if (mtp && decode_bound && width <= recursive_requests)
      for (std::size_t rows = 1; rows <= model.cache_plan.candidate_boundaries; ++rows)
        shapes[{GraphRole::kVerify, TextCompletion::kEmitNext}][rows] = decode_bound;
    // Committed-token reconstruction normalizes lag while retaining each
    // chunk's completion. Strict replay instead preserves its recorded role.
    if constexpr (mtp) if (width == 1)
      for (const auto completion : {TextCompletion::kEmitNext, TextCompletion::kStateOnly})
        for (const auto role : {GraphRole::kMtpCatchupPrefill, GraphRole::kMtpCatchupSingle}) {
          const auto normalized = role == GraphRole::kMtpCatchupPrefill ? GraphRole::kMtpPrefill : GraphRole::kMtpSingle;
          if (const auto *found = prompt_demand.find({role, completion}))
            for (const auto &[tokens, demand] : *found) {
              (void)demand; shapes[{normalized, completion}].try_emplace(tokens, restore_bound);
            }
        }
    if (width == 1 && !model.sampling_configuration(sampling_index).configuration.modifiers.empty()) {
      // A modifier history contains consumed IDs and the original pending/RNG,
      // not old bias/mask bytes. Prepare its state-only reconstruction before
      // storage freezes, including every bounded accepted verifier prefix.
      for (const auto &[key, entries] : prompt_demand.entries()) {
        const auto original = key.first;
        const auto role = original == GraphRole::kMtpCatchupPrefill ? GraphRole::kMtpPrefill
            : original == GraphRole::kMtpCatchupSingle ? GraphRole::kMtpSingle : original;
        if (!supports_state_only(role)) continue;
        for (const auto &[tokens, demand] : entries) {
          (void)demand;
          shapes[{role, TextCompletion::kStateOnly}].try_emplace(tokens, restore_bound);
        }
      }
    }
    for (const auto &[key, controls] : carrier_demand.entries())
      if (width <= Demands::eligible_members(controls, matches_member)) shapes.try_emplace(key);
    for (const auto &[key, entries] : shapes) {
      const auto [role, completion] = key;
      std::vector<std::size_t> chunks; chunks.reserve(entries.size());
      for (const auto &[tokens, uses] : entries) { (void)uses; chunks.push_back(tokens); }
      bool packed = false;
      // Known whole turns of one request already have exact cold variants.
      // Length-aware carriers are needed for batching, bounded chunks or an
      // unknown prefix suffix; do not replace uniform math merely because the
      // more general carrier is available.
      const bool variable_prefill = width > 1 || prefill.bounded() || prefix_enabled;
      if (variable_prefill && (role == GraphRole::kText || is_mtp_prefill(role)) && model.layer0_probe.empty()) {
        std::vector<std::size_t> capacities;
        if (const auto *controls = carrier_demand.find(key))
          Demands::visit_support(*controls, matches_member,
              [&](std::size_t rows, const Demands::Entry &, std::size_t eligible) {
                const auto physical = is_mtp_prefill(role) ? cache::add(rows, draft_lag(role)) : rows;
                if (width <= eligible && physical <= model.cache_plan.max_query_rows / width)
                  capacities.push_back(rows);
              });
        if (width == 1 && completion == TextCompletion::kStateOnly &&
            !model.sampling_configuration(sampling_index).configuration.modifiers.empty()) {
          // A dynamic prefix suffix may have used any real length within its
          // original EmitNext carrier. Rebuild can reuse that cold capacity
          // without retaining historical modifier values or admitting hot L.
          for (const auto &[source_key, source_controls] : carrier_demand.entries()) {
            const auto source_role = source_key.first == GraphRole::kMtpCatchupPrefill
                ? GraphRole::kMtpPrefill : source_key.first;
            if (source_role != role) continue;
            for (const auto &[rows, demand] : source_controls)
              if (rows <= model.cache_plan.max_query_rows && demand.supports(matches_member))
                capacities.push_back(rows);
          }
          std::sort(capacities.begin(), capacities.end());
          capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
        }
        if (!capacities.empty()) packed = model.prepare_prefill_family(capacities, completion, sampling_index, width, role);
        if (packed) {
          for (const auto rows : capacities) {
            const auto uses = carrier_demand.uses(key, rows);
            model.plan_replay_uses(rows, width, cache::add(uses, width == 1 ? restore_bound : 0),
                role, completion, sampling_index, RowPacking::kPrefill);
          }
          // Decode still has its original T1 mathematics and source ports.
          if (role == GraphRole::kText && completion == TextCompletion::kEmitNext && decode_bound && width <= policy_decoding_requests) {
            const std::array decode{std::size_t{1}};
            model.prepare_family(decode, false, width, role, false, completion, sampling_index);
            model.plan_replay_uses(1, width, cache::add(decode_bound, width == 1 ? restore_bound : 0), role, completion, sampling_index);
          }
        }
      }
      if (!packed && !chunks.empty()) {
        model.prepare_family(chunks, false, width, role, false, completion, sampling_index);
        for (const auto &[tokens, uses] : entries) model.plan_replay_uses(tokens, width,
            cache::add(uses, width == 1 ? restore_bound : 0), role, completion, sampling_index);
      }
    }
    if constexpr (mtp) if (decode_bound && width > 1 && width <= policy_decoding_requests &&
        width <= recursive_requests && model.target_kv_layers > 0) {
      const auto capacity = static_cast<std::size_t>(model.cache_plan.candidate_boundaries);
      const std::array capacities{capacity};
      if (model.prepare_family(capacities, false, width, GraphRole::kVerify, false, TextCompletion::kEmitNext,
          sampling_index, RowPacking::kVerify))
        model.plan_replay_uses(capacity, width, decode_bound, GraphRole::kVerify,
            TextCompletion::kEmitNext, sampling_index, RowPacking::kVerify);
    }
    if constexpr (mtp) if (decode_bound && width > 1 && width <= policy_decoding_requests &&
        model.max_drafts() == 1 && !model.recursive_mtp() && width <= policy_decoding_requests - dynamic_decoding_requests) {
      constexpr auto role = GraphRole::kMtpCatchupRound;
      if (model.prepare_family(pair, false, width, role, false, TextCompletion::kEmitNext,
          sampling_index, RowPacking::kDraft))
        model.plan_replay_uses(2, width, decode_bound, role, TextCompletion::kEmitNext, sampling_index, RowPacking::kDraft);
    }
  }
  }
  if (mtp && decode_bound)
    for (std::size_t index = 0; index < model.sampling_configuration_count(); ++index)
      for (const bool dynamic : {false, true})
        if ((dynamic || model.recursive_mtp()) && std::ranges::any_of(requests, [index, dynamic](const auto &request) {
            return request.sampling_index == index && bool(request.logits_provider) == dynamic &&
                std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
          })) model.prepare_recursive_round_programs(index, dynamic);
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
  const auto first_role = prompt_role(first_prefill[0]);
  const auto first_capacity = model.prefill_capacity(first_prefill[0], first_completion, requests.front().sampling_index, 1, first_role);
  if (first_capacity) {
    const std::array carrier{first_capacity};
    model.prepare_prefill_execution(carrier, first_completion, requests.front().sampling_index, 1, first_role);
  } else model.prepare_execution(first_prefill, 1, first_role, first_completion, requests.front().sampling_index);
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
  if (mtp && decode_bound && model.max_drafts() == 1 && !model.recursive_mtp())
    for (std::size_t index = 0; index < model.sampling_configuration_count(); ++index)
      if (std::ranges::any_of(requests, [index](const auto &request) {
          return request.sampling_index == index && !request.logits_provider &&
              std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
        })) for (const auto role : roles) if (drafts_token(role))
          model.prepare_execution(pair, 1, role, TextCompletion::kEmitNext, index);
  if constexpr (mtp) if (decode_overlap && decode_bound && model.cache_plan.in_flight_batches >= 2 &&
      model.max_drafts() == 1 && !model.recursive_mtp())
    for (std::size_t i = 0; i < model.sampling_configuration_count(); ++i) {
      const auto &sampling = model.sampling_configuration(i).configuration;
      if (sampling.enabled || !sampling.modifiers.empty() ||
          std::ranges::none_of(requests, [i](const auto &request) { return request.sampling_index == i; })) continue;
      for (const auto role : {GraphRole::kMtpRound, GraphRole::kMtpCatchupRound})
        model.prepare_execution(pair, 1, role, TextCompletion::kEmitNext, i);
      model.prepare_relay_execution(1, i, GraphRole::kMtpCatchupRound);
    }
  inference::RequestLoop request_loop;
  request_loop.prepare(requests.size(), model.cache_plan.resident_sessions);
  for (std::size_t index = 0; index < requests.size(); ++index)
    requests[index].request_key = request_loop.enqueue(index);
  std::vector<std::size_t> dispatch, dispatch_turns;
  dispatch.reserve(dispatch_bound); dispatch_turns.reserve(dispatch_bound);
  std::vector<std::size_t> evictions; evictions.reserve(total_turns);
  std::size_t completed = 0, admitted = 0;
  struct BatchObservation {
    std::size_t tokens, active_rows, physical_rows; GraphRole role; bool generation;
    hyle::container::SmallVector<std::size_t, 2> members, input_rows;
  };
  std::vector<BatchObservation> batches; batches.reserve(dispatch_bound);
  std::size_t actual_prefill_calls = 0, actual_decode_calls = 0;
  std::size_t actual_fresh_participations = 0, actual_fresh_target_rows = 0;
  double actual_step_sum = 0;
  std::size_t peak_resident = 0, peak_committed = 0;
  const auto resident_count = [&] {
    return static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [](const auto &request) { return bool(request.session); }));
  };
  inference::RequestBatchWorkspace<PreparedGenerationCall> batch_workspace;
  batch_workspace.prepare(model.cache_plan.resident_sessions, model.cache_plan.active_sequences);
  GenerationContinuation continuation; continuation.enabled = decode_overlap;
  if (decode_overlap) continuation.prepare(model.cache_plan.active_sequences);
  const auto ready = Clock::now();
  std::cerr << "generation_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n'
            << "measure_memory name=ready_before_first\n";
  const auto start = Clock::now(); auto endpoint = start;
  double prefill_seconds = 0, decode_seconds = 0;
  const auto observe_capacity = [&] {
    std::size_t candidates = 0;
    for (const auto *step : {model.active_step.get(), model.successor_step.get()})
      if (step) for (const auto &member : step->members) candidates += bool(member.next);
    const auto live = std::size_t(model.snapshot_versions.live);
    if (candidates > live) throw std::logic_error("generation candidates exceed their actual version ownership");
    const auto committed = live - candidates;
    if (committed > model.cache_plan.resident_sessions)
      throw std::runtime_error("generation committed version count exceeds N after publication");
    peak_resident = (std::max)(peak_resident, resident_count());
    peak_committed = (std::max)(peak_committed, committed);
  };
  const auto observe_batch = [&](const GenerationBatchFacts &facts) {
    const auto participations = cache::add(actual_fresh_participations, facts.members.size());
    auto rows = actual_fresh_target_rows;
    BatchObservation batch{facts.input_rows, facts.active_rows, facts.physical_rows, facts.role, facts.generation, {}, {}};
    if (!batch.members.try_reserve(facts.members.size()) || !batch.input_rows.try_reserve(facts.members.size()))
      throw std::bad_alloc{};
    for (const auto &member : facts.members) {
      const auto key = member.request->request_key;
      if (!request_loop.matches(key) || key.slot >= requests.size() || &requests[key.slot] != member.request)
        throw std::runtime_error("actual batch fact lost its original request identity");
      batch.members.push_back(key.slot);
      const auto member_rows = member.actual_target_rows();
      batch.input_rows.push_back(member_rows); rows = cache::add(rows, member_rows);
    }
    if (rows - actual_fresh_target_rows != facts.active_rows || facts.active_rows > facts.physical_rows)
      throw std::logic_error("actual batch row facts differ from its submitted physical carrier");
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
    auto finished = endpoint;
    for (const auto &member : facts.members)
      finished = (std::max)(finished, facts.generation ? member.request->endpoint : member.request->prefill_endpoint);
    const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
    (facts.generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
    observe_capacity();
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<BatchObserver>>) batch_observer(facts);
  };
  const auto execute_ready = [&](inference::RequestWave wave) {
    const auto prepare_call = [&](inference::RequestKey key) {
      auto &request = requests[key.slot];
      std::cerr << "generation_session id=" << request.id << " turn=" << request.progress.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      return prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests,
          request_loop.credit(key, request.progress.generated.size()), max_prefill_rows);
    };
    const auto batch_limit = [&](const PreparedGenerationCall &first) -> std::size_t {
      return first.full_hit ? std::size_t{1} : model.cache_plan.active_sequences;
    };
    const auto compatible_calls = [&](const PreparedGenerationCall &first, const PreparedGenerationCall &other) {
      return other.full_hit == first.full_hit && other.request->sampling_index == first.request->sampling_index &&
          other.generation == first.generation &&
          (other.role == first.role || (first.generation && drafts_token(first.role) && drafts_token(other.role))) &&
          other.completion == first.completion &&
          (other.planned_rows() == first.planned_rows() ||
            (!first.full_hit && first.generation && first.role == GraphRole::kVerify &&
             model.verification_capacity((std::max)(other.planned_rows(), first.planned_rows()),
                 2, first.request->sampling_index)) ||
           (!first.full_hit && !first.generation && (first.role == GraphRole::kText || is_mtp_prefill(first.role)) &&
            model.prefill_capacity((std::max)(other.planned_rows(), first.planned_rows()),
                first.completion, first.request->sampling_index, 2, first.role)));
    };
    const auto fits = [&](std::span<PreparedGenerationCall> members) {
      if (members[0].full_hit) return true;
      const bool unequal = std::ranges::any_of(members,
          [&](const auto &call) { return call.ids().size() != members[0].ids().size(); });
      const bool mixed_roles = std::ranges::any_of(members,
          [&](const auto &call) { return call.role != members[0].role; });
      if (mixed_roles && !model.find_child(members.size(), 2, GraphRole::kMtpCatchupRound, false,
          members[0].completion, members[0].request->sampling_index, RowPacking::kDraft)) return false;
      return (!unequal || generation_prefill_capacity(model, members) || generation_verification_capacity(model, members)) &&
          make_generation_batch_room(model, prefix.get(), members, requests);
    };
    const auto execute_batch = [&](std::span<PreparedGenerationCall> bucket, bool entire_wave) {
      std::cerr << "generation_batch requests=" << bucket.size() << " T=" << bucket[0].planned_rows()
                << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit << '\n';
      try {
        execute_generation_calls(model, bucket, trace_states, prefix.get(), requests,
            nullptr, decode_overlap ? &continuation : nullptr, observe_batch, entire_wave, wave.credits);
      } catch (const inference::EvaluationFailure &failure) {
        // Only the real admitted bucket may attribute a native member failure.
        throw inference::RequestExecutionFailure::from_evaluation(failure, bucket.size(),
            [&](std::size_t member) { return bucket[member].request->request_key; });
      }
      auto finished = endpoint;
      for (const auto &call : bucket) {
        const auto index = call.request->request_key.slot;
        if (call.full_hit) {
          // A cache hit is a logical dispatch, never an invented Graph.
          dispatch.push_back(index); dispatch_turns.push_back(requests[index].progress.turn_index);
        }
        finished = (std::max)(finished, requests[index].endpoint);
      }
      // Real Graph batches report their own completion, including a successor
      // observed on a later pump. A full hit has no Graph.
      if (bucket[0].full_hit) {
        const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
        (bucket[0].generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
      }
      observe_capacity();
    };
    const auto publish_call = [&](PreparedGenerationCall &call) { publish_generation_prompt(model, call, prefix.get()); };
    // These typed callable references bind the original model stages directly;
    // they hold no Model interface, runtime callbacks or execution authority.
    struct Policy final {
      decltype(prepare_call) &prepare;
      decltype(batch_limit) &limit;
      decltype(compatible_calls) &compatible;
      decltype(fits) &fit;
      decltype(execute_batch) &execute;
      decltype(publish_call) &publish;
    } policy{prepare_call, batch_limit, compatible_calls, fits, execute_batch, publish_call};
    inference::execute_request_batches(batch_workspace, wave.keys, policy);
    if (decode_overlap && continuation.ready)
      for (const auto &member : continuation.cohort.members())
        if (!request_loop.credit(member.key, requests[member.key.slot].progress.generated.size())) {
          continuation.release_ready(); break;
        }
  };
  const auto execute_wave = [&](inference::RequestWave wave, std::span<std::size_t> reserved) {
    return advance_generation_wave(model, wave, reserved, trace_states, continuation, execute_ready, nullptr, observe_batch);
  };
  GenerationRequestAdapter adapter{model, prefix.get(), std::span(requests), trace_states, &evictions, execute_wave,
      {}, decode_overlap ? &continuation : nullptr};
  while (completed != requests.size()) {
    const auto pumped = request_loop.pump_once(adapter);
    completed = cache::add(completed, pumped.completed);
    admitted = cache::add(admitted, pumped.admitted);
    if (!pumped.executed && !pumped.waiting_retirement && completed != requests.size())
      throw std::runtime_error("generation admission has no runnable resident request");
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  if ((actual_prefill_calls && !(prefill_seconds > 0)) || (actual_decode_calls && !(decode_seconds > 0)))
    throw std::logic_error("an actual generation phase has no observed completion interval");
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
    std::cout << ']';
    std::cout << ",\"input_rows\":[";
    for (std::size_t j = 0; j < batch.input_rows.size(); ++j) std::cout << (j ? "," : "") << batch.input_rows[j];
    std::cout << "],\"active_rows\":" << batch.active_rows << ",\"physical_rows\":" << batch.physical_rows << '}';
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
        "  --prefill-chunk-tokens N    Maximum real rows per natural prefill episode\n"
        "  --warmup-shapes lazy|planned  Prepare known carriers before ready (default: lazy)\n"
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
    bool warmup_planned_shapes = false, decode_overlap = true;
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
        // Internal benchmark/diagnostic override. Normal use selects the
        // automatic policy and falls back using actual execution facts.
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
    if (max_prefill_rows != SIZE_MAX && !natural_generation)
      throw std::runtime_error("--prefill-chunk-tokens requires natural generation; fixed replay is not supported");
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
      throw std::runtime_error("usage: hyle --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
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
        const auto chunk_rows = prefill.next(tokens).rows;
        planned_chunks.push_back(chunk_rows);
        // The first nonempty prefix leaves at most tokens-1 new rows. Later
        // turns can consume the full carrier plus their historical catch-up.
        if (mtp_enabled) mtp_schema_rows = (std::max)(mtp_schema_rows,
            qwen35::cache::add(chunk_rows, i || (prefix_enabled && tokens > chunk_rows) ? 1 : 0));
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
    bool overlap_selected = decode_overlap && natural_generation &&
        (!mtp_enabled || (max_drafts == 1 && min_draft_probability == 0)) && !progress &&
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
    if (overlap_selected && cache_bytes && plan_for(minimum_residents).backing_bytes > cache_bytes) {
      // Extra in-flight state is optional. A valid single-step configuration
      // must not be rejected merely because automatic overlap needs more room.
      overlap_selected = false;
      selected_cache = plan_for(cached_slots);
    }
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
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected);
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
            // This fixed-input CLI tape has one default sampling configuration.
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
            // The fixed-input tape retains that same CLI-wide configuration.
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
