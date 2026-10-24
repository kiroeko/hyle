// Independent text consumer. All candidate tensor mathematics uses public
// Compute/Graph operations; no native API, provider choice or oracle feedback.
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
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "schedule.h"
#include "session_schedule.h"
#include "session_cache.h"
#include "session_state.h"
#include "prefix_cache.h"
#include "batch_schedule.h"
#include "mtp_graph.h"
#include "mtp_schedule.h"
#include "hyle/container/small_vector.h"

#include "backend.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/progress.h"
#include "hyle/mem/buffer.h"
#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/sched/join.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"

namespace qwen35 {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using V = g::ValueId;
using D = g::Dimension;
constexpr D f(std::int64_t value) noexcept { return D::fixed(value); }
using Type = c::DataType;
constexpr std::int64_t hidden = 2048, intermediate = 6144, vocabulary = 248320;
constexpr char source_id[] = "4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155";
constexpr char score_policy[] = "bf16_inputs_f32_output";
using Clock = std::chrono::steady_clock;
enum class ReplayPolicy { kAuto, kCommands, kPreferCapture };
struct StepObservation {
  std::string name;
  std::size_t input_tokens = 0, visible = 0, capture_recorded = 0, capture_reused = 0;
  std::int32_t next = 0;
  double seconds = 0, submit_seconds = 0, step_seconds = 0, cold_batch_seconds = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool scheduled_progress = false;
  bool prefix_hit = false, graph_executed = true;
  std::size_t reused_tokens = 0, graph_input_tokens = 0;
};
struct PreparationObservation {
  std::size_t slot = 0, phase_slot_prior_uses = 0;
  double layout_seconds = 0, batch_prepare_seconds = 0, replay_prepare_seconds = 0;
  double report_snapshot_seconds = 0;
  std::array<std::uint32_t, 6> nodes{};
  std::array<c::BatchPlanReport, 6> reports{};
};
void print_step_observation(const StepObservation &step) {
  if (step.prefix_hit)
    std::cerr << "prefix_hit name=" << step.name << " reused_tokens=" << step.reused_tokens
              << " graph_calls=" << step.graph_executed << " visible=" << step.visible << '\n';
  if (step.graph_executed) std::cerr << "prepared T=" << (step.graph_input_tokens ? step.graph_input_tokens : step.input_tokens) << " visible=" << step.visible
            << " replay=" << (step.replay_route == g::ReplayRoute::kCaptured ? "captured"
                : step.replay_route == g::ReplayRoute::kHybrid ? "hybrid" : "commands")
            << " cold_batch_seconds=" << step.cold_batch_seconds
            << " capture_recorded=" << step.capture_recorded
            << " capture_reused=" << step.capture_reused << '\n';
  std::cerr << step.name << " tokens=" << step.input_tokens << " visible=" << step.visible
            << " seconds=" << step.seconds << " submit_seconds=" << step.submit_seconds
            << " step_seconds=" << step.step_seconds
            << " progress=" << (step.scheduled_progress ? "scheduler" : "direct")
            << " next=" << step.next << '\n';
}
void print_preparation_observation(const StepObservation &step,
                                   const PreparationObservation &preparation) {
  const auto precision = std::cerr.precision();
  std::cerr << std::setprecision(17)
            << "preparation name=" << step.name << " tokens=" << step.input_tokens
            << " visible=" << step.visible << " slot=" << preparation.slot
            << " phase_slot_prior_uses=" << preparation.phase_slot_prior_uses
            << " native_plan_first_use=" << (preparation.phase_slot_prior_uses == 0)
            << " layout_seconds=" << preparation.layout_seconds
            << " batch_prepare_seconds=" << preparation.batch_prepare_seconds
            << " replay_prepare_seconds=" << preparation.replay_prepare_seconds
            << " report_snapshot_seconds=" << preparation.report_snapshot_seconds
            << " snapshot_after_host_endpoint=1\n";
  for (std::size_t i = 0; i < preparation.reports.size(); ++i) {
    const auto &report = preparation.reports[i];
    const auto &upload = report.metadata_upload;
    std::cerr << "batch_plan name=" << step.name << " attention_index=" << i
              << " node=" << preparation.nodes[i] << " slot=" << preparation.slot
              << " layout_generation=" << report.layout_generation
              << " metadata_staging_bytes=" << report.metadata_staging_bytes
              << " float_workspace_bytes=" << report.float_workspace_bytes
              << " int_workspace_bytes=" << report.int_workspace_bytes
              << " cta_tile_q=" << report.cta_tile_q
              << " padded_batch_size=" << report.padded_batch_size
              << " split_kv=" << report.split_kv
              << " native_metadata_unknown=" << report.native_metadata_unknown
              << " capture_compatible=" << report.capture_compatible
              << " bf16_partials=" << report.bf16_partials
              << " reused_native_plan=" << report.reused_native_plan
              << " host_waits=" << report.host_waits
              << " upload_resources=" << upload.resources << " upload_copies=" << upload.copies
              << " upload_active_copies=" << upload.active_copies << " upload_phases=" << upload.phases
              << " upload_no_op_copies=" << upload.no_op_copies
              << " upload_logical_bytes=" << upload.logical_bytes
              << " upload_physical_bytes=" << upload.physical_bytes
              << " upload_staging_bytes=" << upload.staging_bytes
              << " upload_metadata_bytes=" << upload.metadata_bytes
              << " upload_native_calls=" << upload.native_calls
              << " upload_native_waits=" << upload.native_waits
              << " upload_phase_completions=" << upload.phase_completions
              << " upload_host_waits=" << upload.host_waits
              << " upload_native_metadata_unknown=" << upload.native_metadata_unknown << '\n';
  }
  std::cerr.precision(precision);
}
constexpr bool is_end_of_generation(std::int32_t token) noexcept {
  return token == 248044 || token == 248046; // Original endoftext and tokenizer im_end.
}
constexpr std::array<const char *, 9> layer0_probe_names{
    "01-gdn-output", "02-out-proj-input", "03-out-proj-output",
    "04-down-proj-input", "05-layer-output", "06-post-attn-residual",
    "07-post-attn-norm", "08-gate-proj-output", "09-up-proj-output"};

void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return static_cast<T &&>(*result);
}
c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())));
}
struct Tensor final {
  c::TensorDesc description;
  mem::Buffer buffer;
  c::ReadLease ready;
  Tensor() = default;
  Tensor(mem::MemoryResourceRef resource, c::TensorDesc desc, bool caller = true)
      : description(desc) {
    if (caller) buffer = take(mem::Buffer::create(resource, take(desc.required_bytes()), 256));
  }
  c::TensorView view() { return take(c::TensorView::create(description, buffer.view())); }
  c::ConstTensorView read() {
    return ready ? take(ready.tensor(description)) : view().as_const();
  }
  void freeze(const Backend &backend) {
    check(backend.freeze(buffer, ready));
  }
  void zero() { check(buffer.view().fill(std::byte{})); }
  template<class T> void write(std::span<const T> values) {
    check(buffer.view().write(std::as_bytes(values)));
  }
  void dump(const std::filesystem::path &path) {
    std::vector<std::byte> bytes(buffer.size());
    check(buffer.view().read(bytes));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write trace: " + path.string());
  }
};
struct StateSchema final {
  c::TensorDesc history, recurrent;
};
std::array<StateSchema, 24> state_schema_for(std::size_t capacity) {
  if (capacity == 0 || capacity > 262144) throw std::runtime_error("capacity outside model context limit");
  std::array<StateSchema, 24> result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto &layer = result[i];
    if (i % 4 != 3) {
      layer.history = dense(Type::kBFloat16, {1, 3, 6144});
      layer.recurrent = dense(Type::kFloat32, {1, 16, 128, 128});
    }
  }
  return result;
}
std::size_t recurrent_bytes(const std::array<StateSchema, 24> &schema) {
  std::size_t bytes = 0;
  const auto add = [&](const c::TensorDesc &description, std::size_t copies) {
    const auto size = take(description.required_bytes());
    if (size > ((std::numeric_limits<std::size_t>::max)() - bytes) / copies)
      throw std::runtime_error("session cache byte count overflows");
    bytes += size * copies;
  };
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (i % 4 != 3) { add(schema[i].history, 1); add(schema[i].recurrent, 1); }
  }
  return bytes;
}
struct Recipe final {
  g::GraphBuilder builder = take(g::GraphBuilder::create());
  std::vector<g::TensorSpec> specs;
  std::vector<g::ConstantBinding> constants;
  std::map<std::string, V> weights;
  std::map<std::string, Tensor> &bank;
  std::vector<Tensor> literals;
  mem::MemoryResourceRef resource;
  const Backend &backend;
  c::StorageOwner owner;
  g::DimensionId last_row;
  D token_rows, packed_rows, sample_rows;
  Recipe(std::map<std::string, Tensor> &selected, const Backend &environment,
         std::size_t max_query_rows, std::size_t requests = 1)
      : bank(selected), resource(environment.device()), backend(environment), owner(environment.storage_owner()) {
    // L is a zero-based last-token position, independently bound at startup.
    // T=L+1 avoids subtraction or a second independent (possibly inconsistent)
    // shape binding. All other numeric axes remain fixed model dimensions.
    last_row = take(builder.dimension({0, static_cast<std::int64_t>(max_query_rows - 1), 1}));
    token_rows = D(take(builder.dimension_sum(D(last_row), f(1))));
    packed_rows = requests == 1 ? token_rows :
        D(take(builder.dimension_product(f(static_cast<std::int64_t>(requests)), token_rows)));
    sample_rows = D(take(builder.dimension_sum(token_rows, f(3))));
  }
  static g::TensorSpec dense_spec(Type type, std::span<const D> dimensions) {
    return take(g::TensorSpec::dense(type, dimensions));
  }
  static g::TensorSpec dense_spec(Type type, std::initializer_list<D> dimensions) {
    return dense_spec(type, std::span(dimensions.begin(), dimensions.size()));
  }
  const g::TensorSpec &spec(V value) const { return specs.at(value.ordinal()); }
  V remember(V value, g::TensorSpec description) {
    if (specs.size() <= value.ordinal()) specs.resize(value.ordinal() + 1);
    specs[value.ordinal()] = description;
    return value;
  }
  V input(g::TensorSpec description) { return remember(take(builder.input(description)), description); }
  V input(c::TensorDesc description) { return input(take(g::TensorSpec::concrete(description))); }
  V constant(Tensor &tensor) {
    const auto value = remember(take(builder.constant(tensor.description)),
                                take(g::TensorSpec::concrete(tensor.description)));
    tensor.freeze(backend);
    // Native resources are explicitly borrowed from Model's Backend. The
    // bank/literal owns its lease through every compiled child; the optional
    // StorageOwner+lease pair is supplied only when that owner exists.
    constants.push_back({value, tensor.read(), g::ConstantPolicy::kBorrow,
                         owner, owner ? tensor.ready : c::ReadLease{}});
    return value;
  }
  V weight_absolute(const std::string &name) {
    auto found = weights.find(name);
    if (found != weights.end()) return found->second;
    const auto value = constant(bank.at(name));
    weights.emplace(name, value);
    return value;
  }
  V weight(const std::string &name) {
    return weight_absolute("model.language_model." + name);
  }
  V scalar(float value, Type type = Type::kFloat32) {
    literals.emplace_back(resource, dense(type, {1}));
    if (type == Type::kFloat32) literals.back().write<float>(std::span(&value, 1));
    else {
      auto bits = std::bit_cast<std::uint32_t>(value);
      bits += 0x7fffU + ((bits >> 16) & 1U);
      const auto bf16 = static_cast<std::uint16_t>(bits >> 16);
      literals.back().write<std::uint16_t>(std::span(&bf16, 1));
    }
    return constant(literals.back());
  }
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    std::initializer_list<g::TensorSpec> outputs,
                    g::NodeId *node_id = nullptr) {
    const auto node = take(builder.add_node(operation,
        std::span(arguments.begin(), arguments.size()),
        std::span(outputs.begin(), outputs.size())));
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
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments) {
    std::vector<c::TensorDesc> in;
    for (auto value : arguments) in.push_back(take(spec(value).concrete_description()));
    std::vector<c::TensorDesc> out(c::operation_traits(operation).outputs);
    check(c::infer_output_descriptions(operation, in, out));
    const auto node = take(builder.add_node(operation, std::span(arguments.begin(), arguments.size())));
    std::vector<V> result;
    for (std::size_t i = 0; i < out.size(); ++i)
      result.push_back(remember(take(builder.result(node, i)), take(g::TensorSpec::concrete(out[i]))));
    return result;
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments) {
    return op(c::Operation(operation), arguments).at(0);
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
    const auto output = take(builder.add(view, x));
    return remember(output, take(g::TensorSpec::inferred(spec(x).data_type(),
        std::span(dimensions.begin(), dimensions.size()), spec(x).storage_format())));
  }
  V copy(V x) { return same(c::Copy{}, {x}, x); }
  V concat(std::span<const V> inputs, int axis, g::TensorSpec output) {
    if (inputs.size() == 1) return inputs.front();
    const std::array<g::TensorSpec, 1> outputs{output};
    const auto node = take(builder.add_node(c::Operation(c::Concat{.axis = axis}), inputs, outputs));
    return remember(take(builder.result(node)), std::move(output));
  }
  V slice(V x, int axis, D first, D count) {
    const auto source = spec(x);
    std::vector<D> dimensions(source.dimensions().begin(), source.dimensions().end());
    dimensions.at(static_cast<std::size_t>(axis)) = count;
    const auto output = take(builder.add(take(g::ViewSpec::slice(axis, first, count)), x));
    return remember(output, take(g::TensorSpec::inferred(source.data_type(), dimensions,
                                                       source.storage_format())));
  }
  V linear(V x, const std::string &name) {
    constexpr std::array<std::int32_t, 2> axes{1, 0};
    const auto original = weight(name);
    auto dimensions = std::vector<D>(spec(x).dimensions().begin(), spec(x).dimensions().end());
    dimensions.back() = spec(original).dimensions()[0];
    auto w = one(take(c::View::permute(axes)), {original});
    return one(c::MatMul{}, {x, w}, dense_spec(spec(x).data_type(), dimensions));
  }
  V scores(V x, const std::string &absolute_weight_name) {
    const auto stored = weight_absolute(absolute_weight_name);
    const auto &input = spec(x), &weight = spec(stored);
    if (input.rank() != 2 || weight.rank() != 2 ||
        input.data_type() != Type::kBFloat16 || weight.data_type() != Type::kBFloat16 ||
        input.dimensions()[1] != weight.dimensions()[1])
      throw std::runtime_error("scores require BF16 [rows,K] and stored [vocabulary,K] weights");
    return one(c::model_ops::bfloat_linear_f32(), {x, stored},
        dense_spec(Type::kFloat32, {input.dimensions()[0], weight.dimensions()[0]}));
  }
  V norm(V x, const std::string &name) {
    const auto type = spec(x).data_type();
    auto scale = binary(c::BinaryKind::kAdd, cast(weight(name), Type::kFloat32), scalar(1));
    const auto converted = cast(x, Type::kFloat32);
    return cast(same(c::RmsNorm{.epsilon = 1.e-6F}, {converted, scale}, converted), type);
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
    sum = binary(c::BinaryKind::kAdd, sum, scalar(1.e-6F, Type::kBFloat16));
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
  void publish(V value) { check(builder.publish(value)); }
};

enum class GraphRole { kText, kVerify, kMtpPrefill, kMtpRound, kMtpCatchupRound,
                       kMtpSingle, kMtpCatchupSingle, kMtpCatchupPrefill };
constexpr bool has_mtp(GraphRole role) noexcept { return role >= GraphRole::kMtpPrefill; }
constexpr bool is_mtp_prefill(GraphRole role) noexcept {
  return role == GraphRole::kMtpPrefill || role == GraphRole::kMtpCatchupPrefill;
}
constexpr bool drafts_token(GraphRole role) noexcept {
  return role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound;
}
constexpr bool verifies_two(GraphRole role) noexcept {
  return role == GraphRole::kVerify || drafts_token(role);
}
constexpr std::size_t draft_lag(GraphRole role) noexcept {
  return role == GraphRole::kMtpCatchupRound || role == GraphRole::kMtpCatchupSingle ||
      role == GraphRole::kMtpCatchupPrefill ? 1u : 0u;
}
struct MtpBoundary final {
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t next_id = -1, catchup_token = -1;
  friend bool operator==(const MtpBoundary &, const MtpBoundary &) = default;
};
MtpBoundary mtp_boundary(const cache::Snapshot &head) noexcept {
  return {head.length, head.draft ? head.draft->frontier : 0,
      head.draft ? head.draft->lag : 0, head.next_id,
      head.draft ? head.draft->catchup_token : -1};
}
struct MtpReplayRecord final {
  GraphRole role = GraphRole::kMtpPrefill;
  bool generation = false, committed = false, user_output = true;
  bool imported = false;
  std::shared_ptr<const void> input_owner; // Immutable host data, never a Snapshot/lease.
  std::span<const std::int32_t> prompt_ids;
  std::int32_t pending_input = -1;
  std::size_t quota = 0, turn = 0, accepted = 0, draft_rows = 0;
  std::int32_t draft = -1, target_first = -1, target_second = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  std::array<std::int32_t, 2> emitted{-1, -1};
  std::array<std::int32_t, 2> accepted_inputs{-1, -1};
  std::size_t requests = 1, member_index = 0;
  std::size_t emitted_count = 0;
  MtpBoundary before, after;
  double step_seconds = 0;
  Clock::time_point endpoint{};
  std::span<const std::int32_t> ids() const noexcept {
    return generation ? std::span(&pending_input, 1) : prompt_ids;
  }
  std::span<const std::int32_t> committed_ids() const noexcept {
    if (generation && (accepted == 0 || accepted > accepted_inputs.size())) return {};
    return generation ? std::span<const std::int32_t>(accepted_inputs).first(accepted) : prompt_ids;
  }
  bool same_result(const MtpReplayRecord &other) const noexcept {
    return requests == other.requests && member_index == other.member_index &&
        role == other.role && generation == other.generation && quota == other.quota &&
        before == other.before && after == other.after && accepted == other.accepted &&
        draft_rows == other.draft_rows && draft == other.draft && target_first == other.target_first &&
        target_second == other.target_second && agreed == other.agreed && accepted_draft == other.accepted_draft &&
        stopped == other.stopped && emitted == other.emitted && emitted_count == other.emitted_count;
  }
};
struct MtpReplayLedger final {
  std::vector<MtpReplayRecord> records;
  std::size_t committed_count = 0;
  explicit MtpReplayLedger(std::size_t capacity) : records(capacity) {}
  MtpReplayRecord &next() {
    if (committed_count >= records.size() || records[committed_count].committed)
      throw std::runtime_error("MTP replay ledger has no uncommitted reserved record");
    return records[committed_count];
  }
};
struct FamilyKey final {
  std::size_t requests;
  GraphRole role;
  FamilyKey(std::size_t count, GraphRole selected = GraphRole::kText) noexcept
      : requests(count), role(selected) {}
  auto operator<=>(const FamilyKey &) const = default;
};
struct ExecutionShape final {
  std::size_t requests, tokens;
  GraphRole role = GraphRole::kText;
  auto operator<=>(const ExecutionShape &) const = default;
};
struct StatePorts final {
  V history, recurrent, next_history, next_recurrent;
  V first_history, first_recurrent; // Verify row0 owns separate material storage.
};
struct Ports final {
  V key, value;
  std::vector<StatePorts> sequences;
};
struct FamilyPorts {
  V token_port, position_port, last_ids_port, metadata_port, logits_port, id_port, hidden_port;
  std::array<g::NodeId, 6> attention_nodes;
  std::array<g::NodeId, 14> append_nodes;
  V draft_key_port, draft_value_port, draft_metadata_port, draft_position_port;
  V carry_input_port, prior_input_port, catchup_token_port, draft_id_port;
  std::vector<std::array<V, 2>> draft_hidden_inputs;
  g::NodeId draft_attention_node;
  std::vector<V> state_resource_ports;
  std::vector<std::array<V, 2>> carry_ports;
  std::array<V, 9> probe_ports;
  std::array<Ports, 24> layers;
};
struct FamilyChild final {
  g::CompiledGraph compiled; // One immutable selected child shared by every session.
  std::vector<bool> caller_outputs;
  bool caller_output(V value) const noexcept {
    return value.ordinal() < caller_outputs.size() && caller_outputs[value.ordinal()];
  }
};
struct Family final : FamilyPorts {
  std::size_t requests = 1;
  GraphRole role = GraphRole::kText;
  std::vector<Tensor> literals; // Backing survives every child and its slots.
  std::vector<g::TensorSpec> specs;
  c::BatchSchema attention_schema;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::DimensionId last_row;
  std::vector<std::size_t> chunks;
  std::vector<std::unique_ptr<FamilyChild>> children;
};
// Model admission holds the whole carrier through native retirement and final
// readback. Batch width and retained state boundaries do not add in-flight work.
constexpr std::size_t execution_slot_count = 1;
struct Phase final : FamilyPorts {
  std::size_t query_rows = 0;
  c::BatchSchema attention_schema;
  c::BatchLayoutVersion attention_layout;
  // The selected child is shared immutable model data, not copied per session.
  const FamilyChild *child = nullptr;
  const Family *family = nullptr;
  std::size_t variant_index = 0;
  std::array<g::Invocation, execution_slot_count> slots;
  std::array<std::size_t, execution_slot_count> planned_slot_uses{};
  std::array<std::size_t, execution_slot_count> preparation_uses{};
  std::array<g::ReplayMode, execution_slot_count> replay_modes{};
  Tensor tokens, positions, last_ids, logits, next_id, final_hidden;
  Tensor draft_positions, catchup_token, draft_id;
  std::array<Tensor, 9> probe_outputs;
  std::array<bool, 9> probe_caller{};
};

// One original Graph episode owns all dirty pages and candidate state. Until
// finish replaces the Session head, neither a ready layer nor an error changes
// the committed snapshot. Failure recovery remains attached to that Session.
class Session;
struct StepMember final {
  std::shared_ptr<Session> session;
  cache::SnapshotRef base;
  std::shared_ptr<cache::Snapshot> next;
  std::array<c::PagedPageAppender, 12> next_tails;
  std::array<c::PagedPageAppender, 2> next_draft_tails;
  c::BatchLayoutVersion layout;
  c::BatchLayoutVersion draft_layout;
  std::size_t accepted_tokens = 0;
  std::uint32_t verified_last_page = 0;
  std::int32_t first_input = -1, draft_token = -1;
  MtpReplayRecord *commit_record = nullptr;
  MtpReplayLedger *commit_ledger = nullptr;
};
struct StepState final {
  // Model owns the episode; these strong pins outlive all accepted native work.
  // Session's pending link borrows this owner and never creates a cycle.
  hyle::container::SmallVector<StepMember, 1> members;
  std::array<std::vector<c::PagedPageWriter>, 14> dirty;
  // One inherited partial page per request/bank. The common R1 case stays inline.
  std::array<hyle::container::SmallVector<c::PagedPageAppender, 1>, 14> tails;
  std::array<g::batch::PagedAppendPages, 14> appends;
  std::array<g::NodeId, 14> nodes;
  std::array<bool, 14> bound{};
  c::BatchLayoutVersion draft_layout;
  g::Invocation *run = nullptr;
  Phase *phase = nullptr;
  c::Completion completion;
  hyle::Status terminal;
  Clock::time_point step_start{}, start{};
  double submit_seconds = 0, cold_batch_seconds = 0;
  std::size_t input_tokens = 0, capture_recorded = 0, capture_reused = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool prepared = false, submitted = false, cancelled = false;
  // Model admission is serial even when a cancelled native episode has not
  // retired. Return it only after publication and final result reads, or after
  // complete abort cleanup.
  bool *candidate_admission = nullptr;
  void release_admission() noexcept {
    if (candidate_admission) *std::exchange(candidate_admission, nullptr) = false;
  }

  hyle::Result<bool> try_abort() noexcept {
    if (completion.valid()) {
      auto observed = completion.poll();
      if (observed && *observed == c::CompletionPoll::kPending) return false;
      if (!observed) terminal.Update(observed.error());
      completion = {}; // Error/abandonment keeps actual native guards retained.
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
      *run = {}; // Release failed state-output writers and old input bindings.
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
    for (auto &member : members) { member.next.reset(); member.base.reset(); }
    run = nullptr; phase = nullptr;
    release_admission();
    return true;
  }
};

// Each original device allocation becomes one retained lease pool;
// bounded pinned staging is the only extra payload storage.
struct WeightSourceFile final {
  hyle::io::File file;
  std::uint64_t capacity = 0;
};
hyle::sched::Task<hyle::Result<WeightSourceFile>> open_weight_source(
    mem::AllocatorRef, hyle::io::IoContext *io, std::string path) {
  auto opened = co_await hyle::io::open_file(*io, 0, path,
      {.access = hyle::io::FileAccess::kRead});
  if (!opened) co_return std::unexpected(std::move(opened).error());
  auto file = std::move(*opened);
  auto information = co_await file.info();
  if (!information) co_return std::unexpected(std::move(information).error());
  if (!information->is_regular())
    co_return std::unexpected(hyle::Status::InvalidArgument("checkpoint is not a regular file"));
  co_return WeightSourceFile{std::move(file), information->size};
}
hyle::sched::Task<hyle::Status> close_weight_source(
    mem::AllocatorRef, hyle::io::File *file) {
  co_return co_await file->close();
}
struct WeightLoadReport final {
  std::size_t bytes = 0;
  c::FileStreamReport stream;
};
enum class WeightSelection { kText, kTextAndMtp };
WeightLoadReport load_weight_bank(const std::filesystem::path &manifest,
    Backend &backend, std::map<std::string, Tensor> &weights,
    std::vector<c::LeasePool> &weight_pools,
    WeightSelection selection = WeightSelection::kText) {
  if (selection != WeightSelection::kText && selection != WeightSelection::kTextAndMtp)
    throw std::runtime_error("invalid weight selection");
  const bool include_mtp = selection == WeightSelection::kTextAndMtp;
  const std::uint32_t expected_count = include_mtp ? 335u : 320u;
  const std::size_t expected_bytes = include_mtp ? 3885311680ULL : 3763655360ULL;
  std::ifstream input(manifest);
  std::string signature, shard, identity;
  std::getline(input, signature); std::getline(input, shard); std::getline(input, identity);
  if (signature != (include_mtp ? "HYLE_QWEN35_TEXT_MTP_V1" : "HYLE_QWEN35_TEXT_V1") || identity != source_id)
    throw std::runtime_error("unidentified local weight manifest");

  auto scheduler_status = hyle::Status::Ok();
  hyle::sched::Scheduler loader_scheduler(2, mem::system_allocator(), 256,
                                        &scheduler_status);
  check(scheduler_status);
  hyle::io::IoContext io(loader_scheduler, {.max_blocking_threads = 2});
  struct Shutdown final {
    hyle::sched::Scheduler &scheduler;
    ~Shutdown() { scheduler.shutdown(); }
  } shutdown{loader_scheduler}; // IO's hooks outlive the stopped workers.
  check(io.status());
  auto open_task = open_weight_source(mem::system_allocator(), &io, shard);
  if (!open_task.valid()) throw std::runtime_error("cannot allocate weight open task");
  auto source_file = take(take(hyle::sched::spawn_task_on(loader_scheduler, 0,
      mem::system_allocator(), std::move(open_task)).join()));
  // Borrow this stable File so the actual async close can be observed explicitly
  // after pipeline retirement, before any loader runtime leaves this function.
  auto source = take(c::StreamFile::borrow(source_file.file, source_file.capacity));
  std::vector<c::StreamReadRange> ranges;
  std::vector<std::string> names;
  ranges.reserve(expected_count); names.reserve(expected_count); weight_pools.reserve(expected_count);
  // Original raw checkpoint parameters; MTP shares the target embedding/head.
  struct MtpShape { std::string_view name; std::int64_t rows, columns; };
  constexpr std::array mtp_shapes{
      MtpShape{"mtp.fc.weight", 2048, 4096},
      MtpShape{"mtp.layers.0.input_layernorm.weight", 2048, 0},
      MtpShape{"mtp.layers.0.mlp.down_proj.weight", 2048, 6144},
      MtpShape{"mtp.layers.0.mlp.gate_proj.weight", 6144, 2048},
      MtpShape{"mtp.layers.0.mlp.up_proj.weight", 6144, 2048},
      MtpShape{"mtp.layers.0.post_attention_layernorm.weight", 2048, 0},
      MtpShape{"mtp.layers.0.self_attn.k_norm.weight", 256, 0},
      MtpShape{"mtp.layers.0.self_attn.k_proj.weight", 512, 2048},
      MtpShape{"mtp.layers.0.self_attn.o_proj.weight", 2048, 2048},
      MtpShape{"mtp.layers.0.self_attn.q_norm.weight", 256, 0},
      MtpShape{"mtp.layers.0.self_attn.q_proj.weight", 4096, 2048},
      MtpShape{"mtp.layers.0.self_attn.v_proj.weight", 512, 2048},
      MtpShape{"mtp.norm.weight", 2048, 0},
      MtpShape{"mtp.pre_fc_norm_embedding.weight", 2048, 0},
      MtpShape{"mtp.pre_fc_norm_hidden.weight", 2048, 0}};
  std::size_t bytes = 0, largest = 0, mtp_count = 0, mtp_bytes = 0;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string name, type; std::size_t rank = 0, offset = 0, count = 0;
    fields >> name >> type >> rank;
    if (rank == 0 || rank > c::kMaxTensorRank || (type != "BF16" && type != "F32"))
      throw std::runtime_error("invalid manifest tensor");
    std::vector<std::int64_t> dimensions(rank);
    for (auto &dimension : dimensions) fields >> dimension;
    fields >> offset >> count;
    auto desc = take(c::TensorDesc::dense(type == "BF16" ? Type::kBFloat16 : Type::kFloat32, dimensions));
    if (!fields || take(desc.required_bytes()) != count)
      throw std::runtime_error("manifest tensor extent differs");
    if (name.starts_with("mtp.")) {
      const auto shape = std::find_if(mtp_shapes.begin(), mtp_shapes.end(),
          [&](const auto &candidate) { return candidate.name == name; });
      if (!include_mtp || shape == mtp_shapes.end() || type != "BF16" ||
          rank != (shape->columns ? 2u : 1u) || dimensions[0] != shape->rows ||
          (shape->columns && dimensions[1] != shape->columns))
        throw std::runtime_error("identified MTP tensor differs: " + name);
      ++mtp_count; mtp_bytes += count;
    } else if (!name.starts_with("model.language_model.")) {
      throw std::runtime_error("weight name is outside the selected text inventory");
    }
    if (offset > source_file.capacity || count > source_file.capacity - offset)
      throw std::runtime_error("short checkpoint tensor: " + name);
    if (count > (std::numeric_limits<std::size_t>::max)() - bytes)
      throw std::runtime_error("text weight byte inventory overflows");
    Tensor tensor(backend.device(), desc); // One original device allocation per selected tensor.
    auto pool = take(c::LeasePool::adopt(std::move(tensor.buffer), {.max_blocks = 1}));
    auto writer = take(pool.allocate(count));
    ranges.push_back({.file_offset = offset, .destination = std::move(writer),
        .bytes = count, .description = desc});
    weight_pools.push_back(std::move(pool)); // Admission stays open for Graph reads.
    if (!weights.emplace(name, std::move(tensor)).second)
      throw std::runtime_error("duplicate weight");
    names.push_back(std::move(name)); bytes += count; largest = (std::max)(largest, count);
  }
  if (weights.size() != expected_count || bytes != expected_bytes ||
      mtp_count != (include_mtp ? mtp_shapes.size() : 0u) ||
      mtp_bytes != (include_mtp ? 121656320u : 0u))
    throw std::runtime_error("selected text/MTP weight inventory differs");
  std::vector<c::StreamReadGroup> groups;
  groups.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i)
    groups.push_back({i, std::span(ranges).subspan(i, 1)});
  constexpr std::size_t staging_per_slot = 8u * 1024u * 1024u;
  constexpr std::uint32_t slots = 2;
  if (bytes > (std::numeric_limits<std::size_t>::max)() - slots * staging_per_slot)
    throw std::runtime_error("weight loader backing capacity overflows");
  const std::array targets{c::CopyResourceSpec{backend.device(), largest}};
  auto pipeline = take(backend.create_file_pipeline(io, targets,
      {.max_requests = 1, .max_groups_per_request = expected_count, .max_ranges_per_group = 1,
       .in_flight_slots = slots, .staging_bytes_per_slot = staging_per_slot,
       .max_active_backing_bytes = bytes + slots * staging_per_slot}));
  struct Drain final {
    c::FileResourcePipeline &pipeline;
    ~Drain() { pipeline.drain().IgnoreError(); }
  } drain{pipeline};
  auto ticket = take(pipeline.read(c::StreamSource::file(source), groups));
  std::vector<bool> delivered(groups.size(), false);
  for (std::size_t count = 0; count < groups.size(); ++count) {
    auto group = take(ticket.wait_next_ready());
    if (group.tag() >= names.size() || group.range_count() != 1)
      throw std::runtime_error("weight stream returned an unexpected group");
    const auto index = static_cast<std::size_t>(group.tag());
    if (delivered[index]) throw std::runtime_error("weight stream returned a duplicate group");
    auto ready = take(group.output(0));
    if (ready.size() != ranges[index].bytes)
      throw std::runtime_error("weight stream returned an incomplete tensor");
    weights.at(names[index]).ready = std::move(ready);
    delivered[index] = true;
  }
  const auto result = ticket.wait();
  check(result.status);
  if (!result.complete || !result.retirement_proven || result.read_bytes != bytes ||
      result.copied_bytes != bytes || result.ready_groups != groups.size())
    throw std::runtime_error("weight stream byte inventory or retirement differs");
  ticket = {};
  check(pipeline.drain());
  const auto report = pipeline.report();
  pipeline = {}; // Drop fixed staging, native slots and internal source owners now.
  source.reset();
  auto close_task = close_weight_source(mem::system_allocator(), &source_file.file);
  if (!close_task.valid()) throw std::runtime_error("cannot allocate weight close task");
  check(take(hyle::sched::spawn_task_on(loader_scheduler, 0,
      mem::system_allocator(), std::move(close_task)).join()));
  input.close();
  // Returning destroys the temporary Scheduler/IoContext and joins their actual
  // workers. The constructor's original weight/TTFT timers still surround all of it.
  return {bytes, report};
}

class Model;
// A Session borrows its Model. Destroy all sessions (and retire their episodes)
// before the shared model's Context, weights, families and progress runtimes.
class Session final : public std::enable_shared_from_this<Session> {
  friend class Model;
  Model *model_;
  explicit Session(Model &model) noexcept : model_(&model) {}
public:
  cache::SnapshotRef head;
  std::array<c::PagedPageAppender, 12> tails;
  std::array<c::PagedPageAppender, 2> draft_tails;
  const cache::Snapshot *tail_head = nullptr;
  const cache::Snapshot *draft_tail_head = nullptr;
  StepState *pending = nullptr; // Borrowed from Model; episode pins this Session.
  bool failed = false, stopping = false;
  hyle::Status retirement_status;
  std::size_t length() const noexcept { return head ? head->length : 0; }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  // Stopping any member aborts the whole shared batch. No head is published.
  hyle::Status stop() noexcept;
  hyle::Result<bool> try_retire() noexcept;
  hyle::Status drain() noexcept;
  ~Session();
};
struct EvaluationRequest final {
  std::shared_ptr<Session> session;
  std::span<const std::int32_t> ids;
};
struct EvaluationOutput final {
  std::shared_ptr<Session> session;
  std::string_view name;
  const std::filesystem::path *trace = nullptr;
  bool trace_states = true;
  double *complete_seconds = nullptr;
  Clock::time_point *host_endpoint = nullptr;
  StepObservation *observation = nullptr;
  PreparationObservation *preparation = nullptr;
  std::string_view probe_subdirectory;
  MtpReplayLedger *mtp_ledger = nullptr;
  const MtpReplayRecord *mtp_expected = nullptr;
};
struct MtpRoundResult final {
  std::int32_t pending = -1, draft = -1, target_first = -1, target_second = -1, next_id = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  std::size_t processed_target = 0, processed_draft = 0;
  std::vector<std::int32_t> emitted;
};
struct MtpGenerationSelection final {
  std::shared_ptr<Session> session;
  std::int32_t expected_pending = -1;
  std::size_t output_quota = 0;
};

class Model final {
public:
  Backend execution = take(Backend::create());
  // These original runtimes outlive every phase/slot and its native retirement.
  std::unique_ptr<hyle::sched::Scheduler> scheduler;
  std::unique_ptr<hyle::sched::BlockingPool> workers;
  std::vector<c::LeasePool> weight_pools; // Retain original allocations through all Graph uses.
  std::map<std::string, Tensor> weights;
  std::array<StateSchema, 24> state_schema;
  c::ResourceBudget cache_budget;
  c::ResourceBudget execution_budget; // Opaque native execution is separate from explicit cache pools.
  std::size_t known_cache_backing_bytes = 0;
  bool state_storage_prepared = false;
  std::array<bool, 24> history_writers{}, recurrent_writers{};
  bool hidden_writers = false;
  cache::Plan cache_plan;
  c::BatchSchema attention_schema;
  cache::VersionCount snapshot_versions;
  bool carrier_borrowed = false;
  static constexpr std::size_t slot_index = 0;
  cache::PageInventory page_inventory;
  cache::PageInventory draft_inventory;
  std::array<cache::StatePools, 24> state_pools;
  c::LeasePool hidden_pool;
  // Family/child owners are shared by sessions; no session addresses enter compile.
  std::map<FamilyKey, std::unique_ptr<Family>> families, probe_families;
  // One carrier per actually prepared finite shape. Serial admission borrows
  // its one slot and shared IO through native retirement and all result reads.
  std::map<ExecutionShape, std::unique_ptr<Phase>> phases, probe_phases;
  std::map<ExecutionShape, std::array<std::size_t, execution_slot_count>> planned_slot_uses;
  // Destroy episode pins/borrowed links before carriers and the raw policy
  // owners referenced by Snapshot/PageRef, including exceptional shutdown.
  std::unique_ptr<StepState> active_step;
  std::size_t capacity;
  ReplayPolicy replay_policy = ReplayPolicy::kAuto;
  std::filesystem::path layer0_probe, trace_routes;
  explicit Model(const std::filesystem::path &manifest, std::size_t selected_capacity,
                 cache::Plan selected_cache,
                 std::filesystem::path probe = {}, bool progress = false,
                  ReplayPolicy replay = ReplayPolicy::kAuto,
                  std::size_t max_cache_bytes = (std::numeric_limits<std::size_t>::max)())
       : state_schema(state_schema_for(selected_capacity)), cache_plan(selected_cache), capacity(selected_capacity),
         replay_policy(replay), layer0_probe(std::move(probe)) {
    const std::array limits{c::ResourceBudgetLimit{execution.device(), max_cache_bytes},
        c::ResourceBudgetLimit{mem::system_resource(), (std::numeric_limits<std::size_t>::max)()}};
    cache_budget = take(c::ResourceBudget::create({.limits = limits}));
    if (Backend::native_generated_storage) {
      // Native checked operations can own staging status/metadata buffers.
      // Register the actual backend resources, including aliased resources
      // only once; system allocation is a separate metadata domain.
      std::vector<c::ResourceBudgetLimit> native_limits;
      native_limits.reserve(3);
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        if (std::ranges::none_of(native_limits, [resource](const auto &limit) {
              return limit.resource == resource;
            }))
          native_limits.push_back({resource});
      execution_budget = take(c::ResourceBudget::create({.limits = native_limits, .allow_unattributed = true}));
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
    const auto loading = load_weight_bank(manifest, execution, weights, weight_pools,
        cache_plan.draft_cache ? WeightSelection::kTextAndMtp : WeightSelection::kText);
    const auto weight_load_seconds = std::chrono::duration<double>(Clock::now() - weight_load_start).count();
    std::cerr << "resident text weights: " << weights.size() << " tensors, " << loading.bytes << " bytes\n";
    std::cerr << "weights_load_seconds=" << weight_load_seconds << '\n';
    std::cerr << "weight_loader backend=" << Backend::name() << " staging_bytes=" << loading.stream.staging_bytes
              << " io_calls=" << loading.stream.io_calls
              << " native_calls=" << loading.stream.native_calls
              << " native_waits=" << loading.stream.native_waits
              << " host_waits=" << loading.stream.host_waits
              << " phase_completions=" << loading.stream.phase_completions
              << " copied_physical_bytes=" << loading.stream.copied_physical_bytes
              << " peak_backing_bytes=" << loading.stream.peak_backing_bytes << '\n';
  }

  ~Model() {
    abort_episode().IgnoreError();
    // Sessions retire their borrowed episodes before this owner is destroyed.
    // Scheduler idleness alone does not prove native/completion retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[tokens, phase] : *cache) {
        (void)tokens;
        for (auto &run : phase->slots) run.drain().IgnoreError();
      }
    abort_episode().IgnoreError();
    // A cleanup error is not successful rollback. After draining, detach the
    // remaining host links before destroying carriers or policy inventories;
    // actual native leases retain their own backing/retirement owners.
    if (active_step) detach_episode();
    phases.clear(); probe_phases.clear();
    if (workers) workers->shutdown();
    if (scheduler) scheduler->drain();
  }

  static c::TensorDesc output_description(const g::CompileReport &report, V value) {
    c::TensorDesc description;
    bool found = false;
    for (std::size_t index = 0; index < report.physical_value_count(); ++index) {
      const auto physical = take(report.physical_value(index));
      if (physical.value != value || !physical.external_output) continue;
      if (found && description != physical.description)
        throw std::runtime_error("output has incompatible physical domain descriptions");
      description = physical.description;
      found = true;
    }
    if (!found) throw std::runtime_error("compiled child has no physical output description");
    return description;
  }

  c::ResourceBudgetUsage cache_usage() const {
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes < known_cache_backing_bytes ||
        (!Backend::native_generated_storage && usage.live_bytes != known_cache_backing_bytes))
      throw std::runtime_error("actual cache reservations differ from the shared backing inventory");
    return usage;
  }

  // Report the same accounting scope for ordinary, MTP and scheduled runs.
  // Native execution allocates before a completed result is retained; its
  // temporaries/private allocations do not become a physical-memory cap here.
  std::string cache_report() const {
    const auto usage = cache_usage();
    std::ostringstream out;
    out << ",\"backend\":\"" << Backend::name()
        << "\",\"cache_budget_scope\":\"shared_pools_and_retained_output_backing\""
        << ",\"cache_shared_backing_bytes\":" << known_cache_backing_bytes
        << ",\"cache_retained_output_backing_bytes\":" << usage.live_bytes - known_cache_backing_bytes
        << ",\"cache_live_requested_bytes\":" << usage.live_bytes
        << ",\"cache_peak_requested_bytes\":" << usage.peak_bytes
        << ",\"cache_requested_byte_limit\":";
    if (usage.limit_bytes == SIZE_MAX) out << "null";
    else out << usage.limit_bytes;
    out << ",\"native_execution_private_bytes_unknown\":" << (Backend::native_generated_storage ? "true" : "false")
        << ",\"physical_vram_limit_known\":false";
    return out.str();
  }

  c::ReadLease retain_state_storage(const g::GraphOutput &output) const {
    auto retained = output.storage_lease(cache_budget);
    if (retained) return std::move(*retained);
    const auto usage = cache_usage();
    throw std::runtime_error("cache retained-state admission failed before snapshot publication: " +
        std::string(retained.error().message()) + "; live=" + std::to_string(usage.live_bytes) +
        ", limit=" + std::to_string(usage.limit_bytes) +
        "; increase --cache-bytes or reduce retained sessions/context; old snapshots remain valid");
  }

  void prepare_cache() {
    if (attention_schema.valid()) {
      return;
    }
    const auto storage = take(c::CacheStorage::plain(Type::kBFloat16));
    attention_schema = take(c::BatchSchema::create({
        .storage = c::BatchStorage::kPaged, .query_type = Type::kBFloat16,
        .output_type = Type::kBFloat16, .key_storage = storage, .value_storage = storage,
        .max_requests = cache_plan.max_requests, .max_query_rows = cache_plan.max_query_rows,
        .max_kv_rows = cache_plan.max_kv_rows, .max_pages = cache_plan.max_pages,
        .max_page_refs = cache_plan.max_page_refs, .query_heads = 8, .kv_heads = 2,
        .qk_head_dim = 256, .value_head_dim = 256, .page_size = cache::Plan::page_rows},
        {.resource_budget = cache_budget}));
    page_inventory.prepare(cache_plan.max_pages);
    snapshot_versions.capacity = cache_plan.snapshot_slots;
    for (std::size_t i = 0; i < page_inventory.keys.size(); ++i) {
      page_inventory.keys[i] = take(c::PagedArena::create(attention_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.bank_bytes, .resource_budget = cache_budget}));
      page_inventory.values[i] = take(c::PagedArena::create(attention_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.bank_bytes, .resource_budget = cache_budget}));
    }
    if (cache_plan.draft_cache) {
      draft_inventory.prepare(cache_plan.max_pages, 1);
      draft_inventory.keys[0] = take(c::PagedArena::create(attention_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.bank_bytes, .resource_budget = cache_budget}));
      draft_inventory.values[0] = take(c::PagedArena::create(attention_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.bank_bytes, .resource_budget = cache_budget}));
    }
    known_cache_backing_bytes = cache::multiply(cache_plan.bank_bytes, cache_plan.draft_cache ? 14u : 12u);
  }

  void prepare_state_storage() {
    if (state_storage_prepared) return;
    // Generated results keep their original native owners. These pools cover
    // actual initial zero inputs, and only selected caller-written outputs.
    // Publish together so a failed allocation releases its temporary backing
    // and leaves this Model ready for a later admission attempt.
    std::array<cache::StatePools, 24> prepared_pools;
    c::LeasePool prepared_hidden;
    auto prepared_bytes = known_cache_backing_bytes;
    const auto initial_slots = static_cast<std::uint32_t>(cache_plan.resident_sessions);
    for (std::size_t i = 0; i < state_schema.size(); ++i) if (i % 4 != 3) {
      const auto &description = state_schema[i]; auto &pools = prepared_pools[i];
      const auto history_slots = history_writers[i] ? cache_plan.state_slots : initial_slots;
      const auto recurrent_slots = recurrent_writers[i] ? cache_plan.state_slots : initial_slots;
      const auto history_bytes = cache::add(cache::multiply(take(description.history.required_bytes()), history_slots), 1);
      const auto recurrent_bytes = cache::add(cache::multiply(take(description.recurrent.required_bytes()), recurrent_slots), 1);
      pools.history = take(c::LeasePool::create(execution.device(), history_bytes,
          {.max_blocks = history_slots, .resource_budget = cache_budget}));
      pools.recurrent = take(c::LeasePool::create(execution.device(), recurrent_bytes,
          {.max_blocks = recurrent_slots, .resource_budget = cache_budget}));
      prepared_bytes = cache::add(prepared_bytes, cache::add(history_bytes, recurrent_bytes));
    }
    if (cache_plan.hidden_bytes) {
      const auto slots = hidden_writers ? cache_plan.hidden_slots : initial_slots;
      const auto bytes = cache::add(cache::multiply(4096, slots), 1);
      prepared_hidden = take(c::LeasePool::create(execution.device(), bytes,
          {.max_blocks = slots, .resource_budget = cache_budget}));
      prepared_bytes = cache::add(prepared_bytes, bytes);
    }
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes != prepared_bytes)
      throw std::runtime_error("shared cache backing inventory differs from the actual reserved bytes");
    state_pools = std::move(prepared_pools); hidden_pool = std::move(prepared_hidden);
    known_cache_backing_bytes = prepared_bytes;
    state_storage_prepared = true;
  }

  void prepare_family(std::span<const std::size_t> planned_chunks, bool diagnostic = false,
                      std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    if (state_storage_prepared)
      throw std::runtime_error("compile finite families before preparing their selected state storage");
    const std::size_t ports_per_request = verifies_two(role) ? 114u : 76u;
    if (!requests || requests > cache_plan.active_sequences ||
        requests > (std::numeric_limits<std::size_t>::max)() / ports_per_request)
      throw std::runtime_error("text finite family request count exceeds its configured capacity");
    if (verifies_two(role) && (diagnostic || !layer0_probe.empty() ||
        cache_plan.candidate_boundaries != 2 ||
        std::ranges::any_of(planned_chunks, [](auto t) { return t != 2; })))
      throw std::runtime_error("verification requires T2 and two independently budgeted state boundaries");
    if (has_mtp(role) && (!cache_plan.draft_cache || diagnostic || !layer0_probe.empty()))
      throw std::runtime_error("MTP requires its complete cache plan and a non-diagnostic family");
    if (has_mtp(role) && !drafts_token(role) && !is_mtp_prefill(role) &&
        std::ranges::any_of(planned_chunks, [](auto t) { return t != 1; }))
      throw std::runtime_error("MTP single-token families require T1");
    if (is_mtp_prefill(role) && std::ranges::any_of(planned_chunks, [](auto t) { return t < 2; }))
      throw std::runtime_error("MTP multi-token prefill requires at least two rows; use its single-token family for T1");
    auto &selected = diagnostic ? probe_families[{requests, role}] : families[{requests, role}];
    if (selected) throw std::runtime_error("text finite family was already prepared");
    if (planned_chunks.empty()) throw std::runtime_error("text finite family requires known nonempty chunks");
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.requests = requests; definition.role = role;
    definition.state_resource_ports.reserve(ports_per_request * requests);
    if (verifies_two(role) || has_mtp(role)) definition.carry_ports.resize(requests);
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity)
        throw std::runtime_error("finite text chunk outside nonempty context capacity");
      if (std::find(definition.chunks.begin(), definition.chunks.end(), tokens) == definition.chunks.end())
        definition.chunks.push_back(tokens);
    }
    const auto start = Clock::now();
    const auto max_query_rows = *std::max_element(definition.chunks.begin(), definition.chunks.end());
    Recipe r(weights, execution, max_query_rows, requests);
    const auto t = r.token_rows, packed = r.packed_rows;
    const auto batch = f(static_cast<std::int64_t>(requests));
    definition.last_row = r.last_row;
    if (requests > cache_plan.active_sequences || cache::multiply(requests, max_query_rows) > cache_plan.max_query_rows)
      throw std::runtime_error("finite family exceeds the configured batch capacity");
    prepare_cache();
    definition.attention_schema = attention_schema;
    definition.token_port = r.input(r.dense_spec(Type::kInt32,
        {has_mtp(role) && !is_mtp_prefill(role) ? batch : packed}));
    definition.position_port = requests == 1 ? r.input(r.dense_spec(Type::kInt32, {t})) :
        r.input(r.dense_spec(Type::kInt32, {batch, t}));
    if (requests != 1 && !verifies_two(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    definition.metadata_port = r.input(take(definition.attention_schema.metadata_description()));
    std::array<float, 32> frequency{};
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(10000000.F, float(i) / 32.F);
    r.literals.emplace_back(execution.device(), dense(Type::kFloat32, {32}));
    r.literals.back().write<float>(frequency);
    const auto frequencies = r.constant(r.literals.back());
    V target_tokens = definition.token_port;
    D draft_sequence_rows = f(0), draft_packed_rows = f(0);
    if (has_mtp(role)) {
      std::vector<V> carries, priors;
      definition.draft_hidden_inputs.resize(requests);
      for (auto &inputs : definition.draft_hidden_inputs) {
        inputs[0] = r.input(dense(Type::kBFloat16, {1, hidden}));
        definition.state_resource_ports.push_back(inputs[0]); carries.push_back(inputs[0]);
        if (draft_lag(role)) {
          inputs[1] = r.input(dense(Type::kBFloat16, {1, hidden}));
          definition.state_resource_ports.push_back(inputs[1]); priors.push_back(inputs[1]);
        }
      }
      definition.carry_input_port = r.concat(carries, 0, r.dense_spec(Type::kBFloat16, {batch, f(hidden)}));
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
        definition.prior_input_port = r.concat(priors, 0, r.dense_spec(Type::kBFloat16, {batch, f(hidden)}));
        definition.catchup_token_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      }
      if (drafts_token(role) || !is_mtp_prefill(role)) {
        V ids = definition.token_port, hidden_rows = definition.carry_input_port;
        if (draft_lag(role)) {
          if (requests == 1) {
            ids = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids}, r.dense_spec(Type::kInt32, {f(2)}));
            hidden_rows = r.one(c::Concat{.axis = 0}, {definition.prior_input_port, hidden_rows}, r.dense_spec(Type::kBFloat16, {f(2), f(hidden)}));
          } else {
            ids = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.catchup_token_port, {batch, f(1)}), r.reshape(ids, {batch, f(1)})},
                r.dense_spec(Type::kInt32, {batch, f(2)})), {packed_draft_rows});
            hidden_rows = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.prior_input_port, {batch, f(1), f(hidden)}), r.reshape(hidden_rows, {batch, f(1), f(hidden)})},
                r.dense_spec(Type::kBFloat16, {batch, f(2), f(hidden)})), {packed_draft_rows, f(hidden)});
          }
        }
        const auto projection = mtp::input_projection(r, ids, hidden_rows);
        const mtp::Geometry geometry{requests, draft_rows, packed_draft_rows};
        const auto kv = mtp::project_kv(r, projection, definition.draft_position_port, frequencies, geometry);
        mtp::AppendNodes append;
        if (drafts_token(role)) {
          definition.draft_key_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
          definition.draft_value_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
          definition.draft_metadata_port = r.input(take(attention_schema.metadata_description()));
          const auto head = mtp::full_head(r, projection, kv, definition.draft_position_port, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port, definition.draft_metadata_port},
              f(static_cast<std::int64_t>(draft_lag(role))), geometry);
          append = head.append_nodes; definition.draft_attention_node = head.attention_node;
          definition.draft_id_port = head.ids; r.publish(head.ids);
          target_tokens = requests == 1 ? r.one(c::Concat{.axis = 0},
              {definition.token_port, r.reshape(head.ids, {f(1)})}, r.dense_spec(Type::kInt32, {packed})) :
              r.reshape(r.one(c::Concat{.axis = 1},
                  {r.reshape(definition.token_port, {batch, f(1)}), r.reshape(head.ids, {batch, f(1)})},
                  r.dense_spec(Type::kInt32, {batch, f(2)})), {packed});
        } else append = mtp::append_kv(r, attention_schema, kv);
        definition.append_nodes[12] = append.key; definition.append_nodes[13] = append.value;
      }
    }
    auto x = r.one(c::Gather{.bounds = c::IndexBoundsPolicy::kUnchecked},
        {r.weight("embed_tokens.weight"), target_tokens}, r.dense_spec(Type::kBFloat16, {packed, f(hidden)}));
    // Writer-only draft maintenance has no attention data edge into target.
    // Order its real cache effects before target's first append as well.
    g::NodeId prior_append = has_mtp(role) && !is_mtp_prefill(role)
        ? definition.append_nodes[13] : g::NodeId{};
    bool has_prior_append = has_mtp(role) && !is_mtp_prefill(role);
    for (std::size_t i = 0; i < state_schema.size(); ++i) {
      const auto prefix = "layers." + std::to_string(i) + ".";
      auto normalized = r.norm(x, prefix + "input_layernorm.weight");
      auto &ports = definition.layers[i];
      V mixed;
      if (i % 4 != 3) {
        ports.sequences.resize(requests);
        std::vector<V> history_inputs, recurrent_inputs;
        history_inputs.reserve(requests); recurrent_inputs.reserve(requests);
        for (auto &sequence : ports.sequences) {
          sequence.history = r.input(state_schema[i].history);
          sequence.recurrent = r.input(state_schema[i].recurrent);
          history_inputs.push_back(sequence.history);
          recurrent_inputs.push_back(sequence.recurrent);
        }
        const auto history = r.concat(history_inputs, 0,
            r.dense_spec(Type::kBFloat16, {batch, f(3), f(6144)}));
        const auto state = r.concat(recurrent_inputs, 0,
            r.dense_spec(Type::kFloat32, {batch, f(16), f(128), f(128)}));
        auto qkv = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight"), {batch, t, f(6144)});
        auto samples = r.one(c::Concat{.axis = 1}, {history, qkv},
            r.dense_spec(Type::kBFloat16, {batch, r.sample_rows, f(6144)}));
        // Every published state has independent per-session material storage.
        // R1 preserves its original Copy and direct GDN state result.
        const auto next_history = r.slice(samples, 1, t, f(3));
        for (std::size_t request = 0; request < requests; ++request) {
          ports.sequences[request].next_history = r.copy(requests == 1 ? next_history :
              r.slice(next_history, 0, f(static_cast<std::int64_t>(request)), f(1)));
          if (verifies_two(role)) {
            auto first = r.slice(samples, 1, f(1), f(3));
            if (requests != 1) first = r.slice(first, 0, f(static_cast<std::int64_t>(request)), f(1));
            ports.sequences[request].first_history = r.copy(first);
          }
        }
        auto convolution = r.one(c::model_ops::causal_conv1d(), {r.cast(samples, Type::kFloat32),
            r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"), {f(6144), f(4)}), Type::kFloat32)},
            r.dense_spec(Type::kFloat32, {batch, t, f(6144)}));
        auto activated = r.unary(c::UnaryKind::kSilu, r.cast(convolution, Type::kBFloat16));
        auto q = r.reshape(r.slice(activated, 2, f(0), f(2048)), {batch, t, f(16), f(128)});
        auto k = r.reshape(r.slice(activated, 2, f(2048), f(2048)), {batch, t, f(16), f(128)});
        auto v = r.reshape(r.slice(activated, 2, f(4096), f(2048)), {batch, t, f(16), f(128)});
        auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32), {batch, t, f(16), f(1)});
        auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"), {batch, t, f(16), f(1)});
        auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32), {f(1), f(1), f(16), f(1)});
        auto decay = r.reshape(r.unary(c::UnaryKind::kNeg,
            r.unary(c::UnaryKind::kExp, r.weight(prefix + "linear_attn.A_log"))), {f(1), f(1), f(16), f(1)});
        auto g = r.binary(c::BinaryKind::kMultiply, decay,
            r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
        auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
        auto recurrent = r.op(c::Operation(c::GatedDeltaNet{
            .snapshot_count = verifies_two(role) ? 2u : 1u}),
            {r.cast(r.l2(q), Type::kFloat32), r.cast(r.l2(k), Type::kFloat32),
             r.cast(v, Type::kFloat32), g, beta, state},
            {r.dense_spec(Type::kFloat32, {batch, t, f(16), f(128)}),
             verifies_two(role)
                 ? r.dense_spec(Type::kFloat32, {batch, f(2), f(16), f(128), f(128)}) : r.spec(state)});
        for (std::size_t request = 0; request < requests; ++request) {
          if (verifies_two(role)) {
            auto per_request = r.slice(recurrent[1], 0, f(static_cast<std::int64_t>(request)), f(1));
            ports.sequences[request].first_recurrent = r.copy(r.reshape(
                r.slice(per_request, 1, f(0), f(1)), {f(1), f(16), f(128), f(128)}));
            ports.sequences[request].next_recurrent = r.copy(r.reshape(
                r.slice(per_request, 1, f(1), f(1)), {f(1), f(16), f(128), f(128)}));
          } else ports.sequences[request].next_recurrent = requests == 1 ? recurrent[1] :
              r.copy(r.slice(recurrent[1], 0, f(static_cast<std::int64_t>(request)), f(1)));
        }
        auto y = r.cast(recurrent[0], Type::kBFloat16);
        if (i == 0) definition.probe_ports[0] = y;
        const auto y_f32 = r.cast(y, Type::kFloat32);
        y = r.cast(r.same(c::RmsNorm{.epsilon = 1.e-6F}, {y_f32}, y_f32), Type::kBFloat16);
        // The original checkpoint keeps this scale F32. Torch promotes the
        // rounded BF16 normalized value before multiplying it by that scale.
        y = r.binary(c::BinaryKind::kMultiply, r.cast(y, Type::kFloat32), r.weight(prefix + "linear_attn.norm.weight"));
        auto z = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_z.weight"), {batch, t, f(16), f(128)});
        y = r.cast(r.binary(c::BinaryKind::kMultiply, r.cast(y, Type::kFloat32),
            r.unary(c::UnaryKind::kSilu, r.cast(z, Type::kFloat32))), Type::kBFloat16);
        auto projection_input = r.reshape(y, {packed, f(hidden)});
        if (i == 0) definition.probe_ports[1] = y; // Same BF16 payload before its metadata-only out_proj reshape.
        mixed = r.linear(projection_input, prefix + "linear_attn.out_proj.weight");
        if (i == 0) definition.probe_ports[2] = mixed;
        for (const auto &sequence : ports.sequences) {
          r.publish(sequence.next_history); r.publish(sequence.next_recurrent);
          if (verifies_two(role)) {
            r.publish(sequence.first_history); r.publish(sequence.first_recurrent);
            definition.state_resource_ports.push_back(sequence.first_history);
            definition.state_resource_ports.push_back(sequence.first_recurrent);
          }
          for (const auto port : {sequence.history, sequence.recurrent, sequence.next_history, sequence.next_recurrent})
            definition.state_resource_ports.push_back(port);
        }
      } else {
        ports.key = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
        ports.value = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
        auto projected = r.reshape(r.linear(normalized, prefix + "self_attn.q_proj.weight"), {batch, t, f(8), f(512)});
        auto q = r.norm(r.slice(projected, 3, f(0), f(256)), prefix + "self_attn.q_norm.weight");
        auto gate = r.reshape(r.copy(r.slice(projected, 3, f(256), f(256))), {packed, f(hidden)});
        auto k = r.norm(r.reshape(r.linear(normalized, prefix + "self_attn.k_proj.weight"), {batch, t, f(2), f(256)}), prefix + "self_attn.k_norm.weight");
        auto v = r.linear(normalized, prefix + "self_attn.v_proj.weight");
        const c::RoPE rope{.rotary_dimensions = 64, .sequence_axis = 1, .layout = c::RopeLayout::kHalfSplit};
        q = r.same(rope, {q, definition.position_port, frequencies}, q);
        k = r.same(rope, {k, definition.position_port, frequencies}, k);
        const auto layer = i / 4;
        auto &key_append = definition.append_nodes[layer * 2];
        auto &value_append = definition.append_nodes[layer * 2 + 1];
        r.op(c::Operation::custom(take(g::batch::paged_append(attention_schema, c::BatchCacheRole::kKey))),
            {r.reshape(k, {packed, f(2), f(256)})}, {}, &key_append);
        r.op(c::Operation::custom(take(g::batch::paged_append(attention_schema, c::BatchCacheRole::kValue))),
            {r.reshape(v, {packed, f(2), f(256)})}, {}, &value_append);
        if (has_prior_append) check(r.builder.depends_on(key_append, prior_append));
        check(r.builder.depends_on(value_append, key_append));
        prior_append = value_append; has_prior_append = true;
        auto attention = r.one(c::Operation::custom(c::PagedAttention{
            .capacity = definition.attention_schema.capacity(), .scale = 0.0625F, .causal = true}),
            {r.reshape(q, {packed, f(8), f(256)}), ports.key, ports.value, definition.metadata_port},
            r.dense_spec(Type::kBFloat16, {packed, f(8), f(256)}), &definition.attention_nodes[i / 4]);
        check(r.builder.depends_on(definition.attention_nodes[layer], key_append));
        check(r.builder.depends_on(definition.attention_nodes[layer], value_append));
        auto gated = r.binary(c::BinaryKind::kMultiply, r.reshape(attention, {packed, f(hidden)}), r.unary(c::UnaryKind::kSigmoid, gate));
        mixed = r.linear(gated, prefix + "self_attn.o_proj.weight");
      }
      x = r.ffn(r.binary(c::BinaryKind::kAdd, x, mixed), prefix,
                i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
      if (i == 0) definition.probe_ports[4] = x;
    }
    if (diagnostic) for (std::size_t i = 0; i < definition.probe_ports.size(); ++i) {
      r.publish(definition.probe_ports[i]);
      auto description = r.spec(definition.probe_ports[i]);
      if (description.data_type() != Type::kBFloat16) throw std::runtime_error("layer0 diagnostic boundary is not BF16");
    }
    // The selected child decides caller-written versus generated result storage.
    const auto normalized_last = r.norm(x, "norm.weight");
    if (is_mtp_prefill(role)) {
      const auto kv = mtp::prefill_kv(r, definition.token_port, normalized_last,
          definition.carry_input_port, definition.prior_input_port, definition.catchup_token_port,
          mtp::Geometry{requests, t, packed}, mtp::Geometry{requests, draft_sequence_rows, draft_packed_rows}, definition.draft_position_port,
          frequencies, draft_lag(role) != 0);
      const auto append = mtp::append_kv(r, attention_schema, kv);
      check(r.builder.depends_on(append.key, definition.append_nodes[11]));
      definition.append_nodes[12] = append.key; definition.append_nodes[13] = append.value;
    }
    auto last = verifies_two(role) ? r.copy(normalized_last) :
        (requests == 1 ? r.copy(r.slice(normalized_last, 0, D(r.last_row), f(1))) :
        r.one(c::Gather{}, {normalized_last, definition.last_ids_port},
            r.dense_spec(Type::kBFloat16, {batch, f(hidden)})));
    if (verifies_two(role) || has_mtp(role)) for (std::size_t request = 0; request < requests; ++request)
      for (std::size_t row = 0; row < (verifies_two(role) ? 2u : 1u); ++row) {
        const auto carry = r.copy(r.slice(normalized_last, 0,
            verifies_two(role) ? f(static_cast<std::int64_t>(request * 2 + row)) :
                (requests == 1 ? D(r.last_row) : D(take(r.builder.dimension_sum(
                    D(take(r.builder.dimension_product(t, f(static_cast<std::int64_t>(request))))), D(r.last_row))))), f(1)));
        definition.carry_ports[request][row] = carry;
        r.publish(carry); definition.state_resource_ports.push_back(carry);
      }
    definition.hidden_port = last;
    definition.logits_port = r.scores(last, "model.language_model.embed_tokens.weight");
    definition.id_port = r.op(c::Operation(c::TopK{.count = 1}), {definition.logits_port},
        {r.dense_spec(Type::kFloat32, {verifies_two(role) ? packed : batch, f(1)}),
         r.dense_spec(Type::kInt32, {verifies_two(role) ? packed : batch, f(1)})})[1];
    for (auto value : {definition.hidden_port, definition.logits_port, definition.id_port}) r.publish(value);
    definition.graph = take(std::move(r.builder).finish());
    definition.literals = std::move(r.literals);
    definition.specs = std::move(r.specs);
    std::vector<std::array<g::ShapeBinding, 1>> bindings;
    bindings.reserve(definition.chunks.size());
    for (const auto tokens : definition.chunks)
      bindings.push_back({g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(tokens - 1)}});
    std::vector<g::ExactShapeVariant> variants;
    variants.reserve(bindings.size());
    for (const auto &point : bindings) variants.push_back({point});
    g::CompileOptions compile_options{.shape_policy = g::CompileShapePolicy::kFiniteExact,
        .exact_variants = variants, .constants = r.constants};
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
    definition.compiled = take(execution.compile(definition.graph, r.constants, compile_options));
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
      child->caller_outputs.resize(definition.specs.size());
      for (std::size_t output = 0; output < child->compiled.executable.output_count(); ++output) {
        const auto value = take(child->compiled.executable.output_id(output));
        child->caller_outputs.at(value.ordinal()) =
            take(child->compiled.executable.output_ownership(output)) == g::OutputOwnership::kCaller;
      }
      for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (layer % 4 != 3)
        for (const auto &ports : definition.layers[layer].sequences) {
          history_writers[layer] = history_writers[layer] || child->caller_output(ports.next_history) ||
              (verifies_two(role) && child->caller_output(ports.first_history));
          recurrent_writers[layer] = recurrent_writers[layer] || child->caller_output(ports.next_recurrent) ||
              (verifies_two(role) && child->caller_output(ports.first_recurrent));
        }
      for (const auto &ports : definition.carry_ports) for (const auto value : ports)
        hidden_writers = hidden_writers || (value.valid() && child->caller_output(value));
      definition.children.push_back(std::move(child));
    }
    selected = std::move(result);
  }

  std::shared_ptr<cache::Snapshot> reserve_snapshot() {
    auto version = take(cache::VersionRef::acquire(snapshot_versions));
    auto result = std::make_shared<cache::Snapshot>();
    result->version = std::move(version); result->model = this; result->math = math_identity();
    return result;
  }

  const void *math_identity() const noexcept {
    // Every family uses the fixed score_policy. These per-Model identities also
    // separate probe math; no prefix can cross a Model or its compiled policy.
    return layer0_probe.empty() ? static_cast<const void *>(&families) : static_cast<const void *>(&probe_families);
  }

  // Aggregate ordered call plans before any execution carriers are created.
  // Prefix adoption can skip full-hit calls. Counts describe actual possible
  // uses of the serial slot, not additional execution carriers.
  void plan_execution(std::span<const std::size_t> calls, std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    for (const auto tokens : calls) {
      if (!families.contains({requests, role}) || std::find(families.at({requests, role})->chunks.begin(), families.at({requests, role})->chunks.end(), tokens) == families.at({requests, role})->chunks.end())
        throw std::runtime_error("execution plan chunk is outside the shared finite family");
      auto &uses = planned_slot_uses[{requests, tokens, role}];
      uses[slot_index] = cache::add(uses[slot_index], 1);
    }
  }

  // Finite wave/suffix plans can count many possible restores without
  // materializing repeated host call tapes or execution carriers.
  void plan_replay_uses(std::size_t tokens, std::size_t requests, std::size_t count,
                        GraphRole role = GraphRole::kText) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    if (!families.contains({requests, role}) ||
        std::find(families.at({requests, role})->chunks.begin(), families.at({requests, role})->chunks.end(), tokens) == families.at({requests, role})->chunks.end())
      throw std::runtime_error("execution use shape is outside the shared finite family");
    for (auto &uses : planned_slot_uses[{requests, tokens, role}]) uses = cache::add(uses, count);
  }

  std::shared_ptr<Session> create_session(std::span<const std::size_t> planned_chunks,
                                        cache::SnapshotRef prefix = {},
                                        GraphRole role = GraphRole::kText) {
    prepare_state_storage();
    if (cache_plan.draft_cache != has_mtp(role))
      throw std::runtime_error("session must use its complete target/draft cache family");
    if (!families.contains({1, role}) || (!layer0_probe.empty() && !probe_families.contains({1, role})))
      throw std::runtime_error("session requires the shared text families first");
    if (planned_chunks.empty() && !prefix) throw std::runtime_error("session requires an ordered call plan or a complete prefix");
    if (prefix && (prefix->model != this || prefix->math != math_identity() || prefix->length > capacity ||
        (planned_chunks.empty() && (!prefix->length || prefix->next_id < 0))))
      throw std::runtime_error("prefix belongs to another model or context");
    if (prefix) validate_mtp_snapshot(*prefix);
    auto session = std::shared_ptr<Session>(new Session(*this));
    std::size_t total = prefix ? prefix->length : 0;
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity - total)
        throw std::runtime_error("session plan exceeds nonempty context capacity");
      for (const auto *selected : {families.at({1, role}).get(), layer0_probe.empty() ? nullptr : probe_families.at({1, role}).get()}) {
        if (selected && std::find(selected->chunks.begin(), selected->chunks.end(), tokens) == selected->chunks.end())
          throw std::runtime_error("session chunk is outside the shared finite family");
      }
      total += tokens;
      if (!planned_slot_uses.contains({1, tokens, role}))
        throw std::runtime_error("session chunk has no model execution use plan");
    }
    if (prefix) session->head = std::move(prefix); // Exact evaluated-boundary fork; no payload copy.
    else {
      auto initial = reserve_snapshot();
      const std::array<std::int32_t, 2> empty{0, 0};
      const std::array<std::int32_t, 1> tail{0};
      const auto layout = take(attention_schema.version({empty, empty, {}, tail}));
      for (std::size_t i = 0; i < initial->keys.size(); ++i) {
        initial->keys[i] = take(page_inventory.keys[i].read(layout, {}));
        initial->values[i] = take(page_inventory.values[i].read(layout, {}));
      }
      if (cache_plan.draft_cache) {
        initial->draft.emplace();
        initial->draft->key = take(draft_inventory.keys[0].read(layout, {}));
        initial->draft->value = take(draft_inventory.values[0].read(layout, {}));
        auto zero = take(hidden_pool.allocate(4096));
        check(zero.view()->fill(std::byte{})); initial->carry_hidden = take(zero.publish());
      }
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (i % 4 != 3) {
        auto history = take(state_pools[i].history.allocate(take(state_schema[i].history.required_bytes())));
        auto recurrent = take(state_pools[i].recurrent.allocate(take(state_schema[i].recurrent.required_bytes())));
        check(history.view()->fill(std::byte{})); check(recurrent.view()->fill(std::byte{}));
        initial->state[i].history = take(history.publish());
        initial->state[i].recurrent = take(recurrent.publish());
      }
      session->head = std::move(initial);
    }
    // State admission does not allocate execution carriers. The first actual
    // explicit prepare creates its needed Phase and slot; idle/full-hit cached
    // Sessions never duplicate IO, workspace or native execution preparation.
    return session;
  }

  void validate_mtp_snapshot(const cache::Snapshot &head) const {
    if (cache_plan.draft_cache != head.draft.has_value())
      throw std::runtime_error("snapshot does not contain the configured complete MTP state");
    if (!head.draft) return;
    const auto &draft = *head.draft;
    if (draft.lag > 1 || head.length < draft.lag || draft.frontier != head.length - draft.lag ||
        !draft.key.valid() || !draft.value.valid() || !head.carry_hidden.valid() ||
        (draft.lag && (!head.prior_hidden.valid() || draft.catchup_token < 0 || draft.catchup_token >= vocabulary)) ||
        (!draft.lag && draft.catchup_token != -1))
      throw std::runtime_error("inconsistent MTP frontier, catch-up token or owned target hidden state");
  }

  void initialize_slot(Phase &phase, std::size_t slot) {
    if (phase.slots[slot].valid()) return;
    const auto &definition = *phase.family;
    const std::array shape{g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(phase.query_rows - 1)}};
    auto run = take(definition.compiled.executable.create_invocation(
        {.shape = shape, .resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (run.exact_variant_index() != phase.variant_index)
      throw std::runtime_error("finite family selected an unexpected exact child");
    if (scheduler) check(g::prepare_progress(run, *scheduler, *workers));
    check(run.bind(phase.token_port, phase.tokens.read()));
    check(run.bind(phase.position_port, phase.positions.read()));
    if (definition.requests != 1 && !verifies_two(definition.role)) check(run.bind(phase.last_ids_port, phase.last_ids.read()));
    if (phase.logits.buffer) check(run.bind(phase.logits_port, phase.logits.view()));
    if (phase.next_id.buffer) check(run.bind(phase.id_port, phase.next_id.view()));
    if (phase.final_hidden.buffer) check(run.bind(phase.hidden_port, phase.final_hidden.view()));
    if (has_mtp(definition.role)) {
      check(run.bind(phase.draft_position_port, phase.draft_positions.read()));
      if (draft_lag(definition.role)) check(run.bind(phase.catchup_token_port, phase.catchup_token.read()));
      if (drafts_token(definition.role) && phase.draft_id.buffer) check(run.bind(phase.draft_id_port, phase.draft_id.view()));
    }
    if (!layer0_probe.empty()) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      if (phase.probe_caller[i]) check(run.bind(phase.probe_ports[i], phase.probe_outputs[i].view()));
    // Cold preparation can also run without a pending StepState. Publish only
    // the fully bound slot, so an allocation/binding failure cannot leave an
    // apparently valid half-prepared Invocation in the reusable Model pool.
    phase.slots[slot] = std::move(run);
    phase.preparation_uses[slot] = 0;
  }

  Phase &prepare_child(std::size_t tokens, bool diagnostic, std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    auto &cache = diagnostic ? probe_phases : phases;
    const auto found = cache.find({requests, tokens, role});
    if (found != cache.end()) return *found->second;
    auto &selected = diagnostic ? probe_families[{requests, role}] : families[{requests, role}];
    if (!selected) throw std::runtime_error("text finite family has not been prepared");
    auto &definition = *selected;
    const auto point = std::find(definition.chunks.begin(), definition.chunks.end(), tokens);
    if (point == definition.chunks.end()) throw std::runtime_error("unlisted text chunk has no prepared exact child");
    const auto start = Clock::now();
    const auto index = static_cast<std::size_t>(point - definition.chunks.begin());
    auto result = std::make_unique<Phase>();
    auto &phase = *result;
    phase.query_rows = tokens;
    static_cast<FamilyPorts &>(phase) = static_cast<const FamilyPorts &>(definition);
    phase.planned_slot_uses = planned_slot_uses.at({requests, tokens, role});
    for (std::size_t slot = 0; slot < phase.replay_modes.size(); ++slot) {
      phase.replay_modes[slot] = replay_policy == ReplayPolicy::kCommands
          || (replay_policy == ReplayPolicy::kAuto && phase.planned_slot_uses[slot] <= 1)
          ? g::ReplayMode::kCommands : g::ReplayMode::kPreferCapture;
    }
    phase.attention_schema = definition.attention_schema;
    phase.child = definition.children.at(index).get();
    phase.family = &definition; phase.variant_index = index;
    std::erase_if(phase.state_resource_ports, [&](V value) {
      return value.kind() != g::ValueKind::kInput && !phase.child->caller_output(value);
    });
    const auto t = static_cast<std::int64_t>(tokens);
    phase.tokens = Tensor(execution.device(), dense(Type::kInt32,
        {has_mtp(role) && !is_mtp_prefill(role) ? static_cast<std::int64_t>(requests) : t * static_cast<std::int64_t>(requests)}));
    phase.positions = Tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {t})
        : dense(Type::kInt32, {static_cast<std::int64_t>(requests), t}));
    if (has_mtp(role)) {
      const auto draft_rows = is_mtp_prefill(role) ? t + static_cast<std::int64_t>(draft_lag(role)) : static_cast<std::int64_t>(draft_lag(role) + 1);
      phase.draft_positions = Tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {draft_rows})
          : dense(Type::kInt32, {static_cast<std::int64_t>(requests), draft_rows}));
      if (draft_lag(role)) phase.catchup_token = Tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
      if (drafts_token(role)) phase.draft_id = Tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests), 1}),
          phase.child->caller_output(phase.draft_id_port));
    }
    if (requests != 1 && !verifies_two(role)) {
      phase.last_ids = Tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
      std::vector<std::int32_t> last_ids(requests);
      for (std::size_t j = 0; j < requests; ++j) last_ids[j] = static_cast<std::int32_t>((j + 1) * tokens - 1);
      phase.last_ids.write<std::int32_t>(last_ids);
    }
    phase.logits = Tensor(execution.device(), output_description(phase.child->compiled.report, phase.logits_port),
        phase.child->caller_output(phase.logits_port));
    phase.next_id = Tensor(execution.device(), output_description(phase.child->compiled.report, phase.id_port),
        phase.child->caller_output(phase.id_port));
    phase.final_hidden = Tensor(execution.device(), output_description(phase.child->compiled.report, phase.hidden_port),
        phase.child->caller_output(phase.hidden_port));
    if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
      for (std::size_t output = 0; output < phase.child->compiled.executable.output_count(); ++output) {
        if (take(phase.child->compiled.executable.output_id(output)) != phase.probe_ports[i]) continue;
        phase.probe_caller[i] = take(phase.child->compiled.executable.output_ownership(output)) == g::OutputOwnership::kCaller;
        if (phase.probe_caller[i]) phase.probe_outputs[i] =
            Tensor(execution.device(), output_description(phase.child->compiled.report, phase.probe_ports[i]));
      }
    }
    const auto &route_directory = layer0_probe.empty() ? trace_routes : layer0_probe;
    if (!route_directory.empty()) {
      std::filesystem::create_directories(route_directory);
      const auto label = std::string(diagnostic ? "published" : "default") + (has_mtp(role) ? "-role" + std::to_string(static_cast<int>(role)) : "");
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
              << " slot0_uses=" << phase.planned_slot_uses[0]
              << " slot0=" << (phase.replay_modes[0] == g::ReplayMode::kCommands ? "commands" : "prefer-capture") << '\n';
    std::cerr << "compiled T=" << tokens << " steps=" << phase.child->compiled.report.step_count()
              << " workspace/slot=" << phase.child->compiled.executable.workspace_bytes()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " child_prepare_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    auto *address = result.get(); cache.emplace(ExecutionShape{requests, tokens, role}, std::move(result)); return *address;
  }

  // An engine with a known ordered request can explicitly prepare its actual
  // carriers before admission/ready, preserving the original cold boundary.
  // Candidate prefix suffixes remain unmaterialized until a hit selects one.
  void prepare_execution(std::span<const std::size_t> calls,
                         std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    if (carrier_borrowed)
      throw std::runtime_error("cannot cold-prepare carriers during an active execution loan");
    for (const auto tokens : calls) {
      if (!layer0_probe.empty()) prepare_child(tokens, false, requests, role);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), requests, role);
      initialize_slot(phase, slot_index);
    }
  }

  void detach_episode() noexcept {
    if (!active_step) return;
    for (auto &member : active_step->members) {
      member.session->retirement_status.Update(active_step->terminal);
      member.session->pending = nullptr;
    }
    active_step.reset();
  }

  hyle::Result<bool> try_retire_episode() noexcept {
    if (!active_step) return true;
    // Dropping the public Completion does not remove original native work.
    // Drive its owning environment before checking the unchanged Graph/page
    // retirement gates; an observation error alone never proves retirement.
    active_step->terminal.Update(execution.collect_retired());
    HYLE_ASSIGN_OR_RETURN(auto retired, active_step->try_abort());
    if (retired) detach_episode();
    return retired;
  }

  hyle::Status abort_episode() noexcept {
    // Explicit blocking shutdown/error recovery. The same nonblocking retire
    // path also progresses abandoned native work, without closing the Context
    // or waiting on a Graph whose original observer has already been dropped.
    while (active_step) {
      HYLE_ASSIGN_OR_RETURN(auto retired, try_retire_episode());
      if (retired) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return hyle::Status::Ok();
  }

  // Selection and equal-T bucketing belong to the caller. A batch owns one
  // original Invocation, IO loan, Completion and publication transaction.
  void prepare_draft_append(StepState &step, GraphRole role) {
    auto &phase = *step.phase; auto &run = *step.run;
    const auto width = step.members.size();
    const auto rows = is_mtp_prefill(role) ? cache::add(step.input_tokens, draft_lag(role)) : draft_lag(role) + 1;
    std::vector<std::int32_t> query{0}, old_ptr{0}, next_ptr{0}, old_last, next_last;
    std::vector<std::int32_t> old_ids, next_ids, positions, catchups;
    for (auto *table : {&query, &old_ptr, &next_ptr, &old_last, &next_last}) table->reserve(width + 1);
    positions.reserve(cache::multiply(width, rows)); catchups.reserve(width);
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
      bool reuse = old_tail && session.draft_tail_head == member.base.get() && !old.pages.empty();
      if (reuse) for (const auto &tail : session.draft_tails)
        reuse = reuse && tail.valid() && !tail.reserved_rows() &&
            tail.page() == old.pages.back().page() && tail.initialized_rows() == old_tail;
      next.pages.reserve(count);
      for (std::size_t bank = 12; bank < 14; ++bank) step.dirty[bank].reserve(cache::add(step.dirty[bank].size(), dirty_count));
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
        for (std::size_t bank = 0; bank < 2; ++bank) step.tails[12 + bank].push_back(std::move(session.draft_tails[bank]));
      } else session.draft_tails = {};
      session.draft_tail_head = nullptr;
      for (std::size_t page = reuse ? 1 : 0; page < dirty_count; ++page) {
        auto reservation = take(draft_inventory.acquire()); next.pages.push_back(std::move(reservation.reference));
        step.dirty[12].push_back(std::move(reservation.keys[0])); step.dirty[13].push_back(std::move(reservation.values[0]));
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
      check(run.bind(ports[0], dense(Type::kBFloat16, {1, hidden}), member.base->carry_hidden));
      if (draft_lag(role)) {
        catchups.push_back(old.catchup_token);
        check(run.bind(ports[1], dense(Type::kBFloat16, {1, hidden}), member.base->prior_hidden));
      }
    }
    const auto layout = width == 1 ? step.members[0].draft_layout : take(attention_schema.version({query, next_ptr, next_ids, next_last}));
    step.draft_layout = layout;
    std::vector<std::int32_t> old_query(width + 1, 0);
    const auto old_layout = width == 1 ? c::BatchLayoutVersion{} : take(attention_schema.version({old_query, old_ptr, old_ids, old_last}));
    std::vector<c::PagedArenaRead> sources(width);
    for (std::size_t bank = 12; bank < 14; ++bank) {
      auto &arena = bank == 12 ? draft_inventory.keys[0] : draft_inventory.values[0];
      for (std::size_t j = 0; j < width; ++j) sources[j] = bank == 12 ? step.members[j].base->draft->key : step.members[j].base->draft->value;
      const auto previous = width == 1 ? sources[0] : take(arena.reframe(old_layout, sources));
      step.appends[bank] = take(g::batch::PagedAppendPages::create(arena, layout, previous,
          step.dirty[bank], step.tails[bank], {.resource_budget = cache_budget}));
    }
    phase.draft_positions.write<std::int32_t>(positions);
    if (draft_lag(role)) phase.catchup_token.write<std::int32_t>(catchups);
  }

  void prepare_evaluations(std::span<const EvaluationRequest> requests,
                           PreparationObservation *preparation = nullptr,
                           GraphRole role = GraphRole::kText) {
    if (requests.empty() || requests.size() > cache_plan.active_sequences)
      throw std::runtime_error("batch width exceeds the configured active sequence capacity");
    if (carrier_borrowed || active_step)
      throw std::runtime_error("the original execution carrier has not retired");
    const auto supplied_tokens = requests.front().ids.size();
    const auto tokens = drafts_token(role) ? 2u : supplied_tokens;
    if (cache_plan.draft_cache != has_mtp(role))
      throw std::runtime_error("evaluation must use the configured complete target/draft state");
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      if (!request.session || request.session->model_ != this)
        throw std::runtime_error("batch session belongs to another model");
      const auto &session = *request.session;
      if (session.stopping || session.failed || session.pending)
        throw std::runtime_error("batch session is stopped, failed or already pending");
      if (!supplied_tokens || request.ids.size() != supplied_tokens || tokens > capacity - session.length())
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
    }
    const auto step_start = Clock::now();
    auto candidate = std::make_unique<StepState>();
    if (!candidate->members.try_reserve(requests.size())) throw std::bad_alloc{};
    for (const auto &request : requests) {
      candidate->members.push_back({.session = request.session, .base = request.session->head});
      candidate->members.back().first_input = request.ids.front();
    }
    active_step = std::move(candidate);
    auto &step = *active_step;
    carrier_borrowed = true; step.candidate_admission = &carrier_borrowed;
    step.step_start = step_start; step.input_tokens = tokens;
    for (auto &member : step.members) member.session->pending = &step;
    try {
      const auto width = requests.size(), slot = slot_index;
      if (!layer0_probe.empty()) prepare_child(tokens, false, width, role);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), width, role);
      auto &run = phase.slots[slot];
      step.phase = &phase; step.run = &run; step.nodes = phase.append_nodes;
      initialize_slot(phase, slot);
      std::vector<std::int32_t> packed_ids, positions;
      if (width != 1) packed_ids.reserve(cache::multiply(width, tokens));
      positions.reserve(cache::multiply(width, tokens));
      for (const auto &request : requests) {
        if (width != 1) packed_ids.insert(packed_ids.end(), request.ids.begin(), request.ids.end());
        for (std::size_t row = 0; row < tokens; ++row)
          positions.push_back(static_cast<std::int32_t>(request.session->length() + row));
      }
      phase.tokens.write<std::int32_t>(width == 1 ? requests[0].ids : std::span<const std::int32_t>(packed_ids));
      phase.positions.write<std::int32_t>(positions);
      const auto cold_start = Clock::now();
      hyle::container::SmallVector<std::int32_t, 2> query_indptr{0}, old_indptr{0}, new_indptr{0}, old_last, new_last;
      for (auto *table : {&query_indptr, &old_indptr, &new_indptr, &old_last, &new_last})
        if (!table->try_reserve(width + 1)) throw std::bad_alloc{};
      std::vector<std::int32_t> old_ids, new_ids;
      std::size_t old_refs = 0, new_refs = 0;
      for (const auto &member : step.members) {
        old_refs = cache::add(old_refs, member.base->pages.size());
        new_refs = cache::add(new_refs, (member.base->length + tokens + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
      }
      if (width != 1) old_ids.reserve(old_refs);
      new_ids.reserve(new_refs);
      for (auto &bank : step.tails) if (!bank.try_reserve(width)) throw std::bad_alloc{};
      for (std::size_t j = 0; j < width; ++j) {
        auto &member = step.members[j]; auto &session = *member.session;
        const auto length = session.length();
        member.next = reserve_snapshot(); member.next->length = length + tokens;
        const auto full_prefix_pages = length / cache::Plan::page_rows;
        const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
        const auto dirty_count = page_count - full_prefix_pages;
        const auto tail_rows = length % cache::Plan::page_rows;
        bool reuse_tail = tail_rows && session.tail_head == member.base.get() && !member.base->pages.empty();
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
        } else session.tails = {};
        session.tail_head = nullptr;
        for (std::size_t page = reuse_tail ? 1 : 0; page < dirty_count; ++page) {
          auto reservation = take(page_inventory.acquire());
          member.next->pages.push_back(std::move(reservation.reference));
          for (std::size_t layer = 0; layer < reservation.keys.size(); ++layer) {
            step.dirty[layer * 2].push_back(std::move(reservation.keys[layer]));
            step.dirty[layer * 2 + 1].push_back(std::move(reservation.values[layer]));
          }
        }
        if (width != 1) for (const auto &page : member.base->pages) old_ids.push_back(static_cast<std::int32_t>(page.page()));
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
      phase.attention_layout = width == 1 ? step.members[0].layout
          : take(attention_schema.version({query_indptr, new_indptr, new_ids, new_last}));
      // Old layouts describe no new queries, but retain each independent history.
      const std::vector<std::int32_t> old_query(width == 1 ? 0 : width + 1, 0);
      const auto old_layout = width == 1 ? c::BatchLayoutVersion{}
          : take(attention_schema.version({old_query, old_indptr, old_ids, old_last}));
      std::vector<c::PagedArenaRead> sources(width == 1 ? 0 : width);
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        const auto compose = [&](c::PagedArena &arena, bool key) {
          if (width == 1) return key ? step.members[0].base->keys[layer] : step.members[0].base->values[layer];
          for (std::size_t j = 0; j < width; ++j)
            sources[j] = key ? step.members[j].base->keys[layer] : step.members[j].base->values[layer];
          return take(arena.reframe(old_layout, sources));
        };
        auto &key_arena = page_inventory.keys[layer]; auto &value_arena = page_inventory.values[layer];
        step.appends[layer * 2] = take(g::batch::PagedAppendPages::create(key_arena, phase.attention_layout,
            compose(key_arena, true), step.dirty[layer * 2], step.tails[layer * 2], {.resource_budget = cache_budget}));
        step.appends[layer * 2 + 1] = take(g::batch::PagedAppendPages::create(value_arena, phase.attention_layout,
            compose(value_arena, false), step.dirty[layer * 2 + 1], step.tails[layer * 2 + 1], {.resource_budget = cache_budget}));
      }
      if (has_mtp(role)) prepare_draft_append(step, role);
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (i % 4 != 3)
        for (std::size_t j = 0; j < width; ++j) {
          const auto &ports = phase.layers[i].sequences[j]; const auto &descriptions = state_schema[i];
          check(run.bind(ports.history, descriptions.history, step.members[j].base->state[i].history));
          check(run.bind(ports.recurrent, descriptions.recurrent, step.members[j].base->state[i].recurrent));
          const auto bind_state = [&](V port, const c::TensorDesc &description, c::LeasePool &pool) {
            if (phase.child->caller_output(port))
              check(run.bind(port, description, take(pool.allocate(take(description.required_bytes())))));
          };
          bind_state(ports.next_history, descriptions.history, state_pools[i].history);
          bind_state(ports.next_recurrent, descriptions.recurrent, state_pools[i].recurrent);
          if (verifies_two(role)) {
            bind_state(ports.first_history, descriptions.history, state_pools[i].history);
            bind_state(ports.first_recurrent, descriptions.recurrent, state_pools[i].recurrent);
          }
        }
      if (verifies_two(role) || has_mtp(role)) for (const auto &ports : phase.carry_ports)
        for (std::size_t i = 0; i < (verifies_two(role) ? 2u : 1u); ++i)
          if (phase.child->caller_output(ports[i]))
            check(run.bind(ports[i], dense(Type::kBFloat16, {1, hidden}), take(hidden_pool.allocate(4096))));
      std::vector<g::batch::NodeLayout> layouts;
      layouts.reserve(drafts_token(role) ? 7 : 6);
      for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i) layouts.push_back({phase.attention_nodes[i], phase.attention_layout});
      if (drafts_token(role)) layouts.push_back({phase.draft_attention_node, step.draft_layout});
      const auto queues = execution.queue_bindings();
      if (preparation) { preparation->slot = slot; preparation->phase_slot_prior_uses = phase.preparation_uses[slot]; }
      const auto batch_start = Clock::now();
      check(g::batch::prepare(run, queues, layouts, {.host_workspace = execution.staging()}));
      for (std::size_t layer = 0; layer < phase.attention_nodes.size(); ++layer) {
        const auto &ports = phase.layers[layer * 4 + 3];
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2], ports.key,
            std::move(step.appends[layer * 2]))); step.bound[layer * 2] = true;
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2 + 1], ports.value,
            std::move(step.appends[layer * 2 + 1]))); step.bound[layer * 2 + 1] = true;
      }
      if (has_mtp(role)) for (std::size_t bank = 12; bank < 14; ++bank) {
        if (drafts_token(role)) check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank],
            bank == 12 ? phase.draft_key_port : phase.draft_value_port, std::move(step.appends[bank])));
        else check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank], std::move(step.appends[bank])));
        step.bound[bank] = true;
      }
      const auto batch_end = Clock::now();
      check(run.prepare_replay(queues, {.mode = phase.replay_modes[slot], .host_wait = g::HostWaitPolicy::kAllow}));
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
    } catch (...) { abort_episode().IgnoreError(); throw; }
  }

  void prepare_evaluation(Session &session, std::span<const std::int32_t> ids,
                          PreparationObservation *preparation = nullptr,
                          GraphRole role = GraphRole::kText) {
    const std::array requests{EvaluationRequest{session.shared_from_this(), ids}};
    prepare_evaluations(requests, preparation, role);
  }

  void submit_evaluations() {
    if (!active_step || !active_step->prepared || active_step->submitted || active_step->cancelled || !active_step->run)
      throw std::runtime_error("batch is not prepared or was already submitted/cancelled");
    for (const auto &member : active_step->members)
      if (member.session->stopping || member.session->failed)
        throw std::runtime_error("batch participant stopped before submission");
    const auto queues = execution.queue_bindings();
    auto &step = *active_step;
    step.start = Clock::now(); step.submitted = true;
    step.completion = take(step.run->submit(queues));
    step.submit_seconds = std::chrono::duration<double>(Clock::now() - step.start).count();
  }

  void submit_evaluation(Session &session) {
    if (session.model_ != this || !active_step || session.pending != active_step.get() || active_step->members.size() != 1)
      throw std::runtime_error("single submission requires its original one-request episode");
    submit_evaluations();
  }

  template<class Select>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select) {
    if (!active_step || outputs.size() != active_step->members.size() || next_ids.size() != outputs.size() ||
        active_step->cancelled || !active_step->submitted || !active_step->completion.valid())
      throw std::runtime_error("finish requires its original complete noncancelled batch");
    for (std::size_t j = 0; j < outputs.size(); ++j)
      if (outputs[j].session != active_step->members[j].session || outputs[j].session->stopping || outputs[j].session->failed)
        throw std::runtime_error("finish participant does not match the original active batch");
    auto &step = *active_step; auto &phase = *step.phase; auto &run = *step.run;
    const auto step_start = step.step_start, start = step.start;
    const auto tokens = step.input_tokens;
    const auto submit_seconds = step.submit_seconds, cold_batch_seconds = step.cold_batch_seconds;
    const auto replay_route = step.replay_route;
    const auto capture_recorded = step.capture_recorded, capture_reused = step.capture_reused;
    std::array<c::ConstTensorView, 9> probe_views;
    std::array<c::ReadLease, 9> probe_owners;
    c::ReadLease generated_logits, generated_hidden;
    struct ReturnReadback final { bool *admission = nullptr; ~ReturnReadback() { if (admission) *admission = false; } } return_readback;
    try {
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        if (outputs[j].mtp_ledger) {
          if (!has_mtp(phase.family->role)) throw std::runtime_error("MTP ledger requires an MTP participant");
          for (std::size_t k = 0; k < j; ++k) if (outputs[k].mtp_ledger == outputs[j].mtp_ledger)
            throw std::runtime_error("batch participants cannot share a mutable MTP ledger");
          auto &record = outputs[j].mtp_ledger->next();
          step.members[j].commit_ledger = outputs[j].mtp_ledger; step.members[j].commit_record = &record;
          if (record.role != phase.family->role || record.before != mtp_boundary(*step.members[j].base) ||
              record.ids().size() != (drafts_token(phase.family->role) ? 1u : tokens) ||
              record.ids().empty() || record.ids().front() != step.members[j].first_input)
            throw std::runtime_error("MTP prepared ledger entry differs from its actual episode");
          if (outputs[j].mtp_expected && !outputs[j].mtp_expected->committed)
            throw std::runtime_error("MTP replay expected record is not committed");
        } else if (outputs[j].mtp_expected)
          throw std::runtime_error("MTP expected replay record requires a fresh commit witness");
      }
      check(step.completion.wait()); step.completion = {};
      const auto read_result = [&](V port, Tensor &caller, std::span<std::byte> bytes) {
        if (caller.buffer) { check(execution.read(caller.buffer.view(), bytes)); return; }
        auto result = take(run.result(port));
        auto owner = take(result.storage_lease());
        check(execution.read(take(owner.view()), bytes));
      };
      // Trace storage must be pinned before take_append/unbind revoke Graph
      // output observation. Generated arrays keep their actual native owner.
      if (std::ranges::any_of(outputs, [](const auto &output) { return output.trace && !output.trace->empty(); })) {
        if (!phase.logits.buffer) generated_logits = take(take(run.result(phase.logits_port)).storage_lease());
        if (!phase.final_hidden.buffer) generated_hidden = take(take(run.result(phase.hidden_port)).storage_lease());
      }
      if (drafts_token(phase.family->role)) {
        hyle::container::SmallVector<std::int32_t, 2> drafts;
        if (!drafts.try_resize(outputs.size())) throw std::bad_alloc{};
        read_result(phase.draft_id_port, phase.draft_id, std::as_writable_bytes(std::span(drafts.data(), drafts.size())));
        for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].draft_token = drafts[j];
      }
      // Selection observes only retired original Graph results. No head has
      // been published and all state/append guards still belong to this episode.
      if (verifies_two(phase.family->role)) {
        std::vector<std::int32_t> candidates(outputs.size() * 2);
        read_result(phase.id_port, phase.next_id, std::as_writable_bytes(std::span(candidates)));
        select(std::span<const std::int32_t>(candidates), step, next_ids);
      } else {
        read_result(phase.id_port, phase.next_id, std::as_writable_bytes(next_ids));
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      }
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        auto &member = step.members[j];
        if (!member.accepted_tokens || member.accepted_tokens > tokens ||
            (member.accepted_tokens != tokens && !verifies_two(phase.family->role)) ||
            next_ids[j] < 0 || next_ids[j] >= vocabulary)
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
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (i % 4 != 3)
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          const auto &ports = phase.layers[i].sequences[j];
          const bool first = verifies_two(phase.family->role) && step.members[j].accepted_tokens == 1;
          auto history = take(run.result(first ? ports.first_history : ports.next_history));
          step.members[j].next->state[i].history = retain_state_storage(history);
          auto recurrent = take(run.result(first ? ports.first_recurrent : ports.next_recurrent));
          step.members[j].next->state[i].recurrent = retain_state_storage(recurrent);
        }
      if (verifies_two(phase.family->role) || has_mtp(phase.family->role))
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          const auto &ports = phase.carry_ports[j];
          auto selected = take(run.result(ports[verifies_two(phase.family->role) ? member.accepted_tokens - 1 : 0]));
          member.next->carry_hidden = retain_state_storage(selected);
          if (verifies_two(phase.family->role) && member.accepted_tokens == 2) {
            auto prior = take(run.result(ports[0]));
            member.next->prior_hidden = retain_state_storage(prior);
          }
        }
      if (has_mtp(phase.family->role)) for (auto &member : step.members) {
        auto &draft = *member.next->draft;
        draft.lag = static_cast<std::uint32_t>(member.next->length - draft.frontier);
        draft.catchup_token = draft.lag ? member.draft_token : -1;
      }
      for (std::size_t j = 0; j < outputs.size(); ++j) if (auto *record = step.members[j].commit_record) {
        const auto &member = step.members[j];
        record->requests = outputs.size(); record->member_index = j;
        record->accepted = member.accepted_tokens;
        record->draft_rows = member.next->draft->frontier - member.base->draft->frontier;
        record->after = mtp_boundary(*member.next); record->after.next_id = next_ids[j];
        record->accepted_inputs = {member.first_input, member.accepted_tokens == 2 ? member.draft_token : -1};
        if (!record->generation) {
          record->draft = -1; record->target_first = next_ids[j]; record->target_second = -1;
          record->agreed = false; record->accepted_draft = false;
          record->stopped = is_end_of_generation(next_ids[j]);
          record->emitted = {next_ids[j], -1}; record->emitted_count = 1;
        }
        if (outputs[j].mtp_expected && !record->same_result(*outputs[j].mtp_expected))
          throw std::runtime_error("MTP replay natural decision or state boundary diverged from its recorded episode");
      }
      if (!layer0_probe.empty()) for (std::size_t i = 0; i < probe_views.size(); ++i) {
        if (phase.probe_caller[i]) probe_views[i] = take(run.output(phase.probe_ports[i]));
        else {
          probe_owners[i] = take(take(run.result(phase.probe_ports[i])).storage_lease());
          probe_views[i] = take(probe_owners[i].tensor(output_description(phase.child->compiled.report, phase.probe_ports[i])));
        }
      }
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        for (const auto bank : {layer * 2, layer * 2 + 1}) {
          step.appends[bank] = take(g::batch::take_append(run, step.nodes[bank])); step.bound[bank] = false;
        }
        auto key = take(step.appends[layer * 2].ready()), value = take(step.appends[layer * 2 + 1].ready());
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          member.next->keys[layer] = outputs.size() == 1 && member.accepted_tokens == tokens ? key
              : take(page_inventory.keys[layer].reframe(member.layout, std::span(&key, 1)));
          member.next->values[layer] = outputs.size() == 1 && member.accepted_tokens == tokens ? value
              : take(page_inventory.values[layer].reframe(member.layout, std::span(&value, 1)));
        }
      }
      if (has_mtp(phase.family->role)) {
        for (std::size_t bank = 12; bank < 14; ++bank) {
          step.appends[bank] = take(g::batch::take_append(run, step.nodes[bank])); step.bound[bank] = false;
        }
        auto key = take(step.appends[12].ready()), value = take(step.appends[13].ready());
        std::vector<std::pair<std::uint32_t, std::size_t>> owners; owners.reserve(outputs.size());
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j]; auto &draft = *member.next->draft;
          draft.key = outputs.size() == 1 ? key : take(draft_inventory.keys[0].reframe(member.draft_layout, std::span(&key, 1)));
          draft.value = outputs.size() == 1 ? value : take(draft_inventory.values[0].reframe(member.draft_layout, std::span(&value, 1)));
          owners.emplace_back(draft.pages.back().page(), j);
        }
        std::sort(owners.begin(), owners.end());
        for (std::size_t bank = 12; bank < 14; ++bank)
          for (std::size_t i = 0; i < step.appends[bank].appender_count(); ++i) {
            auto &tail = take(step.appends[bank].appender(i)); if (!tail.valid()) continue;
            const auto found = std::lower_bound(owners.begin(), owners.end(), std::pair{tail.page(), std::size_t{0}});
            if (found == owners.end() || found->first != tail.page())
              throw std::runtime_error("draft append returned a tail outside all final pages");
            auto &member = step.members[found->second]; auto &draft = *member.next->draft;
            if (tail.reserved_rows() || member.next_draft_tails[bank - 12].valid() ||
                tail.initialized_rows() != draft.frontier % cache::Plan::page_rows)
              throw std::runtime_error("draft append returned an invalid committed frontier");
            member.next_draft_tails[bank - 12] = std::move(tail);
          }
        for (const auto &member : step.members) validate_mtp_snapshot(*member.next);
      }
      hyle::container::SmallVector<std::pair<std::uint32_t, std::size_t>, 1> tail_owners;
      if (!tail_owners.try_reserve(step.members.size())) throw std::bad_alloc{};
      for (std::size_t j = 0; j < step.members.size(); ++j)
        tail_owners.emplace_back(step.members[j].verified_last_page, j);
      std::sort(tail_owners.begin(), tail_owners.end());
      for (std::size_t bank = 0; bank < 12; ++bank)
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
      check(run.unbind(phase.state_resource_ports));
      for (const auto &member : step.members)
        if (member.session->head != member.base || member.session->pending != &step || member.session->stopping)
          throw std::runtime_error("batch snapshot expected-base conflict");
      // All fallible native/export/metadata work is complete. The head swaps
      // below cannot throw and publish the whole batch in this host transaction.
      for (std::size_t j = 0; j < step.members.size(); ++j) {
        auto &member = step.members[j]; member.next->next_id = next_ids[j];
        member.session->tails = std::move(member.next_tails);
        member.session->draft_tails = std::move(member.next_draft_tails);
        member.session->head = std::move(member.next);
        member.session->tail_head = member.session->head.get();
        member.session->draft_tail_head = member.session->head.get();
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
    } catch (...) { abort_episode().IgnoreError(); throw; }
    const auto finished = Clock::now();
    for (const auto &output : outputs) if (output.mtp_ledger) {
      auto &record = output.mtp_ledger->records[output.mtp_ledger->committed_count - 1];
      record.endpoint = finished;
      record.step_seconds = std::chrono::duration<double>(finished - step_start).count();
    }
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      const auto &output = outputs[j]; const auto &session = *output.session; const auto next = next_ids[j];
      if (output.complete_seconds) *output.complete_seconds = std::chrono::duration<double>(finished - step_start).count();
      if (output.host_endpoint) *output.host_endpoint = finished;
      if (output.preparation) {
        const auto report_start = Clock::now();
        for (std::size_t i = 0; i < output.preparation->reports.size(); ++i) {
          output.preparation->nodes[i] = phase.attention_nodes[i].ordinal();
          output.preparation->reports[i] = take(g::batch::report(run, phase.attention_nodes[i]));
        }
        output.preparation->report_snapshot_seconds = std::chrono::duration<double>(Clock::now() - report_start).count();
      }
      StepObservation observed;
      auto &observation = output.observation ? *output.observation : observed;
      observation.input_tokens = tokens; observation.visible = session.length(); observation.next = next;
      observation.graph_executed = true; observation.graph_input_tokens = tokens;
      observation.seconds = std::chrono::duration<double>(finished - start).count();
      observation.submit_seconds = submit_seconds; observation.step_seconds = std::chrono::duration<double>(finished - step_start).count();
      observation.cold_batch_seconds = cold_batch_seconds; observation.replay_route = replay_route;
      observation.capture_recorded = capture_recorded; observation.capture_reused = capture_reused;
      observation.scheduled_progress = static_cast<bool>(scheduler);
      if (!output.observation) { observation.name = output.name; print_step_observation(observation); }
      const auto dump_bytes = [&](mem::ConstBufferView view, const std::filesystem::path &path) {
        std::vector<std::byte> bytes(view.size()); check(execution.read(view, bytes));
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write Qwen trace: " + path.string());
      };
      if (output.trace && !output.trace->empty()) {
        const auto directory = *output.trace / output.name; std::filesystem::create_directories(directory);
        const auto dump_row = [&](Tensor &tensor, const c::ReadLease &generated, const char *file) {
          const auto storage = tensor.buffer ? mem::ConstBufferView(tensor.buffer.view()) : take(generated.view());
          const auto bytes = take(tensor.description.required_bytes()) / outputs.size();
          dump_bytes(take(storage.subview(j * bytes, bytes)), directory / file);
        };
        dump_row(phase.logits, generated_logits, "logits.f32");
        dump_row(phase.final_hidden, generated_hidden, "hidden.bf16");
        if (output.trace_states) {
          for (std::size_t i = 0; i < state_schema.size(); ++i) if (i % 4 != 3) {
            const auto prefix = "layer" + std::to_string(i);
            dump_bytes(take(session.head->state[i].history.view()), directory / (prefix + ".conv.bf16"));
            dump_bytes(take(session.head->state[i].recurrent.view()), directory / (prefix + ".gdn.f32"));
          }
          std::ofstream layout_trace(directory / "kv-layout.json");
          layout_trace << "{\"visible_tokens\":" << session.length() << ",\"page_rows\":" << cache::Plan::page_rows << ",\"pages\":[";
          for (std::size_t i = 0; i < session.head->pages.size(); ++i) layout_trace << (i ? "," : "") << session.head->pages[i].page();
          layout_trace << "]}\n"; if (!layout_trace) throw std::runtime_error("cannot write KV layout trace");
        }
      }
      if (!layer0_probe.empty()) {
        const auto directory = output.probe_subdirectory.empty() ? layer0_probe / output.name
            : layer0_probe / std::filesystem::path(output.probe_subdirectory) / output.name;
        std::filesystem::create_directories(directory);
        for (std::size_t i = 0; i < probe_views.size(); ++i) {
          const auto view = probe_views[i]; const auto bytes = take(view.description().required_bytes());
          if ((outputs.size() != 1 && !view.description().is_dense()) || bytes % outputs.size())
            throw std::runtime_error("published batch diagnostic is not request-major dense");
          dump_bytes(take(view.storage().subview(j * (bytes / outputs.size()), bytes / outputs.size())),
              directory / (std::string(layer0_probe_names[i]) + ".bf16"));
        }
      }
    }
  }

  void finish_evaluations(std::span<const EvaluationOutput> outputs,
                         std::span<std::int32_t> next_ids) {
    if (active_step && active_step->phase && verifies_two(active_step->phase->family->role))
      throw std::runtime_error("verification requires explicit accepted-boundary selection");
    finish_selected_evaluations(outputs, next_ids,
        [](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> next) {
          for (std::size_t j = 0; j < step.members.size(); ++j) {
            step.members[j].accepted_tokens = step.input_tokens; next[j] = ids[j];
          }
        });
  }

  // Generation policy observes only the original terminal graph. Pending has
  // already been emitted; an accepted draft and its bonus are new outputs.
  void finish_mtp_rounds(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results) {
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
            result.target_second = -1; result.agreed = false; result.accepted_draft = false;
            if (drafts_token(role)) {
              if (result.draft < 0 || result.draft >= vocabulary) throw std::runtime_error("MTP draft ID outside vocabulary");
              result.target_second = ids[2 * j + 1]; result.agreed = result.draft == result.target_first;
              result.accepted_draft = result.agreed && selection.output_quota >= 2 && !ended(result.target_first);
            }
            const auto accepted = result.accepted_draft ? 2u : 1u;
            member.accepted_tokens = accepted;
            selected[j] = result.next_id = result.accepted_draft ? result.target_second : result.target_first;
            result.processed_target = accepted; result.processed_draft = draft_lag(role) + 1;
            result.emitted.push_back(result.target_first);
            if (result.accepted_draft) result.emitted.push_back(result.target_second);
            result.stopped = ended(result.emitted.back());
            if (outputs[j].mtp_ledger) {
              auto &record = outputs[j].mtp_ledger->next();
              if (!record.generation || record.quota != selection.output_quota)
                throw std::runtime_error("MTP ledger quota differs from its original selection");
              record.draft = result.draft; record.target_first = result.target_first;
              record.target_second = result.target_second; record.agreed = result.agreed;
              record.accepted_draft = result.accepted_draft; record.stopped = result.stopped;
              record.emitted = {result.emitted[0], result.emitted.size() == 2 ? result.emitted[1] : -1};
              record.emitted_count = result.emitted.size();
            }
          }
        });
  }

  MtpRoundResult finish_mtp_round(Session &session, std::int32_t expected_pending,
      std::size_t output_quota, std::span<const std::int32_t> eos = {}, const EvaluationOutput *output = nullptr) {
    const std::array selections{MtpGenerationSelection{session.shared_from_this(), expected_pending, output_quota}};
    const std::array outputs{output ? *output : EvaluationOutput{.session = session.shared_from_this(), .name = "mtp"}};
    std::array<MtpRoundResult, 1> results;
    finish_mtp_rounds(selections, eos, outputs, results);
    return std::move(results[0]);
  }

  std::int32_t finish_evaluation(Session &session, const std::string &name,
                         const std::filesystem::path &trace = {}, bool trace_states = true,
                         double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
                         StepObservation *observation = nullptr, PreparationObservation *preparation = nullptr,
                         std::string_view probe_subdirectory = {},
                         MtpReplayLedger *mtp_ledger = nullptr, const MtpReplayRecord *mtp_expected = nullptr) {
    const std::array outputs{EvaluationOutput{session.shared_from_this(), name, &trace, trace_states,
        complete_seconds, host_endpoint, observation, preparation, probe_subdirectory, mtp_ledger, mtp_expected}};
    std::int32_t next = -1; finish_evaluations(outputs, std::span(&next, 1)); return next;
  }

  std::int32_t evaluate(Session &session, std::span<const std::int32_t> ids, const std::string &name,
                         const std::filesystem::path &trace = {}, bool trace_states = true,
                         double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
                         StepObservation *observation = nullptr, PreparationObservation *preparation = nullptr,
                         std::string_view probe_subdirectory = {}) {
    prepare_evaluation(session, ids, preparation);
    try { submit_evaluation(session); return finish_evaluation(session, name, trace, trace_states,
        complete_seconds, host_endpoint, observation, preparation, probe_subdirectory); }
    catch (...) { abort_episode().IgnoreError(); throw; }
  }

};

hyle::Status Session::stop() noexcept {
  if (stopping) return hyle::Status::Ok();
  stopping = true;
  if (pending) {
    pending->cancelled = true;
    if (pending->completion.valid()) {
      const auto cancelled = pending->completion.request_cancel();
      if (!cancelled.IsUnimplemented()) retirement_status.Update(cancelled);
    }
    if (pending->run) pending->run->close().IgnoreError();
  }
  return retirement_status;
}
hyle::Result<bool> Session::try_retire() noexcept {
  if (!stopping) return std::unexpected(hyle::Status::FailedPrecondition("stop the Qwen session before retirement"));
  if (pending) {
    HYLE_ASSIGN_OR_RETURN(auto retired, model_->try_retire_episode());
    if (!retired) return false;
  }
  tails = {}; draft_tails = {}; tail_head = nullptr; draft_tail_head = nullptr; head.reset(); return true;
}
hyle::Status Session::drain() noexcept {
  hyle::Status status;
  if (pending) status.Update(model_->abort_episode());
  status.Update(retirement_status); return status;
}
Session::~Session() { drain().IgnoreError(); }

std::vector<std::int32_t> read_ids(const std::filesystem::path &path) {
  std::ifstream stream(path); if (!stream) throw std::runtime_error("cannot open token IDs");
  std::vector<std::int32_t> ids; std::int64_t value = 0;
  while (stream >> value) {
    if (value < 0 || value >= vocabulary) throw std::runtime_error("input token outside vocabulary");
    ids.push_back(static_cast<std::int32_t>(value));
  }
  if (!stream.eof()) throw std::runtime_error("malformed input IDs");
  return ids;
}

struct MtpRestoreStats final {
  std::size_t graph_calls = 0, target_rows = 0, draft_rows = 0;
  std::size_t normalized_lag_rows = 0;
  struct DecisionChange { std::size_t episode; std::int32_t historical, fresh; };
  std::vector<DecisionChange> decision_changes;
  double step_seconds = 0, seconds = 0;
};

struct MtpGenerationRequest {
  std::string id, memory_name;
  std::span<const std::int32_t> prompt;
  std::filesystem::path trace;
  std::shared_ptr<Session> session;
  cache::TokenPlanRef input;
  std::shared_ptr<cache::ReplayHistory> history;
  std::vector<std::int32_t> generated;
  std::vector<MtpRoundResult> rounds;
  std::vector<std::vector<std::int32_t>> round_emitted;
  Clock::time_point prefill_start{}, prefill_endpoint{}, endpoint{};
  double prefill_step_seconds = 0, decode_step_sum_seconds = 0;
  std::size_t verified_rows = 0, draft_rows = 0, draft_rounds = 0, agreements = 0, accepts = 0;
  std::size_t visible_tokens = 0, draft_frontier = 0, draft_lag = 0;
  std::size_t reused_tokens = 0, prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0;
  std::size_t prefill_graph_calls = 0, prefill_graph_tokens = 0, prefill_draft_rows = 0;
  std::unique_ptr<MtpReplayLedger> ledger;
  std::size_t ledger_observed = 0;
  std::span<const benchmark::MtpTurn> turns;
  struct TurnObservation {
    Clock::time_point start{}, prefill_endpoint{}, endpoint{};
    std::size_t output_begin = 0, output_end = 0;
    std::size_t context_before = 0, context_after = 0;
  };
  std::vector<TurnObservation> turn_observations;
  std::vector<std::shared_ptr<std::vector<std::int32_t>>> warm_inputs;
  std::size_t turn_index = 0, turn_output_begin = 0, last_idle = 0;
  Clock::time_point admission_start{};
  MtpRestoreStats restored;
  std::size_t restore_count = 0, idle_evictions = 0;
  std::int32_t fresh_next_id = -1;
  bool prefilled = false, done = false;

  MtpGenerationRequest(std::string name, std::span<const std::int32_t> ids,
                       std::filesystem::path output, std::size_t steps)
      : id(std::move(name)), memory_name(id.empty() ? "generation" : "generation_" + id),
        prompt(ids), trace(std::move(output)) {
    generated.reserve(steps); rounds.reserve(steps - 1);
  }
};

MtpReplayRecord &prepare_mtp_replay_record(MtpReplayLedger &ledger, const Session &session,
    GraphRole role, std::span<const std::int32_t> ids, bool generation,
    std::size_t quota, std::size_t turn, std::shared_ptr<const void> owner = {}) {
  if (ids.empty() || !session.head || (generation && ids.size() != 1))
    throw std::runtime_error("MTP replay record requires its actual input and complete base");
  auto &record = ledger.next(); record = {};
  record.role = role; record.generation = generation; record.quota = quota; record.turn = turn;
  record.before = mtp_boundary(*session.head);
  if (generation) record.pending_input = ids.front();
  else { record.input_owner = std::move(owner); record.prompt_ids = ids; }
  return record;
}

void import_mtp_prompt_ledger(MtpReplayLedger &ledger, const cache::ReplayPrefix &prefix,
                              bool full_hit) {
  if (ledger.committed_count || !prefix.history || prefix.chunks > ledger.records.size())
    throw std::runtime_error("MTP initial prompt import requires an empty bounded ledger");
  std::int32_t previous = -1;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    auto &record = ledger.next();
    record.role = part.count == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
    record.input_owner = part.input;
    record.prompt_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
    record.before = {part.first, part.first, 0, previous, -1};
    record.after = {part.first + part.count, part.first + part.count, 0, part.next_id, -1};
    record.accepted = part.count; record.draft_rows = part.count;
    record.target_first = part.next_id; record.emitted = {part.next_id, -1}; record.emitted_count = 1;
    record.stopped = is_end_of_generation(part.next_id);
    record.user_output = full_hit && i + 1 == prefix.chunks;
    record.imported = true; record.committed = true; ++ledger.committed_count;
    record.requests = 0; // Role-only source history does not claim original batch geometry.
    previous = part.next_id;
  }
}

// Consume committed host facts even when a subsequent diagnostic write threw.
// The authoritative ledger was installed by the original head-swap transaction.
void observe_mtp_commits(MtpGenerationRequest &request) {
  if (!request.ledger) return;
  while (request.ledger_observed < request.ledger->committed_count) {
    const auto &record = request.ledger->records[request.ledger_observed];
    if (record.user_output) {
      if (record.generation) {
        MtpRoundResult result;
        result.pending = record.pending_input; result.draft = record.draft;
        result.target_first = record.target_first; result.target_second = record.target_second;
        result.next_id = record.after.next_id; result.agreed = record.agreed;
        result.accepted_draft = record.accepted_draft; result.stopped = record.stopped;
        result.processed_target = record.accepted; result.processed_draft = record.draft_rows;
        result.emitted = std::move(request.round_emitted[request.rounds.size()]);
        result.emitted.assign(record.emitted.begin(), record.emitted.begin() + record.emitted_count);
        request.rounds.push_back(std::move(result));
        request.verified_rows += drafts_token(record.role) ? 2 : 1;
        request.draft_rows += record.draft_rows; request.draft_rounds += drafts_token(record.role) ? 1 : 0;
        request.agreements += record.agreed ? 1 : 0; request.accepts += record.accepted_draft ? 1 : 0;
        request.decode_step_sum_seconds += record.step_seconds;
      } else if (!record.imported) {
        ++request.prefill_graph_calls; request.prefill_graph_tokens += record.ids().size();
        request.prefill_draft_rows += record.draft_rows;
        request.prefill_step_seconds += record.step_seconds;
      }
      request.generated.insert(request.generated.end(), record.emitted.begin(), record.emitted.begin() + record.emitted_count);
    }
    request.visible_tokens = record.after.length; request.draft_frontier = record.after.frontier;
    request.draft_lag = record.after.lag;
    if (!record.imported) request.endpoint = record.endpoint;
    if (!record.generation && !record.imported) {
      request.prefill_endpoint = record.endpoint; request.prefilled = true;
      if (!request.turn_observations.empty()) {
        auto &turn = request.turn_observations[request.turn_index];
        turn.start = request.prefill_start; turn.prefill_endpoint = record.endpoint;
      }
    }
    if (!request.turn_observations.empty()) {
      auto &turn = request.turn_observations[request.turn_index];
      turn.endpoint = request.endpoint; turn.output_end = request.generated.size();
      turn.context_after = request.visible_tokens;
      request.done = request.generated.size() >= request.turn_output_begin + request.turns[request.turn_index].steps ||
          is_end_of_generation(record.after.next_id);
    }
    ++request.ledger_observed;
  }
}

// Only the serialized request owners and this directory retain committed
// Snapshots here. Hit/local Snapshot references are dropped before this gate.
void make_mtp_room(Model &model, cache::PrefixCache *prefix, Session *owner,
                    std::span<const MtpGenerationRequest> requests) {
  if (!prefix) return;
  const auto limit = model.cache_plan.resident_sessions;
  const auto fits = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("MTP admission requires retired committed heads within N");
    if (!owner) return live < limit;
    bool external = prefix->retains(owner->head.get());
    for (const auto &request : requests)
      if (request.session && request.session.get() != owner && request.session->head == owner->head) external = true;
    const auto retires = external ? std::size_t{0} : std::size_t{1};
    if (live < retires) throw std::runtime_error("MTP committed version inventory is inconsistent");
    return live - retires < limit;
  };
  while (!fits()) if (!prefix->evict_one())
    throw std::runtime_error("MTP prompt checkpoints retain all committed versions; no unfinished session may be evicted");
}

std::shared_ptr<Session> restore_mtp_session(Model &model, const MtpReplayLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    MtpRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const MtpGenerationRequest> requests = {}) {
  const auto started = Clock::now();
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP restore requires a complete zero-origin ledger and an idle carrier");
  for (std::size_t i = 0; i < ledger.committed_count; ++i)
    if (ledger.records[i].requests != 1 || ledger.records[i].member_index != 0)
      throw std::runtime_error("strict MTP replay requires recorded original R1 geometry; use committed-token rebuild for cache recovery");
  make_mtp_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before restoring within N committed versions");
  const auto &first = ledger.records[0];
  const std::array calls{first.ids().size()};
  auto session = model.create_session(calls, {}, first.role);
  MtpReplayLedger actual(ledger.committed_count);
  const std::array<std::int32_t, 2> eos{248044, 248046};
  try {
    for (std::size_t index = 0; index < ledger.committed_count; ++index) {
      const auto &expected = ledger.records[index];
      if (!expected.committed) throw std::runtime_error("MTP restore ledger contains an uncommitted gap");
      const auto pending = expected.pending_input;
      const auto ids = expected.generation ? std::span<const std::int32_t>(&pending, 1) : expected.ids();
      prepare_mtp_replay_record(actual, *session, expected.role, ids, expected.generation,
          expected.quota, expected.turn, expected.input_owner);
      make_mtp_room(model, prefix, session.get(), requests);
      model.prepare_evaluation(*session, ids, nullptr, expected.role); model.submit_evaluation(*session);
      if (stats) {
        ++stats->graph_calls;
        stats->target_rows = cache::add(stats->target_rows, drafts_token(expected.role) ? 2 : ids.size());
        stats->draft_rows = cache::add(stats->draft_rows,
            is_mtp_prefill(expected.role) ? cache::add(ids.size(), draft_lag(expected.role)) : draft_lag(expected.role) + 1);
      }
      const auto name = "episode" + std::to_string(index);
      double step_seconds = 0;
      if (expected.generation) {
        const EvaluationOutput output{.session = session, .name = name, .trace = &trace,
            .trace_states = trace_states, .complete_seconds = &step_seconds,
            .mtp_ledger = &actual, .mtp_expected = &expected};
        (void)model.finish_mtp_round(*session, pending, expected.quota, eos, &output);
      } else {
        (void)model.finish_evaluation(*session, name, trace, trace_states, &step_seconds,
            nullptr, nullptr, nullptr, {}, &actual, &expected);
      }
      if (stats) stats->step_seconds += step_seconds;
    }
  } catch (...) {
    session->stop().IgnoreError(); session->drain().IgnoreError();
    auto retired = session->try_retire(); if (!retired) retired.error().IgnoreError();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// Cache reconstruction consumes immutable committed token chunks. It does not
// resample past outputs or claim to replay an old batch's natural decisions.
std::shared_ptr<Session> rebuild_mtp_session(Model &model, const MtpReplayLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    MtpRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const MtpGenerationRequest> requests = {}) {
  const auto started = Clock::now();
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP rebuild requires a complete zero-origin committed history and idle carrier");
  std::size_t committed_length = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i];
    if (!record.committed || !record.accepted || (record.generation &&
        (record.accepted > 2 || record.accepted_inputs[0] != record.pending_input ||
          (record.accepted == 2 && (!record.accepted_draft || record.accepted_inputs[1] != record.draft)))))
      throw std::runtime_error("MTP rebuild has an invalid committed input boundary");
    const auto ids = record.committed_ids();
    if (record.before.length != committed_length || ids.empty() || ids.size() != record.accepted ||
        record.accepted > model.capacity - committed_length)
      throw std::runtime_error("MTP rebuild committed input chunks are not a continuous bounded prefix");
    committed_length = cache::add(committed_length, record.accepted);
    if (record.after.length != committed_length) throw std::runtime_error("MTP rebuild committed length is inconsistent");
    for (auto id : ids) if (id < 0 || id >= vocabulary) throw std::runtime_error("MTP rebuild token is outside vocabulary");
  }
  make_mtp_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before rebuilding within N committed versions");
  if (stats) stats->decision_changes.reserve(cache::add(stats->decision_changes.size(), ledger.committed_count));
  const auto first_ids = ledger.records[0].committed_ids();
  const std::array calls{first_ids.size()};
  auto session = model.create_session(calls, {}, first_ids.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill);
  MtpReplayLedger actual(ledger.committed_count);
  std::size_t observed = 0;
  const auto observe = [&] {
    while (observed < actual.committed_count) {
      const auto &fresh = actual.records[observed]; const auto &old = ledger.records[observed];
      if (stats) {
        ++stats->graph_calls; stats->target_rows = cache::add(stats->target_rows, fresh.accepted);
        stats->draft_rows = cache::add(stats->draft_rows, fresh.draft_rows);
        stats->step_seconds += fresh.step_seconds;
        if (fresh.after.next_id != old.after.next_id)
          stats->decision_changes.push_back({observed, old.after.next_id, fresh.after.next_id});
      }
      ++observed;
    }
  };
  try {
    for (std::size_t i = 0; i < ledger.committed_count; ++i) {
      const auto &source = ledger.records[i]; const auto ids = source.committed_ids();
      const auto role = ids.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
      prepare_mtp_replay_record(actual, *session, role, ids, false, 0, source.turn, source.input_owner);
      make_mtp_room(model, prefix, session.get(), requests);
      model.prepare_evaluation(*session, ids, nullptr, role); model.submit_evaluation(*session);
      (void)model.finish_evaluation(*session, "episode" + std::to_string(i), trace, trace_states,
          nullptr, nullptr, nullptr, nullptr, {}, &actual);
      observe();
    }
    if (session->length() != ledger.records[ledger.committed_count - 1].after.length || session->head->draft->lag)
      throw std::runtime_error("MTP committed-token rebuild has an inconsistent complete boundary");
    if (stats) stats->normalized_lag_rows = cache::add(stats->normalized_lag_rows, ledger.records[ledger.committed_count - 1].after.lag);
  } catch (...) {
    observe(); session->stop().IgnoreError(); session->drain().IgnoreError();
    auto retired = session->try_retire(); if (!retired) retired.error().IgnoreError();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

struct PreparedMtpCall final {
  MtpGenerationRequest *request = nullptr;
  GraphRole role = GraphRole::kMtpPrefill;
  std::span<const std::int32_t> prompt;
  std::int32_t pending = -1;
  std::size_t quota = 0, steps = 0;
  bool generation = false, full_hit = false, initial_prefix = false;
  std::string name;
  double step_seconds = 0;
  std::span<const std::int32_t> ids() const noexcept {
    return generation ? std::span(&pending, 1) : prompt;
  }
};

PreparedMtpCall prepare_mtp_call(Model &model, MtpGenerationRequest &request, std::size_t steps,
    cache::PrefixCache *prefix, std::span<const MtpGenerationRequest> requests) {
  if (request.done) throw std::runtime_error("MTP request has already completed");
  if (request.ledger && (steps > request.generated.capacity() || request.round_emitted.size() < steps - 1))
    throw std::runtime_error("MTP observer slots must be reserved for every turn before execution");
  PreparedMtpCall call; call.request = &request; call.steps = steps;
  call.generation = request.prefilled;
  if (!call.generation) {
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_begin\n";
    request.prefill_start = request.admission_start == Clock::time_point{} ? Clock::now()
        : std::exchange(request.admission_start, Clock::time_point{});
    call.initial_prefix = prefix && request.turn_index == 0;
    if (call.initial_prefix) {
      auto hit = prefix->lookup(*request.input, request.prompt.size(), 1, &model, model.math_identity());
      if (hit) {
        if (!hit.snapshot->draft || hit.snapshot->draft->lag)
          throw std::runtime_error("MTP prompt directory contains a non-prompt frontier");
        request.reused_tokens = hit.snapshot->length; request.prefix_hits = 1;
        request.prefix_full_hits = request.reused_tokens == request.prompt.size() ? 1 : 0;
        request.history = cache::copy_history(hit.replay, model.capacity);
        if (request.ledger) {
          import_mtp_prompt_ledger(*request.ledger, hit.replay, request.prefix_full_hits != 0);
          request.ledger_observed = request.ledger->committed_count;
        }
        const auto suffix = request.prompt.size() - request.reused_tokens;
        const std::array chunks{suffix};
        request.session = model.create_session(suffix ? std::span<const std::size_t>(chunks) : std::span<const std::size_t>{},
            std::move(hit.snapshot), (suffix == 1 || request.prompt.size() == 1) ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill);
        hit.reset();
      } else { request.prefix_misses = 1; request.history = std::make_shared<cache::ReplayHistory>(model.capacity); }
    }
    call.prompt = request.prompt.subspan(request.turn_index == 0 ? request.reused_tokens : 0);
    const auto lag = request.session ? request.session->head->draft->lag : 0;
    call.role = call.prompt.size() == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
    if (!request.session) {
      make_mtp_room(model, prefix, nullptr, requests);
      const std::array chunks{call.prompt.size()}; request.session = model.create_session(chunks, {}, call.role);
    }
    if (!request.turn_observations.empty())
      request.turn_observations[request.turn_index].context_before = request.session->length();
    call.full_hit = call.prompt.empty();
    call.name = request.turn_index ? "round" + std::to_string(request.turn_index + 1) + "_prefill" : "prefill";
  } else {
    call.quota = steps - request.generated.size(); call.pending = request.generated.back();
    const bool one = call.quota == 1 || model.capacity - request.session->length() == 1;
    const auto lag = request.session->head->draft->lag;
    call.role = one ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound);
    call.name = (request.turn_index ? "round" + std::to_string(request.turn_index + 1) + "_" : "") +
        "mtp" + std::to_string(request.rounds.size() + 1);
  }
  return call;
}

bool make_mtp_batch_room(Model &model, cache::PrefixCache *prefix,
    std::span<const PreparedMtpCall> calls, std::span<const MtpGenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const auto fits = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("MTP batch admission requires retired heads within N");
    std::size_t released = 0;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      const auto *head = calls[j].request->session->head.get();
      bool seen = false;
      for (std::size_t k = 0; k < j; ++k) seen = seen || calls[k].request->session->head.get() == head;
      if (seen) continue;
      bool external = prefix && prefix->retains(head);
      for (const auto &request : requests) if (request.session && request.session->head.get() == head) {
        bool selected = false;
        for (const auto &call : calls) selected = selected || call.request->session == request.session;
        external = external || !selected;
      }
      if (!external) ++released;
    }
    if (released > live) throw std::runtime_error("MTP batch version inventory is inconsistent");
    return calls.size() <= limit - (live - released);
  };
  while (!fits()) if (!prefix || !prefix->evict_one()) return false;
  return true;
}

void finish_mtp_call(Model &model, PreparedMtpCall &call, MtpRoundResult *result) {
  auto &request = *call.request;
  if (call.full_hit) {
    request.generated.push_back(request.session->head->next_id);
    request.prefill_endpoint = Clock::now(); request.endpoint = request.prefill_endpoint;
    std::cerr << "prefix_hit name=prefill reused_tokens=" << request.reused_tokens << " graph_calls=0 visible=" << request.session->length() << '\n';
    if (!request.trace.empty()) {
      const auto path = request.trace / "prefill"; std::filesystem::create_directories(path);
      std::ofstream metadata(path / "cache-hit.json");
      if (!metadata) throw std::runtime_error("cannot write MTP prefix-hit metadata");
      metadata << "{\"prefix_hit\":true,\"graph_calls\":0,\"tensor_payload_available\":false,\"reused_tokens\":"
               << request.reused_tokens << ",\"next_id\":" << request.generated.back() << "}\n";
    }
  } else if (request.ledger) observe_mtp_commits(request);
  else if (!call.generation) {
    request.generated.push_back(request.session->head->next_id); request.prefill_step_seconds += call.step_seconds;
    ++request.prefill_graph_calls; request.prefill_graph_tokens += call.ids().size();
    request.prefill_draft_rows += cache::add(call.ids().size(), draft_lag(call.role));
  } else {
    request.decode_step_sum_seconds += call.step_seconds;
    request.verified_rows += drafts_token(call.role) ? 2 : 1; request.draft_rows += result->processed_draft;
    request.draft_rounds += drafts_token(call.role) ? 1 : 0; request.agreements += result->agreed ? 1 : 0;
    request.accepts += result->accepted_draft ? 1 : 0;
    request.generated.insert(request.generated.end(), result->emitted.begin(), result->emitted.end());
    request.rounds.push_back(std::move(*result));
  }
  if (!call.generation) {
    request.prefilled = true; request.endpoint = request.prefill_endpoint;
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_end\n"
              << "memory_phase=" << request.memory_name << "_decode_begin\n";
  }
  request.visible_tokens = request.session->length(); request.draft_frontier = request.session->head->draft->frontier;
  request.draft_lag = request.session->head->draft->lag;
  request.fresh_next_id = request.session->head->next_id;
  request.done = request.generated.size() >= call.steps || is_end_of_generation(request.generated.back());
  if (!request.turn_observations.empty()) {
    auto &turn = request.turn_observations[request.turn_index];
    if (!call.generation) { turn.start = request.prefill_start; turn.prefill_endpoint = request.prefill_endpoint; }
    turn.endpoint = request.endpoint; turn.output_end = request.generated.size();
    turn.context_after = request.visible_tokens;
  }
  if (request.done) std::cerr << "memory_phase=" << request.memory_name << "_decode_end\n";
  (void)model;
}

void execute_mtp_calls(Model &model, std::span<PreparedMtpCall> calls, bool trace_states,
    cache::PrefixCache *prefix = nullptr, std::span<const MtpGenerationRequest> requests = {}) {
  if (calls.empty()) return;
  if (calls.size() == 1 && calls[0].full_hit) { finish_mtp_call(model, calls[0], nullptr); return; }
  if (!make_mtp_batch_room(model, prefix, calls, requests)) throw std::runtime_error("MTP batch exceeds committed head budget");
  hyle::container::SmallVector<EvaluationRequest, 2> inputs;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  if (!inputs.try_reserve(calls.size()) || !outputs.try_reserve(calls.size()) ||
      !selections.try_reserve(calls.size()) || !results.try_resize(calls.size()) || !next.try_resize(calls.size()))
    throw std::bad_alloc{};
  const auto role = calls[0].role; const auto generation = calls[0].generation;
  for (auto &call : calls) {
    auto &request = *call.request;
    if (call.full_hit || call.role != role || call.generation != generation || call.ids().size() != calls[0].ids().size())
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    if (request.ledger) prepare_mtp_replay_record(*request.ledger, *request.session, role, call.ids(), generation,
        call.quota, request.turn_index, generation ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(request.warm_inputs.at(request.turn_index)));
    inputs.push_back({request.session, call.ids()});
    outputs.push_back({.session = request.session, .name = call.name, .trace = &request.trace,
        .trace_states = trace_states, .complete_seconds = &call.step_seconds,
        .host_endpoint = generation ? &request.endpoint : &request.prefill_endpoint, .mtp_ledger = request.ledger.get()});
    if (generation) selections.push_back({request.session, call.pending, call.quota});
  }
  model.prepare_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role); model.submit_evaluations();
  const std::array<std::int32_t, 2> eos{248044, 248046};
  try {
    if (generation) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()));
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()));
    for (std::size_t j = 0; j < calls.size(); ++j) finish_mtp_call(model, calls[j], generation ? &results[j] : nullptr);
  } catch (...) {
    for (auto &call : calls) observe_mtp_commits(*call.request);
    throw;
  }
}

void publish_mtp_prompt(Model &model, PreparedMtpCall &call, cache::PrefixCache *prefix) {
  if (!call.initial_prefix || call.full_hit) return;
  auto &request = *call.request;
  request.history->append(request.input, request.reused_tokens, call.ids().size(), request.session->head->next_id);
  if (request.session->head->draft->lag) throw std::runtime_error("only a complete lag0 MTP prompt may enter the directory");
  prefix->publish(request.input, request.session->head, {request.history, request.history->chunks.size(), request.prompt.size()});
  (void)model;
}

void advance_mtp_request(Model &model, MtpGenerationRequest &request,
    std::size_t steps, bool trace_states, cache::PrefixCache *prefix = nullptr,
    std::span<const MtpGenerationRequest> requests = {}) {
  std::array calls{prepare_mtp_call(model, request, steps, prefix, requests)};
  execute_mtp_calls(model, calls, trace_states, prefix, requests);
  publish_mtp_prompt(model, calls[0], prefix);
}

void retire_mtp_request(MtpGenerationRequest &request) {
  check(request.session->stop());
  if (!take(request.session->try_retire()))
    throw std::runtime_error("finished MTP request still retains an in-flight episode");
  request.session.reset();
}

void print_mtp_rebuild(const MtpRestoreStats &stats) {
  std::cout << ",\"restore_contract\":\"recompute_committed_inputs\",\"restored_graph_calls\":" << stats.graph_calls
      << ",\"restored_target_rows\":" << stats.target_rows << ",\"restored_draft_rows\":" << stats.draft_rows
      << ",\"normalized_lag_rows\":" << stats.normalized_lag_rows << ",\"decision_changes\":[";
  for (std::size_t i = 0; i < stats.decision_changes.size(); ++i) {
    const auto &change = stats.decision_changes[i];
    std::cout << (i ? "," : "") << "{\"episode\":" << change.episode << ",\"historical_next\":" << change.historical
        << ",\"fresh_next\":" << change.fresh << '}';
  }
  std::cout << ']';
}

void print_mtp_request(const Model &model, const MtpGenerationRequest &request, std::size_t steps,
                       Clock::time_point startup, Clock::time_point ready) {
  const auto &prompt = request.prompt;
  const auto &generated = request.generated;
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
    prefill_seconds = 0; decode_seconds = 0; logical_prompt = request.turns.size() - 1; turn_count = request.turns.size();
    for (std::size_t i = 0; i < request.turns.size(); ++i) {
      logical_prompt = cache::add(logical_prompt, request.turns[i].prompt.size());
      const auto &turn = request.turn_observations[i];
      prefill_seconds += std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count();
      decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count();
    }
  }
  const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
  std::cout << std::setprecision(17)
            << "{\"source_id\":\"" << source_id << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << score_policy
            << "\",\"mtp\":true,\"text_tensors\":320,\"mtp_tensors\":15"
            << model.cache_report()
            << ",\"visible_tokens\":" << request.visible_tokens << ",\"draft_frontier\":" << request.draft_frontier
            << ",\"draft_lag\":" << request.draft_lag << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
  std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
            << ",\"session_seconds\":" << session_seconds
            << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
            << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
            << ",\"prefill_step_seconds\":" << prefill_step_seconds
            << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
            << ",\"step_sum_seconds\":" << prefill_step_seconds + decode_step_sum_seconds + request.restored.step_seconds
            << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
            << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated.size() - turn_count) / decode_seconds : 0)
            << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
            << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
            << ",\"generated_decode_tokens\":" << generated.size() - turn_count << ",\"verified_target_rows\":" << verified_rows
            << (request.turn_observations.empty() ? ",\"actual_graph_calls\":" : ",\"graph_participations\":")
            << rounds.size() + request.prefill_graph_calls + request.restored.graph_calls
            << ",\"prefill_graph_calls\":" << request.prefill_graph_calls << ",\"decode_graph_calls\":" << rounds.size()
            << ",\"prefill_target_rows\":" << request.prefill_graph_tokens << ",\"prefill_draft_rows\":" << request.prefill_draft_rows
            << ",\"actual_target_rows\":" << request.prefill_graph_tokens + verified_rows + request.restored.target_rows
            << ",\"actual_draft_rows\":" << request.prefill_draft_rows + draft_rows + request.restored.draft_rows
            << ",\"committed_target_rows\":" << request.visible_tokens
            << ",\"committed_target_decode_rows\":" << request.visible_tokens - logical_prompt
            << ",\"draft_rows\":" << draft_rows << ",\"draft_rounds\":" << draft_rounds
            << ",\"agreements\":" << agreements << ",\"accepted_drafts\":" << accepts
            << ",\"diagnostic\":" << (trace.empty() ? "false" : "true");
  if (!request.turn_observations.empty()) {
    std::cout << ",\"emitted_pending\":" << generated.back() << ",\"fresh_next_id\":" << request.fresh_next_id;
    print_mtp_rebuild(request.restored);
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
      std::cout << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << request.turns[i].steps
          << ",\"new_user_tokens\":" << request.turns[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
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
    std::cout << (i ? "," : "") << "{\"pending\":" << row.pending << ",\"draft\":" << row.draft
              << ",\"target_first\":" << row.target_first << ",\"target_second\":" << row.target_second
              << ",\"accepted_draft\":" << (row.accepted_draft ? "true" : "false")
              << ",\"committed_target_rows\":" << row.processed_target << ",\"draft_rows\":" << row.processed_draft << '}';
  }
  std::cout << "]}";
}

int run_mtp(Model &model, std::span<const std::int32_t> prompt, std::size_t steps,
             const std::filesystem::path &trace, bool trace_states,
             Clock::time_point startup, Clock::time_point constructed) {
  if (prompt.empty()) throw std::runtime_error("MTP requires a nonempty prompt");
  const auto prefill_role = prompt.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
  const std::array prefill_calls{prompt.size()}, single{std::size_t{1}}, pair{std::size_t{2}};
  model.prepare_family(prefill_calls, false, 1, prefill_role);
  model.plan_replay_uses(prompt.size(), 1, 1, prefill_role);
  const std::array roles{GraphRole::kMtpRound, GraphRole::kMtpCatchupRound,
      GraphRole::kMtpSingle, GraphRole::kMtpCatchupSingle};
  if (steps > 1) for (auto role : roles) {
    if (role != prefill_role) model.prepare_family(drafts_token(role) ? std::span(pair) : std::span(single), false, 1, role);
    model.plan_replay_uses(drafts_token(role) ? 2u : 1u, 1, steps - 1, role);
  }
  MtpGenerationRequest request({}, prompt, trace, steps);
  request.session = model.create_session(prefill_calls, {}, prefill_role);
  model.prepare_execution(prefill_calls, 1, prefill_role);
  if (steps > 1) for (auto role : roles) if (drafts_token(role))
    model.prepare_execution(std::span(pair), 1, role);
  const auto ready = Clock::now();
  std::cerr << "mtp_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
  std::cerr << "measure_memory name=ready_before_first\n";
  while (!request.done) advance_mtp_request(model, request, steps, trace_states);
  print_mtp_request(model, request, steps, startup, ready);
  std::cout << '\n';
  retire_mtp_request(request);
  return 0;
}

int run_mtp_sessions(Model &model, std::span<const benchmark::MtpSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans) {
  auto prefix = prefix_enabled ? std::make_unique<cache::PrefixCache>(model.cache_plan.resident_sessions) : nullptr;
  std::vector<MtpGenerationRequest> requests;
  requests.reserve(schedules.size());
  std::size_t total_turns = 0, requested_outputs = 0, decode_bound = 0, restore_bound = 0;
  for (const auto &schedule : schedules) {
    std::size_t outputs = 0;
    for (const auto &turn : schedule.turns) outputs = cache::add(outputs, turn.steps);
    const auto &first = schedule.turns.front().prompt;
    requests.emplace_back(schedule.id, first, trace.empty() ? trace : trace /
        ("session" + std::to_string(requests.size()) + "_" + schedule.id), outputs);
    auto &request = requests.back(); request.turns = schedule.turns;
    request.turn_observations.resize(schedule.turns.size());
    request.warm_inputs.reserve(schedule.turns.size());
    for (std::size_t i = 0; i < schedule.turns.size(); ++i) {
      const auto &ids = schedule.turns[i].prompt;
      auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(ids.size(), i ? 1 : 0));
      std::copy(ids.begin(), ids.end(), owner->begin() + (i ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
    request.prompt = *request.warm_inputs[0];
    const auto ledger_capacity = cache::add(first.size(), outputs);
    request.ledger = std::make_unique<MtpReplayLedger>(ledger_capacity);
    request.round_emitted.resize(outputs - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(2);
    if (prefix) request.input = token_plans[requests.size() - 1];
    total_turns = cache::add(total_turns, schedule.turns.size());
    requested_outputs = cache::add(requested_outputs, outputs);
    decode_bound = cache::add(decode_bound, outputs - schedule.turns.size());
    if (model.cache_plan.resident_sessions < schedules.size())
      restore_bound = cache::add(restore_bound, cache::multiply(schedule.turns.size() - 1, ledger_capacity));
  }
  const std::array pair{std::size_t{2}};
  std::map<GraphRole, std::map<std::size_t, std::size_t>> prompt_uses;
  std::map<GraphRole, std::map<std::size_t, std::vector<std::size_t>>> support;
  const auto note_prompt = [&](GraphRole role, std::size_t tokens, std::size_t member) {
    auto &count = prompt_uses[role][tokens]; count = cache::add(count, 1);
    auto &members = support[role][tokens];
    if (std::find(members.begin(), members.end(), member) == members.end()) members.push_back(member);
  };
  std::size_t decoding_requests = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    std::vector<std::size_t> suffixes{requests[i].prompt.size()};
    if (prefix) for (std::size_t j = 0; j < i; ++j) {
      const auto length = requests[j].prompt.size();
      if (length < requests[i].prompt.size() && cache::same_prefix(*requests[i].input, *requests[j].input, length))
        suffixes.push_back(requests[i].prompt.size() - length);
    }
    std::sort(suffixes.begin(), suffixes.end()); suffixes.erase(std::unique(suffixes.begin(), suffixes.end()), suffixes.end());
    for (auto tokens : suffixes) note_prompt(tokens == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill, tokens, i);
    for (std::size_t turn = 1; turn < requests[i].turns.size(); ++turn) {
      const auto tokens = cache::add(requests[i].turns[turn].prompt.size(), 1);
      for (auto role : {GraphRole::kMtpPrefill, GraphRole::kMtpCatchupPrefill}) note_prompt(role, tokens, i);
    }
    if (std::ranges::any_of(requests[i].turns, [](const auto &turn) { return turn.steps > 1; })) ++decoding_requests;
  }
  const std::array roles{GraphRole::kMtpRound, GraphRole::kMtpCatchupRound,
      GraphRole::kMtpSingle, GraphRole::kMtpCatchupSingle};
  // Normal rebuild preserves original committed chunks, not original proposals.
  // Every natural accept2 can become one teacher-forced T2 prefill chunk.
  prompt_uses[GraphRole::kMtpSingle].try_emplace(1, 0);
  if (model.capacity >= 2) prompt_uses[GraphRole::kMtpPrefill].try_emplace(2, 0);
  for (std::size_t width = 1; width <= model.cache_plan.active_sequences; ++width) {
    std::map<GraphRole, std::map<std::size_t, std::size_t>> shapes;
    for (const auto &[role, tokens] : prompt_uses) for (const auto &[count, uses] : tokens) {
      const auto found_role = support.find(role);
      const auto eligible = found_role == support.end() || !found_role->second.contains(count) ? 0 : found_role->second.at(count).size();
      if (width == 1 || width <= eligible) shapes[role][count] = cache::add(uses, width == 1 ? restore_bound : 0);
    }
    if (decode_bound && width <= decoding_requests) for (auto role : roles) {
      auto &uses = shapes[role][drafts_token(role) ? 2 : 1]; uses = cache::add(uses, decode_bound);
    }
    // R1 restoration also needs lag0 versions of each original warm chunk.
    if (width == 1) for (const auto &[tokens, uses] : prompt_uses[GraphRole::kMtpCatchupPrefill]) {
      (void)uses; shapes[GraphRole::kMtpPrefill].try_emplace(tokens, restore_bound);
    }
    for (const auto &[role, entries] : shapes) {
      std::vector<std::size_t> chunks; chunks.reserve(entries.size());
      for (const auto &[tokens, uses] : entries) { (void)uses; chunks.push_back(tokens); }
      model.prepare_family(chunks, false, width, role);
      for (const auto &[tokens, uses] : entries) model.plan_replay_uses(tokens, width,
          cache::add(uses, width == 1 ? restore_bound : 0), role);
    }
  }
  const std::array first_prefill{requests.front().prompt.size()};
  model.prepare_execution(first_prefill, 1,
      first_prefill[0] == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill);
  if (decode_bound) for (auto role : roles) if (drafts_token(role)) model.prepare_execution(pair, 1, role);
  std::vector<std::size_t> waiting, active, dispatch, dispatch_turns;
  waiting.reserve(total_turns); active.reserve(model.cache_plan.resident_sessions);
  dispatch.reserve(requested_outputs); dispatch_turns.reserve(requested_outputs);
  for (std::size_t i = 0; i < requests.size(); ++i) waiting.push_back(i);
  std::vector<std::size_t> evictions; evictions.reserve(total_turns);
  std::size_t waiting_cursor = 0, completed = 0, admitted = 0, idle_clock = 0;
  struct BatchObservation { std::size_t tokens; GraphRole role; bool generation; hyle::container::SmallVector<std::size_t, 2> members; };
  std::vector<BatchObservation> batches; batches.reserve(requested_outputs);
  std::size_t actual_prefill_calls = 0, actual_decode_calls = 0;
  double actual_step_sum = 0;
  std::size_t peak_resident = 0, peak_committed = 0;
  const auto resident_count = [&] {
    return static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [](const auto &request) { return bool(request.session); }));
  };
  const auto ready = Clock::now();
  std::cerr << "mtp_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n'
            << "measure_memory name=ready_before_first\n";
  const auto start = Clock::now(); auto endpoint = start;
  double prefill_seconds = 0, decode_seconds = 0;
  while (completed != requests.size()) {
    while (active.size() < model.cache_plan.resident_sessions && waiting_cursor < waiting.size()) {
      const auto index = waiting[waiting_cursor++]; auto &request = requests[index];
      request.admission_start = Clock::now();
      if (!request.session) {
        while (resident_count() >= model.cache_plan.resident_sessions) {
          std::size_t victim = requests.size();
          for (std::size_t i = 0; i < requests.size(); ++i)
            if (requests[i].session && std::find(active.begin(), active.end(), i) == active.end() &&
                (victim == requests.size() || requests[i].last_idle < requests[victim].last_idle)) victim = i;
          if (victim == requests.size()) throw std::runtime_error("all MTP heads are generating; no idle owner may be evicted");
          retire_mtp_request(requests[victim]); ++requests[victim].idle_evictions; evictions.push_back(victim);
        }
        if (request.ledger->committed_count) {
          const auto restore_trace = request.trace.empty() ? request.trace : request.trace / "restores" / ("restore" + std::to_string(request.restore_count));
          request.session = rebuild_mtp_session(model, *request.ledger, restore_trace, trace_states, &request.restored, prefix.get(), requests);
          ++request.restore_count;
        }
      }
      if (request.turn_index) {
        if (!request.session) throw std::runtime_error("later MTP turn requires its rebuilt complete head");
        (*request.warm_inputs[request.turn_index])[0] = request.generated.back();
      }
      request.prompt = *request.warm_inputs[request.turn_index]; request.prefilled = false; request.done = false;
      request.turn_output_begin = request.generated.size(); request.turn_observations[request.turn_index].output_begin = request.generated.size();
      active.push_back(index); ++admitted;
    }
    if (active.empty()) throw std::runtime_error("MTP admission has no runnable resident request");
    // Resolve every lookup/admission in this wave before any checkpoint is
    // published. A same-wave prompt cannot observe another member's new head.
    hyle::container::SmallVector<PreparedMtpCall, 2> calls;
    if (!calls.try_reserve(active.size())) throw std::bad_alloc{};
    for (auto index : active) {
      auto &request = requests[index];
      std::cerr << "mtp_session id=" << request.id << " turn=" << request.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      calls.push_back(prepare_mtp_call(model, request, cache::add(request.turn_output_begin, request.turns[request.turn_index].steps), prefix.get(), requests));
    }
    hyle::container::SmallVector<char, 4> executed;
    if (!executed.try_resize(calls.size(), 0)) throw std::bad_alloc{};
    for (std::size_t first = 0; first < calls.size(); ++first) {
      if (executed[first]) continue;
      hyle::container::SmallVector<std::size_t, 4> selected;
      if (!selected.try_reserve(model.cache_plan.active_sequences)) throw std::bad_alloc{};
      for (std::size_t j = first; j < calls.size() && selected.size() < model.cache_plan.active_sequences; ++j)
        if (!executed[j] && calls[j].full_hit == calls[first].full_hit &&
            calls[j].generation == calls[first].generation && calls[j].role == calls[first].role &&
            calls[j].ids().size() == calls[first].ids().size() && (!calls[first].full_hit || selected.empty())) selected.push_back(j);
      hyle::container::SmallVector<PreparedMtpCall, 2> bucket;
      if (!bucket.try_reserve(selected.size())) throw std::bad_alloc{};
      for (auto j : selected) bucket.push_back(calls[j]);
      while (!bucket[0].full_hit && !make_mtp_batch_room(model, prefix.get(), std::span(bucket.data(), bucket.size()), requests)) {
        if (bucket.size() == 1) throw std::runtime_error("MTP committed budget cannot advance its selected request");
        bucket.pop_back(); selected.pop_back();
      }
      std::cerr << "mtp_batch requests=" << bucket.size() << " T=" << (drafts_token(bucket[0].role) ? 2 : bucket[0].ids().size())
                << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit << '\n';
      execute_mtp_calls(model, std::span(bucket.data(), bucket.size()), trace_states, prefix.get(), requests);
      auto finished = endpoint;
      hyle::container::SmallVector<std::size_t, 2> members;
      if (!members.try_reserve(bucket.size())) throw std::bad_alloc{};
      for (std::size_t j = 0; j < bucket.size(); ++j) {
        executed[selected[j]] = true; const auto index = active[selected[j]];
        dispatch.push_back(index); dispatch_turns.push_back(requests[index].turn_index); members.push_back(index);
        finished = (std::max)(finished, requests[index].endpoint);
      }
      const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
      (bucket[0].generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
      if (!bucket[0].full_hit) {
        (bucket[0].generation ? actual_decode_calls : actual_prefill_calls)++;
        actual_step_sum += bucket[0].step_seconds;
        batches.push_back({drafts_token(bucket[0].role) ? 2 : bucket[0].ids().size(), bucket[0].role, bucket[0].generation, std::move(members)});
      }
      peak_resident = (std::max)(peak_resident, resident_count());
      peak_committed = (std::max)(peak_committed, std::size_t(model.snapshot_versions.live));
      if (model.snapshot_versions.live > model.cache_plan.resident_sessions) throw std::runtime_error("MTP committed version count exceeds N after publication");
    }
    for (auto &call : calls) publish_mtp_prompt(model, call, prefix.get());
    for (auto at = active.begin(); at != active.end();) {
      const auto index = *at; auto &request = requests[index];
      if (!request.done) { ++at; continue; }
      at = active.erase(at);
      if (request.turn_index + 1 == request.turns.size()) { retire_mtp_request(request); ++completed; }
      else { request.last_idle = ++idle_clock; ++request.turn_index; waiting.push_back(index); }
    }
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  std::size_t logical_prompt = 0, user_prompt = 0, prefill_tokens = 0, prefill_draft = 0, prefill_calls = 0;
  std::size_t generated = 0, decode_calls = 0, verified = 0, drafted = 0, committed = 0;
  std::size_t draft_rounds = 0, agreements = 0, accepts = 0, hits = 0, misses = 0, full_hits = 0, reused = 0;
  std::size_t restores = 0, recomputed_changes = 0; MtpRestoreStats restored; double step_sum_seconds = 0;
  for (const auto &request : requests) {
    for (const auto &turn : request.turns) user_prompt = cache::add(user_prompt, turn.prompt.size());
    prefill_tokens = cache::add(prefill_tokens, request.prefill_graph_tokens);
    prefill_draft = cache::add(prefill_draft, request.prefill_draft_rows);
    prefill_calls = cache::add(prefill_calls, request.prefill_graph_calls);
    generated = cache::add(generated, request.generated.size()); decode_calls = cache::add(decode_calls, request.rounds.size());
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
  const auto graph_calls = cache::add(cache::add(actual_prefill_calls, actual_decode_calls), restored.graph_calls);
  std::cout << std::setprecision(17) << "{\"source_id\":\"" << source_id << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << score_policy
      << "\",\"mtp\":true,\"text_tensors\":320,\"mtp_tensors\":15"
      << ",\"session_count\":" << requests.size() << ",\"turn_count\":" << total_turns << ",\"active_sequences\":" << model.cache_plan.active_sequences << ",\"max_sessions\":" << model.cache_plan.resident_sessions
      << model.cache_report() << ",\"prefix_cache\":" << (prefix_enabled ? "true" : "false")
      << ",\"admitted_requests\":" << requests.size() << ",\"admitted_turns\":" << admitted << ",\"retired_requests\":" << completed
      << ",\"peak_resident_requests\":" << peak_resident << ",\"peak_committed_versions\":" << peak_committed
      << ",\"live_committed_versions\":" << model.snapshot_versions.live << ",\"idle_evictions\":" << evictions.size()
      << ",\"prefix_hits\":" << hits << ",\"prefix_misses\":" << misses << ",\"prefix_full_hits\":" << full_hits << ",\"reused_tokens\":" << reused
      << ",\"prefix_entries\":" << (prefix ? prefix->size() : 0) << ",\"prefix_evictions\":" << (prefix ? prefix->evictions : 0)
      << ",\"requested_output_tokens\":" << requested_outputs << ",\"generated_tokens\":" << generated << ",\"generated_decode_tokens\":" << generated - total_turns
      << ",\"actual_graph_calls\":" << graph_calls << ",\"prefill_graph_calls\":" << actual_prefill_calls << ",\"decode_graph_calls\":" << actual_decode_calls
      << ",\"graph_participations\":" << prefill_calls + decode_calls
      << ",\"logical_dispatches\":" << dispatch.size() << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"logical_user_prompt_tokens\":" << user_prompt
      << ",\"pending_flushed_tokens\":" << total_turns - requests.size() << ",\"prefill_target_rows\":" << prefill_tokens << ",\"prefill_draft_rows\":" << prefill_draft
      << ",\"verified_target_rows\":" << verified << ",\"draft_rows\":" << drafted
      << ",\"actual_target_rows\":" << cache::add(cache::add(prefill_tokens, verified), restored.target_rows)
      << ",\"actual_draft_rows\":" << cache::add(cache::add(prefill_draft, drafted), restored.draft_rows)
      << ",\"committed_target_rows\":" << committed << ",\"committed_target_decode_rows\":" << committed - logical_prompt
      << ",\"draft_rounds\":" << draft_rounds << ",\"agreements\":" << agreements << ",\"accepted_drafts\":" << accepts
      << ",\"restore_count\":" << restores << ",\"restored_graph_calls\":" << restored.graph_calls << ",\"restored_target_rows\":" << restored.target_rows
      << ",\"restored_draft_rows\":" << restored.draft_rows << ",\"restore_step_seconds\":" << restored.step_seconds << ",\"restore_seconds\":" << restored.seconds
      << ",\"restore_contract\":\"recompute_committed_inputs\",\"normalized_lag_rows\":" << restored.normalized_lag_rows
      << ",\"recomputed_decision_changes\":" << recomputed_changes
      << ",\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << session_seconds
      << ",\"step_sum_seconds\":" << actual_step_sum + restored.step_seconds
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
    for (const auto &turn : request.turns) quota = cache::add(quota, turn.steps);
    std::cout << (i ? "," : "") << "{\"session_id\":\"" << request.id << "\",\"result\":";
    print_mtp_request(model, request, quota, startup, ready);
    std::cout << ",\"restore_count\":" << request.restore_count << ",\"idle_evictions\":" << request.idle_evictions << ",\"ledger\":[";
    std::size_t round = 0;
    for (std::size_t j = 0; j < request.ledger->committed_count; ++j) {
      const auto &record = request.ledger->records[j]; if (record.generation) ++round;
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto token : record.ids()) hash = (hash ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
      const auto prefix_name = record.turn ? "round" + std::to_string(record.turn + 1) + "_" : "";
      const auto name = prefix_name + (record.generation ? "mtp" + std::to_string(round) : "prefill");
      std::cout << (j ? "," : "") << "{\"index\":" << j << ",\"turn\":" << record.turn + 1 << ",\"role\":" << static_cast<int>(record.role)
          << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
          << ",\"generation\":" << (record.generation ? "true" : "false") << ",\"input_rows\":" << record.ids().size()
          << ",\"input_hash\":" << hash << ",\"first_input\":" << record.ids().front() << ",\"quota\":" << record.quota
          << ",\"before_length\":" << record.before.length << ",\"before_frontier\":" << record.before.frontier << ",\"before_lag\":" << record.before.lag
          << ",\"before_next_id\":" << record.before.next_id << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first << ",\"target_second\":" << record.target_second
          << ",\"accepted\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows << ",\"after_length\":" << record.after.length
          << ",\"after_frontier\":" << record.after.frontier << ",\"after_lag\":" << record.after.lag << ",\"next_id\":" << record.after.next_id
          << ",\"committed_input_ids\":[";
      if (record.generation) for (std::size_t k = 0; k < record.accepted; ++k)
        std::cout << (k ? "," : "") << record.accepted_inputs[k];
      std::cout << "],\"imported\":" << (record.imported ? "true" : "false")
          << ",\"source_geometry\":\"" << (record.imported ? "unknown" : "recorded")
          << "\",\"trace_name\":\"" << (record.imported ? "" : name) << "\"}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n"; return 0;
}
// Natural single-session turns use the ordinary Text Graph and the same
// evaluate/publish path as the existing unscheduled consumer. The final emitted
// token is pending (not in KV); append it once before the next user segment.
int run_plain_turns(Model &model, std::span<const benchmark::MtpTurn> turns,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed) {
  struct Turn {
    std::vector<std::int32_t> input;
    std::size_t before = 0, after = 0, begin = 0, end = 0;
    Clock::time_point start{}, prefill{}, endpoint{};
  };
  std::vector<Turn> observations(turns.size());
  std::vector<std::size_t> calls;
  std::size_t quota = 0;
  for (std::size_t i = 0; i < turns.size(); ++i) {
    auto &input = observations[i].input;
    input.resize(cache::add(turns[i].prompt.size(), i ? 1 : 0));
    std::copy(turns[i].prompt.begin(), turns[i].prompt.end(), input.begin() + (i ? 1 : 0));
    calls.push_back(input.size());
    calls.insert(calls.end(), turns[i].steps - 1, 1);
    quota = cache::add(quota, turns[i].steps);
  }
  model.prepare_family(calls); model.plan_execution(calls);
  auto session = model.create_session(calls); model.prepare_execution(calls);
  std::vector<std::int32_t> generated; generated.reserve(quota);
  const auto ready = Clock::now();
  std::cerr << "shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count()
            << "\nmeasure_memory name=ready_before_first\n";
  std::size_t prefill_rows = 0, decode_rows = 0;
  double prefill_seconds = 0, decode_seconds = 0;
  Clock::time_point endpoint{};
  for (std::size_t i = 0; i < turns.size(); ++i) {
    auto &turn = observations[i]; const auto name = "round" + std::to_string(i + 1);
    turn.before = session->length(); turn.begin = generated.size();
    if (i) turn.input.front() = generated.back();
    std::cerr << "memory_phase=" << name << "_prefill_begin\n";
    // Classify the complete between-turn interval as the next prefill, as in
    // the existing MTP episode accounting. No host turn-transition cost vanishes.
    turn.start = i ? endpoint : Clock::now();
    auto pending = model.evaluate(*session, turn.input, name + "_prefill", trace, trace_states, nullptr, &turn.prefill);
    endpoint = turn.prefill; generated.push_back(pending);
    std::cerr << "memory_phase=" << name << "_prefill_end\nmemory_phase=" << name << "_decode_begin\n";
    for (std::size_t n = 1; n < turns[i].steps && !is_end_of_generation(pending); ++n) {
      const auto input = pending;
      pending = model.evaluate(*session, std::span(&input, 1), name + "_decode" + std::to_string(n),
          trace, trace_states, nullptr, &endpoint);
      generated.push_back(pending); ++decode_rows;
    }
    turn.endpoint = endpoint; turn.end = generated.size(); turn.after = session->length();
    if (turn.after != cache::add(cache::add(turn.before, turn.input.size()), turn.end - turn.begin - 1))
      throw std::runtime_error("natural turn committed context differs from actual input/output rows");
    prefill_rows = cache::add(prefill_rows, turn.input.size());
    prefill_seconds += std::chrono::duration<double>(turn.prefill - turn.start).count();
    decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill).count();
    std::cerr << "memory_phase=" << name << "_decode_end\n";
  }
  const auto start = observations.front().start;
  const auto total = std::chrono::duration<double>(endpoint - start).count();
  std::cout << std::setprecision(17) << "{\"source_id\":\"" << source_id
      << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << score_policy
      << "\",\"mtp\":false,\"text_tensors\":320,\"natural_generation\":true,\"fixed_decode\":false"
      << ",\"session_count\":1,\"turn_count\":" << turns.size() << ",\"shared_model_objects\":1"
      << ",\"prefix_cache\":false,\"restore_count\":0,\"requested_context\":" << model.capacity
      << model.cache_report() << ",\"requested_output_tokens\":" << quota << ",\"generated_tokens\":" << generated.size()
      << ",\"generated_decode_tokens\":" << decode_rows << ",\"visible_tokens\":" << session->length()
      << ",\"processed_tokens\":" << session->length() << ",\"committed_target_rows\":" << session->length()
      << ",\"prefill_target_rows\":" << prefill_rows << ",\"prefill_draft_rows\":0,\"draft_rows\":0"
      << ",\"actual_target_rows\":" << prefill_rows + decode_rows << ",\"actual_draft_rows\":0"
      << ",\"verified_target_rows\":" << decode_rows << ",\"committed_target_decode_rows\":" << decode_rows
      << ",\"target_compute_calls\":" << turns.size() + decode_rows << ",\"draft_compute_calls\":0"
      << ",\"actual_graph_calls\":" << turns.size() + decode_rows << ",\"prefill_graph_calls\":" << turns.size()
      << ",\"decode_graph_calls\":" << decode_rows << ",\"pending_flushed_tokens\":" << turns.size() - 1
      << ",\"emitted_pending\":" << generated.back() << ",\"prefill_seconds\":" << prefill_seconds
      << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << total
      << ",\"between_turn_seconds\":" << (std::max)(0.0, total - prefill_seconds - decode_seconds)
      << ",\"weights_loaded_once\":true,\"session_created_once\":true"
      << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
      << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(observations.front().prefill - startup).count()
      << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
  std::cout << "],\"turns\":[";
  for (std::size_t i = 0; i < turns.size(); ++i) {
    const auto &turn = observations[i];
    std::cout << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << turns[i].steps
        << ",\"new_user_tokens\":" << turns[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
        << ",\"context_before\":" << turn.before << ",\"context_after\":" << turn.after
        << ",\"output_begin\":" << turn.begin << ",\"output_end\":" << turn.end
        << ",\"prefill_seconds\":" << std::chrono::duration<double>(turn.prefill - turn.start).count()
        << ",\"decode_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.prefill).count()
        << ",\"session_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.start).count()
        << ",\"stopped_on_EOS\":" << (is_end_of_generation(generated[turn.end - 1]) ? "true" : "false")
        << ",\"input_ids\":[";
    for (std::size_t k = 0; k < turn.input.size(); ++k) std::cout << (k ? "," : "") << turn.input[k];
    std::cout << "]}";
  }
  std::cout << "]}\n";
  check(session->stop());
  if (!take(session->try_retire())) throw std::runtime_error("finished natural session still has an in-flight episode");
  return 0;
}
} // namespace qwen35

int main(int argc, char **argv) {
  const auto startup = qwen35::Clock::now();
  try {
    std::filesystem::path input, continuation, trace, layer0_probe, decode_ids, schedule_file, session_file;
    std::size_t steps = 4, capacity = 128, max_sessions = 0, active_sequences = 0, cache_bytes = 0;
    bool progress = false, trace_states = true, explicit_steps = false;
    bool mtp_enabled = false;
    bool single_schedule = false, multiple_schedules = false;
    std::string prefix_policy;
    bool explicit_continuation = false, explicit_decode_ids = false, preparation_diagnostics = false;
    auto replay = qwen35::ReplayPolicy::kAuto;
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
      else if (option == "--mtp") {
        if (value != "on" && value != "off") throw std::runtime_error("--mtp requires on or off");
        mtp_enabled = value == "on";
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
    const bool natural_plain = !mtp_enabled && single_schedule &&
        qwen35::benchmark::schedule_generates(schedule_file);
    if (natural_plain && prefix_enabled)
      throw std::runtime_error("natural single-session turns require --prefix-cache off");
    if (natural_plain && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("natural single-session turns do not support layer or preparation diagnostics");
    if (scheduled && (explicit_continuation || explicit_decode_ids || (explicit_steps && !mtp_enabled && !natural_plain)))
      throw std::runtime_error("schedule inputs are exclusive with --continuation, --decode-ids and explicit --steps");
    if ((single_schedule && schedule_file.empty()) || (multiple_schedules && session_file.empty()))
      throw std::runtime_error("schedule requires a nonempty manifest path");
    if (preparation_diagnostics && !scheduled)
      throw std::runtime_error("--preparation-diagnostics requires a schedule");
    std::vector<qwen35::benchmark::SessionSchedule> schedules;
    std::vector<qwen35::benchmark::MtpSessionSchedule> mtp_schedules;
    std::vector<qwen35::benchmark::MtpTurn> natural_turns;
    if (mtp_enabled && single_schedule)
      mtp_schedules.push_back({"default", qwen35::benchmark::read_mtp_schedule(schedule_file,
          qwen35::vocabulary, capacity, steps, explicit_steps)});
    else if (mtp_enabled && multiple_schedules)
      mtp_schedules = qwen35::benchmark::read_mtp_session_schedules(session_file, qwen35::vocabulary,
          capacity, std::vector<qwen35::benchmark::MtpSessionSchedule>{}.max_size(), steps, explicit_steps);
    else if (natural_plain)
      natural_turns = qwen35::benchmark::read_mtp_schedule(schedule_file,
          qwen35::vocabulary, capacity, steps, explicit_steps);
    else if (single_schedule)
      schedules.push_back({"default", qwen35::benchmark::read_schedule(schedule_file, qwen35::vocabulary, capacity, true)});
    else if (multiple_schedules)
      schedules = qwen35::benchmark::read_session_schedules(session_file, qwen35::vocabulary, capacity,
          std::vector<qwen35::benchmark::SessionSchedule>{}.max_size(), true);
    const auto input_sessions = natural_plain ? std::size_t{1} : mtp_enabled && scheduled ? mtp_schedules.size() : scheduled ? schedules.size() : std::size_t{1};
    const auto recurrent_payload = qwen35::recurrent_bytes(qwen35::state_schema_for(capacity));
    if (max_sessions == 0) max_sessions = input_sessions;
    const auto cached_slots = (std::min)(max_sessions, input_sessions);
    if (active_sequences > max_sessions)
      throw std::runtime_error("--active-sequences cannot exceed --max-sessions");
    const auto forced = decode_ids.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(decode_ids);
    if (!decode_ids.empty()) {
      if (forced.empty()) throw std::runtime_error("--decode-ids requires a nonempty token tape");
      const auto forced_steps = qwen35::cache::add(forced.size(), 1);
      if (explicit_steps && steps != forced_steps)
        throw std::runtime_error("--steps must equal decode input count plus one");
      steps = forced_steps;
    }
    if (input.empty())
      throw std::runtime_error("usage: hyle_qwen35_text --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
    const auto prompt = scheduled ? std::vector<std::int32_t>{} : qwen35::read_ids(input / "prompt.ids");
    const auto extra = continuation.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(continuation);
    if (!scheduled) {
      if (prompt.empty()) throw std::runtime_error("generation requires a nonempty prompt before loading weights");
      if (prompt.size() > capacity || extra.size() > capacity - prompt.size() ||
          steps - 1 > capacity - prompt.size() - extra.size())
        throw std::runtime_error("prompt, continuation and requested generation exceed --capacity");
    }
    std::vector<std::size_t> planned_chunks;
    std::size_t mtp_schema_rows = 0;
    bool mtp_has_decode = !scheduled && steps > 1;
    if (mtp_enabled && scheduled) {
      for (const auto &request : mtp_schedules) for (std::size_t i = 0; i < request.turns.size(); ++i) {
        const auto tokens = qwen35::cache::add(request.turns[i].prompt.size(), i ? 1 : 0);
        planned_chunks.push_back(tokens);
        mtp_schema_rows = (std::max)(mtp_schema_rows, qwen35::cache::add(tokens, i ? 1 : 0));
        mtp_has_decode = mtp_has_decode || request.turns[i].steps > 1;
      }
    } else if (natural_plain) {
      for (std::size_t i = 0; i < natural_turns.size(); ++i) {
        planned_chunks.push_back(qwen35::cache::add(natural_turns[i].prompt.size(), i ? 1 : 0));
        if (natural_turns[i].steps > 1) planned_chunks.push_back(1);
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
      if (mtp_enabled) for (const auto &request : mtp_schedules) {
        const std::array phases{qwen35::benchmark::Phase{qwen35::benchmark::PhaseKind::kPrompt, request.turns.front().prompt, 1}};
        token_plans.push_back(qwen35::cache::token_plan(phases, capacity));
      } else for (const auto &request : schedules) token_plans.push_back(qwen35::cache::token_plan(request.phases, capacity));
    }
    const auto max_query_rows = (std::max)(*std::max_element(planned_chunks.begin(), planned_chunks.end()), mtp_schema_rows);
    const auto requested_active = (std::min)(active_sequences, input_sessions);
    const auto plan_for = [&](std::size_t residents) {
      return qwen35::cache::plan(capacity, residents,
          mtp_enabled && mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? residents : requested_active, mtp_enabled ? 2 : 1, mtp_enabled);
    };
    const auto minimum_residents = active_sequences == 0 ? std::size_t{1} : requested_active;
    auto selected_cache = plan_for(cached_slots);
    // Caller-written pools have an exact cold formula. Native generated
    // results instead enter the same byte scope after real completion, before
    // any append is taken or head published. Do not estimate native N from the
    // CUDA pool formula or pretend auto bounds opaque execution allocations.
    if (cache_bytes == 0) cache_bytes = qwen35::Backend::native_generated_storage
        ? SIZE_MAX : selected_cache.backing_bytes;
    else if (!qwen35::Backend::native_generated_storage && selected_cache.backing_bytes > cache_bytes) {
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
    const auto model_start = qwen35::Clock::now();
    qwen35::Model model(input / "weights.tsv", capacity, selected_cache, layer0_probe, progress, replay, cache_bytes);
    const auto constructed = qwen35::Clock::now();
    if (!trace.empty()) model.trace_routes = trace / "routes";
    if (mtp_enabled) {
      if (scheduled) return qwen35::run_mtp_sessions(model, mtp_schedules, trace, trace_states,
          startup, constructed, prefix_enabled, token_plans);
      return qwen35::run_mtp(model, prompt, steps, trace, trace_states, startup, constructed);
    }
    if (natural_plain) return qwen35::run_plain_turns(model, natural_turns, trace, trace_states, startup, constructed);

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
    qwen35::benchmark::SessionCache resident_cache({cache_slots, (std::numeric_limits<std::size_t>::max)()});
    // Destroy runs (and their real slots) before this directory and Model.
    auto prefix_cache = prefix_enabled ? std::make_unique<qwen35::cache::PrefixCache>(cache_slots) : nullptr;
    struct SessionRun {
      struct RecomputedDecision { std::size_t chunk; std::int32_t previous, actual; };
      std::shared_ptr<qwen35::Session> session;
      qwen35::benchmark::SessionCache::Pin pin;
      std::vector<std::size_t> chunks;
      std::shared_ptr<qwen35::cache::ReplayHistory> history;
      std::vector<PhaseSummary> summaries;
      std::vector<Call> calls;
      std::vector<std::int32_t> generated, consumed;
      std::vector<qwen35::StepObservation> observations;
      std::vector<qwen35::PreparationObservation> preparation;
      std::filesystem::path trace;
      std::string trace_subdirectory;
      qwen35::Clock::time_point start{}, endpoint{};
      std::size_t visible_tokens = 0, cache_hits = 0, cache_misses = 0;
      std::size_t restored_calls = 0, restored_tokens = 0;
      std::vector<RecomputedDecision> recomputed_decisions;
      std::size_t graph_participations = 0, actual_graph_tokens = 0;
      std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
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
      if (!prefix_cache) return;
      while (model.snapshot_versions.live >= cache_slots) {
        if (!prefix_cache->evict_one())
          throw std::runtime_error("prefix/session heads retain the committed version budget before initial admission");
      }
    };
    const auto make_batch_room = [&](std::span<qwen35::Session *const> owners) {
      if (owners.empty() || model.active_step || model.snapshot_versions.live > cache_slots)
        throw std::runtime_error("batch advance requires retired sessions within the committed version budget");
      for (auto *owner : owners) if (!owner || owner->pending || owner->failed || owner->stopping)
        throw std::runtime_error("batch advance contains a stopped, failed or pending session");
      // With prefix reuse disabled, each resident Session has its own sole
      // committed head. Advancing it replaces one version; no directory scan.
      if (!prefix_cache) return;
      const auto fits = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        if (live <= cache_slots && owners.size() <= cache_slots - live) return true;
        hyle::container::SmallVector<const qwen35::cache::Snapshot *, 4> bases;
        if (!bases.try_reserve(owners.size())) throw std::bad_alloc{};
        for (const auto *owner : owners) bases.push_back(owner->head.get());
        std::sort(bases.begin(), bases.end(), std::less<const qwen35::cache::Snapshot *>{});
        auto unique = std::unique(bases.begin(), bases.end());
        const auto count = static_cast<std::size_t>(unique - bases.begin());
        std::size_t retiring = 0;
        for (std::size_t b = 0; b < count; ++b) {
          const auto *base = bases[b];
          bool retained = prefix_cache->retains(base);
          for (const auto &other : runs) {
            if (retained) break;
            if (other.session && other.session->head.get() == base &&
                std::find(owners.begin(), owners.end(), other.session.get()) == owners.end()) retained = true;
          }
          if (!retained) ++retiring;
        }
        if (retiring > live) throw std::runtime_error("batch base/version inventory is inconsistent");
        return live - retiring <= cache_slots && owners.size() <= cache_slots - (live - retiring);
      };
      // Dropping a directory alias need not free a version. Re-read both live
      // inventory and external retainers after EVERY eviction, including full hits.
      while (!fits()) if (!prefix_cache->evict_one())
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
        if (old.pin) return false;
        if (old.session) {
          old.session->stop().IgnoreError();
          const auto retired = old.session->try_retire();
          if (!retired || !*retired) return false;
        }
        old.session.reset(); // Actual state leases retire before quota returns.
        evicted_sessions.push_back(victim);
        return true;
      });
      if (!admission.accepted())
        throw std::runtime_error(admission.status == qwen35::benchmark::CacheAdmissionStatus::kRetirementUnproven
            ? "session cache cannot prove retirement; the original owner remains charged"
            : "session cache admission exhausted; reduce --active-sequences or increase --max-sessions/--cache-bytes");
      run.pin = resident_cache.pin(s);
      if (!run.pin) throw std::runtime_error("session cache could not pin the admitted owner");
      if (run.session) ++run.cache_hits;
      else {
        ++run.cache_misses;
        if (!defer_restore) {
          make_initial_room();
          run.session = model.create_session(run.chunks);
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
            qwen35::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_call" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = model.evaluate(*run.session, ids, name,
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
        }
      }
      (void)model.cache_usage();
      run.cache_prepare_seconds += std::chrono::duration<double>(qwen35::Clock::now() - begin).count();
    };
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
    std::vector<std::int32_t> generated, consumed;
    if (!scheduled) { generated.reserve(steps); consumed.reserve(steps - 1); }
    const auto ready = qwen35::Clock::now();
    std::cerr << "initialized model_seconds=" << std::chrono::duration<double>(constructed - model_start).count()
              << " shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
              << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
    std::cerr << "cache_limits context=" << capacity << " max_sessions=" << max_sessions
              << " active_sequences=" << active_slots << " effective_cache_sessions=" << cache_slots
              << " recurrent_payload_bytes_per_snapshot=" << recurrent_payload
              << " backend=" << qwen35::Backend::name()
              << " shared_cache_backing_bytes=" << model.known_cache_backing_bytes
              << " native_execution_private_bytes_unknown=" << qwen35::Backend::native_generated_storage
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
            if (hit.snapshot->next_id >= qwen35::vocabulary)
              throw std::runtime_error("prefix next_id is outside the model vocabulary");
            reused = hit.snapshot->length - before;
            const auto suffix = target - hit.snapshot->length;
            std::vector<std::size_t> remaining;
            remaining.reserve(run.chunks.size() - index);
            if (suffix) remaining.push_back(suffix);
            remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index + 1), run.chunks.end());
            // All fallible host/IO preparation precedes withdrawal of the old
            // Session. These are real original Model slots, not shadow state.
            auto history = qwen35::cache::copy_history(hit.replay, capacity);
            auto adopted = model.create_session(remaining, hit.snapshot);
            if (run.session) {
              run.session->stop().IgnoreError();
              const auto retired = run.session->try_retire();
              if (!retired || !*retired)
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
          run.session = model.create_session(remaining);
          note_committed_versions();
          // Recompute actual evaluated chunks, including imported prefixes.
          // Keep any old shared history immutable: new checkpoint metadata
          // must describe the REAL rebuilt head and its recomputed decisions.
          auto rebuilt_history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
          for (std::size_t i = 0; i < run.history->chunks.size(); ++i) {
            const auto &part = run.history->chunks[i];
            const auto restored_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
            qwen35::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_chunk" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = model.evaluate(*run.session, restored_ids, name,
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
        for (const auto &bucket : qwen35::benchmark::bucket_schedule_wave(wave, actual_rows)) {
          requests.clear(); outputs.clear(); owners.clear(); members.clear(); next_ids.resize(bucket.calls.size());
          for (const auto j : bucket.calls) {
            const auto &call = wave.calls[j]; auto &run = runs[call.session]; auto &work = prepared[j];
            requests.push_back({run.session, work.ids});
            owners.push_back(run.session.get()); members.push_back(call.session);
            outputs.push_back({run.session, run.observations[call.call].name, &run.trace, trace_states,
                &work.complete_seconds, &run.endpoint, &run.observations[call.call],
                preparation_diagnostics ? &run.preparation[call.call] : nullptr,
                multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}});
          }
          const auto batch_start = qwen35::Clock::now();
          make_batch_room(owners);
          qwen35::PreparationObservation preparation;
          model.prepare_evaluations(requests, preparation_diagnostics ? &preparation : nullptr);
          if (preparation_diagnostics) for (const auto j : bucket.calls)
            runs[wave.calls[j].session].preparation[wave.calls[j].call] = preparation;
          model.submit_evaluations();
          model.finish_evaluations(outputs, next_ids);
          const auto batch_endpoint = runs[members.front()].endpoint;
          record_batch(bucket.kind, bucket.query_rows, members, false,
              std::chrono::duration<double>(batch_endpoint - batch_start).count());
          for (std::size_t at = 0; at < bucket.calls.size(); ++at) {
            const auto j = bucket.calls[at]; const auto &call = wave.calls[j];
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
        }
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
                output << "{\"prefix_hit\":true,\"graph_calls\":0,\"tensor_payload_available\":false,\"visible_tokens\":"
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
          (void)prefix_cache->publish(token_plans[call.session], run.session->head, history);
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
        for (const auto &observation : run.observations) qwen35::print_step_observation(observation);
        if (preparation_diagnostics) for (std::size_t i = 0; i < run.observations.size(); ++i)
          if (run.observations[i].graph_executed)
            qwen35::print_preparation_observation(run.observations[i], run.preparation[i]);
      }
      // Define after the execution observations. Call once in the single
      // schedule's top-level object, or once in the multi-schedule top-level
      // object; do not call for each nested Session. Existing output sets
      // setprecision(17). Session IDs were validated as ASCII [A-Za-z0-9_-].
      const auto print_execution = [&] {
        std::cout << ",\"actual_graph_calls\":" << executed_graph_calls
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
        std::cout << "{\"source_id\":\"" << qwen35::source_id << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << qwen35::score_policy
                  << "\",\"text_tensors\":320,\"visible_tokens\":"
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
                  << ",\"graph_participations\":" << run.graph_participations << ",\"graph_input_tokens\":" << run.actual_graph_tokens
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
          graph_participations += run.graph_participations;
          prefix_hits += run.prefix_hits; prefix_misses += run.prefix_misses;
          prefix_full_hits += run.prefix_full_hits; reused_tokens += run.reused_tokens;
        }
        std::cout << "{\"source_id\":\"" << qwen35::source_id << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << qwen35::score_policy
                  << "\",\"text_tensors\":320,\"session_count\":" << runs.size()
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
                  << ",\"prefix_evictions\":" << (prefix_cache ? prefix_cache->evictions : 0)
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
    auto next = model.evaluate(*session, prompt, "prefill", trace, trace_states,
        &prefill_step_seconds, &prefill_endpoint);
    auto endpoint = prefill_endpoint;
    std::cerr << "first_token_elapsed_seconds=" << std::chrono::duration<double>(prefill_endpoint - startup).count() << '\n';
    std::cerr << "memory_phase=generation_prefill_end\nmemory_phase=generation_decode_begin\n";
    if (!extra.empty()) {
      const auto continuation_start = qwen35::Clock::now();
      next = model.evaluate(*session, extra, "continuation", trace, trace_states,
          &continuation_step_seconds, &endpoint);
      continuation_seconds = std::chrono::duration<double>(endpoint - continuation_start).count();
    }
    generated.push_back(next);
    for (std::size_t step = 1; step < steps && (!forced.empty() || !qwen35::is_end_of_generation(next)); ++step) {
      const auto previous = forced.empty() ? next : forced[step - 1];
      double step_seconds = 0;
      next = model.evaluate(*session, std::span(&previous, 1), "decode" + std::to_string(step), trace, trace_states,
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
              << "{\"source_id\":\"" << qwen35::source_id << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << qwen35::score_policy
              << "\",\"mtp\":false,\"text_tensors\":320" << model.cache_report()
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
              << ",\"draft_rows\":0,\"draft_rounds\":0,\"agreements\":0,\"accepted_drafts\":0"
              << ",\"decode_seconds_includes_continuation\":" << (extra.empty() ? "false" : "true")
              << ",\"diagnostic\":" << (!trace.empty() || !layer0_probe.empty() ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen3.5 text consumer: " << error.what() << '\n'; return 1;
  }
}
