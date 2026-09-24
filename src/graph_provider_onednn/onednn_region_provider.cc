#include "hyle/graph/providers/onednn.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <utility>

#include "brgemm_internal.h"
#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/region_provider.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/math.h"
#include "normalization_internal.h"
#include "provider_internal.h"

namespace hyle::graph::onednn {
namespace {

using compute::ConstTensorView;
using compute::DataType;
using compute::GatedActivation;
using compute::GatedActivationKind;
using compute::GatedInputKind;
using compute::MatMul;
using compute::Operation;
using compute::ParallelExecutorRef;
using compute::Provider;
using compute::Queue;
using compute::TensorDesc;
using compute::TensorView;
using compute::onednn::internal::BrgemmPlan;
using compute::onednn::internal::BrgemmShape;
using compute::onednn::internal::BrgemmTaskRange;

inline constexpr std::uint64_t kGatedMlpRoute = 1;
inline constexpr std::uint64_t kNormalizationRoute = 2;
inline constexpr std::uint64_t kRoutePolicyNamespace = 0x48594c454f4e4544ULL;
inline constexpr compute::RoutePolicyId kGatedMlpPolicy{kRoutePolicyNamespace,
                                                        1, 1};
inline constexpr compute::RoutePolicyId kNormalizationPolicy{
    kRoutePolicyNamespace, 2, 1};
inline constexpr std::uint32_t kGatedMlpPreference = 150;
inline constexpr std::uint32_t kNormalizationPreference = 150;
namespace norm = compute::onednn::internal;
inline constexpr std::size_t kWorkspaceAlignment =
    compute::onednn::internal::kBrgemmWorkspaceAlignment;

constexpr compute::ProviderRuntimeProperties kRuntimeProperties{
    .mode = compute::ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = compute::ProviderWorkspaceOwner::kCaller,
    .replay_class = compute::ProviderReplayClass::kPrepared,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

struct RegionContext final {
  ParallelExecutorRef executor{};
};

struct ValueUse final {
  std::uint32_t consumers;
  bool published;
  std::uint32_t consumer;
};

struct UseTable final {
  mem::AllocatorRef allocator{};
  ValueUse *values = nullptr;
  std::size_t count = 0;

  UseTable() noexcept = default;
  UseTable(const UseTable &) = delete;
  UseTable &operator=(const UseTable &) = delete;
  ~UseTable() {
    if (values != nullptr)
      allocator.deallocate_n(values, count);
  }
};

struct MlpPattern final {
  std::array<NodeId, 4> nodes{};
  ValueId activation{};
  ValueId gate_weight{};
  ValueId up_weight{};
  ValueId down_weight{};
  ValueId gate{};
  ValueId up{};
  ValueId gated{};
  ValueId output{};
  std::size_t rows = 0;
  std::size_t inner = 0;
  std::size_t hidden = 0;
};

struct WorkspaceLayout final {
  std::size_t gate_offset = 0;
  std::size_t up_offset = 0;
  std::size_t scratch_offset = 0;
  std::size_t scratch_size = 0;
  std::size_t total_size = 0;
};

struct RegionPlan final {
  RegionContext *context = nullptr;
  std::array<std::uint32_t, 4> input_indices{};
  BrgemmPlan *gate_up = nullptr;
  BrgemmPlan *down = nullptr;
  WorkspaceLayout workspace{};
  std::size_t rows = 0;
  std::size_t hidden = 0;
};

struct RegionInvocation final {
  mem::HostMapping workspace_mapping{};
  float *gate = nullptr;
  float *up = nullptr;
  std::byte *scratch = nullptr;
};

bool is_dense_f32_matrix(const TensorDesc &description) noexcept {
  return description.valid() && description.data_type() == DataType::kFloat32 &&
         description.is_plain() && description.is_dense() &&
         description.rank() == 2;
}

Status build_use_table(const Graph &graph, mem::AllocatorRef allocator,
                       UseTable *table) noexcept {
  if (table == nullptr || !allocator.valid())
    return Status::InvalidArgument("oneDNN region use-table owner is invalid");
  table->allocator = allocator;
  table->count = graph.value_count();
  if (table->count == 0)
    return Status::Ok();
  table->values = allocator.allocate_zeroed<ValueUse>(table->count);
  if (table->values == nullptr) {
    return Status::ResourceExhausted(
        "oneDNN region value-use table allocation failed");
  }

  for (std::size_t node_index = 0; node_index < graph.node_count();
       ++node_index) {
    auto node = graph.node_id(node_index);
    if (!node)
      return std::move(node).error();
    auto count = graph.node_input_count(*node);
    if (!count)
      return std::move(count).error();
    for (std::size_t input_index = 0; input_index < *count; ++input_index) {
      auto input = graph.node_input(*node, input_index);
      if (!input)
        return std::move(input).error();
      if (input->ordinal() >= table->count)
        return Status::Internal("graph input ordinal exceeds value table");
      ValueUse &use = table->values[input->ordinal()];
      if (use.consumers == (std::numeric_limits<std::uint32_t>::max)()) {
        return Status::OutOfRange("graph value consumer count overflows");
      }
      ++use.consumers;
      use.consumer = node->ordinal();
    }
  }
  for (std::size_t output_index = 0; output_index < graph.output_count();
       ++output_index) {
    auto output = graph.output_id(output_index);
    if (!output)
      return std::move(output).error();
    if (output->ordinal() >= table->count)
      return Status::Internal("graph output ordinal exceeds value table");
    table->values[output->ordinal()].published = true;
  }
  return Status::Ok();
}

Status match_pattern(const Graph &graph, std::size_t down_index,
                     std::span<const ValueUse> uses, MlpPattern *pattern,
                     bool *matched) noexcept {
  if (pattern == nullptr || matched == nullptr)
    return Status::InvalidArgument("oneDNN region pattern output is null");
  *matched = false;
  *pattern = {};
  if (down_index >= graph.node_count())
    return Status::Ok();

  // Walk the dataflow backwards from the down projection. The two independent
  // projections may be inserted in either order, with unrelated nodes between.
  auto down_node = graph.node_id(down_index);
  if (!down_node)
    return std::move(down_node).error();
  auto down_op = graph.operation(*down_node);
  if (!down_op)
    return std::move(down_op).error();
  if (down_op->get_if<MatMul>() == nullptr)
    return Status::Ok();
  auto down_input_value = graph.node_input(*down_node, 0);
  if (!down_input_value)
    return std::move(down_input_value).error();
  if (down_input_value->kind() != ValueKind::kNodeOutput)
    return Status::Ok();
  auto gated_node = graph.producer(*down_input_value);
  if (!gated_node)
    return std::move(gated_node).error();
  auto gated_op = graph.operation(*gated_node);
  if (!gated_op)
    return std::move(gated_op).error();
  const auto *gated_schema = gated_op->get_if<GatedActivation>();
  if (gated_schema == nullptr ||
      gated_schema->inputs != GatedInputKind::kSeparate)
    return Status::Ok();
  auto gate_value = graph.node_input(*gated_node, 0);
  auto up_value = graph.node_input(*gated_node, 1);
  if (!gate_value)
    return std::move(gate_value).error();
  if (!up_value)
    return std::move(up_value).error();
  if (gate_value->kind() != ValueKind::kNodeOutput ||
      up_value->kind() != ValueKind::kNodeOutput || *gate_value == *up_value)
    return Status::Ok();
  auto gate_node = graph.producer(*gate_value);
  auto up_node = graph.producer(*up_value);
  if (!gate_node)
    return std::move(gate_node).error();
  if (!up_node)
    return std::move(up_node).error();
  pattern->nodes = {*gate_node, *up_node, *gated_node, *down_node};

  std::array<const Operation *, 4> operations{};
  for (std::size_t offset = 0; offset < 4; ++offset) {
    auto operation = graph.operation(pattern->nodes[offset]);
    if (!operation)
      return std::move(operation).error();
    operations[offset] = &*operation;
  }
  const MatMul *gate_operation = operations[0]->get_if<MatMul>();
  const MatMul *up_operation = operations[1]->get_if<MatMul>();
  const GatedActivation *gated_operation =
      operations[2]->get_if<GatedActivation>();
  const MatMul *down_operation = operations[3]->get_if<MatMul>();
  if (gate_operation == nullptr || up_operation == nullptr ||
      down_operation == nullptr || gated_operation == nullptr ||
      gate_operation->accumulation_type != DataType::kFloat32 ||
      up_operation->accumulation_type != DataType::kFloat32 ||
      down_operation->accumulation_type != DataType::kFloat32 ||
      gated_operation->activation != GatedActivationKind::kSilu ||
      gated_operation->inputs != GatedInputKind::kSeparate) {
    return Status::Ok();
  }

  constexpr std::array<std::size_t, 4> kExpectedInputs{2, 2, 2, 2};
  for (std::size_t offset = 0; offset < 4; ++offset) {
    auto input_count = graph.node_input_count(pattern->nodes[offset]);
    auto output_count = graph.node_output_count(pattern->nodes[offset]);
    if (!input_count)
      return std::move(input_count).error();
    if (!output_count)
      return std::move(output_count).error();
    if (*input_count != kExpectedInputs[offset] || *output_count != 1)
      return Status::Ok();
  }

  auto activation = graph.node_input(pattern->nodes[0], 0);
  auto gate_weight = graph.node_input(pattern->nodes[0], 1);
  auto up_activation = graph.node_input(pattern->nodes[1], 0);
  auto up_weight = graph.node_input(pattern->nodes[1], 1);
  auto gate = graph.result(pattern->nodes[0]);
  auto up = graph.result(pattern->nodes[1]);
  auto gated_gate = graph.node_input(pattern->nodes[2], 0);
  auto gated_up = graph.node_input(pattern->nodes[2], 1);
  auto gated = graph.result(pattern->nodes[2]);
  auto down_input = graph.node_input(pattern->nodes[3], 0);
  auto down_weight = graph.node_input(pattern->nodes[3], 1);
  auto output = graph.result(pattern->nodes[3]);
  if (!activation)
    return std::move(activation).error();
  if (!gate_weight)
    return std::move(gate_weight).error();
  if (!up_activation)
    return std::move(up_activation).error();
  if (!up_weight)
    return std::move(up_weight).error();
  if (!gate)
    return std::move(gate).error();
  if (!up)
    return std::move(up).error();
  if (!gated_gate)
    return std::move(gated_gate).error();
  if (!gated_up)
    return std::move(gated_up).error();
  if (!gated)
    return std::move(gated).error();
  if (!down_input)
    return std::move(down_input).error();
  if (!down_weight)
    return std::move(down_weight).error();
  if (!output)
    return std::move(output).error();

  if (*activation != *up_activation || *gate != *gated_gate ||
      *up != *gated_up || *gated != *down_input ||
      gate_weight->kind() != ValueKind::kConstant ||
      up_weight->kind() != ValueKind::kConstant ||
      down_weight->kind() != ValueKind::kConstant ||
      *gate_weight == *up_weight || *gate_weight == *down_weight ||
      *up_weight == *down_weight) {
    return Status::Ok();
  }
  if (gate->ordinal() >= uses.size() || up->ordinal() >= uses.size() ||
      gated->ordinal() >= uses.size() || output->ordinal() >= uses.size()) {
    return Status::Internal("graph region value ordinal exceeds use table");
  }
  if (uses[gate->ordinal()].consumers != 1 || uses[gate->ordinal()].published ||
      uses[up->ordinal()].consumers != 1 || uses[up->ordinal()].published ||
      uses[gated->ordinal()].consumers != 1 ||
      uses[gated->ordinal()].published ||
      (uses[output->ordinal()].consumers == 0 &&
       !uses[output->ordinal()].published)) {
    return Status::Ok();
  }

  auto activation_desc = graph.value_description(*activation);
  auto gate_weight_desc = graph.value_description(*gate_weight);
  auto up_weight_desc = graph.value_description(*up_weight);
  auto gate_desc = graph.value_description(*gate);
  auto up_desc = graph.value_description(*up);
  auto gated_desc = graph.value_description(*gated);
  auto down_weight_desc = graph.value_description(*down_weight);
  auto output_desc = graph.value_description(*output);
  if (!activation_desc)
    return std::move(activation_desc).error();
  if (!gate_weight_desc)
    return std::move(gate_weight_desc).error();
  if (!up_weight_desc)
    return std::move(up_weight_desc).error();
  if (!gate_desc)
    return std::move(gate_desc).error();
  if (!up_desc)
    return std::move(up_desc).error();
  if (!gated_desc)
    return std::move(gated_desc).error();
  if (!down_weight_desc)
    return std::move(down_weight_desc).error();
  if (!output_desc)
    return std::move(output_desc).error();

  const std::array<const TensorDesc *, 8> descriptions{
      &*activation_desc, &*gate_weight_desc, &*up_weight_desc,   &*gate_desc,
      &*up_desc,         &*gated_desc,       &*down_weight_desc, &*output_desc};
  if (!std::ranges::all_of(descriptions, [](const TensorDesc *description) {
        return is_dense_f32_matrix(*description);
      })) {
    return Status::Ok();
  }

  const auto activation_shape = activation_desc->dimensions();
  const auto gate_weight_shape = gate_weight_desc->dimensions();
  const auto up_weight_shape = up_weight_desc->dimensions();
  const auto gate_shape = gate_desc->dimensions();
  const auto up_shape = up_desc->dimensions();
  const auto gated_shape = gated_desc->dimensions();
  const auto down_weight_shape = down_weight_desc->dimensions();
  const auto output_shape = output_desc->dimensions();
  if (activation_shape[0] <= 0 || activation_shape[1] <= 0 ||
      gate_weight_shape[1] <= 0 ||
      gate_weight_shape[0] != activation_shape[1] ||
      up_weight_shape[0] != activation_shape[1] ||
      up_weight_shape[1] != gate_weight_shape[1] ||
      gate_shape[0] != activation_shape[0] ||
      gate_shape[1] != gate_weight_shape[1] ||
      !std::ranges::equal(up_shape, gate_shape) ||
      !std::ranges::equal(gated_shape, gate_shape) ||
      down_weight_shape[0] != gate_weight_shape[1] ||
      down_weight_shape[1] != activation_shape[1] ||
      !std::ranges::equal(output_shape, activation_shape)) {
    return Status::Ok();
  }

  pattern->activation = *activation;
  pattern->gate_weight = *gate_weight;
  pattern->up_weight = *up_weight;
  pattern->down_weight = *down_weight;
  pattern->gate = *gate;
  pattern->up = *up;
  pattern->gated = *gated;
  pattern->output = *output;
  pattern->rows = static_cast<std::size_t>(activation_shape[0]);
  pattern->inner = static_cast<std::size_t>(activation_shape[1]);
  pattern->hidden = static_cast<std::size_t>(gate_weight_shape[1]);
  *matched = true;
  return Status::Ok();
}

Status compute_workspace_layout(std::size_t rows, std::size_t hidden,
                                std::size_t scratch_size,
                                WorkspaceLayout *layout) noexcept {
  if (layout == nullptr)
    return Status::InvalidArgument("oneDNN region workspace output is null");
  const auto elements = math::try_multiply(rows, hidden);
  const auto matrix_bytes =
      elements ? math::try_multiply(*elements, sizeof(float)) : std::nullopt;
  if (!matrix_bytes || !mem::try_align_up(*matrix_bytes, kWorkspaceAlignment,
                                          &layout->up_offset)) {
    return Status::OutOfRange("oneDNN region workspace size overflows size_t");
  }
  const auto after_up = math::try_add(layout->up_offset, *matrix_bytes);
  if (!after_up || !mem::try_align_up(*after_up, kWorkspaceAlignment,
                                      &layout->scratch_offset)) {
    return Status::OutOfRange("oneDNN region workspace size overflows size_t");
  }
  const auto checked_total =
      math::try_add(layout->scratch_offset, scratch_size);
  if (!checked_total) {
    return Status::OutOfRange("oneDNN region workspace size overflows size_t");
  }
  layout->total_size = *checked_total;
  layout->gate_offset = 0;
  layout->scratch_size = scratch_size;
  return Status::Ok();
}

struct NormPattern final {
  std::array<NodeId, norm::kNormMaxPostOps + 1> nodes{};
  std::array<ValueId, norm::kNormMaxInputs> inputs{};
  std::array<TensorDesc, norm::kNormMaxInputs> descriptions{};
  std::array<norm::NormPostOp, norm::kNormMaxPostOps> post_ops{};
  Operation operation{};
  ValueId output{};
  TensorDesc output_description{};
  std::size_t node_count = 0;
  std::size_t norm_inputs = 0;

  std::size_t input_count() const noexcept {
    return norm_inputs + node_count - 1;
  }
  norm::NormRequest request() const noexcept {
    return {operation, std::span(descriptions).first(input_count()),
            output_description, norm_inputs,
            std::span(post_ops).first(node_count - 1)};
  }
};

Status match_norm(const Graph &graph, NodeId node,
                  std::span<const ValueUse> uses, ParallelExecutorRef executor,
                  NormPattern &pattern) noexcept {
  pattern = {};
  auto operation = graph.operation(node);
  if (!operation)
    return std::move(operation).error();
  if (operation->kind() != compute::OperationKind::kRmsNorm &&
      operation->kind() != compute::OperationKind::kLayerNorm)
    return Status::Ok();
  auto count = graph.node_input_count(node);
  auto output = graph.result(node);
  if (!count)
    return std::move(count).error();
  if (!output)
    return std::move(output).error();
  auto description = graph.value_description(*output);
  if (!description)
    return std::move(description).error();
  if (*count == 0 || *count > 3 ||
      description->data_type() != DataType::kFloat32)
    return Status::Ok();
  pattern.operation = *operation;
  pattern.norm_inputs = *count;
  pattern.nodes[0] = node;
  pattern.node_count = 1;
  pattern.output = *output;
  pattern.output_description = *description;
  for (std::size_t index = 0; index < *count; ++index) {
    auto input = graph.node_input(node, index);
    if (!input)
      return std::move(input).error();
    auto input_description = graph.value_description(*input);
    if (!input_description)
      return std::move(input_description).error();
    pattern.inputs[index] = *input;
    pattern.descriptions[index] = *input_description;
  }
  Status status = norm::validate_normalization(pattern.request(), executor);
  if (!status.ok()) {
    pattern.node_count = 0;
    return status.IsUnimplemented() ? Status::Ok() : std::move(status);
  }
  while (pattern.node_count < pattern.nodes.size()) {
    const ValueUse &use = uses[pattern.output.ordinal()];
    if (use.consumers != 1 || use.published)
      break;
    auto consumer = graph.node_id(use.consumer);
    if (!consumer)
      return std::move(consumer).error();
    auto next_operation = graph.operation(*consumer);
    if (!next_operation)
      return std::move(next_operation).error();
    const auto *binary = next_operation->get_if<compute::Binary>();
    if (binary == nullptr || (binary->kind != compute::BinaryKind::kAdd &&
                              binary->kind != compute::BinaryKind::kMultiply))
      break;
    auto left = graph.node_input(*consumer, 0);
    auto right = graph.node_input(*consumer, 1);
    auto next_output = graph.result(*consumer);
    if (!left)
      return std::move(left).error();
    if (!right)
      return std::move(right).error();
    if (!next_output)
      return std::move(next_output).error();
    const ValueId rhs = *left == pattern.output ? *right : *left;
    auto rhs_description = graph.value_description(rhs);
    auto next_description = graph.value_description(*next_output);
    if (!rhs_description)
      return std::move(rhs_description).error();
    if (!next_description)
      return std::move(next_description).error();
    if (*next_description != pattern.output_description)
      break;
    NormPattern candidate = pattern;
    const std::size_t input = candidate.input_count();
    candidate.inputs[input] = rhs;
    candidate.descriptions[input] = *rhs_description;
    candidate.post_ops[candidate.node_count - 1] = {binary->kind, input};
    candidate.nodes[candidate.node_count++] = *consumer;
    candidate.output = *next_output;
    status = norm::validate_normalization(candidate.request(), executor);
    if (status.IsUnimplemented())
      break;
    if (!status.ok())
      return status;
    pattern = candidate;
  }
  return Status::Ok();
}

Status propose_norms(const Graph &graph, std::span<const ValueUse> uses,
                     ParallelExecutorRef executor, mem::AllocatorRef allocator,
                     RegionProposalWriter &writer) noexcept {
  for (std::size_t index = 0; index < graph.node_count(); ++index) {
    auto node = graph.node_id(index);
    if (!node)
      return std::move(node).error();
    NormPattern pattern;
    Status status = match_norm(graph, *node, uses, executor, pattern);
    if (!status.ok())
      return status;
    if (pattern.node_count < 2)
      continue;
    const auto nodes = std::span(pattern.nodes).first(pattern.node_count);
    auto contractible = is_region_contractible(graph, nodes, allocator);
    if (!contractible)
      return std::move(contractible).error();
    if (!*contractible)
      continue;
    auto added = writer.try_add(
        {.nodes = nodes,
         .cost = {.workspace = std::nullopt,
                  .persistent_bytes = std::nullopt,
                  .copy_bytes = 0,
                  .repack_bytes = 0,
                  .synchronization_count = 0,
                  .dispatch_count = 1},
         .route = kNormalizationRoute,
         .replay_class = norm::kNormRuntime.replay_class,
         .implementation =
             compute::ProviderImplementationId{kNormalizationRoute},
         .policy = kNormalizationPolicy,
         .preference = kNormalizationPreference});
    if (!added)
      return std::move(added).error();
    if (!*added)
      break;
  }
  return Status::Ok();
}

Status propose_regions(void *context_value, const Graph &graph,
                       mem::AllocatorRef allocator,
                       RegionProposalWriter &writer) noexcept {
  if (context_value == nullptr || !allocator.valid())
    return Status::Internal("oneDNN region context is invalid");
  UseTable table;
  Status status = build_use_table(graph, allocator, &table);
  if (!status.ok())
    return status;
  const std::span<const ValueUse> uses(table.values, table.count);
  for (std::size_t down = 0; down < graph.node_count(); ++down) {
    MlpPattern pattern;
    bool matched = false;
    status = match_pattern(graph, down, uses, &pattern, &matched);
    if (!status.ok())
      return status;
    if (!matched)
      continue;
    auto contractible = is_region_contractible(graph, pattern.nodes, allocator);
    if (!contractible)
      return std::move(contractible).error();
    if (!*contractible)
      continue;
    auto added = writer.try_add(RegionProposal{
        .nodes = pattern.nodes,
        .cost = {.workspace = std::nullopt,
                 .persistent_bytes = std::nullopt,
                 .copy_bytes = 0,
                 .repack_bytes = 0,
                 .synchronization_count = 0,
                 .dispatch_count = 2},
        .route = kGatedMlpRoute,
        .replay_class = kRuntimeProperties.replay_class,
        .implementation = compute::ProviderImplementationId{kGatedMlpRoute},
        .policy = kGatedMlpPolicy,
        .preference = kGatedMlpPreference,
    });
    if (!added)
      return std::move(added).error();
    if (!*added)
      return Status::Ok();
  }
  return propose_norms(graph, uses,
                       static_cast<RegionContext *>(context_value)->executor,
                       allocator, writer);
}

bool contains_constant(std::span<const ConstantBinding> bindings, ValueId value,
                       const TensorDesc &description) noexcept {
  std::size_t matches = 0;
  for (const ConstantBinding &binding : bindings) {
    if (binding.value == value) {
      if (!binding.tensor.valid() ||
          binding.tensor.description() != description) {
        return false;
      }
      ++matches;
    }
  }
  return matches == 1;
}

void destroy_region_plan(RegionPlan *plan,
                         mem::AllocatorRef allocator) noexcept {
  if (plan == nullptr)
    return;
  compute::onednn::internal::destroy_brgemm_plan(plan->down, allocator);
  compute::onednn::internal::destroy_brgemm_plan(plan->gate_up, allocator);
  plan->~RegionPlan();
  allocator.deallocate(plan, sizeof(RegionPlan), alignof(RegionPlan));
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  destroy_region_plan(static_cast<RegionPlan *>(state), allocator);
}

compute::AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

Status create_invocation(void *prepared_value, mem::AllocatorRef allocator,
                         mem::BufferView workspace,
                         void **invocation_output) noexcept {
  if (invocation_output == nullptr)
    return Status::InvalidArgument("oneDNN region invocation output is null");
  *invocation_output = nullptr;
  auto *plan = static_cast<RegionPlan *>(prepared_value);
  if (plan == nullptr || plan->context == nullptr || plan->gate_up == nullptr ||
      plan->down == nullptr || plan->workspace.total_size == 0) {
    return Status::Internal("oneDNN region plan is invalid");
  }
  void *storage =
      allocator.allocate(sizeof(RegionInvocation), alignof(RegionInvocation));
  if (storage == nullptr) {
    return Status::ResourceExhausted(
        "oneDNN region invocation allocation failed");
  }
  auto *invocation = ::new (storage) RegionInvocation{};
  Status status = Status::Ok();
  if (workspace.size() < plan->workspace.total_size) {
    status = Status::InvalidArgument("oneDNN region workspace is too small");
  } else {
    auto mapping = workspace.map_host(mem::HostAccess::kReadWrite);
    if (!mapping) {
      status = std::move(mapping).error();
    } else {
      invocation->workspace_mapping = std::move(*mapping);
      auto bytes = invocation->workspace_mapping.writable_bytes();
      if (!bytes) {
        status = std::move(bytes).error();
      } else if (bytes->size() < plan->workspace.total_size ||
                 !mem::is_aligned_ptr(bytes->data(), kWorkspaceAlignment)) {
        status = Status::InvalidArgument(
            "oneDNN region workspace must be 64-byte aligned");
      } else {
        invocation->gate = reinterpret_cast<float *>(
            bytes->data() + plan->workspace.gate_offset);
        invocation->up = reinterpret_cast<float *>(bytes->data() +
                                                   plan->workspace.up_offset);
        invocation->scratch = bytes->data() + plan->workspace.scratch_offset;
        status = compute::onednn::internal::validate_brgemm_scratchpad(
            plan->gate_up, invocation->scratch, plan->workspace.scratch_size);
        if (status.ok()) {
          status = compute::onednn::internal::validate_brgemm_scratchpad(
              plan->down, invocation->scratch, plan->workspace.scratch_size);
        }
      }
    }
  }
  if (!status.ok()) {
    invocation->workspace_mapping.close().IgnoreError();
    invocation->~RegionInvocation();
    allocator.deallocate(invocation, sizeof(RegionInvocation),
                         alignof(RegionInvocation));
    return status;
  }
  *invocation_output = invocation;
  return Status::Ok();
}

void destroy_invocation(void *invocation_value,
                        mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<RegionInvocation *>(invocation_value);
  if (invocation == nullptr)
    return;
  invocation->workspace_mapping.close().IgnoreError();
  invocation->~RegionInvocation();
  allocator.deallocate(invocation, sizeof(RegionInvocation),
                       alignof(RegionInvocation));
}

struct ActivationTask final {
  float *gate = nullptr;
  const float *up = nullptr;
};

void apply_swiglu(void *context_value, std::size_t row_count,
                  std::size_t leading_columns, BrgemmTaskRange range) noexcept {
  auto &context = *static_cast<ActivationTask *>(context_value);
  if (range.first_column == 0 && range.column_count == leading_columns) {
    // The validated dense workspace is contiguous across rows. A full-range
    // task needs one batch/tail; partitioned columns must retain row strides.
    simd::swiglu_f32_batch(context.gate, context.up, context.gate,
                           row_count * leading_columns);
    return;
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::size_t offset = row * leading_columns + range.first_column;
    simd::swiglu_f32_batch(context.gate + offset, context.up + offset,
                           context.gate + offset, range.column_count);
  }
}

void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

compute::detail::EnqueueResult
submit(void *prepared_value, void *invocation_value, Queue &,
       std::span<const ConstTensorView> inputs,
       std::span<const TensorView> outputs) noexcept {
  auto *plan = static_cast<RegionPlan *>(prepared_value);
  auto *invocation = static_cast<RegionInvocation *>(invocation_value);
  if (plan == nullptr || plan->context == nullptr || plan->gate_up == nullptr ||
      plan->down == nullptr || invocation == nullptr ||
      invocation->gate == nullptr || invocation->up == nullptr ||
      invocation->scratch == nullptr || inputs.size() != 4 ||
      outputs.size() != 1) {
    return compute::detail::EnqueueResult::accepted_command(
        Status::Internal("oneDNN region execution state is invalid"));
  }

  for (const ConstTensorView &input : inputs) {
    const mem::ConstBufferView storage = input.storage();
    if (!storage.resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent)) {
      Status status = storage.invalidate();
      if (!status.ok())
        return compute::detail::EnqueueResult::accepted_command(
            std::move(status));
    }
  }

  std::array<mem::HostMapping, 4> input_mappings{};
  std::size_t mapped_input_count = 0;
  Status terminal = Status::Ok();
  for (; mapped_input_count < inputs.size(); ++mapped_input_count) {
    auto mapping =
        inputs[plan->input_indices[mapped_input_count]].storage().map_host();
    if (!mapping) {
      terminal = std::move(mapping).error();
      break;
    }
    input_mappings[mapped_input_count] = std::move(*mapping);
  }
  if (!terminal.ok()) {
    while (mapped_input_count != 0) {
      --mapped_input_count;
      preserve_first_error(terminal,
                           input_mappings[mapped_input_count].close());
    }
    return compute::detail::EnqueueResult::accepted_command(
        std::move(terminal));
  }

  auto output_result = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    terminal = std::move(output_result).error();
    for (std::size_t index = input_mappings.size(); index != 0; --index)
      preserve_first_error(terminal, input_mappings[index - 1].close());
    return compute::detail::EnqueueResult::accepted_command(
        std::move(terminal));
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    terminal = std::move(writable).error();
  } else {
    const std::array<const float *, 4> mapped_inputs{
        reinterpret_cast<const float *>(input_mappings[0].bytes().data()),
        reinterpret_cast<const float *>(input_mappings[1].bytes().data()),
        reinterpret_cast<const float *>(input_mappings[2].bytes().data()),
        reinterpret_cast<const float *>(input_mappings[3].bytes().data())};
    auto *mapped_output = reinterpret_cast<float *>(writable->data());
    ActivationTask activation{.gate = invocation->gate, .up = invocation->up};
    terminal = compute::onednn::internal::execute_brgemm_pair(
        plan->gate_up, plan->context->executor, mapped_inputs[0],
        mapped_inputs[1], mapped_inputs[2], invocation->gate, invocation->up,
        invocation->scratch, &apply_swiglu, &activation);
    if (terminal.ok()) {
      terminal = compute::onednn::internal::execute_brgemm(
          plan->down, plan->context->executor, invocation->gate,
          mapped_inputs[3], mapped_output, invocation->scratch);
    }
    if (terminal.ok() && !outputs[0].storage().resource().properties().has(
                             mem::MemoryProperty::kHostDeviceCoherent)) {
      preserve_first_error(terminal, output_mapping.flush());
    }
  }
  preserve_first_error(terminal, output_mapping.close());
  for (std::size_t index = input_mappings.size(); index != 0; --index)
    preserve_first_error(terminal, input_mappings[index - 1].close());
  return compute::detail::EnqueueResult::accepted_command(std::move(terminal));
}

const ProviderRegionPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

struct NormRegionPlan final {
  compute::detail::ProviderPrepared primitive{};
  std::array<std::size_t, norm::kNormMaxInputs> inputs{};
  std::size_t input_count = 0;
  std::size_t boundary_count = 0;
  ProviderRegionPreparedVTable vtable{};
};

void destroy_norm_region(void *value, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<NormRegionPlan *>(value);
  plan->primitive.vtable->destroy_prepared(plan->primitive.state, allocator);
  plan->~NormRegionPlan();
  allocator.deallocate(plan, sizeof(NormRegionPlan), alignof(NormRegionPlan));
}

compute::AliasSupport norm_region_alias(void *value, std::size_t output,
                                        std::size_t input) noexcept {
  const auto &plan = *static_cast<NormRegionPlan *>(value);
  return {.supported = output == 0 && input == plan.inputs[0]};
}

Status create_norm_invocation(void *value, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept {
  auto &prepared = static_cast<NormRegionPlan *>(value)->primitive;
  return prepared.vtable->create_invocation(prepared.state, allocator,
                                            workspace, invocation);
}

// The primitive owns the invocation. Its destroy function is copied into the
// immutable region vtable; a region needs no additional invocation wrapper.
compute::detail::EnqueueResult
submit_norm(void *value, void *invocation, Queue &queue,
            std::span<const ConstTensorView> inputs,
            std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<NormRegionPlan *>(value);
  if (inputs.size() != plan.boundary_count || outputs.size() != 1)
    return compute::detail::EnqueueResult::accepted_command(
        Status::Internal("oneDNN norm region bindings are invalid"));
  std::array<ConstTensorView, norm::kNormMaxInputs> ordered{};
  for (std::size_t index = 0; index < plan.input_count; ++index)
    ordered[index] = inputs[plan.inputs[index]];
  return plan.primitive.vtable->shared_submit(
      plan.primitive.state, invocation, queue,
      std::span(ordered).first(plan.input_count), outputs);
}

compute::ProviderPrepareOutcome
compile_norm(RegionContext &context, const RegionCompileRequest &request,
             mem::AllocatorRef allocator,
             ProviderRegionPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region output is null"));
  *prepared_output = {};
  if (request.graph == nullptr || request.nodes.size() < 2 ||
      request.nodes.size() > norm::kNormMaxPostOps + 1 ||
      request.outputs.size() != 1 ||
      request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != 1 ||
      !request.workspace_resource.valid())
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region request is invalid"));
  UseTable table;
  Status status = build_use_table(*request.graph, allocator, &table);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::error(std::move(status));
  NormPattern pattern;
  status = match_norm(*request.graph, request.nodes.front(),
                      std::span(table.values, table.count), context.executor,
                      pattern);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::error(std::move(status));
  if (pattern.node_count < 2 ||
      !std::ranges::equal(std::span(pattern.nodes).first(pattern.node_count),
                          request.nodes) ||
      pattern.output != request.outputs[0])
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region boundaries do not match"));
  std::array<std::size_t, norm::kNormMaxInputs> bindings{};
  for (std::size_t index = 0; index < pattern.input_count(); ++index) {
    auto found = std::ranges::find(request.inputs, pattern.inputs[index]);
    if (found == request.inputs.end())
      return compute::ProviderPrepareOutcome::error(
          Status::InvalidArgument("oneDNN norm region input is missing"));
    bindings[index] = static_cast<std::size_t>(found - request.inputs.begin());
  }
  for (const auto input : request.inputs) {
    if (std::ranges::find(
            std::span(pattern.inputs).first(pattern.input_count()), input) ==
        std::span(pattern.inputs).first(pattern.input_count()).end())
      return compute::ProviderPrepareOutcome::error(
          Status::InvalidArgument("oneDNN norm region has an unrelated input"));
  }
  compute::detail::ProviderPrepared primitive;
  auto outcome = norm::prepare_normalization(
      pattern.request(), context.executor, allocator, &primitive);
  if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
    return outcome;
  void *storage =
      allocator.allocate(sizeof(NormRegionPlan), alignof(NormRegionPlan));
  if (storage == nullptr) {
    primitive.vtable->destroy_prepared(primitive.state, allocator);
    return compute::ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted(
            "oneDNN norm region wrapper allocation failed"));
  }
  auto *region = ::new (storage) NormRegionPlan{
      .primitive = primitive,
      .inputs = bindings,
      .input_count = pattern.input_count(),
      .boundary_count = request.inputs.size(),
      .vtable = {.destroy_prepared = &destroy_norm_region,
                 .alias_support = &norm_region_alias,
                 .create_invocation = &create_norm_invocation,
                 .destroy_invocation = primitive.vtable->destroy_invocation,
                 .shared_submit = &submit_norm}};
  *prepared_output = {.vtable = &region->vtable,
                      .state = region,
                      .properties = {.workspace = primitive.workspace,
                                     .persistent_bytes = std::nullopt,
                                     .dispatch_count = 1,
                                     .runtime = norm::kNormRuntime,
                                     .invocation_private_bytes = std::nullopt}};
  return compute::ProviderPrepareOutcome::prepared();
}

Status propose_region_resources(
    void *context_value, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<compute::detail::ResourceBindingRequirement> input_requirements,
    std::span<compute::detail::ResourceBindingRequirement> output_requirements,
    RegionResourceProposal *proposal) noexcept {
  if (request.route == kNormalizationRoute) {
    if (context_value == nullptr || request.graph == nullptr ||
        proposal == nullptr || request.nodes.size() < 2 ||
        request.nodes.size() > norm::kNormMaxPostOps + 1 ||
        request.outputs.size() != 1 ||
        input_requirements.size() != request.inputs.size() ||
        output_requirements.size() != 1)
      return Status::InvalidArgument(
          "oneDNN norm resource proposal is invalid");
    for (std::size_t index = 0; index < request.inputs.size(); ++index) {
      auto description =
          request.graph->value_description(request.inputs[index]);
      if (!description)
        return std::move(description).error();
      input_requirements[index] = {
          .required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = description->required_alignment()};
    }
    auto description = request.graph->value_description(request.outputs[0]);
    if (!description)
      return std::move(description).error();
    output_requirements[0] = {
        .required_properties = mem::MemoryProperty::kHostAccessible,
        .minimum_alignment = description->required_alignment()};
    *proposal = {
        .runtime = norm::kNormRuntime,
        .workspace_resource = {
            .required_properties = mem::MemoryProperty::kHostAccessible,
            .minimum_alignment = norm::kNormWorkspaceAlignment}};
    return Status::Ok();
  }
  if (context_value == nullptr || request.graph == nullptr ||
      request.route != kGatedMlpRoute || request.nodes.size() != 4 ||
      request.inputs.size() != 4 || request.outputs.size() != 1 ||
      input_requirements.size() != request.inputs.size() ||
      output_requirements.size() != request.outputs.size() ||
      proposal == nullptr) {
    return Status::InvalidArgument(
        "oneDNN region resource proposal request is invalid");
  }
  for (auto &requirement : input_requirements) {
    requirement = {.required_properties = mem::MemoryProperty::kHostAccessible};
  }
  for (auto &requirement : output_requirements) {
    requirement = {.required_properties = mem::MemoryProperty::kHostAccessible};
  }
  *proposal = RegionResourceProposal{
      .runtime = kRuntimeProperties,
      .workspace_resource =
          {
              .required_properties = mem::MemoryProperty::kHostAccessible,
          },
  };
  return Status::Ok();
}

Status
compile_region_checked(void *context_value, const RegionCompileRequest &request,
                       mem::AllocatorRef allocator,
                       ProviderRegionPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return Status::InvalidArgument("oneDNN region prepared output is null");
  *prepared_output = {};
  auto *context = static_cast<RegionContext *>(context_value);
  if (context == nullptr || request.graph == nullptr ||
      request.route != kGatedMlpRoute || request.nodes.size() != 4 ||
      request.inputs.size() != 4 || request.outputs.size() != 1 ||
      request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size() ||
      !request.workspace_resource.valid()) {
    return Status::InvalidArgument("oneDNN region compile request is invalid");
  }

  UseTable table;
  Status status = build_use_table(*request.graph, allocator, &table);
  if (!status.ok())
    return status;
  MlpPattern pattern;
  bool matched = false;
  status = match_pattern(*request.graph, request.nodes.back().ordinal(),
                         std::span<const ValueUse>(table.values, table.count),
                         &pattern, &matched);
  if (!status.ok())
    return status;
  auto canonical_nodes = pattern.nodes;
  std::sort(canonical_nodes.begin(), canonical_nodes.end(),
            [](NodeId left, NodeId right) noexcept {
              return left.ordinal() < right.ordinal();
            });
  if (!matched || !std::ranges::equal(canonical_nodes, request.nodes) ||
      request.outputs[0] != pattern.output) {
    return Status::InvalidArgument(
        "oneDNN region boundaries do not match the gated MLP");
  }
  const std::array semantic_inputs{pattern.activation, pattern.gate_weight,
                                   pattern.up_weight, pattern.down_weight};
  std::array<std::uint32_t, 4> input_indices{};
  for (std::size_t index = 0; index < semantic_inputs.size(); ++index) {
    const auto found = std::find(request.inputs.begin(), request.inputs.end(),
                                 semantic_inputs[index]);
    if (found == request.inputs.end())
      return Status::InvalidArgument("oneDNN region is missing an input port");
    input_indices[index] =
        static_cast<std::uint32_t>(found - request.inputs.begin());
  }

  auto gate_weight_desc = request.graph->value_description(pattern.gate_weight);
  auto up_weight_desc = request.graph->value_description(pattern.up_weight);
  auto down_weight_desc = request.graph->value_description(pattern.down_weight);
  if (!gate_weight_desc)
    return std::move(gate_weight_desc).error();
  if (!up_weight_desc)
    return std::move(up_weight_desc).error();
  if (!down_weight_desc)
    return std::move(down_weight_desc).error();
  if (!contains_constant(request.constants, pattern.gate_weight,
                         *gate_weight_desc) ||
      !contains_constant(request.constants, pattern.up_weight,
                         *up_weight_desc) ||
      !contains_constant(request.constants, pattern.down_weight,
                         *down_weight_desc)) {
    return Status::InvalidArgument(
        "oneDNN region requires exactly one matching binding per weight");
  }

  const std::size_t gate_up_tasks =
      compute::onednn::internal::recommended_brgemm_task_count(
          pattern.hidden, context->executor.concurrency());
  const std::size_t down_tasks =
      compute::onednn::internal::recommended_brgemm_task_count(
          pattern.inner, context->executor.concurrency());
  BrgemmPlan *gate_up = nullptr;
  status = compute::onednn::internal::create_brgemm_plan(
      BrgemmShape{pattern.rows, pattern.inner, pattern.hidden}, gate_up_tasks,
      allocator, &gate_up);
  if (!status.ok())
    return status;
  BrgemmPlan *down = nullptr;
  status = compute::onednn::internal::create_brgemm_plan(
      BrgemmShape{pattern.rows, pattern.hidden, pattern.inner}, down_tasks,
      allocator, &down);
  if (!status.ok()) {
    compute::onednn::internal::destroy_brgemm_plan(gate_up, allocator);
    return status;
  }

  WorkspaceLayout workspace;
  const std::size_t scratch_size =
      (std::max)(compute::onednn::internal::brgemm_scratchpad_size(gate_up),
                 compute::onednn::internal::brgemm_scratchpad_size(down));
  status = compute_workspace_layout(pattern.rows, pattern.hidden, scratch_size,
                                    &workspace);
  if (!status.ok()) {
    compute::onednn::internal::destroy_brgemm_plan(down, allocator);
    compute::onednn::internal::destroy_brgemm_plan(gate_up, allocator);
    return status;
  }

  void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
  if (storage == nullptr) {
    compute::onednn::internal::destroy_brgemm_plan(down, allocator);
    compute::onednn::internal::destroy_brgemm_plan(gate_up, allocator);
    return Status::ResourceExhausted("oneDNN region plan allocation failed");
  }
  auto *plan = ::new (storage) RegionPlan{.context = context,
                                          .input_indices = input_indices,
                                          .gate_up = gate_up,
                                          .down = down,
                                          .workspace = workspace,
                                          .rows = pattern.rows,
                                          .hidden = pattern.hidden};
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = plan;
  prepared_output->properties = RegionArtifactProperties{
      .workspace = {.size = workspace.total_size,
                    .alignment = kWorkspaceAlignment},
      .persistent_bytes = std::nullopt,
      .packed_constant_count = 0,
      .packed_constant_bytes = 0,
      .copy_bytes = 0,
      .synchronization_count = 0,
      .dispatch_count = 2,
      .runtime = kRuntimeProperties,
  };
  return Status::Ok();
}

compute::ProviderPrepareOutcome
compile_region(void *context, const RegionCompileRequest &request,
               mem::AllocatorRef allocator,
               ProviderRegionPrepared *prepared) noexcept {
  if (request.route == kNormalizationRoute && context != nullptr)
    return compile_norm(*static_cast<RegionContext *>(context), request,
                        allocator, prepared);
  Status status = compile_region_checked(context, request, allocator, prepared);
  return status.ok()
             ? compute::ProviderPrepareOutcome::prepared()
             : compute::ProviderPrepareOutcome::error(std::move(status));
}

const RegionProviderVTable kRegionVTable{
    .propose = &propose_regions,
    .propose_resources = &propose_region_resources,
    .compile = &compile_region,
};

Status create_context(const Provider &provider_value, compute::Device device,
                      void *, mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("oneDNN region context output is null");
  *context_output = nullptr;
  if (device.device_class() != compute::DeviceClass::kCpu) {
    return Status::Unimplemented(
        "oneDNN graph region provider requires a CPU device");
  }
  if (!allocator.valid())
    return Status::FailedPrecondition("oneDNN region allocator is invalid");
  void *storage =
      allocator.allocate(sizeof(RegionContext), alignof(RegionContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN region context allocation failed");
  const Options options =
      compute::detail::ProviderAccess::payload<Options>(provider_value);
  *context_output = ::new (storage) RegionContext{.executor = options.executor};
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<RegionContext *>(context_value);
  if (context == nullptr)
    return;
  context->~RegionContext();
  allocator.deallocate(context, sizeof(RegionContext), alignof(RegionContext));
}

Status propose_single(void *, const compute::detail::ProviderOperationRequest &,
                      std::span<compute::detail::ResourceBindingRequirement>,
                      std::span<compute::detail::ResourceBindingRequirement>,
                      compute::detail::ProviderOperationProposal *) noexcept {
  return Status::Unimplemented(
      "oneDNN graph region provider has no single-operation route");
}

compute::ProviderPrepareOutcome
prepare_single(void *, const compute::detail::ProviderPrepareRequest &,
               mem::AllocatorRef,
               compute::detail::ProviderPrepared *) noexcept {
  return compute::ProviderPrepareOutcome::unsupported(Status::Unimplemented(
      "oneDNN graph region provider has no single-operation route"));
}

Status drain_context(void *) noexcept { return Status::Ok(); }

const void *query_extension(void *, compute::ProviderExtensionId id) noexcept {
  return id == kRegionProviderExtensionId ? &kRegionVTable : nullptr;
}

const compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "onednn-graph-region",
                 .version = "3.13.2"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose_single,
    .prepare = &prepare_single,
    .drain_context = &drain_context,
    .query_extension = &query_extension,
};

} // namespace

compute::Provider provider(Options options) noexcept {
  return compute::detail::ProviderAccess::make(&kProviderVTable, options);
}

} // namespace hyle::graph::onednn
