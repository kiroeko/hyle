#include "hyle/graph/providers/ggml_cuda.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <new>
#include <vector>

#include "hyle/graph/region_provider.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/compute/model_ops.h"
#include "../graph_ggml_shared/constant_packing.h"
#include "../graph/region_discovery.h"
#include "program.h"
#include "constant_expressions.h"

namespace hyle::graph::ggml_cuda {
namespace native = compute::ggml_cuda::internal;
namespace {

inline constexpr compute::RoutePolicyId kGgmlCudaRegionPolicy{
    0x48594c4547474355ULL, 1, 1};
inline constexpr compute::RoutePolicyId kGgmlCudaSiluMultiplyPolicy{
    0x48594c4547474355ULL, 2, 1};
inline constexpr std::uint32_t kGgmlCudaSiluMultiplyPreference = 200;
inline constexpr compute::RoutePolicyId kGgmlCudaDualProjectionPolicy{
    0x48594c4547474355ULL, 3, 1};
inline constexpr std::uint32_t kGgmlCudaDualProjectionPreference = 200;
inline constexpr compute::RoutePolicyId kGgmlCudaCastStableSiluPolicy{
    0x48594c4547474355ULL, 4, 1};
inline constexpr std::uint32_t kGgmlCudaCastStableSiluPreference = 200;
inline constexpr compute::RoutePolicyId kGgmlCudaPairedProjectionPolicy{
    0x48594c4547474355ULL, 5, 1};
inline constexpr std::uint32_t kGgmlCudaPairedProjectionPreference = 201;

Result<bool> supported_node(void *context, const Graph &graph,
                            NodeId node) noexcept {
  auto operation = graph.operation(node);
  if (!operation)
    return false; // Symbolic metadata views are not this exact native domain.
  auto input_count = graph.node_input_count(node);
  auto output_count = graph.node_output_count(node);
  auto effects = graph.node_resource_count(node);
  if (!input_count || !output_count || !effects)
    return std::unexpected(
        Status::InvalidArgument("ggml CUDA region node is invalid"));
  if (*output_count == 0 || *effects != 0)
    return false;
  try {
    std::vector<compute::TensorDesc> inputs(*input_count);
    std::vector<compute::TensorDesc> outputs(*output_count);
    for (std::size_t i = 0; i < *input_count; ++i) {
      auto id = graph.node_input(node, i);
      if (!id)
        return std::unexpected(std::move(id).error());
      auto description = graph.value_description(*id);
      if (!description)
        return false;
      inputs[i] = *description;
    }
    for (std::size_t i = 0; i < *output_count; ++i) {
      auto id = graph.result(node, i);
      if (!id)
        return std::unexpected(std::move(id).error());
      auto description = graph.value_description(*id);
      if (!description)
        return false;
      outputs[i] = *description;
    }
    return native::supports(context, *operation, inputs, outputs);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted(
        "ggml CUDA region descriptor allocation failed"));
  }
}

// Discovery only follows logical metadata ancestry. Graph later supplies the
// checked physical boundary, offset and owner; native lowering does not need to
// reinterpret the external View or assume that a replica is ready and owned.
bool has_constant_origin(const Graph &graph, ValueId value) noexcept {
  for (std::size_t depth = 0; depth <= graph.node_count(); ++depth) {
    if (value.kind() == ValueKind::kConstant) return true;
    if (value.kind() != ValueKind::kNodeOutput) return false;
    const auto producer = graph.producer(value);
    if (!producer) return false;
    const auto kind = graph.operation_kind(*producer);
    const auto inputs = graph.node_input_count(*producer);
    if (!kind || *kind != compute::OperationKind::kView || !inputs || *inputs != 1)
      return false;
    const auto input = graph.node_input(*producer, 0);
    if (!input) return false;
    value = *input;
  }
  return false;
}

bool single_constant_preparation(const Graph &graph, NodeId node) noexcept {
  const auto operation = graph.operation(node);
  if (!operation) return false;
  const auto traits = compute::operation_traits(*operation);
  if (traits.effect != compute::EffectKind::kPure || traits.resource_count != 0 ||
      traits.alias == compute::AliasPolicy::kMustAliasInput0)
    return false;
  const auto inputs = graph.node_input_count(node).value();
  if (inputs == 0) return false;
  for (std::size_t port = 0; port < inputs; ++port)
    if (!has_constant_origin(graph, graph.node_input(node, port).value())) return false;
  for (std::size_t port = 0; port < graph.node_output_count(node).value(); ++port) {
    const auto value = graph.result(node, port).value();
    const auto description = graph.value_description(value).value();
    if (!description.is_plain() || !description.is_dense() ||
        ggml_shared::published_view_of(graph, value)) return false;
  }
  return true;
}

// One discovery-local use index. It proves that delaying a private fused
// chain until its other inputs are ready cannot hide a data/control reader.
// It is not retained by the native recipe or queried during submit.
struct FusionUses final {
  static constexpr std::uint64_t kNone = UINT64_MAX;
  static constexpr std::uint64_t kMany = UINT64_MAX - 1;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> uses;
  container::SmallVector<std::uint64_t, 16, mem::AllocatorRef> control_readers;

  explicit FusionUses(mem::AllocatorRef allocator) noexcept
      : uses(allocator), control_readers(allocator) {}

  Status initialize(const Graph &graph) noexcept {
    if (!uses.try_resize(graph.value_count()) ||
        !control_readers.try_resize(graph.node_count()))
      return Status::ResourceExhausted("ggml CUDA fusion use metadata failed");
    std::ranges::fill(uses, std::uint8_t{0});
    std::ranges::fill(control_readers, kNone);
    for (std::size_t i = 0; i < graph.node_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
      HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, port));
        auto &readers = uses[value.ordinal()];
        if (readers < 2) ++readers;
      }
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(node));
      for (std::size_t index = 0; index < dependencies; ++index) {
        HYLE_ASSIGN_OR_RETURN(auto before, graph.node_dependency(node, index));
        auto &reader = control_readers[before.ordinal()];
        if (reader == kNone) reader = node.ordinal();
        else if (reader != node.ordinal()) reader = kMany;
      }
    }
    for (std::size_t i = 0; i < graph.output_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto value, graph.output_id(i));
      uses[value.ordinal()] = 2;
    }
    return Status::Ok();
  }
};

// This private implementation preserves the public BF16 Cast boundary in
// registers. Its F32 source and terminal SiLU may escape; only the Cast
// result must have no other data/control observer. Native qualification still
// checks the actual metadata, bound storage and marked stable-SiLU contract.
template <class CanElide>
Result<bool> matches_cast_stable_silu(void *context, const Graph &graph,
                                    std::span<const NodeId> nodes,
                                    CanElide can_elide) noexcept {
  if (nodes.size() != 2) return false;
  const auto first = graph.operation(nodes[0]);
  const auto last = graph.operation(nodes[1]);
  const auto *cast = first ? first->get_if<compute::Cast>() : nullptr;
  const auto *unary = last ? last->get_if<compute::Unary>() : nullptr;
  if (!cast || cast->output_type != compute::DataType::kBFloat16 ||
      !unary || unary->kind != compute::UnaryKind::kSilu) return false;
  for (const auto node : nodes) {
    HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
    HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
    if (inputs != 1 || outputs != 1) return false;
  }
  HYLE_ASSIGN_OR_RETURN(auto source, graph.node_input(nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(auto intermediate, graph.result(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(nodes[1], 0));
  if (input != intermediate || !can_elide(nodes[0], nodes[1], intermediate))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto terminal, graph.result(nodes[1]));
  HYLE_ASSIGN_OR_RETURN(auto source_desc, graph.value_description(source));
  HYLE_ASSIGN_OR_RETURN(auto cast_desc, graph.value_description(intermediate));
  HYLE_ASSIGN_OR_RETURN(auto output_desc, graph.value_description(terminal));
  if (!source_desc.is_plain() || !source_desc.is_dense() ||
      source_desc.data_type() != compute::DataType::kFloat32 ||
      !cast_desc.is_plain() || !cast_desc.is_dense() ||
      cast_desc.data_type() != compute::DataType::kBFloat16 ||
      !std::ranges::equal(source_desc.dimensions(), cast_desc.dimensions()) ||
      output_desc != cast_desc) return false;
  HYLE_ASSIGN_OR_RETURN(auto count, source_desc.element_count());
  if (count == 0) return false;
  for (const auto node : nodes) {
    HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
    if (!supported) return false;
  }
  return true;
}

template <class CanElide>
Result<bool> matches_silu_multiply(void *context, const Graph &graph,
                                   std::span<const NodeId> nodes,
                                   CanElide can_elide) noexcept {
  if (nodes.size() != 2) return false;
  const auto silu = graph.operation(nodes[0]);
  const auto multiply = graph.operation(nodes[1]);
  if (!silu || !multiply) return false;
  const auto *unary = silu->get_if<compute::Unary>();
  const auto *binary = multiply->get_if<compute::Binary>();
  if (!unary || unary->kind != compute::UnaryKind::kSilu || !binary ||
      binary->kind != compute::BinaryKind::kMultiply)
    return false;
  HYLE_ASSIGN_OR_RETURN(auto silu_inputs, graph.node_input_count(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto mul_inputs, graph.node_input_count(nodes[1]));
  HYLE_ASSIGN_OR_RETURN(auto silu_outputs, graph.node_output_count(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto mul_outputs, graph.node_output_count(nodes[1]));
  if (silu_inputs != 1 || mul_inputs != 2 || silu_outputs != 1 || mul_outputs != 1)
    return false;
  HYLE_ASSIGN_OR_RETURN(auto gate, graph.node_input(nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(auto intermediate, graph.result(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto left, graph.node_input(nodes[1], 0));
  HYLE_ASSIGN_OR_RETURN(auto right, graph.node_input(nodes[1], 1));
  if ((left != intermediate && right != intermediate) || left == right ||
      !can_elide(nodes[0], nodes[1], intermediate))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto output, graph.result(nodes[1]));
  HYLE_ASSIGN_OR_RETURN(auto description, graph.value_description(gate));
  const auto type = description.data_type();
  if (!description.is_plain() || !description.is_dense() ||
      (type != compute::DataType::kFloat32 &&
       type != compute::DataType::kBFloat16))
    return false;
  for (const auto value : {intermediate, left, right, output}) {
    HYLE_ASSIGN_OR_RETURN(auto actual, graph.value_description(value));
    if (actual != description) return false;
  }
  for (const auto node : nodes) {
    HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
    if (!supported) return false;
  }
  return true;
}

// The same stable F32 SiLU/product implementation may load an exactly widened
// BF16 input and narrow only its final product. Neither F32 intermediate is a
// BF16 rounding boundary. External metadata views remain checked input ports.
template <class CanElide>
Result<bool> matches_typed_silu_multiply(void *context, const Graph &graph,
                                       std::span<const NodeId> nodes,
                                       CanElide can_elide) noexcept {
  if (nodes.size() != 4) return false;
  const auto first = graph.operation(nodes.front());
  const auto last = graph.operation(nodes.back());
  const auto *widen = first ? first->get_if<compute::Cast>() : nullptr;
  const auto *narrow = last ? last->get_if<compute::Cast>() : nullptr;
  if (!widen || widen->output_type != compute::DataType::kFloat32 ||
      !narrow || narrow->output_type != compute::DataType::kBFloat16)
    return false;
  for (const auto node : {nodes.front(), nodes.back()}) {
    HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
    HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
    if (inputs != 1 || outputs != 1) return false;
  }
  HYLE_ASSIGN_OR_RETURN(auto pair, matches_silu_multiply(context, graph,
      nodes.subspan(1, 2), [&](NodeId producer, NodeId, ValueId value) noexcept {
        return can_elide(producer, value);
      }));
  if (!pair) return false;
  HYLE_ASSIGN_OR_RETURN(auto source, graph.node_input(nodes.front(), 0));
  HYLE_ASSIGN_OR_RETURN(auto wide, graph.result(nodes.front()));
  HYLE_ASSIGN_OR_RETURN(auto unary_input, graph.node_input(nodes[1], 0));
  HYLE_ASSIGN_OR_RETURN(auto product, graph.result(nodes[2]));
  HYLE_ASSIGN_OR_RETURN(auto store_input, graph.node_input(nodes.back(), 0));
  HYLE_ASSIGN_OR_RETURN(auto output, graph.result(nodes.back()));
  if (unary_input != wide || store_input != product ||
      !can_elide(nodes.front(), wide) || !can_elide(nodes[2], product))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto source_desc, graph.value_description(source));
  HYLE_ASSIGN_OR_RETURN(auto wide_desc, graph.value_description(wide));
  HYLE_ASSIGN_OR_RETURN(auto output_desc, graph.value_description(output));
  if (!source_desc.is_plain() || !source_desc.is_dense() ||
      source_desc.data_type() != compute::DataType::kBFloat16 ||
      wide_desc.data_type() != compute::DataType::kFloat32 ||
      output_desc != source_desc ||
      !std::ranges::equal(source_desc.dimensions(), wide_desc.dimensions()))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto count, source_desc.element_count());
  if (count == 0) return false;
  for (const auto node : {nodes.front(), nodes.back()}) {
    HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
    if (!supported) return false;
  }
  return true;
}


// Private implementation candidate, not a new public operation. The native
// matcher must still admit the actual device, edges, precision and read/write
// ranges before eliding any store. Descriptors alone do not prove a kernel hit.
template <class CanElide>
Result<bool> matches_bfloat_dual_projection(
    void *context, const Graph &graph, std::span<const NodeId> nodes,
    CanElide can_elide) noexcept {
  if (nodes.size() != 4) return false;
  NodeId silu, multiply;
  std::array<NodeId, 2> projections{};
  std::size_t count = 0;
  for (const auto node : nodes) {
    const auto operation = graph.operation(node);
    if (!operation) return false;
    const auto traits = compute::operation_traits(*operation);
    HYLE_ASSIGN_OR_RETURN(auto resources, graph.node_resource_count(node));
    HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
    if (resources != 0 || outputs != 1 || traits.effect != compute::EffectKind::kPure ||
        traits.resource_count != 0) return false;
    if (const auto *matmul = operation->get_if<compute::MatMul>()) {
      if (count == projections.size() ||
          matmul->accumulation_type != compute::DataType::kFloat32) return false;
      projections[count++] = node;
    } else if (const auto *unary = operation->get_if<compute::Unary>()) {
      if (silu.valid() || unary->kind != compute::UnaryKind::kSilu) return false;
      silu = node;
    } else if (const auto *binary = operation->get_if<compute::Binary>()) {
      if (multiply.valid() || binary->kind != compute::BinaryKind::kMultiply) return false;
      multiply = node;
    } else return false;
  }
  if (count != 2 || !silu.valid() || !multiply.valid()) return false;
  HYLE_ASSIGN_OR_RETURN(auto unary_inputs, graph.node_input_count(silu));
  HYLE_ASSIGN_OR_RETURN(auto multiply_inputs, graph.node_input_count(multiply));
  if (unary_inputs != 1 || multiply_inputs != 2) return false;
  HYLE_ASSIGN_OR_RETURN(auto projected_gate, graph.node_input(silu, 0));
  HYLE_ASSIGN_OR_RETURN(auto activation, graph.result(silu));
  HYLE_ASSIGN_OR_RETURN(auto left, graph.node_input(multiply, 0));
  HYLE_ASSIGN_OR_RETURN(auto right, graph.node_input(multiply, 1));
  if (left == right || (left != activation && right != activation)) return false;
  const auto projected_up = left == activation ? right : left;
  if (projected_gate.kind() != ValueKind::kNodeOutput ||
      projected_up.kind() != ValueKind::kNodeOutput) return false;
  HYLE_ASSIGN_OR_RETURN(auto gate, graph.producer(projected_gate));
  HYLE_ASSIGN_OR_RETURN(auto up, graph.producer(projected_up));
  if (gate == up ||
      !((gate == projections[0] && up == projections[1]) ||
        (gate == projections[1] && up == projections[0]))) return false;
  HYLE_ASSIGN_OR_RETURN(auto output, graph.result(multiply));
  HYLE_ASSIGN_OR_RETURN(auto description, graph.value_description(output));
  if (!description.is_plain() || !description.is_dense() || description.rank() != 2 ||
      description.data_type() != compute::DataType::kBFloat16 ||
      description.dimensions()[0] != 1 || description.dimensions()[1] <= 0 ||
      description.dimensions()[1] > std::numeric_limits<int>::max()) return false;
  for (const auto value : {projected_gate, projected_up, activation}) {
    HYLE_ASSIGN_OR_RETURN(auto actual, graph.value_description(value));
    if (actual != description) return false;
  }
  if (!can_elide(gate, projected_gate) || !can_elide(up, projected_up) ||
      !can_elide(silu, activation)) return false;
  ValueId common_input;
  for (const auto projection : projections) {
    HYLE_ASSIGN_OR_RETURN(auto arity, graph.node_input_count(projection));
    if (arity != 2) return false;
    HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(projection, 0));
    HYLE_ASSIGN_OR_RETURN(auto weight, graph.node_input(projection, 1));
    if (common_input.valid() && input != common_input) return false;
    common_input = input;
    HYLE_ASSIGN_OR_RETURN(auto x, graph.value_description(input));
    HYLE_ASSIGN_OR_RETURN(auto w, graph.value_description(weight));
    if (!x.is_plain() || !x.is_dense() || x.rank() != 2 ||
        x.data_type() != compute::DataType::kBFloat16 || x.dimensions()[0] != 1 ||
        x.dimensions()[1] <= 0 || x.dimensions()[1] % 2 != 0 ||
        x.dimensions()[1] > std::numeric_limits<int>::max() ||
        !w.is_plain() || w.rank() != 2 ||
        w.data_type() != compute::DataType::kBFloat16 ||
        w.dimensions()[0] != x.dimensions()[1] ||
        w.dimensions()[1] != description.dimensions()[1]) return false;
    // The reused MMVF forms row*stride_row in int before widening it.
    // Check the product by division, without overflowing descriptor arithmetic.
    if (description.dimensions()[1] >
        std::numeric_limits<int>::max() / x.dimensions()[1]) return false;
    // Only the metadata transpose of contiguous [N,K] weight rows. Ordinary
    // [K,N], padding and other strides need native CONT and are outside this
    // no-copy implementation even though ordinary MatMul may support them.
    HYLE_ASSIGN_OR_RETURN(auto physical, w.transposed());
    if (!physical.is_dense()) return false;
  }
  for (const auto node : nodes) {
    HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
    if (!supported) return false;
  }
  return true;
}

struct BfloatProjection final {
  NodeId node;
  ValueId activation;
  std::int64_t inner = 0, rows = 0;
};

// Both original stores survive. Publication, outside readers and output order
// are not private-intermediate restrictions for this two-output carrier.
Result<bool> bfloat_projection(void *context, const Graph &graph, NodeId node,
                              BfloatProjection &projection) noexcept {
  const auto operation = graph.operation(node);
  const auto *matmul = operation ? operation->get_if<compute::MatMul>() : nullptr;
  if (!matmul || matmul->accumulation_type != compute::DataType::kFloat32)
    return false;
  const auto traits = compute::operation_traits(*operation);
  HYLE_ASSIGN_OR_RETURN(auto resources, graph.node_resource_count(node));
  HYLE_ASSIGN_OR_RETURN(auto arity, graph.node_input_count(node));
  HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
  if (resources != 0 || arity != 2 || outputs != 1 ||
      traits.effect != compute::EffectKind::kPure || traits.resource_count != 0)
    return false;
  HYLE_ASSIGN_OR_RETURN(auto activation, graph.node_input(node, 0));
  HYLE_ASSIGN_OR_RETURN(auto weight, graph.node_input(node, 1));
  HYLE_ASSIGN_OR_RETURN(auto result, graph.result(node));
  HYLE_ASSIGN_OR_RETURN(auto x, graph.value_description(activation));
  HYLE_ASSIGN_OR_RETURN(auto w, graph.value_description(weight));
  HYLE_ASSIGN_OR_RETURN(auto y, graph.value_description(result));
  if (!x.is_plain() || !x.is_dense() || x.rank() != 2 ||
      x.data_type() != compute::DataType::kBFloat16 || x.dimensions()[0] != 1 ||
      !y.is_plain() || !y.is_dense() || y.rank() != 2 ||
      y.data_type() != compute::DataType::kBFloat16 || y.dimensions()[0] != 1 ||
      !w.is_plain() || w.rank() != 2 ||
      w.data_type() != compute::DataType::kBFloat16)
    return false;
  const auto k = x.dimensions()[1], n = y.dimensions()[1];
  if (k <= 0 || k % 2 != 0 || k > std::numeric_limits<int>::max() || n <= 0 ||
      n > std::numeric_limits<int>::max() / k ||
      w.dimensions()[0] != k || w.dimensions()[1] != n)
    return false;
  HYLE_ASSIGN_OR_RETURN(auto physical, w.transposed());
  if (!physical.is_dense() ||
      !native::supports_bfloat_paired_projection(context, std::uint64_t(k), std::uint64_t(n)))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
  if (!supported) return false;
  projection = {node, activation, k, n};
  return true;
}

Result<bool> matches_bfloat_paired_projection(
    void *context, const Graph &graph, std::span<const NodeId> nodes) noexcept {
  if (nodes.size() != 2 || nodes[0] == nodes[1]) return false;
  std::array<BfloatProjection, 2> projections;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto supported, bfloat_projection(context, graph, nodes[i], projections[i]));
    if (!supported) return false;
  }
  return projections[0].activation == projections[1].activation &&
      projections[0].inner == projections[1].inner && projections[0].rows == projections[1].rows;
}

// Match an already implemented native numerical chain, following its actual
// edges. An external View/Gather stays a boundary; its one activation Cast
// may move to the RMS entry when neither F32 intermediate escapes.
template <class CanElide>
Result<bool> matches_rms_casts(const Graph &graph, std::span<const NodeId> nodes,
                              CanElide can_elide) noexcept {
  if (nodes.size() != 3) return false;
  const auto widen = graph.operation(nodes[0]);
  const auto norm = graph.operation(nodes[1]);
  const auto narrow = graph.operation(nodes[2]);
  if (!widen || !norm || !narrow) return false;
  const auto *first = widen->get_if<compute::Cast>();
  const auto *rms = norm->get_if<compute::RmsNorm>();
  const auto *last = narrow->get_if<compute::Cast>();
  if (!first || first->output_type != compute::DataType::kFloat32 ||
      !rms || rms->accumulation_type != compute::DataType::kFloat32 ||
      !last || last->output_type != compute::DataType::kBFloat16)
    return false;
  HYLE_ASSIGN_OR_RETURN(auto first_inputs, graph.node_input_count(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto rms_inputs, graph.node_input_count(nodes[1]));
  HYLE_ASSIGN_OR_RETURN(auto last_inputs, graph.node_input_count(nodes[2]));
  if (first_inputs != 1 || (rms_inputs != 1 && rms_inputs != 2) || last_inputs != 1)
    return false;
  for (const auto node : nodes) {
    HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
    if (outputs != 1) return false;
  }
  HYLE_ASSIGN_OR_RETURN(auto x, graph.node_input(nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(auto wide, graph.result(nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto rms_input, graph.node_input(nodes[1], 0));
  HYLE_ASSIGN_OR_RETURN(auto normalized, graph.result(nodes[1]));
  HYLE_ASSIGN_OR_RETURN(auto last_input, graph.node_input(nodes[2], 0));
  HYLE_ASSIGN_OR_RETURN(auto output, graph.result(nodes[2]));
  if (rms_input != wide || last_input != normalized ||
      !can_elide(nodes[0], wide) || !can_elide(nodes[1], normalized))
    return false;
  HYLE_ASSIGN_OR_RETURN(auto input_desc, graph.value_description(x));
  HYLE_ASSIGN_OR_RETURN(auto wide_desc, graph.value_description(wide));
  HYLE_ASSIGN_OR_RETURN(auto rms_desc, graph.value_description(normalized));
  HYLE_ASSIGN_OR_RETURN(auto output_desc, graph.value_description(output));
  if (!input_desc.is_plain() || input_desc.data_type() != compute::DataType::kBFloat16 ||
      !wide_desc.is_plain() || !wide_desc.is_dense() ||
      wide_desc.data_type() != compute::DataType::kFloat32 || rms_desc != wide_desc ||
      !output_desc.is_plain() || !output_desc.is_dense() ||
      output_desc.data_type() != compute::DataType::kBFloat16 ||
      !std::ranges::equal(input_desc.dimensions(), wide_desc.dimensions()) ||
      !std::ranges::equal(input_desc.dimensions(), output_desc.dimensions()))
    return false;
  if (rms_inputs == 2) {
    HYLE_ASSIGN_OR_RETURN(auto scale, graph.node_input(nodes[1], 1));
    HYLE_ASSIGN_OR_RETURN(auto scale_desc, graph.value_description(scale));
    if (scale_desc.data_type() != compute::DataType::kFloat32) return false;
  }
  return true;
}

Status propose(void *context, const RegionDiscoveryRequest &request, mem::AllocatorRef allocator,
               RegionProposalWriter &writer) noexcept {
  const auto &graph = request.graph;
  if ((!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count()) ||
      (!request.leaf_preferences.empty() && request.leaf_preferences.size() != graph.node_count()))
    return Status::InvalidArgument("ggml CUDA discovery mask size differs");
  if (graph.node_count() == 0)
    return Status::Ok();
  // Ordinary islands preserve admitted leaf preferences. A specific fusion
  // may cross a nonpreferred leaf because it offers a different implementation.
  constexpr std::uint8_t kSupported = 1, kOrdinary = 2;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> eligible(allocator);
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> ordinary_owner(allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> ffn_projection(allocator);
  if (!eligible.try_resize(graph.node_count()) ||
      !ordinary_owner.try_resize(graph.node_count()) ||
      !ffn_projection.try_resize(graph.node_count()))
    return Status::ResourceExhausted("ggml CUDA discovery eligibility failed");
  std::ranges::fill(ordinary_owner, std::size_t{0});
  std::ranges::fill(ffn_projection, std::uint8_t{0});
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    eligible[i] = 0;
    if (!request.allowed_nodes.empty() && !request.allowed_nodes[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto supported, supported_node(context, graph, node));
    if (!supported) continue;
    eligible[i] = kOrdinary;
    if (!request.leaf_preferences.empty()) {
      const auto &facts = request.leaf_preferences[i];
      if (facts.provider && facts.best && *facts.provider > *facts.best) {
        eligible[i] = kSupported;
      }
    }
  }
  FusionUses uses(allocator);
  HYLE_RETURN_IF_ERROR(uses.initialize(graph));

  // A distinct no-copy implementation may cover two otherwise preferred BLAS
  // leaves. Their ordinary ranks and all nonmatching regions are unchanged.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (!eligible[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto multiply, graph.node_id(i));
    const auto operation = graph.operation(multiply);
    const auto *binary = operation ? operation->get_if<compute::Binary>() : nullptr;
    if (!binary || binary->kind != compute::BinaryKind::kMultiply) continue;
    for (std::size_t port = 0; port < 2; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto activation, graph.node_input(multiply, port));
      HYLE_ASSIGN_OR_RETURN(auto projected_up, graph.node_input(multiply, 1 - port));
      if (activation.kind() != ValueKind::kNodeOutput ||
          projected_up.kind() != ValueKind::kNodeOutput) continue;
      HYLE_ASSIGN_OR_RETURN(auto silu, graph.producer(activation));
      HYLE_ASSIGN_OR_RETURN(auto up, graph.producer(projected_up));
      const auto unary_op = graph.operation(silu);
      const auto *unary = unary_op ? unary_op->get_if<compute::Unary>() : nullptr;
      if (!unary || unary->kind != compute::UnaryKind::kSilu) continue;
      HYLE_ASSIGN_OR_RETURN(auto projected_gate, graph.node_input(silu, 0));
      if (projected_gate.kind() != ValueKind::kNodeOutput) continue;
      HYLE_ASSIGN_OR_RETURN(auto gate, graph.producer(projected_gate));
      const std::array nodes{gate, silu, up, multiply};
      if (std::ranges::any_of(nodes, [&](NodeId node) noexcept {
            return !eligible[node.ordinal()];
          })) continue;
      HYLE_ASSIGN_OR_RETURN(auto matches, matches_bfloat_dual_projection(
          context, graph, nodes, [&](NodeId producer, ValueId value) noexcept {
            const auto control = uses.control_readers[producer.ordinal()];
            return uses.uses[value.ordinal()] == 1 &&
                (control == FusionUses::kNone || std::ranges::any_of(nodes,
                    [&](NodeId node) noexcept { return control == node.ordinal(); }));
          }));
      if (!matches) continue;
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
          .nodes = nodes,
          .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                   .copy_bytes = 0, .repack_bytes = 0,
                   .synchronization_count = 2, .dispatch_count = 1},
          .route = 3,
          .replay_class = compute::ProviderReplayClass::kManaged,
          .implementation = compute::ProviderImplementationId{4},
          .policy = kGgmlCudaDualProjectionPolicy,
          .preference = kGgmlCudaDualProjectionPreference}));
      if (!more) return Status::Ok();
      // Reserve only an actually accepted complete FFN proposal. Sibling
      // pairing must not split its two private projections into another route.
      ffn_projection[gate.ordinal()] = 1;
      ffn_projection[up.ordinal()] = 1;
      break;
    }
  }

  // Same logical activation and geometry, grouped once rather than comparing
  // every pair. Different ValueIds bound to the same address remain distinct.
  container::SmallVector<BfloatProjection, 16, mem::AllocatorRef> projections(allocator);
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (!eligible[i] || ffn_projection[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    BfloatProjection projection;
    HYLE_ASSIGN_OR_RETURN(auto supported, bfloat_projection(context, graph, node, projection));
    if (!supported || single_constant_preparation(graph, node)) continue;
    if (!projections.try_push_back(projection))
      return Status::ResourceExhausted("ggml CUDA paired projection metadata failed");
  }
  const auto same_group = [](const BfloatProjection &a, const BfloatProjection &b) noexcept {
    return a.activation == b.activation && a.inner == b.inner && a.rows == b.rows;
  };
  std::ranges::sort(projections, [](const BfloatProjection &a, const BfloatProjection &b) noexcept {
    if (a.activation != b.activation) return a.activation.ordinal() < b.activation.ordinal();
    if (a.inner != b.inner) return a.inner < b.inner;
    if (a.rows != b.rows) return a.rows < b.rows;
    return a.node.ordinal() < b.node.ordinal();
  });
  for (std::size_t i = 1; i < projections.size(); ++i) {
    if (!same_group(projections[i - 1], projections[i])) continue;
    const std::array nodes{projections[i - 1].node, projections[i].node};
    // The common contraction proof handles nonadjacent data/control paths;
    // do not publish a region which leaves this pair and later re-enters it.
    HYLE_ASSIGN_OR_RETURN(auto contractible, is_region_contractible(graph, nodes, allocator));
    if (!contractible) continue;
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
        .nodes = nodes,
        .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                 .copy_bytes = 0, .repack_bytes = 0,
                 .synchronization_count = 2, .dispatch_count = 1},
        .route = 5,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{6},
        .policy = kGgmlCudaPairedProjectionPolicy,
        .preference = kGgmlCudaPairedProjectionPreference}));
    if (!more) return Status::Ok();
    ++i; // Each successfully proposed sibling participates in this one pair.
  }

  // Preserve the existing stable SiLU/product family while admitting its
  // private typed load/store boundary. The original pair stays available when
  // either Cast or an F32 intermediate has another observer.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (!eligible[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto narrow, graph.node_id(i));
    const auto operation = graph.operation(narrow);
    const auto *cast = operation ? operation->get_if<compute::Cast>() : nullptr;
    if (!cast || cast->output_type != compute::DataType::kBFloat16) continue;
    HYLE_ASSIGN_OR_RETURN(auto product, graph.node_input(narrow, 0));
    if (product.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto multiply, graph.producer(product));
    const auto mul_operation = graph.operation(multiply);
    const auto *binary = mul_operation ? mul_operation->get_if<compute::Binary>() : nullptr;
    if (!binary || binary->kind != compute::BinaryKind::kMultiply ||
        !eligible[multiply.ordinal()]) continue;
    for (std::size_t port = 0; port < 2; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto activated, graph.node_input(multiply, port));
      if (activated.kind() != ValueKind::kNodeOutput) continue;
      HYLE_ASSIGN_OR_RETURN(auto silu, graph.producer(activated));
      const auto silu_operation = graph.operation(silu);
      const auto *unary = silu_operation ? silu_operation->get_if<compute::Unary>() : nullptr;
      if (!unary || unary->kind != compute::UnaryKind::kSilu ||
          !eligible[silu.ordinal()]) continue;
      HYLE_ASSIGN_OR_RETURN(auto wide, graph.node_input(silu, 0));
      if (wide.kind() != ValueKind::kNodeOutput) continue;
      HYLE_ASSIGN_OR_RETURN(auto widen, graph.producer(wide));
      if (!eligible[widen.ordinal()]) continue;
      const std::array nodes{widen, silu, multiply, narrow};
      HYLE_ASSIGN_OR_RETURN(auto matches, matches_typed_silu_multiply(
          context, graph, nodes, [&](NodeId producer, ValueId value) noexcept {
            const auto control = uses.control_readers[producer.ordinal()];
            return uses.uses[value.ordinal()] == 1 &&
                (control == FusionUses::kNone || std::ranges::any_of(nodes,
                    [&](NodeId node) noexcept { return control == node.ordinal(); }));
          }));
      if (!matches) continue;
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
          .nodes = nodes,
          .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                   .copy_bytes = 0, .repack_bytes = 0,
                   .synchronization_count = 2, .dispatch_count = 1},
          .route = 2,
          .replay_class = compute::ProviderReplayClass::kManaged,
          .implementation = compute::ProviderImplementationId{3},
          .policy = kGgmlCudaSiluMultiplyPolicy,
          .preference = kGgmlCudaSiluMultiplyPreference - 1}));
      if (!more) return Status::Ok();
      break;
    }
  }

  // Follow the multiply's actual operands, independent of insertion adjacency.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (!eligible[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto multiply, graph.node_id(i));
    const auto operation = graph.operation(multiply);
    const auto *binary = operation ? operation->get_if<compute::Binary>() : nullptr;
    if (!binary || binary->kind != compute::BinaryKind::kMultiply) continue;
    for (std::size_t port = 0; port < 2; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(multiply, port));
      if (value.kind() != ValueKind::kNodeOutput) continue;
      HYLE_ASSIGN_OR_RETURN(auto silu, graph.producer(value));
      if (!eligible[silu.ordinal()]) continue;
      const std::array nodes{silu, multiply};
      HYLE_ASSIGN_OR_RETURN(auto matches, matches_silu_multiply(context, graph, nodes,
          [&](NodeId first, NodeId last, ValueId intermediate) noexcept {
            const auto control = uses.control_readers[first.ordinal()];
            return uses.uses[intermediate.ordinal()] == 1 &&
                   (control == FusionUses::kNone || control == last.ordinal());
          }));
      if (!matches) continue;
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
          .nodes = nodes,
          .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                   .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
                   .synchronization_count = 2, .dispatch_count = 1},
          .route = 2,
          .replay_class = compute::ProviderReplayClass::kManaged,
          .implementation = compute::ProviderImplementationId{3},
          .policy = kGgmlCudaSiluMultiplyPolicy,
          .preference = kGgmlCudaSiluMultiplyPreference}));
      if (!more) return Status::Ok();
      break;
    }
  }
  const auto emit = [&](std::span<const NodeId> nodes) -> Result<bool> {
    // Immutable preparation also applies to a single ordinary operation;
    // dynamic singleton operations retain their ordinary provider ranks.
    if (nodes.empty() || (nodes.size() == 1 && !single_constant_preparation(graph, nodes[0])))
      return true;
    std::uint32_t preference = 0;
    bool has_math = false;
    for (const auto node : nodes) {
      // The actual supported public View is metadata, not an unranked
      // mathematical provider alternative. Do not invent a provider rank for it.
      const auto operation = graph.operation(node);
      if (operation && operation->get_if<compute::View>()) continue;
      has_math = true;
      if (request.leaf_preferences.empty() ||
          !request.leaf_preferences[node.ordinal()].provider) {
        preference = UINT32_MAX;
        break;
      }
      preference = std::max(preference, *request.leaf_preferences[node.ordinal()].provider);
    }
    if (!has_math) preference = UINT32_MAX;
    return writer.try_add(
        {.nodes = nodes,
         .cost = {.workspace = std::nullopt,
                  .persistent_bytes = std::nullopt,
                  .copy_bytes = std::nullopt,
                  .repack_bytes = std::nullopt,
                  .synchronization_count = 2,
                  .dispatch_count = 1},
         .route = 1,
         .replay_class = compute::ProviderReplayClass::kManaged,
         .implementation = compute::ProviderImplementationId{2},
         .policy = kGgmlCudaRegionPolicy,
         .preference = preference});
  };
  // Record the original ordinary ownership without spending writer budget.
  // A suffix already inside one emitted region needs no duplicate proposal.
  std::size_t next_owner = 1;
  HYLE_RETURN_IF_ERROR(detail::discover_regions(graph, allocator,
      [&](NodeId node) -> Result<bool> {
        return eligible[node.ordinal()] == kOrdinary;
      }, [&](std::span<const NodeId> nodes) -> Result<bool> {
        if (nodes.empty() || (nodes.size() == 1 &&
            !single_constant_preparation(graph, nodes[0]))) return true;
        for (const auto node : nodes) ordinary_owner[node.ordinal()] = next_owner;
        ++next_owner;
        return true;
      }));

  const auto emit_load_suffix = [&](std::span<NodeId> nodes,
                                    NodeId terminal) -> Result<bool> {
    std::ranges::sort(nodes, {}, &NodeId::ordinal);
    const auto owner = ordinary_owner[terminal.ordinal()];
    if (owner != 0 && std::ranges::all_of(nodes, [&](NodeId node) noexcept {
          return ordinary_owner[node.ordinal()] == owner;
        })) return true;
    const auto before = writer.size();
    HYLE_ASSIGN_OR_RETURN(auto more, emit(nodes));
    if (writer.size() != before)
      for (const auto node : nodes)
        eligible[node.ordinal()] = kSupported;
    return more;
  };

  // A GDN may acquire Q/K/decay from different external frontiers after its
  // private V/beta widening was emitted in an earlier ordinary prefix. Delay
  // only those exact Cast edges to the already implemented typed-load entry;
  // external BF16 Views remain original checked Graph boundaries.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (eligible[i] != kOrdinary) continue;
    HYLE_ASSIGN_OR_RETURN(auto gdn, graph.node_id(i));
    const auto operation = graph.operation(gdn);
    if (!operation) continue;
    const bool recurrent = operation->custom_as<compute::model_ops::RecurrentGatedDeltaNet>().has_value();
    if (!operation->get_if<compute::GatedDeltaNet>() && !recurrent) continue;
    std::array<NodeId, 3> suffix;
    std::size_t count = 0;
    for (const std::size_t port : {2U, 4U}) {
      if (recurrent && port == 4) continue; // Keep the original F32 sigmoid.
      HYLE_ASSIGN_OR_RETURN(auto wide, graph.node_input(gdn, port));
      if (wide.kind() != ValueKind::kNodeOutput || uses.uses[wide.ordinal()] != 1)
        continue;
      HYLE_ASSIGN_OR_RETURN(auto widen, graph.producer(wide));
      if (eligible[widen.ordinal()] != kOrdinary) continue;
      const auto cast_operation = graph.operation(widen);
      const auto *cast = cast_operation ? cast_operation->get_if<compute::Cast>() : nullptr;
      if (!cast || cast->output_type != compute::DataType::kFloat32) continue;
      HYLE_ASSIGN_OR_RETURN(auto source, graph.node_input(widen, 0));
      HYLE_ASSIGN_OR_RETURN(auto description, graph.value_description(source));
      const auto control = uses.control_readers[widen.ordinal()];
      if (description.data_type() != compute::DataType::kBFloat16 ||
          (port == 2 ? description.byte_strides().back() != std::int64_t(sizeof(std::uint16_t))
                     : !description.is_dense()) ||
          (control != FusionUses::kNone && control != gdn.ordinal())) continue;
      suffix[count++] = widen;
    }
    if (count == 0) continue;
    suffix[count++] = gdn;
    auto nodes = std::span(suffix).first(count);
    // Same ordinary route, rank and cost. Core proves contraction and retains
    // every escaping value/control edge; no new Graph scheduling rule exists.
    HYLE_ASSIGN_OR_RETURN(auto more, emit_load_suffix(nodes, gdn));
    if (!more) return Status::Ok();
  }

  // Keep prepared F32 weights outside the exact sample-Cast/Conv pair. Its
  // BF16 source may have real history readers; only the widened value is
  // private. Reuse the same accepted-whole-suffix reservation as GDN.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (eligible[i] != kOrdinary) continue;
    HYLE_ASSIGN_OR_RETURN(auto convolution, graph.node_id(i));
    const auto operation = graph.operation(convolution);
    if (!operation || !operation->custom_as<compute::model_ops::CausalConv1d>()) continue;
    HYLE_ASSIGN_OR_RETURN(auto wide, graph.node_input(convolution, 0));
    if (wide.kind() != ValueKind::kNodeOutput || uses.uses[wide.ordinal()] != 1) continue;
    HYLE_ASSIGN_OR_RETURN(auto widen, graph.producer(wide));
    if (eligible[widen.ordinal()] != kOrdinary) continue;
    const auto cast_operation = graph.operation(widen);
    const auto *cast = cast_operation ? cast_operation->get_if<compute::Cast>() : nullptr;
    if (!cast || cast->output_type != compute::DataType::kFloat32) continue;
    HYLE_ASSIGN_OR_RETURN(auto source, graph.node_input(widen, 0));
    HYLE_ASSIGN_OR_RETURN(auto description, graph.value_description(source));
    const auto shape = description.dimensions();
    const auto strides = description.byte_strides();
    const auto control = uses.control_readers[widen.ordinal()];
    if (description.data_type() != compute::DataType::kBFloat16 || shape.size() != 3 ||
        strides[2] != 2 || strides[1] < shape[2] * 2 ||
        strides[0] < (shape[1] - 1) * strides[1] + shape[2] * 2 ||
        (control != FusionUses::kNone && control != convolution.ordinal())) continue;
    std::array nodes{widen, convolution};
    HYLE_ASSIGN_OR_RETURN(auto more, emit_load_suffix(nodes, convolution));
    if (!more) return Status::Ok();
  }

  // Ordinary SiLU keeps its admitted leaf preference (normally SGLang).
  // This exact two-command boundary has a distinct one-dispatch native
  // implementation, so use the same specific-candidate rules as SiLU+Mul.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (!eligible[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto silu, graph.node_id(i));
    const auto operation = graph.operation(silu);
    const auto *unary = operation ? operation->get_if<compute::Unary>() : nullptr;
    if (!unary || unary->kind != compute::UnaryKind::kSilu) continue;
    HYLE_ASSIGN_OR_RETURN(auto intermediate, graph.node_input(silu, 0));
    if (intermediate.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto cast, graph.producer(intermediate));
    if (!eligible[cast.ordinal()]) continue;
    const std::array nodes{cast, silu};
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_cast_stable_silu(context, graph, nodes,
        [&](NodeId first, NodeId last, ValueId value) noexcept {
          const auto control = uses.control_readers[first.ordinal()];
          return uses.uses[value.ordinal()] == 1 &&
                 (control == FusionUses::kNone || control == last.ordinal());
        }));
    if (!matches) continue;
    const auto before = writer.size();
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
        .nodes = nodes,
        .cost = {.workspace = std::nullopt, .persistent_bytes = std::nullopt,
                 .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
                 .synchronization_count = 2, .dispatch_count = 1},
        .route = 4,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{5},
        .policy = kGgmlCudaCastStableSiluPolicy,
        .preference = kGgmlCudaCastStableSiluPreference}));
    if (writer.size() != before)
      for (const auto node : nodes) eligible[node.ordinal()] = kSupported;
    if (!more) return Status::Ok();
  }

  // Accepted suffix nodes are reserved from ordinary prefixes. This also
  // keeps later ordinary groups (including RMS) disjoint from load suffixes;
  // the original leaf alternatives remain available if a region declines.
  std::ranges::fill(ordinary_owner, std::size_t{0});
  next_owner = 1;
  HYLE_RETURN_IF_ERROR(detail::discover_regions(graph, allocator,
      [&](NodeId node) -> Result<bool> {
        return eligible[node.ordinal()] == kOrdinary;
      }, [&](std::span<const NodeId> nodes) -> Result<bool> {
        const auto before = writer.size();
        HYLE_ASSIGN_OR_RETURN(auto more, emit(nodes));
        if (writer.size() != before) {
          for (const auto node : nodes) ordinary_owner[node.ordinal()] = next_owner;
          ++next_owner;
        }
        return more;
      }));
  if (writer.proposal_limit_reached() || writer.node_limit_reached())
    return Status::Ok();

  // Ordinary discovery conservatively cuts on a newly encountered external
  // producer. Keep that cold/ready prefix and add only the three-node suffix
  // which the native RMS cast matcher already implements. This leaves an
  // immutable scale in its existing prepared region, never back in hot math.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (eligible[i] != kOrdinary) continue;
    HYLE_ASSIGN_OR_RETURN(auto narrow, graph.node_id(i));
    const auto operation = graph.operation(narrow);
    const auto *cast = operation ? operation->get_if<compute::Cast>() : nullptr;
    if (!cast || cast->output_type != compute::DataType::kBFloat16) continue;
    HYLE_ASSIGN_OR_RETURN(auto normalized, graph.node_input(narrow, 0));
    if (normalized.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto norm, graph.producer(normalized));
    const auto norm_op = graph.operation(norm);
    if (!norm_op || !norm_op->get_if<compute::RmsNorm>() ||
        eligible[norm.ordinal()] != kOrdinary) continue;
    HYLE_ASSIGN_OR_RETURN(auto wide, graph.node_input(norm, 0));
    if (wide.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto widen, graph.producer(wide));
    if (eligible[widen.ordinal()] != kOrdinary) continue;
    const std::array nodes{widen, norm, narrow};
    const auto owner = ordinary_owner[widen.ordinal()];
    if (owner != 0 && owner == ordinary_owner[norm.ordinal()] &&
        owner == ordinary_owner[narrow.ordinal()]) continue;
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_rms_casts(graph, nodes,
        [&](NodeId producer, ValueId intermediate) noexcept {
          const auto control = uses.control_readers[producer.ordinal()];
          return uses.uses[intermediate.ordinal()] == 1 &&
                 (control == FusionUses::kNone ||
                  std::ranges::any_of(nodes, [&](NodeId node) noexcept {
                    return control == node.ordinal();
                  }));
        }));
    if (!matches) continue;
    // Preserve the same admitted per-leaf preference and fixed writer budget.
    // Core still validates contraction and all boundary/control obligations.
    HYLE_ASSIGN_OR_RETURN(auto more, emit(nodes));
    if (!more) return Status::Ok();
  }

  return Status::Ok();
}

Status qualify(void *context, const RegionCompileRequest &request) noexcept {
  if (!request.graph || (request.route != 1 && request.route != 2 && request.route != 3 && request.route != 4 && request.route != 5) || request.nodes.empty() ||
      request.outputs.empty() || request.has_unknown_effects ||
      !request.effect_accesses.empty())
    return Status::Unimplemented("ggml CUDA region boundary is not admitted");

  if (request.route == 5) {
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_bfloat_paired_projection(
        context, *request.graph, request.nodes));
    if (!matches || request.outputs.size() != 2 || request.outputs[0] == request.outputs[1])
      return Status::Unimplemented("ggml CUDA BF16 paired projection boundary is not admitted");
    const auto first = request.graph->result(request.nodes[0]).value();
    const auto second = request.graph->result(request.nodes[1]).value();
    return ((request.outputs[0] == first && request.outputs[1] == second) ||
            (request.outputs[0] == second && request.outputs[1] == first))
        ? Status::Ok()
        : Status::Unimplemented("ggml CUDA BF16 paired projection output is foreign");
  }

  if (request.route == 4) {
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_cast_stable_silu(
        context, *request.graph, request.nodes,
        [&](NodeId, NodeId, ValueId intermediate) noexcept {
          // Common boundary discovery exposes published/externally read
          // Cast values; contraction retains every external control edge.
          return std::ranges::find(request.outputs, intermediate) == request.outputs.end();
        }));
    if (!matches || request.outputs.size() != 1 ||
        request.outputs.front() != request.graph->result(request.nodes.back()).value())
      return Status::Unimplemented("ggml CUDA Cast stable SiLU boundary is not admitted");
    return Status::Ok();
  }

  if (request.route == 3) {
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_bfloat_dual_projection(
        context, *request.graph, request.nodes,
        [&](NodeId, ValueId value) noexcept {
          // Common boundary discovery exposes every escaping intermediate,
          // including published metadata aliases. Contraction preserves the
          // original control edges. The one discovery-local use/control index
          // already proved this candidate's closure; do not rescan the Graph.
          return std::ranges::find(request.outputs, value) == request.outputs.end();
        }));
    // All three intermediates are private. The single escaping value is the
    // original Multiply, never a rewritten GLU or a placeholder output.
    if (!matches || request.outputs.size() != 1)
      return Status::Unimplemented("ggml CUDA BF16 dual projection boundary is not admitted");
    const auto output = request.outputs.front();
    const auto producer = request.graph->producer(output);
    if (!producer || std::ranges::find(request.nodes, *producer) == request.nodes.end())
      return Status::Unimplemented("ggml CUDA BF16 dual projection output is foreign");
    const auto operation = request.graph->operation(*producer);
    const auto *binary = operation ? operation->get_if<compute::Binary>() : nullptr;
    return binary && binary->kind == compute::BinaryKind::kMultiply ? Status::Ok()
        : Status::Unimplemented("ggml CUDA BF16 dual projection output is not Multiply");
  }

  if (request.route == 2) {
    if (request.nodes.size() == 4) {
      HYLE_ASSIGN_OR_RETURN(auto matches, matches_typed_silu_multiply(
          context, *request.graph, request.nodes,
          [&](NodeId, ValueId intermediate) noexcept {
            return std::ranges::find(request.outputs, intermediate) == request.outputs.end();
          }));
      return matches && request.outputs.size() == 1 && request.outputs.front() ==
                 request.graph->result(request.nodes.back()).value()
          ? Status::Ok()
          : Status::Unimplemented("ggml CUDA typed SiLU multiply boundary is not admitted");
    }
    HYLE_ASSIGN_OR_RETURN(auto matches, matches_silu_multiply(
        context, *request.graph, request.nodes,
        [&](NodeId, NodeId, ValueId intermediate) noexcept {
          // Common boundary discovery exposes every published/externally read
          // value. Its contraction already preserves all control dependencies.
          return std::ranges::find(request.outputs, intermediate) == request.outputs.end();
        }));
    return matches ? Status::Ok()
                   : Status::Unimplemented("ggml CUDA SiLU multiply boundary is not admitted");
  }
  if (request.nodes.size() == 1 &&
      !single_constant_preparation(*request.graph, request.nodes.front()))
    return Status::Unimplemented("ggml CUDA singleton has no immutable preparation");
  for (const auto node : request.nodes) {
    auto supported = supported_node(context, *request.graph, node);
    if (!supported)
      return std::move(supported).error();
    if (!*supported)
      return Status::Unimplemented(
          "ggml CUDA region contains an unsupported node");
    const auto operation = request.graph->operation(node).value();
    const auto *gather = operation.get_if<compute::Gather>();
    const auto elements = operation.custom_as<compute::model_ops::GatherElements>();
    const bool grouped =
        operation.custom_as<compute::model_ops::GroupedMatMul>().has_value();
    if (grouped ||
        (gather && gather->bounds == compute::IndexBoundsPolicy::kChecked) ||
        (elements && elements->bounds == compute::IndexBoundsPolicy::kChecked)) {
      const auto indices = request.graph->node_input(node, grouped ? 2 : 1).value();
      // Native validation runs before this recipe and therefore needs a real
      // input boundary. An internally generated index cannot be checked there.
      // Decline this fusion before physical placement; the ordinary indexed
      // operation retains its checked input and remains a legal fallback.
      if (std::ranges::find(request.inputs, indices) == request.inputs.end())
        return Status::Unimplemented(
            "ggml CUDA checked indices require a region input boundary");
    }
  }
  return Status::Ok();
}
Status resources(void *context, const RegionCompileRequest &request,
                 mem::AllocatorRef,
                 std::span<RegionInputRequirement> inputs,
                 std::span<RegionOutputRequirement> outputs,
                 RegionResourceProposal *proposal) noexcept {
  *proposal = {};
  HYLE_RETURN_IF_ERROR(qualify(context, request));
  std::ranges::fill(inputs, RegionInputRequirement{.resource = native::kBinding});
  std::ranges::fill(outputs, RegionOutputRequirement{.resource = native::kBinding});
  for (auto &binding : inputs)
    binding.resource.minimum_alignment = 1;
  for (auto &binding : outputs)
    binding.resource.minimum_alignment = 1;
  for (const auto node : request.nodes) {
    const auto operation = request.graph->operation(node).value();
    const auto require = [&](ValueId id) {
      const auto type = request.graph->value_description(id)->data_type();
      const auto alignment = native::boundary_alignment(operation, type);
      // Published values may also be read by another native node. Its input
      // requirement is part of the external output's binding contract too.
      for (std::size_t i = 0; i < request.inputs.size(); ++i)
        if (request.inputs[i] == id)
          inputs[i].resource.minimum_alignment =
              std::max(inputs[i].resource.minimum_alignment, alignment);
      for (std::size_t i = 0; i < request.outputs.size(); ++i)
        if (request.outputs[i] == id)
          outputs[i].resource.minimum_alignment =
              std::max(outputs[i].resource.minimum_alignment, alignment);
    };
    for (std::size_t i = 0; i < request.graph->node_input_count(node).value();
         ++i)
      require(request.graph->node_input(node, i).value());
    for (std::size_t i = 0;
         i < request.graph->node_output_count(node).value(); ++i)
      require(request.graph->result(node, i).value());
  }
  *proposal = {.runtime = native::kRuntime,
               .workspace_resource = native::kBinding};
  return Status::Ok();
}

struct Artifact final {
  compute::detail::ProviderPrepared native;
};
void destroy(void *value, mem::AllocatorRef allocator) noexcept {
  auto *artifact = static_cast<Artifact *>(value);
  if (!artifact)
    return;
  if (artifact->native.vtable)
    artifact->native.vtable->destroy_prepared(artifact->native.state, allocator);
  artifact->~Artifact();
  allocator.deallocate(artifact, sizeof(Artifact), alignof(Artifact));
}
compute::AliasSupport alias(void *, std::size_t, std::size_t) noexcept {
  return {};
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  if (!artifact.vtable) { *output = nullptr; return Status::Ok(); }
  return artifact.vtable->create_invocation(artifact.state, allocator,
                                            workspace, output);
}
// Both Direct and Graph use the same invocation callbacks; this table keeps
// only an immutable pointer to that table, never a second execution program.
struct Invocation final {
  const compute::detail::ProviderPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  std::vector<compute::ConstTensorView> input_bindings;
};
Status create_region_invocation(void *value, mem::AllocatorRef allocator,
                                mem::BufferView workspace,
                                void **output) noexcept {
  *output = nullptr;
  const auto &artifact = static_cast<Artifact *>(value)->native;
  if (!artifact.vtable) return Status::Ok();
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return Status::ResourceExhausted(
        "ggml CUDA region invocation allocation failed");
  auto *invocation = ::new (storage) Invocation{artifact.vtable, nullptr, {}};
  *output = invocation;
  return create_invocation(value, allocator, workspace, &invocation->state);
}
void destroy_region_invocation(void *value,
                               mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(value);
  if (!invocation)
    return;
  if (invocation->vtable)
    invocation->vtable->destroy_invocation(invocation->state, allocator);
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
compute::detail::EnqueueResult
submit(void *value, void *lane, compute::detail::QueueSequence &sequence,
       std::span<const compute::ConstTensorView> inputs,
       std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  if (!artifact.vtable) return compute::detail::EnqueueResult::no_op();
  return artifact.vtable->shared_submit(artifact.state,
                                        static_cast<Invocation *>(lane)->state,
                                        sequence, inputs, outputs);
}
Status prepare_bindings(void *value, void *lane,
                        std::span<const RegionValue> inputs,
                        std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  if (!artifact.vtable) return Status::Ok();
  auto &invocation = *static_cast<Invocation *>(lane);
  try {
    invocation.input_bindings.resize(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
      invocation.input_bindings[i] = inputs[i].tensor;
  } catch (...) {
    return Status::ResourceExhausted("ggml CUDA cold region bindings failed");
  }
  return artifact.vtable->prepare_bindings(
      artifact.state, invocation.state, invocation.input_bindings, outputs);
}
const detail::RegionCaptureBindingsVTable capture_bindings{
    [](void *prepared, void *lane) noexcept -> bool {
      if (!prepared || !lane) return false;
      const auto &artifact = static_cast<const Artifact *>(prepared)->native;
      const auto &invocation = *static_cast<const Invocation *>(lane);
      // This exact native program rebinds only external roots/views. Its
      // graph/workspace and patched outer-recording pools retain private
      // addresses until teardown. Commands or private configuration changes
      // must discard all recordings before entering the ordinary encoder.
      return artifact.vtable && artifact.state &&
          artifact.runtime.capture_compatible &&
          artifact.runtime.mode == compute::ProviderRuntimeMode::kSharedQueue &&
          invocation.vtable == artifact.vtable && invocation.state;
    }};
const void *binding_release_extension(void *, compute::ProviderExtensionId id) noexcept {
  if (id == detail::kRegionCaptureBindingsExtensionId)
    return &capture_bindings;
  // This shared carrier passes only borrowed TensorViews to the original
  // Compute slot; its callback ABI cannot import RegionValue lease owners or
  // generated references. Keeping these physical tuples grants no permission;
  // Graph proves every current binding/use before another replay.
  static constexpr detail::RegionBindingReleaseVTable release{
      .validate = [](void *, void *) noexcept { return Status::Ok(); },
      // TensorView metadata owns no leases or native imports. Keep its exact
      // arguments dormant so proven stable recordings can survive withdrawal.
      .release = [](void *, void *) noexcept {},
      .borrows_tensor_views_only = true};
  return id == detail::kRegionBindingReleaseExtensionId ? &release : nullptr;
}
const ProviderRegionPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit,
    .prepare_bindings = &prepare_bindings,
    .query_extension = &binding_release_extension};

compute::ProviderPrepareOutcome
compile(void *context, const RegionCompileRequest &request,
        mem::AllocatorRef allocator, ProviderRegionPrepared *output) noexcept {
  *output = {};
  auto status = qualify(context, request);
  if (!status.ok())
    return status.IsUnimplemented()
        ? compute::ProviderPrepareOutcome::unsupported(std::move(status))
        : compute::ProviderPrepareOutcome::error(std::move(status));
  try {
    native::Recipe recipe;
    std::vector<ValueId> values(request.inputs.begin(), request.inputs.end());
    for (std::uint32_t i = 0; i < values.size(); ++i)
      recipe.inputs.push_back(i);
    for (const auto node : request.nodes)
      for (std::size_t port = 0;
           port < request.graph->node_output_count(node).value(); ++port)
        values.push_back(request.graph->result(node, port).value());
    const auto local = [&](ValueId value) -> std::uint32_t {
      return static_cast<std::uint32_t>(std::ranges::find(values, value) -
                                        values.begin());
    };
    for (const auto id : values)
      recipe.values.push_back(request.graph->value_description(id).value());
    for (const auto node : request.nodes) {
      native::Node lowered{request.graph->operation(node).value(), {}, {}};
      for (std::size_t port = 0;
           port < request.graph->node_output_count(node).value(); ++port)
        lowered.outputs.push_back(
            local(request.graph->result(node, port).value()));
      for (std::size_t i = 0; i < request.graph->node_input_count(node).value();
           ++i) {
        const auto id = local(request.graph->node_input(node, i).value());
        if (id >= values.size())
          return compute::ProviderPrepareOutcome::error(Status::InvalidArgument(
              "ggml CUDA region misses an input boundary"));
        lowered.inputs.push_back(id);
      }
      recipe.nodes.push_back(std::move(lowered));
    }
    for (const auto id : request.outputs)
      recipe.outputs.push_back(local(id));
    if (request.route != 3 && request.route != 5) {
      const auto constants = constant_expressions::prepare(context, request, values, recipe);
      if (constants.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
        return constants;
      const auto packing = ggml_shared::pack_constants(context, request, values, recipe,
          nullptr, &native::pack_matrix_constant);
      if (packing.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
        return packing;
    }
    // Routes3/5 import the original exact BF16 x/weight boundaries. Their dense
    // transposed RHS needs no cold pack/CONT and no duplicate weight bank.
    std::size_t checked_copy_bytes = 0;
    std::uint32_t dispatches = 1;
    const auto add_checked_copy = [&](const compute::TensorDesc &description) {
      const auto bytes = description.required_bytes().value();
      if (bytes > SIZE_MAX - checked_copy_bytes ||
          dispatches == std::numeric_limits<std::uint32_t>::max())
        return false;
      checked_copy_bytes += bytes;
      if (bytes != 0) ++dispatches;
      return true;
    };
    bool checked = false;
    for (const auto &node : recipe.nodes) {
      const auto *gather = node.operation.get_if<compute::Gather>();
      const auto elements = node.operation.custom_as<compute::model_ops::GatherElements>();
      const bool grouped = node.operation.custom_as<compute::model_ops::GroupedMatMul>().has_value();
      if (!grouped && !(gather && gather->bounds == compute::IndexBoundsPolicy::kChecked) &&
          !(elements && elements->bounds == compute::IndexBoundsPolicy::kChecked))
        continue;
      checked = true;
      if (!add_checked_copy(recipe.values[node.inputs[grouped ? 2 : 1]]))
        return compute::ProviderPrepareOutcome::error(
            Status::OutOfRange("ggml CUDA checked copy cost overflows"));
    }
    if (checked)
      for (const auto id : recipe.outputs)
        if (!add_checked_copy(recipe.values[id]))
          return compute::ProviderPrepareOutcome::error(
              Status::OutOfRange("ggml CUDA checked output cost overflows"));
    void *storage = allocator.allocate(sizeof(Artifact), alignof(Artifact));
    if (!storage)
      return compute::ProviderPrepareOutcome::error(
          Status::ResourceExhausted(
              "ggml CUDA region artifact allocation failed"));
    auto *artifact = ::new (storage) Artifact;
    if (recipe.nodes.empty()) {
      // All original outputs have ready artifact bindings. Keep a metadata
      // region for the logical producer/control edges, with no native program,
      // workspace, invocation work or placeholder Queue command.
      *output = {.vtable = &kPrepared, .state = artifact,
          .properties = {.workspace = {}, .persistent_bytes = sizeof(Artifact),
                         .copy_bytes = 0,
                         .synchronization_count = 0, .dispatch_count = 0,
                         .runtime = native::kRuntime, .invocation_private_bytes = 0}};
      return compute::ProviderPrepareOutcome::prepared();
    }
    const auto outcome = native::compile(context, std::move(recipe), allocator,
                                         &artifact->native);
    if (!artifact->native.state) {
      artifact->~Artifact();
      allocator.deallocate(artifact, sizeof(Artifact), alignof(Artifact));
      return outcome;
    }
    *output = {.vtable = &kPrepared,
               .state = artifact,
               .properties = {.workspace = artifact->native.workspace,
                              .persistent_bytes = std::nullopt,
                              // Indexed consumers use private safe IDs and
                              // publish all outputs only after device checks.
                              // Native auxiliary pools/materialization remain unknown.
                              .copy_bytes = checked_copy_bytes,
                              .synchronization_count = 2,
                              .dispatch_count = dispatches,
                              .runtime = artifact->native.runtime,
                              .invocation_private_bytes = std::nullopt}};
    return outcome;
  } catch (const std::bad_alloc &) {
    return compute::ProviderPrepareOutcome::error(
        Status::ResourceExhausted("ggml CUDA region recipe allocation failed"));
  } catch (const std::exception &error) {
    return compute::ProviderPrepareOutcome::error(
        Status::Internal(error.what()));
  }
}
const RegionProviderVTable kRegion{
    .propose = &propose, .propose_resources = &resources, .compile = &compile};
} // namespace

compute::Provider provider() noexcept {
  return native::with_extension(kRegionProviderExtensionId, &kRegion);
}
} // namespace hyle::graph::ggml_cuda
