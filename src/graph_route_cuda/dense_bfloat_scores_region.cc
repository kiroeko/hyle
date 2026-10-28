#include "dense_bfloat_scores_region.h"
#include "ct_packed_linear_region.h"
#include "../graph/mapped_native_region.h"

#include "hyle/compute/providers/cuda.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/region_provider.h"
#include "../compute/provider_internal.h"
#include "../compute_provider_cuda/dense_bfloat_scores.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace hyle::graph::cuda::internal {
namespace {
namespace c = compute;
namespace native = compute::cuda_native::internal;
using c::detail::ProviderAccess;
inline constexpr c::RoutePolicyId kPolicy{0x48594c454e445343ULL, 1, 1};
inline constexpr std::size_t kMaxNodes = 7;
// The existing packed recipes use private region routes 1 and 2.
inline constexpr std::uint64_t kDenseScoreRoute = 3;

const c::detail::ProviderVTable &original() noexcept {
  // The factory has one fixed implementation. Retain the original table, not
  // a configured wrapper or a borrowed Provider object with a different state.
  static const auto *table = ProviderAccess::vtable(c::cuda_native::provider());
  return *table;
}

template <class T>
std::optional<T> operation(const Graph &graph, NodeId node) noexcept {
  const auto value = graph.operation(node);
  if (!value) return std::nullopt;
  const auto *typed = value->get_if<T>();
  return typed ? std::optional<T>{*typed} : std::nullopt;
}
NodeId producer(const Graph &graph, ValueId value) noexcept {
  if (value.kind() != ValueKind::kNodeOutput) return {};
  const auto result = graph.producer(value);
  return result ? *result : NodeId{};
}
bool arity(const Graph &graph, NodeId node, std::size_t inputs) noexcept {
  if (!node.valid()) return false;
  const auto in = graph.node_input_count(node), out = graph.node_output_count(node),
             resources = graph.node_resource_count(node);
  return in && out && resources && *in == inputs && *out == 1 && *resources == 0;
}
bool dimensions(const c::TensorDesc &description,
                std::span<const std::int64_t> shape) noexcept {
  return std::ranges::equal(description.dimensions(), shape);
}
bool typed_dense(const Graph &graph, ValueId value, c::DataType type,
                 std::span<const std::int64_t> shape) noexcept {
  const auto desc = graph.value_description(value);
  return desc && desc->is_plain() && desc->is_dense() &&
      desc->data_type() == type && dimensions(*desc, shape);
}

// One allocator-backed cold index. It retains no Graph or lease and is never
// consulted during submit. A published value is an additional data observer.
struct Uses final {
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> data;
  container::SmallVector<NodeId, 16, mem::AllocatorRef> reader;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> control;
  explicit Uses(mem::AllocatorRef allocator) noexcept
      : data(allocator), reader(allocator), control(allocator) {}
  Status initialize(const Graph &graph) noexcept {
    if (!data.try_resize(graph.value_count()) ||
        !reader.try_resize(graph.value_count()) ||
        !control.try_resize(graph.node_count()))
      return Status::ResourceExhausted("dense score discovery metadata allocation failed");
    std::ranges::fill(data, std::uint8_t{0});
    std::ranges::fill(reader, NodeId{});
    std::ranges::fill(control, std::size_t{0});
    for (std::size_t i = 0; i < graph.node_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
      HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
      for (std::size_t port = 0; port < inputs; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, port));
        auto &count = data[value.ordinal()];
        if (count == 0) reader[value.ordinal()] = node;
        if (count < 2) ++count;
      }
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(node));
      for (std::size_t n = 0; n < dependencies; ++n) {
        HYLE_ASSIGN_OR_RETURN(auto before, graph.node_dependency(node, n));
        if (control[before.ordinal()] == SIZE_MAX)
          return Status::OutOfRange("dense score control reader count overflows");
        ++control[before.ordinal()];
      }
    }
    for (std::size_t n = 0; n < graph.output_count(); ++n) {
      HYLE_ASSIGN_OR_RETURN(auto value, graph.output_id(n));
      data[value.ordinal()] = 2;
    }
    return Status::Ok();
  }
};

struct Pattern final {
  std::array<NodeId, kMaxNodes> nodes{};
  std::size_t size = 0;
  std::array<ValueId, 4> inputs{}; // Actual Q, Kt, Wr, optional F16 mask.
  std::size_t input_count = 3;
  ValueId output{};
  bool masked = false;
  c::DataType output_type = c::DataType::kBFloat16;
  std::span<const NodeId> span() const noexcept { return {nodes.data(), size}; }
  bool contains(NodeId node) const noexcept {
    return std::ranges::find(span(), node) != span().end();
  }
  void append(NodeId node) noexcept { nodes[size++] = node; }
  c::Operation composite() const noexcept {
    return c::Operation::custom(native::DenseBfloatScores{masked, output_type});
  }
};

bool core_pattern(const Graph &graph, NodeId reduction, Pattern &p) noexcept {
  const auto reduce = operation<c::Reduce>(graph, reduction);
  if (!reduce || !arity(graph, reduction, 1) ||
      reduce->kind != c::ReductionKind::kSum || reduce->keep_dimensions ||
      reduce->accumulation_type != c::DataType::kFloat32 ||
      reduce->axes.is_all() || reduce->axes.values().size() != 1 ||
      (reduce->axes.values()[0] != 2 && reduce->axes.values()[0] != -2)) return false;
  const auto weighted = graph.node_input(reduction, 0).value();
  const auto multiply = producer(graph, weighted);
  const auto mul = operation<c::Binary>(graph, multiply);
  if (!mul || mul->kind != c::BinaryKind::kMultiply || !arity(graph, multiply, 2)) return false;
  ValueId relu_value = graph.node_input(multiply, 0).value();
  ValueId weight_view = graph.node_input(multiply, 1).value();
  auto relu = producer(graph, relu_value);
  auto unary = operation<c::Unary>(graph, relu);
  if (!unary || unary->kind != c::UnaryKind::kRelu) {
    std::swap(relu_value, weight_view); relu = producer(graph, relu_value);
    unary = operation<c::Unary>(graph, relu);
  }
  if (!unary || unary->kind != c::UnaryKind::kRelu || !arity(graph, relu, 1)) return false;
  const auto dots = graph.node_input(relu, 0).value();
  const auto matrix = producer(graph, dots);
  const auto mm = operation<c::MatMul>(graph, matrix);
  if (!mm || mm->accumulation_type != c::DataType::kFloat32 || !arity(graph, matrix, 2)) return false;
  const auto q = graph.node_input(matrix, 0).value();
  const auto transposed = graph.node_input(matrix, 1).value();
  // Metadata stays in the original Graph: its zero-dispatch Views may be
  // published or have other readers. Qualify these actual boundary layouts
  // through the same native descriptor instead of reconstructing their roots.
  const auto q_desc = graph.value_description(q),
             key_desc = graph.value_description(transposed),
             weight_desc = graph.value_description(weight_view);
  if (!q_desc || !key_desc || !weight_desc || q_desc->rank() != 4 ||
      key_desc->rank() != 4 || weight_desc->rank() != 4) return false;
  const auto shape = q_desc->dimensions();
  const auto b = shape[0], t = shape[1], h = shape[2], d = shape[3],
             l = key_desc->dimensions()[3];
  const std::array<std::int64_t,4> score_shape{b,t,h,l}, key_shape{b,1,d,l},
      weight_shape{b,t,h,1};
  const std::array<std::int64_t,3> output_shape{b,t,l};
  if (!dimensions(*key_desc,key_shape) || !dimensions(*weight_desc,weight_shape)) return false;
  for (const auto value : {dots, relu_value, weighted})
    if (!typed_dense(graph, value, c::DataType::kBFloat16, score_shape)) return false;
  p.output = graph.result(reduction).value();
  if (!typed_dense(graph, p.output, c::DataType::kBFloat16, output_shape)) return false;
  p.inputs = {q,transposed,weight_view,{}};
  p.append(matrix); p.append(relu); p.append(multiply); p.append(reduction);
  return true;
}

bool closed(const Graph &graph, const Pattern &p, const Uses &uses,
            std::span<const std::uint8_t> allowed) noexcept {
  const auto terminal = producer(graph, p.output);
  for (const auto node : p.span()) {
    if (!allowed.empty() && !allowed[node.ordinal()]) return false;
    if (!arity(graph, node, graph.node_input_count(node).value())) return false;
    if (node == terminal) continue;
    const auto value = graph.result(node).value();
    if (uses.data[value.ordinal()] != 1 || !p.contains(uses.reader[value.ordinal()])) return false;
    std::size_t internal_controls = 0;
    for (const auto after : p.span())
      for (std::size_t n = 0; n < graph.node_dependency_count(after).value(); ++n)
        internal_controls += graph.node_dependency(after,n).value() == node;
    if (uses.control[node.ordinal()] != internal_controls) return false;
  }
  return true;
}
bool append_mask(const Graph &graph, const Uses &uses, Pattern &p) noexcept {
  if (uses.data[p.output.ordinal()] != 1) return false;
  const auto add_node = uses.reader[p.output.ordinal()];
  const auto add = operation<c::Binary>(graph, add_node);
  if (!add || add->kind != c::BinaryKind::kAdd || !arity(graph,add_node,2)) return false;
  const auto a = graph.node_input(add_node,0).value(), b = graph.node_input(add_node,1).value();
  if (a != p.output && b != p.output) return false;
  const auto low_mask = a == p.output ? b : a;
  const auto cast_node = producer(graph,low_mask);
  const auto cast = operation<c::Cast>(graph,cast_node);
  if (!cast || cast->output_type != c::DataType::kBFloat16 || !arity(graph,cast_node,1)) return false;
  const auto mask = graph.node_input(cast_node,0).value();
  const auto desc = graph.value_description(mask);
  if (!desc || desc->data_type() != c::DataType::kFloat16) return false;
  p.masked = true; p.input_count = 4; p.inputs[3] = mask;
  p.append(cast_node); p.append(add_node); p.output = graph.result(add_node).value();
  return true;
}
bool append_widen(const Graph &graph, const Uses &uses, Pattern &p) noexcept {
  if (uses.data[p.output.ordinal()] != 1) return false;
  const auto node = uses.reader[p.output.ordinal()];
  const auto cast = operation<c::Cast>(graph,node);
  if (!cast || cast->output_type != c::DataType::kFloat32 || !arity(graph,node,1)) return false;
  p.output_type = c::DataType::kFloat32; p.append(node); p.output = graph.result(node).value();
  return true;
}

Status native_proposal(void *context, const Graph &graph, const Pattern &p,
    std::array<c::TensorDesc,4> &inputs, c::TensorDesc &output,
    std::array<c::detail::ResourceBindingRequirement,4> &input_resources,
    c::detail::ResourceBindingRequirement &output_resource,
    c::detail::ProviderOperationProposal &proposal,
    const c::WorkloadContext &workload = {}) noexcept {
  for (std::size_t n = 0; n < p.input_count; ++n)
    inputs[n] = graph.value_description(p.inputs[n]).value();
  output = graph.value_description(p.output).value();
  std::array<c::TensorDesc,1> inferred;
  const auto op = p.composite();
  HYLE_RETURN_IF_ERROR(c::infer_output_descriptions(
      op,std::span(inputs).first(p.input_count),inferred));
  if (inferred[0] != output)
    return Status::Unimplemented("dense score composite output differs from the public graph");
  HYLE_RETURN_IF_ERROR(original().propose(context,
      {.operation=op,.inputs=std::span(inputs).first(p.input_count),
       .outputs=std::span(&output,1),.workload=workload},
      std::span(input_resources).first(p.input_count),std::span(&output_resource,1),&proposal));
  if (proposal.cost.persistent_bytes) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, graph::detail::mapped_native_region::artifact_bytes(p.input_count, 1));
    if (*proposal.cost.persistent_bytes > SIZE_MAX - bytes)
      return Status::OutOfRange("dense score region metadata bytes overflow");
    *proposal.cost.persistent_bytes += bytes;
  }
  return Status::Ok();
}

Status propose(void *context, const RegionDiscoveryRequest &request,
    mem::AllocatorRef allocator, RegionProposalWriter &writer) noexcept {
  HYLE_RETURN_IF_ERROR(ct_packed_linear_regions().propose(context, request, allocator, writer));
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size()!=graph.node_count())
    return Status::InvalidArgument("dense score allowed-node count differs");
  Uses uses(allocator); HYLE_RETURN_IF_ERROR(uses.initialize(graph));
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));
    Pattern core;
    if (!core_pattern(graph,node,core)) continue;
    const auto emit = [&](const Pattern &p) -> Result<bool> {
      if (!closed(graph,p,uses,request.allowed_nodes)) return true;
      std::array<c::TensorDesc,4> inputs;
      c::TensorDesc output;
      std::array<c::detail::ResourceBindingRequirement,4> input_resources;
      c::detail::ResourceBindingRequirement output_resource;
      c::detail::ProviderOperationProposal proposal;
      const auto status = native_proposal(context,graph,p,inputs,output,
          input_resources,output_resource,proposal);
      if (!status.ok()) {
        if (status.IsUnimplemented() || status.IsInvalidArgument()) return true;
        return std::unexpected(status);
      }
      return writer.try_add({.nodes=p.span(),.cost=proposal.cost,.route=kDenseScoreRoute,
          .replay_class=proposal.runtime.replay_class,
          .implementation=c::ProviderImplementationId{1},.policy=kPolicy,.preference=200});
    };
    HYLE_ASSIGN_OR_RETURN(auto more,emit(core)); if (!more) return Status::Ok();
    auto widened = core;
    if (append_widen(graph,uses,widened)) {
      HYLE_ASSIGN_OR_RETURN(more,emit(widened)); if (!more) return Status::Ok();
    }
    auto masked = core;
    if (append_mask(graph,uses,masked)) {
      HYLE_ASSIGN_OR_RETURN(more,emit(masked)); if (!more) return Status::Ok();
      if (append_widen(graph,uses,masked)) {
        HYLE_ASSIGN_OR_RETURN(more,emit(masked)); if (!more) return Status::Ok();
      }
    }
  }
  return Status::Ok();
}

// Reconstruct only the chosen bounded recipe. No Graph-wide use scan is
// repeated here: Graph's exact boundary exposes every escaping intermediate,
// and contraction has already validated original data/control ordering.
Status qualify(const RegionCompileRequest &request, Pattern &result) noexcept {
  if (!request.graph || request.route != kDenseScoreRoute || request.outputs.size()!=1 ||
      request.nodes.size()<4 || request.nodes.size()>kMaxNodes ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return Status::Unimplemented("dense score region boundary differs");
  const auto &graph = *request.graph;
  for (const auto node : request.nodes) {
    Pattern p;
    if (!core_pattern(graph,node,p)) continue;
    // Locate optional suffixes only among this exact bounded node set.
    for (std::size_t n = 0; n < 2 && p.output!=request.outputs[0]; ++n) {
      bool appended = false;
      for (const auto next : request.nodes) {
        const auto cast = operation<c::Cast>(graph,next);
        if (cast && cast->output_type==c::DataType::kFloat32 && arity(graph,next,1) &&
            graph.node_input(next,0).value()==p.output) {
          p.append(next);p.output=graph.result(next).value();p.output_type=c::DataType::kFloat32;
          appended=true;break;
        }
        const auto add = operation<c::Binary>(graph,next);
        if (!add || add->kind!=c::BinaryKind::kAdd || !arity(graph,next,2) || p.masked) continue;
        const auto a=graph.node_input(next,0).value(),b=graph.node_input(next,1).value();
        if (a!=p.output && b!=p.output) continue;
        const auto low_mask=a==p.output?b:a;
        const auto mask_node=producer(graph,low_mask);
        const auto narrow=operation<c::Cast>(graph,mask_node);
        if (!narrow || narrow->output_type!=c::DataType::kBFloat16 || !arity(graph,mask_node,1) ||
            std::ranges::find(request.nodes,mask_node)==request.nodes.end()) continue;
        const auto mask=graph.node_input(mask_node,0).value();
        if (graph.value_description(mask)->data_type()!=c::DataType::kFloat16) continue;
        p.masked=true;p.input_count=4;p.inputs[3]=mask;p.append(mask_node);p.append(next);
        p.output=graph.result(next).value();appended=true;break;
      }
      if (!appended) break;
    }
    if (p.output!=request.outputs[0] || p.size!=request.nodes.size()) continue;
    if (!std::ranges::all_of(p.span(),[&](NodeId id) {
          return std::ranges::count(request.nodes,id)==1;
        }) || request.inputs.size()!=p.input_count) continue;
    if (!std::ranges::all_of(std::span(p.inputs).first(p.input_count),[&](ValueId id) {
          return std::ranges::count(request.inputs,id)==1;
        })) continue;
    result=p;return Status::Ok();
  }
  return Status::Unimplemented("dense score region does not preserve the exact public recipe");
}

Status resources(void *context, const RegionCompileRequest &request, mem::AllocatorRef allocator,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *result) noexcept {
  if (request.route != kDenseScoreRoute)
    return ct_packed_linear_regions().propose_resources(context, request, allocator, inputs, outputs, result);
  *result={}; Pattern p; HYLE_RETURN_IF_ERROR(qualify(request,p));
  if (inputs.size()!=p.input_count || outputs.size()!=1)
    return Status::InvalidArgument("dense score resource count differs");
  std::array<c::TensorDesc,4> descriptions;
  c::TensorDesc output;
  std::array<c::detail::ResourceBindingRequirement,4> requirements;
  c::detail::ResourceBindingRequirement destination;
  c::detail::ProviderOperationProposal proposal;
  HYLE_RETURN_IF_ERROR(native_proposal(context,*request.graph,p,descriptions,output,
      requirements,destination,proposal,request.workload));
  for (std::size_t n=0;n<p.input_count;++n) {
    const auto port=std::ranges::find(request.inputs,p.inputs[n])-request.inputs.begin();
    inputs[std::size_t(port)]={.resource=requirements[n]};
  }
  outputs[0]={.resource=destination};
  *result={.runtime=proposal.runtime,.workspace_resource=proposal.workspace_resource,
      .compilation_storage=proposal.cost.compilation_storage};
  return Status::Ok();
}

c::ProviderPrepareOutcome compile(void *context,const RegionCompileRequest &request,
    mem::AllocatorRef allocator,ProviderRegionPrepared *result) noexcept {
  if (request.route != kDenseScoreRoute)
    return ct_packed_linear_regions().compile(context, request, allocator, result);
  *result={};Pattern p;
  const auto qualification=qualify(request,p);
  if (!qualification.ok()) return c::ProviderPrepareOutcome::unsupported(qualification);
  if (request.input_resources.size()!=p.input_count || request.output_resources.size()!=1)
    return c::ProviderPrepareOutcome::error(
        Status::InvalidArgument("dense score physical resource count differs"));
  std::array<c::TensorDesc,4> inputs;
  c::TensorDesc output;
  std::array<c::detail::ResourceBindingRequirement,4> requirements;
  c::detail::ResourceBindingRequirement destination;
  c::detail::ProviderOperationProposal proposal;
  const auto status=native_proposal(context,*request.graph,p,inputs,output,
      requirements,destination,proposal,request.workload);
  if (!status.ok()) return status.IsUnimplemented()
      ? c::ProviderPrepareOutcome::unsupported(status) : c::ProviderPrepareOutcome::error(status);
  std::array<std::size_t,4> ports;
  std::array<mem::MemoryResourceRef,4> resources;
  for (std::size_t n=0;n<p.input_count;++n) {
    ports[n]=std::size_t(std::ranges::find(request.inputs,p.inputs[n])-request.inputs.begin());
    resources[n]=request.input_resources[ports[n]];
  }
  return graph::detail::mapped_native_region::prepare(original(),context,
      {.operation=p.composite(),.inputs=std::span(inputs).first(p.input_count),
       .outputs=std::span(&output,1),.input_resources=std::span(resources).first(p.input_count),
       .output_resources=request.output_resources,.workspace_resource=request.workspace_resource,
       .route=proposal.route,.workload=request.workload},proposal,
      std::span(ports).first(p.input_count),request.inputs.size(),allocator,result);
}
const RegionProviderVTable kRegion{.propose=&propose,.propose_resources=&resources,.compile=&compile};
const void *query_extension(void *context,c::ProviderExtensionId id) noexcept {
  if (id==kRegionProviderExtensionId) return &kRegion;
  return original().query_extension ? original().query_extension(context,id) : nullptr;
}
const c::detail::ProviderVTable &decorated() noexcept {
  static const auto table=[] {
    auto value=original();value.query_extension=&query_extension;return value;
  }();
  return table;
}
} // namespace

compute::Provider dense_bfloat_scores_provider() noexcept {
  return ProviderAccess::with_vtable(c::cuda_native::provider(),&decorated());
}
} // namespace hyle::graph::cuda::internal
