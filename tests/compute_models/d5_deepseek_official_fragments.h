#pragma once

#include "d5_quantization_roundtrip.h"
#include "dspark_stored_projection_fragment.h"

// Official DeepSeek V4 consumer recipes. Host arithmetic below is only the
// independent oracle; every executed node calls an existing mature Provider.
namespace d5_deepseek_official_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace moe = d5_moe_execution_fragments;
namespace core = d5_moe_semantic_fragments;
namespace fp8 = d5_dynamic_fp8_fragments;
namespace quant = d5_quantization_roundtrip;
using sem::ValueId;
using sem::Fragment;

inline ValueId bfloat_boundary(Fragment &f, ValueId value) {
  auto stored = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
  return fp8::visible(f, stored, true);
}
inline std::vector<float> bfloat_boundary(std::vector<float> values) {
  for (auto &x : values) x = fp8::bfloat(x);
  return values;
}
inline ValueId roundtrip(Fragment &f, ValueId value, quant::Format format) {
  value = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
  return fp8::visible(f, f.one(c::model_ops::block_quantization_roundtrip(format), {value}), true);
}

// Original GGUF Q8_0 or explicitly stored BF16 projections are distinct from
// the official block128 FP8 construction below. The callback is independent
// reference math only; computation reads the supplied immutable Graph value.
enum class SharedFfnRole : std::uint8_t { kGate, kUp, kDown };
using PackedProjectionReference=std::function<std::vector<float>(std::span<const float>)>;
struct PackedProjectionParameter final {
  ValueId value;
  std::span<const std::byte> codes;
  PackedProjectionReference reference;
  // Absent preserves the original Q8_0 descriptor and node sequence. A plain
  // BF16 descriptor explicitly selects the BF16-activation projection policy.
  std::optional<c::TensorDesc> storage{};
};
struct SharedFfnParameters final { std::array<PackedProjectionParameter,3> weights; };
// Original global routing stays256-wide. Only physical expert residency is
// compact; the explicit map is supplied with the same immutable upload.
struct RoutedExpertParameters final {
  ValueId router,bias;
  std::array<ValueId,3> weights;
  ValueId map;
  std::int64_t capacity=30;
};
struct RoutedBoundary final {
  ValueId input,ids,weights;
  std::array<ValueId,3> banks;
  ValueId map;
};
struct RoutedExpertPorts final {
  ValueId input,logits,softplus,scores,selection,selected_scores,sum,normalized,ids,weights,map,local_ids;
  ValueId gate_raw,gate,up_raw,up,gate_clipped,up_clipped,activated;
  ValueId weighted_raw,weighted,down_raw,down,combined;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&logits,&softplus,&scores,&selection,&selected_scores,&sum,&normalized,&ids,&weights,&map,&local_ids,
        &gate_raw,&gate,&up_raw,&up,&gate_clipped,&up_clipped,&activated,
        &weighted_raw,&weighted,&down_raw,&down,&combined})visitor(*value);
  }
};
inline c::TensorDesc expert_weight_description(const moe::Case &model,std::size_t role,std::int64_t capacity) {
  if(role>=3 || capacity<=0 || capacity>model.experts)throw std::runtime_error("compact expert geometry");
  return c::gguf_grouped_weight_description(c::GgufQuantizedType::kMxfp4,c::DataType::kFloat32,capacity,
      role==2?model.hidden:model.intermediate,role==2?model.intermediate:model.hidden).value();
}
inline void require_routed_parameters(const Fragment &f,const moe::Case &model,const RoutedExpertParameters &p) {
  const auto require=[&](ValueId value,const c::TensorDesc &desc,bool constant) {
    const auto &set=constant?f.constants:f.inputs;
    if(value.ordinal()>=f.descriptions.size() || f.description(value)!=desc ||
        std::ranges::none_of(set,[&](ValueId member){return member.ordinal()==value.ordinal();}))
      throw std::runtime_error("original routed parameter has wrong description or binding role");
  };
  require(p.router,sem::dense({model.experts,model.hidden},c::DataType::kBFloat16),true);
  require(p.bias,sem::dense({model.experts}),true);
  for(std::size_t i=0;i<3;++i)require(p.weights[i],expert_weight_description(model,i,p.capacity),false);
  require(p.map,sem::dense({model.experts},c::DataType::kInt32),false);
}
inline c::TensorDesc projection_description(const c::TensorDesc &packed,
    const PackedProjectionParameter &parameter) {
  if(!parameter.storage)return packed;
  const auto &actual=*parameter.storage;
  if(actual==packed)return actual;
  c::TensorDesc bfloat;
  const auto shape=packed.dimensions();
  if(packed.rank()==2)bfloat=sem::dense({shape[1],shape[0]},c::DataType::kBFloat16);
  else if(packed.rank()==3)bfloat=sem::dense({shape[0],shape[2],shape[1]},c::DataType::kBFloat16);
  else throw std::runtime_error("original projection rank differs");
  if(actual!=bfloat)throw std::runtime_error("original projection storage does not match its typed geometry");
  return actual;
}
inline void require_packed_parameter(const Fragment &f,const c::TensorDesc &expected,
                                     const PackedProjectionParameter &parameter) {
  const auto actual=projection_description(expected,parameter);
  if(parameter.value.ordinal()>=f.descriptions.size() || f.description(parameter.value)!=actual ||
      parameter.codes.size()!=actual.required_bytes().value() || !parameter.reference ||
      std::ranges::none_of(f.constants,[&](ValueId value){return value.ordinal()==parameter.value.ordinal();}))
    throw std::runtime_error("original projection requires its exact same-Fragment immutable bank and reference");
}
inline c::TensorDesc shared_weight_description(const moe::Case &model,SharedFfnRole role) {
  const bool down=role==SharedFfnRole::kDown;
  return c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,
      down?model.shared_intermediate:model.hidden,down?model.hidden:model.shared_intermediate).value();
}
inline void require_shared_parameter(const Fragment &f,const moe::Case &model,
                                      SharedFfnRole role,const PackedProjectionParameter &parameter) {
  require_packed_parameter(f,shared_weight_description(model,role),parameter);
}
struct SharedFfnPorts final {
  ValueId input,gate_raw,gate_stored,up_raw,up_stored,gate_clipped,up_clipped;
  ValueId activated_raw,activated_stored,down_raw,down_stored,routed,combined,full_raw,full_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&gate_raw,&gate_stored,&up_raw,&up_stored,&gate_clipped,&up_clipped,
        &activated_raw,&activated_stored,&down_raw,&down_stored,&routed,&combined,&full_raw,&full_stored}) visitor(*value);
  }
};
inline SharedFfnPorts shared_ffn_ports(std::span<const ValueId> outputs) {
  if(outputs.size()<20)throw std::runtime_error("shared Q8_0 diagnostic output count differs");
  return {outputs[12],outputs[13],outputs[9],outputs[14],outputs[10],outputs[15],outputs[16],
      outputs[11],outputs[17],outputs[18],outputs[4],outputs[2],outputs[3],outputs[19],outputs[5]};
}

// Full expert selection, original A8 boundary, original FP4 gate/up/down and
// shared block128 FP8. The route weight belongs before down's quantization.
inline Fragment moe_chain(const moe::Case &model, bool use_hash = false,
                           moe::MoeReference *reference_out = nullptr,
                           bool native_routing = true,std::int64_t tokens=1,
                           const SharedFfnParameters *shared_parameters=nullptr,
                           const RoutedExpertParameters *routed_parameters=nullptr) {
  const bool pro=model.hidden==7168 && model.experts==384 && model.intermediate==3072;
  const bool flash=model.hidden==4096 && model.experts==256 && model.intermediate==2048;
  if (model.gguf || !model.profile || model.profile->model_family!="deepseek_v4" ||
      model.profile->format_family!="safetensors" || (!pro && !flash) || model.top_k!=6 ||
      (use_hash && reference_out) || tokens<=0 || (tokens>1 && !reference_out) ||
      (shared_parameters && !reference_out) ||
      (routed_parameters && (!reference_out || !shared_parameters || use_hash)))
    throw std::runtime_error("official DeepSeek MoE profile/replay mismatch");
  Fragment f{.profile = model.profile, .region = use_hash ? 0x0a0b0602u : 0x0a0b0601u,
              .quantized = true};
  const auto input = f.input(sem::dense({tokens, model.hidden}));
  auto x = bfloat_boundary(f, input);
  std::optional<moe::HashInputs> lookup;
  if (use_hash)
    lookup = moe::HashInputs{
        f.input(sem::dense({1}, native_routing ? c::DataType::kInt64
                                              : c::DataType::kInt32)),
        f.input(sem::dense({moe::kHashVocabulary, 6}, c::DataType::kInt32))};
  // The parent owns the readonly banks. Only these three typed child inputs
  // are remapped during ordinary attachment; parent ValueIds never leak into
  // a separate Fragment's external-constant directory.
  std::array<ValueId,3> shared_weights;
  if(shared_parameters) for(std::size_t i=0;i<shared_weights.size();++i) {
    const auto &parameter=shared_parameters->weights[i];
    const auto desc=projection_description(shared_weight_description(model,static_cast<SharedFfnRole>(i)),parameter);
    if(parameter.codes.size()!=desc.required_bytes().value() || !parameter.reference)
      throw std::runtime_error("shared Q8_0 child requires all three complete original banks");
    shared_weights[i]=f.input(desc);
  }
  moe::Routing route;
  std::optional<RoutedExpertPorts> routed_ports;
  std::array<ValueId,3> compact_weights;
  ValueId expert_map{},local_ids{};
  if(routed_parameters) {
    routed_ports.emplace();routed_ports->input=x;
    const auto router=f.input(sem::dense({model.experts,model.hidden},c::DataType::kBFloat16));
    const auto bias=f.input(sem::dense({model.experts}));
    for(std::size_t i=0;i<3;++i)compact_weights[i]=f.input(expert_weight_description(model,i,routed_parameters->capacity));
    expert_map=f.input(sem::dense({model.experts},c::DataType::kInt32));
    const auto stored_input=f.one(c::Operation(c::Cast{c::DataType::kBFloat16}),{x});
    routed_ports->logits=f.one(c::model_ops::bfloat_linear_f32(),{stored_input,router});
    routed_ports->softplus=core::unary(f,c::UnaryKind::kSoftplus,routed_ports->logits);
    routed_ports->scores=core::unary(f,c::UnaryKind::kSqrt,routed_ports->softplus);
    routed_ports->selection=core::binary(f,c::BinaryKind::kAdd,routed_ports->scores,bias);
    const auto selected=f.append(c::Operation(c::TopK{.count=std::size_t(model.top_k)}),{routed_ports->selection},2);
    route.ids=selected[1];
    const auto unbiased=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kUnchecked),{routed_ports->scores,route.ids});
    routed_ports->selected_scores=unbiased;
    routed_ports->sum=moe::sum(f,unbiased,-1);
    routed_ports->normalized=core::binary(f,c::BinaryKind::kDivide,unbiased,routed_ports->sum);
    route.weights=core::binary(f,c::BinaryKind::kMultiply,routed_ports->normalized,
        core::filled(f,sem::dense({1,model.top_k}),model.scaling));
    local_ids=f.one(c::Operation(c::Gather{0,c::IndexBoundsPolicy::kChecked}),{expert_map,route.ids});
    routed_ports->ids=route.ids;routed_ports->weights=route.weights;routed_ports->map=expert_map;routed_ports->local_ids=local_ids;
  } else route=moe::routing(f, model, x, native_routing, 64.0F, lookup);
  const auto resident = model.top_k + 1;
  auto slots = core::reshape(f, x, {tokens, 1, model.hidden});
  sem::Weight wg{},wu{},wd{};
  if(!routed_parameters) {
    wg=d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 71, true);
    wu=d5_official_fp4_storage::weight(resident, model.intermediate, model.hidden, 79, true);
    wd=d5_official_fp4_storage::weight(resident, model.hidden, model.intermediate, 83, true);
  }
  const auto grouped = [&](sem::Weight &w, ValueId value) {
    const auto stored = f.constant(w.description, std::move(w.storage));
    value = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {value});
    return fp8::visible(f, f.one(c::model_ops::grouped_float4_linear(), {stored, value, route.ids}), true);
  };
  ValueId gate,up;
  if(routed_parameters) {
    routed_ports->gate_raw=f.one(c::model_ops::grouped_matmul(),{compact_weights[0],slots,local_ids});
    gate=bfloat_boundary(f,routed_ports->gate_raw);
    routed_ports->up_raw=f.one(c::model_ops::grouped_matmul(),{compact_weights[1],slots,local_ids});
    up=bfloat_boundary(f,routed_ports->up_raw);
    routed_ports->gate=gate;routed_ports->up=up;
  } else {gate=grouped(wg,slots);up=grouped(wu,slots);}
  auto activated = moe::gated(f, model, gate, up);
  if(routed_parameters) {
    const auto &node=f.nodes.back();
    if(!node.operation.get_if<c::GatedActivation>() || node.inputs.size()!=2)throw std::runtime_error("actual expert gated operands");
    routed_ports->gate_clipped=node.inputs[0];routed_ports->up_clipped=node.inputs[1];routed_ports->activated=activated;
  }
  auto weighted = core::binary(f, c::BinaryKind::kMultiply, activated,
                               core::reshape(f, route.weights, {tokens, model.top_k, 1}));
  ValueId routed;
  if(routed_parameters) {
    routed_ports->weighted_raw=weighted;
    const auto down_input=bfloat_boundary(f,weighted);routed_ports->weighted=down_input;
    routed_ports->down_raw=f.one(c::model_ops::grouped_matmul(),{compact_weights[2],down_input,local_ids});
    routed=bfloat_boundary(f,routed_ports->down_raw);routed_ports->down=routed;
  } else routed=grouped(wd,weighted);
  auto combined = f.one(c::Operation(c::Copy{}),
      {core::reshape(f, moe::sum(f, routed, 1), {tokens, model.hidden})});

  fp8::ModelLinear sg{},su{},sd{};
  ValueId shared_gate,shared_up,shared_activated,shared_down;
  ValueId shared_gate_raw{},shared_up_raw{},gate_clipped{},up_clipped{},activated_stored{},shared_down_raw{};
  std::array<PackedProjectionReference,3> supplied_references;
  if(shared_parameters) {
    shared_gate_raw=dspark_stored_projection_fragment::linear(f,x,shared_weights[0]);
    shared_gate=bfloat_boundary(f,shared_gate_raw);
    shared_up_raw=dspark_stored_projection_fragment::linear(f,x,shared_weights[1]);
    shared_up=bfloat_boundary(f,shared_up_raw);
    shared_activated=moe::gated(f,model,shared_gate,shared_up);
    // Observe the existing gated operation's actual bounded operands. This
    // inspects cold model math and creates no alternate activation formula.
    const auto &gated=f.nodes.back();
    if(!gated.operation.get_if<c::GatedActivation>() || gated.inputs.size()!=2 ||
        gated.outputs.size()!=1 || gated.outputs[0].ordinal()!=shared_activated.ordinal())
      throw std::runtime_error("shared Q8_0 diagnostic requires the actual separate gated activation");
    gate_clipped=gated.inputs[0];up_clipped=gated.inputs[1];
    activated_stored=bfloat_boundary(f,shared_activated);
    shared_down_raw=dspark_stored_projection_fragment::linear(f,activated_stored,shared_weights[2]);
    shared_down=bfloat_boundary(f,shared_down_raw);
    for(std::size_t i=0;i<supplied_references.size();++i)supplied_references[i]=shared_parameters->weights[i].reference;
  } else {
    sg=fp8::model_linear(f,x,model.shared_intermediate,model.hidden,113,true);
    su=fp8::model_linear(f,x,model.shared_intermediate,model.hidden,127,true);
    shared_gate=sg.output;shared_up=su.output;
    shared_activated=moe::gated(f,model,shared_gate,shared_up);
    sd=fp8::model_linear(f,shared_activated,model.hidden,model.shared_intermediate,139,true);
    shared_down=sd.output;
  }
  const auto full_raw=core::binary(f,c::BinaryKind::kAdd,combined,shared_down);
  auto output=bfloat_boundary(f,full_raw);
  f.outputs = {moe::gathered_ids(f, model, route.ids), route.weights, routed, combined,
                shared_down, output, gate, up, activated, shared_gate, shared_up,
                shared_activated};
  if(shared_parameters) f.outputs.insert(f.outputs.end(),{x,shared_gate_raw,shared_up_raw,
      gate_clipped,up_clipped,activated_stored,shared_down_raw,full_raw});
  if(routed_ports) {
    routed_ports->combined=combined;
    routed_ports->visit_values([&](ValueId value){f.outputs.push_back(value);});
    *reference_out=[](std::span<const float>)->moe::ReferenceOutputs {
      throw std::runtime_error("original resident experts require actual-predecessor local references");
    };
    return f;
  }
  f.exact_outputs = {0};
  // Keep the established full-chain gate and independent source decoding.
  // The failed ggml composition and its extra Q8 conversion remain in F43 evidence.
  auto evaluate = [model, sg, su, sd, supplied_references, wg = std::move(wg), wu = std::move(wu), wd = std::move(wd)]
      (std::span<const float> incoming, std::span<const std::int32_t> hash_ids) -> moe::ReferenceOutputs {
    const auto source = bfloat_boundary(std::vector<float>(incoming.begin(), incoming.end()));
    auto route_ref = moe::route_reference(model, source, 64.0F, hash_ids);
    auto a8 = quant::reference(source, false, 128);
    std::vector<float> replicated;
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      replicated.insert(replicated.end(), a8.begin(), a8.end());
    auto gate_ref = bfloat_boundary(moe::project(replicated, wg, route_ref.ids, model.intermediate, model.hidden));
    auto up_ref = bfloat_boundary(moe::project(replicated, wu, route_ref.ids, model.intermediate, model.hidden));
    auto act = moe::gate_reference(model, gate_ref, up_ref);
    auto weighted_ref = act;
    for (std::int64_t slot = 0; slot < model.top_k; ++slot)
      for (std::int64_t i = 0; i < model.intermediate; ++i)
        weighted_ref[slot * model.intermediate + i] *= route_ref.weights[slot];
    auto down_ref = quant::reference(bfloat_boundary(std::move(weighted_ref)), false, 128);
    auto routed_ref = bfloat_boundary(moe::project(down_ref, wd, route_ref.ids, model.hidden, model.intermediate));
    std::vector<float> combined_ref(model.hidden);
    // Official accumulation iterates experts by ID. All terms here are BF16
    // and at most six, so this fixed F32 sum preserves its rounding boundary.
    auto order = route_ref.ids;
    std::sort(order.begin(), order.end());
    for (auto id : order) {
      const auto slot = std::find(route_ref.ids.begin(), route_ref.ids.end(), id) - route_ref.ids.begin();
      for (std::int64_t i = 0; i < model.hidden; ++i)
        combined_ref[i] += routed_ref[slot * model.hidden + i];
    }
    auto shared_gate_ref = supplied_references[0]?bfloat_boundary(supplied_references[0](source)):sg.reference(source);
    auto shared_up_ref = supplied_references[1]?bfloat_boundary(supplied_references[1](source)):su.reference(source);
    auto shared_act = moe::gate_reference(model, shared_gate_ref, shared_up_ref);
    auto shared = supplied_references[2]?bfloat_boundary(supplied_references[2](bfloat_boundary(shared_act))):sd.reference(shared_act);
    auto full = combined_ref;
    for (std::int64_t i = 0; i < model.hidden; ++i) full[i] = fp8::bfloat(full[i] + shared[i]);
    return {std::move(route_ref.tags), std::move(route_ref.weights), std::move(routed_ref),
            std::move(combined_ref), std::move(shared), std::move(full), std::move(gate_ref),
            std::move(up_ref), std::move(act), std::move(shared_gate_ref), std::move(shared_up_ref),
            std::move(shared_act)};
  };
  if (reference_out) {
    *reference_out = [evaluate = std::move(evaluate),hidden=model.hidden](std::span<const float> values) {
      moe::ReferenceOutputs result;
      for(std::size_t first=0;first<values.size();first+=hidden) {
        auto row=evaluate(values.subspan(first,hidden),{});
        if(result.empty())result.resize(row.size());
        for(std::size_t port=0;port<row.size();++port)
          result[port].insert(result[port].end(),row[port].begin(),row[port].end());
      }
      return result;
    };
    return f;
  }
  for (int replay = 0; replay < 3; ++replay) {
    auto values = moe::activation(model, replay);
    if (use_hash) {
      const std::array<std::int64_t, 3> hash_tokens{0, moe::kHashVocabulary - 1, 37};
      const std::array token{hash_tokens[replay]};
      auto table = moe::hash_table(resident, replay);
      auto expected = evaluate(values, std::span<const std::int32_t>(table).subspan(token[0] * 6, 6));
      f.replays.push_back({{sem::bytes(values), moe::hash_tokens(token, native_routing), sem::bytes(table)},
                            std::move(expected)});
    } else f.replays.push_back({{sem::bytes(values)}, evaluate(values, {})});
  }
  return f;
}

inline Fragment projection(const moe::Case &model, bool down, bool shared,
                            std::int64_t tokens) {
  const auto n = down ? model.hidden : model.intermediate;
  const auto k = down ? model.intermediate : model.hidden;
  const auto slots = model.top_k, a = shared ? 1 : slots, experts = slots + 1;
  Fragment f{.profile = model.profile, .region = 0x0b0607};
  auto weight = d5_official_fp4_storage::weight(experts, n, k, down ? 83 : 71, true);
  auto w = f.constant(weight.description, std::move(weight.storage));
  auto input = f.input(sem::dense({tokens, a, k}, c::DataType::kBFloat16));
  auto ids = f.input(sem::dense({tokens, slots}, c::DataType::kInt32));
  auto output = f.one(c::model_ops::grouped_float4_linear(), {w, input, ids});
  f.outputs = {fp8::visible(f, output, true)};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    auto values = bfloat_boundary(d5_gguf_test::activation(tokens * a * k, 17 + replay * 31));
    const auto quantized = quant::reference(values, false, 128);
    std::vector<std::int32_t> selected(tokens * slots);
    std::vector<float> expanded;
    for (std::int64_t slot = 0; slot < tokens * slots; ++slot) {
      selected[slot] = std::int32_t((slot * (replay ? 2 : 1) + replay) % experts);
      const auto row = shared ? slot / slots : slot;
      expanded.insert(expanded.end(), quantized.begin() + row * k, quantized.begin() + (row + 1) * k);
    }
    if (replay) selected[1] = selected[0];
    auto expected = bfloat_boundary(moe::project(expanded, weight, selected, n, k));
    f.replays.push_back({{fp8::input_bytes(values, true), sem::bytes(selected)}, {std::move(expected)}});
  }
  return f;
}
} // namespace d5_deepseek_official_fragments
