#include "component.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include "hyle/compute/model_ops.h"

namespace hyle::inference::kimi_k3 {
namespace {
using ValueId=g::ValueId;
using roles::detail::take;
constexpr std::int64_t kHidden=7168,kHeads=96;
constexpr float kEpsilon=1.0e-5F;
void check(hyle::Status status,const char *where) {
  if(!status.ok())throw EvaluationFailure(status.WithContext(where));
}
c::TensorDesc dense(std::initializer_list<std::int64_t> shape,c::DataType type=c::DataType::kFloat32) {
  return take(c::TensorDesc::dense(type,{shape.begin(),shape.size()}),"Kimi dense description");
}
template<class T> Bytes bytes(const std::vector<T> &values) {
  const auto raw=std::as_bytes(std::span(values));return {raw.begin(),raw.end()};
}
// Cold descriptor facts only. Every node is emitted immediately into the
// original GraphBuilder; there is no node list, lowering IR or executor here.
class Assembly final {
 public:
  explicit Assembly(const c::StorageOwner &owner,LiteralBank *shared=nullptr)
      :builder_(take(g::GraphBuilder::create(),"Kimi Graph builder")),owner_(owner),shared_(shared) {}
  std::size_t value_count() const noexcept {return descriptions_.size();}
  const c::TensorDesc &description(ValueId value) const {return descriptions_.at(value.ordinal());}
  ValueId input(c::TensorDesc description) {
    const auto value=take(builder_.input(description),"Kimi Graph input");
    remember(value,description);result_.inputs.push_back({value,description});return value;
  }
  ValueId constant(c::TensorDesc description,Bytes raw) {
    const auto extent=take(description.required_bytes(),"Kimi literal extent");
    if(raw.size()!=extent)throw std::runtime_error("Kimi generated literal extent differs");
    const auto slot=result_.constants.size();
    auto lease=shared_?shared_->bind(slot,description,std::move(raw),owner_):upload(owner_,raw);
    const auto value=take(builder_.constant(description),"Kimi generated Graph constant");
    remember(value,description);
    result_.constants.push_back({.value=value,.tensor=take(lease.tensor(description),"Kimi typed literal"),.owner=owner_,.lease=std::move(lease)});
    return value;
  }
  ValueId constant(const roles::ReadonlyRole &source) {
    if(source.owner.ref()!=owner_.ref())throw std::runtime_error("original Kimi constant owner differs from its cold Graph owner");
    if(source.lease.size()!=take(source.description.required_bytes(),"original Kimi supplied constant extent"))
      throw std::runtime_error("original Kimi constant must have its actual lease and no generated payload");
    const auto value=take(builder_.constant(source.description),"Kimi learned Graph constant");
    remember(value,source.description);result_.constants.push_back(take(source.binding(value),"Kimi original typed binding"));
    return value;
  }
  std::vector<ValueId> append(c::Operation operation,std::initializer_list<ValueId> ports,std::size_t count=1) {
    std::vector<c::TensorDesc> in,out(count);
    for(const auto value:ports)in.push_back(description(value));
    check(c::infer_output_descriptions(operation,in,out),"Kimi operation description");
    const auto node=take(builder_.add_node(operation,{ports.begin(),ports.size()}),"Kimi Graph node");
    std::vector<ValueId> values;values.reserve(count);
    for(std::size_t index=0;index<count;++index) {
      const auto value=take(builder_.result(node,index),"Kimi Graph result");remember(value,out[index]);values.push_back(value);
    }
    return values;
  }
  ValueId one(c::Operation operation,std::initializer_list<ValueId> ports) {return append(operation,ports)[0];}
  GraphDefinition finish(std::span<const ValueId> outputs,std::uint64_t region) {
    for(auto value:outputs) {
      // Preserve the old explicit caller-output Copy and publication order.
      value=one(c::Operation(c::Copy{}),{value});
      check(builder_.publish(value),"Kimi Graph publish");result_.outputs.push_back({value,description(value)});
    }
    result_.region=region;
    result_.workload={take(c::ModelArtifactProfile::create(roles::profile),"Kimi geometry profile"),c::WorkloadPhase::kDecode,{0x48594c454d4f444cULL,region,1}};
    result_.graph=take(std::move(builder_).finish(),"Kimi Graph finish");return std::move(result_);
  }
 private:
  void remember(ValueId value,const c::TensorDesc &description) {
    if(value.ordinal()!=descriptions_.size())throw std::runtime_error("Kimi builder value inventory is not dense");
    descriptions_.push_back(description);
  }
  g::GraphBuilder builder_;
  const c::StorageOwner &owner_;
  LiteralBank *shared_;
  std::vector<c::TensorDesc> descriptions_;
  GraphDefinition result_;
};
ValueId reshape(Assembly &f, ValueId input,
                       std::initializer_list<std::int64_t> shape) {
  return f.one(c::Operation(c::View::reshape(
                   c::TensorShape::create({shape.begin(), shape.size()}).value())
                               .value()),
               {input});
}
ValueId unary(Assembly &f, c::UnaryKind kind, ValueId input) {
  return f.one(c::Operation(c::Unary{kind}), {input});
}
ValueId binary(Assembly &f, c::BinaryKind kind, ValueId a, ValueId b) {
  return f.one(c::Operation(c::Binary{kind}), {a, b});
}
ValueId filled(Assembly &f, const c::TensorDesc &desc, float value) {
  return f.constant(desc, bytes(std::vector<float>(*desc.element_count(), value)));
}
ValueId rows(Assembly &f, ValueId matrix, std::int64_t first,
                    std::int64_t count) {
  std::vector<std::int32_t> indices(count);
  for (std::int64_t i = 0; i < count; ++i)
    indices[i] = static_cast<std::int32_t>(first + i);
  auto ids = f.constant(dense({count}, c::DataType::kInt32), bytes(indices));
  return f.one(c::Operation(c::Gather{}), {matrix, ids});
}void require_value(const Assembly &f, ValueId value,
                          std::initializer_list<std::int64_t> dimensions) {
  if (value.ordinal() >= f.value_count() ||
      !std::ranges::equal(f.description(value).dimensions(), dimensions))
    throw std::runtime_error("supplied KDA/MLA parameter or input shape differs");
}
void require_float_parameter(const Assembly &f, ValueId value,
                                   std::initializer_list<std::int64_t> dimensions) {
  require_value(f, value, dimensions);
  const auto &description = f.description(value);
  if (!description.is_plain() ||
      (description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16))
    throw std::runtime_error("supplied KDA/MLA scalar parameter must be F32 or BF16");
}
void require_projection(const Assembly &f, ValueId value,
                               std::int64_t n, std::int64_t k) {
  if (value.ordinal() >= f.value_count())
    throw std::runtime_error("supplied projection is outside this Assembly");
  const auto &description = f.description(value);
  if ((description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16) ||
      (description.data_type() == c::DataType::kBFloat16 && !description.is_plain()))
    throw std::runtime_error("supplied projection needs its original F32/GGUF or BF16 weight");
  // DenseLinearF32 reads BF16 [N,K]; the existing MatMul RHS is logically
  // [K,N] for both transposed F32 storage and GGUF. Do not transpose it here.
  if (description.data_type() == c::DataType::kBFloat16) require_value(f, value, {n, k});
  else require_value(f, value, {k, n});
}
ValueId float_parameter(Assembly &f, ValueId value) {
  return f.description(value).data_type() == c::DataType::kBFloat16
      ? f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {value}) : value;
}
ValueId project_f32(Assembly &f, ValueId input, ValueId weight) {
  const auto &description = f.description(weight);
  if (description.data_type() == c::DataType::kBFloat16) {
    input = f.one(c::Operation(c::Cast{c::DataType::kBFloat16}), {input});
    return f.one(c::model_ops::dense_linear_f32(), {input, weight});
  }
  return f.one(c::Operation(c::MatMul{}), {input, weight});
}
enum class DecayEncoding { kLog, kNegativeExp };
struct KdaGeometry final {
  std::int64_t hidden, heads, depth;
  float l2_epsilon, norm_epsilon;
};
struct KdaParameters final {
  ValueId q, k, v, f_a, f_b, beta, gate, output;
  ValueId convolution, a, dt_bias, output_norm;
  std::optional<ValueId> gate_a;
  DecayEncoding decay;
};
struct KdaPorts final {
  ValueId history, convolution, state, recurrent, gated, output;
  std::array<ValueId, 6> outputs() const {
    return {history, convolution, state, recurrent, gated, output};
  }
};

KdaPorts append_kda(Assembly &f, ValueId input, ValueId history,
    ValueId state, const KdaParameters &p, const KdaGeometry &geometry) {
  const auto [hidden, heads, depth, l2_epsilon, norm_epsilon] = geometry;
  if (hidden <= 0 || heads <= 0 || depth != 128 ||
      heads > std::numeric_limits<std::int64_t>::max() / depth / 3 ||
      !(l2_epsilon > 0) || !std::isfinite(l2_epsilon) ||
      !(norm_epsilon > 0) || !std::isfinite(norm_epsilon) ||
      (p.decay != DecayEncoding::kLog && p.decay != DecayEncoding::kNegativeExp))
    throw std::runtime_error("invalid supplied KDA geometry or decay encoding");
  const auto width = heads * depth, channels = 3 * width;
  require_value(f, input, {1, hidden});
  require_value(f, history, {3, channels});
  require_value(f, state, {1, heads, depth, depth});
  for (auto value : {input, history, state})
    if (f.description(value).data_type() != c::DataType::kFloat32 || !f.description(value).is_plain())
      throw std::runtime_error("supplied KDA keeps its original plain F32 input/state boundary");
  for (auto value : {p.q, p.k, p.v}) require_projection(f, value, width, hidden);
  require_projection(f, p.f_a, depth, hidden);
  require_projection(f, p.f_b, width, depth);
  require_projection(f, p.beta, heads, hidden);
  if (p.gate_a) require_projection(f, *p.gate_a, depth, hidden);
  require_projection(f, p.gate, width, p.gate_a ? depth : hidden);
  require_projection(f, p.output, hidden, width);
  require_float_parameter(f, p.convolution, {channels, 4});
  require_float_parameter(f, p.dt_bias, {1, 1, heads, depth});
  require_float_parameter(f, p.output_norm, {depth});
  if (p.a.ordinal() >= f.value_count() ||
      f.description(p.a).rank() != 2 || f.description(p.a).dimensions()[0] < heads)
    throw std::runtime_error("supplied KDA decay source does not cover every head");
  require_float_parameter(f, p.a, {f.description(p.a).dimensions()[0], 1});
  if (p.decay == DecayEncoding::kNegativeExp && f.description(p.a).dimensions()[0] != heads)
    throw std::runtime_error("prepared negative-exp KDA decay must have exactly one value per head");

  const auto q = project_f32(f, input, p.q), k = project_f32(f, input, p.k);
  const auto v = project_f32(f, input, p.v);
  const auto fa = project_f32(f, input, p.f_a), fb = project_f32(f, fa, p.f_b);
  const auto beta_input = project_f32(f, input, p.beta);
  const auto gate_input = p.gate_a ? project_f32(f, input, *p.gate_a) : input;
  const auto gate = project_f32(f, gate_input, p.gate);
  auto joined = f.one(c::Operation(c::Concat{.axis = 1}), {q, k});
  joined = f.one(c::Operation(c::Concat{.axis = 1}), {joined, v});
  const auto window = f.one(c::Operation(c::Concat{.axis = 0}), {history, joined});
  const auto next_history = rows(f, window, 1, 3);
  const auto convolution = f.one(c::model_ops::causal_conv1d(),
      {reshape(f, window, {1, 4, channels}), float_parameter(f, p.convolution)});
  const auto activated = f.one(c::Operation(c::GatedActivation{
      .inputs = c::GatedInputKind::kSeparate}),
      {convolution, filled(f, f.description(convolution), 1)});
  const auto channel_rows = reshape(f, activated, {channels, 1});
  auto query = reshape(f, rows(f, channel_rows, 0, width), {heads, depth});
  auto key = reshape(f, rows(f, channel_rows, width, width), {heads, depth});
  const auto value = reshape(f, rows(f, channel_rows, 2 * width, width), {1, 1, heads, depth});
  const auto l2_scale = filled(f, dense({depth}), 1.0F / std::sqrt(float(depth)));
  const c::Operation l2(c::RmsNorm{.epsilon = l2_epsilon / float(depth)});
  query = reshape(f, f.one(l2, {query, l2_scale}), {1, 1, heads, depth});
  key = reshape(f, f.one(l2, {key, l2_scale}), {1, 1, heads, depth});
  auto a = float_parameter(f, p.a);
  if (p.decay == DecayEncoding::kLog) {
    a = rows(f, a, 0, heads);
    a = unary(f, c::UnaryKind::kExp, a);
  } else {
    a = binary(f, c::BinaryKind::kMultiply, a, filled(f, f.description(a), -1));
  }
  a = reshape(f, a, {1, 1, heads, 1});
  const auto gate_description = dense({1, 1, heads, depth});
  const auto raw_gate = reshape(f, fb, {1, 1, heads, depth});
  auto decay = binary(f, c::BinaryKind::kAdd, raw_gate, float_parameter(f, p.dt_bias));
  decay = unary(f, c::UnaryKind::kSigmoid, binary(f, c::BinaryKind::kMultiply, decay, a));
  decay = binary(f, c::BinaryKind::kMultiply, decay, filled(f, gate_description, -5));
  const auto beta = unary(f, c::UnaryKind::kSigmoid, reshape(f, beta_input, {1, 1, heads, 1}));
  const auto recurrent = f.append(c::Operation(c::GatedDeltaNet{}), {query, key, value, decay, beta, state}, 2);
  const auto normed = f.one(c::Operation(c::RmsNorm{.epsilon = norm_epsilon}),
      {recurrent[0], float_parameter(f, p.output_norm)});
  const auto gated = binary(f, c::BinaryKind::kMultiply, normed,
      unary(f, c::UnaryKind::kSigmoid, reshape(f, gate, {1, 1, heads, depth})));
  const auto output = project_f32(f, reshape(f, gated, {1, width}), p.output);
  return {next_history, convolution, recurrent[1], recurrent[0], gated, output};
}
ValueId selected(Assembly &f, ValueId input,
                         std::vector<std::int32_t> indices) {
  const auto count = std::int64_t(indices.size());
  auto ids = f.constant(dense({count}, c::DataType::kInt32), bytes(indices));
  auto flat = reshape(f, input, {std::int64_t(*f.description(input).element_count()), 1});
  return f.one(c::Operation(c::Gather{}), {flat, ids});
}
ValueId slice_heads(Assembly &f, ValueId input, std::int64_t heads,
                           std::int64_t stride, std::int64_t start,
                           std::int64_t count) {
  std::vector<std::int32_t> ids;
  for (std::int64_t h = 0; h < heads; ++h)
    for (std::int64_t d = 0; d < count; ++d)
      ids.push_back(std::int32_t(h * stride + start + d));
  return reshape(f, selected(f, input, std::move(ids)), {1, heads, count});
}ValueId project_heads(Assembly &f, ValueId input, ValueId weight) {
  std::vector<std::int32_t> ids(kHeads);
  std::iota(ids.begin(), ids.end(), 0);
  const auto head_ids = f.constant(dense({1, kHeads}, c::DataType::kInt32), bytes(ids));
  return f.one(c::model_ops::grouped_matmul(), {weight, input, head_ids});
}
struct OrdinaryHistory final {
  std::int64_t capacity, position;
};
std::pair<std::int64_t, std::int64_t> history_extents(
    std::optional<OrdinaryHistory> history) {
  if (history && (history->capacity <= 0 || history->capacity > std::numeric_limits<std::int32_t>::max() ||
      history->position < 0 || history->position >= history->capacity))
    throw std::runtime_error("ordinary Kimi MLA history is outside its cold capacity");
  return {history ? history->capacity : 3, history ? history->position + 1 : 4};
}
struct MlaParameters final {
  ValueId input_norm, q_a, q_norm, q_b, kv_a, kv_norm;
  // Prepared original KV-b K/V layouts; they share source identity but are
  // actual independent grouped operands, not a generated replacement bank.
  ValueId absorbed_key, absorbed_value, gate, output;
};
struct MlaPorts final {
  ValueId cache, query, absorbed_query, attention, projected, gated, output, normalized, gate;
  std::array<ValueId, 9> outputs() const {
    return {cache, query, absorbed_query, attention, projected, gated, output, normalized, gate};
  }
};
void require_head_projection(const Assembly &f, ValueId value,
                                    std::int64_t n, std::int64_t k) {
  require_value(f, value, {kHeads, n, k});
  const auto &description = f.description(value);
  if ((description.data_type() != c::DataType::kFloat32 &&
       description.data_type() != c::DataType::kBFloat16) ||
      (description.data_type() == c::DataType::kBFloat16 && !description.is_plain()))
    throw std::runtime_error("supplied Kimi absorbed weight needs its original F32/GGUF or BF16 storage");
}

MlaPorts append_mla(Assembly &f, ValueId input, ValueId past,
    const MlaParameters &p, std::optional<OrdinaryHistory> history = {}) {
  const auto [past_rows, visible] = history_extents(history);
  require_value(f, input, {1, kHidden});
  require_value(f, past, {past_rows, 576});
  for (auto value : {input, past})
    if (f.description(value).data_type() != c::DataType::kFloat32 || !f.description(value).is_plain())
      throw std::runtime_error("supplied Kimi MLA keeps its original plain F32 input/cache boundary");
  require_float_parameter(f, p.input_norm, {kHidden});
  require_float_parameter(f, p.q_norm, {1536});
  require_float_parameter(f, p.kv_norm, {512});
  require_projection(f, p.q_a, 1536, kHidden);
  require_projection(f, p.q_b, kHeads * 192, 1536);
  require_projection(f, p.kv_a, 576, kHidden);
  require_head_projection(f, p.absorbed_key, 512, 128);
  require_head_projection(f, p.absorbed_value, 128, 512);
  require_projection(f, p.gate, 12288, kHidden);
  require_projection(f, p.output, kHidden, 12288);

  const c::Operation normalization(c::RmsNorm{.epsilon = kEpsilon});
  const auto normalized = f.one(normalization, {input, float_parameter(f, p.input_norm)});
  const auto qa = project_f32(f, normalized, p.q_a);
  const auto nq = f.one(normalization, {qa, float_parameter(f, p.q_norm)});
  const auto qb = project_f32(f, nq, p.q_b);
  const auto kva = project_f32(f, normalized, p.kv_a);
  auto latent = reshape(f, rows(f, reshape(f, kva, {576, 1}), 0, 512), {1, 512});
  const auto pe = reshape(f, rows(f, reshape(f, kva, {576, 1}), 512, 64), {1, 64});
  latent = f.one(normalization, {latent, float_parameter(f, p.kv_norm)});
  const auto current = f.one(c::Operation(c::Concat{.axis = 1}), {latent, pe});
  ValueId cache, next_cache;
  if (history) {
    const auto position = f.constant(dense({1}, c::DataType::kInt32),
        bytes(std::vector<std::int32_t>{std::int32_t(history->position)}));
    next_cache = f.one(c::Operation(c::Scatter{}), {past, position, current});
    cache = rows(f, next_cache, 0, visible);
  } else {
    cache = f.one(c::Operation(c::Concat{.axis = 0}), {past, current});
    next_cache = rows(f, cache, 1, 3);
  }
  const auto query_nope = slice_heads(f, qb, kHeads, 192, 0, 128);
  const auto query_pe = slice_heads(f, qb, kHeads, 192, 128, 64);
  const auto kb = project_heads(f, query_nope, p.absorbed_key);
  const auto query = f.one(c::Operation(c::Concat{.axis = 2}), {kb, query_pe});
  const auto cached_values = slice_heads(f, cache, visible, 576, 0, 512);
  const auto attention = f.one(c::Operation(c::ScaledDotProductAttention{
      .scale = 1.0F / std::sqrt(192.0F), .causal = true,
      .accumulation_type = c::DataType::kFloat32}),
      {reshape(f, query, {1, 1, kHeads, 576}), reshape(f, cache, {1, visible, 1, 576}),
       reshape(f, cached_values, {1, visible, 1, 512})});
  const auto vb = project_heads(f, reshape(f, attention, {1, kHeads, 512}), p.absorbed_value);
  const auto gate = project_f32(f, normalized, p.gate);
  const auto gated = binary(f, c::BinaryKind::kMultiply, reshape(f, vb, {1, 12288}),
      unary(f, c::UnaryKind::kSigmoid, gate));
  const auto output = project_f32(f, gated, p.output);
  return {next_cache, qb, kb, attention, vb, gated, output, normalized, gate};
}

KdaParameters declare_kda(Assembly &f,const roles::KdaView &p) {
  const auto constant=[&](const roles::ReadonlyRole &source) {return f.constant(source);};
  return {constant(p.q), constant(p.k), constant(p.v), constant(p.f_a),
      constant(p.f_b), constant(p.beta), constant(p.gate), constant(p.output),
      constant(p.convolution), constant(p.a_log), constant(p.dt_bias),
      constant(p.output_norm), std::nullopt, DecayEncoding::kLog};
}
MlaParameters declare_mla(Assembly &f,const roles::MlaView &p) {
  const auto constant=[&](const roles::ReadonlyRole &source) {return f.constant(source);};
  return {constant(p.input_norm), constant(p.q_a), constant(p.q_norm),
      constant(p.q_b), constant(p.kv_a), constant(p.kv_norm),
      constant(p.absorbed_key), constant(p.absorbed_value), constant(p.gate), constant(p.output)};
}
} // namespace

GraphDefinition build_kda_graph(const roles::Kda &original,const c::StorageOwner &owner) {
  const auto source=original.view();roles::validate(source);
  Assembly f(owner);
  const auto input=f.input(dense({1,7168}));
  const auto history=f.input(dense({3,kChannels}));
  const auto state=f.input(dense({1,96,128,128}));
  const auto parameters=declare_kda(f,source);
  const auto result=append_kda(f,input,history,state,parameters,{7168,96,128,1.e-5F,1.e-5F});
  const auto outputs=result.outputs();return f.finish(outputs,0x030602);
}
GraphDefinition build_mla_graph(std::int32_t position,const roles::Mla &original,
    const c::StorageOwner &owner,LiteralBank &small_literals) {
  // Keep the cold history check before source upload/Graph mutation.
  (void)history_extents(OrdinaryHistory{kCapacity,position});
  const auto source=original.view();roles::validate(source);
  Assembly f(owner,&small_literals);
  const auto input=f.input(dense({1,7168}));
  const auto past=f.input(dense({kCapacity,576}));
  const auto parameters=declare_mla(f,source);
  const auto result=append_mla(f,input,past,parameters,OrdinaryHistory{kCapacity,position});
  const auto ports=result.outputs();std::vector<ValueId> outputs(ports.begin(),ports.end());
  const auto cursor=f.input(dense({1},c::DataType::kInt32));
  const auto one=f.constant(dense({1},c::DataType::kInt32),bytes(std::vector<std::int32_t>{1}));
  const auto joined=f.one(c::Operation(c::Concat{.axis=0}),{cursor,one});
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),{joined});
  outputs.push_back(f.one(c::Operation(c::Gather{}),{scanned,one}));
  return f.finish(outputs,0x070701);
}
} // namespace hyle::inference::kimi_k3
