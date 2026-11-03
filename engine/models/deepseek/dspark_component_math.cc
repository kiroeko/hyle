#include "dspark_component_math.h"

namespace dspark_component_math {
using sem::ops::reshape;
using sem::ops::binary;
using sem::ops::unary;
using sem::ops::filled;
using sem::dense;
using sem::bytes;
ValueId bfloat_boundary(Fragment &f,ValueId value) {
  auto stored=f.one(c::Operation(c::Cast{c::DataType::kBFloat16}),{value});
  return f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{stored});
}
ValueId mean(Fragment &f, ValueId x) {
  const std::int32_t axis = std::int32_t(f.description(x).rank() - 1);
  return f.one(c::Operation(c::Reduce{.kind = c::ReductionKind::kMean,
      .axes = c::Axes::from(std::span(&axis, 1)).value(), .keep_dimensions = true,
      .accumulation_type = c::DataType::kFloat32}), {x});
}

ValueId query_norm(Fragment &f, ValueId x, QueryNormPorts *ports) {
  const auto square_raw=binary(f, c::BinaryKind::kMultiply, x, x);
  auto square = bfloat_boundary(f, square_raw);
  const auto mean_raw=mean(f,square);
  const auto average=bfloat_boundary(f,mean_raw);
  const auto variance_raw=binary(f,c::BinaryKind::kAdd,average,filled(f,dense({1}),1.0e-6F));
  auto variance = bfloat_boundary(f, variance_raw);
  const auto inverse_raw=unary(f,c::UnaryKind::kRsqrt,variance);
  auto inverse = bfloat_boundary(f, inverse_raw);
  const auto output_raw=binary(f,c::BinaryKind::kMultiply,x,inverse);
  const auto output=bfloat_boundary(f,output_raw);
  if(ports)*ports={x,square_raw,square,mean_raw,average,variance_raw,variance,inverse_raw,inverse,output_raw,output};
  return output;
}

std::vector<float> rope_frequencies(bool compressed, bool inverse) {
  const float base = compressed ? 160000.0F : 10000.0F;
  const auto correction = [&](double rotations) {
    return 64*std::log(65536/(rotations*2*std::acos(-1.0)))/(2*std::log(double(base)));
  };
  const auto low = std::max(0.0, std::floor(correction(32)));
  const auto high = std::min(63.0, std::ceil(correction(1)));
  std::vector<float> result(32);
  for (std::size_t i = 0; i < result.size(); ++i) {
    // Original torch F32 power/reciprocal and YaRN expression, not the
    // iterative frequency recurrence used by the separate GGUF consumer.
    auto frequency = 1.0F/std::pow(base, float(i)/32.0F);
    if (compressed) {
      const auto smooth = 1.0F-std::clamp((float(i)-float(low))/float(high == low ? 0.001 : high-low), 0.0F, 1.0F);
      frequency = frequency/16.0F*(1.0F-smooth)+frequency*smooth;
    }
    result[i] = inverse ? -frequency : frequency;
  }
  return result;
}

ValueId rope(Fragment &f, ValueId x, ValueId positions, bool compressed, bool inverse,
             RopePorts *ports) {
  const auto depth = f.description(x).dimensions().back();
  auto prefix = slice(f, x, 3, 0, depth-64), tail = slice(f, x, 3, depth-64, 64);
  auto frequencies = f.constant(dense({32}), bytes(rope_frequencies(compressed, inverse)));
  auto rotated = f.one(c::Operation(c::RoPE{.rotary_dimensions = 64,
      .sequence_axis = 1, .layout = c::RopeLayout::kInterleaved}), {tail, positions, frequencies});
  const auto raw=concat(f,prefix,rotated,3);
  const auto stored=bfloat_boundary(f,raw);
  if(ports)*ports={x,raw,stored};
  return stored;
}

ValueId inverse_rms(Fragment &f, ValueId input) {
  auto square = binary(f, c::BinaryKind::kMultiply, input, input);
  return unary(f, c::UnaryKind::kRsqrt, binary(f, c::BinaryKind::kAdd,
      mean(f, square), filled(f, dense({1}), 1.0e-6F)));
}

Hyper hyper_pre(Fragment &f, ValueId state, std::int64_t hidden,
                const HyperValues &parameters, bool bfloat) {
  // Graph assembly requires typed operands. Host coefficient spans belong to
  // the independent reference and are not needed to execute these Values.
  const auto require_value=[&](ValueId value,const c::TensorDesc &description) {
    if(value.ordinal()>=f.descriptions.size()||f.description(value)!=description)
      throw std::runtime_error("hyper-connection parameter has a different typed layout");
  };
  require_value(parameters.fn,dense({24,4*hidden}));
  require_value(parameters.scale,dense({3}));
  require_value(parameters.base,dense({24}));
  const auto tokens = f.description(state).dimensions()[0];
  const auto flat = reshape(f, state, {tokens, 4 * hidden});
  const auto product = f.one(c::Operation(c::MatMul{}), {flat, permute(f, parameters.fn, {1, 0})});
  Projection fn{product};
  const auto inverse = inverse_rms(f, flat);
  const auto mixes = binary(f, c::BinaryKind::kMultiply, product, inverse);
  Hyper result{{}, {}, {}, std::move(fn), hidden};
  result.bfloat_outputs = bfloat;
  result.combination = f.one(c::model_ops::hyper_connection_combine(1.0e-6F, 20),
      {mixes, parameters.scale, parameters.base});
  const auto gate = [&](int start, float multiplier, float offset) {
    const auto part = slice(f, mixes, 1, start, 4);
    const auto scale = slice(f, parameters.scale, 0, start / 4, 1);
    const auto base = reshape(f, slice(f, parameters.base, 0, start, 4), {1, 4});
    const auto affine = binary(f, c::BinaryKind::kAdd,
        binary(f, c::BinaryKind::kMultiply, part, scale), base);
    const auto sigmoid = unary(f, c::UnaryKind::kSigmoid, affine);
    return binary(f, c::BinaryKind::kAdd,
        binary(f, c::BinaryKind::kMultiply, sigmoid, filled(f, dense({1}), multiplier)),
        filled(f, dense({1}), offset));
  };
  result.post = gate(4, 2, 0);
  const auto pre = gate(0, 1, 1.0e-6F);
  const auto collapsed_raw = f.one(c::model_ops::hyper_connection_pre(), {state, pre});
  result.collapsed = bfloat ? bfloat_boundary(f, collapsed_raw) : collapsed_raw;
  result.ports = {state, flat, product, inverse, mixes, pre, result.post,
      result.combination, collapsed_raw, result.collapsed};
  return result;
}

ValueId hyper_post(Fragment &f, const Hyper &mix, ValueId state, ValueId branch,
                   HyperPostPorts *ports) {
  auto result=f.one(c::model_ops::hyper_connection_post(), {branch, state, mix.post, mix.combination});
  const auto stored = mix.bfloat_outputs?bfloat_boundary(f,result):result;
  if (ports) *ports = {branch, state, result, stored};
  return stored;
}
}
