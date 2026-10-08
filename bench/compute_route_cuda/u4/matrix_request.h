#pragma once
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace c = hyle::compute;
template <class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> dimensions) {
  return take(c::TensorDesc::dense(type, std::span(dimensions.begin(), dimensions.size())));
}
struct Request {
  c::Operation operation;
  std::vector<c::TensorDesc> inputs;
  c::TensorDesc output;
  std::string name;
  int tokens;
  int hidden;
};
Request request(std::string op, c::DataType type, int tokens, int hidden,
                c::DataType output_type = c::DataType::kUnknown) {
  Request r{.operation = c::Operation(c::Copy{}), .inputs = {dense(type, {tokens, hidden})},
            .output = {}, .name = op, .tokens = tokens, .hidden = hidden};
  const auto set = [&](auto value) { r.operation = c::Operation(std::move(value)); };
  const std::array unary_names{"abs", "neg", "exp", "log", "sqrt", "rsqrt", "tanh",
      "sigmoid", "silu", "gelu", "gelu_tanh", "softplus"};
  const std::array unary_kinds{c::UnaryKind::kAbs, c::UnaryKind::kNeg, c::UnaryKind::kExp,
      c::UnaryKind::kLog, c::UnaryKind::kSqrt, c::UnaryKind::kRsqrt, c::UnaryKind::kTanh,
      c::UnaryKind::kSigmoid, c::UnaryKind::kSilu, c::UnaryKind::kGelu,
      c::UnaryKind::kGeluTanh, c::UnaryKind::kSoftplus};
  for (std::size_t i = 0; i < unary_names.size(); ++i)
    if (op == unary_names[i]) set(c::Unary{unary_kinds[i]});
  const std::array binary_names{"add", "sub", "mul", "div"};
  const std::array binary_kinds{c::BinaryKind::kAdd, c::BinaryKind::kSubtract,
      c::BinaryKind::kMultiply, c::BinaryKind::kDivide};
  for (std::size_t i = 0; i < binary_names.size(); ++i)
    if (op == binary_names[i]) {
      set(c::Binary{binary_kinds[i]}); r.inputs.push_back(r.inputs[0]);
    }
  if (op == "cast") set(c::Cast{output_type});
  if (op == "rms" || op == "weighted_rms") {
    set(c::RmsNorm{.epsilon = 1.0e-6F});
    if (op == "weighted_rms") r.inputs.push_back(dense(type, {hidden}));
  }
  if (op == "layer" || op == "affine_layer") {
    set(c::LayerNorm{.epsilon = 1.0e-6F});
    if (op == "affine_layer") {
      r.inputs.push_back(dense(type, {hidden})); r.inputs.push_back(r.inputs.back());
    }
  }
  if (op == "sum" || op == "mean") {
    const std::array<std::int32_t, 1> axes{-1};
    set(c::Reduce{.kind = op == "sum" ? c::ReductionKind::kSum : c::ReductionKind::kMean,
        .axes = take(c::Axes::from(axes)), .accumulation_type = c::DataType::kFloat32});
  }
  if (op == "softmax" || op == "masked_softmax") {
    set(c::Softmax{.mask = op == "masked_softmax" ? c::MaskKind::kAdditive : c::MaskKind::kNone});
    if (op == "masked_softmax") r.inputs.push_back(dense(type, {1, hidden}));
  }
  if (op == "concat") {
    r.inputs = {dense(type, {tokens, hidden / 2}), dense(type, {tokens, hidden / 2})};
    set(c::Concat{.axis = -1});
  }
  if (op.starts_with("packed_") || op.starts_with("separate_")) {
    const bool packed = op.starts_with("packed_");
    const auto kind = op.ends_with("gelu_tanh") ? c::GatedActivationKind::kGeluTanh
                    : op.ends_with("gelu") ? c::GatedActivationKind::kGelu
                                             : c::GatedActivationKind::kSilu;
    r.inputs = {dense(type, {tokens, hidden * (packed ? 2 : 1)})};
    if (!packed) r.inputs.push_back(r.inputs[0]);
    set(c::GatedActivation{.activation = kind,
        .inputs = packed ? c::GatedInputKind::kPackedHalves : c::GatedInputKind::kSeparate});
  }
  if (op == "gather") {
    r.inputs = {dense(type, {7, hidden}), dense(c::DataType::kInt32, {tokens})};
    set(c::Gather{});
  }
  r.output = take(c::infer_single_output(r.operation, r.inputs));
  return r;
}
struct Candidate { const char *name; c::Provider provider; };
std::vector<std::byte> pack(const c::TensorDesc &desc, const std::vector<float> &values) {
  std::vector<std::byte> bytes(take(desc.required_bytes()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    switch (desc.data_type()) {
    case c::DataType::kFloat32: std::memcpy(bytes.data() + i * 4, &values[i], 4); break;
    case c::DataType::kFloat16: {
      const auto x = __float2half_rn(values[i]); std::memcpy(bytes.data() + i * 2, &x, 2); break;
    }
    case c::DataType::kBFloat16: {
      const auto x = __float2bfloat16_rn(values[i]); std::memcpy(bytes.data() + i * 2, &x, 2); break;
    }
    case c::DataType::kInt32: {
      const auto x = static_cast<std::int32_t>(values[i]); std::memcpy(bytes.data() + i * 4, &x, 4); break;
    }
    default: throw std::runtime_error("unsupported fixture storage");
    }
  }
  return bytes;
}
std::vector<float> unpack(c::DataType type, const std::vector<std::byte> &bytes) {
  std::vector<float> values(bytes.size() / c::data_type_size(type));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (type == c::DataType::kFloat32) std::memcpy(&values[i], bytes.data() + i * 4, 4);
    else if (type == c::DataType::kFloat16) {
      __half x; std::memcpy(&x, bytes.data() + i * 2, 2); values[i] = __half2float(x);
    } else if (type == c::DataType::kBFloat16) {
      __nv_bfloat16 x; std::memcpy(&x, bytes.data() + i * 2, 2); values[i] = __bfloat162float(x);
    } else {
      std::int32_t x; std::memcpy(&x, bytes.data() + i * 4, 4); values[i] = static_cast<float>(x);
    }
  }
  return values;
}
std::vector<std::vector<float>> payload(const Request &r, int seed) {
  std::vector<std::vector<float>> result;
  for (std::size_t port = 0; port < r.inputs.size(); ++port) {
    const auto &desc = r.inputs[port];
    std::vector<float> values(take(desc.element_count()));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto phase = static_cast<int>((i * 7 + static_cast<std::size_t>(seed * 11)) % 97) - 48;
      float value = static_cast<float>(phase) / 64.F;
      if (r.name == "gather" && port == 1) value = static_cast<float>((i + seed) % 7);
      else if (r.name == "weighted_rms" && port == 1)
        value = .75F + static_cast<float>(i % 17) / 64.F;
      else if (r.name == "affine_layer" && port != 0)
        value = port == 1 ? .75F + static_cast<float>(i % 17) / 64.F : value / 8.F;
      else if (r.name == "masked_softmax" && port == 1) value /= 8.F;
      else if (r.name == "log" || r.name == "sqrt" || r.name == "rsqrt" ||
               (r.name == "div" && port == 1)) value = 1.F + std::abs(value);
      values[i] = value;
    }
    result.push_back(unpack(desc.data_type(), pack(desc, values)));
  }
  return result;
}
double activation(std::string_view kind, double x) {
  if (kind.ends_with("gelu_tanh"))
    return .5 * x * (1 + std::tanh(std::sqrt(2. / 3.14159265358979323846) * (x + .044715 * x * x * x)));
  if (kind.ends_with("gelu")) return .5 * x * std::erfc(-x / std::sqrt(2.));
  return x / (1 + std::exp(-x));
}
std::vector<double> oracle(const Request &r, const std::vector<std::vector<float>> &p) {
  std::vector<double> out(take(r.output.element_count()));
  const auto width = static_cast<std::size_t>(r.hidden);
  for (std::size_t row = 0; row < static_cast<std::size_t>(r.tokens); ++row) {
    double sum = 0, square = 0, maximum = -1.e30;
    if (r.name == "rms" || r.name == "weighted_rms" || r.name == "layer" || r.name == "affine_layer" ||
        r.name == "sum" || r.name == "mean" || r.name == "softmax" || r.name == "masked_softmax") {
      for (std::size_t col = 0; col < width; ++col) {
        const double x = p[0][row * width + col]; sum += x; square += x * x;
        maximum = std::max(maximum, x + (r.name == "masked_softmax" ? p[1][col] : 0));
      }
      if (r.name == "sum" || r.name == "mean") {
        out[row] = r.name == "sum" ? sum : sum / width; continue;
      }
    }
    double variance = 0, denominator = 0;
    if (r.name == "layer" || r.name == "affine_layer")
      for (std::size_t col = 0; col < width; ++col) {
        const double centered = p[0][row * width + col] - sum / width; variance += centered * centered;
      }
    if (r.name == "softmax" || r.name == "masked_softmax")
      for (std::size_t col = 0; col < width; ++col)
        denominator += std::exp(p[0][row * width + col] + (r.name == "masked_softmax" ? p[1][col] : 0) - maximum);
    for (std::size_t col = 0; col < width; ++col) {
      const auto i = row * width + col;
      const double x = r.name == "concat" ? 0 : r.name == "gather" ? p[0][static_cast<std::size_t>(p[1][row]) * width + col]
          : r.name.starts_with("packed_") ? p[0][row * width * 2 + col] : p[0][i];
      double value = x;
      if (r.name == "concat") value = p[col < width / 2 ? 0 : 1][row * (width / 2) + col % (width / 2)];
      else if (r.name == "abs") value = std::abs(x);
      else if (r.name == "neg") value = -x;
      else if (r.name == "exp") value = std::exp(x);
      else if (r.name == "log") value = std::log(x);
      else if (r.name == "sqrt") value = std::sqrt(x);
      else if (r.name == "rsqrt") value = 1 / std::sqrt(x);
      else if (r.name == "tanh") value = std::tanh(x);
      else if (r.name == "sigmoid") value = 1 / (1 + std::exp(-x));
      else if (r.name == "silu" || r.name == "gelu" || r.name == "gelu_tanh") value = activation(r.name, x);
      else if (r.name == "softplus") value = std::max(x, 0.) + std::log1p(std::exp(-std::abs(x)));
      else if (r.name == "add") value = x + p[1][i];
      else if (r.name == "sub") value = x - p[1][i];
      else if (r.name == "mul") value = x * p[1][i];
      else if (r.name == "div") value = x / p[1][i];
      else if (r.name == "rms" || r.name == "weighted_rms") {
        value = x / std::sqrt(square / width + 1.e-6);
        if (r.name == "weighted_rms") value *= p[1][col];
      } else if (r.name == "layer" || r.name == "affine_layer") {
        value = (x - sum / width) / std::sqrt(variance / width + 1.e-6);
        if (r.name == "affine_layer") value = value * p[1][col] + p[2][col];
      } else if (r.name == "softmax" || r.name == "masked_softmax")
        value = std::exp(x + (r.name == "masked_softmax" ? p[1][col] : 0) - maximum) / denominator;
      else if (r.name.starts_with("packed_") || r.name.starts_with("separate_"))
        value = activation(r.name, x) * (r.name.starts_with("packed_") ? p[0][row * width * 2 + width + col] : p[1][i]);
      out[i] = value;
    }
  }
  return out;
}
