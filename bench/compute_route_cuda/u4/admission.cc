#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
};
Request request(std::string op, c::DataType type, int tokens, int hidden,
                c::DataType output_type = c::DataType::kUnknown) {
  Request r{.operation = c::Operation(c::Copy{}), .inputs = {dense(type, {tokens, hidden})}, .output = {}};
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
void inspect(const std::string &id, const Request &r, const Candidate &candidate) {
  auto context = c::Context::create(take(c::cuda::device(0)), candidate.provider);
  if (!context) throw std::runtime_error(std::string(context.error().message()));
  auto memory = take(c::cuda::MemoryResource::create(*context, c::cuda::MemoryKind::kDevice));
  auto prepared = c::prepare(*context, r.operation, r.inputs, std::span(&r.output, 1), memory.ref());
  if (!prepared) {
    std::printf("{\"request\":\"%s\",\"provider\":\"%s\",\"admission\":\"%s\"}\n",
        id.c_str(), candidate.name, prepared.error().IsUnimplemented() ? "DECLINED" : "ERROR");
    if (!prepared.error().IsUnimplemented())
      throw std::runtime_error(std::string(prepared.error().message()));
    return;
  }
  // Instantiate original native programs too: declaration alone is not admission.
  auto workspace = take(hyle::mem::Buffer::create(memory.ref(), prepared->workspace_requirement().size,
      prepared->workspace_requirement().alignment));
  auto slot = prepared->workspace_requirement().size == 0
      ? prepared->create_invocation() : prepared->create_invocation(workspace.view());
  if (!slot) throw std::runtime_error(std::string(slot.error().message()));
  const auto runtime = prepared->runtime_properties();
  std::printf("{\"request\":\"%s\",\"provider\":\"%s\",\"admission\":\"PREPARED\",\"capture\":%s,\"prepared_replay\":%s,\"blocks_host\":%s,\"workspace\":%zu,\"private_known\":%s}\n",
      id.c_str(), candidate.name, runtime.capture_compatible ? "true" : "false",
      runtime.replay_class == c::ProviderReplayClass::kPrepared ? "true" : "false",
      runtime.blocks_host ? "true" : "false", prepared->workspace_requirement().size,
      prepared->invocation_private_bytes().has_value() ? "true" : "false");
}
int main() {
  try {
    const std::array candidates{Candidate{"hyle", c::cuda_native::provider()},
        Candidate{"ggml", c::ggml_cuda::provider()}, Candidate{"sglang", c::cuda_native::sglang_provider()},
        Candidate{"torch", c::torch_cuda::provider()}, Candidate{"cccl", c::cuda_native::cccl_provider()}};
    const std::array ops{"copy", "abs", "neg", "exp", "log", "sqrt", "rsqrt", "tanh", "sigmoid",
        "silu", "gelu", "gelu_tanh", "softplus", "add", "sub", "mul", "div", "rms", "weighted_rms",
        "layer", "affine_layer", "sum", "mean", "softmax", "masked_softmax", "concat", "packed_silu",
        "separate_silu", "packed_gelu", "separate_gelu", "packed_gelu_tanh", "separate_gelu_tanh"};
    const std::array types{c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16};
    const std::array names{"f32", "f16", "bf16"};
    for (auto op : ops)
      for (std::size_t t = 0; t < types.size(); ++t)
        for (int tokens : {1, 32}) {
          const int width = std::string(op).starts_with("packed_") || std::string(op).starts_with("separate_")
              ? 6144 : tokens == 1 ? 2048 : 4096;
          const auto id = std::string(op) + "-" + names[t] + "-t" + std::to_string(tokens) + "-h" + std::to_string(width);
          const auto r = request(op, types[t], tokens, width);
          for (const auto &candidate : candidates) inspect(id, r, candidate);
        }
    for (std::size_t source = 0; source < types.size(); ++source)
      for (std::size_t target = 0; target < types.size(); ++target) {
        if (source == target) continue;
        for (int tokens : {1, 32}) {
          const int width = tokens == 1 ? 2048 : 4096;
          const auto id = std::string("cast-") + names[source] + "-" + names[target] + "-t" + std::to_string(tokens) + "-h" + std::to_string(width);
          const auto r = request("cast", types[source], tokens, width, types[target]);
          for (const auto &candidate : candidates) inspect(id, r, candidate);
        }
      }
    for (int tokens : {1, 32}) {
      const int width = tokens == 1 ? 2048 : 4096;
      const auto id = std::string("gather-f32-t") + std::to_string(tokens) + "-h" + std::to_string(width);
      const auto r = request("gather", c::DataType::kFloat32, tokens, width);
      for (const auto &candidate : candidates) inspect(id, r, candidate);
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what()); return 2;
  }
}
