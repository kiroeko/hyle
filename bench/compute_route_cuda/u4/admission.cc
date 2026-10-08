#include "matrix_request.h"

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
  auto queue = take(context->create_queue());
  std::vector<hyle::mem::Buffer> buffers;
  std::vector<c::ConstTensorView> inputs;
  for (const auto &desc : r.inputs) {
    buffers.push_back(take(hyle::mem::Buffer::create(memory.ref(), take(desc.required_bytes()), 256)));
    inputs.push_back(take(c::ConstTensorView::create(desc, buffers.back().view())));
  }
  auto output = take(hyle::mem::Buffer::create(memory.ref(), take(r.output.required_bytes()), 256));
  auto view = take(c::TensorView::create(r.output, output.view()));
  for (int seed : {5, 17}) {
    const auto values = payload(r, seed);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto status = buffers[i].view().write(pack(r.inputs[i], values[i]));
      if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    }
    auto done = take(slot->submit(queue, inputs, std::span(&view, 1)));
    const auto terminal = done.wait();
    if (!terminal.ok()) throw std::runtime_error(std::string(terminal.message()));
    std::vector<std::byte> bytes(output.size());
    const auto read = output.view().read(bytes);
    if (!read.ok()) throw std::runtime_error(std::string(read.message()));
    const auto actual = unpack(r.output.data_type(), bytes);
    const auto expected = oracle(r, values);
    const double tolerance = r.output.data_type() == c::DataType::kBFloat16 ? .008
        : r.output.data_type() == c::DataType::kFloat16 ? .001 : 3.e-5;
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance * (1 + std::abs(expected[i]))) {
        std::printf("{\"request\":\"%s\",\"provider\":\"%s\",\"admission\":\"NUMERIC_FAIL\",\"index\":%zu,\"actual\":%.9g,\"expected\":%.9g}\n",
            id.c_str(), candidate.name, i, static_cast<double>(actual[i]), expected[i]);
        return;
      }
  }
  const auto runtime = prepared->runtime_properties();
  std::printf("{\"request\":\"%s\",\"provider\":\"%s\",\"admission\":\"CORRECT\",\"capture\":%s,\"prepared_replay\":%s,\"blocks_host\":%s,\"workspace\":%zu,\"private_known\":%s}\n",
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
