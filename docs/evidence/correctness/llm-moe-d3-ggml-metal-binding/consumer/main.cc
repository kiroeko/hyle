#include <hyle/compute/apple.h>
#include <hyle/compute/low_precision.h>
#include <hyle/compute/operation.h>
#include <hyle/graph/apple.h>
#include <hyle/graph/graph.h>

#include <mlx/array.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace c = hyle::compute;
namespace a = hyle::compute::apple;
namespace g = hyle::graph;
namespace ga = hyle::graph::apple;
namespace mx = mlx::core;

namespace {

bool is_zero(const mx::array &value) {
  if (value.dtype() != mx::float32)
    return false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value.data<float>()[index] != 0.0F)
      return false;
  }
  return true;
}

int direct_q4_k() {
  constexpr int m = 2;
  constexpr int k = 256;
  constexpr int n = 8;
  auto context = a::Context::create();
  if (!context)
    return 10;
  auto weight = c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
                                           c::DataType::kFloat32, k, n);
  if (!weight)
    return 11;
  auto packed_spec = a::input_spec(*weight);
  if (!packed_spec)
    return 12;
  const std::array specs{a::InputSpec{{m, k}, mx::float32}, *packed_spec};
  auto program = context->prepare(c::MatMul{}, specs);
  if (!program)
    return 13;
  auto invocation = program->create_invocation();
  if (!invocation)
    return 14;
  std::vector<float> activation(m * k, 0.25F);
  std::vector<std::uint8_t> packed(weight->required_bytes().value());
  const std::array inputs{
      mx::array(activation.data(), mx::Shape{m, k}),
      mx::array(packed.data(), mx::Shape{static_cast<int>(packed.size())},
                mx::uint8)};
  for (int replay = 0; replay < 2; ++replay) {
    if (!invocation->bind(inputs).ok())
      return 15;
    auto work = invocation->submit();
    if (!work)
      return 16;
    auto result = work->wait();
    if (!result || result->get()->arrays().size() != 1 ||
        result->get()->arrays()[0].shape() != mx::Shape{m, n} ||
        !is_zero(result->get()->arrays()[0]))
      return 17;
  }
  return 0;
}

int graph_iq4_xs() {
  constexpr int m = 2;
  constexpr int k = 256;
  constexpr int n = 8;
  auto context = a::Context::create();
  if (!context)
    return 20;
  auto activation = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{m, k});
  auto weight = c::gguf_weight_description(c::GgufQuantizedType::kIq4Xs,
                                           c::DataType::kFloat32, k, n);
  if (!activation || !weight)
    return 21;
  auto builder_result = g::GraphBuilder::create();
  if (!builder_result)
    return 22;
  auto builder = std::move(*builder_result);
  auto input = builder.input(*activation);
  auto weights = builder.constant(*weight);
  if (!input || !weights)
    return 23;
  auto output = builder.add(c::MatMul{}, *input, *weights);
  if (!output || !builder.publish(*output).ok())
    return 24;
  auto graph = std::move(builder).finish();
  if (!graph)
    return 25;
  std::vector<std::uint8_t> packed(weight->required_bytes().value());
  const std::array constants{
      mx::array(packed.data(), mx::Shape{static_cast<int>(packed.size())},
                mx::uint8)};
  ga::CompileReport report;
  auto program = ga::compile(*context, *graph, constants, {.report = &report});
  if (!program || report.implementation_policy.find("ggml Metal") ==
                      std::string::npos)
    return 26;
  auto invocation = program->create_invocation();
  if (!invocation)
    return 27;
  std::vector<float> host(m * k, 0.5F);
  const std::array inputs{mx::array(host.data(), mx::Shape{m, k})};
  for (int replay = 0; replay < 2; ++replay) {
    if (!invocation->bind(inputs).ok())
      return 28;
    auto work = invocation->submit();
    if (!work)
      return 29;
    auto result = work->wait();
    if (!result || result->get()->arrays().size() != 1 ||
        result->get()->arrays()[0].shape() != mx::Shape{m, n} ||
        !is_zero(result->get()->arrays()[0]))
      return 30;
  }
  return 0;
}

} // namespace

int main() {
  if (const int status = direct_q4_k(); status != 0)
    return status;
  return graph_iq4_xs();
}
