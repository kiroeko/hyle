#include <hyle/compute/low_precision.h>
#include <hyle/compute/operation.h>
#include <hyle/compute/vulkan.h>
#include <hyle/graph/graph.h>
#include <hyle/graph/vulkan.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace c = hyle::compute;
namespace v = hyle::compute::vulkan;
namespace g = hyle::graph;
namespace gv = hyle::graph::vulkan;

namespace {

bool all_zero(std::span<const float> values) {
  for (const float value : values)
    if (!std::isfinite(value) || value != 0.0F)
      return false;
  return true;
}

int direct_q4_k(v::Context &context) {
  constexpr std::int64_t m = 2;
  constexpr std::int64_t k = 256;
  constexpr std::int64_t n = 8;
  auto activation_desc =
      c::TensorDesc::dense(c::DataType::kFloat32, std::array{m, k});
  auto weight_desc = c::gguf_weight_description(
      c::GgufQuantizedType::kQ4K, c::DataType::kFloat32, k, n);
  if (!activation_desc || !weight_desc)
    return 10;

  std::vector<std::byte> packed(*weight_desc->required_bytes());
  auto weight = context.upload(*weight_desc, packed);
  if (!weight)
    return 11;
  auto program = v::prepare(context, c::MatMul{}, *activation_desc, *weight);
  if (!program)
    return 12;
  auto invocation = program->create_invocation();
  if (!invocation)
    return 13;

  for (int replay = 1; replay <= 2; ++replay) {
    std::vector<float> values(static_cast<std::size_t>(m * k),
                              0.125F * static_cast<float>(replay));
    auto activation =
        context.upload(*activation_desc, std::as_bytes(std::span(values)));
    if (!activation || !invocation->bind(*activation).ok())
      return 14;
    auto work = invocation->submit();
    if (!work)
      return 15;
    auto result = work->wait();
    if (!result || result->get()->tensors().size() != 1)
      return 16;
    std::vector<float> observed(static_cast<std::size_t>(m * n));
    if (!context
             .download(result->get()->tensors()[0],
                       std::as_writable_bytes(std::span(observed)))
             .ok() ||
        !all_zero(observed))
      return 17;
  }
  return 0;
}

int graph_iq4_xs(v::Context &context) {
  constexpr std::int64_t m = 2;
  constexpr std::int64_t k = 256;
  constexpr std::int64_t n = 8;
  auto activation_desc =
      c::TensorDesc::dense(c::DataType::kFloat32, std::array{m, k});
  auto weight_desc = c::gguf_weight_description(
      c::GgufQuantizedType::kIq4Xs, c::DataType::kFloat32, k, n);
  if (!activation_desc || !weight_desc)
    return 20;

  auto builder_result = g::GraphBuilder::create();
  if (!builder_result)
    return 21;
  auto builder = std::move(*builder_result);
  auto input = builder.input(*activation_desc);
  auto weight_value = builder.constant(*weight_desc);
  if (!input || !weight_value)
    return 22;
  auto output = builder.add(c::MatMul{}, *input, *weight_value);
  if (!output || !builder.publish(*output).ok())
    return 23;
  auto graph = std::move(builder).finish();
  if (!graph)
    return 24;

  std::vector<std::byte> packed(*weight_desc->required_bytes());
  auto weight = context.upload(*weight_desc, packed);
  if (!weight)
    return 25;
  gv::CompileReport report;
  auto program = gv::compile(context, *graph, std::array{*weight},
                             {.report = &report});
  if (!program || report.managed_region_count != 1 ||
      report.numerical_policy.find("GGUF") == std::string::npos)
    return 26;
  auto invocation = program->create_invocation();
  if (!invocation)
    return 27;

  for (int replay = 1; replay <= 2; ++replay) {
    std::vector<float> values(static_cast<std::size_t>(m * k),
                              0.25F * static_cast<float>(replay));
    auto activation =
        context.upload(*activation_desc, std::as_bytes(std::span(values)));
    if (!activation || !invocation->bind(*activation).ok())
      return 28;
    auto work = invocation->submit();
    if (!work)
      return 29;
    auto result = work->wait();
    if (!result || result->get()->tensors().size() != 1)
      return 30;
    std::vector<float> observed(static_cast<std::size_t>(m * n));
    if (!context
             .download(result->get()->tensors()[0],
                       std::as_writable_bytes(std::span(observed)))
             .ok() ||
        !all_zero(observed))
      return 31;
  }
  return 0;
}

} // namespace

int main() {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
#endif
  auto context = v::Context::create();
  if (!context)
    return 1;
  std::cout << context->environment().device_name << '\n';
  if (const int status = direct_q4_k(*context); status != 0)
    return status;
  return graph_iq4_xs(*context);
}
