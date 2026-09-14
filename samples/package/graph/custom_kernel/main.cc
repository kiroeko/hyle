#include "kernel.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <utility>

#include <hyle/compute/backends/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

int main() {
  using namespace hyle::compute;
  auto provider = example::make_provider();
  if (!provider)
    return 1;
  auto context = Context::create(cpu::device(), std::move(*provider));
  if (!context)
    return 2;
  auto queue = context->create_queue();
  auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4});
  auto source = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64);
  auto target = hyle::mem::Buffer::create(hyle::mem::system_resource(), 16, 64);
  if (!queue || !desc || !source || !target)
    return 3;
  const std::array<float, 4> values{1, 2, 3, 4};
  if (!source->view().write(std::as_bytes(std::span(values))).ok())
    return 4;
  auto input = ConstTensorView::create(*desc, source->view());
  auto output = TensorView::create(*desc, target->view());
  if (!input || !output)
    return 5;
  const auto operation = Operation::custom(example::Affine{3});
  auto prepared =
      prepare(*context, operation, std::array{*input}, std::array{*output});
  if (!prepared) {
    const auto message = prepared.error().message();
    std::fprintf(stderr, "custom prepare: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    return 6;
  }
  auto invocation = prepared->create_invocation();
  if (!invocation)
    return 7;
  auto done = invocation->submit(*queue, *input, *output);
  if (!done || !done->wait().ok())
    return 8;
  std::array<float, 4> actual{};
  if (!target->view().read(std::as_writable_bytes(std::span(actual))).ok() ||
      actual != std::array<float, 4>{5, 7, 9, 11})
    return 9;

  auto builder = hyle::graph::GraphBuilder::create();
  if (!builder)
    return 10;
  auto in = builder->constant(*desc);
  if (!in)
    return 11;
  auto out = builder->add(operation, *in);
  if (!out || !builder->publish(*out).ok())
    return 12;
  auto graph = std::move(*builder).finish();
  if (!graph)
    return 13;
  const auto family = context->provider_identity(0);
  const auto instance = context->provider_instance_id(0);
  const auto node = graph->node_id(0);
  if (!family || !instance || !node)
    return 14;
  const std::array allowed{family->id};
  const std::array required{
      hyle::graph::NodeProviderConstraint{*node, family->id, *instance}};
  const std::array constants{hyle::graph::ConstantBinding{
      *in, *input, hyle::graph::ConstantPolicy::kMaterialize}};
  auto compiled = hyle::graph::compile(*graph, *context,
                                       {.constants = constants,
                                        .allowed_providers = allowed,
                                        .node_providers = required,
                                        .max_operation_proposals = 1});
  if (!compiled || compiled->report.search().operation_proposals != 1 ||
      compiled->report.search().placement_attempts != 1 ||
      compiled->report.search().optimization_limited())
    return 14;
  if (!source->reset().ok() ||
      compiled->report.borrowed_constant_count() != 0 ||
      compiled->report.materialized_constant_bytes() != 16)
    return 19;
  const std::string provider_name(compiled->report.provider().name);
  const std::string provider_version(compiled->report.provider().version);
  auto run = compiled->executable.create_invocation();
  if (!run || !run->bind(*out, *output).ok())
    return 15;
  compiled->executable = {}; // Invocation retains the kernel/factory owner.
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto complete = run->submit(*queue);
    if (!complete || !complete->wait().ok())
      return 16;
    if (!target->view().read(std::as_writable_bytes(std::span(actual))).ok() ||
        actual != std::array<float, 4>{5, 7, 9, 11})
      return 17;
  }
  // Diagnostic reports own their text, not the factory or execution resources.
  *done = {};
  *invocation = {};
  *prepared = {};
  *run = {};
  *queue = {};
  *context = {};
  if (compiled->report.provider().name != provider_name ||
      compiled->report.step(0)->provider.version != provider_version ||
      compiled->report.step(0)->profile_key !=
          hyle::compute::ProviderRouteKey{1} ||
      compiled->report.constant()->policy !=
          hyle::graph::ConstantPolicy::kMaterialize)
    return 18;
  return 0;
}
