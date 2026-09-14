#include "kernel.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <utility>

#include <hyle/compute/cpu.h>
#include <hyle/compute/route_profile.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/buffer.h>

namespace {

// This second journey exercises the public domain API through the same source,
// static and shared package boundaries; it does not use a private Provider SPI.
hyle::Status bounded_normalization() {
  using namespace hyle::compute;
  using namespace hyle::graph;
  HYLE_ASSIGN_OR_RETURN(auto context, cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto builder, GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows, builder.dimension({0, 4, 1}));
  HYLE_ASSIGN_OR_RETURN(
      auto spec,
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)}));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(RmsNorm{}, spec, input));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(
      auto compiled,
      compile(graph, context,
              {.shape_policy = CompileShapePolicy::kDeclaredDomain}));
  if (!compiled.executable.is_shape_domain() ||
      !compiled.report.is_shape_domain() ||
      compiled.executable.specialization().valid())
    return hyle::Status::Internal("domain report has an exact signature");
  graph = {};
  HYLE_ASSIGN_OR_RETURN(auto source, hyle::mem::Buffer::create(
                                         hyle::mem::system_resource(), 64, 64));
  HYLE_ASSIGN_OR_RETURN(
      auto destination,
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 64, 64));
  std::array<float, 16> values;
  values.fill(1.0F);
  HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(values))));
  for (std::int64_t count : {0, 1, 4}) {
    const std::array shape{ShapeBinding{rows, count}};
    HYLE_ASSIGN_OR_RETURN(auto run,
                          compiled.executable.create_invocation(shape));
    HYLE_ASSIGN_OR_RETURN(auto binding, run.specialization().binding());
    if (binding != shape[0])
      return hyle::Status::Internal("domain instance has the wrong shape");
    HYLE_ASSIGN_OR_RETURN(
        auto description,
        TensorDesc::dense(DataType::kFloat32,
                          std::array{count, std::int64_t{4}}));
    HYLE_ASSIGN_OR_RETURN(auto in,
                          ConstTensorView::create(description, source.view()));
    HYLE_ASSIGN_OR_RETURN(auto out,
                          TensorView::create(description, destination.view()));
    HYLE_RETURN_IF_ERROR(run.bind(input, in));
    HYLE_RETURN_IF_ERROR(run.bind(output, out));
    for (int repeat = 0; repeat < 3; ++repeat) {
      HYLE_ASSIGN_OR_RETURN(auto completion, run.submit(queue));
      HYLE_RETURN_IF_ERROR(completion.wait());
    }
    std::array<float, 16> result{};
    HYLE_RETURN_IF_ERROR(out.storage().read(std::as_writable_bytes(
        std::span(result).first(static_cast<std::size_t>(count * 4)))));
    for (std::int64_t i = 0; i < count * 4; ++i)
      if (std::abs(result[static_cast<std::size_t>(i)] -
                   1.0F / std::sqrt(1.0F + RmsNorm{}.epsilon)) > 1.0e-5F)
        return hyle::Status::Internal("domain normalization output mismatch");
  }
  return hyle::Status::Ok();
}

} // namespace

int main() {
  using namespace hyle::compute;
  // This example has no qualified timing dataset. Missing identity must not
  // turn into a zero-cost match, including through an installed shared SDK.
  const auto unqualified = match_route_profile({}, {});
  if (unqualified.reason != ProfileMatchReason::kIncompleteIdentity ||
      unqualified.cost || find_route_profile({}, {}).cost ||
      !ProfileTimingInterval{}.valid())
    return 20;
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
  // A report can expose structural facts without pretending this deployment
  // has a qualified build/device/workload profile.
  const hyle::graph::CompileProfileContext profile_context;
  auto compiled = hyle::graph::compile(*graph, *context,
                                       {.constants = constants,
                                        .allowed_providers = allowed,
                                        .node_providers = required,
                                        .max_operation_proposals = 1,
                                        .profile_context = &profile_context});
  if (!compiled || compiled->report.search().operation_proposals != 1 ||
      compiled->report.search().placement_attempts != 1 ||
      compiled->report.search().optimization_limited())
    return 14;
  if (compiled->report.profile_identity().route.valid() ||
      compiled->report.profile_identity().inputs.size() != 1 ||
      compiled->report.search().selected_profile_cost)
    return 21;
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
  const auto bounded = bounded_normalization();
  if (!bounded.ok()) {
    const auto message = bounded.message();
    std::fprintf(stderr, "bounded graph: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    return 22;
  }
  return 0;
}
