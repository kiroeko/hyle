#include "cuda_kernel.h"
#include "kernel.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>

#include <hyle/compute/backends/cuda.h>
#include <hyle/error/result.h>
#include <hyle/graph/executable.h>

namespace {
using namespace hyle::compute;
using hyle::Result;
using hyle::Status;

constexpr ProviderRuntimeProperties runtime{
    .replay_class = ProviderReplayClass::kPrepared, .capture_compatible = true};

struct Configuration final {
  unsigned block_size = 128;
  bool report_after_launch = false;
  mutable std::atomic<unsigned> proposals{0}, preparations{0};
};

struct Kernel final {
  const Configuration *configuration;
  example::Affine operation;
  struct Invocation {};

  Result<Invocation> create_invocation(hyle::mem::BufferView,
                                       hyle::mem::AllocatorRef) const noexcept {
    return Invocation{};
  }
  AliasSupport alias_support(std::size_t output,
                             std::size_t input) const noexcept {
    return {output == 0 && input == 0, 0};
  }
  KernelCommandResult
  enqueue(Invocation &, Queue &queue, std::span<const ConstTensorView> inputs,
          std::span<const TensorView> outputs) const noexcept {
    auto stream = cuda::stream(queue);
    if (!stream)
      return KernelCommandResult::rejected(std::move(stream).error());
    auto input = cuda::pointer<float>(inputs[0].storage());
    if (!input)
      return KernelCommandResult::rejected(std::move(input).error());
    auto output = cuda::pointer<float>(outputs[0].storage());
    if (!output)
      return KernelCommandResult::rejected(std::move(output).error());
    const auto count = inputs[0].description().element_count().value();
    if (count == 0)
      return KernelCommandResult::no_op();
    // The launch uses the Queue's stream. No nested cuda::submit, device-wide
    // synchronization or extra Completion; an uncertain error retains work.
    auto status = cuda::status(launch_affine(*stream, *input, *output, count,
                                             operation.scale, operation.bias,
                                             configuration->block_size));
    if (status.ok() && configuration->report_after_launch)
      status = Status::DataLoss("injected error after accepted native launch");
    return KernelCommandResult::accepted_command(std::move(status));
  }
};

struct Factory final {
  std::shared_ptr<const Configuration> configuration;
  Result<KernelProposal>
  propose(const KernelRequest &request, std::span<ResourceRequirement> inputs,
          std::span<ResourceRequirement> outputs) const noexcept {
    ++configuration->proposals;
    if (!request.operation.custom_as<example::Affine>() ||
        request.inputs.size() != 1 || request.outputs.size() != 1 ||
        request.inputs[0] != request.outputs[0] ||
        !request.inputs[0].is_dense() ||
        request.inputs[0].data_type() != DataType::kFloat32 ||
        request.inputs[0].element_count().value() > 1048576)
      return std::unexpected(
          Status::Unimplemented("example requires small dense f32 affine"));
    inputs[0] = outputs[0] = {.required_properties =
                                  hyle::mem::MemoryProperty::kDeviceAccessible,
                              .minimum_alignment = alignof(float)};
    return KernelProposal{.runtime = runtime,
                          .cost = {.workspace = WorkspaceRequirement{},
                                   .persistent_bytes = 0,
                                   .copy_bytes = 0,
                                   .repack_bytes = 0,
                                   .synchronization_count = 0,
                                   .dispatch_count = 1},
                          .profile_key = ProviderRouteKey{1}};
  }
  KernelPreparation prepare(const KernelPrepareRequest &request,
                            hyle::mem::AllocatorRef allocator) const noexcept {
    ++configuration->preparations;
    const auto operation = request.operation.custom_as<example::Affine>();
    if (!operation)
      return KernelPreparation::error(
          Status::Internal("selected schema changed"));
    auto kernel = make_custom_kernel(
        Kernel{configuration.get(), *operation},
        {.runtime = runtime, .persistent_bytes = 0}, allocator);
    if (!kernel)
      return KernelPreparation::error(std::move(kernel).error());
    return KernelPreparation::prepared(std::move(*kernel));
  }
};

Status exercise(bool report_after_launch) {
  auto configuration = std::make_shared<Configuration>();
  configuration->report_after_launch = report_after_launch;
  const std::weak_ptr<const Configuration> lifetime = configuration;
  {
    HYLE_ASSIGN_OR_RETURN(
        auto provider,
        make_custom_provider(
            {{0x4558414d504c45ULL, 3}, "example-affine-cuda", "1"},
            cuda::kBackendId, Factory{configuration}));
    HYLE_ASSIGN_OR_RETURN(auto device, cuda::device(0));
    HYLE_ASSIGN_OR_RETURN(auto context,
                          Context::create(device, std::move(provider)));
    HYLE_ASSIGN_OR_RETURN(auto memory, cuda::MemoryResource::create(
                                           context, cuda::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto second_queue, context.create_queue());
    HYLE_RETURN_IF_ERROR(queue.reserve_completions(4));
    HYLE_RETURN_IF_ERROR(second_queue.reserve_completions(4));
    constexpr std::size_t count = 257; // includes a partial final CUDA block
    HYLE_ASSIGN_OR_RETURN(
        auto desc, TensorDesc::dense(DataType::kFloat32,
                                     std::array<std::int64_t, 1>{count}));
    HYLE_ASSIGN_OR_RETURN(
        auto input,
        hyle::mem::Buffer::create(memory.ref(), count * sizeof(float), 256));
    HYLE_ASSIGN_OR_RETURN(
        auto output,
        hyle::mem::Buffer::create(memory.ref(), count * sizeof(float), 256));
    HYLE_ASSIGN_OR_RETURN(
        auto second_output,
        hyle::mem::Buffer::create(memory.ref(), count * sizeof(float), 256));
    std::array<float, count> values{}, actual{};
    for (std::size_t index = 0; index < count; ++index)
      values[index] = static_cast<float>(index);
    HYLE_RETURN_IF_ERROR(input.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto in, ConstTensorView::create(desc, input.view()));
    HYLE_ASSIGN_OR_RETURN(auto out, TensorView::create(desc, output.view()));
    HYLE_ASSIGN_OR_RETURN(auto out2,
                          TensorView::create(desc, second_output.view()));
    const auto operation =
        Operation::custom(example::Affine{.bias = 3, .scale = 2});
    HYLE_ASSIGN_OR_RETURN(
        auto prepared,
        prepare(context, operation, std::array{in}, std::array{out}));
    HYLE_ASSIGN_OR_RETURN(auto direct, prepared.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto other, prepared.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto first, direct.submit(queue, in, out));
    HYLE_ASSIGN_OR_RETURN(auto second, other.submit(second_queue, in, out2));
    for (auto *completion : {&first, &second}) {
      const auto status = completion->wait();
      if (report_after_launch ? !status.IsDataLoss() : !status.ok())
        return Status::Internal(
            "custom native completion lost its terminal result");
    }
    HYLE_RETURN_IF_ERROR(
        output.view().read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t index = 0; index < count; ++index)
      if (actual[index] != values[index] * 2 + 3)
        return Status::DataLoss("custom native direct output mismatch");
    HYLE_RETURN_IF_ERROR(
        second_output.view().read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t index = 0; index < count; ++index)
      if (actual[index] != values[index] * 2 + 3)
        return Status::DataLoss("custom native concurrent output mismatch");

    if (!report_after_launch) {
      for (const bool materialize : {false, true}) {
        HYLE_ASSIGN_OR_RETURN(auto builder,
                              hyle::graph::GraphBuilder::create());
        HYLE_ASSIGN_OR_RETURN(auto graph_in, materialize
                                                 ? builder.constant(desc)
                                                 : builder.input(desc));
        HYLE_ASSIGN_OR_RETURN(auto intermediate,
                              builder.add(operation, graph_in));
        HYLE_ASSIGN_OR_RETURN(auto graph_out,
                              builder.add(operation, intermediate));
        HYLE_RETURN_IF_ERROR(builder.publish(graph_out));
        HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
        const std::array storage{
            hyle::graph::StorageDomain{.resource = memory.ref()}};
        const std::array targets{hyle::graph::ExecutionTarget{
            .context = &context,
            .default_storage = hyle::graph::StorageDomainId{0}}};
        HYLE_ASSIGN_OR_RETURN(auto family, context.provider_identity(0));
        HYLE_ASSIGN_OR_RETURN(auto instance, context.provider_instance_id(0));
        HYLE_ASSIGN_OR_RETURN(auto first_node, graph.node_id(0));
        const std::array required{hyle::graph::NodeProviderConstraint{
            first_node, family.id, instance}};
        const std::array constants{hyle::graph::ConstantBinding{
            graph_in, in, hyle::graph::ConstantPolicy::kMaterialize}};
        HYLE_ASSIGN_OR_RETURN(
            auto compiled,
            hyle::graph::compile(
                graph, {.storage_domains = storage, .targets = targets},
                {.constants =
                     materialize
                         ? std::span<const hyle::graph::ConstantBinding>(
                               constants)
                         : std::span<const hyle::graph::ConstantBinding>{},
                 .node_providers = required,
                 .max_operation_proposals = 2}));
        if (materialize) {
          HYLE_RETURN_IF_ERROR(input.reset());
          if (compiled.report.borrowed_constant_count() != 0 ||
              compiled.report.materialized_constant_bytes() !=
                  count * sizeof(float))
            return Status::Internal("custom CUDA constant ownership mismatch");
        }
        if (compiled.report.search().operation_proposals != 2 ||
            compiled.report.search().placement_attempts != 2 ||
            compiled.report.search().optimization_limited())
          return Status::Internal("custom CUDA candidate discovery mismatch");
        for (std::size_t index = 0; index < compiled.report.step_count();
             ++index) {
          HYLE_ASSIGN_OR_RETURN(auto step, compiled.report.step(index));
          if (step.profile_key != ProviderRouteKey{1})
            return Status::Internal("custom CUDA implementation key changed");
        }
        HYLE_ASSIGN_OR_RETURN(auto invocation,
                              compiled.executable.create_invocation());
        if (!materialize)
          HYLE_RETURN_IF_ERROR(invocation.bind(graph_in, in));
        HYLE_RETURN_IF_ERROR(invocation.bind(graph_out, out));
        const std::array queues{hyle::graph::QueueBinding{
            .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
        HYLE_RETURN_IF_ERROR(invocation.prepare_replay(
            queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}));
        if (invocation.replay_route() != hyle::graph::ReplayRoute::kCaptured)
          return Status::Internal("custom CUDA graph was not captured");
        const auto proposals = configuration->proposals.load();
        const auto preparations = configuration->preparations.load();
        compiled.executable = {};
        for (int iteration = 0; iteration < 3; ++iteration) {
          HYLE_ASSIGN_OR_RETURN(auto done, invocation.submit(queues));
          HYLE_RETURN_IF_ERROR(done.wait());
          HYLE_RETURN_IF_ERROR(
              output.view().read(std::as_writable_bytes(std::span(actual))));
          for (std::size_t index = 0; index < count; ++index)
            if (actual[index] != values[index] * 4 + 9)
              return Status::DataLoss("custom captured graph output mismatch");
        }
        if (configuration->proposals != proposals ||
            configuration->preparations != preparations)
          return Status::Internal(
              "replay unexpectedly prepared a kernel again");
      }
    }
    configuration.reset();
    if (lifetime.expired())
      return Status::Internal(
          "custom CUDA configuration released before Context");
  }
  if (!lifetime.expired())
    return Status::Internal("custom CUDA configuration leaked");
  return Status::Ok();
}

Status exercise_transfer_pipeline() {
  namespace graph = hyle::graph;
  auto configuration = std::make_shared<Configuration>();
  HYLE_ASSIGN_OR_RETURN(
      auto provider,
      make_custom_provider({{0x4558414d504c45ULL, 3}, "example-affine-cuda", "1"},
                           cuda::kBackendId, Factory{configuration}));
  HYLE_ASSIGN_OR_RETURN(auto device, cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context,
                        Context::create(device, std::move(provider)));
  HYLE_ASSIGN_OR_RETURN(auto memory, cuda::MemoryResource::create(
                                         context, cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(
      auto pinned,
      cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost));
  constexpr std::size_t count = 257;
  HYLE_ASSIGN_OR_RETURN(auto description,
                        TensorDesc::dense(DataType::kFloat32,
                                          std::array<std::int64_t, 1>{count}));
  HYLE_ASSIGN_OR_RETURN(auto builder, graph::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input_id, builder.input(description));
  const auto operation =
      Operation::custom(example::Affine{.bias = 3, .scale = 2});
  HYLE_ASSIGN_OR_RETURN(auto intermediate, builder.add(operation, input_id));
  HYLE_ASSIGN_OR_RETURN(auto output_id, builder.add(operation, intermediate));
  HYLE_RETURN_IF_ERROR(builder.publish(output_id));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  const std::array storage{graph::StorageDomain{memory.ref()},
                           graph::StorageDomain{pinned.ref()}};
  const std::array targets{
      graph::ExecutionTarget{&context, graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{graph::ExecutionTargetId{0},
                           graph::ExecutionLaneRole::kCompute},
      graph::ExecutionLane{graph::ExecutionTargetId{0},
                           graph::ExecutionLaneRole::kTransfer}};
  const std::array external{
      graph::ExternalResourceContract{input_id, graph::StorageDomainId{1}},
      graph::ExternalResourceContract{output_id, graph::StorageDomainId{1}}};
  HYLE_ASSIGN_OR_RETURN(
      auto compiled,
      graph::compile(
          logical,
          {.storage_domains = storage, .targets = targets, .lanes = lanes},
          {.external_resources = external,
           .provider_replay = graph::ProviderReplayPolicy::kRequirePrepared}));
  if (compiled.report.target_count() != 1 ||
      compiled.report.storage_domain_count() != 2 ||
      compiled.report.lane_count() != 2 ||
      compiled.report.segment_count() != 3 ||
      compiled.report.step_count() != 4 ||
      compiled.report.copy_bytes() != 2 * count * sizeof(float) ||
      compiled.executable.required_queue_count() != 2)
    return Status::Internal(
        "CUDA copy/compute/copy plan is not the declared topology");

  std::array<hyle::mem::Buffer, 2> inputs, outputs;
  std::array<Queue, 2> compute_queues, transfer_queues;
  std::array<graph::Invocation, 2> runs;
  std::array<std::array<graph::QueueBinding, 2>, 2> bindings;
  for (std::size_t i = 0; i < runs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(
        inputs[i],
        hyle::mem::Buffer::create(pinned.ref(), count * sizeof(float), 256));
    HYLE_ASSIGN_OR_RETURN(
        outputs[i],
        hyle::mem::Buffer::create(pinned.ref(), count * sizeof(float), 256));
    HYLE_ASSIGN_OR_RETURN(compute_queues[i], context.create_queue());
    HYLE_ASSIGN_OR_RETURN(transfer_queues[i], context.create_queue());
    HYLE_ASSIGN_OR_RETURN(runs[i], compiled.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(
        auto input, ConstTensorView::create(description, inputs[i].view()));
    HYLE_ASSIGN_OR_RETURN(auto output,
                          TensorView::create(description, outputs[i].view()));
    HYLE_RETURN_IF_ERROR(runs[i].bind(input_id, input));
    HYLE_RETURN_IF_ERROR(runs[i].bind(output_id, output));
    bindings[i] = {
        graph::QueueBinding{graph::ExecutionLaneId{0}, &compute_queues[i]},
        graph::QueueBinding{graph::ExecutionLaneId{1}, &transfer_queues[i]}};
  }
  // The runs retain the immutable plan; buffers/resources and the still-open
  // Context remain caller-owned. Each run gets its own scratch and capture.
  logical = graph::Graph{};
  compiled = graph::CompiledGraph{};
  for (const auto mode :
       {graph::ReplayMode::kCommands, graph::ReplayMode::kRequireCapture}) {
    for (std::size_t i = 0; i < runs.size(); ++i) {
      HYLE_RETURN_IF_ERROR(runs[i].prepare_replay(
          bindings[i],
          {.mode = mode, .host_wait = graph::HostWaitPolicy::kReject}));
      HYLE_ASSIGN_OR_RETURN(auto report, runs[i].replay_report());
      const auto expected = mode == graph::ReplayMode::kCommands
                                ? graph::ReplayRoute::kCommands
                                : graph::ReplayRoute::kCaptured;
      if (report.route != expected || report.queued_transfer_count != 2 ||
          report.device_wait_dependency_count < 2 ||
          report.host_wait_dependency_count != 0)
        return Status::Internal(
            "CUDA pipeline did not preserve asynchronous lane dependencies");
    }
    const auto proposals = configuration->proposals.load();
    const auto preparations = configuration->preparations.load();
    for (std::size_t replay = 0; replay < 3; ++replay) {
      std::array<std::array<float, count>, 2> values;
      std::array<Completion, 2> completions;
      for (std::size_t i = 0; i < runs.size(); ++i) {
        for (std::size_t j = 0; j < count; ++j)
          values[i][j] =
              static_cast<float>(replay * 1000 + i * 17 + j) * 0.25F - 32;
        HYLE_RETURN_IF_ERROR(
            inputs[i].view().write(std::as_bytes(std::span(values[i]))));
        std::array<float, count> poison;
        poison.fill(12345.0F);
        HYLE_RETURN_IF_ERROR(
            outputs[i].view().write(std::as_bytes(std::span(poison))));
        HYLE_ASSIGN_OR_RETURN(completions[i], runs[i].submit(bindings[i]));
      }
      // Both independent queue pairs have work in flight. Only the final
      // per-run Completion, not a device-wide synchronization, releases host
      // IO.
      for (std::size_t i = 0; i < runs.size(); ++i) {
        HYLE_RETURN_IF_ERROR(completions[i].wait());
        std::array<float, count> actual;
        HYLE_RETURN_IF_ERROR(
            outputs[i].view().read(std::as_writable_bytes(std::span(actual))));
        for (std::size_t j = 0; j < count; ++j)
          if (actual[j] != values[i][j] * 4 + 9)
            return Status::DataLoss(
                "CUDA multi-lane replay produced stale or shared output");
      }
    }
    if (configuration->proposals != proposals ||
        configuration->preparations != preparations)
      return Status::Internal(
          "CUDA pipeline unexpectedly repeated cold routing");
  }
  return Status::Ok();
}
} // namespace

int main() {
  for (bool fail_after_launch : {false, true}) {
    const auto status = exercise(fail_after_launch);
    if (!status.ok()) {
      const auto message = status.message();
      std::fprintf(stderr, "custom CUDA journey: %.*s\n",
                   static_cast<int>(message.size()), message.data());
      return 1;
    }
  }
  const auto pipeline = exercise_transfer_pipeline();
  if (!pipeline.ok()) {
    const auto message = pipeline.message();
    std::fprintf(stderr, "custom CUDA pipeline: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    return 1;
  }
}
