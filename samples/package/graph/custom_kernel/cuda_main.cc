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
                                   .dispatch_count = 1}};
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
      HYLE_ASSIGN_OR_RETURN(auto builder, hyle::graph::GraphBuilder::create());
      HYLE_ASSIGN_OR_RETURN(auto graph_in, builder.input(desc));
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
      const std::array required{
          hyle::graph::NodeProviderConstraint{first_node, family.id, instance}};
      HYLE_ASSIGN_OR_RETURN(
          auto compiled,
          hyle::graph::compile(graph,
                               {.storage_domains = storage, .targets = targets},
                               {.node_providers = required}));
      HYLE_ASSIGN_OR_RETURN(auto invocation,
                            compiled.executable.create_invocation());
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
        return Status::Internal("replay unexpectedly prepared a kernel again");
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
}
