#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "backend_internal.h"
#include "completion_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/buffer.h"
#include "provider_internal.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::compute::AliasSupport;
using hyle::compute::Completion;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::Operation;
using hyle::compute::Provider;
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileFallbackReason;
using hyle::graph::CompileOptions;
using hyle::graph::CompileReport;
using hyle::graph::CompileStepKind;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::OperationFallbackPolicy;
using hyle::graph::ProviderRegionPrepared;
using hyle::graph::RegionCompileRequest;
using hyle::graph::RegionNodeRange;
using hyle::graph::RegionProposal;
using hyle::graph::RegionProposalWriter;
using hyle::graph::RegionSelectionPolicy;
using hyle::graph::ValueId;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

constexpr hyle::compute::ProviderId kRegionProviderId{0x544553542d475241ULL,
                                                      0x50482d524547494fULL};

enum class ProposalMode : std::uint8_t {
  kPairs = 0,
  kWeightedOverlap,
  kOverflow,
  kCompileFailure,
  kInvocationFailure,
  kAsyncPairs,
  kAsyncRejectSecond,
};

struct RegionProbe final {
  ProposalMode mode = ProposalMode::kPairs;
  std::optional<std::uint64_t> proposal_savings = 10;
  std::uint32_t proposal_dispatch_count = 1;
  std::size_t propose_count = 0;
  std::size_t compile_count = 0;
  std::size_t single_prepare_count = 0;
  std::size_t create_invocation_count = 0;
  std::size_t destroy_invocation_count = 0;
  std::size_t submit_count = 0;
  std::size_t submit_attempt_count = 0;
  std::size_t completion_poll_count = 0;
  std::size_t completion_wait_count = 0;
  std::size_t completion_destroy_count = 0;
  std::size_t destroy_prepared_count = 0;
  std::size_t completed_prefix = 0;
  bool cancelled = false;
  hyle::compute::detail::CompletionLifetime abandoned_lifetime{};
  Status abandoned_hint{};
  hyle::graph::RegionArtifactProperties artifact_properties{};
  std::uint32_t compiled_begin = (std::numeric_limits<std::uint32_t>::max)();
  std::uint32_t compiled_count = 0;
};

struct ProviderPayload final {
  RegionProbe *probe = nullptr;
};

struct RegionPlan final {
  RegionProbe *probe = nullptr;
  std::uint32_t negation_count = 0;
};

struct RegionRun final {
  RegionProbe *probe = nullptr;
};

Status async_terminal(const RegionProbe &probe) noexcept {
  return probe.cancelled ? Status::Cancelled("region sequence was cancelled")
                         : Status::Ok();
}

void destroy_async_completion(
    void *state, hyle::compute::detail::CompletionLifetime lifetime,
    Status terminal_hint) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_destroy_count;
  if (!lifetime.valid())
    return;
  if (probe.completed_prefix < probe.submit_count) {
    probe.abandoned_lifetime = lifetime;
    probe.abandoned_hint = std::move(terminal_hint);
    return;
  }
  Status terminal = async_terminal(probe);
  if (terminal.ok() && !terminal_hint.ok())
    terminal = std::move(terminal_hint);
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

Result<hyle::compute::CompletionPoll>
poll_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_poll_count;
  if (probe.completed_prefix < probe.submit_count)
    return hyle::compute::CompletionPoll::kPending;
  const Status terminal = async_terminal(probe);
  if (!terminal.ok())
    return std::unexpected(terminal);
  return hyle::compute::CompletionPoll::kComplete;
}

Status wait_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_wait_count;
  probe.completed_prefix = probe.submit_count;
  return async_terminal(probe);
}

Status cancel_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  probe.cancelled = true;
  probe.completed_prefix = probe.submit_count;
  return Status::Ok();
}

const hyle::compute::detail::CompletionVTable kAsyncCompletionVTable{
    .destroy = &destroy_async_completion,
    .poll = &poll_async_completion,
    .wait = &wait_async_completion,
    .request_cancel = &cancel_async_completion,
};

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor result;
  result.buffer = Buffer::create(hyle::mem::system_resource(),
                                 description.required_bytes().value(), 64)
                      .value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

void write(OwnedTensor &tensor, std::span<const float> values) {
  ASSERT_TRUE(tensor.buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const OwnedTensor &tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(tensor.buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

bool is_negation(const Graph &graph, std::size_t node_index) {
  auto node = graph.node_id(node_index);
  if (!node)
    return false;
  auto operation = graph.operation(*node);
  if (!operation)
    return false;
  const Unary *unary = operation->get_if<Unary>();
  return unary != nullptr && unary->kind == UnaryKind::kNeg;
}

Status add_proposal(RegionProposalWriter &writer, std::uint32_t begin,
                    std::uint32_t count, std::optional<std::uint64_t> savings,
                    std::uint32_t dispatch_count = 1) noexcept {
  return writer.add(RegionProposal{
      .nodes = RegionNodeRange{begin, count},
      .cost = {.estimated_net_savings_ns = savings,
               .dispatch_count = dispatch_count},
      .route = (static_cast<std::uint64_t>(begin) << 32) | count,
  });
}

Status propose_regions(void *context, const Graph &graph,
                       RegionProposalWriter &writer) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.propose_count;
  if (probe.mode == ProposalMode::kOverflow) {
    Status first = add_proposal(writer, 0, 2, 10);
    if (!first.ok())
      return first;
    return add_proposal(writer, 0, 2, 11);
  }
  if (probe.mode == ProposalMode::kWeightedOverlap) {
    Status first = add_proposal(writer, 0, 2, 10);
    if (!first.ok())
      return first;
    return add_proposal(writer, 1, 2, 20);
  }
  for (std::size_t index = 0; index + 1 < graph.node_count(); ++index) {
    if (!is_negation(graph, index) || !is_negation(graph, index + 1))
      continue;
    Status status =
        add_proposal(writer, static_cast<std::uint32_t>(index), 2,
                     probe.proposal_savings, probe.proposal_dispatch_count);
    if (!status.ok())
      return status;
    ++index;
  }
  return Status::Ok();
}

Status create_region_invocation(void *prepared, AllocatorRef allocator,
                                hyle::mem::BufferView, void **output) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  void *storage = allocator.allocate(sizeof(RegionRun), alignof(RegionRun));
  if (storage == nullptr)
    return Status::ResourceExhausted("region invocation allocation failed");
  *output = ::new (storage) RegionRun{plan.probe};
  ++plan.probe->create_invocation_count;
  if (plan.probe->mode == ProposalMode::kInvocationFailure) {
    return Status::Internal("injected region invocation failure");
  }
  return Status::Ok();
}

void destroy_region_invocation(void *invocation,
                               AllocatorRef allocator) noexcept {
  if (invocation == nullptr)
    return;
  auto *run = static_cast<RegionRun *>(invocation);
  ++run->probe->destroy_invocation_count;
  run->~RegionRun();
  allocator.deallocate(invocation, sizeof(RegionRun), alignof(RegionRun));
}

void destroy_region_prepared(void *prepared, AllocatorRef allocator) noexcept {
  auto *plan = static_cast<RegionPlan *>(prepared);
  if (plan == nullptr)
    return;
  ++plan->probe->destroy_prepared_count;
  plan->~RegionPlan();
  allocator.deallocate(plan, sizeof(RegionPlan), alignof(RegionPlan));
}

AliasSupport region_alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

hyle::compute::detail::EnqueueResult
submit_region(void *prepared, void *invocation, Queue &,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return hyle::compute::detail::EnqueueResult::accepted_command(
        Status::Internal("region binding mismatch"));
  if (plan.probe->mode == ProposalMode::kAsyncPairs ||
      plan.probe->mode == ProposalMode::kAsyncRejectSecond) {
    ++plan.probe->submit_attempt_count;
    if (plan.probe->mode == ProposalMode::kAsyncRejectSecond &&
        plan.probe->submit_attempt_count == 2) {
      return hyle::compute::detail::EnqueueResult::rejected(
          Status::Unavailable("injected second-region rejection"));
    }
    ++plan.probe->submit_count;
    return hyle::compute::detail::EnqueueResult::accepted_command();
  }
  auto input_mapping = inputs[0].storage().map_host();
  if (!input_mapping)
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(input_mapping).error());
  auto output_mapping =
      outputs[0].storage().map_host(hyle::mem::HostAccess::kWrite);
  if (!output_mapping) {
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(output_mapping).error());
  }
  auto writable = output_mapping->writable_bytes();
  if (!writable) {
    Status error = std::move(writable).error();
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(error));
  }
  const std::span<const std::byte> source = input_mapping->bytes();
  if (source.size() != writable->size() || source.size() % sizeof(float) != 0) {
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        Status::Internal("region byte size mismatch"));
  }
  const auto input_values =
      std::span(reinterpret_cast<const float *>(source.data()),
                source.size() / sizeof(float));
  const auto output_values =
      std::span(reinterpret_cast<float *>(writable->data()),
                writable->size() / sizeof(float));
  const float sign = (plan.negation_count & 1U) == 0 ? 1.0F : -1.0F;
  for (std::size_t index = 0; index < input_values.size(); ++index)
    output_values[index] = sign * input_values[index];
  Status terminal = output_mapping->close();
  const Status input_close = input_mapping->close();
  if (terminal.ok() && !input_close.ok())
    terminal = input_close;
  ++plan.probe->submit_count;
  return hyle::compute::detail::EnqueueResult::accepted_command(
      std::move(terminal));
}

const hyle::graph::ProviderRegionPreparedVTable kPreparedRegionVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .submit = &submit_region,
};

Status compile_region(void *context, const RegionCompileRequest &request,
                      AllocatorRef allocator,
                      ProviderRegionPrepared *output) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (output == nullptr || request.graph == nullptr ||
      request.nodes.count < 2 || request.inputs.size() != 1 ||
      request.outputs.size() != 1) {
    return Status::InvalidArgument("region compile request is invalid");
  }
  for (std::uint32_t offset = 0; offset < request.nodes.count; ++offset) {
    if (!is_negation(*request.graph, request.nodes.begin + offset))
      return Status::InvalidArgument("region contains a non-negation node");
  }
  void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
  if (storage == nullptr)
    return Status::ResourceExhausted("region plan allocation failed");
  auto *plan = ::new (storage)
      RegionPlan{.probe = &probe, .negation_count = request.nodes.count};
  *output = ProviderRegionPrepared{
      .vtable = &kPreparedRegionVTable,
      .state = plan,
      .properties = probe.artifact_properties,
  };
  ++probe.compile_count;
  probe.compiled_begin = request.nodes.begin;
  probe.compiled_count = request.nodes.count;
  if (probe.mode == ProposalMode::kCompileFailure)
    return Status::Internal("injected region compile failure");
  return Status::Ok();
}

const hyle::graph::RegionProviderVTable kRegionVTable{
    .propose = &propose_regions,
    .compile = &compile_region,
};

Status create_provider_context(const Provider &provider, hyle::compute::Device,
                               void *, AllocatorRef, void **output) noexcept {
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  *output = payload.probe;
  return Status::Ok();
}

void destroy_provider_context(void *, AllocatorRef) noexcept {}

Status prepare_single(void *context, Operation, std::span<const TensorDesc>,
                      std::span<const TensorDesc>, AllocatorRef,
                      hyle::compute::detail::ProviderPrepared *) noexcept {
  ++static_cast<RegionProbe *>(context)->single_prepare_count;
  return Status::Unimplemented("region provider has no single-op route");
}

const void *query_extension(void *,
                            hyle::compute::ProviderExtensionId id) noexcept {
  return id == hyle::graph::kRegionProviderExtensionId ? &kRegionVTable
                                                       : nullptr;
}

const hyle::compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = kRegionProviderId,
                 .name = "test-region",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .prepare = &prepare_single,
    .query_extension = &query_extension,
};

Provider region_provider(RegionProbe &probe) noexcept {
  return hyle::compute::detail::ProviderAccess::make(&kProviderVTable,
                                                     ProviderPayload{&probe});
}

struct AsyncBackendPayload final {
  RegionProbe *probe = nullptr;
};

constexpr hyle::compute::BackendId kAsyncBackendId{
    0x544553542d475241ULL, 0x50482d4153594e43ULL};

Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend, std::size_t index) noexcept;

Result<std::size_t>
async_device_count(const hyle::compute::Backend &) noexcept {
  return std::size_t{1};
}

bool same_async_device(const hyle::compute::Device &left,
                       const hyle::compute::Device &right) noexcept {
  return hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(left)
             .probe ==
         hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(right)
             .probe;
}

Result<hyle::compute::DeviceProperties>
async_device_properties(const hyle::compute::Device &device) noexcept {
  if (hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("test graph backend device is invalid"));
  }
  return hyle::compute::DeviceProperties{
      .backend = kAsyncBackendId,
      .device_class = hyle::compute::DeviceClass::kCpu,
      .backend_name = "test-graph-async",
      .name = "test Graph async sequence backend",
      .unified_memory = true,
      .asynchronous_queues = true,
  };
}

Status validate_async_resource(const hyle::compute::Device &,
                               hyle::mem::MemoryResourceRef resource) noexcept {
  if (!resource.valid())
    return Status::InvalidArgument("test graph resource is invalid");
  if (!resource.properties().has(hyle::mem::MemoryProperty::kHostAccessible)) {
    return Status::FailedPrecondition(
        "test graph backend requires host-accessible storage");
  }
  return Status::Ok();
}

Status create_async_backend_context(const hyle::compute::Device &device,
                                    AllocatorRef, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("test graph context output is null");
  *output = hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(
                device)
                .probe;
  return *output != nullptr
             ? Status::Ok()
             : Status::InvalidArgument("test graph backend has no probe");
}

void destroy_async_backend_context(void *, AllocatorRef) noexcept {}

Status create_async_queue(void *context, hyle::compute::QueueOptions,
                          AllocatorRef, void **output) noexcept {
  if (context == nullptr || output == nullptr)
    return Status::InvalidArgument("test graph queue output is invalid");
  *output = context;
  return Status::Ok();
}

void destroy_async_queue(void *, AllocatorRef) noexcept {}

Status begin_async_sequence(void *queue, void **sequence) noexcept {
  if (queue == nullptr || sequence == nullptr)
    return Status::InvalidArgument("test graph sequence is invalid");
  *sequence = queue;
  return Status::Ok();
}

Completion finish_async_sequence(void *, void *sequence, bool accepted,
                                 Status terminal) noexcept {
  if (!accepted)
    return Completion::completed(std::move(terminal));
  Completion completion = hyle::compute::detail::CompletionAccess::make(
      &kAsyncCompletionVTable, sequence);
  hyle::compute::detail::CompletionAccess::append_terminal(
      completion, std::move(terminal));
  return completion;
}

Status reserve_async_completions(void *, std::size_t) noexcept {
  return Status::Ok();
}

Result<hyle::compute::TransferProperties>
async_transfer_properties(void *, hyle::mem::BufferView,
                          hyle::mem::ConstBufferView) noexcept {
  return hyle::compute::TransferProperties{
      .mode = hyle::compute::TransferMode::kImmediate};
}

hyle::compute::detail::EnqueueResult
async_copy(void *, hyle::mem::BufferView destination,
           hyle::mem::ConstBufferView source) noexcept {
  return hyle::compute::detail::EnqueueResult::accepted_command(
      destination.copy_from(source));
}

hyle::compute::detail::EnqueueResult
async_fill(void *, hyle::mem::BufferView destination,
           std::byte value) noexcept {
  return hyle::compute::detail::EnqueueResult::accepted_command(
      destination.fill(value));
}


hyle::compute::detail::EnqueueResult async_wait_completion(
    void *, const hyle::compute::detail::CompletionVTable *, void *) noexcept {
  return hyle::compute::detail::EnqueueResult::rejected(
      Status::Unimplemented("test graph backend has no dependency route"));
}

const void *query_async_extension(void *,
                                  hyle::compute::BackendExtensionId) noexcept {
  return nullptr;
}

Status drain_async(void *state) noexcept {
  if (state != nullptr) {
    auto &probe = *static_cast<RegionProbe *>(state);
    probe.completed_prefix = probe.submit_count;
  }
  return Status::Ok();
}

const hyle::compute::detail::BackendVTable kAsyncBackendVTable{
    .id = kAsyncBackendId,
    .name = "test-graph-async",
    .device_class = hyle::compute::DeviceClass::kCpu,
    .device_count = &async_device_count,
    .device = &async_device,
    .same_device = &same_async_device,
    .properties = &async_device_properties,
    .validate_resource = &validate_async_resource,
    .create_context = &create_async_backend_context,
    .destroy_context = &destroy_async_backend_context,
    .create_queue = &create_async_queue,
    .destroy_queue = &destroy_async_queue,
    .begin_sequence = &begin_async_sequence,
    .finish_sequence = &finish_async_sequence,
    .reserve_completions = &reserve_async_completions,
    .transfer_properties = &async_transfer_properties,
    .copy_async = &async_copy,
    .fill_async = &async_fill,
    .wait_completion = &async_wait_completion,
    .query_extension = &query_async_extension,
    .drain_context = &drain_async,
    .drain_queue = &drain_async,
};

Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend, std::size_t index) noexcept {
  if (index != 0) {
    return std::unexpected(
        Status::OutOfRange("test graph device index is out of range"));
  }
  return hyle::compute::detail::DeviceAccess::make(
      &kAsyncBackendVTable,
      hyle::compute::detail::BackendAccess::payload<AsyncBackendPayload>(
          backend));
}

hyle::compute::Device async_device(RegionProbe &probe) noexcept {
  const auto backend = hyle::compute::detail::BackendAccess::make(
      &kAsyncBackendVTable, AsyncBackendPayload{&probe});
  return backend.device(0).value();
}

struct NegationGraph final {
  Graph graph{};
  ValueId input{};
  ValueId output{};
};

NegationGraph make_negations(std::size_t count, bool finish_with_abs = false) {
  GraphBuilder builder = GraphBuilder::create().value();
  NegationGraph result;
  result.input = builder.input(row(8)).value();
  ValueId value = result.input;
  for (std::size_t index = 0; index < count; ++index)
    value = builder.add(Unary{UnaryKind::kNeg}, value).value();
  if (finish_with_abs)
    value = builder.add(Unary{UnaryKind::kAbs}, value).value();
  result.output = value;
  EXPECT_TRUE(builder.publish(value).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

Context make_context(RegionProbe &probe) {
  const std::array providers{region_provider(probe),
                             hyle::compute::native::provider()};
  const bool asynchronous = probe.mode == ProposalMode::kAsyncPairs ||
                            probe.mode == ProposalMode::kAsyncRejectSecond;
  return Context::create(asynchronous ? async_device(probe)
                                      : hyle::compute::cpu::device(),
                         {.providers = providers})
      .value();
}

Context make_context(RegionProbe &first, RegionProbe &second) {
  const std::array providers{region_provider(first), region_provider(second),
                             hyle::compute::native::provider()};
  return Context::create(hyle::compute::cpu::device(), {.providers = providers})
      .value();
}

TEST(GraphRegions, CompilesAndSubmitsOneArtifactWithoutInternalStorage) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 2u);
  EXPECT_EQ(compiled.report.step_count(), 1u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  EXPECT_EQ(compiled.report.provider().id, kRegionProviderId);
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  EXPECT_EQ(probe.single_prepare_count, 0u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
  EXPECT_EQ(probe.submit_count, 1u);
}

TEST(GraphRegions, MixesACompiledRegionWithSingleOperationFallback) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2, true);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 3u);
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 1u);
  EXPECT_TRUE(compiled.report.mixed_providers());
  EXPECT_FALSE(compiled.report.provider().valid());
  EXPECT_EQ(probe.single_prepare_count, 1u);
  auto fallback = compiled.report.step(1);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(fallback->fallback_reason,
            CompileFallbackReason::kNoEligibleRegion);
  EXPECT_FALSE(fallback->persistent_bytes.has_value());

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> expected{4, 3, 2, 1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions,
     SelectsTheHigherBenefitOverlappingIntervalDeterministically) {
  RegionProbe probe{.mode = ProposalMode::kWeightedOverlap};
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection =
                             RegionSelectionPolicy::kRequireEstimatedBenefit})
          .value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(probe.compiled_count, 2u);
}

TEST(GraphRegions, DefaultPolicyAcceptsUnknownEstimateAsExplicitOptIn) {
  RegionProbe probe{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, StrictPolicyRejectsUnknownEstimate) {
  RegionProbe probe{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection =
                             RegionSelectionPolicy::kRequireEstimatedBenefit})
          .value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 2u);
}

TEST(GraphRegions, ExplicitZeroEstimateIsNotSelectedByDefault) {
  RegionProbe probe{.proposal_savings = 0};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyUsesEarlierProviderAsTheFinalTieBreak) {
  RegionProbe first{.proposal_savings = std::nullopt};
  RegionProbe second{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyPrefersFewerDispatchEpisodesBeforeProvider) {
  RegionProbe first{.proposal_savings = std::nullopt,
                    .proposal_dispatch_count = 2};
  RegionProbe second{.proposal_savings = std::nullopt,
                     .proposal_dispatch_count = 1};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
}

TEST(GraphRegions, RejectsAnUnknownRegionSelectionPolicy) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.region_selection =
                         static_cast<RegionSelectionPolicy>(0xff)});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  EXPECT_EQ(probe.propose_count, 0u);
}

TEST(GraphRegions, DisabledPolicyUsesOnlyExplicitSingleOperationFallback) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection = RegionSelectionPolicy::kDisabled})
          .value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.propose_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 2u);
  auto fallback = compiled.report.step(0);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->fallback_reason,
            CompileFallbackReason::kRegionSelectionDisabled);
}

TEST(GraphRegions, ProposalBudgetExhaustionFailsInsteadOfChangingThePlan) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context, CompileOptions{.max_region_proposals = 1});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(probe.compile_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, PerProviderProposalBudgetIsAlsoAHardLimit) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.max_region_proposals = 4,
                     .max_region_proposals_per_provider = 1});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, RejectFallbackPolicyRequiresCompleteRegionCoverage) {
  RegionProbe probe;
  Context context = make_context(probe);
  const NegationGraph complete = make_negations(2);
  EXPECT_TRUE(
      hyle::graph::compile(complete.graph, context,
                           CompileOptions{.operation_fallback =
                                              OperationFallbackPolicy::kReject})
          .has_value());

  const NegationGraph partial = make_negations(2, true);
  auto rejected = hyle::graph::compile(
      partial.graph, context,
      CompileOptions{.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsUnimplemented());
}

TEST(GraphRegions, DetailedReportOwnsAndEnumeratesTheCompiledPlan) {
  RegionProbe probe;
  probe.artifact_properties = {
      .persistent_bytes = 128,
      .copy_bytes = 64,
      .synchronization_count = 2,
      .dispatch_count = 3,
      .blocks_host = true,
      .uses_private_threads = true,
      .supports_concurrent_invocations = false,
  };
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  CompileReport retained = compiled.report;
  compiled.report = {};
  compiled.executable = {};
  EXPECT_EQ(probe.destroy_prepared_count, 1u);

  ASSERT_TRUE(retained.valid());
  ASSERT_EQ(retained.step_count(), 1u);
  EXPECT_EQ(retained.copy_bytes(), 64u);
  EXPECT_EQ(retained.synchronization_count(), 2u);
  auto step = retained.step(0);
  ASSERT_TRUE(step.has_value()) << step.error().message();
  EXPECT_EQ(step->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(step->provider.id, kRegionProviderId);
  EXPECT_EQ(step->node_begin, 0u);
  EXPECT_EQ(step->node_count, 2u);
  ASSERT_TRUE(step->persistent_bytes.has_value());
  EXPECT_EQ(*step->persistent_bytes, 128u);
  EXPECT_EQ(step->copy_bytes, 64u);
  EXPECT_EQ(step->synchronization_count, 2u);
  EXPECT_EQ(step->dispatch_count, 3u);
  EXPECT_TRUE(step->blocks_host);
  EXPECT_TRUE(step->uses_private_threads);
  EXPECT_FALSE(step->supports_concurrent_invocations);
  EXPECT_TRUE(retained.step(1).error().IsOutOfRange());

  retained = {};
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
}

TEST(GraphRegions, CostFactsPreserveUnknownAndExactZero) {
  const NegationGraph logical = make_negations(2);

  RegionProbe unknown_probe;
  Context unknown_context = make_context(unknown_probe);
  auto unknown = hyle::graph::compile(logical.graph, unknown_context).value();
  auto unknown_step = unknown.report.step(0);
  ASSERT_TRUE(unknown_step.has_value()) << unknown_step.error().message();
  EXPECT_FALSE(unknown_step->persistent_bytes.has_value());

  RegionProbe zero_probe;
  zero_probe.artifact_properties.persistent_bytes = 0;
  Context zero_context = make_context(zero_probe);
  auto zero = hyle::graph::compile(logical.graph, zero_context).value();
  auto zero_step = zero.report.step(0);
  ASSERT_TRUE(zero_step.has_value()) << zero_step.error().message();
  ASSERT_TRUE(zero_step->persistent_bytes.has_value());
  EXPECT_EQ(*zero_step->persistent_bytes, 0u);
}

TEST(GraphRegions, RegionExtensionVersionDoesNotAliasTheV1Table) {
  constexpr hyle::compute::ProviderExtensionId version_one{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 1};
  constexpr hyle::compute::ProviderExtensionId version_two{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 2};
  EXPECT_EQ(query_extension(nullptr, version_one), nullptr);
  EXPECT_EQ(query_extension(nullptr, version_two), nullptr);
  EXPECT_EQ(query_extension(nullptr, hyle::graph::kRegionProviderExtensionId),
            &kRegionVTable);
}

TEST(GraphRegions, RollsBackPartiallyCreatedProviderState) {
  {
    RegionProbe probe{.mode = ProposalMode::kCompileFailure};
    const NegationGraph logical = make_negations(2);
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsInternal());
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
  }
  {
    RegionProbe probe{.mode = ProposalMode::kInvocationFailure};
    const NegationGraph logical = make_negations(2);
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context).value();
    auto invocation = compiled.executable.create_invocation();
    ASSERT_FALSE(invocation.has_value());
    EXPECT_TRUE(invocation.error().IsInternal());
    EXPECT_EQ(probe.create_invocation_count, 1u);
    EXPECT_EQ(probe.destroy_invocation_count, 1u);
  }
}

TEST(GraphRegions, InvocationRetainsItsCompiledRegionAndWorkspaceOwner) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  Queue queue = context.create_queue().value();
  hyle::graph::Invocation invocation;
  {
    auto compiled = hyle::graph::compile(logical.graph, context).value();
    invocation = compiled.executable.create_invocation().value();
  }
  EXPECT_EQ(probe.destroy_prepared_count, 0u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));

  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
}

TEST(GraphRegions,
     OrderedSequenceDoesNotHostWaitAndRetainsInvocationUntilTerminal) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.region_count(), 2u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();

  auto submitted = invocation.submit(queue);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
  EXPECT_EQ(probe.submit_count, 2u);
  EXPECT_EQ(probe.completion_wait_count, 0u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(invocation.reset().IsFailedPrecondition());
  EXPECT_TRUE(
      invocation.bind(logical.input, input.view).IsFailedPrecondition());
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
  auto pending = submitted->poll();
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(*pending, hyle::compute::CompletionPoll::kPending);
  EXPECT_TRUE(submitted->wait().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_EQ(probe.completion_destroy_count, 1u);
  ASSERT_TRUE(invocation.reset().ok());

  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Completion abandoned = invocation.submit(queue).value();
  EXPECT_EQ(probe.submit_count, 4u);
  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 0u);
  abandoned = {};
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.destroy_invocation_count, 0u);

  probe.completed_prefix = probe.submit_count;
  Status terminal = async_terminal(probe);
  if (terminal.ok() && !probe.abandoned_hint.ok())
    terminal = probe.abandoned_hint;
  probe.abandoned_lifetime.terminal(probe.abandoned_lifetime.state, terminal);
  probe.abandoned_lifetime.release(probe.abandoned_lifetime.state);
  probe.abandoned_lifetime = {};
  EXPECT_EQ(probe.destroy_invocation_count, 2u);
}

TEST(GraphRegions, RejectedTailIsReportedAfterTheAcceptedAsyncPrefix) {
  RegionProbe probe{.mode = ProposalMode::kAsyncRejectSecond};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();

  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_EQ(probe.submit_attempt_count, 2u);
  EXPECT_EQ(probe.submit_count, 1u);
  EXPECT_EQ(probe.completion_wait_count, 0u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(completion->wait().IsUnavailable());
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_FALSE(invocation.in_flight());
}

} // namespace
