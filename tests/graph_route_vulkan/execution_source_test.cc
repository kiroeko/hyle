#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "hyle/graph/vulkan.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/source.h"
#include "hyle/compute/vulkan_memory.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph/region_discovery.h"
#include "hyle/graph/detail/recording_access.h"
#include "../../src/graph_route_vulkan/vulkan_source.h"
#include "../../src/graph_route_vulkan/native_value.h"
#include "../../src/compute_route_vulkan/vulkan_composition_access.h"
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "hyle/simd/dtype.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gd = g::detail;

void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take_at(hyle::Result<T> value, const char *stage) {
  if (!value) throw std::runtime_error(std::string(stage) + ": " + std::string(value.error().message()));
  return std::move(*value);
}
void check_at(hyle::Status status, const char *stage) {
  if (!status.ok()) throw std::runtime_error(std::string(stage) + ": " +
      std::string(hyle::status_code_name(status.code())) + ": " + std::string(status.message()));
}
struct NativeEndpoint final {
  v::Context context = v::Context::create().value();
  v::MemoryResource device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  v::MemoryResource staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  c::StorageOwner device_owner = device.retain_storage_owner(context).value();
  c::StorageOwner staging_owner = staging.retain_storage_owner(context).value();
  g::ExecutionSource source = g::vulkan::execution_source(context).value();
  const std::array<g::StorageDomain, 2> domains{{
      {.resource = device.ref(), .owner = device_owner},
      {.resource = staging.ref(), .owner = staging_owner}}};
  const std::array<g::ExecutionTarget, 1> targets{{{source, g::StorageDomainId{0}}}};

  template<class T, std::size_t N>
  c::ReadLease ready(const std::array<T, N> &values) {
    auto bytes = hyle::mem::Buffer::create(device.ref(), sizeof(values), 256).value();
    check(bytes.view().write(std::as_bytes(std::span(values))));
    return device_owner.adopt_buffer(std::move(bytes)).value();
  }
};
struct NativeSourceProgram final {
  NativeEndpoint &endpoint;
  c::TensorDesc input_description, output_description, indices_description;
  g::ValueId input, output, delivery, indices;
  g::Graph graph;
  g::CompiledGraph compiled;
  c::LeasePool outputs, deliveries;

  NativeSourceProgram(NativeEndpoint &native, std::int64_t width, bool fail = false,
                      bool prefix_views = false)
      : endpoint(native),
        input_description(c::TensorDesc::dense(c::DataType::kFloat32,
            std::array<std::int64_t, 1>{width}).value()),
        output_description(c::TensorDesc::dense(c::DataType::kFloat32,
            std::array<std::int64_t, 1>{fail ? 1 : width}).value()),
        indices_description(c::TensorDesc::dense(c::DataType::kInt32,
            std::array<std::int64_t, 1>{1}).value()) {
    auto builder = g::GraphBuilder::create().value();
    input = builder.input(input_description).value();
    auto math = input;
    if (prefix_views) {
      math = builder.add(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 2>{1, width}).value()).value(), math).value();
      math = builder.add(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 1>{width}).value()).value(), math).value();
    }
    if (fail) {
      indices = builder.input(indices_description).value();
      math = builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked},
          math, indices).value();
    } else math = builder.add(c::Unary{c::UnaryKind::kNeg}, math).value();
    output = builder.add(c::Copy{}, math).value();
    delivery = builder.add(c::Copy{}, output).value();
    check(builder.publish(output)); check(builder.publish(delivery));
    graph = std::move(builder).finish().value();
    const std::array external{g::ExternalResourceContract{output, g::StorageDomainId{0}},
        g::ExternalResourceContract{delivery, g::StorageDomainId{1}}};
    compiled = g::compile(graph, {.storage_domains = native.domains, .targets = native.targets},
        {.external_resources = external}).value();
    outputs = c::LeasePool::create(native.device_owner.ref(), 1024,
        {.max_blocks = 4, .backing_alignment = 256}).value();
    deliveries = c::LeasePool::create(native.staging_owner.ref(), 1024,
        {.max_blocks = 4, .backing_alignment = 256}).value();
  }
  g::Invocation invocation() {
    auto result = compiled.executable.create_invocation().value();
    check(result.prepare_resources());
    const auto bytes = output_description.required_bytes().value();
    check(result.bind(output, output_description, outputs.allocate(bytes, 256).value()));
    check(result.bind(delivery, output_description, deliveries.allocate(bytes, 256).value()));
    return result;
  }
  std::vector<float> read(g::Invocation &invocation) {
    const auto result = invocation.result(delivery).value();
    const auto lease = result.storage_lease().value();
    std::vector<float> values(output_description.element_count().value());
    check(lease.view().value().read(std::as_writable_bytes(std::span(values))));
    return values;
  }
};

// Hold only successful fence observation, not hardware execution. The real
// work and fences still run. This isolates the original cancellation/retirement
// boundary and never claims physical overlap or a performance measurement.
struct HoldObservation final {
  v::detail::Device &device;
  PFN_vkGetFenceStatus original;
  bool held = true;
  static inline HoldObservation *active = nullptr;
  explicit HoldObservation(v::Context &context)
      : device(*v::detail::Access::runtime(context)->device), original(device.fn.GetFenceStatus) {
    active = this; device.fn.GetFenceStatus = poll;
  }
  ~HoldObservation() { device.fn.GetFenceStatus = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL poll(VkDevice device, VkFence fence) {
    const auto result = active->original(device, fence);
    return active->held && result == VK_SUCCESS ? VK_NOT_READY : result;
  }
};

std::pair<std::size_t, g::CompileStepKind> compiled_step(
    const g::CompiledGraph &compiled, g::NodeId node) {
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    const auto step = compiled.report.step(index).value();
    for (std::size_t port = 0; port < step.node_count; ++port)
      if (compiled.report.step_node(index, port).value() == node)
        return {index, step.kind};
  }
  throw std::runtime_error("original node missing from compiled report");
}

TEST(VulkanRegionDiscovery, PrefixQualificationRechecksAnEmptyCandidateAfterControlCut) {
  auto builder = g::GraphBuilder::create().value();
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const auto x = builder.input(desc).value();
  const auto outside = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{x}).value();
  const auto outside_value = builder.result(outside).value();
  const auto calculation = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kExp}), std::array{x}).value();
  const auto data = builder.result(calculation).value();
  const auto shape = c::TensorShape::create(std::array<std::int64_t, 2>{2, 2}).value();
  const auto metadata = builder.add_node(c::Operation(c::View::reshape(shape).value()), std::array{data}).value();
  const auto view = builder.result(metadata).value();
  const auto a = builder.add(c::Unary{c::UnaryKind::kNeg}, view).value();
  const auto b = builder.add(c::Unary{c::UnaryKind::kExp}, view).value();
  // Keep the builder's actual identities and original control edge.
  check(builder.depends_on(metadata, outside));
  check(builder.publish(outside_value)); check(builder.publish(a)); check(builder.publish(b));
  const auto graph = std::move(builder).finish().value();
  ASSERT_EQ(graph.producer(data).value(), calculation);
  ASSERT_EQ(graph.producer(view).value(), metadata);
  std::vector<std::vector<g::NodeId>> ordinary, prefix;
  std::vector<unsigned> ordinary_calls(graph.node_count());
  check(gd::discover_regions(graph, hyle::mem::system_allocator(),
      [&](g::NodeId node) -> hyle::Result<bool> {
        ++ordinary_calls[node.ordinal()]; return node != outside;
      }, [&](std::span<const g::NodeId> nodes) -> hyle::Result<bool> {
        ordinary.emplace_back(nodes.begin(), nodes.end()); return true;
      }));
  unsigned nonempty_view = 0, empty_view = 0;
  check(gd::discover_regions(graph, hyle::mem::system_allocator(),
      [&](g::NodeId node, std::span<const g::NodeId> candidate) -> hyle::Result<bool> {
        if (node == outside) return false;
        if (node != metadata) return true;
        if (candidate.empty()) ++empty_view; else ++nonempty_view;
        return !candidate.empty();
      }, [&](std::span<const g::NodeId> nodes) -> hyle::Result<bool> {
        prefix.emplace_back(nodes.begin(), nodes.end()); return true;
      }));
  const auto first = graph.producer(a).value(), second = graph.producer(b).value();
  const std::vector<std::vector<g::NodeId>> old_expected{{calculation}, {metadata, first, second}};
  const std::vector<std::vector<g::NodeId>> prefix_expected{{calculation}, {first, second}};
  EXPECT_EQ(ordinary, old_expected);
  EXPECT_EQ(prefix, prefix_expected);
  EXPECT_EQ(nonempty_view, 1U); EXPECT_EQ(empty_view, 1U);
  for (const auto calls : ordinary_calls) EXPECT_EQ(calls, 1U);
}

TEST(VulkanExecutionSource, InputReshapeUsesGraphReadPermissionAndOriginalColdWriterHandoff) {
  NativeEndpoint endpoint;
  const auto ids = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{1, 1}).value();
  const auto scalar = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{1}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(ids).value();
  const auto flat = builder.add(c::View::reshape(c::TensorShape::create(
      std::array<std::int64_t, 1>{1}).value()).value(), input).value();
  const auto cast = builder.add(c::Cast{c::DataType::kFloat32}, flat).value();
  check(builder.publish(flat)); check(builder.publish(cast));
  const auto graph = std::move(builder).finish().value();
  const auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets}).value();
  const auto flat_step = compiled_step(compiled, graph.producer(flat).value());
  const auto cast_step = compiled_step(compiled, graph.producer(cast).value());
  EXPECT_EQ(flat_step.second, g::CompileStepKind::kView);
  EXPECT_EQ(cast_step.second, g::CompileStepKind::kProviderRegion);
  auto pool = c::LeasePool::create(endpoint.device_owner.ref(), 1024,
      {.max_blocks = 4, .backing_alignment = 256}).value();
  auto writer = pool.allocate(sizeof(std::int32_t), 256).value();
  const auto original = writer.view().value();
  const auto generation = writer.generation();
  const std::array<std::int32_t, 1> initial_token{11};
  check(writer.view()->write(std::as_bytes(std::span(initial_token))));
  auto ready = writer.publish().value();
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  check(invocation.bind(input, ids, ready));
  check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto reclaimed = gd::RecordingAccess::reclaim_input_writer(invocation, input, ids, ready);
  ASSERT_TRUE(reclaimed) << reclaimed.error().message();
  EXPECT_EQ(reclaimed->generation(), generation);
  EXPECT_EQ(reclaimed->view()->native_handle(), original.native_handle());
  EXPECT_EQ(reclaimed->view()->offset(), original.offset());
  EXPECT_EQ(reclaimed->view()->size(), original.size());
  const std::array<std::int32_t, 1> token{13};
  check(reclaimed->view()->write(std::as_bytes(std::span(token))));
  ready = reclaimed->publish().value();
  check(invocation.bind(input, ids, ready));
  check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto done = invocation.submit().value(); check(done.wait()); done = {};
  auto alias_read = invocation.result(flat)->storage_lease().value();
  auto old_float = invocation.result(cast)->storage_lease().value();
  std::array<std::int32_t, 1> actual_ids{};
  std::array<float, 1> actual_float{};
  check(alias_read.view()->read(std::as_writable_bytes(std::span(actual_ids))));
  check(old_float.view()->read(std::as_writable_bytes(std::span(actual_float))));
  EXPECT_EQ(actual_ids, token); EXPECT_EQ(actual_float[0], 13.F);
  invocation = {};
  // The View's escaped read keeps the original input permission, even after
  // every Graph/native slot is gone. A copied output owns independent bytes.
  EXPECT_FALSE(ready.try_write());
  alias_read.reset();
  auto next = ready.try_write(); ASSERT_TRUE(next) << next.error().message();
  const std::array<std::int32_t, 1> next_token{19};
  check(next->view()->write(std::as_bytes(std::span(next_token))));
  auto next_ready = next->publish().value();
  auto next_invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  check(next_invocation.bind(input, ids, next_ready));
  check(next_invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto next_done = next_invocation.submit().value(); check(next_done.wait()); next_done = {};
  auto next_float = next_invocation.result(cast)->storage_lease().value();
  check(next_float.view()->read(std::as_writable_bytes(std::span(actual_float))));
  EXPECT_EQ(actual_float[0], 19.F);
  check(old_float.view()->read(std::as_writable_bytes(std::span(actual_float))));
  EXPECT_EQ(actual_float[0], 13.F);
}

TEST(VulkanExecutionSource, ComputedViewsStayInsideTheirOriginalNativeRegion) {
  NativeEndpoint endpoint;
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  const auto view = builder.add(c::View::reshape(c::TensorShape::create(
      std::array<std::int64_t, 2>{2, 2}).value()).value(), negative).value();
  check(builder.publish(view));
  const auto graph = std::move(builder).finish().value();
  const auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets}).value();
  const auto math = compiled_step(compiled, graph.producer(negative).value());
  EXPECT_EQ(math.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled_step(compiled, graph.producer(view).value()), math);
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  check(invocation.bind(input, desc, endpoint.ready(std::array{1.F, -2.F, 3.F, -4.F})));
  check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto done = invocation.submit().value(); check(done.wait()); done = {};
  auto read = invocation.result(view)->storage_lease().value();
  std::array<float, 4> actual{};
  check(read.view()->read(std::as_writable_bytes(std::span(actual))));
  EXPECT_EQ(actual, (std::array{-1.F, 2.F, -3.F, 4.F}));
}

TEST(VulkanExecutionSource, ComputedBoolSliceAcrossExternalControlRetainsItsNativeRoot) {
  NativeEndpoint endpoint;
  const auto values = take_at(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}), "computed Bool: F32 description");
  auto builder = take_at(g::GraphBuilder::create(), "computed Bool: GraphBuilder");
  const auto left = take_at(builder.input(values), "computed Bool: left input");
  const auto right = take_at(builder.input(values), "computed Bool: right input");
  const auto external = take_at(builder.input(values), "computed Bool: control input");
  // A real floating != comparison, using the existing Subtract/Bool Cast
  // recipe. The complete computed Bool[2] belongs to this original region.
  const auto difference = take_at(builder.add(c::Binary{c::BinaryKind::kSubtract}, left, right),
      "computed Bool: subtraction");
  const auto predicate = take_at(builder.add(c::Cast{c::DataType::kBool}, difference),
      "computed Bool: comparison Cast");
  const auto shape = take_at(c::TensorShape::create(std::array<std::int64_t, 2>{1, 2}),
      "computed Bool: control shape");
  const auto reshape = take_at(c::View::reshape(shape), "computed Bool: control reshape");
  const auto control = take_at(builder.add_node(c::Operation(reshape), std::array{external}),
      "computed Bool: control View");
  const auto select = take_at(c::View::slice(0, 1, 1), "computed Bool: Slice descriptor");
  const auto slice = take_at(builder.add_node(c::Operation(select), std::array{predicate}),
      "computed Bool: Slice node");
  check_at(builder.depends_on(slice, control), "computed Bool: external control edge");
  const auto selected = take_at(builder.result(slice), "computed Bool: Slice result");
  // Bool->F32 is the existing native tensor Cast; Bool->I32 is unsupported.
  // Its input still performs the same checked word32 load of Bool byte1.
  const auto output = take_at(builder.add(c::Cast{c::DataType::kFloat32}, selected),
      "computed Bool: result Cast");
  check_at(builder.publish(output), "computed Bool: publish");
  const auto graph = take_at(std::move(builder).finish(), "computed Bool: finish Graph");
  const auto compiled = take_at(g::compile(graph, {endpoint.domains, endpoint.targets}),
      "computed Bool: compile Graph");
  const auto producer_step = compiled_step(compiled,
      take_at(graph.producer(predicate), "computed Bool: producer node"));
  const auto slice_step = compiled_step(compiled, slice);
  const auto cast_step = compiled_step(compiled,
      take_at(graph.producer(output), "computed Bool: result Cast node"));
  EXPECT_EQ(compiled.report.region_count(), 2U);
  EXPECT_EQ(producer_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled_step(compiled, control).second, g::CompileStepKind::kView);
  EXPECT_EQ(slice_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(slice_step, cast_step);
  EXPECT_NE(producer_step.first, slice_step.first);
  EXPECT_NE(compiled_step(compiled, control).first, slice_step.first);

  auto pool = take_at(c::LeasePool::create(endpoint.device_owner.ref(), 1024,
      {.max_blocks = 4, .backing_alignment = 256}), "computed Bool: input pool");
  auto writer = take_at(pool.allocate(2 * sizeof(float), 256), "computed Bool: input writer");
  const auto generation = writer.generation();
  const auto original = take_at(writer.view(), "computed Bool: original writer view");
  const std::array<std::array<float, 2>, 3> inputs{{{2.F, 1.F}, {3.F, 0.F}, {-1.F, -2.F}}};
  check_at(original.write(std::as_bytes(std::span(inputs[0]))), "computed Bool: initialize writer");
  auto first = take_at(writer.publish(), "computed Bool: publish input");
  auto invocation = take_at(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),
      "computed Bool: create Graph invocation");
  const auto upload = [&](const std::array<float, 2> &data) {
    const auto tensor = take_at(endpoint.context.upload(values, std::as_bytes(std::span(data))),
        "computed Bool: upload ready input");
    return take_at(endpoint.device.import_readonly(tensor), "computed Bool: import ready input");
  };
  const auto rhs = upload(std::array{2.F, 0.F});
  const auto control_input = upload(std::array{3.F, 4.F});
  for (std::size_t replay = 0; replay < inputs.size(); ++replay) {
    SCOPED_TRACE(replay);
    check_at(invocation.bind(left, values, replay == 0 ? first : upload(inputs[replay])), "computed Bool: bind left");
    check_at(invocation.bind(right, values, rhs), "computed Bool: bind right");
    check_at(invocation.bind(external, values, control_input), "computed Bool: bind control");
    check_at(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}), "computed Bool: prepare replay");
    if (replay == 0) {
      HoldObservation hold(endpoint.context);
      auto done = take_at(invocation.submit(), "computed Bool: submit held replay");
      EXPECT_EQ(take_at(done.poll(), "computed Bool: poll held replay"), c::CompletionPoll::kPending);
      EXPECT_FALSE(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}).ok());
      EXPECT_FALSE(first.try_write());
      hold.held = false;
      check_at(done.wait(), "computed Bool: retire held replay"); done = {};
    } else {
      auto done = take_at(invocation.submit(), "computed Bool: submit replay");
      check_at(done.wait(), "computed Bool: retire replay"); done = {};
    }
    const auto result = take_at(invocation.result(output), "computed Bool: Graph result");
    auto read = take_at(result.storage_lease(), "computed Bool: result lease");
    std::array<float, 1> actual{};
    const auto view = take_at(read.view(), "computed Bool: result lease view");
    check_at(view.read(std::as_writable_bytes(std::span(actual))), "computed Bool: read F32 result");
    EXPECT_EQ(actual[0], replay == 1 ? 0.F : 1.F);
  }
  invocation = {};
  auto recovered = take_at(first.try_write(), "computed Bool: reclaim original writer");
  EXPECT_EQ(recovered.generation(), generation);
  const auto restored = take_at(recovered.view(), "computed Bool: restored writer view");
  EXPECT_EQ(restored.native_handle(), original.native_handle());
  EXPECT_EQ(restored.offset(), original.offset());
}

TEST(VulkanExecutionSource, CallerBoolSubviewCannotBorrowItsUnleasedWordTail) {
  NativeEndpoint endpoint;
  const auto bit = take_at(c::TensorDesc::dense(c::DataType::kBool,
      std::array<std::int64_t, 1>{1}), "caller Bool: description");
  auto builder = take_at(g::GraphBuilder::create(), "caller Bool: GraphBuilder");
  const auto input = take_at(builder.input(bit), "caller Bool: Graph input");
  const auto output = take_at(builder.add(c::Cast{c::DataType::kFloat32}, input),
      "caller Bool: supported F32 Cast");
  check_at(builder.publish(output), "caller Bool: publish");
  const auto graph = take_at(std::move(builder).finish(), "caller Bool: finish Graph");
  const auto compiled = take_at(g::compile(graph, {endpoint.domains, endpoint.targets}), "caller Bool: compile Graph");
  auto pool = take_at(c::LeasePool::create(endpoint.device_owner.ref(), 1024,
      {.max_blocks = 4, .backing_alignment = 256}), "caller Bool: pool");
  auto writer = take_at(pool.allocate(4, 256), "caller Bool: writer");
  const auto generation = writer.generation();
  const std::array<std::uint8_t, 4> bits{0, 1, 0, 1};
  const auto original = take_at(writer.view(), "caller Bool: writer view");
  check_at(original.write(std::as_bytes(std::span(bits))), "caller Bool: initialize backing");
  auto backing = take_at(writer.publish(), "caller Bool: publish backing");
  auto exact = take_at(backing.subview(1, 1), "caller Bool: exact one-byte sublease");
  ASSERT_EQ(exact.size(), 1U);
  EXPECT_EQ(take_at(exact.view(), "caller Bool: exact lease view").offset(),
      take_at(backing.view(), "caller Bool: backing lease view").offset() + 1);
  auto invocation = take_at(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),
      "caller Bool: create Graph invocation");
  auto refused = invocation.bind(input, bit, exact);
  if (refused.ok()) refused = invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands});
  // The physical allocation has four bytes; the actual caller lease grants
  // only byte1. It cannot authorize a shader's containing 32-bit word load.
  EXPECT_TRUE(refused.IsInvalidArgument()) << "caller Bool: exact lease bind/replay: "
      << hyle::status_code_name(refused.code()) << ": " << refused.message();
  const auto submitted = invocation.submit();
  EXPECT_FALSE(submitted) << "caller Bool: refused exact lease must not submit";
  EXPECT_FALSE(backing.try_write());
  invocation = {}; exact.reset();
  auto recovered = take_at(backing.try_write(), "caller Bool: reclaim backing writer");
  EXPECT_EQ(recovered.generation(), generation);
}

TEST(VulkanExecutionSource, ComputedBoolViewOnlyOutputRetainsItsInputAliasDetachRestriction) {
  NativeEndpoint endpoint;
  const auto floats = take_at(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}), "Bool alias: F32 description");
  const auto bits = take_at(c::TensorDesc::dense(c::DataType::kBool,
      std::array<std::int64_t, 1>{2}), "Bool alias: Bool description");
  const std::array<float, 2> values{0.F, 1.F};
  const auto input = take_at(endpoint.context.upload(floats, std::as_bytes(std::span(values))), "Bool alias: upload");
  const auto producer_program = take_at(v::prepare(endpoint.context, c::Cast{c::DataType::kBool}, floats),
      "Bool alias: prepare producer");
  auto producer = take_at(producer_program.create_invocation(), "Bool alias: create producer");
  check_at(producer.bind(input), "Bool alias: bind producer");
  v::Tensor computed;
  {
    auto submitted = take_at(producer.submit(), "Bool alias: submit producer");
    const auto ready = take_at(submitted.wait(), "Bool alias: retire producer");
    computed = ready->tensors()[0];
  }
  auto original_read = take_at(endpoint.device.import_readonly(computed), "Bool alias: import original read");
  const auto imported = take_at(endpoint.device.binding(original_read, bits), "Bool alias: original native binding");
  const auto slice = take_at(c::View::slice(0, 1, 1), "Bool alias: Slice descriptor");
  const auto program = take_at(v::prepare(endpoint.context, slice, bits), "Bool alias: prepare View");
  const auto &plan = *v::detail::Access::program(program);
  ASSERT_EQ(plan.inputs.size(), 1U); ASSERT_EQ(plan.outputs.size(), 1U);
  // Computed in the producer does not mean owned by this receiving region.
  // Graph's native output contract classifies this same root as BoundGenerated.
  EXPECT_EQ(plan.aliases[plan.outputs[0]].root, plan.inputs[0]);
  auto invocation = take_at(program.create_invocation(), "Bool alias: create invocation");
  check_at(invocation.bind(imported), "Bool alias: bind imported tensor");
  {
    const auto alias = take_at(v::detail::CompositionAccess::output(invocation, 0), "Bool alias: observe cold output");
    EXPECT_EQ(alias.native_buffer(), computed.native_buffer());
    EXPECT_EQ(alias.native_offset(), computed.native_offset() + 1);
    EXPECT_EQ(alias.description().data_type(), c::DataType::kBool);
    EXPECT_EQ(take_at(alias.description().required_bytes(), "Bool alias: output bytes"), 1U);
  } // Drop the observation before sealing the original private-owner baseline.
  check_at(v::detail::CompositionAccess::seal(invocation), "Bool alias: seal original baseline");
  const auto cold_refusal = v::detail::PagedBatchAccess::can_detach_inputs(invocation, {});
  EXPECT_TRUE(cold_refusal.IsUnimplemented()) << "Bool alias: cold detach: "
      << hyle::status_code_name(cold_refusal.code()) << ": " << cold_refusal.message();
  auto submitted = take_at(invocation.submit(), "Bool alias: submit View");
  {
    const auto ready = take_at(submitted.wait(), "Bool alias: retire View");
    std::array<std::byte, 1> actual{};
    check_at(endpoint.context.download(ready->tensors()[0], actual), "Bool alias: read original byte");
    EXPECT_EQ(actual[0], std::byte{1});
  }
  auto retired = c::detail::ResultAccess::take_storage(std::move(submitted));
  const auto retired_refusal = v::detail::PagedBatchAccess::can_detach_inputs(invocation, retired.owner);
  EXPECT_TRUE(retired_refusal.IsUnimplemented()) << "Bool alias: retired detach: "
      << hyle::status_code_name(retired_refusal.code()) << ": " << retired_refusal.message();
  // Neither readiness nor the computed origin grants detach permission.
  // Keep the original imported read, alias carrier and result owners intact.
}

TEST(VulkanExecutionSource, EscapingComputedIdsViewsStayMetadataBesideARealNativeDispatch) {
  NativeEndpoint endpoint;
  const auto matrix = take_at(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}), "escaping IDs: matrix description");
  const auto control_description = take_at(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}), "escaping IDs: control description");
  auto builder = take_at(g::GraphBuilder::create(), "escaping IDs: GraphBuilder");
  const auto input = take_at(builder.input(matrix), "escaping IDs: matrix input");
  const auto external = take_at(builder.input(control_description), "escaping IDs: control input");
  const auto top = take_at(builder.add_node(c::Operation(c::TopK{.count = 1}), std::array{input}),
      "escaping IDs: TopK node");
  const auto ids = take_at(builder.result(top, 1), "escaping IDs: original rank2 IDs");
  const auto row_shape = take_at(c::TensorShape::create(std::array<std::int64_t, 2>{1, 2}),
      "escaping IDs: row shape");
  const auto row_view = take_at(c::View::reshape(row_shape), "escaping IDs: row reshape");
  const auto control = take_at(builder.add_node(c::Operation(row_view), std::array{external}),
      "escaping IDs: external metadata control");
  const auto flat_shape = take_at(c::TensorShape::create(std::array<std::int64_t, 1>{2}),
      "escaping IDs: flat shape");
  const auto flat_view = take_at(c::View::reshape(flat_shape), "escaping IDs: flat reshape");
  const auto flat_node = take_at(builder.add_node(c::Operation(flat_view), std::array{ids}),
      "escaping IDs: flat View node");
  check_at(builder.depends_on(flat_node, control), "escaping IDs: real control frontier");
  const auto flat = take_at(builder.result(flat_node), "escaping IDs: flat result");
  const auto row_node = take_at(builder.add_node(c::Operation(row_view), std::array{ids}),
      "escaping IDs: second escaping View node");
  const auto row = take_at(builder.result(row_node), "escaping IDs: row result");
  // A real, supported dispatch is not ownership evidence for either sibling
  // alias. The over-broad candidate exports both I32 Views and this F32 output.
  const auto widened = take_at(builder.add(c::Cast{c::DataType::kFloat32}, flat),
      "escaping IDs: native I32-to-F32 Cast");
  for (const auto output : {flat, row, widened})
    check_at(builder.publish(output), "escaping IDs: publish exact output");
  const auto graph = take_at(std::move(builder).finish(), "escaping IDs: finish Graph");
  const auto compiled = take_at(g::compile(graph, {endpoint.domains, endpoint.targets}),
      "escaping IDs: cold Graph qualification");
  const auto top_step = compiled_step(compiled, top);
  const auto cast_step = compiled_step(compiled,
      take_at(graph.producer(widened), "escaping IDs: Cast producer"));
  EXPECT_EQ(compiled.report.region_count(), 2U);
  EXPECT_EQ(top_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(cast_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_NE(top_step.first, cast_step.first);
  EXPECT_EQ(compiled_step(compiled, control).second, g::CompileStepKind::kView);
  EXPECT_EQ(compiled_step(compiled, flat_node).second, g::CompileStepKind::kView);
  EXPECT_EQ(compiled_step(compiled, row_node).second, g::CompileStepKind::kView);
  for (const auto output : {flat, row}) {
    bool found = false;
    for (std::size_t index = 0; index < compiled.report.physical_value_count(); ++index) {
      const auto value = take_at(compiled.report.physical_value(index), "escaping IDs: physical View");
      if (value.value != output) continue;
      found = true;
      ASSERT_TRUE(value.alias_source.has_value());
      const auto root = take_at(compiled.report.physical_value(*value.alias_source), "escaping IDs: original alias root");
      EXPECT_EQ(root.value, ids);
      EXPECT_EQ(root.description.data_type(), c::DataType::kInt32);
      EXPECT_EQ(root.description.rank(), 2U);
      EXPECT_EQ(value.storage, root.storage);
      EXPECT_FALSE(value.constant);
    }
    EXPECT_TRUE(found);
  }

  const std::array<std::array<float, 4>, 3> inputs{{
      {1.F, 3.F, 4.F, 2.F}, {5.F, 2.F, 1.F, 6.F}, {-2.F, -1.F, -8.F, -3.F}}};
  const std::array<std::array<std::int32_t, 2>, 3> expected{{{1, 0}, {0, 1}, {1, 1}}};
  auto pool = take_at(c::LeasePool::create(endpoint.device_owner.ref(), 1024,
      {.max_blocks = 4, .backing_alignment = 256}), "escaping IDs: input pool");
  auto writer = take_at(pool.allocate(sizeof(inputs[0]), 256), "escaping IDs: original writer");
  const auto generation = writer.generation();
  const auto original = take_at(writer.view(), "escaping IDs: original writer view");
  check_at(original.write(std::as_bytes(std::span(inputs[0]))), "escaping IDs: initialize first input");
  auto first = take_at(writer.publish(), "escaping IDs: publish first input");
  const std::array<float, 2> controls{3.F, 4.F};
  const auto control_tensor = take_at(endpoint.context.upload(control_description, std::as_bytes(std::span(controls))),
      "escaping IDs: upload control");
  const auto control_read = take_at(endpoint.device.import_readonly(control_tensor), "escaping IDs: control read");
  auto invocation = take_at(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),
      "escaping IDs: create original invocation");
  c::ReadLease escaped;
  for (std::size_t replay = 0; replay < inputs.size(); ++replay) {
    SCOPED_TRACE(replay);
    auto ready = first;
    if (replay != 0) {
      const auto uploaded = take_at(endpoint.context.upload(matrix, std::as_bytes(std::span(inputs[replay]))),
          "escaping IDs: upload next input");
      ready = take_at(endpoint.device.import_readonly(uploaded), "escaping IDs: next input read");
    }
    check_at(invocation.bind(input, matrix, ready), "escaping IDs: bind matrix");
    check_at(invocation.bind(external, control_description, control_read), "escaping IDs: bind control");
    check_at(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}), "escaping IDs: prepare replay");
    auto done = take_at(invocation.submit(), "escaping IDs: submit real native graph");
    check_at(done.wait(), "escaping IDs: retire original graph"); done = {};
    for (const auto output : {flat, row}) {
      const auto result = take_at(invocation.result(output), "escaping IDs: metadata result");
      auto read = take_at(result.storage_lease(), "escaping IDs: original alias read");
      const auto view = take_at(read.view(), "escaping IDs: alias read view");
      std::array<std::int32_t, 2> actual{};
      check_at(view.read(std::as_writable_bytes(std::span(actual))), "escaping IDs: read I32 alias");
      EXPECT_EQ(actual, expected[replay]);
      if (replay + 1 == inputs.size() && output == flat) escaped = std::move(read);
    }
    const auto result = take_at(invocation.result(widened), "escaping IDs: owned Cast result");
    const auto read = take_at(result.storage_lease(), "escaping IDs: owned Cast read");
    const auto view = take_at(read.view(), "escaping IDs: owned Cast view");
    std::array<float, 2> actual{};
    check_at(view.read(std::as_writable_bytes(std::span(actual))), "escaping IDs: read real Cast output");
    for (std::size_t index = 0; index < actual.size(); ++index)
      EXPECT_EQ(actual[index], static_cast<float>(expected[replay][index]));
  }
  invocation = {};
  std::array<std::int32_t, 2> retained{};
  const auto retained_view = take_at(escaped.view(), "escaping IDs: escaped read after invocation destruction");
  check_at(retained_view.read(std::as_writable_bytes(std::span(retained))), "escaping IDs: retained original IDs");
  EXPECT_EQ(retained, expected.back());
  escaped.reset();
  const auto recovered = take_at(first.try_write(), "escaping IDs: reclaim original input after retirement");
  EXPECT_EQ(recovered.generation(), generation);
}

void constant_weight_view_case(bool control_frontier, bool metadata_chain) {
  using BF16 = hyle::simd::bfloat16_t;
  constexpr std::size_t k = 64, n = 128, prefix = 256;
  constexpr std::size_t weight_bytes = k * n * sizeof(BF16);
  NativeEndpoint endpoint;
  const auto activation_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{1, k}).value();
  const auto weight_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{n, k}).value();
  const auto right_desc = weight_desc.transposed().value();
  auto pool = c::LeasePool::create(endpoint.device_owner.ref(), weight_bytes + prefix,
      {.max_blocks = 2, .backing_alignment = 256}).value();
  auto writer = pool.allocate(weight_bytes + prefix, 256).value();
  const auto generation = writer.generation();
  std::array<BF16, (weight_bytes + prefix) / sizeof(BF16)> bytes{};
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < k; ++column)
      bytes[prefix / sizeof(BF16) + row * k + column] = hyle::simd::f32_to_bf16(.25F * float(1 + row % 3));
  check(writer.view()->write(std::as_bytes(std::span(bytes))));
  auto weight_read = writer.publish().value();
  auto weight_span = weight_read.subview(prefix, weight_bytes).value();
  auto native_weight = endpoint.device.binding(weight_span, weight_desc).value();
  const auto original_buffer = native_weight.native_buffer();
  const auto original_offset = native_weight.native_offset();
  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  const auto weight = builder.constant(weight_desc).value();
  // A genuine preceding compute node keeps this View in the old maximal
  // native prefix instead of letting the planner choose a standalone View.
  const auto transformed = builder.add(c::Unary{c::UnaryKind::kNeg}, activation).value();
  const auto permuted = builder.add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(), weight).value();
  auto view = permuted;
  if (metadata_chain)
    view = builder.add(c::View::permute(std::array<std::int32_t, 2>{0, 1}).value(), view).value();
  g::NodeId control;
  if (control_frontier) {
    // This original metadata node is outside the native candidate. Its actual
    // control edge cuts the old constant View region before the MatMul.
    control = builder.add_node(c::Operation(c::View::reshape(c::TensorShape::create(
        std::array<std::int64_t, 1>{k}).value()).value()), std::array{activation}).value();
  }
  const auto matrix = builder.add_node(c::Operation(c::MatMul{}), std::array{transformed, view}).value();
  if (control_frontier) check(builder.depends_on(matrix, control));
  const auto output = builder.result(matrix).value();
  check(builder.publish(output));
  const auto graph = std::move(builder).finish().value();
  std::array constants{g::ConstantBinding{weight,
      c::ConstTensorView::create(weight_desc, weight_span.view().value()).value(),
      g::ConstantPolicy::kBorrow, endpoint.device_owner, weight_span}};
  auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets}, {.constants = constants}).value();
  const auto view_step = compiled_step(compiled, graph.producer(view).value());
  const auto matrix_step = compiled_step(compiled, matrix);
  EXPECT_EQ(view_step.second, g::CompileStepKind::kView);
  EXPECT_EQ(compiled_step(compiled, graph.producer(permuted).value()).second, g::CompileStepKind::kView);
  EXPECT_EQ(matrix_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_NE(view_step.first, matrix_step.first);
  if (!control_frontier) {
    // Immutable metadata at entry is always available. It introduces no
    // external producer/control dependency and must not split real math.
    EXPECT_EQ(compiled.report.region_count(), 1U);
    EXPECT_EQ(compiled_step(compiled, graph.producer(transformed).value()), matrix_step);
  }
  EXPECT_EQ(compiled.report.step(matrix_step.first)->borrowed_constant_count, 1U);
  bool physical_view = false;
  for (std::size_t i = 0; i < compiled.report.physical_value_count(); ++i) {
    const auto value = compiled.report.physical_value(i).value();
    if (value.value != view) continue;
    physical_view = true;
    EXPECT_TRUE(value.constant); EXPECT_EQ(value.description, right_desc);
    EXPECT_TRUE(value.alias_source.has_value());
    if (value.alias_source) {
      auto root = compiled.report.physical_value(*value.alias_source).value();
      EXPECT_EQ(root.value, metadata_chain ? permuted : weight);
      if (metadata_chain) {
        EXPECT_TRUE(root.constant); EXPECT_TRUE(root.alias_source.has_value());
        if (root.alias_source) root = compiled.report.physical_value(*root.alias_source).value();
      }
      EXPECT_EQ(root.value, weight); EXPECT_TRUE(root.constant);
      EXPECT_EQ(root.storage, value.storage);
    }
  }
  EXPECT_TRUE(physical_view);
  // Only the compiled Plan now owns the source's additional read permission.
  constants[0].lease.reset(); weight_span.reset(); native_weight = {};
  EXPECT_FALSE(weight_read.try_write());
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  struct StorageWitness final {
    VkBuffer weight;
    VkDeviceSize offset;
    std::size_t native_uses = 0, full_weight_blocks = 0;
  } witness{original_buffer, original_offset};
  const auto inspect = [](void *opaque, const gd::RegionAccessView &selected) noexcept -> hyle::Status {
    auto &facts = *static_cast<StorageWitness *>(opaque);
    if (!selected.prepared_vtable || !selected.prepared_vtable->output_value)
      return hyle::Status::FailedPrecondition("selected Vulkan MatMul has no original native output");
    auto value = selected.prepared_vtable->output_value(selected.prepared, selected.invocation, 0);
    if (!value) return value.error();
    // This exact selected MatMul belongs to the sole original Vulkan source
    // in this test. Its shared private port type is NativeValue, not a copied
    // Artifact layout. Inspect only the retired original Run's metadata.
    const auto *native = static_cast<const g::vulkan::detail::NativeValue *>(value->native_handle);
    if (!value->native_type || !native || !native->tensor || !native->run || !*native->run)
      return hyle::Status::FailedPrecondition("selected Vulkan MatMul native port is absent");
    const auto &run = **native->run;
    if (run.pending || run.retiring || !run.terminal.ok())
      return hyle::Status::FailedPrecondition("selected Vulkan MatMul has not actually retired");
    std::array<const v::detail::Buffer *, 32> counted{};
    std::size_t count = 0;
    for (const auto &buffer : run.retained_buffers) {
      if (!buffer || buffer->bytes < weight_bytes || buffer->is_view || buffer->external_owner ||
          buffer->memory == VK_NULL_HANDLE) continue;
      if (std::ranges::find(counted, buffer.get()) != counted.end()) continue;
      if (count == counted.size()) return hyle::Status::ResourceExhausted("test storage witness is full");
      counted[count++] = buffer.get(); ++facts.full_weight_blocks;
    }
    for (const auto &stage : run.stages)
      for (const auto &binding : stage.bindings)
        if (binding.handle == facts.weight && binding.offset == facts.offset && binding.capacity >= weight_bytes)
          ++facts.native_uses;
    return hyle::Status::Ok();
  };
  for (unsigned replay = 1; replay <= 3; ++replay) {
    SCOPED_TRACE(replay);
    std::array<BF16, k> input{}; input.fill(hyle::simd::f32_to_bf16(float(replay)));
    check(invocation.bind(activation, activation_desc, endpoint.ready(input)));
    check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
    auto done = invocation.submit().value(); check(done.wait()); done = {};
    std::array<BF16, n> actual{};
    auto read = invocation.result(output)->storage_lease().value();
    check(read.view()->read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t row = 0; row < n; ++row)
      EXPECT_EQ(hyle::simd::bf16_to_f32(actual[row]), -float(replay * k) * .25F * float(1 + row % 3));
    if (replay == 1) {
      check(gd::RegionAccess::visit_retired(invocation, matrix, inspect, &witness));
      std::printf("constant view witness borrowed=%zu native_original_uses=%zu full_weight_blocks=%zu original_offset=%llu\n",
          compiled.report.step(matrix_step.first)->borrowed_constant_count, witness.native_uses,
          witness.full_weight_blocks, static_cast<unsigned long long>(original_offset));
      EXPECT_GT(witness.native_uses, 0U); EXPECT_EQ(witness.full_weight_blocks, 0U);
    }
  }
  invocation = {}; compiled = {};
  auto recovered = weight_read.try_write(); ASSERT_TRUE(recovered) << recovered.error().message();
  EXPECT_EQ(recovered->generation(), generation);
}

TEST(VulkanExecutionSource, ConstantWeightViewAcrossControlFrontierBorrowsOriginalStridedStorage) {
  constant_weight_view_case(true, false);
}

TEST(VulkanExecutionSource, ConstantViewEntryChainDoesNotFragmentTheNativeComputation) {
  // Actual compute prefix -> external immutable Constant/Permute/Permute
  // metadata -> MatMul using both. There is no external control frontier.
  // Check one native region AND original strided storage/lease/no-copy facts
  // so neither needless fragmentation nor hiding the View inside native work
  // can satisfy this regression fixture.
  constant_weight_view_case(false, true);
}

TEST(VulkanExecutionSource, ConstantPrefixViewWithExternalControlRetainsItsOriginalNativeRoute) {
  NativeEndpoint endpoint;
  const auto flat = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const auto square = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(flat).value();
  const auto control = builder.add_node(c::Operation(c::View::reshape(
      c::TensorShape::create(std::array<std::int64_t, 2>{2, 2}).value()).value()), std::array{input}).value();
  const auto constant = builder.constant(square).value();
  const auto view_node = builder.add_node(c::Operation(c::View::permute(
      std::array<std::int32_t, 2>{1, 0}).value()), std::array{constant}).value();
  check(builder.depends_on(view_node, control));
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, builder.result(view_node).value()).value();
  check(builder.publish(output));
  const auto graph = std::move(builder).finish().value();
  const auto read = endpoint.ready(std::array{1.F, 2.F, 3.F, 4.F});
  const std::array constants{g::ConstantBinding{constant,
      c::ConstTensorView::create(square, read.view().value()).value(),
      g::ConstantPolicy::kBorrow, endpoint.device_owner, read}};
  const auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets}, {.constants = constants}).value();
  const auto controlled_view_step = compiled_step(compiled, view_node);
  EXPECT_EQ(controlled_view_step.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled_step(compiled, control).second, g::CompileStepKind::kView);
  EXPECT_NE(controlled_view_step.first, compiled_step(compiled, control).first);
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  check(invocation.bind(input, flat, endpoint.ready(std::array{0.F, 0.F, 0.F, 0.F})));
  check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto done = invocation.submit().value(); check(done.wait()); done = {};
  std::array<float, 4> actual{};
  auto result = invocation.result(output)->storage_lease().value();
  check(result.view()->read(std::as_writable_bytes(std::span(actual))));
  EXPECT_EQ(actual, (std::array{-1.F, -3.F, -2.F, -4.F}));
}

TEST(VulkanExecutionSource, SymbolicConstantSliceKeepsActualOffsetAcrossPreparedSpecializations) {
  NativeEndpoint endpoint;
  const auto backing = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{8}).value();
  const auto output_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto start = builder.dimension({0, 1, 1}).value();
  const auto constant = builder.constant(backing).value();
  const auto selected = builder.add(g::ViewSpec::slice(0, g::Dimension(start),
      g::Dimension::fixed(4)).value(), constant).value();
  const auto view = builder.add(g::ViewSpec::permute(std::array<std::int32_t, 1>{0}).value(), selected).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg},
      g::TensorSpec::concrete(output_desc).value(), view).value();
  check(builder.publish(output));
  const auto graph = std::move(builder).finish().value();
  const auto read = endpoint.ready(std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F});
  const std::array constants{g::ConstantBinding{constant,
      c::ConstTensorView::create(backing, read.view().value()).value(),
      g::ConstantPolicy::kBorrow, endpoint.device_owner, read}};
  for (const std::int64_t offset : {0, 1, 0}) {
    SCOPED_TRACE(offset);
    const std::array shape{g::ShapeBinding{start, offset}};
    const auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets},
        {.shape = shape, .constants = constants}).value();
    auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
    auto done = invocation.submit().value(); check(done.wait()); done = {};
    std::array<float, 4> actual{};
    auto result = invocation.result(output)->storage_lease().value();
    check(result.view()->read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], -float(offset + i + 1));
  }
}

TEST(VulkanExecutionSource, ExternalViewWithInternalControlDoesNotLeaveAndReenterItsRegion) {
  NativeEndpoint endpoint;
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto first_node = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{input}).value();
  const auto first = builder.result(first_node).value();
  const auto view_node = builder.add_node(c::Operation(c::View::reshape(c::TensorShape::create(
      std::array<std::int64_t, 2>{2, 2}).value()).value()), std::array{input}).value();
  check(builder.depends_on(view_node, first_node));
  const auto view = builder.result(view_node).value();
  const auto end = builder.add(c::Unary{c::UnaryKind::kNeg}, view).value();
  check(builder.publish(first)); check(builder.publish(end));
  const auto graph = std::move(builder).finish().value();
  const auto compiled = g::compile(graph, {endpoint.domains, endpoint.targets}).value();
  const auto original = compiled_step(compiled, first_node);
  EXPECT_EQ(original.second, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled_step(compiled, view_node), original);
  EXPECT_EQ(compiled_step(compiled, graph.producer(end).value()), original);
  auto invocation = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  check(invocation.bind(input, desc, endpoint.ready(std::array{1.F, -2.F, 3.F, -4.F})));
  check(invocation.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto done = invocation.submit().value(); check(done.wait()); done = {};
  auto read = invocation.result(end)->storage_lease().value();
  std::array<float, 4> actual{};
  check(read.view()->read(std::as_writable_bytes(std::span(actual))));
  EXPECT_EQ(actual, (std::array{-1.F, 2.F, -3.F, 4.F}));
}

TEST(VulkanExecutionSource, ActualOwnerResourceAdmissionAndStableBacking) {
  auto context = v::Context::create().value();
  auto foreign = v::Context::create().value();
  auto source = g::vulkan::execution_source(context).value();
  const auto again = g::vulkan::execution_source(context).value();
  const auto other = g::vulkan::execution_source(foreign).value();
  EXPECT_TRUE(source.same_owner(again));
  EXPECT_TRUE(source.same_device(again));
  EXPECT_FALSE(source.same_owner(other));
  EXPECT_FALSE(source.same_device(other));
  const g::vulkan::CustomLowering custom = [](const c::Operation &,
      std::span<const c::TensorDesc>, const c::TensorDesc &)
      -> hyle::Result<g::vulkan::CustomDispatch> {
    return std::unexpected(hyle::Status::Unimplemented("unused cold custom configuration"));
  };
  const auto configured = g::vulkan::execution_source(context, {.custom = custom}).value();
  const auto configured_copy = configured;
  const auto independently_configured =
      g::vulkan::execution_source(context, {.custom = custom}).value();
  EXPECT_TRUE(configured.same_owner(configured_copy));
  EXPECT_FALSE(configured.same_owner(source));
  EXPECT_FALSE(configured.same_owner(independently_configured));
  EXPECT_TRUE(configured.same_device(source));
  EXPECT_TRUE(configured.same_device(independently_configured));
  const auto facts = source.device_properties().value();
  EXPECT_EQ(facts.name, context.environment().device_name);
  EXPECT_EQ(facts.backend, source.backend_id());
  EXPECT_FALSE(gd::ExecutionSourceAccess::compute(source));
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto wrong_memory = v::MemoryResource::create(foreign, v::MemoryKind::kDevice).value();
  EXPECT_TRUE(source.validate_resource(memory.ref()).ok());
  EXPECT_TRUE(source.validate_resource(staging.ref()).ok());
  EXPECT_FALSE(source.validate_resource(wrong_memory.ref()).ok());
  EXPECT_FALSE(source.validate_resource(hyle::mem::system_resource()).ok());
  static const unsigned capability = 7;
  const auto region_source = gd::GraphRegionSource::native(source, &capability).value();
  EXPECT_TRUE(region_source.is_native());
  EXPECT_EQ(region_source.shared_extension(), nullptr);
  EXPECT_FALSE(region_source.provider_instance_id().valid());
  struct CapabilityCall final {
    const void *owner;
    unsigned calls = 0;
  } native_call{gd::ExecutionSourceAccess::native_context(source).value()};
  const auto invoke_capability = [](void *owner, const void *table, void *opaque) noexcept {
    auto &call = *static_cast<CapabilityCall *>(opaque);
    if (owner != call.owner || table != &capability)
      return hyle::Status::InvalidArgument("native source lost its actual owner/capability");
    ++call.calls;
    return hyle::Status::Ok();
  };
  ASSERT_TRUE(region_source.call(invoke_capability, &native_call).ok());

  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const std::array<float, 4> values{0.25f, -1.5f, 3.75f, -0.0f};
  const auto input = context.upload(description, std::as_bytes(std::span(values))).value();
  const auto program = v::prepare(context, c::Unary{c::UnaryKind::kNeg}, description).value();
  auto slot = program.create_invocation().value();
  EXPECT_FALSE(g::vulkan::detail::output_backing(other, slot, 0));
  // This handle is uninitialized backing, not a ready ReadLease. Do not read
  // or import it until the original native result reports successful retirement.
  const auto backing = g::vulkan::detail::output_backing(source, slot, 0).value();
  const auto native_buffer = backing.native_buffer();
  const auto native_offset = backing.native_offset();
  ASSERT_TRUE(g::vulkan::detail::seal_outputs(source, slot).ok());
  ASSERT_TRUE(slot.bind(input).ok());
  v::Tensor old_output;
  {
    auto submitted = slot.submit();
    ASSERT_TRUE(submitted) << submitted.error().message();
    auto result = submitted->wait();
    ASSERT_TRUE(result) << result.error().message();
    old_output = result->get()->tensors()[0];
    EXPECT_EQ(old_output.native_buffer(), native_buffer);
    EXPECT_EQ(old_output.native_offset(), native_offset);
    std::array<float, 4> observed{};
    ASSERT_TRUE(context.download(old_output, std::as_writable_bytes(std::span(observed))).ok());
    for (std::size_t i = 0; i < values.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(observed[i]),
          std::bit_cast<std::uint32_t>(-values[i]));
  }
  EXPECT_FALSE(slot.submit()); // Native Tensor owner still blocks overwrite.
  old_output = {};
  {
    auto submitted = slot.submit();
    ASSERT_TRUE(submitted) << submitted.error().message();
    ASSERT_TRUE(submitted->wait());
  }
  unsigned calls = 0;
  const auto invoke = [](void *state) noexcept {
    ++*static_cast<unsigned *>(state);
    return hyle::Status::Ok();
  };
  EXPECT_TRUE(gd::ExecutionSourceAccess::call(source, invoke, &calls).ok());
  ASSERT_TRUE(context.close().ok());
  EXPECT_FALSE(source.is_open());
  EXPECT_FALSE(gd::ExecutionSourceAccess::call(source, invoke, &calls).ok());
  EXPECT_FALSE(region_source.call(invoke_capability, &native_call).ok());
  EXPECT_EQ(native_call.calls, 1U);
  EXPECT_EQ(calls, 1U);
  EXPECT_FALSE(g::vulkan::execution_source(context));
  // Closing admission does not invalidate a retained backing owner.
  std::array<float, 4> observed{};
  EXPECT_TRUE(context.download(backing, std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_TRUE(source.same_owner(again));
}

TEST(VulkanExecutionSource, RetiredCrossDomainCopyUnbindsAndUsesNewBacking) {
  auto context = v::Context::create().value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  const auto source = g::vulkan::execution_source(context).value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{staging.ref()}, g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{1}}};
  const std::array external{g::ExternalResourceContract{input, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets},
      {.external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = take_at(compiled->executable.create_invocation(), "cross-domain create_invocation");
  ASSERT_TRUE(invocation.prepare_resources().ok());
  const std::array<std::array<float, 4>, 2> values{{
      {0.25f, -1.5f, 3.75f, -0.0f}, {2.5f, -8.0f, 0.0f, 9.0f}}};
  std::array<hyle::mem::Buffer, 2> backings;
  for (auto &backing : backings)
    backing = hyle::mem::Buffer::create(staging.ref(), sizeof(values[0]), 64).value();
  ASSERT_NE(backings[0].view().native_handle(), backings[1].view().native_handle());
  for (std::size_t episode = 0; episode < values.size(); ++episode) {
    SCOPED_TRACE(episode);
    ASSERT_TRUE(backings[episode].view().write(std::as_bytes(std::span(values[episode]))).ok());
    const auto binding = c::ConstTensorView::create(description, backings[episode].view()).value();
    ASSERT_TRUE(invocation.bind(input, binding).ok());
    ASSERT_TRUE(invocation.prepare_replay({}).ok());
    {
      auto done = invocation.submit();
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      auto result = take_at(invocation.result(output), "cross-domain result");
      auto native = take_at(device.binding(take_at(result.storage_lease(), "cross-domain result lease"), description), "cross-domain native binding");
      std::array<float, 4> actual{};
      ASSERT_TRUE(context.download(native, std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[index]),
            std::bit_cast<std::uint32_t>(-values[episode][index]));
    }
    // This must release the native copy's actual VkBuffer pins. The second
    // episode reuses this slot with another backing, without reset or drain.
    const auto released = invocation.unbind(input);
    ASSERT_TRUE(released.ok()) << released.message();
  }
}
TEST(VulkanExecutionSource, NativeSubmissionDependencyUsesOriginalOwnersAndWholeRetirement) {
  NativeEndpoint endpoint, foreign;
  NativeSourceProgram program(endpoint, 4), foreign_program(foreign, 4);
  auto producer = program.invocation(), consumer = program.invocation();
  auto other = foreign_program.invocation();
  const std::array<float, 4> values{1, -2, 3, -4};
  check(producer.bind(program.input, program.input_description, endpoint.ready(values)));
  check(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  const auto eligible = g::SubmissionSource::check_native_dependency(producer, consumer);
  ASSERT_TRUE(eligible.ok()) << eligible.message();
  EXPECT_TRUE(g::SubmissionSource::check_native_dependency(producer, other).IsUnimplemented());
  auto submitted = producer.submit().value();
  const auto wrong = g::SubmissionSource::create(producer, std::move(submitted), foreign.device_owner);
  EXPECT_FALSE(wrong); ASSERT_TRUE(submitted.valid());
  auto source = g::SubmissionSource::create(producer, std::move(submitted), endpoint.device_owner).value();
  auto view = g::OutputSource::create(source, program.output, endpoint.device_owner, program.outputs).value();
  EXPECT_FALSE(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}).ok());
  EXPECT_FALSE(g::OutputSource::create(source, program.output, endpoint.staging_owner, program.outputs));
  const std::array bindings{view.reader_binding(program.input)};
  check(g::bind_sources(consumer, bindings)); check(g::prepare_sources(consumer, {}));
  const auto report = consumer.input_source_report().value();
  EXPECT_EQ(report.source_count, 1U); EXPECT_GT(report.device_wait_count, 0U);
  EXPECT_EQ(report.host_wait_count, 0U);
  EXPECT_FALSE(consumer.submit());
  auto finished = g::submit_sources(consumer, {}).value();
  check(finished.wait());
  EXPECT_EQ(program.read(consumer), std::vector<float>(values.begin(), values.end()));
  finished = {};
  check(source.wait()); check(view.detach()); check(g::release_sources(consumer)); check(source.detach());
  EXPECT_EQ(source.poll().value(), c::CompletionPoll::kComplete);
}

TEST(VulkanExecutionSource, PendingNativeDependencyPreservesCheckedProducerFailure) {
  for (const bool prefix_views : {false, true}) {
  SCOPED_TRACE(prefix_views);
  NativeEndpoint endpoint;
  NativeSourceProgram failing(endpoint, 4, true, prefix_views), consuming(endpoint, 1, false, prefix_views);
  auto producer = failing.invocation(), consumer = consuming.invocation();
  check(producer.bind(failing.input, failing.input_description, endpoint.ready(std::array{1.F,2.F,3.F,4.F})));
  check(producer.bind(failing.indices, failing.indices_description, endpoint.ready(std::array<std::int32_t,1>{-1})));
  check(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  ASSERT_TRUE(g::SubmissionSource::check_native_dependency(producer, consumer).ok());
  c::Completion submitted;
  g::SubmissionSource source;
  g::OutputSource view;
  HoldObservation hold(endpoint.context);
  submitted = producer.submit().value();
  source = g::SubmissionSource::create(producer, std::move(submitted), endpoint.device_owner).value();
  view = g::OutputSource::create(source, failing.output, endpoint.device_owner, failing.outputs).value();
  const std::array bindings{view.reader_binding(consuming.input)};
  check(g::bind_sources(consumer, bindings)); check(g::prepare_sources(consumer, {}));
  const auto report = consumer.input_source_report().value();
  EXPECT_GT(report.device_wait_count, 0U); EXPECT_EQ(report.host_wait_count, 0U);
  auto finished = g::submit_sources(consumer, {}).value();
  hold.held = false;
  const auto failure = finished.wait();
  EXPECT_TRUE(failure.IsOutOfRange() || failure.IsInvalidArgument()) << failure.message();
  EXPECT_EQ(source.wait().code(), failure.code());
  EXPECT_FALSE(producer.result(failing.output)); EXPECT_FALSE(consumer.result(consuming.output));
  finished = {};
  EXPECT_FALSE(view.detach().ok()); view = {};
  check(g::release_sources(consumer)); check(source.detach());
  EXPECT_FALSE(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}).ok());
  auto retry = failing.invocation();
  check(retry.bind(failing.input, failing.input_description, endpoint.ready(std::array{1.F,2.F,3.F,4.F})));
  check(retry.bind(failing.indices, failing.indices_description, endpoint.ready(std::array<std::int32_t,1>{0})));
  check(retry.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto fresh = retry.submit().value(); check(fresh.wait());
  EXPECT_EQ(failing.read(retry), std::vector<float>{1.F});
  }
}

TEST(VulkanExecutionSource, OriginalCancellationDoesNotAuthorizeEarlySourceOrSlotReuse) {
  for (const bool prefix_views : {false, true}) {
  SCOPED_TRACE(prefix_views);
  NativeEndpoint endpoint;
  NativeSourceProgram program(endpoint, 4, false, prefix_views);
  auto producer = program.invocation();
  check(producer.bind(program.input, program.input_description, endpoint.ready(std::array{1.F,2.F,3.F,4.F})));
  check(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  c::Completion submitted;
  g::SubmissionSource source;
  g::OutputSource view;
  HoldObservation hold(endpoint.context);
  submitted = producer.submit().value();
  source = g::SubmissionSource::create(producer, std::move(submitted), endpoint.device_owner).value();
  view = g::OutputSource::create(source, program.output, endpoint.device_owner, program.outputs).value();
  EXPECT_EQ(source.poll().value(), c::CompletionPoll::kPending);
  check(source.request_cancel());
  EXPECT_FALSE(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}).ok());
  EXPECT_FALSE(view.detach().ok());
  hold.held = false;
  EXPECT_TRUE(source.wait().IsCancelled());
  EXPECT_FALSE(producer.result(program.output));
  view = {}; check(source.detach());
  auto retry = program.invocation();
  check(retry.bind(program.input, program.input_description, endpoint.ready(std::array{2.F,3.F,4.F,5.F})));
  check(retry.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto fresh = retry.submit().value(); check(fresh.wait());
  EXPECT_EQ(program.read(retry), (std::vector<float>{-2.F,-3.F,-4.F,-5.F}));
  }
}
} // namespace
