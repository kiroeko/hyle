#include "native_graph.h"
#include "gallocr_faults.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "ggml-backend.h"
#include "ggml.h"

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeGraph;
using hyle::compute::ggml_vulkan::detail::NativeOptions;

struct ColdGraph {
  MetadataArena arena;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *weight = nullptr;
  ggml_tensor *output = nullptr;
};
hyle::Result<ColdGraph> make_chain(int nodes = 256) {
  ColdGraph result;
  HYLE_ASSIGN_OR_RETURN(result.arena, MetadataArena::create(
      (nodes + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes + 8, false),
      hyle::mem::system_allocator()));
  auto *context = result.arena.get();
  result.input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  result.weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  ggml_set_input(result.input);
  ggml_set_input(result.weight);
  result.output = result.input;
  for (int i = 0; i < nodes; ++i)
    result.output = ggml_add(context, result.output, result.weight);
  ggml_set_output(result.output);
  result.graph = ggml_new_graph_custom(context, nodes + 8, false);
  ggml_build_forward_expand(result.graph, result.output);
  return result;
}

TEST(GgmlVulkanStorage, LivenessReuseAndPendingLeaseRetireQuota) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1 << 20);
  NativeOptions options;
  options.quota = &quota;
  auto device = NativeDevice::create(options);
  ASSERT_TRUE(device);
  const auto device_charge = quota.used();
  auto graph = make_chain();
  ASSERT_TRUE(graph);
  auto storage = NativeGraph::allocate(*device, std::move(graph->arena), graph->graph);
  ASSERT_TRUE(storage);
  EXPECT_GT(storage->storage_bytes(), 0u);
  // A live-range planner needs much less than one allocation per node.
  EXPECT_LT(storage->storage_bytes(), 256u * 2048u * sizeof(float) / 4);
  auto lease = storage->lease();
  std::weak_ptr<const void> alive = lease;
  auto work = device->create_work();
  ASSERT_TRUE(work);
  {
    auto active = device->activate();
    const std::vector<float> input(2048, 0.5f), weight(2048, 0.25f);
    ggml_backend_tensor_set(graph->input, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(graph->weight, weight.data(), 0, weight.size() * sizeof(float));
  }
  ASSERT_TRUE(work->submit(storage->get(), std::move(lease)).status.ok());
  *storage = {};
  EXPECT_FALSE(alive.expired());
  EXPECT_GT(quota.used(), device_charge);
  // Keep the graph through readback, independently of the completed work.
  auto retained = alive.lock();
  ASSERT_TRUE(work->wait().status.ok());
  {
    auto active = device->activate();
    std::vector<float> output(2048);
    ggml_backend_tensor_get(graph->output, output.data(), 0, output.size() * sizeof(float));
    for (const auto value : output) ASSERT_EQ(value, 64.5f);
  }
  retained.reset();
  EXPECT_TRUE(alive.expired());
  *work = {};
  *device = {};
  EXPECT_EQ(quota.used(), 0u);
}

TEST(GgmlVulkanStorage, ColdQuotaFailureAndMissingOutputReturnErrors) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  NativeOptions options;
  options.quota = &quota;
  auto device = NativeDevice::create(options);
  ASSERT_TRUE(device);
  auto graph = make_chain(4);
  ASSERT_TRUE(graph);
  auto rejected = NativeGraph::allocate(*device, std::move(graph->arena), graph->graph);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::ResourceExhausted);
  EXPECT_EQ(quota.used(), 0u);
  graph = make_chain(4);
  ASSERT_TRUE(graph);
  graph->output->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
  rejected = NativeGraph::allocate(*device, std::move(graph->arena), graph->graph);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
  EXPECT_EQ(quota.used(), 0u);
}

TEST(GgmlVulkanStorage, EveryColdPlannerHostAllocationFailureIsRecoverable) {
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto active = device->activate();
  auto backend = device->create_backend();
  ASSERT_TRUE(backend);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> owner(*backend, ggml_backend_free);
  auto *type = ggml_backend_get_default_buffer_type(owner.get());
  auto attempt = [&](int fail_at) {
    auto graph = make_chain(8);
    EXPECT_TRUE(graph);
    if (!graph) return false;
    hyle_test_gallocr_fail_at(fail_at);
    auto *allocator = hyle_test_gallocr_new(type);
    const bool ready = allocator && hyle_test_gallocr_reserve(allocator, graph->graph);
    const bool allocated = ready && hyle_test_gallocr_alloc_graph(allocator, graph->graph);
    hyle_test_gallocr_free(allocator);
    EXPECT_EQ(hyle_test_gallocr_live(), 0);
    return allocated;
  };
  ASSERT_TRUE(attempt(0));
  const auto allocations = hyle_test_gallocr_calls();
  ASSERT_GT(allocations, 5);
  for (int allocation = 1; allocation <= allocations; ++allocation) {
    SCOPED_TRACE(allocation);
    EXPECT_FALSE(attempt(allocation));
  }
  EXPECT_TRUE(attempt(0));
  RecordProperty("host_allocation_failure_points", allocations);
}

TEST(GgmlVulkanStorage, ImportedStorageNeedsAnOwnerAndTheSameDevice) {
  auto first = NativeDevice::create();
  auto second = NativeDevice::create();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto allocation = first->allocate_buffer(2048 * sizeof(float));
  ASSERT_TRUE(allocation);
  std::shared_ptr<ggml_backend_buffer> buffer(*allocation,
      [device = *first](ggml_backend_buffer *buffer) {
        auto active = device.activate();
        ggml_backend_buffer_free(buffer);
      });
  auto bind = [&](ColdGraph &graph) {
    auto active = first->activate();
    return ggml_backend_tensor_alloc(buffer.get(), graph.input,
                                    ggml_backend_buffer_get_base(buffer.get()));
  };
  auto graph = make_chain(4);
  ASSERT_TRUE(graph);
  ASSERT_EQ(bind(*graph), GGML_STATUS_SUCCESS);
  auto rejected = NativeGraph::allocate(*first, std::move(graph->arena), graph->graph);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
  graph = make_chain(4);
  ASSERT_TRUE(graph);
  ASSERT_EQ(bind(*graph), GGML_STATUS_SUCCESS);
  rejected = NativeGraph::allocate(*second, std::move(graph->arena), graph->graph, buffer);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
  graph = make_chain(4);
  ASSERT_TRUE(graph);
  ASSERT_EQ(bind(*graph), GGML_STATUS_SUCCESS);
  auto accepted = NativeGraph::allocate(*first, std::move(graph->arena), graph->graph, buffer);
  ASSERT_TRUE(accepted);
  std::weak_ptr<ggml_backend_buffer> alive = buffer;
  buffer.reset();
  EXPECT_FALSE(alive.expired());
  *accepted = {};
  EXPECT_TRUE(alive.expired());
}

TEST(GgmlVulkanStorage, FragmentedPlannerCapacityReturnsFailureWithoutAbort) {
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto active = device->activate();
  auto backend = device->create_backend();
  ASSERT_TRUE(backend);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> owner(*backend, ggml_backend_free);
  auto arena = MetadataArena::create(
      1024 * ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false),
      hyle::mem::system_allocator());
  ASSERT_TRUE(arena);
  auto *context = arena->get();
  auto *graph = ggml_new_graph_custom(context, 1024, false);
  auto *shape = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1024);
  for (int i = 0; i < 300; ++i) {
    auto *input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    auto *keeper = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    auto *output = ggml_repeat(context, input, shape);
    ggml_set_output(keeper);
    ggml_set_output(output);
    // Live keeper leaves separate the small freed input blocks. The large
    // outputs cannot consume those holes, exceeding upstream's finite table.
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, keeper);
  }
  hyle_test_gallocr_fail_at(0);
  auto *allocator = hyle_test_gallocr_new(ggml_backend_get_default_buffer_type(owner.get()));
  ASSERT_NE(allocator, nullptr);
  EXPECT_FALSE(hyle_test_gallocr_reserve(allocator, graph));
  hyle_test_gallocr_free(allocator);
  EXPECT_EQ(hyle_test_gallocr_live(), 0);
}
} // namespace
