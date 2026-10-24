#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <thread>

#include "hyle/graph/vulkan.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/progress.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../../src/compute_route_vulkan/vulkan_weights_internal.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/vulkan_batch_access.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace vn = v::detail;
using namespace std::chrono_literals;
const auto weight_description = c::TensorDesc::dense(c::DataType::kFloat32,
    std::array<std::int64_t, 1>{4}).value();

struct HostWeight final {
  m::Buffer storage;
  c::ConstTensorView tensor;
};
HostWeight host_weight(const std::array<float, 4> &values) {
  auto storage = m::Buffer::create(m::system_resource(), sizeof(values), 4).value();
  EXPECT_TRUE(storage.view().write(std::as_bytes(std::span(values))).ok());
  auto tensor = c::ConstTensorView::create(weight_description, storage.view()).value();
  return {std::move(storage), tensor};
}
struct WeightGraph final {
  g::Graph graph;
  g::ValueId left, right, output;
};
WeightGraph weight_graph() {
  auto builder = g::GraphBuilder::create().value();
  const auto left = builder.input(weight_description).value();
  const auto right = builder.input(weight_description).value();
  const auto output = builder.add(c::Binary{c::BinaryKind::kAdd}, left, right).value();
  EXPECT_TRUE(builder.publish(output).ok());
  return {std::move(builder).finish().value(), left, right, output};
}
void expect_weight(const v::Context &context, const g::GraphOutput &output,
                   const std::array<float, 4> &expected) {
  auto projected = output.tensor();
  ASSERT_TRUE(projected) << projected.error().message();
  auto native = vn::MemoryAccess::tensor_view(vn::Access::runtime(context), *projected,
                                             m::system_allocator());
  ASSERT_TRUE(native) << native.error().message();
  std::array<float, 4> actual{};
  ASSERT_TRUE(context.download(*native, std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
              std::bit_cast<std::uint32_t>(expected[i])) << i;
}
template<class Predicate> bool eventually(Predicate predicate) {
  const auto end = std::chrono::steady_clock::now() + 5s;
  do {
    if (predicate()) return true;
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < end);
  return predicate();
}

// Observe the real upload fence through the real API, but withhold successful
// observation while armed. This proves readiness/ownership behavior, not that
// the physical transfer remains in flight or a throughput result.
struct HoldFenceObservation final {
  vn::Device &device;
  PFN_vkGetFenceStatus original_poll;
  PFN_vkWaitForFences original_wait;
  std::atomic<bool> held{true};
  std::atomic<unsigned> polls{0}, waits{0};
  static inline HoldFenceObservation *active = nullptr;
  explicit HoldFenceObservation(const v::Context &context)
      : device(*vn::Access::runtime(context)->device),
        original_poll(device.fn.GetFenceStatus), original_wait(device.fn.WaitForFences) {
    active = this;
    device.fn.GetFenceStatus = poll;
    device.fn.WaitForFences = wait;
  }
  ~HoldFenceObservation() {
    release();
    device.fn.GetFenceStatus = original_poll;
    device.fn.WaitForFences = original_wait;
    active = nullptr;
  }
  void release() { held.store(false, std::memory_order_release); }
  static VKAPI_ATTR VkResult VKAPI_CALL poll(VkDevice device, VkFence fence) {
    auto &self = *active;
    const auto result = self.original_poll(device, fence);
    self.polls.fetch_add(1, std::memory_order_release);
    if (self.held.load(std::memory_order_acquire) &&
        (result == VK_SUCCESS || result == VK_NOT_READY)) return VK_NOT_READY;
    return result;
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device, std::uint32_t count,
      const VkFence *fences, VkBool32 all, std::uint64_t timeout) {
    auto &self = *active;
    self.waits.fetch_add(1, std::memory_order_release);
    return self.original_wait(device, count, fences, all, timeout);
  }
};


// Every is_view owner exposes its direct Buffer parent. Its control block may
// independently retain a lease, exact page frame, or immutable weight selection.
// Keep the structural ASSERT before CompileReport's Buffer-only parent walk:
// the old implementation must fail safely rather than dereference a wrong type.
void expect_constant_report(const v::Context &context, const v::Tensor &tensor,
                            const vn::Buffer &root) {
  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.constant(tensor.description()).value();
  const auto b = builder.constant(tensor.description()).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, a, b).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  auto graph = std::move(builder).finish().value();
  g::vulkan::CompileReport report;
  auto program = g::vulkan::compile(context, graph, std::array{tensor, tensor},
                                   {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  ASSERT_TRUE(report.status.ok()) << report.status.message();
  EXPECT_EQ(report.retained_constant_owner_count, 1U);
  EXPECT_EQ(report.retained_constant_capacity_bytes, root.bytes);
  EXPECT_EQ(report.owned_constant_allocation_bytes, root.allocation_bytes);
  EXPECT_EQ(report.has_borrowed_constants, root.memory == VK_NULL_HANDLE);
}

TEST(VulkanWeightBinding, LeaseViewExposesBufferParentAndKeepsOriginalLease) {
  auto context = v::Context::create().value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const std::array<float, 4> values{1, -2, 3, 4};
  auto source = context.upload(weight_description, std::as_bytes(std::span(values))).value();
  auto root = vn::Access::buffer(source);
  std::weak_ptr<const void> retained = root;
  auto ready = device.import_readonly(source).value();
  auto view = device.binding(ready, weight_description).value();
  const auto &native = vn::Access::buffer(view);
  ASSERT_TRUE(native->is_view);
  ASSERT_EQ(native->external_owner.get(), root.get());
  // The alias pointer changes, but its control block remains the lease owner.
  EXPECT_TRUE(native->external_owner.owner_before(root) ||
              root.owner_before(native->external_owner));
  ready.reset(); source = {};
  EXPECT_FALSE(retained.expired());
  expect_constant_report(context, view, *root);
  std::array<float, 4> actual{};
  ASSERT_TRUE(context.download(view, std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, values);
  root.reset();
  EXPECT_FALSE(retained.expired());
  view = {};
  EXPECT_TRUE(retained.expired());
}

TEST(VulkanWeightBinding, PagedViewExposesBufferParentAndRetainsExactPage) {
  auto context = v::Context::create().value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  c::BatchCapacity capacity;
  capacity.qk_head_dim = capacity.value_head_dim = 4;
  capacity.page_size = 1;
  auto schema = c::BatchSchema::create(capacity).value();
  auto arena = c::PagedArena::create(schema, c::BatchCacheRole::kKey, device.ref(),
                                    {.max_backing_bytes = 64}).value();
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat16,
                                        std::array<std::int64_t, 4>{1, 1, 1, 4}).value();
  auto writer = arena.acquire(0).value();
  auto writable = writer.lease().tensor(desc).value();
  auto original = vn::MemoryAccess::tensor_view(vn::Access::runtime(context),
      writable.as_const(), m::system_allocator()).value();
  const auto *root = vn::Access::buffer(original).get();
  // tensor_view preserves the ordinary Buffer-only parent chain. The arena
  // includes its sentinel byte, so this exact page may be a regular subview.
  while (root->is_view)
    root = static_cast<const vn::Buffer *>(root->external_owner.get());
  const std::array<std::uint16_t, 4> values{0x4000, 0x4400, 0xbc00, 0x4800};
  ASSERT_TRUE(context.upload(original, std::as_bytes(std::span(values))).ok());
  auto ready = writer.publish(1).value();
  const std::array<std::int32_t, 2> indptr{0, 1};
  const std::array<std::int32_t, 1> page{0}, tail{1};
  auto layout = schema.version({indptr, indptr, page, tail}).value();
  auto frame = arena.read(layout, std::array{ready}).value();
  auto view = vn::PagedBatchAccess::binding(device, frame).value();
  ASSERT_TRUE(vn::Access::buffer(view)->is_view);
  ASSERT_EQ(vn::Access::buffer(view)->external_owner.get(), root);
  frame = {}; ready.reset(); original = {}; writer = {};
  EXPECT_FALSE(arena.acquire(0));
  expect_constant_report(context, view, *root);
  view = {};
  auto reused = arena.acquire(0);
  ASSERT_TRUE(reused) << reused.error().message();
}

TEST(VulkanWeightBinding, WeightViewsExposeBufferParentAndRetainSelection) {
  auto context = v::Context::create().value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const std::array catalog{v::WeightSpec{10, weight_description}};
  auto weights = v::WeightSlots::create(context, staging, device, catalog, 1024).value();
  auto host = host_weight({1, -2, 3, 4});
  const std::array selection{v::WeightFragment{10, host.tensor}};
  auto batch = weights.upload(0, selection).value();
  ASSERT_TRUE(batch.wait().ok());
  auto public_view = batch.binding(10).value();
  auto binding = vn::WeightAccess::binding(batch, 10).value();
  auto private_view = vn::MemoryAccess::tensor_view(vn::Access::runtime(context),
      binding.tensor, m::system_allocator()).value();
  const auto &public_buffer = vn::Access::buffer(public_view);
  const auto &private_buffer = vn::Access::buffer(private_view);
  ASSERT_TRUE(public_buffer->is_view);
  ASSERT_TRUE(private_buffer->is_view);
  // WeightState is deliberately opaque. Prove neither alias exposes that
  // control object before following either Buffer-only parent pointer.
  EXPECT_NE(public_buffer->external_owner.get(), binding.owner.get());
  EXPECT_NE(private_buffer->external_owner.get(), binding.owner.get());
  if (public_buffer->external_owner.get() == binding.owner.get() ||
      private_buffer->external_owner.get() == binding.owner.get()) return;
  ASSERT_EQ(public_buffer->external_owner.get(), private_buffer->external_owner.get());
  const auto *root = static_cast<const vn::Buffer *>(public_buffer->external_owner.get());
  ASSERT_NE(root, nullptr);
  ASSERT_FALSE(root->is_view);
  EXPECT_EQ(root->handle, public_view.native_buffer());
  EXPECT_EQ(root->handle, private_view.native_buffer());
  EXPECT_FALSE(public_buffer->external_owner.owner_before(binding.owner));
  EXPECT_FALSE(binding.owner.owner_before(public_buffer->external_owner));
  EXPECT_FALSE(private_buffer->external_owner.owner_before(binding.owner));
  EXPECT_FALSE(binding.owner.owner_before(private_buffer->external_owner));
  expect_constant_report(context, public_view, *root);
  expect_constant_report(context, private_view, *root);
  batch = {}; binding = {};
  EXPECT_FALSE(weights.upload(0, selection));
  public_view = {};
  EXPECT_FALSE(weights.upload(0, selection));
  private_view = {};
  auto reused = weights.upload(0, selection);
  ASSERT_TRUE(reused) << reused.error().message();
  ASSERT_TRUE(reused->wait().ok());
}

TEST(VulkanWeightBinding, OriginalInvocationRetainsRealSelectionAndDeduplicatesOwner) {
  auto context = v::Context::create().value();
  RecordProperty("vulkan_device", context.environment().device_name);
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  m::QuotaAllocator weight_metadata(m::system_allocator());
  const std::array catalog{v::WeightSpec{10, weight_description}, v::WeightSpec{20, weight_description}};
  auto weights = v::WeightSlots::create(context, staging, device, catalog, 1024,
      {.allocator = weight_metadata.get()}).value();
  const auto a = host_weight({1.0f, -2.0f, 3.5f, 0.25f});
  const auto b = host_weight({-0.5f, 4.0f, 2.5f, -0.25f});
  const auto c0 = host_weight({2.0f, 0.5f, -1.0f, 4.0f});
  const auto d = host_weight({-1.0f, -0.25f, 2.0f, -2.0f});
  const std::array selection0{v::WeightFragment{10, a.tensor}, v::WeightFragment{20, b.tensor}};
  const std::array selection1{v::WeightFragment{10, c0.tensor}, v::WeightFragment{20, d.tensor}};
  auto batch0 = weights.upload(0, selection0).value();
  auto batch1 = weights.upload(1, selection1).value();
  auto definition = weight_graph();
  const auto source = g::vulkan::execution_source(context).value();
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(definition.graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->executable.required_queue_count(), 0U);
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation().value();
  ASSERT_TRUE(g::vulkan::bind(first, definition.left, batch0, 10).ok());
  ASSERT_TRUE(g::vulkan::bind(first, definition.right, batch0, 20).ok());
  const auto missing = g::vulkan::bind(first, definition.left, batch0, 99);
  EXPECT_TRUE(missing.IsNotFound()) << missing.message();
  {
    auto foreign = v::Context::create().value();
    auto foreign_staging = v::MemoryResource::create(foreign, v::MemoryKind::kStaging).value();
    auto foreign_device = v::MemoryResource::create(foreign, v::MemoryKind::kDevice).value();
    auto foreign_weights = v::WeightSlots::create(foreign, foreign_staging, foreign_device,
                                                 catalog, 1024).value();
    auto foreign_batch = foreign_weights.upload(0, selection0).value();
    const auto wrong_owner = g::vulkan::bind(first, definition.left, foreign_batch, 10);
    EXPECT_TRUE(wrong_owner.IsInvalidArgument()) << wrong_owner.message();
    ASSERT_TRUE(foreign_batch.wait().ok());
  }
  const auto rejected = first.prepare_replay({}, {.host_wait = g::HostWaitPolicy::kReject});
  EXPECT_TRUE(rejected.IsUnimplemented()) << rejected.message();
  ASSERT_TRUE(first.prepare_replay({}).ok());
  const auto prepared_metadata_bytes = weight_metadata.used();
  ASSERT_TRUE(first.prepare_replay({}).ok());
  EXPECT_EQ(weight_metadata.used(), prepared_metadata_bytes);
  auto report = first.input_source_report().value();
  EXPECT_EQ(report.source_count, 1U);
  EXPECT_EQ(report.host_wait_count, 1U);
  EXPECT_EQ(report.device_wait_count, 0U);
  {
    auto accepted = first.submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    ASSERT_TRUE(accepted->wait().ok());
  }
  auto old = first.result(definition.output).value();
  expect_weight(context, old, {0.5f, 2.0f, 6.0f, 0.0f});
  EXPECT_FALSE(g::vulkan::bind(first, definition.left, batch1, 10).ok());
  EXPECT_FALSE(first.reset().ok());
  ASSERT_TRUE(g::vulkan::bind(second, definition.left, batch1, 10).ok());
  ASSERT_TRUE(g::vulkan::bind(second, definition.right, batch0, 20).ok());
  ASSERT_TRUE(second.prepare_replay({}).ok());
  report = second.input_source_report().value();
  EXPECT_EQ(report.source_count, 2U);
  EXPECT_EQ(report.host_wait_count, 2U);
  {
    auto accepted = second.submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    ASSERT_TRUE(accepted->wait().ok());
  }
  expect_weight(context, second.result(definition.output).value(), {1.5f, 4.5f, 1.5f, 3.75f});
  expect_weight(context, old, {0.5f, 2.0f, 6.0f, 0.0f});
  old.reset();
  ASSERT_TRUE(g::vulkan::bind(first, definition.left, batch1, 10).ok());
  ASSERT_TRUE(g::vulkan::bind(first, definition.right, batch1, 20).ok());
  ASSERT_TRUE(first.prepare_replay({}).ok());
  {
    auto accepted = first.submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    ASSERT_TRUE(accepted->wait().ok());
  }
  expect_weight(context, first.result(definition.output).value(), {1.0f, 0.25f, 1.0f, 2.0f});
  batch0 = {};
  batch1 = {};
  // Actual native input arrays also retain the upload. Destroy both original
  // slots, rather than assuming reset has erased every provider-side alias.
  EXPECT_FALSE(weights.upload(0, selection0));
  first = {}; second = {};
  auto reused = weights.upload(0, selection0);
  ASSERT_TRUE(reused) << reused.error().message();
  ASSERT_TRUE(reused->wait().ok());
}

TEST(VulkanWeightBinding, PendingNativeUploadUsesOriginalProgressAndExplicitHostWait) {
  auto context = v::Context::create().value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const std::array catalog{v::WeightSpec{10, weight_description}, v::WeightSpec{20, weight_description}};
  auto weights = v::WeightSlots::create(context, staging, device, catalog, 1024).value();
  const auto a = host_weight({1.0f, -2.0f, 3.5f, 0.25f});
  const auto b = host_weight({-0.5f, 4.0f, 2.5f, -0.25f});
  const std::array selection{v::WeightFragment{10, a.tensor}, v::WeightFragment{20, b.tensor}};
  auto batch = weights.upload(0, selection).value();
  auto definition = weight_graph();
  const auto source = g::vulkan::execution_source(context).value();
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(definition.graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  hyle::sched::Scheduler scheduler{1, m::system_allocator(), 64};
  hyle::sched::BlockingPool pool{m::system_allocator(), 1};
  HoldFenceObservation hold(context);
  struct ReleaseAndDrain final {
    HoldFenceObservation &hold;
    g::Invocation &run;
    ~ReleaseAndDrain() { hold.release(); run.drain().IgnoreError(); }
  } cleanup{hold, run};
  ASSERT_TRUE(g::vulkan::bind(run, definition.left, batch, 10).ok());
  ASSERT_TRUE(g::vulkan::bind(run, definition.right, batch, 20).ok());
  const auto rejected = run.prepare_replay({}, {.host_wait = g::HostWaitPolicy::kReject});
  EXPECT_TRUE(rejected.IsUnimplemented()) << rejected.message();
  ASSERT_TRUE(run.prepare_replay({}).ok());
  ASSERT_TRUE(g::prepare_progress(run, scheduler, pool).ok());
  EXPECT_EQ(hold.polls.load(), 0U);
  EXPECT_EQ(hold.waits.load(), 0U);
  auto accepted = run.submit();
  ASSERT_TRUE(accepted) << accepted.error().message();
  ASSERT_TRUE(eventually([&] { return hold.polls.load(std::memory_order_acquire) != 0; }));
  auto pending = accepted->poll();
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  EXPECT_FALSE(run.result(definition.output));
  EXPECT_EQ(hold.waits.load(), 0U);
  std::atomic<bool> ordinary{false};
  ASSERT_TRUE(scheduler.spawn_detached([&]() noexcept { ordinary.store(true); }).ok());
  EXPECT_TRUE(eventually([&] { return ordinary.load(); }));
  hold.release();
  ASSERT_TRUE(accepted->wait().ok());
  auto result = run.result(definition.output);
  ASSERT_TRUE(result) << result.error().message();
  expect_weight(context, *result, {0.5f, 2.0f, 6.0f, 0.0f});
  result->reset();
  ASSERT_TRUE(run.drain().ok());
}
} // namespace
