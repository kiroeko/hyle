#include <gtest/gtest.h>

#include "tensor_test_support.h"
#include "vulkan_internal.h"

namespace {
using namespace vulkan_tensor_test;
namespace d = v::detail;

TEST(VulkanView, MetadataAliasesSixStorageTypesWithoutDeviceAllocation) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1U << 22);
  auto context = vulkan_test::context(&quota).value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16,
        c::DataType::kInt32, c::DataType::kInt64, c::DataType::kBool}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto desc = c::TensorDesc::strided_bytes(
                          type, std::array<std::int64_t, 2>{3, 5},
                          std::array<std::int64_t, 2>{5 * size + 1, size})
                          .value();
    std::vector<std::uint64_t> raw(15);
    const auto mask =
        size == 8 ? UINT64_MAX : (std::uint64_t{1} << (8 * size)) - 1;
    for (std::size_t i = 0; i < raw.size(); ++i)
      raw[i] = type == c::DataType::kBool ? i % 2
                                          : (0xfffff00012340000ULL + i) & mask;
    const auto original = payload(desc, raw);
    auto source = context.upload(desc, original).value();
    const auto charged = quota.used();
    auto rows = source.view(c::View::slice(0, 1, 2).value()).value();
    auto selected = rows.view(c::View::slice(-1, 1, 2, 2).value()).value();
    auto transposed =
        selected
            .view(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value())
            .value();
    EXPECT_EQ(quota.used(), charged);
    EXPECT_EQ(selected.native_buffer(), source.native_buffer());
    EXPECT_EQ(selected.native_offset(), source.native_offset() + 6 * size + 1);
    EXPECT_EQ(transposed.native_offset(), selected.native_offset());
    EXPECT_EQ(bits(context, transposed),
              (std::vector<std::uint64_t>{raw[6], raw[11], raw[8], raw[13]}));
    auto copied =
        execute(context, c::Operation(c::Copy{}), std::array{transposed});
    ASSERT_TRUE(copied) << copied.error().message();
    EXPECT_EQ(bits(context, *copied), bits(context, transposed));
    auto one = rows.view(c::View::slice(0, 0, 1).value())
                   .value()
                   .view(c::View::slice(1, 2, 1).value())
                   .value();
    const auto shape = c::TensorShape::create(
                           std::array<std::int64_t, 8>{1, 2, 1, 3, 1, 1, 1, 1})
                           .value();
    auto broadcast = one.view(c::View::broadcast_to(shape).value()).value();
    EXPECT_EQ(bits(context, broadcast), std::vector<std::uint64_t>(6, raw[7]));
    EXPECT_FALSE(source.view(
        c::View::reshape(
            c::TensorShape::create(std::array<std::int64_t, 1>{15}).value())
            .value()));
    auto empty = source.view(c::View::slice(0, 3, 0).value()).value();
    EXPECT_EQ(empty.native_offset(), source.native_offset());
    EXPECT_TRUE(bits(context, empty).empty());
    EXPECT_FALSE(source.view(c::View::slice(0, 3, 1).value()));
    EXPECT_EQ(bytes(context, source), original);
    const auto handle = source.native_buffer();
    source = {};
    rows = {};
    selected = {};
    one = {};
    broadcast = {};
    empty = {};
    EXPECT_EQ(transposed.native_buffer(), handle);
    EXPECT_EQ(bits(context, transposed),
              (std::vector<std::uint64_t>{raw[6], raw[11], raw[8], raw[13]}));
    auto dense_source =
        context.upload(dense(type, {3, 5}), payload(dense(type, {3, 5}), raw))
            .value();
    auto reshaped =
        dense_source
            .view(c::View::reshape(
                      c::TensorShape::create(std::array<std::int64_t, 1>{15})
                          .value())
                      .value())
            .value();
    EXPECT_EQ(reshaped.native_buffer(), dense_source.native_buffer());
    EXPECT_EQ(bits(context, reshaped), raw);
  }
}

TEST(VulkanView, PreparedAliasAndFixedRecorderRequireRealSharedRanges) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat16, {17});
  std::vector<double> numbers(17);
  for (int i = 0; i < 17; ++i)
    numbers[i] = i - 8;
  const auto source = make(context, desc, numbers);
  const auto operation = c::View::slice(0, 1, 7, 2).value();
  auto alias = execute(context, c::Operation(operation), std::array{source});
  ASSERT_TRUE(alias) << alias.error().message();
  EXPECT_EQ(alias->native_buffer(), source.native_buffer());
  EXPECT_EQ(alias->native_offset(), source.native_offset() + 2);
  expect(context, *alias, std::array{-7., -5., -3., -1., 1., 3., 5.}, true);
  auto program = v::prepare(context, operation, desc).value();
  auto recorder = v::Recorder::create(context).value();
  const auto separate = context.allocate(alias->description()).value();
  EXPECT_FALSE(
      recorder.append(program, std::array{source}, std::array{separate}).ok());
  ASSERT_TRUE(
      recorder.append(program, std::array{source}, std::array{*alias}).ok());
  auto recording = recorder.finish().value();
  auto work = recording.submit().value();
  ASSERT_TRUE(work.wait().ok());
  EXPECT_EQ(alias->native_buffer(), source.native_buffer());
  auto dense_view = source.view(c::View::slice(0, 1, 7).value()).value();
  auto copy = v::prepare(context, c::Copy{}, dense_view.description()).value();
  auto rejected = v::Recorder::create(context).value();
  EXPECT_TRUE(
      rejected.append(copy, std::array{dense_view}, std::array{dense_view})
          .IsUnimplemented());
}

struct RecordingProbe {
  inline static RecordingProbe *active = nullptr;
  std::shared_ptr<d::Device> device;
  d::DeviceFunctions original;
  int begins = 0, buffers = 0, pools = 0;
  bool fail_begin = false;
  explicit RecordingProbe(const v::Context &context)
      : device(d::Access::runtime(context)->device), original(device->fn) {
    active = this;
    device->fn.BeginCommandBuffer = begin;
    device->fn.CreateBuffer = buffer;
    device->fn.CreateDescriptorPool = pool;
  }
  ~RecordingProbe() {
    device->fn = original;
    active = nullptr;
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  begin(VkCommandBuffer command, const VkCommandBufferBeginInfo *info) {
    ++active->begins;
    if (active->fail_begin) {
      active->fail_begin = false;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return active->original.BeginCommandBuffer(command, info);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  buffer(VkDevice device, const VkBufferCreateInfo *info,
         const VkAllocationCallbacks *callbacks, VkBuffer *out) {
    ++active->buffers;
    return active->original.CreateBuffer(device, info, callbacks, out);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  pool(VkDevice device, const VkDescriptorPoolCreateInfo *info,
       const VkAllocationCallbacks *callbacks, VkDescriptorPool *out) {
    ++active->pools;
    return active->original.CreateDescriptorPool(device, info, callbacks, out);
  }
};

TEST(VulkanView, OriginChangesReuseResourcesAndFailedRecordingCannotSubmit) {
  auto context = vulkan_test::context().value();
  const auto backing_desc = dense(c::DataType::kBool, {300});
  std::vector<std::uint64_t> raw(300), other(300);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = i % 2;
    other[i] = 1 - raw[i];
  }
  const auto source =
      context.upload(backing_desc, payload(backing_desc, raw)).value();
  const auto second =
      context.upload(backing_desc, payload(backing_desc, other)).value();
  auto view = source.view(c::View::slice(0, 1, 129).value()).value();
  auto program = v::prepare(context, c::Copy{}, view.description()).value();
  auto slot = program.create_invocation().value();
  RecordingProbe probe(context);
  for (int step = 0; step < 4; ++step) {
    const auto start = step == 0 ? 0 : step == 3 ? 2 : 1;
    const auto &parent = step == 2 ? second : source;
    view = parent.view(c::View::slice(0, start, 129).value()).value();
    const auto before = probe.begins, buffers = probe.buffers,
               pools = probe.pools;
    ASSERT_TRUE(slot.bind(view).ok());
    EXPECT_EQ(probe.begins - before, step == 1 || step == 3 ? 1 : 0);
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(probe.buffers, buffers);
    EXPECT_EQ(probe.pools, pools);
    EXPECT_EQ(probe.begins - before, step == 1 || step == 3 ? 1 : 0);
    const auto &expected = step == 2 ? other : raw;
    EXPECT_EQ(bits(context, result->get()->tensors()[0]),
              std::vector<std::uint64_t>(expected.begin() + start,
                                         expected.begin() + start + 129));
  }
  view = source.view(c::View::slice(0, 3, 129).value()).value();
  probe.fail_begin = true;
  EXPECT_TRUE(slot.bind(view).IsResourceExhausted());
  EXPECT_FALSE(slot.submit());
  ASSERT_TRUE(slot.bind(view).ok());
  auto work = slot.submit().value();
  auto result = work.wait().value();
  EXPECT_EQ(bits(context, result.get()->tensors()[0]),
            std::vector<std::uint64_t>(raw.begin() + 3, raw.begin() + 132));
}
} // namespace
