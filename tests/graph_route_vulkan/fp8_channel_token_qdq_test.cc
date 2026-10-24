#include "../compute_models/fp8_channel_token_qdq_fixture.h"
#include "../compute_route_vulkan/fixture.h"
#include "hyle/graph/vulkan.h"
#include "../compute_route_vulkan/tensor_test_support.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"

namespace {
template<class Run> void run_vulkan(Run run) {
  namespace f = fp8_channel_token_qdq_test;
  namespace v = f::c::vulkan;
  auto context = f::take(vulkan_test::context(), "Vulkan context");
  ::testing::Test::RecordProperty("device", context.environment().device_name);
  auto device = f::take(v::MemoryResource::create(context, v::MemoryKind::kDevice), "Vulkan device memory");
  auto staging = f::take(v::MemoryResource::create(context, v::MemoryKind::kStaging), "Vulkan staging memory");
  constexpr std::size_t capacity = 64U * 1024U;
  auto transfer = f::take(v::TransferSlot::create(context, staging, capacity), "Vulkan transfer slot");
  auto host = f::take(f::m::Buffer::create(f::m::system_resource(), capacity, 64), "transfer host bytes");
  const auto description = [](std::size_t size) {
    return f::c::TensorDesc::dense(f::Type::kUInt8,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(size)});
  };
  const auto write = [&](f::m::BufferView destination,
                         std::span<const std::byte> bytes) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, description(bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto source_bytes, host.view().subview(0, bytes.size()));
    HYLE_RETURN_IF_ERROR(source_bytes.write(bytes));
    HYLE_ASSIGN_OR_RETURN(auto target, f::c::ConstTensorView::create(desc, destination));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context), target, f::m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto source, f::c::ConstTensorView::create(desc, source_bytes));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.upload(native, source));
    return done.wait();
  };
  const auto read = [&](f::m::ConstBufferView source,
                        std::span<std::byte> bytes) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto desc, description(bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto view, f::c::ConstTensorView::create(desc, source));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context), view, f::m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.download(native));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto target_bytes, host.view().subview(0, bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto target, f::c::TensorView::create(desc, target_bytes));
    HYLE_RETURN_IF_ERROR(transfer.read(target));
    return target_bytes.read(bytes);
  };
  const std::array domains{f::g::StorageDomain{device.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::vulkan::execution_source(context), "Vulkan execution source"), f::g::StorageDomainId{0}}};
  run(device.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {.storage_domains = domains, .targets = targets}, std::move(options));
      }, write, read,
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> queues) {
        return invocation.prepare_replay(queues);
      });
}
} // namespace

TEST(VulkanFp8ChannelTokenQdqGraph, RankThreeBfloatReferenceKeepsTypedRoundingAndLifetimes) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    fp8_channel_token_qdq_test::run(resource, queues, compile, write, read, prepare);
  });
}

TEST(VulkanFp8ChannelTokenQdqGraph, OrdinaryBoolCastAndBroadcastSelectKeepRawStorage) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    fp8_channel_token_qdq_test::run_boolean_selection(resource, queues, compile, write, read, prepare);
  });
}

TEST(VulkanFp8ChannelTokenStorage, BfloatFp8BridgeKeepsRneAndRejectsFp8Arithmetic) {
  namespace t = vulkan_tensor_test;
  namespace c = t::c;
  auto context=vulkan_test::context().value();
  const auto from=t::dense(c::DataType::kBFloat16,{8});
  const auto code_desc=t::dense(c::DataType::kFloat8E4M3FN,{8});
  const std::array<std::uint64_t,8> input_bits{0,0x8000,0x3f88,0x3f98,0x43fa,0x7f80,0xffc1,0x8001};
  const auto input=context.upload(from,t::payload(from,input_bits)).value();
  const auto encoded=t::execute(context,c::Operation(c::Cast{c::DataType::kFloat8E4M3FN}),std::array{input});
  ASSERT_TRUE(encoded)<<encoded.error().message();
  EXPECT_EQ(t::bits(context,*encoded),(std::vector<std::uint64_t>{0,0x80,0x38,0x3a,0x7e,0x7e,0xff,0x80}));
  const auto widened=t::execute(context,c::Operation(c::Cast{c::DataType::kBFloat16}),std::array{*encoded});
  ASSERT_TRUE(widened)<<widened.error().message();
  EXPECT_EQ(t::bits(context,*widened),(std::vector<std::uint64_t>{0,0x8000,0x3f80,0x3fa0,0x43e0,0x43e0,0xffc0,0x8000}));
  EXPECT_EQ(t::bits(context,input),std::vector<std::uint64_t>(input_bits.begin(),input_bits.end()));
  EXPECT_FALSE(t::v::prepare(context,c::Binary{c::BinaryKind::kAdd},code_desc,code_desc));
  EXPECT_FALSE(t::v::prepare(context,c::Cast{c::DataType::kBool},code_desc));
  EXPECT_FALSE(t::v::prepare(context,c::Cast{c::DataType::kFloat8E4M3FN},t::dense(c::DataType::kBool,{8})));
}
