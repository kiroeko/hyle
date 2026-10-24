#include "../compute_models/static_fp8_reference_fixture.h"
#include "../compute_route_vulkan/fixture.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph/static_fp8_region.h"

#include <algorithm>

namespace {
namespace f = static_fp8_reference_test;
namespace gd = hyle::graph::detail;

// Select only among the real native source's cold proposals. Whole-program
// prepare/submit/resources stay original; this is no product mode or provider.
struct Selection final {
  gd::ExecutionSourceState *state;
  const gd::ExecutionSourceVTable *original;
  gd::ExecutionSourceVTable table;
  const f::g::RegionProviderVTable *regions;
  f::g::RegionProviderVTable selected;
  bool packed;
  static inline Selection *current = nullptr;
  Selection(f::g::ExecutionSource &source, bool use_packed)
      : state(gd::ExecutionSourceAccess::state(source)), original(state->table),
        table(*original), regions(original->region_provider(*state)),
        selected(*regions), packed(use_packed) {
    current = this;
    selected.propose = filter;
    table.region_provider = [](const gd::ExecutionSourceState &) noexcept
        -> const f::g::RegionProviderVTable * {
      return &current->selected;
    };
    state->table = &table;
  }
  ~Selection() { state->table = original; current = nullptr; }
  static hyle::Status filter(void *owner, const f::g::RegionDiscoveryRequest &request,
      f::m::AllocatorRef allocator, f::g::RegionProposalWriter &writer) noexcept {
    std::array<f::g::RegionProposal, 128> proposals;
    std::array<f::g::NodeId, 1024> nodes;
    f::g::RegionProposalWriter candidates(proposals, nodes);
    HYLE_RETURN_IF_ERROR(current->regions->propose(owner, request, allocator, candidates));
    if (candidates.proposal_limit_reached() || candidates.node_limit_reached())
      return hyle::Status::ResourceExhausted("static FP8 test proposal scratch exceeded");
    const auto is_packed = [](const auto &candidate) {
      return candidate.policy == gd::StaticFloat8Pattern::kPolicy;
    };
    const bool has_packed = std::ranges::any_of(candidates.proposals(), is_packed);
    for (const auto &candidate : candidates.proposals()) {
      if ((!current->packed && is_packed(candidate)) ||
          (current->packed && has_packed && !is_packed(candidate))) continue;
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add(candidate));
      if (!more) break;
    }
    return hyle::Status::Ok();
  }
};

template<class Run> void run_vulkan(Run run, bool packed = true) {
  namespace f = static_fp8_reference_test;
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
  auto source = f::take(f::g::vulkan::execution_source(context), "Vulkan execution source");
  Selection selection(source, packed);
  const std::array domains{f::g::StorageDomain{device.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      source, f::g::StorageDomainId{0}}};
  run(device.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {.storage_domains = domains, .targets = targets}, std::move(options));
      }, write, read,
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> queues) {
        return invocation.prepare_replay(queues);
      });
}
} // namespace

TEST(VulkanStaticFp8ReferenceGraph, StoredScalesAndTypedOutputsSurviveSlotsAndUpdates) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    static_fp8_reference_test::run_lifetimes(resource, queues, compile, write, read, prepare);
  });
}
TEST(VulkanStaticFp8ReferenceGraph, IeeeScaleAndCodeBoundariesRetainTheDeclaredTree) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    static_fp8_reference_test::run_runtime_domain(resource, queues, compile, write, read, prepare);
  });
}
TEST(VulkanStaticFp8ReferenceGraph, PublishedCodesAndControlEscapesUseTheOriginalGraph) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    static_fp8_reference_test::run_escape(resource, queues, compile, write, read, prepare);
  }, false);
}

TEST(VulkanStaticFp8ReferenceGraph, OddStorageCastWordsRespectExplicitPaddingOwnership) {
  namespace c = hyle::compute;
  namespace v = c::vulkan;
  auto context = vulkan_test::context().value();
  const auto alignment = (std::max)(VkDeviceSize{4},
      v::detail::Access::runtime(context)->device->properties.limits.minStorageBufferOffsetAlignment);
  const std::array<std::uint32_t, 7> source_bits{
      0x00000000u, 0x80000000u, 0xffc00123u, 0x43fa0000u,
      0xc3fa0000u, 0x7f800000u, 0xff800000u}; // ±0, -NaN, ±500, ±Inf.
  const std::array<std::uint8_t, 7> encoded{0x00, 0x80, 0xff, 0x7e, 0xfe, 0x7e, 0xfe};
  const std::array<std::uint32_t, 7> decoded{
      0x00000000u, 0x80000000u, 0xffc00000u, 0x43e00000u,
      0xc3e00000u, 0x43e00000u, 0xc3e00000u};
  for (const std::int64_t count : {5, 7}) {
    SCOPED_TRACE(count);
    const auto from = c::TensorDesc::dense(c::DataType::kFloat32, std::array{count}).value();
    const auto to = c::TensorDesc::dense(c::DataType::kFloat8E4M3FN, std::array{count}).value();
    const auto input = context.upload(from,
        std::as_bytes(std::span(source_bits).first(static_cast<std::size_t>(count)))).value();
    constexpr VkDeviceSize capacity = 8;
    std::vector<std::byte> guarded(static_cast<std::size_t>(alignment + capacity + 8), std::byte{0x5a});
    const auto backing_desc = c::TensorDesc::dense(c::DataType::kFloat8E4M3FN,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(guarded.size())}).value();
    const auto backing = context.upload(backing_desc, guarded).value();
    // This call explicitly grants an independent padded eight-byte slice.
    // Merely owning the enclosing VkBuffer never grants that tail permission.
    auto borrowed = context.borrow_ready(to, backing.native_buffer(), alignment,
        capacity, backing.native_owner());
    ASSERT_TRUE(borrowed) << borrowed.error().message();
    EXPECT_FALSE(context.borrow_ready(to, backing.native_buffer(), alignment,
        capacity - 1, backing.native_owner()));
    const auto program = v::prepare(context, c::Cast{c::DataType::kFloat8E4M3FN}, from).value();
    const auto clipped = backing.view(c::View::slice(0, static_cast<std::int64_t>(alignment), count).value()).value();
    auto invalid_recorder = v::Recorder::create(context).value();
    EXPECT_FALSE(invalid_recorder.append(program, std::array{input}, std::array{clipped}).ok());
    auto recorder = v::Recorder::create(context).value();
    ASSERT_TRUE(recorder.append(program, std::array{input}, std::array{*borrowed}).ok());
    auto recorded = recorder.finish().value();
    auto submitted = recorded.submit().value();
    ASSERT_TRUE(submitted.wait().ok());
    std::vector<std::byte> observed(guarded.size());
    ASSERT_TRUE(context.download(backing, observed).ok());
    for (std::size_t i = 0; i < observed.size(); ++i) {
      if (i < alignment || i >= alignment + capacity)
        EXPECT_EQ(observed[i], guarded[i]) << i;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
      EXPECT_EQ(std::to_integer<std::uint8_t>(observed[static_cast<std::size_t>(alignment) + i]), encoded[i]) << i;
    const auto reverse = v::prepare(context, c::Cast{c::DataType::kFloat32}, to).value();
    auto lane = reverse.create_invocation().value();
    ASSERT_TRUE(lane.bind(std::array{*borrowed}).ok());
    auto work = lane.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<std::uint32_t> widened(static_cast<std::size_t>(count));
    ASSERT_TRUE(context.download(result->get()->tensors()[0], std::as_writable_bytes(std::span(widened))).ok());
    EXPECT_TRUE(std::ranges::equal(widened, std::span(decoded).first(widened.size())));
    std::vector<std::uint32_t> original(static_cast<std::size_t>(count));
    ASSERT_TRUE(context.download(input, std::as_writable_bytes(std::span(original))).ok());
    EXPECT_TRUE(std::ranges::equal(original, std::span(source_bits).first(original.size())));
  }
}
