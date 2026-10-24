#include "../compute_models/gptq_reference_fixture.h"
#include "../compute_route_vulkan/fixture.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph/artifact_packed_region.h"

#include <algorithm>

namespace {
namespace f = gptq_reference_test;
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
      return hyle::Status::ResourceExhausted("GPTQ test proposal scratch exceeded");
    const auto is_packed = [](const auto &candidate) {
      return candidate.policy == gd::ArtifactPackedPattern::kPolicy;
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
  namespace f = gptq_reference_test;
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

TEST(VulkanGptqReferenceGraph, OriginalQuartetAndTypedProductsSurviveSlotsAndUpdates) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    gptq_reference_test::run_lifetimes(resource, queues, compile, write, read, prepare);
  });
}
TEST(VulkanGptqReferenceGraph, AllGroupIndicesValidateBeforeAnySourceAddress) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    gptq_reference_test::run_runtime_domain(resource, queues, compile, write, read, prepare);
  });
}
TEST(VulkanGptqReferenceGraph, PublishedDecodeRetainsTheOriginalCheckedPath) {
  run_vulkan([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    gptq_reference_test::run_escape(resource, queues, compile, write, read, prepare);
  }, false);
}
