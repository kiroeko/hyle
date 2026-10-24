#include "../compute_models/expert_working_set_fixture.h"
#include "../compute_route_vulkan/fixture.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"

namespace {
namespace f = expert_working_set_test;
namespace v = f::c::vulkan;
struct Backend final {
  using Batch = v::WeightBatch;
  const v::Context &context;
  const v::MemoryResource &device, &staging;
  v::TransferSlot &transfer;
  f::m::Buffer &host;
  f::m::MemoryResourceRef resource;
  std::span<const f::g::QueueBinding> queues{};
  hyle::Result<v::WeightSlots> create_slots(std::span<const f::BankSpec> banks,std::size_t capacity) {
    std::vector<v::WeightSpec> catalog;
    for (const auto &bank:banks) catalog.push_back({bank.id,bank.description});
    return v::WeightSlots::create(context,staging,device,catalog,capacity,{.max_fragments=64});
  }
  hyle::Result<Batch> upload(v::WeightSlots &slots,std::size_t slot,
      std::span<const f::c::WeightSlice> slices) { return slots.upload_slices(slot,slices); }
  hyle::Status bind(f::g::Invocation &invocation,std::span<const f::g::WeightBinding> banks,
      const Batch &batch) { return f::g::vulkan::bind(invocation,banks,batch); }
  hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph) {
    const std::array domains{f::g::StorageDomain{resource}};
    HYLE_ASSIGN_OR_RETURN(auto source,f::g::vulkan::execution_source(context));
    const std::array targets{f::g::ExecutionTarget{std::move(source),f::g::StorageDomainId{0}}};
    return f::g::compile(graph,{.storage_domains=domains,.targets=targets});
  }
  hyle::Status prepare(f::g::Invocation &invocation) { return invocation.prepare_replay(queues); }
  static hyle::Result<f::c::TensorDesc> byte_desc(std::size_t bytes) {
    return f::c::TensorDesc::dense(f::Type::kUInt8,
        std::array<std::int64_t,1>{static_cast<std::int64_t>(bytes)});
  }
  hyle::Status write(f::m::BufferView destination,std::span<const std::byte> bytes) {
    // Only small dynamic A and router IDs use this helper; packed original
    // weights/scales/map take the real WeightSlots sliced native transfer.
    HYLE_ASSIGN_OR_RETURN(auto desc,byte_desc(bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto staging_bytes,host.view().subview(0,bytes.size()));
    HYLE_RETURN_IF_ERROR(staging_bytes.write(bytes));
    HYLE_ASSIGN_OR_RETURN(auto target,f::c::ConstTensorView::create(desc,destination));
    HYLE_ASSIGN_OR_RETURN(auto native,v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context),target,f::m::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto source,f::c::ConstTensorView::create(desc,staging_bytes));
    HYLE_ASSIGN_OR_RETURN(auto done,transfer.upload(native,source));
    return done.wait();
  }
  hyle::Status read(f::m::ConstBufferView source,std::span<std::byte> bytes) {
    HYLE_ASSIGN_OR_RETURN(auto desc,byte_desc(bytes.size()));
    HYLE_ASSIGN_OR_RETURN(auto view,f::c::ConstTensorView::create(desc,source));
    HYLE_ASSIGN_OR_RETURN(auto native,v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context),view,f::m::system_allocator()));
    // The ephemeral download releases the result owner after each read; a
    // persistent TransferSlot download would itself be another retained view.
    return context.download(native,bytes);
  }
};
template<class Run> void run_vulkan(Run run) {
  auto context=f::take(vulkan_test::context(),"Vulkan context");
  ::testing::Test::RecordProperty("device",context.environment().device_name);
  auto device=f::take(v::MemoryResource::create(context,v::MemoryKind::kDevice),"Vulkan device");
  auto staging=f::take(v::MemoryResource::create(context,v::MemoryKind::kStaging),"Vulkan staging");
  constexpr std::size_t capacity=64U*1024U;
  auto transfer=f::take(v::TransferSlot::create(context,staging,capacity),"dynamic input transfer");
  auto host=f::take(f::m::Buffer::create(f::m::system_resource(),capacity,64),"dynamic host input");
  Backend api{context,device,staging,transfer,host,device.ref(),{}};
  run(api);
}
} // namespace
TEST(VulkanExpertWorkingSet, RealRoleGeometryLoadsOnlyFourOf512ExpertsAcrossGenerations) {
  run_vulkan([](auto &api) { f::run_lifetimes(api,f::Family::kNvfp4); });
}
TEST(VulkanExpertWorkingSet, OddCtZerosAreExplicitlyPackedBeforeSelectedBankUpload) {
  run_vulkan([](auto &api) { f::run_lifetimes(api,f::Family::kCtOdd); });
}
TEST(VulkanExpertWorkingSet, MissingIdsPartialSourcesAndRetainedGenerationsCannotPublish) {
  run_vulkan([](auto &api) { f::run_failures(api); });
}
