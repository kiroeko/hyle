#include "../compute_models/glm_flash_mixed_mlx_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace f = expert_working_set_test;
namespace cu = f::c::cuda;
struct Backend final {
  using Batch = cu::WeightBatch;
  using Slots = cu::WeightSlots;
  f::c::Context &context;
  cu::MemoryResource &host, &device;
  f::c::Queue &transfer;
  f::m::MemoryResourceRef resource;
  std::span<const f::g::QueueBinding> queues;
  hyle::Result<Slots> create_slots(std::span<const f::BankSpec> banks,
                                  std::size_t capacity) {
    std::vector<cu::WeightSpec> catalog;
    for (const auto &bank : banks) catalog.push_back({bank.id,bank.description});
    return Slots::create(context,host,device,catalog,capacity,{.max_fragments=128});
  }
  hyle::Result<Batch> upload(Slots &slots,std::size_t slot,
                             std::span<const f::c::WeightSlice> slices) {
    return slots.upload_slices(slot,slices,transfer);
  }
  hyle::Status bind(f::g::Invocation &invocation,
      std::span<const f::g::WeightBinding> bindings,const Batch &batch) {
    return f::g::cuda::bind(invocation,bindings,batch);
  }
  hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph) {
    return f::g::cuda::compile(graph,context,resource);
  }
  hyle::Status prepare(f::g::Invocation &invocation) {
    auto status=f::g::cuda::prepare_replay(invocation,*queues[0].queue);
    if(status.ok())
      ::testing::Test::RecordProperty("replay_route",static_cast<int>(invocation.replay_route()));
    return status;
  }
  hyle::Status write(f::m::BufferView destination,std::span<const std::byte> bytes) {
    return destination.write(bytes);
  }
  hyle::Status read(f::m::ConstBufferView source,std::span<std::byte> bytes) {
    return source.read(bytes);
  }
};
template<class Run> void run_cuda(Run run) {
  auto context=f::take(f::g::cuda::create_context(),"CUDA context");
  auto host=f::take(cu::MemoryResource::create(context,cu::MemoryKind::kPinnedHost),"original pinned staging resource");
  auto device=f::take(cu::MemoryResource::create(context,cu::MemoryKind::kDevice),"original device resource");
  auto compute=f::take(context.create_queue(),"original compute queue");
  auto transfer=f::take(context.create_queue(),"original transfer queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&compute}};
  Backend api{context,host,device,transfer,device.ref(),queues};
  run(api);
}
} // namespace

TEST(CudaGlmFlashMixedMlx, RoleBanksKeepPromotedCompleteMoe) {
  run_cuda([](auto &api) { glm_flash_mixed_mlx_test::run_gpu(api); });
}

TEST(CudaWeightBankAlignment, WholeAndSlicedBanksKeepNativePortAlignment) {
  run_cuda([](auto &api) {
    const auto first_desc = f::dense(f::Type::kFloat32, {3});
    const auto second_desc = f::dense(f::Type::kFloat32, {1});
    const std::array<std::uint32_t, 3> first_bits{0x3f800000u, 0x80000000u, 0x7fc12345u};
    const std::array<std::uint32_t, 1> second_bits{0xbf000000u};
    auto first = f::take(f::m::Buffer::create(f::m::system_resource(), sizeof(first_bits), 64),
                         "first original host bank");
    auto second = f::take(f::m::Buffer::create(f::m::system_resource(), sizeof(second_bits), 64),
                          "second original host bank");
    f::check(first.view().write(std::as_bytes(std::span(first_bits))), "fill first bank");
    f::check(second.view().write(std::as_bytes(std::span(second_bits))), "fill second bank");
    const auto first_source = f::take(f::c::ConstTensorView::create(first_desc, first.view()),
                                      "first source description");
    const auto second_source = f::take(f::c::ConstTensorView::create(second_desc, second.view()),
                                       "second source description");
    const std::array catalog{cu::WeightSpec{7, first_desc}, cu::WeightSpec{8, second_desc}};
    const std::array whole{cu::WeightFragment{7, first_source}, cu::WeightFragment{8, second_source}};
    const std::array sliced{f::c::WeightSlice{7, first_source, 0},
                            f::c::WeightSlice{8, second_source, 0}};
    for (const bool slices : {false, true}) {
      SCOPED_TRACE(slices ? "sliced" : "whole");
      // Only bank starts are aligned: 12 payload bytes, 244 padding, 4 payload.
      auto slots = f::take(cu::WeightSlots::create(api.context, api.host, api.device, catalog, 260),
                           "exact aligned bank capacity");
      auto batch = f::take(slices ? slots.upload_slices(0, sliced, api.transfer)
                                 : slots.upload(0, whole, api.transfer), "aligned original upload");
      f::check(batch.wait(), "retire aligned upload");
      const auto first_binding = f::take(batch.binding(7), "first actual device binding");
      const auto second_binding = f::take(batch.binding(8), "second actual device binding");
      EXPECT_EQ(first_binding.storage().offset(), 0u);
      EXPECT_EQ(second_binding.storage().offset(), 256u);
      const auto first_pointer = f::take(cu::pointer<std::byte>(first_binding.storage()), "first bank address");
      const auto second_pointer = f::take(cu::pointer<std::byte>(second_binding.storage()), "second bank address");
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first_pointer) % 256, 0u);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second_pointer) % 256, 0u);
      std::array<std::uint32_t, 3> first_actual{};
      std::array<std::uint32_t, 1> second_actual{};
      f::check(api.read(first_binding.storage(), std::as_writable_bytes(std::span(first_actual))),
               "read first bank without conversion");
      f::check(api.read(second_binding.storage(), std::as_writable_bytes(std::span(second_actual))),
               "read second bank without conversion");
      EXPECT_EQ(first_actual, first_bits);
      EXPECT_EQ(second_actual, second_bits);
      const auto report = f::take(batch.report(), "aligned upload report");
      EXPECT_EQ(report.logical_bytes, 16u);
      EXPECT_EQ(report.staged_bytes, 16u);
      EXPECT_EQ(report.submitted_bytes, 260u);
      EXPECT_EQ(report.completed_bytes, 260u);
      EXPECT_FALSE(report.copied_bytes_unknown);

      auto short_slots = f::take(cu::WeightSlots::create(api.context, api.host, api.device, catalog, 259),
                                 "one-byte-short aligned capacity");
      auto rejected = slices ? short_slots.upload_slices(0, sliced, api.transfer)
                             : short_slots.upload(0, whole, api.transfer);
      ASSERT_FALSE(rejected);
      const auto rejected_report = f::take(short_slots.report(0), "pre-copy capacity rejection");
      EXPECT_EQ(rejected_report.staged_bytes, 0u);
      EXPECT_EQ(rejected_report.submitted_bytes, 0u);
      EXPECT_EQ(rejected_report.completed_bytes, 0u);
      f::check(first.view().read(std::as_writable_bytes(std::span(first_actual))), "unchanged first source");
      f::check(second.view().read(std::as_writable_bytes(std::span(second_actual))), "unchanged second source");
      EXPECT_EQ(first_actual, first_bits);
      EXPECT_EQ(second_actual, second_bits);
    }
  });
}
