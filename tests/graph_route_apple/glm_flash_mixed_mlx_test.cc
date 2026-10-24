#include "../compute_models/glm_flash_mixed_mlx_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

namespace {
namespace f = expert_working_set_test;
namespace a = hyle::compute::apple;
struct Backend final {
  using Slots = a::WeightSlots;
  using Batch = a::WeightBatch;
  a::Context &context;
  a::MemoryResource &memory;
  f::m::MemoryResourceRef resource;
  std::span<const f::g::QueueBinding> queues{};
  f::g::ExecutionSource source;
  hyle::Result<Slots> create_slots(std::span<const f::BankSpec> banks,
                                    std::size_t capacity) {
    std::vector<a::WeightSpec> catalog;
    catalog.reserve(banks.size());
    for (const auto &bank : banks) catalog.push_back({bank.id, bank.description});
    return Slots::create(context, memory, catalog, capacity, {.max_fragments = 128});
  }
  hyle::Result<Batch> upload(Slots &slots, std::size_t slot,
                             std::span<const f::c::WeightSlice> slices) {
    return slots.upload_slices(slot, slices);
  }
  hyle::Status bind(f::g::Invocation &invocation,
      std::span<const f::g::WeightBinding> bindings, const Batch &batch) {
    return f::g::apple::bind(invocation, bindings, batch);
  }
  hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph) {
    const std::array domains{f::g::StorageDomain{resource}};
    const std::array targets{f::g::ExecutionTarget{source, f::g::StorageDomainId{0}}};
    return f::g::compile(graph, {domains, targets});
  }
  hyle::Status prepare(f::g::Invocation &invocation) {
    return invocation.prepare_replay(queues);
  }
  hyle::Status write(f::m::BufferView destination, std::span<const std::byte> bytes) {
    return destination.write(bytes);
  }
  hyle::Status read(f::m::ConstBufferView source_view, std::span<std::byte> bytes) {
    return source_view.read(bytes);
  }
};
template<class Run> void run_apple(Run run) {
  auto context = f::take(a::Context::create(), "Apple context");
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = f::take(a::MemoryResource::create(context), "Apple shared memory");
  Backend api{context, memory, memory.ref(), {},
      f::take(f::g::apple::execution_source(context), "original Apple execution source")};
  run(api);
}
} // namespace

TEST(AppleGlmFlashMixedMlx, OriginalRolesKeepTypedPromotionAndWeightGeneration) {
  run_apple([](auto &api) { glm_flash_mixed_mlx_test::run_gpu(api); });
}
