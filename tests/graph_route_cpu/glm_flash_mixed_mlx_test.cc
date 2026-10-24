#include "../compute_models/glm_flash_mixed_mlx_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
namespace f=glm_flash_mixed_mlx_test;
struct Backend final {
  f::c::Context &context;
  f::m::MemoryResourceRef resource;
  std::span<const f::g::QueueBinding> queues;
  hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph) {
    return f::g::cpu::compile(graph,context);
  }
  hyle::Status prepare(f::g::Invocation &invocation) {
    return invocation.prepare_replay(queues);
  }
  hyle::Status write(f::m::BufferView destination,std::span<const std::byte> bytes) {
    return destination.write(bytes);
  }
  hyle::Status read(f::m::ConstBufferView source,std::span<std::byte> bytes) {
    return source.read(bytes);
  }
};
} // namespace

TEST(CpuGlmFlashMixedMlx, RoleBanksKeepPromotedCompleteMoe) {
  auto context=f::take(f::g::cpu::create_context(),"CPU full FFN context");
  auto queue=f::take(context.create_queue(),"CPU full FFN queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  Backend api{context,f::m::system_resource(),queues};
  f::run_cpu(api);
}
