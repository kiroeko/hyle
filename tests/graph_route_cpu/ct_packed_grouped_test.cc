#include "../compute_models/ct_packed_grouped_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
namespace f = ct_packed_grouped_test;
void exercise(unsigned journey) {
  auto context = f::take(f::g::cpu::create_context(), "CPU Context");
  auto queue = f::take(context.create_queue(), "CPU Queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  const auto compile = [&](const f::g::Graph &graph, f::g::CompileOptions options,
                           bool) {
    return f::g::cpu::compile(graph, context, options);
  };
  const auto write = [](f::m::BufferView destination, std::span<const std::byte> bytes) {
    return destination.write(bytes);
  };
  const auto read = [](f::m::ConstBufferView source, std::span<std::byte> bytes) {
    return source.read(bytes);
  };
  const auto prepare = [](f::g::Invocation &slot,
                          std::span<const f::g::QueueBinding> bindings, bool) {
    return slot.prepare_replay(bindings,
        {.mode = f::g::ReplayMode::kCommands, .host_wait = f::g::HostWaitPolicy::kAllow});
  };
  if (journey == 0) f::run(f::m::system_resource(), queues, compile, write, read, prepare);
  else if (journey == 1) f::run_domain(f::m::system_resource(), queues, compile, write, read, prepare);
  else f::run_escape(f::m::system_resource(), queues, compile, write, read, prepare);
}
TEST(CpuCtPackedGrouped, OriginalBanksKeepRoundingStridesSlotsAndResults) { exercise(0); }
TEST(CpuCtPackedGrouped, CompleteIdsValidateBeforeSelectedBankAccess) { exercise(1); }
TEST(CpuCtPackedGrouped, PublishedDecodeRetainsTheOriginalGraph) { exercise(2); }
} // namespace
