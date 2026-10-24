#include "../compute_models/e8m0_weight_import_fixture.h"
#include "hyle/graph/cpu.h"

TEST(CpuE8M0WeightImport, ColdCanonicalBankUsesOriginalLinearAndResultLifetime) {
  namespace f=e8m0_weight_import_test;
  auto context=f::take(f::g::cpu::create_context(), "CPU import consumer");
  auto queue=f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::cpu::compile(graph,context,options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}

TEST(CpuE8M0WeightImport, StoredFloat32ScalesKeepDynamicActivationQuantization) {
  namespace f=e8m0_weight_import_test;
  auto context=f::take(f::g::cpu::create_context(), "CPU stored-scale consumer");
  auto queue=f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::cpu::compile(graph,context,options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      },{},true);
}
