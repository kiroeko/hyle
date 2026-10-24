#include "../compute_models/paged_e4_kv_fixture.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=paged_e4_kv_test;
  auto context=f::take(f::c::Context::create(f::c::cpu::device(),
      f::c::ggml_cpu::provider({.native_threads=2})), "CPU paged context");
  auto queue=f::take(context.create_queue(), "CPU paged queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const std::array domains{f::g::StorageDomain{f::m::system_resource()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::ExecutionSource::acquire(context), "CPU source"),f::g::StorageDomainId{0}}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::batch::compile(graph,{domains,targets},options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);});
  f::check(context.drain(), "CPU context retirement");
}
}
TEST(CpuPagedE4Kv, EncoderBytesRespectFixedScale) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    paged_e4_kv_test::encoder(resource,queues,compile,write,read);
  });
}
TEST(CpuPagedE4Kv, EncoderFailureRetiresAndFreshInvocationRecovers) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    paged_e4_kv_test::encoder(resource,queues,compile,write,read,true);
  });
}
TEST(CpuPagedE4Kv, OrdinaryPagesKeepSeparateScalesAndOldResults) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    paged_e4_kv_test::paged(resource,queues,compile,write,read);
  });
}
TEST(CpuPagedE4Kv, FixedQ1GrowsHistoryThroughOriginalReader) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read) {
    paged_e4_kv_test::paged(resource,queues,compile,write,read,true);
  });
}
