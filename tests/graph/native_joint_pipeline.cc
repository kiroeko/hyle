#include <hyle/graph/cpu.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/blocking.h>
#if defined(HYLE_G09_APPLE)
#include <hyle/graph/apple.h>
#else
#include <hyle/graph/vulkan.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
#if defined(HYLE_G09_APPLE)
namespace ng = g::apple;
namespace n = c::apple;
#else
namespace ng = g::vulkan;
namespace n = c::vulkan;
#endif
using hyle::Status;
constexpr g::ExecutionTargetId native_target{0},cpu_target{1};
constexpr g::StorageDomainId host_storage{0},native_storage{1};
// Only cold resource arguments. Plan, mutable slot, results and progress remain
// the original common Graph owners.
struct PipelineResources final {
  const c::Context &cpu;
  const n::Context &native;
  const n::MemoryResource &device;
  const n::MemoryResource *staging = nullptr;
};
struct Tensor { m::Buffer buffer; c::TensorView view; };
hyle::Result<Tensor> tensor(c::TensorDesc desc) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto buffer, m::Buffer::create(m::system_resource(), bytes, 64));
  HYLE_ASSIGN_OR_RETURN(auto view, c::TensorView::create(desc, buffer.view()));
  return Tensor{std::move(buffer), view};
}
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

// Public consumption only: independent real math, mandatory join, two original
// slots, changed inputs, shared readonly native source and retained old results.
// This fixture proves correctness/progress; it makes no device-overlap claim.
Status branches(PipelineResources resources, bool cpu_first, unsigned workers, bool native_input,
    unsigned iterations = 65, unsigned active_slots = 2,
    Status (*observe)(unsigned, bool) = nullptr, bool exp_branch = false,
    bool second_native = false, unsigned wait_pool_capacity = 2) {
  HYLE_RETURN_IF_ERROR(require(active_slots == 1 || active_slots == 2, "invalid observation slot count"));
  constexpr std::size_t elements = 4 * 1024 * 1024;
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{elements}));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(desc));
  std::vector<float> values(elements, 1.25f);
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(values))));
#if defined(HYLE_G09_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto resident, resources.device.allocate(desc));
  HYLE_RETURN_IF_ERROR(resources.device.copy_from_host(resident, source.view.as_const()));
  HYLE_ASSIGN_OR_RETURN(auto resident_read, resources.device.import_readonly(resident,desc));
#else
  HYLE_ASSIGN_OR_RETURN(auto resident, resources.native.upload(desc, std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto resident_read, resources.device.import_readonly(resident));
#endif
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto xi, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto ai, builder.input(desc));
  g::ValueId cv, av;
  const c::Unary cpu_math{exp_branch ? c::UnaryKind::kExp : c::UnaryKind::kAbs};
  if (cpu_first) {
    HYLE_ASSIGN_OR_RETURN(cv, builder.add(cpu_math, xi));
    HYLE_ASSIGN_OR_RETURN(av, builder.add(c::Unary{c::UnaryKind::kNeg}, ai));
  } else {
    HYLE_ASSIGN_OR_RETURN(av, builder.add(c::Unary{c::UnaryKind::kNeg}, ai));
    HYLE_ASSIGN_OR_RETURN(cv, builder.add(cpu_math, xi));
  }
  auto cpu_result = cv;
  if (second_native) {
    HYLE_ASSIGN_OR_RETURN(cpu_result, builder.add(c::Unary{c::UnaryKind::kNeg}, cv));
  }
  HYLE_ASSIGN_OR_RETURN(auto out, builder.add(c::Binary{}, cpu_result, av));
  HYLE_RETURN_IF_ERROR(builder.publish(out));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  std::vector<g::NodePlacement> placements{
      g::NodePlacement{*graph.node_id(cpu_first ? 0 : 1), cpu_target, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(cpu_first ? 1 : 0), native_target, g::PlacementStrength::kRequired}};
  if (second_native)
    placements.push_back({*graph.node_id(2), native_target, g::PlacementStrength::kRequired});
  placements.push_back({*graph.node_id(second_native ? 3 : 2), cpu_target, g::PlacementStrength::kRequired});
  std::printf("G09 begin order=%s workers=%u native_input=%u second_native=%u\n",
      cpu_first ? "cpu-first" : "gpu-first", workers, unsigned(native_input), unsigned(second_native));
  std::fflush(stdout);
  HYLE_ASSIGN_OR_RETURN(auto native_source,ng::execution_source(resources.native));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(resources.cpu));
  std::vector<g::StorageDomain> domains{{m::system_resource()},{resources.device.ref()}};
  if(resources.staging)domains.push_back({resources.staging->ref()});
  const std::array targets{g::ExecutionTarget{native_source,native_storage},g::ExecutionTarget{cpu_source,host_storage}};
  const std::array external{g::ExternalResourceContract{xi,host_storage},
      g::ExternalResourceContract{ai,native_input?native_storage:host_storage},g::ExternalResourceContract{out,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto program,g::compile(graph,{.storage_domains=domains,.targets=targets},
      {.node_placements=placements,.external_resources=external}));
  std::size_t cpu_step=SIZE_MAX,join_step=SIZE_MAX,native_step=SIZE_MAX,second_native_step=SIZE_MAX;
  for(std::size_t i=0;i<program.report.step_count();++i){
    HYLE_ASSIGN_OR_RETURN(auto step,program.report.step(i));
    for(std::size_t index=0;index<step.node_count;++index){
      HYLE_ASSIGN_OR_RETURN(auto node,program.report.step_node(i,index));
      if(node==*graph.node_id(cpu_first?0:1))cpu_step=i;
      if(node==*graph.node_id(cpu_first?1:0))native_step=i;
      if(node==*graph.node_id(second_native?3:2))join_step=i;
      if(second_native && node==*graph.node_id(2))second_native_step=i;
    }
  }
  const auto segment_of=[&](std::size_t step)->hyle::Result<std::size_t>{
    for(std::size_t i=0;i<program.report.segment_count();++i){
      HYLE_ASSIGN_OR_RETURN(auto segment,program.report.segment(i));
      if(step>=segment.step_begin && step-segment.step_begin<segment.step_count)return i;
    }return std::unexpected(Status::Internal("pipeline mathematical step has no original segment"));
  };
  HYLE_ASSIGN_OR_RETURN(auto cpu_segment,segment_of(cpu_step));
  HYLE_ASSIGN_OR_RETURN(auto join_segment,segment_of(join_step));
  HYLE_RETURN_IF_ERROR(require(cpu_segment!=join_segment,
      "independent CPU prefix still inherits the native join wait"));
  if(second_native){
    HYLE_ASSIGN_OR_RETURN(auto first_native_segment,segment_of(native_step));
    HYLE_ASSIGN_OR_RETURN(auto following_native_segment,segment_of(second_native_step));
    HYLE_RETURN_IF_ERROR(require(first_native_segment!=following_native_segment,
        "independent native branch waits for the CPU-produced native successor"));
  }
  hyle::sched::Scheduler scheduler(workers, m::system_allocator(), 64);
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(), wait_pool_capacity);
  std::array<c::Queue, 2> queues;
  std::array<Tensor, 2> x,output;
  std::array<g::Invocation, 2> slots;
  std::array<std::vector<g::QueueBinding>,2> bindings;
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain() { scheduler.drain(); } } drain{scheduler};
  struct CloseSlots { std::array<g::Invocation,2>&slots; ~CloseSlots(){for(auto&slot:slots)slot.close().IgnoreError();} } close_slots{slots};
  for (unsigned i = 0; i < slots.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(queues[i], resources.cpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(x[i], tensor(desc));
    HYLE_ASSIGN_OR_RETURN(output[i],tensor(desc));
    for(std::size_t index=0;index<program.report.lane_count();++index){
      HYLE_ASSIGN_OR_RETURN(auto lane,program.report.lane(index));
      if(lane.queue_required){
        HYLE_RETURN_IF_ERROR(require(lane.target==cpu_target && lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
            "native pipeline requested a fabricated Compute Queue"));bindings[i].push_back({lane.lane,&queues[i]});
      }
    }
    HYLE_ASSIGN_OR_RETURN(slots[i], program.executable.create_invocation({.resources=g::ResourceOptions{}}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(slots[i],scheduler,completion_waiters));
  }
  auto bind = [&](unsigned slot, unsigned step) -> Status {
    for (std::size_t i = 0; i < elements; ++i)
      values[i] = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
    HYLE_RETURN_IF_ERROR(x[slot].buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(xi, x[slot].view.as_const()));
    if (native_input) HYLE_RETURN_IF_ERROR(slots[slot].bind(ai, desc, resident_read));
    else HYLE_RETURN_IF_ERROR(slots[slot].bind(ai, source.view.as_const()));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(out,output[slot].view));
    return slots[slot].prepare_replay(bindings[slot]);
  };
  auto verify = [&](g::GraphOutput &result, unsigned slot, unsigned step) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto view, result.tensor());
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(values))));
    for (std::size_t i = 0; i < elements; ++i) {
      const auto source_value = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
      const auto cpu_value = exp_branch ? std::exp(double(source_value)) : double(std::abs(source_value));
      const auto expected = (second_native ? -cpu_value : cpu_value) - 1.25;
      const auto differs = exp_branch
          ? !std::isfinite(values[i]) || std::abs(double(values[i]) - expected) > 2e-6 + 2e-6 * std::abs(expected)
          : double(values[i]) != expected;
      if (differs)
        return Status::Internal("independent CPU/native branch+join oracle differs");
    }
    return Status::Ok();
  };
  for (unsigned step = 0; step < iterations; ++step) {
    HYLE_RETURN_IF_ERROR(bind(0, step));
    if (active_slots == 2) HYLE_RETURN_IF_ERROR(bind(1, step));
    if (step == 0) { std::printf("G09 prepared %u slots\n", active_slots); std::fflush(stdout); }
    {
      if (observe != nullptr) HYLE_RETURN_IF_ERROR(observe(step, true));
      HYLE_ASSIGN_OR_RETURN(auto first, slots[0].submit(bindings[0]));
      c::Completion second;
      if (active_slots == 2) { HYLE_ASSIGN_OR_RETURN(second, slots[1].submit(bindings[1])); }
      if (step == 0) { std::printf("G09 submitted %u slots\n", active_slots); std::fflush(stdout); }
      HYLE_RETURN_IF_ERROR(first.wait());
      HYLE_ASSIGN_OR_RETURN(auto old,slots[0].result(out));
      g::GraphOutput other;
      if (active_slots == 2) { HYLE_RETURN_IF_ERROR(second.wait());HYLE_ASSIGN_OR_RETURN(other,slots[1].result(out)); }
      if (observe != nullptr) HYLE_RETURN_IF_ERROR(observe(step, false));
      HYLE_RETURN_IF_ERROR(require(!slots[0].bind(xi, x[0].view.as_const()).ok(),
          "retained result did not prevent slot overwrite"));
      HYLE_RETURN_IF_ERROR(verify(old, 0, step));
      if (active_slots == 2) HYLE_RETURN_IF_ERROR(verify(other, 1, step));
    }
  }
  std::printf("G09 %u-step oracle passed; cancelling\n", iterations); std::fflush(stdout);
  // Cancellation may lose to successful completion. Either result must retire
  // before close, and successful completion must still satisfy the oracle.
  HYLE_RETURN_IF_ERROR(bind(0, 91));
  {
    HYLE_ASSIGN_OR_RETURN(auto pending, slots[0].submit(bindings[0]));
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    auto result = pending.wait();
    if (result.ok()) {HYLE_ASSIGN_OR_RETURN(auto output_result,slots[0].result(out));HYLE_RETURN_IF_ERROR(verify(output_result, 0, 91));}
    else HYLE_RETURN_IF_ERROR(require(result.IsCancelled(), "cancellation lost its status"));
  }
  const auto closed = slots[0].close();
  std::puts("G09 cancelled slot closed"); std::fflush(stdout);
  HYLE_RETURN_IF_ERROR(require(closed.ok() || closed.IsCancelled(), "cancelled slot did not close"));
  HYLE_RETURN_IF_ERROR(slots[1].close());
  scheduler.drain();
  std::printf("{\"case\":\"%s\",\"workers\":%u,\"native_input\":%s,\"oracle\":\"PASS\",\"slots\":%u,\"changed_steps\":%u,\"cpu_math\":\"%s\",\"second_native\":%s}\n",
      cpu_first ? "cpu-first" : "gpu-first", workers, native_input ? "true" : "false", active_slots, iterations,
      exp_branch ? "exp" : "abs", second_native ? "true" : "false");
  return Status::Ok();
}

Status contracts() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto native, n::Context::create());
#if defined(HYLE_G09_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto memory, n::MemoryResource::create(native));
  const PipelineResources environment{cpu,native,memory};
#else
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(native, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(native, n::MemoryKind::kStaging));
  const PipelineResources environment{cpu,native,device,&staging};
#endif
  for (unsigned workers : {1u, 2u}) {
    HYLE_RETURN_IF_ERROR(branches(environment, true, workers, false));
    HYLE_RETURN_IF_ERROR(branches(environment, false, workers, false));
  }
  HYLE_RETURN_IF_ERROR(branches(environment, false, 2, true));
  // The CPU branch now enables another native fragment before the final join.
  // Two slots and a single Scheduler worker exercise repeated notification and
  // request handoffs while the root must keep admitting newly ready work.
  HYLE_RETURN_IF_ERROR(branches(environment, true, 1, false, 8, 2, nullptr, false, true));
  HYLE_RETURN_IF_ERROR(branches(environment, false, 1, true, 8, 2, nullptr, false, true));
#if !defined(HYLE_G09_APPLE)
  // A caller may deliberately budget one blocking worker. Slots may queue, but
  // finite graphs without a pool dependency cycle must still drain and reuse.
  HYLE_RETURN_IF_ERROR(branches(environment, false, 1, false, 4, 2, nullptr, false, true, 1));
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = contracts();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
