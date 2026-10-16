#include <hyle/graph/cpu.h>
#include <hyle/sched/scheduler.h>
#if defined(HYLE_G09_APPLE)
#include <hyle/graph/cpu_apple.h>
#else
#include <hyle/graph/cpu_vulkan.h>
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
namespace b = g::cpu_apple;
namespace n = c::apple;
constexpr auto native_target = b::kAppleTarget;
#else
namespace b = g::cpu_vulkan;
namespace n = c::vulkan;
constexpr auto native_target = b::kVulkanTarget;
#endif
using hyle::Status;
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
Status branches(b::Environment environment, bool cpu_first, unsigned workers, bool native_input,
    unsigned iterations = 65, unsigned active_slots = 2,
    Status (*observe)(unsigned, bool) = nullptr, bool exp_branch = false) {
  HYLE_RETURN_IF_ERROR(require(active_slots == 1 || active_slots == 2, "invalid observation slot count"));
  constexpr std::size_t elements = 4 * 1024 * 1024;
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{elements}));
  HYLE_ASSIGN_OR_RETURN(auto source, tensor(desc));
  std::vector<float> values(elements, 1.25f);
  HYLE_RETURN_IF_ERROR(source.buffer.view().write(std::as_bytes(std::span(values))));
#if defined(HYLE_G09_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto resident, environment.native->allocate(desc));
  HYLE_RETURN_IF_ERROR(environment.native->copy_from_host(resident, source.view.as_const()));
#else
  HYLE_ASSIGN_OR_RETURN(auto resident, environment.vulkan->upload(desc, std::as_bytes(std::span(values))));
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
  HYLE_ASSIGN_OR_RETURN(auto out, builder.add(c::Binary{}, cv, av));
  HYLE_RETURN_IF_ERROR(builder.publish(out));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placements{
      g::NodePlacement{*graph.node_id(cpu_first ? 0 : 1), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(cpu_first ? 1 : 0), native_target, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), b::kCpuTarget, g::PlacementStrength::kRequired}};
  std::printf("G09 begin order=%s workers=%u native_input=%u\n", cpu_first ? "cpu-first" : "gpu-first", workers, unsigned(native_input));
  std::fflush(stdout);
  b::CompileReport report;
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment,
      {.node_placements = placements}, &report));
  HYLE_RETURN_IF_ERROR(require(report.fragments.size() == 3,
      "independent CPU prefix still inherits the native join wait"));
  hyle::sched::Scheduler scheduler(workers, m::system_allocator(), 64);
  std::array<c::Queue, 2> queues;
  std::array<Tensor, 2> x;
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain() { scheduler.drain(); } } drain{scheduler};
  std::array<b::Invocation, 2> slots;
  for (unsigned i = 0; i < slots.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(queues[i], environment.cpu->create_queue());
    HYLE_ASSIGN_OR_RETURN(x[i], tensor(desc));
    HYLE_ASSIGN_OR_RETURN(slots[i], program.create_invocation(scheduler, queues[i]));
  }
  auto bind = [&](unsigned slot, unsigned step) -> Status {
    for (std::size_t i = 0; i < elements; ++i)
      values[i] = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
    HYLE_RETURN_IF_ERROR(x[slot].buffer.view().write(std::as_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(slots[slot].bind(xi, x[slot].view.as_const()));
    if (native_input) HYLE_RETURN_IF_ERROR(slots[slot].bind(ai, resident));
    else HYLE_RETURN_IF_ERROR(slots[slot].bind(ai, source.view.as_const()));
    return slots[slot].prepare_replay();
  };
  auto verify = [&](const b::Results &result, unsigned slot, unsigned step) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto view, result.output(out));
    HYLE_RETURN_IF_ERROR(view.storage().read(std::as_writable_bytes(std::span(values))));
    for (std::size_t i = 0; i < elements; ++i) {
      const auto source_value = float(int(i % 101) - 50) * 0.125f + float(step) * 0.0625f + float(slot) * 0.25f;
      const auto expected = exp_branch ? std::exp(double(source_value)) - 1.25
                                       : double(std::abs(source_value) - 1.25f);
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
      HYLE_ASSIGN_OR_RETURN(auto first, slots[0].submit());
      c::OwnedResult<b::Results> second;
      if (active_slots == 2) { HYLE_ASSIGN_OR_RETURN(second, slots[1].submit()); }
      if (step == 0) { std::printf("G09 submitted %u slots\n", active_slots); std::fflush(stdout); }
      HYLE_ASSIGN_OR_RETURN(auto old, first.wait());
      c::ResultLease<b::Results> other;
      if (active_slots == 2) { HYLE_ASSIGN_OR_RETURN(other, second.wait()); }
      if (observe != nullptr) HYLE_RETURN_IF_ERROR(observe(step, false));
      HYLE_RETURN_IF_ERROR(require(!slots[0].bind(xi, x[0].view.as_const()).ok(),
          "retained result did not prevent slot overwrite"));
      HYLE_RETURN_IF_ERROR(verify(*old, 0, step));
      if (active_slots == 2) HYLE_RETURN_IF_ERROR(verify(*other, 1, step));
    }
  }
  std::printf("G09 %u-step oracle passed; cancelling\n", iterations); std::fflush(stdout);
  // Cancellation may lose to successful completion. Either result must retire
  // before close, and successful completion must still satisfy the oracle.
  HYLE_RETURN_IF_ERROR(bind(0, 91));
  {
    HYLE_ASSIGN_OR_RETURN(auto pending, slots[0].submit());
    HYLE_RETURN_IF_ERROR(pending.request_cancel());
    auto result = pending.wait();
    if (result) HYLE_RETURN_IF_ERROR(verify(**result, 0, 91));
    else HYLE_RETURN_IF_ERROR(require(result.error().IsCancelled(), "cancellation lost its status"));
  }
  const auto closed = slots[0].close();
  std::puts("G09 cancelled slot closed"); std::fflush(stdout);
  HYLE_RETURN_IF_ERROR(require(closed.ok() || closed.IsCancelled(), "cancelled slot did not close"));
  HYLE_RETURN_IF_ERROR(slots[1].close());
  scheduler.drain();
  std::printf("{\"case\":\"%s\",\"workers\":%u,\"native_input\":%s,\"oracle\":\"PASS\",\"slots\":%u,\"changed_steps\":%u,\"cpu_math\":\"%s\"}\n",
      cpu_first ? "cpu-first" : "gpu-first", workers, native_input ? "true" : "false", active_slots, iterations,
      exp_branch ? "exp" : "abs");
  return Status::Ok();
}

Status contracts() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto native, n::Context::create());
#if defined(HYLE_G09_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto memory, n::MemoryResource::create(native));
  const b::Environment environment{.cpu = &cpu, .apple = &native, .native = &memory};
#else
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(native, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(native, n::MemoryKind::kStaging));
  const b::Environment environment{.cpu = &cpu, .vulkan = &native, .device = &device, .staging = &staging};
#endif
  for (unsigned workers : {1u, 2u}) {
    HYLE_RETURN_IF_ERROR(branches(environment, true, workers, false));
    HYLE_RETURN_IF_ERROR(branches(environment, false, workers, false));
  }
  return branches(environment, false, 2, true);
}
} // namespace

int main() {
  const auto status = contracts();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
