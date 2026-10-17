// Private source test: observe the owner through CompletionDependency, never
// concurrently settle the same move-only Completion/OwnedResult wrapper.
#define main hyle_publication_sample_main
#if defined(HYLE_COMPLETION_PUBLICATION_APPLE)
#include "../../samples/package/graph/cpu_apple/main.cc"
#else
#include "../../samples/package/graph/cpu_vulkan/main.cc"
#endif
#undef main

#include <hyle/compute/detail/generated_result.h>
#include <hyle/platform/thread.h>
#include <hyle/sched/sync.h>

#include "../../src/compute/completion_internal.h"

#include <atomic>
#include <chrono>

namespace {
namespace publication_cases {
using hyle::sched::Event;
using hyle::sched::EventResetMode;
constexpr auto watchdog = std::chrono::seconds(10);

Status check(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

struct CpuGate final {
  Event entered{EventResetMode::kManual}, release{EventResetMode::kManual};
  std::atomic<bool> armed{false};
  std::atomic<unsigned> calls{0};
  static void execute(void *opaque, std::size_t count, c::ParallelTaskFn task,
                      void *context) noexcept {
    auto &self = *static_cast<CpuGate *>(opaque);
    self.calls.fetch_add(1, std::memory_order_relaxed);
    if (self.armed.exchange(false, std::memory_order_acq_rel)) {
      self.entered.signal();
      self.release.wait();
    }
    for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  }
  void arm() noexcept {
    entered.reset(); release.reset();
    armed.store(true, std::memory_order_release);
  }
};

struct TailGate final {
  Event entered{EventResetMode::kManual}, release{EventResetMode::kManual};
  Event released{EventResetMode::kManual};
  c::detail::CompletionRetention retention;
  Status terminal;
  TailGate() noexcept {
    retention.own = {
        this,
        [](void *opaque, const Status &status) noexcept {
          auto &self = *static_cast<TailGate *>(opaque);
          self.terminal = status;
          self.entered.signal();
          self.release.wait();
        },
        [](void *opaque) noexcept {
          static_cast<TailGate *>(opaque)->released.signal();
        }};
  }
};

struct Observer final {
  c::detail::CompletionDependency source;
  Event entered{EventResetMode::kManual}, done{EventResetMode::kManual};
  std::atomic<bool> returned{false};
  Status terminal;
  static void run(void *opaque) noexcept {
    auto &self = *static_cast<Observer *>(opaque);
    self.entered.signal();
    self.terminal = self.source.vtable->wait(self.source.state);
    self.returned.store(true, std::memory_order_release);
    self.done.signal();
  }
};

// Construct after the borrowed observer/gate records and invocation. Every
// failure path releases both gates and joins before their storage disappears.
struct Cleanup final {
  CpuGate &cpu;
  TailGate &tail;
  hyle::sched::Scheduler &scheduler;
  hyle::platform::Thread &observer, &closer;
  ~Cleanup() {
    cpu.release.signal(); tail.release.signal();
    if (observer.joinable()) observer.join();
    if (closer.joinable()) closer.join();
    scheduler.drain();
  }
};

template <class CreateInvocation>
Status run(b::Environment environment, CreateInvocation create) {
  CpuGate cpu_gate;
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context(
      {.executor = c::ParallelExecutorRef{&cpu_gate, 2, &CpuGate::execute}}));
  environment.cpu = &cpu;
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(desc));
  HYLE_ASSIGN_OR_RETURN(auto weight, tensor(desc));
  std::array<float, 4096> values, diagonal{};
  values.fill(1.0f);
  for (std::size_t i = 0; i < 64; ++i) diagonal[64 * i + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(weight.buffer.view().write(std::as_bytes(std::span(diagonal))));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, x, w));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kAbs}, product));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  // Exercise the real joint Graph owner without submitting a GPU command.
  const std::array placements{
      g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), b::kCpuTarget, g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment,
      {.node_placements = placements}));

  for (bool close_observer : {false, true}) {
    HYLE_ASSIGN_OR_RETURN(auto invocation, create(program, scheduler, queue));
    const auto prepare = [&]() -> Status {
      HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
      HYLE_RETURN_IF_ERROR(invocation.bind(w, weight.view.as_const()));
      return invocation.prepare_replay();
    };
    HYLE_RETURN_IF_ERROR(prepare());
    { HYLE_ASSIGN_OR_RETURN(auto warm, invocation.run()); }
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load(std::memory_order_relaxed) != 0,
        "CPU MatMul did not reach the real executor gate"));

    TailGate tail;
    std::array<Observer, 4> observers;
    Event sentinel{EventResetMode::kManual};
    Event close_entered{EventResetMode::kManual}, close_done{EventResetMode::kManual};
    std::atomic<bool> close_returned{false};
    Status close_terminal;
    hyle::platform::Thread external, closer;
    Cleanup cleanup{cpu_gate, tail, scheduler, external, closer};
    HYLE_RETURN_IF_ERROR(prepare());
    cpu_gate.arm();
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_RETURN_IF_ERROR(check(cpu_gate.entered.wait_for(watchdog),
        "CPU prefix did not reach the controlled gate"));
    const auto source = c::detail::CompletionAccess::dependency(work.completion());
    HYLE_RETURN_IF_ERROR(check(source.pending(), "joint Graph did not expose its pending owner"));
    HYLE_RETURN_IF_ERROR(c::detail::CompletionAccess::attach_retention(
        c::detail::ResultAccess::completion(work), tail.retention));
    work = {}; // Registers the real abandoned callback, keeping result retention.
    cpu_gate.release.signal();
    HYLE_RETURN_IF_ERROR(check(tail.entered.wait_for(watchdog),
        "abandoned completion did not enter its callback tail"));
    HYLE_RETURN_IF_ERROR(tail.terminal);
    HYLE_ASSIGN_OR_RETURN(auto pending, source.vtable->poll(source.state));
    HYLE_RETURN_IF_ERROR(check(pending == c::CompletionPoll::kPending,
        "Graph published completion before its callback tail retired"));
    HYLE_RETURN_IF_ERROR(check(!invocation.bind(x, input.view.as_const()).ok(),
        "Graph allowed reuse while its old callback tail was active"));

    // Independent private owner observers are safe here: they do not mutate
    // any Completion wrapper, and invocation/driver retain the borrowed state.
    for (auto &observer : observers)
      observer.source = {source.vtable, source.state, nullptr};
    for (std::size_t i = 0; i < 3; ++i)
      HYLE_RETURN_IF_ERROR(scheduler.spawn_detached(&Observer::run, &observers[i]));
    HYLE_ASSIGN_OR_RETURN(external, hyle::platform::Thread::spawn(&Observer::run, &observers[3]));
    for (auto &observer : observers)
      HYLE_RETURN_IF_ERROR(check(observer.entered.wait_for(watchdog), "owner observer did not start"));
    if (close_observer) {
      HYLE_ASSIGN_OR_RETURN(closer, hyle::platform::Thread::spawn([&] {
        close_entered.signal();
        close_terminal = invocation.close();
        close_returned.store(true, std::memory_order_release);
        close_done.signal();
      }));
      HYLE_RETURN_IF_ERROR(check(close_entered.wait_for(watchdog), "close observer did not start"));
    }
    HYLE_RETURN_IF_ERROR(scheduler.spawn_detached(
        [](void *opaque) noexcept { static_cast<Event *>(opaque)->signal(); }, &sentinel));
    HYLE_RETURN_IF_ERROR(check(sentinel.wait_for(watchdog),
        "completion wait stranded the only Scheduler worker"));
    for (auto &observer : observers)
      HYLE_RETURN_IF_ERROR(check(!observer.returned.load(std::memory_order_acquire),
          "owner observer escaped the live callback tail"));
    HYLE_RETURN_IF_ERROR(check(!close_returned.load(std::memory_order_acquire),
        "close escaped the live callback tail"));

    tail.release.signal();
    for (auto &observer : observers) {
      HYLE_RETURN_IF_ERROR(check(observer.done.wait_for(watchdog), "owner observer did not finish"));
      HYLE_RETURN_IF_ERROR(observer.terminal);
    }
    HYLE_RETURN_IF_ERROR(check(tail.released.wait_for(watchdog), "callback release tail did not finish"));
    if (close_observer) {
      HYLE_RETURN_IF_ERROR(check(close_done.wait_for(watchdog), "close did not join the callback tail"));
      HYLE_RETURN_IF_ERROR(close_terminal);
    } else {
      // Reuse immediately after the owner observers return. The old publisher
      // may still be unwinding, but must not signal/write this new episode.
      values.fill(3.0f);
      HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
      HYLE_RETURN_IF_ERROR(prepare());
      {
        HYLE_ASSIGN_OR_RETURN(auto next, invocation.run());
        HYLE_ASSIGN_OR_RETURN(auto actual, next->output(output));
        HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(values))));
        for (auto value : values)
          HYLE_RETURN_IF_ERROR(check(value == 6.0f, "reused episode lost its changed-input result"));
      }
      HYLE_RETURN_IF_ERROR(invocation.close());
    }
  }
  // The watchdogs bound a broken test; elapsed time is never a correctness
  // assertion. The old timer implementation may pass these lifetime checks.
  std::puts("Graph completion publication: callback tails, independent owner observers, one-worker progress, close and immediate reuse passed");
  return Status::Ok();
}
} // namespace publication_cases

Status publication_contracts() {
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
#if defined(HYLE_COMPLETION_PUBLICATION_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto native, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(native));
  const b::Environment environment{.cpu = &cpu, .apple = &native, .native = &memory};
  return publication_cases::run(environment,
      [](const b::Executable &program, hyle::sched::Scheduler &scheduler, c::Queue &queue) {
        return program.create_invocation(scheduler, queue);
      });
#else
  HYLE_ASSIGN_OR_RETURN(auto native, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(native, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(native, v::MemoryKind::kStaging));
  const b::Environment environment{.cpu = &cpu, .vulkan = &native, .device = &device, .staging = &staging};
  hyle::sched::BlockingPool pool(m::system_allocator(), 1);
  return publication_cases::run(environment,
      [&](const b::Executable &program, hyle::sched::Scheduler &scheduler, c::Queue &queue) {
        return program.create_invocation(scheduler, queue, pool);
      });
#endif
}
} // namespace

int main() {
  const auto status = publication_contracts();
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
