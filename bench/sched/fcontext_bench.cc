// Same-contract context-switch evidence for the Boost.Context 1.92 upgrade.
//
// All rows allocate the same guarded HYLE FiberStack, build a context with the
// exact same vendored make_fcontext assembly, and execute the same number of
// caller -> context -> caller round trips.  The direct row is the primitive
// floor; the Fiber row measures only HYLE's consumer-facing bookkeeping above
// that floor.  This is intentionally a validation-only use of sched's internal
// fcontext shim, not a second public fiber API.

#include <nanobench.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "fcontext_bench_seam.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/fiber.h"

namespace {
namespace nb = ankerl::nanobench;

using hyle::mem::FiberStack;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;

FiberStack allocate_stack() {
  auto stack = FiberStack::allocate(kDefaultStackSize);
  if (!stack) std::abort();
  return *std::move(stack);
}

void fiber_yield_body(void* argument) {
  const int rounds = *static_cast<const int*>(argument);
  for (int i = 0; i < rounds; ++i) Fiber::yield();
}

void run_hyle_round_trips(int rounds) {
  Fiber fiber(allocate_stack(), &fiber_yield_body, &rounds);
  for (int i = 0; i <= rounds; ++i) fiber.resume();
  nb::doNotOptimizeAway(fiber.stack_size());
}

struct HandleYieldArguments {
  HyleBenchFcontextSeamContext* context = nullptr;
  int rounds = 0;
};

void handle_yield_body(void* argument) {
  auto* state = static_cast<HandleYieldArguments*>(argument);
  for (int i = 0; i < state->rounds; ++i)
    hyle_bench_fcontext_seam_yield_handle(state->context);
}

void current_yield_body(void* argument) {
  const int rounds = *static_cast<const int*>(argument);
  for (int i = 0; i < rounds; ++i) hyle_bench_fcontext_seam_yield_current();
}

void run_handle_seam_round_trips(int rounds) {
  alignas(kHyleBenchFcontextSeamStorageAlignment)
      std::byte storage[kHyleBenchFcontextSeamStorageSize];
  HandleYieldArguments arguments{.rounds = rounds};
  HyleBenchFcontextSeamContext* context =
      hyle_bench_fcontext_seam_create(storage, sizeof(storage),
                                      &handle_yield_body, &arguments, false);
  arguments.context = context;
  for (int i = 0; i <= rounds; ++i) hyle_bench_fcontext_seam_resume(context);
  hyle_bench_fcontext_seam_destroy(context);
}

void run_current_seam_round_trips(int rounds) {
  alignas(kHyleBenchFcontextSeamStorageAlignment)
      std::byte storage[kHyleBenchFcontextSeamStorageSize];
  HyleBenchFcontextSeamContext* context =
      hyle_bench_fcontext_seam_create(storage, sizeof(storage),
                                      &current_yield_body, &rounds, true);
  for (int i = 0; i <= rounds; ++i) hyle_bench_fcontext_seam_resume(context);
  hyle_bench_fcontext_seam_destroy(context);
}

void run_raw_round_trips(int rounds) {
  hyle_bench_raw_fcontext_round_trips(rounds);
}

}  // namespace

int main() {
  std::fprintf(stderr,
               "# HYLE_SCHED_FCONTEXT_BENCH peer=boost.context-1.92.0 "
               "source=third_party/fcontext contract=resume-yield-roundtrip\n");

  constexpr int kRoundTrips = 1000;
  nb::Bench bench;
  bench.title("same vendored fcontext resume/yield round trip")
      .relative(true)
      .batch(kRoundTrips)
      .minEpochIterations(2000);
  bench.run("raw Boost.Context 1.92 jump_fcontext",
            [] { run_raw_round_trips(kRoundTrips); });
  bench.run("raw consumer seam + explicit handle",
            [] { run_handle_seam_round_trips(kRoundTrips); });
  bench.run("raw consumer seam + static yield/TLS current",
            [] { run_current_seam_round_trips(kRoundTrips); });
  bench.run("HYLE Fiber::resume/Fiber::yield",
            [] { run_hyle_round_trips(kRoundTrips); });
  return 0;
}
