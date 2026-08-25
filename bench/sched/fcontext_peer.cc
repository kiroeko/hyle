// Validation-only raw driver for the exact fcontext assembly selected by the
// hyle_sched product target.  A shared HYLE build puts the peer in a separate
// image; a static HYLE build reuses the fcontext symbols already selected into
// the final executable.

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "fcontext_bench_seam.h"
#include "fcontext.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/fiber.h"

struct HyleBenchFcontextSeamContext {
  using Entry = void (*)(void*);

  explicit HyleBenchFcontextSeamContext(hyle::mem::FiberStack&& owned_stack,
                                        Entry fn, void* fn_arg,
                                        bool current_semantics)
      : stack(std::move(owned_stack)),
        entry(fn),
        argument(fn_arg),
        tracks_current(current_semantics) {}

  hyle::mem::FiberStack stack;
  hyle::sched::fcontext_t suspended = nullptr;
  hyle::sched::fcontext_t caller = nullptr;
  Entry entry = nullptr;
  void* argument = nullptr;
  bool done = false;
  bool tracks_current = false;
};

static_assert(sizeof(HyleBenchFcontextSeamContext) <=
              kHyleBenchFcontextSeamStorageSize);
static_assert(alignof(HyleBenchFcontextSeamContext) <=
              kHyleBenchFcontextSeamStorageAlignment);

namespace {

using hyle::mem::FiberStack;
using hyle::sched::fcontext_t;
using hyle::sched::jump_fcontext;
using hyle::sched::kDefaultStackSize;
using hyle::sched::make_fcontext;
using hyle::sched::transfer_t;

FiberStack allocate_stack() {
  auto stack = FiberStack::allocate(kDefaultStackSize);
  if (!stack) std::abort();
  return *std::move(stack);
}

struct RawRoundTrips {
  int rounds = 0;
  bool done = false;
  fcontext_t suspended = nullptr;
};

thread_local HyleBenchFcontextSeamContext* t_current_seam = nullptr;

void raw_entry(transfer_t transfer) {
  auto* state = static_cast<RawRoundTrips*>(transfer.data);
  for (int i = 0; i < state->rounds; ++i) {
    transfer = jump_fcontext(transfer.fctx, state);
  }
  state->done = true;
  (void)jump_fcontext(transfer.fctx, state);
  std::abort();
}

void seam_entry(transfer_t transfer) {
  auto* context = static_cast<HyleBenchFcontextSeamContext*>(transfer.data);
  context->caller = transfer.fctx;
  context->entry(context->argument);
  context->done = true;
  (void)jump_fcontext(context->caller, context);
  std::abort();
}

}  // namespace

extern "C" HYLE_BENCH_PEER_API void hyle_bench_raw_fcontext_round_trips(
    int rounds) {
  FiberStack stack = allocate_stack();
  RawRoundTrips state{.rounds = rounds};
  state.suspended = make_fcontext(stack.top(), stack.size(), &raw_entry);
  while (!state.done) {
    const transfer_t transfer = jump_fcontext(state.suspended, &state);
    if (!state.done) state.suspended = transfer.fctx;
  }
}

extern "C" HYLE_BENCH_PEER_API HyleBenchFcontextSeamContext*
hyle_bench_fcontext_seam_create(void* storage, std::size_t storage_size,
                                HyleBenchFcontextSeamContext::Entry entry,
                                void* argument, bool tracks_current) {
  if (storage == nullptr ||
      storage_size < sizeof(HyleBenchFcontextSeamContext) ||
      reinterpret_cast<std::uintptr_t>(storage) %
              alignof(HyleBenchFcontextSeamContext) !=
          0) {
    std::abort();
  }
  auto* context = ::new (storage) HyleBenchFcontextSeamContext(
      allocate_stack(), entry, argument, tracks_current);
  context->suspended =
      make_fcontext(context->stack.top(), context->stack.size(), &seam_entry);
  return context;
}

extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_destroy(
    HyleBenchFcontextSeamContext* context) {
  if (context == nullptr) std::abort();
  context->~HyleBenchFcontextSeamContext();
}

extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_resume(
    HyleBenchFcontextSeamContext* context) {
  if (context == nullptr || context->done) std::abort();
  HyleBenchFcontextSeamContext* previous = nullptr;
  if (context->tracks_current) {
    previous = t_current_seam;
    t_current_seam = context;
  }
  const transfer_t transfer = jump_fcontext(context->suspended, context);
  if (context->tracks_current) t_current_seam = previous;
  if (!context->done) context->suspended = transfer.fctx;
}

extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_yield_handle(
    HyleBenchFcontextSeamContext* context) {
  if (context == nullptr) std::abort();
  const transfer_t transfer = jump_fcontext(context->caller, nullptr);
  context->caller = transfer.fctx;
}

extern "C" HYLE_BENCH_PEER_API void
hyle_bench_fcontext_seam_yield_current() {
  HyleBenchFcontextSeamContext* context = t_current_seam;
  if (context == nullptr) std::abort();
  const transfer_t transfer = jump_fcontext(context->caller, nullptr);
  context->caller = transfer.fctx;
}
