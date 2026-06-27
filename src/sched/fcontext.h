#pragma once

#include <cstddef>

// Minimal HYLE-side declarations for boost.context's three assembly routines
// (vendored under third_party/fcontext/). This shim is what lets us use the raw
// asm with ZERO boost headers: fcontext_t and transfer_t are plain PODs, and the
// only "boost-ism" (the BOOST_CONTEXT_DECL/CALLDECL macros) is simply omitted --
// on x64 the default C calling convention is what the asm expects.
//
// Semantics (see sched-design.md §1):
//   make_fcontext(sp, size, fn): build a context on the stack whose top is `sp`
//     (highest address; stack grows down), entry `fn` receiving a transfer_t.
//   jump_fcontext(to, vp): save the current context, switch to `to`, deliver vp
//     as transfer_t.data; returns the transfer_t handed back on the next switch.
//   ontop_fcontext: jump but run fn on top of the target (cleanup/unwind) -- not
//     used yet by the skeleton; declared for completeness.

namespace hyle::sched {

using fcontext_t = void*;

struct transfer_t {
  fcontext_t fctx;  // the context we switched away from (resume target)
  void* data;       // payload passed across the switch
};

extern "C" transfer_t jump_fcontext(fcontext_t to, void* vp);
extern "C" fcontext_t make_fcontext(void* sp, std::size_t size,
                                    void (*fn)(transfer_t));
extern "C" transfer_t ontop_fcontext(fcontext_t to, void* vp,
                                     transfer_t (*fn)(transfer_t));

}  // namespace hyle::sched
