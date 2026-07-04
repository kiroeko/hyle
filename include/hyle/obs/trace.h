#pragma once

// obs-02: instrumentation / trace hooks (capability C6). A COMPILE-TIME-GATED seam
// for zone timing, counters/plots, fiber tracking, and frame marks. This is the
// "hot-path" half of obs/ (log.h is the cold structured-log half).
//
// ZERO-COST LAW (the load-bearing constraint, ratified 2026-07-03): these hooks
// are meant to be embedded in the FIBER SWITCH (~21ns) and io-reap nanosecond hot
// paths. So the gate is COMPILE-TIME, not the runtime `should_log` gate log.h uses
// (a relaxed load + a cross-DLL call -- fine for logging, far too much for a
// nanosecond path). When HYLE_TRACE is undefined (the default, and every shipped/
// benchmarked build) EVERY macro below expands to nothing -- `(void)0` / a discarded
// value -- so the compiler emits IDENTICAL code with and without the hooks present.
// This mirrors tracy's TRACY_ENABLE model exactly (tracy macros vanish when it is
// undefined). Acceptance is by objdump: the fiber-switch assembly must be
// instruction-for-instruction identical gated on/off, and scheduler_ext_bench must
// not move. "Embed in the hot path from day one" only holds because the OFF state
// is provably free.
//
// TWO LAYERS:
//   1. This header is HYLE's OWN stable hook surface (HYLE_TRACE_*). Product code
//      and the scheduler/io hot paths call ONLY these -- never tracy directly --
//      so the profiler backend can change without touching a single call site.
//   2. A backend fills them in. The first backend is tracy (HYLE_TRACE_TRACY),
//      borrowed whole (header + one TU), not reinvented. Selected at build time.
//
// Enable with -DHYLE_TRACE (+ the backend's own switch, e.g. -DHYLE_TRACE_TRACY
// which also defines TRACY_ENABLE for the vendored client). Off => this header
// pulls in NOTHING (no backend include), keeping the default build dependency-free.

// ---- backend selection ------------------------------------------------------
// HYLE_TRACE is the master gate. A backend macro (HYLE_TRACE_TRACY) picks who fills
// the hooks. Defining a backend implies the master gate (so callers need only pass
// the backend switch). With no backend, HYLE_TRACE alone still gates everything off
// -- the hooks stay empty (useful to prove the seam compiles with zero backend).
#if defined(HYLE_TRACE_TRACY) && !defined(HYLE_TRACE)
#  define HYLE_TRACE
#endif

#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)

// tracy backend. Fiber tracking needs TRACY_FIBERS (the TracyFiber* macros only
// exist when it is set) -- turn it on here so HYLE's fiber hooks always resolve.
#  ifndef TRACY_FIBERS
#    define TRACY_FIBERS
#  endif
#  include <tracy/Tracy.hpp>

// Zone timing (RAII: times the enclosing scope). tracy caches a static constexpr
// SourceLocationData per site -- zero allocation, the runtime cost is one lock-free
// queue write at scope entry and one at exit.
#  define HYLE_TRACE_ZONE() ZoneScoped
#  define HYLE_TRACE_ZONE_NAMED(name) ZoneScopedN(name)

// A numeric time series (tracy plot). `name` MUST be a string literal / long-lived
// pointer (tracy stores the pointer by identity, not a copy).
#  define HYLE_TRACE_PLOT(name, value) TracyPlot(name, value)
// tracy has no discrete incremental counter; a counter IS a plot of its running
// value. Callers that want an increment maintain the sum and plot it (documented).
#  define HYLE_TRACE_COUNTER(name, value) TracyPlot(name, value)

// Fiber tracking: mark which fiber the current OS thread is running, so zones/plots
// attribute to the FIBER track rather than the physical worker thread -- exactly
// what an M:N fiber scheduler needs. `name` identifies the fiber and MUST be a
// stable, long-lived pointer (tracy keys the fiber by pointer identity; the SAME
// fiber must always pass the SAME pointer, even across worker migration). Enter is
// idempotent per switch; a direct fiber->fiber switch may Enter again without an
// intervening Leave (tracy allows it). Leave when returning to non-fiber code.
#  define HYLE_TRACE_FIBER_ENTER(name) TracyFiberEnter(name)
#  define HYLE_TRACE_FIBER_LEAVE() TracyFiberLeave

// Frame boundary (a serving loop iteration / a graph tick).
#  define HYLE_TRACE_FRAME() FrameMark
#  define HYLE_TRACE_FRAME_NAMED(name) FrameMarkNamed(name)

// A transient message on the timeline.
#  define HYLE_TRACE_MESSAGE(msg) TracyMessageL(msg)

#else  // ---- OFF (default) or a gate with no backend: every hook vanishes -------

// Each macro expands to a statement/expression that generates NO code. Args are
// NOT evaluated (a plot value / message that is itself expensive must not run when
// tracing is off) -- consistent with tracy's off-state, which also drops its args.
#  define HYLE_TRACE_ZONE() ((void)0)
#  define HYLE_TRACE_ZONE_NAMED(name) ((void)0)
#  define HYLE_TRACE_PLOT(name, value) ((void)0)
#  define HYLE_TRACE_COUNTER(name, value) ((void)0)
#  define HYLE_TRACE_FIBER_ENTER(name) ((void)0)
#  define HYLE_TRACE_FIBER_LEAVE() ((void)0)
#  define HYLE_TRACE_FRAME() ((void)0)
#  define HYLE_TRACE_FRAME_NAMED(name) ((void)0)
#  define HYLE_TRACE_MESSAGE(msg) ((void)0)

#endif
