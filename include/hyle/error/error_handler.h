#pragma once

#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#ifndef HYLE_ERROR_API
#  if defined(HYLE_STATIC)
#    define HYLE_ERROR_API
#  elif defined(HYLE_ERROR_BUILD)
#    define HYLE_ERROR_API HYLE_EXPORT
#  else
#    define HYLE_ERROR_API HYLE_IMPORT
#  endif
#endif

// Pluggable diagnostic sink (clay sys/error.h error_handler). This is the
// human-readable DIAGNOSTIC channel -- "report this message somewhere a dev will
// see it" -- distinct from Status/expected (the machine-readable RETURN-VALUE
// channel). An embedder (host app, or another DLL) installs a handler to route
// HYLE's warnings/errors/fatals into its own logger, an error dialog, or a test
// recorder, across the C-ABI seam.
//
// Three improvements over clay:
//  * vtable is NON-variadic. clay's errorf/fatal were variadic function pointers
//    -- passing varargs through a cross-DLL function pointer is an ABI hazard
//    (calling convention / register-vs-stack rules differ). HYLE formats at the
//    call site (snprintf into a stack buffer) and the vtable takes a finished
//    const char*. Stable across compilers.
//  * one report() + a Severity, not two near-identical function pointers.
//  * clean fatal. clay's default fatal did rand()%1 + *(int*)0 = 10 + exit(-1)
//    (a deliberate null write to trip the debugger). HYLE traps via HYLE_TRAP()
//    in the dispatcher AFTER the handler runs -- the handler still gets to log /
//    show a dialog, but the crash is guaranteed regardless of which handler is
//    installed (charter §5 fail-fast).

namespace hyle {

enum class Severity : std::uint8_t { Warning = 0, Error = 1, Fatal = 2 };

typedef struct error_handler_o error_handler_o;

// Diagnostic sink vtable. `report` receives an already-formatted message; it
// must not assume the pointer outlives the call (copy if retaining).
struct error_handler_t {
  error_handler_o* inst;
  void (*report)(error_handler_o* inst, Severity sev, const char* file,
                 std::uint32_t line, const char* message);
};

// The process-wide default sink (a stderr logger until replaced). Dispatch and
// replacement are serialized in the .cc, so after set/restore returns the
// displaced handler is no longer executing and may be destroyed safely.
HYLE_ERROR_API const error_handler_t* error_default_handler() noexcept;
// Install `h` as the default; returns the previous one (restore it to undo).
// Passing nullptr resets to the built-in stderr handler.
HYLE_ERROR_API const error_handler_t* error_set_default_handler(
    const error_handler_t* h) noexcept;

// Format (printf-style) and dispatch to the current default handler. Severity
// Fatal additionally traps after the handler returns (never returns). Used by
// the macros below; call directly for a runtime Severity.
HYLE_ERROR_API void error_report(Severity sev, const char* file,
                                 std::uint32_t line, const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 4, 5)))
#endif
    ;

// ---- recording handler (clay create_record_handler, fixed-size) -------------
// Captures diagnostics for inspection (tests, deferred reporting) WITHOUT
// allocating: counts per severity + the last message inline. Optionally forwards
// to `backing` (e.g. keep logging while also counting). clay stored every message
// in a carray<char*> via an Allocator; HYLE keeps the last + counts, no heap.
struct HYLE_ERROR_API ErrorRecord {
  static constexpr std::size_t kMsgCap = 256;

  void install();                 // set as default, remembering the previous
  void restore() noexcept;        // reinstate the previous default
  std::uint32_t count(Severity sev) const noexcept;
  std::uint32_t total() const noexcept;
  const char* last_message() const noexcept { return last_; }

  const error_handler_t* backing = nullptr;  // optional forward target

 private:
  static void on_report(error_handler_o* inst, Severity sev, const char* file,
                        std::uint32_t line, const char* message);
  std::uint32_t counts_[3] = {0, 0, 0};
  char last_[kMsgCap] = {0};
  error_handler_t self_{};                 // vtable bound to this
  const error_handler_t* prev_ = nullptr;  // default to restore
};

}  // namespace hyle

// ---- macros -----------------------------------------------------------------
// HYLE_WARN / HYLE_ERROR: report a formatted diagnostic at this site.
#define HYLE_WARN(...) \
  ::hyle::error_report(::hyle::Severity::Warning, __FILE__, __LINE__, __VA_ARGS__)
#define HYLE_ERROR(...) \
  ::hyle::error_report(::hyle::Severity::Error, __FILE__, __LINE__, __VA_ARGS__)
// HYLE_FATAL: report then trap (never returns). The formatted complement to
// macros.h's HYLE_FATAL_ASSERT (which only takes a bare condition).
#define HYLE_FATAL(...) \
  ::hyle::error_report(::hyle::Severity::Fatal, __FILE__, __LINE__, __VA_ARGS__)

// HYLE_ASSERT (charter §5): evaluates to bool, SURVIVES release (unlike <cassert>
// assert, which compiles out under NDEBUG), and on failure REPORTS rather than
// invoking UB. Use it in a condition: `if (!HYLE_ASSERT(p != nullptr)) return ...;`
// -- the program keeps control and can recover, vs HYLE_FATAL_ASSERT which aborts.
#define HYLE_ASSERT(cond)                                                    \
  (HYLE_LIKELY(static_cast<bool>(cond))                                      \
       ? true                                                                \
       : (::hyle::error_report(::hyle::Severity::Error, __FILE__, __LINE__,  \
                               "assertion failed: %s", #cond),               \
          false))

// ---- stub / deprecation convenience (HYLE names; not clay identifiers) ------
// Temporary unfinished API body: log which function is missing, then mark the
// path unreachable. NOT a recoverable error -- for that return
// Status(StatusCode::Unimplemented, ...).
#define HYLE_NOT_IMPLEMENTED()                                                 \
  do {                                                                         \
    ::hyle::error_report(::hyle::Severity::Error, __FILE__, __LINE__,          \
                         "'%s' not yet implemented", __func__);               \
    HYLE_UNREACHABLE();                                                        \
  } while (0)

// Soft deprecation notice (does not abort; function may continue).
#define HYLE_DEPRECATED_WARNING(message)                                       \
  ::hyle::error_report(::hyle::Severity::Warning, __FILE__, __LINE__,          \
                       "function `%s` has been deprecated. %s", __func__,      \
                       (message))
