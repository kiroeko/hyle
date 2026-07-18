// C-ABI bridge for the error module (include/hyle/c/error.h). Pure forwarding:
// no extra policy. Lives in hyle_error so HYLE_ERROR_BUILD exports the symbols.

#include "hyle/c/error.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>  // std::nothrow for the push/pop handler token
#include <string_view>

#include "hyle/error/error_handler.h"
#include "hyle/error/status.h"

namespace {

hyle::StatusCode to_cpp_code(hyle_status_code c) noexcept {
  // Values are identical to StatusCode by construction (static_assert'd below),
  // so this is a plain cast -- no switch to keep in sync. An out-of-range FFI
  // input is a contract violation, deliberately not clamped to Unknown here
  // (status_code_name still maps unknown values to "Unknown").
  return static_cast<hyle::StatusCode>(static_cast<std::uint16_t>(c));
}

hyle_status_code to_c_code(hyle::StatusCode c) noexcept {
  return static_cast<hyle_status_code>(static_cast<std::uint16_t>(c));
}

// Single-source-of-truth guard: the hand-written C enum (hyle/c/error.h) and the
// X-macro-generated C++ StatusCode must agree value-for-value, since both casts
// above depend on it. Every row of the shared list is checked here -- add or
// renumber a code on one side without the other and this fails to compile.
#define HYLE_STATUS_ABI_MATCH(name, screaming, value)                          \
  static_assert(static_cast<std::uint16_t>(hyle::StatusCode::name) ==          \
                    static_cast<std::uint16_t>(HYLE_STATUS_##screaming),       \
                "C hyle_status_code drifted from C++ StatusCode: " #screaming);
HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_ABI_MATCH)
#undef HYLE_STATUS_ABI_MATCH

hyle_status steal_to_c(hyle::Status&& s) noexcept {
  hyle_status out;
  out.rep = s.release_rep();
  return out;
}

hyle::Status adopt_from_c(hyle_status s) noexcept {
  return hyle::Status::adopt_rep(s.rep);
}

// ---- C diagnostic handler (process-wide, single slot) -----------------------
struct CHandlerState {
  hyle_error_report_fn fn = nullptr;
  void* user = nullptr;
  hyle::error_handler_t vtable{};
  const hyle::error_handler_t* prev = nullptr;
  bool installed = false;
};

CHandlerState& c_handler_state() noexcept {
  static CHandlerState st;
  return st;
}

void c_handler_on_report(hyle::error_handler_o* inst, hyle::Severity sev,
                         const char* file, std::uint32_t line,
                         const char* message) {
  auto* st = reinterpret_cast<CHandlerState*>(inst);
  if (st == nullptr || st->fn == nullptr) return;
  st->fn(st->user, static_cast<hyle_severity>(static_cast<std::uint8_t>(sev)),
         file, line, message);
}


size_t copy_sv(std::string_view sv, char* buf, size_t cap) noexcept {
  if (buf == nullptr || cap == 0) return sv.size();
  size_t n = sv.size();
  size_t write = n < cap - 1 ? n : cap - 1;
  if (write > 0) std::memcpy(buf, sv.data(), write);
  buf[write] = '\0';
  return n;
}

}  // namespace

// Nesting-safe push/pop token (v3). Unlike the single static c_handler_state(),
// each push OWNS its own state + vtable on the heap, so overrides nest (LIFO):
// the token records the exact handler it displaced and reinstates it on pop.
struct hyle_error_saved_handler_o {
  CHandlerState state;                               // this override's fn/user + vtable
  const hyle::error_handler_t* displaced = nullptr;  // reinstated on pop
};

extern "C" {

hyle_status hyle_status_ok(void) { return steal_to_c(hyle::Status()); }

hyle_status hyle_status_new(hyle_status_code code, const char* msg) {
  if (code == HYLE_STATUS_OK) return hyle_status_ok();
  std::string_view m = msg != nullptr ? std::string_view(msg) : std::string_view();
  return steal_to_c(hyle::Status(to_cpp_code(code), m));
}

void hyle_status_free(hyle_status* s) {
  if (s == nullptr) return;
  hyle::Status owned = adopt_from_c(*s);
  s->rep = 0;  // clear C handle before owned dtor releases the heap rep
  (void)owned;
}

hyle_status hyle_status_clone(hyle_status s) {
  hyle::Status tmp = adopt_from_c(s);
  hyle::Status copy = tmp;          // retain if heap
  (void)tmp.release_rep();          // give ownership of original word back to caller
  return steal_to_c(std::move(copy));
}

void hyle_status_move(hyle_status* dst, hyle_status* src) {
  if (dst == nullptr) return;
  if (src == nullptr) {
    hyle_status_free(dst);
    return;
  }
  if (dst == src) return;
  hyle_status_free(dst);
  dst->rep = src->rep;
  src->rep = 0;
}

int hyle_status_is_ok(hyle_status s) {
  hyle::Status tmp = adopt_from_c(s);
  int ok = tmp.ok() ? 1 : 0;
  (void)tmp.release_rep();
  return ok;
}

hyle_status_code hyle_status_get_code(hyle_status s) {
  hyle::Status tmp = adopt_from_c(s);
  hyle_status_code c = to_c_code(tmp.code());
  (void)tmp.release_rep();
  return c;
}

int hyle_status_has_message(hyle_status s) {
  hyle::Status tmp = adopt_from_c(s);
  int h = tmp.has_message() ? 1 : 0;
  (void)tmp.release_rep();
  return h;
}

int hyle_status_equals(hyle_status a, hyle_status b) {
  hyle::Status aa = adopt_from_c(a);
  hyle::Status bb = adopt_from_c(b);
  int eq = (aa == bb) ? 1 : 0;
  (void)aa.release_rep();
  (void)bb.release_rep();
  return eq;
}

const char* hyle_status_code_name(hyle_status_code code) {
  // status_code_name returns a string_view into a static literal table.
  std::string_view n = hyle::status_code_name(to_cpp_code(code));
  return n.data();  // NUL-terminated string literals from #name
}

size_t hyle_status_message(hyle_status s, char* buf, size_t cap) {
  hyle::Status tmp = adopt_from_c(s);
  size_t n = copy_sv(tmp.message(), buf, cap);
  (void)tmp.release_rep();
  return n;
}

size_t hyle_status_to_string(hyle_status s, char* buf, size_t cap) {
  hyle::Status tmp = adopt_from_c(s);
  size_t n = tmp.ToString(buf, cap);
  (void)tmp.release_rep();
  return n;
}

const char* hyle_status_message_cstr(hyle_status s) {
  hyle::Status tmp = adopt_from_c(s);
  std::string_view m = tmp.message();
  (void)tmp.release_rep();
  // Heap message bytes are NUL-terminated and owned by the caller's handle, so
  // the pointer stays valid until it is freed/moved; OK/inline codes carry no
  // message -> return the empty string rather than a NULL data() pointer.
  return m.empty() ? "" : m.data();
}

void hyle_error_set_handler(hyle_error_report_fn fn, void* user) {
  CHandlerState& st = c_handler_state();
  if (fn == nullptr) {
    if (st.installed) {
      hyle::error_set_default_handler(st.prev);
      st.installed = false;
      st.prev = nullptr;
      st.fn = nullptr;
      st.user = nullptr;
    }
    return;
  }
  st.fn = fn;
  st.user = user;
  st.vtable.inst = reinterpret_cast<hyle::error_handler_o*>(&st);
  st.vtable.report = &c_handler_on_report;
  if (!st.installed) {
    st.prev = hyle::error_set_default_handler(&st.vtable);
    st.installed = true;
  } else {
    // Already installed: vtable pointer is stable; just update fn/user.
    (void)hyle::error_set_default_handler(&st.vtable);
  }
}

hyle_error_saved_handler_o* hyle_error_push_handler(hyle_error_report_fn fn,
                                                    void* user) {
  auto* tok = new (std::nothrow) hyle_error_saved_handler_o{};
  if (tok == nullptr) return nullptr;  // still-installed handler unchanged
  tok->state.fn = fn;
  tok->state.user = user;
  if (fn != nullptr) {
    tok->state.vtable.inst =
        reinterpret_cast<hyle::error_handler_o*>(&tok->state);
    tok->state.vtable.report = &c_handler_on_report;
    // Install this token's own vtable; capture whatever it displaced.
    tok->displaced = hyle::error_set_default_handler(&tok->state.vtable);
  } else {
    // Push a "reset to built-in" scope: install default, remember the previous.
    tok->displaced = hyle::error_set_default_handler(nullptr);
  }
  tok->state.installed = true;
  return tok;
}

void hyle_error_pop_handler(hyle_error_saved_handler_o* saved) {
  if (saved == nullptr) return;
  // Reinstate exactly the handler this push displaced (LIFO restore).
  hyle::error_set_default_handler(saved->displaced);
  delete saved;
}

void hyle_error_report(hyle_severity severity, const char* file, uint32_t line,
                       const char* message) {
  const char* msg = message != nullptr ? message : "";
  const char* f = file != nullptr ? file : "";
  // error_report is printf-style; pass message as %s data.
  hyle::error_report(static_cast<hyle::Severity>(severity), f, line, "%s", msg);
}

void hyle_error_reportf(hyle_severity severity, const char* file, uint32_t line,
                        const char* fmt, ...) {
  const char* f = file != nullptr ? file : "";
  char buf[1024];
  buf[0] = '\0';
  if (fmt != nullptr) {
    va_list ap;
    va_start(ap, fmt);
    // vsnprintf is C++-safe here; cap leaves room for NUL.
#if defined(_MSC_VER)
    (void)_vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
#else
    (void)std::vsnprintf(buf, sizeof(buf), fmt, ap);
#endif
    va_end(ap);
  }
  hyle::error_report(static_cast<hyle::Severity>(severity), f, line, "%s", buf);
}

}  // extern "C"
