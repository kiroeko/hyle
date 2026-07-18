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
  using hyle::StatusCode;
  switch (c) {
    case HYLE_STATUS_OK:
      return StatusCode::Ok;
    case HYLE_STATUS_CANCELLED:
      return StatusCode::Cancelled;
    case HYLE_STATUS_UNKNOWN:
      return StatusCode::Unknown;
    case HYLE_STATUS_INVALID_ARGUMENT:
      return StatusCode::InvalidArgument;
    case HYLE_STATUS_DEADLINE_EXCEEDED:
      return StatusCode::DeadlineExceeded;
    case HYLE_STATUS_NOT_FOUND:
      return StatusCode::NotFound;
    case HYLE_STATUS_ALREADY_EXISTS:
      return StatusCode::AlreadyExists;
    case HYLE_STATUS_PERMISSION_DENIED:
      return StatusCode::PermissionDenied;
    case HYLE_STATUS_RESOURCE_EXHAUSTED:
      return StatusCode::ResourceExhausted;
    case HYLE_STATUS_FAILED_PRECONDITION:
      return StatusCode::FailedPrecondition;
    case HYLE_STATUS_ABORTED:
      return StatusCode::Aborted;
    case HYLE_STATUS_OUT_OF_RANGE:
      return StatusCode::OutOfRange;
    case HYLE_STATUS_UNIMPLEMENTED:
      return StatusCode::Unimplemented;
    case HYLE_STATUS_INTERNAL:
      return StatusCode::Internal;
    case HYLE_STATUS_UNAVAILABLE:
      return StatusCode::Unavailable;
    case HYLE_STATUS_DATA_LOSS:
      return StatusCode::DataLoss;
  }
  return StatusCode::Unknown;
}

hyle_status_code to_c_code(hyle::StatusCode c) noexcept {
  return static_cast<hyle_status_code>(static_cast<std::uint16_t>(c));
}

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

int hyle_error_c_abi_version(void) { return HYLE_ERROR_C_ABI_VERSION; }

int hyle_error_api_get(hyle_error_api* out, int expected_abi) {
  if (out == nullptr) return -1;
  if (expected_abi != HYLE_ERROR_C_ABI_VERSION) return -1;
  out->abi_version = HYLE_ERROR_C_ABI_VERSION;
  out->status_ok = &hyle_status_ok;
  out->status_new = &hyle_status_new;
  out->status_free = &hyle_status_free;
  out->status_clone = &hyle_status_clone;
  out->status_move = &hyle_status_move;
  out->status_is_ok = &hyle_status_is_ok;
  out->status_get_code = &hyle_status_get_code;
  out->status_has_message = &hyle_status_has_message;
  out->status_equals = &hyle_status_equals;
  out->status_code_name = &hyle_status_code_name;
  out->status_message = &hyle_status_message;
  out->status_to_string = &hyle_status_to_string;
  out->error_set_handler = &hyle_error_set_handler;
  out->error_report = &hyle_error_report;
  out->error_reportf = &hyle_error_reportf;
  out->c_abi_version = &hyle_error_c_abi_version;

  /* v3 appended */
  out->error_push_handler = &hyle_error_push_handler;
  out->error_pop_handler = &hyle_error_pop_handler;
  return 0;
}

}  // extern "C"
