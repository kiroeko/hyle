#include "hyle/error/error_handler.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace hyle {

namespace {

// Built-in default: format to stderr. Severity tag mirrors clay's "error:" line.
void stderr_report(error_handler_o*, Severity sev, const char* file,
                   std::uint32_t line, const char* message) {
  const char* tag = sev == Severity::Fatal     ? "fatal"
                    : sev == Severity::Warning  ? "warning"
                                                : "error";
  if (file != nullptr) {
    std::fprintf(stderr, "%s:%u: %s: %s\n", file, line, tag, message);
  } else {
    std::fprintf(stderr, "%s: %s\n", tag, message);
  }
}

constexpr error_handler_t kStderrHandler{nullptr, stderr_report};

// Atomic so a handler swap (e.g. a test recorder) races cleanly with reporters.
std::atomic<const error_handler_t*> g_default{&kStderrHandler};

}  // namespace

const error_handler_t* error_default_handler() noexcept {
  return g_default.load(std::memory_order_acquire);
}

const error_handler_t* error_set_default_handler(
    const error_handler_t* h) noexcept {
  if (h == nullptr) h = &kStderrHandler;
  return g_default.exchange(h, std::memory_order_acq_rel);
}

void error_report(Severity sev, const char* file, std::uint32_t line,
                  const char* format, ...) {
  char buf[1024];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  const error_handler_t* h = g_default.load(std::memory_order_acquire);
  h->report(h->inst, sev, file, line, buf);

  // Fatal always crashes, regardless of handler -- the handler logged/dialoged,
  // now honor fail-fast. (clay tripped a null write; HYLE_TRAP is the clean form.)
  if (sev == Severity::Fatal) HYLE_TRAP();
}

// ---- ErrorRecord ------------------------------------------------------------

void ErrorRecord::on_report(error_handler_o* inst, Severity sev,
                            const char* file, std::uint32_t line,
                            const char* message) {
  auto* self = reinterpret_cast<ErrorRecord*>(inst);
  unsigned idx = static_cast<unsigned>(sev);
  if (idx < 3) ++self->counts_[idx];

  std::size_t n = std::strlen(message);
  if (n >= kMsgCap) n = kMsgCap - 1;
  std::memcpy(self->last_, message, n);
  self->last_[n] = '\0';

  if (self->backing != nullptr) {
    self->backing->report(self->backing->inst, sev, file, line, message);
  }
}

void ErrorRecord::install() {
  self_.inst = reinterpret_cast<error_handler_o*>(this);
  self_.report = &ErrorRecord::on_report;
  prev_ = error_set_default_handler(&self_);
}

void ErrorRecord::restore() noexcept { error_set_default_handler(prev_); }

std::uint32_t ErrorRecord::count(Severity sev) const noexcept {
  unsigned idx = static_cast<unsigned>(sev);
  return idx < 3 ? counts_[idx] : 0;
}

std::uint32_t ErrorRecord::total() const noexcept {
  return counts_[0] + counts_[1] + counts_[2];
}

}  // namespace hyle
