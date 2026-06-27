#include "hyle/error/status.h"

#include <array>
#include <cstdio>

#include "hyle/error/result.h"

namespace hyle {

std::string_view status_code_name(StatusCode code) noexcept {
  switch (code) {
#define HYLE_STATUS_NAME_CASE(name, value) \
  case StatusCode::name:                   \
    return #name;
    HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_NAME_CASE)
#undef HYLE_STATUS_NAME_CASE
  }
  return "Unknown";
}

std::string Status::ToString() const {
  if (ok()) return "OK";
  std::string out(status_code_name(code_));
  if (msg_ && !msg_->empty()) {
    out += ": ";
    out += *msg_;
  }
  return out;
}

namespace detail {

void log_assert_failure(const char* expr, const char* file, int line) noexcept {
  std::fprintf(stderr, "HYLE assert failed: %s\n  at %s:%d\n", expr, file, line);
}

}  // namespace detail
}  // namespace hyle
