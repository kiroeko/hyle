// C-ABI bridge for Status (include/hyle/c/error.h). The C handle owns the same
// tagged word as hyle::Status; bridge operations transfer or borrow that
// ownership without exposing the heap representation. Native C++ consumers
// inline context no-ops; this DLL-local bridge enters the exported method.

#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/c/error.h"
#include "hyle/error/status.h"
#include "status_access.h"

namespace {

static_assert(sizeof(hyle_status) == sizeof(std::uintptr_t));
static_assert(alignof(hyle_status) == alignof(std::uintptr_t));
static_assert(std::is_standard_layout_v<hyle_status>);
static_assert(std::is_trivially_copyable_v<hyle_status>);

hyle::StatusCode to_cpp_code(hyle_status_code code) noexcept {
  const auto raw = static_cast<std::uint16_t>(code);
  return hyle::detail::normalize_status_code(
      static_cast<hyle::StatusCode>(raw));
}

hyle_status_code to_c_code(hyle::StatusCode code) noexcept {
  return static_cast<hyle_status_code>(static_cast<std::uint16_t>(code));
}

#define HYLE_STATUS_ABI_MATCH(name, screaming, value)          \
  static_assert(                                               \
      static_cast<std::uint16_t>(hyle::StatusCode::name) ==    \
          static_cast<std::uint16_t>(HYLE_STATUS_##screaming), \
      "C hyle_status_code drifted from C++ StatusCode: " #screaming);
HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_ABI_MATCH)
#undef HYLE_STATUS_ABI_MATCH

hyle_status steal_to_c(hyle::Status &&status) noexcept {
  return hyle_status{hyle::detail::StatusAccess::release(status)};
}

hyle::Status adopt_from_c(hyle_status status) noexcept {
  return hyle::detail::StatusAccess::adopt(status.rep);
}

std::size_t copy_sv(std::string_view value, char *buf,
                    std::size_t cap) noexcept {
  if (buf == nullptr || cap == 0) return value.size();
  const std::size_t write = value.size() < cap - 1 ? value.size() : cap - 1;
  if (write != 0) std::memcpy(buf, value.data(), write);
  buf[write] = '\0';
  return value.size();
}

}  // namespace

extern "C" {

hyle_status hyle_status_ok(void) { return steal_to_c(hyle::Status()); }

hyle_status hyle_status_new(hyle_status_code code, const char *msg) {
  const hyle::StatusCode normalized = to_cpp_code(code);
  if (normalized == hyle::StatusCode::Ok) return hyle_status_ok();
  const std::string_view message =
      msg == nullptr ? std::string_view() : std::string_view(msg);
  return steal_to_c(hyle::Status(normalized, message));
}

hyle_status_code hyle_status_code_from_errno(int error_number) {
  return to_c_code(hyle::status_code_from_errno(error_number));
}

hyle_status hyle_status_from_errno(int error_number, const char *context) {
  return steal_to_c(hyle::status_from_errno(
      error_number, context == nullptr ? std::string_view()
                                      : std::string_view(context)));
}

void hyle_status_free(hyle_status *status) {
  if (status == nullptr) return;
  hyle::Status owned = adopt_from_c(*status);
  status->rep = 0;
}

hyle_status hyle_status_clone(hyle_status status) {
  hyle::Status borrowed = adopt_from_c(status);
  hyle::Status copy = borrowed;
  (void)hyle::detail::StatusAccess::release(borrowed);
  return steal_to_c(std::move(copy));
}

void hyle_status_move(hyle_status *dst, hyle_status *src) {
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

void hyle_status_add_context(hyle_status *status, const char *context) {
  if (status == nullptr || context == nullptr || context[0] == '\0') return;
  hyle::Status owned = adopt_from_c(*status);
  status->rep = 0;
  owned = std::move(owned).WithContext(context);
  *status = steal_to_c(std::move(owned));
}

int hyle_status_is_ok(hyle_status status) {
  hyle::Status borrowed = adopt_from_c(status);
  const int result = borrowed.ok() ? 1 : 0;
  (void)hyle::detail::StatusAccess::release(borrowed);
  return result;
}

hyle_status_code hyle_status_get_code(hyle_status status) {
  hyle::Status borrowed = adopt_from_c(status);
  const hyle_status_code result = to_c_code(borrowed.code());
  (void)hyle::detail::StatusAccess::release(borrowed);
  return result;
}

int hyle_status_has_message(hyle_status status) {
  hyle::Status borrowed = adopt_from_c(status);
  const int result = borrowed.has_message() ? 1 : 0;
  (void)hyle::detail::StatusAccess::release(borrowed);
  return result;
}

int hyle_status_equals(hyle_status a, hyle_status b) {
  hyle::Status aa = adopt_from_c(a);
  hyle::Status bb = adopt_from_c(b);
  const int result = aa == bb ? 1 : 0;
  (void)hyle::detail::StatusAccess::release(aa);
  (void)hyle::detail::StatusAccess::release(bb);
  return result;
}

const char *hyle_status_code_name(hyle_status_code code) {
  return hyle::status_code_name(to_cpp_code(code)).data();
}

size_t hyle_status_message(hyle_status status, char *buf, size_t cap) {
  hyle::Status borrowed = adopt_from_c(status);
  const size_t result = copy_sv(borrowed.message(), buf, cap);
  (void)hyle::detail::StatusAccess::release(borrowed);
  return result;
}

size_t hyle_status_to_string(hyle_status status, char *buf, size_t cap) {
  hyle::Status borrowed = adopt_from_c(status);
  const size_t result = borrowed.ToString(buf, cap);
  (void)hyle::detail::StatusAccess::release(borrowed);
  return result;
}

const char *hyle_status_message_cstr(hyle_status status) {
  hyle::Status borrowed = adopt_from_c(status);
  const std::string_view message = borrowed.message();
  (void)hyle::detail::StatusAccess::release(borrowed);
  return message.empty() ? "" : message.data();
}

}  // extern "C"
