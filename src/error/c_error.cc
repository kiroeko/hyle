// Direct Native FFI bridge for Status. Foreign runtimes see an opaque owning
// pointer with NULL == OK; the C++ tagged-word representation remains private.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

#include "hyle/c/error.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "status_access.h"

struct hyle_status_handle {
  std::uint16_t code;
};

namespace {

// Code-only failures remain allocation-free after crossing the FFI seam.
hyle_status_handle g_inline_handles[] = {
    {HYLE_STATUS_CANCELLED},           {HYLE_STATUS_UNKNOWN},
    {HYLE_STATUS_INVALID_ARGUMENT},    {HYLE_STATUS_DEADLINE_EXCEEDED},
    {HYLE_STATUS_NOT_FOUND},           {HYLE_STATUS_ALREADY_EXISTS},
    {HYLE_STATUS_PERMISSION_DENIED},   {HYLE_STATUS_RESOURCE_EXHAUSTED},
    {HYLE_STATUS_FAILED_PRECONDITION}, {HYLE_STATUS_ABORTED},
    {HYLE_STATUS_OUT_OF_RANGE},        {HYLE_STATUS_UNIMPLEMENTED},
    {HYLE_STATUS_INTERNAL},            {HYLE_STATUS_UNAVAILABLE},
    {HYLE_STATUS_DATA_LOSS},           {HYLE_STATUS_UNAUTHENTICATED},
};

static_assert(sizeof(g_inline_handles) / sizeof(g_inline_handles[0]) == 16);

bool decode_inline_handle(hyle_status status,
                          hyle::StatusCode &code) noexcept {
  for (std::size_t i = 0;
       i < sizeof(g_inline_handles) / sizeof(g_inline_handles[0]); ++i) {
    if (status == &g_inline_handles[i]) {
      code = static_cast<hyle::StatusCode>(i + 1);
      return true;
    }
  }
  return false;
}

hyle::Status borrow_from_ffi(hyle_status status) noexcept {
  if (status == nullptr)
    return hyle::Status{};
  hyle::StatusCode code = hyle::StatusCode::Unknown;
  if (decode_inline_handle(status, code))
    return hyle::Status(code);
  return hyle::detail::StatusAccess::adopt(
      reinterpret_cast<std::uintptr_t>(status));
}

void forget_borrow(hyle::Status &status) noexcept {
  (void)hyle::detail::StatusAccess::release(status);
}

hyle_status_code to_c_code(hyle::StatusCode code) noexcept {
  return static_cast<hyle_status_code>(static_cast<std::uint16_t>(code));
}

hyle_native_error_domain
to_c_domain(hyle::NativeErrorDomain domain) noexcept {
  return static_cast<hyle_native_error_domain>(
      static_cast<std::uint16_t>(domain));
}

#define HYLE_STATUS_ABI_MATCH(name, screaming, value)                         \
  static_assert(static_cast<std::uint16_t>(hyle::StatusCode::name) ==         \
                    HYLE_STATUS_##screaming,                                  \
                "Native FFI status code drift: " #screaming);
HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_ABI_MATCH)
#undef HYLE_STATUS_ABI_MATCH

#define HYLE_NATIVE_DOMAIN_ABI_MATCH(cpp_name, c_name)                        \
  static_assert(static_cast<std::uint16_t>(                                   \
                    hyle::NativeErrorDomain::cpp_name) ==                     \
                    static_cast<std::uint16_t>(HYLE_NATIVE_ERROR_##c_name),    \
                "Native FFI native domain drift: " #c_name)
HYLE_NATIVE_DOMAIN_ABI_MATCH(None, NONE);
HYLE_NATIVE_DOMAIN_ABI_MATCH(PosixErrno, POSIX_ERRNO);
HYLE_NATIVE_DOMAIN_ABI_MATCH(Win32, WIN32);
HYLE_NATIVE_DOMAIN_ABI_MATCH(Vulkan, VULKAN);
HYLE_NATIVE_DOMAIN_ABI_MATCH(CudaRuntime, CUDA_RUNTIME);
HYLE_NATIVE_DOMAIN_ABI_MATCH(CudaDriver, CUDA_DRIVER);
HYLE_NATIVE_DOMAIN_ABI_MATCH(Nvrtc, NVRTC);
HYLE_NATIVE_DOMAIN_ABI_MATCH(HResult, HRESULT);
HYLE_NATIVE_DOMAIN_ABI_MATCH(Unknown, UNKNOWN);
#undef HYLE_NATIVE_DOMAIN_ABI_MATCH

std::uint64_t copy_view(std::string_view value, char *buffer,
                        std::uint64_t capacity) noexcept {
  if (buffer != nullptr && capacity != 0) {
    const std::uint64_t writable = capacity - 1;
    const std::size_t copied =
        value.size() < writable ? value.size()
                                : static_cast<std::size_t>(writable);
    if (copied != 0)
      std::memcpy(buffer, value.data(), copied);
    buffer[copied] = '\0';
  }
  return static_cast<std::uint64_t>(value.size());
}

} // namespace

namespace hyle::detail {

hyle_status status_to_ffi(Status &&status) noexcept {
  const std::uintptr_t rep = StatusAccess::release(status);
  if (rep == 0)
    return nullptr;
  if ((rep & std::uintptr_t{1}) != 0) {
    const std::size_t code = static_cast<std::size_t>(rep >> 2);
    HYLE_FATAL_ASSERT(code >= 1 && code <= 16);
    return &g_inline_handles[code - 1];
  }
  return reinterpret_cast<hyle_status>(rep);
}

} // namespace hyle::detail

extern "C" {

uint32_t HYLE_ERROR_C_CALL hyle_error_ffi_abi_version(void) {
  return HYLE_FFI_ABI_VERSION;
}

void HYLE_ERROR_C_CALL hyle_status_destroy(hyle_status status) {
  if (status == nullptr)
    return;
  hyle::StatusCode ignored = hyle::StatusCode::Unknown;
  if (decode_inline_handle(status, ignored))
    return;
  hyle::detail::status_release(reinterpret_cast<std::uintptr_t>(status));
}

hyle_status_code HYLE_ERROR_C_CALL hyle_status_get_code(hyle_status status) {
  hyle::Status borrowed = borrow_from_ffi(status);
  const hyle_status_code result = to_c_code(borrowed.code());
  forget_borrow(borrowed);
  return result;
}

int32_t HYLE_ERROR_C_CALL hyle_status_get_native_error(
    hyle_status status, hyle_native_error_domain *out_domain,
    int64_t *out_code) {
  hyle::Status borrowed = borrow_from_ffi(status);
  const hyle::NativeError native = borrowed.native_error();
  forget_borrow(borrowed);
  if (out_domain != nullptr)
    *out_domain = to_c_domain(native.domain);
  if (out_code != nullptr)
    *out_code = native.code;
  return native.present() ? 1 : 0;
}

uint64_t HYLE_ERROR_C_CALL hyle_status_message(hyle_status status,
                                               char *buffer,
                                               uint64_t capacity) {
  hyle::Status borrowed = borrow_from_ffi(status);
  const std::uint64_t result = copy_view(borrowed.message(), buffer, capacity);
  forget_borrow(borrowed);
  return result;
}

uint64_t HYLE_ERROR_C_CALL hyle_status_to_string(hyle_status status,
                                                 char *buffer,
                                                 uint64_t capacity) {
  hyle::Status borrowed = borrow_from_ffi(status);
  const std::size_t result = borrowed.ToString(
      buffer, static_cast<std::size_t>(capacity));
  forget_borrow(borrowed);
  return static_cast<std::uint64_t>(result);
}

} // extern "C"
