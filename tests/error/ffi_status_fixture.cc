#include <cstdint>
#include <string_view>

#include "../../src/error/status_access.h"
#include "hyle/c/error.h"
#include "hyle/error/status.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#define HYLE_ERROR_FFI_FIXTURE_API __declspec(dllexport)
#define HYLE_ERROR_FFI_FIXTURE_CALL __cdecl
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_ERROR_FFI_FIXTURE_API __attribute__((visibility("default")))
#define HYLE_ERROR_FFI_FIXTURE_CALL
#else
#define HYLE_ERROR_FFI_FIXTURE_API
#define HYLE_ERROR_FFI_FIXTURE_CALL
#endif

extern "C" {

// This test-only producer is a separate shared artifact. Managed probes must
// accept its bootstrap before resolving any of the deliberate failure sources.
HYLE_ERROR_FFI_FIXTURE_API std::uint32_t HYLE_ERROR_FFI_FIXTURE_CALL
hyle_error_status_fixture_ffi_abi_version(void) {
  return HYLE_FFI_ABI_VERSION;
}

HYLE_ERROR_FFI_FIXTURE_API hyle_status HYLE_ERROR_FFI_FIXTURE_CALL
hyle_error_status_fixture_ok(void) {
  return hyle::detail::status_to_ffi(hyle::Status::Ok());
}

HYLE_ERROR_FFI_FIXTURE_API hyle_status HYLE_ERROR_FFI_FIXTURE_CALL
hyle_error_status_fixture_code_only(void) {
  return hyle::detail::status_to_ffi(hyle::Status::ResourceExhausted());
}

HYLE_ERROR_FFI_FIXTURE_API hyle_status HYLE_ERROR_FFI_FIXTURE_CALL
hyle_error_status_fixture_message_native(void) {
  static constexpr char kMessage[] = {'d', 'e', 'v', '\0', 'l', 'o', 's', 't'};
  return hyle::detail::status_to_ffi(hyle::Status::FromNative(
      hyle::StatusCode::Unavailable, {hyle::NativeErrorDomain::Vulkan, -4},
      std::string_view(kMessage, sizeof(kMessage))));
}

} // extern "C"
