#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define HYLE_CORE_FFI_FIXTURE_API __declspec(dllexport)
#define HYLE_CORE_FFI_FIXTURE_CALL __cdecl
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_CORE_FFI_FIXTURE_API __attribute__((visibility("default")))
#define HYLE_CORE_FFI_FIXTURE_CALL
#else
#define HYLE_CORE_FFI_FIXTURE_API
#define HYLE_CORE_FFI_FIXTURE_CALL
#endif

/* Deliberately incompatible and deliberately the fixture's only export.  A
 * correct binding rejects this ABI major before resolving product symbols. */
HYLE_CORE_FFI_FIXTURE_API uint32_t HYLE_CORE_FFI_FIXTURE_CALL
hyle_ffi_abi_version(void) {
  return 0x00020000u;
}
