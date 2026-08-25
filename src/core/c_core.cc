// Native FFI bridge for the core module (include/hyle/c/core.h). Lives in
// hyle_core so HYLE_CORE_BUILD exports current unmangled candidate symbols for
// managed and other foreign-language bindings. Compatibility begins only when
// a binding product is released.
#include "hyle/c/core.h"
#include "hyle/core/version.h"

extern "C" {

uint32_t HYLE_CORE_C_CALL hyle_ffi_abi_version(void) {
  return HYLE_FFI_ABI_VERSION;
}

uint32_t HYLE_CORE_C_CALL hyle_version(void) { return hyle::version(); }

const char* HYLE_CORE_C_CALL hyle_version_string(void) {
  return hyle::version_string();
}

}  // extern "C"
