// C-ABI bridge for the core module (include/hyle/c/core.h). Lives in hyle_core
// so HYLE_CORE_BUILD exports the symbols. Pure forwarding, no extra policy.
#include "hyle/c/core.h"
#include "hyle/core/version.h"

extern "C" {

uint32_t hyle_version(void) { return hyle::version(); }

const char* hyle_version_string(void) { return hyle::version_string(); }

}  // extern "C"
