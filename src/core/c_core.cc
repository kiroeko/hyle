// C-ABI bridge for the core module (include/hyle/c/core.h). Lives in hyle_core
// so HYLE_CORE_BUILD exports the symbols. Pure forwarding, no extra policy.
#include "hyle/c/core.h"

#include <string>

#include "hyle/core/version.h"

extern "C" {

uint32_t hyle_version(void) { return hyle::version(); }

const char* hyle_version_string(void) {
  // Built once from the packed version; static storage lives for the process so
  // the returned pointer stays valid with no per-call allocation.
  static const std::string s = [] {
    uint32_t v = hyle::version();
    return std::to_string((v >> 16) & 0xFFu) + "." +
           std::to_string((v >> 8) & 0xFFu) + "." + std::to_string(v & 0xFFu);
  }();
  return s.c_str();
}

}  // extern "C"
