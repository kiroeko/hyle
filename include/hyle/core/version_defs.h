#ifndef HYLE_CORE_VERSION_DEFS_H_
#define HYLE_CORE_VERSION_DEFS_H_

// Public compile-time product version of the headers. This identifies the HYLE
// release; it is deliberately independent from the Native FFI ABI version
// below. Managed bindings must check both before resolving the rest of their
// required symbols.
#define HYLE_VERSION_MAJOR 0
#define HYLE_VERSION_MINOR 1
#define HYLE_VERSION_PATCH 0

#define HYLE_VERSION_PACK(major, minor, patch) \
  ((((major) & 0xFFu) << 16) | (((minor) & 0xFFu) << 8) | ((patch) & 0xFFu))
#define HYLE_VERSION_PACKED \
  HYLE_VERSION_PACK(HYLE_VERSION_MAJOR, HYLE_VERSION_MINOR, HYLE_VERSION_PATCH)

#define HYLE_VERSION_STRINGIZE_INNER(value) #value
#define HYLE_VERSION_STRINGIZE(value) HYLE_VERSION_STRINGIZE_INNER(value)
#define HYLE_VERSION_STRING                  \
  HYLE_VERSION_STRINGIZE(HYLE_VERSION_MAJOR) \
  "." HYLE_VERSION_STRINGIZE(HYLE_VERSION_MINOR) "." HYLE_VERSION_STRINGIZE(HYLE_VERSION_PATCH)

// Stable contract between hyle native libraries and separately distributed
// Java/.NET/Rust/Python bindings. A major change may remove or alter symbols;
// a minor change is additive. Bindings accept the same major and a runtime
// minor greater than or equal to the minor they were generated against.
#define HYLE_FFI_ABI_VERSION_MAJOR 1
#define HYLE_FFI_ABI_VERSION_MINOR 0
#define HYLE_FFI_ABI_VERSION_PACK(major, minor) \
  ((((major) & 0xFFFFu) << 16) | ((minor) & 0xFFFFu))
#define HYLE_FFI_ABI_VERSION \
  HYLE_FFI_ABI_VERSION_PACK(HYLE_FFI_ABI_VERSION_MAJOR, \
                            HYLE_FFI_ABI_VERSION_MINOR)

#endif  // HYLE_CORE_VERSION_DEFS_H_
