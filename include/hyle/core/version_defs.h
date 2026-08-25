#ifndef HYLE_CORE_VERSION_DEFS_H_
#define HYLE_CORE_VERSION_DEFS_H_

// Public compile-time product version of the headers. This identifies the HYLE
// release; it is deliberately independent from the Native FFI ABI version
// below. Managed bindings may use it for diagnostics after the FFI ABI
// handshake; it does not participate in compatibility negotiation.
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

// Stable contract between HYLE native libraries and separately distributed
// Java/.NET/Rust/Python bindings. A major change may remove or alter ordinary
// ABI symbols; a minor change is additive. HYLE has not released a binding
// product yet, so this pre-release protocol may still change without backward
// compatibility. Once a binding is published, the bootstrap symbol it uses,
// its C calling convention, and its uint32_t(void) signature remain available
// so that old bindings can reject incompatible runtimes safely. Bindings accept
// the same major and a runtime minor at least as new as their required minor.
#define HYLE_FFI_ABI_VERSION_MAJOR 1
#define HYLE_FFI_ABI_VERSION_MINOR 0
#define HYLE_FFI_ABI_VERSION_PACK(major, minor) \
  ((((major) & 0xFFFFu) << 16) | ((minor) & 0xFFFFu))
#define HYLE_FFI_ABI_VERSION \
  HYLE_FFI_ABI_VERSION_PACK(HYLE_FFI_ABI_VERSION_MAJOR, \
                            HYLE_FFI_ABI_VERSION_MINOR)

#endif  // HYLE_CORE_VERSION_DEFS_H_
