#ifndef HYLE_CORE_VERSION_DEFS_H_
#define HYLE_CORE_VERSION_DEFS_H_

// Public compile-time version of the headers. The hyle_core runtime exposes the
// version it was built with separately, so a plugin/FFI loader can diagnose a
// header/runtime mismatch without turning this pre-release API into a versioned
// function table.
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

#endif  // HYLE_CORE_VERSION_DEFS_H_
