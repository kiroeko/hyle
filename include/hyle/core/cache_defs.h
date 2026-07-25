#ifndef HYLE_CORE_CACHE_DEFS_H_
#define HYLE_CORE_CACHE_DEFS_H_

// Stable compile-time spacing for objects whose layout must prevent false
// sharing. This is deliberately a preprocessor constant so C and C++ see the
// same layout. Apple Silicon has 128-byte L1 data-cache lines; supported x86
// and other ARM64 targets use 64-byte lines today.
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
#  define HYLE_STATIC_CACHE_LINE_SIZE 128
#else
#  define HYLE_STATIC_CACHE_LINE_SIZE 64
#endif

#endif /* HYLE_CORE_CACHE_DEFS_H_ */
