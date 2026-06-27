#pragma once

// Symbol visibility: tri-state export/import, used by per-module HYLE_<MOD>_API.
// Fallback to empty on unknown compilers (degrade safe, never fail to compile).

#if defined(_WIN32) || defined(__CYGWIN__)
#  define HYLE_EXPORT __declspec(dllexport)
#  define HYLE_IMPORT __declspec(dllimport)
#  define HYLE_HIDDEN
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_EXPORT __attribute__((visibility("default")))
#  define HYLE_IMPORT __attribute__((visibility("default")))
#  define HYLE_HIDDEN __attribute__((visibility("hidden")))
#else
#  define HYLE_EXPORT
#  define HYLE_IMPORT
#  define HYLE_HIDDEN
#endif
