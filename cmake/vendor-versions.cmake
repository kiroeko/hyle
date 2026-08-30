# Compatibility projection of the central dependency ledger for the standalone
# benchmark fetchers (simd-20 X4+).
# Highway and xsimd are locked product sources. Benchmarks still use clean sibling
# checkouts at the same pins so candidate and peer include paths cannot accidentally
# collapse into one implementation.
include("${CMAKE_CURRENT_LIST_DIR}/HyleDependencyVersions.cmake")
set(HYLE_BENCH_VENDOR_HIGHWAY_REPOSITORY "${HYLE_DEP_HIGHWAY_REPOSITORY}")
set(HYLE_BENCH_VENDOR_HIGHWAY_REV "${HYLE_DEP_HIGHWAY_COMMIT}")
set(HYLE_BENCH_VENDOR_XSIMD_REPOSITORY "${HYLE_DEP_XSIMD_REPOSITORY}")
set(HYLE_BENCH_VENDOR_XSIMD_REV "${HYLE_DEP_XSIMD_COMMIT}")

# The shell/PowerShell fetchers use this machine-readable manifest so the pin has
# one source of truth. It is intentionally inert during normal CMake configuration.
if(DEFINED HYLE_BENCH_VENDOR_OUTPUT_FILE)
  file(WRITE "${HYLE_BENCH_VENDOR_OUTPUT_FILE}"
    "highway|${HYLE_BENCH_VENDOR_HIGHWAY_REPOSITORY}|${HYLE_BENCH_VENDOR_HIGHWAY_REV}\n"
    "xsimd|${HYLE_BENCH_VENDOR_XSIMD_REPOSITORY}|${HYLE_BENCH_VENDOR_XSIMD_REV}\n")
endif()

# Fetch helpers call back into the same lock after checkout so a matching HEAD
# alone cannot hide a wrong root tree, dirty/untracked files or license drift.
if(DEFINED HYLE_VERIFY_BENCH_DEPENDENCY_NAME OR
   DEFINED HYLE_VERIFY_BENCH_DEPENDENCY_DIR)
  if(NOT DEFINED HYLE_VERIFY_BENCH_DEPENDENCY_NAME OR
     NOT DEFINED HYLE_VERIFY_BENCH_DEPENDENCY_DIR)
    message(FATAL_ERROR
      "Pass both HYLE_VERIFY_BENCH_DEPENDENCY_NAME and "
      "HYLE_VERIFY_BENCH_DEPENDENCY_DIR")
  endif()
  if(NOT HYLE_VERIFY_BENCH_DEPENDENCY_NAME STREQUAL "highway" AND
     NOT HYLE_VERIFY_BENCH_DEPENDENCY_NAME STREQUAL "xsimd")
    message(FATAL_ERROR
      "'${HYLE_VERIFY_BENCH_DEPENDENCY_NAME}' is not a SIMD benchmark source")
  endif()
  hyle_verify_locked_git_checkout(
    "${HYLE_VERIFY_BENCH_DEPENDENCY_NAME}"
    "${HYLE_VERIFY_BENCH_DEPENDENCY_DIR}")
  message(STATUS
    "HYLE verified benchmark checkout: "
    "${HYLE_VERIFY_BENCH_DEPENDENCY_NAME}")
endif()
