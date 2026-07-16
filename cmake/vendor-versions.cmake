# Pinned benchmark rivals for simd-20 vendor extraction.
# Keep these revisions independent of the copies temporarily consumed by product
# sources: benchmark rows are the unchanged external reference during each port.
set(HYLE_BENCH_VENDOR_HIGHWAY_REPOSITORY "https://github.com/google/highway.git")
set(HYLE_BENCH_VENDOR_HIGHWAY_REV "83eff39371e5e6494b305b253f8a3c06f831faf4")
set(HYLE_BENCH_VENDOR_XSIMD_REPOSITORY "https://github.com/xtensor-stack/xsimd.git")
set(HYLE_BENCH_VENDOR_XSIMD_REV "5d2490f515cf773f984da9ec0a1aec103d00d238")

# The shell/PowerShell fetchers use this machine-readable manifest so the pin has
# one source of truth. It is intentionally inert during normal CMake configuration.
if(DEFINED HYLE_BENCH_VENDOR_OUTPUT_FILE)
  file(WRITE "${HYLE_BENCH_VENDOR_OUTPUT_FILE}"
    "highway|${HYLE_BENCH_VENDOR_HIGHWAY_REPOSITORY}|${HYLE_BENCH_VENDOR_HIGHWAY_REV}\n"
    "xsimd|${HYLE_BENCH_VENDOR_XSIMD_REPOSITORY}|${HYLE_BENCH_VENDOR_XSIMD_REV}\n")
endif()
