#!/bin/bash
# sve-qemu-test.sh -- cross-compile the SVE backend gtest (tests/simd/sve_test.cc) for aarch64 and
# run it under qemu-aarch64 at multiple SVE vector lengths (128/256/384/512/2048-bit). This is the "one
# binary, every width" scalable proof: SVE's vector length is a runtime property, so the SAME
# binary must pass at every VL. There is no SVE dev machine in this project, so qemu's -cpu
# max,sve<bits>=on is the correctness gate (analogous to Intel SDE for AVX-512).
#
# Prereqs (WSL Ubuntu, already installed here):
#   aarch64-linux-gnu-g++-14, qemu-aarch64-static, gtest source under .deps/googletest-src.
# Usage: run from the repo root on a Linux box with the aarch64 cross toolchain:
#   bash scripts/sve-qemu-test.sh
#
# On a REAL SVE host (Graviton3, etc.) you don't need this -- configure with
# -DHYLE_TEST_SIMD_SVE=ON
# and `ctest -R simd_sve` runs it natively.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GT=${GTEST_SOURCE_DIR:-"$REPO/.deps/googletest-src/googletest"}
if [ ! -f "$GT/src/gtest-all.cc" ]; then
  echo "ERROR: gtest source not found at $GT (configure the CMake build once to fetch it)"; exit 1
fi

CXX=${CXX_AARCH64:-aarch64-linux-gnu-g++-14}
QEMU=${QEMU_AARCH64:-qemu-aarch64-static}
OUT=${OUT_DIR:-/tmp/hyle-sve}
mkdir -p "$OUT"
HWY=${HIGHWAY_SOURCE_DIR:-"$REPO/../highway"}
if [ ! -f "$HWY/hwy/highway.h" ]; then
  echo "ERROR: locked Highway source not found at $HWY (set HIGHWAY_SOURCE_DIR)"; exit 1
fi

# tier: base (armv8-a+sve, default) or crypto (armv8-a+sve2-aes, unlocks the SVE2 AES/CLMul test
# guarded by __ARM_FEATURE_SVE2_AES). qemu -cpu max enables sve2-aes at every VL.
TIER=${1:-base}
ARCH="armv8-a+sve"
if [ "$TIER" = "crypto" ]; then ARCH="armv8-a+sve2-aes"; fi

COMMON_FLAGS="-std=c++23 -O2 -march=armv8-a -DHYLE_SIMD_HAS_SVE_VARIANT=1 -pthread \
  -I$REPO/include -I$REPO -I$HWY -I$GT/include -I$GT"
FLAGS="$COMMON_FLAGS -march=$ARCH -DHYLE_SIMD_TARGET_SVE"

echo "[1/3] cross-compiling register + product-math probes for aarch64 ($ARCH) ..."
$CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all.o"
$CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main.o"
$CXX $FLAGS -c "$REPO/tests/simd/sve_test.cc" -o "$OUT/sve_test.o"
$CXX $FLAGS -c "$REPO/tests/simd/scalable_math_product_test.cc" -o "$OUT/scalable_math_product_test.o"
$CXX $FLAGS -c "$REPO/tests/simd/scalable_dtype_product_test.cc" -o "$OUT/scalable_dtype_product_test.o"
$CXX $FLAGS -c "$REPO/tests/simd/narrow_float_test.cc" -o "$OUT/narrow_float_test.o"
$CXX $FLAGS -c "$REPO/tests/simd/scalable_bitpack_product_test.cc" -o "$OUT/scalable_bitpack_product_test.o"
$CXX $FLAGS -c "$REPO/tests/simd/scalable_sort_product_test.cc" -o "$OUT/scalable_sort_product_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/dispatch_test.cc" -o "$OUT/dispatch_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/runtime_test.cc" -o "$OUT/runtime_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/ops_test.cc" -o "$OUT/ops_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/search_test.cc" -o "$OUT/search_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/narrow_ops_test.cc" -o "$OUT/narrow_ops_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/scan_test.cc" -o "$OUT/scan_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/fixed128_bitpack_reference.cc" -o "$OUT/fixed128_bitpack_reference.o"
$CXX $FLAGS -c "$REPO/src/simd/math/transcendental_highway_scalable.cc" -o "$OUT/math_sve.o"
$CXX $FLAGS -c "$REPO/src/simd/dtype/cvt_scalable.cc" -o "$OUT/dtype_sve.o"
$CXX $FLAGS -c "$REPO/src/simd/bitpack/bitpack_hyle_scalable.cc" -o "$OUT/bitpack_sve.o"
$CXX $FLAGS -c "$REPO/src/simd/sort/vqsort_hyle_sve.cc" -o "$OUT/sort_sve.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/scalable_math_dispatch_test.cc" -o "$OUT/scalable_math_dispatch_test.o"
$CXX $COMMON_FLAGS -c "$REPO/tests/simd/neon_math_product_test.cc" -o "$OUT/neon_math_product_test.o"
$CXX $COMMON_FLAGS -c "$REPO/src/platform/cpu.cc" -o "$OUT/cpu.o"
$CXX $COMMON_FLAGS -DHYLE_ERROR_BUILD -c "$REPO/src/error/status.cc" -o "$OUT/error_status.o"
printf '#include "hyle/platform/thread.h"\nnamespace hyle::platform { unsigned hardware_concurrency() noexcept { return 1; } }\n' |
  $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/thread_test_stub.o"
printf '#define HYLE_SIMD_VARIANT_SUFFIX scalar\n#include "tests/simd/fat_kernel_fixture.h"\n' |
  $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/fat_scalar.o"
printf '#define HYLE_SIMD_TARGET_NEON\n#define HYLE_SIMD_VARIANT_SUFFIX neon\n#include "tests/simd/fat_kernel_fixture.h"\n' |
  $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/fat_neon.o"
printf '#define HYLE_SIMD_VARIANT_SUFFIX sve\n#include "tests/simd/fat_kernel_fixture.h"\n' |
  $CXX $FLAGS -x c++ -c - -o "$OUT/fat_sve.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/runtime.cc" -o "$OUT/runtime.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/math/math_dispatch.cc" -o "$OUT/math_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/math/transcendental_scalar.cc" -o "$OUT/math_scalar.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/math/transcendental_highway_neon.cc" -o "$OUT/math_neon.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/dtype_dispatch.cc" -o "$OUT/dtype_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/cvt_scalar.cc" -o "$OUT/dtype_scalar.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/cvt_neon.cc" -o "$OUT/dtype_neon.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_dispatch.cc" -o "$OUT/bitpack_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_scalar.cc" -o "$OUT/bitpack_scalar.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_hyle_neon.cc" -o "$OUT/bitpack_neon.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/ops/dispatch_ops.cc" -o "$OUT/ops_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/random/random_dispatch.cc" -o "$OUT/random_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/search/search_dispatch.cc" -o "$OUT/search_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/narrow/narrow_dispatch.cc" -o "$OUT/narrow_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/scan/scan_dispatch.cc" -o "$OUT/scan_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/sort_dispatch.cc" -o "$OUT/sort_dispatch.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/vqsort_scalar.cc" -o "$OUT/sort_scalar.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/vqsort_hyle_neon.cc" -o "$OUT/sort_neon.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/radix_sort.cc" -o "$OUT/radix_sort.o"
$CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/stream_topk.cc" -o "$OUT/stream_topk.o"
for module in ops random scan search narrow; do
  printf '#define HYLE_SIMD_VARIANT_SUFFIX scalar\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
    $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/${module}_scalar.o"
  printf '#define HYLE_SIMD_TARGET_NEON\n#define HYLE_SIMD_VARIANT_SUFFIX neon\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
    $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/${module}_neon.o"
  printf '#define HYLE_SIMD_VARIANT_SUFFIX sve\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
    $CXX $FLAGS -x c++ -c - -o "$OUT/${module}_sve.o"
done
$CXX $FLAGS "$OUT/sve_test.o" \
  "$OUT/scalable_math_product_test.o" "$OUT/scalable_math_dispatch_test.o" \
  "$OUT/neon_math_product_test.o" \
  "$OUT/scalable_dtype_product_test.o" \
  "$OUT/narrow_float_test.o" \
  "$OUT/scalable_bitpack_product_test.o" \
  "$OUT/scalable_sort_product_test.o" \
  "$OUT/dispatch_test.o" \
  "$OUT/runtime_test.o" \
  "$OUT/ops_test.o" \
  "$OUT/search_test.o" \
  "$OUT/narrow_ops_test.o" \
  "$OUT/scan_test.o" \
  "$OUT/fixed128_bitpack_reference.o" \
  "$OUT/math_sve.o" "$OUT/math_neon.o" \
  "$OUT/math_scalar.o" "$OUT/math_dispatch.o" "$OUT/cpu.o" "$OUT/error_status.o" \
  "$OUT/thread_test_stub.o" "$OUT/fat_scalar.o" "$OUT/fat_neon.o" "$OUT/fat_sve.o" \
  "$OUT/runtime.o" \
  "$OUT/dtype_sve.o" "$OUT/dtype_neon.o" "$OUT/dtype_scalar.o" "$OUT/dtype_dispatch.o" \
  "$OUT/bitpack_sve.o" "$OUT/bitpack_neon.o" "$OUT/bitpack_scalar.o" "$OUT/bitpack_dispatch.o" \
  "$OUT/scan_scalar.o" "$OUT/scan_neon.o" "$OUT/scan_sve.o" "$OUT/scan_dispatch.o" \
  "$OUT/sort_sve.o" "$OUT/sort_neon.o" "$OUT/sort_scalar.o" "$OUT/sort_dispatch.o" \
  "$OUT/radix_sort.o" "$OUT/stream_topk.o" \
  "$OUT/ops_scalar.o" "$OUT/ops_neon.o" "$OUT/ops_sve.o" "$OUT/ops_dispatch.o" \
  "$OUT/random_scalar.o" "$OUT/random_neon.o" "$OUT/random_sve.o" "$OUT/random_dispatch.o" \
  "$OUT/search_scalar.o" "$OUT/search_neon.o" "$OUT/search_sve.o" "$OUT/search_dispatch.o" \
  "$OUT/narrow_scalar.o" "$OUT/narrow_neon.o" "$OUT/narrow_sve.o" "$OUT/narrow_dispatch.o" \
  "$OUT/gtest-all.o" \
  "$OUT/gtest-main.o" -static -o "$OUT/sve_gtest" -pthread

echo "[2/3] running under qemu at VL 128 / 256 / 384 / 512 / 2048 ..."
rc=0
FEATURE_ENV=(HYLE_EXPECT_SVE=1)
if [ "$TIER" = "crypto" ]; then
  FEATURE_ENV+=(HYLE_EXPECT_SVE2=1 HYLE_EXPECT_SVE_AES=1 HYLE_EXPECT_SVE_PMULL=1)
fi
for bits in 128 256 384 512 2048; do
  echo "----- SVE VL=${bits}-bit -----"
  env "${FEATURE_ENV[@]}" HYLE_EXPECT_SVE_BITS="$bits" \
    "$QEMU" -cpu "max,sve${bits}=on" -L /usr/aarch64-linux-gnu "$OUT/sve_gtest" || rc=$?
done

echo "[3/3] done (exit=$rc)"
exit $rc
