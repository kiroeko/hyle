#!/bin/bash
# rvv-qemu-test.sh -- cross-compile the RVV backend gtest (tests/simd/rvv_test.cc) and
# run it under qemu-riscv64 at VLEN 128/256/512/1024.
#
# Tiers:
#   base  (default) : -march=rv64gcv            (existing P0-P5 surface)
#   zvbb            : -march=rv64gcv_zvbb       (simd-16 LeadingZero/TrailingZero/ReverseBits/Rotate)
#   zvbc            : -march=rv64gcv_zvbc       (simd-16 CLMulLower/Upper)
#   zvkned          : -march=rv64gcv_zvkned     (native AES round operations)
#   all             : run base + zvbb + zvbc + zvkned
#
# Usage:
#   bash scripts/rvv-qemu-test.sh            # base only
#   bash scripts/rvv-qemu-test.sh zvbb
#   bash scripts/rvv-qemu-test.sh zvbc
#   bash scripts/rvv-qemu-test.sh zvkned
#   bash scripts/rvv-qemu-test.sh all
#   bash scripts/rvv-qemu-test.sh fat    # direct-Highway source-fat seam only
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GT=${GTEST_SOURCE_DIR:-"$REPO/.deps/googletest-src/googletest"}
if [ ! -f "$GT/src/gtest-all.cc" ]; then
  echo "ERROR: gtest source not found at $GT (configure the CMake build once to fetch it)"; exit 1
fi

CXX=${CXX_RISCV64:-riscv64-linux-gnu-g++-14}
QEMU=${QEMU_RISCV64:-qemu-riscv64-static}
OBJDUMP=${OBJDUMP_RISCV64:-riscv64-linux-gnu-objdump}
OUT=${OUT_DIR:-/tmp/hyle-rvv}
mkdir -p "$OUT"
HWY=${HIGHWAY_SOURCE_DIR:-"$REPO/../highway"}
if [ ! -f "$HWY/hwy/highway.h" ]; then
  echo "ERROR: locked Highway source not found at $HWY (set HIGHWAY_SOURCE_DIR)"; exit 1
fi

TIER=${1:-base}
rc=0

run_fat_fixture() {
  local common_flags="-std=c++23 -O2 -march=rv64gc -pthread \
    -I$REPO/include -I$REPO -I$HWY"
  local flags="$common_flags -march=rv64gcv -DHYLE_SIMD_TARGET_RVV"
  local bin="$OUT/fat_rvv_only"

  echo "[fat] compiling direct-Highway RVV source-fat fixture ..."
  printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX rvv\n#include "tests/simd/fat_kernel_fixture.h"\n' |
    $CXX $flags -x c++ -c - -o "$OUT/fat_rvv_only.o"
  $CXX $common_flags -DHYLE_TEST_FAT_RVV=1 \
    -DHYLE_SIMD_FAT_EXACT_VARIANT_RVV=1 -DHYLE_SIMD_EXACT_TARGET_RVV=1 \
    -c "$REPO/tests/simd/fat_kernel_scalable_driver.cc" \
    -o "$OUT/fat_driver_rvv.o"
  $CXX $flags "$OUT/fat_driver_rvv.o" "$OUT/fat_rvv_only.o" \
    -static -o "$bin" -pthread
  for bits in 128 256 512 1024; do
    echo "----- direct Highway RVV VLEN=${bits}-bit -----"
    "$QEMU" -cpu "rv64,v=true,vlen=${bits},elen=64" \
      -L /usr/riscv64-linux-gnu "$bin"
  done
}

run_tier() {
  local name="$1"
  local march="$2"
  local qemu_cpu="$3"
  local bin="$OUT/rvv_gtest_${name}"

  local COMMON_FLAGS="-std=c++23 -O2 -march=rv64gc -DHYLE_SIMD_HAS_RVV_VARIANT=1 -pthread \
    -I$REPO/include -I$REPO -I$HWY -I$GT/include -I$GT"
  local FLAGS="$COMMON_FLAGS -march=${march} -DHYLE_SIMD_TARGET_RVV"

  echo "[${name}] cross-compiling register + product-math probes (-march=${march}) ..."
  $CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all_${name}.o"
  $CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/rvv_test.cc" -o "$OUT/rvv_test_${name}.o"
  $OBJDUMP -d "$OUT/rvv_test_${name}.o" > "$OUT/rvv_test_${name}.dis"
  if [ "$name" = "zvkned" ]; then
    for op in vaesem vaesef vaesdm vaesdf; do
      if ! grep -q "${op}\\.vv" "$OUT/rvv_test_${name}.dis"; then
        echo "ERROR: Zvkned object did not emit ${op}.vv"; exit 1
      fi
    done
  elif grep -Eq 'vaes(em|ef|dm|df)\.vv' "$OUT/rvv_test_${name}.dis"; then
    echo "ERROR: optional Zvkned instruction leaked into ${name} object"; exit 1
  fi
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_math_product_test.cc" -o "$OUT/scalable_math_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_dtype_product_test.cc" -o "$OUT/scalable_dtype_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/narrow_float_test.cc" -o "$OUT/narrow_float_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_bitpack_product_test.cc" -o "$OUT/scalable_bitpack_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_sort_product_test.cc" -o "$OUT/scalable_sort_product_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/dispatch_test.cc" -o "$OUT/dispatch_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/runtime_test.cc" -o "$OUT/runtime_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/ops_test.cc" -o "$OUT/ops_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/search_test.cc" -o "$OUT/search_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/narrow_ops_test.cc" -o "$OUT/narrow_ops_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/scan_test.cc" -o "$OUT/scan_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/fixed128_bitpack_reference.cc" -o "$OUT/fixed128_bitpack_reference_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/math/transcendental_highway_scalable.cc" -o "$OUT/math_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/dtype/cvt_scalable.cc" -o "$OUT/dtype_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/bitpack/bitpack_hyle_scalable.cc" -o "$OUT/bitpack_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/sort/vqsort_hyle_rvv.cc" -o "$OUT/sort_rvv_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/scalable_math_dispatch_test.cc" -o "$OUT/scalable_math_dispatch_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/platform/cpu.cc" -o "$OUT/cpu_${name}.o"
  $CXX $COMMON_FLAGS -DHYLE_ERROR_BUILD -c "$REPO/src/error/status.cc" -o "$OUT/error_status_${name}.o"
  printf '#include "hyle/platform/thread.h"\nnamespace hyle::platform { unsigned hardware_concurrency() noexcept { return 1; } }\n' |
    $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/thread_test_stub_${name}.o"
  printf '#define HWY_COMPILE_ONLY_SCALAR 1\n#define HYLE_SIMD_VARIANT_SUFFIX scalar\n#include "tests/simd/fat_kernel_fixture.h"\n' |
    $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/fat_scalar_${name}.o"
  printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX rvv\n#include "tests/simd/fat_kernel_fixture.h"\n' |
    $CXX $FLAGS -x c++ -c - -o "$OUT/fat_rvv_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/runtime.cc" -o "$OUT/runtime_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/math/math_dispatch.cc" -o "$OUT/math_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/math/transcendental_scalar.cc" -o "$OUT/math_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/dtype_dispatch.cc" -o "$OUT/dtype_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/cvt_scalar.cc" -o "$OUT/dtype_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_dispatch.cc" -o "$OUT/bitpack_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_scalar.cc" -o "$OUT/bitpack_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/ops/dispatch_ops.cc" -o "$OUT/ops_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/random/random_dispatch.cc" -o "$OUT/random_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/search/search_dispatch.cc" -o "$OUT/search_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/narrow/narrow_dispatch.cc" -o "$OUT/narrow_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/scan/scan_dispatch.cc" -o "$OUT/scan_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/sort_dispatch.cc" -o "$OUT/sort_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/vqsort_scalar.cc" -o "$OUT/sort_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/radix_sort.cc" -o "$OUT/radix_sort_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/stream_topk.cc" -o "$OUT/stream_topk_${name}.o"
  for module in ops random scan search narrow; do
    printf '#define HYLE_SIMD_VARIANT_SUFFIX scalar\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
      $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/${module}_scalar_${name}.o"
    printf '#define HYLE_SIMD_VARIANT_SUFFIX rvv\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
      $CXX $FLAGS -x c++ -c - -o "$OUT/${module}_rvv_${name}.o"
  done
  $CXX $FLAGS "$OUT/rvv_test_${name}.o" \
    "$OUT/scalable_math_product_test_${name}.o" "$OUT/scalable_math_dispatch_test_${name}.o" \
    "$OUT/scalable_dtype_product_test_${name}.o" \
    "$OUT/narrow_float_test_${name}.o" \
    "$OUT/scalable_bitpack_product_test_${name}.o" \
    "$OUT/scalable_sort_product_test_${name}.o" \
    "$OUT/dispatch_test_${name}.o" \
    "$OUT/runtime_test_${name}.o" \
    "$OUT/ops_test_${name}.o" \
    "$OUT/search_test_${name}.o" \
    "$OUT/narrow_ops_test_${name}.o" \
    "$OUT/scan_test_${name}.o" \
    "$OUT/fixed128_bitpack_reference_${name}.o" \
    "$OUT/math_rvv_${name}.o" "$OUT/math_scalar_${name}.o" "$OUT/math_dispatch_${name}.o" \
    "$OUT/dtype_rvv_${name}.o" "$OUT/dtype_scalar_${name}.o" "$OUT/dtype_dispatch_${name}.o" \
    "$OUT/bitpack_rvv_${name}.o" "$OUT/bitpack_scalar_${name}.o" "$OUT/bitpack_dispatch_${name}.o" \
    "$OUT/scan_rvv_${name}.o" "$OUT/scan_scalar_${name}.o" "$OUT/scan_dispatch_${name}.o" \
    "$OUT/sort_rvv_${name}.o" "$OUT/sort_scalar_${name}.o" "$OUT/sort_dispatch_${name}.o" \
    "$OUT/radix_sort_${name}.o" "$OUT/stream_topk_${name}.o" \
    "$OUT/cpu_${name}.o" "$OUT/error_status_${name}.o" "$OUT/thread_test_stub_${name}.o" \
    "$OUT/fat_scalar_${name}.o" "$OUT/fat_rvv_${name}.o" "$OUT/runtime_${name}.o" \
    "$OUT/ops_scalar_${name}.o" "$OUT/ops_rvv_${name}.o" "$OUT/ops_dispatch_${name}.o" \
    "$OUT/random_scalar_${name}.o" "$OUT/random_rvv_${name}.o" "$OUT/random_dispatch_${name}.o" \
    "$OUT/search_scalar_${name}.o" "$OUT/search_rvv_${name}.o" "$OUT/search_dispatch_${name}.o" \
    "$OUT/narrow_scalar_${name}.o" "$OUT/narrow_rvv_${name}.o" "$OUT/narrow_dispatch_${name}.o" \
    "$OUT/gtest-all_${name}.o" "$OUT/gtest-main_${name}.o" \
    -static -o "$bin" -pthread

  echo "[${name}] running under qemu-riscv64 at VLEN 128 / 256 / 512 / 1024 (cpu=${qemu_cpu}) ..."
  local feature_env=(HYLE_EXPECT_RVV=1)
  # qemu-linux-user 8.2 executes Zvbb/Zvbc instructions but does not emulate the
  # riscv_hwprobe optional-extension result (the syscall safely reports no bit).
  # A newer emulator or a full-system guest kernel may opt into the positive
  # detector assertion; real Linux uses the same official hwprobe ABI.
  if [ "${HYLE_QEMU_EXPECT_OPTIONAL_FEATURES:-0}" = "1" ]; then
    if [ "$name" = "zvbb" ]; then feature_env+=(HYLE_EXPECT_RVV_ZVBB=1); fi
    if [ "$name" = "zvbc" ]; then feature_env+=(HYLE_EXPECT_RVV_ZVBC=1); fi
    if [ "$name" = "zvkned" ]; then feature_env+=(HYLE_EXPECT_RVV_ZVKNED=1); fi
  fi
  for bits in 128 256 512 1024; do
    echo "----- RVV ${name} VLEN=${bits}-bit -----"
    env "${feature_env[@]}" "$QEMU" \
      -cpu "rv64,v=true,${qemu_cpu}vlen=${bits},elen=64" -L /usr/riscv64-linux-gnu \
      "$bin" || rc=$?
  done
}

case "$TIER" in
  fat)
    run_fat_fixture
    ;;
  base)
    run_tier base rv64gcv ""
    ;;
  zvbb)
    run_tier zvbb rv64gcv_zvbb "zvbb=true,"
    ;;
  zvbc)
    run_tier zvbc rv64gcv_zvbc "zvbc=true,"
    ;;
  zvkned)
    run_tier zvkned rv64gcv_zvkned "zvkned=true,"
    ;;
  all)
    run_tier base rv64gcv ""
    run_tier zvbb rv64gcv_zvbb "zvbb=true,"
    run_tier zvbc rv64gcv_zvbc "zvbc=true,"
    run_tier zvkned rv64gcv_zvkned "zvkned=true,"
    ;;
  *)
    echo "Usage: $0 [base|zvbb|zvbc|zvkned|all|fat]"; exit 2
    ;;
esac

echo "done (exit=$rc)"
exit $rc
