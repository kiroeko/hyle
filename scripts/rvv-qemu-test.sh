#!/bin/bash
# rvv-qemu-test.sh -- cross-compile the RVV backend gtest (tests/simd/rvv_test.cc) and
# run it under qemu-riscv64 at VLEN 128/256/512/1024.
#
# Tiers:
#   base  (default) : -march=rv64gcv            (existing P0-P5 surface)
#   zvbb            : -march=rv64gcv_zvbb       (simd-16 LeadingZero/TrailingZero/ReverseBits/Rotate)
#   zvbc            : -march=rv64gcv_zvbc       (simd-16 CLMulLower/Upper)
#   all             : run base + zvbb + zvbc
#
# Usage:
#   bash scripts/rvv-qemu-test.sh            # base only
#   bash scripts/rvv-qemu-test.sh zvbb
#   bash scripts/rvv-qemu-test.sh zvbc
#   bash scripts/rvv-qemu-test.sh all
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GT="$REPO/.deps/googletest-src/googletest"
if [ ! -f "$GT/src/gtest-all.cc" ]; then
  echo "ERROR: gtest source not found at $GT (configure the CMake build once to fetch it)"; exit 1
fi

CXX=${CXX_RISCV64:-riscv64-linux-gnu-g++-14}
QEMU=${QEMU_RISCV64:-qemu-riscv64-static}
OUT=${OUT_DIR:-/tmp/hyle-rvv}
mkdir -p "$OUT"

TIER=${1:-base}
rc=0

run_tier() {
  local name="$1"
  local march="$2"
  local qemu_cpu="$3"
  local bin="$OUT/rvv_gtest_${name}"

  local COMMON_FLAGS="-std=c++23 -O2 -march=rv64gc -DHYLE_SIMD_HAS_RVV_VARIANT=1 -pthread \
    -I$REPO/include -I$REPO -I$GT/include -I$GT"
  local FLAGS="$COMMON_FLAGS -march=${march} -DHYLE_SIMD_TARGET_RVV"

  echo "[${name}] cross-compiling register + product-math probes (-march=${march}) ..."
  $CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all_${name}.o"
  $CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/rvv_test.cc" -o "$OUT/rvv_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_math_compile.cc" -o "$OUT/scalable_math_compile_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_math_product_test.cc" -o "$OUT/scalable_math_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_dtype_product_test.cc" -o "$OUT/scalable_dtype_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_bitpack_product_test.cc" -o "$OUT/scalable_bitpack_product_test_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/scalable_sort_product_test.cc" -o "$OUT/scalable_sort_product_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/fixed128_bitpack_reference.cc" -o "$OUT/fixed128_bitpack_reference_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/math/transcendental_highway_scalable.cc" -o "$OUT/math_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/dtype/cvt_scalable.cc" -o "$OUT/dtype_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/bitpack/bitpack_hyle_scalable.cc" -o "$OUT/bitpack_rvv_${name}.o"
  $CXX $FLAGS -c "$REPO/src/simd/sort/vqsort_hyle_rvv.cc" -o "$OUT/sort_rvv_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/tests/simd/scalable_math_dispatch_test.cc" -o "$OUT/scalable_math_dispatch_test_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/platform/cpu.cc" -o "$OUT/cpu_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/runtime.cc" -o "$OUT/runtime_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/math/math_dispatch.cc" -o "$OUT/math_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/math/transcendental_scalar.cc" -o "$OUT/math_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/dtype_dispatch.cc" -o "$OUT/dtype_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/dtype/cvt_scalar.cc" -o "$OUT/dtype_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_dispatch.cc" -o "$OUT/bitpack_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/bitpack/bitpack_scalar.cc" -o "$OUT/bitpack_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/sort_dispatch.cc" -o "$OUT/sort_dispatch_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/vqsort_scalar.cc" -o "$OUT/sort_scalar_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/radix_sort.cc" -o "$OUT/radix_sort_${name}.o"
  $CXX $COMMON_FLAGS -c "$REPO/src/simd/sort/stream_topk.cc" -o "$OUT/stream_topk_${name}.o"
  for module in ops random search; do
    printf '#define HYLE_SIMD_VARIANT_SUFFIX scalar\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
      $CXX $COMMON_FLAGS -x c++ -c - -o "$OUT/${module}_scalar_${name}.o"
    printf '#define HYLE_SIMD_VARIANT_SUFFIX rvv\n#include "src/simd/%s/kernel_impl.h"\n' "$module" |
      $CXX $FLAGS -x c++ -c - -o "$OUT/${module}_rvv_${name}.o"
  done
  $CXX $FLAGS "$OUT/rvv_test_${name}.o" "$OUT/scalable_math_compile_${name}.o" \
    "$OUT/scalable_math_product_test_${name}.o" "$OUT/scalable_math_dispatch_test_${name}.o" \
    "$OUT/scalable_dtype_product_test_${name}.o" \
    "$OUT/scalable_bitpack_product_test_${name}.o" \
    "$OUT/scalable_sort_product_test_${name}.o" \
    "$OUT/fixed128_bitpack_reference_${name}.o" \
    "$OUT/math_rvv_${name}.o" "$OUT/math_scalar_${name}.o" "$OUT/math_dispatch_${name}.o" \
    "$OUT/dtype_rvv_${name}.o" "$OUT/dtype_scalar_${name}.o" "$OUT/dtype_dispatch_${name}.o" \
    "$OUT/bitpack_rvv_${name}.o" "$OUT/bitpack_scalar_${name}.o" "$OUT/bitpack_dispatch_${name}.o" \
    "$OUT/sort_rvv_${name}.o" "$OUT/sort_scalar_${name}.o" "$OUT/sort_dispatch_${name}.o" \
    "$OUT/radix_sort_${name}.o" "$OUT/stream_topk_${name}.o" \
    "$OUT/cpu_${name}.o" "$OUT/runtime_${name}.o" \
    "$OUT/ops_scalar_${name}.o" "$OUT/ops_rvv_${name}.o" \
    "$OUT/random_scalar_${name}.o" "$OUT/random_rvv_${name}.o" \
    "$OUT/search_scalar_${name}.o" "$OUT/search_rvv_${name}.o" \
    "$OUT/gtest-all_${name}.o" "$OUT/gtest-main_${name}.o" \
    -static -o "$bin" -pthread

  echo "[${name}] running under qemu-riscv64 at VLEN 128 / 256 / 512 / 1024 (cpu=${qemu_cpu}) ..."
  for bits in 128 256 512 1024; do
    echo "----- RVV ${name} VLEN=${bits}-bit -----"
    "$QEMU" -cpu "rv64,v=true,${qemu_cpu}vlen=${bits},elen=64" -L /usr/riscv64-linux-gnu \
      "$bin" || rc=$?
  done
}

case "$TIER" in
  base)
    run_tier base rv64gcv ""
    ;;
  zvbb)
    run_tier zvbb rv64gcv_zvbb "zvbb=true,"
    ;;
  zvbc)
    run_tier zvbc rv64gcv_zvbc "zvbc=true,"
    ;;
  all)
    run_tier base rv64gcv ""
    run_tier zvbb rv64gcv_zvbb "zvbb=true,"
    run_tier zvbc rv64gcv_zvbc "zvbc=true,"
    ;;
  *)
    echo "Usage: $0 [base|zvbb|zvbc|all]"; exit 2
    ;;
esac

echo "done (exit=$rc)"
exit $rc
