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

  local FLAGS="-std=c++23 -O2 -march=${march} -DHYLE_SIMD_TARGET_RVV -pthread \
    -I$REPO/include -I$REPO -I$GT/include -I$GT"

  echo "[${name}] cross-compiling gtest + rvv_test.cc (-march=${march}) ..."
  $CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all_${name}.o"
  $CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main_${name}.o"
  $CXX $FLAGS -c "$REPO/tests/simd/rvv_test.cc" -o "$OUT/rvv_test_${name}.o"
  $CXX $FLAGS "$OUT/rvv_test_${name}.o" "$OUT/gtest-all_${name}.o" "$OUT/gtest-main_${name}.o" \
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
