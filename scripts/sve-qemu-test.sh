#!/bin/bash
# sve-qemu-test.sh -- cross-compile the SVE backend gtest (tests/simd/sve_test.cc) for aarch64 and
# run it under qemu-aarch64 at multiple SVE vector lengths (128/256/512-bit). This is the "one
# binary, every width" scalable proof: SVE's vector length is a runtime property, so the SAME
# binary must pass at every VL. There is no SVE dev machine in this project, so qemu's -cpu
# max,sve<bits>=on is the correctness gate (analogous to Intel SDE for AVX-512).
#
# Prereqs (WSL Ubuntu, already installed here):
#   aarch64-linux-gnu-g++-14, qemu-aarch64-static, gtest source under .deps/googletest-src.
# Usage: run from the repo root on a Linux box with the aarch64 cross toolchain:
#   bash scripts/sve-qemu-test.sh
#
# On a REAL SVE host (Graviton3, etc.) you don't need this -- configure with -DHYLE_SIMD_SVE=ON
# and `ctest -R simd_sve` runs it natively.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GT="$REPO/.deps/googletest-src/googletest"
if [ ! -f "$GT/src/gtest-all.cc" ]; then
  echo "ERROR: gtest source not found at $GT (configure the CMake build once to fetch it)"; exit 1
fi

CXX=${CXX_AARCH64:-aarch64-linux-gnu-g++-14}
QEMU=${QEMU_AARCH64:-qemu-aarch64-static}
OUT=${OUT_DIR:-/tmp/hyle-sve}
mkdir -p "$OUT"

FLAGS="-std=c++23 -O2 -march=armv8-a+sve -DHYLE_SIMD_TARGET_SVE -pthread \
  -I$REPO/include -I$REPO -I$GT/include -I$GT"

echo "[1/3] cross-compiling gtest runtime + sve_test.cc for aarch64+sve ..."
$CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all.o"
$CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main.o"
$CXX $FLAGS -c "$REPO/tests/simd/sve_test.cc" -o "$OUT/sve_test.o"
$CXX $FLAGS "$OUT/sve_test.o" "$OUT/gtest-all.o" "$OUT/gtest-main.o" -static -o "$OUT/sve_gtest" -pthread

echo "[2/3] running under qemu at VL 128 / 256 / 512 ..."
rc=0
for bits in 128 256 512; do
  echo "----- SVE VL=${bits}-bit -----"
  "$QEMU" -cpu "max,sve${bits}=on" -L /usr/aarch64-linux-gnu "$OUT/sve_gtest" || rc=$?
done

echo "[3/3] done (exit=$rc)"
exit $rc
