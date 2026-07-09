#!/bin/bash
# rvv-qemu-test.sh -- cross-compile the RVV backend gtest (tests/simd/rvv_test.cc) for rv64gcv and
# run it under qemu-riscv64. RVV's vector length (VLEN) is a runtime property; the test is
# VL-agnostic (queries vsetvlmax), so it is correct at whatever VLEN qemu provides (default 128).
# There is no RISC-V dev machine in this project, so qemu is the correctness gate (analogous to
# Intel SDE for AVX-512 and qemu-aarch64 for SVE).
#
# Prereqs (WSL Ubuntu, already installed here):
#   riscv64-linux-gnu-g++-14, qemu-riscv64-static, gtest source under .deps/googletest-src.
# Usage from the repo root on a Linux box with the riscv64 cross toolchain:
#   bash scripts/rvv-qemu-test.sh
#
# On a REAL RVV host you don't need this -- configure with -DHYLE_SIMD_RVV=ON and
# `ctest -R simd_rvv` runs it natively.
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

FLAGS="-std=c++23 -O2 -march=rv64gcv -DHYLE_SIMD_TARGET_RVV -pthread \
  -I$REPO/include -I$REPO -I$GT/include -I$GT"

echo "[1/3] cross-compiling gtest runtime + rvv_test.cc for rv64gcv ..."
$CXX $FLAGS -c "$GT/src/gtest-all.cc"  -o "$OUT/gtest-all.o"
$CXX $FLAGS -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main.o"
$CXX $FLAGS -c "$REPO/tests/simd/rvv_test.cc" -o "$OUT/rvv_test.o"
$CXX $FLAGS "$OUT/rvv_test.o" "$OUT/gtest-all.o" "$OUT/gtest-main.o" -static -o "$OUT/rvv_gtest" -pthread

echo "[2/3] running under qemu-riscv64 ..."
"$QEMU" -L /usr/riscv64-linux-gnu "$OUT/rvv_gtest"
rc=$?

echo "[3/3] done (exit=$rc)"
exit $rc
