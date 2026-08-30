#!/bin/bash
# Build the single-leg SVE/RVV public + native contract without CMake and run it at several
# vector lengths. This complements the fat scalable harnesses: the exact artifact must link only
# its requested suffix and must not retain a CPU probe or a lower-tier runtime table.
#
# Usage:
#   bash scripts/simd-scalable-exact-qemu-test.sh sve
#   bash scripts/simd-scalable-exact-qemu-test.sh rvv
#   HYLE_SCALABLE_SEARCH_SCAN_ONLY=1 bash scripts/simd-scalable-exact-qemu-test.sh sve
#     # SIMD-09 bounded exact-artifact gate
#   HYLE_SCALABLE_BITPACK_RANDOM_ONLY=1 bash scripts/simd-scalable-exact-qemu-test.sh sve
#     # SIMD-10 bounded exact-artifact gate
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GT="${GTEST_SOURCE_DIR:-}"
KIND="${1:-}"
OUT="${OUT_DIR:-/tmp/hyle-scalable-exact-$KIND}"
mkdir -p "$OUT"

case "$KIND" in
  sve)
    CXX="${CXX_AARCH64:-aarch64-linux-gnu-g++-14}"
    QEMU="${QEMU_AARCH64:-qemu-aarch64-static}"
    BASE_ARCH="armv8-a"
    TARGET_ARCH="armv8-a+sve"
    EXACT_DEFINE="HYLE_SIMD_EXACT_TARGET_SVE"
    VARIANT_DEFINE="HYLE_SIMD_HAS_SVE_VARIANT"
    TARGET_DEFINE="HYLE_SIMD_TARGET_SVE"
    SUFFIX="sve"
    ISA_NAME="sve"
    LENGTHS=(128 256 384 512 2048)
    ;;
  rvv)
    CXX="${CXX_RISCV64:-riscv64-linux-gnu-g++-14}"
    QEMU="${QEMU_RISCV64:-qemu-riscv64-static}"
    BASE_ARCH="rv64gc"
    TARGET_ARCH="rv64gcv"
    EXACT_DEFINE="HYLE_SIMD_EXACT_TARGET_RVV"
    VARIANT_DEFINE="HYLE_SIMD_HAS_RVV_VARIANT"
    TARGET_DEFINE="HYLE_SIMD_TARGET_RVV"
    SUFFIX="rvv"
    ISA_NAME="rvv"
    LENGTHS=(128 256 512 1024)
    ;;
  *)
    echo "usage: $0 sve|rvv" >&2
    exit 2
    ;;
esac

if [ ! -f "$GT/src/gtest-all.cc" ]; then
  echo "ERROR: gtest source not found at $GT" >&2
  exit 1
fi
HWY="${HIGHWAY_SOURCE_DIR:-}"
if [ ! -f "$HWY/hwy/highway.h" ]; then
  echo "ERROR: locked Highway source not found at $HWY (set HIGHWAY_SOURCE_DIR)" >&2
  exit 1
fi

COMMON=(-std=c++23 -O2 "-march=$BASE_ARCH" -pthread
  "-I$REPO/include" "-I$REPO" "-I$HWY" "-I$GT/include" "-I$GT"
  -DHWY_HEADER_ONLY=1 -DHYLE_SIMD_EXACT=1 "-D$EXACT_DEFINE=1"
  "-D$VARIANT_DEFINE=1")
TARGET=("${COMMON[@]}" "-march=$TARGET_ARCH" "-D$TARGET_DEFINE=1")
SCALAR_COMMON=(-std=c++23 -O2 "-march=$BASE_ARCH" -pthread
  "-I$REPO/include" "-I$REPO" -DHYLE_SIMD_EXACT=1
  -DHYLE_SIMD_EXACT_TARGET_SCALAR=1)

echo "[$KIND exact] compiling one runtime table and one kernel suffix ..."
"$CXX" "${COMMON[@]}" -c "$GT/src/gtest-all.cc" -o "$OUT/gtest-all.o"
"$CXX" "${COMMON[@]}" -c "$GT/src/gtest_main.cc" -o "$OUT/gtest-main.o"
for test in runtime ops search scan narrow_ops bitpack; do
  "$CXX" "${COMMON[@]}" -c "$REPO/tests/simd/${test}_test.cc" -o "$OUT/${test}_test.o"
done
"$CXX" "${TARGET[@]}" -c "$REPO/tests/simd/native_test.cc" -o "$OUT/native_test.o"
"$CXX" "${TARGET[@]}" -c "$REPO/tests/simd/scalable_bitpack_product_test.cc" \
  -o "$OUT/scalable_bitpack_product_test.o"

"$CXX" "${COMMON[@]}" -c "$REPO/src/simd/runtime.cc" -o "$OUT/runtime.o"
"$CXX" "${COMMON[@]}" -c "$REPO/src/platform/cpu.cc" -o "$OUT/cpu_test_only.o"
"$CXX" "${COMMON[@]}" -DHYLE_ERROR_BUILD -c "$REPO/src/error/status.cc" \
  -o "$OUT/error_status.o"
# cpu.cc's topology diagnostics use this platform entry, while exact dispatch
# itself never probes it. Keep this standalone artifact focused on SIMD instead
# of pulling thread.cc and the whole error/platform dependency graph.
printf '#include "hyle/platform/thread.h"\nnamespace hyle::platform { unsigned hardware_concurrency() noexcept { return 1; } }\n' |
  "$CXX" "${COMMON[@]}" -x c++ -c - -o "$OUT/thread_test_stub.o"
printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX %s\n#include "tests/simd/fat_kernel_fixture.h"\n' \
  "$SUFFIX" |
  "$CXX" "${TARGET[@]}" -x c++ -c - -o "$OUT/fat_fixture.o"
for product in math/math_dispatch dtype/dtype_dispatch sort/sort_dispatch \
               bitpack/bitpack_dispatch; do
  "$CXX" "${COMMON[@]}" -c "$REPO/src/simd/$product.cc" \
    -o "$OUT/product_$(basename "$product").o"
done
"$CXX" "${SCALAR_COMMON[@]}" -c "$REPO/src/simd/math/math_dispatch.cc" \
  -o "$OUT/product_math_dispatch_scalar_exact.o"
for module in random; do
  dispatch_source="${module}_dispatch.cc"
  "$CXX" "${COMMON[@]}" -c "$REPO/src/simd/$module/$dispatch_source" \
    -o "$OUT/${module}_dispatch.o"
  printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX %s\n#include "src/simd/%s/kernel_impl.h"\n' \
    "$SUFFIX" "$module" |
    "$CXX" "${TARGET[@]}" -x c++ -c - -o "$OUT/${module}_${SUFFIX}.o"
done
for module in ops search scan narrow; do
  dispatch_source="${module}_dispatch.cc"
  if [ "$module" = "ops" ]; then dispatch_source="dispatch_ops.cc"; fi
  "$CXX" "${COMMON[@]}" -c "$REPO/src/simd/$module/$dispatch_source" \
    -o "$OUT/${module}_dispatch.o"
  printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX %s\n#include "src/simd/%s/kernel_impl.h"\n' \
    "$SUFFIX" "$module" |
    "$CXX" "${TARGET[@]}" -x c++ -c - -o "$OUT/${module}_${SUFFIX}.o"
done
printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX %s\n#include "src/simd/dtype/conversion_kernel_impl.h"\n' \
  "$SUFFIX" |
  "$CXX" "${TARGET[@]}" -x c++ -c - -o "$OUT/dtype_${SUFFIX}.o"
printf '#define HWY_COMPILE_ONLY_STATIC 1\n#define HYLE_SIMD_VARIANT_SUFFIX %s\n#include "src/simd/bitpack/kernel_impl.h"\n' \
  "$SUFFIX" |
  "$CXX" "${TARGET[@]}" -x c++ -c - -o "$OUT/bitpack_${SUFFIX}.o"

if nm -u "$OUT/runtime.o" "$OUT/"*_dispatch.o |
    grep -Eq 'cpu_features|best_isa|supports'; then
  echo "ERROR: exact dispatch retains a CPU probe" >&2
  exit 1
fi
if [ "$KIND" = "sve" ]; then
  FOREIGN_SUFFIX='_(neon|rvv|sse4|avx2|avx512)\('
else
  FOREIGN_SUFFIX='_(neon|sve|sse4|avx2|avx512)\('
fi
if nm -u -C "$OUT"/product_*.o | grep -Eq "$FOREIGN_SUFFIX"; then
  echo "ERROR: exact product dispatcher retains a foreign target suffix" >&2
  exit 1
fi
if nm -u -C "$OUT/product_math_dispatch_scalar_exact.o" |
    grep -Eq '_(neon|sve|rvv|sse4|avx2|avx512)\('; then
  echo "ERROR: architecture-hosted exact scalar math retains a SIMD suffix" >&2
  exit 1
fi

"$CXX" "${TARGET[@]}" "$OUT/gtest-all.o" "$OUT/gtest-main.o" \
  "$OUT/runtime_test.o" "$OUT/ops_test.o" "$OUT/search_test.o" \
  "$OUT/scan_test.o" "$OUT/narrow_ops_test.o" "$OUT/bitpack_test.o" \
  "$OUT/scalable_bitpack_product_test.o" "$OUT/native_test.o" "$OUT/runtime.o" \
  "$OUT/cpu_test_only.o" "$OUT/thread_test_stub.o" \
  "$OUT/error_status.o" "$OUT/fat_fixture.o" \
  "$OUT/ops_dispatch.o" "$OUT/random_dispatch.o" \
  "$OUT/search_dispatch.o" "$OUT/scan_dispatch.o" \
  "$OUT/narrow_dispatch.o" "$OUT/product_bitpack_dispatch.o" \
  "$OUT/ops_${SUFFIX}.o" "$OUT/random_${SUFFIX}.o" \
  "$OUT/search_${SUFFIX}.o" "$OUT/scan_${SUFFIX}.o" \
  "$OUT/narrow_${SUFFIX}.o" "$OUT/dtype_${SUFFIX}.o" \
  "$OUT/bitpack_${SUFFIX}.o" \
  -static -o "$OUT/exact_gtest" -pthread

RUN_ARGS=(--gtest_brief=1)
if [ "${HYLE_SCALABLE_SEARCH_SCAN_ONLY:-0}" = "1" ]; then
  RUN_ARGS+=(--gtest_filter=SimdRuntime.TableMetadataAndExactTierLookup:Search.*:Scan.*)
elif [ "${HYLE_SCALABLE_BITPACK_RANDOM_ONLY:-0}" = "1" ]; then
  RUN_ARGS+=(--gtest_filter=SimdRuntime.TableMetadataAndExactTierLookup:SimdRuntime.Random*:BitPack.*:ScalableBitpackProduct.*)
fi

for bits in "${LENGTHS[@]}"; do
  echo "----- $KIND exact ${bits}-bit -----"
  if [ "$KIND" = "sve" ]; then
    "$QEMU" -cpu "max,sve-max-vq=$((bits / 128))" "$OUT/exact_gtest" \
      "${RUN_ARGS[@]}"
  else
    "$QEMU" -cpu "rv64,v=true,vlen=$bits,elen=64,vext_spec=v1.0" \
      "$OUT/exact_gtest" "${RUN_ARGS[@]}"
  fi
done

echo "exact scalable artifact OK: isa=$ISA_NAME"
