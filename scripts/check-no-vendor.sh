#!/usr/bin/env bash
# check-no-vendor.sh — simd-20 X4 gate.
# Assert product sources (include/ + src/) never #include Highway or xsimd headers.
# Bench code under bench/ may use external HYLE_BENCH_VENDOR_DIR copies; that is fine.
#
# Usage: scripts/check-no-vendor.sh
# Exit 0 = clean; exit 1 = product still depends on vendor headers or in-tree third_party.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

echo "== product #include gate (include/ + src/) =="
# Only real vendor header paths: "hwy/..." or "xsimd/..." (angle or quote).
# HYLE port filenames like math_xsimd_hyle-inl.h must NOT trip this.
if hits=$(grep -RIn --include='*.h' --include='*.hh' --include='*.hpp' \
    --include='*.cc' --include='*.cpp' --include='*.inc' \
    -E '#[[:space:]]*include[[:space:]]*[<"](hwy/|xsimd/)' include src 2>/dev/null || true); then
  if [[ -n "${hits}" ]]; then
    echo "FAIL: product still includes Highway/xsimd headers:"
    echo "${hits}"
    fail=1
  else
    echo "OK: no #include <hwy/...> or #include <xsimd/...> in include/ or src/"
  fi
fi

echo "== in-tree third_party/{highway,xsimd} must be gone =="
for d in third_party/highway third_party/xsimd; do
  if [[ -e "$d" ]]; then
    echo "FAIL: ${d} still exists (delete for simd-20 X4)"
    fail=1
  else
    echo "OK: ${d} absent"
  fi
done

echo "== product CMake must not add third_party highway/xsimd include paths =="
# Match real CMake path arguments, not comments that mention the historical names.
if hits=$(grep -RIn --include='CMakeLists.txt' --include='*.cmake' \
    -E 'third_party/(highway|xsimd)' cmake src CMakeLists.txt 2>/dev/null \
    | grep -v '^\s*#' | grep -v '#[^"]*third_party' || true); then
  # Filter comment-only lines more carefully: keep only lines that look like path usage.
  real=""
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # Skip pure comments
    if [[ "$line" =~ ^[^:]*:[0-9]+:[[:space:]]*# ]]; then
      continue
    fi
    # Skip lines whose only third_party mention is inside a comment tail
    code="${line#*:*:}"
    code_no_comment="${code%%//*}"
    code_no_comment="${code_no_comment%%#*}"
    if [[ "$code_no_comment" =~ third_party/(highway|xsimd) ]]; then
      real+="$line"$'\n'
    fi
  done <<< "$hits"
  if [[ -n "${real}" ]]; then
    echo "FAIL: CMake still references third_party/highway or third_party/xsimd:"
    printf '%s' "$real"
    fail=1
  else
    echo "OK: no product CMake include of third_party/highway or third_party/xsimd"
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  echo
  echo "check-no-vendor: FAILED"
  exit 1
fi
echo
echo "check-no-vendor: PASSED"
exit 0
