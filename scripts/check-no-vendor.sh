#!/usr/bin/env bash
# check-no-vendor.sh — SIMD vendor-boundary gate (historical filename).
# Installed public headers must never expose vendor headers or types. Product
# sources may include centrally locked Highway/xsimd only in explicit
# target-bound implementation bodies; baseline dispatch/API files may not.
# Bench code under bench/ may use external HYLE_BENCH_VENDOR_DIR copies.
#
# Usage: scripts/check-no-vendor.sh
# Exit 0 = boundary clean; exit 1 = an unowned vendor edge or in-tree copy exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

echo "== public vendor include gate (include/) =="
if hits=$(grep -RIn --include='*.h' --include='*.hh' --include='*.hpp' \
    -E '#[[:space:]]*include[[:space:]]*[<"](hwy/|xsimd/)' include 2>/dev/null || true); then
  if [[ -n "${hits}" ]]; then
    echo "FAIL: public headers expose Highway/xsimd headers:"
    echo "${hits}"
    fail=1
  else
    echo "OK: public headers do not expose Highway/xsimd"
  fi
fi

echo "== private product vendor include allowlist (src/) =="
if hits=$(grep -RIn --include='*.h' --include='*.hh' --include='*.hpp' \
    --include='*.cc' --include='*.cpp' --include='*.inc' \
    -E '#[[:space:]]*include[[:space:]]*[<"](hwy/|xsimd/)' src 2>/dev/null || true); then
  unexpected=$(printf '%s\n' "${hits}" | grep -Ev \
    '^src/simd/(provider/[^:]+|bitpack/kernel_impl\.h|dtype/(conversion_kernel_impl\.h|highway_narrow\.h)|math/(highway_math_kernel-inl\.h|complex_hyle-inl\.h|complex_typed_hyle-inl\.h|provider_[^/:]+\.h)|narrow/kernel_impl\.h|ops/kernel_impl\.h|random/kernel_impl\.h|scan/kernel_impl\.h|search/kernel_impl\.h):[0-9]+:#[[:space:]]*include[[:space:]]*[<"](hwy/|xsimd/)|^src/container/(bitset_ops_impl\.h|str_search_impl\.h):[0-9]+:#[[:space:]]*include[[:space:]]*[<"]hwy/' || true)
  if [[ -n "${unexpected}" ]]; then
    echo "FAIL: product vendor includes escaped target-bound implementation bodies:"
    echo "${unexpected}"
    fail=1
  else
    echo "OK: locked Highway/xsimd includes are confined to target-bound implementation bodies"
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
  echo "check-no-vendor (vendor-boundary): FAILED"
  exit 1
fi
echo
echo "check-no-vendor (vendor-boundary): PASSED"
exit 0
