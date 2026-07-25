#!/bin/bash
# Build + install + run sample 04 in BOTH link modes on Linux. Run from this directory:
#   bash build_and_run.sh   (uses g++-14 via CXX/CC if set)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${CXX:=g++-14}" "${CC:=gcc-14}"
export CXX CC
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hyle-sample04.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0
build_hyle() { # $1=mode(shared|static) $2=builddir $3=prefix
  local extra=""
  [ "$1" = "static" ] && extra="-DHYLE_STATIC=ON"
  cmake -S "$REPO" -B "$2" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DHYLE_COMPONENTS='simd;mem' -DHYLE_BUILD_TESTS=OFF -DHYLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="$3" $extra >/dev/null 2>&1 \
    && cmake --build "$2" --target hyle_simd >/dev/null 2>&1 \
    && cmake --install "$2" >/dev/null 2>&1
}
run_mode() { # $1=mode $2=prefix
  local cdir="$WORK/consumer-$1"
  cmake -S "$HERE" -B "$cdir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$2" >/dev/null 2>&1 \
    && cmake --build "$cdir" >/dev/null 2>&1
  if [ -x "$cdir/installed_package_consumer" ]; then
    echo "--- $1 package, run: ---"; "$cdir/installed_package_consumer"; echo "exit=$?"
  else
    echo "--- $1 link FAILED to produce consumer ---"; fail=1
  fi
}

echo "=== building HYLE shared + static ==="
build_hyle shared "$WORK/hyle-shared" "$WORK/prefix-shared" || { echo "shared HYLE install failed"; fail=1; }
build_hyle static "$WORK/hyle-static" "$WORK/prefix-static" || { echo "static HYLE install failed"; fail=1; }
echo "=== consumer via find_package ==="
run_mode shared "$WORK/prefix-shared"
run_mode static "$WORK/prefix-static"
exit $fail
