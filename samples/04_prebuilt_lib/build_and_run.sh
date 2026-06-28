#!/bin/bash
# Build + run sample 04 in BOTH link modes on Linux. Builds HYLE twice (shared, static),
# then links this consumer against each prebuilt hyle_simd. Run from this directory:
#   bash build_and_run.sh   (uses g++-14 via CXX/CC if set)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${CXX:=g++-14}" "${CC:=gcc-14}"
export CXX CC

fail=0
build_hyle() { # $1=mode(shared|static) $2=builddir
  local extra=""
  [ "$1" = "static" ] && extra="-DHYLE_STATIC=ON"
  rm -rf "$2"  # fresh dir: never reconfigure a stale cache (source-dir mismatch etc.)
  cmake -S "$REPO" -B "$2" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHYLE_BUILD_TESTS=OFF $extra >/dev/null 2>&1 \
    && cmake --build "$2" --target hyle_simd >/dev/null 2>&1
}
run_mode() { # $1=mode $2=hylebuild
  local cdir="$HERE/build-$1"
  rm -rf "$cdir"
  cmake -S "$HERE" -B "$cdir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DHYLE_REPO_DIR="$REPO" -DHYLE_PREBUILT_DIR="$2" -DHYLE_LINK_MODE="$1" >/dev/null 2>&1 \
    && cmake --build "$cdir" >/dev/null 2>&1
  if [ -x "$cdir/prebuilt_consumer" ]; then
    echo "--- $1 link, run: ---"; "$cdir/prebuilt_consumer"; echo "exit=$?"
  else
    echo "--- $1 link FAILED to produce consumer ---"; fail=1
  fi
}

echo "=== building HYLE shared + static ==="
build_hyle shared /tmp/hyle-shared || { echo "shared HYLE build failed"; fail=1; }
build_hyle static /tmp/hyle-static || { echo "static HYLE build failed"; fail=1; }
echo "=== consumer vs prebuilt ==="
run_mode shared /tmp/hyle-shared
run_mode static /tmp/hyle-static
exit $fail
