#!/bin/bash
# Build + install + run sample 04 in BOTH link modes on Linux. Run from this directory:
#   bash build_and_run.sh   (uses g++-14 via CXX/CC if set)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${CXX:=g++-14}" "${CC:=gcc-14}"
export CXX CC
# Reuse this sample's two linkage-specific build trees. An optional absolute
# first argument selects another workspace; no cache is deleted on exit.
WORK="${1:-$REPO/.scratch/sample04}"
mkdir -p "$WORK" || exit 1

fail=0
build_hyle() { # $1=mode(shared|static) $2=builddir $3=prefix
  local static=OFF
  [ "$1" = "static" ] && static=ON
  cmake -S "$REPO" -B "$2" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DHYLE_COMPONENTS='simd;mem' -DHYLE_INSTALL=ON "-DHYLE_STATIC=$static" \
    -DCMAKE_INSTALL_PREFIX="$3" \
    && cmake --build "$2" \
    && cmake --install "$2"
}
run_mode() { # $1=mode $2=prefix
  local cdir="$WORK/consumer-$1"
  if ! cmake -S "$HERE" -B "$cdir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$2" || ! cmake --build "$cdir"; then
    echo "--- $1 consumer build FAILED ---"; fail=1
    return
  fi
  if [ -x "$cdir/installed_package_consumer" ]; then
    echo "--- $1 package, run: ---"
    "$cdir/installed_package_consumer"
    local result=$?
    echo "exit=$result"
    [ "$result" -eq 0 ] || fail=1
  else
    echo "--- $1 link FAILED to produce consumer ---"; fail=1
  fi
}

for mode in shared static; do
  echo "=== building HYLE $mode ==="
  if build_hyle "$mode" "$WORK/hyle-$mode" "$WORK/prefix-$mode"; then
    run_mode "$mode" "$WORK/prefix-$mode"
  else
    echo "$mode HYLE install failed"
    fail=1
  fi
done
exit $fail
