#!/usr/bin/env bash
# setup-bench-libs.sh -- clone the SOTA reference libraries HYLE's *gated*
# benchmarks compare against (abseil/folly/moodycamel/oneTBB/marl/
# crossbeam-deque/rigtorp-MPMCQueue/xxHash/wyhash/mimalloc/liburing/asio/libuv).
#
# NOTE: Highway/xsimd use the separately pinned fetch-bench-vendors.{sh,ps1}
# flow because every SIMD bench needs their unchanged external copies. simdutf
# remains vendored in third_party/. (SLEEF was removed from the project entirely.)
#
# WHY: the gated benches (-DHYLE_BENCH_ABSL / _TBB / _MARL / _MOODYCAMEL /
# _RIGTORP / _BOOST, and the Rust crossbeam crate) point
# at these libs via the HYLE_<LIB>_DIR CMake cache vars, whose DEFAULTS are a
# sibling of this repo (../<lib>). That made benchmarking impossible on a fresh
# machine that didn't happen to have the exact sibling layout. This script
# reproduces that layout anywhere, idempotently, with no hard-coded absolute path.
#
# USAGE:
#   scripts/setup-bench-libs.sh            # clone into the repo's parent dir
#   HYLE_DEPS_DIR=/some/where scripts/setup-bench-libs.sh   # custom location
#   scripts/setup-bench-libs.sh --shallow  # --depth 1 (faster, less history)
#   scripts/setup-bench-libs.sh marl moodycamel-ConcurrentQueue   # only the named libs
#
# Then point the build at $HYLE_DEPS_DIR (only needed if NOT the sibling default):
#   cmake ... -DHYLE_BENCH_MARL=ON -DHYLE_MARL_DIR=$HYLE_DEPS_DIR/marl
#
# Cross-platform: pure `git`, works in git-bash (Windows) and Linux/macOS.
# Idempotent: a lib that already has a .git is left untouched (no clobber).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Default deps dir = repo's parent (the sibling layout the CMake defaults expect).
DEPS_DIR="${HYLE_DEPS_DIR:-$(cd "$REPO_DIR/.." && pwd)}"

SHALLOW=""
SELECT=()
for arg in "$@"; do
  case "$arg" in
    --shallow) SHALLOW="--depth 1" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) SELECT+=("$arg") ;;
  esac
done

# name|url -- URLs are the upstreams actually used by this project's clones.
# (moodycamel + crossbeam-deque are community forks pinned for reproducibility;
#  swap to cameron314/moodycamel or crossbeam-rs/crossbeam if you prefer canon.)
LIBS=(
  "abseil-cpp|https://github.com/abseil/abseil-cpp.git"
  "folly|https://github.com/facebook/folly.git"
  "moodycamel-ConcurrentQueue|https://github.com/ikiller1/moodycamel-ConcurrentQueue.git"
  "oneTBB|https://github.com/uxlfoundation/oneTBB.git"
  "marl|https://github.com/google/marl.git"
  "crossbeam-deque|https://github.com/k-sareen/crossbeam-deque.git"
  "rigtorp-MPMCQueue|https://github.com/rigtorp/MPMCQueue.git"
  "xxHash|https://github.com/Cyan4973/xxHash.git"
  "wyhash|https://github.com/wangyi-fudan/wyhash.git"
  "mimalloc|https://github.com/microsoft/mimalloc.git"
  "liburing|https://github.com/axboe/liburing.git"
  "asio|https://github.com/chriskohlhoff/asio.git"
  "libuv|https://github.com/libuv/libuv.git"
)

echo "HYLE bench deps -> $DEPS_DIR"
mkdir -p "$DEPS_DIR"

want() {  # returns 0 if $1 should be processed (no filter, or in SELECT)
  [ "${#SELECT[@]}" -eq 0 ] && return 0
  for s in "${SELECT[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

cloned=0 skipped=0 failed=0
for entry in "${LIBS[@]}"; do
  name="${entry%%|*}"; url="${entry#*|}"
  want "$name" || continue
  dst="$DEPS_DIR/$name"
  if [ -d "$dst/.git" ] || [ -f "$dst/.git" ]; then
    echo "  [skip]  $name (already present)"
    skipped=$((skipped + 1))
    continue
  fi
  echo "  [clone] $name <- $url"
  if git clone $SHALLOW "$url" "$dst"; then
    cloned=$((cloned + 1))
  else
    echo "  [FAIL]  $name" >&2
    failed=$((failed + 1))
  fi
done

echo "done: cloned=$cloned skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ] || exit 1
