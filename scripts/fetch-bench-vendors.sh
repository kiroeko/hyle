#!/usr/bin/env bash
# Fetch the pinned external Highway/xsimd checkouts used only by SIMD benchmarks.
# Product sources deliberately do not read this directory during simd-20 migration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSIONS_FILE="$REPO_DIR/cmake/vendor-versions.cmake"

usage() {
  cat <<'EOF'
Usage: scripts/fetch-bench-vendors.sh [--dir DIRECTORY]

Fetches Highway and xsimd at the exact revisions in cmake/vendor-versions.cmake.
The directory must contain the two checkout names: highway/ and xsimd/.
Defaults: C:/public/project-hyle on Windows Git Bash; ~/vendor on macOS.
Linux/WSL requires --dir or HYLE_BENCH_VENDOR_DIR.
EOF
}

VENDOR_DIR="${HYLE_BENCH_VENDOR_DIR:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dir)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      VENDOR_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$VENDOR_DIR" ]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) VENDOR_DIR="C:/public/project-hyle" ;;
    Darwin) VENDOR_DIR="$HOME/vendor" ;;
    *)
      printf '%s\n' "Linux/WSL requires --dir or HYLE_BENCH_VENDOR_DIR." >&2
      exit 2
      ;;
  esac
fi

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
cmake -DHYLE_BENCH_VENDOR_OUTPUT_FILE="$manifest" -P "$VERSIONS_FILE"

mkdir -p "$VENDOR_DIR"
while IFS='|' read -r name repository revision; do
  [ -n "$name" ] || continue
  # CMake writes the manifest with the host line ending; trim CR on Git Bash.
  revision="${revision%$'\r'}"
  checkout="$VENDOR_DIR/$name"
  if [ -e "$checkout" ] && [ ! -d "$checkout/.git" ] && [ ! -f "$checkout/.git" ]; then
    printf '%s\n' "$checkout exists but is not a Git checkout; refusing to overwrite it." >&2
    exit 1
  fi

  if [ ! -e "$checkout" ]; then
    git clone "$repository" "$checkout"
  else
    status="$(git -C "$checkout" status --porcelain)"
    if [ -n "$status" ]; then
      printf '%s\n' "$checkout has local changes; refusing to overwrite it." >&2
      exit 1
    fi
  fi

  git -C "$checkout" fetch --depth=1 origin "$revision"
  git -C "$checkout" checkout --detach "$revision"
  actual="$(git -C "$checkout" rev-parse HEAD)"
  [ "$actual" = "$revision" ] || {
    printf '%s\n' "$name resolved to $actual, expected $revision." >&2
    exit 1
  }
  printf '%s\n' "[ok] $name $actual -> $checkout"
done < "$manifest"
