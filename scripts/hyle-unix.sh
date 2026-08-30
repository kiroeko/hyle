#!/usr/bin/env bash
# Select an already-installed supported Unix host toolchain, then execute the
# requested command without rewriting its arguments.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/hyle-unix.sh --print-toolchain
  scripts/hyle-unix.sh [--] COMMAND [ARG...]

The wrapper never installs software or changes an existing CMake cache.
Set both CC and CXX to choose an explicit supported compiler pair. When both
are unset, Linux prefers versioned GCC 14+ and then Clang 18+; macOS uses the
system clang/clang++ pair.
EOF
}

fail() {
  printf 'hyle-unix: %s\n' "$*" >&2
  exit 2
}

probe_compiler() {
  local executable="$1"
  local banner lower family version major

  command -v "$executable" >/dev/null 2>&1 || return 1
  banner="$("$executable" --version 2>&1)" || return 1
  banner="${banner%%$'\n'*}"
  lower="$(printf '%s\n' "$banner" | tr '[:upper:]' '[:lower:]')"

  case "$lower" in
    *clang*)
      family=clang
      version="$(printf '%s\n' "$banner" |
        sed -nE 's/.*version[[:space:]]+([0-9]+)(\.[0-9]+)*.*/\1/p')"
      ;;
    *)
      # GCC's unversioned cc/c++ driver banner need not contain the words gcc
      # or g++; its dump-version interface is the stable discriminator here.
      family=gcc
      version="$("$executable" -dumpfullversion -dumpversion 2>/dev/null)" ||
        return 1
      version="${version%%$'\n'*}"
      version="${version%%.*}"
      ;;
  esac

  major="${version%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] || return 1
  printf '%s|%s\n' "$family" "$major"
}

pair_supported() {
  local cc="$1" cxx="$2"
  local cc_probe cxx_probe cc_family cc_major cxx_family cxx_major floor

  cc_probe="$(probe_compiler "$cc")" || return 1
  cxx_probe="$(probe_compiler "$cxx")" || return 1
  IFS='|' read -r cc_family cc_major <<<"$cc_probe"
  IFS='|' read -r cxx_family cxx_major <<<"$cxx_probe"
  [[ "$cc_family" == "$cxx_family" ]] || return 1
  [[ "$cc_major" == "$cxx_major" ]] || return 1

  case "$cxx_family" in
    gcc) floor=14 ;;
    clang) floor=18 ;;
    *) return 1 ;;
  esac
  ((cc_major >= floor && cxx_major >= floor))
}

validate_explicit_pair() {
  local cc_probe cxx_probe cc_family cc_major cxx_family cxx_major floor

  cc_probe="$(probe_compiler "$CC")" ||
    fail "CC='$CC' is missing or is not a supported GCC/Clang driver"
  cxx_probe="$(probe_compiler "$CXX")" ||
    fail "CXX='$CXX' is missing or is not a supported GCC/Clang driver"
  IFS='|' read -r cc_family cc_major <<<"$cc_probe"
  IFS='|' read -r cxx_family cxx_major <<<"$cxx_probe"
  [[ "$cc_family" == "$cxx_family" ]] ||
    fail "CC='$CC' and CXX='$CXX' are from different compiler families"
  [[ "$cc_major" == "$cxx_major" ]] ||
    fail "CC='$CC' and CXX='$CXX' have different major versions ($cc_major vs $cxx_major)"

  case "$cxx_family" in
    gcc) floor=14 ;;
    clang) floor=18 ;;
    *) fail "unsupported compiler family '$cxx_family'" ;;
  esac
  ((cc_major >= floor && cxx_major >= floor)) ||
    fail "$cxx_family requires major version $floor+ (CC=$cc_major, CXX=$cxx_major)"
}

select_linux_pair() {
  local version cc cxx

  # Prefer the project's oldest supported GCC when it is installed. This keeps
  # normal WSL validation on the documented GCC 14 baseline, while still
  # accepting newer versioned toolchains on other distributions.
  for version in {14..30}; do
    cc="gcc-$version"
    cxx="g++-$version"
    if pair_supported "$cc" "$cxx"; then
      CC="$cc"
      CXX="$cxx"
      export CC CXX
      return
    fi
  done
  for version in {18..30}; do
    cc="clang-$version"
    cxx="clang++-$version"
    if pair_supported "$cc" "$cxx"; then
      CC="$cc"
      CXX="$cxx"
      export CC CXX
      return
    fi
  done
  if pair_supported cc c++; then
    CC=cc
    CXX=c++
    export CC CXX
    return
  fi

  fail "no supported compiler pair found; install GCC 14+ or Clang 18+, or set both CC and CXX"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${CC:-}" || -n "${CXX:-}" ]]; then
  [[ -n "${CC:-}" && -n "${CXX:-}" ]] ||
    fail "CC and CXX must be set together"
else
  case "$(uname -s)" in
    Linux) select_linux_pair ;;
    Darwin)
      CC=clang
      CXX=clang++
      export CC CXX
      ;;
    *) fail "supported host systems are Linux and macOS" ;;
  esac
fi
validate_explicit_pair

if [[ "${1:-}" == "--print-toolchain" ]]; then
  [[ "$#" -eq 1 ]] || fail "--print-toolchain does not accept a command"
  printf 'CC=%s\nCXX=%s\n' "$CC" "$CXX"
  exit 0
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
[[ "$#" -gt 0 ]] || {
  usage >&2
  exit 2
}

printf 'HYLE Unix toolchain: CC=%s CXX=%s\n' "$CC" "$CXX" >&2
exec "$@"
