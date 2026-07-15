#!/bin/bash
# simd-19 C18: three NEON bench TUs × dual order × 3 runs, serial (jetsam-safe).
set -u
export PATH=/opt/homebrew/bin:$PATH
OUT="$HOME/hyle/oplab/s19_c18"
LOG="$OUT/c18_runner.log"
mkdir -p "$OUT"
: > "$LOG"
echo "C18 start $(date)" | tee -a "$LOG"

BIN_parity="$HOME/hyle/build-mac/bin/hyle_op_parity_bench_neon"
BIN_full="$HOME/hyle/build-mac/bin/hyle_op_parity_full_bench_neon"
BIN_widen="$HOME/hyle/build-mac/bin/hyle_op_parity_widen_bench_neon"

# expected line counts filled after first successful run per TU
EXPECT_parity=""
EXPECT_full=""
EXPECT_widen=""

order_ok() {
  local f="$1" order="$2"
  local first
  first=$(grep -E 'HYLE#|HWY#' "$f" | head -1 || true)
  if [ -z "$first" ]; then
    echo "ORDER_CHECK missing markers in $f" | tee -a "$LOG"
    return 1
  fi
  if [ "$order" = "hwyfirst" ]; then
    if echo "$first" | grep -q 'HWY#'; then
      echo "ORDER_OK hwyfirst ($f)" | tee -a "$LOG"
      return 0
    fi
    echo "ORDER_FAIL hwyfirst first_line=[$first] ($f)" | tee -a "$LOG"
    return 1
  else
    if echo "$first" | grep -q 'HYLE#'; then
      echo "ORDER_OK normal ($f)" | tee -a "$LOG"
      return 0
    fi
    echo "ORDER_FAIL normal first_line=[$first] ($f)" | tee -a "$LOG"
    return 1
  fi
}

get_expect() {
  case "$1" in
    parity) echo "$EXPECT_parity" ;;
    full) echo "$EXPECT_full" ;;
    widen) echo "$EXPECT_widen" ;;
  esac
}

set_expect() {
  case "$1" in
    parity) EXPECT_parity="$2" ;;
    full) EXPECT_full="$2" ;;
    widen) EXPECT_widen="$2" ;;
  esac
}

run_one() {
  local tu="$1" bin="$2" order="$3" n="$4"
  local out="$OUT/s19_${tu}_${order}_${n}.txt"
  echo "START tu=$tu order=$order n=$n $(date)" | tee -a "$LOG"
  if [ "$order" = "hwyfirst" ]; then
    env HYLE_BENCH_HWYFIRST=1 "$bin" >"$out" 2>&1
  else
    "$bin" >"$out" 2>&1
  fi
  local rc=$?
  local lines
  lines=$(wc -l <"$out" | tr -d ' ')
  echo "DONE tu=$tu order=$order n=$n rc=$rc lines=$lines $(date)" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "WARN non-zero rc=$rc for $out" | tee -a "$LOG"
  fi
  local exp
  exp=$(get_expect "$tu")
  if [ -z "$exp" ]; then
    set_expect "$tu" "$lines"
    echo "EXPECT[$tu]=$lines (baseline from first run)" | tee -a "$LOG"
  else
    if [ "$lines" -lt "$exp" ]; then
      echo "INCOMPLETE short lines=$lines < expect=$exp for $out" | tee -a "$LOG"
    elif [ "$lines" -ne "$exp" ]; then
      echo "NOTE lines=$lines != expect=$exp for $out" | tee -a "$LOG"
    else
      echo "COMPLETE lines=$lines match expect for $out" | tee -a "$LOG"
    fi
  fi
  order_ok "$out" "$order" || true
}

for n in 1 2 3; do
  for order in normal hwyfirst; do
    run_one parity "$BIN_parity" "$order" "$n"
    run_one full "$BIN_full" "$order" "$n"
    run_one widen "$BIN_widen" "$order" "$n"
  done
done

echo "C18 all runs finished $(date)" | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
ls -la "$OUT"/s19_*.txt | tee -a "$LOG"
wc -l "$OUT"/s19_*.txt | tee -a "$LOG"
echo DONE > "$OUT/c18_FINISHED"
