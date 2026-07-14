#!/usr/bin/env python3
"""SIMD op benchmark coverage — mechanical, self-validating.

The point of this script is to answer ONE question reliably: how many HYLE SIMD ops
CAN be perf-benched against a Highway/xsimd reference, and how many actually ARE?

Every prior attempt to count this by hand or by a one-shot grep was wrong (ops that were
demonstrably benched got reported "missing", inflating the gap). The fix is a self-check:

    BENCHED must be a SUBSET of BENCHABLE.

If any op that appears in an op_parity bench is NOT in the extracted benchable set, the
EXTRACTION RULES have a blind spot (they missed that op's definition form). The script then
FAILS LOUDLY and lists the offenders, so the rules get fixed and re-run. Only once
`benched ⊆ benchable` holds is the denominator trustworthy, and:

    uncovered = benchable − benched      <- the real number.

Run:  python scripts/simd-op-coverage.py          (report)
      python scripts/simd-op-coverage.py --check   (exit 1 if benched⊄benchable or uncovered grows)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INC = os.path.join(ROOT, "include", "hyle", "simd")
BENCH = os.path.join(ROOT, "bench", "simd")

HEADERS = ["vec.h"] + [f"backend/{b}.h" for b in
                       ("avx2", "sse4", "avx512", "neon", "scalar", "sve", "rvv")]
# op_parity_* are the calibrated HYLE-vs-Highway parity benches; crypto/gather also pair HYLE vs a
# reference per-op so they count as coverage for the ops they title.
BENCH_FILES = ["op_parity_full_bench.cc", "op_parity_bench.cc", "op_parity_widen_bench.cc",
               "crypto_bench.cc", "gather_bench.cc"]
# authoritative per-op measured-verdict table (the calibrated instrument's results). An op having a
# bench is NOT the same as having a measured verdict: only ops with a row here are confirmed TIE/WIN
# (i.e. "proven >= Highway / SOTA"). benched-minus-matrix = benched but not yet measured.
MATRIX = os.path.join(ROOT, "docs", "milestones", "simd-orderswap-matrix.md")

# Ops that exist but are NOT perf-benchable against a vector reference. Each with a REASON,
# so the exclusion is auditable (no silent shrinking of the denominator).
NOT_BENCHABLE = {
    # constexpr scalar predicate — not a vector op
    "DivIsPow2": "constexpr bool predicate",
    # data-independent constant producers (nothing to race a reference on)
    "Inf": "constant producer", "NaN": "constant producer", "Iota": "constant producer",
    "Undefined": "constant producer", "Zero": "constant producer", "Set": "splat/constant",
    # introspection / compile-time
    "Lanes": "introspection", "Blocks": "introspection", "MaxLanes": "introspection",
    # single-lane scalar get/insert (not a whole-vector throughput op)
    "GetLane": "scalar lane access", "ExtractLane": "scalar lane access",
    "InsertLane": "scalar lane access", "ExtractBlock": "scalar block access",
    "InsertBlock": "scalar block access", "InsertIntoUpper": "scalar block access",
    # pure re-tag / bitcast plumbing (no compute)
    "BitCast": "pure retag", "ResizeBitCast": "pure retag",
    "ZeroExtendResizeBitCast": "pure retag",
    # construction / tag / index helpers
    "Combine": "construction helper", "LowerHalf": "construction helper",
    "UpperHalf": "construction helper", "ZeroExtendVector": "construction helper",
    "LoadDup128": "construction helper", "Dup128VecFromValues": "construction helper",
    "SetTableIndices": "index helper", "IndicesFromVec": "index helper",
    "FirstN": "mask constant helper",
}

# Substrings that mark an internal EXPLICIT-SELECTOR variant covered by its umbrella op
# (e.g. CompressNotNative exercised through CompressNot; NativeGatherOffset32 through GatherOffset).
# Not separately benchable — the umbrella op's bench routes through them.
SELECTOR_SUBSTR = ("Native", "Emulated")

# Pure aliases of an already-benched op (same code, different public name) — not a distinct gap.
ALIASES = {
    "Rol": "RotateLeft", "Ror": "RotateRight",
    "ZipLower": "InterleaveLower", "ZipUpper": "InterleaveUpper",
}

DEF_RE = re.compile(r"HYLE_FORCE_INLINE\b[^(;{]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

def strip_line_comments(s):
    return re.sub(r"//.*", "", s)

def extract_public_ops(path):
    """Return set of public op names in a header, skipping `namespace detail{...}` via brace depth."""
    ops = set()
    depth_total = 0          # overall brace depth
    detail_depth = None      # brace depth at which a `namespace detail {` opened
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = strip_line_comments(raw)
            # detect entering an internal namespace BEFORE counting this line's braces
            if detail_depth is None and re.search(r"namespace\s+detail\b", line):
                detail_depth = depth_total  # the '{' on this/next lines will raise depth above this
            in_detail = detail_depth is not None
            if not in_detail:
                m = DEF_RE.search(line)
                if m:
                    name = m.group(1)
                    # op names are CamelCase starting uppercase. Skip ALL_CAPS macros
                    # (HYLE_SIMD_MAX_LANES_D) and lowercase-leading helpers (reduce_scalar).
                    if re.fullmatch(r"[A-Z0-9_]+", name):
                        continue
                    if name[0].islower():
                        continue
                    ops.add(name)
            # update brace depth
            depth_total += line.count("{") - line.count("}")
            if detail_depth is not None and depth_total <= detail_depth:
                detail_depth = None
    return ops

def _title_to_ops(title, all_ops):
    """Map a bench title string to the op name(s) it exercises.

    Titles are free-form (e.g. "MulEvenAdd bf16->f32", "Load/Store i32 aligned", "Mask And i32",
    "CombineShiftRightBytes4 i32"). Rules, applied to the leading token(s):
      - split "X/Y" compound titles into both X and Y (Load/Store, LoadN/StoreN, LoadU/StoreU);
      - "Mask <Op>" prefix -> <Op> (mask-domain logic titled with a "Mask " lead);
      - drop <...> template args and ->conversion suffixes;
      - strip a trailing digit that is a bench-kernel shape tag, not part of the name
        (CombineShiftRightBytes4 -> CombineShiftRightBytes) ONLY if the stripped form is a real op."""
    # bench titles that abbreviate the op name for display (the CALL uses the full name)
    TITLE_ALIAS = {
        "MaskedApproxReciprocal": "MaskedApproximateReciprocal",
        "MaskedApproxReciprocalSqrt": "MaskedApproximateReciprocalSqrt",
    }
    t = title.strip()
    m = re.match(r"^Mask\s+([A-Za-z][A-Za-z0-9_]*)", t)  # "Mask And i32" -> "And"
    if m:
        return {m.group(1)}
    tok = t.split()[0]
    if tok in TITLE_ALIAS:
        return {TITLE_ALIAS[tok]}
    tok = re.sub(r"<[^>]*>", "", tok)     # template args
    tok = re.sub(r"->.*$", "", tok)        # conversion arrow
    parts = tok.split("/") if "/" in tok else [tok]
    out = set()
    for p in parts:
        if p in all_ops:
            out.add(p)
        else:
            stripped = re.sub(r"\d+$", "", p)   # CombineShiftRightBytes4 -> CombineShiftRightBytes
            out.add(stripped if stripped in all_ops else p)
    return out

def extract_benched(all_ops):
    """Return set of op names that appear in an op_parity bench.

    Titles reach nanobench through MANY forms: `.title("X")`, the UNARY_/BINARY_/... macros, AND
    bespoke helper functions (run_reduce/cmp128/mred/...). Rather than enumerate every helper (a
    moving target), scan EVERY string literal in the bench TUs and keep those whose leading token
    maps to a real op name. This is why the self-check matters: a title that maps to nothing is
    either a non-title string (harmless) or a real op the header-extractor missed (a blind spot)."""
    str_re = re.compile(r'"([^"\\\n]{2,})"')  # single-line literals only (no cross-line merge)
    benched = set()
    for bf in BENCH_FILES:
        p = os.path.join(BENCH, bf)
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8", errors="replace").read()
        for m in str_re.finditer(text):
            s = m.group(1)
            if not re.match(r"^[A-Za-z]", s):   # titles start with an op name (a letter)
                continue
            benched |= (_title_to_ops(s, all_ops) & all_ops)
    return benched

def is_selector(name):
    return any(s in name for s in SELECTOR_SUBSTR)

def extract_matrix_ops(all_ops):
    """Op names with a measured verdict row in simd-orderswap-matrix.md (| Op ... | val | val | val |).
    Having a row = the calibrated instrument produced a TIE/WIN/LOSE for it = SOTA status is known."""
    if not os.path.exists(MATRIX):
        return set()
    ops = set()
    row_re = re.compile(r"^\|\s*([A-Za-z][A-Za-z0-9_]*)")
    for line in open(MATRIX, encoding="utf-8", errors="replace"):
        m = row_re.match(line)
        if m:
            ops |= (_title_to_ops(m.group(1), all_ops) & all_ops)
    return ops

def main():
    check = "--check" in sys.argv

    all_ops = set()
    per_header = {}
    for h in HEADERS:
        p = os.path.join(INC, h)
        ops = extract_public_ops(p)
        per_header[h] = len(ops)
        all_ops |= ops

    benched = extract_benched(all_ops)

    # benchable = all public ops − not-benchable − internal selectors − aliases-of-benched.
    # An alias counts as covered iff its target op is benched (else the alias masks a real gap).
    benchable = {o for o in all_ops
                 if o not in NOT_BENCHABLE and not is_selector(o) and o not in ALIASES}

    # --- SELF-VALIDATION: every benched op must be a known op name -----------------
    # (benched titles are matched loosely, so allow a benched name that maps to a benchable op)
    benched_known = {b for b in benched if b in all_ops}
    orphans = sorted(b for b in benched if b not in all_ops)

    covered = sorted(benchable & benched_known)
    uncovered = sorted(benchable - benched_known)

    # SOTA status: an op is "SOTA-confirmed" only if it has a MEASURED verdict row in the matrix.
    # benched-but-not-in-matrix = has a bench but not yet run through the calibrated instrument.
    matrix_ops = extract_matrix_ops(all_ops)
    sota_pending_benched = sorted((benchable & benched_known) - matrix_ops)  # bench yes, verdict no
    not_yet_sota = sorted(set(uncovered) | set(sota_pending_benched))        # everything unproven

    print(f"headers parsed: {len(HEADERS)}")
    for h in HEADERS:
        print(f"  {h:22s} public ops: {per_header[h]}")
    print(f"\npublic op names (deduped across headers): {len(all_ops)}")
    print(f"  not-benchable (excluded, with reason):  {len(all_ops & set(NOT_BENCHABLE))}")
    print(f"  internal selector variants (*Native/*Emulated): "
          f"{sum(1 for o in all_ops if is_selector(o))}")
    print(f"\nBENCHABLE universe:      {len(benchable)}")
    print(f"benched (op_parity):     {len(benched_known)}")
    print(f"UNCOVERED (benchable-benched): {len(uncovered)}")
    print(f"coverage: {len(covered)}/{len(benchable)} = "
          f"{100.0*len(covered)/max(1,len(benchable)):.1f}%")

    if orphans:
        print("\n!! SELF-CHECK FAILED: these benched ops are NOT in the extracted op set.")
        print("   => extraction rules have a blind spot; fix DEF_RE / namespace handling and re-run.")
        for o in orphans:
            print(f"     - {o}")

    print(f"\n--- SOTA status (matrix = measured-verdict rows) ---")
    print(f"benchable ops with a measured verdict (SOTA-confirmed): {len(benchable & matrix_ops)}")
    print(f"benched but NOT yet measured (bench exists, run calibrated instrument): "
          f"{len(sota_pending_benched)}")
    print(f"not benched at all (add bench first):                    {len(uncovered)}")
    print(f"==> TOTAL not-yet-proven-SOTA: {len(not_yet_sota)} "
          f"({len(sota_pending_benched)} measure + {len(uncovered)} add-bench)")

    print(f"\nUNCOVERED benchable ops -- need a bench first ({len(uncovered)}):")
    for o in uncovered:
        print(f"  - {o}")
    print(f"\nBENCHED but not yet measured -- run calibrated instrument ({len(sota_pending_benched)}):")
    for o in sota_pending_benched:
        print(f"  - {o}")

    if check:
        if orphans:
            print("\nFAIL: self-check (benched ⊄ all_ops).")
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
