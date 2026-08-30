#!/usr/bin/env python3
"""SIMD op benchmark coverage - mechanical, self-validating.

The point of this script is to answer ONE question reliably: how many HYLE SIMD ops
CAN be perf-benched against a Highway/xsimd reference, and how many actually ARE?

Every prior attempt to count this by hand or by a one-shot grep was wrong (ops that were
demonstrably benched got reported "missing", inflating the gap). The fix is a self-check:

    BENCHED must be a SUBSET of the extracted public-op set.

If any op that appears in an op_parity bench is NOT in the extracted public-op set, the
EXTRACTION RULES have a blind spot (they missed that op's definition form). The script then
FAILS LOUDLY and lists the offenders, so the rules get fixed and re-run. Only once
`benched <= benchable` holds is the denominator trustworthy, and:

    covered   = benchable intersection benched
    uncovered = benchable minus benched      <- the real number.

Some construction/retag operations are deliberately excluded from the performance denominator
even though a parity title exercises them. Therefore the raw "benched op names" count can be
larger than the covered subset; the report prints both instead of pretending they are identical.

Run:  python scripts/simd-op-coverage.py          (report)
      python scripts/simd-op-coverage.py --check   (exit 1 if benched is not a subset or uncovered grows)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
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
# Historical per-op matrix ledger. An op having a source benchmark is not the same as appearing in
# this table, and a historical row is not automatically a current-HEAD SOTA certificate: v4 may
# classify the new measurement as INCONCLUSIVE. Current execution coverage is audited via --reports.
MATRIX = os.path.join(ROOT, "docs", "milestones", "simd-orderswap-matrix.md")
MANIFEST = os.path.join(ROOT, "scripts", "simd-op-manifest.json")
ROUTE_LEDGER = os.path.join(ROOT, "scripts", "simd-op-routing.json")
PROVIDER_MANIFEST = os.path.join(
    INC, "provider", "operation_manifest.h")
FAMILY_ROUTES = os.path.join(ROOT, "scripts", "simd-family-routing.json")
PROVIDER_ROUTES = os.path.join(INC, "provider", "routes_generated.h")
LEDGER_SCHEMA = "hyle.simd.operation-routing/v2"

HEADER_TARGET = {
    "vec.h": "FACADE",
    "backend/avx2.h": "AVX2",
    "backend/sse4.h": "SSE4",
    "backend/avx512.h": "AVX512",
    "backend/neon.h": "NEON",
    "backend/scalar.h": "SCALAR",
    "backend/sve.h": "SVE",
    "backend/rvv.h": "RVV",
}

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

def inline_definition_name(line):
    """Return the public CamelCase function after HYLE_FORCE_INLINE on one line.

    The old regex stopped at the first ``(`` after the inline macro. That silently
    dropped valid definitions whose return type contains a parenthesized template
    expression, for example ``Vec128<T, (N / 4)> ExtractBlock(...)``. Scan lexical
    nesting instead: the function parameter list is the first top-level CamelCase
    identifier followed by ``(``; calls in ``decltype`` or default arguments are
    nested and therefore ignored.
    """
    marker = "HYLE_FORCE_INLINE"
    marker_pos = line.find(marker)
    if marker_pos < 0:
        return None
    suffix = line[marker_pos + len(marker):]
    paren_depth = 0
    angle_depth = 0
    bracket_depth = 0
    pos = 0
    while pos < len(suffix):
        char = suffix[pos]
        if char == "(":
            paren_depth += 1
            pos += 1
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            pos += 1
            continue
        if char == "[":
            bracket_depth += 1
            pos += 1
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            pos += 1
            continue
        if char == "<" and paren_depth == 0:
            angle_depth += 1
            pos += 1
            continue
        if char == ">" and paren_depth == 0 and angle_depth:
            angle_depth -= 1
            pos += 1
            continue
        if (char.isalpha() or char == "_") and not (
                paren_depth or angle_depth or bracket_depth):
            end = pos + 1
            while end < len(suffix) and (suffix[end].isalnum() or suffix[end] == "_"):
                end += 1
            name = suffix[pos:end]
            next_pos = end
            while next_pos < len(suffix) and suffix[next_pos].isspace():
                next_pos += 1
            if (name[0].isupper() and next_pos < len(suffix) and
                    suffix[next_pos] == "("):
                return name
            pos = end
            continue
        pos += 1
    return None

def strip_line_comments(s):
    return re.sub(r"//.*", "", s)

def extract_public_op_sites(text, header):
    """Return public op definition sites, skipping `namespace detail{...}` by brace depth."""
    sites = {}
    depth_total = 0          # overall brace depth
    detail_depth = None      # brace depth at which a `namespace detail {` opened
    previous_nonempty = ""
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = strip_line_comments(raw)
        # detect entering an internal namespace BEFORE counting this line's braces
        if detail_depth is None and re.search(r"namespace\s+detail\b", line):
            detail_depth = depth_total  # the '{' on this/next lines will raise depth above this
        in_detail = detail_depth is not None
        if not in_detail:
            name = inline_definition_name(line)
            if name:
                # op names are CamelCase starting uppercase. Skip ALL_CAPS macros
                # (HYLE_SIMD_MAX_LANES_D) and lowercase-leading helpers (reduce_scalar).
                if not re.fullmatch(r"[A-Z0-9_]+", name) and not name[0].islower():
                    declaration = line.strip()
                    if previous_nonempty.startswith("template"):
                        declaration = f"{previous_nonempty} {declaration}"
                    sites.setdefault(name, []).append({
                        "id": f"{header}:{line_number}",
                        "header": header,
                        "line": line_number,
                        "declaration": re.sub(r"\s+", " ", declaration),
                    })
        # update brace depth
        depth_total += line.count("{") - line.count("}")
        if detail_depth is not None and depth_total <= detail_depth:
            detail_depth = None
        if line.strip():
            previous_nonempty = line.strip()
    return sites

def extract_public_ops(path):
    """Return set of public op names in a current-worktree header."""
    with open(path, encoding="utf-8", errors="replace") as src:
        return set(extract_public_op_sites(src.read(), os.path.relpath(path, INC)))

def git_text(commit, relative_path):
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path.replace(os.sep, '/')}"] ,
        cwd=ROOT, check=False, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read {relative_path} at {commit}: {message}")
    return result.stdout.decode("utf-8", errors="replace")

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

def extract_bench_maps(all_ops, loader=None):
    """Return op -> source title strings and op -> bench files for parity benches.

    Titles reach nanobench through MANY forms: `.title("X")`, the UNARY_/BINARY_/... macros, AND
    bespoke helper functions (run_reduce/cmp128/mred/...). Rather than enumerate every helper (a
    moving target), scan EVERY string literal in the bench TUs and keep those whose leading token
    maps to a real op name. This is why the self-check matters: a title that maps to nothing is
    either a non-title string (harmless) or a real op the header-extractor missed (a blind spot)."""
    str_re = re.compile(r'"([^"\\\n]{2,})"')  # single-line literals only (no cross-line merge)
    titles = {}
    files = {}
    for bf in BENCH_FILES:
        p = os.path.join(BENCH, bf)
        if loader is None:
            if not os.path.exists(p):
                continue
            source = open(p, encoding="utf-8", errors="replace").read()
        else:
            source = loader(f"bench/simd/{bf}")
        for m in str_re.finditer(source):
            s = m.group(1)
            if not re.match(r"^[A-Za-z]", s):   # titles start with an op name (a letter)
                continue
            for op in (_title_to_ops(s, all_ops) & all_ops):
                titles.setdefault(op, set()).add(s)
                files.setdefault(op, set()).add(bf)
    return (
        {name: sorted(values) for name, values in titles.items()},
        {name: sorted(values) for name, values in files.items()},
    )

def extract_bench_title_map(all_ops, loader=None):
    return extract_bench_maps(all_ops, loader=loader)[0]

def extract_benched(all_ops):
    """Return set of op names that appear in an op_parity bench."""
    return set(extract_bench_title_map(all_ops))

def extract_reported(report_paths, all_ops):
    """Map titles that actually appeared in raw/aggregated reports back to canonical op names."""
    title_re = re.compile(r"\|\s*relative\s*\|.*\|\s*([^|]+?)\s*$")
    reported = set()
    for path in report_paths:
        with open(path, "rb") as src:
            raw = src.read()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
        for line in raw.decode(encoding, errors="replace").splitlines():
            match = title_re.match(line)
            if match:
                reported |= (_title_to_ops(match.group(1).strip(), all_ops) & all_ops)
                continue
            # op-parity-report TSV: first field is the exact nanobench title.
            if "\t" in line and not line.startswith("op\t"):
                title = line.split("\t", 1)[0]
                if title.startswith("# "):
                    title = title[2:]
                reported |= (_title_to_ops(title, all_ops) & all_ops)
    return reported

def is_selector(name):
    return any(s in name for s in SELECTOR_SUBSTR)

def extract_matrix_ops(all_ops):
    """Op names with a historical row in simd-orderswap-matrix.md."""
    if not os.path.exists(MATRIX):
        return set()
    ops = set()
    row_re = re.compile(r"^\|\s*([A-Za-z][A-Za-z0-9_]*)")
    for line in open(MATRIX, encoding="utf-8", errors="replace"):
        m = row_re.match(line)
        if m:
            ops |= (_title_to_ops(m.group(1), all_ops) & all_ops)
    return ops

def load_manifest():
    with open(MANIFEST, encoding="utf-8") as src:
        doc = json.load(src)
    return doc.get("ops", {})

def load_manifest_at(commit):
    return json.loads(git_text(commit, "scripts/simd-op-manifest.json")).get("ops", {})

def comparison_kind(name, manifest):
    if name in NOT_BENCHABLE:
        return "STRUCTURAL"
    if is_selector(name):
        return "SELECTOR"
    item = manifest.get(name, {})
    if name in ALIASES or item.get("class") == "alias":
        return "ALIAS"
    if item.get("class") in {"synthetic", "paired_info"}:
        return "COMPOSITION"
    if item.get("class") == "algorithm":
        return "ALGORITHM_WORKLOAD"
    return "PERF_ROW"

def frozen_ledger_projection(document):
    return {
        "baseline_commit": document.get("baseline", {}).get("commit"),
        "operations": [
            {
                "name": item.get("name"),
                "comparison_kind": item.get("comparison_kind"),
                "legacy_sites": item.get("legacy_sites"),
                "legacy_targets": item.get("legacy_targets"),
                "source_bench_titles": item.get("source_bench_titles"),
                "source_bench_files": item.get("source_bench_files"),
                "classification": item.get("classification"),
            }
            for item in document.get("operations", [])
        ],
    }

def ledger_digest(document):
    payload = json.dumps(
        frozen_ledger_projection(document), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_provider_operation_manifest(document):
    """Render the compile-time operation universe from the frozen ledger.

    Route winners deliberately do not appear here yet: SIMD-15 may split one
    operation by lane type, target and compiler.  The route substrate uses this
    exhaustive list plus a fail-closed PENDING primary template, and generated
    terminal specializations are added only after qualification.
    """
    operations = document.get("operations", [])
    names = [item.get("name") for item in operations]
    if len(names) != 387 or len(names) != len(set(names)):
        raise RuntimeError(
            "provider manifest requires exactly 387 unique operation names")
    invalid = [name for name in names
               if not isinstance(name, str) or
               re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None]
    if invalid:
        raise RuntimeError(
            f"provider manifest has non-C++ operation identifiers: {invalid}")
    digest = ledger_digest(document)
    lines = [
        "/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */",
        "#pragma once",
        "",
        "// Generated by scripts/simd-op-coverage.py from the frozen SIMD-v3",
        "// operation ledger. Do not hand-edit, reorder, add, or remove rows.",
        f'#define HYLE_SIMD_OPERATION_INVENTORY_SHA256 "{digest}"',
        f"#define HYLE_SIMD_OPERATION_COUNT {len(names)}",
        "#define HYLE_SIMD_OPERATION_LIST(X) \\",
    ]
    for index, name in enumerate(names):
        continuation = " \\" if index + 1 != len(names) else ""
        lines.append(f"  X({name}){continuation}")
    lines.append("")
    return "\n".join(lines)


def validate_provider_operation_manifest(document, path):
    try:
        expected = render_provider_operation_manifest(document)
    except RuntimeError as error:
        return [str(error)]
    try:
        with open(path, encoding="utf-8") as src:
            actual = src.read()
    except OSError as error:
        return [f"cannot read generated provider operation manifest {path}: {error}"]
    if actual != expected:
        return [
            "generated provider operation manifest differs from the frozen route ledger; "
            "run scripts/simd-op-coverage.py --write-provider-manifest"
        ]
    return []


_ROUTE_CPP = {
    ("KEEP_HYLE", "HYLE"): ("Provider::kHyle", "Decision::kKeepHyle"),
    ("ROUTE_HIGHWAY", "HIGHWAY"): (
        "Provider::kHighway", "Decision::kRouteHighway"),
    ("ROUTE_XSIMD", "XSIMD"): ("Provider::kXsimd", "Decision::kRouteXsimd"),
    ("REMOVE_REDUNDANT", "COMPOSITION"): (
        "Provider::kNone", "Decision::kRemoveRedundant"),
    ("REMOVE_REDUNDANT", "NONE"): (
        "Provider::kNone", "Decision::kRemoveRedundant"),
}

_TYPE_CPP = {
    "i8": "std::int8_t", "u8": "std::uint8_t",
    "i16": "std::int16_t", "u16": "std::uint16_t",
    "i32": "std::int32_t", "u32": "std::uint32_t",
    "i64": "std::int64_t", "u64": "std::uint64_t",
    "f32": "float", "f64": "double",
}

_TARGET_CPP = {
    "SCALAR": "Target::kScalar", "SSE4": "Target::kSse4",
    "AVX2": "Target::kAvx2", "AVX512": "Target::kAvx512",
    "NEON": "Target::kNeon", "SVE": "Target::kSve", "RVV": "Target::kRvv",
}

_COMPILER_CPP = {
    "MSVC": "Compiler::kMsvc", "CLANG_CL": "Compiler::kClangCl",
    "CLANG": "Compiler::kClang", "APPLE_CLANG": "Compiler::kAppleClang",
    "GCC": "Compiler::kGcc", "OTHER": "Compiler::kOther",
}


def _route_cpp(decision, provider):
    key = (decision, provider)
    if key not in _ROUTE_CPP:
        raise RuntimeError(f"cannot generate terminal provider route {key}")
    return _ROUTE_CPP[key]


def _variant_cpp(selector):
    variant = selector.get("variant", {})
    kind = variant.get("kind")
    if kind == "ANY":
        if variant.get("contract_tags", []):
            raise RuntimeError("ANY provider route has unsupported contract tags")
        return "AnyVariant"
    if kind == "LANE":
        lane = variant.get("lane")
        if lane not in _TYPE_CPP:
            raise RuntimeError(f"provider route cannot express lane type {lane!r}")
        if variant.get("contract_tags", []):
            raise RuntimeError("LANE provider route has unsupported contract tags")
        return f"LaneVariant<{_TYPE_CPP[lane]}>"
    if kind == "CONVERSION":
        source = variant.get("from")
        destination = variant.get("to")
        if source not in _TYPE_CPP or destination not in _TYPE_CPP:
            raise RuntimeError("provider conversion route has unsupported lane types")
        tags = variant.get("contract_tags", [])
        if tags == []:
            contract = "ConversionContract::kDefault"
        elif tags == ["two-input-saturating"]:
            contract = "ConversionContract::kTwoInputSaturating"
        else:
            raise RuntimeError(f"provider conversion route has unsupported tags {tags}")
        return (
            f"ConversionVariant<{_TYPE_CPP[source]}, {_TYPE_CPP[destination]}, "
            f"{contract}>")
    if kind == "INVARIANT_DIVISOR":
        lane = variant.get("lane")
        if lane not in _TYPE_CPP:
            raise RuntimeError(f"provider invariant route cannot express lane type {lane!r}")
        if variant.get("contract_tags", []) != ["reusable-precomputed-divisor"]:
            raise RuntimeError("provider invariant route lacks its reusable-divisor contract")
        return f"InvariantDivisorVariant<{_TYPE_CPP[lane]}>"
    raise RuntimeError(f"provider route has unsupported selector kind {kind!r}")


def _render_default_route(key, decision, provider):
    provider_cpp, decision_cpp = _route_cpp(decision, provider)
    return [
        "template <typename Variant, Target kTarget, Compiler kCompiler,",
        "          std::uint32_t kCompilerVersion>",
        f"struct RouteFor<RouteKey<{key}>, Variant, kTarget, kCompiler, kCompilerVersion> final {{",
        f"  inline static constexpr Route value{{{provider_cpp}, {decision_cpp}}};",
        "};",
        "",
    ]


def _render_exact_routes(key, route):
    selector = route.get("selector")
    if not isinstance(selector, dict):
        raise RuntimeError(f"exact provider route {key} lacks a selector")
    variant_cpp = _variant_cpp(selector)
    provider_cpp, decision_cpp = _route_cpp(
        route.get("decision"), route.get("provider"))
    lines = []
    for target in selector.get("runtime_targets", []):
        if target not in _TARGET_CPP:
            raise RuntimeError(f"provider route has unsupported target {target!r}")
        for compiler_range in selector.get("compiler_ranges", []):
            compiler = compiler_range.get("kind")
            if compiler not in _COMPILER_CPP:
                raise RuntimeError(f"provider route has unsupported compiler {compiler!r}")
            minimum = compiler_range.get("min")
            maximum = compiler_range.get("max")
            if not isinstance(minimum, int) or not isinstance(maximum, int):
                raise RuntimeError("provider route compiler range is not integral")
            lines.extend([
                "template <std::uint32_t kCompilerVersion>",
                f"  requires (kCompilerVersion >= {minimum}u &&",
                f"            kCompilerVersion <= {maximum}u)",
                f"struct RouteFor<RouteKey<{key}>, {variant_cpp}, {_TARGET_CPP[target]},",
                f"                {_COMPILER_CPP[compiler]}, kCompilerVersion> final {{",
                f"  inline static constexpr Route value{{{provider_cpp}, {decision_cpp}}};",
                "};",
                "",
            ])
    if not lines:
        raise RuntimeError(f"exact provider route {key} expands to no specialization")
    return lines


def render_provider_routes(document, family_path):
    """Generate compile-time terminal routes from the reviewed ledgers."""
    with open(family_path, encoding="utf-8") as src:
        families = json.load(src)
    if families.get("schema") != "hyle.simd.family-routing/v1":
        raise RuntimeError("SIMD family route ledger has an unsupported schema")
    expected_revisions = {
        provider: document.get("candidate_providers", {}).get(provider)
        for provider in ("HIGHWAY", "XSIMD")
    }
    if families.get("provider_revisions") != expected_revisions:
        raise RuntimeError("SIMD family routes do not match locked provider revisions")
    family_rows = families.get("families")
    if not isinstance(family_rows, list) or not family_rows:
        raise RuntimeError("SIMD family route ledger is empty")
    family_names = [row.get("name") for row in family_rows]
    if len(family_names) != len(set(family_names)) or any(
            not isinstance(name, str) or
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            for name in family_names):
        raise RuntimeError("SIMD family routes contain invalid or duplicate names")

    lines = [
        "/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */",
        "#pragma once",
        "",
        "// Generated by scripts/simd-op-coverage.py from terminal SIMD-v3",
        "// operation and family route ledgers. Include only from route.h after",
        "// the RouteFor/RouteKey primary templates; do not hand-edit.",
        "",
        "enum class Family : std::uint8_t {",
    ]
    lines.extend(f"  k{name}," for name in family_names)
    lines.extend(["};", ""])

    for operation in document.get("operations", []):
        name = operation.get("name")
        routes = operation.get("routes", [])
        defaults = [route for route in routes if "selector" not in route]
        if not defaults:
            raise RuntimeError(f"{name}: terminal provider routes lack a default remainder")
        default_pairs = {
            (route.get("decision"), route.get("provider")) for route in defaults
        }
        if len(default_pairs) != 1 or any("PENDING" in pair for pair in default_pairs):
            raise RuntimeError(f"{name}: default provider routes are not one terminal decision")
        decision, provider = next(iter(default_pairs))
        key = f"Operation::k{name}"
        lines.extend(_render_default_route(key, decision, provider))
        for route in routes:
            if "selector" in route:
                lines.extend(_render_exact_routes(key, route))

    for family in family_rows:
        name = family["name"]
        default = family.get("default", {})
        key = f"Family::k{name}"
        lines.extend(_render_default_route(
            key, default.get("decision"), default.get("provider")))
        exceptions = family.get("exceptions", [])
        if not isinstance(exceptions, list):
            raise RuntimeError(f"family {name} exceptions must be an array")
        for route in exceptions:
            evidence = route.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise RuntimeError(f"family {name} exception lacks evidence")
            lines.extend(_render_exact_routes(key, route))
    return "\n".join(lines)


def validate_provider_routes(document, family_path, generated_path):
    try:
        expected = render_provider_routes(document, family_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError,
            RuntimeError) as error:
        return [str(error)]
    try:
        with open(generated_path, encoding="utf-8") as src:
            actual = src.read()
    except OSError as error:
        return [f"cannot read generated provider routes {generated_path}: {error}"]
    if actual != expected:
        return [
            "generated provider routes differ from the terminal ledgers; "
            "run scripts/simd-op-coverage.py --write-provider-routes"
        ]
    return []

def build_route_ledger(commit):
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("--baseline-commit must be a full lowercase Git commit id")

    sites_by_op = {}
    for header in HEADERS:
        relative = f"include/hyle/simd/{header}"
        header_sites = extract_public_op_sites(git_text(commit, relative), header)
        for name, sites in header_sites.items():
            sites_by_op.setdefault(name, []).extend(sites)

    all_ops = set(sites_by_op)
    title_map, bench_file_map = extract_bench_maps(
        all_ops, loader=lambda path: git_text(commit, path))
    manifest = load_manifest_at(commit)
    operations = []
    for name in sorted(all_ops):
        sites = sorted(sites_by_op[name], key=lambda item: (item["header"], item["line"]))
        targets = sorted({HEADER_TARGET[item["header"]] for item in sites})
        manifest_item = manifest.get(name, {})
        classification = {
            "legacy_manifest_class": manifest_item.get("class"),
            "covered_by": manifest_item.get("covered_by", []),
            "legacy_evidence": manifest_item.get("evidence", ""),
            "structural_reason": NOT_BENCHABLE.get(name, ""),
            "source_alias_of": ALIASES.get(name, ""),
        }
        site_ids = [item["id"] for item in sites]
        operations.append({
            "name": name,
            "comparison_kind": comparison_kind(name, manifest),
            "legacy_sites": sites,
            "legacy_targets": targets,
            "source_bench_titles": title_map.get(name, []),
            "source_bench_files": bench_file_map.get(name, []),
            "classification": classification,
            "candidate_audit": {
                "HIGHWAY": {"availability": "PENDING", "evidence": []},
                "XSIMD": {"availability": "PENDING", "evidence": []},
                "HYLE": {"availability": "SOURCE_PRESENT", "evidence": ["legacy_sites"]},
            },
            "audit_group": "PENDING",
            "public_mapping": "PENDING",
            "variant_audit": "PENDING",
            "routes": [{
                "scope_id": "legacy-all",
                "legacy_site_ids": site_ids,
                "variant_scope": "all types and overloads represented by the frozen legacy sites",
                "targets": targets,
                "provider_assessments": {
                    "HIGHWAY": {"contract": "PENDING", "evidence": []},
                    "XSIMD": {"contract": "PENDING", "evidence": []},
                    "HYLE": {"contract": "PENDING", "evidence": []},
                },
                "decision": "PENDING",
                "provider": "PENDING",
                "evidence_group": "PENDING",
                "evidence": [],
            }],
        })

    kind_counts = {}
    for item in operations:
        kind = item["comparison_kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    document = {
        "schema": LEDGER_SCHEMA,
        "generated_by": "scripts/simd-op-coverage.py",
        "baseline": {
            "commit": commit,
            "sources": [f"include/hyle/simd/{header}" for header in HEADERS],
            "operation_count": len(operations),
            "definition_site_count": sum(len(item["legacy_sites"]) for item in operations),
        },
        "candidate_providers": {
            "HIGHWAY": "f503c485cec6b4f2f596fa60d83484797c9eeb5c",
            "XSIMD": "3b3151ff665514a00a437cc985f47b7de99b772b",
            "HYLE": commit,
        },
        "comparison_kind_counts": dict(sorted(kind_counts.items())),
        "evidence_groups": {},
        "operations": operations,
    }
    document["baseline"]["inventory_sha256"] = ledger_digest(document)
    return document


def refresh_frozen_inventory(ledger_path):
    """Apply an additive parser erratum without discarding qualified route work.

    This is deliberately narrower than regenerating the ledger. Operation names may
    not change, an existing site may not disappear, and every operation gaining a
    site must still have one PENDING ``legacy-all`` route. Thus a parser correction
    cannot silently widen an already-qualified decision.
    """
    ledger_path = os.path.abspath(ledger_path)
    with open(ledger_path, encoding="utf-8") as src:
        document = json.load(src)
    commit = document.get("baseline", {}).get("commit")
    expected = build_route_ledger(commit)
    operations = document.get("operations", [])
    expected_operations = expected.get("operations", [])
    names = [item.get("name") for item in operations]
    expected_names = [item.get("name") for item in expected_operations]
    if names != expected_names:
        raise RuntimeError("inventory refresh may not add, remove or reorder operations")
    baseline = document.get("baseline", {})
    if (baseline.get("definition_site_count") ==
            expected["baseline"]["definition_site_count"] and
            baseline.get("inventory_sha256") ==
            expected["baseline"]["inventory_sha256"]):
        _, errors = validate_route_ledger(ledger_path)
        if errors:
            raise RuntimeError("current inventory does not validate: " + "; ".join(errors))
        return []

    immutable_fields = (
        "name", "comparison_kind", "legacy_sites", "legacy_targets",
        "source_bench_titles", "source_bench_files", "classification",
    )
    additions = []
    for operation, expected_operation in zip(operations, expected_operations):
        old_ids = {site["id"] for site in operation.get("legacy_sites", [])}
        new_ids = {site["id"] for site in expected_operation.get("legacy_sites", [])}
        removed = sorted(old_ids - new_ids)
        added = sorted(new_ids - old_ids)
        if removed:
            raise RuntimeError(
                f"{operation['name']}: inventory refresh would remove sites {removed}")
        if added:
            routes = operation.get("routes", [])
            if (len(routes) != 1 or routes[0].get("scope_id") != "legacy-all" or
                    routes[0].get("decision") != "PENDING"):
                raise RuntimeError(
                    f"{operation['name']}: added sites cannot widen a qualified/split route")
            routes[0]["legacy_site_ids"] = [
                site["id"] for site in expected_operation["legacy_sites"]]
            routes[0]["targets"] = expected_operation["legacy_targets"]
            additions.append({"operation": operation["name"], "site_ids": added})
        for field in immutable_fields:
            operation[field] = expected_operation[field]

    old_baseline = document.get("baseline", {})
    old_count = old_baseline.get("definition_site_count")
    old_digest = old_baseline.get("inventory_sha256")
    document["baseline"] = expected["baseline"]
    document.setdefault("inventory_errata", []).append({
        "kind": "ADDITIVE_INLINE_RETURN_TYPE_PARSER_FIX",
        "old_definition_site_count": old_count,
        "new_definition_site_count": expected["baseline"]["definition_site_count"],
        "old_inventory_sha256": old_digest,
        "new_inventory_sha256": expected["baseline"]["inventory_sha256"],
        "added_sites": additions,
        "safety_rule": "same 387 operations; zero removed sites; only PENDING legacy-all routes widened",
    })

    temp_path = ledger_path + ".inventory-refresh.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as dst:
            json.dump(document, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
        _, errors = validate_route_ledger(temp_path)
        if errors:
            raise RuntimeError("inventory refresh would invalidate ledger: " + "; ".join(errors))
        os.replace(temp_path, ledger_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return additions

def validate_route_ledger(path, require_decisions=False):
    errors = []
    try:
        with open(path, encoding="utf-8") as src:
            document = json.load(src)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"cannot read route ledger {path}: {error}"]

    if document.get("schema") != LEDGER_SCHEMA:
        errors.append(f"invalid route ledger schema {document.get('schema')!r}")
    baseline = document.get("baseline", {})
    commit = baseline.get("commit", "")
    try:
        expected = build_route_ledger(commit)
    except RuntimeError as error:
        return document, [str(error)]

    operations = document.get("operations")
    if not isinstance(operations, list):
        return document, errors + ["route ledger operations must be an array"]
    names = [item.get("name") for item in operations if isinstance(item, dict)]
    if len(names) != len(set(names)):
        errors.append("route ledger contains duplicate operation names")
    expected_names = [item["name"] for item in expected["operations"]]
    if names != expected_names:
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        errors.append(f"route ledger op set/order differs from baseline; missing={missing}, extra={extra}")
    declared_count = baseline.get("operation_count")
    if declared_count != len(operations) or declared_count != 387:
        errors.append(
            f"route ledger must contain the frozen 387 operations, got declared={declared_count} "
            f"actual={len(operations)}")
    actual_site_count = sum(
        len(item.get("legacy_sites", [])) for item in operations if isinstance(item, dict))
    expected_site_count = expected.get("baseline", {}).get("definition_site_count")
    declared_site_count = baseline.get("definition_site_count")
    if (declared_site_count != actual_site_count or
            declared_site_count != expected_site_count):
        errors.append(
            "route ledger definition-site count differs from the frozen Git sources; "
            f"declared={declared_site_count} actual={actual_site_count} "
            f"expected={expected_site_count}")
    expected_digest = ledger_digest(expected)
    if baseline.get("inventory_sha256") != expected_digest:
        errors.append("route ledger baseline inventory digest does not match the frozen Git sources")
    if ledger_digest(document) != expected_digest:
        errors.append("route ledger immutable operation/source projection was edited or is incomplete")

    evidence_groups = document.get("evidence_groups", {})
    if not isinstance(evidence_groups, dict):
        errors.append("route ledger evidence_groups must be an object")
        evidence_groups = {}
    for group_id, group in evidence_groups.items():
        if not isinstance(group, dict) or group.get("status") not in {"PLANNED", "QUALIFIED"}:
            errors.append(f"invalid evidence group {group_id!r}")
            continue
        if not isinstance(group.get("contract"), str) or not group.get("contract"):
            errors.append(f"evidence group {group_id!r} lacks a contract")
        if group.get("status") == "QUALIFIED" and (
                not isinstance(group.get("evidence"), list) or not group.get("evidence")):
            errors.append(f"qualified evidence group {group_id!r} lacks evidence")
    allowed_decisions = {
        "PENDING", "KEEP_HYLE", "ROUTE_HIGHWAY", "ROUTE_XSIMD", "REMOVE_REDUNDANT"
    }
    allowed_providers = {"PENDING", "HYLE", "HIGHWAY", "XSIMD", "COMPOSITION", "NONE"}
    allowed_availability = {
        "PENDING", "SOURCE_HIT", "DIRECT", "ADAPTER", "COMPOSITION", "UNAVAILABLE",
        "SOURCE_PRESENT"
    }
    terminal_availability = {
        "DIRECT", "ADAPTER", "COMPOSITION", "UNAVAILABLE", "SOURCE_PRESENT"
    }
    allowed_contracts = {
        "PENDING", "SAME_CONTRACT", "ADAPTED_SAME_CONTRACT",
        "COMPOSED_SAME_CONTRACT", "NOT_AVAILABLE", "NOT_COMPARABLE",
    }
    terminal_contracts = allowed_contracts - {"PENDING"}
    decision_provider = {
        "KEEP_HYLE": {"HYLE"},
        "ROUTE_HIGHWAY": {"HIGHWAY"},
        "ROUTE_XSIMD": {"XSIMD"},
        "REMOVE_REDUNDANT": {"COMPOSITION", "NONE"},
    }
    pending_ops = []
    for item in operations:
        if not isinstance(item, dict):
            errors.append("route ledger operation row must be an object")
            continue
        name = item.get("name", "<unknown>")
        audit_group = item.get("audit_group")
        if not isinstance(audit_group, str) or not audit_group:
            errors.append(f"{name}: audit_group must be a non-empty string")
        public_mapping = item.get("public_mapping")
        if not isinstance(public_mapping, str) or not public_mapping:
            errors.append(f"{name}: public_mapping must be a non-empty string")
        variant_audit = item.get("variant_audit")
        if variant_audit not in {"PENDING", "INVENTORIED", "COMPLETE"}:
            errors.append(f"{name}: invalid variant_audit {variant_audit!r}")
        candidate_audit = item.get("candidate_audit")
        if not isinstance(candidate_audit, dict) or set(candidate_audit) != {
                "HIGHWAY", "XSIMD", "HYLE"}:
            errors.append(f"{name}: candidate_audit must cover HIGHWAY, XSIMD and HYLE")
            candidate_audit = {}
        for provider_name, candidate in candidate_audit.items():
            availability = candidate.get("availability") if isinstance(candidate, dict) else None
            if availability not in allowed_availability:
                errors.append(f"{name}: invalid {provider_name} availability {availability!r}")
            if require_decisions and availability not in terminal_availability:
                errors.append(
                    f"{name}: {provider_name} candidate availability is not terminal: "
                    f"{availability}")
            if require_decisions and (not isinstance(candidate.get("evidence"), list) or
                                      not candidate.get("evidence")):
                errors.append(f"{name}: {provider_name} candidate audit lacks evidence")
        legacy_ids = {site.get("id") for site in item.get("legacy_sites", [])}
        routes = item.get("routes")
        if not isinstance(routes, list) or not routes:
            errors.append(f"{name}: routes must be a non-empty array")
            continue
        routed_ids = set()
        scope_ids = set()
        item_pending = (audit_group == "PENDING" or public_mapping == "PENDING" or
                        variant_audit != "COMPLETE")
        for route in routes:
            scope_id = route.get("scope_id")
            if not isinstance(scope_id, str) or not scope_id:
                errors.append(f"{name}: route scope_id must be a non-empty string")
            elif scope_id in scope_ids:
                errors.append(f"{name}: duplicate route scope_id {scope_id!r}")
            else:
                scope_ids.add(scope_id)
            variant_scope = route.get("variant_scope")
            if not isinstance(variant_scope, str) or not variant_scope:
                errors.append(f"{name}/{scope_id}: variant_scope must be a non-empty string")
            selector = route.get("selector")
            if selector is not None:
                if not isinstance(selector, dict):
                    errors.append(f"{name}/{scope_id}: selector must be an object")
                else:
                    variant = selector.get("variant")
                    if not isinstance(variant, dict) or variant.get("kind") not in {
                            "ANY", "LANE", "CONVERSION", "INVARIANT_DIVISOR"}:
                        errors.append(
                            f"{name}/{scope_id}: selector variant must have a supported kind")
                    else:
                        kind = variant["kind"]
                        lane_types = {
                            "i8", "u8", "i16", "u16", "i32", "u32",
                            "i64", "u64", "f16", "bf16", "f32", "f64",
                        }
                        if kind in {"LANE", "INVARIANT_DIVISOR"} and (
                                variant.get("lane") not in lane_types):
                            errors.append(
                                f"{name}/{scope_id}: selector has an invalid lane type")
                        if kind == "CONVERSION" and (
                                variant.get("from") not in lane_types or
                                variant.get("to") not in lane_types):
                            errors.append(
                                f"{name}/{scope_id}: conversion selector has invalid types")
                        tags = variant.get("contract_tags", [])
                        if not isinstance(tags, list) or not all(
                                isinstance(tag, str) and tag for tag in tags):
                            errors.append(
                                f"{name}/{scope_id}: selector contract_tags must be strings")
                    compiler_ranges = selector.get("compiler_ranges")
                    if not isinstance(compiler_ranges, list) or not compiler_ranges:
                        errors.append(
                            f"{name}/{scope_id}: selector compiler_ranges must be non-empty")
                    else:
                        for compiler_range in compiler_ranges:
                            if (not isinstance(compiler_range, dict) or
                                    compiler_range.get("kind") not in {
                                        "MSVC", "CLANG_CL", "CLANG",
                                        "APPLE_CLANG", "GCC", "OTHER"} or
                                    not isinstance(compiler_range.get("min"), int) or
                                    not isinstance(compiler_range.get("max"), int) or
                                    compiler_range["min"] > compiler_range["max"]):
                                errors.append(
                                    f"{name}/{scope_id}: invalid selector compiler range")
                    runtime_targets = selector.get("runtime_targets")
                    if (not isinstance(runtime_targets, list) or not runtime_targets or
                            not all(target in {
                                "SCALAR", "SSE4", "AVX2", "AVX512",
                                "NEON", "SVE", "RVV"} for target in runtime_targets)):
                        errors.append(
                            f"{name}/{scope_id}: selector runtime_targets must be supported targets")
            route_ids = set(route.get("legacy_site_ids", []))
            if not route_ids:
                errors.append(f"{name}/{scope_id}: route must own at least one legacy site")
            unknown_ids = sorted(route_ids - legacy_ids)
            if unknown_ids:
                errors.append(f"{name}: route references unknown legacy sites {unknown_ids}")
            routed_ids |= route_ids
            targets = route.get("targets")
            if not isinstance(targets, list) or not targets:
                errors.append(f"{name}/{scope_id}: targets must be a non-empty array")
            else:
                unknown_targets = sorted(set(targets) - set(item.get("legacy_targets", [])))
                if unknown_targets:
                    errors.append(
                        f"{name}/{scope_id}: route references unknown targets {unknown_targets}")

            assessments = route.get("provider_assessments")
            if not isinstance(assessments, dict) or set(assessments) != {
                    "HIGHWAY", "XSIMD", "HYLE"}:
                errors.append(
                    f"{name}/{scope_id}: provider_assessments must cover HIGHWAY, XSIMD and HYLE")
                assessments = {}
            for provider_name, assessment in assessments.items():
                contract = assessment.get("contract") if isinstance(assessment, dict) else None
                if contract not in allowed_contracts:
                    errors.append(
                        f"{name}/{scope_id}: invalid {provider_name} contract {contract!r}")
                if contract == "PENDING":
                    item_pending = True
                if require_decisions and contract not in terminal_contracts:
                    errors.append(
                        f"{name}/{scope_id}: {provider_name} contract is not terminal: {contract}")
                if require_decisions and (
                        not isinstance(assessment.get("evidence"), list) or
                        not assessment.get("evidence")):
                    errors.append(
                        f"{name}/{scope_id}: {provider_name} contract lacks evidence")
            decision = route.get("decision")
            provider = route.get("provider")
            if decision not in allowed_decisions:
                errors.append(f"{name}: invalid route decision {decision!r}")
            if provider not in allowed_providers:
                errors.append(f"{name}: invalid route provider {provider!r}")
            if decision in decision_provider and provider not in decision_provider[decision]:
                errors.append(f"{name}: decision {decision} is inconsistent with provider {provider}")
            selected_contracts = {
                "SAME_CONTRACT", "ADAPTED_SAME_CONTRACT", "COMPOSED_SAME_CONTRACT"
            }
            selected_assessment = assessments.get(provider) if provider in assessments else None
            if decision in {"KEEP_HYLE", "ROUTE_HIGHWAY", "ROUTE_XSIMD"} and (
                    not isinstance(selected_assessment, dict) or
                    selected_assessment.get("contract") not in selected_contracts):
                errors.append(
                    f"{name}/{scope_id}: selected provider {provider} lacks a qualified contract")
            group = route.get("evidence_group")
            if decision == "PENDING" or provider == "PENDING" or group == "PENDING":
                item_pending = True
            elif group not in evidence_groups:
                errors.append(f"{name}: unknown evidence group {group!r}")
            elif require_decisions and evidence_groups[group].get("status") != "QUALIFIED":
                errors.append(f"{name}: evidence group {group!r} is not QUALIFIED")
            evidence = route.get("evidence")
            if require_decisions and (not isinstance(evidence, list) or not evidence):
                errors.append(f"{name}: terminal route lacks per-operation evidence references")
        missing_sites = sorted(legacy_ids - routed_ids)
        if missing_sites:
            errors.append(f"{name}: legacy definition sites have no route {missing_sites}")
        if item_pending:
            pending_ops.append(name)

    if require_decisions and pending_ops:
        errors.append(f"terminal route ledger still has {len(pending_ops)} PENDING operations: " +
                      ", ".join(pending_ops))
    return document, errors

def camel_to_snake(name):
    step1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1).lower()

def provider_source_files(root, provider):
    base = os.path.join(root, "hwy") if provider == "HIGHWAY" else os.path.join(root, "include", "xsimd")
    suffixes = (".h", ".inc") if provider == "HIGHWAY" else (".hpp", ".h")
    result = []
    for directory, _, files in os.walk(base):
        for filename in files:
            if filename.endswith(suffixes):
                result.append(os.path.join(directory, filename))
    return sorted(result)


def provider_public_source_files(root, provider):
    """Return the narrow, callable public surface used to triage source hits.

    Highway intentionally defines its public target API in the per-architecture
    headers, so its full indexed set is already the relevant surface. xsimd has
    a central public API plus batch operators; hits found only in arch/common
    implementation files must not be presented as directly callable candidates.
    This remains discovery, not a semantic verdict.
    """
    if provider == "HIGHWAY":
        return provider_source_files(root, provider)
    relative_paths = (
        "include/xsimd/types/xsimd_api.hpp",
        "include/xsimd/types/xsimd_batch.hpp",
    )
    return [os.path.join(root, path) for path in relative_paths
            if os.path.exists(os.path.join(root, path))]

def provider_tokens(name, provider):
    if provider == "HIGHWAY":
        return [name]
    # Exact xsimd spellings whose superficially similar names carry a different
    # contract. In particular rotate_left/right move whole batch elements,
    # whereas HYLE RotateLeft/Right and Rol/Ror rotate bits within each lane.
    # Keep the wrong same-name token out of discovery instead of relying on a
    # later reviewer to notice the false positive.
    exact_tokens = {
        "Rol": ["rotl"],
        "Ror": ["rotr"],
        "RotateLeft": ["rotl"],
        "RotateLeftSame": ["rotl"],
        "RotateRight": ["rotr"],
        "RotateRightSame": ["rotr"],
        "ZipLower": ["zip_lo"],
        "ZipUpper": ["zip_hi"],
    }
    if name in exact_tokens:
        return exact_tokens[name]
    operator_tokens = {
        "Add": ["operator+"], "Sub": ["operator-"], "Mul": ["operator*"],
        "Div": ["operator/"], "Eq": ["operator=="], "Ne": ["operator!="],
        "Lt": ["operator<"], "Le": ["operator<="], "Gt": ["operator>"],
        "Ge": ["operator>="], "And": ["operator&"], "Or": ["operator|"],
        "Xor": ["operator^"], "Not": ["operator~"],
    }
    semantic_tokens = {
        "AllFalse": ["none"], "AllTrue": ["all"], "CountTrue": ["count"],
        "AndNot": ["bitwise_andnot"], "BitwiseIfThenElse": ["select"],
        "IfThenElse": ["select"], "IfThenElseZero": ["select"],
        "IfThenZeroElse": ["select"], "RebindMask": ["batch_bool_cast"],
        "MaskFromVec": ["bitwise_cast"], "VecFromMask": ["bitwise_cast"],
        "BitCast": ["bitwise_cast"], "ResizeBitCast": ["bitwise_cast"],
        "ConvertTo": ["batch_cast"], "ConvertInRangeTo": ["batch_cast"],
        "PromoteTo": ["widen", "batch_cast"], "DemoteTo": ["batch_cast"],
        "AverageRound": ["avgr"], "Clamp": ["clip"],
        "CopySign": ["copysign"], "SignBit": ["bitofsign"],
        "MaxNumber": ["fmax"], "MinNumber": ["fmin"],
        "MulHigh": ["mul_hi"], "MulHighLow": ["mul_hilo"],
        "SaturatedAdd": ["sadd"], "SaturatedSub": ["ssub"],
        "MulAdd": ["fma"], "NegMulAdd": ["fnma"], "MulSub": ["fms"],
        "NegMulSub": ["fnms"], "ReduceSum": ["reduce_add"],
        "MinOfLanes": ["reduce_min"], "MaxOfLanes": ["reduce_max"],
        "ReduceMul": ["reduce_mul"], "NearestInt": ["nearbyint"],
        "RoundToInt": ["nearbyint_as_int"], "Round": ["round"],
        "Trunc": ["trunc"], "PopulationCount": ["popcount"],
        "LeadingZeroCount": ["countl_zero"], "TrailingZeroCount": ["countr_zero"],
        "Load": ["load_aligned"], "LoadU": ["load_unaligned"],
        "Store": ["store_aligned"], "StoreU": ["store_unaligned"],
        "GatherIndex": ["gather"], "ScatterIndex": ["scatter"],
        "InterleaveLower": ["zip_lo"], "InterleaveUpper": ["zip_hi"],
        "TableLookupLanes": ["swizzle", "shuffle"],
        "Slide1Down": ["slide_right"], "Slide1Up": ["slide_left"],
        "ShiftLeft": ["bitwise_lshift"], "ShiftRight": ["bitwise_rshift"],
        "Shl": ["bitwise_lshift"], "Shr": ["bitwise_rshift"],
        "ComplexConj": ["conj"], "ReciprocalNewton": ["reciprocal"],
        "ReciprocalSqrtNewton": ["rsqrt"],
    }
    return sorted(set([camel_to_snake(name), name.lower()] +
                      operator_tokens.get(name, []) + semantic_tokens.get(name, [])))

def discover_provider_sources(ledger_path, roots):
    with open(ledger_path, encoding="utf-8") as src:
        document = json.load(src)
    discovery = document.setdefault("provider_discovery", {})
    for provider, root in roots.items():
        root = os.path.abspath(root)
        result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], check=False,
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"cannot identify {provider} source checkout at {root}")
        revision = result.stdout.strip()
        expected_revision = document.get("candidate_providers", {}).get(provider)
        if revision != expected_revision:
            raise RuntimeError(
                f"{provider} checkout is {revision}, expected locked {expected_revision}")
        files = provider_source_files(root, provider)
        if not files:
            raise RuntimeError(f"no {provider} headers found under {root}")
        public_files = {
            os.path.normcase(os.path.normpath(path))
            for path in provider_public_source_files(root, provider)
        }
        indexed = []
        for path in files:
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, encoding="utf-8", errors="replace") as src:
                indexed.append((
                    relative,
                    os.path.normcase(os.path.normpath(path)) in public_files,
                    src.read().splitlines()))
        discovery[provider] = {
            "revision": revision,
            "header_count": len(files),
            "public_header_count": len(public_files),
            "method": "case-aware source token discovery only; hits are not semantic equivalence",
        }
        operations = document.get("operations", [])
        tokens_by_op = {
            operation["name"]: provider_tokens(operation["name"], provider)
            for operation in operations
        }
        ops_by_token = {}
        for name, tokens in tokens_by_op.items():
            for token in tokens:
                ops_by_token.setdefault(token, set()).add(name)
        call_re = re.compile(
            r"(?<![A-Za-z0-9_])((?:operator\s*(?:==|!=|<=|>=|[+\-*/<>&|^~]))|"
            r"[A-Za-z_][A-Za-z0-9_]*)\s*\(")
        hits_by_op = {operation["name"]: [] for operation in operations}
        counts_by_op = {operation["name"]: 0 for operation in operations}
        public_hits_by_op = {operation["name"]: [] for operation in operations}
        public_counts_by_op = {operation["name"]: 0 for operation in operations}
        for relative, is_public, lines in indexed:
            for line_number, line in enumerate(lines, start=1):
                called = {
                    match.group(1).replace(" ", "") for match in call_re.finditer(line)
                }
                for token in called:
                    for name in ops_by_token.get(token, ()):
                        counts_by_op[name] += 1
                        if len(hits_by_op[name]) < 12:
                            hits_by_op[name].append(
                                f"{provider.lower()}@{revision}:{relative}:{line_number}")
                        if is_public:
                            public_counts_by_op[name] += 1
                            if len(public_hits_by_op[name]) < 12:
                                public_hits_by_op[name].append(
                                    f"{provider.lower()}@{revision}:{relative}:{line_number}")
        for operation in operations:
            name = operation["name"]
            tokens = tokens_by_op[name]
            hits = hits_by_op[name]
            total_hits = counts_by_op[name]
            candidate = operation["candidate_audit"][provider]
            candidate["source_tokens"] = tokens
            candidate["source_hit_count"] = total_hits
            candidate["source_hits"] = hits
            candidate["public_source_hit_count"] = public_counts_by_op[name]
            candidate["public_source_hits"] = public_hits_by_op[name]
            if document.get("audit_stage") != "GROUPED":
                candidate["availability"] = "SOURCE_HIT" if total_hits else "PENDING"
                candidate["evidence"] = hits
            elif (total_hits and candidate.get("availability") == "UNAVAILABLE" and
                  candidate.get("evidence") == [
                      f"provider_discovery:{provider}:no-same-contract-source-token"]):
                # A corrected spelling may invalidate a discovery-derived
                # negative after grouping. Reopen it as SOURCE_HIT; never
                # promote the hit to a semantic adapter/direct verdict here.
                candidate["availability"] = "SOURCE_HIT"
                candidate["evidence"] = hits
        if document.get("audit_stage") == "GROUPED":
            for operation in operations:
                operation["audit_group"] = (
                    f"{operation['comparison_kind']}|"
                    f"HIGHWAY={operation['candidate_audit']['HIGHWAY']['availability']}|"
                    f"XSIMD={operation['candidate_audit']['XSIMD']['availability']}|"
                    "HYLE=SOURCE_PRESENT")
    with open(ledger_path, "w", encoding="utf-8", newline="\n") as dst:
        json.dump(document, dst, ensure_ascii=False, indent=2)
        dst.write("\n")

def seed_audit_groups(ledger_path):
    with open(ledger_path, encoding="utf-8") as src:
        document = json.load(src)
    if set(document.get("provider_discovery", {})) != {"HIGHWAY", "XSIMD"}:
        raise RuntimeError("seed audit groups only after locked Highway/xsimd discovery")

    selector_target = {
        "CompressEmulated": "Compress",
        "CompressNotNative": "CompressNot",
        "DemoteToNearestIntNative": "DemoteToNearestInt",
        "GatherIndexEmulated": "GatherIndex",
        "HighestSetBitIndexNative": "HighestSetBitIndex",
        "MaxOfLanesNative": "MaxOfLanes",
        "MinOfLanesNative": "MinOfLanes",
        "NativeGatherOffset32": "GatherOffset",
        "NativeMaskedGatherOr32": "MaskedGatherIndexOr",
        "ReciprocalNewtonNative": "ReciprocalNewton",
        "ReciprocalSqrtNewtonNative": "ReciprocalSqrtNewton",
        "SatWidenMulAccumFixedPointNative": "SatWidenMulAccumFixedPoint",
        "SumOfLanesNative": "SumOfLanes",
    }
    xsimd_composition = {
        "BitwiseIfThenElse", "IfThenElseZero", "IfThenZeroElse",
        "MaskFromVec", "VecFromMask", "ReciprocalNewton", "ReciprocalSqrtNewton",
    }
    evidence_groups = {}
    document["schema"] = LEDGER_SCHEMA

    for operation in document.get("operations", []):
        name = operation["name"]
        kind = operation["comparison_kind"]
        highway = operation["candidate_audit"]["HIGHWAY"]
        xsimd = operation["candidate_audit"]["XSIMD"]
        if highway.get("source_hit_count", 0):
            highway["availability"] = "DIRECT"
        elif kind == "SELECTOR":
            highway["availability"] = "UNAVAILABLE"
            highway["evidence"] = ["provider_discovery:HIGHWAY:no-public-selector"]
        else:
            highway["availability"] = "COMPOSITION"
            highway["evidence"] = [
                "provider_discovery:HIGHWAY:no-same-name-entry",
                "legacy_sites",
            ]
        if xsimd.get("source_hit_count", 0):
            xsimd["availability"] = (
                "COMPOSITION" if name in xsimd_composition else "ADAPTER")
        else:
            xsimd["availability"] = "UNAVAILABLE"
            xsimd["evidence"] = ["provider_discovery:XSIMD:no-same-contract-source-token"]

        classification = operation["classification"]
        covered_by = classification.get("covered_by", [])
        alias_of = classification.get("source_alias_of", "")
        if kind == "ALIAS":
            targets = [alias_of] if alias_of else covered_by
            operation["public_mapping"] = "MERGE_ALIAS:" + "+".join(targets)
        elif kind == "SELECTOR":
            operation["public_mapping"] = "INTERNAL_SELECTOR:" + selector_target[name]
        elif kind == "COMPOSITION":
            operation["public_mapping"] = "COMPOSE:" + "+".join(covered_by or [name])
        elif kind == "STRUCTURAL":
            operation["public_mapping"] = "STRUCTURAL_REVIEW:" + name
        else:
            operation["public_mapping"] = "CAPABILITY:" + name
        operation["variant_audit"] = "INVENTORIED"
        operation["audit_group"] = (
            f"{kind}|HIGHWAY={highway['availability']}|XSIMD={xsimd['availability']}|HYLE=SOURCE_PRESENT")

        if kind == "PERF_ROW":
            bench_files = operation.get("source_bench_files", [])
            if not bench_files:
                raise RuntimeError(f"{name} is PERF_ROW but has no source benchmark file")
            providers = ["HYLE", "HIGHWAY"]
            if xsimd["availability"] != "UNAVAILABLE":
                providers.append("XSIMD")
            file_key = "+".join(os.path.splitext(value)[0] for value in bench_files)
            group_id = f"P1:{file_key}:{'+'.join(providers)}"
            group = {
                "status": "PLANNED",
                "kind": "CURRENT_PER_OPERATION_REPORT_BUNDLE",
                "providers": providers,
                "bench_files": bench_files,
                "contract": "each operation keeps its own qualified title/row; the bundle only shares build and run",
            }
        elif kind == "ALGORITHM_WORKLOAD":
            group_id = f"P1:algorithm:{name}"
            group = {
                "status": "PLANNED",
                "kind": "CURRENT_WORKLOAD",
                "providers": ["HYLE", "HIGHWAY"],
                "bench_files": operation.get("source_bench_files", []),
                "contract": "whole-range algorithm semantics and resource cost",
            }
        else:
            group_id = f"P0:{kind.lower()}"
            group = {
                "status": "PLANNED",
                "kind": f"{kind}_PROOF",
                "providers": ["HYLE", "HIGHWAY", "XSIMD"],
                "bench_files": [],
                "contract": "per-operation structural, alias, selector or composition proof",
            }
        evidence_groups.setdefault(group_id, group)
        operation["routes"][0]["evidence_group"] = group_id
        operation["routes"][0]["provider_assessments"] = {
            "HIGHWAY": {"contract": "PENDING", "evidence": []},
            "XSIMD": {"contract": "PENDING", "evidence": []},
            "HYLE": {"contract": "PENDING", "evidence": []},
        }

    document["audit_stage"] = "GROUPED"
    document["audit_note"] = (
        "Provider source discovery and public mapping are frozen inputs, not semantic, route or "
        "performance verdicts. Every route scope separately keeps HIGHWAY, XSIMD and HYLE contract "
        "assessments; SIMD-15 must qualify all three before any decision leaves PENDING.")
    document["evidence_groups"] = dict(sorted(evidence_groups.items()))
    with open(ledger_path, "w", encoding="utf-8", newline="\n") as dst:
        json.dump(document, dst, ensure_ascii=False, indent=2)
        dst.write("\n")

def apply_qualification_receipt(ledger_path, receipt_path):
    """Apply a reviewed qualification receipt to mutable route-ledger fields.

    The frozen operation/site projection is never accepted from the receipt. This keeps
    semantic review in small, human-readable evidence files while the exhaustive ledger
    remains a mechanically checked projection instead of a 60k-line hand-editing surface.
    """
    ledger_path = os.path.abspath(ledger_path)
    receipt_path = os.path.abspath(receipt_path)
    if os.path.commonpath((ROOT, receipt_path)) != ROOT:
        raise RuntimeError("qualification receipt must live inside the repository")
    with open(ledger_path, encoding="utf-8") as src:
        document = json.load(src)
    with open(receipt_path, encoding="utf-8") as src:
        receipt = json.load(src)

    baseline = document.get("baseline", {})
    if receipt.get("baseline_commit") != baseline.get("commit"):
        raise RuntimeError("qualification baseline_commit does not match the route ledger")
    if receipt.get("inventory_sha256") != baseline.get("inventory_sha256"):
        raise RuntimeError("qualification inventory_sha256 does not match the route ledger")
    revisions = receipt.get("provider_revisions")
    expected_revisions = {
        provider: document.get("candidate_providers", {}).get(provider)
        for provider in ("HIGHWAY", "XSIMD")
    }
    if revisions != expected_revisions:
        raise RuntimeError("qualification provider revisions do not match the locked candidates")

    group = receipt.get("ledger_evidence_group")
    if not isinstance(group, dict) or not all(
            isinstance(group.get(field), str) and group.get(field)
            for field in ("id", "kind", "contract")):
        raise RuntimeError("qualification lacks a complete ledger_evidence_group")
    operations = {item.get("name"): item for item in document.get("operations", [])}
    scope_replacements = receipt.get("scope_replacements", [])
    if not isinstance(scope_replacements, list):
        raise RuntimeError("qualification scope_replacements must be an array")
    replaced_scopes = set()
    for replacement in scope_replacements:
        if not isinstance(replacement, dict):
            raise RuntimeError("qualification scope replacement must be an object")
        name = replacement.get("name")
        if name not in operations:
            raise RuntimeError(
                f"qualification scope replacement references unknown operation {name!r}")
        source_scope_id = replacement.get("source_scope_id")
        replacement_key = (name, source_scope_id)
        if replacement_key in replaced_scopes:
            raise RuntimeError(
                f"qualification repeats scope replacement {name}/{source_scope_id}")
        replaced_scopes.add(replacement_key)
        operation = operations[name]
        matching = [
            (index, route) for index, route in enumerate(operation.get("routes", []))
            if route.get("scope_id") == source_scope_id
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"{name}: expected exactly one source route scope {source_scope_id!r}")
        route_index, source_route = matching[0]
        if source_route.get("decision") != "PENDING":
            raise RuntimeError(
                f"{name}/{source_scope_id}: only a PENDING route scope may be split")
        scopes = replacement.get("routes")
        if not isinstance(scopes, list) or len(scopes) < 2:
            raise RuntimeError(
                f"{name}/{source_scope_id}: replacement must contain at least two scopes")
        source_ids = set(source_route.get("legacy_site_ids", []))
        source_targets = set(source_route.get("targets", []))
        replacement_ids = set()
        new_routes = []
        new_scope_ids = {
            route.get("scope_id") for route in operation.get("routes", [])
            if route is not source_route
        }
        for scope in scopes:
            if not isinstance(scope, dict):
                raise RuntimeError(
                    f"{name}/{source_scope_id}: replacement route must be an object")
            scope_id = scope.get("scope_id")
            if not isinstance(scope_id, str) or not scope_id or scope_id in new_scope_ids:
                raise RuntimeError(
                    f"{name}/{source_scope_id}: invalid or duplicate replacement scope {scope_id!r}")
            new_scope_ids.add(scope_id)
            legacy_site_ids = scope.get("legacy_site_ids")
            if not isinstance(legacy_site_ids, list) or not legacy_site_ids:
                raise RuntimeError(f"{name}/{scope_id}: legacy_site_ids must be non-empty")
            route_ids = set(legacy_site_ids)
            if not route_ids <= source_ids:
                raise RuntimeError(
                    f"{name}/{scope_id}: replacement route references sites outside source scope")
            replacement_ids |= route_ids
            targets = scope.get("targets")
            if not isinstance(targets, list) or not targets or not set(targets) <= source_targets:
                raise RuntimeError(
                    f"{name}/{scope_id}: replacement targets must be a non-empty source subset")
            variant_scope = scope.get("variant_scope")
            if not isinstance(variant_scope, str) or not variant_scope:
                raise RuntimeError(f"{name}/{scope_id}: variant_scope must be non-empty")
            new_route = {
                "scope_id": scope_id,
                "legacy_site_ids": legacy_site_ids,
                "variant_scope": variant_scope,
                "targets": targets,
                "decision": "PENDING",
                "provider": "PENDING",
                "evidence_group": source_route.get("evidence_group", "PENDING"),
                "evidence": [],
                "provider_assessments": {
                    provider: {"contract": "PENDING", "evidence": []}
                    for provider in ("HIGHWAY", "XSIMD", "HYLE")
                },
            }
            if "selector" in scope:
                new_route["selector"] = scope["selector"]
            new_routes.append(new_route)
        if replacement_ids != source_ids:
            raise RuntimeError(
                f"{name}/{source_scope_id}: replacement scope union must preserve every source site")
        operation["routes"][route_index:route_index + 1] = new_routes

    findings = receipt.get("operations")
    if not isinstance(findings, list) or not findings:
        raise RuntimeError("qualification must contain at least one operation finding")

    receipt_rel = os.path.relpath(receipt_path, ROOT).replace(os.sep, "/")
    document.setdefault("evidence_groups", {})[group["id"]] = {
        "status": "QUALIFIED",
        "kind": group["kind"],
        "providers": ["HYLE", "HIGHWAY", "XSIMD"],
        "bench_files": [],
        "contract": group["contract"],
        "evidence": [receipt_rel],
    }
    seen_scopes = set()
    for finding in findings:
        name = finding.get("name") if isinstance(finding, dict) else None
        if name not in operations:
            raise RuntimeError(f"qualification references unknown operation {name!r}")
        operation = operations[name]
        scope_id = finding.get("scope_id")
        key = (name, scope_id)
        if key in seen_scopes:
            raise RuntimeError(f"qualification repeats {name}/{scope_id}")
        seen_scopes.add(key)

        availability = finding.get("candidate_availability")
        if not isinstance(availability, dict) or set(availability) != {
                "HIGHWAY", "XSIMD", "HYLE"}:
            raise RuntimeError(f"{name}: candidate_availability must cover all providers")
        for provider, verdict in availability.items():
            operation["candidate_audit"][provider]["availability"] = verdict
        operation["audit_group"] = (
            f"{operation['comparison_kind']}|"
            f"HIGHWAY={availability['HIGHWAY']}|"
            f"XSIMD={availability['XSIMD']}|"
            f"HYLE={availability['HYLE']}")
        public_mapping = finding.get("public_mapping")
        if not isinstance(public_mapping, str) or not public_mapping:
            raise RuntimeError(f"{name}: qualification lacks public_mapping")
        operation["public_mapping"] = public_mapping
        if finding.get("variant_audit") != "COMPLETE":
            raise RuntimeError(f"{name}: qualification must explicitly complete variant_audit")
        operation["variant_audit"] = "COMPLETE"

        routes = [route for route in operation.get("routes", [])
                  if route.get("scope_id") == scope_id]
        if len(routes) != 1:
            raise RuntimeError(f"{name}: expected exactly one route scope {scope_id!r}")
        route = routes[0]
        decision = finding.get("decision")
        provider = finding.get("provider")
        if not isinstance(decision, str) or not isinstance(provider, str):
            raise RuntimeError(f"{name}: qualification lacks decision/provider")
        route["decision"] = decision
        route["provider"] = provider
        route["evidence_group"] = group["id"]
        route["evidence"] = [f"{receipt_rel}#{name}"]
        for provider_name in ("HIGHWAY", "XSIMD", "HYLE"):
            provider_finding = finding.get(provider_name.lower())
            if not isinstance(provider_finding, dict):
                raise RuntimeError(f"{name}: qualification lacks {provider_name} assessment")
            contract = provider_finding.get("contract")
            evidence = provider_finding.get("evidence")
            if not isinstance(contract, str) or not isinstance(evidence, list) or not evidence:
                raise RuntimeError(f"{name}: incomplete {provider_name} assessment")
            route["provider_assessments"][provider_name] = {
                "contract": contract,
                "evidence": evidence + [f"{receipt_rel}#{name}"],
            }

    temp_path = ledger_path + ".qualification.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as dst:
            json.dump(document, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
        _, errors = validate_route_ledger(temp_path)
        if errors:
            raise RuntimeError("qualification would invalidate ledger: " + "; ".join(errors))
        os.replace(temp_path, ledger_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reports", nargs="+", metavar="PATH",
                        help="audit op names that actually appear in these raw/aggregated reports")
    parser.add_argument("--ledger", default=ROUTE_LEDGER,
                        help="path to the frozen all-operation route ledger")
    parser.add_argument("--write-ledger", action="store_true",
                        help="generate the initial immutable inventory and PENDING route rows")
    parser.add_argument("--force-write-ledger", action="store_true",
                        help="allow --write-ledger to replace an existing ledger")
    parser.add_argument("--refresh-frozen-inventory", action="store_true",
                        help="apply an additive parser erratum while preserving qualified routes")
    parser.add_argument("--baseline-commit",
                        help="full Git id used by --write-ledger")
    parser.add_argument("--check-ledger", action="store_true",
                        help="validate the ledger against its immutable Git baseline")
    parser.add_argument("--require-route-decisions", action="store_true",
                        help="also fail unless every operation has terminal routes and evidence")
    parser.add_argument("--discover-providers", action="store_true",
                        help="record non-semantic source-token discovery for locked Highway/xsimd")
    parser.add_argument("--highway-root",
                        help="locked Highway checkout used by --discover-providers")
    parser.add_argument("--xsimd-root",
                        help="locked xsimd checkout used by --discover-providers")
    parser.add_argument("--seed-audit-groups", action="store_true",
                        help="classify all discovered rows into bounded provider audit groups")
    parser.add_argument("--apply-qualification", action="append", metavar="RECEIPT",
                        help="apply a reviewed in-repository qualification receipt to mutable routes")
    parser.add_argument("--provider-manifest", default=PROVIDER_MANIFEST,
                        help="generated compile-time operation manifest path")
    parser.add_argument("--write-provider-manifest", action="store_true",
                        help="write the exhaustive compile-time operation list from the ledger")
    parser.add_argument("--check-provider-manifest", action="store_true",
                        help="fail if the compile-time operation list differs from the ledger")
    parser.add_argument("--family-routes", default=FAMILY_ROUTES,
                        help="reviewed whole-buffer provider route ledger")
    parser.add_argument("--provider-routes", default=PROVIDER_ROUTES,
                        help="generated compile-time terminal route header")
    parser.add_argument("--write-provider-routes", action="store_true",
                        help="generate terminal operation/family RouteFor specializations")
    parser.add_argument("--check-provider-routes", action="store_true",
                        help="fail if generated terminal routes differ from reviewed ledgers")
    args = parser.parse_args()
    check = args.check

    if args.write_ledger:
        if not args.baseline_commit:
            parser.error("--write-ledger requires --baseline-commit")
        if os.path.exists(args.ledger) and not args.force_write_ledger:
            parser.error(
                f"route ledger already exists: {args.ledger}; use --force-write-ledger only "
                "when intentionally regenerating the frozen baseline")
        try:
            document = build_route_ledger(args.baseline_commit)
        except RuntimeError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        with open(args.ledger, "w", encoding="utf-8", newline="\n") as dst:
            json.dump(document, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
        print(f"wrote {len(document['operations'])} frozen SIMD operation rows to {args.ledger}")

    if args.discover_providers:
        if not args.highway_root or not args.xsimd_root:
            parser.error("--discover-providers requires --highway-root and --xsimd-root")
        if not os.path.exists(args.ledger):
            parser.error("--discover-providers requires an existing route ledger")
        try:
            discover_provider_sources(args.ledger, {
                "HIGHWAY": args.highway_root,
                "XSIMD": args.xsimd_root,
            })
        except RuntimeError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        print("recorded locked Highway/xsimd source-token discovery; "
              "SOURCE_HIT is not a semantic or performance verdict")

    if args.refresh_frozen_inventory:
        if not os.path.exists(args.ledger):
            parser.error("--refresh-frozen-inventory requires an existing route ledger")
        try:
            additions = refresh_frozen_inventory(args.ledger)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"FAIL: cannot refresh frozen inventory: {error}", file=sys.stderr)
            return 1
        print(
            "applied additive frozen-inventory parser erratum: "
            f"{sum(len(item['site_ids']) for item in additions)} sites across "
            f"{len(additions)} operations")

    if args.seed_audit_groups:
        if not os.path.exists(args.ledger):
            parser.error("--seed-audit-groups requires an existing route ledger")
        try:
            seed_audit_groups(args.ledger)
        except (KeyError, RuntimeError) as error:
            print(f"FAIL: cannot seed audit groups: {error}", file=sys.stderr)
            return 1
        print("seeded 387 provider audit groups; route decisions remain PENDING")

    for receipt in args.apply_qualification or []:
        if not os.path.exists(args.ledger):
            parser.error("--apply-qualification requires an existing route ledger")
        try:
            apply_qualification_receipt(args.ledger, receipt)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"FAIL: cannot apply qualification {receipt}: {error}", file=sys.stderr)
            return 1
        print(f"applied reviewed SIMD qualification receipt: {receipt}")

    if args.write_provider_manifest:
        document, errors = validate_route_ledger(args.ledger)
        if errors:
            for error in errors:
                print(f"FAIL: {error}", file=sys.stderr)
            return 1
        try:
            contents = render_provider_operation_manifest(document)
        except RuntimeError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(args.provider_manifest)), exist_ok=True)
        with open(args.provider_manifest, "w", encoding="utf-8", newline="\n") as dst:
            dst.write(contents)
        print(f"wrote exhaustive provider operation manifest to {args.provider_manifest}")

    if args.write_provider_routes:
        document, errors = validate_route_ledger(
            args.ledger, require_decisions=True)
        if errors:
            for error in errors:
                print(f"FAIL: {error}", file=sys.stderr)
            return 1
        try:
            contents = render_provider_routes(document, args.family_routes)
        except (OSError, KeyError, ValueError, json.JSONDecodeError,
                RuntimeError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(args.provider_routes)), exist_ok=True)
        with open(args.provider_routes, "w", encoding="utf-8", newline="\n") as dst:
            dst.write(contents)
        print(f"wrote terminal provider routes to {args.provider_routes}")

    ledger_document = None
    ledger_errors = []
    if args.check_ledger or args.require_route_decisions or os.path.exists(args.ledger):
        ledger_document, ledger_errors = validate_route_ledger(
            args.ledger, require_decisions=args.require_route_decisions)
        if ledger_document is not None and (
                args.check or args.check_ledger or args.require_route_decisions or
                args.check_provider_manifest or args.check_provider_routes):
            ledger_errors.extend(validate_provider_operation_manifest(
                ledger_document, args.provider_manifest))
        if ledger_document is not None and (
                args.check or args.check_provider_routes):
            ledger_errors.extend(validate_provider_routes(
                ledger_document, args.family_routes, args.provider_routes))

    # SIMD-16 physically retires the legacy register facade after its frozen
    # inventory has been fully transcribed into the terminal route ledger. Do
    # not make post-migration ledger/route verification depend on resurrecting
    # deleted product sources or benchmark-shaped test files. Historical source
    # coverage mode remains fail-closed when explicitly requested with --check.
    legacy_inventory_present = all(
        os.path.exists(os.path.join(INC, header)) for header in HEADERS)
    if not legacy_inventory_present:
        print("legacy SIMD register sources retired; validating frozen route ledger only")
        if args.check or args.reports:
            print("FAIL: historical source-coverage mode is unavailable after SIMD-16 migration; "
                  "use --check-ledger/--check-provider-routes", file=sys.stderr)
            return 1
        if ledger_errors:
            for error in ledger_errors:
                print(f"FAIL: {error}", file=sys.stderr)
            return 1
        if ledger_document is not None:
            routes = [route for item in ledger_document.get("operations", [])
                      for route in item.get("routes", [])]
            print(f"operation rows: {len(ledger_document.get('operations', []))}")
            print(f"route scopes: {len(routes)}; pending: "
                  f"{sum(route.get('decision') == 'PENDING' for route in routes)}")
        return 0

    all_ops = set()
    per_header = {}
    for h in HEADERS:
        p = os.path.join(INC, h)
        ops = extract_public_ops(p)
        per_header[h] = len(ops)
        all_ops |= ops

    benched = extract_benched(all_ops)
    manifest = load_manifest()

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
    benched_excluded = sorted(benched_known - benchable)

    valid_classes = {"native", "alias", "synthetic", "paired_info", "algorithm"}
    manifest_unknown = sorted(set(manifest) - all_ops)
    manifest_bad_class = sorted(
        name for name, item in manifest.items() if item.get("class") not in valid_classes)
    manifest_native_missing = sorted(
        name for name, item in manifest.items()
        if item.get("class") == "native" and name not in benched_known)
    manifest_non_native_benched = sorted(
        name for name, item in manifest.items()
        if item.get("class") in {"alias", "synthetic"} and name in benched_known)
    manifest_unclassified = sorted(set(uncovered) - set(manifest))
    manifest_stale = sorted(
        name for name, item in manifest.items()
        if item.get("class") != "native" and name not in uncovered)
    manifest_broken_coverage = []
    manifest_missing_evidence = []
    for name, item in manifest.items():
        for target in item.get("covered_by", []):
            if target not in benched_known:
                manifest_broken_coverage.append(f"{name} -> {target}")
        if item.get("class") == "algorithm":
            evidence = item.get("evidence", "")
            if not evidence or not os.path.exists(os.path.join(ROOT, evidence)):
                manifest_missing_evidence.append(f"{name}: {evidence or '<missing>'}")

    # Historical matrix bookkeeping. Current certification comes from v4 report files, not merely
    # from the presence of an old row.
    matrix_available = os.path.exists(MATRIX)
    matrix_ops = extract_matrix_ops(all_ops) if matrix_available else set()
    sota_pending_benched = (
        sorted((benchable & benched_known) - matrix_ops) if matrix_available else [])
    not_yet_sota = sorted(set(uncovered) | set(sota_pending_benched))

    print(f"headers parsed: {len(HEADERS)}")
    for h in HEADERS:
        print(f"  {h:22s} public ops: {per_header[h]}")
    print(f"\npublic op names (deduped across headers): {len(all_ops)}")
    print(f"  not-benchable (excluded, with reason):  {len(all_ops & set(NOT_BENCHABLE))}")
    print(f"  internal selector variants (*Native/*Emulated): "
          f"{sum(1 for o in all_ops if is_selector(o))}")
    print(f"\nBENCHABLE universe:      {len(benchable)}")
    print(f"benched op names (raw):  {len(benched_known)}")
    print(f"  covered + benchable:   {len(covered)}")
    print(f"  exercised but excluded from perf denominator: {len(benched_excluded)}")
    if benched_excluded:
        print(f"    ({', '.join(benched_excluded)})")
    print(f"UNCOVERED (benchable-benched): {len(uncovered)}")
    print(f"coverage: {len(covered)}/{len(benchable)} = "
          f"{100.0*len(covered)/max(1,len(benchable)):.1f}%")
    class_counts = {kind: sum(1 for item in manifest.values() if item.get("class") == kind)
                    for kind in sorted(valid_classes)}
    classified_non_bench = sorted(set(uncovered) & set(manifest))
    print(f"\nexplicit manifest:       {len(manifest)} classified gap names")
    print("  " + ", ".join(f"{kind}={count}" for kind, count in class_counts.items()))
    print(f"  independently benched native rows: "
          f"{class_counts['native'] - len(manifest_native_missing)}/{class_counts['native']}")
    print(f"  intentionally non-parity names still classified: {len(classified_non_bench)}")
    print(f"  truly unclassified benchable names: {len(manifest_unclassified)}")

    if ledger_document is not None:
        ledger_ops = ledger_document.get("operations", [])
        pending = [
            item.get("name") for item in ledger_ops
            if item.get("audit_group") == "PENDING" or
            item.get("public_mapping") == "PENDING" or
            item.get("variant_audit") != "COMPLETE" or
            any(route.get("decision") == "PENDING" or
                route.get("provider") == "PENDING" or
                route.get("evidence_group") == "PENDING"
                for route in item.get("routes", []))
        ]
        print(f"\n--- frozen multi-provider route ledger ---")
        print(f"baseline: {ledger_document.get('baseline', {}).get('commit', '<missing>')}")
        print(f"operation rows: {len(ledger_ops)}")
        print(f"pending operation rows: {len(pending)}")
        route_scopes = [route for item in ledger_ops for route in item.get("routes", [])]
        terminal_scopes = [route for route in route_scopes
                           if route.get("decision") != "PENDING" and
                           route.get("provider") != "PENDING"]
        print(f"route scopes: {len(route_scopes)} total, {len(terminal_scopes)} terminal")
        for provider in ("HYLE", "HIGHWAY", "XSIMD"):
            contract_counts = {}
            for route in route_scopes:
                contract = route.get("provider_assessments", {}).get(
                    provider, {}).get("contract", "MISSING")
                contract_counts[contract] = contract_counts.get(contract, 0) + 1
            print(f"  {provider.lower()} contracts: " + ", ".join(
                f"{name}={count}" for name, count in sorted(contract_counts.items())))
        xsimd_hits = sum(
            item.get("candidate_audit", {}).get("XSIMD", {}).get(
                "source_hit_count", 0) > 0 for item in ledger_ops)
        xsimd_public_hits = sum(
            item.get("candidate_audit", {}).get("XSIMD", {}).get(
                "public_source_hit_count", 0) > 0 for item in ledger_ops)
        print(f"  xsimd discovery: all-header hits={xsimd_hits}, "
              f"public-surface token hits={xsimd_public_hits} (neither is a contract verdict)")
        print(f"ledger structural errors: {len(ledger_errors)}")
        for error in ledger_errors:
            print(f"  - {error}")

    if orphans:
        print("\n!! SELF-CHECK FAILED: these benched ops are NOT in the extracted op set.")
        print("   => extraction rules have a blind spot; fix the inline parser / namespace handling and re-run.")
        for o in orphans:
            print(f"     - {o}")
    if manifest_unknown or manifest_bad_class or manifest_native_missing or manifest_non_native_benched or manifest_unclassified or manifest_stale or manifest_broken_coverage or manifest_missing_evidence:
        print("\n!! MANIFEST CHECK FAILED")
        for label, values in (
                ("unknown names", manifest_unknown),
                ("invalid classes", manifest_bad_class),
                ("native rows still missing", manifest_native_missing),
                ("alias/synthetic accidentally benched", manifest_non_native_benched),
                ("unclassified uncovered names", manifest_unclassified),
                ("stale non-native entries no longer uncovered", manifest_stale),
                ("covered_by target not source-benched", manifest_broken_coverage),
                ("algorithm evidence missing", manifest_missing_evidence)):
            if values:
                print(f"  {label}:")
                for value in values:
                    print(f"    - {value}")

    print(f"\n--- historical matrix ledger (not a current SOTA certificate) ---")
    if matrix_available:
        print(f"benchable ops with a historical matrix row: {len(benchable & matrix_ops)}")
        print(f"benched but absent from the historical matrix: "
              f"{len(sota_pending_benched)}")
        print(f"not independently benched (classified aliases/synthetic/algorithms included): {len(uncovered)}")
        print(f"==> TOTAL absent from historical matrix or source bench: {len(not_yet_sota)} "
              f"({len(sota_pending_benched)} measure + {len(uncovered)} add-bench)")
    else:
        print("historical per-op matrix is no longer a current repository artifact; ")
        print("current execution and deletion gates come from --reports and the v3 route ledger")

    print(f"\nNOT INDEPENDENTLY BENCHED ({len(uncovered)}; see manifest class, not all need a row):")
    for o in uncovered:
        kind = manifest.get(o, {}).get("class", "UNCLASSIFIED")
        print(f"  - {o} [{kind}]")
    if matrix_available:
        print(f"\nBENCHED but absent from historical matrix ({len(sota_pending_benched)}):")
        for o in sota_pending_benched:
            print(f"  - {o}")

    report_missing = []
    if args.reports:
        reported = extract_reported(args.reports, all_ops)
        report_missing = sorted(benched_known - reported)
        print(f"\n--- executed-report audit ({len(args.reports)} files) ---")
        print(f"source-benched names: {len(benched_known)}")
        print(f"names observed in reports: {len(benched_known & reported)}")
        print(f"source-benched but absent from report union: {len(report_missing)}")
        for op in report_missing:
            print(f"  - {op}")

    if check:
        if orphans:
            print("\nFAIL: self-check (benched is not a subset of all_ops).")
            return 1
        if report_missing:
            print("\nFAIL: executed reports do not cover every source-benched op name.")
            return 1
        if (manifest_unknown or manifest_bad_class or manifest_native_missing or
                manifest_non_native_benched or manifest_unclassified or manifest_stale or
                manifest_broken_coverage or manifest_missing_evidence):
            print("\nFAIL: explicit SIMD op manifest is incomplete or inconsistent.")
            return 1
        if ledger_errors:
            print("\nFAIL: frozen SIMD route ledger is incomplete or inconsistent.")
            return 1
    if (args.check_ledger or args.require_route_decisions) and ledger_errors:
        print("\nFAIL: frozen SIMD route ledger is incomplete or inconsistent.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
