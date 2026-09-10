#!/usr/bin/env python3
"""Select and execute bounded validation from Git change impact (no default P1)."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = "cmake/HyleComponents.cmake"
REGISTRY = "docs/implementation/module-status.json"
NATIVE = {".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".inl", ".inc", ".S", ".asm"}
NATIVE_GATES = {"build", "correctness", "headers", "delivery", "performance"}
NATIVE_ENV = ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CL", "_CL_",
              "INCLUDE", "LIB", "LIBPATH", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET",
              "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "CMAKE_TOOLCHAIN_FILE", "PATH")


class SelectionError(RuntimeError):
    pass


_PROJECT_GRAPH_CACHE = {}
_COMMITTED_SNAPSHOT_CACHE = {}


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def command(root, argv, *, env=None):
    result = subprocess.run(argv, cwd=root, env=env, capture_output=True)
    if result.returncode:
        raise SelectionError(f"{argv[0]} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def git(root, *args, env=None):
    return command(root, ["git", *args], env=env)


class Snapshot:
    def __init__(self, root, revision):
        self.root = root
        self.commit = None
        if revision == "WORKTREE":
            # A private index preserves the user's staged/unstaged split.
            with tempfile.TemporaryDirectory(prefix="hyle-validation-index-") as temp:
                env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
                git(root, "read-tree", "HEAD", env=env)
                git(root, "add", "--all", "--", ".", env=env)
                self.tree = git(root, "write-tree", env=env).decode().strip()
        else:
            if revision.startswith("-"):
                raise SelectionError("revision cannot be an option")
            self.commit = git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
            self.tree = git(root, "rev-parse", self.commit + "^{tree}").decode().strip()
        self.files = {}
        for row in git(root, "ls-tree", "-rz", self.tree).split(b"\0"):
            if row:
                metadata, path = row.split(b"\t", 1)
                mode, kind, oid = metadata.decode().split()
                self.files[path.decode()] = (mode, kind, oid)

    def read(self, path):
        entry = self.files.get(path)
        if not entry or entry[1] != "blob":
            return None
        return git(self.root, "cat-file", "blob", entry[2])


def remember_snapshot(snapshot):
    if snapshot.commit is not None:
        _COMMITTED_SNAPSHOT_CACHE[(str(Path(snapshot.root).resolve()), snapshot.commit)] = snapshot
    return snapshot


def committed_snapshot(root, revision):
    root_key = str(Path(root).resolve())
    commit = git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    key = (root_key, commit)
    if key not in _COMMITTED_SNAPSHOT_CACHE:
        _COMMITTED_SNAPSHOT_CACHE[key] = Snapshot(root, commit)
    return _COMMITTED_SNAPSHOT_CACHE[key]


def project_graph(snapshot):
    data = snapshot.read(MANIFEST)
    if data is None:
        raise SelectionError(f"missing central manifest in {snapshot.tree}")
    cache_key = hashlib.sha256(data).hexdigest()
    if cache_key in _PROJECT_GRAPH_CACHE:
        return _PROJECT_GRAPH_CACHE[cache_key]
    # The historical manifest couples metadata resolution to configure-time
    # source auditing. Evaluate its exact metadata prefix with CMake, stopping
    # at the named audit-function boundary; never interpret dependency syntax.
    boundary = b"function(_hyle_component_public_headers name out_var)"
    if data.count(boundary) != 1:
        raise SelectionError("central manifest metadata boundary changed; review the projection adapter")
    data, tail = data.split(boundary)
    if re.search(rb"(?m)^\s*_hyle_declare_(?:component|product)\(", tail):
        raise SelectionError("central declarations moved beyond the metadata boundary")
    with tempfile.TemporaryDirectory(prefix="hyle-validation-graph-") as temp:
        path = Path(temp)
        (path / "manifest.cmake").write_bytes(data)
        command(snapshot.root, ["cmake", f"-DHYLE_VALIDATION_MANIFEST={path / 'manifest.cmake'}",
                               f"-DHYLE_VALIDATION_GRAPH={path / 'graph.txt'}", "-P",
                               str(SCRIPT_DIR / "validation-graph.cmake")])
        graph = {"factors": {}, "products": {}}
        for row in (path / "graph.txt").read_text().splitlines():
            fields = row.split("|")
            if fields[0] == "factor" and len(fields) == 7:
                _, name, queued, public, private, build, headers = fields
                graph["factors"][name] = {
                    "queued": queued == "TRUE",
                    "deps": [x for x in (public + ";" + private + ";" + build).split(";") if x],
                    "headers": ["include/" + x for x in headers.split(";") if x],
                }
            elif fields[0] == "product" and len(fields) == 3:
                graph["products"][fields[1]] = fields[2].split(";")
            else:
                raise SelectionError("unrecognized central graph projection")
        if not graph["factors"]:
            raise SelectionError("empty central graph projection")
        _PROJECT_GRAPH_CACHE[cache_key] = graph
        return graph


def owner(path, graph):
    for name, factor in graph["factors"].items():
        if path in factor["headers"]:
            return name
    parts = path.split("/")
    name = None
    if len(parts) >= 3 and parts[0] in {"src", "tests", "bench"}:
        name = parts[1]
    elif len(parts) >= 4 and parts[:2] == ["include", "hyle"]:
        name = parts[2]
    if name in graph["factors"] and not graph["factors"][name]["queued"]:
        return name
    return None


def reverse_consumers(seeds, graphs, recursive=False):
    found, frontier = set(), set(seeds)
    while frontier:
        next_nodes = {name for graph in graphs for name, factor in graph["factors"].items()
                      if not factor["queued"] and set(factor["deps"]) & frontier}
        next_nodes -= found | set(seeds)
        found.update(next_nodes)
        if not recursive:
            break
        frontier = next_nodes
    return found


def formatting_only(path, before, after):
    if before is None or after is None:
        return False
    if path.endswith(".json"):
        try:
            return digest(json.loads(before)) == digest(json.loads(after))
        except (ValueError, UnicodeError):
            return False
    if Path(path).suffix not in NATIVE:
        return False
    try:
        old, new = before.decode(), after.decode()
    except UnicodeError:
        return False
    # Only line-preserving indentation/trailing-space and CRLF changes are
    # automatically proven. Raw strings, continuations and line-sensitive
    # constructs are deliberately excluded; token equivalence is insufficient
    # for C++ stringification, preprocessing or Python indentation semantics.
    if any(token in old + new for token in ('R"', "\\\n", "\\\r", "__LINE__", "#line")):
        return False
    return [line.strip(" \t") for line in old.splitlines()] == [
        line.strip(" \t") for line in new.splitlines()]


def vendored_owners(path, policy):
    matches = [
        owners
        for prefix, owners in policy["vendored_input_owners"].items()
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix))
    ]
    if len(matches) != 1:
        return None
    return list(matches[0])


def classify(before, after, graphs, policy):
    changes, unknown = [], []
    for path in sorted(before.files.keys() | after.files.keys()):
        if before.files.get(path) == after.files.get(path):
            continue
        old_entry, new_entry = before.files.get(path), after.files.get(path)
        same_mode = old_entry and new_entry and old_entry[:2] == new_entry[:2]
        # A newly added split header has no owner in the old tree, and a
        # deleted one has no owner in the new tree.  Consulting both manifests
        # for a path that did not exist manufactures a fallback owner from its
        # directory prefix (for example, a new container_text header appears
        # to have moved from container).  Attribute only snapshots that
        # actually contain the path; real renames still retain their old and
        # new owners as two independent changed paths.
        factor_set = set()
        for snapshot, graph in zip((before, after), graphs):
            if path in snapshot.files:
                factor = owner(path, graph)
                if factor:
                    factor_set.add(factor)
        factors = sorted(factor_set)
        kind, reason, suites = "unknown", "no reviewed ownership or gate mapping", []
        if path == REGISTRY or path == "docs/implementation/status.md":
            kind, reason = "state", "registry/projection maintenance; no product execution"
        elif path.startswith(("docs/", "evidence/")) or path.endswith(".md"):
            kind, reason = "docs", "documentation/evidence only"
        elif path in policy["format_config"]:
            kind, reason = "format", "editor/format policy; no product semantics"
        elif same_mode and formatting_only(path, before.read(path), after.read(path)):
            kind, reason = "format", "proven JSON or line-preserving native formatting equivalence"
        elif path == "scripts/check-docs.py":
            kind, reason = "tool", "documentation checker executes in the mandatory docs gate"
        else:
            for suite in policy["tool_suites"]:
                if any(fnmatch.fnmatchcase(path, pattern) for pattern in suite["paths"]):
                    suites.extend(suite["tests"])
            if suites:
                kind, reason = "tool", "mapped maintainer tool contract; no product build or P1"
            elif (
                path.startswith("scripts/")
                and old_entry is not None
                and new_entry is None
                and path not in policy["package_tools"]
            ):
                kind, reason = (
                    "tool",
                    "unmapped maintenance helper retired; additions and modifications remain fail-closed",
                )
            elif path == MANIFEST:
                kind, reason = "graph", "central component or ownership contract changed"
            elif path.startswith("bench/contracts/") and path.endswith(".json"):
                kind, reason = (
                    "benchmark",
                    "performance contract requires an explicit frozen P1 card; no automatic sampling",
                )
            elif path.startswith("scripts/validate-") and path.endswith("-bench.py"):
                kind, reason = (
                    "benchmark",
                    "benchmark semantic validator requires an explicit frozen P1 card",
                )
            elif path.startswith("samples/package/"):
                product = path.split("/")[2]
                if product in graphs[1]["products"]:
                    kind, reason = "sample", "real installed/source consumer changed"
                    factors = graphs[1]["products"][product]
            elif path.startswith("third_party/"):
                reviewed = vendored_owners(path, policy)
                if reviewed is None:
                    kind, reason = "unknown", "vendored input lacks unique reviewed ownership"
                else:
                    kind, reason = "package", "reviewed vendored product or instrument input changed"
                    factors = sorted(reviewed)
            elif path in policy["package_tools"] or path in {"CMakeLists.txt", "CMakePresets.json", "tests/CMakeLists.txt", "src/CMakeLists.txt", "bench/CMakeLists.txt", "include/hyle/obs/trace.h"} or path.startswith("cmake/"):
                kind, reason = "package", "shared build/package or external product input changed"
            elif factors:
                if path.startswith("tests/"):
                    kind, reason = "test", "factor test source or CMake change; owner tests only"
                elif path.endswith(("CMakeLists.txt", ".cmake")):
                    kind, reason = "factor_seam", "factor product build/package seam changed"
                elif path.startswith("include/"):
                    kind, reason = "public", "owned public header/API may change consumer closure"
                elif path.startswith("src/"):
                    kind, reason = "source", "implementation change; owner and one-hop direct consumers only"
                elif path.startswith("bench/"):
                    kind, reason = "benchmark", "harness change requires explicit frozen P1 card; no automatic sampling"
            elif path in {"include/hyle/version.h", "include/hyle/visibility.h"}:
                kind, reason = "package", "shared public version/visibility contract changed"
        if kind == "unknown":
            unknown.append(path)
        changes.append({"path": path, "kind": kind, "factors": factors,
                        "reason": reason, "suites": sorted(set(suites))})
    return changes, unknown


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SelectionError(f"invalid JSON {path}: {exc}") from exc


def registry_states(snapshot):
    raw = snapshot.read(REGISTRY)
    if raw is None:
        raise SelectionError("candidate is missing the module registry")
    try:
        payload = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise SelectionError(f"candidate module registry is invalid JSON: {exc}") from exc
    modules = payload.get("modules") if isinstance(payload, dict) else None
    if not isinstance(modules, list):
        raise SelectionError("candidate module registry has no modules array")
    states = {}
    for row in modules:
        if not isinstance(row, dict) or not isinstance(row.get("module"), str) or not isinstance(row.get("state"), str):
            raise SelectionError("candidate module registry has an invalid module/state row")
        if row["module"] in states:
            raise SelectionError(f"candidate module registry repeats {row['module']!r}")
        states[row["module"]] = row["state"]
    return states


def product_change_owners(changes, graphs, policy):
    owners = {
        factor
        for change in changes
        if change["kind"] in {"source", "public", "factor_seam"}
        for factor in change["factors"]
    }
    for change in changes:
        if change["path"].startswith("third_party/"):
            reviewed = vendored_owners(change["path"], policy)
            if reviewed:
                owners.update(reviewed)
    if any(change["path"] == MANIFEST for change in changes):
        before, after = graphs
        for factor in before["factors"].keys() | after["factors"].keys():
            if before["factors"].get(factor) != after["factors"].get(factor):
                # Registering a new dormant identity is governance only.  Making
                # it buildable, or changing an existing factor contract, is a
                # product-boundary change and needs an audited owner.
                current = after["factors"].get(factor)
                if current is not None and not (
                    factor not in before["factors"] and current["queued"]
                ):
                    owners.add(factor)
        for product in before["products"].keys() | after["products"].keys():
            old_roots = set(before["products"].get(product, []))
            new_roots = set(after["products"].get(product, []))
            if old_roots != new_roots:
                owners.update(old_roots | new_roots)
    return owners


def authorization_report(root, before, after, policy, expected_owners):
    """Check product changes at the commit where they were introduced."""
    if not expected_owners:
        return {"owners": [], "states": {}, "issues": []}
    remember_snapshot(before)
    remember_snapshot(after)
    snapshot_cache = {before.tree: before, after.tree: after}

    def snapshot(revision):
        item = committed_snapshot(root, revision)
        return snapshot_cache.setdefault(item.tree, item)

    if after.commit is None:
        steps = [(snapshot("HEAD"), after, "WORKTREE")]
    else:
        start = before.commit
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", start, after.commit],
            cwd=root, capture_output=True, check=False
        ).returncode:
            start = git(root, "merge-base", start, after.commit).decode().strip()
        commits = git(root, "rev-list", "--reverse", "--topo-order",
                      f"{start}..{after.commit}", "--", "include", "src", "third_party",
                      MANIFEST).decode().splitlines()
        steps = []
        for commit in commits:
            family = git(root, "rev-list", "--parents", "-n", "1", commit).decode().split()
            parents = family[1:]
            if not parents:
                continue
            current = snapshot(commit)
            if len(parents) == 1:
                steps.append((snapshot(parents[0]), current, commit))
                continue
            # Branch commits are already visited above. Only a path whose merge
            # result differs from every parent can contain new conflict-resolution
            # product bytes and therefore needs an additional authorization step.
            parent_snapshots = [snapshot(parent) for parent in parents]
            novel = {
                path for path in current.files
                if all(current.files.get(path) != parent.files.get(path)
                       for parent in parent_snapshots)
            }
            novel.update(
                path for parent in parent_snapshots for path in parent.files
                if path not in current.files
                and all(other.files.get(path) != current.files.get(path)
                        for other in parent_snapshots)
            )
            steps.append((parent_snapshots[0], current, commit, novel))

    owners, issues = set(), []
    for step in steps:
        previous, current, label, *filters = step
        graphs = [project_graph(previous), project_graph(current)]
        changes, _ = classify(previous, current, graphs, policy)
        if filters:
            changes = [change for change in changes if change["path"] in filters[0]]
        changed_owners = product_change_owners(changes, graphs, policy)
        if not changed_owners:
            continue
        owners.update(changed_owners)
        old_states = registry_states(previous)
        new_states = registry_states(current)
        for factor in sorted(changed_owners):
            old_state = old_states.get(factor, "MISSING")
            new_state = new_states.get(factor, "MISSING")
            if old_state != "AUDITED" or new_state != "AUDITED":
                issues.append(
                    f"{factor}: product input changed at {label[:12]} while state was "
                    f"{old_state} -> {new_state}; AUDITED must be committed before "
                    "implementation and retained until the product commit is complete"
                )
    missing = set(expected_owners) - owners
    for factor in sorted(missing):
        issues.append(
            f"{factor}: net product change could not be attributed to an introducing commit"
        )
    owners.update(expected_owners)
    endpoint_states = registry_states(after)
    return {
        "owners": sorted(owners),
        "states": {factor: endpoint_states.get(factor, "MISSING") for factor in sorted(owners)},
        "issues": issues,
    }


def environment_identity(context=None, *, native=False):
    observed = {"system": platform.system(), "release": platform.release(),
                "machine": platform.machine(), "node": platform.node(),
                "python": sys.version, "python_executable": str(Path(sys.executable).resolve())}
    if context:
        identity = read_json(context)
        for key in ("toolchain", "machine_profile"):
            if not isinstance(identity.get(key), dict) or not identity[key].get("id"):
                raise SelectionError(f"context requires {key} object with a non-empty id")
        if native:
            toolchain = identity["toolchain"]
            for key in ("compiler_path", "compiler_id", "compiler_version"):
                if not isinstance(toolchain.get(key), str) or not toolchain[key].strip():
                    raise SelectionError(f"native context requires toolchain.{key}")
            compiler = Path(toolchain["compiler_path"])
            if not compiler.is_absolute() or not compiler.is_file():
                raise SelectionError("native context requires an existing absolute compiler_path")
            stdlib = toolchain.get("stdlib")
            if not isinstance(stdlib, dict) or not stdlib.get("id"):
                raise SelectionError("native context requires a concrete stdlib id")
            flags = toolchain.get("flags")
            required = {"CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE"}
            if not isinstance(flags, dict) or set(flags) != required or not all(isinstance(v, str) for v in flags.values()) or not flags["CMAKE_CXX_FLAGS_RELEASE"].strip():
                raise SelectionError("native context requires explicit CMAKE_CXX_FLAGS and non-empty CMAKE_CXX_FLAGS_RELEASE")
            observed["native_environment"] = {key: os.environ.get(key) for key in NATIVE_ENV}
        identity["observed_host"] = observed
        return identity
    return {"toolchain": {"id": "python-maintenance", "version": sys.version},
            "machine_profile": observed}


def p1_scope(snapshot, graph, policy, factor):
    dependencies, pending = set(), {factor}
    while pending:
        name = pending.pop()
        if name in dependencies:
            continue
        if name not in graph["factors"] or graph["factors"][name]["queued"]:
            raise SelectionError(f"P1 input scope has no implemented owner: {name}")
        dependencies.add(name)
        pending.update(graph["factors"][name]["deps"])
    shared = {"CMakeLists.txt", "CMakePresets.json", "src/CMakeLists.txt",
              "include/hyle/version.h", "include/hyle/visibility.h", "include/hyle/obs/trace.h",
              *policy["package_tools"]}
    inputs = {}
    for path, entry in snapshot.files.items():
        if path.startswith("third_party/"):
            owners = vendored_owners(path, policy)
            if owners is None:
                raise SelectionError(f"P1 vendored product input lacks unique reviewed ownership: {path}")
            relevant = bool(dependencies & set(owners))
        else:
            relevant = path in shared or path.startswith("cmake/") or (
                path.startswith(("include/", "src/")) and owner(path, graph) in dependencies)
        if relevant:
            inputs[path] = entry
    input_hash = digest({"factor": factor, "dependencies": sorted(dependencies), "files": inputs})
    return {"factor": factor, "dependencies": sorted(dependencies),
            "product_input_hash": input_hash, "input_paths": sorted(inputs),
            "scope_hash": digest({"factor": factor, "product_input_hash": input_hash})}


def make_plan(root, baseline, candidate, *, context=None, card_path=None, cache=None):
    before, after = Snapshot(root, baseline), Snapshot(root, candidate)
    policy = read_json(SCRIPT_DIR / "validation-policy.json")
    graphs = [project_graph(before), project_graph(after)]
    changes, unknown = classify(before, after, graphs, policy)
    authorization = authorization_report(
        root, before, after, policy, product_change_owners(changes, graphs, policy)
    )
    available = {k for k, v in graphs[1]["factors"].items() if not v["queued"]}
    own, consumers, expanded, header_roots, delivery = set(), set(), set(), set(), set()
    suites, impact_reasons = set(), {}
    broad = False
    for change in changes:
        kind, factors = change["kind"], set(change["factors"])
        suites.update(change["suites"])
        if kind in {"source", "public", "factor_seam", "test"}:
            own.update(factors)
            for factor in factors:
                impact_reasons.setdefault(factor, []).append(change["path"])
        if kind == "source":
            consumers.update(reverse_consumers(factors, graphs))
        if kind in {"public", "factor_seam"}:
            expanded.update(factors | reverse_consumers(factors, graphs, True))
            header_roots.update(factors)
        if kind in {"graph", "package"}:
            broad = True
        if kind == "sample":
            delivery.add(change["path"].split("/")[2])
    if broad:
        # Global build semantics can change even if the projected edges do not.
        expanded.update(available)
        header_roots.update(available)
    own &= available
    consumers &= available
    expanded &= available
    selected_factors = own | consumers | expanded
    for product, roots in graphs[1]["products"].items():
        if set(roots) & expanded:
            delivery.add(product)
    identity = environment_identity(context)
    native_issue = None
    if context and (selected_factors or broad or delivery):
        try:
            identity = environment_identity(context, native=True)
        except SelectionError as exc:
            native_issue = str(exc)
    tool_hash = digest({p.name: digest(p.read_bytes()) for p in SCRIPT_DIR.glob("validation-*") if p.is_file()})
    gates = []

    def gate(name, kind, selected, reason, commands=(), factors=()):
        row = {"id": name, "kind": kind, "status": "SELECTED" if selected else "SKIPPED",
               "reason": reason, "factors": sorted(factors), "commands": list(commands)}
        row["identity"] = {"candidate_tree": after.tree,
                           "contract_hash": digest({"gate": name, "commands": row["commands"],
                                                    "policy": policy, "card": read_json(card_path) if card_path else None,
                                                    "baseline_registry": before.files.get(REGISTRY) if name == "closure" else None}),
                           "toolchain": identity["toolchain"], "machine_profile": identity["machine_profile"],
                           "observed_host": identity.get("observed_host"), "selector_hash": tool_hash}
        row["cache_key"] = digest(row["identity"])
        gates.append(row)
        return row

    gate("docs", "maintenance", True, "mandatory documentation contract",
         [["{python}", "scripts/check-docs.py"]])
    registry_changed = before.files.get(REGISTRY) != after.files.get(REGISTRY)
    closure_argv = ["{python}", "scripts/check-module-closure.py"]
    if registry_changed:
        if not before.commit:
            raise SelectionError("registry transition requires an immutable Git baseline")
        closure_argv += ["--previous-ref", before.commit]
    gate("closure", "maintenance", True,
         "mandatory registry contract" + (" including explicit baseline snapshot transition check" if registry_changed else ""),
         [closure_argv])
    for suite in sorted(suites):
        gate("tool:" + suite, "tool", True, "changed tool mapped to this focused regression suite",
             [["{python}", suite]])
    configure = ["cmake", "-S", ".", "-B", "{build}", "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DHYLE_DEV_FACTORS=" + ",".join(sorted(selected_factors)),
                 "-DHYLE_BUILD_TESTS=ON", "-DHYLE_BUILD_BENCH=OFF", "-DHYLE_INSTALL=OFF",
                 "-DHYLE_TEST_INSTRUMENTATION=OFF", "-DHYLE_WERROR=ON"]
    if context and selected_factors and not native_issue:
        configure += ["-DCMAKE_CXX_COMPILER=" + identity["toolchain"]["compiler_path"]]
        configure += ["-D" + key + "=" + value for key, value in sorted(identity["toolchain"]["flags"].items())]
    gate("configure", "build", bool(selected_factors),
         "configure exact factor roots; dependency artifacts are prerequisites, not additional test roots"
         if selected_factors else "no product or factor test impact", [configure], selected_factors)
    for factor in sorted(available):
        why = ("owning factor: " + ", ".join(impact_reasons[factor]) if factor in own else
               "public/API/component/package boundary requires reverse consumer closure" if factor in expanded else
               "real one-hop direct edges to changed implementation: " + ", ".join(sorted({d for graph in graphs
                   for d in graph["factors"].get(factor, {}).get("deps", []) if d in own})) if factor in consumers else
               "outside the change and its required consumer boundary")
        gate("factor:" + factor, "correctness", factor in selected_factors, why,
             [["cmake", "--build", "{build}", "--config", "Release", "--target", f"hyle_{factor}_tests", "--parallel", "2"],
              ["ctest", "--test-dir", "{build}", "-C", "Release", "--output-on-failure", "--no-tests=error",
               "-L", "^" + factor + "$", "-LE", "headers"]], [factor])
        fragment = after.read("tests/" + factor + "/CMakeLists.txt") or b""
        has_header_suite = b"hyle_add_header_self_contained_test(" in fragment
        gate("headers:" + factor, "headers", factor in header_roots and has_header_suite,
             "no independently registered header suite; selected consumer/delivery gates cover the boundary, not header self-containment"
             if factor in header_roots and not has_header_suite else
             "public header or delivery seam changed" if factor in header_roots else "no public-header seam change for this factor",
             [["ctest", "--test-dir", "{build}", "-C", "Release", "--output-on-failure", "--no-tests=error",
               "-L", "^" + factor + "$", "-R", "_header_self_contained_test$"]], [factor])
    gate("component-contract", "delivery", broad,
         "shared component/build/package semantics changed" if broad else "no global component/package seam change",
         [["cmake", "-DHYLE_SOURCE_DIR={root}", "-DHYLE_SMOKE_ROOT={scratch}/component",
           "-P", "cmake/HyleCiComponentContract.cmake"]])
    for product in sorted(graphs[1]["products"]):
        gate("delivery:" + product, "delivery", product in delivery,
             "public boundary or real package consumer changed; reuse existing applicable source/static/shared smoke"
             if product in delivery else "no delivery boundary change for this product",
             [["cmake", "-DHYLE_SOURCE_DIR={root}", "-DHYLE_SMOKE_ROOT={scratch}/delivery-" + product,
               "-DHYLE_SMOKE_COMPONENT=" + product, "-P", "cmake/HyleCiSelectedComponentSmoke.cmake"]])
    p1 = gate("p1", "performance", False, "no explicit frozen P1 card; sampling is never automatic")
    p1_scopes = {}
    if card_path:
        card = read_json(card_path)
        material = any(c["kind"] in {"source", "public", "factor_seam", "graph", "package", "benchmark"} for c in changes)
        issues = []
        if not material:
            p1["reason"] = "docs/state/format/tools/test-only diff cannot trigger P1, even with a card"
        else:
            if not after.commit or card.get("candidate_commit") != after.commit or card.get("candidate_tree") != after.tree:
                issues.append("card must name this exact immutable candidate commit/tree")
            if card.get("candidate_fixed") is not True:
                issues.append("P1 card must confirm candidate_fixed=true")
            if not context:
                issues.append("P1 requires explicit toolchain and machine profile context")
            elif not all(identity["toolchain"].get(key) for key in
                         ("compiler_path", "compiler_id", "compiler_version", "stdlib", "flags")):
                issues.append("P1 toolchain must identify compiler, standard library and build flags")
            machines = after.read("scripts/perf-machines.json")
            profiles = json.loads(machines).get("machine_profiles", {}) if machines else {}
            machine_profile = identity["machine_profile"]
            if machine_profile.get("id") not in profiles or machine_profile.get("profile_hash") != digest(profiles.get(machine_profile.get("id"))):
                issues.append("P1 machine profile hash must match the candidate central machine registry")
            factors = card.get("factors")
            if not isinstance(factors, list) or not factors or not all(isinstance(f, str) for f in factors) or len(set(factors)) != len(factors) or not set(factors) <= selected_factors:
                issues.append("P1 factors must have selected candidate correctness gates")
            else:
                try:
                    p1_scopes = {f: p1_scope(after, graphs[1], policy, f) for f in sorted(factors)}
                except SelectionError as exc:
                    issues.append(str(exc))
            contract_path = card.get("contract", "")
            contract_bytes = after.read(contract_path)
            if contract_bytes is None or card.get("contract_hash") != digest(contract_bytes):
                issues.append("frozen contract hash must match the candidate Git blob")
            peers = card.get("peers", {})
            if not isinstance(peers, dict) or not all(isinstance(k, str) and re.fullmatch(r"[\w.-]+", k)
                                                     and isinstance(v, str) and v for k, v in peers.items()):
                issues.append("P1 peers must map peer IDs to locked checkout paths")
            argv = ["{python}", "scripts/hyle-perf.py", "qualify-platform", "--contract", contract_path,
                    "--machine", identity["machine_profile"].get("id", ""), "--source", "{root}",
                    "--build", "{scratch}/p1-build", "--output", "{scratch}/p1-evidence"]
            if not issues:
                for peer, path in sorted(peers.items()):
                    argv += ["--peer", peer + "=" + path]
            p1.update(status="BLOCKED" if issues else "SELECTED", reason="; ".join(issues) if issues else
                      "immutable API/algorithm candidate; bounded P1 bundle after every selected prerequisite passes",
                      commands=[argv] if not issues else [], factors=factors or [])
            p1["identity"]["contract_hash"] = digest({"card": card, "contract_blob": digest(contract_bytes or b"")})
            p1["cache_key"] = digest(p1["identity"])
    for row in gates:
        if row["status"] == "SELECTED" and row["kind"] in NATIVE_GATES and native_issue:
            row.update(status="BLOCKED", reason=native_issue)
        # A context-free dry plan must never reuse compiler-dependent evidence.
        if row["kind"] in NATIVE_GATES and not context:
            continue
        if row["status"] != "SELECTED" or not cache or not after.commit:
            continue
        receipt = cached_receipt(cache, row, root)
        if receipt:
            row.update(status="REUSED", reason="complete PASS evidence with identical tree/contract/toolchain/machine/tool and verified log hashes",
                       receipt=str(Path(cache) / (row["cache_key"] + ".json")))
    configure_gate = next(g for g in gates if g["id"] == "configure")
    if configure_gate["status"] == "REUSED" and any(g["status"] == "SELECTED" and
            g["kind"] in {"correctness", "headers"} for g in gates):
        configure_gate.update(status="SELECTED", reason="pending native gates need local build prerequisites; evidence alone is not a build tree")
    return {"schema_version": 2, "baseline_commit": before.commit, "baseline_tree": before.tree,
            "candidate_commit": after.commit, "candidate_tree": after.tree,
            "changes": changes, "selected_factors": sorted(selected_factors), "p1_scopes": p1_scopes,
            "unknown_paths": unknown,
            "product_authorization": authorization,
            "gates": gates,
            "verdict": "BLOCKED" if unknown or authorization["issues"] or any(g["status"] == "BLOCKED" for g in gates) else "READY"}


def cached_receipt(cache, gate, root):
    path = Path(cache) / (gate["cache_key"] + ".json")
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt.get("identity") != gate["identity"] or receipt.get("verdict") != "PASS":
        return None
    logs = receipt.get("logs", [])
    if len(logs) != len(gate["commands"]) or not logs:
        return None
    for log in logs:
        name = log.get("path", "")
        if not name or Path(name).name != name:
            return None
        file = Path(cache) / name
        if not file.is_file() or digest(file.read_bytes()) != log.get("sha256"):
            return None
    if gate["id"] == "p1":
        artifact = receipt.get("performance_manifest", {})
        path = Path(artifact.get("path", ""))
        if not path.is_file() or digest(path.read_bytes()) != artifact.get("sha256"):
            return None
        try:
            contract = gate["commands"][0][gate["commands"][0].index("--contract") + 1]
            checked = json.loads(command(root, [sys.executable, "scripts/hyle-perf.py", "verify", "--contract", contract,
                                               "--manifest", str(path)]))
            if checked.get("paths_verified") is not True or checked.get("qualification") != "QUALIFIED" or checked.get("current_certification_policy_eligible") is not True:
                return None
        except (SelectionError, ValueError):
            return None
    return receipt


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace on the same volume; a interrupted write cannot become a receipt.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        json.dump(data, temp, indent=2, ensure_ascii=False)
        temp.write("\n")
        staging = Path(temp.name)
    os.replace(staging, path)


def execute(root, plan, cache, scratch, *, context=None):
    if plan["verdict"] == "BLOCKED":
        return
    common = Path(git(root, "rev-parse", "--git-common-dir").decode().strip())
    native = any(g["status"] in {"SELECTED", "REUSED"} and g["kind"] in NATIVE_GATES for g in plan["gates"])
    key = digest({"tree": plan["candidate_tree"], "context": environment_identity(context, native=native)})
    lease = root / common / "hyle-validation-executions" / (key + ".lock")
    lease.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lease.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "candidate_tree": plan["candidate_tree"]}, stream)
    except FileExistsError:
        raise SelectionError(f"candidate validation is already running or interrupted; inspect execution lease {lease}")
    try:
        _execute_locked(root, plan, cache, scratch, context=context)
    finally:
        lease.unlink()


def _execute_locked(root, plan, cache, scratch, *, context=None):
    if plan["verdict"] == "BLOCKED":
        return
    expensive = any(g["status"] in {"SELECTED", "REUSED"} and g["kind"] in NATIVE_GATES for g in plan["gates"])
    if expensive and (not plan["candidate_commit"] or not context):
        raise SelectionError("product execution requires a committed candidate and explicit --context toolchain/machine identity")
    execution_identity = environment_identity(context, native=expensive)
    for row in plan["gates"]:
        if row["status"] in {"SELECTED", "REUSED"} and any(row["identity"].get(k) != execution_identity.get(k) for k in ("toolchain", "machine_profile", "observed_host")):
            raise SelectionError("execution environment differs from plan; regenerate with the current context")
    current = Snapshot(root, "WORKTREE")
    if current.tree != plan["candidate_tree"]:
        raise SelectionError("checkout differs from planned candidate; use its clean checkout or WORKTREE for maintenance only")
    cache = Path(cache).resolve()
    scratch = (root / scratch).resolve()
    scratch /= digest({"tree": plan["candidate_tree"], "context": execution_identity})[:20]
    cache.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    build = scratch / ("build-" + digest({"tree": plan["candidate_tree"], "context": execution_identity,
                                       "factors": plan["selected_factors"]})[:20])
    replacements = {"{python}": sys.executable, "{root}": str(root), "{build}": str(build),
                    "{scratch}": str(scratch)}
    native_env = None
    if expensive:
        # A native smoke/qualifier may launch nested CMake processes. Bind the
        # same explicit compiler/flags there as on the direct factor configure.
        native_env = dict(os.environ)
        toolchain = execution_identity["toolchain"]
        variables = {"CMAKE_CXX_COMPILER": toolchain["compiler_path"].replace("\\", "/"), **toolchain["flags"]}
        lines = []
        for key, value in variables.items():
            equals = "="
            while "]" + equals + "]" in value:
                equals += "="
            lines.append(f'set({key} [{equals}[{value}]{equals}] CACHE STRING "selector native context" FORCE)')
        native_file = scratch / "native-context.cmake"
        native_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        native_env["CMAKE_TOOLCHAIN_FILE"] = str(native_file)
    failure = False
    for row in plan["gates"]:
        if row["status"] != "SELECTED":
            continue
        if failure:
            row.update(status="BLOCKED", reason="a selected prerequisite failed; this gate did not run")
            continue
        if Snapshot(root, "WORKTREE").tree != plan["candidate_tree"]:
            row.update(status="BLOCKED", reason="checkout changed before this gate; regenerate the candidate plan")
            failure = True
            continue
        if expensive and environment_identity(context, native=True) != execution_identity:
            raise SelectionError("native context changed before gate; no stale receipt may be signed")
        if row["id"] == "p1":
            if Snapshot(root, "WORKTREE").tree != plan["candidate_tree"]:
                row.update(status="BLOCKED", reason="candidate drifted while prerequisites ran; no P1 attempt started")
                failure = True
                continue
        logs, verdict = [], "PASS"
        receipt_path = cache / (row["cache_key"] + ".json")
        write_json(receipt_path, {"identity": row["identity"], "verdict": "RUNNING", "logs": []})
        for index, template in enumerate(row["commands"]):
            argv = []
            for token in template:
                for key, value in replacements.items():
                    token = token.replace(key, value)
                argv.append(token)
            name = row["cache_key"] + f"-{index}.log"
            log_path = cache / name
            print(f"RUN {row['id']}: {argv}", file=sys.stderr, flush=True)
            with log_path.open("wb") as stream:
                try:
                    result = subprocess.run(argv, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                            env=native_env if row["kind"] in NATIVE_GATES else None)
                    if result.returncode:
                        verdict = "FAIL"
                except OSError as exc:
                    stream.write(str(exc).encode())
                    verdict = "FAIL"
            logs.append({"path": name, "sha256": digest(log_path.read_bytes())})
            if verdict == "FAIL":
                break
        if Snapshot(root, "WORKTREE").tree != plan["candidate_tree"]:
            verdict = "FAIL"
        if expensive and environment_identity(context, native=True) != execution_identity:
            verdict = "FAIL"
        receipt = {"identity": row["identity"], "candidate_commit": plan["candidate_commit"],
                   "verdict": verdict, "logs": logs}
        if row["id"] == "p1" and verdict == "PASS":
            try:
                # The canonical qualifier returns nonzero on either failed
                # gate. Also require its explicit verdict and retain its raw
                # manifest; exit zero from a diagnostic is never evidence.
                output = log_path.read_text(encoding="utf-8")
                result = json.loads(output[output.rfind("\n{") + 1:] if "\n{" in output else output)
                manifest = Path(result["manifest"])
                if result["qualification_gate"] != "PASS" or result["performance_gate"] != "PASS":
                    raise SelectionError("P1 qualifier did not pass both gates")
                receipt["performance_manifest"] = {"path": str(manifest.resolve()), "sha256": digest(manifest.read_bytes())}
                if Snapshot(root, "WORKTREE").tree != plan["candidate_tree"]:
                    raise SelectionError("candidate drifted during P1")
            except (OSError, ValueError, KeyError, SelectionError):
                verdict = receipt["verdict"] = "FAIL"
        write_json(receipt_path, receipt)
        row.update(status=verdict, receipt=str(receipt_path))
        failure |= verdict != "PASS"
    plan["verdict"] = "BLOCKED" if failure else "PASS"


def public_report(plan):
    report = dict(plan)
    report["selected_gates"] = [g for g in plan["gates"] if g["status"] in {"SELECTED", "PASS", "FAIL", "BLOCKED"}]
    report["skipped_gates"] = [g for g in plan["gates"] if g["status"] in {"SKIPPED", "REUSED"}]
    del report["gates"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", default="HEAD", help="Git commit or WORKTREE (maintenance only)")
    parser.add_argument("--context", type=Path, help="exact toolchain and machine_profile JSON")
    parser.add_argument("--p1-card", type=Path, help="explicit frozen API/algorithm candidate and P1 qualification command")
    parser.add_argument("--cache", type=Path, help="persistent PASS evidence cache; defaults to Git common dir")
    parser.add_argument("--scratch", type=Path, default=Path(".scratch/validation"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        common = Path(git(root, "rev-parse", "--git-common-dir").decode().strip())
        cache = args.cache or (root / common / "hyle-validation")
        plan = make_plan(root, args.base, args.candidate, context=args.context, card_path=args.p1_card, cache=cache)
        if args.action == "run":
            execute(root, plan, cache, args.scratch, context=args.context)
        report = public_report(plan)
        if args.output:
            write_json(args.output, report)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        selected = [g["id"] for g in report["selected_gates"]]
        print(f"{plan['verdict']}: selected={','.join(selected)}; skipped/reused={len(report['skipped_gates'])}", file=sys.stderr)
        return 2 if plan["verdict"] == "BLOCKED" else 0
    except (SelectionError, OSError, KeyError, TypeError, ValueError) as exc:
        report = {"schema_version": 1, "verdict": "BLOCKED", "reason": str(exc), "selected_gates": [], "skipped_gates": []}
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
