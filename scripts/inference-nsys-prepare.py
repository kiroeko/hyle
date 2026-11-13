#!/usr/bin/env python3
"""Create single-request diagnostic sources with exact prefill NVTX markers.

This inserts observation only; it does not build or run either consumer.
The reference source must be the frozen matched-input consumer. The generated
Hyle source ends collection when the original prefill publication is observed,
before the same pump can prepare decode. Source anchors must be unique and
every edit must reverse exactly, so changed callers fail instead of guessing.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REDIRECT = r'''
#include <cstdio>
#include <cstdlib>
namespace hyle_profile_io {
bool redirect(const char *key, FILE *stream) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
  const auto *path = std::getenv(key);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  if (!path || !*path) return true;
  FILE *opened = nullptr;
  return ::freopen_s(&opened, path, "wb", stream) == 0 && opened;
}
bool initialize() {
  if (!redirect("HYLE_DIAGNOSTIC_STDERR", stderr)) return false;
  return redirect("HYLE_DIAGNOSTIC_STDOUT", stdout);
}
}
'''
TRACE = r'''
#pragma push_macro("interface")
#pragma push_macro("min")
#pragma push_macro("max")
#undef interface
#undef min
#undef max
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#pragma pop_macro("max")
#pragma pop_macro("min")
#pragma pop_macro("interface")
namespace hyle_profile_trace {
class Collection final {
  bool active_ = false, started_ = false;
 public:
  void begin(const char *whole, const char *body) {
    if (started_) return;
    const auto error = cudaProfilerStart();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    nvtxRangePushA(whole); nvtxRangePushA(body);
    active_ = started_ = true;
  }
  void finish() {
    if (!active_) return;
    nvtxRangePop(); nvtxRangePop(); active_ = false;
    const auto error = cudaProfilerStop();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }
  bool active() const noexcept { return active_; }
  ~Collection() { if (active_) { nvtxRangePop(); nvtxRangePop(); (void)cudaProfilerStop(); } }
};
inline Collection *prefill = nullptr;
class HostRange final {
  bool active_ = false;
 public:
  explicit HostRange(const char *name) : active_(prefill && prefill->active()) {
    if (active_) nvtxRangePushA(name);
  }
  void next(const char *name) {
    if (active_) { nvtxRangePop(); nvtxRangePushA(name); }
  }
  void finish() { if (active_) { nvtxRangePop(); active_ = false; } }
  ~HostRange() { finish(); }
};
}
'''


def annotate(origin, arm):
    before = origin.read_text(encoding="utf-8")
    text, changes = before, []

    def edit(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"{arm}: missing or ambiguous source anchor: {old[:100]!r}")
        text = text.replace(old, new, 1)
        changes.append((old, new))

    if arm == "hyle":
        # Flatten only the diagnostic consumer and the internal headers which
        # include the instrumented core. Preserve pragma-once semantics across
        # their shared dependencies; all other headers remain ordinary includes.
        expanded = set()
        inline = {"generation_runtime.cc", "generation_report.cc", "generation_state.h",
                  "generation_diagnostics.h", "model_core.h", "cache_report.h", "device_mtp_planning.h"}

        def expand(source, parent):
            def include(match):
                resolved = (parent / match.group(1)).resolve()
                if not resolved.is_file():
                    return match.group(0)
                if resolved.name not in inline:
                    return '#include "' + resolved.as_posix() + '"'
                if resolved in expanded:
                    return "// already expanded: " + resolved.name
                expanded.add(resolved)
                body = resolved.read_text(encoding="utf-8")
                if resolved.suffix == ".h":
                    if not body.startswith("#pragma once\n\n"):
                        raise ValueError("hyle: unexpected internal header: " + resolved.name)
                    body = body.removeprefix("#pragma once\n\n")
                return expand(body, resolved.parent).rstrip("\n")
            return re.sub(r'^#include "([^\"]+)"', include, source, flags=re.MULTILINE)

        if re.search(r'^#include "', text, flags=re.MULTILINE):
            edit(before, expand(before, origin.parent))
        marker = "namespace qwen35 {\nnamespace sampling = hyle::inference::sampling;"
        edit(marker, REDIRECT + TRACE + "\n" + marker)
        marker = "  const auto start = Clock::now(); auto endpoint = start;"
        edit(marker, "  hyle_profile_trace::Collection prefill_trace;\n"
                     "  hyle_profile_trace::prefill = &prefill_trace;\n" + marker)
        marker = "      return prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests,"
        edit(marker, '      if (!request.progress.prefilled) prefill_trace.begin("hyle.prefill", "hyle.prefill.body");\n'
                     '      else nvtxMarkA("hyle.generation.prepare");\n' + marker)
        marker = "    request.prefill_endpoint = record.endpoint;"
        edit(marker, marker + '\n    if (hyle_profile_trace::prefill) {\n'
                     '      nvtxMarkA("hyle.prefill.ledger_endpoint.observed");\n'
                     '      nvtxMarkA("hyle.prefill.capture_closed_at_ledger");\n'
                     '      hyle_profile_trace::prefill->finish();\n    }')
        marker = "      const auto cold_start = Clock::now();"
        edit(marker, marker + '\n      hyle_profile_trace::HostRange host_cold("hyle.host.cold_batch");\n'
                     '      hyle_profile_trace::HostRange host_stage("hyle.host.layout");')
        marker = "      const auto batch_start = Clock::now();"
        edit(marker, '      host_stage.next("hyle.host.batch_prepare");\n' + marker)
        marker = "      const auto batch_end = Clock::now();"
        edit(marker, '      host_stage.next("hyle.host.replay_prepare");\n' + marker)
        marker = "      const auto replay_end = Clock::now();"
        edit(marker, marker + '\n      host_stage.finish();')
        marker = "      step.cold_batch_seconds = std::chrono::duration<double>(Clock::now() - cold_start).count();"
        edit(marker, marker + '\n      host_cold.finish();')
        entry = ('\nint main(int argc, char **argv) { if (!hyle_profile_io::initialize()) return 1; '
                 'return qwen35::run(argc, argv); }\n')
        text += entry
        changes.append(("", entry))
    elif arm == "peer":
        edit("namespace peer_mtp {", REDIRECT + TRACE + "\nnamespace peer_mtp {")
        marker = "int main(int argc, char ** argv) {\n"
        edit(marker, marker + "    if (!hyle_profile_io::initialize()) return 1;\n")
        marker = "    Point journey_begin{}, endpoint{}, first_prefill_end{};"
        edit(marker, marker + "\n    hyle_profile_trace::Collection prefill_trace;")
        marker = '        marker(label("prefill_begin").c_str(), Clock::now());'
        edit(marker, '        prefill_trace.begin("peer.prefill", "peer.prefill.body");\n' + marker)
        marker = '        marker(label("prefill_end").c_str(), prefill_end); marker(label("decode_begin").c_str(), decode_begin);'
        edit(marker, marker + '\n        nvtxMarkA("peer.prefill.endpoint.observed");\n        prefill_trace.finish();')
        marker = "            const auto begin = Clock::now(); const auto base = length, old_frontier = frontier, old_lag = lag;"
        edit(marker, '            nvtxMarkA("peer.generation.prepare");\n' + marker)
    else:
        raise ValueError("unknown arm")

    restored = text
    for old, new in reversed(changes):
        if restored.count(new) != 1:
            raise ValueError(f"{arm}: annotation cannot be reversed uniquely")
        restored = restored.replace(new, old, 1)
    if restored != before:
        raise ValueError(f"{arm}: annotation changes the underlying source")
    return before, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyle-source", type=Path, default=ROOT / "engine/models/qwen35/main.cc")
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory under repository .scratch")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        scratch = (ROOT / ".scratch").resolve()
        if not output.is_relative_to(scratch) or output == scratch or output.exists():
            raise ValueError("Output must be a new directory under repository .scratch")
        # Prepare and verify both sources before creating any output.
        generated = []
        for arm, source in (("hyle", args.hyle_source), ("peer", args.reference_source)):
            source = source.resolve()
            before, after = annotate(source, arm)
            generated.append((arm, source, before, after))
        output.mkdir(parents=True)
        notes = []
        for arm, source, before, after in generated:
            destination = output / (arm + "_decode.cc")
            destination.write_text(after, encoding="utf-8", newline="\n")
            (output / (arm + ".diff")).write_text("".join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile=str(source), tofile=str(destination))),
                encoding="utf-8")
            notes.append(f"{arm}: source={source}; exact_reverse=PASS")
        notes.append("HEAD=" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        notes.append("Single-request diagnostic only; generated source keeps original warmup, math and publication.")
        notes.append("Hyle closes capture upon observing the original prefill ledger publication before decode preparation.")
        (output / "sources.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print("\n".join(notes))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"inference-nsys-prepare: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
