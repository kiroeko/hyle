# CPU final F3: Mac carrier partial qualification

Current task: [F3 platform carriers](../../../docs/implementation/tasks/compute/cpu-final-f3-platform-carriers.md).
This is not a terminal record or the final CPU coverage certificate.

The product comes from the immutable Windows archive of
`42898c37ef90005944b73f4bf174f9b7f1070550`. The MacBook Air validation copy runs
ARM64 CPU ggml and official LibTorch 2.8.0 with the HYLE Apple route disabled.
The managed SDK retains the original archive hash and derives a loader-relative
libomp dependency; no external user SDK was modified.

Actual shared/source checks:

- Provider owning groups pass. Ordinary Direct 12 / Graph 11 cases pass without skips,
  including a projection using the canonical local Qwen3.5 checkpoint fixture.
- C5 Direct originally passed 652 of 653 cases. The original single failure remains
  in the archive. Correction `6359804abc6e0543f3fef9a4b6b622f01a8b98db` passes the
  complete failed case with every head shard; the prior 652 cases are reused.
- C5 Graph passes all 653 cases without skips after the same correction.

The correction changes the downstream independent oracle to consume the actual
upstream activation that feeds the downstream program. The upstream retains its
independent oracle and original tolerance/state feedback; a legal F32 difference
crossed one BF16 rounding boundary. No tolerance, dtype, weight or branch changed.
The diagnostic first-shard run is not a qualification result.

[Receipt and member hashes](receipt.json) bind the
[original logs/XML, corrections, source identity and selected plan](mac-carriers-c5.zip).
The original aggregate Direct failure is not rewritten as a passing full rerun.
These contracts cover actual quantization and state fragments, not a claim that
all target large-model weights were resident and inferred end to end.

Installed static/shared relocation, Linux current-version checks and final
cross-platform journeys remain pending. No new performance WIN/P1 is claimed.
The unified script reports Compute + Graph implementation **61,242 lines / 251
files**, excluding tests/vendor/bench.

[Cleanup receipt](cleanup.json): original 62.6 MiB Direct failure log/XML now remain
in the verified archive; redundant scratch copies were removed after matching
each member hash. Active source, producer and fixture inputs remain available.

## Mac three C++ deliveries

Candidate `9a9936e8271fc834f4b090316bc0a5d7f9b25982` is packaged by the existing
Windows complete-source script with a commit manifest and archive digest. Source
embedding runs both ordinary public consumers. Each shared/static SDK is moved
before independent Direct and Graph consumers configure and execute. All six
consumer checks pass without skips; the shared scan checks 16 HYLE libraries and
1,262 exports.

[Delivery receipt](mac-cpp-delivery-receipt.json) and
[raw delivery logs/XML/loader inspection](mac-cpp-delivery.zip) retain the one
missing-Ninja-PATH environment failure and its bounded continuation. Successful
producers were not rebuilt for it. Four ARM64 Torch CPU dylibs were inspected;
torch_cpu loads its adjacent libomp. The original machine CPU SDK is a separately
resolved dependency of the installed shim, not bundled into the HYLE binary
package. Full C5 on installed shapes and Linux current-version results remain
pending, so this section does not close F3.

The same relocated shared/static SDKs each pass the reused C3 numerical format
consumer's 36 Direct/Graph checks: eleven GGUF additions, NF4/nested/AWQ/CT decode
and projection, plus affine4/MXFP4/MXFP8 selected experts. Its exact source hash is
recorded in the delivery archive. These are format/delivery checks; complete model
state qualification remains the separate C5/official fragment proof.

## Linux current mature carriers

The same Windows candidate/archive is forwarded to an idle ext4 verification
copy, with GCC 14.2, oneDNN 3.13.4, ggml 0.26 and LibTorch 2.8.0+cpu. Eleven owning
groups pass. Direct's original full run passes 821 of 828 cases; all seven failures
are the same fixture-directory error. Correcting only the environment to the
archive's `fixture/` child passes those seven cases in 1.561 seconds. The original
aggregate FAIL is retained, and the other 821 cases are reused.

Graph's first full run passes **846 cases without skips**, including the current
official mixed-precision main programs, three-stage DSpark/MTP sidecars and C5
quantized/state contracts. [Linux receipt](linux-current-carriers-receipt.json)
and [raw logs/XML/source identity](linux-current-carriers.zip) bind this evidence.
No product, tolerance or fixture-byte change was needed, and no second producer
build or duplicate full Direct run was performed. WSL is a correctness environment
and not an independent physical formal performance machine. Current installed
Linux journeys, remaining Mac model branches and the current full Windows route
batch/delivery audit are still pending.

## Mac remaining model branches and precision repair

[Precision/composition receipt](mac-precision-composition-receipt.json) and
[raw failures, two bounded probes and final qualifications](mac-precision-composition.zip)
record the remaining Direct 159 / Graph 176 cases. Initially 157 / 171 pass. The
actual ARM half-dot path accumulates in F16 despite the requested F32 metadata;
the product now declines only ARM plain F16 MatMul so the qualified original
ATen F32 program handles ordinary calls. Packed codecs and x86 remain unchanged.
Owning checks and both entries' affected 23 F16 contracts qualify this admission.

The Q8 sidecar's two bounded probes locate error growth from its first projection
through the three stages. The independent norm oracle used Double intermediate
arithmetic where the original model operation crosses explicit F32 square,
mean/rsqrt and multiplication boundaries before Q8 encoding. Test candidate
`3c62c77b9b93f127fb8f71d265cbc89247f427c0` retains those original boundaries and
passes the complete Q8/BF16 sidecars in both entries under the unchanged **4%**
whole-fragment gate. Prefix stages still have independent oracles; internal
reference/state advances independently and actual state feedback/reset remain.

All original aggregate failures and diagnostic-only runs remain in the archive.
The final Graph targeted run also passes the migrated provider/fusion assertions
and full half-storage vocabulary head. Successful earlier cases are reused;
the aggregate failures are not rewritten as full passing reruns. The current
unified implementation is **61,246 lines / 251 files**. New ARM delivery-delta,
Windows full-route and Linux current installed checks remain pending; no new
performance winner or goal closure is claimed.

## Windows original-program ISA oracle correction

[ISA receipt](windows-isa-correction-receipt.json) and
[original full Direct failure, removed norm experiment and corrected qualification](windows-isa-correction.zip)
retain the initial 827/828 Direct result. A narrow norm routing experiment did
not pass the full gate and was physically removed; no unqualified model rule
remains in the product.

Source audit found that the product's selected original AVX2 program encodes
Q8_0 activation with `127/max` and nearest-even rounding. The test's linked
baseline used `1/(max/127)` and `roundf` (ties away). These can encode different
inputs, so the baseline is not a same-contract oracle for the selected program.
The sidecar now expresses its selected native activation definition with an
independent scalar oracle; the independent dot, original weights, inputs,
three stages, state feedback and **4%** gate remain. The new midpoint check,
complete Q8/BF16 sidecar and sixteen generic entry regressions pass (18 cases).
The original aggregate failure is retained and the 827 successful cases reused.
Graph's first full batch is still pending in this partial receipt. This is not
a performance WIN or a new product codec implementation.

The first full Windows Graph run subsequently passes **847 cases without
skips** in 759.797 seconds. The receipt/archive now include its complete XML and
log. The product's generic defaults remain restored; installed-delivery and
the corresponding Linux/Mac test-seam deltas remain pending.

## Linux and Mac current C++ delivery deltas

[Linux delivery receipt](linux-cpp-delivery-receipt.json) and
[full logs/XML, original failures and consumer source](linux-cpp-delivery.zip)
qualify source consumption and relocated installed shared/static. Each shape
passes 40 Direct/Graph format/precision checks. Current ordinary samples also
pass, with actual numeric results and two changed-input Graph replays. Loader
audits have no unresolved libraries and no LD_LIBRARY_PATH dependency; the
original managed CPU runtime remains separate. The actual shared export check
passes 19 libraries / 1,391 exports after the bounded private-entry tool fix.

[Mac current delivery receipt](mac-current-cpp-delivery-receipt.json) and
[incremental producer/consumer/XML/source identities](mac-current-cpp-delivery.zip)
qualify the ARM precision admission repair in source/static/shared with the same
40 checks. Current ordinary samples were recompiled before counting their runs;
their matrix/gate output and changed-input replays pass. Original source archive
and late sample/tool deltas are separately bound, never claimed to be one unchanged
Git tree. Windows current installed and Linux/Mac T11 test-seam checks remain
pending, and F3/whole-goal closure is not claimed.

[Windows current delivery receipt](windows-cpp-delivery-receipt.json) and
[producer/install/relocation/consumer/raw DLL execution logs](windows-cpp-delivery.zip)
qualify source/shared and source/static with 40 checks each, the corresponding
relocated installed SDKs with the same checks, and current ordinary Direct/Graph
samples. Installed execution uses only System32/Windows in PATH and required
CPU DLLs reside beside the executable. All 27 recorded steps pass, including
the actual shared export scan. No dependency rebuild or performance race is
introduced by this delivery verdict. T11 Linux/Mac test-seam qualification,
F1 mixed envelopes and the remaining goal phases remain separate.

[Current Linux/Mac ISA-delta receipt](current-isa-deltas-receipt.json) and
[targeted compile/log/XML/source identities](current-isa-deltas.zip) pass both
full sidecars and midpoint definition in exactly two tests per entry/platform,
without skips. These complete the T11 test-seam deltas. The product snapshot
remains unchanged; no further model/quantization/runtime probe was run.
