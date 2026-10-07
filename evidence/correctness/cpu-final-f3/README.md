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
