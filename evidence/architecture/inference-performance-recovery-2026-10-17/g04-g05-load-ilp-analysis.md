# G04/G05: D256 K-load instruction-level parallelism diagnosis

2026-10-17; canonical HEAD `8e04eb0c0`. New user-authorized bounded diagnosis;
no product, registry, old SDK, old contract or old performance receipt is changed.
The G04/G05 final terminal and CE21–CE28/C1 stopping boundaries were read first.
This record does not reopen implementation or replace the original verdict.

## Bound identities and measured facts

- Full product `669121418669b31f6849c74f64362aaf8d6eb919`, FI
  `0a29d119426b2975d48c238c37a2994675e0ba9f`, original CE28 math
  `830dd9e01829e3e5850ddff7eaa6355ee7c8fd0d`.
- Original attempt14 is QUALIFIED: prefill direction LOSS, 5% noninferiority
  UNRESOLVED. It compares immutable FP8/F32-partial warm execution with an
  additional cold-expanded BF16 snapshot and BF16 partials; storage and numerical
  constraints differ. It is not a llama.cpp engine comparison.
- Recomputed from attempt14 summary: eight prefill median ratios imply
  4.018–4.935% latency excess. The two unresolved lower confidence bounds,
  0.9511281428694289 and 0.9513110560004704, are below 1/1.05. At their reported
  absolute medians, the deficit corresponds approximately to 0.140 and 0.120 us.
  This is an arithmetic scale illustration, not a new CI or permission to tune
  the test. All medians themselves are inside the original 5% latency bound.
- D1 actual product timeline (12,288 episodes) found prefill attention median
  differences 3.950–4.134 us and merge 0.895–1.111 us. Its CPU/GPU clock offset
  errors prevent treating wall-minus-kernel-sum as pure host cost. D3 cropped
  counterfactual differences are not pure causal contributions: C1 invalidated
  that attribution with an unchanged-control bridge offset.

## New source and actual-SASS clue

Current `third_party/flashinfer_cache/patches.json` contains the CE28 direct-stage
producer. For D256 each thread handles two original packed 16-byte vectors per
row. The helper combines validity, load, unchanged `vec_cast<16>` and two native
BF16 shared stores. Its K `kNoFill` early return is repeated for each `j`; the row
validity predicate is identical for `j=0,1`.

The **actual product control**, not a cropped bridge, is preserved in C1
`raw.zip`, member `private-records/direct/control-sass.txt`. C1 proved its
normalized kernel instructions equal the shipped SDK kernel. In the loop:

- `0x3bc0 @P1 BRA`, `0x3bd0 LDG.E.128 R4,[R72]`, then immediately `0x3be0 PRMT`;
- first conversion ends in K stores `0x3ec0/0x3ee0` to shared `+0x8000`;
- only after another same-predicate branch does `0x3f40` load `[R72+0x80]`, again
  into R4. There are 55 instruction slots between the two global loads.
- By contrast the same kernel's V path issues its two predicated packed loads
  at `0x4540` and `0x4570` before conversion, storing a separate tile `+0xa000`.

This proves a K source-dependency ordering difference. It does **not** prove
memory latency is the dominant stall or predict a speedup. It provides a more
specific hypothesis than another FP8 conversion-opcode experiment.

## One bounded compile-only counterfactual

For the existing direct-stage/F32/E4M3/equal-D256 K producer only:

1. Evaluate the common row-valid predicate once. Invalid K rows return without
   any global load or shared write, exactly preserving `kNoFill`.
2. Load both original packed vectors before calling the original decoder on
   either vector; keep all output coordinates, special codes, scales and stores.
3. Preserve D128, V, other dtypes/shapes, FP8 backing, F32 partial/merge, final
   scale/cast, page permissions, tile, planner, public API and all barriers.

Expected additional live payload is four 32-bit registers; current product
uses 248 registers. This is a register-pressure risk, not a resource promise.
Acceptance of the compile-only step requires inspecting actual emitted order,
register count, local spill, shared bytes and original kernel identity. If the
second load remains serialized, or new spills/resources appear, report that
fact and stop; do not scan alternative variants.

Use one disposable header tree and the original full `native_cache.cu` TU,
original SM89/O3/C++17/CUDA13.3 command with only include/output paths changed,
at most four CPU cores and nice 10. Reuse the actual unchanged archive-member
control. Do not execute either object, touch driver permissions, launch GPU work,
rebuild ggml/the SDK, or repeat C1. A GPU slot and separate numerical/permission
validation must be provided by the coordinating owner before any execution.

Compile-only source, command, hashes, logs and codegen result are placed in the
owned scratch directory `.scratch/inference-performance-recovery-2026-10-17/`.
No runtime or performance PASS is produced by this step.

## Compile-only result

The one original full TU compiled successfully in **291.811 seconds**, on CPU
affinity 0–3, nice 10, one top-level nvcc invocation with the original
`--split-compile=3`. CUDA13.3/SM89/O3/C++17 flags and original source path were
retained; only the disposable header include/output paths and verbose ptxas
reporting differ. No GPU, SDK/ggml rebuild, driver query or benchmark executed.

| Target prefill codegen | Original actual SDK object | Load-ILP diagnostic |
|---|---:|---:|
| Registers | 248 | 250 |
| Stack / local / spill loads / spill stores | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Dynamic shared footprint | 49,152 bytes, original runtime identity | Original shared type and launch-size expression unchanged; no new runtime query |
| Static instruction count | 1,809 | 1,809 |
| BRA / BSSY / BSYNC | 14 / 8 / 8 | 12 / 6 / 6 |
| PRMT / HFMA2 / HMMA / STS | 32 / 64 / 65 / 16 | 32 / 64 / 65 / 16 |
| K loop packed loads | `0x3bd0`, `0x3f40`; conversion between | `0x3be0`, `0x3bf0`; conversion starts `0x3c00` |

The candidate therefore makes the intended two global loads adjacent before
either decode, using two extra registers and no spill. The source-level shared
layout and existing 128-thread block are unchanged; with the original 49,152-byte
shared requirement neither register count reduces the two-block-per-SM resource
ceiling. This is a static resource bound, not a device occupancy measurement.

The control object SHA256 is
`ff214a327ca8c9d6d22e8657bf9510586f2ae00eb0466f0ebae1a464a60b716f`.
Its normalized target SASS SHA256 is
`3e3ea55d912657cf12b7873dc7f06d84de1689069fad126fcdffd83033615ea9`,
exactly matching C1's actual SDK control. Candidate object SHA256 is
`106c5a7e005d375a58f596b945ca6f5734323a1bc0df3f6ee1af7cfd061c4061`;
candidate normalized target SASS SHA256 is
`7f555eae49d6c662123410e3d7e423c4d7227ae9e4ab054fd60a700aa7af15a6`.
Original/candidate prefill hashes are
`8146bd05db566d6f3c999a3aa3b83d91f7a565a343d06780edf2abd5faa0840f` /
`395e579a37968b041c37ea5eb68434d3f39ba9907ef5c2cfaf23f8818f3ce0b5`.

The local candidate object is
`/root/hyle-scratch/compute-unified/g04-load-ilp-2026-10-17/native_cache_load_ilp.o`.
Owned Windows scratch contains `g04-load-ilp-compile-receipt.json`, the exact
variant patch, compilation log, both resource/SASS dumps and
`g04-load-ilp-codegen-analysis.json`. The coordinating owner was notified that
compilation ended and the serialized heavy-work slot is free.

This passes only the intended load-order/no-spill **codegen** checkpoint.
Correctness on device, tail/readonly/race behavior and latency benefit remain
unmeasured. Do not promote this into product or claim a fix without the separate
finite execution and legal owner/card transition.

## Proposed finite execution (pending coordinating-owner review)

Preparation/linking may proceed; **no GPU execution is authorized by this
proposal until the coordinating owner reviews the actual source and launcher**.

Two executables use the same cloned, previously audited C1 public lifecycle /
independent-double oracle / readonly validator. The control explicitly links the
actual SDK `native_cache_control.o`; treatment links only the new original-full-
TU load-ILP object. All remaining SDK, consumer source and link options match.
The BF16-expanded provider/reference object/library is removed entirely, not
merely omitted from timing. Each executable constructs two FP8/F32 contexts using
its one native implementation: these two identical implementations are an
instance/order/allocation negative control, **not** a control-vs-treatment pair.
The implementation comparison is explicitly between separately linked processes.
Do not falsely report same-process A/B.

Freeze these original three shapes and original Data/oracle/permissions:

- window: Q1/17, KV1024/768, D256/Dv256, causal, window16;
- prefill: Q64/128, KV1024/768, D256/Dv256, causal, unbounded;
- holdout: Q17/1, KV7/47, D64/Dv128, noncausal.

All retain page16, GQA8/2, original shared-page/reverse-ID/tail layout, FP8
key/value scales .5/2, BF16 Q/output, F32 partial/merge, two invocation slots,
actual checked Completion, owning output release and normal drain/close.
Only prefill's selected F32 D256 kernel has the load-ILP intervention; the window
and holdout are unchanged-shape/codegen negative controls, not new variants.

Bounded runs, all kept including failures:

1. One VERIFY_ONLY process per executable: four changed-slot replays per shape,
   two identical contexts, independent oracle `max_abs_error <= 8e-3`, finite
   outputs and byte-identical original KV pages. Coordinating/independent review
   added exactly one necessary D256 partial-tail case, Q64/128 KV1023/767,
   causal/unbounded, only in VERIFY_ONLY and the sanitizer paths. It keeps the
   original sentinel/shared/reverse-page Data/oracle. Require all **32** oracle
   records, eight complete context teardowns and one tail-only marker. This case
   is never added to timing, which still has only the original three shapes.
2. One memcheck VERIFY_ONLY process per executable and one treatment racecheck
   VERIFY_ONLY process, original tools and error-exit86. Stop on any mismatch,
   numerical failure, device error, hazard, skipped shape or incomplete shutdown.
3. Four paired short process groups covering execution-order and allocation-order
   factors. Each group runs one control and one treatment process; implementation
   process order is balanced C,T / T,C / T,C / C,T. Exactly eight timeline
   processes total, each same 64 warmups then eight groups of64 episodes/context/
   shape, original NVTX/correlation labels and nsys CUDA/NVTX trace with no CPU
   sampling. No NCU/hardware-counter request, driver setting or device policy
   change. No n15/P1, duration extension, optional extra sample, tile or opcode.
4. Extract actual object-specific runtime kernel names, grid/block/shared/register
   identity, per-context attention/merge GPU durations and complete host episode
   durations. All CPU/GPU clock offset limits remain. Show all four paired
   process outcomes and both same-implementation contexts; include the two
   unchanged shapes and self-instance differences. A cross-process clock/control
   mismatch prevents causal attribution; do not hide it by normalization or
   favorable subsampling. No new statistical performance verdict is produced.

Host binding: serialize all GPU work with the coordinating owner; same fixed
WSL CUDA13.3/RTX4070 SM89/driver identity, pin the process to the original available
Linux caller CPU2, report actual affinity/priority/environment and binary/object/
header hashes before and after. Build remains four CPU cores/nice10/max2 top-level
jobs. Runtime uses the original diagnostic priority, does not change external
processes/overlays/driver, and records visible-device identity. GPU residency and
input bytes are identical for control/treatment; cold setup stays outside timing
and is disclosed.

The latency question is whether this exact FP8/F32 load-order change improves
the actual prefill kernel/complete lifecycle while its unchanged cases remain
stable. An apparent gain need not erase the full original 4–5% cross-representation
gap, and cannot retroactively pass attempt14. No discernible gain or noisy
negative controls terminates this one mechanism without more samples. Positive
evidence still requires a separately audited product FIX and applicable tests /
formal-contract recovery before adoption.

### Link and review binding

The consumer was linked against the existing G06 relocated static SDK without
rebuilding the SDK or CUDA TU. Original configure/link logs and the two later
host-only relink logs (review-mandated tail, explicit successful-exit markers)
are retained in owned scratch. The final consumer SHA256 is
`3911915b43d8b657f0d4fb743fb77342d61d300eb726d4514b0e68eb400c8af5`.
Final control/treatment executable SHA256 values are
`087e54190af639fe22a0d6cb78faaa38efb7a4a53a6eb28c2586fa11ecd4d9b5` /
`cf8015adf4649f2081862c58664d09e50fb87e7700d9d1b8089249fb90c4d970`.
Both explicit `.o` files still equal their compile receipts. Link maps show
the explicit full TU object and **no extraction** of SDK archive member
`libhyle_compute_provider_flashinfer.a(native_cache.cu.o)`. Link commands/maps
exclude the original BF16 comparator library and reference TU; nm finds no
original-BF16-peer namespace.

The actual linked control/treatment CUDA functions were extracted before the
last host-marker-only relink. Window is CTA64, not CTA16: its emitted template
has **BF16 `DTypeO`**, so the existing `is_same_v<DTypeO,float>` direct-stage gate
is false. No claim that window masking itself excludes the patch is made.
Normalized SASS is exactly equal for the linked window, holdout and F32 merge:

- window: `b2094d597d01969614df3b0dcb8bf0a0eed0e824c7591d94a9b3238681c0ee8e`;
- holdout: `b374212989815c1525b06a59f7111039f4e1145de2b988e96cc802c3c60cd5bf`;
- merge: `add79d37f8ae25d145360368e7b15331668df6dfacb13eb9b2b29b61b798e6aa`.

`cuobjdump` emitted per-fatbin “function not found” warnings for other embedded
modules; the parser independently required exactly one of each requested
function overall and obtained each full nonempty instruction stream. No missing
kernel was silently accepted. Runtime dispatch identity remains to be checked in
the proposed capture; source and linked SASS are not a new GPU execution PASS.

The actual launcher is owned scratch `g04-load-ilp-run.py`, guarded by explicit
`--execute-reviewed-gpu`. It freezes launcher/source/objects/binaries/compile and
link metadata hashes, records driver/UUID/tool versions and actual CUDA ENV,
checks them after execution, and kills its own `start_new_session` process group
on timeout/exception before wait. Each normal application must publish one
`LOAD_ILP_APP_EXIT code=0` and all expected context teardowns. There is no GPU
execution result in that preparation receipt; actual execution is recorded below.

### One reviewed runtime batch and stop

The coordinating owner authorized the final launcher SHA256
`07be8eb852a3b6b5d00be5ebe49b92925f65e4cc3c4d9bb7dcb9aecd3cab1acf`
after the independent execution review passed. Exactly one frozen batch ran:
five numerical/tool processes and eight timeline processes. All 13 exited zero
with the unique application-success marker and complete lifecycle teardown;
no timeout, cleanup, retry or extra capture occurred. All 14 frozen input
artifacts matched before and after. The control and treatment each passed 32
oracle records / eight teardowns, including the one D256 partial-tail case.
Both memchecks and the treatment racecheck reported zero errors/hazards.
Maximum absolute oracle error across every process was `0.000994979551`, against
the unchanged `8e-3` bound; original KV pages remained byte-identical.

Actual host was WSL2 Linux `6.18.54.1`, RTX4070
`GPU-e7376560-6f5d-ec19-6ee4-3dd3a2c81ef8`, driver `617.14`, CUDA `13.3.73`,
Nsight Systems `2026.1.3.425`, compute-sanitizer `2026.2.1.0`; caller affinity
was CPU2 and original diagnostic nice0. No runtime clock/frequency/power telemetry
was collected. No driver/device policy or external process was changed.

The eight SQLite traces contain **37,184 raw kernel records**. Only the **32,768
kernel records correlated to 24,576 measured episode ranges** enter the timing
summaries; 4,416 records are outside measured episodes (the trace also captures
oracle/warmup work). Each measured process contributes 3,072 episodes and 4,096
mapped kernels; both same-implementation contexts have 512 episodes per shape.
Actual runtime names matched the four expected attention/merge templates, with
no BF16 comparator. Prefill registers were 248 to 250; window241, holdout165 and
merge48 were unchanged. Grid/block/shared resources matched, and local memory
was zero. Static ptxas/SASS receipts independently show no spill.

All numbers below are raw median microseconds over both contexts, control to
treatment; percentages are `(treatment/control - 1) * 100`. GPU sum is the median
per-episode sum of attention and merge durations. Host is the complete NVTX
episode duration, measured on its own timebase; it is not CPU overhead.

| Pair / allocation / execution / process order | Prefill attention, us (%) | Prefill GPU sum, us (%) | Prefill host episode, us (%) |
| --- | --- | --- | --- |
| 0 / 0 / 0 / C,T | 42.457 to 41.658 (-1.882%) | 52.3095 to 51.697 (-1.171%) | 102.6615 to 88.050 (-14.233%) |
| 1 / 0 / 1 / T,C | 42.511 to 40.797 (-4.032%) | 52.420 to 50.611 (-3.451%) | 87.4485 to 85.326 (-2.427%) |
| 2 / 1 / 0 / T,C | 42.450 to 39.501 (-6.947%) | 52.309 to 49.026 (-6.276%) | 97.011 to 84.2375 (-13.167%) |
| 3 / 1 / 1 / C,T | 41.679 to 40.693 (-2.366%) | 51.396 to 50.483 (-1.776%) | 89.150 to 85.244 (-4.381%) |

The unchanged negative controls must be read alongside every pair:

| Pair | Window attention, us (%) | Holdout attention, us (%) | Window host episode, us (%) | Holdout host episode, us (%) |
| --- | --- | --- | --- | --- |
| 0 | 6.034 to 6.148 (+1.889%) | 4.429 to 4.513 (+1.897%) | 53.625 to 39.8285 (-25.728%) | 28.579 to 48.851 (+70.933%) |
| 1 | 6.069 to 6.030 (-0.643%) | 4.455 to 4.426 (-0.651%) | 39.6095 to 39.4145 (-0.492%) | 31.020 to 29.4635 (-5.018%) |
| 2 | 6.037 to 5.834 (-3.363%) | 4.431 to 4.282 (-3.363%) | 39.519 to 39.0715 (-1.132%) | 48.269 to 28.1585 (-41.663%) |
| 3 | 5.932 to 6.015 (+1.399%) | 4.354 to 4.415 (+1.401%) | 58.2115 to 39.9125 (-31.435%) | 31.553 to 29.586 (-6.234%) |

Every context's median/mean, all eight group64 means, self-instance ratios and
all four comparisons are retained in `g04-load-ilp-runtime-analysis.json` inside
the source/raw archive. Independent read-only recomputation used same-host-thread
correlation launches and unique containing episodes, and agreed with every
reported value (maximum difference zero microseconds). Its separately owned
[review](g04-load-ilp-independent-review.json) also verified all raw names,
resource identities, counts, logs and frozen hashes.

The finite result is **codegen change proven; numerical/permission/lifecycle
checks passed; shape-local raw timing signal retained; causal performance
qualification unresolved**. Prefill attention and GPU sum improved in all four
raw process pairs, but the two unchanged kernels drift together by -3.363% to
+1.897%, and negative-control host durations vary greatly. This establishes an
unresolved cross-process timing/resource perturbation; it does not establish a
clock cause. No negative-control subtraction, host-minus-GPU arithmetic or
favorable pair/context selection is used to manufacture a pure load-ILP benefit.
This is not P1, a new confidence interval, a product FIX, or a recovery of the
original attempt14 noninferiority verdict. G04/G05 retain their prior performance
verdicts. Per the frozen stop rule this one variant is stopped, with
no further sample, bridge, tile/opcode variant or clock intervention authorized.

One CUDA full-TU compile took 291.811 seconds; CMake configure 30.934 seconds;
initial host consumer link 3.760 seconds; the necessary tail-only and exit-marker
host relinks 3.959 and 3.360 seconds. All five tools/checks plus eight timelines
took 89.906 seconds total, including the 51.717-second racecheck. Offline
analysis/review/archive time is separate and is not claimed as measured runtime.
No compile/link/GPU/numerical/permission/identity failure occurred. Read-only
setup initially queried the absent WSL distro `Ubuntu` and unavailable WSL `rg`;
the work used `Ubuntu-24.04` and available text readers. The already documented
per-other-fatbin `cuobjdump` warnings were retained. No failed work was dropped
or reclassified as a pass.

The complete owned source/header/patch/launcher/log/map/codegen evidence, all
eight raw SQLite files and their original `.nsys-rep` files are sealed in
`source-and-raw.zip`. Its file manifest, archive hash, recorded finite costs and
failure/cleanup accounting are in `g04-load-ilp-receipt.json`; no SDK archive or
CUDA object/executable payload is duplicated. This evidence is reproducibility
input, not a new authorization to execute it.
