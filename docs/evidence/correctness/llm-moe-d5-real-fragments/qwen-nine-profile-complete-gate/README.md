# Nine Qwen profiles: complete recurrent gate and ordinary entries

The [receipt](receipt.json) qualifies the required fragments for the nine fixed
Qwen3.5-2B, Qwen3.8-27B and Qwen3.8-2.4T official/GGUF profiles through CPU,
CUDA, Apple and Vulkan Direct/Graph: **72 ordinary-entry cells**. This is
fragment correctness and lifecycle coverage, not whole-model inference,
three-delivery acceptance or performance qualification.

## Complete recurrent boundary

The shared recipe executes convolution, activation, Q/K normalization and
official/GGUF head mapping, raw a/b gate plus original ggml GDN, state update,
output normalization and gating. Each profile runs eight changed-input steps
and feeds its actual output state into the next call. The independent oracle
advances its own state. Official 2B uses local checkpoint convolution and gate
parameters; other profiles use nonzero synthetic payloads with their fixed
real widths, formats and state dimensions.

Public recurrent outputs retain `abs_error <= 2e-6 + 4e-5*abs(reference)`.
The separate [native numerical probe](../native-recurrent-gate-qualification/README.md)
retains its failed intermediate-gate verdict; it cannot qualify a standalone
gate API. The new operation returns attended output and state, as required by
the approved complete-region contract. Generic Softplus accuracy is unchanged.

## Executed checks

| Route | Ordinary suites | Additional checks |
|---|---|---|
| CPU | Direct 82, Graph 97, all pass | Ops 53; ggml 41; common/backend/native/oneDNN/Graph owners; header and oneDNN consumers |
| CUDA | Direct 81, Graph 98, all pass | ggml 28, Graph ggml 8, backend 27, cuBLAS 6, existing CUDA Provider 42; late-failure CTest |
| Apple | Direct 111, Graph 84, native Provider 7, all pass | Metal static Provider/fault suite 11; after resource fix, shared Provider 7 plus both nine-profile tests |
| Vulkan | Graph 106 pass; Direct 183/195 initially, all 12 failures covered by a passing 16-test correction | ggml 22 plus separately enabled validation-layer case; IREE 7 |

CPU/CUDA full suites bind candidate `83be4e1b`; Apple full suites bind
`2ed078ce`, with native-buffer correction and targeted shared regression at
`969c56c0`. Vulkan correction and Graph bind `cb16f5669`. Metal final static
fault assertions bind tests-only `877c9f5b`; product bytes are unchanged since
`969c56c0`. Full commit IDs and raw logs are in the receipt.

Vulkan initially traversed inactive slots in its newly enlarged input array
and retained an obsolete custom single-output guard. The correction limits
inspection to active inputs and admits two outputs for this named operation.
The initial failure remains recorded; this is reuse plus targeted correction,
not a claimed fresh 195-test passing invocation.

All runs use reused Release trees, HYLE warnings as errors, low build
parallelism and serial heavy validation. Windows RTX 4070 runs CUDA/Vulkan;
MacBook Air Mac16,12 runs native Apple. These are manually batched checks,
not a fabricated full selector-run receipt. No unrun delivery/P1 gate is signed.

## Metal cold-allocation correction

The new workspace fault test exposed a real null dereference in original
`ggml_metal_buffer_init`. MB1-MB3 add a null check and use the existing cleanup
at partial initialization and C++ backend-wrapper failure. Original math,
kernels and fusion are unchanged. The initial crash report, failure logs,
dependency-notice hash correction and final static/shared logs are retained.

Final static tests inject every reached C metadata and C++ allocation failure,
check failure categories and native allocation balance, and reach successful
construction after the failure points. Workspace failures are
`ResourceExhausted`; the existing native backend constructor exposes only null
and remains `Unavailable`. An intermediate test incorrectly required every
backend null to be `ResourceExhausted`; its failure is retained and the
assertion now follows the existing boundary. No product error was hidden.
MTLBuffer and residency-set system failures have source-audited cleanup; these
OS failures were not injected by the C/C++ allocation hooks.

Only the Apple-conditional manifest changed. The common, CUDA and Vulkan
manifests and non-Metal product inputs remain byte-identical. Their completed
checks are reused; the conservative dependency-lock selector plan is retained.

## Coverage and limits

Non-R05 six-profile regions have both retained earlier receipts and passing
ordinary suites here. The 2.4T suites additionally cover all 512 router scores,
top10 changing/near-tied selections, native grouped experts and shared expert,
full stored projection roles, causal attention and MTP. Vocabulary checks keep
all 248320 rows; only permitted token/expert/context capacity is reduced.
The receipt maps these tests to required regions. The prior 48-cell R05
retraction remains in ledger history and is replaced by this complete boundary.

The ledger is now **72 RUN_PASS / 104 VALIDATION_NOT_RUN**. The other 13
profiles, D5-A5 external source/static/shared consumers, D5-A6 applicable P1
and D6 final review remain open. No new HYLE operator math was added.

The unified [LOC report](loc.json) counts **58,255 implementation lines in
228 files** for all Compute + Graph, excluding tests and vendored code.
