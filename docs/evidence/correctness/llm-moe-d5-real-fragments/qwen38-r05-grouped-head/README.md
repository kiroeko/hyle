# D5 Qwen3.8-27B official BF16 R05 grouped-head partial receipt

Candidate `d7cb7e016b3c620ed3fd9e6839a748257cdca42a` fixes a Provider admission defect exposed by the
official Qwen3.8-27B Gated DeltaNet shape. The public operation and pinned ggml implementation both accept
K16/V48 grouped heads at depth 128, but HYLE's four ggml adapters had incorrectly required equal K/V head counts.

The shared fixture fixes `Qwen/Qwen3.8-27B@1d4bf0f`, the official config and safetensors-index digests, R05,
decode T=1, the full K16/V48×128 head geometry, nonzero inputs and recurrent state. An independent double-precision
oracle checks both the output and new state. Every route prepares through its ordinary Direct and Graph entry,
submits the same prepared program twice with changed inputs, and reports the pinned mature ggml Provider.

- CPU Direct/Graph: PASS on Windows; targeted runs reported 15/16 ms. ggml Provider 33/33 PASS; route suites
  13/13 and 28/28 applicable tests PASS.
- CUDA Direct/Graph: PASS on RTX 4070, CUDA 13.3; targeted runs reported 96/95 ms. ggml Provider 21/21,
  ggml Graph Provider 8/8, route suites 10/10 and 29/29 applicable tests PASS.
- Vulkan Direct/Graph: PASS on RTX 4070, driver 610.62; targeted runs reported 221/240 ms. ggml Provider 19/19,
  IREE Provider 7/7, route suites 127/127 and 40/40 applicable tests PASS.
- Apple Direct/Graph: PASS on formal Air `Mac16,12`, Apple M4, macOS 27.0; targeted runs reported 94/60 ms.
  Route suites 46/46 and 19/19 applicable tests PASS.

The listed suite counts exclude pre-existing, explicitly skipped tests whose external Qwen3.5 fixture or optional
Vulkan validation-layer environment was not supplied. No applicable Qwen3.8 R05 test was skipped.

This is a **partial R05 receipt**. It proves the grouped-head region on all four routes, but does not complete any
of the eight Qwen3.8 profile/route Direct/Graph ledger cells because the other required regions, three delivery
forms and P1 remain outstanding. No HYLE math, kernel, shader, fallback or hot-path search was added.

Machine-readable details are in [receipt.json](receipt.json). The immutable-candidate Compute + Graph LOC report
is in [compute-graph-loc.json](compute-graph-loc.json).
