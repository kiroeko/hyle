# C1-Y17–Y19: ggml Vulkan pointwise bindings, W1

Audited at `4add4d1d`; product and all runs use clean `4c309c6e`.
Seven additional mappings use original ggml Abs/Neg/Exp/Tanh/Sigmoid and Sub/Div.
Scalar descriptions map to one element. Scalar/vector-left Sub/Div preserve
operand order through an actual vendor Repeat, including its storage and work.
Ordinary Direct and Graph no longer require a norm/gate anchor for this admitted
F32 dense rank0–4 region. No new HYLE kernel or vendor algorithm patch was added.

| W1 Release / WERROR | Static | Shared |
|---|---:|---:|
| Provider cases | 19 PASS | 19 PASS |
| Direct cases | 40 PASS | 40 PASS |
| Graph cases | 17 PASS | 16 PASS |
| Source contract / MASM composition CTests | 2 PASS | 2 PASS |

The one-case difference is the existing static-only Graph global-new fault
fixture. No test was skipped; no VUID or synchronization validation diagnostic
was found. Runtime DLLs were beside each EXE, PATH was unset, and error dialogs
were disabled. Builds used parallel2 and tests ran serially in the two existing
fixed build directories. There was no new correctness correction/retry.

Independent math covers exceptional values, scalar/rank1/2/4, tails, both
noncommutative broadcast orders and Recorder replay. Ordinary Graph checks
published intermediate values, two slots, changed inputs and retained outputs.
The native-prefix/failure fixture now uses still-native ReLU so it continues to
exercise real mixed-provider failure and drain behavior.

`result.json` and `raw-bundle.zip` preserve commands, outputs, structured test
results, caches, artifact hashes and selector choices. These are local
correctness facts. Air/Mini, actual model-weight fragments, independent complete
source/static/shared deliveries and qualified performance remain unfinished.
No default performance winner, Native deletion or complete C1 coverage cell is
certified. Existing origin copies and the optional Repeat remain real costs.

The uniform LOC script reports **46,989 Compute+Graph implementation lines in
193 files**, excluding tests/bench/vendor; self-check passed. Product code grew
by 34 lines for this batch. Full-library implementation is 142,174 lines in 544
files; repository code is 910,729 lines in 2,643 files.
