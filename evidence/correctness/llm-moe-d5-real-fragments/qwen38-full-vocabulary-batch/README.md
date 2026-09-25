# D5 Qwen3.8 full-vocabulary BF16 batch checkpoint

Immutable tests-only candidate `c4e2b0d0f661dd2d0181af68c3d8d30a1ab477fa`
adds a shared consumer for the pinned official Qwen3.8-27B and
Qwen3.8-2.4T-A95B profiles. Their logical vocabulary has 248,320 rows;
hidden widths are 5120 and 8192 respectively. The consumer uses 16,384-row
weight shards to fit the available machines, but traverses every shard and
checks all 248,320 output logits in global row order for two changed BF16
activations. It reuses prepared full- and tail-shard programs. Gather checks
global IDs 0, 16,385, 131,071, and 248,317–319, including the nonuniform
tail. The payload is deterministic sparse BF16 data with one nonzero per
full-width row, not checkpoint bytes; the selected Provider still executes
its ordinary dense MatMul or Gather. Independent row and embedding oracles
check the results. Numeric comparison handles equivalent signed zeros.

All 32 targeted cases passed: CPU, CUDA, Apple, and Vulkan; Direct and Graph;
two official profiles; full head and global Gather. Existing isolated owner
builds and suites produced these results:

| Route | Direct owner | Graph owner |
| --- | ---: | ---: |
| CPU (Windows) | 52 PASS, 6 SKIP | 64 PASS, 6 SKIP |
| CUDA (Windows RTX 4070) | 54 PASS, 6 SKIP | 71 PASS, 6 SKIP |
| Apple (MacBook Air M4) | 84 PASS, 6 SKIP | 57 PASS, 6 SKIP |
| Vulkan (Windows RTX 4070) | 168 PASS, 6 SKIP | 79 PASS, 6 SKIP |

Each group of six skips requires `HYLE_D5_QWEN35_FIXTURE`; those Qwen3.5
cases were not exercised in this batch. All newly added Qwen3.8 cases ran.
The candidate was built with WERROR on CUDA and Vulkan; the reused CPU and
Apple trees had WERROR off. Windows is canonical. The Apple clone was synced
to the exact candidate and used only for verification.

An early full Windows selector run passed its common checks and configure,
then entered a fresh all-route dependency build for the first CPU factor.
That expensive bootstrap was stopped after the bounded run budget. The
selector has no complete verdict for this candidate. This partial checkpoint
does not justify repeating a full selector now: the D5 owner card requires
one consolidated run when the complete profile consumer is stable. Source,
static and shared external delivery and applicable P1 were not run.

The full-vocabulary global out-of-range Gather failure remains to be added.
Quantized artifact formats and the other profiles have not been qualified by
this BF16 test. No D5 profile/route/entry cell changes status; the ledger is
0/176 `RUN_PASS` and 176/176 `VALIDATION_NOT_RUN`. No product math or Provider
route changed. The canonical Compute + Graph implementation count is 58,043
lines in 228 files, excluding tests. See [receipt.json](receipt.json) for the
machine-readable checkpoint.
