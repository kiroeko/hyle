# D5 Qwen full-vocabulary checked Gather boundary

Immutable tests-only candidate `d2250624a825013322d52ace58a2ee6470708c18`
extends the common BF16 vocabulary consumer for official Qwen3.5-2B,
Qwen3.8-27B and Qwen3.8-2.4T-A95B. The logical vocabulary is 248,320 rows.
Consumer mapping rejects global IDs `-1` and `248320` before dispatch.
To test the underlying checked Provider boundary as well, each route submits
local index `2560` to the 2560-row tail shard. The completion reports
out-of-range; no new result replaces the previously published output.

The exact candidate passed 24/24 targeted Gather cases: three profiles ×
four routes × Direct/Graph. Windows CPU, CUDA and Vulkan were built
incrementally in existing isolated trees and ran on the canonical machine;
Apple Direct and Graph were built incrementally from the same commit in the
clean MacBook Air M4 verification clone. The Windows change-impact selector
plan selected only Vulkan factors for this test-header change, so the other
affected route tests were invoked explicitly. The selector itself was not
run for this partial checkpoint.

This closes the identified BF16 global boundary check in the test consumer,
not the complete R01 or model profile qualification. Quantized artifact
vocabulary formats, whole-profile owner suites, external source/static/shared
delivery and applicable P1 remain unrun for this candidate. No product
implementation or Provider math changed; the ledger remains 0/176
`RUN_PASS`, 176/176 `VALIDATION_NOT_RUN`. The canonical Compute + Graph
implementation count is 58,043 lines in 228 files, excluding tests. See
[receipt.json](receipt.json).
