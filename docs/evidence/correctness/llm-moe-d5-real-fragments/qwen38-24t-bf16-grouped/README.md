# D5 Qwen3.8-2.4T official BF16 R11 grouped-expert partial receipt

Candidate `d707af1e562e5fd4c183d187238a58bb8eab6641` admits the official
Qwen3.8-2.4T plain BF16 expert-weight domain into the existing mature
`ggml_mul_mat_id` implementation. The public operation retains F32 activation,
output and accumulation plus I32 expert ids. No weight conversion, fallback,
HYLE math, kernel or shader was added.

The shared R11 fixture fixes `Qwen/Qwen3.8-2.4T-A95B@207bd685`, H=8192,
expert intermediate=2048, 512-way routing and top-10 semantics. The capacity
fixture instantiates the ten selected experts with distinct ids 0..9; it does
not collapse the model to one expert or shrink either matrix width. All ten
weights use exact-width deterministic BF16 rows, and two changing-input
replays are checked against an independent accumulation oracle.

- CPU Direct/Graph: PASS on Windows. ggml CPU Provider 39/39, Direct 49/49,
  Graph 64/64.
- CUDA Direct/Graph: PASS on RTX 4070 / SM89 with CUDA 13.3. ggml CUDA Provider
  27/27, Direct 48/48, Graph 65/65.
- Vulkan Direct/Graph: PASS on RTX 4070 with Vulkan SDK 1.4.357.0 and driver
  610.62. ggml Vulkan Provider 21 applicable PASS with one explicit optional
  validation-layer skip, Direct 162/162, Graph 73/73.
- Apple Direct/Graph: PASS on formal Air `Mac16,12`, Apple M4, macOS 27.0.
  ggml Metal Provider 6/6, Direct 78/78, Graph 51/51. The run loaded
  `kernel_mul_mv_id_bf16_f32_4`, proving the BF16 weight executed in the
  upstream Metal implementation.

The first formal Metal run rejected the input before graph creation because
the adapter omitted BF16 from its native dtype map. Commit `d707af1e` adds that
single mapping; the full Provider, Direct and Graph suites then passed. The
existing negative suite continues to reject unqualified format, layout and
rank domains.

The public API change-impact smoke also passed on Windows for both Compute and
Graph: source embedding plus relocated shared and static CPU consumers built
and ran with Ninja/MSVC 19.50. oneDNN was disabled in this bounded smoke
because it was unchanged; the package consumers exercised the ordinary API.
This does not sign D5-A5 delivery for each model route.

This is a **partial R11 receipt**. It completes no 176-cell ledger entry:
R01-R05, R10 and R13-R15, route-level source/static/shared delivery and P1 are
still required. The ledger therefore remains 48 `RUN_PASS` and 128
`VALIDATION_NOT_RUN`. Machine-readable details are in [receipt.json](receipt.json),
and the immutable-candidate Compute + Graph implementation count is in
[compute-graph-loc.json](compute-graph-loc.json).
