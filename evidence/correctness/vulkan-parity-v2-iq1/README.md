# Vulkan original IQ1 XS/XXS/XXXS source packet

Owner scope: [current finite I1–I5](../../../docs/implementation/tasks/compute/vulkan-parity-v2-iq1-bindings.md).
Original source: Unsloth `a809594fd2035979ad2b7ef23398127c1d07e891`, registered
GPU grids and `kernel_mul_mv_iq1_*_f32_impl` in `third_party/ggml_metal_iq1`.
The standard tensor program retains original lane-strided 32-element subgroup
contributions, source scale/delta and final reduction. The independent test
reference uses original CPU grid bytes and original dequantization equations.

Offline source: three entries, four deduplicated original LocalSize64 schedules,
compiler/runtime `6ef606b6fa3e12bd2c12d36a07463069eb71185f`, descriptor Vulkan
1.3 with the original IEEE32 special-value mode. No runtime compiler, new HYLE
shader, quantizer or whole-bank float expansion. Initial SSA/syntax diagnostics
and the test-only private-symbol link failure are retained; no numeric gate changed.

`run_windows_batch.py` runs only new IQ1 cases and the affected native owner,
serially with validation layers, hidden children and adjacent DLLs. Raw candidate
logs/XML/receipt record the actual outcome; an unrun case is not PASS.
The eight catalogue geometries in the domains test are **cold preparation only**.
Full-bank actual GPU qualification, remaining catalogue/regions/state, WSL/Air,
three deliveries, defaults and formal P1 remain pending in the parent round.

## Windows packet result

Product candidate `250758cef` (final test-only revision `11c2b11d`): Direct and
Graph numerical/checked-ID delta each 1/1 PASS, three formats times three K
values (256/1024/8192) times two slot counts, two changed weight/activation/ID
replays, and 54 exact OutOfRange cases per entrance. Independent requests,
retained output lease, cancellation/replay, close and reserve-all OOM each
passed at `250758cef`; the affected native owner 7/7 and eight actual cold
geometries per entrance passed at `6920e7a35`. No validation-layer errors.

The native source program did not change after initial offline generation.
One real adapter defect was fixed: saving the output TensorDesc by value before
appending checked temporaries, avoiding an invalidated vector reference. The
fixture-only I32 upload and existing OutOfRange category were corrected; all
original failures are retained. No precision threshold or source algorithm was
changed, and no P1/WIN is claimed. The compile receipt records its original
generator hash; subsequent explicit UTF-8 input reading changes no grid or IR.

The change-impact selector conservatively selects a wider package matrix for
the vendor/CMake seam. This packet runs its affected native owner and ordinary
Direct/Graph consumers. Other platform/delivery selections are deferred to the
parent V4 and remain unproven, rather than being labeled skipped/PASS.
