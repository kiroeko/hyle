# C1-AA oneDNN normalization candidate comparison

Clean source `e33b6fea4b80d4b82adb6a267dc52ba2ad7a73f9`, W1-WIN, MSVC 19.50,
oneDNN 3.13.2 at `71094df451909e95ceeefc8f64c275cdcbbcdff5`.
Qualification PASS; performance FAIL. Native is the unchanged `2c9338e1` implementation.
Both arms use the same public HYLE Direct or Graph submit/wait contract, input and FP64 oracle.
The Graph candidate fuses normalization, vector multiply and full-shape residual add; Native executes
the same three nodes with intermediate storage. Vendor runtime metadata is charged; cold setup is excluded.

The original instrument used n=15, two placements, both execution orders and both allocation orders.
All qualification controls passed; no n=31 upgrade or additional timing retry followed material losses.
Ratio >1 favors oneDNN. Each range below contains all eight stratum medians; full CI values and
descriptive absolute latencies are in directions.json, with the original statistics in summary.json.

| Workload | Ratio range | Strict direction | Noninferiority |
|---|---|---|---|
| direct-rms-1x2048 | 0.3053–0.3125 | LOSS | FAIL |
| graph-rms-1x2048 | 0.2193–0.2249 | LOSS | FAIL |
| direct-rms-17x2048 | 1.2994–1.3480 | WIN | PASS |
| graph-rms-17x2048 | 0.0064–0.0067 | LOSS | FAIL |
| direct-rms-128x2048 | 1.5725–1.5913 | WIN | PASS |
| graph-rms-128x2048 | 0.0096–0.0103 | LOSS | FAIL |
| direct-rms-17x257 | 0.4525–0.4761 | LOSS | FAIL |
| graph-rms-17x257 | 0.0077–0.0081 | LOSS | FAIL |
| direct-layer-1x2048 | 0.3471–0.3577 | LOSS | FAIL |
| graph-layer-1x2048 | 0.2487–0.2578 | LOSS | FAIL |
| direct-layer-17x2048 | 1.5272–1.5586 | WIN | PASS |
| graph-layer-17x2048 | 0.0084–0.0086 | LOSS | FAIL |
| direct-layer-128x2048 | 1.8775–1.9096 | WIN | PASS |
| graph-layer-128x2048 | 0.0114–0.0123 | LOSS | FAIL |
| direct-layer-17x257 | 0.6195–0.6365 | LOSS | FAIL |
| graph-layer-17x257 | 0.0101–0.0106 | LOSS | FAIL |

## Source diagnosis and limits

The cold descriptor probe queried the actual linked vendor library in 64 cases (RMS/LayerNorm,
four shapes, 2D/3D descriptors and four post-op patterns). Batch full-shape Add selects `simple:any`;
normalization alone and vector Multiply retain `jit:uni`. The pinned JIT broadcast set excludes
full-shape/no_broadcast, while the simple implementation invokes generic reference post-ops per element.
The earlier tentative `ref:any` diagnosis is superseded by the measured identity `simple:any`.
This probe performs no execution or timing, and does not prove a corrected Graph performance win.
The next correction must preserve the optimized prefix and qualify the complete resulting Graph.

This is serial F32 shape replay, not BF16/F16 vendor qualification or real-weight/model acceptance.
No default rule or Native deletion is certified by this record.

## Reproduction and preservation

The archive contains the exact driver, original instrument outputs, all product/control logs,
preflight failures, the finite diagnostic and cold-probe source/commands. The initial CRLF configure
failure and the first workload A/A failure remain recorded; the latter produced zero P1 samples.
One unchanged full retry was declared and completed; its retry budget is now exhausted.
The prior diagnostic executables remain in scratch with hashes in their receipt, not duplicated here.

`verify.json` records a successful original deep verification before product changes:

```text
python scripts/hyle-perf.py verify --contract bench/contracts/onednn-normalization-p1-v1.json --manifest .scratch/perf-runs/onednn-normalization-p1/manifest.json
```

The original manifest retains its absolute paths. The qualified build and source peers are preserved;
this archive does not claim portable Formal certification independent of those artifacts.
`loc.json` counts 46,941 Compute+Graph implementation lines in 193 files, excluding tests and benchmarks.
