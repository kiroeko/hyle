# Apple M2-B1 complete-region P1

- Product candidate: `fd03ba10da3d08b7f8d1fa010e92bc82bc250a4e`.
- Qualification harness/validator checkout:
  `a8c9d8aa91df1f4a0cf9006897767f3ced41b907`; commits after the product
  candidate only change tests, performance contracts and execution records.
- Machine: registered `M1-MAC` MacBook Air `Mac16,12`, Apple M4,
  macOS 27.0 build 26A428, AppleClang 21.0.0.21000334 and Metal 32023.883.
- Mature implementation: MLX 0.32.2 at
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, plus the exact admitted
  `third_party/mlx_cast/patches.json` BF16 storage-boundary patch set.
- Contract: `graph-apple-b1-p1-v1`, SHA-256
  `9375fc6260f091d184d00aaf8256098bc33a8faf4277be6f1bb8cc69c173d43e`.

The one measured bundle covers BF16 conversion boundaries, checked
Scatter/Gather state, RMSNorm, separate SiLU gate and residual in a complete
`state[37,2048]` / `updates[17,2048]` Graph region. Candidate and direct MLX
peer use separate Contexts, two reusable invocation slots, the same original
MLX primitive sequence, two checked-index reductions and terminal completion.
Both complete outputs were bitwise equal for two changed resident input sets
before timing.

The qualified n=15 run passed A/A, 5%/10% known-delta and timer-floor controls.
All four execution/allocation strata are `PARITY` with non-inferiority `PASS`:
median candidate/peer ratios are 1.00414, 1.00193, 1.00618 and 0.99899; their
99.26% confidence intervals remain within 0.98866–1.00845. The finite schedule
did not escalate to n=31. `hyle-perf verify` replayed the manifest and every
referenced file on the Air with exact provenance and current child-environment
policy.

`raw.zip` contains the complete generated manifest, receipts and raw logs. Its
SHA-256 is
`4a914aedd921c20974036f285ea248c9c5a9cc7ba70aef991294002af22cb82c`.
The Air profile signs engineering qualification only; this evidence does not
claim Formal independent-hardware certification or another Apple generation.
