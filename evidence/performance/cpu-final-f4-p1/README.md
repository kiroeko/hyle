# CPU final F4: bounded current-provider comparisons

The completed Windows batch used candidate `dc52cbcc5fd99695a23441a32f9ef4bd570eaf2a`,
the immutable CPU product/SDK `560eb1d2332bb69ee5148e899b000fae85d80074`,
oneDNN 3.13.4, ggml ffa4e8b8 plus the original IQ1 program, and LibTorch 2.8 CPU.
The exact sealed image and staged DLLs match the final workload A/A qualification.

- 128 admitted, A/A-qualified pairs; four execution/allocation strata, n=15 only.
- Instrument qualification **PASS**: A/A, known 5%/10% deltas, timer floor,
  SDK/artifact/build validators and before/after source/artifact/host identity pass.
- Pair verdicts: **114 WIN, 12 LOSE, 2 INCONCLUSIVE**. Noninferiority: 116 PASS,
  12 FAIL. The original aggregate performance gate remains **FAIL**; this is a
  candidate-comparison batch containing losers, not a certificate that all candidates win.
- All 453 fixed numerical requests retain at least one mature carrier. 33 have
  a single carrier. Of the remaining requests, 43 have a complete qualified
  dominance choice across all admitted candidates: 41 Native, one ggml, one ATen.
  The other 377 require a justified engineering default; a partial pair win is
  not a full-candidate best-of claim.

`p1-pair-verdicts.json` preserves exact per-stratum estimates and verdicts.
`complete-provider-choices.json` joins those verdicts to the unchanged admission
matrix. A dominance path consists only of qualified wins for the same request.
`admission-family-ledger.json` is mathematical admission evidence, not proof of
equivalent strict resource capabilities or authorization to delete a source family.

`receipt.json` binds the archive SHA. `raw-windows.zip` contains the full unified
manifest, raw products/controls, observations, host, contract lock, consumer and
drivers, together with rejected entry logs. No rejected entry has been rewritten
as PASS. No n=31, additional A/A, self optimization or instrument rewrite followed.

Scope: W1-WIN/i9-13900K, MSVC 19.50.35728, the exact static Release recipe,
one caller/native thread and the exact original dtype/layout/shape/attributes.
All submit-to-terminal costs remain charged. These leaf results do not qualify
whole Graph regions, other hardware, shared builds, Linux or Mac. Unknown domains
use mature engineering defaults and never inherit a statistical WIN label.

Next: F4 cold rule binding and ordinary-entry/resource/delivery validation, then
F5 source retirement. This evidence does not close F4 or the CPU goal.
