# CPU final F4 named-provider preflight

Task: [F4](../../../docs/implementation/tasks/compute/cpu-final-f4-best-of.md).
This page contains capability/numerical preflight, not a performance WIN or
self-retirement verdict.

The current Windows static SDK was exercised with four named factories using
the same descriptions, bytes and terminal completion: 453 requests, 767
admitted pairs, no numerical mismatch. The 33 single-carrier requests all use
ATen and do not require a timing race. Current versions: oneDNN3.13.4, ggml
ffa4e8b8 plus original IQ1 port, locked LibTorch2.8 CPU. All actual declines and
caller/private resource facts remain in the raw log.

Two early consumer limitations are retained: buffered stdout/stderr split a
metadata line, then the explicit ggml consumer lacked its already exported SDK
implementation target. Ordinary CPU routing already used ggml; the developer
consumer was corrected without changing product/build governance or math.
Those incomplete rows never qualified a WIN or ggml loss.

[Receipt and exact scope](windows-named-provider-preflight-receipt.json),
[initial/current raw logs, admissions/pairs, consumer and drivers](windows-named-provider-preflight.zip).
F4's bounded [immutable P1](../../performance/cpu-final-f4-p1/README.md) is now
complete; full rule delivery/other-platform verdicts remain pending.
F1/F3 proof supplies the existing carrier correctness boundary;
this short preflight does not expand it or qualify arbitrary input values.

## Windows cold-rule delta

Product `a7ccfc6639483aee4a76403a35fb32d1732441ec` binds the 43 complete choices
to the measured software/static recipe/hardware/single logical CPU2/one-thread
conditions. Old 187/17 tables are removed. Unknown conditions keep mature math;
Copy uses qualified ggml prepared storage first and Select qualified ATen byte
selection first, with real capability/resource admission determining alternatives.

[Rule receipt](windows-rule-delta-receipt.json) and
[source/driver/raw archive](windows-rule-delta.zip) preserve working-tree
validation identity and its limits. Pinned property=1 and unrestricted property=0
are both observed. All exact rule requests, thread/shape/epsilon/layout misses,
changed-input replay and actual Graph leaf/whole-region priority pass. Final
ordinary Direct19/Graph20 and Native owner2 regressions pass without skips.
The original test-fixture schema FAIL is retained; only the separate-gate test
input was corrected. The initial include-owner rejection is a transcript
reconstruction, explicitly not original raw output.

No math, hot execution/objects, P1 repetition or public governance changed.
Windows installed static/shared delta, Linux/Mac rule delta and F4 retirement
handoff remain incomplete; this evidence does not close F4 or the CPU goal.
