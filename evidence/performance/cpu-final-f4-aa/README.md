# CPU final F4 workload-matched A/A

Task: [F4](../../../docs/implementation/tasks/compute/cpu-final-f4-best-of.md).
The fixed eight runs completed: candidate/peer program substitution crossed
with both execution orders and both allocation orders, caller pinned to CPU2.
All 767 admitted/numerically matching pairs were retained. Exactly 334 passed
the short A/A limits; 433 did not. Aggregate gate: **INCONCLUSIVE**.

The gate checks same-program fixed factor medians against 5% and individual
gross instability against 10%, using the shared nanobench parser. It does not
produce an algorithm winner, loss, parity or permission to optimize self math.
No A/B was sampled, no n31 escalation or repeated qualification was used.

[Scope and artifact receipt](receipt.json), [qualified subset](qualified.json),
[inconclusive reasons](inconclusive.json), [all raw runs/ratios and consumer](raw-aa.zip).
Failed rows retain their original status; mature engineering-default decisions
remain possible under the user's accepted bounded policy, but may not be called
statistical best-of. Independent machine/controls and immutable P1 qualification
are still required for any performance rule. Full regions and other platforms
remain in F4 scope.
