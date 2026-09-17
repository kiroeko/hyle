# TaskScope first-error test ordering

The original R6 Windows selector stopped at `factor:sched`: 84 selected gates had
passed, one failed, and 26 had not run. The Sched executable ran 251 tests: 250
passed and `TaskScope.AggregatesFirstStatusFailureAndStillDrains` failed because
the aggregate held `later child` instead of `first child`.

`original-selector.zip` retains the original BLOCKED result, plan, controller,
all 85 executed gate receipts and their logs, and the exact original test source.
All receipt/log hashes and all 218 archived entries were checked. SHA-256:
`79059b54958e83e466b11ee29e30a487d16f11cc4bfceed51292ede7aea4ec29`.
`original-identity.json` records the actual snapshot and counts.

Source inspection establishes the missing ordering: typed JoinState publication
occurs in `ScopedTaskDriver::entry`; TaskScope error aggregation occurs later in
the Scheduler finish hook, after Task/Fiber cleanup. Waiting for `first.join()`
does not prove the first child has published its scope completion. The corrected
test waits for that public completion snapshot before admitting the second child,
while retaining the original first-error and full-drain assertions.

This is bounded test maintenance under the
[task record](../../../docs/implementation/tasks/sched/sched-first-error-test-order.md).
No Sched product/API or performance change is authorized. Corrected validation
and the remaining R6 selector gates are pending; the original failure is not PASS.
