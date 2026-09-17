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
No Sched product/API or performance change was made. The original failure is
not PASS; remaining R6 selector gates are validated separately.

## Corrected test validation

Immutable test snapshot `756d62206dbfd412ee95d1446a4f6fe64b73a90d` passed the four
selected docs/closure/configure/Sched gates. Actual CTest `LastTest.log` records
all **251 Sched tests passing**. The same executable then passed **32 bounded
repetitions** of the corrected case. No extra sampling or product P1 was run.

`corrected-sched.zip` retains the exact commands/receipts, full native test log,
repeat log, executable identity, changed test source and product identity proof.
All 17 archived entries and all public receipt log hashes were verified. SHA-256:
`a92b0665c086143fe2023013d859a007cf474c458ba2418c913174740530ea5b`.
The raw Windows CTest log, including its local-encoding timestamp, is preserved
byte for byte; only ASCII test-count markers were parsed for the count check.

`corrected-identity.json` records the actual results and seven unchanged shipping
Git trees/files: top-level CMake, cmake/include/src/third_party/scripts/samples.
Only `tests/sched/task_scope_test.cc` changed within tests. This supplies a
corrected test obligation for the unchanged R6/Sched product; it does not turn
the original aggregate BLOCKED selector into a fictitious PASS receipt.
