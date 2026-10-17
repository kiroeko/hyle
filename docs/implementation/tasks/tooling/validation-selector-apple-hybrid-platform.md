状态：**CLOSED，2026-10-17**。独立维护工具任务；root 已对下述有限补丁及红绿证据完成独立源码审查，无 must。

# Selector Apple hybrid-route platform policy correction

This is an independent maintenance-tool proposal, prepared in `.scratch` for root review. It does not change the canonical registry, product state, HEAD, or existing G09 evidence. `SCOPED` and `AUDITED` below describe the proposed tool scope and its source audit; they do not claim a merged or validated product candidate.

## SCOPED

- Owner: change-impact selector maintenance policy; no new product component.
- Baseline: canonical `6c6222e510853b07b30fb65dd9dcf1054e86bd22`, tree `f9456b27c91ceeca3254880dea413e660eff1891`.
- Trigger: `.scratch/g09-completion-wake-correction/selector-windows-plan.json` selects `factor:graph_route_cpu_apple` on Windows, while the corresponding test target contains no actual correctness tests unless `HYLE_COMPUTE_APPLE_ENABLED AND HYLE_COMPUTE_CPU`. The gate's `ctest --no-tests=error -L ^graph_route_cpu_apple$` therefore cannot succeed as genuine route correctness on that platform.
- Existing contract: `docs/developer/review/change-impact-validation.md` assigns native-route OS applicability to `factor_supported_systems`; unsupported correctness is `SKIPPED` with its reason. Existing `compute_route_apple` and `graph_route_apple` rows restrict correctness to Darwin. The hybrid Apple route is missing its equivalent row.
- Normal consumer: the existing G09 ordinary selector plan, where an actual CPU+Vulkan public API change reaches the CPU+Apple reverse consumer through the central graph. Other native-route ownership, optional-provider selection and platform failure rules remain those of the current selector.
- Scope: exactly one policy row and extension of the existing three-platform table test. No selector algorithm, cache identity, graph, feature probing, build/cache system, product API, or G09 performance changes.
- Performance: P0 source reasoning only. Policy lookup shape remains unchanged; no P1/P2 or product build is applicable to this maintenance-only patch.
- Budget frozen before unit execution: one red execution of the extended existing platform-table test with the old policy; one full selector unittest suite with the corrected policy. If either result is unexpected, stop with the named tool failure and return for review. Do not start a product build, shader generation, benchmark, remote run, or additional cache project.

## AUDITED (proposal; independent root review pending)

| FIX | Evidence | Frozen change | Acceptance |
|---|---|---|---|
| T1 | `scripts/validation-policy.json` has Darwin-only native Apple siblings, but no hybrid Apple row; `scripts/validation-select.py` computes unsupported factors solely from that explicit table. `tests/graph_route_cpu_apple/CMakeLists.txt` guards its correctness registrations on Apple-enabled plus CPU. | Add `"graph_route_cpu_apple": ["Darwin"]` beside `graph_route_apple`. | The existing platform table skips hybrid Apple correctness on Windows/Linux and selects it on Darwin. Off-route header-selection behavior stays as asserted by that existing synthetic fixture. |
| T2 | `test_native_platforms_keep_off_headers_without_false_correctness` already covers Windows/Linux/Darwin but omits this factor. | Add the factor to the fixture list and its expected Darwin-only row. | With T2 alone the platform test fails on Windows and Linux for `SELECTED` versus `SKIPPED`; with T1+T2 the entire selector unittest suite passes. |

The seven review concerns reduce to a metadata applicability error: no third-party change; an actual G09 plan is the consumer; the existing platform table/test is reused; the lookup complexity and execution protocol are unchanged; no C/FFI or public C++ surface changes; no source/static/shared artifact changes; user-facing semantics remain explicit unsupported-route skips rather than an empty correctness pass.

The isolated copy uses canonical Git objects read-only and copies only selector scripts, their existing unit test, central metadata, registry and workflow inputs. Unit fixtures create disposable synthetic repositories and use their existing process spies. CMake is allowed only for the existing graph-metadata `-P` projection; no real product configure/build or native execution is authorized.

## Validation and boundary

- Local unit evidence: the extended platform test with the old policy exits 1 with exactly the Windows/Linux unsupported-platform failures; the corrected policy passes the full existing suite, **60 tests in 122.537 seconds**. Commands, exits, hashes and limits are in `review.json`, `red-command.json`, `green-command.json`, and the corresponding raw logs.
- The final copied policy preserves the baseline adjacent row's LF. The initially tested copy had one extra CR inserted there by the scratch writer; removing only that introduced CR is proved byte-identical after newline normalization and `formatting_only` under the existing selector's JSON proof. This whitespace correction does not consume another unit run. Both final `git diff --check` and reverse patch applicability checks pass.
- Canonical HEAD/tree and clean tracked status are unchanged after execution. The final patch is one policy insertion plus two test insertions/one replaced test-list line; SHA256 `0ba03ef6cd82c7774f3cfd58e0b6ee5745113d95402b388882d7991fa3f313be`.
- Delivery artifact: `selector-platform.patch`, limited to the two files in T1/T2. This patch is not merged, and the copied runtime is not a product candidate.
- The old Windows plan remains a static `READY` plan that was never executed. This tool correction does not turn it into PASS, and does not reuse native receipts across altered policy/tool/context identities.
- After independent approval and a separate merge boundary, root must generate a fresh plan under the actual prepared VS environment and retain all skipped reasons. Tool unit PASS does not satisfy any G09 product correctness, delivery or performance gate.
- Stop condition: after the bounded red/green evidence and proposal handoff. Product or ordinary-selector resource decisions belong to root; this task does not expand cache/build tooling.

## Canonical adoption

Root approved and applied only T1/T2 after the native checkpoint stopped. The canonical maintenance change is recorded by this commit; no selector algorithm, build/cache system, product API or registry transition is part of it. The isolated proposal and validation above remain historical facts. Durable raw evidence is in `evidence/correctness/validation-selector-apple-hybrid-platform/`. The G09 native failure, candidate identities, missing runtime gates and original performance budget are unchanged. A new native checkpoint and actual VS-context plan are required; this tool task is CLOSED, while G09 remains AUDITED.

The exact raw patch is preserved as the selector-platform.patch member of selector-platform.patch.zip, with its original SHA unchanged. The initial staged diff check rejected its required blank context line when stored as a text artifact; lossless ZIP storage resolves only that artifact-format check.
