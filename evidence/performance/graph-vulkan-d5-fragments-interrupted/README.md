# Vulkan D5 performance: completed first stage, interrupted escalation

Candidate `6d2412d2a28eadd3b6fc0f85bd881e4ffa24a792`; unchanged product SDK
`9273bc71017a1febb69db240b6ea8970b88dd77c`; Direct reference
`c0de38c9b9d6fa32b19fb9791171ce5b8df3cb42`. Windows RTX 4070, original frozen
[contract](contract.json), two complete resident fragments and four execution/allocation strata.

## Actual outcome

- Workload A/A: all eight processes completed, **PASS**.
- Initial n15 stage: all 60 product processes and 30 common-control processes completed.
- Offline replay through the existing canonical parser, estimator and control evaluator:
  all four common controls PASS; Qwen BF16 FFN noninferiority remains UNRESOLVED;
  DeepSeek Flash mixed GGUF MoE has diagnostic noninferiority PASS.
- The runner automatically began its predeclared n31 escalation. The user needed the
  desktop back. The launcher and its benchmark child were stopped; eight n31 product
  logs had been persisted, with no n31 controls. Partial n31 observations are preserved
  separately and are **not pooled into the complete n15 diagnostic summary**.
- No HYLE test process remained after stopping. CUDA was not restarted. No user
  application/service was stopped by HYLE and no system power or privilege setting changed.

**Overall: INTERRUPTED / INCONCLUSIVE, diagnostic evidence only.** The runner did not
reach final artifact/host qualification or emit a completed manifest, so there is no
strict manifest replay or qualified product performance PASS/FAIL. Intermediate
statistical decisions in the diagnostic JSON do not override this boundary. Correctness,
three-delivery evidence, and the qualified CPU/Apple P1 results remain unchanged.

## Preserved evidence

[Index](index.json) records the exact candidate, identities, stage counts and SHA256 of
all archived files. `run/` preserves workload A/A and every existing raw log, including
the interrupted second stage; gzip files decompress to the original bytes. `build/`
preserves the consumer build inputs and A/A marker. The qualified build is not claimed;
the existing consumer build and SDK are retained to avoid unnecessary recompilation.

[First-stage summary](diagnostic/n15-summary.json),
[common controls](diagnostic/n15-common-controls.json), and
[observations](diagnostic/n15-observations.jsonl) are an offline diagnostic replay only:
no new measurements, altered thresholds, sample deletion or new statistics implementation.

## Follow-up boundary

This sampling attempt is stopped, not silently retried. The task returns to its AUDITED
validation plan with product FIX zero. Before any future GPU measurement, reconcile the
remaining finite budget and reuse this complete first stage; do not rerun CPU/Apple P1
or A5. Normal desktop variation is an environment fact, not a requirement for the user
to keep applications closed. This turn starts no additional Vulkan or CUDA measurement.
