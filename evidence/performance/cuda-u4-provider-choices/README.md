# CUDA U4 provider choice: engineering defaults, no measured winner

Three fixed overlaps / six shapes: SGLang vs native BF16 weighted RMSNorm and packed SiLU; ggml vs native F32 RMSNorm. Each uses tokens1/32 and independent resident owners, two payloads, terminal Completion, and full output checks.

All six preflight mathematical/ownership checks passed. The original P1 attempt failed telemetry representation; its logs remain. The corrected artifact performed all eight same-route A/A processes, with fixed P-core caller2, parent mask0x14, above-normal priority, both allocation and execution orders. A/A **FAIL**: same implementation against itself exceeds the frozen5% bias and10% individual stability gates.

This is **DIAGNOSTIC / INCONCLUSIVE**, not a provider WIN or LOSS. The sampler was explicitly stopped after observing failed A/A; the common runner had begun product rows despite its global validator failure. No full n15 qualification is claimed and no further sampling/optimization is authorized by this card. The original runner is unchanged. `raw-bundle.zip` preserves both attempts and all interrupted early product rows; `summary.json` binds its hash and exact qualification errors.

The user-authorized budget rule chooses named mature **ENGINEERING_DEFAULT** for these unresolved overlaps. No native branch obtains an advantage certificate from this evidence. Strict capture/no-private/dense-parameter domains remain separately audited in U5; this ordinary contract cannot withdraw them.

No new CUDA mathematics, no all-hardware SOTA, no model throughput claim.
