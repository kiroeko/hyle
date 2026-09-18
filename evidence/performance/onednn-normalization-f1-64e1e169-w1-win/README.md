# C1-AA04-F1: qualification failure retained

Frozen source `64e1e1690f2ced52e16cfc7d373ca68e6c37be01`; product correction `ae270af9`.
Correctness and Graph source/static/shared delivery passed separately. Performance
qualification FAIL; engineering performance INCONCLUSIVE. No speed winner is signed.

The workload A/A passed before sampling. All 120 product children (n=15 in eight
strata) and 30 common controls completed. All five common controls pass; source,
artifacts and host identity remain stable. The final workload A/A contains one
gross outlier among 256 ratios: oneDNN versus itself, primary placement,
peer-first execution and allocation, Direct LayerNorm 1x2048, ratio 1.21435.
Every aggregate/factor A/A median remains inside its existing 5% band. The outlier's
own paired-round interval excludes equality, so it is not dismissed as a timer tick.

One bounded four-child same-artifact diagnostic crossed both orders and allocations.
Its affected-row ratios were 1.00731, 0.99646, 0.95003 and 1.01484. The original
deviation did not recur, but its OS, code/data-placement or runtime cause is unknown.
This diagnostic is not a retry, qualification or permission to discard the failure.
No complete run was repeated, no threshold changed and no outlier removed.

The corrected Graph actually uses an optimized norm/vector-Mul region and a Native
residual node for multiple rows. All unqualified observations are retained for
source/cost investigation only; they cannot sign routing, deletion or a SIMD defect.
The earlier qualified 4 wins/12 losses remain a distinct historical result.

Both actual normalization builds use Release /O2. Upstream /Od applies only to
unrelated generated GEMM files. Compiler rules and the complete source/cost bill
are in raw-bundle.zip, together with all logs, scripts and failed qualification.
Deep verify reports DIAGNOSTIC / EXACT / paths_verified at verification time.
Rotating the build later does not preserve its absolute artifact paths.

A separate bounded qualification task must establish an auditable recovery before
another candidate performance run. Default rules and Native removal remain unsigned.
The global C1–C7 goal is not complete. Unified Compute+Graph LOC: 46,955 / 193 files,
excluding tests, benchmarks and vendor code; self-check PASS.
