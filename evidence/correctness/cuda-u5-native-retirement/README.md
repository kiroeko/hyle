# CUDA U5 borrowed activation qualification

Product `657f291f`; test/SDK consumer `b1f3fdc1`. Six duplicated local SiLU/GELU/GELU-tanh gate/Unary formulas physically removed in favor of the same locked SGLang inline math. Scalar/separate/tail/stride indexing, F32 intermediates, final RNE, alignment, aliases, empty tensors and native Completion remain. Other required unique CUDA math/resources remain explicitly listed in the U5 card; this is not deletion of all native operators or a measured native win.

- Windows and WSL: original activation/resource9, ordinary Direct2, Graph3 tests passed on each OS. Shapes, tolerance, nonfinite gates and full output oracle unchanged.
- Each independently relocated/offline static/shared SDK on both OSes linked and loaded unchanged original test sources: activation/resource9 passed, plus ordinary Direct cold-policy/complete ggml Graph changing-payload replay. Windows system-only PATH, DLL neighbors; WSL no LD_LIBRARY_PATH.
- 42 declared executed steps latest PASS; one original Windows shared test missing cuBLAS link-dependency failure retained. Only the test DEPS corrected. Two duplicate native Graph requests explicitly skipped because the identical cases were already in completed source graph3; they are not separate PASS rows.
- `receipt.json` binds all results and `validation.zip`; original logs/XML/job inputs/failed attempt remain. No P1 resampling.
- `u6-native-audit.json`: original full native archive and per-input hash check plus exact Linux651/649 union XML. `u6-delivery-audit.json`: eight U23 delivery archives and1147 internal hashes verified. `u6-product-impact-audit.json`: every later product change classified, original vendor math/model programs unchanged.
- Uniform non-test Compute + Graph LOC **60,774 /255 files** in `loc.json`.

Scope and complete-vs-partial replacement decision: [U5 card](../../../docs/implementation/tasks/compute/cuda-final-u5-native-retirement.md).
