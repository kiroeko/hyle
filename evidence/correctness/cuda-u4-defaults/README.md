# CUDA U4 D1--D3 qualification

Product candidate `8aa1c6796c9979ec2f24ce03f959b7095b70741f`.

- Windows: CCCL10, ggml34, native region8, ordinary Direct10; original Graph51 had46 passes and5 structure failures. Only those5 plus the FFN capture/resource case were corrected and all6 passed.
- WSL: CCCL10, ggml35, native region8, Direct10, Graph51 passed.
- Each OS additionally passed two selected public-header checks.
- Source checkpoint preserves original failed logs and corrected results in `source-validation.zip` / `source-receipt.json`. This is D1--D3 evidence, not a completed U4 performance verdict.
- 28 SDK steps (7 per Windows/WSL static/shared form) passed: install, actual prefix rename, independent offline configure/link/load and ordinary changed-input Direct/complete Graph replay. DLLs neighbor the Windows exe; WSL has no LD_LIBRARY_PATH. See `sdk-receipt.json` and `sdk-validation.zip`.
- Full U2/U3 arithmetic, fixed formats/model branches and lifecycle evidence is reused for unchanged original programs; the small SDK consumer does not substitute for it.
- Uniform Compute + Graph non-test implementation LOC: **60,779**, **255** files.

The bounded D4 provider comparison is separate and still pending.
