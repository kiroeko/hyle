# Qwen3.5 GGUF region applicability audit

The fixed `unsloth/Qwen3.5-2B-GGUF@f6d5376` primary and fidelity artifacts
each contain 320 tensors and 1,881,825,088 logical parameters. Their complete
GGUF header tensor-name lists are identical. Neither list contains an MTP or
NextN tensor, and the repository tree at the fixed revision contains no MTP
sidecar.

The official `Qwen/Qwen3.5-2B@15852e8` checkpoint still contains its MTP
branch and retains R13. R13 is inapplicable only to the two fixed GGUF
artifacts:

- `UD-Q4_K_XL`
- `UD-Q8_K_XL`

This correction separates model-family capability from artifact contents. It
does not claim execution coverage and does not change the 8 `RUN_PASS` / 168
`VALIDATION_NOT_RUN` cell totals.

The machine-readable audit is in [receipt.json](receipt.json).
