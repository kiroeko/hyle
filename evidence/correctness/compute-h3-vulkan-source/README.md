# H3 source qualification

Windows is canonical. Product `ae0283e3b890d96945ac5116fd5afd3294a8f1cc`; prior wiring `6422daf35`.
Installed static/shared and WSL build-only are outstanding. This evidence does not close H3.

Windows MSVC 19.50.35728, RTX 4070, driver 617.14: the initial five-entry source batch linked successfully,
then original Compute/Graph and migrated-weight Direct MatMul passed. Both coarse consumers found a retained
internal download owner absent from the cold reference baseline. The fix freezes that owner before sealing;
it does not disable external Tensor protection. Only two affected targets were rebuilt (6.09 s) and both
coarse consumers passed (1.31 s). The changed weight close path's Direct consumer was also rerun and passed
(0.19 s total); the two unchanged original entrances reuse their prior result. Initial FAIL remains recorded.

MacBook Air Mac16,12, 16 GiB, macOS 27.0.1/26A434, AppleClang 21: canonical archive hash appears in the
transport identity. Equal files retain source mtimes. All five source entries pass, including original native
math, Direct migration and both coarse math/resource/lifetime consumers. Build1/test1, owned peak 646 MiB;
no product is copied back from this disposable validation source.

Receipts and logs are retained in this directory. `loc.json` uses the unified immutable implementation count:
Compute+Graph 68,095 lines/301 files, library implementation 163,322 lines/652 files, excluding tests/samples.
