# Qwen3.5 CPU Direct R02 gate admission audit

Baseline `4dd37f0a0b5dabad3925b85f02993742cb4d9fb4` has mature-provider coverage
for the official BF16 Qwen3.5-2B R02 normalization and three matrix
multiplications, but its ordinary CPU Direct `GatedActivation(SiLU,
separate)` binds `hyle-native`.

The exact 2048/6144 test consumer was built in the existing Windows D5 tree
and run with the checkpoint fixture generated from `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Four of five complete Direct
tests passed; the FFN result matched its independent BF16-boundary oracle
apart from the deliberate mature-provider assertion:

```text
CpuDefaultDirect.D5Qwen35OfficialFfnUsesMaturePreparedCalls
gated-activation: hyle-native
```

The pinned ggml `456172ec733a135778adcd32d00e576a58232e45-hyle` source already exposes
`ggml_swiglu` and `ggml_swiglu_split`. Its CPU dispatch accepts F16/F32, while
the HYLE adapter currently neither qualifies nor lowers `GatedActivation`.
The existing adapter already uses upstream Cast nodes at BF16 boundaries for
other mature operations. This evidence fixes the D5-F5 scope to the adapter
mapping recorded in the D5 card; it does not authorize new math, upstream
changes, route reordering, or other activation kinds.

Machine: Windows 11, Intel64 Family 6 Model 183 Stepping 1. Baseline tree:
`ad9fc3d73eb546f6240896f98860690a7d59218a`.
