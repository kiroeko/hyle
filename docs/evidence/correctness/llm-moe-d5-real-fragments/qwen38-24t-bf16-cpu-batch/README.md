# D5 Qwen3.8-2.4T official BF16 CPU fragment batch

Immutable tests-only candidate `041d9e63a` reuses the Qwen3.8 Direct and Graph
consumers with a profile-derived shape case for the pinned official
`Qwen/Qwen3.8-2.4T-A95B@207bd685` artifact. The test capacity uses two tokens,
seven selected embedding/head rows, and the ten active experts from the
separate R11 receipt. Model inner widths remain H=8192, shared expert N=2048,
GQA 64/4 x 256, Gated DeltaNet K16/V128 x 128, and MTP input 16384.

The same prepared Direct calls and compiled Graph artifacts each run two
changing-input replays through mature CPU Providers. The batch checks:

- embedding Gather, BF16 normalization/residual, dense projection and head;
- the BF16 shared expert gate/up/SiLU/down chain;
- partial interleaved RoPE, query/key norm, 64/4 GQA and output gate;
- MTP concat/projection/norm/cast at the exact 16384-to-8192 width;
- the 16-key/128-value-head Gated DeltaNet state update against an independent
  F32 oracle, including the state output.

The existing R11 receipt separately proves BF16 grouped expert execution on
all four routes. This batch adds CPU fragment evidence; it does not prove the
R10 router/dispatch/combine chain or complete the official profile cell. The
R05 check covers the state-update operation with F32 runtime ports; its model
projection and storage boundaries still need the full profile assembly.

Windows Ninja/MSVC 19.50 with `/WX` built both affected factor binaries. With
the existing local Qwen3.5 fixture supplied, the complete CPU Direct suite
passed 54/54 and the complete CPU Graph suite passed 69/69, with no skips. The
shared inputs tool passed 13/13; module closure and documentation checks
passed. No product implementation changed, so the immutable Compute + Graph
implementation count remains 57,938 lines. Details are in
[receipt.json](receipt.json).
