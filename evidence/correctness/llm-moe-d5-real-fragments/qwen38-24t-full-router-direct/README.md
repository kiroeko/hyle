# Qwen3.8-2.4T BF16 full MoE Direct batch (partial D5 receipt)

Immutable Windows product/test candidate `719fb96317822eb481acc98ff1f0e7fba7f46b3b`
uses the pinned official `Qwen/Qwen3.8-2.4T-A95B@207bd685` BF16 profile.
The shared fixture preserves H=8192, 512 router experts, top-10, and expert
width 2048. Eleven experts remain resident across replays, including the
near-tie eleventh expert. The payload is format-valid synthetic data; it is
not the full checkpoint.

The test assembles one Compute-only typed Direct recipe. It prepares the
router BF16-to-F32 score, Softmax, TopK, renormalization, ID-driven grouped
gate/up/down projections, weighted combination, independent sigmoid-gated
shared expert, and final addition through ordinary route Direct calls.
Metadata `View` steps use the route's native tensor view. Three changed
activations replay the prepared calls. An independent CPU oracle checks IDs,
per-slot routed results, combined output, shared output, and final sum.
The CPU and CUDA Direct tests explicitly reject frozen HYLE math providers;
the Apple and Vulkan tests use their route's mature prepared path.

| Route | Machine | Targeted full-chain test | Full Direct owner suite |
|---|---|---|---|
| CPU | Windows / MSVC 19.50 | PASS, 1/1 | PASS, 70.78 s |
| CUDA | Windows / RTX 4070 / MSVC 19.50 | PASS, 1/1 | PASS, 67.56 s |
| Vulkan | Windows / RTX 4070 / MSVC 19.50 | PASS, 1/1 | PASS, 89.86 s |
| Apple | MacBook Air M4 / AppleClang 21 / MLX 0.32.2 / ggml Metal 0.24.0 | PASS, 1/1 | PASS, 61.46 s |

Apple Direct initially rejected a lazy MLX `reshape` result at
`Invocation::bind`, which requires successfully evaluated inputs. The test
consumer now evaluates and synchronizes each metadata view before binding it
to the next prepared call. The targeted replay and complete Apple Direct
owner suite passed. The Mac verification checkout was checked against the
exact pushed Windows candidate, then reset to that commit; it contains no
product edits.

The full-chain Graph receipt is
[here](../qwen38-24t-full-router-graph/README.md). These two receipts prove
the official BF16 R10/R11 MoE chain through all eight route/entry paths.
They do **not** sign an entire D5 profile cell: R01-R05/R13-R15, relocated
source/static/shared consumers, applicable P1, and the full change-impact
selector remain subject to their own batch acceptance. The ledger therefore
stays at 48 `RUN_PASS` / 128 `VALIDATION_NOT_RUN`.
