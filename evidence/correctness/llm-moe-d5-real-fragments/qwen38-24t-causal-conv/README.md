# Qwen3.8-2.4T BF16 causal convolution (partial D5 receipt)

Windows candidate `55ff8c517` adds the official Qwen3.8-2.4T-A95B
Gated DeltaNet causal Conv1d shape to the ordinary Direct and Graph consumers.
The pinned [official config](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json)
has 16 key heads, 128 value heads, head depth 128 and kernel size 4. The
[model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
concatenates two key-width projections and one value-width projection before
convolution, giving `(2 * 16 + 128) * 128 = 20,480` channels. The fixture keeps
that width and kernel while reducing only the token window to one output token.
It uses nonzero deterministic F32 runtime data and an independent double
oracle, not checkpoint bytes.

| Route | Direct | Graph | Machine |
|---|---:|---:|---|
| CPU | 1/1 pass | 1/1 pass | Windows, MSVC 19.50 |
| CUDA | 1/1 pass | 1/1 pass | Windows, RTX 4070, CUDA 13.3.73, SM89 |
| Apple | 1/1 pass | 1/1 pass | MacBook Air M4, AppleClang 21, MLX 0.32.2 |
| Vulkan | 1/1 pass | 1/1 pass | Windows, RTX 4070 |

Each prepared entry runs two changed inputs and compares every output
channel to the oracle. The CPU, CUDA and Vulkan tests ran in isolated factor
trees; CUDA and Vulkan used WERROR-enabled trees, while the existing CPU and
Apple trees had WERROR disabled. The Mac verification clone ran the exact
pushed Windows candidate and contains no product edit.

This is a focused R05 connection check, not an entire profile acceptance.
The complete owner selector, external source/static/shared consumers and P1
were not run for this candidate. The existing R01 boundary test also uses a
seven-row vocabulary table and output head; it does not prove the official
248,320-row vocabulary width. The D5 ledger remains unchanged at 48/176
`RUN_PASS` cells, with all eight Qwen3.8-2.4T official cells still
`VALIDATION_NOT_RUN`.
