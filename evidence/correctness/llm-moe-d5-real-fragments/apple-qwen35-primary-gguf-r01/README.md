# Qwen3.5 primary GGUF Apple R01 partial acceptance

Immutable candidate `94264ba2822a7bbcf8abc3b86db7f5f52181bfb3` completes the
R01 quantized embedding fragment for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary Apple Direct and Graph
entry points.

- The profile identity is cold-bound to the fixed official model and unsloth
  artifact revisions.
- The exact Q6_K `[7,2048]` table preserves the model hidden width and uses
  format-valid nonzero bytes. Direct and Graph both bind `ggml-metal`, replay
  two changed shared-buffer I32 index sets, and match an oracle decoded from
  the packed bytes.
- Checked out-of-range input is validated before the upstream native graph can
  write output. Completion fails, the prior output remains unchanged, and a
  later valid invocation recovers normally.
- The provider owner suite also executes Q4_K, Q6_K, and Q8_0 through the same
  upstream `ggml_get_rows` adapter. Q6_K/Q8_0 retain a `1e-6` comparison;
  fixed-upstream Metal Q4_K is qualified against the independent decode with
  its measured `0.4% + 2e-4` bound, including the near-zero error floor.
- The immutable MacBook Air M4 / AppleClang 21 WERROR selector passed docs,
  closure, configure, `compute_provider_ggml_metal`, `compute_route_apple`,
  and `graph_route_apple`.

This receipt claims R01 only. R02-R05, R13-R15, the complete profile cells,
three deliveries and P1 remain D5 work. Machine-readable details are in
[receipt.json](receipt.json).
