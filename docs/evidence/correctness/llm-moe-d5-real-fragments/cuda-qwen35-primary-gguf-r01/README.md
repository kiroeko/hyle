# Qwen3.5 primary GGUF CUDA R01 partial acceptance

Immutable candidate `889bb81a26b08c962af7e5adde0076f202c6b38e` completes the
R01 quantized embedding fragment for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary CUDA Direct and Graph
entry points.

- The profile identity is cold-bound to the fixed official model and unsloth
  artifact revisions.
- The exact Q6_K `[7,2048]` table keeps the model hidden width and uses
  format-valid nonzero bytes. Direct and Graph both bind `ggml-cuda`, replay
  two changed device-resident I32 index sets, and match an oracle decoded from
  those packed bytes.
- Checked out-of-range input is validated before the upstream native graph can
  write output. Completion fails and the prior output remains unchanged.
  Graph replay performs no metadata allocation.
- The provider owner suite additionally runs Q4_K, Q6_K, and Q8_0 tables
  through the same upstream `ggml_get_rows` adapter path.
- The immutable Windows i9-13900K / RTX 4070 / CUDA 13.3.73 / driver 610.62 /
  SM89 / MSVC 19.50 WERROR selector passed docs, closure, configure,
  `compute_provider_ggml_cuda`, `compute_route_cuda`,
  `graph_provider_ggml_cuda`, and `graph_route_cuda`.

This receipt claims R01 only. R02-R05, R13-R15, the complete profile cells,
three deliveries and P1 remain D5 work. Machine-readable details are in
[receipt.json](receipt.json).
