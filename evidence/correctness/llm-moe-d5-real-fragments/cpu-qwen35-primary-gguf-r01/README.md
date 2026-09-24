# Qwen3.5 primary GGUF CPU R01 partial acceptance

Immutable candidate `42f0320f64d88b7e0e9b78daa17f8790a88f53cc` completes the
R01 quantized embedding fragment for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary CPU Direct and Graph
entry points.

- The profile identity is cold-bound to the fixed official model and unsloth
  artifact revisions.
- The exact Q6_K `[7,2048]` table keeps the model hidden width and uses
  format-valid nonzero bytes. Direct and Graph both bind `ggml-cpu`, replay
  two changed I32 index sets, and match an oracle decoded from those packed
  bytes.
- Checked out-of-range input fails before publication and preserves the prior
  output. Graph replay performs no metadata allocation.
- The provider owner suite additionally runs Q4_K, Q6_K, and Q8_0 tables
  through the same upstream `ggml_get_rows` adapter path.
- The exact Windows i9-13900K / RTX 4070 / MSVC 19.50 WERROR selector passed
  docs, closure, configure, `compute_provider_ggml`, `compute_route_cpu`, and
  `graph_route_cpu`.

This receipt claims R01 only. R02-R05, R13-R15, the complete profile cells,
three deliveries and P1 remain D5 work. Machine-readable details are in
[receipt.json](receipt.json).
