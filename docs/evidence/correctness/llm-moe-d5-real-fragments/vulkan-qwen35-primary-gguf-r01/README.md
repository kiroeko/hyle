# Qwen3.5 primary GGUF Vulkan R01 partial acceptance

Immutable candidate `8af3a29da32243748fdd0bfab407097b3fcb8059` completes the
R01 quantized embedding fragment for
`qwen35_2b:primary_gguf:UD-Q4_K_XL` through ordinary Vulkan Direct and Graph
entry points.

- The profile identity is cold-bound to the fixed official model and unsloth
  artifact revisions.
- The exact Q6_K `[7,2048]` table preserves the model hidden width and uses
  format-valid nonzero bytes. Direct and Graph bind one fixed `ggml-vulkan`
  region, replay two changed device-resident I32 index sets, and match an
  oracle decoded independently from the packed bytes.
- The existing GPU bounds check converts every invalid index to a safe value
  before the upstream `GET_ROWS` region. The provider writes only a private
  temporary result; checked publication returns `OutOfRange` without changing
  the retained public result, and a fresh valid invocation succeeds.
- The route owner suite also executes Q4_K, Q6_K, and Q8_0 through the same
  upstream `ggml_get_rows` path. The provider admits only its internal
  unchecked boundary; ordinary HYLE entry points remain checked.
- The immutable Windows i9-13900K / RTX 4070 / Vulkan SDK 1.4.357.0 / driver
  610.62 / MSVC 19.50 WERROR selector passed docs, closure, configure,
  `compute_provider_ggml_vulkan`, the selected IREE Vulkan one-hop factor,
  `compute_route_vulkan`, and `graph_route_vulkan`.

No HYLE dequantizer, shader, mathematical kernel, host readback, fallback, or
execution-time provider search was added. This receipt claims R01 only.
R02-R05, R13-R15, the complete profile cells, three deliveries and P1 remain
D5 work. The unified immutable-candidate report counts **57,920** Compute +
Graph implementation lines in 228 files, excluding tests and third-party
source, with self-check PASS; see
[compute-graph-loc.json](compute-graph-loc.json). Machine-readable acceptance
details are in [receipt.json](receipt.json).
