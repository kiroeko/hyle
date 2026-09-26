# Pinned SGLang activation, normalization and model routing

`hash_topk.cuh` retains the complete DeepSeek token-table routing program,
stable sqrt-softplus, warp SUM and sum+1e-20 normalization. HYLE adds token
and selected-expert bounds guards using its existing asynchronous device-status
channel before either indirect payload read. Valid arithmetic and table order
are unchanged. TVM host binding, PDL and fused shared experts are excluded;
model output scaling stays in the consumer. D5-F32 qualification remains open.

The complete `moe_gate.cuh` small/large-token programs now serve DeepSeek's
sqrt-softplus routing. Score arithmetic, biased selection, unbiased weights,
SUM reduction and launch policy are retained. The original score load that
preceded an expert-ID range check is moved inside that check to prevent an
out-of-range read for invalid/extreme model data. TVM/Python host wrappers are
replaced by the existing HYLE CUDA resource/stream/completion binding.

Origin: SGLang `06992c92ad4494fe9f9636e75be3e35040512447`, Apache-2.0.
`provenance.json` records the exact upstream Git blobs and adaptations.
`cmake/HyleDependencyLock.json` binds the official archive and this adapted tree.

The owning adapter is `src/compute_provider_cuda`; its standalone device
compatibility code remains in HYLE implementation sources. Packed
SiLU/GELU/GELU-tanh-times-value uses FP16/BF16/FP32, 16-byte vectors,
PDL/filter/reuse disabled, and one final output conversion.

C1-AB also binds the pinned half RMSNorm warp/CTA/double-load bodies for
64/128/256, multiples of 256 through 8192, and multiples of 512 through 16384.
Input/weight/output use the same FP16/BF16 type, FP32 accumulation and one
final RNE conversion. No cast-before-weight HF variant is used. The upstream
pre-Blackwell launch policy is bound during preparation; the separate
Blackwell-only 32-byte path is excluded from this qualification.

Unary SiLU/GELU/GELU-tanh uses the original vector kernel shell, explicitly
adapted stable-tail math and -Inf limiting value; its restrict qualifiers are
removed for the ordinary exact in-place contract. Packed gate math is unchanged.
No fast-math is enabled. Uninstantiated upstream branches do not expose extra
HYLE operations. TVM/Python runtime is excluded. Added C1 domains' qualification
and performance remain in the owning task, not inferred from source presence.

HYLE maintains this narrow source adapter. Upgrades require review of
arithmetic, indexing, flags, vector eligibility, namespace isolation and
the host ABI together; renew blob/archive/tree hashes, oracle and boundary
tests, native lifetime/capture, three C++ deliveries and applicable complete
consumer performance. Replacing this directory alone is not an upgrade.
