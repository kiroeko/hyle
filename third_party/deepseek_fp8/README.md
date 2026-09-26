# Pinned DeepSeek FP8 weight preparation

Original `weight_dequant_kernel` from DeepSeek-V3
`9b4e9788e4a3a731f7567338ed15d3ec549ce03b`, MIT code license.
`provenance.json` records the source/body/archive identities. Only standalone
Triton imports replace model-level Python imports; arithmetic and masks are unchanged.

The two checked-in cubins use Triton3.8.0, CUDA SM89, original block128,
4warps/3stages, BF16 or F32 output. The native ABI has three pointers, M/N I32,
and the compiler's two null scratch slots. No Python/Triton runtime ships.

HYLE's separate `deepseek-cuda` Provider owns descriptor/resource checks and
native module lifetime. Decode is an explicit ordinary operation used once
before binding model-owned cached weights. Model loading and K/V partition
policy are outside this package. D5-F41 entry, resource, delivery and full
consumer qualification are recorded in the owning task, not inferred here.

Rebuild offline with Triton3.8.0 installed: `python rebuild.py --output <dir>`.
Compare both cubin SHA256 and the generated header before accepting an upgrade.
No tuning, runtime JIT, or HYLE decoding implementation is included.
