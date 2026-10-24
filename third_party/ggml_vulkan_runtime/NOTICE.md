# ggml Vulkan integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

HYLE does not vendor the runtime here. `patches.json` records exact source and
result hashes and the complete, context-matched changes applied only to a
build-local copy of authenticated upstream source. The immutable source cache
and upstream checkout are never modified.

The private page16 attention tag retains physical K/V root descriptors and
carries logical KV length separately. Its scalar shader maps every original
K/V row load through an owned I32 page table. The unused mask-opt binding and
mask-only push fields carry that table and its offset without extending the
128-byte push packet. Paged causal masking uses logical positions. Only the
private full-F32 paged identity admits these fields; normal flash and
coopmat identities keep their original addressing and arithmetic.

These backend edits disable nested shader/compiler parallelism. Common context
failure handling and the pinned build identity are maintained separately in
`third_party/ggml_runtime`; this manifest contains only backend-specific edits.

The SiLU and separate SwiGLU shaders additionally map negative infinity to the
activation's positive-zero limit, as required by HYLE's common semantics.
The gate still multiplies that activation by its other input, preserving
zero-times-infinity/NaN behavior. Their negative finite branch uses exp(x) to
avoid intermediate overflow, as does Sigmoid. Tanh uses the GLSL intrinsic
instead of a subtractive formula that cancels tiny inputs. Softplus uses a
small-tail expansion shared with HYLE LogSoftmax, then a compensated log1p
expression outside that domain, with explicit NaN/Inf handling; Log and Sqrt also
handle the common negative-domain and signed-zero boundaries explicitly.
These are declared numerical adaptations, including changes to finite math.
The original dispatch, fusion and resource planning remain in place.

Plain F32/F32 matrix multiplication uses the original full-FP32 shader with
the original generic subgroup tiles, including on devices that also support
Float16. Tile admission accounts for the actual F32 shared-memory footprint
and workgroup limits. This preserves F32 operands as well as accumulation;
the original half-internal shader could narrow an F32 FFN activation before
the down projection. Other storage and quantized pipeline maps are unchanged.
No new multiplication shader or algorithm is introduced.

HYLE's tagged full-F32 attention mode reuses the original scalar FP32 flash
shader. The private slot tag is distinct from accumulation precision, because
the normal native FP32 accumulator may still narrow Q/K/V or probabilities.
The tagged F16/F32, same-D64/128/256 domain selects scalar tuning and the
original FP32 SPIR-V, accounts for float shared storage, and has a separate
pipeline-cache key. Other ggml calls and device capability flags are unchanged.
No attention shader, kernel or replacement queue/runtime is introduced.

The tagged cache domain additionally admits canonical Q8_0 K/V with F16 Q
and output, independent (Dqk,Dv)=(64,128), and causal inclusive left windows.
The original scalar F32 shader decodes Q8 blocks as F32 at the row/tile load;
this domain excludes the native full-history F16 dequantization branch.
Ragged requests retain the aligned physical root and carry a row offset;
paged requests use the original owned table. Logical length, offset and window
reuse mask-only push fields and remain in the private pipeline identity.
The original Q8 copy shader has a tagged reference roundf mode, including exact
ties and adjacent values. Other Q8 copies keep their original rounding.

Plain F16 cache attention additionally admits (Dqk,Dv)=(512,512) and
(576,512) through that same tagged scalar F32 shader. Head dimensions are
pipeline specialization constants; the shader, page addressing and arithmetic
are unchanged, and admission retains the actual F32 shared-memory check.
This extension does not admit high-dimensional Q8 caches or change ordinary
attention's selection between fused and mathematical paths.

The same tagged scalar domain also admits paired plain BF16 K/V, including
those two high-dimensional pairs. Its existing FaTypeK/FaTypeV specialization
loads the original BF16 words as F32 within each tile; neither a full-cache
F16/F32 conversion nor new SPIR-V is introduced. Q is converted to F32 and
the private flash result remains F32. The HYLE route then uses its existing
IREE nearest-even conversion for the public BF16 result. Original page-table,
window, shared-memory, workgroup and physical storage checks remain in force;
ordinary untagged attention and high-dimensional Q8 admission are unchanged.

The two Argsort shaders retain the upstream bitonic sorting networks, barriers,
dispatch and scratch management. Their comparison uses original F32 storage
keys and source indices to provide stable ties, equal signed zeros, NaN-last in
either order, and valid elements before tail padding. Descending order is
encoded in that comparison; output reversal is removed to preserve tie order.
This is an explicit semantic adaptation, not an unmodified upstream algorithm.

Argsort's host padding exponent uses integer bit width and an unsigned shift.
The original F32 log2 calculation rounds widths such as 2097153 down to the
previous power of two and silently omits the last input. The integer correction
preserves the original sorting kernels, passes and scratch ownership.

The I32 GetRows pipeline's X dispatch divisor is corrected from 1024 to 512
to match the upstream shader's fixed 512-lane workgroup. The original uint
loads/stores and shader are unchanged. This prevents incomplete rows when a
bit-preserving Gather spans more than one workgroup, including BF16 embeddings.

The large Softmax second and third shaders accumulate their shared maximum at
every reduction step. The original loops reused the initial lane-local maximum
and could discard a preceding reduction's peak (for example block 64 at width
32769). Only the two max expressions change; the upstream three-pass algorithm,
dispatch, numerical storage and scratch ownership remain in place.

The native embedding edits isolate Vulkan instance/dispatcher/registry state,
publish fully initialized devices, roll back partial buffer/backend creation,
account device memory through optional HYLE reservations, and permit an owned
native-buffer/queue bridge. The host retains its terminate handler and supplies
loader function pointers. The bridge implementation remains private HYLE code.
Device creation also enables `VK_KHR_portability_subset` when the physical
device advertises it, as required by Vulkan VUID 04451. This uses the existing
extension enumeration; devices without that extension do not request it.
These changes do not certify asynchronous partial-submission failure handling
or the ordinary Direct/Graph adapter; those have separate remaining acceptance.

The native owner now queries and enables synchronization2 when the physical
device and instance API both support Vulkan 1.3. It publishes only features
actually enabled at device creation for IREE's external-device contract;
1.2-only devices retain their existing path without advertising synchronization2.
No shader or numerical behavior is changed by this resource integration.

The private full-F32 resident SDPA Work may retain the original non-one-time
compute commands in its own backend command/descriptor pools after actual
retirement. Recording is limited to the fixed attention/mask/view/cast graph
without staging or semaphore/event dependencies. Rebinding a byte origin,
buffer/capacity or strong input owner invalidates the record before releasing
old imports. Other graphs retain original managed execution; this is not a
public capture promise or a change to any shader or mathematical domain.

The CacheEncode-only Q8 exact-copy branch canonicalizes every block whose
rounded F16 scale is zero to 34 zero bytes before computing a reciprocal.
Nonzero-scale blocks keep the original two rounded F32 quotients and ties-away
codes. Ordinary Copy and SetRows are unchanged. This explicitly changes only
zero-scale encoded payload bytes, not their decoded zero value.

The explicit-frequency RoPE's original computed F32 phase is retained. Sin and
Cos reduce finite arguments with absolute value at most 2^20 by a three-part
2pi constant and explicit FMAs before using the existing GLSL intrinsic. This
prevents large-angle intrinsic error at normal inference positions without
adding storage or dispatches. Sin preserves signed zero. Nonfinite arguments
and values outside this finite domain retain the native intrinsic; this patch
does not qualify their accuracy. The original independent F32 phase oracle,
attention tolerances, quantized storage and state feedback are unchanged.

The indexed matrix and vector drivers retain actual direct activation byte
spans and token/slot strides, including shared activation views with padded
token pitches. Direct noncontiguous matrix inputs use the existing unaligned
pipeline; converted/quantized staging keeps its compact layout. This changes
host addressing only, without a new shader, weight decode or arithmetic path.

Indexed matrix dispatch covers the full slot/token occurrence upper bound,
including repeated expert IDs. Cold support checks the widened product and
actual created tile/device grid limits before native graph allocation. The
original push constants, scalar arithmetic and output addressing are unchanged.

Ordinary page16 E4M3FN cache reads use a separate private scalar-F32 variant
of the same flash shader and original queue/graph/lifetime path. Raw I8 K/V
ports are admitted only with the explicit E4 storage tag. Fixed independent
K/V scales form part of the pipeline key and specialization constants; each
addressed E4 value is widened and scaled in F32 before the original flash
math. No complete-cache decode, intermediate BF16 cache or new executor is
introduced. The numerical include is generated from HYLE's shared integer
IEEE helpers and E4 storage codec, preserving subnormal scale multiplication.
The original F16/BF16/Q8 shader modules and compact MLA storage remain separate.

The same private E4 scalar shader additionally admits compact MLA rows with
CKV512 or CKV512+KPE64 and a V512 prefix of the same physical row. The host
requires the same view root, byte origin, row pitch and fixed scale for K/V;
it does not admit arbitrary padded E4 inputs. Only this tagged support check
changes: the existing shader reads the original pitch, and no V bank,
complete-cache decode, new SPIR-V or executor is introduced. Ragged compact
rows use the same private cache identity with a logical row offset instead of
a page table. Plain and ordinary separate-K/V E4 attention keep their
previous domain and behavior.
