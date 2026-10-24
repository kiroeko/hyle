# ggml CUDA integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

The owning CMake applies the exact context-matched edits in `patches.json` to
a build-local copy of authenticated upstream source. Shared source caches and
upstream checkouts remain unchanged. The manifest pins both original and
result hashes; this directory does not contain a second runtime distribution.

Edits preserve native CUDA Runtime, Driver and cuBLAS errors at the private
C++ adapter boundary. Resource owners roll back partial construction; cleanup
does not throw. Capture begin/end/encoding failures release the global capture
counter and discard incomplete graphs. Virtual-memory mapping failures release
their allocation handle and partial mapping. Non-owning buffer wrappers never
free external tensor payload. Backend streams, pools, algorithms, fusion and
capture selection remain upstream-owned. The precision and mathematical
changes are documented below.

The private bridge implementation is in `src/compute_provider_ggml_cuda` and
is compiled into the upstream translation unit. It is not an installed API.
Numerical behavior, Direct/Graph execution and failure recovery still require
tests of the operations and lifetimes being used.

The narrow IQ1_XS/XXS/XXXS CUDA programs in `iq1-patches.json` and the three
MMQ template instances originate from Unsloth llama.cpp
`46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f` (MIT, same LICENSE).
Original CUDA tables, conversion, dot products, MMVQ/MMQ math and dispatch
blocks remain unchanged. The newer core's three block-size traits and the
template include path are ABI/build adaptations. `iq1-origins.json` retains
the original block and template SHA-256 identities. They share the existing
physical types/codecs with the original CPU port, not a second ggml runtime.
Only the CUDA route compiles these programs; CPU-only/Metal/Vulkan pins and
math remain unchanged. This source port is not runtime or performance proof.

The F32 MMF tiles in this pin execute TF32 multiplication. The original regular
MUL_MAT, MUL_MAT_ID and corresponding capture-suitability selection are guarded
so a requested native F32 operand contract on F32 weights reaches the original
F32 cuBLAS/sorting program instead. An explicit F32 source request also uses
DEFAULT_MATH within the F32 cuBLAS call: COMPUTE_32F alone does not disable the
handle's TF32 mode. The private handle's previous mode is restored on return
and exception unwind. Default precision and GEMM kernels remain upstream.
These are precision-selection changes, not a new GEMM implementation.

The original CUDA DIAG_MASK_INF kernel subtracted FLT_MAX. This does not
exclude a future score containing positive infinity, and can leave a very
large finite score visible. The exact patch replaces every excluded element
with negative infinity and preserves each permitted value. The original
launch geometry and attention scale, additive-bias, softmax and PV programs
remain unchanged; no mask-content scan or host fallback is added.

The existing float elementwise templates also accept homogeneous BF16 inputs
and outputs for Add/Sub/Mul/Div, their bit-preserving Repeat broadcasts, and
Abs/Neg/Exp/Log/Sqrt/Tanh/Sigmoid. Values widen to F32 on load and round to BF16
on each operation's store. BF16 binary Add/Sub/Mul/Div use round-to-nearest F32
instructions without flush-to-zero, preserving subnormal quantization scales.
Consecutive BF16 Add/Mul nodes retain each intermediate BF16 rounding in
registers during native fusion. Repeat, typed unary arithmetic and launch
geometry remain unchanged, as do F32/F16 dispatch, math and fusion, the
separate SiLU/Mul fusion, and composite Rsqrt.

Caller-owned CUDA capture records the same factored native encoder without
private graph warmup, instantiation or launch. The exact lowered graph uses
the original graph-enabled metadata and capture-compatibility checks. Creating
or refreshing that original metadata entry performs no model mathematics or
native graph instantiation. It preserves the original Mean CUB threshold in
Commands and outer capture alike; environment/architecture-disabled native
graphs remain ineligible rather than taking a different reduction algorithm.
Original lazy handles, streams and caching pools stay owned by that backend.
Only this outer recording permits their allocation in a local Relaxed scope;
pool OOM does not synchronize/flush, and its 256-entry retaining limit counts
both cached and checked-out buffers before allocating. Original Commands and
private native capture keep their previous allocator and selection behavior.
The original preparation Completion holds the process-wide cuBLAS destruction
exclusion through outer End/abort, then observes any failed auxiliary prefix.
No model mathematics executes during recording. A captured graph must be
destroyed before reusing its invocation's pool through a new preparation;
the owning Graph slot already enforces that order.

The original MMVF template additionally accepts a narrow homogeneous BF16
weight/activation/output carrier: one output column, no IDs or fusion,
requested F32 accumulation, dense native layouts and the existing MMVF
shape/device predicate. Activation pairs widen exactly on load; the original
F32 accumulation and warp/shared reduction are reused, followed by an explicit
BF16 round-to-nearest-even store. No whole-tensor input conversion or weight
copy is introduced. Existing float activation/output instantiations retain
their arithmetic and launch selection. This extension does not add a GLU or
ordinary SiLU epilogue, change public descriptors, or establish performance
or model-quality results. Default projection routing is qualified separately.

The original RMSNorm template also carries a checked BF16 widen/RMSNorm/
optional F32-scale/narrow chain. It reads BF16 directly, keeps the original
F32 partial sums, block reduction, epsilon and reciprocal square root, and
stores with explicit BF16 round-to-nearest-even. A scale inside the original
F32 chain is applied before that final rounding. A separate BF16 norm output
followed by a scale retains its intermediate rounding and is not folded into
the weighted variant. Original F32 RMSNorm, multiply and add paths retain
their existing arithmetic and launch choice.

Matching keeps published intermediate values, extra readers and views, and
checks output overlap with every actual input, including OP_NONE leaves.
The BF16 input may share exactly the output's dense shape, strides and byte
range: each row's original block reduction synchronizes its reads before
stores, and each epilogue thread reads and writes only its own columns.
Partial, strided and broadcast overlaps remain ineligible; the F32 scale
must always be disjoint. CPY self references are counted explicitly. The extension
uses the original backend graph, allocator and encoder; it neither changes
public descriptors nor establishes throughput, workspace or quality results.

The composed BF16 square/SUM_ROWS/epsilon/Rsqrt/multiply pattern has a separate
NVIDIA CUDA carrier. It retains the original eight partial sums, launch-dependent
reduction tree and thread-zero result, and every BF16 round-to-nearest-even
boundary. Explicit RN sqrt without FTZ keeps the precise unary object's sqrt
edge separate from the original fast-math division. Runtime epsilon and the
original repeated numerator remain actual inputs in the original dense
variant. Scalar and private-output variants are described below. Matching retains published values and extra readers,
counts CPY self references, and accepts only identical dense reshape origins;
output may alias its dense BF16 input exactly, but cannot overlap epsilon or
numerator. Original per-operation math remains the fallback.

The dense single-vector BF16 dual-projection implementation shares the original
MMVF carrier through its private header and instantiates its new epilogue in the
existing precise unary translation unit. Both F32 dot accumulations round to
BF16 independently; ordinary stable SiLU rounds again before the final multiply
and BF16 store. The floating dot reduction is an implementation variant, not a
bitwise promise against cuBLASLt. Four actual closed BF16 nodes, declared F32
accumulation, device/layout/alignment and disjoint output/read ranges are checked.
The implementation reads both original weights without packing or copying them.
Default MMVF, quantized GLU and other native paths retain their old epilogues.
The matcher also admits the other projection's actual readonly weight transpose
between the four math nodes, retaining its exact shape, strides, origin and sole
reader. It consumes the real native span without bypassing casts, copies, CONT
or mathematical values; the existing private intermediate checks remain intact.

Explicit-frequency RoPE has a private single-kernel NVIDIA CUDA carrier in the
existing precise unary translation unit. It preserves I32-to-F32 RN conversion,
F32 frequency multiplication, separate strict sin/cos, the original independent
RN/FTZ products and add/subtract, and the final typed rounding. Trigonometry is
shared by heads and matching batches without scratch allocation. Exact internal
dense input reuse is safe; partial overlaps and position/frequency overlap are
rejected through the existing native failure boundary. The private marker keeps
these explicit frequencies out of inverse-factor/YaRN kernels and fusers. Other
backends retain the existing decomposed graph. Runtime and performance results
remain separate from this source integration description.

The composed BF16 L2 implementation shares its original reduction with optional
F32 output and scalar numerator variants. F32 output explicitly widens the final
BF16-rounded product; scalar mode reads the real input behind a readonly exact
zero-stride view. The private closure and output/input range checks protect
published BF16 intermediates and other readers. F32 output must be disjoint from
BF16 input; the original BF16 exact-in-place case remains valid. Generic F32
scalar-left division and Rsqrt preparation use the same native broadcast view
without a repeated payload. Other binary fusion/layout domains are unchanged.

Native matrix-vector bias and GLU fusion qualification checks the real layout
consumed by the fused epilogue. Shape equality alone does not establish dense
bias addressing; unsupported layouts retain the original stride-aware math.
The checks preserve inactive-axis freedom and validate the mapped expert axis
for indexed bias. Actual reads cannot depend on eliminated producers and must
obey their real output/read alias contract, including original leaf tensors.
The quantized matvec kernel receives its real output row count for bias-prefetch
and final-store masks; column pitch is not a logical bound. Arithmetic and launch
tuning are unchanged. No model-specific rule, new executor or public operation
is introduced.

MMVF receives the original MatMul accumulation precision explicitly, rather
than reading a fused ADD/GLU node's unrelated parameter storage. Unequal F16
projection domains retain their original independent arithmetic. The existing
private fusion arguments now carry the actual OAI alpha and limit to the same
floating, quantized and multi-token expert epilogues. The private by-value
argument layout changes, so main/MMVF/MMVQ and unary shared-template instances
must be compiled together. No public precision policy, compatibility overload,
new executor or GPU shadow buffer is introduced.

The original MMVF/MMVQ matcher now reports its actual weight, activation,
index, bias and scale storage roots to Hyle's cold allocation planner. A
private allocation-only graph extends those read lifetimes through each
potential fused output; execution and capture keep the original mathematical
graph. Measurement and actual invocation use the same dependency metadata.
Final encoding retains the bound-address, alignment, layout and RESTRICT
checks, including explicit caller aliases. No input OUTPUT flags, payload
copies, public API, extra executor or upstream optimizer are introduced.
The upstream weighted-MoE optimizer keeps its original registration behavior.

Indexed quantized MMVQ fusion uses the token axis for its batch qualification;
the selected expert slots remain channels. Launch qualification uses the actual
device grid.y limits captured during the existing device-info initialization,
for both activation staging and output slots. Backend support rejects only the
MMVQ domain that would otherwise select an invalid launch; larger-token MMQ
qualification, plain floating MMVF, numerical math, and shared CUDA ABI remain
unchanged. Hyle's cold lowering shares checked IDs only for identical original
tensor/count/extent, interns dense metadata by actual tensor and shape, and
retains native grouped producer edges for eligible separate GLU. Public values,
extra readers, duplicate IDs, per-consumer checks and failure publication remain
protected; no payload copies or additional execution mechanism are introduced.

The existing GDN kernel accepts independent BF16 or F32 V and beta carriers
on its private CUDA node. BF16 values widen at their original scalar loads,
using the same conversion helper as the materialized copy. Q, K, decay,
initial state, attention output and snapshot storage remain F32. Decay and
beta have separate element strides; V uses its actual scalar element size.
The original MAC expressions, warp reduction, launch geometry, recurrence
and snapshot order are unchanged, including the original cache-copy fusion.
Cold support checks metadata without requiring addresses; encoding checks
actual owners, device, alignment, capacity and output overlaps. An ineligible
typed cache destination retains the ordinary GDN plus copy. The public GDN
descriptor and original F32 constructor remain unchanged; this source change
alone is not an executed numeric, performance or full-model coverage result.

The existing SSM convolution kernels accept private BF16 sample carriers with
their actual time/channel/batch strides. Sample values widen at the original
scalar loads; weights, register/shared windows, j-order MAC, bias and output
remain F32 in the original fast-math compilation domain. Private stride and
offset arithmetic uses signed64-bit values. The long-tile loader bounds both
F32 and BF16 reads to the actual final tile and zeroes only unused columns.
Shared cold metadata qualification reflects implemented K templates, shapes,
strides and cached device launch limits; actual typed encoding checks original
CUDA owners, device, capacity, alignment and RESTRICT output/read disjointness.
The stable ordinary SiLU exclusion and explicit output rounding are retained.
No public CausalConv1d descriptor, shared CUDA structure or GDN implementation
is changed. This source change alone establishes neither executed numerical
coverage nor a full-model, workspace or throughput improvement.

A private F32-to-BF16 Cast followed by marked ordinary stable BF16 SiLU can
execute as one kernel in the existing strict unary compilation unit. It first
performs the original BF16 round-to-nearest-even, widens that rounded value,
calls the existing stable ordinary SiLU helper, then performs the final BF16
round-to-nearest-even. The original convolution math and F32 materialization
remain unchanged. Cold qualification preserves CPY destination self-reference,
private actual readers, published values and escaping view/OP_NONE aliases.
The actual F32 source read is retained through the fused terminal using the
existing allocation-only dependency mechanism. Encoding checks real CUDA
owners, device, capacity, alignment, launch limits and disjoint input/output
ranges. Existing private main launch caps are named for their shared purpose;
no shared device-info or other by-value structure changes. General Cast,
Unary, GDN, SSM and old marked-SiLU exclusion behavior remain intact.
No public operation, model-specific admission, extra executor or precision
policy is introduced. This source change alone is not an executed numeric,
artifact, full-model or throughput result.

NVFP4 selected-row decoding reuses the original 64-value block arithmetic:
four UE4M3 scale subgroups, the original FP4 codebook and destination cast.
The existing conversion and get_rows kernels share this one device helper.
The existing get_rows k-quant carrier has a compile-time block-size parameter
whose default remains QK_K; only NVFP4 instantiates QK_NVFP4 (64). Thus legal
64/128-value rows do not inherit a 256-value restriction. Selected rows retain
the existing index/stride handling and never materialize a dense weight bank.
No public/native structure ABI, old codec arithmetic, executor or shared source
cache is changed. Product admission and tests are maintained by Hyle; this
source patch by itself does not assert executed numeric or model coverage.

The native MUL_MAT_ID metadata sorter retains every (token, selected slot)
occurrence, including repeated expert IDs within one token. A two-pass warp
prefix builds the existing forward/inverse maps and expert bounds in the
original token/slot order. This removes the old per-token unique-expert/shared
store assumption without allocating a larger shared store or changing public
or native interfaces. A one-channel activation keeps the original broadcast
mapping. Quantization, decoded weights, matrix arithmetic and output rounding
are unchanged. Obsolete 22/10-bit packed-store limits are removed; flattened
I32 map capacity is checked and actual Hyle stride/capacity qualification stays. This applies to the original generic and fixed-slot
metadata paths; it does not flatten all model activations or copy weight banks.

The original host-sorted F32 MUL_MAT_ID fallback retains every selected slot,
including repeated expert IDs in one token. Its existing T*S map and sorted
buffer capacities already cover those occurrences; only the premature break
is removed. Input broadcast mapping, F32 cuBLAS arithmetic, explicit precision,
stream waits and capture qualification stay unchanged. This lets provider
lowering preserve the actual activation/slot shape instead of flattening all
plain-weight slots into separate artificial tokens.

NVIDIA indexed plain-F32 small-token matrices additionally use the original
MMVF GPU-ID carrier when its existing device/shape preference and actual pair
layout/index bounds are satisfied. One predicate is shared by dispatch and
capture eligibility. The original TF32 MMF guard and larger-token host-sorted
fallback remain. That fallback now preserves both parent accumulation and
activation-source precision in each internal expert MatMul instead of resetting
them to DEFAULT. Every repeated slot remains an independent output.

Only the new NVIDIA F32-ID, non-fused specialization explicitly emits RN F32
FMA and reduction adds without FTZ, preserving subnormal operands/partials under
the existing fast-math TU flags. It reuses the original float2 loads, ID/slot
mapping, warp/shared reduction tree and launch geometry. Default-false template
instantiations, existing fusion, BF16/F16 and other backends retain their old
arithmetic. No activation expansion, ID host readback, new executor or public
operation is added in the admitted vector path. This is source/precision
selection, not a measured performance or complete prefill improvement claim.

An indexed NVIDIA BF16-weight/F32-activation/F32-output operation requesting
both F32 accumulation and F32 source1 now retains that actual mixed contract.
The original small-token MMVF reads each BF16 pair by exact bit widening and
reuses the strict RN no-FTZ FMA and original reduction tree. Its metadata
qualification measures pair alignment, strides and index spans in the actual
weight element width. Only this strict mixed contract reuses the original
bounded F32 arithmetic preference, since a BF16 tensor-core alternative would
narrow its activation; default callers retain their original preferences.
Existing default-false arithmetic, the strict F32-ID path and other backends
remain unchanged. The old indexed fused epilogues are declined only for this
new strict mixed domain so their default fast-math dot cannot bypass it.

Other shapes, including odd reduction widths and larger token counts, keep
the original host-sorted expert execution and both parent precision markers.
Strict mixed activations sort as F32, BF16 MMF is not used, and each nonempty
expert calls the original F32 cuBLAS carrier. Its existing conversion expands
only that expert's N*K slice into a pooled temporary; it does not construct a
persistent F32 E*N*K bank or narrow F32 activations. Sorting, synchronization,
pool/workspace ownership and checked publication remain real costs. A previous
widened-F32 bank could admit different fusion/preferences, so no universal
latency or fusion-count improvement is implied by this storage reduction.

The existing marked stable F32 SiLU/product may consume an exact BF16 load
and store the original final BF16 result directly. Its closed private
BF16-to-F32 Cast / stable F32 SiLU / F32 Multiply / BF16 Cast retains both
F32 intermediate rounding boundaries and only the original final BF16 RNE.
The old same-type unary/product and F32-to-BF16 Cast/SiLU contracts remain
distinct. Cold allocation dependencies retain the actual BF16 source and
F32 multiplier through the final store; encode rechecks CUDA device, bounds
and disjointness. Escaping intermediates or unsupported layouts keep their
ordinary operations. No weight projection or reduction is changed.

Two independent BF16 vector MatMul outputs sharing the same BF16 activation
may use the original MMVF dual-accumulator body with two explicit BF16 RNE
stores. Both original outputs remain independently bound and published;
there is no joined weight bank, hidden output, SiLU/product or weight copy.
The precise unary translation unit instantiates this two-output mode. The
original pair loads, F32 accumulations, reduction tree and dynamic block
selection remain; default-false carrier instantiations retain their old math.
This MMVF reduction tree can differ from an ordinary cuBLAS implementation,
so the shared F32-accumulation type alone does not establish bit equality.

The initial native domain is unbatched, even-width, dense BF16 vector inputs
and equal-sized output row banks, bounded by the cached device SM count and
the original MMVF preference and launch limits. This is a hardware-derived
small-row domain, not a measured universal crossover or model whitelist.
Cold dependency reporting retains the three actual read roots through the
second MatMul. Encoding rechecks the same metadata plus actual CUDA owners,
device, ranges, alignment and read/write and write/write disjointness. Only
an exact external weight transpose can occur between the two MatMul nodes;
complete existing FFN fusion keeps priority. Other programs keep their
ordinary operations. No public API, executor or user build option is added.

Cold read-lifetime reporting checks the complete BF16 FFN structure before a
two-output projection prefix, matching encoder priority. It conservatively
holds both projections' real weight and activation roots through the final
Multiply, including before bindings exist. This closes the older complete-FFN
dependency gap as well as its interaction with pairing. Structural liveness
qualification does not grant execution eligibility: encoding still requires
the original full device, layout, pointer and numerical checks.

For a cold-closed single-snapshot GDN split, the existing native fused-cache
carrier may write directly into the real independently bound F32 state CONT
output. The original packed allocation and score CONT copy remain; only the
state payload copy is omitted. Both real endpoints may be published or have
additional readers. Other packed-root/tail observers, intervening math or CPY,
unsupported layouts/devices and every input/output or output/output overlap
retain the original operations. The old rollback-cache CPY route is unchanged.

Cold preparation clears and recomputes an Invocation-local private qualifier
in GDN op-parameter slot one; slot zero remains the original snapshot count,
and the unchanged GDN kernel never reads slot one. Cold closure retains all
six actual input storage roots and packed score source through the terminal
CONT. Encoding checks the local pattern, qualifier and actual CUDA storage
again, without rediscovering graph-wide consumers. The original kernel math,
typed loads, reduction, nonfinite behavior and F32 stores are unchanged. New
state storage must be disjoint from old state. Existing Invocation, capture,
Completion and failed-prefix retirement ownership remain in effect; no new
public API, executor, state version or peak-memory reduction is introduced.
