# MLX BF16 conversion integration

This locked integration also carries HYLE's private exact page16 attention
adapter. A distinct inference-only primitive supplies an owned I32 table and
logical KV length to the original Metal SDPA dispatch helpers. The original
one-pass/two-pass/grouped-query vector reads, older steel tile loader and NAX
16-row fragment loads map logical rows to physical pages. A 32-row NAX K tile
loads its two original fragments from their respective pages. Original MMA,
softmax, split-depth and reduction arithmetic remain unchanged. Normal SDPA
primitive/AttnParams layouts receive no page fields; an optional function
constant and gated buffer arguments isolate the page shader identity. There
is no CPU fallback, full-history gather, new attention formula or executor.

Upstream MLX (MIT; see LICENSE):
https://github.com/ml-explore/mlx at
`1f8e74e3f12f31365464a6867c6579f0e9b29d85` (0.32.2).

The exact `float_to_bfloat_bits` function is borrowed without changes from
`mlx/backend/metal/kernels/bf16.h` at upstream
`726dbd926770b7c78cd23480e71f1a21c07a9bc0` (v0.20.0):
https://github.com/ml-explore/mlx/blob/726dbd926770b7c78cd23480e71f1a21c07a9bc0/mlx/backend/metal/kernels/bf16.h
The source file SHA-256 is
`a56b2cc67ad44e8faa1a20418feae0af1e00eeed76c84016de58e0c06f4f236c`.
The current upstream host BF16 type uses the same integer rounding method.

On the qualified Apple GPU, native F32-to-BF16 conversion flushes subnormal
inputs even with safe math. Returning native bfloat after software conversion
also loses subnormal results. The patch adds the original conversion function
and connects only the F32-to-BF16 copy pair to it with an integer storage port.
The original copy loops, array dtype, layout and owner remain unchanged.
Ordinary numeric F32-to-UInt16 casts have a different native library identity.

Compiled fusion stops at this F32-to-BF16 storage boundary so the original copy
primitive executes. This may add a native dispatch and materialized storage;
all other operations remain eligible for upstream fusion. There is no host
payload transfer, new HYLE kernel or new conversion algorithm. The source and
cost change is explicitly part of the owning Apple route's validation scope.

The owning Apple CMake authenticates this manifest and upstream source before
using the existing exact-patch materializer on a build-local copy. Shared cache
originals are never changed. Edited files carry before/after hashes;
upgrades require re-audit. Source, static and shared consumers receive the same
patched runtime, original MIT license and this notice.

The finite HYLE cache adapter additionally supports Q8_0 block32/34-byte
rows, separate QK/VO depths and causal inclusive left windows. The original
vector/two-pass/steel loaders decode inline half scales and signed codes in
F32, without materializing the historical K/V arrays. Encoded queries and
outputs use bounded active-row F32 conversion; two-pass partials stay F32.
Steel uses independent BD/BV tiles, with a finite smaller D256 F32 tile to
respect Metal threadgroup memory limits. Plain coupled/unbounded NAX and GQA
paths remain original. Window loops restrict the effective row range; cache
keys include encoding/paging/window function constants. These adapters retain
the original MLX runtime, queues, MMA/softmax and completion ownership.

Plain F16/BF16 private paged/ragged attention also admits QK512/576 with
V512. It selects the existing compact Steel BQ8/BK8/WM1/WN1 template
before pipeline creation, including Q1, and checks actual pipeline static
threadgroup memory against the device limit. QK, online softmax and PV
accumulate in F32; only the original output store narrows. Low-dimensional,
Q8, ordinary attention and NAX selection remain unchanged.

The same Apple integration carries `metal-shared-ptr.json` for Apple's
Apache-2.0 metal-cpp at `f567ed836e4cbb85788c42115a2682bbe68097ee`:
https://github.com/apple/metal-cpp/tree/f567ed836e4cbb85788c42115a2682bbe68097ee
See `metal-cpp-LICENSE.txt`. Its original `NS::SharedPtr` accepts empty and
moved-from values, but nullable retain/release calls enter C++ member functions
through null pointers. Clang21 UBSan reproduces this without HYLE or MLX.
The exact single-header edits guard those call sites, preserving nonnull
reference-count ordering, layout and API; no sanitizer check is disabled.
The original source and shared cache remain immutable. MLX compilation and
the HYLE installed Foundation/Metal/QuartzCore headers use the same build-local
derived source. This adds nullable-pointer branches, no allocation or state.

The MLX Metal allocator's external active/peak/cache/limit readers also take
its existing mutex, matching the original write-side protection. Internal
allocation/free paths already holding that mutex read the same fields/cache
directly, avoiding recursive locking. This fixes the actual host TSan race
between asynchronous evaluation and destruction of an old array. It adds no
lock object, allocation, policy change or whole-execution serialization.

Ordinary F32 D256 causal attention without an array mask or sinks additionally
uses the existing Steel BQ16/BK8/WM2/WN1 template when its query length is greater
than eight and does not exceed KV length. The original NAX dispatch remains
first. The selected pipeline is checked against the device's actual static
threadgroup-memory limit before dispatch. HYLE forces this bounded F32 route
only on known pre-NAX Apple GPU generations 13-16; unknown/new architectures
retain native selection. Query, probabilities, online softmax and PV remain
F32; no new shader, half conversion, synchronization or public API is added.

Ordinary F32 D192 attention without an array mask or sinks uses the same
existing BQ16/BK8/WM2/WN1 float template for queries longer than eight, including
noncausal attention. Causal attention still requires Q length not exceeding KV.
Its actual pipeline threadgroup usage is checked before dispatch. HYLE uses
the same known-generation guard; the existing D256, NAX and page routes remain
unchanged. This reuses the original JIT template without adding a shader.

The common reshape planner also preserves a nonempty one-stored-element
broadcast as a zero-stride shared-buffer view. Its original empty and
row-contiguous branches remain first. This avoids a full logical copy when
large broadcast axes cannot be collapsed into one native int dimension;
ordinary strided arrays and reshape arithmetic remain unchanged. HYLE uses
this view for the isolated zero-valued preparation carriers of its two
official grouped FP8/FP4 formats. Real bindings retain their required byte
capacity.

The GPU byte view also handles a single stored U8 byte broadcast as complete
four-byte words when reinterpreting it as F32. It checks the physical byte
count, zero strides on every nonsingleton axis, last-axis size and readable
allocation range, then uses the original scalar copy to build one aligned
four-byte word. The output shares that word with zero strides; it does not
reinterpret the original one-byte allocation or accept general strided views.
HYLE's grouped preparation retains the original one-byte code-bank carrier.
Scale slices no longer request an additional contiguous allocation: real
row-contiguous scales share the original bank and offset, while the bounded
broadcast-byte case needs only one physical word. No numeric conversion,
shader, or public binding-capacity change is introduced.

The exact-match manifest also restores the original copyright symbol in four
whole-file edit pairs for `scaled_dot_product_attention.cpp` and `sdpa_vector.h`.
This corrects corrupted match text; replay produces the already authenticated
source hashes, with no native body or shader change.

The shared vector two-pass attention helper now stores its partial output in
F32 for ordinary SDPA and private paged/ragged calls. The host intermediate
uses float32; both the normal and optimized GQA first passes store their
existing float accumulators through float pointers, and the second pass reads
float partials. Query/key/value storage, page/window indexing, block selection,
max/sum combination, launch order and final declared-type output cast remain
unchanged. The one-pass, Steel and NAX paths are unchanged. This removes the
intermediate F16/BF16 rounding/overflow boundary; low-precision output bits may
therefore change, and numerical tolerances still require actual validation.

The partial shape remains [B,Hq,Q,blocks,Dv]. F16/BF16 partial payload grows
from 2 to 4 bytes per element, adding 2*B*Hq*Q*blocks*Dv requested bytes;
F32 partial payload is unchanged. Sums/maxs were already F32. The original
encoder retains this same temporary until completion; allocator rounding,
cache and other private costs are not represented as zero. Host and base
metallib must be rebuilt as a matching pair because these precompiled kernels
share the revised partial-buffer contract. Kernel names, template parameters,
function constants and public API are unchanged; no alternate shader,
precision switch, executor or full-history KV copy is added.

The existing optimized GQA first pass also includes the batch index in its
K/V head address, matching the normal vector pass. The original host already
ensures contiguous batch/head storage for multi-batch, multi-KV-head inputs;
the combined batch/head index uses that existing stride contract. Ordinary
B>1 calls no longer reuse batch0 K/V for later batches. Per-request B1 private
paged/ragged calls retain their original physical addressing.

Ordinary paged E4M3FN K/V use independent fixed positive finite scalar scales.
The existing HylePagedAttention primitive selects its original F32 Steel tiles
for both decode and prefill. Addressed bytes are decoded and scaled in F32 at
the page loader, with no resident floating cache or intermediate BF16 cache.
The loader shares HYLE's private integer IEEE F32 numerical source from
`src/compute_shader_shared/nvfp4_f32.h`; function constants and pipeline keys
include the exact K/V scale bits. The old plain/Q8 entries remain unchanged.

Uniform E4 compact MLA reuses the same physical row for CKV512 and its optional
KPE64 tail; V is a strided alias of the first512 values, with the same scalar
interpretation. Original F16/BF16 query precision is retained in the small
Steel shared-memory tile, while decoded K/V, QK, softmax and PV stay F32. BF16
query fragments widen to F32; there is no BF16 cache/probability narrowing.
The BQ8/BK8 D576 shared allocation is27,648 bytes with16-bit query storage,
instead of36,864 bytes for two F32 tiles, within the actual32KiB Air limit.

Compact ragged MLA uses the original ragged region with two payloads (Q and
one shared cache), plus metadata. Per-request slices retain the physical
CKV/KPE pitch and V is an internal prefix view. The original owned-layout
update, input imports, native result observer and retirement gates are reused.
