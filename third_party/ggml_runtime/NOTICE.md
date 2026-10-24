# ggml common resource integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

The owning CMake authenticates the source and this exact patch manifest before
materializing a build-local copy. Shared cache originals are never modified.
Every edit has before/after hashes and unique context; upgrades require a fresh
resource audit. This directory contains integration edits, not a second ggml
source distribution.

The common edits return NULL for context/pool allocation failures, validate
context alignment/padding and thread counts, preserve native initialization
errors, and roll back partially created worker threads. Apple deallocation
uses the original allocation size. A private pool-creation entry carries the
native failure code to HYLE; it is not an installed public C API.
The official gallocr keeps native liveness planning and returns allocation
failures instead of aborting, including partial-construction cleanup. CPU and
Vulkan adapters share these resource edits; the planner algorithm is unchanged.
Graph metadata sizing advances integer addresses, including its zero-based
size-only cursor, without undefined null-pointer arithmetic; layout is unchanged.

CPU initialization calls the original one-time time initialization without
allocating a temporary context on every execution. The original math tables,
operators, worker execution, barriers and fusion algorithms are unchanged.
The build identity is supplied by the lock instead of probing the parent's Git
repository. Vulkan backend-specific patches remain in its separate manifest.

The conditional `metal-patches.json` owns the M1-A Apple resource seams: retained
native MTLBuffer imports (including heap-backed buffers), checked completion,
and failure-safe encoding. Original Metal kernels, operator semantics and
fusion remain unchanged. C++ allocation failures are caught before crossing
libdispatch callbacks; accepted commands are drained before owners are released.
The bridge symbols are private. The original embedded Metal sources provide
relocatable delivery; library/pipeline construction stays in cold preparation.
The imported array remains owned and allocated by MLX, never freed by ggml.
Native workspace buffer creation also checks metadata allocation and releases
partially initialized Metal/host resources, including a failed C++ backend
buffer wrapper. These changes affect cold allocation and failure paths only.

The CPU paged-address integration maps logical K/V rows through a cold owned
I32 physical page table for private tagged batch graphs. It retains the original
CPU flash vec-dot, tiled GEMM, online softmax, split reduction and workspace;
only effective row counts and four K/V address reads are adapted. Ordinary
untagged CPU attention keeps its original data access. This is not a new math
kernel or a public ggml ABI promise; Vulkan/Metal algorithms are unchanged.
The generic CPU loop resolves one physical row per effective KV token and
applies the independent K/V strides to that row, avoiding duplicate page-table
resolution across the original dot product.

Ordinary CPU causal attention can use a private HYCA tag without a Q-by-KV
half-mask payload. The vector loop limits logical KV rows; tiled attention
constructs the equivalent zero/negative-infinity values in its existing tile
scratch. Native tiled/split selection and QK/softmax/PV arithmetic, including
the vector Half PV path, are retained. The boxed-window tag remains separate.
These CPU-only changes do not alter Vulkan or Metal attention kernels.

Metal grouped matrix metadata emits every token/slot occurrence, including repeated expert IDs. The original matrix kernel reads a compact T*S index map plus per-expert counts/offsets; no repeated activation or selected-weight bank is introduced. Shared F32 activation slots are normalized to their true A1 storage at cold lowering. Floating-point dot and output math remain unchanged.

GGUF NVFP4 keeps its original 64-value/36-byte encoding and four UE4M3 scale
groups. A private decoder is included only by the existing mul_mv, mul_mm and
quantize Metal kinds. Existing MV/MVID, tiled SIMD MM/MMID and selected-row
carriers read packed blocks directly; the NV matrix specialization retains F32
activation, dequantized tile, accumulation and output, including on Metal 4.
Its 12 KiB shared tile and actual pipeline thread limits are checked during
the original NativeProgram cold preparation. Existing non-NV numerical
specializations and compact duplicate-ID mapping are retained. No whole-bank
F32 expansion, new executor or installed C API is introduced.
