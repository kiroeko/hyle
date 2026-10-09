# ggml Vulkan integration patches

Upstream: https://github.com/ggml-org/ggml at
`456172ec733a135778adcd32d00e576a58232e45` (MIT; see LICENSE).

HYLE does not vendor the runtime here. `patches.json` records exact source and
result hashes and the complete, context-matched changes applied only to a
build-local copy of authenticated upstream source. The immutable source cache
and upstream checkout are never modified.

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
The CPU ggml provider remains governed by its separate blocked record.

The native owner now queries and enables synchronization2 when the physical
device and instance API both support Vulkan 1.3. It publishes only features
actually enabled at device creation for IREE's external-device contract;
1.2-only devices retain their existing path without advertising synchronization2.
No shader or numerical behavior is changed by this resource integration.
