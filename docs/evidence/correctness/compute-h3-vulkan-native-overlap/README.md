# H3 native timeline: NOT QUALIFIED for overlap

The source-only P0 probe decorates original Vulkan recording PFNs with timestamps. It executes 512 dispatches
of the existing public sample affine shader and an actual WeightSlots VkBuffer upload of 16,776,960 bytes.
Scalar fixed-point arithmetic and every uploaded value are checked. It is not a new product operator, a speed
comparison, or evidence of mature-provider performance. Raw query brackets are preserved in the logs.

RTX 4070, driver 617.14, original compute queue family 2, 64 valid timestamp bits, period 1 ns: both arithmetic
and payload pass. No dispatch bracket intersects the upload. A bounded diagnostic moves query reset entirely
before the measured submissions; the no-overlap result remains. The final compute bracket ends only 7.68 us
before transfer begins in that diagnostic. GPU copy/compute overlap is **not qualified**.

Source inspection found that the original ggml device already creates a transfer queue; HYLE Native and
with_queue expose only compute. H3-07 scopes the missing native queue/handoff seam. This result does not prove
that the hardware lacks an independent transfer engine and does not establish an operator performance FAIL.
The reset scope was checked against the [Vulkan reference](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResetQueryPool.html).

The probe is test-only and is not an installed C++ API or delivery journey. Source/contract identity and the
fixed candidate must accompany any subsequent qualification; no repeated profile search is authorized.
