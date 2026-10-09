# Unified Compute C3 partial delivery receipts

This is ongoing validation for the [active delivery card](../../../docs/implementation/tasks/build/compute-unified-delivery.md).
It is not a terminal verdict for all platforms/routes, hybrid execution, or asynchronous expert transfers.

[Delivery matrix](delivery-matrix.json) fixes the declared applicable scope: 15 platform/route groups,
45 source/shared/static shapes, each with the existing Direct and Graph consumer. Thirty consumer entries
are currently PASS and the other 60 are UNRUN; earlier Vulkan failures remain archived. A mixed SDK entry runs each selected native route independently
and cannot certify the later joint execution or asynchronous transfer work.

[Conditional source-asset producer counterexample](apple-conditional-producer/receipt.json) records a
C2-01 build-graph defect: an inactive link suppresses copying but still builds its source asset owner.
This is a CMake fixture, not a Metal or math verdict. The [applied product correction](apple-conditional-producer/product-fix-receipt.json)
passed the metadata regression and actual conditional cross-directory asset builds on Windows Ninja and
Air Unix Makefiles. Existing non-Apple math receipts retain their original product identities; actual
Apple native delivery remains unrun.
The [disposable correction experiment](apple-conditional-producer/design-experiment/receipt.json) passed
inactive/active source asset cases on Windows Ninja and Air Unix Makefiles, with the real library owner
and generated asset in a separate directory. It validates the finite dependency approach, not an applied
product fix or native Metal execution.

- [Windows CPU receipt](windows-cpu-receipt.json): six actual public math consumers passed,
  covering Direct and Graph in embedded source, relocated shared SDK and relocated static SDK.
- [Per-entry result](windows-cpu-partial.json): exact observed test logs and distinct product identities;
  the later shipping-test-default correction does not change these explicitly tests-OFF math recipes.
- [Archived logs and package metadata](windows-cpu/): CTest logs, real HyleConfig outputs and three bare configuration checks.
- [Actual SDK requirement checks](windows-sdk-requirements.json): shared/static installed SDKs accept CPU,
  reject a requested absent CUDA route and reject a retired Provider option. Exported target route/recipe
  properties agree with package variables; these six configure verdicts required no rebuild or P1.
- [Actual SDK offline and missing-dependency checks](windows-sdk-offline.json): eight configure cases passed
  across the same relocated shared/static CPU SDKs. A populated offline cache is accepted; an empty offline
  cache, invalid explicit Torch root and missing original ggml runtime are rejected with their specific
  diagnostics. [Logs](windows-sdk-offline/) preserve the observed outputs; no native rebuild or P1 was run.
- [Windows CUDA receipt](windows-cuda-partial.json): the embedded-source and relocated shared/static SDK
  Direct and Graph FFN consumers passed on the actual RTX 4070. [Logs and package metadata](windows-cuda/)
  preserve their six CTest verdicts. Required DLLs are adjacent to the EXEs; the CUDA-only producer excludes
  the HYLE CPU backend and oneDNN. The batch completed with exit 0. Shared export hygiene passed for
  18 HYLE libraries and 985 exports.

The producer used MSVC 19.50.35728.0, full CPU route, locked oneDNN 3.13.4, ggml CPU core/port and
LibTorch 2.8.0+cpu. Source/static/shared share capability semantics, while recipe identities reflect linkage/configuration.
Actual runtime staging placed DLLs beside each EXE. Static HYLE still requires the qualified dynamic Torch runtime.
The required Torch runtime remains a locked machine SDK acquisition, not part of the HYLE archive; offline
consumption needs its populated cache or a qualified local root/archive. Route requirements do not remove it.
The shared SDK export check inspected 17 HYLE libraries and 1289 exports; no new P1 was run.

Windows CPU and CUDA three-delivery consumer batches are complete. Remaining route/platform/mixed SDK entries have no PASS here.
[Windows Vulkan source failure receipt](windows-vulkan-failure/receipt.json) preserves both actual failures:
the sample's old native-dispatch count and the 16-row F32 FFN numerical mismatch. An independent host
rounding bill matches the observed GPU result after narrowing the down-projection activation to F16;
disabling cooperative-matrix variants does not resolve it. The [bounded completion receipt](windows-vulkan-partial.json)
now signs source Direct/Graph PASS after binding the original full-F32 shaders and preserving explicitly
requested F32 operands before cooperative-path conversion. The original numerical gate is retained.
The relocated shared/static SDK Direct/Graph FFN consumers also passed; the resumed SDK loop exited 0.
[Additional F32 range/transposed/replay receipt](windows-vulkan-range.json) covers all three deliveries after
only the needed small consumers were rebuilt. SDKs were not rebuilt for that test addition. Existing native
resource and other-dtype regressions passed in [22 selected cases](windows-vulkan-regressions.json);
[eight matrix/ID and quantized regression cases](windows-vulkan-quant-regressions.json) also passed, including
the new plain-F32 grouped matrix path with non-F16-exact and out-of-F16-range operands. These subsets had
no fixture skips. Other platforms remain unrun. No self multiplication is added.
Read-only verification copies use a 42,956,800-byte source-only archive from the Windows product candidate
79137d31, SHA256 `6ca08a1ea6728ee340267b5fa10e0f1b9c681e726f2bf407cf2abadd8d0ffefe`.
The archive excludes large historical evidence and is not a reverse source of product changes.
The remaining WSL/Air entries now use fresh `source-2614676e` directories from the corrected product:
42,967,040-byte source archive, SHA256 `94cd4aff04573073ffc848ecf71d67933ed705d03160ba93f8ce1b3da64b1bcc`.
Both received archives were checked against that hash; no native math PASS follows from transport.
The [Air Apple partial receipt](air-apple-partial.json) signs actual source Direct/Graph consumers on
Mac16,12 Apple M4. Shared/static also passed and the serial batch exited 0. Its newer complete validation
source snapshot is c8874688, archive SHA256 `3770e2f2147120f527ff96a45af4a096ed846c791624e720582cb2eb4a88d4f5`;
product c96c7c6e and test/document-only follow-up identities remain distinct.
The [Air Vulkan receipt](air-vulkan-partial.json) covers six actual source/relocated shared/static consumers
after route-owned default loader discovery, without DYLD overrides. Its original loader failure remains
archived. Product a1c22ac3 and validation snapshot 55652ab2 are bound to the complete archive hash; only the
changed owner TU and links were rebuilt for the source correction.
