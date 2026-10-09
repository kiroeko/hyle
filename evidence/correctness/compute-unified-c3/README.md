# Unified Compute C3 partial delivery receipts

This is ongoing validation for the [active delivery card](../../../docs/implementation/tasks/build/compute-unified-delivery.md).
It is not a terminal verdict for all platforms/routes, hybrid execution, or asynchronous expert transfers.

[Delivery matrix](delivery-matrix.json) fixes the declared applicable scope: 15 platform/route groups,
45 source/shared/static shapes, each with the existing Direct and Graph consumer. Twelve consumer entries
are currently PASS; the other 78 are UNRUN. A mixed SDK entry runs each selected native route independently
and cannot certify the later joint execution or asynchronous transfer work.

[Conditional source-asset producer counterexample](apple-conditional-producer/receipt.json) records a
C2-01 build-graph defect: an inactive link suppresses copying but still builds its source asset owner.
This is a CMake fixture, not a Metal or math verdict. A finite owner-authorized correction is pending
after the current immutable Windows CUDA batch; existing non-Apple math receipts are retained.
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
Read-only verification copies use a 42,956,800-byte source-only archive from the Windows product candidate
79137d31, SHA256 `6ca08a1ea6728ee340267b5fa10e0f1b9c681e726f2bf407cf2abadd8d0ffefe`.
The archive excludes large historical evidence and is not a reverse source of product changes.
