# Unified Compute C3 partial delivery receipts

This is ongoing validation for the [active delivery card](../../../docs/implementation/tasks/build/compute-unified-delivery.md).
It is not a terminal verdict for all platforms/routes, hybrid execution, or asynchronous expert transfers.

- [Windows CPU receipt](windows-cpu-receipt.json): six actual public math consumers passed,
  covering Direct and Graph in embedded source, relocated shared SDK and relocated static SDK.
- [Per-entry result](windows-cpu-partial.json): exact observed test logs and distinct product identities;
  the later shipping-test-default correction does not change these explicitly tests-OFF math recipes.
- [Archived logs and package metadata](windows-cpu/): CTest logs, real HyleConfig outputs and three bare configuration checks.

The producer used MSVC 19.50.35728.0, full CPU route, locked oneDNN 3.13.4, ggml CPU core/port and
LibTorch 2.8.0+cpu. Source/static/shared share capability semantics, while recipe identities reflect linkage/configuration.
Actual runtime staging placed DLLs beside each EXE. Static HYLE still requires the qualified dynamic Torch runtime.
The shared SDK export check inspected 17 HYLE libraries and 1289 exports; no new P1 was run.

Windows CUDA is in progress. Remaining route/platform/mixed SDK entries have no PASS here.
Read-only verification copies use a 42,956,800-byte source-only archive from the Windows product candidate
79137d31, SHA256 `6ca08a1ea6728ee340267b5fa10e0f1b9c681e726f2bf407cf2abadd8d0ffefe`.
The archive excludes large historical evidence and is not a reverse source of product changes.
