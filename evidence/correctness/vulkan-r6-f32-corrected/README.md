# R6 F32 resource correction and immutable candidate

Candidate: `8e45b10423686a0753aaa9d461c4ef3b117412c5`.
This is correctness and source-cost evidence; formal P1 is still pending.

The retained Air `82c14c7e` failure rejected a 34,816-byte shared-memory pipeline
on the 32,768-byte M4 device. Initial review had confused the upstream host recipe's
ID3=16 with the actual F32 shader's hard-coded BK32. The correction follows the
original shader's storage declarations and selects a legal 64 tile on that device;
it does not alter vendor shaders, precision or acceptance thresholds.

| Check | Result |
|---|---|
| Windows 14 Compute + 3 Graph GTests, two public headers | PASS |
| Air 14 Compute + 3 Graph GTests, two public headers | PASS |
| Actual Khronos synchronization validation, both systems | PASS, no VUID/SYNC-HAZARD after correction |
| Air Direct/Graph and both raw-cost full FFN programs, M1/M16, two inputs, independent double oracle | PASS |
| Resident Graph device allocation | Same 541,362,176 / 545,816,576 bytes per arm for M1/M16 |
| Windows full FFN oracle | Reused from preceding checkpoint; M1/M16 shader and 48 KiB device choices are unchanged |

The Air verification controller finished every configure/build/test/oracle command
successfully, then failed copying a missing `compile_commands.json`. The original
failure record remains unchanged. `verification-summary.json` checks all command
exits, oracle counts and artifact hashes; a configure-only invocation exports the
compile inventory, with no rebuild/retest and unchanged executables.

`identity.json` hashes all 28 entries in `raw.zip`; archive SHA-256:
`0f7a210446b487236233c166808c489fad9988ef7910b50e02a3abcb78a3f699`.
The archive retains the failed native run, corrected raw logs, controllers, compiler
commands, cache and reference-shader input identities. No GPU timing verdict is
claimed by these tests. Air's measured Vulkan device API is 1.2.334, loader 1.4.350,
MoltenVK 1.4.1, glslang 16.3.0; use actual telemetry rather than earlier admission
shorthand when binding performance evidence.
