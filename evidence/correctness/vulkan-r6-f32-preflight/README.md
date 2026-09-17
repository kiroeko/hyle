# Vulkan R6 F32 implementation preflight — Windows

Baseline `e408536690ea7fc76153f9013bf5aa300feada22`, with the V3 worktree changes
identified by the archived input hashes. This is correctness/P0 evidence, not P1
or route closure. The [performance contract](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)
records the source findings before timing.

- `native-f32-preflight-01`: BK16/shared-memory accounting and safe complete-tile
  vector loads; 16 native tests and two public-header checks, actual Khronos
  synchronization validation, no VUID/SYNC-HAZARD.
- `performance-build-01`: additionally release unused original constants after
  packing. 14 Compute + 3 Graph tests and two public-header checks pass; the new
  public owner/quota test then executes the prepared packed matrix. Six benchmark
  executables compile under MSVC/WERROR at parallelism two.
- `performance-oracle-01`: four Direct/Graph and raw-cost executables, each with
  M1/M16, K4096/H11008/O4096 and two different inputs. Both arms pass the complete
  independent double oracle; all candidate/reference outputs agree exactly here.
  Largest oracle error is about `2.55e-5`, against the frozen elementwise tolerance.
  These runs set `HYLE_BENCH_VERIFY_ONLY=1`; there are no timing samples.

Graph resident VkDeviceMemory is equal for both arms: 541,362,176 bytes (M1),
545,816,576 bytes (M16). GPU cold packing has a higher peak than the reference's
CPU packing/upload, recorded explicitly; this is not a total GPU/driver memory
budget. Two resident weight copies are no longer retained after preparation.

`identity.json` hashes every archive entry. `raw.zip` SHA-256:
`4b8cd9e0bf34ce84b52cf3dc5e06982deac2f702b3e7ce13f4568b470a2315ac`
(19 entries, 79,501 bytes). The oracle controller proves its recorded inputs stayed
unchanged during the run and retains executable identities. Later benchmark-only
formatting, host-policy telemetry names and link-map output do not extend these
receipts to a different artifact; final A/A binds its actual executables separately.

MoltenVK verification of these V3 changes, finite A/A, immutable-candidate P1 and
the remaining delivery/selector review are still pending at this checkpoint.

Subsequent Air verification rejected the initial shared-memory estimate: the
original `DATA_A_F32` shader hard-codes BK32 despite host recipe ID3=16. This
Windows receipt remains a historical pass on its 48 KiB device, and does not
qualify that incorrect resource rule on the Air's 32 KiB device. See the linked
performance contract for the correction and retained failure.
