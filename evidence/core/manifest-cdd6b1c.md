# Core v3 `cdd6b1c6` verification manifest

This manifest binds the Core v3 engineering closure to one immutable product
candidate. It certifies the scoped Core contract only; it does not certify the
algorithms or aggregate performance of downstream modules.

## Identity and boundary

| Item | Value |
|---|---|
| candidate commit | `cdd6b1c612c8e318220972fa0936f23e28045447` |
| candidate tree | `3855da3095dcd52417aeb46b8822346755b28abd` |
| baseline commit | `0fc5fbc0cc8a86ef043788400e6305e5e6afa0e4` |
| product shape | header-only `hyle::core`; no archive, DSO, C API or Native FFI |
| implementation boundary | `include/hyle/core/**`, `include/hyle/version.h`, `include/hyle/visibility.h`, `src/core/**` |
| validation date | 2026-09-01, Asia/Shanghai |

The Windows checkout was a detached worktree at the candidate. WSL used a
candidate-only native Git clone after a Windows-created worktree exposed an
invalid cross-host `.git` pointer to the unrelated LOC tooling test. macOS used
a detached worktree fetched from the pushed candidate. No remote/WSL/macOS file
was copied back as a product change.

## Signed claims

- Bounded typed fatal context, the image-local failure observer and default
  formatter execute only after a failed invariant. The success path remains one
  cold branch and performs no allocation, lock, system call, formatting or
  observer atomic load.
- Disabled debug assertions do not evaluate their condition, message or fields;
  debug/sanitizer/analyzer builds activate them. Always-on fatal assertions keep
  their release invariant semantics.
- Compiler annotations and current-translation-unit environment facts are
  compile-site contracts with no runtime registry or metadata. Runtime CPU,
  topology, NUMA and ISA dispatch remain outside Core.
- Source embedding and relocated static/shared SDKs expose the same C++ headers
  through `hyle::core`. A shared package intentionally creates no Core DSO.
- Core has no product third-party dependency and no public C/managed bridge.
  GoogleTest is test-only; mature libraries are comparison inputs, not linked
  Core mechanisms.

These are P0 structural claims. No nanosecond peer race, universal SOTA claim,
cache-hit guarantee or downstream throughput claim is made.

## Platform, correctness and code-shape gates

| Profile | Toolchain / journey | Result |
|---|---|---|
| W1-WIN | MSVC 19.50.35728 Release, WERROR, full Core build scope | `18/18 PASS` |
| W1-WIN | MSVC 19.50.35728 Debug, WERROR, Core label | `6/6 PASS`; optimized codegen test skipped by its declared Debug precondition |
| W1-WIN | clang-cl 20.1.8 Release, WERROR, full Core build scope | `18/18 PASS` |
| W1-WIN | Mem/Container/Sched direct consumer closure, MSVC WERROR | `38/38 PASS` |
| W1-WSL | Ubuntu 24.04, GCC 14.2 Release, WERROR, native candidate clone | `18/18 PASS` |
| W1-WSL | GCC 14.2 Debug ASan and UBSan, Core label | `5/5 PASS` for each; no sanitizer diagnostic |
| W1-WSL | GCC 14.2 TSan, `setarch x86_64 -R` | Core `14/14 PASS` plus release-assert `1/1 PASS`; no race diagnostic |
| W1-WSL | Mem/Container/Sched normal consumer closure | `38/38 PASS` |
| M1-MAC | Apple M4, AppleClang 21 arm64 Release, WERROR | `18/18 PASS` |
| M1-MAC | Mem/Container/Sched direct consumer closure, WERROR | `38/38 PASS` |

The first WSL TSan launch without `setarch -R` failed before `main` with the
known runtime message `unexpected memory mapping`; the documented ASLR-disabled
launch is the signed run. The WSL direct-consumer WERROR build compiled the
Mem/Container/Sched product targets, then stopped in an existing Container
`SmallVector` test-only GCC `-Warray-bounds` diagnostic. Replaying that consumer
suite with its normal warning policy passed `38/38`; Core's own GCC WERROR gate
remained `18/18`. Neither event was rewritten as a Core product failure or PASS.

## Delivery and dependency gates

`HyleCiSelectedComponentSmoke.cmake` passed on W1-WIN, W1-WSL and M1-MAC for all
three journeys:

| Journey | Result | Artifact contract |
|---|---|---|
| parent-project source embedding | `1/1 PASS` on each platform | links only `hyle::core` |
| relocated installed shared SDK | `1/1 PASS` on each platform | installed headers/imported INTERFACE target; no Core DSO |
| relocated installed static SDK | `1/1 PASS` on each platform | installed headers/imported INTERFACE target; no Core archive |

`cmake -P cmake/HyleDependencyVersions.cmake` passed for the candidate lock.
Core-only builds fetch GoogleTest only because tests were enabled; tests-off
source and installed consumers have no third-party Core product route.

## Rejected candidate and red-team disposition

Candidate 1, `cbc2951ca34ffa961d9db8660b45f17eac2a0104` / tree
`21697748299939378a258a5c5aee02ef02582c15`, was rejected rather than patched in
place. clang-cl found an invalid array-reference `lifetimebound` placement and a
test that confused compiler frontend identity with Windows CRT identity.
Candidate 2 removed the redundant array annotation and selected `tmpfile_s` by
the OS/ABI fact. No public API, normal-path algorithm or cost contract changed.

The final red team found no remaining Core MEDIUM-or-higher issue. Accepted
limits are: borrowed fatal fields live only through the synchronous callback;
the observer is image-local and not signal-safe; unsupported compiler
annotations may degrade to no-op; compiler promises retain their documented
preconditions; and owning consumer workloads must separately prove whether a
hint improves throughput.

GitHub Actions was deliberately not invoked because the monthly quota was
nearly exhausted; every implementation/state commit used `[skip ci]` and the
declared local/remote machine gates above were run directly.
