# Core v5 `4b7f28c5` verification manifest

This manifest binds the Core v5 assertion-macro repair to one immutable
candidate. It certifies the scoped C++ contract and its normal-path structural
cost only; it is not a nanosecond benchmark or a universal SOTA claim.

## Identity and boundary

| Item | Value |
|---|---|
| candidate commit | `4b7f28c55e9ca118cf87b80058ce291261033ec5` |
| candidate tree | `074617e1b1dc9d11843493c1e2240da7fd798d9a` |
| baseline commit | `b94b451e3209dc65a6887f0888dc0688b8c86391` |
| product shape | header-only `hyle::core`; no archive, DSO, C API or Native FFI |
| validation date | 2026-09-01, Asia/Shanghai |

Windows used a detached worktree at the exact candidate. WSL used a new native
ext4 clone, and macOS used the disposable native clone reset to the same pushed
commit. All three checkouts were clean and reported the full candidate hash
before validation. No WSL or macOS source change was copied into the canonical
Windows checkout.

## Signed claims

- Context assertions declare no fixed macro-local identifier, so a consumer
  may use `hyle_fatal_fields` or any other legal local name in a field
  expression without capture.
- Enabled and Release-disabled Debug assertions accept and reject the same
  contextual-Boolean condition, message and `FatalField` expression shapes.
  Disabled assertions evaluate none of them.
- Context assertions accept one through eight fields and reject zero or nine
  in Release as well as Debug. Field construction remains failure-only and
  left-to-right.
- The failure observer remains an image-local, synchronous borrowed callback.
  The guide now assigns callback-code lifetime and host quiescence before
  plugin unload to the embedding host.
- Public names, signatures, layouts, HYLE dependency edges and delivery shapes
  are unchanged.

## Candidate verification

| Profile | Toolchain / gate | Result |
|---|---|---|
| W1-WIN | MSVC 19.50.35728, Release WERROR Core tests | `6/6 PASS` |
| W1-WIN | MSVC 19.50.35728, Debug WERROR Core tests | `5 PASS`, Release-only codegen test `SKIP` as declared |
| W1-WIN | Release compile contract | valid dependent use accepted; six invalid condition/message/field/count cases rejected |
| W1-WIN | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| W1-WIN | full component contract | `PASS` |
| W1-WSL | Ubuntu 24.04.4 x86-64, GCC 14.2, Release WERROR Core tests | `6/6 PASS` |
| W1-WSL | GCC 14.2, Debug WERROR Core tests | `5 PASS`, Release-only codegen test `SKIP` as declared |
| W1-WSL | Release compile contract | valid dependent use accepted; all six invalid cases rejected |
| W1-WSL | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| W1-WSL | full component contract | `PASS` |
| W1-WSL | GCC 14.2 ASan / UBSan Debug Core tests | each `5/5 PASS`; optimized codegen gate is not registered under sanitizers |
| M1-MAC | macOS 26.6.2 arm64, AppleClang 21, Release WERROR Core tests | `6/6 PASS` |
| M1-MAC | AppleClang 21, Debug WERROR Core tests | `5 PASS`, Release-only codegen test `SKIP` as declared |
| M1-MAC | Release compile contract | valid dependent use accepted; all six invalid cases rejected |
| M1-MAC | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| M1-MAC | full component contract | `PASS` |

The compile contract runs during every fresh Core configure. Its six negative
translation units cover non-Boolean condition, invalid dependent condition,
invalid dependent message, invalid dependent field, zero fields and nine
fields. Runtime/death coverage additionally observes a signed value from a
caller local named `hyle_fatal_fields` and checks enabled single evaluation and
disabled zero evaluation.

## P0 structural performance

The existing fail-closed codegen audit was extended with a same-signature
identity control. On MSVC x64, GCC x86-64 and AppleClang arm64, the Release
disabled-Debug probe has exactly the identity control's normalized instruction
sequence. The same audit verifies that an always-on Fatal success path has a
conditional failure edge, the diagnostic/termination calls do not enter the
hot caller, force-inline/noinline and assume contracts remain intact, and each
supported prefetch intent retains one target hint.

Therefore the scoped engineering verdict is `P0 STRUCTURAL PASS`: the repair
adds no steady-state instruction, allocation, lock, syscall, dispatch, object
metadata or ABI/layout cost. Timing peer comparison is
`NOT_APPLICABLE`; no sub-nanosecond result was used to shape the API or
implementation.

## Usability and independent red-team disposition

The public assertion families still map directly to normal tasks: always-on
invariants, debug-only expensive checks, optional static messages, bounded
typed context and host fatal observation. The candidate diff contains no
benchmark-named public seam, new option, dependency, binary export or normal
metadata. The independent post-candidate diff review found no fixed
caller-scope macro name, no product path outside the frozen four fixes and no
remaining medium-or-higher finding.

The accepted limitations are deliberate: arbitrary formatting, comparison
CHECK matrices, backtraces/crash management, process-wide cross-DSO observer
coordination, observer RAII/quiescence machinery and Native FFI remain outside
Core until a named consumer admits them. The observer exchange itself is not a
quiescence barrier.

## LOC snapshot

`python scripts/loc-report.py --module core --commit
4b7f28c55e9ca118cf87b80058ce291261033ec5 --self-check --json` used cloc 2.08
and passed its immutable-tree self-check:

| Metric | Code |
|---|---:|
| full repository | 414,925 |
| self-written including tests | 99,927 |
| self-written excluding tests | 62,142 |
| Core implementation | 762 |

GitHub Actions was not invoked because the project is preserving its monthly
quota. The declared local/remote machine gates were run directly, and every
implementation/state commit in this round uses `[skip ci]`.
