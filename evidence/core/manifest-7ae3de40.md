# Core v4 `7ae3de40` verification manifest

This manifest binds the Core v4 build-contract projection to one immutable
candidate. It certifies the scoped Core product and delivery contract only; it
does not re-certify downstream algorithms or claim a timing advantage.

## Identity and boundary

| Item | Value |
|---|---|
| candidate commit | `7ae3de400b5d7b598972ebf4913d336b60517483` |
| candidate tree | `b6c04d0e97c93724a9136c34e4ea55b6e4449092` |
| baseline commit | `fe944155e6ada4e68997aaa3fa7d52f27b0769e0` |
| product shape | header-only `hyle::core`; no archive, DSO, C API or Native FFI |
| validation date | 2026-09-01, Asia/Shanghai |

The Windows run used a detached worktree at the exact candidate. WSL and macOS
used disposable native clones reset to the same pushed commit. No WSL or macOS
source modification was copied into the canonical Windows checkout.

## Signed claims

- Core's public headers, API, algorithms, layout and runtime behavior are
  unchanged from the v3 candidate. The v4 product delta is build projection.
- `hyle::core` no longer exports a Container-only SSSE3 permission or chooses
  the parent application's exception mode.
- A Core-only consumer can compile in its ordinary mode or explicitly disable
  exceptions; the latter proves `HYLE_HAS_EXCEPTIONS == 0` at compile time.
- Container still directly owns and exports the same compiler-conditioned
  SSSE3 permission required by its public inline probe.
- Source embedding and relocated static/shared SDKs expose the same Core C++
  headers and create no Core binary artifact.

These are P0 structural claims. No nanosecond benchmark, universal SOTA claim,
cache-hit guarantee or downstream-throughput claim is made.

## Candidate verification

| Profile | Toolchain / gate | Result |
|---|---|---|
| W1-WIN | MSVC 19.50.35728, Release WERROR Core tests | `6/6 PASS` |
| W1-WIN | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| W1-WIN | full component contract | `PASS` |
| W1-WIN | Container source / relocated shared / relocated static | each journey `1/1 PASS` |
| W1-WSL | Ubuntu 24.04, GCC 14.2, Release WERROR Core tests | `6/6 PASS` |
| W1-WSL | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| W1-WSL | full component contract | `PASS` |
| W1-WSL | Container source / relocated shared / relocated static | each journey `1/1 PASS` |
| M1-MAC | Apple M4, AppleClang 21, Release WERROR Core tests | `6/6 PASS` |
| M1-MAC | Core source / relocated shared / relocated static | each journey normal + no-exception, `2/2 PASS` |
| M1-MAC | full component contract | `PASS` |
| M1-MAC | Container source / relocated shared / relocated static | each journey `1/1 PASS` |

The full component contract checks selected source/static/shared journeys,
installed target/header/artifact inventories and target-property ownership. On
macOS every generated child configure inherits the project's existing 14.4
deployment floor from `HylePlatformRequirements.cmake`.

## Usability and anti-benchmark disposition

The public families remain fatal/debug typed context, the failure observer,
compiler contracts, cache layout and three prefetch intents, current-TU
environment facts, version and visibility. Each answers a normal developer or
host-integration task and is available through a task-specific header. There is
no benchmark-named, test-only or timing-selected public seam, and Core adds no
steady-state metadata or runtime dispatch.

The audit compared these tasks with Highway `63f4b16c`, Abseil `b7adf83d`,
Folly `eeac00030`, ggml `d4716378` and llama.cpp `0eadefeb`. It rejected API
count matching and did not admit CHECK matrices, arbitrary formatting,
backtraces, runtime CPU/NUMA, an assume-aligned wrapper, observer RAII or Native
FFI without a named owning consumer. Dangerous assume/restrict/prefetch
preconditions and the exception boundary are documented in the Core guide.

## Rejected first candidate

Candidate 1, `350fc39a25adf953c0f13c7404635b4624d5c6f1` / tree
`dafe004eb0651ac0fccc500c658b7cd577e52f4f`, passed its Core and Container
journeys on all three platforms. Its macOS full-component gate stopped before
the central assertions because a generated SIMD child omitted the required
deployment floor. The candidate was rejected, the tool defect was added as the
single finite fix, and all gates were replayed from candidate 2.

## LOC snapshot

`python scripts/loc-report.py --module core --commit
7ae3de400b5d7b598972ebf4913d336b60517483 --self-check --json` used cloc 2.08
and passed its immutable-tree self-check:

| Metric | Code |
|---|---:|
| full repository | 414,457 |
| self-written including tests | 99,891 |
| self-written excluding tests | 62,143 |
| Core implementation | 763 |

GitHub Actions was not invoked because the project was preserving its monthly
quota; the declared local/remote machine gates were run directly and all
implementation/state commits use `[skip ci]`.
