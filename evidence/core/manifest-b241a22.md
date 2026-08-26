# Core v2 `b241a22` verification manifest

This manifest binds the Core v2 closure evidence to one frozen implementation
candidate. It certifies the scoped Core contract; it does not certify the
feature completeness or performance of downstream modules.

## Identity and scope

| Item | Value |
|---|---|
| candidate commit | `b241a228d3311f0faa65cde68123de7da3688c2f` |
| candidate tree | `e73472e2cefedd70e41eca6243871b2343db7d74` |
| Core product shape | header-only `hyle::core`; no archive, DSO, runtime query, C API, or Native FFI |
| implementation boundary | `include/hyle/core/**`, `include/hyle/version.h`, `src/core/**` |
| validation date | 2026-08-26, Asia/Shanghai |

Windows used the canonical workspace. Its only non-candidate working-tree edit
was the user's unrelated `bench/error/status_bench.cc`, with benchmarks disabled;
it was neither staged nor an input to a Core target. Linux sanitizer evidence was
built from `git archive` of the full candidate hash. Linux and macOS ordinary
evidence used one-way copies of the same candidate inputs and recorded the
matching files/toolchains in their raw logs.

## Signed claims

- Core's public capability set is limited to compiler/invariant primitives,
  compile-target cache layout/prefetch primitives, and header product identity.
- Normal paths are constant-time compile-time/inline operations with no lock,
  allocation, registry, metadata, or Core runtime artifact.
- Release codegen contains the intended read/one-shot/write prefetch intents;
  `HYLE_ASSUME` does not add a validation branch; fatal success paths branch to
  a cold, non-exported, non-preemptible local sink.
- Source, relocated static SDK, and relocated shared SDK consumers use the same
  C++ surface. Core does not propagate trace/testing/Tracy state.
- Core Native FFI is `N/A`: these compile-site facilities do not form a useful
  runtime type-erased foreign-language service.

The claim is intentionally bounded. Prefetch is best-effort and its throughput
benefit belongs to the owning consumer workload. Fatal diagnostics are not
async-signal-safe. RISC-V/unknown-target layout remains conservative. ARM64EC
and Windows ARM64 fcontext are outside the current supported matrix.

## Toolchains and gates

| Platform | Toolchain / shape | Result |
|---|---|---|
| Windows x64 | MSVC 19.50 Release/Werror + Debug | Core 5/5; build-scope 6/6; Debug real-object codegen skipped by design |
| Windows x64 | clang-cl 20.1.8 Release | build-scope 6/6 |
| Windows ARM64 cross | MSVC 19.50 Release | exact codegen: 9 functions, 3 ARM prefetch intents |
| Ubuntu 24.04 WSL x64 | GCC 14.2 Release/Werror | build-scope 6/6; parser 14/14; exact codegen and ELF visibility pass |
| Ubuntu 24.04 WSL x64 | GCC 14.2 ASan/UBSan RelWithDebInfo | Core-only 5/5 for each; no sanitizer diagnostic |
| Linux AArch64 cross | GCC 14.2, Werror | compile/link and static/codegen checks pass; no QEMU/runtime claim |
| macOS 26.5.2 arm64 | AppleClang 21 Release/Debug, Werror | Core 5/5 in each; exact codegen and Mach-O visibility pass |
| Windows/macOS | source + relocated static/shared package contract | 1/1 each shape; no Core binary or instrumentation propagation |
| Windows direct consumers | all selected components, Release/Werror | 66/66 regression pass; not a downstream closure certificate |
| independent red team | API, ownership, complexity, package, codegen and symbol audit | no remaining MEDIUM+ finding |

The AArch64 NEON compile log only closes a portability blocker exposed by target
normalization. It is not a SIMD functional or performance certificate.

## Raw evidence hashes

| Evidence | SHA-256 |
|---|---|
| `core-v2-ctest-msvc-release-b241a22.log` | `ed64484d24badb626addfe4781c6e6cc8d822d81729199e2c1e40b0e605582ee` |
| `core-v2-ctest-msvc-debug-b241a22.log` | `93bf9eb9318e82940d35424fa466db0a5ff3f9cea2e0c281b0d8888e7c67a055` |
| `core-v2-ctest-clangcl-release-b241a22.log` | `c2a3e59937e343c3b81048a5e3dc686676f5bcb61116835ec69761ebc4d4cfa0` |
| `core-v2-codegen-msvc-b241a22.txt` | `57c127e351501d9820b7accc7d0491a54159fda469a4d20b4f42dbe7955b33a8` |
| `core-v2-codegen-msvc-arm64-b241a22.txt` | `91165ff4a1cc2c7b62bd4ef835540cb8040139b4e7c7595259124f1c782838e4` |
| `core-v2-symbol-visibility-msvc-b241a22.txt` | `a57f82aed526d017c72e3657a841ad4901ba237a388b27922ae49c120aaa4062` |
| `core-v2-component-contract-msvc-b241a22.log` | `4b9541d90e6cc270fffba942b21a39855cdcab3c202971aa73424eedb24d22e4` |
| `core-v2-direct-consumers-msvc-b241a22.log` | `d071a6a09c14b16168a2f8070e80603ff103a79268721020f45945553d27a27e` |
| `core-v2-ctest-release-gcc-b241a22.log` | `3c4a7cb03b03e4f11f82ce5ad04030c4b771bd88be647652a6ab32cca6bf45cb` |
| `core-v2-codegen-gcc-b241a22.txt` | `5ff6060db7374b8a9b6673c5430b40342cacf4158961507370ffb1dba0da3c17` |
| `core-v2-symbol-visibility-gcc-b241a22.txt` | `3e930cdf384614baf16f08eb287860a54edd5e7b5b26a970f2161fc17d62c1b3` |
| `core-v2-asan-gcc-b241a22.log` | `efcc82af802331acf06e3a970635e24dc3e6332c34a597e627c3005af2ace109` |
| `core-v2-ubsan-gcc-b241a22.log` | `f55eb50f31ca3316f1c7308cedaf463256ce49f9d7957c86d3d7f4c7cadc8b79` |
| `core-v2-target-platform-gcc-b241a22.log` | `65af0805e5297d215e9333256790d9a0dbe501b1a461e3bf2b0164f219c956a8` |
| `core-v2-aarch64-cross-gcc-b241a22.log` | `6b5a6cea2c63661d77eadeb9a7e5d2d6d9b8f20b4bf7a1768cdd9fed8bc6b7ac` |
| `core-v2-ctest-appleclang-b241a22.log` | `e153065f594a16fe03e96895bdd92cf23ca1bea42d35fad50043f08c71970324` |
| `core-v2-ctest-debug-appleclang-b241a22.log` | `4bee9b2d53fc7521ab90ff10e98ac36f21957e97f5234d187f451b79481bed32` |
| `core-v2-codegen-appleclang-b241a22-audit-output.txt` | `a43bb1130f8f3c3c6c7f4a42cbc7861679aa7557afc1ee265d867a5ef08c433d` |
| `core-v2-codegen-appleclang-b241a22-otool-rv.txt` | `3acccf613a379ab82356de7998e82c772aba1b02a34f08006fd853d965a72051` |
| `core-v2-codegen-appleclang-b241a22-otool-tvV.txt` | `b55def06b51d28d2b385ee0a2f9e05ae2ca7ed7df9264e4f23a25ad0c85daf8e` |
| `core-v2-codegen-appleclang-b241a22-verdict.txt` | `86b62d6a05d17b282b7179226ddc2a05eaf4b893931b828343f29c53583ba442` |
| `core-v2-symbol-visibility-appleclang-b241a22.txt` | `a640695df06c70bce9e664f8de9913f689315fc1adcb639cd69bc77893a73271` |
| `core-v2-component-contract-appleclang-b241a22.txt` | `f3581aff456917be1ce380867165ecf26dd7da417222db2cdc624592c1815058` |
| `core-v2-trace-isolation-appleclang-b241a22.log` | `2419ac567e5603a79bf88a67d221861c4670bdbade49739276800d36b3296161` |
| `core-v2-neon-compile-appleclang-b241a22.log` | `e13df2551bb84cf1a30962104dcfe96ec5025fbdb0c951406583cf8c2b004e79` |

The earlier `manifest-c0b9a84.md` describes the retired runtime/managed-FFI
Core design and is retained only as historical task evidence. It must not be
used as the current product certificate. GitHub Actions was deliberately not
invoked because the monthly quota was nearly exhausted; all implementation and
evidence commits use `[skip ci]`.
