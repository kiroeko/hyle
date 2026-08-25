# core `c0b9a84` verification manifest

This manifest binds the 2026-08-25 core closure evidence to one committed
source snapshot. It is an engineering certificate for a pre-release candidate,
not a compatibility promise to an existing native-C or managed customer.

## Identity and scope

| Item | Value |
|---|---|
| source/public-contract commit | `c0b9a84d81ad6df1fb81fcec6235425a8f3448ab` |
| source tree | `795fc79e925df13c6df9750129afea7fc2029c3d` |
| compiled core behavior base | `9459d3fa91e75b95d261067284451c45fe96f54a` / tree `925b516d00226a20767361ec59b3fee057ae1b21` |
| later source changes | comments, test/evidence machinery, probe comments and sample classification; no change to core runtime behavior or claim-bearing hot paths |
| Windows source state | repository `HEAD` was the commit above; the code-bearing validation set (`CMakeLists.txt`, `cmake/`, `include/`, `src/`, `tests/`, managed probe source and runner) had no uncommitted change |
| WSL source state | detached `HEAD` and tree printed exactly the commit/tree above; `git status --short` was empty |
| macOS source state | detached `HEAD` and tree printed exactly the commit/tree above; `git status --short` was empty |

The WSL build directory still contains `3a30069` and the macOS build directory
still contains `7ed488a` in their **directory names** because they were reused as
incremental caches. Before the recorded configure/build/test, both source
worktrees were fetched and detached at the full `c0b9a84...` commit; the
`rev-parse` results above are the provenance, not those cache labels.

## Toolchains and commands

| Platform | Toolchain | Effective configure/build/test contract |
|---|---|---|
| Windows 11 x64 | MSVC `19.50.35728.0`, CMake/Ninja, Release | `HYLE_COMPONENTS=core`, shared, tests on; build `hyle_core_tests` and the incompatible fixture, then `ctest -L core`; separately install the tests-off shared/static package and run installed C++ shared/static consumers plus a complete-source static C++ consumer |
| Ubuntu 24.04 WSL x64 | GCC/G++ `14.2.0`, CMake/Ninja, Release | `-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DHYLE_COMPONENTS=core -DHYLE_BUILD_TESTS=ON -DHYLE_STATIC=OFF`, then build core tests/fixture and `ctest -L core` |
| macOS arm64 | AppleClang `21.0.0`, CMake, Release | `-DHYLE_COMPONENTS=core -DHYLE_BUILD_TESTS=ON -DHYLE_STATIC=OFF`, then build core tests/fixture and `ctest -L core` |
| Ubuntu 24.04 WSL x64 sanitizer | GCC/G++ `14.2.0`, RelWithDebInfo | the WSL contract above, in independent builds with `HYLE_SANITIZER=asan` and `HYLE_SANITIZER=ubsan`; core is not a concurrent runtime, so TSan is not an applicable current core gate |

The managed runner command was:

```text
python scripts/run-core-ffi-smoke.py
  --library <installed-shared-prefix>/bin/hyle_core.dll
  --incompatible-library <test-build>/bin/hyle_core_ffi_incompatible_fixture.dll
  --evidence-json <output>
```

It ran Java 25 FFM and .NET SDK 10 targeting `net8.0`, each with one positive
exact-path load and one incompatible-major rejection. The negative path also
confirmed that both product-identity symbols were absent before reporting PASS.

## Source and artifact hashes

| Input/artifact | SHA-256 |
|---|---|
| `include/hyle/c/core.h` | `bfde24217510f092d46594d0114b9a527083c2879deb85e34ac33726da5c45b3` |
| `include/hyle/core/version_defs.h` | `075010e62663da751cd9640344f03f81ade087febd5bf9b13128cdf7849578c3` |
| `include/hyle/core/version.h` | `8bc3cc5f80bed169f59f660d2216289993c3cafae54247f7bf6ebcc175a27d6e` |
| `src/core/c_core.cc` | `c2912e181c476a57738f92bef37b459d56dd83f365479e496d48da608e29a04b` |
| incompatible fixture source | `6d5fc20ef7f4f929b39e74b89a1ffda8f52f5d3d1089f72458e36fded5adc81d` |
| managed runner | `8e6ad4fa6a94d50f37829c03bc479ee23ded407015e4107940d1c45cfa79b8ee` |
| Java probe source | `8c950ee08a27162f0c29e26141bbe4323f06f8e6d8169192378a68628a494647` |
| .NET probe source | `3f5a24dcfd2f94120d166f003649afac24b143cc123827590dba2fc89a3c7b21` |
| .NET project | `daef8201fffae3cf6672d49fe179f247a33c8a5d9d5f273eb32b89a4e24320c7` |
| installed Windows `hyle_core.dll` | `5c1cd086f4d77db311bd40492cc78c9a76b6165bbac8cf7913ae2cfce00b4006` |
| Windows incompatible fixture | `4bc59955eab5bc74136e4f3f729e3d011ebea384a82f9c9471e30b127d1e7edf` |
| WSL `libhyle_core.so.0.1.0` | `1e7db551c17428c0c19b5fa36243cc77e856baecb2b52bb45eeed8a8560f8535` |
| WSL incompatible fixture | `42779c37aed6b85a864b0dc67bb8aad7b5527cf7543e89d06106e16ff30588ab` |
| macOS `libhyle_core.dylib` | `389b5a2bf18f1e48e516825525d787ae2c32ce8e66443d456c8c5d5a83731980` |
| macOS incompatible fixture | `3706587f3dd2c80275827dd7c0e5bf2269306f56023f3b1720e83d51e0b1e5da` |

## Tracked evidence

| Evidence | Result | SHA-256 |
|---|---|---|
| [Windows core CTest](./core-ctest-windows-c0b9a84.log) | 4/4 PASS | `46b96064eaccb5606800d20c3c9fe0419762d1770d35cd48cf67ac89141b6ba3` |
| [WSL core CTest](./core-ctest-wsl-c0b9a84.log) | 4/4 PASS | `1c235fbc8d8a3370d688203762b75151be7724e58c00bb08075a98eab75acebf` |
| [macOS core CTest](./core-ctest-macos-c0b9a84.log) | 4/4 PASS | `da442ceb06194178ed55f838330be19c39ea6877d06793aea1e7824912e9aea1` |
| [WSL ASan CTest](./core-asan-wsl-c0b9a84.log) | 4/4 PASS | `e038d4dd66f08f64aa958acd88530805be17b4523f97263d5e047cbb87abc0e2` |
| [WSL UBSan CTest](./core-ubsan-wsl-c0b9a84.log) | 4/4 PASS | `30970e63d80c466e895019a7a4b3199aaea010593256df9fb3869671742c5d3e` |
| [installed shared C++ CTest](./cpp-installed-shared-windows-c0b9a84.log) | 1/1 PASS | `65a6c153849671979e663cb50f8fb51ea363fb69aa19ef9f4ec98df97d696a69` |
| [installed static C++ CTest](./cpp-installed-static-windows-c0b9a84.log) | 1/1 PASS | `3763381a2f15b52c823ddb003433ebe9708612a7106fad42150fc9e5c426752f` |
| [complete-source C++ CTest](./cpp-source-windows-c0b9a84.log) | 1/1 PASS | `dcfc450ccd10a36af56a73aefd2be25df6d812b0927e2b4470359244fa3bf47b` |
| [managed FFI structured evidence](./managed-ffi-windows-c0b9a84.json) | Java/.NET positive + negative 4/4 PASS | `0ed17aed817b1f39fcdee7bca0eb83974acb2ed9393050156bbce50eac13de4f` |
| [FFI export transcript](./ffi-exports-c0b9a84.log) | product 3 symbols; each incompatible fixture only bootstrap | `f4b0cf1ba2dccb8bb2857ddee7eb9777e2c781b6946fb09737876057f6769e4b` |

GitHub Actions was deliberately not invoked: every implementation/evidence
commit in this round used `[skip ci]` because the monthly quota was nearly
exhausted. These are local/remote-machine release validations, not a claim that
a new continuous CI job ran.
