# Managed-runtime direct core FFI probes

These samples verify the smallest real managed-runtime boundary of the shared
`hyle_core` library. They are deliberately **not C product examples** and are
not full .NET or Java bindings. Their job is to prove that a foreign runtime
can load one exact native artifact, resolve the stable Native FFI bootstrap
symbol, and reject an incompatible ABI before using a wider module surface.

Both probes first validate the global ABI bootstrap using the compatibility
rule “same major and runtime minor greater than or equal to the binding's
required minor”; only then do they resolve/call the product-identity symbols.
For the current source tree the expected values are:

- `hyle_ffi_abi_version()` = `0x00010000`;
- `hyle_version()` = `0x00000100`;
- `hyle_version_string()` = `0.1.0`.

Exact product-version equality is only this repository probe's artifact-identity
assertion. It is not the compatibility rule; production bindings negotiate only
with `hyle_ffi_abi_version` and may read product identity for diagnostics.

The .NET 8 probe uses source-generated `LibraryImport` stubs plus a
`DllImportResolver`. The Java 25 probe uses the final Foreign Function & Memory
API and `SymbolLookup.libraryLookup(Path, Arena)`. Neither probe searches for a
bare `hyle_core` name: the runner canonicalizes and passes the exact `.dll`,
`.so`, or `.dylib` path so an older system-installed copy cannot satisfy the
test accidentally.

## Run

Build shared `hyle_core` and the test-only
`hyle_core_ffi_incompatible_fixture` target first, then run from the HYLE
repository root. The fixture exports ABI major 2 and deliberately omits both
product symbols, so a passing negative probe proves bootstrap rejection happened
before wider symbol resolution. It is never installed.

```powershell
python scripts/run-core-ffi-smoke.py `
  --library C:\absolute\path\to\hyle_core.dll `
  --incompatible-library C:\absolute\path\to\hyle_core_ffi_incompatible_fixture.dll
```

```sh
python3 scripts/run-core-ffi-smoke.py \
  --library /absolute/path/to/libhyle_core.so \
  --incompatible-library /absolute/path/to/libhyle_core_ffi_incompatible_fixture.so
```

Use `--only dotnet` or `--only java` to select one probe. The default is
`--only all`. The runner requires a .NET SDK capable of targeting `net8.0` and
a JDK 25 `javac`/`java` pair. Java is launched with
`--enable-native-access=ALL-UNNAMED`, as required for an intentional FFM
downcall.

All copied sources, compiler intermediates, and outputs live in one temporary
directory that is removed on exit. Running these probes does not create build
artifacts under `samples/` or elsewhere in the repository.
