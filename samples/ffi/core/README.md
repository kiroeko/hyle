# Managed-runtime direct core FFI probes

These samples verify the smallest real managed-runtime boundary of the shared
`hyle_core` library. They are deliberately **not C product examples** and are
not full .NET or Java bindings. Their job is to prove that a foreign runtime
can load one exact native artifact, resolve the stable Native FFI bootstrap
symbols, and reject a binding/package version mismatch before using a wider
module surface.

Both probes call and validate:

- `hyle_ffi_abi_version()` = `0x00010000`;
- `hyle_version()` = `0x00000100`;
- `hyle_version_string()` = `0.1.0`.

The .NET 8 probe uses source-generated `LibraryImport` stubs plus a
`DllImportResolver`. The Java 25 probe uses the final Foreign Function & Memory
API and `SymbolLookup.libraryLookup(Path, Arena)`. Neither probe searches for a
bare `hyle_core` name: the runner canonicalizes and passes the exact `.dll`,
`.so`, or `.dylib` path so an older system-installed copy cannot satisfy the
test accidentally.

## Run

Build a shared `hyle_core` first, then run from the HYLE repository root:

```powershell
python scripts/run-core-ffi-smoke.py `
  --library C:\absolute\path\to\hyle_core.dll
```

```sh
python3 scripts/run-core-ffi-smoke.py \
  --library /absolute/path/to/libhyle_core.so
```

Use `--only dotnet` or `--only java` to select one probe. The default is
`--only all`. The runner requires a .NET SDK capable of targeting `net8.0` and
a JDK 25 `javac`/`java` pair. Java is launched with
`--enable-native-access=ALL-UNNAMED`, as required for an intentional FFM
downcall.

All copied sources, compiler intermediates, and outputs live in one temporary
directory that is removed on exit. Running these probes does not create build
artifacts under `samples/` or elsewhere in the repository.
