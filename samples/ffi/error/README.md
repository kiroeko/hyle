# Managed-runtime direct error FFI probes

These samples are executable binding checks for the shared `hyle_error`
artifact. They are not a native-C product API and not general-purpose Java or
.NET packages. They prove the current pre-release managed FFI seam end to end:

- `NULL` is the complete OK value;
- a code-only `RESOURCE_EXHAUSTED` failure crosses as an allocation-free
  process token;
- an `UNAVAILABLE` failure preserves Vulkan native code `-4` and the eight
  message bytes `d e v NUL l o s t`;
- every non-NULL owning handle is destroyed exactly once by Java
  `AutoCloseable` or a .NET `SafeHandle`;
- an ABI-major-2 artifact is rejected before any status symbol is used.

The managed programs consume only the six exports declared by
`hyle/c/error.h`: the artifact-local bootstrap plus destroy, code, native
detail, message, and string queries. They cannot manufacture errors. The
separate `hyle_error_ffi_status_fixture` test DSO is a deliberate failure
producer: inside native test code it converts real C++ `Status` values through
the source-internal `status_to_ffi` bridge and exports three result sources.
Neither fixture is installed.

Both clients load exact absolute paths. They first call
`hyle_error_ffi_abi_version()` from the selected error artifact and accept only
the same ABI major with at least the required minor. In the positive journey,
they independently call
`hyle_error_status_fixture_ffi_abi_version()` before using any producer entry
point. Native entry points are resolved lazily in .NET and explicitly after
bootstrap in Java. The negative journey calls only the incompatible
artifact's bootstrap before rejecting it; post-rejection symbol lookups merely
verify that the fixture really has a one-symbol surface.

The .NET 8 program uses source-generated `LibraryImport` calls and one
`DllImportResolver` that maps private import names to the two exact files. It
keeps both native library handles alive until process exit and gives each
failure pointer to a zero-invalid `SafeHandle`; borrowing query imports accept
that SafeHandle directly so .NET pins its lifetime for every native call. The Java 25 program
uses the final Foreign Function & Memory API with exact-path
`SymbolLookup.libraryLookup` scopes and an owning `AutoCloseable`. Message
lengths, not C-string scans, control both bindings so the embedded NUL remains
observable.

## Run

Configure a shared HYLE build, then build `hyle_error_tests` (or the two
fixture targets directly). Pass the three resulting shared artifacts to the
runner:

```powershell
python scripts/run-error-ffi-smoke.py `
  --library C:\absolute\path\to\hyle_error.dll `
  --producer-library C:\absolute\path\to\hyle_error_ffi_status_fixture.dll `
  --incompatible-library C:\absolute\path\to\hyle_error_ffi_incompatible_fixture.dll
```

```sh
python3 scripts/run-error-ffi-smoke.py \
  --library /absolute/path/to/libhyle_error.so \
  --producer-library /absolute/path/to/libhyle_error_ffi_status_fixture.so \
  --incompatible-library /absolute/path/to/libhyle_error_ffi_incompatible_fixture.so
```

Use `--only dotnet` or `--only java` to select one runtime; the default runs
all four positive/negative processes. The runner requires a .NET SDK capable
of targeting `net8.0` and JDK 25 or newer. Java is launched with
`--enable-native-access=ALL-UNNAMED`. `--evidence-json <path>` atomically
writes hashes, tool versions, commands, and captured probe results, but only
after every selected process succeeds.

Managed sources and compiler outputs are copied into a temporary directory and
removed on exit. The runner does not place generated artifacts under
`samples/`.
