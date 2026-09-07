# Fixture 06 — installed C++ io package with HTTP/2

This C++ consumer requests the installed `io` component, audits the package's
`io_http2` feature manifest, constructs the native HTTP/2 client seam and links
the complete private protocol dependency closure. It uses only the installed
prefix; neither nghttp2's checkout nor HYLE's source tree is visible.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/hyle-prefix
cmake --build build
./build/installed_io_http2_consumer
```

The same project consumes shared and static HYLE packages through `hyle::io`; internal
factor artifact names are package details. The shared product closure contains the private
nghttp2 archive and needs no nghttp2 DLL. A static package propagates the pinned archive
through its imported target graph.
