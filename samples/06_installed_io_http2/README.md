# Sample 06 — installed io package with HTTP/2

This pure-C consumer requests the installed `io` component, audits the package's
`io_http2` feature manifest, links `hyle::io`, and verifies the HTTP/2 protocol core at
runtime. It uses only the installed prefix; neither nghttp2's checkout nor HYLE's source
tree is visible to the consumer.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/hyle-prefix
cmake --build build
./build/installed_io_http2_consumer
```

The same project consumes shared and static HYLE packages. A shared `hyle_io` contains the
private nghttp2 archive and needs no nghttp2 DLL. A static package propagates the pinned
archive through its imported target graph.
