# HYLE third-party notices

HYLE's proprietary notice applies only to original HYLE materials. The components below
retain their own copyright and license terms. This index is informational; the referenced
license text and the notices in each source file control.

## In-tree vendored implementations

| Component | HYLE use | License | Full text in source / installed package |
|---|---|---|---|
| mimalloc | `mem` system allocator backend | MIT | `third_party/mimalloc/LICENSE` / `mimalloc-LICENSE` |
| xxHash | `container` byte hashing | BSD-2-Clause | `third_party/xxhash/LICENSE` / `xxhash-LICENSE` |
| wyhash | `container` long-input hashing | The Unlicense | `third_party/wyhash/LICENSE` / `wyhash-LICENSE` |
| simdutf | `container` UTF-8 validation/counting | MIT OR Apache-2.0 | `third_party/simdutf/LICENSE-{MIT,APACHE}` / `simdutf-LICENSE-{MIT,APACHE}` |
| Boost.Context fcontext assembly | `sched` context switching | BSL-1.0 | `third_party/fcontext/LICENSE_1_0.txt` / `fcontext-LICENSE_1_0.txt` |
| picohttpparser | `io` HTTP/1 parser | MIT option of MIT OR Perl | `third_party/picohttpparser/LICENSE` / `picohttpparser-LICENSE` |

## Code ported into HYLE sources

| Origin | HYLE files | Selected license | Full text in source / installed package |
|---|---|---|---|
| Google Highway | `src/simd/sort/vqsort_hyle/`, `src/simd/bitpack/bitpack_hyle-inl.h`, and most of `src/simd/math/math_hyle-inl.h` | Apache-2.0 option of Apache-2.0 OR BSD-3-Clause | `third_party/simdutf/LICENSE-APACHE` / `highway-Apache-2.0-LICENSE`; source files retain upstream copyright, origin, and commit notices |
| JPEG XL | `Cbrt` portion of `src/simd/math/math_hyle-inl.h` | BSD-3-Clause | `third_party/ported/LICENSE-JPEG-XL-BSD-3-Clause` / same basename |
| xsimd | `src/simd/math/math_xsimd_hyle-inl.h` | BSD-3-Clause | `third_party/ported/LICENSE-XSIMD-BSD-3-Clause` / same basename |

The proprietary terms for HYLE's modifications and surrounding original code do not remove
the permissions or conditions applying to these third-party portions.

## Optional external or fetched dependencies

Trace builds may compile and redistribute Tracy's client and public headers; its checkout's
`LICENSE` must accompany that package. OpenSSL, CUDA, and GPU driver runtimes are normally
provided by the target system and remain governed by their suppliers. nghttp2, ggml, volk,
Vulkan-Headers, and glslang are pinned build dependencies for optional product features; an
installable HYLE package is rejected until the selected dependency's artifacts and license
payload are both represented in the export graph.

GoogleTest, nanobench, Highway, and xsimd checkouts used only as tests or benchmark opponents
are not part of a production HYLE installation unless their code is explicitly listed above
as ported into HYLE.
