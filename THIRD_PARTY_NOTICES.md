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

## Pinned fetched implementations included in packages

| Component | HYLE use | License | Full text in checkout / installed package |
|---|---|---|---|
| Google Highway | `simd` portable register, target and mathematics implementation | Apache-2.0 OR BSD-3-Clause | fetched checkout `LICENSE` / `highway-upstream-LICENSE` |
| nghttp2 | `io` HTTP/2 protocol core | MIT | fetched checkout `COPYING` / `nghttp2-LICENSE` |
| ggml | `compute` CPU/CUDA operator implementation | MIT | fetched checkout `LICENSE` / `ggml-LICENSE` |
| volk | `compute` Vulkan meta-loader | MIT | fetched checkout `LICENSE.md` / `volk-LICENSE.md` |
| Vulkan-Headers | Vulkan API declarations compiled with volk | Apache-2.0 OR MIT, by file | fetched checkout `LICENSE.md`, `LICENSES/` / `Vulkan-Headers/{LICENSE.md,Apache-2.0.txt,MIT.txt}` |

## Code ported into HYLE sources

| Origin | HYLE files | Selected license | Full text in source / installed package |
|---|---|---|---|
| Google Highway | `src/simd/sort/vqsort_hyle/` | Apache-2.0 option of Apache-2.0 OR BSD-3-Clause | `third_party/ported/LICENSE-HIGHWAY` / `highway-LICENSE`; source files retain upstream copyright, origin, and commit notices |

The proprietary terms for HYLE's modifications and surrounding original code do not remove
the permissions or conditions applying to these third-party portions.

## Optional external or fetched dependencies

Trace builds may compile and redistribute Tracy's client and public headers; its checkout's
`LICENSE` must accompany that package. OpenSSL, CUDA, and GPU driver runtimes are normally
provided by the target system and remain governed by their suppliers. glslang is fetched only
when HYLE's Vulkan tests or benchmarks regenerate shaders; it is not linked into or installed
with a production HYLE package.

GoogleTest, nanobench, and xsimd checkouts used only as tests or benchmark opponents are not
part of a production HYLE installation. Highway is different: SIMD product builds consume the
centrally locked source directly, while parity benchmarks still use a separate clean checkout
of the same identity so candidate and peer cannot accidentally share include paths.
