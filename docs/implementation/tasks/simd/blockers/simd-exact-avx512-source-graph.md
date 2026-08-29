# SIMD exact AVX512 source-graph blocker

- state：`RESOLVED`
- discovered by：SIMD-10 exact delivery gate
- affected journey：clang/GCC `HYLE_SIMD_EXACT_TARGET=AVX512`
- product/API impact：none；source-selection correctness only

## Qualification

The exact AVX512 graph compiled both `transcendental_highway_avx512.cc` and the AVX2-only
`transcendental_highway.cc`. The library-wide exact minimum flags are intentionally visible to baseline
and ordinary sources, so the latter TU saw AVX512 feature macros even though its source property also
appended AVX2 flags. Locked Highway then selected AVX3 and correctly required AVX512CD, while HYLE's
canonical AVX512 tier deliberately does not promise CD. The generated AVX512 bitpack target had already
compiled; this unrelated math source prevented the shared artifact from linking.

Adding CD to the canonical tier is rejected: it would silently strengthen the CPU contract without a
matching Platform probe and would erase the intended AVX2/AVX512 implementation boundary. The finite fix
is to include the AVX2 math source only when AVX2 is a compiled target or classic cl.exe needs its existing
AVX2 fallback. clang/GCC exact AVX512 already owns a dedicated AVX512 math TU.

## Frozen validation

- reconfigure and build the clang-cl exact AVX512 SIMD test artifact;
- run `hyle_simd_exact_artifact_check` and confirm the graph has one canonical target and no forbidden
  lower-tier source;
- retain the already-passing exact scalar/SSE4/AVX2 focused bitpack/random gates;
- do not change ISA flags, Platform feature detection, Runtime metadata, public API or performance tools.

## Resolution

`src/simd/CMakeLists.txt` now includes the AVX2 math TU for an AVX2 target or the existing classic
cl.exe fallback only. The target also emits a private `HYLE_SIMD_CLASSIC_CL_AVX512_FALLBACK`
front-end contract; `math_dispatch.cc` and the dedicated AVX512 TU consume that contract instead of
mistaking clang-cl's `_MSC_VER` compatibility definition for cl.exe. Canonical AVX512 remains
F/BW/VL/DQ/BMI2 and does not acquire an unprobed AVX512CD requirement.

The clang-cl exact AVX512 shared library and `hyle_simd_exact_artifact_check` pass. The clang-cl fat
library's ordinary/SSE4/AVX2/SSE4-AES tests pass, and the classic cl.exe source/shared/static selected
component journeys compile and run. No public API, Platform feature bit or ISA tier changed.
