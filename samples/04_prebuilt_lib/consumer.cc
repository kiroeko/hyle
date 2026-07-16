// Sample 04 -- consume HYLE as a PREBUILT library (shared OR static), not as a source
// subdirectory. This is the integration mode that VALIDATES HYLE's export boundary:
//
//  - SHARED: the consumer links libhyle_simd.so / hyle_simd.dll and calls symbols marked
//    HYLE_SIMD_API. On Windows that resolves to __declspec(dllimport) here / dllexport in
//    the DLL -- if the export macros were wrong, this link would FAIL. So a clean link +
//    run is the proof the cross-DLL surface works.
//  - STATIC: links libhyle_simd.a / hyle_simd.lib; HYLE_SIMD_API expands to nothing
//    (HYLE_STATIC). Proves the static path is consistent too.
//
// HONEST SCOPE: this exercises HYLE's C++ API across the dllexport/import boundary. The
// PURE extern "C" C-ABI shim (a flat C header, no name mangling) is a planned milestone
// (P6), NOT done yet -- so this is "C++ ABI across a DLL", not "a C ABI". We call a plain
// free function (dot) whose signature is C-friendly to keep the surface minimal.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/runtime.h"

int main() {
  constexpr std::size_t n = 512;
  std::vector<float> x(n, 1.5f), y(n, 2.0f);
  // dot lives in the prebuilt hyle_simd library; reaching it proves the export/import
  // (shared) or static-archive symbol resolution works.
  const float dot = hyle::simd::dot(x.data(), y.data(), n);
  std::printf("linked prebuilt hyle_simd OK\n");
  std::printf("  active SIMD variant : %s\n",
              hyle::simd::isa_name(hyle::simd::runtime().isa));
  std::printf("  dot(1.5*2.0 x512)   : %.1f (expect %.1f)\n", dot, 1.5f * 2.0f * n);
  return (dot == 1.5f * 2.0f * static_cast<float>(n)) ? 0 : 1;
}
