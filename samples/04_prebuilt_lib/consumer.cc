// Sample 04 -- consume an installed HYLE package (shared OR static), not the source
// tree. This validates the full package/export boundary:
//
//  - SHARED: the consumer links libhyle_simd.so / hyle_simd.dll and calls symbols marked
//    HYLE_SIMD_API. On Windows that resolves to __declspec(dllimport) here / dllexport in
//    the DLL -- if the export macros were wrong, this link would FAIL. So a clean link +
//    run is the proof the cross-DLL surface works.
//  - STATIC: links libhyle_simd.a / hyle_simd.lib; HYLE_SIMD_API expands to nothing
//    (HYLE_STATIC). Proves the static path is consistent too.
//
// find_package supplies the correct HYLE_STATIC definition, include directory and
// transitive platform/error targets; the consumer must not reconstruct any of those.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/platform/cpu.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/runtime.h"

int main() {
  hyle::mem::AllocatorRef allocator = hyle::mem::system_allocator();
  auto* package_probe = allocator.allocate_n<std::uint64_t>(8);
  if (package_probe == nullptr) return 2;
  allocator.deallocate_n(package_probe, 8);

  constexpr std::size_t n = 512;
  std::vector<float> x(n, 1.5f), y(n, 2.0f);
  // dot lives in the installed hyle_simd library; reaching it proves the imported
  // target's shared/static symbol resolution works.
  const float dot = hyle::simd::dot(x.data(), y.data(), n);
  std::printf("linked installed hyle::simd + hyle::mem OK\n");
  std::printf("  active SIMD variant : %s\n",
              hyle::simd::kernel_target_name(
                  hyle::simd::selected_target().target));
  std::printf("  dot(1.5*2.0 x512)   : %.1f (expect %.1f)\n", dot, 1.5f * 2.0f * n);
  return (dot == 1.5f * 2.0f * static_cast<float>(n)) ? 0 : 1;
}
