// Preserve the primary benchmark source/binary and shift only this artifact.
#define main hyle_owned_fragment_main
#include "owned_fragment_bench.cc"
#undef main

extern "C" int hyle_vulkan_placement_prefix(int);
int main(int argc, char **argv) {
  ankerl::nanobench::doNotOptimizeAway(hyle_vulkan_placement_prefix(argc));
  return hyle_owned_fragment_main(argc, argv);
}
