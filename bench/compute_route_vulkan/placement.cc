// Retained cold code preceding the benchmark TU changes text placement without
// adding padding/instructions to measured functions or changing their flags.
extern "C" int hyle_vulkan_placement_prefix(int value) {
  return value * 17 + 3;
}
