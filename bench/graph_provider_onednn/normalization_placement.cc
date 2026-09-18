// A retained cold contribution shifts code addresses without changing the
// measured functions, their compiler flags, or hot-path instruction counts.
extern "C" int hyle_normalization_placement_prefix(int value) {
  return value * 29 + 7;
}
