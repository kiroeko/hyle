// Fixed Philox4x32-10 equations, Random123 BSD-3-Clause; see installed license.
// Shared unchanged by the product shader and the private endpoint probe.
uvec4 philox(uvec4 value, uvec2 key) {
  for (uint round = 0; round < 10; ++round) {
    uint high0, low0, high1, low1;
    umulExtended(0xd2511f53u, value.x, high0, low0);
    umulExtended(0xcd9e8d57u, value.z, high1, low1);
    value = uvec4(high1 ^ value.y ^ key.x, low1,
                  high0 ^ value.w ^ key.y, low0);
    key += uvec2(0x9e3779b9u, 0xbb67ae85u);
  }
  return value;
}
float gumbel(uint hi, uint lo) {
  bool upper = (hi & 0x80000000u) != 0;
  if (upper) { hi = ~hi; lo = ~lo; }
  float small = float(hi) * 2.3283064365386962890625e-10 +
                  (float(lo) + .5) * 5.421010862427522170037264e-20;
  precise float t = 1.0 - small;
  // Compensated log1p equation, retaining positive exponential noise even
  // when 1-small rounds to one. Never clamp an already rounded U to an end.
  float exponential = upper ? (t == 1.0 ? small : small * (log(t) / (t - 1.0)))
                              : -log(small);
  return -log(exponential);
}
