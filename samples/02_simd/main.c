/* Stable, runtime-dispatched SIMD C ABI. One binary runs on unknown customer CPUs. */
#include <stdio.h>

#include "hyle/c/simd.h"

int main(void) {
  float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float y[8] = {8, 7, 6, 5, 4, 3, 2, 1};
  float out[8];
  hyle_simd_runtime_info runtime = HYLE_SIMD_RUNTIME_INFO_INIT;

  if (!hyle_simd_runtime_info_get(&runtime)) {
    fprintf(stderr, "SIMD runtime query failed: %s\n", hyle_simd_last_error());
    return 1;
  }
  hyle_simd_add_f32(x, y, out, 8);
  printf("active ISA=%u, dot=%.1f, out[0]=%.1f\n",
         (unsigned)runtime.isa, hyle_simd_dot_f32(x, y, 8), out[0]);
  return 0;
}
