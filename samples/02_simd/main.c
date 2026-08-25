/* Pre-release SIMD FFI schema/package conformance fixture.
 * This is not a native-C product sample or a certified managed binding. */
#include <stdio.h>

#include "hyle/c/error_helpers.h"
#include "hyle/c/simd.h"

int main(void) {
  float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float y[8] = {8, 7, 6, 5, 4, 3, 2, 1};
  float out[8];
  hyle_simd_runtime_info runtime = HYLE_SIMD_RUNTIME_INFO_INIT;

  hyle_status status = hyle_simd_runtime_info_get(&runtime);
  if (!hyle_status_is_ok(status)) {
    char message[256];
    hyle_status_message(status, message, sizeof(message));
    fprintf(stderr, "SIMD runtime query failed: %s\n", message);
    hyle_status_free(&status);
    return 1;
  }
  hyle_status_free(&status);
  hyle_simd_add_f32(x, y, out, 8);
  printf("active ISA=%u, dot=%.1f, out[0]=%.1f\n",
         (unsigned)runtime.isa, hyle_simd_dot_f32(x, y, 8), out[0]);
  return 0;
}
