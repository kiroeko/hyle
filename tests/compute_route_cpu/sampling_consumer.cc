#include "../compute_models/sampling_common_consumer.h"

int main() {
  try { sampling_test::run<sampling_test::CommonSession>(); return 0; }
  catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL CPU sampling: %s\n", error.what()); return 1;
  }
}
