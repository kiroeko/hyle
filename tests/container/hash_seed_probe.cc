#include <cinttypes>
#include <cstdio>

#include "hyle/container/hash.h"

int main() {
  const hyle::container::StringHash hash;
  std::printf("%" PRIx64 "\n", hash("process-seed-probe"));
  return 0;
}
