#include "hyle/core/version.h"

int main() {
  return hyle::version() == hyle::kVersion ? 0 : 1;
}
