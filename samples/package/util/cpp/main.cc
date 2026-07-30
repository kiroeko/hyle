#include "hyle/util/demangle.h"
#include "hyle/util/registry.h"

#include <cstddef>
#include <cstring>

namespace {

void* create_value() {
  static int value = 7;
  return &value;
}

}  // namespace

int main() {
  hyle::util::Registry<void*> registry;
  if (!registry.register_creator("sample", &create_value,
                                 hyle::util::RegistryPriority::kDefault,
                                 "installed C++ package probe")
           .ok()) {
    return 1;
  }
  const auto frozen = registry.freeze();
  if (frozen.create("sample") == nullptr) {
    return 2;
  }

  char readable[32];
  const std::size_t required =
      hyle::util::demangle("plain_name", readable, sizeof(readable));
  return required == 10 && std::strcmp(readable, "plain_name") == 0 ? 0 : 3;
}
