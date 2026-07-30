#include "hyle/util/demangle.h"
#include "hyle/util/registry.h"

#include <cstddef>
#include <cstring>

namespace {

hyle::Result<void*> create_value(bool available) {
  if (!available) {
    return hyle::make_unexpected(
        hyle::Status::Unavailable("sample backend unavailable"));
  }
  static int value = 7;
  return &value;
}

}  // namespace

int main() {
  hyle::util::Registry<hyle::Result<void*>, bool> registry;
  if (!registry.register_creator("sample", &create_value,
                                 hyle::util::RegistryPriority::kDefault,
                                 "installed C++ package probe")
           .ok()) {
    return 1;
  }
  const auto frozen = registry.freeze();
  const auto created = frozen.create("sample", true);
  if (!created.has_value() || *created == nullptr) {
    return 2;
  }
  const auto failed = frozen.create("sample", false);
  const auto missing = frozen.create("missing", true);
  if (failed.has_value() || !failed.error().IsUnavailable() ||
      missing.has_value() || !missing.error().IsNotFound()) {
    return 3;
  }

  char readable[32];
  const std::size_t required =
      hyle::util::demangle("plain_name", readable, sizeof(readable));
  return required == 10 && std::strcmp(readable, "plain_name") == 0 ? 0 : 4;
}
