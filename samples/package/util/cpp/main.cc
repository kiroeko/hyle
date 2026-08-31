#include "hyle/container/string_view.h"
#include "hyle/util/demangle.h"
#include "hyle/util/registry.h"

#include <cstddef>
#include <cstring>
#include <string>

namespace {

hyle::Result<void*> create_value(bool available) noexcept {
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
  const auto entry = frozen.find("sample");
  if (!entry.has_value() ||
      entry->key != hyle::container::StringView{"sample"} ||
      entry->creator != &create_value ||
      entry->priority != hyle::util::RegistryPriority::kDefault) {
    return 2;
  }
  const auto creator = entry->creator;
  const auto created = creator(true);
  if (!created.has_value() || *created == nullptr) {
    return 3;
  }
  const auto failed = frozen.create("sample", false);
  const auto missing = frozen.create("missing", true);
  if (failed.has_value() || !failed.error().IsUnavailable() ||
      missing.has_value() || !missing.error().IsNotFound()) {
    return 4;
  }

  char readable[32];
  const std::size_t required =
      hyle::util::demangle("plain_name", readable, sizeof(readable));
  if (required != 10 || std::strcmp(readable, "plain_name") != 0) {
    return 5;
  }
  const std::string owned = hyle::util::demangle("plain_name");
  return owned == "plain_name" ? 0 : 6;
}
