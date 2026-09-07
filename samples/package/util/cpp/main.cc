#include <cstddef>
#include <cstring>
#include <memory>
#include <string>

#include "hyle/container/string_view.h"
#include "hyle/util/demangle.h"
#include "hyle/util/registry.h"
#include "hyle/util/scope_guard.h"

namespace {
struct Backend {
  std::shared_ptr<int> image;
};
using Registry = hyle::util::Registry<hyle::Result<Backend>, bool>;

// Version compatibility belongs to the typed backend owner, before committing
// any entry. A real image owner also keeps its unload operation here.
hyle::Status register_backend(Registry& registry, unsigned version,
                              std::shared_ptr<int> image) noexcept {
  if (version != 1) {
    return hyle::Status::InvalidArgument("incompatible backend version");
  }
  return registry.register_creator(
      "sample",
      [image =
           std::move(image)](bool available) noexcept -> hyle::Result<Backend> {
        if (!available) {
          return hyle::make_unexpected(
              hyle::Status::Unavailable("sample backend unavailable"));
        }
        return Backend{image};
      },
      hyle::util::RegistryPriority::kDefault, "installed C++ package probe");
}
}  // namespace

int main() {
  bool scope_finished = false;
  {
    hyle::util::ScopeGuard cleanup{[&]() noexcept { scope_finished = true; }};
    (void)cleanup;
  }
  if (!scope_finished) return 1;

  std::weak_ptr<int> image_lifetime;
  Backend object;
  {
    Registry registry;
    auto image = std::make_shared<int>(7);
    image_lifetime = image;
    if (!register_backend(registry, 2, image).IsInvalidArgument() ||
        registry.size() != 0)
      return 2;
    if (!register_backend(registry, 1, std::move(image)).ok()) return 3;
    const auto frozen = registry.freeze();
    const auto entry = frozen.find("sample");
    if (!entry || !entry->creator ||
        entry->key != hyle::container::StringView{"sample"})
      return 4;
    const auto created = entry->creator(true);
    if (!created || *created->image != 7) return 5;
    object = *created;
    const auto failed = frozen.create("sample", false);
    const auto missing = frozen.create("missing", true);
    if (failed || !failed.error().IsUnavailable() || missing ||
        !missing.error().IsNotFound())
      return 6;
  }
  if (image_lifetime.expired()) return 7;
  object.image.reset();
  if (!image_lifetime.expired()) return 8;

  char readable[32];
  const std::size_t required =
      hyle::util::demangle("plain_name", readable, sizeof(readable));
  if (required != 10 || std::strcmp(readable, "plain_name") != 0) return 9;
  return hyle::util::demangle("plain_name") == "plain_name" ? 0 : 10;
}
