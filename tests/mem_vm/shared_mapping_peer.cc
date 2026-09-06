#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include "hyle/mem/mapping.h"
#include "hyle/platform/virtual_memory.h"

namespace {

bool parse_size(std::string_view text, std::size_t* value) noexcept {
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed > (std::numeric_limits<std::size_t>::max)()) {
    return false;
  }
  *value = static_cast<std::size_t>(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 10;
  std::size_t size = 0;
  std::size_t expected = 0;
  if (!parse_size(argv[2], &size) || !parse_size(argv[3], &expected) ||
      size == 0 || expected > 255) {
    return 11;
  }

  auto opened = hyle::mem::SharedMapping::open(argv[1], size);
  if (!opened) return 12;
  hyle::mem::SharedMapping mapping = std::move(*opened);
  if (mapping.access() != hyle::mem::MappingAccess::kReadOnly ||
      reinterpret_cast<std::uintptr_t>(mapping.data()) %
              hyle::platform::page_size() !=
          0 ||
      !mapping.writable_bytes().error().IsPermissionDenied()) {
    return 13;
  }
  const auto bytes = mapping.bytes();
  for (const std::byte value : bytes) {
    if (value != static_cast<std::byte>(expected)) return 14;
  }
  return mapping.reset().ok() ? 0 : 15;
}
