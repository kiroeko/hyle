#include <cstdint>
#include <string>
#include <string_view>

#include "hyle/container/string_algo.h"
#include "hyle/container/flat_hash_map.h"

int main() {
  using hyle::container::FlatHashMap;
  using hyle::container::HardenedStringHash;
  using hyle::container::StringEqual;
  using hyle::container::StringHash;

  if (!hyle::container::utf8_validate(
          hyle::container::string_ref{"installed-\xE2\x9C\x93"})) {
    return 1;
  }

  FlatHashMap<std::string, int, StringHash, StringEqual> trusted;
  trusted.insert_or_assign(std::string("alpha"), 7);
  const int* trusted_value = trusted.find(std::string_view("alpha"));
  if (trusted_value == nullptr || *trusted_value != 7 ||
      !trusted.contains("alpha")) {
    return 2;
  }

  using HardenedMap =
      FlatHashMap<std::string, int, HardenedStringHash, StringEqual>;
  // Fixed words make this package smoke reproducible. Production code must
  // obtain an unpredictable 128-bit per-process/table key from its CSPRNG or
  // secret manager; never copy these constants into an untrusted-key table.
  HardenedMap untrusted(
      hyle::mem::SystemAllocator{},
      HardenedStringHash{UINT64_C(0x0123456789abcdef),
                         UINT64_C(0xfedcba9876543210)},
      StringEqual{});
  untrusted.insert_or_assign(std::string("request-key"), 11);
  const int* hardened_value =
      untrusted.find(std::string_view("request-key"));
  return hardened_value != nullptr && *hardened_value == 11 ? 0 : 3;
}
