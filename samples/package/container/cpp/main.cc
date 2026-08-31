#include <cstdint>
#include <string>
#include <string_view>

#include "hyle/container/circular_buffer.h"
#include "hyle/container/dynamic_bitset.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/string_algo.h"

int main() {
  using hyle::container::FlatHashMap;
  using hyle::container::HardenedStringHash;
  using hyle::container::StringEqual;
  using hyle::container::StringHash;

  hyle::container::CircularBuffer<int> recent(3);
  for (int value = 0; value < 5; ++value) recent.push_back(value);
  if (recent.capacity() != 3 || recent.size() != 3 || recent.front() != 2 ||
      recent.back() != 4) {
    return 1;
  }

  if (!hyle::container::utf8_validate(
          hyle::container::StringView{"installed-\xE2\x9C\x93"})) {
    return 2;
  }

  // A dense active-ID mask is the natural route when the universe is known
  // and bitwise scans matter. Keep this in the installed consumer so the
  // public ownership/lifetime shape is exercised outside the unit tests.
  hyle::container::DynamicBitSet active(128);
  active.set(3).set(65);
  if (active.count() != 2 || !active.test(65)) return 3;

  FlatHashMap<std::string, int, StringHash, StringEqual> trusted;
  trusted.insert_or_assign(std::string("alpha"), 7);
  const int* trusted_value = trusted.find(std::string_view("alpha"));
  if (trusted_value == nullptr || *trusted_value != 7 ||
      !trusted.contains("alpha")) {
    return 4;
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
  return hardened_value != nullptr && *hardened_value == 11 ? 0 : 5;
}
