// Sample 01 -- mem + container: the default allocation path used by HYLE containers.
//
// FlatHashMap is HYLE's public flat-hash-map shape (control bytes + SIMD probing).
// Like the other header-inline containers, it defaults to the compile-time-bound,
// stateless SystemAllocator. Normal users therefore default-construct it; a runtime
// mem::Allocator is an explicit template-level opt-in for quota/stats/custom backing.
//
// Build: see ../README.md. Run: prints a tiny word-count, demonstrating insert/find.

#include <cstdint>
#include <cstdio>

#include "hyle/container/swiss_table.h"

using hyle::container::FlatHashMap;

int main() {
  // A word -> count map. Keyed by uint64 id here to keep the sample dependency-free
  // (StringView keys work too; see the container guide). The default allocator is
  // the fast SystemAllocator path, so no allocator argument is needed.
  FlatHashMap<std::uint64_t, std::uint64_t> counts;

  // Simulate a stream of token ids.
  const std::uint64_t stream[] = {7, 3, 7, 7, 42, 3, 7, 42, 99};
  for (std::uint64_t id : stream) {
    if (std::uint64_t* c = counts.find(id)) {
      ++*c;  // find returns a pointer into the table -- mutate in place, no re-hash
    } else {
      counts.insert(id, 1);
    }
  }

  // Read back.
  std::printf("token counts (FlatHashMap, %zu distinct):\n", counts.size());
  for (std::uint64_t id : {3ull, 7ull, 42ull, 99ull, 123ull}) {
    std::uint64_t* c = counts.find(id);
    if (c)
      std::printf("  id %3llu -> %llu\n", static_cast<unsigned long long>(id),
                  static_cast<unsigned long long>(*c));
    else
      std::printf("  id %3llu -> (absent)\n", static_cast<unsigned long long>(id));
  }

  // contains() is the bool shorthand.
  std::printf("has 7? %s   has 123? %s\n", counts.contains(7) ? "yes" : "no",
              counts.contains(123) ? "yes" : "no");
  return 0;
}
