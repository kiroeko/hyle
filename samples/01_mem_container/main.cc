// Sample 01 -- mem + container: the two patterns every HYLE consumer touches first.
//
//  (1) EXPLICIT ALLOCATOR INJECTION. HYLE has no hidden global heap -- every allocating
//      API takes an `Allocator` by value. system_allocator() is the process default
//      (mimalloc-backed). You can pass an arena/pool instead to control lifetime.
//  (2) SwissTable -- HYLE's flat hashmap (control-byte + SIMD probe), the absl-class
//      replacement for std::unordered_map. It takes the allocator explicitly.
//
// Build: see ../README.md. Run: prints a tiny word-count, demonstrating insert/find.

#include <cstdint>
#include <cstdio>

#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::SwissTable;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

int main() {
  // The allocator is a value handle ({fn,state}) -- pass it wherever HYLE allocates.
  Allocator alloc = system_allocator();

  // A word -> count map. Keyed by uint64 id here to keep the sample dependency-free
  // (string_ref keys work too; see usage-guide). Allocator injected at construction.
  SwissTable<std::uint64_t, std::uint64_t> counts(alloc);

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
  std::printf("token counts (SwissTable, %zu distinct):\n", counts.size());
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
