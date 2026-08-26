#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"

// Kept in a separate, non-IPO translation unit so a benchmark callback can
// make both its linked and unlinked node-storage phases observable without an
// O(N) verification scan. The runner rejects interprocedural optimization and
// independently verifies teardown/unlink work in the final benchmark object.
extern "C" HYLE_NOINLINE std::uintptr_t
hyle_container_intrusive_observe_storage_phase(
    const void *storage, std::size_t byte_count) noexcept {
  return reinterpret_cast<std::uintptr_t>(storage) ^ byte_count;
}
