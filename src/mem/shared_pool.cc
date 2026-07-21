// Out-of-line home for SharedPoolCached's process-global TLS-registry lock.
//
// The pools are otherwise header-only, but the cached pool's TLS-freelist registry
// lock must be ONE instance across the whole process (and across DLL boundaries): it
// serializes the thread-exit teardown of a ThreadCache against the destruction of the
// SharedPoolCached that node points at. A header-inline static would give each DLL its
// own lock -- two parties racing on the SAME node through DIFFERENT locks is no
// synchronization at all. Exporting lock/unlock from hyle_mem makes every inlined
// consumer serialize on the same mutex. It is a COLD-path lock (attach / detach /
// reset / dtor / thread-exit); the alloc/free hot path never takes it. Magic-static
// init is thread-safe; the function-local static outlives all threads that could use it.

#include "hyle/mem/shared_pool.h"

#include <mutex>

namespace hyle::mem::detail {

namespace {
std::mutex& reg_mutex() noexcept {
  static std::mutex m;
  return m;
}
}  // namespace

void tls_registry_lock() noexcept { reg_mutex().lock(); }
void tls_registry_unlock() noexcept { reg_mutex().unlock(); }

}  // namespace hyle::mem::detail
