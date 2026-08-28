#include <memory_resource>

#include <oneapi/tbb/scalable_allocator.h>

namespace {

// Keep this provider outside owner_consumer_bench.cc.  With interprocedural
// optimization disabled, the caller sees only memory_resource* and therefore
// exercises oneTBB's intended runtime-erased PMR route instead of a compiler-
// devirtualized scalable_resource_impl.
std::pmr::memory_resource* tbb_runtime_resource_impl() noexcept {
  return oneapi::tbb::scalable_memory_resource();
}

}  // namespace

std::pmr::memory_resource* tbb_runtime_resource() noexcept {
  return tbb_runtime_resource_impl();
}
