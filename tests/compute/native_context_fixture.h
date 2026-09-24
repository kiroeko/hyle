#pragma once

#include <array>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"

namespace hyle::compute::test {

struct NativeContextOptions final {
  mem::AllocatorRef metadata_allocator{};
  ParallelExecutorRef executor{};
};

// Implementation-specific tests keep a stable Native oracle and resource cost.
// Tests of the ordinary route use compute/cpu.h or graph/cpu.h directly.
inline Result<Context>
native_context(NativeContextOptions options = {}) noexcept {
  const std::array providers{hyle::compute::ProviderBinding{
      native::provider({.executor = options.executor}), 100}};
  return Context::create(cpu::device(),
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = providers});
}

} // namespace hyle::compute::test
