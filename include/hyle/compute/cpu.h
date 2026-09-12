#pragma once

#include <array>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"

namespace hyle::compute::cpu {

struct DefaultContextOptions final {
  mem::AllocatorRef metadata_allocator{};
  ParallelExecutorRef executor{};
};

// Natural CPU route: explicit CPU backend plus the HYLE native provider. More
// advanced users construct ContextOptions with their own ordered provider span
// (for example {onednn, native}) and call Context::create(cpu::device(), ...).
inline Result<Context>
create_context(DefaultContextOptions options = {}) noexcept {
  const std::array providers{native::provider({.executor = options.executor})};
  return Context::create(
      device(), ContextOptions{.metadata_allocator = options.metadata_allocator,
                               .providers = providers});
}

} // namespace hyle::compute::cpu
