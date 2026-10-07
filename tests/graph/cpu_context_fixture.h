#pragma once

#include <array>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace hyle::graph::test {

// Graph mechanism tests retain Native RMS/Copy resource guarantees while
// retired math uses an actual mature carrier. This fixture grants no WIN.
inline Result<compute::Context> cpu_context(
    compute::test::NativeContextOptions options = {}) noexcept {
  const std::array providers{
      compute::ProviderBinding{
          compute::native::provider({.executor = options.executor}), 100},
      compute::ProviderBinding{compute::ggml_cpu::provider(), 200},
      compute::ProviderBinding{compute::aten_cpu::provider(), 300}};
  return compute::Context::create(
      compute::cpu::device(),
      {.metadata_allocator = options.metadata_allocator,
       .providers = providers});
}

} // namespace hyle::graph::test
