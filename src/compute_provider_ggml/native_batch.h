#pragma once
#include "hyle/compute/detail/batch_provider.h"

namespace hyle::compute::ggml_cpu::internal {
const detail::BatchProviderVTable *native_batch_table() noexcept;
std::size_t native_invocation_bytes(void *, void *) noexcept;
bool native_invocation_reads_kv(void *) noexcept;
}
