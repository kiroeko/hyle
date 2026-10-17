#pragma once

#include <cstdint>
#include <optional>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/device.h"
#include "hyle/compute/file_stream.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_ROUTE_CPU_API
#elif defined(HYLE_COMPUTE_ROUTE_CPU_BUILD)
#define HYLE_COMPUTE_ROUTE_CPU_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_ROUTE_CPU_API HYLE_IMPORT
#endif

namespace hyle::compute::cpu {

enum class RuntimeThreadBackend : std::uint8_t {
  kUnavailable,
  kOpenMP,
  kNative,
};

// Facts about the shared ATen CPU runtime, not live OS-thread counts or an
// aggregate bound on concurrent callers, BLAS, executor or ggml workers.
struct RuntimeThreadInfo final {
  RuntimeThreadBackend backend = RuntimeThreadBackend::kUnavailable;
  std::optional<std::uint32_t> configured_intra_op_threads{};
  std::optional<std::uint32_t> intra_op_limit{};
  std::optional<std::uint32_t> inter_op_limit{};
  bool configuration_open = false;
  bool usage_started = false;
};

// Explicit startup configuration before any related CPU Context attempt or
// external ATen use. The setting is shared by the resolved ATen runtime;
// the admission guard belongs to this loaded HYLE native implementation.
// Applications must coordinate foreign users and separately embedded HYLE
// copies. Context destruction never restores or reopens this setting.
// This CPU factory freezes admission before outer metadata allocation; expert
// Context factories freeze it when the relevant provider callback is entered.
// A successful same-value request is idempotent only while the observed limit
// still matches. A failed setter can have partial foreign effects: its window
// stays closed, with no rollback or automatic retry. Zero or >INT_MAX is invalid.
// This controls intra-op settings only; simultaneous callers can have multiple
// OpenMP teams. It is not a per-Context policy or a process-wide thread quota.
HYLE_COMPUTE_ROUTE_CPU_API Status
configure_runtime_threads(std::uint32_t intra_op_threads) noexcept;

// Cold observation outside a parallel region. Before configuration/first use,
// limits remain unknown and the runtime is not initialized by this query.
// Afterwards limits describe the current caller; foreign drift is reported,
// never repaired by a setter. An unavailable runtime reports unknown limits.
HYLE_COMPUTE_ROUTE_CPU_API Result<RuntimeThreadInfo>
runtime_thread_info() noexcept;

struct DefaultContextOptions final {
  // Both backings are borrowed through all derived artifacts and invocations.
  mem::AllocatorRef metadata_allocator{};
  // Empty uses the caller thread. Native/oneDNN may split work through this
  // executor; ggml dispatches one complete program through it.
  ParallelExecutorRef executor{};
  // ggml's independent native budget, including the submitting thread. A
  // value above one explicitly prepares persistent workers per invocation.
  // The borrowed executor's concurrency never implicitly creates such a pool.
  std::uint32_t native_threads = 1;
};

// CPU Direct Compute with fixed implementation rules. Prepare
// binds eligible mature oneDNN/ggml programs or the existing Native fallback.
HYLE_COMPUTE_ROUTE_CPU_API Result<Context>
create_context(DefaultContextOptions options = {}) noexcept;

// Borrows the original Context/Queue and IoContext runtime. Ordinary CPU stage
// storage remains unpinned; this factory does not select a math Provider.
HYLE_COMPUTE_ROUTE_CPU_API Result<FileResourcePipeline> create_file_pipeline(
    io::IoContext &, const Context &, Queue &, mem::MemoryResourceRef staging,
    std::span<const CopyResourceSpec> targets, FileStreamOptions options = {}) noexcept;

} // namespace hyle::compute::cpu
