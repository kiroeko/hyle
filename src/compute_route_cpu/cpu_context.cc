#include "hyle/compute/cpu.h"
#include "default_policy.h"

#include "../compute_provider_native/torch_cpu_program.h"
#include <array>

#include "hyle/compute/providers/native.h"
#if defined(HYLE_COMPUTE_GGML_CPU)
#include "hyle/compute/providers/ggml_cpu.h"
#endif
#if defined(HYLE_COMPUTE_ONEDNN)
#include "hyle/compute/providers/onednn.h"
#endif

namespace hyle::compute::cpu {

Status configure_runtime_threads(std::uint32_t intra_op_threads) noexcept {
  return native::internal::configure_thread_runtime(intra_op_threads);
}

Result<RuntimeThreadInfo> runtime_thread_info() noexcept {
  auto result = native::internal::thread_runtime_info();
  if (!result)
    return std::unexpected(std::move(result).error());
  RuntimeThreadBackend backend = RuntimeThreadBackend::kUnavailable;
  switch (result->backend) {
  case native::internal::ThreadRuntimeBackend::kUnavailable:
    break;
  case native::internal::ThreadRuntimeBackend::kOpenMP:
    backend = RuntimeThreadBackend::kOpenMP;
    break;
  case native::internal::ThreadRuntimeBackend::kNative:
    backend = RuntimeThreadBackend::kNative;
    break;
  }
  return RuntimeThreadInfo{
      .backend = backend,
      .configured_intra_op_threads = result->configured_intra_op_threads,
      .intra_op_limit = result->intra_op_limit,
      .inter_op_limit = result->inter_op_limit,
      .configuration_open = result->configuration_open,
      .usage_started = result->usage_started};
}

Result<Context> create_context(DefaultContextOptions options) noexcept {
#if defined(HYLE_COMPUTE_GGML_CPU)
  auto ggml = ggml_cpu::provider({.executor = options.executor,
                                 .native_threads = options.native_threads});
#else
  Provider ggml;
#endif
  return internal::create_context(options, internal::DefaultScope::kDirect,
                                   std::move(ggml));
}

Result<Context> internal::create_context(DefaultContextOptions options,
                                         DefaultScope scope, Provider ggml,
                                         ProviderBinding region,
                                         Provider native_provider) noexcept {
  // A failed outer metadata allocation must not reopen the startup contract.
  if (scope == DefaultScope::kDirect)
    native::internal::mark_thread_runtime_used();
  const bool single_native_lane = options.native_threads == 1 &&
                             options.executor.concurrency() <= 1;
  const auto configured = [single_native_lane, scope](Provider provider, DefaultProvider family) {
    return internal::with_defaults(std::move(provider), family,
                                   scope, single_native_lane);
  };
  std::array<ProviderBinding, 5> providers{};
  std::size_t count = 0;
  if (region.provider.valid())
    providers[count++] = std::move(region);
#if defined(HYLE_COMPUTE_ONEDNN)
  providers[count++] =
      {configured(onednn::provider({.executor = options.executor}),
                  DefaultProvider::kOneDnn), 900};
#endif
  if (ggml.valid())
    providers[count++] = {configured(std::move(ggml), DefaultProvider::kGgml), 900};
  auto aten = aten_cpu::provider();
  if (aten.valid())
    providers[count++] = {configured(std::move(aten), DefaultProvider::kAten), 900};
  if (!native_provider.valid())
    native_provider = native::provider({.executor = options.executor});
  providers[count++] = {configured(std::move(native_provider), DefaultProvider::kNative), 1000};
  return Context::create(device(),
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = std::span(providers).first(count)});
}

} // namespace hyle::compute::cpu
