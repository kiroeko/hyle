#include "../../engine/inference/backend.h"
#include "../../src/compute/context_state.h"
#include "../../src/compute/provider_internal.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
std::size_t checks = 0;
void require(bool value) {
  if (!value) throw std::runtime_error("device batch capability contract failed");
  ++checks;
}
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}

// Replace only the selected provider's test vtable; its original Context,
// state and ownership stay intact, and are restored before the Backend dies.
struct IncompleteBatchProvider final {
  using Provider = hyle::compute::Provider;
  using Slot = hyle::compute::detail::ContextState::ProviderSlot;
  using VTable = hyle::compute::detail::ProviderVTable;
  using BatchTable = hyle::compute::detail::BatchProviderVTable;
  Slot &slot;
  Provider original;
  const VTable *base;
  VTable replacement;
  BatchTable batch{};
  unsigned missing;
  static inline IncompleteBatchProvider *active = nullptr;

  IncompleteBatchProvider(Slot &selected, unsigned field)
      : slot(selected), original(selected.provider),
        base(hyle::compute::detail::ProviderAccess::vtable(original)), replacement(*base), missing(field) {
    active = this;
    replacement.query_extension = [](void *native, hyle::compute::ProviderExtensionId id) noexcept -> const void * {
      auto &self = *active;
      const auto *value = self.base->query_extension(native, id);
      if (!value || id != hyle::compute::detail::kBatchPlanningExtensionId) return value;
      self.batch = *static_cast<const BatchTable *>(value);
      if (self.missing == 0) self.batch.update = nullptr;
      if (self.missing == 1) self.batch.plan = nullptr;
      if (self.missing == 2) self.batch.run = nullptr;
      if (self.missing == 3) { self.batch.uses_caller_workspace = true; self.batch.workspace = nullptr; }
      if (self.missing == 4) { self.batch.uses_caller_workspace = false; self.batch.workspace = nullptr; }
      return &self.batch;
    };
    slot.provider = hyle::compute::detail::ProviderAccess::with_vtable(original, &replacement);
  }
  ~IncompleteBatchProvider() { slot.provider = original; active = nullptr; }
};
}

int main() {
  try {
  namespace c = hyle::compute;
  namespace i = hyle::inference;
  c::BatchCapacity capacity;
  capacity.query_type = capacity.output_type = c::DataType::kBFloat16;
  capacity.key_storage = capacity.value_storage = take(c::CacheStorage::plain(c::DataType::kBFloat16));
  capacity.max_query_rows = 4;
  capacity.max_kv_rows = 32768;
  capacity.max_pages = 2048;
  capacity.max_page_refs = 512;
  capacity.query_heads = 8;
  capacity.kv_heads = 2;
  capacity.page_size = 64;
  i::Backend empty;
  require(!empty.supports_device_lengths_batch(capacity));
  const auto device = c::cuda::device(0);
  if (!device) {
    const auto &error = device.error();
    if (error.IsUnavailable() || error.IsNotFound() || error.IsOutOfRange() || error.IsUnimplemented()) {
      std::printf("SKIP: CUDA device unavailable: %.*s\n", int(error.message().size()), error.message().data());
      return 77;
    }
    throw std::runtime_error(std::string(error.message()));
  }
  auto created = i::Backend::create({.primary = i::BackendKind::kCuda});
  if (!created && created.error().IsUnimplemented()) {
    std::printf("SKIP: CUDA device does not support this backend\n");
    return 77;
  }
  auto backend = take(std::move(created));
  for (const auto depth : {64u, 128u, 256u}) {
    capacity.qk_head_dim = capacity.value_head_dim = depth;
    require(take(backend.supports_device_lengths_batch(capacity)));
  }
  capacity.qk_head_dim = capacity.value_head_dim = 64;
  auto *context = take(backend.common_context());
  auto *state = c::detail::ContextAccess::state(*context);
  std::size_t selected = state->provider_count;
  const auto provider_count = state->provider_count;
  for (std::size_t index = 0; index < provider_count; ++index) {
    const auto lengths = c::detail::ProviderExtensionAccess::query(*context, index,
        c::detail::kBatchDeviceLengthsExtensionId);
    const auto planning = c::detail::ProviderExtensionAccess::query(*context, index,
        c::detail::kBatchPlanningExtensionId);
    if (lengths && planning) {
      struct RestoreCount final {
        c::detail::ContextState *state;
        std::size_t count;
        ~RestoreCount() { state->provider_count = count; }
      } restore{state, state->provider_count};
      state->provider_count = index + 1;
      if (take(backend.supports_device_lengths_batch(capacity))) { selected = index; break; }
    }
  }
  require(selected < state->provider_count);
  {
    // Isolate the first compatible provider. Later providers may legitimately
    // supply a fallback, so they must not mask this provider's missing method.
    struct RestoreCount final {
      c::detail::ContextState *state;
      std::size_t count;
      ~RestoreCount() { state->provider_count = count; }
    } restore{state, state->provider_count};
    state->provider_count = selected + 1;
    for (unsigned missing = 0; missing != 4; ++missing) {
      IncompleteBatchProvider incomplete(state->providers()[selected], missing);
      require(!take(backend.supports_device_lengths_batch(capacity)));
    }
    {
      IncompleteBatchProvider provider_owned_workspace(state->providers()[selected], 4);
      require(take(backend.supports_device_lengths_batch(capacity)));
    }
  }
  require(take(backend.supports_device_lengths_batch(capacity)));
  capacity.qk_head_dim = capacity.value_head_dim = 96;
  require(!take(backend.supports_device_lengths_batch(capacity)));
  capacity.qk_head_dim = 256; capacity.value_head_dim = 128;
  require(!take(backend.supports_device_lengths_batch(capacity)));
  capacity.qk_head_dim = 64; capacity.value_head_dim = 128;
  require(!take(backend.supports_device_lengths_batch(capacity)));
  capacity.qk_head_dim = capacity.value_head_dim = 512;
  require(!take(backend.supports_device_lengths_batch(capacity)));
  capacity.qk_head_dim = capacity.value_head_dim = 256;
  const auto host = hyle::mem::system_resource();
  const std::array tight_limits{c::ResourceBudgetLimit{host, 64u << 20},
      c::ResourceBudgetLimit{backend.device(), SIZE_MAX}};
  const auto tight = take(c::ResourceBudget::create({.limits = tight_limits, .allow_unattributed = true}));
  const auto before_tight = take(tight.usage(host));
  auto pressure = take(tight.reserve(host, before_tight.limit_bytes - before_tight.live_bytes));
  const auto pressured = take(tight.usage(host)).live_bytes;
  const auto rejected = backend.supports_device_lengths_batch(capacity, tight);
  require(!rejected && rejected.error().IsResourceExhausted());
  require(take(tight.usage(host)).live_bytes == pressured);
  pressure.reset();
  require(take(backend.supports_device_lengths_batch(capacity, tight)));
  require(take(tight.usage(host)).live_bytes == before_tight.live_bytes);
  const std::array limits{c::ResourceBudgetLimit{host, 64u << 20},
      c::ResourceBudgetLimit{backend.device(), SIZE_MAX}};
  const auto budget = take(c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  const auto before = take(budget.usage(host));
  require(take(backend.supports_device_lengths_batch(capacity, budget)));
  const auto usage = take(budget.usage(host));
  require(usage.live_bytes == before.live_bytes && usage.peak_bytes >= before.peak_bytes);
  std::printf("Device batch capability: %zu checks passed\n", checks);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Device batch capability after %zu checks: %s\n", checks, error.what());
    return 1;
  }
}
