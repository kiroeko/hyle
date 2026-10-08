#include "best_of.h"

#include <algorithm>
#include <array>
#include <new>
#include <string_view>

#include "../compute/provider_internal.h"
#include "../compute_backend_cuda/native_context.h"
#include "../compute_provider_cuda/qualified_prepare.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"

#if HYLE_OS_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hyle::compute::cuda::internal {
namespace {
enum class Family : std::uint8_t { kUnary, kBinary, kRms, kLayer, kAffineLayer };
struct Rule final {
  Family family;
  unsigned variant;
  DataType type;
  std::int64_t rows, width;
};
using U = UnaryKind;
using B = BinaryKind;
using D = DataType;
constexpr std::array kRules{
    Rule{Family::kUnary, unsigned(U::kAbs), D::kFloat16, 1, 2048},
    Rule{Family::kUnary, unsigned(U::kNeg), D::kFloat16, 32, 4096},
    Rule{Family::kUnary, unsigned(U::kNeg), D::kBFloat16, 32, 4096},
    Rule{Family::kUnary, unsigned(U::kExp), D::kBFloat16, 32, 4096},
    Rule{Family::kUnary, unsigned(U::kLog), D::kFloat16, 1, 2048},
    Rule{Family::kUnary, unsigned(U::kLog), D::kBFloat16, 1, 2048},
    Rule{Family::kUnary, unsigned(U::kSqrt), D::kBFloat16, 1, 2048},
    Rule{Family::kUnary, unsigned(U::kSqrt), D::kBFloat16, 32, 4096},
    Rule{Family::kUnary, unsigned(U::kRsqrt), D::kBFloat16, 32, 4096},
    Rule{Family::kUnary, unsigned(U::kSoftplus), D::kFloat16, 1, 2048},
    Rule{Family::kUnary, unsigned(U::kSoftplus), D::kBFloat16, 32, 4096},
    Rule{Family::kBinary, unsigned(B::kAdd), D::kFloat16, 1, 2048},
    Rule{Family::kBinary, unsigned(B::kSubtract), D::kFloat16, 32, 4096},
    Rule{Family::kBinary, unsigned(B::kSubtract), D::kBFloat16, 1, 2048},
    Rule{Family::kBinary, unsigned(B::kMultiply), D::kFloat16, 32, 4096},
    Rule{Family::kBinary, unsigned(B::kMultiply), D::kBFloat16, 1, 2048},
    Rule{Family::kBinary, unsigned(B::kMultiply), D::kBFloat16, 32, 4096},
    Rule{Family::kBinary, unsigned(B::kDivide), D::kBFloat16, 32, 4096},
    Rule{Family::kRms, 0, D::kFloat16, 32, 4096},
    Rule{Family::kLayer, 0, D::kFloat32, 1, 2048},
    Rule{Family::kLayer, 0, D::kFloat32, 32, 4096},
    Rule{Family::kAffineLayer, 0, D::kFloat32, 32, 4096},
    Rule{Family::kAffineLayer, 0, D::kBFloat16, 1, 2048},
};

bool matches(const Rule &rule, Operation operation,
             std::span<const TensorDesc> in,
             std::span<const TensorDesc> out) noexcept {
  if (in.empty() || out.size() != 1 || !in[0].is_plain() ||
      !in[0].is_dense() || in[0].data_type() != rule.type ||
      in[0].rank() != 2 || in[0].dimensions()[0] != rule.rows ||
      in[0].dimensions()[1] != rule.width || out[0] != in[0])
    return false;
  switch (rule.family) {
  case Family::kUnary: {
    const auto *op = operation.get_if<Unary>();
    return op && in.size() == 1 && unsigned(op->kind) == rule.variant;
  }
  case Family::kBinary: {
    const auto *op = operation.get_if<Binary>();
    return op && in.size() == 2 && in[1] == in[0] &&
           unsigned(op->kind) == rule.variant;
  }
  case Family::kRms: {
    const auto *op = operation.get_if<RmsNorm>();
    return op && in.size() == 1 && op->epsilon == 1.e-6F &&
           op->accumulation_type == D::kFloat32;
  }
  case Family::kLayer:
  case Family::kAffineLayer: {
    const auto *op = operation.get_if<LayerNorm>();
    if (!op || op->epsilon != 1.e-6F || op->accumulation_type != D::kFloat32)
      return false;
    if (rule.family == Family::kLayer)
      return in.size() == 1;
    return in.size() == 3 && in[1].is_plain() && in[1].is_dense() &&
           in[1].data_type() == rule.type && in[1].rank() == 1 &&
           in[1].dimensions()[0] == rule.width && in[2] == in[1];
  }
  }
  return false;
}

struct State final {
  void *original;
  bool environment;
};
struct Tables final {
  const compute::detail::ProviderVTable *original;
  compute::detail::ProviderVTable configured;
};
template <DefaultScope S> const Tables &tables(const Provider &provider = {}) noexcept;

template <DefaultScope S>
Status create_context(const Provider &provider, Device device, void *backend,
                      mem::AllocatorRef allocator, void **output) noexcept {
  if (!output)
    return Status::InvalidArgument("CUDA winner context output is null");
  *output = nullptr;
  const auto *table = tables<S>().original;
  void *original = nullptr;
  HYLE_RETURN_IF_ERROR(table->create_context(provider, device, backend, allocator, &original));
  void *storage = allocator.allocate(sizeof(State), alignof(State));
  if (!storage) {
    table->destroy_context(original, allocator);
    return Status::ResourceExhausted("CUDA winner context allocation failed");
  }
  const auto *environment = detail::borrow_environment(backend);
  *output = ::new (storage) State{original,
      environment && qualified_environment(*environment)};
  return Status::Ok();
}

template <DefaultScope S>
void destroy_context(void *context, mem::AllocatorRef allocator) noexcept {
  auto *state = static_cast<State *>(context);
  if (!state)
    return;
  tables<S>().original->destroy_context(state->original, allocator);
  state->~State();
  allocator.deallocate(state, sizeof(State), alignof(State));
}

template <DefaultScope S>
Status propose(void *context, const compute::detail::ProviderOperationRequest &request,
               std::span<compute::detail::ResourceBindingRequirement> in,
               std::span<compute::detail::ResourceBindingRequirement> out,
               compute::detail::ProviderOperationProposal *proposal) noexcept {
  const auto *state = static_cast<State *>(context);
  const auto policy = qualified_policy(request.operation, request.inputs, request.outputs);
  if (!state->environment || !request.shape.exact() || !policy.valid() ||
      !qualified_caller())
    return Status::Unimplemented("CUDA measured rule does not match this request");
  HYLE_RETURN_IF_ERROR(tables<S>().original->propose(state->original, request, in, out, proposal));
  // A separate ordinary instance retains managed/low-alignment capability.
  for (auto bindings : {in, out})
    for (auto &binding : bindings) {
      binding.forbidden_properties |= mem::MemoryProperty::kManaged |
                                      mem::MemoryProperty::kHostAccessible;
      binding.minimum_alignment = (std::max)(binding.minimum_alignment, std::size_t{256});
    }
  proposal->policy = policy;
  // An isolated leaf WIN does not undercut the original complete region (200).
  proposal->preference = S == DefaultScope::kGraph ? 250 : 50;
  return Status::Ok();
}

template <DefaultScope S>
ProviderPrepareOutcome prepare(void *context,
    const compute::detail::ProviderPrepareRequest &request,
    mem::AllocatorRef allocator, compute::detail::ProviderPrepared *output) noexcept {
  return cuda_native::internal::prepare_qualified(
      static_cast<State *>(context)->original, request, allocator, output);
}
template <DefaultScope S> Status drain_context(void *context) noexcept {
  return tables<S>().original->drain_context(static_cast<State *>(context)->original);
}
template <DefaultScope S> const Tables &tables(const Provider &provider) noexcept {
  static const Tables value = [&]() noexcept {
    const auto *original = compute::detail::ProviderAccess::vtable(provider);
    auto configured = *original;
    configured.create_context = &create_context<S>;
    configured.destroy_context = &destroy_context<S>;
    configured.propose = &propose<S>;
    configured.prepare = &prepare<S>;
    configured.drain_context = &drain_context<S>;
    return Tables{original, configured};
  }();
  return value;
}
template <DefaultScope S> Provider decorate(const Provider &provider) noexcept {
  const auto &table = tables<S>(provider);
  if (compute::detail::ProviderAccess::vtable(provider) != table.original)
    return {};
  return compute::detail::ProviderAccess::with_vtable(provider, &table.configured);
}
} // namespace

RoutePolicyId qualified_policy(Operation operation, std::span<const TensorDesc> in,
                                std::span<const TensorDesc> out) noexcept {
  for (std::size_t i = 0; i < kRules.size(); ++i)
    if (matches(kRules[i], operation, in, out))
      return {0x48594c4543555752ULL, i + 1, 1};
  return {};
}

bool qualified_recipe() noexcept {
#if defined(HYLE_CUDA_DEFAULT_C2_RECIPE) && HYLE_OS_WINDOWS && HYLE_BUILD_RELEASE && \
    defined(_MSC_FULL_VER) && _MSC_FULL_VER == 195035728
  return true;
#else
  return false;
#endif
}

bool qualified_environment(const Environment &environment) noexcept {
#if HYLE_OS_WINDOWS
  if (!qualified_recipe())
    return false;
  const auto &cpu = platform::cpu_identity();
  std::string_view brand(cpu.brand.data());
  while (!brand.empty() && brand.front() == ' ') brand.remove_prefix(1);
  while (!brand.empty() && brand.back() == ' ') brand.remove_suffix(1);
  DWORD_PTR mask = 0, system = 0;
  return cpu.family == 6 && cpu.model == 183 && cpu.stepping == 1 &&
      std::string_view(cpu.vendor.data()) == "GenuineIntel" &&
      brand == "13th Gen Intel(R) Core(TM) i9-13900K" &&
      ::GetProcessAffinityMask(::GetCurrentProcess(), &mask, &system) && mask == 0x14 &&
      ::GetPriorityClass(::GetCurrentProcess()) == ABOVE_NORMAL_PRIORITY_CLASS &&
      std::string_view(environment.device_name.data()) == "NVIDIA GeForce RTX 4070" &&
      environment.sm_major == 8 && environment.sm_minor == 9 &&
      environment.driver_cuda_version == 13040 && environment.runtime_version == 13040 &&
      environment.build_toolkit_version == 13030 &&
      std::string_view(environment.driver_release.data()) == "617.14";
#else
  (void)environment;
  return false;
#endif
}

bool qualified_caller() noexcept {
#if HYLE_OS_WINDOWS
  unsigned caller = 0;
  const auto count = platform::this_thread::affinity(&caller, 1);
  return count && *count == 1 && caller == 2 &&
      ::GetThreadPriority(::GetCurrentThread()) == THREAD_PRIORITY_NORMAL;
#else
  return false;
#endif
}

Provider with_qualified_defaults(Provider provider, DefaultScope scope) noexcept {
  // Extension callbacks receive the provider context directly. Never decorate
  // a future extension-bearing implementation with this context wrapper.
  const auto *table = compute::detail::ProviderAccess::vtable(provider);
  if (!qualified_recipe() || !table || table->query_extension ||
      table->identity.name != "hyle-cuda" ||
      table->identity.version != "v12-dense-retirement")
    return {};
  return scope == DefaultScope::kGraph ? decorate<DefaultScope::kGraph>(provider)
                                      : decorate<DefaultScope::kDirect>(provider);
}
} // namespace hyle::compute::cuda::internal
