#include "default_policy.h"

#include "../compute/provider_internal.h"

namespace hyle::compute::cpu::internal {
namespace {
using detail::ProviderOperationProposal;
using detail::ProviderOperationRequest;
using detail::ProviderVTable;
using Family = DefaultProvider;
using Scope = DefaultScope;

// Engineering baselines, not measured WIN labels. Real admission runs first.
Family baseline(Operation operation) noexcept {
  switch (operation.kind()) {
  case OperationKind::kMatMul:
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm:
  case OperationKind::kCopy:
  case OperationKind::kSelect:
    return Family::kOneDnn;
  case OperationKind::kUnary:
    switch (operation.get_if<Unary>()->kind) {
    case UnaryKind::kSigmoid:
    case UnaryKind::kGelu:
    case UnaryKind::kGeluTanh:
    case UnaryKind::kSilu:
      return Family::kAten;
    case UnaryKind::kSoftplus:
      return Family::kOneDnn;
    default:
      return Family::kGgml;
    }
  case OperationKind::kBinary:
    if (const auto kind = operation.get_if<Binary>()->kind;
        kind == BinaryKind::kMinimum || kind == BinaryKind::kMaximum)
      return Family::kAten;
    return Family::kGgml;
  case OperationKind::kGatedActivation:
  case OperationKind::kRoPE:
    return Family::kAten;
  case OperationKind::kSoftmax: {
    const auto &softmax = *operation.get_if<Softmax>();
    return softmax.kind == SoftmaxKind::kLogSoftmax ||
                   softmax.mask == MaskKind::kBoolean
               ? Family::kAten
               : Family::kGgml;
  }
  case OperationKind::kScatter:
    return operation.get_if<Scatter>()->kind == ScatterKind::kAdd
               ? Family::kAten
               : Family::kGgml;
  default:
    return Family::kGgml;
  }
}

Family baseline(const ProviderOperationRequest &request) noexcept {
  // Packed/blocked matrices preserve the original mature codec rather than
  // preferring a plain GEMM merely because both requests say MatMul.
  if (request.operation.kind() == OperationKind::kMatMul &&
      request.inputs.size() > 1 && !request.inputs[1].is_plain())
    return Family::kGgml;
  return baseline(request.operation);
}

std::uint32_t rank(Family family, Family primary) noexcept {
  if (family == primary)
    return 100;
  // Qualified alternatives when the primary cannot meet this exact contract.
  if (primary == Family::kOneDnn)
    return family == Family::kGgml ? 200 : 350;
  if (primary == Family::kGgml)
    return family == Family::kOneDnn ? 200 : 350;
  return family == Family::kGgml ? 200 : 350;
}

struct Tables final {
  const ProviderVTable *original;
  ProviderVTable configured;
};
template <Family F, Scope S>
const Tables &tables(const Provider &provider = {}) noexcept;

template <Family F, Scope S>
Status propose(void *context, const ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               ProviderOperationProposal *proposal) noexcept {
  auto status = tables<F, S>().original->propose(context, request, inputs,
                                                outputs, proposal);
  if (!status.ok() || !proposal->policy.empty())
    return status;
  const auto primary = baseline(request);
  proposal->preference = rank(F, primary) + (S == Scope::kGraph ? 200 : 0);
  proposal->policy = {
      0x48594c4543505544ULL,
      1 + (static_cast<std::uint64_t>(request.operation.kind()) << 16) +
          (static_cast<std::uint64_t>(primary) << 8) +
          static_cast<std::uint64_t>(F),
      1};
  return status;
}

template <Family F, Scope S>
const Tables &tables(const Provider &provider) noexcept {
  static const Tables value = [&]() noexcept {
    const auto *original = detail::ProviderAccess::vtable(provider);
    auto configured = *original;
    configured.propose = &propose<F, S>;
    return Tables{original, configured};
  }();
  return value;
}

template <Family F, Scope S>
Provider decorate(const Provider &provider) noexcept {
  const auto &table = tables<F, S>(provider);
  // A different implementation loaded into this process gets its own original
  // behavior, never a stale callback table from an earlier implementation.
  if (detail::ProviderAccess::vtable(provider) != table.original)
    return provider;
  return detail::ProviderAccess::with_vtable(provider, &table.configured);
}
template <Family F>
Provider decorate(const Provider &provider, Scope scope) noexcept {
  return scope == Scope::kDirect ? decorate<F, Scope::kDirect>(provider)
                                : decorate<F, Scope::kGraph>(provider);
}
} // namespace

Provider with_defaults(Provider provider, Family family, Scope scope) noexcept {
  if (!provider.valid())
    return provider;
  switch (family) {
  case Family::kOneDnn:
    return decorate<Family::kOneDnn>(provider, scope);
  case Family::kGgml:
    return decorate<Family::kGgml>(provider, scope);
  case Family::kAten:
    return decorate<Family::kAten>(provider, scope);
  }
  return provider;
}
} // namespace hyle::compute::cpu::internal
