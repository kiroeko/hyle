#include "default_policy.h"

#include "../compute/provider_internal.h"

#include <limits>

namespace hyle::compute::cpu::internal {
namespace {
using detail::ProviderOperationProposal;
using detail::ProviderOperationRequest;
using detail::ProviderVTable;
using Family = DefaultProvider;
using Scope = DefaultScope;

bool flat_f32_exp(const ProviderOperationRequest &request) noexcept {
  const auto *unary = request.operation.get_if<Unary>();
  return unary && unary->kind == UnaryKind::kExp && request.inputs.size() == 1 &&
         request.outputs.size() == 1 && request.inputs[0].is_plain() &&
         request.inputs[0].is_dense() &&
         request.inputs[0].data_type() == DataType::kFloat32 &&
         request.inputs[0] == request.outputs[0];
}

// Selected-byte Gather is an algorithm/format choice. Admission has already
// checked the real index, layout and native storage contracts. Prefer this
// route to whole-bank packing or a narrow -> F32 -> narrow round trip.
RoutePolicyId storage_gather_policy(const ProviderOperationRequest &request) noexcept {
  const auto in = request.inputs, out = request.outputs;
  if (!request.operation.get_if<Gather>() || in.size() != 2 || out.size() != 1 ||
      !in[0].valid() || !in[0].is_plain() || !out[0].is_plain() ||
      !out[0].is_dense() || in[0].data_type() != out[0].data_type() ||
      data_type_size(in[0].data_type()) == 0)
    return {};
  const auto type = in[0].data_type();
  if (in[0].is_dense() && type != DataType::kFloat16 &&
      type != DataType::kBFloat16)
    return {};
  return {0x48594c4553544f52ULL, 1, 1};
}

// Engineering baselines, not measured WIN labels. Real admission runs first.
Family baseline(Operation operation) noexcept {
  switch (operation.kind()) {
  case OperationKind::kMatMul:
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm:
    return Family::kOneDnn;
  case OperationKind::kCopy:
    // Original prepared storage copy avoids primitive/managed replay costs.
    // Actual stride/storage admission still selects a qualified alternative.
    return Family::kGgml;
  case OperationKind::kSelect:
    // Current complete-cost evidence rejects the slower oneDNN byte primitive;
    // ATen's qualified original byte selection preserves all scalar payloads.
    return Family::kAten;
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
    return Family::kAten;
  case OperationKind::kRoPE:
    return Family::kGgml;
  case OperationKind::kSoftmax: {
    const auto &softmax = *operation.get_if<Softmax>();
    return softmax.kind == SoftmaxKind::kLogSoftmax ||
                   softmax.mask == MaskKind::kBoolean
               ? Family::kAten
               : Family::kGgml;
  }
  case OperationKind::kScatter:
    // The checked original ggml indexed program has prepared replay. Preserve
    // it before a managed tensor alternative when both satisfy the contract.
    return Family::kGgml;
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

RoutePolicyId engineering_policy(Operation operation, Family primary,
                                 Family selected) noexcept {
  return {0x48594c4543505544ULL,
          1 + (static_cast<std::uint64_t>(operation.kind()) << 16) +
              (static_cast<std::uint64_t>(primary) << 8) +
              static_cast<std::uint64_t>(selected),
          1};
}

struct Tables final {
  const ProviderVTable *original;
  ProviderVTable configured;
};
template <Family F, Scope S, bool SingleNativeLane>
const Tables &tables(const Provider &provider = {}) noexcept;

template <Family F, Scope S, bool SingleNativeLane>
Status propose(void *context, const ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               ProviderOperationProposal *proposal) noexcept {
  auto status = tables<F, S, SingleNativeLane>().original->propose(context, request, inputs,
                                                outputs, proposal);
  if (!status.ok() || !proposal->policy.empty())
    return status;
  if constexpr (F == Family::kNative) {
    if (request.shape.exact()) {
      const auto policy = storage_gather_policy(request);
      if (policy.valid()) {
        proposal->preference = S == Scope::kDirect ? 75 : 275;
        proposal->policy = policy;
        return status;
      }
    }
    // SingleNativeLane means ggml native_threads==1 and no multi-lane executor;
    // it neither configures nor asserts an ATen global thread limit.
    if constexpr (SingleNativeLane) {
      if (request.shape.exact() && flat_f32_exp(request)) {
        // The admitted kernel is one flat SIMD map for every rank and N.
        // SIMD dispatch owns executable ISA checks; routing adds no CPU brand,
        // affinity, compiler, model shape, value-domain or measured-WIN gate.
        proposal->preference = S == Scope::kDirect ? 50 : 175;
        proposal->policy = engineering_policy(request.operation, F, F);
        return status;
      }
    }
  }
  const auto primary = baseline(request);
  if constexpr (F == Family::kNative) {
    // Preserve actual algorithm priorities such as packed matvec1001.
    // Native's ordinary fallback1000 must not tie a mature leaf at350.
    if (proposal->preference == (std::numeric_limits<std::uint32_t>::max)())
      proposal->preference = 1000;
  } else {
    proposal->preference = rank(F, primary) + (S == Scope::kGraph ? 200 : 0);
  }
  proposal->policy = engineering_policy(request.operation, primary, F);
  return status;
}

template <Family F, Scope S, bool SingleNativeLane>
const Tables &tables(const Provider &provider) noexcept {
  static const Tables value = [&]() noexcept {
    const auto *original = detail::ProviderAccess::vtable(provider);
    auto configured = *original;
    configured.propose = &propose<F, S, SingleNativeLane>;
    return Tables{original, configured};
  }();
  return value;
}

template <Family F, Scope S, bool SingleNativeLane>
Provider decorate(const Provider &provider) noexcept {
  const auto &table = tables<F, S, SingleNativeLane>(provider);
  // A different implementation loaded into this process gets its own original
  // behavior, never a stale callback table from an earlier implementation.
  if (detail::ProviderAccess::vtable(provider) != table.original)
    return provider;
  return detail::ProviderAccess::with_vtable(provider, &table.configured);
}
template <Family F, bool SingleNativeLane>
Provider decorate(const Provider &provider, Scope scope) noexcept {
  return scope == Scope::kDirect ? decorate<F, Scope::kDirect, SingleNativeLane>(provider)
                                : decorate<F, Scope::kGraph, SingleNativeLane>(provider);
}
template <Family F>
Provider decorate(const Provider &provider, Scope scope, bool single_native_lane) noexcept {
  return single_native_lane ? decorate<F, true>(provider, scope)
                       : decorate<F, false>(provider, scope);
}
} // namespace

Provider with_defaults(Provider provider, Family family, Scope scope,
                       bool single_native_lane) noexcept {
  if (!provider.valid())
    return provider;
  switch (family) {
  case Family::kOneDnn:
    return decorate<Family::kOneDnn>(provider, scope, single_native_lane);
  case Family::kGgml:
    return decorate<Family::kGgml>(provider, scope, single_native_lane);
  case Family::kAten:
    return decorate<Family::kAten>(provider, scope, single_native_lane);
  case Family::kNative:
    return decorate<Family::kNative>(provider, scope, single_native_lane);
  }
  return provider;
}
} // namespace hyle::compute::cpu::internal
