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

bool original_q8_sidecar_norm(const ProviderOperationRequest &request) noexcept {
  const auto *norm = request.operation.get_if<RmsNorm>();
  if (!norm || norm->epsilon != 1.0e-6F ||
      norm->accumulation_type != DataType::kFloat32 ||
      request.inputs.size() != 2 || request.outputs.size() != 1)
    return false;
  for (const auto &input : request.inputs)
    if (!input.is_plain() || !input.is_dense() ||
        input.data_type() != DataType::kFloat32)
      return false;
  const auto artifact = request.workload.profile.description();
  return artifact.format_family == "gguf" &&
         artifact.model_repository == "deepseek-ai/DeepSeek-V4-Flash-0731" &&
         artifact.model_name == "DeepSeek-V4-Flash-0731" &&
         artifact.model_revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062" &&
         artifact.artifact_repository == "unsloth/DeepSeek-V4-Flash-0731-GGUF" &&
         artifact.artifact_revision == "fbbb5b93fb787c21338159b0af3318bb3f4d9768" &&
         artifact.variant == "sidecar:dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" &&
         artifact.manifest_digest ==
             "b77793866ec63e166212f588ce5b74d9edb464bdd0787f54bc6cab2779ad8eb1";
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
  // Keep the original model's quantization-sensitive boundary. This is a
  // cold engineering rule; submission retains the prepared provider path.
  const bool original_norm = original_q8_sidecar_norm(request);
  const auto primary = original_norm ? Family::kGgml : baseline(request);
  proposal->preference = rank(F, primary) + (S == Scope::kGraph ? 200 : 0);
  proposal->policy = {
      0x48594c4543505544ULL,
      1 + (static_cast<std::uint64_t>(request.operation.kind()) << 16) +
          (static_cast<std::uint64_t>(primary) << 8) +
          static_cast<std::uint64_t>(F) +
          (original_norm ? (std::uint64_t{1} << 32) : 0),
      original_norm ? 2U : 1U};
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
