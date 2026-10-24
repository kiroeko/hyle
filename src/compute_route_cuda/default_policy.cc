#include "default_policy.h"

#include <algorithm>

#include "../compute/provider_internal.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::cuda::internal {
namespace {
using Family = DefaultProvider;
using detail::ProviderOperationRequest;
using detail::ProviderVTable;

bool scalar_f32_binary(const ProviderOperationRequest &request) noexcept {
  const auto *binary = request.operation.get_if<Binary>();
  if (!binary || (binary->kind != BinaryKind::kMultiply &&
                  binary->kind != BinaryKind::kDivide) ||
      request.inputs.size() != 2 || request.outputs.size() != 1)
    return false;
  const auto floating_dense = [](const TensorDesc &description) noexcept {
    return description.is_plain() && description.is_dense() &&
           description.data_type() == DataType::kFloat32;
  };
  if (!floating_dense(request.outputs[0]) ||
      !std::ranges::all_of(request.inputs, floating_dense))
    return false;
  const auto scalar = [](const TensorDesc &description) noexcept {
    return description.rank() == 0 ||
           (description.rank() == 1 && description.dimensions()[0] == 1);
  };
  for (std::size_t input = 0; input < 2; ++input)
    if (scalar(request.inputs[input]) &&
        std::ranges::equal(request.inputs[1 - input].dimensions(),
                           request.outputs[0].dimensions()))
      return true;
  return false;
}

// ENGINEERING_DEFAULT: these choices are not measured WIN certificates.
// The actual port/resource admission runs before this cost-based preference.
Family baseline(const ProviderOperationRequest &request) noexcept {
  const auto operation = request.operation;
  // Scalar normalization and scale epilogues retain F32 subnormal/IEEE values.
  // The original native leaf is compiled without ggml's fast-math FTZ mode.
  if (scalar_f32_binary(request))
    return Family::kNative;
  switch (operation.kind()) {
  case OperationKind::kMatMul:
    return request.inputs.size() == 2 && request.inputs[1].is_plain() &&
                   request.inputs[0].rank() == 2 && request.inputs[1].rank() == 2
               ? Family::kCublas : Family::kGgml;
  case OperationKind::kRmsNorm:
    if (request.inputs.size() == 2 &&
        (request.inputs[0].data_type() == DataType::kFloat16 ||
         request.inputs[0].data_type() == DataType::kBFloat16))
      return Family::kSglang;
    return Family::kGgml;
  case OperationKind::kGatedActivation:
    return operation.get_if<GatedActivation>()->inputs == GatedInputKind::kPackedHalves
               ? Family::kSglang : Family::kGgml;
  case OperationKind::kUnary:
    switch (operation.get_if<Unary>()->kind) {
    case UnaryKind::kSilu:
    case UnaryKind::kGelu:
    case UnaryKind::kGeluTanh: return Family::kSglang;
    case UnaryKind::kSoftplus: return Family::kTorch;
    default: return Family::kGgml;
    }
  case OperationKind::kScan:
  case OperationKind::kTopK: return Family::kCccl;
  case OperationKind::kGather:
    return !request.inputs.empty() && request.inputs[0].is_plain()
               ? Family::kCccl : Family::kGgml;
  case OperationKind::kScaledDotProductAttention: return Family::kFlashInfer;
  default: break;
  }
  if (operation.custom_as<model_ops::ArtifactWeightDecode>() ||
      operation.custom_as<model_ops::MlxQuantized>())
    return Family::kTorch;
  if (operation.custom_as<model_ops::SqrtSoftplusRouting>())
    return Family::kSglang;
  // Original DeepSeek admits its own exact custom program before ranking.
  return Family::kGgml;
}

std::uint32_t rank(Family family, Family primary,
                   const ProviderOperationRequest &request) noexcept {
  if (family == Family::kNative)
    return family == primary ? 100 : 1000;
  if (family == primary)
    return 100;
  // Admission stays with the provider: its fused BF16 attention precedes
  // ggml's materialized score graph, while admitted FlashInfer stays first.
  if (family == Family::kTorch &&
      request.operation.kind() == OperationKind::kScaledDotProductAttention &&
      !request.inputs.empty() &&
      request.inputs[0].data_type() == DataType::kBFloat16)
    return 150;
  if (family == Family::kGgml)
    return 200;
  if (family == Family::kCublas || family == Family::kSglang ||
      family == Family::kDeepSeek || family == Family::kCccl)
    return 250;
  return 350;
}

bool admitted_matrix_candidate(Family family,
                               const ProviderOperationRequest &request) noexcept {
  if (family != Family::kCublas || request.inputs.size() != 2 ||
      request.outputs.size() != 1)
    return false;
  const auto plain_matrix = [](const TensorDesc &description) noexcept {
    return description.is_plain() && description.rank() == 2;
  };
  return std::ranges::all_of(request.inputs, plain_matrix) &&
         std::ranges::all_of(request.outputs, plain_matrix);
}

struct Tables final {
  const ProviderVTable *original;
  ProviderVTable configured;
};
template <Family F, DefaultScope S>
const Tables &tables(const Provider &provider = {}) noexcept;

template <Family F, DefaultScope S>
Status propose(void *context, const ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  const auto status = tables<F, S>().original->propose(context, request, inputs, outputs, proposal);
  if (!status.ok())
    return status;
  const auto primary = baseline(request);
  // Semantic, dtype and layout support belongs to the successful provider
  // admission above. Prefer its dedicated matrix implementation without
  // duplicating that support table in the default policy. The Graph offset
  // keeps specific fused regions ahead of these ordinary leaves.
  const auto preference = admitted_matrix_candidate(F, request)
                              ? 99u : rank(F, primary, request);
  proposal->preference = preference +
                         (S == DefaultScope::kGraph ? 200 : 0);
  proposal->policy = {0x48594c4543554444ULL,
      1 + (static_cast<std::uint64_t>(request.operation.kind()) << 16) +
          (static_cast<std::uint64_t>(primary) << 8) + static_cast<std::uint64_t>(F), 2};
  return status;
}

template <Family F, DefaultScope S>
const Tables &tables(const Provider &provider) noexcept {
  static const Tables value = [&]() noexcept {
    const auto *original = detail::ProviderAccess::vtable(provider);
    auto configured = *original;
    configured.propose = &propose<F, S>;
    return Tables{original, configured};
  }();
  return value;
}
template <Family F, DefaultScope S>
Provider decorate(const Provider &provider) noexcept {
  const auto &table = tables<F, S>(provider);
  if (detail::ProviderAccess::vtable(provider) != table.original)
    return provider; // Another loaded implementation keeps its own table.
  return detail::ProviderAccess::with_vtable(provider, &table.configured);
}
template <Family F>
Provider decorate(const Provider &provider, DefaultScope scope) noexcept {
  return scope == DefaultScope::kGraph ? decorate<F, DefaultScope::kGraph>(provider)
                                      : decorate<F, DefaultScope::kDirect>(provider);
}
} // namespace

Provider with_defaults(Provider provider, Family family, DefaultScope scope) noexcept {
  if (!provider.valid())
    return provider;
  switch (family) {
  case Family::kCublas: return decorate<Family::kCublas>(provider, scope);
  case Family::kGgml: return decorate<Family::kGgml>(provider, scope);
  case Family::kTorch: return decorate<Family::kTorch>(provider, scope);
  case Family::kSglang: return decorate<Family::kSglang>(provider, scope);
  case Family::kDeepSeek: return decorate<Family::kDeepSeek>(provider, scope);
  case Family::kCccl: return decorate<Family::kCccl>(provider, scope);
  case Family::kFlashInfer: return decorate<Family::kFlashInfer>(provider, scope);
  case Family::kNative: return decorate<Family::kNative>(provider, scope);
  }
  return provider;
}
} // namespace hyle::compute::cuda::internal
