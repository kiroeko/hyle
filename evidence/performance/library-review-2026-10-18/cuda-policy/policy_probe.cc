#include "src/compute_route_cuda/default_policy.h"
#include "src/compute/provider_internal.h"
#include <array>
#include <cstdio>
namespace c = hyle::compute;
namespace p = c::cuda::internal;
namespace d = c::detail;
static hyle::Status propose(void *reject, const d::ProviderOperationRequest &,
    std::span<d::ResourceBindingRequirement>, std::span<d::ResourceBindingRequirement>,
    d::ProviderOperationProposal *proposal) noexcept {
  if (reject) return hyle::Status::Unimplemented("original admission rejected");
  *proposal = {}; return hyle::Status::Ok();
}
int main() {
  static const d::ProviderVTable table{.propose=&propose};
  const auto provider=d::ProviderAccess::make(&table, 0);
  unsigned checks=0;
  for (const auto scope : {p::DefaultScope::kDirect,p::DefaultScope::kGraph}) {
    const auto offset=scope==p::DefaultScope::kGraph ? 200u : 0u;
    for (const auto type : {c::DataType::kBFloat16,c::DataType::kFloat16,c::DataType::kFloat32}) {
      const auto desc=c::TensorDesc::dense(type,std::array<std::int64_t,4>{1,3,4,64}).value();
      const std::array inputs{desc,desc,desc}; const std::array outputs{desc};
      const d::ProviderOperationRequest request{.operation=c::Operation(c::ScaledDotProductAttention{}),.inputs=inputs,.outputs=outputs};
      for (const auto family : {p::DefaultProvider::kFlashInfer,p::DefaultProvider::kTorch,p::DefaultProvider::kGgml}) {
        const auto configured=p::with_defaults(provider,family,scope);
        d::ProviderOperationProposal facts;
        const auto status=d::ProviderAccess::vtable(configured)->propose(nullptr,request,{},{},&facts);
        const auto expected=family==p::DefaultProvider::kFlashInfer ? 100u : family==p::DefaultProvider::kGgml ? 200u : type==c::DataType::kBFloat16 ? 150u : 350u;
        if (!status.ok() || facts.preference!=expected+offset) return 1;
        int rejected=1;
        if (!d::ProviderAccess::vtable(configured)->propose(&rejected,request,{},{},&facts).IsUnimplemented()) return 2;
        checks+=2;
      }
      const d::ProviderOperationRequest unary{.operation=c::Operation(c::Unary{c::UnaryKind::kNeg}),.inputs=std::span(inputs).first(1),.outputs=outputs};
      const auto torch=p::with_defaults(provider,p::DefaultProvider::kTorch,scope);
      d::ProviderOperationProposal facts;
      if (!d::ProviderAccess::vtable(torch)->propose(nullptr,unary,{},{},&facts).ok() || facts.preference!=350u+offset) return 3;
      ++checks;
    }
  }
  std::printf("CUDA default policy: %u checks passed (no GPU execution)\n",checks);
}
