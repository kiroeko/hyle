#pragma once
#include "glm_mlx_mtp_fragment.h"
#include "glm_original_moe_roles.h"
#include "model_plane_resident_weights.h"

namespace glm_mlx_runtime_source {
namespace c=hyle::compute;namespace roles=glm_mlx_original_roles;
namespace weights=model_plane_resident_weights;namespace model=glm_mlx_mtp_fragment;
inline constexpr c::ModelArtifactProfileDescription kProfile{
    .model_publisher="mlx-community",.model_repository=roles::kRepository,.model_family="glm5_next_text",
    .model_name="GLM-5.3-Flash-4bit",.model_revision=roles::kRevision,
    .configuration_digest="ff382668c5c53c774c718ed662bf336b08a41c01f515bde9a8a6c7948377f2eb",
    .artifact_repository=roles::kRepository,.artifact_revision=roles::kRevision,
    .variant="affine-group64-mixed4-5-6-F16-metadata",
    .manifest_digest="cacc0f91d635087c63adf6136e51935e251f12b810fbba2f77551c7aaececa1a",
    .format_family="safetensors"};
struct Source final {
  roles::Bound parameters;
  std::array<roles::Readonly,2> prepared;
  std::array<roles::Readonly,3> convolutions;
  c::LeasePool prepared_pool;
  std::unique_ptr<weights::CpuSupply> cpu_supply;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  std::unique_ptr<weights::CudaSupply> cuda_supply;
#endif
  std::function<void(std::shared_ptr<roles::Data>,std::span<const std::int32_t>)> before_upload;
  bool full_checked=false;
  std::size_t cold_submissions=0,completed_episodes=0,head_calls=0,pending_checks=0,projection_values=0;
  std::vector<glm_original_moe_roles::Selection> selections;
  std::vector<std::size_t> union_sizes;
  void ensure(std::span<const std::int32_t> ids)const {
    const auto missing=parameters.source->missing(ids);if(missing.empty())return;
    if(!before_upload)throw std::runtime_error("MLX original selected experts require explicit external frontier supply");
    before_upload(parameters.source,ids);
    if(!parameters.source->missing(ids).empty())throw std::runtime_error("MLX source hook did not attach every actual ID");
  }
};
inline weights::Layout layout() {
  std::array<c::TensorDesc,9> planes;
  for(std::size_t i=0;i<planes.size();++i)planes[i]=roles::expert_description(roles::Leg(i/3),roles::Plane(i%3));
  return roles::raw::take(weights::Layout::create(planes,roles::kDomain,roles::kCapacity),"MLX original9-plane resident layout");
}
} // namespace glm_mlx_runtime_source
