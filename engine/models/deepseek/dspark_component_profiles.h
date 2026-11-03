#pragma once
#include "hyle/compute/workload.h"

namespace dspark_component_profiles {
inline constexpr hyle::compute::ModelArtifactProfileDescription pro_profile{
  .model_publisher="deepseek-ai",
  .model_repository="deepseek-ai/DeepSeek-V4-Pro-0813",
  .model_family="deepseek_v4",
  .model_name="DeepSeek-V4-Pro-0813",
  .model_revision="72e1d3230f6c080a530b0a1d46f8eb4602340597",
  .configuration_digest="9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
  .artifact_repository="deepseek-ai/DeepSeek-V4-Pro-0813",
  .artifact_revision="72e1d3230f6c080a530b0a1d46f8eb4602340597",
  .variant="official-fp4-fp8",
  .manifest_digest="8953e03efe9c1f85191e8c874f1fb2da0abd53374fcb8ce46147556bdae1550a",
  .format_family="safetensors",
};
inline constexpr hyle::compute::ModelArtifactProfileDescription flash_profile{
  .model_publisher="deepseek-ai",
  .model_repository="deepseek-ai/DeepSeek-V4-Flash-0731",
  .model_family="deepseek_v4",
  .model_name="DeepSeek-V4-Flash-0731",
  .model_revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
  .configuration_digest="6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
  .artifact_repository="deepseek-ai/DeepSeek-V4-Flash-0731",
  .artifact_revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
  .variant="official-fp4-fp8",
  .manifest_digest="acd54480d2c4921a990f9e0636b1b29b2455228feb27bddf6ae9ca7d84d40914",
  .format_family="safetensors",
};
}
