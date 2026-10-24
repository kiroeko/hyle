#pragma once

#include "dspark_original_role_source.h"

// Metadata from the fixed original S03 GGUF header, not payload hashes.
namespace dspark_bfloat_sidecar_roles {
namespace source=dspark_original_role_source;
inline constexpr std::size_t kStaticRoles=72,kOriginalRoles=81,kStaticBytes=1041922972;
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kFile="dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf";
inline constexpr std::string_view kCachedHeaderTableSha256="c93fa628f656654ba421e1d7d161c3edb754646e9da634b3f0cad592bb0d91ed";
inline constexpr std::string_view kOriginalHeaderSha256="0a530d8e43675e4d83695d2a9397c9179d8a12100ffebc4b55df6a932d0cd185";
inline constexpr std::string_view kStaticPlanSha256="2e7eb0a9fc9c19995d41d2d41c93dd928c082a3e9e5b756aaf9c26132310e428";
inline constexpr std::string_view kExpertSourcePlanSha256="e3cfe35aec5870034d0b384553fd90b242806e63fedab435d18c16969324984f";
inline constexpr std::array<source::Spec,kStaticRoles> kStaticSpecs{
  source::Spec{"blk.0.attn_sinks.weight",source::c::DataType::kFloat32,64,0},
  source::Spec{"blk.0.attn_kv_a_norm.weight",source::c::DataType::kFloat32,512,0},
  source::Spec{"blk.0.attn_q_a_norm.weight",source::c::DataType::kFloat32,1024,0},
  source::Spec{"blk.0.attn_kv.weight",source::c::DataType::kBFloat16,512,4096},
  source::Spec{"blk.0.attn_output_a.weight",source::c::DataType::kBFloat16,8192,4096},
  source::Spec{"blk.0.attn_output_b.weight",source::c::DataType::kBFloat16,4096,8192},
  source::Spec{"blk.0.attn_q_a.weight",source::c::DataType::kBFloat16,1024,4096},
  source::Spec{"blk.0.attn_q_b.weight",source::c::DataType::kBFloat16,32768,1024},
  source::Spec{"blk.0.attn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"blk.0.exp_probs_b.bias",source::c::DataType::kFloat32,256,0},
  source::Spec{"blk.0.ffn_gate_inp.weight",source::c::DataType::kBFloat16,256,4096},
  source::Spec{"blk.0.ffn_gate_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.0.ffn_down_shexp.weight",source::c::DataType::kBFloat16,4096,2048},
  source::Spec{"blk.0.ffn_up_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.0.ffn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"blk.0.hc_attn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.0.hc_attn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.0.hc_attn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"blk.0.hc_ffn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.0.hc_ffn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.0.hc_ffn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"enc.output_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"fc.weight",source::c::DataType::kBFloat16,4096,12288},
  source::Spec{"blk.1.attn_sinks.weight",source::c::DataType::kFloat32,64,0},
  source::Spec{"blk.1.attn_kv_a_norm.weight",source::c::DataType::kFloat32,512,0},
  source::Spec{"blk.1.attn_q_a_norm.weight",source::c::DataType::kFloat32,1024,0},
  source::Spec{"blk.1.attn_kv.weight",source::c::DataType::kBFloat16,512,4096},
  source::Spec{"blk.1.attn_output_a.weight",source::c::DataType::kBFloat16,8192,4096},
  source::Spec{"blk.1.attn_output_b.weight",source::c::DataType::kBFloat16,4096,8192},
  source::Spec{"blk.1.attn_q_a.weight",source::c::DataType::kBFloat16,1024,4096},
  source::Spec{"blk.1.attn_q_b.weight",source::c::DataType::kBFloat16,32768,1024},
  source::Spec{"blk.1.attn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"blk.1.exp_probs_b.bias",source::c::DataType::kFloat32,256,0},
  source::Spec{"blk.1.ffn_gate_inp.weight",source::c::DataType::kBFloat16,256,4096},
  source::Spec{"blk.1.ffn_gate_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.1.ffn_down_shexp.weight",source::c::DataType::kBFloat16,4096,2048},
  source::Spec{"blk.1.ffn_up_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.1.ffn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"blk.1.hc_attn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.1.hc_attn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.1.hc_attn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"blk.1.hc_ffn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.1.hc_ffn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.1.hc_ffn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"blk.2.attn_sinks.weight",source::c::DataType::kFloat32,64,0},
  source::Spec{"blk.2.attn_kv_a_norm.weight",source::c::DataType::kFloat32,512,0},
  source::Spec{"blk.2.attn_q_a_norm.weight",source::c::DataType::kFloat32,1024,0},
  source::Spec{"blk.2.attn_kv.weight",source::c::DataType::kBFloat16,512,4096},
  source::Spec{"blk.2.attn_output_a.weight",source::c::DataType::kBFloat16,8192,4096},
  source::Spec{"blk.2.attn_output_b.weight",source::c::DataType::kBFloat16,4096,8192},
  source::Spec{"blk.2.attn_q_a.weight",source::c::DataType::kBFloat16,1024,4096},
  source::Spec{"blk.2.attn_q_b.weight",source::c::DataType::kBFloat16,32768,1024},
  source::Spec{"blk.2.attn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"conf_proj.weight",source::c::DataType::kBFloat16,1,4352},
  source::Spec{"blk.2.exp_probs_b.bias",source::c::DataType::kFloat32,256,0},
  source::Spec{"blk.2.ffn_gate_inp.weight",source::c::DataType::kBFloat16,256,4096},
  source::Spec{"blk.2.ffn_gate_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.2.ffn_down_shexp.weight",source::c::DataType::kBFloat16,4096,2048},
  source::Spec{"blk.2.ffn_up_shexp.weight",source::c::DataType::kBFloat16,2048,4096},
  source::Spec{"blk.2.ffn_norm.weight",source::c::DataType::kFloat32,4096,0},
  source::Spec{"blk.2.hc_attn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.2.hc_attn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.2.hc_attn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"blk.2.hc_ffn_base.weight",source::c::DataType::kFloat32,24,0},
  source::Spec{"blk.2.hc_ffn_fn.weight",source::c::DataType::kFloat32,24,16384},
  source::Spec{"blk.2.hc_ffn_scale.weight",source::c::DataType::kFloat32,3,0},
  source::Spec{"output_hc_base.weight",source::c::DataType::kFloat32,4,0},
  source::Spec{"output_hc_fn.weight",source::c::DataType::kFloat32,4,16384},
  source::Spec{"output_hc_scale.weight",source::c::DataType::kFloat32,1,0},
  source::Spec{"markov_w1.weight",source::c::DataType::kBFloat16,129280,256},
  source::Spec{"markov_w2.weight",source::c::DataType::kBFloat16,129280,256},
  source::Spec{"output_norm.weight",source::c::DataType::kFloat32,4096,0},
};
} // namespace dspark_bfloat_sidecar_roles
