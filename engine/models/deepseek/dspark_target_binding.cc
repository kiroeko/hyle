#include "dspark_target_binding.h"

namespace dspark_target_binding {
Status qualify_identity(const TargetIdentity &target,
                               const SidecarIdentity &sidecar,
                               const SidecarIdentity &expected) {
  if (target.model_repository != kModelRepository || target.model_revision != kModelRevision ||
      target.artifact_repository.empty() || target.artifact_revision.empty() || target.variant.empty() ||
      expected.repository.empty()||expected.revision.empty()||expected.file.empty()||sidecar!=expected)
    return Status::InvalidArgument("DSpark requires the matching formal Flash0731 target and explicitly pinned sidecar identity");
  return Status::Ok();
}
}

namespace dspark_target_binding {
Result<c::TensorDesc> plain(std::span<const std::int64_t> shape) {
  return c::TensorDesc::dense(c::DataType::kFloat32, shape);
}
}

namespace dspark_target_binding {
Status qualify_role(const ReadonlyRole &role, std::string_view name,
                           const c::TensorDesc &description) {
  if (role.name != name || role.description != description || !role.owner.valid() || !role.lease.valid())
    return Status::InvalidArgument("DSpark supplied role has the wrong name, typed layout or readonly owner");
  HYLE_ASSIGN_OR_RETURN(const auto bytes, description.required_bytes());
  if (role.lease.size() != bytes)
    return Status::InvalidArgument("DSpark supplied role byte extent differs from its exact typed layout");
  HYLE_ASSIGN_OR_RETURN(const auto view, role.lease.tensor(description));
  if (view.storage().resource() != role.owner.ref())
    return Status::InvalidArgument("DSpark supplied role owner does not own its actual readonly resource");
  return Status::Ok();
}
}

namespace dspark_target_binding {
Result<g::ConstantBinding> constant(g::ValueId value, const ReadonlyRole &role) {
  HYLE_ASSIGN_OR_RETURN(auto view, role.lease.tensor(role.description));
  return g::ConstantBinding{.value = value, .tensor = view, .owner = role.owner, .lease = role.lease};
}
}

namespace dspark_target_binding {
Result<Projection> make_projection(ProjectionRoles roles, std::int64_t rows,
    c::DataType mean_storage) {
  HYLE_RETURN_IF_ERROR(qualify_identity(roles.target, roles.sidecar,roles.expected_sidecar));
  if (rows <= 0) return std::unexpected(Status::InvalidArgument("DSpark capture rows must be positive"));
  if(mean_storage!=c::DataType::kFloat32 && mean_storage!=c::DataType::kBFloat16)
    return std::unexpected(Status::InvalidArgument("DSpark capture mean requires explicit F32 or BF16 storage"));
  HYLE_ASSIGN_OR_RETURN(auto packed, c::gguf_weight_description(
      c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32, 3 * kHidden, kHidden));
  const auto weight=roles.projection.description;
  if(weight!=packed&&!dspark_stored_projection_fragment::bfloat_matrix(weight,kHidden,3*kHidden))
    return std::unexpected(Status::InvalidArgument("DSpark fc has an unsupported actual typed description"));
  HYLE_ASSIGN_OR_RETURN(auto norm, plain(std::array{kHidden}));
  HYLE_RETURN_IF_ERROR(qualify_role(roles.projection, "fc.weight", weight));
  HYLE_RETURN_IF_ERROR(qualify_role(roles.norm, "enc.output_norm.weight", norm));
  Projection result{.roles = std::move(roles), .rows = rows, .mean_storage=mean_storage};
  HYLE_ASSIGN_OR_RETURN(result.capture, plain(std::array{rows, std::int64_t{4}, kHidden}));
  HYLE_ASSIGN_OR_RETURN(result.output, plain(std::array{rows, kHidden}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto axis, c::Axes::from(std::array<std::int32_t, 1>{1}));
  std::array<g::ValueId, 3> means;
  for (std::size_t i = 0; i < means.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(result.taps[i], builder.input(result.capture));
    HYLE_ASSIGN_OR_RETURN(means[i], builder.add(c::Reduce{.kind = c::ReductionKind::kMean,
        .axes = axis, .keep_dimensions = false, .accumulation_type = c::DataType::kFloat32}, result.taps[i]));
    if(mean_storage==c::DataType::kBFloat16) {
      // The trained source calls h.mean(stream) on actual BF16 four-stream h.
      // Its result is stored in BF16 before the three layer means are joined.
      HYLE_ASSIGN_OR_RETURN(result.mean_raw[i],builder.add(c::Copy{},means[i]));
      HYLE_ASSIGN_OR_RETURN(auto stored,builder.add(c::Cast{c::DataType::kBFloat16},means[i]));
      HYLE_ASSIGN_OR_RETURN(result.mean_stored[i],builder.add(c::Copy{},stored));
      HYLE_ASSIGN_OR_RETURN(means[i],builder.add(c::Cast{c::DataType::kFloat32},stored));
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto joined01, builder.add(c::Concat{.axis = 1}, means[0], means[1]));
  HYLE_ASSIGN_OR_RETURN(auto joined, builder.add(c::Concat{.axis = 1}, joined01, means[2]));
  HYLE_ASSIGN_OR_RETURN(result.parameters[0], builder.constant(weight));
  HYLE_ASSIGN_OR_RETURN(result.parameters[1], builder.constant(norm));
  HYLE_ASSIGN_OR_RETURN(auto projected, dspark_stored_projection_fragment::linear(
      builder,joined,result.parameters[0],weight));
  HYLE_ASSIGN_OR_RETURN(result.main_x, builder.add(c::RmsNorm{.epsilon = 1.0e-6F}, projected, result.parameters[1]));
  HYLE_RETURN_IF_ERROR(builder.publish(result.main_x));
  if(mean_storage==c::DataType::kBFloat16)for(std::size_t i=0;i<means.size();++i) {
    HYLE_RETURN_IF_ERROR(builder.publish(result.mean_raw[i]));
    HYLE_RETURN_IF_ERROR(builder.publish(result.mean_stored[i]));
  }
  if(weight.is_plain()) {
    HYLE_ASSIGN_OR_RETURN(auto observed,builder.add(c::Copy{},projected));
    result.raw_projection_port=mean_storage==c::DataType::kBFloat16?7u:1u;
    HYLE_RETURN_IF_ERROR(builder.publish(observed));
  }
  HYLE_ASSIGN_OR_RETURN(result.graph, std::move(builder).finish());
  return result;
}
}

namespace dspark_target_binding {
Result<MainKvPart> append_main_kv(g::GraphBuilder &builder,
                                       const StageKvRoles &roles,
                                       std::int64_t rows, g::ValueId main_x) {
  HYLE_RETURN_IF_ERROR(qualify_identity(roles.target, roles.sidecar,roles.expected_sidecar));
  if (rows <= 0 || roles.stage < 0 || roles.stage >= 3)
    return std::unexpected(Status::InvalidArgument("DSpark main KV requires positive rows and a declared stage"));
  HYLE_ASSIGN_OR_RETURN(auto packed, c::gguf_weight_description(
      c::GgufQuantizedType::kQ8_0, c::DataType::kFloat32, kHidden, kKv));
  const auto weight=roles.projection.description;
  if(weight!=packed&&!dspark_stored_projection_fragment::bfloat_matrix(weight,kKv,kHidden))
    return std::unexpected(Status::InvalidArgument("DSpark wkv has an unsupported actual typed description"));
  HYLE_ASSIGN_OR_RETURN(auto norm, plain(std::array{kKv}));
  const auto prefix = "blk." + std::to_string(roles.stage);
  HYLE_RETURN_IF_ERROR(qualify_role(roles.projection, prefix + ".attn_kv.weight", weight));
  HYLE_RETURN_IF_ERROR(qualify_role(roles.norm, prefix + ".attn_kv_a_norm.weight", norm));
  MainKvPart result;
  HYLE_ASSIGN_OR_RETURN(result.parameters[0], builder.constant(weight));
  HYLE_ASSIGN_OR_RETURN(result.parameters[1], builder.constant(norm));
  HYLE_ASSIGN_OR_RETURN(auto projected, dspark_stored_projection_fragment::linear(
      builder,main_x,result.parameters[0],weight));
  result.raw_projection=projected;
  HYLE_ASSIGN_OR_RETURN(result.kv, builder.add(c::RmsNorm{.epsilon = 1.0e-6F}, projected, result.parameters[1]));
  return result;
}
}

namespace dspark_target_binding {
Result<MainKv> make_main_kv(StageKvRoles roles, std::int64_t rows) {
  if (rows <= 0)
    return std::unexpected(Status::InvalidArgument("DSpark main KV requires positive rows"));
  MainKv result{.roles = std::move(roles)};
  HYLE_ASSIGN_OR_RETURN(result.input, plain(std::array{rows, kHidden}));
  HYLE_ASSIGN_OR_RETURN(result.output, plain(std::array{rows, kKv}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.main_x, builder.input(result.input));
  HYLE_ASSIGN_OR_RETURN(auto part, append_main_kv(builder, result.roles, rows, result.main_x));
  result.parameters = part.parameters;
  result.kv = part.kv;
  HYLE_RETURN_IF_ERROR(builder.publish(result.kv));
  HYLE_ASSIGN_OR_RETURN(result.graph, std::move(builder).finish());
  return result;
}
}
