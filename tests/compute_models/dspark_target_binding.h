#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "dspark_stored_projection_fragment.h"

// A test-consumer boundary for the selected Flash0731 Q8_0 sidecar. This
// supplies ordinary Graph values and retained constant/input permissions;
// it does not load files, schedule work, identify providers or own an executor.
namespace dspark_target_binding {
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::Result;
using hyle::Status;
inline constexpr std::int64_t kHidden = 4096, kKv = 512;
inline constexpr auto kModelRepository = "deepseek-ai/DeepSeek-V4-Flash-0731";
inline constexpr auto kModelRevision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
inline constexpr auto kSidecarRepository = "unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr auto kSidecarRevision = "fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr auto kSidecarFile = "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
inline constexpr std::array<std::int32_t, 3> kBlocks{40, 41, 42};

struct TargetIdentity final {
  std::string model_repository, model_revision;
  std::string artifact_repository, artifact_revision, variant;
  bool operator==(const TargetIdentity &) const = default;
};
struct SidecarIdentity final {
  std::string repository, revision, file;
  bool operator==(const SidecarIdentity &) const = default;
};
struct ReadonlyRole final {
  std::string name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
};
struct ProjectionRoles final {
  TargetIdentity target;
  SidecarIdentity sidecar;
  ReadonlyRole projection, norm;
  SidecarIdentity expected_sidecar{kSidecarRepository,kSidecarRevision,kSidecarFile};
};
struct StageKvRoles final {
  TargetIdentity target;
  SidecarIdentity sidecar;
  std::int32_t stage = 0;
  ReadonlyRole projection, norm;
  SidecarIdentity expected_sidecar{kSidecarRepository,kSidecarRevision,kSidecarFile};
};

inline Status qualify_identity(const TargetIdentity &target,
                               const SidecarIdentity &sidecar,
                               const SidecarIdentity &expected={kSidecarRepository,kSidecarRevision,kSidecarFile}) {
  if (target.model_repository != kModelRepository || target.model_revision != kModelRevision ||
      target.artifact_repository.empty() || target.artifact_revision.empty() || target.variant.empty() ||
      expected.repository.empty()||expected.revision.empty()||expected.file.empty()||sidecar!=expected)
    return Status::InvalidArgument("DSpark requires the matching formal Flash0731 target and explicitly pinned sidecar identity");
  return Status::Ok();
}
inline Result<c::TensorDesc> plain(std::span<const std::int64_t> shape) {
  return c::TensorDesc::dense(c::DataType::kFloat32, shape);
}
inline Status qualify_role(const ReadonlyRole &role, std::string_view name,
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
inline Result<g::ConstantBinding> constant(g::ValueId value, const ReadonlyRole &role) {
  HYLE_ASSIGN_OR_RETURN(auto view, role.lease.tensor(role.description));
  return g::ConstantBinding{.value = value, .tensor = view, .owner = role.owner, .lease = role.lease};
}

enum class LayerConvention { kOfficialBlockZeroBased, kGgufHiddenStateOneBased };
struct CaptureFrame final {
  TargetIdentity target;
  std::uint64_t session = 0, branch = 0, epoch = 0;
  std::int64_t first_position = 0, rows = 0;
  bool operator==(const CaptureFrame &) const = default;
};
struct TargetTap final {
  CaptureFrame frame;
  LayerConvention convention = LayerConvention::kOfficialBlockZeroBased;
  std::int32_t layer = 0;
  c::TensorDesc description;
  c::ReadLease lease;
};

struct Projection final {
  g::Graph graph;
  c::TensorDesc capture, output;
  std::array<g::ValueId, 3> taps;
  std::array<g::ValueId, 2> parameters;
  g::ValueId main_x;
  ProjectionRoles roles;
  std::int64_t rows = 0;
  c::DataType mean_storage = c::DataType::kFloat32;
  // Optional observations of the explicit BF16 capture-mean boundary. The
  // F32 fixture keeps its existing single-output Graph and arithmetic.
  std::array<g::ValueId,3> mean_raw,mean_stored;
  std::optional<std::size_t> raw_projection_port;

  Result<std::array<g::ConstantBinding, 2>> constants() const {
    HYLE_ASSIGN_OR_RETURN(auto weight, constant(parameters[0], roles.projection));
    HYLE_ASSIGN_OR_RETURN(auto norm, constant(parameters[1], roles.norm));
    return std::array{std::move(weight), std::move(norm)};
  }
  // Validate the whole tuple before the caller binds anything. ReadLease
  // ownership proves ready immutable storage, not that its label is authentic:
  // the actual target consumer supplies these provenance facts and its epoch.
  Result<std::array<c::ReadLease, 3>> qualify(const CaptureFrame &expected,
                                           const std::array<TargetTap, 3> &actual) const {
    if (expected.target != roles.target || expected.rows != rows || expected.epoch == 0 ||
        expected.first_position < 0 || rows > std::numeric_limits<std::int64_t>::max() - expected.first_position)
      return std::unexpected(Status::InvalidArgument("DSpark target capture frame does not match this projection"));
    std::array<c::ReadLease, 3> result;
    std::array<bool, 3> seen{};
    HYLE_ASSIGN_OR_RETURN(const auto bytes, capture.required_bytes());
    for (const auto &tap : actual) {
      if (tap.frame != expected || tap.description != capture || !tap.lease.valid() || tap.lease.size() != bytes)
        return std::unexpected(Status::InvalidArgument("DSpark tap target/session/branch/epoch/position/shape differs"));
      std::int32_t block = tap.layer;
      if (tap.convention == LayerConvention::kGgufHiddenStateOneBased) {
        if (block <= 0) return std::unexpected(Status::InvalidArgument("invalid one-based DSpark capture layer"));
        --block;
      } else if (tap.convention != LayerConvention::kOfficialBlockZeroBased) {
        return std::unexpected(Status::InvalidArgument("unknown DSpark capture layer convention"));
      }
      std::size_t slot = 0;
      while (slot < kBlocks.size() && kBlocks[slot] != block) ++slot;
      if (slot == kBlocks.size() || seen[slot])
        return std::unexpected(Status::InvalidArgument("DSpark capture layer is missing, duplicated or belongs to another block"));
      HYLE_ASSIGN_OR_RETURN(const auto view, tap.lease.tensor(capture));
      (void)view;
      seen[slot] = true;
      result[slot] = tap.lease;
    }
    return result;
  }
};

inline Result<Projection> make_projection(ProjectionRoles roles, std::int64_t rows,
    c::DataType mean_storage=c::DataType::kFloat32) {
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

// Exactly the selected sidecar stage's wkv -> kv_norm boundary, before RoPE
// or a physical ring write. A producer main_x lease can bind directly here.
struct MainKv final {
  g::Graph graph;
  c::TensorDesc input, output;
  g::ValueId main_x, kv;
  std::array<g::ValueId, 2> parameters;
  StageKvRoles roles;
  Result<std::array<g::ConstantBinding, 2>> constants() const {
    HYLE_ASSIGN_OR_RETURN(auto weight, constant(parameters[0], roles.projection));
    HYLE_ASSIGN_OR_RETURN(auto norm, constant(parameters[1], roles.norm));
    return std::array{std::move(weight), std::move(norm)};
  }
};
struct MainKvPart final {
  std::array<g::ValueId, 2> parameters;
  g::ValueId kv,raw_projection;
};
// Share the model's actual math when embedding it in a larger cold graph.
// Callers need neither the standalone graph's node order nor its node count.
inline Result<MainKvPart> append_main_kv(g::GraphBuilder &builder,
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
inline Result<MainKv> make_main_kv(StageKvRoles roles, std::int64_t rows) {
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
} // namespace dspark_target_binding
