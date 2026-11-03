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

Status qualify_identity(const TargetIdentity &target,
                               const SidecarIdentity &sidecar,
                               const SidecarIdentity &expected={kSidecarRepository,kSidecarRevision,kSidecarFile});
Result<c::TensorDesc> plain(std::span<const std::int64_t> shape);
Status qualify_role(const ReadonlyRole &role, std::string_view name,
                           const c::TensorDesc &description);
Result<g::ConstantBinding> constant(g::ValueId value, const ReadonlyRole &role);

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

Result<Projection> make_projection(ProjectionRoles roles, std::int64_t rows,
    c::DataType mean_storage=c::DataType::kFloat32);

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
Result<MainKvPart> append_main_kv(g::GraphBuilder &builder,
                                       const StageKvRoles &roles,
                                       std::int64_t rows, g::ValueId main_x);
Result<MainKv> make_main_kv(StageKvRoles roles, std::int64_t rows);
} // namespace dspark_target_binding
