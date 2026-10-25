#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <variant>

#include "model_geometry.h"
#include "precision_profile.h"
#include "hyle/compute/model_ops.h"

namespace qwen35 {
struct Tensor;
struct LoadedWeight;

// These cold bindings borrow the Tensor children of stable LoadedWeight map
// nodes. They own no storage, leases or graph ValueIds. Do not erase or replace the referenced weights
// while a binding/Recipe exists; rebuild the bindings after an idle reload.
struct StoredProjection final { Tensor *weight = nullptr; };
struct GptqProjection final {
  Tensor *codes = nullptr, *zeros = nullptr, *scales = nullptr, *groups = nullptr;
  hyle::compute::model_ops::ArtifactWeightDecode decode;
};
struct CtPackedProjection final {
  Tensor *codes = nullptr, *zeros = nullptr, *scales = nullptr;
  hyle::compute::model_ops::ArtifactWeightDecode decode;
};
struct MlxAffineProjection final {
  Tensor *codes = nullptr, *scales = nullptr, *biases = nullptr;
  hyle::compute::model_ops::MlxQuantized equation;
};
struct Nf4Projection final {
  std::array<Tensor *, 6> original{};
  hyle::compute::model_ops::ArtifactWeightDecode decode;
};
struct ModelOptNvfp4Projection final {
  Tensor *codes = nullptr, *scales = nullptr, *weight_global = nullptr, *input_global = nullptr;
};
struct LocalNvfp4Projection final {
  Tensor *codes = nullptr, *scales = nullptr, *weight_global = nullptr, *input_global = nullptr;
  hyle::compute::model_ops::LocalNvfp4Linear equation;
};
struct BlockFp8Projection final {
  Tensor *carrier = nullptr;
  hyle::compute::model_ops::Float8BlockLinear equation;
};
// The source codec certifies the fixed CT BF16 token/channel QDQ reader;
// this is not the distinct block128 FP8 scaling or a hardware FP8 dot.
struct ChannelTokenFp8Projection final { Tensor *codes = nullptr, *channel_scales = nullptr; };
// Selected stored-scale software reader: F32(A)*(1/SA), SATFINITE E4,
// F32 code dot, separate *SA then *SW, BF16 output. Not producer fake-QDQ
// or a claim that a native scaled_mm has the same intermediate arithmetic.
struct ModelOptStaticFp8Projection final {
  Tensor *codes = nullptr, *activation_scale = nullptr, *weight_scale = nullptr;
};

using ProjectionBinding = std::variant<StoredProjection, GptqProjection,
    CtPackedProjection, MlxAffineProjection, Nf4Projection, ModelOptNvfp4Projection, LocalNvfp4Projection,
    BlockFp8Projection, ChannelTokenFp8Projection, ModelOptStaticFp8Projection>;
using ProjectionBindings = std::map<std::string, ProjectionBinding, std::less<>>;

// Definitions require the complete model-private LoadedWeight. Resolve one logical
// .weight group for component consumers, or all linear inventory entries plus
// the actual vocabulary head for a fully loaded ordinary/MTP model. Invalid
// physical groups fail during cold preparation with std::runtime_error.
ProjectionBinding resolve_projection(std::map<std::string, LoadedWeight> &bank,
                                     std::string_view logical_name);
ProjectionBindings prepare_projection_bindings(std::map<std::string, LoadedWeight> &bank,
                                                bool mtp, const ModelGeometry &geometry);
std::string_view score_policy(const ProjectionBinding &binding, PrecisionProfile precision) noexcept;
} // namespace qwen35
