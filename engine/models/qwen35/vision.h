#pragma once

// Qwen-private image equations and source-bound cold assets. Reuses the
// original Backend/Graph/weight stream. It owns no request, Session or KV cache.
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "graph_builder.h"
#include "../../inference/prompt_attachment.h"
#include "../../inference/backend.h"

namespace qwen35 {
struct VisionGeometry final {
  std::size_t layers = 0, hidden = 0, intermediate = 0, heads = 0;
  std::size_t patch = 0, temporal = 0, merge = 0, positions = 0, output = 0;
  std::int32_t image_id = -1, start_id = -1, end_id = -1;
  std::array<std::size_t, 3> mrope{};
};
struct VisionSource final {
  VisionGeometry geometry;
  prepared::Roles roles;
  std::string identity;
};
VisionSource read_vision_source(const std::filesystem::path &path, const prepared::Roles &target);

// Shared model-private equation for both ViT and language three-axis RoPE.
// cos/sin are explicit F32 [B,T,1,rotary/2] values from the source recipe.
V apply_position_angles(Recipe &recipe, V value, V cos, V sin, std::size_t rotary);

class VisionEncoder final : public std::enable_shared_from_this<VisionEncoder> {
public:
  VisionEncoder(hyle::inference::Backend &backend, const prepared::Roles &target,
      const std::filesystem::path &path, const c::ResourceBudget &budget = {});
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder &) = delete;
  VisionEncoder &operator=(const VisionEncoder &) = delete;
  const VisionGeometry &geometry() const noexcept { return source_.geometry; }
  const std::string &identity() const noexcept { return source_.identity; }
  // Called by the serialized model owner at input preparation. Output stays on
  // the original endpoint, in the requested LM activation type, through prefill.
  Tensor encode(const hyle::inference::PromptAttachment &part, Type output_type);
  void validate(const hyle::inference::PromptAttachment &part) const;
private:
  struct Program;
  hyle::inference::Backend backend_;
  c::ResourceBudget budget_;
  VisionSource source_;
  std::map<std::string, LoadedWeight> weights_;
  std::vector<c::LeasePool> pools_;
  ProjectionBindings projections_;
  std::map<std::array<std::size_t, 3>, std::unique_ptr<Program>> programs_;
  std::unique_ptr<Program> prepare(std::size_t height, std::size_t width, Type output_type);
};

struct ConditionedImage final { std::size_t begin = 0, rows = 0; Tensor features; };
struct ConditionedPrompt final {
  std::vector<ConditionedImage> images;
  std::vector<std::array<std::int32_t,3>> positions;
};
ConditionedPrompt prepare_image_prompt(VisionEncoder &encoder, std::span<const std::int32_t> ids,
    std::span<const hyle::inference::PromptAttachment> images, std::size_t pending_rows,
    std::int64_t initial_position, Type activation_type);
// Pure source geometry/placeholder planning. Pixel validation and execution
// remain separate, so admission can reject malformed media without submitting.
ConditionedPrompt image_position_plan(const VisionGeometry &geometry, std::span<const std::int32_t> ids,
    std::span<const hyle::inference::PromptAttachment> images, std::size_t pending_rows,
    std::int64_t initial_position);
} // namespace qwen35
