#pragma once

#include "../../inference/mtp_policy.h"
#include "mtp_capabilities.h"
#include "precision_profile.h"
#include "weights.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwen35 {

// Cold source selection. This chooses paths only; prepare_weight_source later
// validates the selected vocabulary, original source and optional head.
struct DraftAssetSelection final {
  hyle::inference::mtp::DefaultSelection policy;
  std::filesystem::path vocabulary;
  std::filesystem::path head;
  bool auto_head = false;
};

inline DraftAssetSelection select_draft_assets(const std::filesystem::path &input,
    const hyle::inference::mtp::DefaultRequest &request,
    std::filesystem::path vocabulary, std::filesystem::path head) {
  namespace mtp = hyle::inference::mtp;
  const auto inspect = [](const std::filesystem::path &path, const char *label) {
    std::error_code error;
    const auto present = std::filesystem::exists(path, error);
    if (error) throw std::runtime_error(std::string(label) + error.message());
    return present;
  };
  const auto prepared_vocabulary = input / "mtp-draft-vocabulary.ids";
  const bool has_vocabulary = mtp::inspect_prepared_vocabulary(request) &&
      inspect(prepared_vocabulary, "cannot inspect prepared MTP defaults: ");
  bool original_projection_eligible = false;
  if (!has_vocabulary && request.mode == mtp::RequestedMode::kAuto &&
      mtp::select_default(request, true).automatic) {
    const auto roles = read_weight_roles(input / "weights.tsv", false);
    original_projection_eligible = roles.representation == prepared::SourceRepresentation::kGgufConverted &&
        roles.geometry.mtp_layers == 1 &&
        PrecisionProfile::from_roles(roles).activation_type() == hyle::compute::DataType::kFloat32 &&
        supports_stored_draft_projection(roles);
  }
  const auto policy = mtp::select_default(request, has_vocabulary, original_projection_eligible);
  bool auto_head = false;
  if (head == "original") head.clear();
  else if (head == "auto") {
    head.clear();
    auto_head = policy.enabled && request.max_drafts && request.greedy && !original_projection_eligible;
  }
  if (vocabulary == "full") vocabulary.clear();
  else if (vocabulary == "auto") {
    vocabulary.clear();
    if (policy.prepared_assets) vocabulary = prepared_vocabulary;
    else if (head.empty() && policy.enabled && request.max_drafts && request.greedy &&
             inspect(prepared_vocabulary, "cannot inspect prepared draft vocabulary: "))
      vocabulary = prepared_vocabulary;
  }
  if (!vocabulary.empty() && (!policy.enabled || !request.max_drafts || !request.greedy))
    throw std::runtime_error("--mtp-draft-vocabulary requires --mtp on, active drafting, unmodified greedy sampling and zero minimum probability");
  if (!head.empty() && (!policy.enabled || !request.max_drafts || !request.greedy))
    throw std::runtime_error("--mtp-draft-head requires active unmodified greedy MTP and zero minimum probability");
  return {policy, std::move(vocabulary), std::move(head), auto_head};
}

} // namespace qwen35
