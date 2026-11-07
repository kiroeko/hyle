#pragma once

#include "../../inference/mtp_policy.h"
#include "mtp_capabilities.h"
#include "precision_profile.h"
#include "weights.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace qwen35 {

// Cold source selection. This chooses paths only; prepare_weight_source later
// validates the selected vocabulary, original source and optional head.
struct DraftAssetSelection final {
  hyle::inference::mtp::DefaultSelection policy;
  std::filesystem::path vocabulary;
  std::filesystem::path head;
  bool auto_head = false;
  std::string_view vocabulary_reason;
};

inline DraftAssetSelection select_draft_assets(const std::filesystem::path &input,
    const hyle::inference::mtp::DefaultRequest &request,
    std::filesystem::path vocabulary, std::filesystem::path head, bool encoded_rows_on_backend) {
  namespace mtp = hyle::inference::mtp;
  const auto inspect = [](const std::filesystem::path &path, const char *label) {
    std::error_code error;
    const auto present = std::filesystem::exists(path, error);
    if (error) throw std::runtime_error(std::string(label) + error.message());
    return present;
  };
  const auto prepared_vocabulary = input / "mtp-draft-vocabulary.ids";
  const bool explicit_asset_selection = vocabulary != "auto" || head != "auto";
  const bool explicit_file = (vocabulary != "auto" && vocabulary != "full") ||
      (head != "auto" && head != "original");
  // An implicit map must never make a previously usable source/backend fail.
  // Explicit files still reach the loader and report their actual incompatibility.
  const bool may_use_automatic_vocabulary = request.mode != mtp::RequestedMode::kOff &&
      request.max_drafts && request.greedy && request.automatic_vocabulary &&
      (request.mode == mtp::RequestedMode::kOn || request.automatic_backend_eligible);
  const bool present_automatic_vocabulary = may_use_automatic_vocabulary &&
      inspect(prepared_vocabulary, "cannot inspect prepared MTP defaults: ");
  bool usable_automatic_vocabulary = false;
  if (present_automatic_vocabulary) {
    const auto roles = read_weight_roles(input / "weights.tsv", false);
    usable_automatic_vocabulary = supports_draft_vocabulary(roles, encoded_rows_on_backend);
  }
  const bool has_vocabulary = mtp::inspect_prepared_vocabulary(request) && usable_automatic_vocabulary;
  bool original_projection_eligible = false;
  if (!has_vocabulary && !explicit_asset_selection && request.mode == mtp::RequestedMode::kAuto &&
      mtp::select_default(request, true).automatic) {
    const auto roles = read_weight_roles(input / "weights.tsv", false);
    original_projection_eligible = roles.representation == prepared::SourceRepresentation::kGgufConverted &&
        roles.geometry.mtp_layers == 1 &&
        PrecisionProfile::from_roles(roles).activation_type() == hyle::compute::DataType::kFloat32 &&
        supports_stored_draft_projection(roles);
  }
  const auto policy = mtp::select_default(request, has_vocabulary, original_projection_eligible,
      explicit_asset_selection);
  auto vocabulary_reason = vocabulary == "full" ? std::string_view{"explicit_full"} :
      vocabulary != "auto" ? std::string_view{"explicit_file"} :
      !may_use_automatic_vocabulary ? std::string_view{"not_inspected"} :
      !present_automatic_vocabulary ? std::string_view{"not_present"} :
      !usable_automatic_vocabulary ? std::string_view{"unsupported_source_or_backend"} :
      std::string_view{"not_selected"};
  if (!policy.enabled || !request.max_drafts) {
    if (explicit_file)
      throw std::runtime_error("explicit MTP asset cannot be used: " +
          std::string(request.max_drafts ? policy.reason : "drafting_disabled"));
    return {policy, {}, {}, false, vocabulary_reason};
  }
  bool auto_head = false;
  if (head == "original") head.clear();
  else if (head == "auto") {
    head.clear();
    auto_head = request.greedy && !original_projection_eligible;
  }
  if (vocabulary == "full") vocabulary.clear();
  else if (vocabulary == "auto") {
    vocabulary.clear();
    if (policy.prepared_assets) vocabulary = prepared_vocabulary;
    else if (head.empty() && usable_automatic_vocabulary)
      vocabulary = prepared_vocabulary;
    if (!vocabulary.empty()) vocabulary_reason = "prepared_vocabulary";
  }
  if (!vocabulary.empty() && !request.greedy)
    throw std::runtime_error("--mtp-draft-vocabulary requires --mtp on, active drafting, unmodified greedy sampling and zero minimum probability");
  if (!head.empty() && !request.greedy)
    throw std::runtime_error("--mtp-draft-head requires active unmodified greedy MTP and zero minimum probability");
  return {policy, std::move(vocabulary), std::move(head), auto_head, vocabulary_reason};
}

} // namespace qwen35
