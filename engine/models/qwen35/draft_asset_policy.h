#pragma once

#include "../../inference/mtp_policy.h"
#include "mtp_default_policy.h"
#include "mtp_capabilities.h"
#include "precision_profile.h"
#include "weights.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwen35 {

// A selected file is always a file, even when its name is "auto" or "full".
// Entry points translate any textual policy keywords before this boundary.
enum class DraftAssetKind { kAutomatic, kOriginal, kFile };
struct DraftAssetRequest final {
  DraftAssetKind kind = DraftAssetKind::kAutomatic;
  std::filesystem::path file;

  static DraftAssetRequest automatic() { return {}; }
  static DraftAssetRequest original() { return {DraftAssetKind::kOriginal, {}}; }
  static DraftAssetRequest from_file(std::filesystem::path path) {
    if (path.empty()) throw std::invalid_argument("explicit draft asset path is empty");
    return {DraftAssetKind::kFile, std::move(path)};
  }
};

// Cold source selection. This chooses paths only; prepare_weight_source later
// validates the selected vocabulary, original source and optional head.
struct DraftAssetSelection final {
  hyle::inference::mtp::DefaultSelection policy;
  std::filesystem::path vocabulary;
  std::filesystem::path head;
  bool auto_head = false;
  hyle::inference::RouteReason vocabulary_reason = hyle::inference::RouteReason::kNone;
};

inline DraftAssetSelection select_draft_assets(const std::filesystem::path &input,
    const MtpDefaultRequest &request,
    DraftAssetRequest vocabulary_request, DraftAssetRequest head_request, bool encoded_rows_on_backend) {
  namespace mtp = hyle::inference::mtp;
  const auto valid = [](const DraftAssetRequest &asset) {
    switch (asset.kind) {
      case DraftAssetKind::kAutomatic:
      case DraftAssetKind::kOriginal: return asset.file.empty();
      case DraftAssetKind::kFile: return !asset.file.empty();
    }
    return false;
  };
  if (!valid(vocabulary_request) || !valid(head_request))
    throw std::invalid_argument("invalid draft asset selection");
  const auto inspect = [](const std::filesystem::path &path, const char *label) {
    std::error_code error;
    const auto present = std::filesystem::exists(path, error);
    if (error) throw std::runtime_error(std::string(label) + error.message());
    return present;
  };
  const auto prepared_vocabulary = input / "mtp-draft-vocabulary.ids";
  const bool explicit_asset_selection = vocabulary_request.kind != DraftAssetKind::kAutomatic ||
      head_request.kind != DraftAssetKind::kAutomatic;
  const bool explicit_file = vocabulary_request.kind == DraftAssetKind::kFile || head_request.kind == DraftAssetKind::kFile;
  // An implicit map must never make a previously usable source/backend fail.
  // Explicit files still reach the loader and report their actual incompatibility.
  const bool may_use_automatic_vocabulary = request.mode != mtp::RequestedMode::kOff &&
      request.max_drafts && request.unmodified_greedy_drafting() && request.automatic_vocabulary &&
      (request.mode == mtp::RequestedMode::kOn || request.automatic_backend_eligible);
  const bool present_automatic_vocabulary = may_use_automatic_vocabulary &&
      inspect(prepared_vocabulary, "cannot inspect prepared MTP defaults: ");
  bool usable_automatic_vocabulary = false;
  if (present_automatic_vocabulary) {
    const auto roles = read_weight_roles(input / "weights.tsv", false);
    usable_automatic_vocabulary = supports_draft_vocabulary(roles, encoded_rows_on_backend);
  }
  const bool has_vocabulary = inspect_prepared_vocabulary(request) && usable_automatic_vocabulary;
  bool original_projection_eligible = false;
  if (!has_vocabulary && !explicit_asset_selection && request.mode == mtp::RequestedMode::kAuto &&
      select_mtp_default(request, true).automatic) {
    const auto roles = read_weight_roles(input / "weights.tsv", false);
    original_projection_eligible = roles.representation == prepared::SourceRepresentation::kGgufConverted &&
        roles.geometry.mtp_layers == 1 &&
        PrecisionProfile::from_roles(roles).activation_type() == hyle::compute::DataType::kFloat32 &&
        supports_stored_draft_projection(roles);
  }
  const auto policy = select_mtp_default(request, has_vocabulary, original_projection_eligible,
      explicit_asset_selection);
  auto vocabulary_reason = vocabulary_request.kind == DraftAssetKind::kOriginal ? hyle::inference::RouteReason::kExplicitFull :
      vocabulary_request.kind == DraftAssetKind::kFile ? hyle::inference::RouteReason::kExplicitFile :
      !may_use_automatic_vocabulary ? hyle::inference::RouteReason::kNotInspected :
      !present_automatic_vocabulary ? hyle::inference::RouteReason::kNotPresent :
      !usable_automatic_vocabulary ? hyle::inference::RouteReason::kUnsupportedSourceOrBackend :
      hyle::inference::RouteReason::kNotSelected;
  if (!policy.enabled || !request.max_drafts) {
    if (explicit_file)
      throw std::runtime_error("explicit MTP asset cannot be used: " +
          std::string(hyle::inference::route_reason_name(request.max_drafts ? policy.reason :
              hyle::inference::RouteReason::kDraftingDisabled)));
    return {policy, {}, {}, false, vocabulary_reason};
  }
  auto vocabulary = std::move(vocabulary_request.file);
  auto head = std::move(head_request.file);
  bool auto_head = false;
  if (head_request.kind == DraftAssetKind::kAutomatic) {
    auto_head = request.unmodified_greedy_drafting() && !original_projection_eligible;
  }
  if (vocabulary_request.kind == DraftAssetKind::kAutomatic) {
    if (policy.prepared_assets) vocabulary = prepared_vocabulary;
    else if (head.empty() && usable_automatic_vocabulary)
      vocabulary = prepared_vocabulary;
    if (!vocabulary.empty()) vocabulary_reason = hyle::inference::RouteReason::kPreparedVocabulary;
  }
  if (!vocabulary.empty() && !request.unmodified_greedy_drafting())
    throw std::runtime_error("--mtp-draft-vocabulary requires --mtp on, active drafting and unmodified greedy sampling");
  if (!head.empty() && !request.unmodified_greedy_drafting())
    throw std::runtime_error("--mtp-draft-head requires active unmodified greedy MTP");
  return {policy, std::move(vocabulary), std::move(head), auto_head, vocabulary_reason};
}

} // namespace qwen35
