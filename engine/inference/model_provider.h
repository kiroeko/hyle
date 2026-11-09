#pragma once

#include "model_executor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hyle::inference {

enum class BackendKind : std::uint8_t;

// Asset roles are provider-defined; this common contract does not prescribe
// a draft architecture or require every model to use a vocabulary projection.
enum class ModelAssetMode : std::uint8_t { kFile, kAutomatic };
struct ModelAsset final {
  std::string role;
  ModelAssetMode mode = ModelAssetMode::kFile;
  std::filesystem::path path;
};

// Immutable source assets are separate from execution capacity and from
// prepared sampling programs. A provider validates roles and source identity.
struct ModelAssets final {
  std::filesystem::path inputs;
  std::vector<ModelAsset> auxiliary;
  void set_file(std::string role, std::filesystem::path path) {
    if (role.empty() || path.empty()) throw std::invalid_argument("model asset requires a role and nonempty path");
    for (auto &asset : auxiliary) if (asset.role == role) {
      asset = {std::move(role), ModelAssetMode::kFile, std::move(path)};
      return;
    }
    auxiliary.push_back({std::move(role), ModelAssetMode::kFile, std::move(path)});
  }
};

inline void validate_model_assets(const ModelAssets &assets) {
  if (assets.inputs.empty()) throw std::invalid_argument("model input path is required");
  for (std::size_t index = 0; index < assets.auxiliary.size(); ++index) {
    const auto &asset = assets.auxiliary[index];
    if (asset.role.empty() || (asset.mode == ModelAssetMode::kFile && asset.path.empty()) ||
        (asset.mode == ModelAssetMode::kAutomatic && !asset.path.empty()) ||
        (asset.mode != ModelAssetMode::kFile && asset.mode != ModelAssetMode::kAutomatic))
      throw std::invalid_argument("invalid model asset role, mode or path");
    for (std::size_t previous = 0; previous < index; ++previous)
      if (assets.auxiliary[previous].role == asset.role)
        throw std::invalid_argument("duplicate model asset role");
  }
}

inline const ModelAsset *find_model_asset(const ModelAssets &assets, std::string_view role) noexcept {
  for (const auto &asset : assets.auxiliary) if (asset.role == role) return &asset;
  return nullptr;
}

struct ExecutionPreparation final {
  BackendKind backend{};
  std::size_t context = 0, resident_sessions = 1, active_sequences = 1;
  std::size_t prefill_rows = (std::numeric_limits<std::size_t>::max)();
  std::optional<std::size_t> draft_count;
  std::optional<bool> overlap;
};

// Describes a program prepared for later request selection, not an RNG
// cursor. Per-request RNG identity belongs to its request/session.
struct PreparedSamplingPolicy final {
  bool random = false;
  float temperature = 1.F, top_p = 1.F;
  std::size_t top_k = 0;
};

struct ModelLoadConfig final {
  ModelAssets assets;
  ExecutionPreparation execution;
  PreparedSamplingPolicy sampling;
  // Index zero is sampling; additional entries are immutable prepared choices
  // which requests can select independently.
  std::vector<PreparedSamplingPolicy> additional_sampling;
};

// Providers are composed at application startup. Matching may read source
// metadata, but may not load weights or mutate the input. A match is only a
// cold dispatch hint: the provider validates the complete source when loaded.
struct ModelProvider final {
  std::string_view name;
  bool (*matches)(const std::filesystem::path &source);
  std::unique_ptr<ModelExecutor> (*load)(const ModelLoadConfig &);
};

inline const ModelProvider &select_model_provider(std::span<const ModelProvider> providers,
                                                   const std::filesystem::path &source) {
  const ModelProvider *selected = nullptr;
  for (const auto &provider : providers) {
    if (provider.name.empty() || !provider.matches || !provider.load)
      throw std::invalid_argument("model provider is incomplete");
    if (!provider.matches(source)) continue;
    if (selected) throw std::runtime_error("prepared model matches multiple providers");
    selected = &provider;
  }
  if (!selected) throw std::runtime_error("no model provider recognizes the prepared model");
  return *selected;
}

inline std::unique_ptr<ModelExecutor> load_model(std::span<const ModelProvider> providers,
                                                 const ModelLoadConfig &config) {
  validate_model_assets(config.assets);
  auto executor = select_model_provider(providers, config.assets.inputs).load(config);
  if (!executor) throw std::runtime_error("model provider returned no executor");
  return executor;
}

} // namespace hyle::inference
