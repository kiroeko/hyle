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
#include <string_view>

namespace hyle::inference {

enum class BackendKind : std::uint8_t;

// Cold configuration. A model provider validates its own supported inputs;
// no model-specific tensor names or state layout cross this boundary.
struct ModelLoadConfig final {
  std::filesystem::path inputs, draft_vocabulary, draft_head;
  BackendKind backend{};
  std::size_t context = 0, resident_sessions = 1, active_sequences = 1;
  std::size_t prefill_rows = (std::numeric_limits<std::size_t>::max)();
  std::optional<std::size_t> draft_count;
  bool auto_draft_head = false, random = false;
  std::optional<bool> overlap;
  float temperature = 1.F, top_p = 1.F;
  std::size_t top_k = 0;
  std::uint64_t seed = 0;
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
  if (config.inputs.empty()) throw std::invalid_argument("model input path is required");
  auto executor = select_model_provider(providers, config.inputs).load(config);
  if (!executor) throw std::runtime_error("model provider returned no executor");
  return executor;
}

} // namespace hyle::inference
