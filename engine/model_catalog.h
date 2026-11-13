#pragma once

#include "inference/model_provider.h"
#include "models/qwen35/prepared_executor.h"

#include <array>
#include <memory>

namespace hyle::engine {

// Static application composition, not a runtime plugin system. Adding a model
// provider changes this boundary rather than the common worker or Graph.
inline std::span<const inference::ModelProvider> model_providers() noexcept {
  static constexpr std::array providers{
      inference::ModelProvider{"qwen35", qwen35::matches_prepared_model, qwen35::load_model_executor}};
  return providers;
}

inline std::unique_ptr<inference::ModelExecutor> load_model(const inference::ModelLoadConfig &config) {
  return inference::load_model(model_providers(), config);
}

} // namespace hyle::engine
