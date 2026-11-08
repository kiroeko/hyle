#pragma once

#include "inference/model_provider.h"
#include "models/qwen35/driver.h"
#include "models/qwen35/prepared_executor.h"

#include <array>
#include <memory>

namespace hyle::engine {

// Static application composition, not a runtime plugin system. Adding a model
// provider changes this boundary rather than the common worker or Graph.
inline std::unique_ptr<inference::ModelExecutor> load_model(const inference::ModelLoadConfig &config) {
  static constexpr std::array providers{
      inference::ModelProvider{"qwen35", qwen35::matches_prepared_model, qwen35::load_model_executor}};
  return inference::load_model(providers, config);
}

// The original CLI remains a Qwen-specific entry until it migrates to the
// common request interface. It is kept separate from model provider dispatch.
inline int run_legacy_cli(int argc, char **argv) { return qwen35::run(argc, argv); }

} // namespace hyle::engine
