#pragma once

#include <memory>

#include "../../inference/model_provider.h"

namespace qwen35 {
class Model;
// Model remains the sole execution owner and must outlive the returned adapter.
// Create at most one adapter for a Model; do not execute that Model directly
// or through another adapter while this adapter exists.
std::unique_ptr<hyle::inference::ModelExecutor> make_prepared_executor(
    Model &model, bool mtp, std::size_t prefill_rows, bool overlap);

// Owns its model and original native resources for the returned executor's
// lifetime. It neither supplies the frontend scheduler nor retains sessions
// across separate execute calls.
bool matches_prepared_model(const std::filesystem::path &source);
std::unique_ptr<hyle::inference::ModelExecutor> load_model_executor(const hyle::inference::ModelLoadConfig &config);
} // namespace qwen35
