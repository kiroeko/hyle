#pragma once

#include <cstddef>
#include <memory>

#include "../../inference/model_executor.h"

namespace qwen35 {
class Model;
// Model remains the sole execution owner and must outlive the returned adapter.
// Create at most one adapter for a Model; do not execute that Model directly
// or through another adapter while this adapter exists.
std::unique_ptr<hyle::inference::ModelExecutor> make_prepared_executor(
    Model &model, bool mtp, std::size_t prefill_rows, bool overlap);
} // namespace qwen35
