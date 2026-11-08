#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>

#include "../../inference/model_executor.h"

namespace hyle::inference { enum class BackendKind : std::uint8_t; }

namespace qwen35 {
class Model;
// Model remains the sole execution owner and must outlive the returned adapter.
// Create at most one adapter for a Model; do not execute that Model directly
// or through another adapter while this adapter exists.
std::unique_ptr<hyle::inference::ModelExecutor> make_prepared_executor(
    Model &model, bool mtp, std::size_t prefill_rows, bool overlap);

// Model-adapter inputs. Unspecified drafting and overlap use the existing
// cold policies; explicit values select the requested configuration. Tokenizer,
// templates and scheduling are owned by the engine.
struct ModelLoadConfig final {
  std::filesystem::path inputs, draft_vocabulary, draft_head;
  hyle::inference::BackendKind backend{};
  std::size_t context = 0, resident_sessions = 1, active_sequences = 1;
  std::size_t prefill_rows = (std::numeric_limits<std::size_t>::max)();
  std::optional<std::size_t> draft_count;
  bool auto_draft_head = false, random = false;
  // Unspecified selects the existing single-session ordinary route in cold preparation.
  std::optional<bool> overlap;
  float temperature = 1.F, top_p = 1.F;
  std::size_t top_k = 0;
  std::uint64_t seed = 0;
};

// Owns its model and original native resources for the returned executor's
// lifetime. It neither supplies the frontend scheduler nor retains sessions
// across separate execute calls.
std::unique_ptr<hyle::inference::ModelExecutor> load_model_executor(const ModelLoadConfig &config);
} // namespace qwen35
