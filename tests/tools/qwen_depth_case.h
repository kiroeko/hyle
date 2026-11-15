#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

// A finite real-weight consumer of the same loader/request boundary as the
// product. The synthetic token pattern fixes depth, not model quality. Ignore
// EOS explicitly through the existing request policy so every backend runs
// the same128-output quota, including its actual MTP acceptance/repair work.
namespace qwen_depth_case {
namespace q = qwen35;
namespace i = hyle::inference;

struct Tokens final {
  std::array<std::int32_t, 128> ids{};
  std::size_t count = 0, endings = 0, waits = 0, vocabulary = 0;
  bool invalid = false, cancelled = false;
  std::size_t max_pending() const noexcept { return ids.size(); }
  bool try_write(i::RequestKey, std::int32_t token) noexcept {
    invalid |= token < 0 || static_cast<std::size_t>(token) >= vocabulary || count >= ids.size();
    if (count < ids.size()) ids[count] = token;
    ++count;
    return true;
  }
  bool cancel_requested(i::RequestKey) const noexcept { return false; }
  void finished(i::RequestKey, bool stopped) noexcept { ++endings; cancelled |= stopped; }
  void wait() noexcept { ++waits; std::this_thread::yield(); }
};
static_assert(i::TokenConsumer<Tokens>);

inline int run(const std::filesystem::path &inputs, q::BackendKind backend,
               bool draft, std::size_t cpu_layers) {
  const auto start = std::chrono::steady_clock::now();
  i::ModelLoadConfig config;
  config.assets.inputs = inputs;
  config.execution = {.backend = backend, .context = 8192, .resident_sessions = 1,
      .active_sequences = 1, .prefill_rows = 512, .cpu_ffn_layers = cpu_layers};
  config.execution_profiles[0].draft_count = draft ? 3 : 0;
  auto model = q::load_model_executor(config);
  const auto description = model->describe();
  i::validate_model_description(description);
  const auto &profile = description.execution_profiles.at(0);
  if (description.context != 8192 || description.placement.primary != backend ||
      description.placement.cpu_ffn_layers != cpu_layers ||
      profile.algorithm != (draft ? "mtp" : "ordinary") ||
      profile.max_draft_tokens != (draft ? 3u : 0u) || description.vocabulary < 4)
    throw std::runtime_error("depth consumer did not load its requested native program");

  std::vector<std::int32_t> prompt(4096 + 512);
  for (std::size_t index = 0; index < prompt.size(); ++index)
    prompt[index] = static_cast<std::int32_t>(1 + index % 3);
  const std::array schedules{i::GenerationSessionSchedule{"depth4096-pp512", {{std::move(prompt), 128}}}};
  const std::array stops{i::RequestStopPolicy(std::span<const std::int32_t>{}, true)};
  const i::GenerationBatch batch{.sessions = schedules, .stop_policies = stops};
  Tokens tokens;
  tokens.vocabulary = description.vocabulary;
  i::TokenConsumerRef output(tokens);
  std::cout << "DEPTH_CONFIG context=8192 prefix=4096 pp=512 output_quota=128"
               " ignore_model_eos=1 synthetic_pattern=1,2,3 algorithm=" << profile.algorithm
            << " drafts=" << profile.max_draft_tokens << " cpu_ffn_layers=" << cpu_layers << '\n' << std::flush;
  model->validate(batch);
  model->execute(batch, output);
  if (tokens.invalid || tokens.cancelled || tokens.count != 128 || tokens.endings != 1)
    throw std::runtime_error("depth consumer lost its token quota, domain or terminal retirement");
  model.reset();
  std::cout << "DEPTH_OUTPUT_IDS";
  for (auto token : tokens.ids) std::cout << ' ' << token;
  std::cout << "\nPASS finite_depth context=8192 prefix=4096 pp=512 outputs=" << tokens.count
            << " drafts=" << (draft ? 3 : 0) << " native_owner_retired=1 consumer_waits=" << tokens.waits
            << " consumer_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << '\n' << std::flush;
  return 0;
}
} // namespace qwen_depth_case
