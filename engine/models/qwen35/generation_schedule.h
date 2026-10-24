#pragma once

#include "session_schedule.h"
#include "../../inference/request_progress.h"
#include <string_view>

namespace qwen35::benchmark {
enum class ScheduleGrammar { kPromptOnly, kGeneration, kFixedInput };

namespace generation_schedule_detail {
inline std::size_t output_quota(std::string_view value) {
  std::size_t count = 0;
  const auto converted = std::from_chars(value.data(), value.data() + value.size(), count);
  if (converted.ec == std::errc::result_out_of_range)
    throw std::runtime_error("generate output quota exceeds the supported size range");
  if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || !count)
    throw std::runtime_error("generate requires a positive integer output quota");
  return count;
}
} // namespace generation_schedule_detail

// Inspect every directive before choosing an execution grammar. A single
// prompt is neutral: it uses natural defaults unless a session peer explicitly
// selects fixed decode inputs. Token files are read by the selected parser.
inline ScheduleGrammar schedule_grammar(const std::filesystem::path &path) {
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open schedule: " + path.string());
  ScheduleGrammar grammar = ScheduleGrammar::kPromptOnly;
  bool needs_output = false, has_prompt = false;
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(manifest, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto error = [&](const std::string &message) {
      return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + message);
    };
    std::istringstream fields(line); std::string kind, value, extra;
    if (!(fields >> kind >> std::quoted(value)) || value.empty() || (fields >> extra))
      throw error("expected prompt IDSFILE, generate COUNT or decode IDSFILE");
    if (kind == "prompt") {
      if (needs_output) throw error("expected generate COUNT or decode IDSFILE after prompt");
      if (value.find('\0') != std::string::npos) throw error("prompt path contains a null byte");
      needs_output = true; has_prompt = true;
    } else if (kind == "generate" || kind == "decode") {
      if (!needs_output) throw error("expected prompt IDSFILE before output directive");
      const auto selected = kind == "generate" ? ScheduleGrammar::kGeneration : ScheduleGrammar::kFixedInput;
      if (grammar != ScheduleGrammar::kPromptOnly && grammar != selected)
        throw error("cannot mix generate quotas with fixed decode token tapes");
      if (kind == "generate") {
        try { (void)generation_schedule_detail::output_quota(value); }
        catch (const std::runtime_error &exception) { throw error(exception.what()); }
      } else if (value.find('\0') != std::string::npos) throw error("decode path contains a null byte");
      grammar = selected; needs_output = false;
    } else throw error("expected prompt IDSFILE, generate COUNT or decode IDSFILE");
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading schedule: " + path.string());
  if (!has_prompt) throw std::runtime_error("schedule requires a prompt: " + path.string());
  if (needs_output && grammar == ScheduleGrammar::kGeneration)
    throw std::runtime_error("generation schedule requires complete prompt/generate turns: " + path.string());
  return grammar;
}

namespace generation_schedule_detail {
inline ScheduleGrammar session_grammar(const std::filesystem::path &path,
    const std::vector<SessionManifestEntry> &entries) {
  auto grammar = ScheduleGrammar::kPromptOnly;
  for (const auto &entry : entries) {
    try {
      const auto selected = schedule_grammar(entry.path);
      if (selected == ScheduleGrammar::kPromptOnly) continue;
      if (grammar != ScheduleGrammar::kPromptOnly && grammar != selected)
        throw std::runtime_error("cannot mix generate quotas with fixed decode token tapes across sessions");
      grammar = selected;
    } catch (const std::runtime_error &exception) {
      throw std::runtime_error(path.string() + ":" + std::to_string(entry.line) + ": session " + entry.id + ": " + exception.what());
    }
  }
  return grammar;
}
} // namespace generation_schedule_detail

inline ScheduleGrammar session_schedule_grammar(const std::filesystem::path &path,
    std::size_t max_sessions) {
  return generation_schedule_detail::session_grammar(path, read_session_manifest(path, max_sessions));
}

// decode IDSFILE is never reinterpreted as an output quota or generated tape.
inline std::vector<hyle::inference::GenerationTurn> read_generation_schedule(const std::filesystem::path &path,
    std::int64_t vocabulary, std::size_t capacity, std::size_t default_outputs,
    bool explicit_steps = false) {
  if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)() ||
      !capacity || capacity > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()) ||
      !default_outputs)
    throw std::runtime_error("generation schedule requires positive vocabulary/context within I32 positions and a positive generation quota");
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open generation schedule: " + path.string());
  std::vector<hyle::inference::GenerationTurn> turns;
  std::size_t line_number = 0;
  bool needs_generate = false, has_generate = false;
  std::string line;
  while (std::getline(manifest, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto error = [&](const std::string &message) {
      return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + message);
    };
    std::istringstream fields(line);
    std::string kind, value, extra;
    if (!(fields >> kind >> std::quoted(value)) || value.empty() || (fields >> extra))
      throw error("expected prompt IDSFILE or generate COUNT");
    if (kind == "decode") throw error("decode IDSFILE is a fixed token tape; natural generation uses generate COUNT");
    if (kind == "prompt" && !needs_generate) {
      if (value.find('\0') != std::string::npos) throw error("prompt path contains a null byte");
      const auto input_path = path.parent_path() / std::filesystem::path(value);
      hyle::inference::GenerationTurn turn;
      // A later turn consumes its actual preceding pending token even when
      // the user supplies no new suffix. The initial prompt remains nonempty.
      try { turn.prompt = read_schedule_ids(input_path, vocabulary, capacity, !turns.empty()); }
      catch (const std::runtime_error &exception) { throw error(exception.what()); }
      turns.push_back(std::move(turn)); needs_generate = true;
    } else if (kind == "generate" && needs_generate) {
      std::size_t count;
      try { count = generation_schedule_detail::output_quota(value); }
      catch (const std::runtime_error &exception) { throw error(exception.what()); }
      if (explicit_steps) throw error("explicit generate quotas are exclusive with --steps");
      turns.back().max_outputs = count; needs_generate = false; has_generate = true;
    } else throw error(needs_generate ? "expected generate COUNT after prompt" : "expected prompt IDSFILE");
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading generation schedule: " + path.string());
  if (turns.empty()) throw std::runtime_error("generation schedule requires a prompt");
  if (needs_generate) {
    if (has_generate || turns.size() != 1) throw std::runtime_error("generation schedule requires complete prompt/generate turns");
    turns.front().max_outputs = default_outputs; // Original one-prompt/--steps shorthand.
  }
  std::size_t maximum_committed = 0;
  for (std::size_t i = 0; i < turns.size(); ++i) {
    // A turn emits its first token at prefill. Its remaining steps consume
    // context, and a later turn consumes the preceding pending token once.
    // Compare each positive extent before adding: even SIZE_MAX quotas fail
    // against the real remaining capacity without wrapping the total.
    const auto consume = [&](std::size_t count) {
      if (count > capacity - maximum_committed) throw std::runtime_error("generation turns plus quotas exceed context capacity");
      maximum_committed += count;
    };
    if (i) consume(1); // Last emitted pending from the previous turn, once.
    consume(turns[i].prompt.size()); consume(turns[i].max_outputs - 1);
  }
  return turns;
}

inline std::vector<hyle::inference::GenerationSessionSchedule> read_generation_session_schedules(
    const std::filesystem::path &path, std::int64_t vocabulary, std::size_t capacity,
    std::size_t max_sessions, std::size_t default_outputs, bool explicit_steps = false) {
  const auto entries = read_session_manifest(path, max_sessions);
  if (generation_schedule_detail::session_grammar(path, entries) == ScheduleGrammar::kFixedInput)
    throw std::runtime_error("natural generation sessions require generate COUNT; fixed decode token tapes use the fixed-input parser");
  std::vector<hyle::inference::GenerationSessionSchedule> result;
  result.reserve(entries.size());
  for (const auto &entry : entries) {
    try {
      result.push_back({entry.id, read_generation_schedule(entry.path, vocabulary, capacity, default_outputs, explicit_steps)});
    } catch (const std::runtime_error &exception) {
      throw std::runtime_error(path.string() + ":" + std::to_string(entry.line) + ": session " + entry.id + ": " + exception.what());
    }
  }
  return result;
}
} // namespace qwen35::benchmark
