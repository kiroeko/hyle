#pragma once

#include "session_schedule.h"

namespace qwen35::benchmark {
struct MtpTurn final {
  std::vector<std::int32_t> prompt;
  std::size_t steps = 0;
};
struct MtpSessionSchedule final {
  std::string id;
  std::vector<MtpTurn> turns;
};

// Natural generation has an explicit grammar. In particular decode
// IDSFILE is never reinterpreted as an output quota or a generated token tape.
// The first non-prompt directive selects a grammar before any model is loaded;
// the selected parser still validates every line, quota and capacity bound.
inline bool schedule_generates(const std::filesystem::path &path) {
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open schedule: " + path.string());
  std::string line;
  while (std::getline(manifest, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream fields(line); std::string kind; fields >> kind;
    if (kind == "generate") return true;
    if (kind != "prompt") return false;
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading schedule: " + path.string());
  return false;
}

inline std::vector<MtpTurn> read_mtp_schedule(const std::filesystem::path &path,
    std::int64_t vocabulary, std::size_t capacity, std::size_t default_steps,
    bool explicit_steps = false) {
  if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)() ||
      !capacity || capacity > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()) ||
      !default_steps || default_steps > 128)
    throw std::runtime_error("MTP schedule requires valid vocabulary/context and a generation quota in [1,128]");
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open MTP schedule: " + path.string());
  std::vector<MtpTurn> turns;
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
    if (kind == "decode") throw error("decode IDSFILE is a fixed token tape; natural MTP uses generate COUNT");
    if (kind == "prompt" && !needs_generate) {
      if (value.find('\0') != std::string::npos) throw error("prompt path contains a null byte");
      const auto input_path = path.parent_path() / std::filesystem::path(value);
      MtpTurn turn;
      try { turn.prompt = read_schedule_ids(input_path, vocabulary, capacity); }
      catch (const std::runtime_error &exception) { throw error(exception.what()); }
      turns.push_back(std::move(turn)); needs_generate = true;
    } else if (kind == "generate" && needs_generate) {
      std::size_t count = 0;
      const auto converted = std::from_chars(value.data(), value.data() + value.size(), count);
      if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || !count || count > 128)
        throw error("generate requires an output quota in [1,128]");
      if (explicit_steps) throw error("explicit generate quotas are exclusive with --steps");
      turns.back().steps = count; needs_generate = false; has_generate = true;
    } else throw error(needs_generate ? "expected generate COUNT after prompt" : "expected prompt IDSFILE");
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading MTP schedule: " + path.string());
  if (turns.empty()) throw std::runtime_error("MTP schedule requires a prompt");
  if (needs_generate) {
    if (has_generate || turns.size() != 1) throw std::runtime_error("MTP schedule requires complete prompt/generate turns");
    turns.front().steps = default_steps; // Original one-prompt/--steps shorthand.
  }
  std::size_t maximum_committed = 0;
  for (std::size_t i = 0; i < turns.size(); ++i) {
    const auto consume = [&](std::size_t count) {
      if (count > capacity - maximum_committed) throw std::runtime_error("MTP turns plus generation quotas exceed context capacity");
      maximum_committed += count;
    };
    if (i) consume(1); // Last emitted pending from the previous turn, once.
    consume(turns[i].prompt.size()); consume(turns[i].steps - 1);
  }
  return turns;
}

inline std::vector<MtpSessionSchedule> read_mtp_session_schedules(
    const std::filesystem::path &path, std::int64_t vocabulary, std::size_t capacity,
    std::size_t max_sessions, std::size_t default_steps, bool explicit_steps = false) {
  const auto entries = read_session_manifest(path, max_sessions);
  std::vector<MtpSessionSchedule> result;
  result.reserve(entries.size());
  for (const auto &entry : entries) {
    try {
      result.push_back({entry.id, read_mtp_schedule(entry.path, vocabulary, capacity, default_steps, explicit_steps)});
    } catch (const std::runtime_error &exception) {
      throw std::runtime_error(path.string() + ":" + std::to_string(entry.line) + ": session " + entry.id + ": " + exception.what());
    }
  }
  return result;
}
} // namespace qwen35::benchmark
