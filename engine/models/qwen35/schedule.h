// Fixed-input benchmark schedule shared by the two test consumers.
// This is consumer input handling, not a Compute/Graph execution interface.
#pragma once

#include "token_ids.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace qwen35::benchmark {

enum class PhaseKind { kPrompt, kDecode };

struct Phase {
  PhaseKind kind;
  std::vector<std::int32_t> ids;
  std::size_t turn;
};

// Both schedule grammars use the same bounded token-file contract. The
// caller adds its manifest line/phase context to these input diagnostics.
inline std::vector<std::int32_t> read_schedule_ids(const std::filesystem::path &path,
    std::int64_t vocabulary, std::size_t max_rows, bool allow_empty = false) {
  std::vector<std::int32_t> ids;
  try { ids = qwen35::read_ids(path, vocabulary, max_rows); }
  catch (const std::runtime_error &exception) {
    throw std::runtime_error(std::string(exception.what()) + ": " + path.string());
  }
  if (ids.empty() && !allow_empty)
    throw std::runtime_error("phase requires nonempty token IDs: " + path.string());
  return ids;
}

inline std::vector<Phase> read_schedule(const std::filesystem::path &path,
    std::int64_t vocabulary, std::size_t capacity, bool allow_terminal_prompt = false) {
  if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)() ||
      capacity == 0 || capacity > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()))
    throw std::runtime_error("schedule requires positive vocabulary and context capacity within I32 positions");
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open schedule: " + path.string());
  std::vector<Phase> phases;
  std::size_t total = 0, line_number = 0, turn = 0;
  std::string line;
  while (std::getline(manifest, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto error = [&](const std::string &message) {
      return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + message);
    };
    std::istringstream fields(line);
    std::string kind, filename, extra;
    if (!(fields >> kind >> std::quoted(filename)) || filename.empty() || (fields >> extra))
      throw error("expected prompt IDSFILE or decode IDSFILE (quote paths containing spaces)");
    const auto expected = phases.size() % 2 == 0 ? "prompt" : "decode";
    if (kind != expected)
      throw error(std::string("expected ") + expected + "; each turn is prompt followed by decode");
    const auto input_path = path.parent_path() / std::filesystem::path(filename);
    Phase phase{kind == "prompt" ? PhaseKind::kPrompt : PhaseKind::kDecode, {}, turn};
    if (phase.kind == PhaseKind::kPrompt) phase.turn = ++turn;
    else phase.turn = turn;
    try { phase.ids = read_schedule_ids(input_path, vocabulary, capacity - total); }
    catch (const std::runtime_error &exception) { throw error(exception.what()); }
    total += phase.ids.size();
    phases.push_back(std::move(phase));
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading schedule: " + path.string());
  if (phases.empty() || (!allow_terminal_prompt && phases.back().kind != PhaseKind::kDecode))
    throw std::runtime_error("schedule requires complete prompt/decode turns");
  return phases;
}

// local_index is zero based, including single-token decode inputs.
inline std::string phase_name(const Phase &phase, std::size_t local_index) {
  const auto prefix = "round" + std::to_string(phase.turn);
  return phase.kind == PhaseKind::kPrompt ? prefix + "_prefill"
      : prefix + "_decode" + std::to_string(local_index + 1);
}

} // namespace qwen35::benchmark
