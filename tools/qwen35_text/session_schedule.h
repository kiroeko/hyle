// Consumer input handling only. Execution and session state remain with the
// original model/Invocation owners; this file does not schedule native work.
#pragma once

#include "schedule.h"

namespace qwen35::benchmark {

struct SessionSchedule {
  std::string id;
  std::vector<Phase> phases;
};

struct SessionManifestEntry {
  std::string id;
  std::filesystem::path path;
  std::size_t line;
};

// Shared identity/path validation for fixed-input and natural-generation
// schedules. The selected consumer parser retains ownership of its grammar.
inline std::vector<SessionManifestEntry> read_session_manifest(
    const std::filesystem::path &path, std::size_t max_sessions) {
  std::vector<SessionManifestEntry> entries;
  if (!max_sessions)
    throw std::runtime_error("session manifest requires a finite positive max_sessions");
  std::ifstream manifest(path);
  if (!manifest) throw std::runtime_error("cannot open session manifest: " + path.string());
  const auto alphanumeric = [](char value) {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
        (value >= '0' && value <= '9');
  };
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(manifest, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto error = [&](const std::string &message) {
      return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + message);
    };
    std::istringstream fields(line);
    std::string id, filename, extra;
    if (!(fields >> id >> std::quoted(filename)) || filename.empty() ||
        filename.find('\0') != std::string::npos || (fields >> extra))
      throw error("expected SESSION_ID SCHEDULE_FILE (quote paths containing spaces)");
    if (!alphanumeric(id.front())) throw error("session ID must start with an ASCII letter or digit");
    for (const char value : id)
      if (!alphanumeric(value) && value != '_' && value != '-')
        throw error("session ID permits only ASCII letters, digits, underscore and hyphen");
    for (const auto &entry : entries)
      if (entry.id == id) throw error("duplicate session ID: " + id);
    if (entries.size() == max_sessions || entries.size() == entries.max_size())
      throw error("session manifest exceeds max_sessions");
    entries.push_back({std::move(id), path.parent_path() / std::filesystem::path(filename), line_number});
  }
  if (!manifest.eof()) throw std::runtime_error("failed reading session manifest: " + path.string());
  if (entries.empty()) throw std::runtime_error("session manifest requires at least one session");
  return entries;
}

// Read and validate the entire manifest before constructing/loading a Model.
// Each non-comment line is SESSION_ID SCHEDULE_FILE. Schedule paths are
// relative to this manifest, and token paths remain relative to that schedule.
// The caller supplies explicit per-session context and session-count limits.
inline std::vector<SessionSchedule> read_session_schedules(
    const std::filesystem::path &path, std::int64_t vocabulary,
    std::size_t capacity, std::size_t max_sessions, bool allow_terminal_prompt = false) {
  std::vector<SessionSchedule> sessions;
  if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)() ||
      capacity == 0 || capacity > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()))
    throw std::runtime_error("session manifest requires positive vocabulary and context capacity within I32 positions");
  const auto entries = read_session_manifest(path, max_sessions);
  sessions.reserve(entries.size());
  for (const auto &entry : entries) {
    try {
      auto phases = read_schedule(entry.path, vocabulary, capacity, allow_terminal_prompt);
      sessions.push_back({entry.id, std::move(phases)});
    } catch (const std::runtime_error &exception) {
      throw std::runtime_error(path.string() + ":" + std::to_string(entry.line) + ": session " + entry.id + ": " + exception.what());
    }
  }
  return sessions;
}

} // namespace qwen35::benchmark
