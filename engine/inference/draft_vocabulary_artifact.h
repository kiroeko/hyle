#pragma once

#include "draft_vocabulary.h"
#include "evaluation_failure.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hyle::inference {

enum class DraftHeadPreference { kOriginal, kPrepared };

struct DraftVocabularyPreferences final {
  std::size_t recommended_drafts = 0; // Zero means no recommendation.
  DraftHeadPreference head = DraftHeadPreference::kOriginal;
};

struct DraftVocabularyArtifact final {
  DraftVocabulary vocabulary;
  DraftVocabularyPreferences preferences;
};

// Parse one bounded cold artifact and bind it to the already-selected source.
// The caller owns source roles, head compatibility and all execution resources.
inline DraftVocabularyArtifact read_draft_vocabulary_artifact(const std::filesystem::path &path,
    std::string_view source_identity, std::int64_t source_vocabulary) {
  if (source_vocabulary <= 0 || source_vocabulary > (std::numeric_limits<std::int32_t>::max)())
    throw std::runtime_error("draft vocabulary requires a positive I32 source vocabulary");
  std::error_code error;
  const auto exists = std::filesystem::exists(path, error);
  if (error) throw std::runtime_error("cannot inspect draft vocabulary: " + path.string() + ": " + error.message());
  if (!exists) throw std::runtime_error("draft vocabulary file does not exist: " + path.string());
  const auto bytes = std::filesystem::file_size(path, error);
  if (error) throw std::runtime_error("cannot size draft vocabulary: " + path.string() + ": " + error.message());
  const auto full_vocab = static_cast<std::size_t>(source_vocabulary);
  const auto maximum_bytes = std::uint64_t{256} + std::uint64_t(full_vocab) * 32;
  if (bytes > maximum_bytes || bytes > (std::numeric_limits<std::size_t>::max)() ||
      bytes > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
    throw std::runtime_error("draft vocabulary text exceeds its full-vocabulary byte bound");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open draft vocabulary: " + path.string());
  std::string text(static_cast<std::size_t>(bytes), '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!input || static_cast<std::size_t>(input.gcount()) != text.size() ||
      input.peek() != std::char_traits<char>::eof() || input.bad())
    throw std::runtime_error("draft vocabulary changed extent or failed reading");
  std::size_t cursor = 0;
  const auto line = [&](std::size_t limit) {
    if (cursor >= text.size()) throw std::runtime_error("draft vocabulary header is incomplete");
    const auto newline = text.find('\n', cursor);
    const auto end = newline == std::string::npos ? text.size() : newline;
    if (end - cursor > limit) throw std::runtime_error("draft vocabulary header exceeds its byte bound");
    auto value = text.substr(cursor, end - cursor);
    cursor = newline == std::string::npos ? text.size() : newline + 1;
    if (value.ends_with('\r')) value.pop_back();
    return value;
  };
  if (line(32) != "hyle-draft-vocabulary-v2")
    throw std::runtime_error("draft vocabulary signature differs from hyle-draft-vocabulary-v2");
  const auto identity = line(65);
  if (identity.size() != 64 || !std::ranges::all_of(identity, [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
      })) throw std::runtime_error("invalid prepared role source identity");
  if (identity != source_identity)
    throw std::runtime_error("draft vocabulary source identity differs from selected weights");
  const auto vocabulary_text = line(21);
  std::int64_t vocabulary = 0;
  const auto parsed = std::from_chars(vocabulary_text.data(), vocabulary_text.data() + vocabulary_text.size(), vocabulary);
  if (parsed.ec != std::errc{} || parsed.ptr != vocabulary_text.data() + vocabulary_text.size() ||
      vocabulary != source_vocabulary)
    throw std::runtime_error("draft vocabulary full size differs from selected model vocabulary");
  DraftVocabularyPreferences preferences;
  const auto recommendation = line(64);
  constexpr std::string_view prefix = "recommended_drafts ";
  if (!recommendation.starts_with(prefix))
    throw std::runtime_error("draft vocabulary recommendation field is invalid");
  const std::string_view depth{recommendation.data() + prefix.size(), recommendation.size() - prefix.size()};
  if (depth != "none") {
    std::size_t value = 0;
    const auto parsed_depth = std::from_chars(depth.data(), depth.data() + depth.size(), value);
    if (parsed_depth.ec != std::errc{} || parsed_depth.ptr != depth.data() + depth.size() || !value)
      throw std::runtime_error("draft vocabulary recommendation must be none or a positive draft count");
    preferences.recommended_drafts = value;
  }
  const auto head_preference = line(32);
  if (head_preference == "head prepared") preferences.head = DraftHeadPreference::kPrepared;
  else if (head_preference != "head original")
    throw std::runtime_error("draft vocabulary head preference must be original or prepared");
  std::vector<std::int32_t> ids;
  ids.reserve((std::min)(full_vocab, static_cast<std::size_t>(bytes)));
  while (cursor < text.size()) {
    while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) ++cursor;
    if (cursor == text.size()) break;
    if (ids.size() == full_vocab)
      throw std::runtime_error("draft vocabulary contains more IDs than the full vocabulary");
    const auto first = cursor;
    while (cursor < text.size() && !std::isspace(static_cast<unsigned char>(text[cursor]))) ++cursor;
    if (cursor - first > 16) throw std::runtime_error("draft vocabulary token exceeds an I32 text field");
    std::int32_t id = 0;
    const auto token = std::from_chars(text.data() + first, text.data() + cursor, id);
    if (token.ec != std::errc{} || token.ptr != text.data() + cursor)
      throw std::runtime_error("draft vocabulary contains an invalid or overflowing I32 token ID");
    ids.push_back(id);
  }
  auto mapping = DraftVocabulary::create(vocabulary, ids);
  if (!mapping) throw EvaluationFailure(static_cast<Status>(mapping.error()));
  return {std::move(*mapping), preferences};
}

} // namespace hyle::inference
