#include "../../engine/inference/draft_vocabulary_artifact.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>

namespace i = hyle::inference;

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected the workspace scratch directory");
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto folder = std::filesystem::path(argv[1]) / ("draft-vocabulary-artifact-" + std::to_string(suffix));
    std::filesystem::create_directories(folder);
    const auto path = folder / "vocabulary.ids";
    const std::string identity(64, 'a');
    const auto write = [&](std::string_view value) {
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      output << value;
      if (!output) throw std::runtime_error("cannot write test artifact");
    };
    const auto text = [&](std::string_view source, std::string_view recommendation,
                          std::string_view head, std::string_view ids) {
      return "hyle-draft-vocabulary-v2\n" + std::string(source) + "\n8\nrecommended_drafts " +
          std::string(recommendation) + "\nhead " + std::string(head) + "\n" + std::string(ids) + "\n";
    };
    write(text(identity, "3", "prepared", "7 2 5 0"));
    auto selected = i::read_draft_vocabulary_artifact(path, identity, 8);
    if (!std::ranges::equal(selected.vocabulary.ids(), std::array<std::int32_t,4>{0,2,5,7}) ||
        selected.preferences.recommended_drafts != 3 || selected.preferences.head != i::DraftHeadPreference::kPrepared)
      throw std::runtime_error("validated mapping or source recommendation differs");
    write(text(identity, "none", "original", "5 1"));
    auto unspecified = i::read_draft_vocabulary_artifact(path, identity, 8);
    if (unspecified.preferences.recommended_drafts || unspecified.preferences.head != i::DraftHeadPreference::kOriginal)
      throw std::runtime_error("unspecified recommendation differs");
    const auto refuse = [&](const std::string &value) {
      write(value);
      try { (void)i::read_draft_vocabulary_artifact(path, identity, 8); }
      catch (const std::exception &) { return; }
      throw std::runtime_error("invalid vocabulary artifact was accepted");
    };
    refuse(text(std::string(64,'b'), "3", "prepared", "0 1"));
    refuse(text(identity, "0", "original", "0 1"));
    refuse(text(identity, "18446744073709551616", "original", "0 1"));
    refuse(text(identity, "none", "other", "0 1"));
    refuse(text(identity, "none", "original", "1 1"));
    refuse(text(identity, "none", "original", "8"));
    refuse("hyle-draft-vocabulary-v1\n" + identity + "\n8\n0 1\n");
    write(text(identity, "2", "original", "1 3"));
    if (i::read_draft_vocabulary_artifact(path, identity, 8).preferences.recommended_drafts != 2)
      throw std::runtime_error("valid artifact could not be read after refusals");
    std::cout << "PASS model-independent draft vocabulary source, preference and bounds\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft vocabulary artifact: " << error.what() << '\n';
    return 1;
  }
}
