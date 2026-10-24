// Standard-library-only parser regression. No model, SDK, or device is loaded.
// Build this single translation unit and pass a scratch directory to the test.
#include "../../engine/models/qwen35/generation_schedule.h"

#include <array>
#include <cstdio>
#include <string_view>
#include <type_traits>

namespace {
namespace b = qwen35::benchmark;
namespace fs = std::filesystem;
namespace inference = hyle::inference;

static_assert(std::is_same_v<decltype(b::read_generation_schedule(fs::path{}, 10, 10, 4)),
    std::vector<inference::GenerationTurn>>);
static_assert(std::is_same_v<decltype(b::read_generation_session_schedules(fs::path{}, 10, 10, 2, 4)),
    std::vector<inference::GenerationSessionSchedule>>);

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void write(const fs::path &path, std::string_view contents) {
  std::ofstream output(path);
  output << contents;
  require(static_cast<bool>(output), "cannot write schedule parser fixture");
}

void host_progress_keeps_reserved_outputs_and_borrowed_turns() {
  const std::array turns{inference::GenerationTurn{{1, 2}, 3}, inference::GenerationTurn{{3}, 2}};
  inference::RequestProgress progress(5);
  progress.bind_turns(turns);
  progress.initial_sampling = {123, 7, 0};
  const auto capacity = progress.generated.capacity();
  const auto initial = progress.initial_sampling;
  std::array<std::int32_t, 2> first_input{1, 2};
  progress.begin_turn(first_input);
  require(progress.turn_index == 0 && progress.turn_output_begin == 0 &&
      !progress.prefilled && !progress.turn_done, "initial host turn progress is inconsistent");
  const std::array<std::int32_t, 3> first_output{36, 8, 22};
  progress.observe_commit(0, first_output, true, true, false);
  const auto *storage = progress.generated.data();
  progress.advance_turn();
  std::array<std::int32_t, 2> second_input{-1, 3};
  progress.begin_turn(second_input, 22);
  require(progress.turn_index == 1 && progress.turn_output_begin == 3 &&
      progress.generated == std::vector<std::int32_t>{36, 8, 22} &&
      !progress.prefilled && !progress.turn_done, "beginning the next turn erased outputs or retained old turn flags");
  const std::array<std::int32_t, 2> second_output{19, 23};
  progress.observe_commit(1, second_output, true, true, false);
  require(progress.generated.data() == storage && progress.generated.capacity() == capacity &&
      progress.turns().data() == turns.data() && progress.turns()[1].prompt == std::vector<std::int32_t>{3} &&
      progress.initial_sampling == initial,
      "host progress reallocated reserved outputs, copied turn ownership or advanced the model RNG");
}

template<class Read>
void rejects(Read &&read, std::string_view diagnostic) {
  try { read(); }
  catch (const std::runtime_error &error) {
    if (std::string_view(error.what()).find(diagnostic) != std::string_view::npos) return;
    throw std::runtime_error(std::string("unexpected parser diagnostic: ") + error.what());
  }
  throw std::runtime_error("invalid schedule was accepted");
}

void positive_quotas_use_the_real_context(const fs::path &directory) {
  const auto explicit_turn = directory / "explicit.turns";
  write(explicit_turn, "prompt two.ids\ngenerate 129\n");
  const auto parsed = b::read_generation_schedule(explicit_turn, 10, 130, 4);
  require(parsed.size() == 1 && parsed[0].prompt == std::vector<std::int32_t>{1, 2} &&
      parsed[0].max_outputs == 129, "a valid explicit quota above 128 was rejected or altered");
  require(b::schedule_grammar(explicit_turn) == b::ScheduleGrammar::kGeneration, "natural generation grammar was lost");
  rejects([&] { (void)b::read_generation_schedule(explicit_turn, 10, 129, 4); }, "exceed context capacity");
  rejects([&] { (void)b::read_generation_schedule(explicit_turn, 10, 130, 4, true); }, "exclusive with --steps");

  const auto shorthand = directory / "default.turns";
  write(shorthand, "prompt two.ids\n");
  const auto defaults = b::read_generation_schedule(shorthand, 10, 130, 129, true);
  require(defaults.size() == 1 && defaults[0].max_outputs == 129,
      "a valid shorthand --steps quota above 128 was rejected or altered");
  rejects([&] { (void)b::read_generation_schedule(shorthand, 10, 129, 129); }, "exceed context capacity");
}

void multiturn_and_session_bounds_count_pending_once(const fs::path &directory) {
  const auto turns = directory / "two-turns.turns";
  write(turns, "prompt two.ids\ngenerate 129\nprompt two.ids\ngenerate 129\n");
  // Two prompts + 258 outputs - the last unconsumed pending = 261 rows.
  const auto parsed = b::read_generation_schedule(turns, 10, 261, 4);
  require(parsed.size() == 2 && parsed[0].max_outputs == 129 && parsed[1].max_outputs == 129,
      "two-turn quotas lost their actual per-turn output counts");
  rejects([&] { (void)b::read_generation_schedule(turns, 10, 260, 4); }, "exceed context capacity");
  const auto sessions = directory / "sessions.txt";
  write(sessions, "a two-turns.turns\nb explicit.turns\n");
  const auto multiple = b::read_generation_session_schedules(sessions, 10, 261, 2, 4);
  require(multiple.size() == 2 && multiple[0].turns.size() == 2 &&
      multiple[1].turns[0].max_outputs == 129, "session wrapper altered a valid large quota");
  rejects([&] { (void)b::read_generation_session_schedules(sessions, 10, 260, 2, 4); }, "session a:");
}

void quota_and_position_overflow_is_rejected(const fs::path &directory) {
  const auto path = directory / "limits.turns";
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  write(path, "prompt two.ids\ngenerate " + std::to_string(maximum) + "\n");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 130, 4); }, "exceed context capacity");
  write(path, "prompt two.ids\ngenerate " + std::to_string(maximum) + "0\n");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 130, 4); }, "supported size range");
  write(path, "prompt two.ids\n");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 130, maximum); }, "exceed context capacity");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 130, 0); }, "positive generation quota");

  const auto positions = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  write(path, "prompt one.ids\ngenerate " + std::to_string(positions) + "\n");
  // Parsing the quota never allocates a generated-token array of that size.
  const auto bounded = b::read_generation_schedule(path, 10, positions, 4);
  require(bounded.size() == 1 && bounded[0].max_outputs == positions,
      "I32 context limit acquired an unrelated artificial output bound");
  rejects([&] { (void)b::read_generation_schedule(path, 10, positions + 1, 4); }, "within I32 positions");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 0, 4); }, "within I32 positions");
}

void invalid_quota_syntax_still_fails_before_execution(const fs::path &directory) {
  const auto path = directory / "invalid.turns";
  for (const auto quota : {"0", "-1", "129x"}) {
    write(path, std::string("prompt two.ids\ngenerate ") + quota + "\n");
    rejects([&] { (void)b::read_generation_schedule(path, 10, 130, 4); }, "positive integer output quota");
  }
  write(path, "prompt two.ids\ndecode one.ids\n");
  rejects([&] { (void)b::read_generation_schedule(path, 10, 130, 4); }, "natural generation uses generate");
}

void whole_grammar_is_checked_before_selection(const fs::path &directory) {
  const auto path = directory / "grammar.turns";
  write(path, "prompt missing.ids\ngenerate 1\nprompt missing.ids\ndecode missing.ids\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "cannot mix generate quotas with fixed decode");
  write(path, "prompt missing.ids\ndecode missing.ids\nprompt missing.ids\ngenerate 1\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "cannot mix generate quotas with fixed decode");
  write(path, "prompt missing.ids\ngenerate 1\nprompt missing.ids\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "complete prompt/generate turns");
  write(path, "prompt missing.ids\ngenerate 1\nunknown extra\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "expected prompt IDSFILE");
  write(path, "prompt missing.ids\ngenerate 1\nprompt missing.ids\ngenerate 0\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "positive integer output quota");
  write(path, "prompt missing.ids\nprompt missing.ids\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "after prompt");
  write(path, "\n# no prompt\r\n");
  rejects([&] { (void)b::schedule_grammar(path); }, "requires a prompt");
}

void session_grammar_and_neutral_prompts(const fs::path &directory) {
  const auto plain = directory / "neutral.turns";
  const auto natural = directory / "natural.turns";
  const auto fixed = directory / "fixed.turns";
  const auto sessions = directory / "grammar-sessions.txt";
  write(plain, "# prompt-only keeps its default quota\r\nprompt \"two tokens.ids\"\r\n");
  write(natural, "prompt one.ids\ngenerate 2\n");
  write(fixed, "prompt one.ids\ndecode two.ids\n");
  require(b::schedule_grammar(plain) == b::ScheduleGrammar::kPromptOnly, "prompt-only is not neutral");
  require(b::schedule_grammar(fixed) == b::ScheduleGrammar::kFixedInput, "fixed decode grammar was lost");
  write(sessions, "neutral neutral.turns\n");
  require(b::session_schedule_grammar(sessions, 2) == b::ScheduleGrammar::kPromptOnly,
      "all prompt-only sessions should remain neutral");
  const auto defaults = b::read_generation_session_schedules(sessions, 10, 20, 2, 4, true);
  require(defaults.size() == 1 && defaults[0].turns[0].prompt == std::vector<std::int32_t>{1, 2} &&
      defaults[0].turns[0].max_outputs == 4, "neutral shorthand lost quoted relative inputs/default quota");
  write(sessions, "neutral neutral.turns\nnatural natural.turns\n");
  require(b::session_schedule_grammar(sessions, 2) == b::ScheduleGrammar::kGeneration,
      "natural peer did not select generation for neutral prompt");
  const auto generated = b::read_generation_session_schedules(sessions, 10, 20, 2, 4);
  require(generated[0].turns[0].max_outputs == 4 && generated[1].turns[0].max_outputs == 2,
      "session grammar selection altered per-session quotas");
  write(sessions, "neutral neutral.turns\nfixed fixed.turns\n");
  require(b::session_schedule_grammar(sessions, 2) == b::ScheduleGrammar::kFixedInput,
      "fixed peer did not select fixed inputs for neutral prompt");
  const auto tapes = b::read_session_schedules(sessions, 10, 20, 2, true);
  require(tapes[0].phases.size() == 1 && tapes[1].phases.size() == 2,
      "neutral terminal prompt changed the fixed-input parser");
  rejects([&] { (void)b::read_generation_session_schedules(sessions, 10, 20, 2, 4); }, "fixed decode token tapes");
  // Missing token paths prove cross-session grammar is checked before reading IDs.
  write(natural, "prompt missing.ids\ngenerate 1\n");
  write(fixed, "prompt missing.ids\ndecode missing.ids\n");
  for (const auto order : {"a natural.turns\nb fixed.turns\n", "a fixed.turns\nb natural.turns\n"}) {
    write(sessions, order);
    rejects([&] { (void)b::session_schedule_grammar(sessions, 2); }, "across sessions");
    rejects([&] { (void)b::read_generation_session_schedules(sessions, 10, 20, 2, 4); }, "across sessions");
  }
  write(sessions, "a natural.turns\nb grammar.turns\n");
  rejects([&] { (void)b::session_schedule_grammar(sessions, 2); }, "session b:");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("pass the scratch fixture directory");
    const fs::path directory = argv[1];
    fs::create_directories(directory);
    write(directory / "one.ids", "1\n");
    write(directory / "two.ids", "1 2\n");
    write(directory / "two tokens.ids", "1 2\n");
    host_progress_keeps_reserved_outputs_and_borrowed_turns();
    positive_quotas_use_the_real_context(directory);
    multiturn_and_session_bounds_count_pending_once(directory);
    quota_and_position_overflow_is_rejected(directory);
    invalid_quota_syntax_still_fails_before_execution(directory);
    whole_grammar_is_checked_before_selection(directory);
    session_grammar_and_neutral_prompts(directory);
    std::puts("PASS host progress/capacity/turn ownership, generation quotas/context/pending, size/I32 limits, full grammar/session preflight, neutral prompts, fixed tape separation");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
