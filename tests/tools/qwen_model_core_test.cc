// A real model/Graph consumer that links the model core without the CLI or
// request driver. Pass a prepared tiny local input; no weights are downloaded.
#include "../../engine/models/qwen35/model_core.h"
#include "../../engine/models/qwen35/token_ids.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace q = qwen35;
namespace i = hyle::inference;

const void *qwen_model_core_probe_names() noexcept;

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

std::string fixture_from_environment() {
#if defined(_MSC_VER)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_QWEN_CORE_INPUTS"))
    throw std::runtime_error("cannot read HYLE_QWEN_CORE_INPUTS");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto *value = std::getenv("HYLE_QWEN_CORE_INPUTS");
  return value ? value : "";
#endif
}
} // namespace

int main(int argc, char **argv) try {
  if (argc > 3) throw std::runtime_error("usage: qwen_model_core_test [PREPARED_INPUT] [--cuda]");
  const auto input = argc > 1 ? std::string(argv[1]) : fixture_from_environment();
  if (input.empty()) {
    std::cerr << "SKIP: pass prepared input or set HYLE_QWEN_CORE_INPUTS\n";
    return 77;
  }
  const auto backend = argc > 2 && std::string_view(argv[2]) == "--cuda"
      ? i::BackendKind::kCuda : i::BackendKind::kCpu;
  if (argc > 2 && std::string_view(argv[2]) != "--cuda") throw std::runtime_error("unknown backend option");
  if (std::ranges::find(i::Backend::compiled_backends(), backend) == i::Backend::compiled_backends().end()) {
    std::cerr << "SKIP: requested backend was not compiled\n";
    return 77;
  }
  const auto directory = std::filesystem::path(input);
  require(qwen_model_core_probe_names() == q::layer0_probe_names.data(),
      "model core inline constants differ between translation units");
  auto source = q::prepare_weight_source(directory / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto prompt = q::read_ids(directory / "prompt.ids", geometry.vocabulary, 16);
  require(prompt.size() >= 2, "core consumer needs at least two input IDs");
  constexpr std::size_t capacity = 16;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry));
  q::Model model(std::move(source), geometry, capacity, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, {}, 0, {.primary = backend});
  bool missing_verifier_rejected = false, missing_timer_rejected = false;
  try { model.bind_device_verification_inputs(q::GraphRole::kVerify, {}); }
  catch (const std::logic_error &) { missing_verifier_rejected = true; }
  try { model.start_device_verification_timing(); }
  catch (const std::logic_error &) { missing_timer_rejected = true; }
  require(missing_verifier_rejected && missing_timer_rejected && !model.active_step,
      "device verification must not invent a prepared target episode");
  const std::array<std::size_t, 2> chunks{2, 1};
  require(model.prepare_family(chunks), "core consumer could not prepare its model family");
  model.plan_execution(chunks);
  model.prepare_execution(chunks);
  auto session = model.create_session(chunks);
  std::int32_t next = -1;
  const auto evaluate = [&](std::span<const std::int32_t> ids) {
    model.prepare_evaluation(*session, ids);
    model.submit_evaluation(*session);
    const std::array outputs{q::EvaluationOutput{.session = session}};
    model.finish_evaluations(outputs, std::span(&next, 1),
        [](std::size_t, const q::CompletedEvaluationView &) {});
  };
  evaluate(std::span(prompt).first(2));
  require(session->length() == 2 && session->head->next_id == next && next >= 0,
      "prefill did not publish its complete state and next ID");
  const auto previous = next;
  evaluate(std::span(&previous, 1));
  require(session->length() == 3 && session->head->next_id == next && next >= 0,
      "decode did not publish its complete state and next ID");
  require(session->head->sampling == q::cache::SamplingCursor{}, "greedy core advanced RNG");
  q::check(session->stop());
  const auto retired = session->drain();
  q::check(retired.status);
  require(retired.retired, "core consumer did not retire its original session");
  session.reset();
  require(!model.active_step && !model.successor_step && !model.snapshot_versions.live,
      "core consumer retained a step or committed version");
  std::cout << "PASS Qwen model core: prefill, decode, greedy RNG and actual session retirement\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "qwen model core: " << error.what() << '\n';
  return 1;
}
