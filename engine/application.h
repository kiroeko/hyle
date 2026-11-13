#pragma once

#include "inference/model_provider.h"
#include "inference/token_worker.h"

#include <array>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <locale.h>
#include <type_traits>
#endif

namespace hyle::engine {

struct ApplicationOptions final {
  inference::ModelLoadConfig model;
  std::optional<std::uint64_t> seed;
  std::size_t waiting = 64, output_capacity = 16, output_limit = 64;
  std::size_t execution_profile = 0, sampling_index = 0;
  std::filesystem::path tokens;
  bool worker = false, help = false;
};

template<class T> T application_number(std::string_view option, std::string_view value) {
  T result{};
  bool valid = false;
#if defined(__APPLE__)
  if constexpr (std::is_floating_point_v<T>) {
    // libc++ floating from_chars requires macOS26, above our deployment floor.
    // Parse directly to the requested precision in a private C numeric locale.
    // No process locale change or double rounding through an intermediate.
    const std::string text(value);
    const auto locale = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
    if (!locale) throw std::runtime_error("C numeric locale is unavailable");
    char *end = nullptr;
    errno = 0;
    if constexpr (std::is_same_v<T, float>) result = ::strtof_l(text.c_str(), &end, locale);
    else if constexpr (std::is_same_v<T, double>) result = ::strtod_l(text.c_str(), &end, locale);
    else result = ::strtold_l(text.c_str(), &end, locale);
    const auto range_error = errno == ERANGE;
    ::freelocale(locale);
    auto magnitude = value;
    if (magnitude.starts_with('-')) magnitude.remove_prefix(1);
    // Match general from_chars: no leading whitespace/plus or hexadecimal.
    valid = !value.empty() && value.front() != '+' && value.find_first_of(" \t\r\n\f\v") != 0 &&
        !magnitude.starts_with("0x") && !magnitude.starts_with("0X") && end == text.data() + text.size() &&
        !(range_error && (result == 0 || !std::isfinite(result)));
  } else
#endif
  {
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    valid = parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
  }
  if (!valid)
    throw std::invalid_argument(std::string(option) + " requires a valid number");
  return result;
}

inline inference::PreparedSamplingPolicy application_sampling(std::string_view value) {
  if (value == "greedy") return {};
  if (value == "random") return {.random = true};
  if (!value.starts_with("random,"))
    throw std::invalid_argument("additional sampling requires greedy, random or random,temperature,top-k,top-p");
  value.remove_prefix(7);
  std::array<std::string_view, 3> parts;
  for (std::size_t index = 0; index < parts.size(); ++index) {
    const auto separator = value.find(',');
    if ((index + 1 < parts.size() && separator == std::string_view::npos) ||
        (index + 1 == parts.size() && separator != std::string_view::npos))
      throw std::invalid_argument("additional random sampling requires temperature, top-k and top-p");
    parts[index] = value.substr(0, separator);
    if (separator != std::string_view::npos) value.remove_prefix(separator + 1);
  }
  return {.random = true, .temperature = application_number<float>("additional temperature", parts[0]),
          .top_p = application_number<float>("additional top-p", parts[2]),
          .top_k = application_number<std::size_t>("additional top-k", parts[1])};
}

inline ApplicationOptions parse_application_options(int argc, char **argv) {
  ApplicationOptions options;
  options.model.execution.context = 4096;
  int begin = 1;
  if (argc > 1 && std::string_view(argv[1]) == "--engine-worker") { options.worker = true; begin = 2; }
  else if (argc > 1 && std::string_view(argv[1]) == "generate") begin = 2;
  options.model.execution.resident_sessions = options.model.execution.active_sequences = options.worker ? 2 : 1;
  bool request_options = false;
  for (int at = begin; at < argc; at += 2) {
    const std::string_view option(argv[at]);
    if (option == "--help" || option == "-h") { options.help = true; return options; }
    if (at + 1 == argc) throw std::invalid_argument(std::string(option) + " requires a value");
    const std::string_view value(argv[at + 1]);
    auto &config = options.model;
    auto &profile = config.execution_profiles.back();
    if (option == "--inputs") config.assets.inputs = value;
    else if (option == "--backend") {
      using Kind = inference::BackendKind;
      if (value == "auto") config.execution.backend = Kind::kDefault;
      else if (value == "cpu") config.execution.backend = Kind::kCpu;
      else if (value == "cuda") config.execution.backend = Kind::kCuda;
      else if (value == "apple") config.execution.backend = Kind::kApple;
      else if (value == "vulkan") config.execution.backend = Kind::kVulkan;
      else throw std::invalid_argument("backend must be auto, cpu, cuda, apple or vulkan");
    } else if (option == "--capacity") config.execution.context = application_number<std::size_t>(option, value);
    else if (option == "--max-sessions") config.execution.resident_sessions = application_number<std::size_t>(option, value);
    else if (option == "--active-sequences") config.execution.active_sequences = application_number<std::size_t>(option, value);
    else if (option == "--prefill-chunk-tokens") config.execution.prefill_rows = application_number<std::size_t>(option, value);
    else if (option == "--prefix-cache-entries") config.execution.prefix_cache_entries = application_number<std::size_t>(option, value);
    else if (option == "--draft-count") profile.draft_count = application_number<std::size_t>(option, value);
    else if (option == "--overlap") {
      if (value != "on" && value != "off") throw std::invalid_argument("overlap requires on or off");
      profile.overlap = value == "on";
    } else if (option == "--profile") config.execution_profiles.push_back({.name = std::string(value)});
    else if (option == "--asset" || option == "--profile-asset") {
      const auto equal = value.find('=');
      if (equal == std::string_view::npos || !equal || equal + 1 == value.size())
        throw std::invalid_argument("asset requires role=path");
      auto role = std::string(value.substr(0, equal));
      const std::filesystem::path path(value.substr(equal + 1));
      if (option == "--asset") config.assets.set_file(std::move(role), path);
      else {
        auto found = std::find_if(profile.assets.begin(), profile.assets.end(), [&](const auto &asset) { return asset.role == role; });
        inference::ModelAsset asset{std::move(role), inference::ModelAssetMode::kFile, path};
        if (found == profile.assets.end()) profile.assets.push_back(std::move(asset)); else *found = std::move(asset);
      }
    } else if (option == "--draft-vocabulary") config.assets.set_file("draft.vocabulary", std::filesystem::path(value));
    else if (option == "--draft-head") config.assets.set_file("draft.head", std::filesystem::path(value));
    else if (option == "--waiting-limit") options.waiting = application_number<std::size_t>(option, value);
    else if (option == "--output-capacity") options.output_capacity = application_number<std::size_t>(option, value);
    else if (option == "--sampling") {
      if (value != "greedy" && value != "random") throw std::invalid_argument("sampling requires greedy or random");
      config.sampling.random = value == "random";
    } else if (option == "--temperature") config.sampling.temperature = application_number<float>(option, value);
    else if (option == "--top-p") config.sampling.top_p = application_number<float>(option, value);
    else if (option == "--top-k") config.sampling.top_k = application_number<std::size_t>(option, value);
    else if (option == "--additional-sampling") config.additional_sampling.push_back(application_sampling(value));
    else if (option == "--seed") options.seed = application_number<std::uint64_t>(option, value);
    else if (option == "--tokens") { options.tokens = value; request_options = true; }
    else if (option == "--max-output-tokens") { options.output_limit = application_number<std::size_t>(option, value); request_options = true; }
    else if (option == "--execution-profile") { options.execution_profile = application_number<std::size_t>(option, value); request_options = true; }
    else if (option == "--sampling-index") { options.sampling_index = application_number<std::size_t>(option, value); request_options = true; }
    else throw std::invalid_argument("unknown option: " + std::string(option));
  }
  if (options.worker && request_options)
    throw std::invalid_argument("worker request options belong in token frames, not startup arguments");
  if (!options.worker && (options.tokens.empty() || !options.output_limit))
    throw std::invalid_argument("generation requires --tokens FILE|- and a positive --max-output-tokens");
  return options;
}

inline int run_application(std::span<const inference::ModelProvider> providers, int argc, char **argv,
                           std::istream &input, std::ostream &output, std::ostream &error) {
  try {
    const auto options = parse_application_options(argc, argv);
    if (options.help) {
      output << "Hyle text inference\nUsage: hyle generate --inputs SOURCE --tokens FILE|- [options]\n"
          "  --backend auto|cpu|cuda|apple|vulkan  --capacity N\n"
          "  --max-sessions N  --active-sequences N  --max-output-tokens N\n"
          "  --sampling greedy|random  --temperature T  --top-k N  --top-p P  --seed N\n"
          "  --additional-sampling greedy|random|random,T,K,P  --sampling-index N\n"
          "  --draft-count N  --draft-vocabulary FILE  --draft-head FILE\n"
          "  --asset ROLE=FILE  --profile NAME  --profile-asset ROLE=FILE\n"
          "  --execution-profile N  --overlap on|off  --prefill-chunk-tokens N\n"
          "  --prefix-cache-entries N  (complete-state prefix directory; default 0)\n"
          "  --engine-worker starts the internal token transport with the same model preparation.\n"
          "Model replay and tracing use their explicit diagnostic targets.\n";
      return 0;
    }
    const auto &config = options.model;
    auto executor = inference::load_model(providers, config);
    if (options.worker) {
      inference::TokenWorker worker(std::move(executor), config.execution.resident_sessions,
          options.waiting, options.output_capacity, config.execution.context, input, output, options.seed);
      return worker.run();
    }
    std::ifstream file;
    if (options.tokens != "-") { file.open(options.tokens); if (!file) throw std::invalid_argument("cannot open token input"); }
    std::istream &tokens = options.tokens == "-" ? input : file;
    std::vector<std::int32_t> prompt;
    std::string token_text;
    const auto description = executor->describe();
    const auto vocabulary = description.vocabulary;
    if (!vocabulary || !description.context) throw std::invalid_argument("loaded text model has invalid limits");
    while (tokens >> token_text) {
      const auto token = application_number<std::int64_t>("input token", token_text);
      if (token < 0 || static_cast<std::uint64_t>(token) >= vocabulary)
        throw std::invalid_argument("input token is outside the loaded model vocabulary");
      if (prompt.size() >= description.context) throw std::invalid_argument("input exceeds the loaded model context");
      prompt.push_back(static_cast<std::int32_t>(token));
    }
    if (!tokens.eof() || prompt.empty()) throw std::invalid_argument("token input requires a nonempty list of integer IDs");
    inference::GenerationDispatcher dispatcher(std::move(executor), config.execution.resident_sessions,
        options.waiting, options.output_capacity);
    inference::GenerationSessionSchedule schedule{"cli", {{std::move(prompt), options.output_limit}}};
    schedule.sampling_seed = options.seed; schedule.execution_profile = options.execution_profile;
    auto request = dispatcher.submit(std::move(schedule), {}, options.sampling_index);
    bool first = true;
    for (;;) {
      const auto event = request.read();
      using Kind = inference::GenerationRequestHandle::EventKind;
      if (event.kind == Kind::kToken) { output << (first ? "" : " ") << event.token; first = false; }
      else if (event.kind == Kind::kCompleted) { output << '\n'; return 0; }
      else if (event.failure) std::rethrow_exception(event.failure);
      else throw std::runtime_error("generation did not complete");
      if (!output) throw std::runtime_error("token output failed");
    }
  } catch (const std::exception &failure) { error << "hyle: " << failure.what() << '\n'; return 1; }
}

} // namespace hyle::engine
