#include "model_catalog.h"
#include "inference/backend.h"
#include "inference/token_worker.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
template<class T> T worker_number(std::string_view option, std::string_view value) {
  T result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::invalid_argument(std::string(option) + " requires a valid number");
  return result;
}

int token_worker(int argc, char **argv) {
  hyle::inference::ModelLoadConfig config;
  config.execution.context = 4096; config.execution.resident_sessions = 2; config.execution.active_sequences = 2;
  std::optional<std::uint64_t> default_seed;
  std::size_t waiting = 64, output_capacity = 16;
  for (int at = 2; at < argc; at += 2) {
    const std::string_view option(argv[at]);
    if (at + 1 == argc) throw std::invalid_argument(std::string(option) + " requires a value");
    const std::string_view value(argv[at + 1]);
    if (option == "--inputs") config.assets.inputs = value;
    else if (option == "--backend") {
      using Kind = hyle::inference::BackendKind;
      if (value == "auto") config.execution.backend = Kind::kDefault;
      else if (value == "cpu") config.execution.backend = Kind::kCpu;
      else if (value == "cuda") config.execution.backend = Kind::kCuda;
      else if (value == "apple") config.execution.backend = Kind::kApple;
      else if (value == "vulkan") config.execution.backend = Kind::kVulkan;
      else throw std::invalid_argument("worker backend must be auto, cpu, cuda, apple or vulkan");
    } else if (option == "--capacity") config.execution.context = worker_number<std::size_t>(option, value);
    else if (option == "--max-sessions") config.execution.resident_sessions = worker_number<std::size_t>(option, value);
    else if (option == "--active-sequences") config.execution.active_sequences = worker_number<std::size_t>(option, value);
    else if (option == "--prefill-chunk-tokens") config.execution.prefill_rows = worker_number<std::size_t>(option, value);
    else if (option == "--draft-count") config.execution.draft_count = worker_number<std::size_t>(option, value);
    else if (option == "--draft-vocabulary") config.assets.set_file("draft.vocabulary", std::filesystem::path(value));
    else if (option == "--draft-head") config.assets.set_file("draft.head", std::filesystem::path(value));
    else if (option == "--waiting-limit") waiting = worker_number<std::size_t>(option, value);
    else if (option == "--output-capacity") output_capacity = worker_number<std::size_t>(option, value);
    else if (option == "--overlap") {
      if (value != "on" && value != "off") throw std::invalid_argument("worker overlap requires on or off");
      config.execution.overlap = value == "on";
    } else if (option == "--sampling") {
      if (value != "greedy" && value != "random") throw std::invalid_argument("worker sampling requires greedy or random");
      config.sampling.random = value == "random";
    } else if (option == "--temperature") config.sampling.temperature = worker_number<float>(option, value);
    else if (option == "--top-p") config.sampling.top_p = worker_number<float>(option, value);
    else if (option == "--top-k") config.sampling.top_k = worker_number<std::size_t>(option, value);
    else if (option == "--seed") default_seed = worker_number<std::uint64_t>(option, value);
    else throw std::invalid_argument("unknown worker option: " + std::string(option));
  }
  auto executor = hyle::engine::load_model(config);
  hyle::inference::TokenWorker worker(std::move(executor), config.execution.resident_sessions,
      waiting, output_capacity, config.execution.context, std::cin, std::cout, default_seed);
  return worker.run();
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--engine-worker") {
    try { return token_worker(argc, argv); }
    catch (const std::exception &error) { std::cerr << "engine worker: " << error.what() << '\n'; return 1; }
  }
  return hyle::engine::run_legacy_cli(argc, argv);
}
