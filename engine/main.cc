#include "models/qwen35/driver.h"
#include "models/qwen35/prepared_executor.h"
#include "inference/backend.h"
#include "inference/token_worker.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
  qwen35::ModelLoadConfig config;
  config.context = 4096; config.resident_sessions = 2; config.active_sequences = 2;
  std::size_t waiting = 64, output_capacity = 16;
  for (int at = 2; at < argc; at += 2) {
    const std::string_view option(argv[at]);
    if (at + 1 == argc) throw std::invalid_argument(std::string(option) + " requires a value");
    const std::string_view value(argv[at + 1]);
    if (option == "--inputs") config.inputs = value;
    else if (option == "--backend") {
      using Kind = hyle::inference::BackendKind;
      if (value == "auto") config.backend = Kind::kDefault;
      else if (value == "cpu") config.backend = Kind::kCpu;
      else if (value == "cuda") config.backend = Kind::kCuda;
      else if (value == "apple") config.backend = Kind::kApple;
      else if (value == "vulkan") config.backend = Kind::kVulkan;
      else throw std::invalid_argument("worker backend must be auto, cpu, cuda, apple or vulkan");
    } else if (option == "--capacity") config.context = worker_number<std::size_t>(option, value);
    else if (option == "--max-sessions") config.resident_sessions = worker_number<std::size_t>(option, value);
    else if (option == "--active-sequences") config.active_sequences = worker_number<std::size_t>(option, value);
    else if (option == "--prefill-chunk-tokens") config.prefill_rows = worker_number<std::size_t>(option, value);
    else if (option == "--draft-count") config.draft_count = worker_number<std::size_t>(option, value);
    else if (option == "--draft-vocabulary") config.draft_vocabulary = value;
    else if (option == "--draft-head") config.draft_head = value;
    else if (option == "--waiting-limit") waiting = worker_number<std::size_t>(option, value);
    else if (option == "--output-capacity") output_capacity = worker_number<std::size_t>(option, value);
    else if (option == "--overlap") {
      if (value != "on" && value != "off") throw std::invalid_argument("worker overlap requires on or off");
      config.overlap = value == "on";
    } else if (option == "--sampling") {
      if (value != "greedy" && value != "random") throw std::invalid_argument("worker sampling requires greedy or random");
      config.random = value == "random";
    } else if (option == "--temperature") config.temperature = worker_number<float>(option, value);
    else if (option == "--top-p") config.top_p = worker_number<float>(option, value);
    else if (option == "--top-k") config.top_k = worker_number<std::size_t>(option, value);
    else if (option == "--seed") config.seed = worker_number<std::uint64_t>(option, value);
    else throw std::invalid_argument("unknown worker option: " + std::string(option));
  }
  auto executor = qwen35::load_model_executor(config);
  hyle::inference::TokenWorker worker(std::move(executor), config.resident_sessions,
      waiting, output_capacity, config.context, std::cin, std::cout);
  return worker.run();
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--engine-worker") {
    try { return token_worker(argc, argv); }
    catch (const std::exception &error) { std::cerr << "engine worker: " << error.what() << '\n'; return 1; }
  }
  return qwen35::run(argc, argv);
}
