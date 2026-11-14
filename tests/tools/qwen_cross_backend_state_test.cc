// A bounded complete-state consumer, reusing the existing host-image checks.
// No CLI, model diagnostics, fixture generation or download is included.
#include "../../engine/models/qwen35/generation_runtime.cc"
#include "qwen_host_image_case.h"

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error(
        "usage: qwen_cross_backend_state_test cpu|cuda|apple|vulkan INPUTS off|K3 CPU_FFN_LAYERS");
    const std::string_view name(argv[1]), mode(argv[3]);
    auto kind = qwen35::BackendKind::kDefault;
    if (name == "cpu") kind = qwen35::BackendKind::kCpu;
    else if (name == "cuda") kind = qwen35::BackendKind::kCuda;
    else if (name == "apple") kind = qwen35::BackendKind::kApple;
    else if (name == "vulkan") kind = qwen35::BackendKind::kVulkan;
    else throw std::runtime_error("unknown native backend");
    if (mode != "off" && mode != "K3") throw std::runtime_error("expected ordinary or K3 state consumer");
    std::size_t consumed = 0;
    const auto layers = std::stoull(argv[4], &consumed);
    if (consumed != std::string_view(argv[4]).size()) throw std::runtime_error("invalid CPU placement");
    for (const bool random : {false, true}) {
      const auto report = qwen_host_image_case::run_mode(argv[2], kind, mode == "K3", 3, layers, random);
      std::cout << "PASS complete_state backend=" << name << " cpu_ffn_layers=" << layers
                << " algorithm=" << mode << " sampling=" << (random ? "random" : "greedy")
                << " payloads=" << report.payloads << " bytes=" << report.total_bytes
                << " versions=0 images=0 historical_boundary=preserved restored_continuation=equal\n" << std::flush;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "complete state consumer failed: " << error.what() << '\n';
    return 1;
  }
}
