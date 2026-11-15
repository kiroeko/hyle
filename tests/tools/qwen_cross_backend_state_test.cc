// A bounded complete-state consumer, reusing the existing host-image checks.
// No CLI, model diagnostics, fixture generation or download is included.
#include "../../engine/models/qwen35/generation_runtime.cc"
#include "qwen_host_image_case.h"
#include "qwen_mtp_batch_state_case.h"
#include "qwen_depth_case.h"
#include "inference_program_carrier_case.h"
#include "inference_native_handoff_case.h"

int main(int argc, char **argv) {
  try {
    const bool depth = argc == 6 && std::string_view(argv[5]) == "--depth-probe";
    const bool carrier = argc == 3 && std::string_view(argv[2]) == "--carrier-probe";
    const bool handoff = argc == 3 && std::string_view(argv[2]) == "--native-handoff-probe";
    const bool prefix_repair = argc == 6 && std::string_view(argv[5]) == "--prefix-repair-probe";
    if (argc != 5 && !depth && !carrier && !handoff && !prefix_repair) throw std::runtime_error(
        "usage: qwen_cross_backend_state_test cpu|cuda|apple|vulkan INPUTS off|K3 CPU_FFN_LAYERS [--depth-probe|--prefix-repair-probe]; or BACKEND --carrier-probe|--native-handoff-probe");
    const std::string_view name(argv[1]);
    auto kind = qwen35::BackendKind::kDefault;
    if (name == "cpu") kind = qwen35::BackendKind::kCpu;
    else if (name == "cuda") kind = qwen35::BackendKind::kCuda;
    else if (name == "apple") kind = qwen35::BackendKind::kApple;
    else if (name == "vulkan") kind = qwen35::BackendKind::kVulkan;
    else throw std::runtime_error("unknown native backend");
    if (carrier) return inference_program_carrier_case::run(kind);
    if (handoff) return inference_native_handoff_case::run(kind);
    const std::string_view mode(argv[3]);
    if (mode != "off" && mode != "K3") throw std::runtime_error("expected ordinary or K3 state consumer");
    std::size_t consumed = 0;
    const auto layers = std::stoull(argv[4], &consumed);
    if (consumed != std::string_view(argv[4]).size()) throw std::runtime_error("invalid CPU placement");
    if (depth) return qwen_depth_case::run(argv[2], kind, mode == "K3", layers);
    if (prefix_repair) {
      if (mode != "K3") throw std::runtime_error("prefix repair requires the prepared three-draft consumer");
      for (const bool random : {false, true})
        qwen_mtp_batch_state_case::run(argv[2], kind, layers, 4, random);
      return 0;
    }
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
