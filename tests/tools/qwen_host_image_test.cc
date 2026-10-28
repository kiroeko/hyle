#include "../../engine/models/qwen35/main.cc"
#include "qwen_host_image_case.h"
#include "qwen_state_reuse_case.h"
#include "qwen_mtp_batch_state_case.h"
#include "qwen_packed_prefill_case.h"
#include "qwen_ragged_prefill_case.h"
#include "qwen_mtp_packed_prefill_case.h"
#include "qwen_mtp_ragged_prefill_case.h"
#include "qwen_mtp_chunked_prefill_case.h"
#include "qwen_logits_preflight_case.h"
#include "qwen_mtp_stop_case.h"
#include "qwen_mtp_sampling_case.h"
#include "qwen_mtp_logits_case.h"
#include "qwen_mtp_e4_case.h"
#include "inference_history_math_case.h"
#include "qwen_history_logits_case.h"
#include "qwen_history_replay_case.h"
#include "qwen_history_control_case.h"
#include "inference_program_carrier_case.h"
#include "qwen_generation_wave_case.h"
#include "qwen_sampling_policy_case.h"
#include "qwen_mtp_group_return_case.h"
#include "inference_cold_demand_case.h"
#include <cstdlib>

int main(int argc, char **argv) {
  try {
    const bool lagged = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-lag1";
    const bool state_reuse = argc >= 3 && std::string_view(argv[argc - 1]) == "--state-reuse";
    const bool mtp_batch_state = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-batch-state";
    const bool packed_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--packed-prefill";
    const bool ragged_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--ragged-prefill";
    const bool mtp_packed_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-packed-prefill";
    const bool mtp_ragged_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-ragged-prefill";
    const bool mtp_chunked_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-chunked-prefill";
    const bool logits_preflight = argc >= 3 && std::string_view(argv[argc - 1]) == "--logits-preflight";
    const bool mtp_stop = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-stop";
    const bool mtp_sampling = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-sampling";
    const bool mtp_logits = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-logits";
    const bool mtp_e4 = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-e4";
    const bool program_carrier = argc >= 3 && std::string_view(argv[argc - 1]) == "--program-carrier";
    const bool history_logits = argc >= 3 && std::string_view(argv[argc - 1]) == "--history-logits";
    const bool history_ordinary = argc >= 3 && std::string_view(argv[argc - 1]) == "--history-ordinary";
    const bool history_replay = argc >= 3 && std::string_view(argv[argc - 1]) == "--history-replay";
    const bool history_control = argc >= 3 && std::string_view(argv[argc - 1]) == "--history-control";
    const bool generation_wave = argc >= 3 && std::string_view(argv[argc - 1]) == "--generation-wave";
    const bool sampling_policies = argc >= 3 && std::string_view(argv[argc - 1]) == "--sampling-policies";
    const bool mtp_group_return = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-group-return";
    const bool cold_demand = argc >= 3 && std::string_view(argv[argc - 1]) == "--cold-demand";
    const auto input_argc = argc - static_cast<int>(lagged || state_reuse || mtp_batch_state || packed_prefill || ragged_prefill || mtp_packed_prefill || mtp_ragged_prefill || mtp_chunked_prefill || logits_preflight || mtp_stop || mtp_sampling || mtp_logits || mtp_e4 || program_carrier || history_logits || history_ordinary || history_replay || history_control || generation_wave || sampling_policies || mtp_group_return || cold_demand);
    if (input_argc < 2 || input_argc > 3)
      throw std::runtime_error("usage: qwen_host_image_test --cpu|--cuda|--apple|--vulkan [prepared_input_directory] [--mtp-lag1|--state-reuse|--mtp-batch-state|--packed-prefill|--ragged-prefill|--mtp-packed-prefill|--mtp-ragged-prefill|--mtp-chunked-prefill|--logits-preflight|--mtp-stop|--mtp-sampling|--mtp-logits|--mtp-e4|--program-carrier|--history-logits|--history-ordinary|--history-replay|--history-control|--generation-wave|--sampling-policies|--mtp-group-return|--cold-demand]");
    const std::string_view selected(argv[1]);
    auto backend = qwen35::BackendKind::kDefault;
    if (selected == "--cpu") backend = qwen35::BackendKind::kCpu;
    else if (selected == "--cuda") backend = qwen35::BackendKind::kCuda;
    else if (selected == "--apple") backend = qwen35::BackendKind::kApple;
    else if (selected == "--vulkan") backend = qwen35::BackendKind::kVulkan;
    else throw std::runtime_error("unknown host-image-test backend");
    const auto compiled = qwen35::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), backend) == compiled.end()) {
      std::cerr << "SKIP: selected route is not built\n"; return 77;
    }
    if (program_carrier) return inference_program_carrier_case::run(backend);
    if (cold_demand) return inference_cold_demand_case::run();
    std::string input;
    if (input_argc == 3) input = argv[2];
    else {
#if defined(_MSC_VER)
      char *value = nullptr; std::size_t length = 0;
      if (_dupenv_s(&value, &length, "HYLE_QWEN_HOST_IMAGE_INPUTS"))
        throw std::runtime_error("cannot read HYLE_QWEN_HOST_IMAGE_INPUTS");
      const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
      if (value) input = value;
#else
      if (const auto *value = std::getenv("HYLE_QWEN_HOST_IMAGE_INPUTS")) input = value;
#endif
    }
    if (input.empty()) {
      std::cerr << "SKIP: pass prepared inputs or set HYLE_QWEN_HOST_IMAGE_INPUTS\n"; return 77;
    }
    return mtp_group_return ? qwen_mtp_group_return_case::run(input, backend)
           : generation_wave ? qwen_generation_wave_case::run(input, backend)
           : sampling_policies ? qwen_sampling_policy_case::run(input, backend)
           : history_control ? qwen_history_control_case::run(input, backend)
           : history_replay ? qwen_history_replay_case::run(input, backend)
           : history_ordinary ? qwen_history_logits_case::run(input, backend, false)
           : history_logits ? qwen_history_logits_case::run(input, backend)
           : mtp_e4 ? qwen_mtp_e4_case::run(input, backend)
           : mtp_logits ? qwen_mtp_logits_case::run(input, backend)
           : mtp_sampling ? qwen_mtp_sampling_case::run(input, backend)
           : mtp_stop ? qwen_mtp_stop_case::run(input, backend)
           : logits_preflight ? qwen_logits_preflight_case::run(input, backend)
           : mtp_chunked_prefill ? qwen_mtp_chunked_prefill_case::run(input, backend)
           : mtp_ragged_prefill ? qwen_mtp_ragged_prefill_case::run(input, backend)
           : mtp_packed_prefill ? qwen_mtp_packed_prefill_case::run(input, backend)
           : ragged_prefill ? qwen_ragged_prefill_case::run(input, backend)
           : packed_prefill ? qwen_packed_prefill_case::run(input, backend)
           : mtp_batch_state ? qwen_mtp_batch_state_case::run(input, backend)
           : state_reuse ? qwen_state_reuse_case::run(input, backend)
           : lagged ? qwen_host_image_case::run_lagged(input, backend)
                  : qwen_host_image_case::run(input, backend);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
