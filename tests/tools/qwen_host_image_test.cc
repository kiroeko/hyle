#include "../../engine/models/qwen35/main.cc"
#include "qwen_host_image_case.h"
#include "qwen_state_reuse_case.h"
#include "qwen_mtp_batch_state_case.h"
#include "qwen_packed_prefill_case.h"
#include <cstdlib>

int main(int argc, char **argv) {
  try {
    const bool lagged = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-lag1";
    const bool state_reuse = argc >= 3 && std::string_view(argv[argc - 1]) == "--state-reuse";
    const bool mtp_batch_state = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-batch-state";
    const bool packed_prefill = argc >= 3 && std::string_view(argv[argc - 1]) == "--packed-prefill";
    const auto input_argc = argc - static_cast<int>(lagged || state_reuse || mtp_batch_state || packed_prefill);
    if (input_argc < 2 || input_argc > 3)
      throw std::runtime_error("usage: qwen_host_image_test --cpu|--cuda|--apple|--vulkan [prepared_input_directory] [--mtp-lag1|--state-reuse|--mtp-batch-state|--packed-prefill]");
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
    return packed_prefill ? qwen_packed_prefill_case::run(input, backend)
           : mtp_batch_state ? qwen_mtp_batch_state_case::run(input, backend)
           : state_reuse ? qwen_state_reuse_case::run(input, backend)
           : lagged ? qwen_host_image_case::run_lagged(input, backend)
                  : qwen_host_image_case::run(input, backend);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
