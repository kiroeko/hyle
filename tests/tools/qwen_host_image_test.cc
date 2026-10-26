#include "../../engine/models/qwen35/main.cc"
#include "qwen_host_image_case.h"
#include <cstdlib>

int main(int argc, char **argv) {
  try {
    const bool lagged = argc >= 3 && std::string_view(argv[argc - 1]) == "--mtp-lag1";
    const auto input_argc = argc - static_cast<int>(lagged);
    if (input_argc < 2 || input_argc > 3 ||
        (std::string_view(argv[1]) != "--cpu" && std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_host_image_test --cpu|--cuda [prepared_input_directory] [--mtp-lag1]");
    const auto backend = std::string_view(argv[1]) == "--cpu"
        ? qwen35::BackendKind::kCpu : qwen35::BackendKind::kCuda;
#if !defined(HYLE_COMPUTE_CPU)
    if (backend == qwen35::BackendKind::kCpu) { std::cerr << "SKIP: CPU route is not built\n"; return 77; }
#endif
#if !defined(HYLE_COMPUTE_CUDA)
    if (backend == qwen35::BackendKind::kCuda) { std::cerr << "SKIP: CUDA route is not built\n"; return 77; }
#endif
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
    return lagged ? qwen_host_image_case::run_lagged(input, backend)
                  : qwen_host_image_case::run(input, backend);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
