#include "../../engine/models/qwen35/main.cc"
#include "qwen_cpu_ahead_case.h"

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass prepared local inputs\n"; return 77; }
    if (argc != 2) throw std::runtime_error("usage: qwen_cpu_ahead_test PREPARED_LOCAL_INPUTS");
    const auto report = qwen_cpu_ahead_case::exercise(argv[1], hyle::inference::BackendKind::kCuda);
    qwen_cpu_ahead_case::write_json(report, hyle::inference::BackendKind::kCuda);
    std::cout << '\n';
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
