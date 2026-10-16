#define HYLE_G08_CUDA 1
#include "../compute_models/sampling_common_consumer.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  try {
#if defined(HYLE_G08_GRAPH)
    if (argc == 2 && std::string(argv[1]) == "--ports-only") {
      sampling_test::graph_port_liveness<sampling_test::CommonSession>(); return 0;
    }
#endif
    sampling_test::run<sampling_test::CommonSession>(argc == 2 ? std::string_view(argv[1]) : std::string_view{}); return 0;
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL CUDA sampling: %s\n", error.what()); return 1;
  }
}
