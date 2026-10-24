#include "deepseek_sparse_prefix_fixture.h"

int main(int argc, char **argv) {
  using namespace deepseek_sparse_prefix_fixture;
  try {
    const bool wide_only = argc == 2 && std::string_view(argv[1]) == "--ratio128";
    if (argc > 1 && !wide_only) return 2;
    auto status = wide_only ? Status::Ok() : run(kIndexed);
    if (status.ok()) status = run(kWide);
    if (status.ok()) return 0;
    char message[1024]{}; status.ToString(message, sizeof(message)); std::fprintf(stderr, "%s\n", message);
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); }
  return 1;
}
