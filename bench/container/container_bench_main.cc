// Single entry point for the container benchmark suite. Each bench lives in its
// own TU (swiss_table_bench.cc / string_buf_bench.cc / string_view_bench.cc /
// path_bench.cc / string_algo_bench.cc) as a function; this runs them in sequence
// so they share one executable (hyle_container_bench).
#include <cstdlib>
#include <string_view>

int bench_swiss_table();
int bench_string_buf();
int bench_string_view();
int bench_path();
int bench_string_algo();

int main() {
  if (const char* filter = std::getenv("HYLE_CONTAINER_BENCH_FILTER")) {
    const std::string_view selected(filter);
    if (selected == "swiss") return bench_swiss_table();
    if (selected == "string-buf") return bench_string_buf();
    if (selected == "string-view") return bench_string_view();
    if (selected == "path") return bench_path();
    if (selected == "string-algo") return bench_string_algo();
    return 2;
  }
  bench_swiss_table();
  bench_string_buf();
  bench_string_view();
  bench_path();
  bench_string_algo();
  return 0;
}
