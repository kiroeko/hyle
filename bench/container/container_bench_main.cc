// Single entry point for the container benchmark suite. Each bench lives in its
// own TU (swiss_table_bench.cc / string_buf_bench.cc / string_ref_bench.cc /
// path_bench.cc / string_algo_bench.cc) as a function; this runs them in sequence
// so they share one executable (hyle_container_bench).
int bench_swiss_table();
int bench_string_buf();
int bench_string_ref();
int bench_path();
int bench_string_algo();

int main() {
  bench_swiss_table();
  bench_string_buf();
  bench_string_ref();
  bench_path();
  bench_string_algo();
  return 0;
}
