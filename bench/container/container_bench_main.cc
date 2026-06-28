// Single entry point for the container benchmark suite. Each bench lives in its
// own TU (swiss_table_bench.cc / string_buf_bench.cc) as a function; this runs
// them in sequence so they share one executable (hyle_container_bench).
int bench_swiss_table();
int bench_string_buf();

int main() {
  bench_swiss_table();
  bench_string_buf();
  return 0;
}
