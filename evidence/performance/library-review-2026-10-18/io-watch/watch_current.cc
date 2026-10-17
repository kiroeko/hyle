#define main hyle_watch_legacy_main
#include "/mnt/c/public/project-hyle/hyle/bench/io_watch/watch_p1_bench.cc"
#undef main
#include <fstream>

int thread_count() {
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line))
    if (line.starts_with("Threads:")) return std::stoi(line.substr(8));
  return -1;
}

template<class Harness>
void measure(const char *route, std::size_t episodes, int rows) {
  Harness harness(Shape::kExactReplace);
  harness.warmup();
  std::this_thread::sleep_for(kRouteQuiescence);
  const int threads = thread_count();
  for (int row = 0; row < rows; ++row) {
    const auto begin = std::chrono::steady_clock::now();
    harness.run(episodes);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    std::printf("{\"route\":\"%s\",\"row\":%d,\"episodes\":%zu,\"elapsed_ns\":%lld,\"ns_per_change\":%.9f,\"threads\":%d}\n",
                route, row, episodes, static_cast<long long>(elapsed),
                static_cast<double>(elapsed) / static_cast<double>(episodes), threads);
    std::fflush(stdout);
  }
}

int main(int argc, char **argv) {
  const std::string_view route = argc > 1 ? argv[1] : "hyle";
  const auto episodes = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 4096u;
  const int rows = argc > 3 ? std::stoi(argv[3]) : 5;
  if (route == "hyle") measure<HyleHarness>("hyle", episodes, rows);
  else if (route == "libuv") measure<LibuvHarness>("libuv", episodes, rows);
  else if (route == "hyle-libuv") {
    measure<HyleHarness>("hyle", episodes, rows);
    std::this_thread::sleep_for(kRouteQuiescence);
    measure<LibuvHarness>("libuv", episodes, rows);
  } else if (route == "libuv-hyle") {
    measure<LibuvHarness>("libuv", episodes, rows);
    std::this_thread::sleep_for(kRouteQuiescence);
    measure<HyleHarness>("hyle", episodes, rows);
  } else return 2;
}
