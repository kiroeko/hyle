#include <nanobench.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "hyle/mem/reclamation.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kReadsPerBatch = 4096;

struct Snapshot {
  std::array<std::uint64_t, 8> values{};
};

std::uint64_t consume(const Snapshot &snapshot, std::size_t index) noexcept {
  return snapshot.values[index & (snapshot.values.size() - 1U)];
}

} // namespace

int main() {
  Snapshot snapshot{{11, 23, 37, 53, 71, 89, 107, 127}};
  std::atomic<Snapshot *> published{&snapshot};
  hyle::mem::ReclamationDomain domain(0);
  std::atomic<std::shared_ptr<const Snapshot>> shared{
      std::make_shared<const Snapshot>(snapshot)};

  // Preheat the process-image TLS lease and the standard control block. First
  // participation/setup is intentionally outside the steady-state read task.
  { auto guard = domain.read(); }
  static_cast<void>(shared.load(std::memory_order_acquire));

  auto hyle_read = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t i = 0; i != kReadsPerBatch; ++i) {
      auto guard = domain.read();
      checksum += consume(*guard.protect(published), i);
    }
    nb::doNotOptimizeAway(checksum);
  };
  auto shared_read = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t i = 0; i != kReadsPerBatch; ++i) {
      std::shared_ptr<const Snapshot> current =
          shared.load(std::memory_order_acquire);
      checksum += consume(*current, i);
    }
    nb::doNotOptimizeAway(checksum);
  };
  auto raw_floor = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t i = 0; i != kReadsPerBatch; ++i) {
      checksum += consume(*published.load(std::memory_order_acquire), i);
    }
    nb::doNotOptimizeAway(checksum);
  };

  nb::Bench bench;
  bench.title("immutable snapshot: warmed single-reader batch")
      .relative(true)
      .batch(kReadsPerBatch)
      .minEpochIterations(100)
      .minEpochTime(std::chrono::milliseconds(20));

  const bool rival_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  if (rival_first) {
    bench.run("std atomic<shared_ptr> natural route", shared_read);
    bench.run("HYLE ReclamationDomain read guard", hyle_read);
  } else {
    bench.run("HYLE ReclamationDomain read guard", hyle_read);
    bench.run("std atomic<shared_ptr> natural route", shared_read);
  }
  bench.run("raw atomic pointer lifetime-unsafe floor", raw_floor);
}
