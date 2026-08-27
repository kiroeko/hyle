#include <nanobench.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/core/compiler.h"
#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

#if defined(HYLE_BENCH_BOOST)
#include <boost/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#endif

#if defined(HYLE_BENCH_TBB)
#include <oneapi/tbb/spin_mutex.h>
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kAggregate = 4096;
constexpr std::size_t kShuffleCount = 1024;
constexpr std::size_t kDiagnosticIterations = 100'000;

nb::Bench make_bench(const char *title, std::size_t batch) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(batch).minEpochTime(
      std::chrono::milliseconds(8));
  return bench;
}

bool environment_present(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

// Independent scalar spelling of the public xoshiro256++ reference algorithm.
// It intentionally has the same 32-byte state and SplitMix64 seed contract as
// HYLE. This row detects wrapper/codegen regressions; it is not a contest
// between unrelated PRNG quality or state-size choices.
class ScalarXoshiro256pp {
 public:
  explicit ScalarXoshiro256pp(std::uint64_t seed) noexcept {
    for (auto &word : state_) word = splitmix64(seed);
  }

  HYLE_FORCE_INLINE std::uint64_t operator()() noexcept {
    const std::uint64_t result =
        rotate_left(state_[0] + state_[3], 23) + state_[0];
    const std::uint64_t temporary = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= temporary;
    state_[3] = rotate_left(state_[3], 45);
    return result;
  }

 private:
  static HYLE_FORCE_INLINE std::uint64_t rotate_left(std::uint64_t value,
                                                     int amount) noexcept {
    return (value << amount) | (value >> (64 - amount));
  }

  static HYLE_FORCE_INLINE std::uint64_t splitmix64(
      std::uint64_t &seed) noexcept {
    std::uint64_t value = (seed += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
  }

  std::uint64_t state_[4];
};

static_assert(sizeof(ScalarXoshiro256pp) == sizeof(hyle::util::Rng));

struct HyleWidget : hyle::util::RefCounted<HyleWidget> {
  explicit HyleWidget(std::uint64_t input) noexcept : value(input) {}
  std::uint64_t value;
};

#if defined(HYLE_BENCH_BOOST)
struct BoostWidget
    : boost::intrusive_ref_counter<BoostWidget, boost::thread_safe_counter> {
  explicit BoostWidget(std::uint64_t input) noexcept : value(input) {}
  std::uint64_t value;
};
#endif

template <class Owner>
HYLE_NOINLINE std::uint64_t owning_handoff(Owner owner) noexcept {
  return owner->value;
}

template <class Lock>
HYLE_NOINLINE std::uint64_t short_state_transition(
    Lock &lock, std::uint64_t &state) noexcept {
  lock.lock();
  state = state * 0x9e3779b97f4a7c15ull + 1;
  lock.unlock();
  return state;
}

template <class Lock>
std::uint64_t contended_increment(std::size_t thread_count,
                                  std::size_t iterations) {
  Lock lock;
  std::uint64_t value = 0;
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      for (std::size_t index = 0; index < iterations; ++index) {
        lock.lock();
        ++value;
        lock.unlock();
      }
    });
  }
  for (auto &thread : threads) thread.join();
  return value;
}

bool is_permutation(const std::array<std::uint32_t, kShuffleCount> &values) {
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    if (sorted[index] != index) return false;
  }
  return true;
}

int validate_contract() {
  bool rng_equal = true;
  hyle::util::Rng hyle_rng{7};
  ScalarXoshiro256pp reference_rng{7};
  for (std::size_t index = 0; index < 64; ++index) {
    rng_equal = rng_equal && hyle_rng.next_u64() == reference_rng();
  }

  hyle::util::Rng bounded_rng{9};
  bool bounded = true;
  for (std::uint64_t upper = 0; upper < 257; ++upper) {
    const std::uint64_t value = bounded_rng.range_u64(0, upper);
    bounded = bounded && value <= upper;
  }

  std::array<std::uint32_t, kShuffleCount> values{};
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(index);
  }
  hyle::util::Rng shuffle_rng{11};
  shuffle_rng.shuffle(values.begin(), values.end());
  const bool shuffle = is_permutation(values);

  auto hyle_owner = hyle::util::make_ref<HyleWidget>(13);
  const std::uint64_t hyle_owner_value = owning_handoff(hyle_owner);
  const bool hyle_ref = hyle_owner_value == 13 && hyle_owner->use_count() == 1;

  hyle::util::SpinLock hyle_lock;
  std::uint64_t hyle_state = 1;
  const bool hyle_spin =
      short_state_transition(hyle_lock, hyle_state) == hyle_state;

  bool boost_ref = false;
  std::uint64_t boost_owner_value = 0;
#if defined(HYLE_BENCH_BOOST)
  boost::intrusive_ptr<BoostWidget> boost_owner(new BoostWidget(13));
  boost_owner_value = owning_handoff(boost_owner);
  boost_ref = boost_owner_value == 13 && boost_owner->use_count() == 1;
#endif

  bool tbb_spin = false;
  std::uint64_t tbb_state = 1;
#if defined(HYLE_BENCH_TBB)
  oneapi::tbb::spin_mutex tbb_lock;
  tbb_spin = short_state_transition(tbb_lock, tbb_state) == tbb_state;
#endif

  const bool layout = sizeof(hyle::util::Rng) == 32 &&
                      sizeof(hyle::util::Ref<HyleWidget>) == sizeof(void *) &&
                      sizeof(hyle::util::SpinLock) == 1;
  const bool peers = boost_ref && tbb_spin;
  const bool passed = rng_equal && bounded && shuffle && hyle_ref &&
                      hyle_spin && peers && layout;
  std::printf(
      "# HYLE_UTIL_CORE_CONTRACT schema=1 rng=%s bounded=%s shuffle=%s "
      "hyle_ref=%s boost_ref=%s hyle_spin=%s tbb_spin=%s layout=%s "
      "hyle_owner=%" PRIu64 " boost_owner=%" PRIu64 " verdict=%s\n",
      rng_equal ? "PASS" : "FAIL", bounded ? "PASS" : "FAIL",
      shuffle ? "PASS" : "FAIL", hyle_ref ? "PASS" : "FAIL",
      boost_ref ? "PASS" : "FAIL", hyle_spin ? "PASS" : "FAIL",
      tbb_spin ? "PASS" : "FAIL", layout ? "PASS" : "FAIL", hyle_owner_value,
      boost_owner_value, passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

void bench_rng_next(bool peer_first) {
  hyle::util::Rng candidate_rng{12345};
  ScalarXoshiro256pp peer_rng{12345};
  nb::Bench bench =
      make_bench("util-core scalar xoshiro256++ aggregate N=4096", kAggregate);
  const auto candidate = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= candidate_rng.next_u64();
    }
    nb::doNotOptimizeAway(checksum);
  };
  const auto peer = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= peer_rng();
    }
    nb::doNotOptimizeAway(checksum);
  };
  run_pair(bench, peer_first, "HYLE Rng::next_u64", candidate,
           "scalar xoshiro256++ reference", peer);
}

void bench_rng_bounded(bool peer_first) {
  std::array<std::uint64_t, 256> bounds{};
  for (std::size_t index = 0; index < bounds.size(); ++index) {
    bounds[index] = 17 + index * 7919;
  }
  hyle::util::Rng candidate_rng{7};
  hyle::util::Rng peer_rng{7};
  std::uniform_int_distribution<std::uint64_t> distribution;
  nb::Bench bench =
      make_bench("util-core varying bounded u64 aggregate N=4096", kAggregate);
  const auto candidate = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= candidate_rng.range_u64(0, bounds[index & 255]);
    }
    nb::doNotOptimizeAway(checksum);
  };
  const auto peer = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= distribution(
          peer_rng, decltype(distribution)::param_type{0, bounds[index & 255]});
    }
    nb::doNotOptimizeAway(checksum);
  };
  run_pair(bench, peer_first, "HYLE Rng::range_u64", candidate,
           "std::uniform_int_distribution(Rng)", peer);
}

void bench_rng_shuffle(bool peer_first) {
  std::array<std::uint32_t, kShuffleCount> candidate_values{};
  std::array<std::uint32_t, kShuffleCount> peer_values{};
  for (std::size_t index = 0; index < kShuffleCount; ++index) {
    candidate_values[index] = static_cast<std::uint32_t>(index);
    peer_values[index] = static_cast<std::uint32_t>(index);
  }
  hyle::util::Rng candidate_rng{9};
  hyle::util::Rng peer_rng{9};
  nb::Bench bench = make_bench("util-core shuffle N=1024", kShuffleCount);
  const auto candidate = [&] {
    candidate_rng.shuffle(candidate_values.begin(), candidate_values.end());
    nb::doNotOptimizeAway(candidate_values);
  };
  const auto peer = [&] {
    std::shuffle(peer_values.begin(), peer_values.end(), peer_rng);
    nb::doNotOptimizeAway(peer_values);
  };
  run_pair(bench, peer_first, "HYLE Rng::shuffle", candidate,
           "std::shuffle(Rng)", peer);
}

#if defined(HYLE_BENCH_BOOST)
void bench_ref(bool peer_first) {
  auto candidate_owner = hyle::util::make_ref<HyleWidget>(17);
  boost::intrusive_ptr<BoostWidget> peer_owner(new BoostWidget(17));
  nb::Bench bench = make_bench(
      "util-core intrusive owning handoff aggregate N=4096", kAggregate);
  const auto candidate = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum += owning_handoff(candidate_owner);
    }
    nb::doNotOptimizeAway(checksum);
  };
  const auto peer = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum += owning_handoff(peer_owner);
    }
    nb::doNotOptimizeAway(checksum);
  };
  run_pair(bench, peer_first, "HYLE Ref", candidate, "Boost intrusive_ptr",
           peer);
}
#endif

#if defined(HYLE_BENCH_TBB)
void bench_spin(bool peer_first) {
  hyle::util::SpinLock candidate_lock;
  oneapi::tbb::spin_mutex peer_lock;
  std::uint64_t candidate_state = 1;
  std::uint64_t peer_state = 1;
  nb::Bench bench = make_bench(
      "util-core FiberStackPool-shaped state aggregate N=4096", kAggregate);
  const auto candidate = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= short_state_transition(candidate_lock, candidate_state);
    }
    nb::doNotOptimizeAway(checksum);
  };
  const auto peer = [&] {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kAggregate; ++index) {
      checksum ^= short_state_transition(peer_lock, peer_state);
    }
    nb::doNotOptimizeAway(checksum);
  };
  run_pair(bench, peer_first, "HYLE SpinLock", candidate, "oneTBB spin_mutex",
           peer);
}

void bench_spin_diagnostics(bool peer_first) {
  for (std::size_t threads : {std::size_t{2}, std::size_t{8}}) {
    nb::Bench bench = make_bench(
        threads == 2 ? "diagnostic spin handoff 2 threads"
                     : "diagnostic sustained spin contention 8 threads",
        threads * kDiagnosticIterations);
    bench.epochs(7).minEpochIterations(1);
    const auto candidate = [=] {
      nb::doNotOptimizeAway(contended_increment<hyle::util::SpinLock>(
          threads, kDiagnosticIterations));
    };
    const auto peer = [=] {
      nb::doNotOptimizeAway(contended_increment<oneapi::tbb::spin_mutex>(
          threads, kDiagnosticIterations));
    };
    run_pair(bench, peer_first, "HYLE SpinLock diagnostic", candidate,
             "oneTBB spin_mutex diagnostic", peer);
  }
}
#endif

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    return validate_contract();
  }

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool formal_only =
      environment_present("HYLE_UTIL_CORE_BENCH_FORMAL_ONLY");
  std::fprintf(stderr,
               "# HYLE_UTIL_CORE_RUN execution_order=%s apple_qos=%s "
               "qos_ok=%s affinity=external formal_only=%d run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               formal_only ? 1 : 0, hyle::bench::perf::run_nonce());

  // The formal contract only admits peers compiled from their pinned source
  // trees. The scalar/standard-library RNG rows remain useful diagnostics, but
  // cannot borrow Boost or oneTBB's source identity.
  if (!formal_only) {
    bench_rng_next(peer_first);
    bench_rng_bounded(peer_first);
    bench_rng_shuffle(peer_first);
  }
#if defined(HYLE_BENCH_BOOST)
  bench_ref(peer_first);
#endif
#if defined(HYLE_BENCH_TBB)
  bench_spin(peer_first);
  if (!formal_only) bench_spin_diagnostics(peer_first);
#endif
  return 0;
}
