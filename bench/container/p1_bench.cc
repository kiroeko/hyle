#include <nanobench.h>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <boost/dynamic_bitset.hpp>
#include <rigtorp/MPMCQueue.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "../common/perf_process.h"
#include "hyle/container/bounded_queue.h"
#include "hyle/container/dynamic_bitset.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/hash.h"
#include "hyle/container/small_vector.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/system_inline.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kBatch = 4096;
constexpr std::size_t kInlineValues = 8;
constexpr std::size_t kSpillValues = 64;
constexpr std::size_t kInlineLists = 256;
constexpr std::size_t kSpillLists = 32;
constexpr std::size_t kMapEntries = 32768;
constexpr std::size_t kQueueCapacity = 1024;
constexpr std::size_t kBitBlocks = kBatch;
constexpr std::size_t kBitCount = kBitBlocks * 64;
constexpr std::size_t kHashKeys = 256;
constexpr std::size_t kHashPasses = kBatch / kHashKeys;
constexpr std::uint64_t kHashSeed = UINT64_C(0x9e3779b97f4a7c15);
constexpr std::uint64_t kQueueExpected =
    (kBatch * (kBatch + 1)) / 2;

static_assert(kInlineLists * kInlineValues +
                  kSpillLists * kSpillValues ==
              kBatch);
static_assert(kHashKeys * kHashPasses == kBatch);

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

inline void spin_pause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

nb::Bench make_bench(const char* title) {
  nb::Bench bench;
  bench.title(title)
      .relative(true)
      .batch(kBatch)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

using SequenceInput = std::array<std::uint32_t, kSpillValues>;

HYLE_NOINLINE std::uint64_t sequence_candidate(
    const SequenceInput& input) {
  std::uint64_t checksum = 0;
  for (std::size_t list = 0; list < kInlineLists; ++list) {
    hyle::container::SmallVector<std::uint32_t, 16> values;
    for (std::size_t index = 0; index < kInlineValues; ++index)
      values.push_back(input[(index + list) & (kSpillValues - 1)]);
    for (const std::uint32_t value : values) checksum += value;
  }
  for (std::size_t list = 0; list < kSpillLists; ++list) {
    hyle::container::SmallVector<std::uint32_t, 16> values;
    for (std::size_t index = 0; index < kSpillValues; ++index)
      values.push_back(input[(index + list) & (kSpillValues - 1)]);
    for (const std::uint32_t value : values) checksum += value;
  }
  return checksum;
}

HYLE_NOINLINE std::uint64_t sequence_peer(const SequenceInput& input) {
  std::uint64_t checksum = 0;
  for (std::size_t list = 0; list < kInlineLists; ++list) {
    absl::InlinedVector<std::uint32_t, 16> values;
    for (std::size_t index = 0; index < kInlineValues; ++index)
      values.push_back(input[(index + list) & (kSpillValues - 1)]);
    for (const std::uint32_t value : values) checksum += value;
  }
  for (std::size_t list = 0; list < kSpillLists; ++list) {
    absl::InlinedVector<std::uint32_t, 16> values;
    for (std::size_t index = 0; index < kSpillValues; ++index)
      values.push_back(input[(index + list) & (kSpillValues - 1)]);
    for (const std::uint32_t value : values) checksum += value;
  }
  return checksum;
}

class AssociativeReplay {
 public:
  explicit AssociativeReplay(bool peer_allocation_first = false) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<PeerMap>();
      candidate_ = std::make_unique<CandidateMap>();
    } else {
      candidate_ = std::make_unique<CandidateMap>();
      peer_ = std::make_unique<PeerMap>();
    }
    candidate_->reserve(kMapEntries);
    peer_->reserve(kMapEntries);
    for (std::size_t index = 0; index < kMapEntries; ++index) {
      const std::uint64_t key = index * 2;
      const std::uint64_t value = index * 17 + 3;
      candidate_->insert_or_assign(key, value);
      peer_->insert_or_assign(key, value);
    }
    for (std::size_t index = 0; index < queries_.size(); ++index) {
      const std::uint64_t base = ((index * 101) % kMapEntries) * 2;
      queries_[index] = base + (index & 1);
    }
  }

  HYLE_NOINLINE std::uint64_t candidate() const noexcept {
    std::uint64_t checksum = 0;
    for (const std::uint64_t key : queries_) {
      if (const std::uint64_t* value = candidate_->find(key)) checksum += *value;
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() const noexcept {
    std::uint64_t checksum = 0;
    for (const std::uint64_t key : queries_) {
      const auto found = peer_->find(key);
      if (found != peer_->end()) checksum += found->second;
    }
    return checksum;
  }

  std::size_t candidate_capacity() const noexcept {
    return candidate_->capacity();
  }
  std::size_t peer_capacity() const noexcept { return peer_->capacity(); }

 private:
  using CandidateMap =
      hyle::container::FlatHashMap<std::uint64_t, std::uint64_t>;
  using PeerMap = absl::flat_hash_map<std::uint64_t, std::uint64_t>;

  std::unique_ptr<CandidateMap> candidate_;
  std::unique_ptr<PeerMap> peer_;
  std::array<std::uint64_t, kBatch> queries_{};
};

class QueueReplay {
 public:
  explicit QueueReplay(bool peer_allocation_first) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<PeerQueue>(kQueueCapacity);
      candidate_ = std::make_unique<CandidateQueue>(kQueueCapacity);
    } else {
      candidate_ = std::make_unique<CandidateQueue>(kQueueCapacity);
      peer_ = std::make_unique<PeerQueue>(kQueueCapacity);
    }
    if (!candidate_->valid()) std::abort();
    worker_ = std::thread([this] { worker_loop(); });
    while (!ready_.load(std::memory_order_acquire))
      spin_pause();
  }

  ~QueueReplay() {
    command_.store(Command::kStop, std::memory_order_relaxed);
    const std::uint64_t generation =
        issued_.fetch_add(1, std::memory_order_release) + 1;
    while (completed_.load(std::memory_order_acquire) != generation)
      spin_pause();
    worker_.join();
  }

  QueueReplay(const QueueReplay&) = delete;
  QueueReplay& operator=(const QueueReplay&) = delete;

  HYLE_NOINLINE std::uint64_t run_candidate() noexcept {
    return dispatch(Command::kCandidate, *candidate_);
  }

  HYLE_NOINLINE std::uint64_t run_peer() noexcept {
    return dispatch(Command::kPeer, *peer_);
  }

 private:
  using CandidateQueue =
      hyle::container::BoundedQueue<std::uint64_t>;
  using PeerQueue = rigtorp::mpmc::Queue<std::uint64_t>;
  enum class Command : unsigned char { kNone, kCandidate, kPeer, kStop };

  template <class Queue>
  std::uint64_t dispatch(Command command, Queue& queue) noexcept {
    command_.store(command, std::memory_order_relaxed);
    const std::uint64_t generation =
        issued_.fetch_add(1, std::memory_order_release) + 1;
    for (std::uint64_t value = 1; value <= kBatch; ++value) {
      while (!queue.try_push(value)) spin_pause();
    }
    while (completed_.load(std::memory_order_acquire) != generation)
      spin_pause();
    if (last_checksum_ != kQueueExpected) std::abort();
    return last_checksum_;
  }

  template <class Queue>
  static std::uint64_t consume(Queue& queue) noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < kBatch; ++index) {
      std::uint64_t value = 0;
      while (!queue.try_pop(value)) spin_pause();
      checksum += value;
    }
    return checksum;
  }

  void worker_loop() noexcept {
    (void)hyle::bench::perf::configure_thread_affinity(
        "HYLE_PERF_WORKER_CPU");
    ready_.store(true, std::memory_order_release);
    std::uint64_t observed = 0;
    for (;;) {
      std::uint64_t generation = issued_.load(std::memory_order_acquire);
      while (generation == observed) {
        spin_pause();
        generation = issued_.load(std::memory_order_acquire);
      }
      observed = generation;
      const Command command = command_.load(std::memory_order_relaxed);
      if (command == Command::kStop) {
        completed_.store(observed, std::memory_order_release);
        return;
      }
      if (command == Command::kCandidate) {
        last_checksum_ = consume(*candidate_);
      } else if (command == Command::kPeer) {
        last_checksum_ = consume(*peer_);
      } else {
        std::abort();
      }
      completed_.store(observed, std::memory_order_release);
    }
  }

  std::unique_ptr<CandidateQueue> candidate_;
  std::unique_ptr<PeerQueue> peer_;
  std::thread worker_;
  alignas(64) std::atomic<std::uint64_t> issued_{0};
  alignas(64) std::atomic<std::uint64_t> completed_{0};
  std::atomic<Command> command_{Command::kNone};
  std::atomic<bool> ready_{false};
  std::uint64_t last_checksum_ = 0;
};

class ActiveSetReplay {
 public:
  explicit ActiveSetReplay(bool peer_allocation_first = false) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<PeerBits>(kBitCount);
      candidate_ = std::make_unique<CandidateBits>(kBitCount);
    } else {
      candidate_ = std::make_unique<CandidateBits>(kBitCount);
      peer_ = std::make_unique<PeerBits>(kBitCount);
    }
    for (std::size_t index = 0; index < kBitCount; ++index) {
      if (((index * 0x9e3779b9u) ^ (index >> 3)) % 11 < 4) {
        candidate_->set(index);
        peer_->set(index);
      }
    }
    if (candidate_->count() != peer_->count()) std::abort();
  }

  HYLE_NOINLINE std::size_t candidate() const noexcept {
    return candidate_->count();
  }
  HYLE_NOINLINE std::size_t peer() const noexcept { return peer_->count(); }

 private:
  using CandidateBits = hyle::container::DynamicBitSet;
  using PeerBits = boost::dynamic_bitset<std::uint64_t>;

  std::unique_ptr<CandidateBits> candidate_;
  std::unique_ptr<PeerBits> peer_;
};

class HashReplay {
 public:
  HashReplay() {
    std::uint64_t state = UINT64_C(0x243f6a8885a308d3);
    for (std::size_t key = 0; key < keys_.size(); ++key) {
      for (char& byte : keys_[key]) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        byte = static_cast<char>((state * UINT64_C(2685821657736338717)) >> 56);
      }
    }
  }

  HYLE_NOINLINE std::uint64_t candidate() const noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t pass = 0; pass < kHashPasses; ++pass) {
      for (const auto& key : keys_)
        checksum ^= hyle::container::hash_bytes(
            key.data(), key.size(), kHashSeed + pass);
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() const noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t pass = 0; pass < kHashPasses; ++pass) {
      for (const auto& key : keys_)
        checksum ^= XXH3_64bits_withSeed(
            key.data(), key.size(), kHashSeed + pass);
    }
    return checksum;
  }

 private:
  std::array<std::array<char, 64>, kHashKeys> keys_{};
};

struct RecordingAllocatorState {
  std::size_t bytes = 0;
};

struct RecordingAllocator {
  RecordingAllocatorState* state;

  void* allocate(std::size_t size, std::size_t alignment) const noexcept {
    state->bytes = size;
    return hyle::mem::SystemAllocator::allocate(size, alignment);
  }
  void deallocate(void* pointer, std::size_t size,
                  std::size_t alignment) const noexcept {
    hyle::mem::SystemAllocator::deallocate(pointer, size, alignment);
  }
};

std::size_t candidate_queue_storage_bytes() {
  RecordingAllocatorState state;
  {
    hyle::container::BoundedQueue<std::uint64_t, RecordingAllocator> queue(
        RecordingAllocator{&state}, kQueueCapacity);
    if (!queue.valid()) std::abort();
  }
  return state.bytes;
}

int validate_contract() {
  SequenceInput input{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint32_t>(index * 7 + 1);
  const bool sequence = sequence_candidate(input) == sequence_peer(input);

  AssociativeReplay associative;
  const bool map = associative.candidate() == associative.peer();

  QueueReplay queue_replay(false);
  const std::uint64_t candidate_queue_checksum =
      queue_replay.run_candidate();
  const std::uint64_t peer_queue_checksum = queue_replay.run_peer();
  const bool queue = candidate_queue_checksum == peer_queue_checksum;

  ActiveSetReplay active;
  const bool bitset = active.candidate() == active.peer();

  HashReplay hash;
  const std::uint64_t candidate_hash = hash.candidate();
  const std::uint64_t peer_hash = hash.peer();
  const bool hashing = candidate_hash != 0 && peer_hash != 0;
  const bool passed = sequence && map && queue && bitset && hashing;
  std::printf(
      "# HYLE_CONTAINER_P1_CONTRACT schema=1 sequence=%s map=%s queue=%s "
      "bitset=%s hash=%s map_candidate_capacity=%zu map_peer_capacity=%zu "
      "queue_candidate_storage=%zu queue_peer_storage=%zu verdict=%s\n",
      sequence ? "PASS" : "FAIL", map ? "PASS" : "FAIL",
      queue ? "PASS" : "FAIL", bitset ? "PASS" : "FAIL",
      hashing ? "PASS" : "FAIL", associative.candidate_capacity(),
      associative.peer_capacity(), candidate_queue_storage_bytes(),
      (kQueueCapacity + 1) *
          sizeof(rigtorp::mpmc::Slot<std::uint64_t>),
      passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

void bench_sequence(bool peer_first, const SequenceInput& input) {
  nb::Bench bench = make_bench(
      "container inline and spill batch aggregate N=4096");
  run_pair(
      bench, peer_first, "HYLE SmallVector batch N=4096",
      [&] { nb::doNotOptimizeAway(sequence_candidate(input)); },
      "Abseil InlinedVector batch N=4096",
      [&] { nb::doNotOptimizeAway(sequence_peer(input)); });
}

void bench_associative(bool peer_first, bool peer_allocation_first) {
  AssociativeReplay replay(peer_allocation_first);
  nb::Bench bench = make_bench(
      "container metadata flat lookup aggregate N=4096");
  run_pair(
      bench, peer_first, "HYLE FlatHashMap lookup N=4096",
      [&] { nb::doNotOptimizeAway(replay.candidate()); },
      "Abseil flat_hash_map lookup N=4096",
      [&] { nb::doNotOptimizeAway(replay.peer()); });
}

void bench_queue(bool peer_first, bool peer_allocation_first) {
  QueueReplay replay(peer_allocation_first);
  for (int index = 0; index < 32; ++index) {
    (void)replay.run_candidate();
    (void)replay.run_peer();
  }
  nb::Bench bench = make_bench(
      "container bounded MPMC handoff aggregate N=4096");
  run_pair(
      bench, peer_first, "HYLE BoundedQueue handoff N=4096",
      [&] { nb::doNotOptimizeAway(replay.run_candidate()); },
      "rigtorp MPMCQueue handoff N=4096",
      [&] { nb::doNotOptimizeAway(replay.run_peer()); });
}

void bench_active_set(bool peer_first, bool peer_allocation_first) {
  ActiveSetReplay replay(peer_allocation_first);
  nb::Bench bench = make_bench(
      "container dense active mask count blocks N=4096");
  run_pair(
      bench, peer_first, "HYLE DynamicBitSet count N=4096",
      [&] { nb::doNotOptimizeAway(replay.candidate()); },
      "Boost dynamic_bitset count N=4096",
      [&] { nb::doNotOptimizeAway(replay.peer()); });
}

void bench_hash(bool peer_first) {
  HashReplay replay;
  nb::Bench bench = make_bench(
      "container metadata byte hash aggregate N=4096");
  run_pair(
      bench, peer_first, "HYLE hash_bytes 64B N=4096",
      [&] { nb::doNotOptimizeAway(replay.candidate()); },
      "xxHash XXH3 64B N=4096",
      [&] { nb::doNotOptimizeAway(replay.peer()); });
}

}  // namespace

int main(int argc, char** argv) {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract")
    return validate_contract();

  const char* worker_cpu = std::getenv("HYLE_PERF_WORKER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  AssociativeReplay resource_map(peer_allocation_first);
  const std::size_t queue_candidate_storage =
      candidate_queue_storage_bytes();
  std::fprintf(
      stderr,
      "# HYLE_CONTAINER_P1_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s "
      "affinity_topology=%s caller_cpu=%s worker_cpu=%s affinity_ok=%s "
      "batch=%zu small_candidate_object=%zu small_peer_object=%zu "
      "map_candidate_capacity=%zu map_peer_capacity=%zu "
      "queue_candidate_storage=%zu queue_peer_storage=%zu "
      "bitset_payload_bytes=%zu queue_progress=lock-free run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok,
      affinity.topology, affinity.cpu,
      worker_cpu != nullptr ? worker_cpu : "external",
      worker_cpu != nullptr ? affinity.ok : "not-applicable", kBatch,
      sizeof(hyle::container::SmallVector<std::uint32_t, 16>),
      sizeof(absl::InlinedVector<std::uint32_t, 16>),
      resource_map.candidate_capacity(), resource_map.peer_capacity(),
      queue_candidate_storage,
      (kQueueCapacity + 1) *
          sizeof(rigtorp::mpmc::Slot<std::uint64_t>),
      kBitBlocks * sizeof(std::uint64_t),
      hyle::bench::perf::run_nonce());

  SequenceInput input{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint32_t>(index * 7 + 1);
  bench_sequence(peer_first, input);
  bench_associative(peer_first, peer_allocation_first);
  bench_queue(peer_first, peer_allocation_first);
  bench_active_set(peer_first, peer_allocation_first);
  bench_hash(peer_first);
  return 0;
}
