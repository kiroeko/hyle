#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <vector>

#include "hyle/container/byte_ring_buffer.h"
#include "hyle/container/ring_deque.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::container::ByteRingBuffer;
using hyle::container::RingDeque;

constexpr std::size_t kQueueCapacity = 4096;
constexpr std::size_t kQueueOperations = 1u << 16;
constexpr std::size_t kByteCapacity = 1u << 16;
constexpr std::size_t kByteBacklog = 1u << 15;
constexpr std::size_t kChunk = 4096;
constexpr std::size_t kRounds = 64;
} // namespace

int main() {
  RingDeque<std::uint64_t> hyle_queue(kQueueCapacity);
  std::deque<std::uint64_t> std_queue;
  std::vector<std::uint64_t> floor_storage(kQueueCapacity);
  for (std::size_t i = 0; i < kQueueCapacity / 2; ++i) {
    hyle_queue.push_back(i);
    std_queue.push_back(i);
    floor_storage[i] = i;
  }
  std::size_t floor_head = 0;
  std::size_t floor_tail = kQueueCapacity / 2;
  {
    nb::Bench bench;
    bench.title("single-thread fixed-capacity FIFO")
        .relative(true)
        .batch(kQueueOperations)
        .minEpochIterations(2048)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE RingDeque bounded", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t i = 0; i < kQueueOperations; ++i) {
        hyle_queue.try_push_back_bounded(i);
        checksum += hyle_queue.front();
        hyle_queue.pop_front();
      }
      nb::doNotOptimizeAway(checksum);
    });
    bench.run("std::deque<uint64_t>", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t i = 0; i < kQueueOperations; ++i) {
        std_queue.push_back(i);
        checksum += std_queue.front();
        std_queue.pop_front();
      }
      nb::doNotOptimizeAway(checksum);
    });
    bench.run("portable bounded ring (Folly shape)", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t i = 0; i < kQueueOperations; ++i) {
        if (floor_tail - floor_head < kQueueCapacity) {
          floor_storage[floor_tail & (kQueueCapacity - 1)] = i;
          ++floor_tail;
        }
        checksum += floor_storage[floor_head & (kQueueCapacity - 1)];
        ++floor_head;
      }
      nb::doNotOptimizeAway(checksum);
    });
  }

  std::array<std::byte, kChunk> input{};
  std::array<std::byte, kChunk> output{};
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::byte>(i & 0xffu);

  ByteRingBuffer hyle_bytes(kByteCapacity);
  std::vector<std::byte> initial(kByteBacklog, std::byte{0x5a});
  hyle_bytes.write(initial);
  std::vector<std::byte> linear(kByteCapacity);
  std::memcpy(linear.data(), initial.data(), initial.size());
  std::size_t linear_size = initial.size();
  std::deque<std::byte> deque_bytes(initial.begin(), initial.end());
  {
    nb::Bench bench;
    bench.title("stream staging: 32KiB backlog, 4KiB write+read")
        .relative(true)
        .batch(kRounds * kChunk * 2)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE ByteRingBuffer copy API", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t round = 0; round < kRounds; ++round) {
        hyle_bytes.write(input);
        hyle_bytes.read(output);
        checksum += std::to_integer<unsigned>(output[round & (kChunk - 1)]);
      }
      nb::doNotOptimizeAway(checksum);
    });
    bench.run("linear byte buffer + memmove", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t round = 0; round < kRounds; ++round) {
        std::memcpy(linear.data() + linear_size, input.data(), input.size());
        linear_size += input.size();
        std::memcpy(output.data(), linear.data(), output.size());
        std::memmove(linear.data(), linear.data() + output.size(),
                     linear_size - output.size());
        linear_size -= output.size();
        checksum += std::to_integer<unsigned>(output[round & (kChunk - 1)]);
      }
      nb::doNotOptimizeAway(checksum);
    });
    bench.run("std::deque<byte> element API", [&] {
      std::uint64_t checksum = 0;
      for (std::size_t round = 0; round < kRounds; ++round) {
        for (std::byte value : input)
          deque_bytes.push_back(value);
        for (std::size_t i = 0; i < output.size(); ++i) {
          output[i] = deque_bytes.front();
          deque_bytes.pop_front();
        }
        checksum += std::to_integer<unsigned>(output[round & (kChunk - 1)]);
      }
      nb::doNotOptimizeAway(checksum);
    });
  }
}
