// Public-API scaling probe. Build the same TU against before/after libraries.
// Setup, payload allocation and import stay outside the warm timing interval.
#include "hyle/compute/state_versions.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace c = hyle::compute;
namespace m = hyle::mem;
using Clock = std::chrono::steady_clock;

int main() {
#if defined(_WIN32)
  if (!SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << 4)) return 2;
#endif
  for (const std::uint32_t count : {128u, 512u, 2048u}) {
    for (const std::uint32_t roots : {1u, 4u}) {
      const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 1>{4}).value();
      std::vector<c::StateEntrySpec> specs(count, {.description = desc});
      auto schema = c::StateSchema::create(specs).value();
      auto store = c::StateStore::create(schema, {.max_versions = 2, .max_drafts = 1,
                                                .max_retained_blocks = count}).value();
      std::vector<c::LeasePool> pools;
      for (std::uint32_t i = 0; i < roots; ++i)
        pools.push_back(c::LeasePool::create(m::system_resource(), count / roots * 64,
            {.max_blocks = count / roots, .backing_alignment = 64}).value());
      std::vector<c::StateMember> members;
      for (std::uint32_t i = 0; i < count; ++i) {
        auto writer = pools[i % roots].allocate(16, 64).value();
        if (!writer.view()->fill(std::byte{7}).ok()) return 3;
        members.push_back({schema.entry(i).value(), writer.publish().value()});
      }
      auto version = store.import(members).value();
      const auto expected = store.stats();
      if (expected.retained_blocks != count || expected.backing_roots != roots ||
          expected.backing_bytes != count * 64u || expected.versions != 1) return 4;
      const auto rounds = 32768u / count;
      std::uint64_t checksum = 0;
      // Warm up the exact operation, without resetting or changing its inputs.
      for (unsigned i = 0; i < 8; ++i) checksum += store.stats().backing_bytes;
      const auto start = Clock::now();
      for (unsigned i = 0; i < rounds; ++i) {
        const auto stats = store.stats();
        checksum += stats.backing_bytes + stats.retained_blocks + stats.backing_roots;
      }
      const auto elapsed = Clock::now() - start;
      const auto ns = std::chrono::duration<double, std::nano>(elapsed).count();
      std::printf("{\"blocks\":%u,\"roots\":%u,\"rounds\":%u,\"total_ns\":%.0f,"
                  "\"ns_per_call\":%.3f,\"checksum\":%llu}\n",
                  count, roots, rounds, ns, ns / rounds,
                  static_cast<unsigned long long>(checksum));
    }
  }
}
